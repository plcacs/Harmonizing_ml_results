import os
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, List, Mapping, Optional, TypeVar, Union, cast

from typing_extensions import Protocol, runtime_checkable

from dbt import deprecations
from dbt.adapters.contracts.connection import QueryComment
from dbt.clients.yaml_helper import load_yaml_text
from dbt.config.selectors import SelectorDict
from dbt.config.utils import normalize_warn_error_options
from dbt.constants import (
    DBT_PROJECT_FILE_NAME,
    DEPENDENCIES_FILE_NAME,
    PACKAGE_LOCK_HASH_KEY,
    PACKAGES_FILE_NAME,
)
from dbt.contracts.project import PackageConfig
from dbt.contracts.project import Project as ProjectContract
from dbt.contracts.project import ProjectFlags, ProjectPackageMetadata, SemverString
from dbt.exceptions import (
    DbtExclusivePropertyUseError,
    DbtProjectError,
    DbtRuntimeError,
    ProjectContractBrokenError,
    ProjectContractError,
)
from dbt.flags import get_flags
from dbt.graph import SelectionSpec
from dbt.node_types import NodeType
from dbt.utils import MultiDict, coerce_dict_str, md5
from dbt.version import get_installed_version
from dbt_common.clients.system import load_file_contents, path_exists
from dbt_common.dataclass_schema import ValidationError
from dbt_common.exceptions import SemverError
from dbt_common.helper_types import NoValue
from dbt_common.semver import VersionSpecifier, versions_compatible

from .renderer import DbtProjectYamlRenderer, PackageRenderer
from .selectors import (
    SelectorConfig,
    selector_config_from_data,
    selector_data_from_root,
)

INVALID_VERSION_ERROR = """\
This version of dbt is not supported with the '{package}' package.
  Installed version of dbt: {installed}
  Required version of dbt for '{package}': {version_spec}
Check for a different version of the '{package}' package, or run dbt again with \
--no-version-check
"""


IMPOSSIBLE_VERSION_ERROR = """\
The package version requirement can never be satisfied for the '{package}
package.
  Required versions of dbt for '{package}': {version_spec}
Check for a different version of the '{package}' package, or run dbt again with \
--no-version-check
"""

MALFORMED_PACKAGE_ERROR = """\
The packages.yml file in this project is malformed. Please double check
the contents of this file and fix any errors before retrying.

You can find more information on the syntax for this file here:
https://docs.getdbt.com/docs/package-management

Validator Error:
{error}
"""

MISSING_DBT_PROJECT_ERROR = """\
No {DBT_PROJECT_FILE_NAME} found at expected path {path}
Verify that each entry within packages.yml (and their transitive dependencies) contains a file named {DBT_PROJECT_FILE_NAME}
"""


@runtime_checkable
class IsFQNResource(Protocol):
    fqn: List[str]
    resource_type: NodeType
    package_name: str


def _load_yaml(path: str) -> Any:
    contents = load_file_contents(path)
    return load_yaml_text(contents)


def load_yml_dict(file_path: str) -> Dict[str, Any]:
    ret: Dict[str, Any] = {}
    if path_exists(file_path):
        ret = _load_yaml(file_path) or {}
    return ret


def package_and_project_data_from_root(project_root: str) -> tuple[Dict[str, Any], str]:
    packages_yml_dict = load_yml_dict(f"{project_root}/{PACKAGES_FILE_NAME}")
    dependencies_yml_dict = load_yml_dict(f"{project_root}/{DEPENDENCIES_FILE_NAME}")

    if "packages" in packages_yml_dict and "packages" in dependencies_yml_dict:
        msg = "The 'packages' key cannot be specified in both packages.yml and dependencies.yml"
        raise DbtProjectError(msg)
    if "projects" in packages_yml_dict:
        msg = "The 'projects' key cannot be specified in packages.yml"
        raise DbtProjectError(msg)

    packages_specified_path = PACKAGES_FILE_NAME
    packages_dict: Dict[str, Any] = {}
    if "packages" in dependencies_yml_dict:
        packages_dict["packages"] = dependencies_yml_dict["packages"]
        packages_specified_path = DEPENDENCIES_FILE_NAME
    else:  # don't check for "packages" here so we capture invalid keys in packages.yml
        packages_dict = packages_yml_dict

    return packages_dict, packages_specified_path


def package_config_from_data(
    packages_data: Dict[str, Any],
    unrendered_packages_data: Optional[Dict[str, Any]] = None,
) -> PackageConfig:
    if not packages_data:
        packages_data = {"packages": []}

    # this depends on the two lists being in the same order
    if unrendered_packages_data:
        unrendered_packages_data = deepcopy(unrendered_packages_data)
        for i in range(0, len(packages_data.get("packages", []))):
            packages_data["packages"][i]["unrendered"] = unrendered_packages_data["packages"][i]

    if PACKAGE_LOCK_HASH_KEY in packages_data:
        packages_data.pop(PACKAGE_LOCK_HASH_KEY)
    try:
        PackageConfig.validate(packages_data)
        packages = PackageConfig.from_dict(packages_data)
    except ValidationError as e:
        raise DbtProjectError(MALFORMED_PACKAGE_ERROR.format(error=str(e.message))) from e
    return packages


def _parse_versions(versions: Union[List[str], str]) -> List[VersionSpecifier]:
    """Parse multiple versions as read from disk. The versions value may be any
    one of:
        - a single version string ('>0.12.1')
        - a single string specifying multiple comma-separated versions
            ('>0.11.1,<=0.12.2')
        - an array of single-version strings (['>0.11.1', '<=0.12.2'])

    Regardless, this will return a list of VersionSpecifiers
    """
    if isinstance(versions, str):
        versions = versions.split(",")
    return [VersionSpecifier.from_version_string(v) for v in versions]


def _all_source_paths(*args: List[str]) -> List[str]:
    paths = chain(*args)
    # Strip trailing slashes since the path is the same even though the name is not
    stripped_paths = map(lambda s: s.rstrip("/"), paths)
    return list(set(stripped_paths))


T = TypeVar("T")


def flag_or(flag: Optional[T], value: Optional[T], default: T) -> T:
    if flag is None:
        return value_or(value, default)
    else:
        return flag


def value_or(value: Optional[T], default: T) -> T:
    if value is None:
        return default
    else:
        return value


def load_raw_project(project_root: str) -> Dict[str, Any]:
    project_root = os.path.normpath(project_root)
    project_yaml_filepath = os.path.join(project_root, DBT_PROJECT_FILE_NAME)

    # get the project.yml contents
    if not path_exists(project_yaml_filepath):
        raise DbtProjectError(
            MISSING_DBT_PROJECT_ERROR.format(
                path=project_yaml_filepath, DBT_PROJECT_FILE_NAME=DBT_PROJECT_FILE_NAME
            )
        )

    project_dict = _load_yaml(project_yaml_filepath)

    if not isinstance(project_dict, dict):
        raise DbtProjectError(f"{DBT_PROJECT_FILE_NAME} does not parse to a dictionary")

    if "tests" in project_dict and "data_tests" not in project_dict:
        project_dict["data_tests"] = project_dict.pop("tests")

    return project_dict


def _query_comment_from_cfg(
    cfg_query_comment: Union[QueryComment, NoValue, str, None]
) -> QueryComment:
    if not cfg_query_comment:
        return QueryComment(comment="")

    if isinstance(cfg_query_comment, str):
        return QueryComment(comment=cfg_query_comment)

    if isinstance(cfg_query_comment, NoValue):
        return QueryComment()

    return cfg_query_comment


def validate_version(dbt_version: List[VersionSpecifier], project_name: str) -> None:
    """Ensure this package works with the installed version of dbt."""
    installed = get_installed_version()
    if not versions_compatible(*dbt_version):
        msg = IMPOSSIBLE_VERSION_ERROR.format(
            package=project_name, version_spec=[x.to_version_string() for x in dbt_version]
        )
        raise DbtProjectError(msg)

    if not versions_compatible(installed, *dbt_version):
        msg = INVALID_VERSION_ERROR.format(
            package=project_name,
            installed=installed.to_version_string(),
            version_spec=[x.to_version_string() for x in dbt_version],
        )
        raise DbtProjectError(msg)


def _get_required_version(
    project_dict: Dict[str, Any],
    verify_version: bool,
) -> List[VersionSpecifier]:
    dbt_raw_version: Union[List[str], str] = ">=0.0.0"
    required = project_dict.get("require-dbt-version")
    if required is not None:
        dbt_raw_version = required

    try:
        dbt_version = _parse_versions(dbt_raw_version)
    except SemverError as e:
        raise DbtProjectError(str(e)) from e

    if verify_version:
        # no name is also an error that we want to raise
        if "name" not in project_dict:
            raise DbtProjectError(
                'Required "name" field not present in project',
            )
        validate_version(dbt_version, project_dict["name"])

    return dbt_version


@dataclass
class RenderComponents:
    project_dict: Dict[str, Any] = field(metadata=dict(description="The project dictionary"))
    packages_dict: Dict[str, Any] = field(metadata=dict(description="The packages dictionary"))
    selectors_dict: Dict[str, Any] = field(metadata=dict(description="The selectors dictionary"))


@dataclass
class PartialProject(RenderComponents):
    # This class includes the project_dict, packages_dict, selectors_dict, etc from RenderComponents
    profile_name: Optional[str] = field(
        metadata=dict(description="The unrendered profile name in the project, if set"))
    project_name: Optional[str] = field(
        metadata=dict(
            description=(
                "The name of the project. This should always be set and will not be rendered"
            )
        )
    )
    project_root: str = field(
        metadata=dict(description="The root directory of the project"),
    )
    verify_version: bool = field(
        metadata=dict(description=("If True, verify the dbt version matches the required version"))
    )
    packages_specified_path: str = field(
        metadata=dict(description="The filename where packages were specified"))
    )

    def render_profile_name(self, renderer: DbtProjectYamlRenderer) -> Optional[str]:
        if self.profile_name is None:
            return None
        return renderer.render_value(self.profile_name)

    def get_rendered(
        self,
        renderer: DbtProjectYamlRenderer,
    ) -> RenderComponents:
        rendered_project = renderer.render_project(self.project_dict, self.project_root)
        rendered_packages = renderer.render_packages(
            self.packages_dict, self.packages_specified_path
        )
        rendered_selectors = renderer.render_selectors(self.selectors_dict)

        return RenderComponents(
            project_dict=rendered_project,
            packages_dict=rendered_packages,
            selectors_dict=rendered_selectors,
        )

    # Called by Project.from_project_root which first calls PartialProject.from_project_root
    def render(self, renderer: DbtProjectYamlRenderer) -> "Project":
        try:
            rendered = self.get_rendered(renderer)
            return self.create_project(rendered)
        except DbtProjectError as exc:
            if exc.path is None:
                exc.path = os.path.join(self.project_root, DBT_PROJECT_FILE_NAME)
            raise

    def render_package_metadata(self, renderer: PackageRenderer) -> ProjectPackageMetadata:
        packages_data = renderer.render_data(self.packages_dict)
        packages_config = package_config_from_data(packages_data, self.packages_dict)
        if not self.project_name:
            raise DbtProjectError(f"Package defined in {DBT_PROJECT_FILE_NAME} must have a name!")
        return ProjectPackageMetadata(self.project_name, packages_config.packages)

    def check_config_path(
        self, 
        project_dict: Dict[str, Any], 
        deprecated_path: str, 
        expected_path: Optional[str] = None, 
        default_value: Any = None
    ) -> None:
        if deprecated_path in project_dict:
            if expected_path in project_dict:
                msg = (
                    "{deprecated_path} and {expected_path} cannot both be defined. The "
                    "`{deprecated_path}` config has been deprecated in favor of `{expected_path}`. "
                    f"Please update your `{DBT_PROJECT_FILE_NAME}` configuration to reflect this "
                    "change."
                )
                raise DbtProjectError(
                    msg.format(deprecated_path=deprecated_path, expected_path=expected_path)
                )
            # this field is no longer supported, but many projects may specify it with the default value
            # if so, let's only raise this deprecation warning if they set a custom value
            if not default_value or project_dict[deprecated_path] != default_value:
                kwargs = {"deprecated_path": deprecated_path}
                if expected_path:
                    kwargs.update({"exp_path": expected_path})
                deprecations.warn(f"project-config-{deprecated_path}", **kwargs)

    def create_project(self, rendered: RenderComponents) -> "Project":
        unrendered = RenderComponents(
            project_dict=self.project_dict,
            packages_dict=self.packages_dict,
            selectors_dict=self.selectors_dict,
        )
        dbt_version = _get_required_version(
            rendered.project_dict,
            verify_version=self.verify_version,
        )

        self.check_config_path(rendered.project_dict, "source-paths", "model-paths")
        self.check_config_path(rendered.project_dict, "data-paths", "seed-paths")
        self.check_config_path(rendered.project_dict, "log-path", default_value="logs")
        self.check_config_path(rendered.project_dict, "target-path", default_value="target")

        try:
            ProjectContract.validate(rendered.project_dict)
            cfg = ProjectContract.from_dict(rendered.project_dict)
        except ValidationError as e:
            raise ProjectContractError(e) from e
        # name/version are required in the Project definition, so we can assume
        # they are present
        name = cfg.name
        version = cfg.version
        # this is added at project_dict parse time and should always be here
        # once we see it.
        if cfg.project_root is None:
            raise DbtProjectError("cfg must have a project root!")
        else:
            project_root = cfg.project_root
        # this is only optional in the sense that if it's not present, it needs
        # to have been a cli argument.
        profile_name = cfg.profile
        # these are all the defaults

        # `source_paths` is deprecated but still allowed. Copy it into
        # `model_paths` to simlify logic throughout the rest of the system.
        model_paths: List[str] = value_or(
            cfg.model_paths if "model-paths" in rendered.project_dict else cfg.source_paths,
            ["models"],
        )
        macro_paths: List[str] = value_or(cfg.macro_paths, ["macros"])
        # `data_paths` is deprecated but still allowed. Copy it into
        # `seed_paths` to simlify logic throughout the rest of the system.
        seed_paths: List[str] = value_or(
            cfg.seed_paths if "seed-paths" in rendered.project_dict else cfg.data_paths, ["seeds"]
        )
        test_paths: List[str] = value_or(cfg.test_paths, ["tests"])
        analysis_paths: List[str] = value_or(cfg.analysis_paths, ["analyses"])
        snapshot_paths: List[str] = value_or(cfg.snapshot_paths, ["snapshots"])

        all_source_paths: List[str] = _all_source_paths(
            model_paths, seed_paths, snapshot_paths, analysis_paths, macro_paths, test_paths
        )

        docs_paths: List[str] = value_or(cfg.docs_paths, all_source_paths)
        asset_paths: List[str] = value_or(cfg.asset_paths, [])
        global_flags = get_flags()

        flag_target_path = str(global_flags.TARGET_PATH) if global_flags.TARGET_PATH else None
        target_path: str = flag_or(flag_target_path, cfg.target_path, "target")
        log_path: str = str(global_flags.LOG_PATH)

        clean_targets: List[str] = value_or(cfg.clean_targets, [target_path])
        packages_install_path: str = value_or(cfg.packages_install_path, "dbt_packages")
        # in the default case we'll populate this once we know the adapter type
        # It would be nice to just pass along a Quoting here, but that would
        # break many things
        quoting: Dict[str, Any] = {}
        if cfg.quoting is not None:
            quoting = cfg.quoting.to_dict(omit_none=True)

        dispatch: List[Dict[str, Any]]
        models: Dict[str, Any]
        seeds: Dict[str, Any]
        snapshots: Dict[str, Any]
        sources: Dict[str, Any]
        data_tests: Dict[str, Any]
        unit_tests: Dict[str, Any]
        metrics: Dict[str, Any]
        semantic_models: Dict[str, Any]
        saved_queries: Dict[str, Any]
        exposures: Dict[str, Any]
        vars_value: VarProvider
        dbt_cloud: Dict[str, Any]

        dispatch = cfg
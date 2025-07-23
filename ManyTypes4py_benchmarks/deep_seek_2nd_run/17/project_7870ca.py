import os
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, List, Mapping, Optional, TypeVar, Union, Set, cast
from typing_extensions import Protocol, runtime_checkable
from dbt import deprecations
from dbt.adapters.contracts.connection import QueryComment
from dbt.clients.yaml_helper import load_yaml_text
from dbt.config.selectors import SelectorDict
from dbt.config.utils import normalize_warn_error_options
from dbt.constants import DBT_PROJECT_FILE_NAME, DEPENDENCIES_FILE_NAME, PACKAGE_LOCK_HASH_KEY, PACKAGES_FILE_NAME
from dbt.contracts.project import PackageConfig
from dbt.contracts.project import Project as ProjectContract
from dbt.contracts.project import ProjectFlags, ProjectPackageMetadata, SemverString
from dbt.exceptions import DbtExclusivePropertyUseError, DbtProjectError, DbtRuntimeError, ProjectContractBrokenError, ProjectContractError
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
from .selectors import SelectorConfig, selector_config_from_data, selector_data_from_root

INVALID_VERSION_ERROR: str = "This version of dbt is not supported with the '{package}' package.\n  Installed version of dbt: {installed}\n  Required version of dbt for '{package}': {version_spec}\nCheck for a different version of the '{package}' package, or run dbt again with --no-version-check\n"
IMPOSSIBLE_VERSION_ERROR: str = "The package version requirement can never be satisfied for the '{package}\npackage.\n  Required versions of dbt for '{package}': {version_spec}\nCheck for a different version of the '{package}' package, or run dbt again with --no-version-check\n"
MALFORMED_PACKAGE_ERROR: str = 'The packages.yml file in this project is malformed. Please double check\nthe contents of this file and fix any errors before retrying.\n\nYou can find more information on the syntax for this file here:\nhttps://docs.getdbt.com/docs/package-management\n\nValidator Error:\n{error}\n'
MISSING_DBT_PROJECT_ERROR: str = 'No {DBT_PROJECT_FILE_NAME} found at expected path {path}\nVerify that each entry within packages.yml (and their transitive dependencies) contains a file named {DBT_PROJECT_FILE_NAME}\n'

@runtime_checkable
class IsFQNResource(Protocol):
    pass

def _load_yaml(path: str) -> Dict[str, Any]:
    contents: str = load_file_contents(path)
    return cast(Dict[str, Any], load_yaml_text(contents))

def load_yml_dict(file_path: str) -> Dict[str, Any]:
    ret: Dict[str, Any] = {}
    if path_exists(file_path):
        ret = _load_yaml(file_path) or {}
    return ret

def package_and_project_data_from_root(project_root: str) -> tuple[Dict[str, Any], str]:
    packages_yml_dict: Dict[str, Any] = load_yml_dict(f'{project_root}/{PACKAGES_FILE_NAME}')
    dependencies_yml_dict: Dict[str, Any] = load_yml_dict(f'{project_root}/{DEPENDENCIES_FILE_NAME}')
    if 'packages' in packages_yml_dict and 'packages' in dependencies_yml_dict:
        msg: str = "The 'packages' key cannot be specified in both packages.yml and dependencies.yml"
        raise DbtProjectError(msg)
    if 'projects' in packages_yml_dict:
        msg = "The 'projects' key cannot be specified in packages.yml"
        raise DbtProjectError(msg)
    packages_specified_path: str = PACKAGES_FILE_NAME
    packages_dict: Dict[str, Any] = {}
    if 'packages' in dependencies_yml_dict:
        packages_dict['packages'] = dependencies_yml_dict['packages']
        packages_specified_path = DEPENDENCIES_FILE_NAME
    else:
        packages_dict = packages_yml_dict
    return (packages_dict, packages_specified_path)

def package_config_from_data(packages_data: Dict[str, Any], unrendered_packages_data: Optional[Dict[str, Any]] = None) -> PackageConfig:
    if not packages_data:
        packages_data = {'packages': []}
    if unrendered_packages_data:
        unrendered_packages_data = deepcopy(unrendered_packages_data)
        for i in range(0, len(packages_data.get('packages', []))):
            packages_data['packages'][i]['unrendered'] = unrendered_packages_data['packages'][i]
    if PACKAGE_LOCK_HASH_KEY in packages_data:
        packages_data.pop(PACKAGE_LOCK_HASH_KEY)
    try:
        PackageConfig.validate(packages_data)
        packages: PackageConfig = PackageConfig.from_dict(packages_data)
    except ValidationError as e:
        raise DbtProjectError(MALFORMED_PACKAGE_ERROR.format(error=str(e.message))) from e
    return packages

def _parse_versions(versions: Union[str, List[str]]) -> List[VersionSpecifier]:
    """Parse multiple versions as read from disk. The versions value may be any
    one of:
        - a single version string ('>0.12.1')
        - a single string specifying multiple comma-separated versions
            ('>0.11.1,<=0.12.2')
        - an array of single-version strings (['>0.11.1', '<=0.12.2'])

    Regardless, this will return a list of VersionSpecifiers
    """
    if isinstance(versions, str):
        versions = versions.split(',')
    return [VersionSpecifier.from_version_string(v) for v in versions]

def _all_source_paths(*args: List[str]) -> List[str]:
    paths = chain(*args)
    stripped_paths = map(lambda s: s.rstrip('/'), paths)
    return list(set(stripped_paths))

T = TypeVar('T')

def flag_or(flag: Optional[str], value: Optional[str], default: str) -> str:
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
    if not path_exists(project_yaml_filepath):
        raise DbtProjectError(MISSING_DBT_PROJECT_ERROR.format(path=project_yaml_filepath, DBT_PROJECT_FILE_NAME=DBT_PROJECT_FILE_NAME))
    project_dict: Dict[str, Any] = _load_yaml(project_yaml_filepath)
    if not isinstance(project_dict, dict):
        raise DbtProjectError(f'{DBT_PROJECT_FILE_NAME} does not parse to a dictionary')
    if 'tests' in project_dict and 'data_tests' not in project_dict:
        project_dict['data_tests'] = project_dict.pop('tests')
    return project_dict

def _query_comment_from_cfg(cfg_query_comment: Union[str, NoValue, QueryComment, None]) -> QueryComment:
    if not cfg_query_comment:
        return QueryComment(comment='')
    if isinstance(cfg_query_comment, str):
        return QueryComment(comment=cfg_query_comment)
    if isinstance(cfg_query_comment, NoValue):
        return QueryComment()
    return cfg_query_comment

def validate_version(dbt_version: List[VersionSpecifier], project_name: str) -> None:
    """Ensure this package works with the installed version of dbt."""
    installed: VersionSpecifier = get_installed_version()
    if not versions_compatible(*dbt_version):
        msg: str = IMPOSSIBLE_VERSION_ERROR.format(package=project_name, version_spec=[x.to_version_string() for x in dbt_version])
        raise DbtProjectError(msg)
    if not versions_compatible(installed, *dbt_version):
        msg = INVALID_VERSION_ERROR.format(package=project_name, installed=installed.to_version_string(), version_spec=[x.to_version_string() for x in dbt_version])
        raise DbtProjectError(msg)

def _get_required_version(project_dict: Dict[str, Any], verify_version: bool) -> List[VersionSpecifier]:
    dbt_raw_version: str = '>=0.0.0'
    required: Optional[Union[str, List[str]]] = project_dict.get('require-dbt-version')
    if required is not None:
        dbt_raw_version = required
    try:
        dbt_version: List[VersionSpecifier] = _parse_versions(dbt_raw_version)
    except SemverError as e:
        raise DbtProjectError(str(e)) from e
    if verify_version:
        if 'name' not in project_dict:
            raise DbtProjectError('Required "name" field not present in project')
        validate_version(dbt_version, project_dict['name'])
    return dbt_version

@dataclass
class RenderComponents:
    project_dict: Dict[str, Any] = field(metadata=dict(description='The project dictionary'))
    packages_dict: Dict[str, Any] = field(metadata=dict(description='The packages dictionary'))
    selectors_dict: Dict[str, Any] = field(metadata=dict(description='The selectors dictionary'))

@dataclass
class PartialProject(RenderComponents):
    profile_name: Optional[str] = field(metadata=dict(description='The unrendered profile name in the project, if set'))
    project_name: Optional[str] = field(metadata=dict(description='The name of the project. This should always be set and will not be rendered'))
    project_root: str = field(metadata=dict(description='The root directory of the project'))
    verify_version: bool = field(metadata=dict(description='If True, verify the dbt version matches the required version'))
    packages_specified_path: str = field(metadata=dict(description='The filename where packages were specified'))

    def render_profile_name(self, renderer: DbtProjectYamlRenderer) -> Optional[str]:
        if self.profile_name is None:
            return None
        return renderer.render_value(self.profile_name)

    def get_rendered(self, renderer: DbtProjectYamlRenderer) -> RenderComponents:
        rendered_project: Dict[str, Any] = renderer.render_project(self.project_dict, self.project_root)
        rendered_packages: Dict[str, Any] = renderer.render_packages(self.packages_dict, self.packages_specified_path)
        rendered_selectors: Dict[str, Any] = renderer.render_selectors(self.selectors_dict)
        return RenderComponents(project_dict=rendered_project, packages_dict=rendered_packages, selectors_dict=rendered_selectors)

    def render(self, renderer: DbtProjectYamlRenderer) -> 'Project':
        try:
            rendered: RenderComponents = self.get_rendered(renderer)
            return self.create_project(rendered)
        except DbtProjectError as exc:
            if exc.path is None:
                exc.path = os.path.join(self.project_root, DBT_PROJECT_FILE_NAME)
            raise

    def render_package_metadata(self, renderer: DbtProjectYamlRenderer) -> ProjectPackageMetadata:
        packages_data: Dict[str, Any] = renderer.render_data(self.packages_dict)
        packages_config: PackageConfig = package_config_from_data(packages_data, self.packages_dict)
        if not self.project_name:
            raise DbtProjectError(f'Package defined in {DBT_PROJECT_FILE_NAME} must have a name!')
        return ProjectPackageMetadata(self.project_name, packages_config.packages)

    def check_config_path(self, project_dict: Dict[str, Any], deprecated_path: str, expected_path: Optional[str] = None, default_value: Optional[Any] = None) -> None:
        if deprecated_path in project_dict:
            if expected_path in project_dict:
                msg: str = f'{{deprecated_path}} and {{expected_path}} cannot both be defined. The `{{deprecated_path}}` config has been deprecated in favor of `{{expected_path}}`. Please update your `{DBT_PROJECT_FILE_NAME}` configuration to reflect this change.'
                raise DbtProjectError(msg.format(deprecated_path=deprecated_path, expected_path=expected_path))
            if not default_value or project_dict[deprecated_path] != default_value:
                kwargs: Dict[str, Any] = {'deprecated_path': deprecated_path}
                if expected_path:
                    kwargs.update({'exp_path': expected_path})
                deprecations.warn(f'project-config-{deprecated_path}', **kwargs)

    def create_project(self, rendered: RenderComponents) -> 'Project':
        unrendered: RenderComponents = RenderComponents(project_dict=self.project_dict, packages_dict=self.packages_dict, selectors_dict=self.selectors_dict)
        dbt_version: List[VersionSpecifier] = _get_required_version(rendered.project_dict, verify_version=self.verify_version)
        self.check_config_path(rendered.project_dict, 'source-paths', 'model-paths')
        self.check_config_path(rendered.project_dict, 'data-paths', 'seed-paths')
        self.check_config_path(rendered.project_dict, 'log-path', default_value='logs')
        self.check_config_path(rendered.project_dict, 'target-path', default_value='target')
        try:
            ProjectContract.validate(rendered.project_dict)
            cfg: ProjectContract = ProjectContract.from_dict(rendered.project_dict)
        except ValidationError as e:
            raise ProjectContractError(e) from e
        name: str = cfg.name
        version: Optional[SemverString] = cfg.version
        if cfg.project_root is None:
            raise DbtProjectError('cfg must have a project root!')
        else:
            project_root: str = cfg.project_root
        profile_name: Optional[str] = cfg.profile
        model_paths: List[str] = value_or(cfg.model_paths if 'model-paths' in rendered.project_dict else cfg.source_paths, ['models'])
        macro_paths: List[str] = value_or(cfg.macro_paths, ['macros'])
        seed_paths: List[str] = value_or(cfg.seed_paths if 'seed-paths' in rendered.project_dict else cfg.data_paths, ['seeds'])
        test_paths: List[str] = value_or(cfg.test_paths, ['tests'])
        analysis_paths: List[str] = value_or(cfg.analysis_paths, ['analyses'])
        snapshot_paths: List[str] = value_or(cfg.snapshot_paths, ['snapshots'])
        all_source_paths: List[str] = _all_source_paths(model_paths, seed_paths, snapshot_paths, analysis_paths, macro_paths, test_paths)
        docs_paths: List[str] = value_or(cfg.docs_paths, all_source_paths)
        asset_paths: List[str] = value_or(cfg.asset_paths, [])
        global_flags: Any = get_flags()
        flag_target_path: Optional[str] = str(global_flags.TARGET_PATH) if global_flags.TARGET_PATH else None
        target_path: str = flag_or(flag_target_path, cfg.target_path, 'target')
        log_path: str = str(global_flags.LOG_PATH)
        clean_targets: List[str] = value_or(cfg.clean_targets, [target_path])
        packages_install_path: str = value_or(cfg.packages_install_path, 'dbt_packages')
        quoting: Dict[str, Any] = {}
        if cfg.quoting is not None:
            quoting = cfg.quoting.to_dict(omit_none=True)
        dispatch: List[Dict[str, Any]] = cfg.dispatch
        models: Dict[str, Any] = cfg.models
        seeds: Dict[str, Any] = cfg.seeds
        snapshots: Dict[str, Any] = cfg.snapshots
        sources: Dict[str, Any] = cfg.sources
        data_tests: Dict[str, Any] = cfg.data_tests if 'data_tests' in rendered.project_dict else cfg.tests
        unit_tests: Dict[str, Any] = cfg.unit_tests
        metrics: Dict[str, Any] = cfg.metrics
        semantic_models: Dict[str, Any] = cfg.semantic_models
        saved_queries: Dict[str, Any] = cfg.saved_queries
        exposures: Dict[str, Any] = cfg.exposures
        vars_dict: Dict[str, Any] = {} if cfg.vars is None else cfg.vars
        vars_value: VarProvider = VarProvider(vars_dict)
        project_env_vars: Dict[str, str] = {}
        on_run_start: List[str] = value_or(cfg.on_run_start, [])
        on_run_end: List[str] = value_or(cfg.on_run_end, [])
        query_comment: QueryComment = _query_comment_from_cfg(cfg.query_comment)
        packages: PackageConfig = package_config_from_data(rendered.packages_dict, unrendered.packages_dict)
        selectors: SelectorConfig = selector_config_from_data(rendered.selectors_dict)
        manifest_selectors: Dict[str, Any] = {}
        if rendered.selectors_dict and rendered.selectors_dict['selectors']:
            manifest_selectors = SelectorDict.parse_from_selectors_list(rendered.selectors_dict['selectors'])
        dbt_cloud: Dict[str, Any] = cfg.dbt_cloud
        flags: ProjectFlags = cfg.flags
        project: Project = Project(project_name=name, version=version, project_root=project_root, profile_name=profile_name, model_paths=model_paths, macro_paths=macro_paths, seed_paths=seed_paths, test_paths=test_paths, analysis
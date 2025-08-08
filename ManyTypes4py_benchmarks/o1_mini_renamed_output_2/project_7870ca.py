import os
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, List, Mapping, Optional, TypeVar, Union, Tuple
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
from dbt.contracts.project import (
    PackageConfig,
    Project as ProjectContract,
    ProjectFlags,
    ProjectPackageMetadata,
    SemverString,
)
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
from .selectors import SelectorConfig, selector_config_from_data, selector_data_from_root

INVALID_VERSION_ERROR = """This version of dbt is not supported with the '{package}' package.
  Installed version of dbt: {installed}
  Required version of dbt for '{package}': {version_spec}
Check for a different version of the '{package}' package, or run dbt again with --no-version-check
"""
IMPOSSIBLE_VERSION_ERROR = """The package version requirement can never be satisfied for the '{package}
package.
  Required versions of dbt for '{package}': {version_spec}
Check for a different version of the '{package}' package, or run dbt again with --no-version-check
"""
MALFORMED_PACKAGE_ERROR = """The packages.yml file in this project is malformed. Please double check
the contents of this file and fix any errors before retrying.

You can find more information on the syntax for this file here:
https://docs.getdbt.com/docs/package-management

Validator Error:
{error}
"""
MISSING_DBT_PROJECT_ERROR = """No {DBT_PROJECT_FILE_NAME} found at expected path {path}
Verify that each entry within packages.yml (and their transitive dependencies) contains a file named {DBT_PROJECT_FILE_NAME}
"""


@runtime_checkable
class IsFQNResource(Protocol):
    pass


def func_o7vkfxs1(path: str) -> Any:
    contents = load_file_contents(path)
    return load_yaml_text(contents)


def func_tqmmo8nd(file_path: str) -> Dict[str, Any]:
    ret: Dict[str, Any] = {}
    if path_exists(file_path):
        ret = func_o7vkfxs1(file_path) or {}
    return ret


def func_zks8i0ya(project_root: str) -> Tuple[Dict[str, Any], str]:
    packages_yml_dict: Dict[str, Any] = func_tqmmo8nd(f"{project_root}/{PACKAGES_FILE_NAME}")
    dependencies_yml_dict: Dict[str, Any] = func_tqmmo8nd(
        f"{project_root}/{DEPENDENCIES_FILE_NAME}"
    )
    if 'packages' in packages_yml_dict and 'packages' in dependencies_yml_dict:
        msg = (
            "The 'packages' key cannot be specified in both packages.yml and dependencies.yml"
        )
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
    return packages_dict, packages_specified_path


def func_ecuqzm2q(packages_data: Dict[str, Any], unrendered_packages_data: Optional[Dict[str, Any]] = None) -> PackageConfig:
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


def func_yukxwcri(versions: Union[str, List[str]]) -> List[VersionSpecifier]:
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


def func_yrxq0138(*args: List[str]) -> List[str]:
    paths = chain(*args)
    stripped_paths = map(lambda s: s.rstrip('/'), paths)
    return list(set(stripped_paths))


T = TypeVar('T')


def func_zadv2wlh(flag: Optional[str], value: Optional[str], default: str) -> str:
    if flag is None:
        return func_grr4oqre(value, default)
    else:
        return flag


def func_grr4oqre(value: Optional[str], default: str) -> str:
    if value is None:
        return default
    else:
        return value


def func_5ire24cq(project_root: str) -> Dict[str, Any]:
    project_root = os.path.normpath(project_root)
    project_yaml_filepath = os.path.join(project_root, DBT_PROJECT_FILE_NAME)
    if not path_exists(project_yaml_filepath):
        raise DbtProjectError(
            MISSING_DBT_PROJECT_ERROR.format(
                path=project_yaml_filepath, DBT_PROJECT_FILE_NAME=DBT_PROJECT_FILE_NAME
            )
        )
    project_dict: Dict[str, Any] = func_o7vkfxs1(project_yaml_filepath)
    if not isinstance(project_dict, dict):
        raise DbtProjectError(f"{DBT_PROJECT_FILE_NAME} does not parse to a dictionary")
    if 'tests' in project_dict and 'data_tests' not in project_dict:
        project_dict['data_tests'] = project_dict.pop('tests')
    return project_dict


def func_856q5x7b(cfg_query_comment: Optional[Union[str, NoValue, QueryComment]]) -> QueryComment:
    if not cfg_query_comment:
        return QueryComment(comment='')
    if isinstance(cfg_query_comment, str):
        return QueryComment(comment=cfg_query_comment)
    if isinstance(cfg_query_comment, NoValue):
        return QueryComment()
    return cfg_query_comment


def func_6oyympen(dbt_version: List[VersionSpecifier], project_name: str) -> None:
    """Ensure this package works with the installed version of dbt."""
    installed = get_installed_version()
    if not versions_compatible(*dbt_version):
        msg = IMPOSSIBLE_VERSION_ERROR.format(
            package=project_name,
            version_spec=[x.to_version_string() for x in dbt_version],
        )
        raise DbtProjectError(msg)
    if not versions_compatible(installed, *dbt_version):
        msg = INVALID_VERSION_ERROR.format(
            package=project_name,
            installed=installed.to_version_string(),
            version_spec=[x.to_version_string() for x in dbt_version],
        )
        raise DbtProjectError(msg)


def func_56i1a9aq(project_dict: Dict[str, Any], verify_version: bool) -> List[VersionSpecifier]:
    dbt_raw_version: str = '>=0.0.0'
    required = project_dict.get('require-dbt-version')
    if required is not None:
        dbt_raw_version = required
    try:
        dbt_version = func_yukxwcri(dbt_raw_version)
    except SemverError as e:
        raise DbtProjectError(str(e)) from e
    if verify_version:
        if 'name' not in project_dict:
            raise DbtProjectError('Required "name" field not present in project')
        func_6oyympen(dbt_version, project_dict['name'])
    return dbt_version


@dataclass
class RenderComponents:
    project_dict: Dict[str, Any] = field(metadata={'description': 'The project dictionary'})
    packages_dict: Dict[str, Any] = field(metadata={'description': 'The packages dictionary'})
    selectors_dict: Dict[str, Any] = field(metadata={'description': 'The selectors dictionary'})


@dataclass
class PartialProject(RenderComponents):
    profile_name: Optional[str] = field(metadata={'description': 'The unrendered profile name in the project, if set'})
    project_name: Optional[str] = field(metadata={'description': 'The name of the project. This should always be set and will not be rendered'})
    project_root: str = field(metadata={'description': 'The root directory of the project'})
    verify_version: bool = field(metadata={'description': 'If True, verify the dbt version matches the required version'})
    packages_specified_path: str = field(metadata={'description': 'The filename where packages were specified'})

    def func_qpcyb2ck(self, renderer: DbtProjectYamlRenderer) -> Optional[str]:
        if self.profile_name is None:
            return None
        return renderer.render_value(self.profile_name)

    def func_xg5l5wa2(self, renderer: DbtProjectYamlRenderer) -> RenderComponents:
        rendered_project = renderer.render_project(self.project_dict, self.project_root)
        rendered_packages = renderer.render_packages(self.packages_dict, self.packages_specified_path)
        rendered_selectors = renderer.render_selectors(self.selectors_dict)
        return RenderComponents(
            project_dict=rendered_project,
            packages_dict=rendered_packages,
            selectors_dict=rendered_selectors,
        )

    def func_msv937ci(self, renderer: DbtProjectYamlRenderer) -> 'Project':
        try:
            rendered = self.func_xg5l5wa2(renderer)
            return self.create_project(rendered)
        except DbtProjectError as exc:
            if exc.path is None:
                exc.path = os.path.join(self.project_root, DBT_PROJECT_FILE_NAME)
            raise

    def func_c6x1br0b(self, renderer: DbtProjectYamlRenderer) -> ProjectPackageMetadata:
        packages_data = renderer.render_data(self.packages_dict)
        packages_config = func_ecuqzm2q(packages_data, self.packages_dict)
        if not self.project_name:
            raise DbtProjectError(
                f'Package defined in {DBT_PROJECT_FILE_NAME} must have a name!'
            )
        return ProjectPackageMetadata(self.project_name, packages_config.packages)

    def func_etz09swz(
        self,
        project_dict: Dict[str, Any],
        deprecated_path: str,
        expected_path: Optional[str] = None,
        default_value: Optional[Any] = None,
    ) -> None:
        if deprecated_path in project_dict:
            if expected_path in project_dict:
                msg = (
                    f'{{deprecated_path}} and {{expected_path}} cannot both be defined. The `{{deprecated_path}}` config has been deprecated in favor of `{{expected_path}}`. Please update your `{DBT_PROJECT_FILE_NAME}` configuration to reflect this change.'
                )
                raise DbtProjectError(msg.format(deprecated_path=deprecated_path, expected_path=expected_path))
            if not default_value or project_dict[deprecated_path] != default_value:
                kwargs: Dict[str, Any] = {'deprecated_path': deprecated_path}
                if expected_path:
                    kwargs.update({'exp_path': expected_path})
                deprecations.warn(f'project-config-{deprecated_path}', **kwargs)

    def func_uod83hw6(self, rendered: RenderComponents) -> 'Project':
        unrendered = RenderComponents(
            project_dict=self.project_dict,
            packages_dict=self.packages_dict,
            selectors_dict=self.selectors_dict,
        )
        dbt_version = func_56i1a9aq(rendered.project_dict, verify_version=self.verify_version)
        self.func_etz09swz(rendered.project_dict, 'source-paths', 'model-paths')
        self.func_etz09swz(rendered.project_dict, 'data-paths', 'seed-paths')
        self.func_etz09swz(rendered.project_dict, 'log-path', default_value='logs')
        self.func_etz09swz(rendered.project_dict, 'target-path', default_value='target')
        try:
            ProjectContract.validate(rendered.project_dict)
            cfg: ProjectContract = ProjectContract.from_dict(rendered.project_dict)
        except ValidationError as e:
            raise ProjectContractError(e) from e
        name: str = cfg.name
        version: str = cfg.version
        if cfg.project_root is None:
            raise DbtProjectError('cfg must have a project root!')
        else:
            project_root = cfg.project_root
        profile_name: Optional[str] = cfg.profile
        model_paths: List[str] = func_grr4oqre(cfg.model_paths if 'model-paths' in rendered.project_dict else cfg.source_paths, ['models'])
        macro_paths: List[str] = func_grr4oqre(cfg.macro_paths, ['macros'])
        seed_paths: List[str] = func_grr4oqre(cfg.seed_paths if 'seed-paths' in rendered.project_dict else cfg.data_paths, ['seeds'])
        test_paths: List[str] = func_grr4oqre(cfg.test_paths, ['tests'])
        analysis_paths: List[str] = func_grr4oqre(cfg.analysis_paths, ['analyses'])
        snapshot_paths: List[str] = func_grr4oqre(cfg.snapshot_paths, ['snapshots'])
        all_source_paths: List[str] = func_yrxq0138(model_paths, seed_paths, snapshot_paths, analysis_paths, macro_paths, test_paths)
        docs_paths: List[str] = func_grr4oqre(cfg.docs_paths, all_source_paths)
        asset_paths: List[str] = func_grr4oqre(cfg.asset_paths, [])
        global_flags = get_flags()
        flag_target_path: Optional[str] = str(global_flags.TARGET_PATH) if global_flags.TARGET_PATH else None
        target_path: str = func_zadv2wlh(flag_target_path, cfg.target_path, 'target')
        log_path: str = str(global_flags.LOG_PATH)
        clean_targets: List[str] = func_grr4oqre(cfg.clean_targets, [target_path])
        packages_install_path: List[str] = func_grr4oqre(cfg.packages_install_path, 'dbt_packages')
        quoting: Dict[str, Any] = {}
        if cfg.quoting is not None:
            quoting = cfg.quoting.to_dict(omit_none=True)
        dispatch = cfg.dispatch
        models = cfg.models
        seeds = cfg.seeds
        snapshots = cfg.snapshots
        sources = cfg.sources
        data_tests: Any = (cfg.data_tests if 'data_tests' in rendered.project_dict else cfg.tests)
        unit_tests = cfg.unit_tests
        metrics = cfg.metrics
        semantic_models = cfg.semantic_models
        saved_queries = cfg.saved_queries
        exposures = cfg.exposures
        if cfg.vars is None:
            vars_dict: Dict[str, Any] = {}
        else:
            vars_dict = cfg.vars
        vars_value = VarProvider(vars_dict)
        project_env_vars: Dict[str, Any] = {}
        on_run_start: List[str] = func_grr4oqre(cfg.on_run_start, [])
        on_run_end: List[str] = func_grr4oqre(cfg.on_run_end, [])
        query_comment: QueryComment = func_856q5x7b(cfg.query_comment)
        packages: PackageConfig = func_ecuqzm2q(rendered.packages_dict, unrendered.packages_dict)
        selectors: SelectorConfig = selector_config_from_data(rendered.selectors_dict)
        manifest_selectors: Dict[str, Any] = {}
        if rendered.selectors_dict and rendered.selectors_dict.get('selectors'):
            manifest_selectors = SelectorDict.parse_from_selectors_list(rendered.selectors_dict['selectors'])
        dbt_cloud = cfg.dbt_cloud
        flags = cfg.flags
        project = Project(
            project_name=name,
            version=version,
            project_root=project_root,
            profile_name=profile_name,
            model_paths=model_paths,
            macro_paths=macro_paths,
            seed_paths=seed_paths,
            test_paths=test_paths,
            analysis_paths=analysis_paths,
            docs_paths=docs_paths,
            asset_paths=asset_paths,
            target_path=target_path,
            snapshot_paths=snapshot_paths,
            clean_targets=clean_targets,
            log_path=log_path,
            packages_install_path=packages_install_path,
            packages_specified_path=self.packages_specified_path,
            quoting=quoting,
            models=models,
            on_run_start=on_run_start,
            on_run_end=on_run_end,
            dispatch=dispatch,
            seeds=seeds,
            snapshots=snapshots,
            dbt_version=dbt_version,
            packages=packages,
            manifest_selectors=manifest_selectors,
            selectors=selectors,
            query_comment=query_comment,
            sources=sources,
            data_tests=data_tests,
            unit_tests=unit_tests,
            metrics=metrics,
            semantic_models=semantic_models,
            saved_queries=saved_queries,
            exposures=exposures,
            vars=vars_value,
            config_version=cfg.config_version,
            unrendered=unrendered,
            project_env_vars=project_env_vars,
            restrict_access=cfg.restrict_access,
            dbt_cloud=dbt_cloud,
            flags=flags,
        )
        project.validate()
        return project

    @classmethod
    def func_cy9ol71o(
        cls,
        project_root: str,
        project_dict: Dict[str, Any],
        packages_dict: Dict[str, Any],
        selectors_dict: Dict[str, Any],
        *,
        verify_version: bool = False,
        packages_specified_path: str = PACKAGES_FILE_NAME,
    ) -> 'PartialProject':
        """Construct a partial project from its constituent dicts."""
        project_name: Optional[str] = project_dict.get('name')
        profile_name: Optional[str] = project_dict.get('profile')
        return cls(
            profile_name=profile_name,
            project_name=project_name,
            project_root=project_root,
            project_dict=project_dict,
            packages_dict=packages_dict,
            selectors_dict=selectors_dict,
            verify_version=verify_version,
            packages_specified_path=packages_specified_path,
        )

    @classmethod
    def func_ki5vjym9(cls, project_root: str, *, verify_version: bool = False) -> 'PartialProject':
        project_root = os.path.normpath(project_root)
        project_dict: Dict[str, Any] = func_5ire24cq(project_root)
        packages_dict: Dict[str, Any]
        packages_specified_path: str
        packages_dict, packages_specified_path = func_zks8i0ya(project_root)
        selectors_dict: Dict[str, Any] = selector_data_from_root(project_root)
        return cls.func_cy9ol71o(
            project_root=project_root,
            project_dict=project_dict,
            selectors_dict=selectors_dict,
            packages_dict=packages_dict,
            verify_version=verify_version,
            packages_specified_path=packages_specified_path,
        )


class VarProvider:
    """Var providers are tied to a particular Project."""

    def __init__(self, vars: Dict[str, Any]) -> None:
        self.vars: Dict[str, Any] = vars

    def func_h4p4rv0n(self, node: Any, adapter_type: str) -> MultiDict:
        merged: MultiDict = MultiDict([self.vars])
        merged.add(self.vars.get(node.package_name, {}))
        return merged

    def func_e20yzvhw(self) -> Dict[str, Any]:
        return self.vars


@dataclass
class Project:
    project_name: str
    version: str
    project_root: str
    profile_name: Optional[str]
    model_paths: List[str]
    macro_paths: List[str]
    seed_paths: List[str]
    test_paths: List[str]
    analysis_paths: List[str]
    docs_paths: List[str]
    asset_paths: List[str]
    target_path: str
    snapshot_paths: List[str]
    clean_targets: List[str]
    log_path: str
    packages_install_path: List[str]
    packages_specified_path: str
    quoting: Dict[str, Any]
    models: Any
    on_run_start: List[str]
    on_run_end: List[str]
    dispatch: Any
    seeds: Any
    snapshots: Any
    sources: Any
    data_tests: Any
    unit_tests: Any
    metrics: Any
    semantic_models: Any
    saved_queries: Any
    exposures: Any
    vars: VarProvider
    config_version: Any
    unrendered: RenderComponents
    project_env_vars: Dict[str, Any]
    restrict_access: Any
    dbt_cloud: Any
    flags: Any

    @property
    def func_ni1vim36(self) -> List[str]:
        return func_yrxq0138(self.model_paths, self.seed_paths, self.snapshot_paths, self.analysis_paths, self.macro_paths, self.test_paths)

    @property
    def func_uhggcer2(self) -> List[str]:
        generic_test_paths: List[str] = []
        for test_path in self.test_paths:
            generic_test_paths.append(os.path.join(test_path, 'generic'))
        return generic_test_paths

    @property
    def func_qeqs0kdr(self) -> List[str]:
        fixture_paths: List[str] = []
        for test_path in self.test_paths:
            fixture_paths.append(os.path.join(test_path, 'fixtures'))
        return fixture_paths

    def __str__(self) -> str:
        cfg = self.func_w61tf60m(with_packages=True)
        return str(cfg)

    def __eq__(self, other: Any) -> bool:
        if not (isinstance(other, self.__class__) and isinstance(self, other.__class__)):
            return False
        return self.func_w61tf60m(with_packages=True) == other.func_w61tf60m(with_packages=True)

    def func_w61tf60m(self, with_packages: bool = False) -> Dict[str, Any]:
        """Return a dict representation of the config that could be written to
        disk with `yaml.safe_dump` to get this configuration.

        :param with_packages bool: If True, include the serialized packages
            file in the root.
        :returns dict: The serialized profile.
        """
        result: Dict[str, Any] = deepcopy({
            'name': self.project_name,
            'version': self.version,
            'project-root': self.project_root,
            'profile': self.profile_name,
            'model-paths': self.model_paths,
            'macro-paths': self.macro_paths,
            'seed-paths': self.seed_paths,
            'test-paths': self.test_paths,
            'analysis-paths': self.analysis_paths,
            'docs-paths': self.docs_paths,
            'asset-paths': self.asset_paths,
            'target-path': self.target_path,
            'snapshot-paths': self.snapshot_paths,
            'clean-targets': self.clean_targets,
            'log-path': self.log_path,
            'quoting': self.quoting,
            'models': self.models,
            'on-run-start': self.on_run_start,
            'on-run-end': self.on_run_end,
            'dispatch': self.dispatch,
            'seeds': self.seeds,
            'snapshots': self.snapshots,
            'sources': self.sources,
            'data_tests': self.data_tests,
            'unit_tests': self.unit_tests,
            'metrics': self.metrics,
            'semantic-models': self.semantic_models,
            'saved-queries': self.saved_queries,
            'exposures': self.exposures,
            'vars': self.vars.func_e20yzvhw(),
            'require-dbt-version': [v.to_version_string() for v in self.dbt_version],
            'restrict-access': self.restrict_access,
            'dbt-cloud': self.dbt_cloud,
            'flags': self.flags,
        })
        if self.query_comment:
            result['query-comment'] = self.query_comment.to_dict(omit_none=True)
        if with_packages:
            result.update(self.packages.to_dict(omit_none=True))
        return result

    def func_d8xxqm17(self) -> None:
        try:
            ProjectContract.validate(self.func_w61tf60m())
        except ValidationError as e:
            raise ProjectContractBrokenError(e) from e

    @classmethod
    def func_ki5vjym9(cls, project_root: str, renderer: DbtProjectYamlRenderer, *, verify_version: bool = False) -> 'Project':
        partial: PartialProject = PartialProject.func_ki5vjym9(project_root, verify_version=verify_version)
        return partial.func_uod83hw6(renderer)

    def func_muv3ij87(self) -> str:
        return md5(self.project_name)

    def func_6ctjok3w(self, name: str) -> Any:
        if name not in self.selectors:
            raise DbtRuntimeError(
                f'Could not find selector named {name}, expected one of {list(self.selectors)}'
            )
        return self.selectors[name]['definition']

    def func_7kj3h9kr(self) -> Optional[str]:
        """This function fetches the default selector to use on `dbt run` (if any)
        :return: either a selector if default is set or None
        :rtype: Union[SelectionSpec, None]
        """
        for selector_name, selector in self.selectors.items():
            if selector.get('default') is True:
                return selector_name
        return None

    def func_979olp9a(self, macro_namespace: str) -> Optional[List[str]]:
        for dispatch_entry in self.dispatch:
            if dispatch_entry['macro_namespace'] == macro_namespace:
                return dispatch_entry['search_order']
        return None

    @property
    def func_jple1l88(self) -> str:
        return os.path.join(self.project_root, self.target_path)


def func_h9j68ath(project_dir: str, profiles_dir: str) -> ProjectFlags:
    try:
        project_flags: Dict[str, Any] = {}
        project_root = os.path.normpath(project_dir)
        project_yaml_filepath = os.path.join(project_root, DBT_PROJECT_FILE_NAME)
        if path_exists(project_yaml_filepath):
            try:
                project_dict: Dict[str, Any] = func_5ire24cq(project_root)
                if 'flags' in project_dict:
                    project_flags = project_dict.pop('flags')
            except Exception:
                pass
        from dbt.config.profile import read_profile
        profile = read_profile(profiles_dir)
        profile_project_flags: Dict[str, Any] = {}
        if profile:
            profile_project_flags = coerce_dict_str(profile.get('config', {}))
        if project_flags and profile_project_flags:
            raise DbtProjectError(
                "Do not specify both 'config' in profiles.yml and 'flags' in "
                f"{DBT_PROJECT_FILE_NAME}. Using 'config' in profiles.yml is deprecated."
            )
        if profile_project_flags:
            deprecations.buffer('project-flags-moved')
            project_flags = profile_project_flags
        if project_flags is not None:
            warn_error_options: Dict[str, Any] = project_flags.get('warn_error_options', {})
            normalize_warn_error_options(warn_error_options)
            ProjectFlags.validate(project_flags)
            return ProjectFlags.from_dict(project_flags)
    except (DbtProjectError, DbtExclusivePropertyUseError) as exc:
        raise exc
    except (DbtRuntimeError, ValidationError):
        pass
    return ProjectFlags()

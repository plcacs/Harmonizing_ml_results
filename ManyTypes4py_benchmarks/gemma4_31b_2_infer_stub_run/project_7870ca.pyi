import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, Protocol, runtime_checkable
from dbt.adapters.contracts.connection import QueryComment
from dbt.config.selectors import SelectorDict
from dbt.contracts.project import PackageConfig, Project as ProjectContract, ProjectFlags, ProjectPackageMetadata, SemverString
from dbt_common.semver import VersionSpecifier

T = TypeVar('T')

INVALID_VERSION_ERROR: str = "This version of dbt is not supported with the '{package}' package.\n  Installed version of dbt: {installed}\n  Required version of dbt for '{package}': {version_spec}\nCheck for a different version of the '{package}' package, or run dbt again with --no-version-check\n"
IMPOSSIBLE_VERSION_ERROR: str = "The package version requirement can never be satisfied for the '{package}\npackage.\n  Required versions of dbt for '{package}': {version_spec}\nCheck for a different version of the '{package}' package, or run dbt again with --no-version-check\n"
MALFORMED_PACKAGE_ERROR: str = 'The packages.yml file in this project is malformed. Please double check\nthe contents of this file and fix any errors before retrying.\n\nYou can find more information on the syntax for this file here:\nhttps://docs.getdbt.com/docs/package-management\n\nValidator Error:\n{error}\n'
MISSING_DBT_PROJECT_ERROR: str = 'No {DBT_PROJECT_FILE_NAME} found at expected path {path}\nVerify that each entry within packages.yml (and their transitive dependencies) contains a file named {DBT_PROJECT_FILE_NAME}\n'

@runtime_checkable
class IsFQNResource(Protocol):
    ...

def _load_yaml(path: str) -> Any: ...

def load_yml_dict(file_path: str) -> Dict[str, Any]: ...

def package_and_project_data_from_root(project_root: str) -> Tuple[Dict[str, Any], str]: ...

def package_config_from_data(packages_data: Dict[str, Any], unrendered_packages_data: Optional[Dict[str, Any]] = None) -> PackageConfig: ...

def _parse_versions(versions: Union[str, List[str]]) -> List[VersionSpecifier]: ...

def _all_source_paths(*args: List[str]) -> List[str]: ...

def flag_or(flag: Optional[T], value: T, default: T) -> T: ...

def value_or(value: Optional[T], default: T) -> T: ...

def load_raw_project(project_root: str) -> Dict[str, Any]: ...

def _query_comment_from_cfg(cfg_query_comment: Any) -> QueryComment: ...

def validate_version(dbt_version: List[VersionSpecifier], project_name: str) -> None: ...

def _get_required_version(project_dict: Dict[str, Any], verify_version: bool) -> List[VersionSpecifier]: ...

@dataclass
class RenderComponents:
    project_dict: Dict[str, Any]
    packages_dict: Dict[str, Any]
    selectors_dict: Dict[str, Any]

@dataclass
class PartialProject(RenderComponents):
    profile_name: Optional[str]
    project_name: str
    project_root: str
    verify_version: bool
    packages_specified_path: str

    def render_profile_name(self, renderer: Any) -> Optional[str]: ...
    def get_rendered(self, renderer: Any) -> RenderComponents: ...
    def render(self, renderer: Any) -> 'Project': ...
    def render_package_metadata(self, renderer: Any) -> ProjectPackageMetadata: ...
    def check_config_path(self, project_dict: Dict[str, Any], deprecated_path: str, expected_path: Optional[str] = None, default_value: Optional[Any] = None) -> None: ...
    def create_project(self, rendered: RenderComponents) -> 'Project': ...

    @classmethod
    def from_dicts(cls, project_root: str, project_dict: Dict[str, Any], packages_dict: Dict[str, Any], selectors_dict: Dict[str, Any], *, verify_version: bool = False, packages_specified_path: str = 'packages.yml') -> 'PartialProject': ...
    @classmethod
    def from_project_root(cls, project_root: str, *, verify_version: bool = False) -> 'PartialProject': ...

class VarProvider:
    def __init__(self, vars: Dict[str, Any]) -> None: ...
    def vars_for(self, node: Any, adapter_type: str) -> Any: ...
    def to_dict(self) -> Dict[str, Any]: ...

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
    packages_install_path: str
    packages_specified_path: str
    quoting: Dict[str, Any]
    models: Any
    on_run_start: List[Any]
    on_run_end: List[Any]
    dispatch: List[Dict[str, Any]]
    seeds: Any
    snapshots: Any
    dbt_version: List[VersionSpecifier]
    packages: PackageConfig
    manifest_selectors: SelectorDict
    selectors: Dict[str, Any]
    query_comment: QueryComment
    sources: Any
    data_tests: Any
    unit_tests: Any
    metrics: Any
    semantic_models: Any
    saved_queries: Any
    exposures: Any
    vars: VarProvider
    config_version: str
    unrendered: RenderComponents
    project_env_vars: Dict[str, Any]
    restrict_access: bool
    dbt_cloud: Any
    flags: ProjectFlags

    @property
    def all_source_paths(self) -> List[str]: ...
    @property
    def generic_test_paths(self) -> List[str]: ...
    @property
    def fixture_paths(self) -> List[str]: ...
    @property
    def project_target_path(self) -> str: ...

    def __str__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def to_project_config(self, with_packages: bool = False) -> Dict[str, Any]: ...
    def validate(self) -> None: ...
    def hashed_name(self) -> str: ...
    def get_selector(self, name: str) -> Any: ...
    def get_default_selector_name(self) -> Optional[str]: ...
    def get_macro_search_order(self, macro_namespace: str) -> Optional[List[str]]: ...

    @classmethod
    def from_project_root(cls, project_root: str, renderer: Any, *, verify_version: bool = False) -> 'Project': ...

def read_project_flags(project_dir: str, profiles_dir: str) -> ProjectFlags: ...
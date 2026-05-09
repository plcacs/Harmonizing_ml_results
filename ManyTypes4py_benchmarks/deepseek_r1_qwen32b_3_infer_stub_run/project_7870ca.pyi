from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable, TypeVar
from typing_extensions import Protocol
from dbt.contracts.project import PackageConfig, ProjectContract, ProjectFlags, SemverString
from dbt.exceptions import DbtProjectError
from dbt_common.semver import VersionSpecifier
from dbt_common.helper_types import NoValue
from dbt.adapters.contracts.connection import QueryComment
from dbt.contracts.project import ProjectPackageMetadata

@runtime_checkable
class IsFQNResource(Protocol):
    pass

def _load_yaml(path: str) -> dict:
    ...

def load_yml_dict(file_path: str) -> dict:
    ...

def package_and_project_data_from_root(project_root: str) -> Tuple[dict, str]:
    ...

def package_config_from_data(packages_data: dict, unrendered_packages_data: Optional[dict] = None) -> PackageConfig:
    ...

def _parse_versions(versions: Union[str, List[str]]) -> List[VersionSpecifier]:
    ...

def _all_source_paths(*args: Iterable[str]) -> List[str]:
    ...

T = TypeVar('T')

def flag_or(flag: T, value: T, default: T) -> T:
    ...

def value_or(value: T, default: T) -> T:
    ...

def load_raw_project(project_root: str) -> dict:
    ...

def _query_comment_from_cfg(cfg_query_comment: Union[str, NoValue, QueryComment]) -> QueryComment:
    ...

def validate_version(dbt_version: List[VersionSpecifier], project_name: str) -> None:
    ...

def _get_required_version(project_dict: dict, verify_version: bool) -> List[VersionSpecifier]:
    ...

@dataclass
class RenderComponents:
    project_dict: dict
    packages_dict: dict
    selectors_dict: dict

@dataclass
class PartialProject(RenderComponents):
    profile_name: Optional[str]
    project_name: Optional[str]
    project_root: str
    verify_version: bool
    packages_specified_path: str

    def render_profile_name(self, renderer: Any) -> Optional[str]:
        ...

    def get_rendered(self, renderer: Any) -> RenderComponents:
        ...

    def render(self, renderer: Any) -> Project:
        ...

    def render_package_metadata(self, renderer: Any) -> ProjectPackageMetadata:
        ...

    def check_config_path(self, project_dict: dict, deprecated_path: str, expected_path: Optional[str] = None, default_value: Optional[Any] = None) -> None:
        ...

    def create_project(self, rendered: RenderComponents) -> Project:
        ...

    @classmethod
    def from_dicts(cls, project_root: str, project_dict: dict, packages_dict: dict, selectors_dict: dict, verify_version: bool = False, packages_specified_path: str = PACKAGES_FILE_NAME) -> PartialProject:
        ...

    @classmethod
    def from_project_root(cls, project_root: str, verify_version: bool = False) -> PartialProject:
        ...

@dataclass
class Project:
    project_name: str
    version: SemverString
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
    models: Dict[str, Any]
    on_run_start: List[str]
    on_run_end: List[str]
    dispatch: List[Dict[str, Any]]
    seeds: Dict[str, Any]
    snapshots: Dict[str, Any]
    dbt_version: List[VersionSpecifier]
    packages: PackageConfig
    manifest_selectors: Dict[str, Any]
    selectors: SelectorConfig
    query_comment: QueryComment
    sources: Dict[str, Any]
    data_tests: Dict[str, Any]
    unit_tests: Dict[str, Any]
    metrics: Dict[str, Any]
    semantic_models: Dict[str, Any]
    saved_queries: Dict[str, Any]
    exposures: Dict[str, Any]
    vars: VarProvider
    config_version: int
    unrendered: RenderComponents
    project_env_vars: Dict[str, str]
    restrict_access: bool
    dbt_cloud: Dict[str, Any]
    flags: ProjectFlags

    @property
    def all_source_paths(self) -> List[str]:
        ...

    @property
    def generic_test_paths(self) -> List[str]:
        ...

    @property
    def fixture_paths(self) -> List[str]:
        ...

    def __str__(self) -> str:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def to_project_config(self, with_packages: bool = False) -> dict:
        ...

    def validate(self) -> None:
        ...

    @classmethod
    def from_project_root(cls, project_root: str, renderer: Any, verify_version: bool = False) -> Project:
        ...

    def hashed_name(self) -> str:
        ...

    def get_selector(self, name: str) -> Dict[str, Any]:
        ...

    def get_default_selector_name(self) -> Optional[str]:
        ...

    def get_macro_search_order(self, macro_namespace: str) -> Optional[List[str]]:
        ...

    @property
    def project_target_path(self) -> str:
        ...

def read_project_flags(project_dir: str, profiles_dir: str) -> ProjectFlags:
    ...
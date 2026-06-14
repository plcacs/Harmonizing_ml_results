import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, TypeVar, Union

from typing_extensions import Protocol, runtime_checkable

from dbt.adapters.contracts.connection import QueryComment
from dbt.config.selectors import SelectorDict
from dbt.contracts.project import PackageConfig
from dbt.contracts.project import Project as ProjectContract
from dbt.contracts.project import ProjectFlags, ProjectPackageMetadata, SemverString
from dbt.graph import SelectionSpec
from dbt.node_types import NodeType
from dbt.utils import MultiDict
from dbt_common.helper_types import NoValue
from dbt_common.semver import VersionSpecifier

from .renderer import DbtProjectYamlRenderer, PackageRenderer
from .selectors import SelectorConfig

INVALID_VERSION_ERROR: str
IMPOSSIBLE_VERSION_ERROR: str
MALFORMED_PACKAGE_ERROR: str
MISSING_DBT_PROJECT_ERROR: str

@runtime_checkable
class IsFQNResource(Protocol): ...

def _load_yaml(path: str) -> Any: ...
def load_yml_dict(file_path: str) -> Dict[str, Any]: ...
def package_and_project_data_from_root(project_root: str) -> tuple[Dict[str, Any], str]: ...
def package_config_from_data(
    packages_data: Optional[Dict[str, Any]],
    unrendered_packages_data: Optional[Dict[str, Any]] = ...,
) -> PackageConfig: ...
def _parse_versions(versions: Union[str, List[str]]) -> List[VersionSpecifier]: ...
def _all_source_paths(*args: List[str]) -> List[str]: ...

T = TypeVar("T")

def flag_or(flag: Optional[T], value: Optional[T], default: T) -> T: ...
def value_or(value: Optional[T], default: T) -> T: ...
def load_raw_project(project_root: str) -> Dict[str, Any]: ...
def _query_comment_from_cfg(
    cfg_query_comment: Optional[Union[str, NoValue, QueryComment]],
) -> QueryComment: ...
def validate_version(dbt_version: List[VersionSpecifier], project_name: str) -> None: ...
def _get_required_version(
    project_dict: Dict[str, Any], verify_version: bool
) -> List[VersionSpecifier]: ...

@dataclass
class RenderComponents:
    project_dict: Dict[str, Any]
    packages_dict: Dict[str, Any]
    selectors_dict: Dict[str, Any]

@dataclass
class PartialProject(RenderComponents):
    profile_name: Optional[str]
    project_name: Optional[str]
    project_root: str
    verify_version: bool
    packages_specified_path: str

    def render_profile_name(self, renderer: DbtProjectYamlRenderer) -> Optional[str]: ...
    def get_rendered(self, renderer: DbtProjectYamlRenderer) -> RenderComponents: ...
    def render(self, renderer: DbtProjectYamlRenderer) -> Project: ...
    def render_package_metadata(self, renderer: DbtProjectYamlRenderer) -> ProjectPackageMetadata: ...
    def check_config_path(
        self,
        project_dict: Dict[str, Any],
        deprecated_path: str,
        expected_path: Optional[str] = ...,
        default_value: Optional[Any] = ...,
    ) -> None: ...
    def create_project(self, rendered: RenderComponents) -> Project: ...
    @classmethod
    def from_dicts(
        cls,
        project_root: str,
        project_dict: Dict[str, Any],
        packages_dict: Dict[str, Any],
        selectors_dict: Dict[str, Any],
        *,
        verify_version: bool = ...,
        packages_specified_path: str = ...,
    ) -> PartialProject: ...
    @classmethod
    def from_project_root(
        cls, project_root: str, *, verify_version: bool = ...
    ) -> PartialProject: ...

class VarProvider:
    vars: Dict[str, Any]
    def __init__(self, vars: Dict[str, Any]) -> None: ...
    def vars_for(self, node: Any, adapter_type: str) -> MultiDict: ...
    def to_dict(self) -> Dict[str, Any]: ...

@dataclass
class Project:
    project_name: str
    version: Optional[Union[str, SemverString]]
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
    project_env_vars: Dict[str, Any]
    restrict_access: bool
    dbt_cloud: Dict[str, Any]
    flags: ProjectFlags

    @property
    def all_source_paths(self) -> List[str]: ...
    @property
    def generic_test_paths(self) -> List[str]: ...
    @property
    def fixture_paths(self) -> List[str]: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def to_project_config(self, with_packages: bool = ...) -> Dict[str, Any]: ...
    def validate(self) -> None: ...
    @classmethod
    def from_project_root(
        cls,
        project_root: str,
        renderer: DbtProjectYamlRenderer,
        *,
        verify_version: bool = ...,
    ) -> Project: ...
    def hashed_name(self) -> str: ...
    def get_selector(self, name: str) -> SelectionSpec: ...
    def get_default_selector_name(self) -> Optional[str]: ...
    def get_macro_search_order(self, macro_namespace: str) -> Optional[List[str]]: ...
    @property
    def project_target_path(self) -> str: ...

def read_project_flags(project_dir: str, profiles_dir: str) -> ProjectFlags: ...
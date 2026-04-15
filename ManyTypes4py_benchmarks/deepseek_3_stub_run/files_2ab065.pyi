import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from mashumaro.types import SerializableType
from dbt.artifacts.resources.base import FileHash
from dbt.constants import MAXIMUM_SEED_SIZE
from dbt_common.dataclass_schema import StrEnum, dbtClassMixin
from .util import SourceKey

class ParseFileType(StrEnum):
    Macro: str = ...
    Model: str = ...
    Snapshot: str = ...
    Analysis: str = ...
    SingularTest: str = ...
    GenericTest: str = ...
    Seed: str = ...
    Documentation: str = ...
    Schema: str = ...
    Hook: str = ...
    Fixture: str = ...

parse_file_type_to_parser: Dict[ParseFileType, str] = ...

@dataclass
class FilePath(dbtClassMixin):
    project_root: str
    searched_path: str
    relative_path: str

    @property
    def search_key(self) -> str: ...

    @property
    def full_path(self) -> str: ...

    @property
    def absolute_path(self) -> str: ...

    @property
    def original_file_path(self) -> str: ...

    def seed_too_large(self) -> bool: ...

@dataclass
class RemoteFile(dbtClassMixin):
    path: str
    path_end: str

    def __init__(self, language: str) -> None: ...

    @property
    def searched_path(self) -> str: ...

    @property
    def relative_path(self) -> str: ...

    @property
    def absolute_path(self) -> str: ...

    @property
    def original_file_path(self) -> str: ...

    @property
    def modification_time(self) -> str: ...

@dataclass
class BaseSourceFile(dbtClassMixin, SerializableType):
    path: Union[FilePath, RemoteFile]
    checksum: FileHash
    project_name: Optional[str] = None
    parse_file_type: Optional[str] = None
    contents: Optional[str] = None

    @property
    def file_id(self) -> Optional[str]: ...

    @property
    def original_file_path(self) -> str: ...

    def _serialize(self) -> Dict[str, Any]: ...

    @classmethod
    def _deserialize(cls, dct: Dict[str, Any]) -> "BaseSourceFile": ...

    def __post_serialize__(self, dct: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]: ...

@dataclass
class SourceFile(BaseSourceFile):
    nodes: List[Any] = field(default_factory=list)
    docs: List[Any] = field(default_factory=list)
    macros: List[Any] = field(default_factory=list)
    env_vars: List[Any] = field(default_factory=list)

    @classmethod
    def big_seed(cls, path: FilePath) -> "SourceFile": ...

    def add_node(self, value: Any) -> None: ...

    @classmethod
    def remote(cls, contents: str, project_name: str, language: str) -> "SourceFile": ...

@dataclass
class SchemaSourceFile(BaseSourceFile):
    dfy: Dict[str, Any] = field(default_factory=dict)
    data_tests: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    sources: List[Any] = field(default_factory=list)
    exposures: List[Any] = field(default_factory=list)
    metrics: List[Any] = field(default_factory=list)
    snapshots: List[Any] = field(default_factory=list)
    generated_metrics: List[str] = field(default_factory=list)
    metrics_from_measures: Dict[str, List[str]] = field(default_factory=dict)
    groups: List[Any] = field(default_factory=list)
    ndp: List[str] = field(default_factory=list)
    semantic_models: List[Any] = field(default_factory=list)
    unit_tests: List[Any] = field(default_factory=list)
    saved_queries: List[Any] = field(default_factory=list)
    mcp: Dict[str, Any] = field(default_factory=dict)
    sop: List[Any] = field(default_factory=list)
    env_vars: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    unrendered_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unrendered_databases: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unrendered_schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pp_dict: Optional[Dict[str, Any]] = None
    pp_test_index: Optional[Any] = None

    @property
    def dict_from_yaml(self) -> Dict[str, Any]: ...

    @property
    def node_patches(self) -> List[str]: ...

    @property
    def macro_patches(self) -> Dict[str, Any]: ...

    @property
    def source_patches(self) -> List[Any]: ...

    def __post_serialize__(self, dct: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]: ...

    def append_patch(self, yaml_key: str, unique_id: str) -> None: ...

    def add_test(self, node_unique_id: str, test_from: Dict[str, Any]) -> None: ...

    def remove_tests(self, yaml_key: str, name: str) -> None: ...

    def get_tests(self, yaml_key: str, name: str) -> List[str]: ...

    def add_metrics_from_measures(self, semantic_model_name: str, metric_unique_id: str) -> None: ...

    def fix_metrics_from_measures(self) -> None: ...

    def get_key_and_name_for_test(self, test_unique_id: str) -> tuple[Optional[str], Optional[str]]: ...

    def get_all_test_ids(self) -> List[str]: ...

    def add_unrendered_config(self, unrendered_config: Any, yaml_key: str, name: str, version: Optional[int] = None) -> None: ...

    def get_unrendered_config(self, yaml_key: str, name: str, version: Optional[int] = None) -> Optional[Any]: ...

    def delete_from_unrendered_configs(self, yaml_key: str, name: str) -> None: ...

    def add_env_var(self, var: str, yaml_key: str, name: str) -> None: ...

    def delete_from_env_vars(self, yaml_key: str, name: str) -> None: ...

    def add_unrendered_database(self, yaml_key: str, name: str, unrendered_database: Any) -> None: ...

    def get_unrendered_database(self, yaml_key: str, name: str) -> Optional[Any]: ...

    def add_unrendered_schema(self, yaml_key: str, name: str, unrendered_schema: Any) -> None: ...

    def get_unrendered_schema(self, yaml_key: str, name: str) -> Optional[Any]: ...

@dataclass
class FixtureSourceFile(BaseSourceFile):
    fixture: Optional[Any] = None
    unit_tests: List[Any] = field(default_factory=list)

    def add_unit_test(self, value: Any) -> None: ...

AnySourceFile = Union[SchemaSourceFile, SourceFile, FixtureSourceFile]
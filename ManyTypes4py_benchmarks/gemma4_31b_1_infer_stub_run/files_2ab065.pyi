import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple, overload
from mashumaro.types import SerializableType
from dbt.artifacts.resources.base import FileHash
from dbt_common.dataclass_schema import StrEnum, dbtClassMixin
from .util import SourceKey

class ParseFileType(StrEnum):
    Macro = 'macro'
    Model = 'model'
    Snapshot = 'snapshot'
    Analysis = 'analysis'
    SingularTest = 'singular_test'
    GenericTest = 'generic_test'
    Seed = 'seed'
    Documentation = 'docs'
    Schema = 'schema'
    Hook = 'hook'
    Fixture = 'fixture'

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
    project_name: Optional[str]
    parse_file_type: Optional[ParseFileType]
    contents: Optional[str]
    path: Union[FilePath, RemoteFile]
    checksum: Optional[str]

    @property
    def file_id(self) -> Optional[str]: ...
    @property
    def original_file_path(self) -> str: ...
    def _serialize(self) -> Dict[str, Any]: ...
    @classmethod
    def _deserialize(cls, dct: Dict[str, Any]) -> Union['SourceFile', 'SchemaSourceFile', 'FixtureSourceFile']: ...
    def __post_serialize__(self, dct: Dict[str, Any], context: Any = None) -> Dict[str, Any]: ...

@dataclass
class SourceFile(BaseSourceFile):
    nodes: List[Any]
    docs: List[Any]
    macros: List[Any]
    env_vars: List[Any]

    @classmethod
    def big_seed(cls, path: FilePath) -> 'SourceFile': ...
    def add_node(self, value: Any) -> None: ...
    @classmethod
    def remote(cls, contents: str, project_name: str, language: str) -> 'SourceFile': ...

@dataclass
class SchemaSourceFile(BaseSourceFile):
    dfy: Dict[str, Any]
    data_tests: Dict[str, Dict[str, List[str]]]
    sources: List[Any]
    exposures: List[Any]
    metrics: List[Any]
    snapshots: List[Any]
    generated_metrics: List[str]
    metrics_from_measures: Dict[str, List[str]]
    groups: List[Any]
    ndp: List[str]
    semantic_models: List[Any]
    unit_tests: List[Any]
    saved_queries: List[Any]
    mcp: Dict[str, Any]
    sop: List[str]
    env_vars: Dict[str, Dict[str, List[str]]]
    unrendered_configs: Dict[str, Dict[str, Any]]
    unrendered_databases: Dict[str, Dict[str, Any]]
    unrendered_schemas: Dict[str, Dict[str, Any]]
    pp_dict: Optional[Any]
    pp_test_index: Optional[Any]

    @property
    def dict_from_yaml(self) -> Dict[str, Any]: ...
    @property
    def node_patches(self) -> List[str]: ...
    @property
    def macro_patches(self) -> Dict[str, Any]: ...
    @property
    def source_patches(self) -> List[str]: ...
    def __post_serialize__(self, dct: Dict[str, Any], context: Any = None) -> Dict[str, Any]: ...
    def append_patch(self, yaml_key: str, unique_id: str) -> None: ...
    def add_test(self, node_unique_id: str, test_from: Dict[str, str]) -> None: ...
    def remove_tests(self, yaml_key: str, name: str) -> None: ...
    def get_tests(self, yaml_key: str, name: str) -> List[str]: ...
    def add_metrics_from_measures(self, semantic_model_name: str, metric_unique_id: str) -> None: ...
    def fix_metrics_from_measures(self) -> None: ...
    def get_key_and_name_for_test(self, test_unique_id: str) -> Tuple[Optional[str], Optional[str]]: ...
    def get_all_test_ids(self) -> List[str]: ...
    def add_unrendered_config(self, unrendered_config: Any, yaml_key: str, name: str, version: Optional[Union[int, str]] = None) -> None: ...
    def get_unrendered_config(self, yaml_key: str, name: str, version: Optional[Union[int, str]] = None) -> Optional[Any]: ...
    def delete_from_unrendered_configs(self, yaml_key: str, name: str) -> None: ...
    def add_env_var(self, var: str, yaml_key: str, name: str) -> None: ...
    def delete_from_env_vars(self, yaml_key: str, name: str) -> None: ...
    def add_unrendered_database(self, yaml_key: str, name: str, unrendered_database: Any) -> None: ...
    def get_unrendered_database(self, yaml_key: str, name: str) -> Optional[Any]: ...
    def add_unrendered_schema(self, yaml_key: str, name: str, unrendered_schema: Any) -> None: ...
    def get_unrendered_schema(self, yaml_key: str, name: str) -> Optional[Any]: ...

@dataclass
class FixtureSourceFile(BaseSourceFile):
    fixture: Optional[Any]
    unit_tests: List[Any]

    def add_unit_test(self, value: Any) -> None: ...

AnySourceFile = Union[SchemaSourceFile, SourceFile, FixtureSourceFile]
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from mashumaro.types import SerializableType
from dbt.artifacts.resources.base import FileHash
from dbt_common.dataclass_schema import StrEnum, dbtClassMixin

from .util import SourceKey


class ParseFileType(StrEnum):
    Macro: str
    Model: str
    Snapshot: str
    Analysis: str
    SingularTest: str
    GenericTest: str
    Seed: str
    Documentation: str
    Schema: str
    Hook: str
    Fixture: str

parse_file_type_to_parser: Dict[ParseFileType, str]

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
    project_name: Optional[str]
    parse_file_type: Optional[str]
    contents: Optional[str]

    @property
    def file_id(self) -> Optional[str]: ...
    @property
    def original_file_path(self) -> str: ...
    def _serialize(self) -> Dict[str, Any]: ...
    @classmethod
    def _deserialize(cls, dct: Dict[str, Any]) -> Union[SchemaSourceFile, FixtureSourceFile, SourceFile]: ...
    def __post_serialize__(self, dct: Dict[str, Any], context: Optional[Any] = ...) -> Dict[str, Any]: ...

@dataclass
class SourceFile(BaseSourceFile):
    nodes: List[str]
    docs: List[str]
    macros: List[str]
    env_vars: List[str]

    @classmethod
    def big_seed(cls, path: FilePath) -> SourceFile: ...
    def add_node(self, value: str) -> None: ...
    @classmethod
    def remote(cls, contents: str, project_name: str, language: str) -> SourceFile: ...

@dataclass
class SchemaSourceFile(BaseSourceFile):
    dfy: Dict[str, Any]
    data_tests: Dict[str, Dict[str, List[str]]]
    sources: List[str]
    exposures: List[str]
    metrics: List[str]
    snapshots: List[str]
    generated_metrics: List[str]
    metrics_from_measures: Dict[str, List[str]]
    groups: List[str]
    ndp: List[str]
    semantic_models: List[str]
    unit_tests: List[str]
    saved_queries: List[str]
    mcp: Dict[str, str]
    sop: List[str]
    env_vars: Dict[str, Dict[str, List[str]]]
    unrendered_configs: Dict[str, Dict[str, Any]]
    unrendered_databases: Dict[str, Dict[str, Any]]
    unrendered_schemas: Dict[str, Dict[str, Any]]
    pp_dict: Optional[Dict[str, Any]]
    pp_test_index: Optional[Dict[str, Any]]

    @property
    def dict_from_yaml(self) -> Dict[str, Any]: ...
    @property
    def node_patches(self) -> List[str]: ...
    @property
    def macro_patches(self) -> Dict[str, str]: ...
    @property
    def source_patches(self) -> List[str]: ...
    def __post_serialize__(self, dct: Dict[str, Any], context: Optional[Any] = ...) -> Dict[str, Any]: ...
    def append_patch(self, yaml_key: str, unique_id: str) -> None: ...
    def add_test(self, node_unique_id: str, test_from: Dict[str, str]) -> None: ...
    def remove_tests(self, yaml_key: str, name: str) -> None: ...
    def get_tests(self, yaml_key: str, name: str) -> List[str]: ...
    def add_metrics_from_measures(self, semantic_model_name: str, metric_unique_id: str) -> None: ...
    def fix_metrics_from_measures(self) -> None: ...
    def get_key_and_name_for_test(self, test_unique_id: str) -> Tuple[Optional[str], Optional[str]]: ...
    def get_all_test_ids(self) -> List[str]: ...
    def add_unrendered_config(self, unrendered_config: Any, yaml_key: str, name: str, version: Optional[Any] = ...) -> None: ...
    def get_unrendered_config(self, yaml_key: str, name: str, version: Optional[Any] = ...) -> Optional[Any]: ...
    def delete_from_unrendered_configs(self, yaml_key: str, name: str) -> None: ...
    def add_env_var(self, var: str, yaml_key: str, name: str) -> None: ...
    def delete_from_env_vars(self, yaml_key: str, name: str) -> None: ...
    def add_unrendered_database(self, yaml_key: str, name: str, unrendered_database: Any) -> None: ...
    def get_unrendered_database(self, yaml_key: str, name: str) -> Optional[Any]: ...
    def add_unrendered_schema(self, yaml_key: str, name: str, unrendered_schema: Any) -> None: ...
    def get_unrendered_schema(self, yaml_key: str, name: str) -> Optional[Any]: ...

@dataclass
class FixtureSourceFile(BaseSourceFile):
    fixture: Optional[str]
    unit_tests: List[str]

    def add_unit_test(self, value: str) -> None: ...

AnySourceFile = Union[SchemaSourceFile, SourceFile, FixtureSourceFile]
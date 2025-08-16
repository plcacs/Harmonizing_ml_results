from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from mashumaro.types import SerializableType
from dbt.artifacts.resources.base import FileHash
from dbt.constants import MAXIMUM_SEED_SIZE
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
    def search_key(self) -> str:
        return self.absolute_path

    @property
    def full_path(self) -> str:
        return os.path.join(self.project_root, self.searched_path, self.relative_path)

    @property
    def absolute_path(self) -> str:
        return os.path.abspath(self.full_path)

    @property
    def original_file_path(self) -> str:
        return os.path.join(self.searched_path, self.relative_path)

    def seed_too_large(self) -> bool:
        return os.stat(self.full_path).st_size > MAXIMUM_SEED_SIZE

@dataclass
class RemoteFile(dbtClassMixin):
    path_end: str
    path: str

    def __init__(self, language: str):
        ...

    @property
    def searched_path(self) -> str:
        return self.path

    @property
    def relative_path(self) -> str:
        return self.path

    @property
    def absolute_path(self) -> str:
        return self.path

    @property
    def original_file_path(self) -> str:
        return self.path

    @property
    def modification_time(self) -> str:
        return self.path

@dataclass
class BaseSourceFile(dbtClassMixin, SerializableType):
    project_name: Optional[str]
    parse_file_type: Optional[ParseFileType]
    contents: Optional[str]

    @property
    def file_id(self) -> Optional[str]:
        ...

    @property
    def original_file_path(self) -> str:
        ...

    def _serialize(self) -> Dict[str, Any]:
        ...

    @classmethod
    def _deserialize(cls, dct: Dict[str, Any]) -> 'BaseSourceFile':
        ...

    def __post_serialize__(self, dct: Dict[str, Any], context=None) -> Dict[str, Any]:
        ...

@dataclass
class SourceFile(BaseSourceFile):
    nodes: List[str]
    docs: List[str]
    macros: List[str]
    env_vars: List[str]

    @classmethod
    def big_seed(cls, path: FilePath) -> 'SourceFile':
        ...

    def add_node(self, value: str):
        ...

    @classmethod
    def remote(cls, contents: str, project_name: str, language: str) -> 'SourceFile':
        ...

@dataclass
class SchemaSourceFile(BaseSourceFile):
    dfy: Dict[str, Any]
    data_tests: Dict[str, Any]
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
    mcp: Dict[str, Any]
    sop: List[str]
    env_vars: Dict[str, Any]
    unrendered_configs: Dict[str, Any]
    unrendered_databases: Dict[str, Any]
    unrendered_schemas: Dict[str, Any]
    pp_dict: Optional[Any]
    pp_test_index: Optional[Any]

    @property
    def dict_from_yaml(self) -> Dict[str, Any]:
        ...

    @property
    def node_patches(self) -> List[str]:
        ...

    @property
    def macro_patches(self) -> Dict[str, Any]:
        ...

    @property
    def source_patches(self) -> List[str]:
        ...

    def append_patch(self, yaml_key: str, unique_id: str):
        ...

    def add_test(self, node_unique_id: str, test_from: Dict[str, Any]):
        ...

    def remove_tests(self, yaml_key: str, name: str):
        ...

    def get_tests(self, yaml_key: str, name: str) -> List[str]:
        ...

    def add_metrics_from_measures(self, semantic_model_name: str, metric_unique_id: str):
        ...

    def fix_metrics_from_measures(self):
        ...

    def get_key_and_name_for_test(self, test_unique_id: str) -> Tuple[str, str]:
        ...

    def get_all_test_ids(self) -> List[str]:
        ...

    def add_unrendered_config(self, unrendered_config: Any, yaml_key: str, name: str, version: Optional[int] = None):
        ...

    def get_unrendered_config(self, yaml_key: str, name: str, version: Optional[int] = None) -> Any:
        ...

    def delete_from_unrendered_configs(self, yaml_key: str, name: str):
        ...

    def add_env_var(self, var: str, yaml_key: str, name: str):
        ...

    def delete_from_env_vars(self, yaml_key: str, name: str):
        ...

    def add_unrendered_database(self, yaml_key: str, name: str, unrendered_database: Any):
        ...

    def get_unrendered_database(self, yaml_key: str, name: str) -> Any:
        ...

    def add_unrendered_schema(self, yaml_key: str, name: str, unrendered_schema: Any):
        ...

    def get_unrendered_schema(self, yaml_key: str, name: str) -> Any:
        ...

@dataclass
class FixtureSourceFile(BaseSourceFile):
    fixture: Optional[str]
    unit_tests: List[str]

    def add_unit_test(self, value: str):
        ...

AnySourceFile = Union[SchemaSourceFile, SourceFile, FixtureSourceFile]

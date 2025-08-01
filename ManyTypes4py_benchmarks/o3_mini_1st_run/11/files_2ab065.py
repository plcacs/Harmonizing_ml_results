import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from mashumaro.types import SerializableType
from dbt.artifacts.resources.base import FileHash
from dbt.constants import MAXIMUM_SEED_SIZE
from dbt_common.dataclass_schema import StrEnum, dbtClassMixin
from .util import SourceKey

T = TypeVar("T", bound="BaseSourceFile")

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

parse_file_type_to_parser: Dict[ParseFileType, str] = {
    ParseFileType.Macro: 'MacroParser',
    ParseFileType.Model: 'ModelParser',
    ParseFileType.Snapshot: 'SnapshotParser',
    ParseFileType.Analysis: 'AnalysisParser',
    ParseFileType.SingularTest: 'SingularTestParser',
    ParseFileType.GenericTest: 'GenericTestParser',
    ParseFileType.Seed: 'SeedParser',
    ParseFileType.Documentation: 'DocumentationParser',
    ParseFileType.Schema: 'SchemaParser',
    ParseFileType.Hook: 'HookParser',
    ParseFileType.Fixture: 'FixtureParser'
}

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
        """Return whether the file this represents is over the seed size limit"""
        return os.stat(self.full_path).st_size > MAXIMUM_SEED_SIZE

@dataclass
class RemoteFile(dbtClassMixin):
    path: str = field(init=False)
    path_end: str = field(init=False)

    def __init__(self, language: str) -> None:
        if language == 'sql':
            self.path_end = '.sql'
        elif language == 'python':
            self.path_end = '.py'
        else:
            raise RuntimeError(f'Invalid language for remote File {language}')
        self.path = f'from remote system{self.path_end}'

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
    """Define a source file in dbt"""
    project_name: Optional[str] = None
    parse_file_type: Optional[str] = None
    contents: Optional[str] = None
    path: Union[FilePath, RemoteFile, None] = None
    checksum: Optional[FileHash] = None

    @property
    def file_id(self) -> Optional[str]:
        if isinstance(self.path, RemoteFile):
            return None
        # Assuming self.path is of type FilePath when not RemoteFile.
        return f'{self.project_name}://{self.path.original_file_path}'  # type: ignore

    @property
    def original_file_path(self) -> Optional[str]:
        return self.path.original_file_path if self.path is not None else None

    def _serialize(self) -> Dict[str, Any]:
        dct: Dict[str, Any] = self.to_dict()  # type: ignore
        return dct

    @classmethod
    def _deserialize(cls: Type[T], dct: Dict[str, Any]) -> T:
        if dct['parse_file_type'] == 'schema':
            sf = SchemaSourceFile.from_dict(dct)  # type: ignore
        elif dct['parse_file_type'] == 'fixture':
            sf = FixtureSourceFile.from_dict(dct)  # type: ignore
        else:
            sf = SourceFile.from_dict(dct)  # type: ignore
        return sf

    def __post_serialize__(self, dct: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        dct = super().__post_serialize__(dct, context)  # type: ignore
        dct_keys = list(dct.keys())
        for key in dct_keys:
            if isinstance(dct[key], list) and (not dct[key]):
                del dct[key]
        if 'contents' in dct:
            del dct['contents']
        return dct

@dataclass
class SourceFile(BaseSourceFile):
    nodes: List[Any] = field(default_factory=list)
    docs: List[Any] = field(default_factory=list)
    macros: List[Any] = field(default_factory=list)
    env_vars: List[Any] = field(default_factory=list)

    @classmethod
    def big_seed(cls: Type["SourceFile"], path: FilePath) -> "SourceFile":
        """Parse seeds over the size limit with just the path"""
        instance = cls(
            path=path,
            checksum=FileHash.path(path.original_file_path)
        )
        instance.contents = ''
        return instance

    def add_node(self, value: Any) -> None:
        if value not in self.nodes:
            self.nodes.append(value)

    @classmethod
    def remote(cls: Type["SourceFile"], contents: str, project_name: str, language: str) -> "SourceFile":
        instance = cls(
            path=RemoteFile(language),
            checksum=FileHash.from_contents(contents),
            project_name=project_name,
            contents=contents
        )
        return instance

@dataclass
class SchemaSourceFile(BaseSourceFile):
    dfy: Dict[Any, Any] = field(default_factory=dict)
    data_tests: Dict[Any, Any] = field(default_factory=dict)
    sources: List[Any] = field(default_factory=list)
    exposures: List[Any] = field(default_factory=list)
    metrics: List[Any] = field(default_factory=list)
    snapshots: List[Any] = field(default_factory=list)
    generated_metrics: List[Any] = field(default_factory=list)
    metrics_from_measures: Dict[Any, List[Any]] = field(default_factory=dict)
    groups: List[Any] = field(default_factory=list)
    ndp: List[Any] = field(default_factory=list)
    semantic_models: List[Any] = field(default_factory=list)
    unit_tests: List[Any] = field(default_factory=list)
    saved_queries: List[Any] = field(default_factory=list)
    mcp: Dict[Any, Any] = field(default_factory=dict)
    sop: List[Any] = field(default_factory=list)
    env_vars: Dict[Any, Any] = field(default_factory=dict)
    unrendered_configs: Dict[Any, Any] = field(default_factory=dict)
    unrendered_databases: Dict[Any, Any] = field(default_factory=dict)
    unrendered_schemas: Dict[Any, Any] = field(default_factory=dict)
    pp_dict: Optional[Any] = None
    pp_test_index: Optional[Any] = None

    @property
    def dict_from_yaml(self) -> Dict[Any, Any]:
        return self.dfy

    @property
    def node_patches(self) -> List[Any]:
        return self.ndp

    @property
    def macro_patches(self) -> Dict[Any, Any]:
        return self.mcp

    @property
    def source_patches(self) -> List[Any]:
        return self.sop

    def __post_serialize__(self, dct: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        dct = super().__post_serialize__(dct, context)  # type: ignore
        for key in ('pp_test_index', 'pp_dict'):
            if key in dct:
                del dct[key]
        return dct

    def append_patch(self, yaml_key: Any, unique_id: Any) -> None:
        self.node_patches.append(unique_id)

    def add_test(self, node_unique_id: Any, test_from: Dict[str, Any]) -> None:
        name: Any = test_from['name']
        key: Any = test_from['key']
        if key not in self.data_tests:
            self.data_tests[key] = {}
        if name not in self.data_tests[key]:
            self.data_tests[key][name] = []
        self.data_tests[key][name].append(node_unique_id)

    def remove_tests(self, yaml_key: Any, name: Any) -> None:
        if yaml_key in self.data_tests:
            if name in self.data_tests[yaml_key]:
                del self.data_tests[yaml_key][name]

    def get_tests(self, yaml_key: Any, name: Any) -> List[Any]:
        if yaml_key in self.data_tests:
            if name in self.data_tests[yaml_key]:
                return self.data_tests[yaml_key][name]
        return []

    def add_metrics_from_measures(self, semantic_model_name: Any, metric_unique_id: Any) -> None:
        if self.generated_metrics:
            self.fix_metrics_from_measures()
        if semantic_model_name not in self.metrics_from_measures:
            self.metrics_from_measures[semantic_model_name] = []
        self.metrics_from_measures[semantic_model_name].append(metric_unique_id)

    def fix_metrics_from_measures(self) -> None:
        generated_metrics = self.generated_metrics
        self.generated_metrics = []
        for metric_unique_id in generated_metrics:
            parts = metric_unique_id.split('.')
            metric_name = parts[-1]
            if 'semantic_models' in self.dict_from_yaml:
                for sem_model in self.dict_from_yaml['semantic_models']:
                    if 'measures' in sem_model:
                        for measure in sem_model['measures']:
                            if measure['name'] == metric_name:
                                self.add_metrics_from_measures(sem_model['name'], metric_unique_id)
                                break

    def get_key_and_name_for_test(self, test_unique_id: Any) -> (Optional[Any], Optional[Any]):
        yaml_key: Optional[Any] = None
        block_name: Optional[Any] = None
        for key in self.data_tests.keys():
            for name in self.data_tests[key]:
                for unique_id in self.data_tests[key][name]:
                    if unique_id == test_unique_id:
                        yaml_key = key
                        block_name = name
                        break
        return (yaml_key, block_name)

    def get_all_test_ids(self) -> List[Any]:
        test_ids: List[Any] = []
        for key in self.data_tests.keys():
            for name in self.data_tests[key]:
                test_ids.extend(self.data_tests[key][name])
        return test_ids

    def add_unrendered_config(self, unrendered_config: Any, yaml_key: Any, name: Any, version: Optional[Any] = None) -> None:
        versioned_name = f'{name}_v{version}' if version is not None else name
        if yaml_key not in self.unrendered_configs:
            self.unrendered_configs[yaml_key] = {}
        if versioned_name not in self.unrendered_configs[yaml_key]:
            self.unrendered_configs[yaml_key][versioned_name] = unrendered_config

    def get_unrendered_config(self, yaml_key: Any, name: Any, version: Optional[Any] = None) -> Optional[Any]:
        versioned_name = f'{name}_v{version}' if version is not None else name
        if yaml_key not in self.unrendered_configs:
            return None
        if versioned_name not in self.unrendered_configs[yaml_key]:
            return None
        return self.unrendered_configs[yaml_key][versioned_name]

    def delete_from_unrendered_configs(self, yaml_key: Any, name: Any) -> None:
        if self.get_unrendered_config(yaml_key, name):
            del self.unrendered_configs[yaml_key][name]
            version_names_to_delete: List[Any] = []
            for potential_version_name in self.unrendered_configs[yaml_key]:
                if potential_version_name.startswith(f'{name}_v'):
                    version_names_to_delete.append(potential_version_name)
            for version_name in version_names_to_delete:
                del self.unrendered_configs[yaml_key][version_name]
            if not self.unrendered_configs[yaml_key]:
                del self.unrendered_configs[yaml_key]

    def add_env_var(self, var: Any, yaml_key: Any, name: Any) -> None:
        if yaml_key not in self.env_vars:
            self.env_vars[yaml_key] = {}
        if name not in self.env_vars[yaml_key]:
            self.env_vars[yaml_key][name] = []
        if var not in self.env_vars[yaml_key][name]:
            self.env_vars[yaml_key][name].append(var)

    def delete_from_env_vars(self, yaml_key: Any, name: Any) -> None:
        if yaml_key in self.env_vars and name in self.env_vars[yaml_key]:
            del self.env_vars[yaml_key][name]
            if not self.env_vars[yaml_key]:
                del self.env_vars[yaml_key]

    def add_unrendered_database(self, yaml_key: Any, name: Any, unrendered_database: Any) -> None:
        if yaml_key not in self.unrendered_databases:
            self.unrendered_databases[yaml_key] = {}
        self.unrendered_databases[yaml_key][name] = unrendered_database

    def get_unrendered_database(self, yaml_key: Any, name: Any) -> Optional[Any]:
        if yaml_key not in self.unrendered_databases:
            return None
        return self.unrendered_databases[yaml_key].get(name)

    def add_unrendered_schema(self, yaml_key: Any, name: Any, unrendered_schema: Any) -> None:
        if yaml_key not in self.unrendered_schemas:
            self.unrendered_schemas[yaml_key] = {}
        self.unrendered_schemas[yaml_key][name] = unrendered_schema

    def get_unrendered_schema(self, yaml_key: Any, name: Any) -> Optional[Any]:
        if yaml_key not in self.unrendered_schemas:
            return None
        return self.unrendered_schemas[yaml_key].get(name)

@dataclass
class FixtureSourceFile(BaseSourceFile):
    fixture: Optional[Any] = None
    unit_tests: List[Any] = field(default_factory=list)

    def add_unit_test(self, value: Any) -> None:
        if value not in self.unit_tests:
            self.unit_tests.append(value)

AnySourceFile = Union[SchemaSourceFile, SourceFile, FixtureSourceFile]
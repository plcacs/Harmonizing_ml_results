import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from mashumaro.types import SerializableType
from dbt.artifacts.resources.base import FileHash
from dbt.constants import MAXIMUM_SEED_SIZE
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
parse_file_type_to_parser = {ParseFileType.Macro: 'MacroParser', ParseFileType.Model: 'ModelParser', ParseFileType.Snapshot: 'SnapshotParser', ParseFileType.Analysis: 'AnalysisParser', ParseFileType.SingularTest: 'SingularTestParser', ParseFileType.GenericTest: 'GenericTestParser', ParseFileType.Seed: 'SeedParser', ParseFileType.Documentation: 'DocumentationParser', ParseFileType.Schema: 'SchemaParser', ParseFileType.Hook: 'HookParser', ParseFileType.Fixture: 'FixtureParser'}

@dataclass
class FilePath(dbtClassMixin):

    @property
    def search_key(self):
        return self.absolute_path

    @property
    def full_path(self) -> str:
        return os.path.join(self.project_root, self.searched_path, self.relative_path)

    @property
    def absolute_path(self):
        return os.path.abspath(self.full_path)

    @property
    def original_file_path(self):
        return os.path.join(self.searched_path, self.relative_path)

    def seed_too_large(self) -> bool:
        """Return whether the file this represents is over the seed size limit"""
        return os.stat(self.full_path).st_size > MAXIMUM_SEED_SIZE

@dataclass
class RemoteFile(dbtClassMixin):

    def __init__(self, language: Union[str, LocalizedString, None]) -> None:
        if language == 'sql':
            self.path_end = '.sql'
        elif language == 'python':
            self.path_end = '.py'
        else:
            raise RuntimeError(f'Invalid language for remote File {language}')
        self.path = f'from remote system{self.path_end}'

    @property
    def searched_path(self):
        return self.path

    @property
    def relative_path(self):
        return self.path

    @property
    def absolute_path(self):
        return self.path

    @property
    def original_file_path(self):
        return self.path

    @property
    def modification_time(self):
        return self.path

@dataclass
class BaseSourceFile(dbtClassMixin, SerializableType):
    """Define a source file in dbt"""
    project_name = None
    parse_file_type = None
    contents = None

    @property
    def file_id(self) -> Union[None, typing.Text]:
        if isinstance(self.path, RemoteFile):
            return None
        return f'{self.project_name}://{self.path.original_file_path}'

    @property
    def original_file_path(self):
        return self.path.original_file_path

    def _serialize(self) -> dict:
        dct = self.to_dict()
        return dct

    @classmethod
    def _deserialize(cls: Union[dict, typing.Mapping, typing.Type], dct: Union[dict, dict[str, typing.Any]]):
        if dct['parse_file_type'] == 'schema':
            sf = SchemaSourceFile.from_dict(dct)
        elif dct['parse_file_type'] == 'fixture':
            sf = FixtureSourceFile.from_dict(dct)
        else:
            sf = SourceFile.from_dict(dct)
        return sf

    def __post_serialize__(self, dct: Union[cmk.base.events.EventContext, typing.Mapping, dict], context: Union[None, dict, cmk.base.events.EventContext, dict[str, str]]=None) -> Union[dict, dict[str, typing.Any]]:
        dct = super().__post_serialize__(dct, context)
        dct_keys = list(dct.keys())
        for key in dct_keys:
            if isinstance(dct[key], list) and (not dct[key]):
                del dct[key]
        if 'contents' in dct:
            del dct['contents']
        return dct

@dataclass
class SourceFile(BaseSourceFile):
    nodes = field(default_factory=list)
    docs = field(default_factory=list)
    macros = field(default_factory=list)
    env_vars = field(default_factory=list)

    @classmethod
    def big_seed(cls: str, path: str) -> SourceFile:
        """Parse seeds over the size limit with just the path"""
        self = cls(path=path, checksum=FileHash.path(path.original_file_path))
        self.contents = ''
        return self

    def add_node(self, value: Union[str, int, Vertex, None]) -> None:
        if value not in self.nodes:
            self.nodes.append(value)

    @classmethod
    def remote(cls: Union[str, dict[str, bool]], contents: Union[str, dict[str, bool]], project_name: Union[str, dict[str, bool]], language: Union[str, dict[str, bool]]) -> SourceFile:
        self = cls(path=RemoteFile(language), checksum=FileHash.from_contents(contents), project_name=project_name, contents=contents)
        return self

@dataclass
class SchemaSourceFile(BaseSourceFile):
    dfy = field(default_factory=dict)
    data_tests = field(default_factory=dict)
    sources = field(default_factory=list)
    exposures = field(default_factory=list)
    metrics = field(default_factory=list)
    snapshots = field(default_factory=list)
    generated_metrics = field(default_factory=list)
    metrics_from_measures = field(default_factory=dict)
    groups = field(default_factory=list)
    ndp = field(default_factory=list)
    semantic_models = field(default_factory=list)
    unit_tests = field(default_factory=list)
    saved_queries = field(default_factory=list)
    mcp = field(default_factory=dict)
    sop = field(default_factory=list)
    env_vars = field(default_factory=dict)
    unrendered_configs = field(default_factory=dict)
    unrendered_databases = field(default_factory=dict)
    unrendered_schemas = field(default_factory=dict)
    pp_dict = None
    pp_test_index = None

    @property
    def dict_from_yaml(self):
        return self.dfy

    @property
    def node_patches(self):
        return self.ndp

    @property
    def macro_patches(self):
        return self.mcp

    @property
    def source_patches(self):
        return self.sop

    def __post_serialize__(self, dct: Union[cmk.base.events.EventContext, typing.Mapping, dict], context: Union[None, dict, cmk.base.events.EventContext, dict[str, str]]=None) -> Union[dict, dict[str, typing.Any]]:
        dct = super().__post_serialize__(dct, context)
        for key in ('pp_test_index', 'pp_dict'):
            if key in dct:
                del dct[key]
        return dct

    def append_patch(self, yaml_key: Union[str, None, skeema.types.KeyValueDef], unique_id: Union[str, None, list[str]]) -> None:
        self.node_patches.append(unique_id)

    def add_test(self, node_unique_id: Union[str, dict, dict[str, typing.Any]], test_from: Union[str, dict]) -> None:
        name = test_from['name']
        key = test_from['key']
        if key not in self.data_tests:
            self.data_tests[key] = {}
        if name not in self.data_tests[key]:
            self.data_tests[key][name] = []
        self.data_tests[key][name].append(node_unique_id)

    def remove_tests(self, yaml_key: Union[str, dict, typing.Callable[str, str]], name: Union[str, dict, dict[str, typing.Any]]) -> None:
        if yaml_key in self.data_tests:
            if name in self.data_tests[yaml_key]:
                del self.data_tests[yaml_key][name]

    def get_tests(self, yaml_key: Union[str, dict, typing.Mapping], name: Union[str, dict[str, typing.Any]]) -> list:
        if yaml_key in self.data_tests:
            if name in self.data_tests[yaml_key]:
                return self.data_tests[yaml_key][name]
        return []

    def add_metrics_from_measures(self, semantic_model_name: Union[str, dict], metric_unique_id: Union[list, set[tuple[str]], str, None]) -> None:
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

    def get_key_and_name_for_test(self, test_unique_id: Union[str, int, None, list]) -> tuple[None]:
        yaml_key = None
        block_name = None
        for key in self.data_tests.keys():
            for name in self.data_tests[key]:
                for unique_id in self.data_tests[key][name]:
                    if unique_id == test_unique_id:
                        yaml_key = key
                        block_name = name
                        break
        return (yaml_key, block_name)

    def get_all_test_ids(self) -> list:
        test_ids = []
        for key in self.data_tests.keys():
            for name in self.data_tests[key]:
                test_ids.extend(self.data_tests[key][name])
        return test_ids

    def add_unrendered_config(self, unrendered_config: str, yaml_key: str, name: Union[str, None, dict[str, typing.Any]], version: Union[None, str, dict[str, typing.Any]]=None) -> None:
        versioned_name = f'{name}_v{version}' if version is not None else name
        if yaml_key not in self.unrendered_configs:
            self.unrendered_configs[yaml_key] = {}
        if versioned_name not in self.unrendered_configs[yaml_key]:
            self.unrendered_configs[yaml_key][versioned_name] = unrendered_config

    def get_unrendered_config(self, yaml_key: Union[str, list, dict], name: Union[str, None, dict[str, str], list[str]], version: Union[None, str, dict[str, str], list[str]]=None) -> None:
        versioned_name = f'{name}_v{version}' if version is not None else name
        if yaml_key not in self.unrendered_configs:
            return None
        if versioned_name not in self.unrendered_configs[yaml_key]:
            return None
        return self.unrendered_configs[yaml_key][versioned_name]

    def delete_from_unrendered_configs(self, yaml_key: Union[str, list], name: Union[str, list, cmk.base.api.agent_based.type_defs.RuleSetName]) -> None:
        if self.get_unrendered_config(yaml_key, name):
            del self.unrendered_configs[yaml_key][name]
            version_names_to_delete = []
            for potential_version_name in self.unrendered_configs[yaml_key]:
                if potential_version_name.startswith(f'{name}_v'):
                    version_names_to_delete.append(potential_version_name)
            for version_name in version_names_to_delete:
                del self.unrendered_configs[yaml_key][version_name]
            if not self.unrendered_configs[yaml_key]:
                del self.unrendered_configs[yaml_key]

    def add_env_var(self, var: Union[str, None, list[str]], yaml_key: str, name: Union[str, dict[str, typing.Any]]) -> None:
        if yaml_key not in self.env_vars:
            self.env_vars[yaml_key] = {}
        if name not in self.env_vars[yaml_key]:
            self.env_vars[yaml_key][name] = []
        if var not in self.env_vars[yaml_key][name]:
            self.env_vars[yaml_key][name].append(var)

    def delete_from_env_vars(self, yaml_key: str, name: str) -> None:
        if yaml_key in self.env_vars and name in self.env_vars[yaml_key]:
            del self.env_vars[yaml_key][name]
            if not self.env_vars[yaml_key]:
                del self.env_vars[yaml_key]

    def add_unrendered_database(self, yaml_key: Union[str, dict], name: Union[str, None], unrendered_database: Union[str, None]) -> None:
        if yaml_key not in self.unrendered_databases:
            self.unrendered_databases[yaml_key] = {}
        self.unrendered_databases[yaml_key][name] = unrendered_database

    def get_unrendered_database(self, yaml_key: Union[str, dict[str, typing.Any], None], name: Union[str, dict[str, typing.Any], skeema.types.KeyValueDef]) -> None:
        if yaml_key not in self.unrendered_databases:
            return None
        return self.unrendered_databases[yaml_key].get(name)

    def add_unrendered_schema(self, yaml_key: Union[str, dict, None], name: Union[str, dict, None], unrendered_schema: Union[str, dict, None]) -> None:
        if yaml_key not in self.unrendered_schemas:
            self.unrendered_schemas[yaml_key] = {}
        self.unrendered_schemas[yaml_key][name] = unrendered_schema

    def get_unrendered_schema(self, yaml_key: Union[str, None, dict], name: Union[str, dict, dict[str, typing.Any]]) -> None:
        if yaml_key not in self.unrendered_schemas:
            return None
        return self.unrendered_schemas[yaml_key].get(name)

@dataclass
class FixtureSourceFile(BaseSourceFile):
    fixture = None
    unit_tests = field(default_factory=list)

    def add_unit_test(self, value: Union[str, list, list[str]]) -> None:
        if value not in self.unit_tests:
            self.unit_tests.append(value)
AnySourceFile = Union[SchemaSourceFile, SourceFile, FixtureSourceFile]
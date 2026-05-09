import os
import unittest
from argparse import Namespace
from copy import deepcopy
from unittest import mock
import yaml
from dbt import tracking
from dbt.artifacts.resources import ModelConfig, RefArgs
from dbt.artifacts.resources.v1.model import ModelBuildAfter, ModelFreshnessDependsOnOptions
from dbt.context.context_config import ContextConfig
from dbt.contracts.files import FileHash, FilePath, SchemaSourceFile, SourceFile
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.model_config import NodeConfig, SnapshotConfig, TestConfig
from dbt.contracts.graph.nodes import AnalysisNode, DependsOn, Macro, ModelNode, SingularTestNode, SnapshotNode, UnpatchedSourceDefinition
from dbt.exceptions import CompilationError, ParsingError, SchemaConfigError
from dbt.flags import set_from_args
from dbt.node_types import NodeType
from dbt.parser import AnalysisParser, GenericTestParser, MacroParser, ModelParser, SchemaParser, SingularTestParser, SnapshotParser
from dbt.parser.common import YamlBlock
from dbt.parser.models import _get_config_call_dict, _get_exp_sample_result, _get_sample_result, _get_stable_sample_result, _shift_sources
from dbt.parser.schemas import AnalysisPatchParser, MacroPatchParser, ModelPatchParser, SourceParser, TestablePatchParser, yaml_from_file
from dbt.parser.search import FileBlock
from dbt.parser.sources import SourcePatcher
from tests.unit.utils import MockNode, config_from_parts_or_dicts, generate_name_macros, normalize

def get_abs_os_path(unix_path: str) -> str:
    return normalize(os.path.abspath(unix_path))

class BaseParserTest(unittest.TestCase):
    maxDiff: int = None

    def _generate_macros(self) -> list[Macro]:
        name_sql: dict[str, str] = {}
        for component in ('database', 'schema', 'alias'):
            if component == 'alias':
                source = 'node.name'
            else:
                source = f'target.{component}'
            name = f'generate_{component}_name'
            sql = f'{{% macro {name}(value, node) %}} {{% if value %}} {{{{ value }}}} {{% else %}} {{{{ {source} }}}} {{% endif %}} {{% endmacro %}}'
            name_sql[name] = sql
        for name, sql in name_sql.items():
            pm = Macro(name=name, resource_type=NodeType.Macro, unique_id=f'macro.root.{name}', package_name='root', original_file_path=normalize('macros/macro.sql'), path=normalize('macros/macro.sql'), macro_sql=sql)
            yield pm

    def setUp(self) -> None:
        set_from_args(Namespace(warn_error=True, state_modified_compare_more_unrendered_values=False), None)
        tracking.do_not_track()
        self.maxDiff = None
        profile_data = {'target': 'test', 'quoting': {}, 'outputs': {'test': {'type': 'postgres', 'host': 'localhost', 'schema': 'analytics', 'user': 'test', 'pass': 'test', 'dbname': 'test', 'port': 1}}}
        root_project = {'name': 'root', 'version': '0.1', 'profile': 'test', 'project-root': normalize('/usr/src/app'), 'config-version': 2}
        self.root_project_config = config_from_parts_or_dicts(project=root_project, profile=profile_data, cli_vars={'test_schema_name': 'foo'})
        snowplow_project = {'name': 'snowplow', 'version': '0.1', 'profile': 'test', 'project-root': get_abs_os_path('./dbt_packages/snowplow'), 'config-version': 2}
        self.snowplow_project_config = config_from_parts_or_dicts(project=snowplow_project, profile=profile_data)
        self.all_projects = {'root': self.root_project_config, 'snowplow': self.snowplow_project_config}
        self.root_project_config.dependencies = self.all_projects
        self.snowplow_project_config.dependencies = self.all_projects
        self.patcher = mock.patch('dbt.context.providers.get_adapter')
        self.factory = self.patcher.start()
        self.parser_patcher = mock.patch('dbt.parser.base.get_adapter')
        self.factory_parser = self.parser_patcher.start()
        self.manifest = Manifest(macros={m.unique_id: m for m in generate_name_macros('root')})

    def tearDown(self) -> None:
        self.parser_patcher.stop()
        self.patcher.stop()

    def source_file_for(self, data: str, filename: str, searched: str) -> SchemaSourceFile | SourceFile:
        root_dir = get_abs_os_path('./dbt_packages/snowplow')
        filename = normalize(filename)
        path = FilePath(searched_path=searched, relative_path=filename, project_root=root_dir, modification_time=0.0)
        sf_cls = SchemaSourceFile if filename.endswith('.yml') else SourceFile
        source_file = sf_cls(path=path, checksum=FileHash.from_contents(data), project_name='snowplow')
        source_file.contents = data
        return source_file

    def file_block_for(self, data: str, filename: str, searched: str = 'models') -> FileBlock:
        source_file = self.source_file_for(data, filename, searched)
        return FileBlock(file=source_file)

    def assert_has_manifest_lengths(self, manifest: Manifest, macros: int = 3, nodes: int = 0, sources: int = 0, docs: int = 0, disabled: int = 0, unit_tests: int = 0) -> None:
        self.assertEqual(len(manifest.macros), macros)
        self.assertEqual(len(manifest.nodes), nodes)
        self.assertEqual(len(manifest.sources), sources)
        self.assertEqual(len(manifest.docs), docs)
        self.assertEqual(len(manifest.disabled), disabled)
        self.assertEqual(len(manifest.unit_tests), unit_tests)

def assertEqualNodes(node_one: dict, node_two: dict) -> bool:
    node_one_dict = node_one.to_dict()
    if 'created_at' in node_one_dict:
        del node_one_dict['created_at']
    if 'relation_name' in node_one_dict:
        del node_one_dict['relation_name']
    node_two_dict = node_two.to_dict()
    if 'created_at' in node_two_dict:
        del node_two_dict['created_at']
    if 'relation_name' in node_two_dict:
        del node_two_dict['relation_name']
    if 'config' in node_one_dict and 'packages' in node_one_dict['config']:
        if 'config' not in node_two_dict and 'packages' in node_two_dict['config']:
            return False
        node_one_dict['config']['packages'] = set(node_one_dict['config']['packages'])
        node_two_dict['config']['packages'] = set(node_two_dict['config']['packages'])
        node_one_dict['unrendered_config']['packages'] = set(node_one_dict['config']['packages'])
        node_two_dict['unrendered_config']['packages'] = set(node_two_dict['config']['packages'])
        if 'packages' in node_one_dict['config_call_dict']:
            node_one_dict['config_call_dict']['packages'] = set(node_one_dict['config_call_dict']['packages'])
            node_two_dict['config_call_dict']['packages'] = set(node_two_dict['config_call_dict']['packages'])
    assert node_one_dict == node_two_dict

# Rest of the code remains the same

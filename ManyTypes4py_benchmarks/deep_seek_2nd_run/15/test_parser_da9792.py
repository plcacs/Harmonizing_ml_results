import os
import unittest
from argparse import Namespace
from copy import deepcopy
from unittest import mock
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Generator, Iterator
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

set_from_args(Namespace(warn_error=False, state_modified_compare_more_unrendered_values=False), None)

def get_abs_os_path(unix_path: str) -> str:
    return normalize(os.path.abspath(unix_path))

class BaseParserTest(unittest.TestCase):
    maxDiff: Optional[int] = None

    def _generate_macros(self) -> Generator[Macro, None, None]:
        name_sql: Dict[str, str] = {}
        for component in ('database', 'schema', 'alias'):
            if component == 'alias':
                source = 'node.name'
            else:
                source = f'target.{component}'
            name = f'generate_{component}_name'
            sql = f'{{% macro {name}(value, node) %}} {{% if value %}} {{{{ value }}}} {{% else %}} {{{{ {source} }}}} {{% endif %}} {{% endmacro %}}'
            name_sql[name] = sql
        for name, sql in name_sql.items():
            pm = Macro(
                name=name,
                resource_type=NodeType.Macro,
                unique_id=f'macro.root.{name}',
                package_name='root',
                original_file_path=normalize('macros/macro.sql'),
                path=normalize('macros/macro.sql'),
                macro_sql=sql
            )
            yield pm

    def setUp(self) -> None:
        set_from_args(Namespace(warn_error=True, state_modified_compare_more_unrendered_values=False), None)
        tracking.do_not_track()
        self.maxDiff = None
        profile_data = {
            'target': 'test',
            'quoting': {},
            'outputs': {
                'test': {
                    'type': 'postgres',
                    'host': 'localhost',
                    'schema': 'analytics',
                    'user': 'test',
                    'pass': 'test',
                    'dbname': 'test',
                    'port': 1
                }
            }
        }
        root_project = {
            'name': 'root',
            'version': '0.1',
            'profile': 'test',
            'project-root': normalize('/usr/src/app'),
            'config-version': 2
        }
        self.root_project_config = config_from_parts_or_dicts(
            project=root_project,
            profile=profile_data,
            cli_vars={'test_schema_name': 'foo'}
        )
        snowplow_project = {
            'name': 'snowplow',
            'version': '0.1',
            'profile': 'test',
            'project-root': get_abs_os_path('./dbt_packages/snowplow'),
            'config-version': 2
        }
        self.snowplow_project_config = config_from_parts_or_dicts(
            project=snowplow_project,
            profile=profile_data
        )
        self.all_projects = {
            'root': self.root_project_config,
            'snowplow': self.snowplow_project_config
        }
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

    def source_file_for(self, data: str, filename: str, searched: str) -> Union[SchemaSourceFile, SourceFile]:
        root_dir = get_abs_os_path('./dbt_packages/snowplow')
        filename = normalize(filename)
        path = FilePath(
            searched_path=searched,
            relative_path=filename,
            project_root=root_dir,
            modification_time=0.0
        )
        sf_cls = SchemaSourceFile if filename.endswith('.yml') else SourceFile
        source_file = sf_cls(
            path=path,
            checksum=FileHash.from_contents(data),
            project_name='snowplow'
        )
        source_file.contents = data
        return source_file

    def file_block_for(self, data: str, filename: str, searched: str) -> FileBlock:
        source_file = self.source_file_for(data, filename, searched)
        return FileBlock(file=source_file)

    def assert_has_manifest_lengths(
        self,
        manifest: Manifest,
        macros: int = 3,
        nodes: int = 0,
        sources: int = 0,
        docs: int = 0,
        disabled: int = 0,
        unit_tests: int = 0
    ) -> None:
        self.assertEqual(len(manifest.macros), macros)
        self.assertEqual(len(manifest.nodes), nodes)
        self.assertEqual(len(manifest.sources), sources)
        self.assertEqual(len(manifest.docs), docs)
        self.assertEqual(len(manifest.disabled), disabled)
        self.assertEqual(len(manifest.unit_tests), unit_tests)

def assertEqualNodes(node_one: Any, node_two: Any) -> None:
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

SINGLE_TABLE_SOURCE = '\nsources:\n    - name: my_source\n      tables:\n        - name: my_table\n'
MULTIPLE_TABLE_SOURCE_META = '\nsources:\n    - name: my_source\n      meta:\n        source_field: source_value\n        shared_field: shared_field_default\n      tables:\n        - name: my_table_shared_field_default\n          meta:\n            table_field: table_value\n        - name: my_table_shared_field_override\n          meta:\n            shared_field: shared_field_table_override\n            table_field: table_value\n'
SINGLE_TABLE_SOURCE_TESTS = "\nsources:\n    - name: my_source\n      tables:\n        - name: my_table\n          description: A description of my table\n          columns:\n            - name: color\n              data_tests:\n                - not_null:\n                    severity: WARN\n                - accepted_values:\n                    values: ['red', 'blue', 'green']\n"
SINGLE_TABLE_MODEL_TESTS = "\nmodels:\n    - name: my_model\n      description: A description of my model\n      columns:\n        - name: color\n          description: The color value\n          data_tests:\n            - not_null:\n                severity: WARN\n            - accepted_values:\n                description: Only primary colors are allowed in here\n                values: ['red', 'blue', 'green']\n            - foreign_package.test_case:\n                arg: 100\n"
SINGLE_TABLE_MODEL_TESTS_WRONG_SEVERITY = "\nmodels:\n    - name: my_model\n      description: A description of my model\n      columns:\n        - name: color\n          description: The color value\n          data_tests:\n            - not_null:\n                severity: WARNING\n            - accepted_values:\n                values: ['red', 'blue', 'green']\n            - foreign_package.test_case:\n                arg: 100\n"
SINGLE_TALBE_MODEL_FRESHNESS = '\nmodels:\n    - name: my_model\n      description: A description of my model\n      freshness:\n        build_after: {count: 1, period: day}\n'
SINGLE_TALBE_MODEL_FRESHNESS_ONLY_DEPEND_ON = '\nmodels:\n    - name: my_model\n      description: A description of my model\n      freshness:\n        build_after:\n            depends_on: all\n'
MULTIPLE_TABLE_VERSIONED_MODEL_TESTS = "\nmodels:\n    - name: my_model\n      description: A description of my model\n      data_tests:\n        - unique:\n            column_name: color\n      columns:\n        - name: color\n          description: The color value\n          data_tests:\n            - not_null:\n                severity: WARN\n        - name: location_id\n          data_type: int\n      versions:\n        - v: 1\n          defined_in: arbitrary_file_name\n          data_tests: []\n          columns:\n            - include: '*'\n            - name: extra\n        - v: 2\n          columns:\n            - include: '*'\n              exclude: ['location_id']\n            - name: extra\n"
MULTIPLE_TABLE_VERSIONED_MODEL = "\nmodels:\n    - name: my_model\n      description: A description of my model\n      config:\n        materialized: table\n        sql_header: test_sql_header\n      columns:\n        - name: color\n          description: The color value\n        - name: location_id\n          data_type: int\n      versions:\n        - v: 1\n          defined_in: arbitrary_file_name\n          columns:\n            - include: '*'\n            - name: extra\n        - v: 2\n          config:\n            materialized: view\n          columns:\n            - include: '*'\n              exclude: ['location_id']\n            - name: extra\n"
MULTIPLE_TABLE_VERSIONED_MODEL_CONTRACT_ENFORCED = '\nmodels:\n    - name: my_model\n      config:\n        contract:\n            enforced: true\n      versions:\n        - v: 0\n          defined_in: arbitrary_file_name\n        - v: 2\n'
MULTIPLE_TABLE_VERSIONED_MODEL_V0 = '\nmodels:\n    - name: my_model\n      versions:\n        - v: 0\n          defined_in: arbitrary_file_name\n        - v: 2\n'
MULTIPLE_TABLE_VERSIONED_MODEL_V0_LATEST_VERSION = '\nmodels:\n    - name: my_model\n      latest_version: 0\n      versions:\n        - v: 0\n          defined_in: arbitrary_file_name\n        - v: 2\n'
SINGLE_TABLE_SOURCE_PATCH = '\nsources:\n  - name: my_source\n    overrides: snowplow\n    tables:\n      - name: my_table\n        columns:\n          - name: id\n            data_tests:\n              - not_null\n              - unique\n'
SOURCE_CUSTOM_FRESHNESS_AT_SOURCE = '\nsources:\n  - name: my_source\n    loaded_at_query: "select 1 as id"\n    tables:\n      - name: my_table\n'
SOURCE_CUSTOM_FRESHNESS_AT_SOURCE_FIELD_AT_TABLE = '\nsources:\n  - name: my_source\n    loaded_at_query: "select 1 as id"\n    tables:\n      - name: my_table\n        loaded_at_field: test\n'
SOURCE_FIELD_AT_SOURCE_CUSTOM_FRESHNESS_AT_TABLE = '\nsources:\n  - name: my_source\n    loaded_at_field: test\n    tables:\n      - name: my_table\n        loaded_at_query: "select 1 as id"\n'
SOURCE_FIELD_AT_CUSTOM_FRESHNESS_BOTH_AT_TABLE = '\nsources:\n  - name: my_source\n    loaded_at_field: test\n    tables:\n      - name: my_table\n        loaded_at_query: "select 1 as id"\n        loaded_at_field: test\n'
SOURCE_FIELD_AT_CUSTOM_FRESHNESS_BOTH_AT_SOURCE = '\nsources:\n  - name: my_source\n    loaded_at_field: test\n    loaded_at_query: "select 1 as id"\n    tables:\n      - name: my_table\n        loaded_at_field: test\n'

class SchemaParserTest(BaseParserTest):
    def setUp(self) -> None:
        super().setUp()
        self.parser = SchemaParser(
            project=self.snowplow_project_config,
            manifest=self.manifest,
            root_project=self.root_project_config
        )
        self.source_patcher = SourcePatcher(
            root_project=self.root_project_config,
            manifest=self.manifest
        )

    def file_block_for(self, data: str, filename: str, searched: str = 'models') -> FileBlock:
        return super().file_block_for(data, filename, searched)

    def yaml_block_for(self, test_yml: str, filename: str) -> YamlBlock:
        file_block = self.file_block_for(data=test_yml, filename=filename)
        return YamlBlock.from_file_block(src=file_block, data=yaml.safe_load(test_yml))

class SchemaParserSourceTest(SchemaParserTest):
    def test__read_basic_source(self) -> None:
        block = self.yaml_block_for(SINGLE_TABLE_SOURCE, 'test_one.yml')
        analysis_blocks = AnalysisPatchParser(self.parser, block, 'analyses').parse().test_blocks
        model_blocks = ModelPatchParser(self.parser, block, 'models').parse().test_blocks
        source_blocks = SourceParser(self.parser, block, 'sources').parse().test_blocks
        macro_blocks = MacroPatchParser(self.parser, block, 'macros').parse().test_blocks
        self.assertEqual(len(analysis_blocks), 0)
        self.assertEqual(len(model_blocks), 0)
        self.assertEqual(len(source_blocks), 0)
        self.assertEqual(len(macro_blocks), 0)
        self.assertEqual(len(list(self.parser.manifest.nodes)), 0)
        source_values = list(self.parser.manifest.sources.values())
        self.assertEqual(len(source_values), 1)
        self.assertEqual(source_values[0].source.name, 'my_source')
        self.assertEqual(source_values[0].table.name, 'my_table')
        self.assertEqual(source_values[0].table.description, '')
        self.assertEqual(len(source_values[0].table.columns), 0)

    @mock.patch('dbt.parser.sources.get_adapter')
    def test_parse_source_custom_freshness_at_source(self, _: Any) -> None:
        block = self.file_block_for(SOURCE_CUSTOM_FRESHNESS_AT_SOURCE, 'test_one.yml')
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        unpatched_src_default = self.parser.manifest.sources['source.snowplow.my_source.my_table']
        src_default = self.source_patcher.parse_source(unpatched_src_default)
        assert src_default.loaded_at_query == 'select 1 as id'

    @mock.patch('dbt.parser.sources.get_adapter')
    def test_parse_source_custom_freshness_at_source_field_at_table(self, _: Any) -> None:
        block = self.file_block_for(SOURCE_CUSTOM_FRESHNESS_AT_SOURCE_FIELD_AT_TABLE, 'test_one.yml')
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        unpatched_src_default = self.parser.manifest.sources['source.snowplow.my_source.my_table']
        src_default = self.source_patcher.parse_source(unpatched
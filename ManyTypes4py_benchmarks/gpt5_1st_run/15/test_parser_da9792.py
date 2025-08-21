import os
import unittest
from argparse import Namespace
from copy import deepcopy
from unittest import mock
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

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
                macro_sql=sql,
            )
            yield pm

    def setUp(self) -> None:
        set_from_args(Namespace(warn_error=True, state_modified_compare_more_unrendered_values=False), None)
        tracking.do_not_track()
        self.maxDiff = None
        profile_data: Dict[str, Any] = {
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
                    'port': 1,
                }
            },
        }
        root_project: Dict[str, Any] = {
            'name': 'root',
            'version': '0.1',
            'profile': 'test',
            'project-root': normalize('/usr/src/app'),
            'config-version': 2,
        }
        self.root_project_config = config_from_parts_or_dicts(
            project=root_project, profile=profile_data, cli_vars={'test_schema_name': 'foo'}
        )
        snowplow_project: Dict[str, Any] = {
            'name': 'snowplow',
            'version': '0.1',
            'profile': 'test',
            'project-root': get_abs_os_path('./dbt_packages/snowplow'),
            'config-version': 2,
        }
        self.snowplow_project_config = config_from_parts_or_dicts(project=snowplow_project, profile=profile_data)
        self.all_projects: Dict[str, Any] = {'root': self.root_project_config, 'snowplow': self.snowplow_project_config}
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

    def source_file_for(self, data: str, filename: str, searched: str) -> Union[SourceFile, SchemaSourceFile]:
        root_dir = get_abs_os_path('./dbt_packages/snowplow')
        filename = normalize(filename)
        path = FilePath(searched_path=searched, relative_path=filename, project_root=root_dir, modification_time=0.0)
        sf_cls = SchemaSourceFile if filename.endswith('.yml') else SourceFile
        source_file = sf_cls(path=path, checksum=FileHash.from_contents(data), project_name='snowplow')
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
        unit_tests: int = 0,
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


SINGLE_TABLE_SOURCE: str = '\nsources:\n    - name: my_source\n      tables:\n        - name: my_table\n'
MULTIPLE_TABLE_SOURCE_META: str = '\nsources:\n    - name: my_source\n      meta:\n        source_field: source_value\n        shared_field: shared_field_default\n      tables:\n        - name: my_table_shared_field_default\n          meta:\n            table_field: table_value\n        - name: my_table_shared_field_override\n          meta:\n            shared_field: shared_field_table_override\n            table_field: table_value\n'
SINGLE_TABLE_SOURCE_TESTS: str = "\nsources:\n    - name: my_source\n      tables:\n        - name: my_table\n          description: A description of my table\n          columns:\n            - name: color\n              data_tests:\n                - not_null:\n                    severity: WARN\n                - accepted_values:\n                    values: ['red', 'blue', 'green']\n"
SINGLE_TABLE_MODEL_TESTS: str = "\nmodels:\n    - name: my_model\n      description: A description of my model\n      columns:\n        - name: color\n          description: The color value\n          data_tests:\n            - not_null:\n                severity: WARN\n            - accepted_values:\n                description: Only primary colors are allowed in here\n                values: ['red', 'blue', 'green']\n            - foreign_package.test_case:\n                arg: 100\n"
SINGLE_TABLE_MODEL_TESTS_WRONG_SEVERITY: str = "\nmodels:\n    - name: my_model\n      description: A description of my model\n      columns:\n        - name: color\n          description: The color value\n          data_tests:\n            - not_null:\n                severity: WARNING\n            - accepted_values:\n                values: ['red', 'blue', 'green']\n            - foreign_package.test_case:\n                arg: 100\n"
SINGLE_TALBE_MODEL_FRESHNESS: str = '\nmodels:\n    - name: my_model\n      description: A description of my model\n      freshness:\n        build_after: {count: 1, period: day}\n'
SINGLE_TALBE_MODEL_FRESHNESS_ONLY_DEPEND_ON: str = '\nmodels:\n    - name: my_model\n      description: A description of my model\n      freshness:\n        build_after:\n            depends_on: all\n'
MULTIPLE_TABLE_VERSIONED_MODEL_TESTS: str = "\nmodels:\n    - name: my_model\n      description: A description of my model\n      data_tests:\n        - unique:\n            column_name: color\n      columns:\n        - name: color\n          description: The color value\n          data_tests:\n            - not_null:\n                severity: WARN\n        - name: location_id\n          data_type: int\n      versions:\n        - v: 1\n          defined_in: arbitrary_file_name\n          data_tests: []\n          columns:\n            - include: '*'\n            - name: extra\n        - v: 2\n          columns:\n            - include: '*'\n              exclude: ['location_id']\n            - name: extra\n"
MULTIPLE_TABLE_VERSIONED_MODEL: str = "\nmodels:\n    - name: my_model\n      description: A description of my model\n      config:\n        materialized: table\n        sql_header: test_sql_header\n      columns:\n        - name: color\n          description: The color value\n        - name: location_id\n          data_type: int\n      versions:\n        - v: 1\n          defined_in: arbitrary_file_name\n          columns:\n            - include: '*'\n            - name: extra\n        - v: 2\n          config:\n            materialized: view\n          columns:\n            - include: '*'\n              exclude: ['location_id']\n            - name: extra\n"
MULTIPLE_TABLE_VERSIONED_MODEL_CONTRACT_ENFORCED: str = '\nmodels:\n    - name: my_model\n      config:\n        contract:\n            enforced: true\n      versions:\n        - v: 0\n          defined_in: arbitrary_file_name\n        - v: 2\n'
MULTIPLE_TABLE_VERSIONED_MODEL_V0: str = '\nmodels:\n    - name: my_model\n      versions:\n        - v: 0\n          defined_in: arbitrary_file_name\n        - v: 2\n'
MULTIPLE_TABLE_VERSIONED_MODEL_V0_LATEST_VERSION: str = '\nmodels:\n    - name: my_model\n      latest_version: 0\n      versions:\n        - v: 0\n          defined_in: arbitrary_file_name\n        - v: 2\n'
SINGLE_TABLE_SOURCE_PATCH: str = '\nsources:\n  - name: my_source\n    overrides: snowplow\n    tables:\n      - name: my_table\n        columns:\n          - name: id\n            data_tests:\n              - not_null\n              - unique\n'
SOURCE_CUSTOM_FRESHNESS_AT_SOURCE: str = '\nsources:\n  - name: my_source\n    loaded_at_query: "select 1 as id"\n    tables:\n      - name: my_table\n'
SOURCE_CUSTOM_FRESHNESS_AT_SOURCE_FIELD_AT_TABLE: str = '\nsources:\n  - name: my_source\n    loaded_at_query: "select 1 as id"\n    tables:\n      - name: my_table\n        loaded_at_field: test\n'
SOURCE_FIELD_AT_SOURCE_CUSTOM_FRESHNESS_AT_TABLE: str = '\nsources:\n  - name: my_source\n    loaded_at_field: test\n    tables:\n      - name: my_table\n        loaded_at_query: "select 1 as id"\n'
SOURCE_FIELD_AT_CUSTOM_FRESHNESS_BOTH_AT_TABLE: str = '\nsources:\n  - name: my_source\n    loaded_at_field: test\n    tables:\n      - name: my_table\n        loaded_at_query: "select 1 as id"\n        loaded_at_field: test\n'
SOURCE_FIELD_AT_CUSTOM_FRESHNESS_BOTH_AT_SOURCE: str = '\nsources:\n  - name: my_source\n    loaded_at_field: test\n    loaded_at_query: "select 1 as id"\n    tables:\n      - name: my_table\n        loaded_at_field: test\n'


class SchemaParserTest(BaseParserTest):
    def setUp(self) -> None:
        super().setUp()
        self.parser: SchemaParser = SchemaParser(
            project=self.snowplow_project_config, manifest=self.manifest, root_project=self.root_project_config
        )
        self.source_patcher: SourcePatcher = SourcePatcher(
            root_project=self.root_project_config, manifest=self.manifest
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
        src_default = self.source_patcher.parse_source(unpatched_src_default)
        assert src_default.loaded_at_query is None

    @mock.patch('dbt.parser.sources.get_adapter')
    def test_parse_source_field_at_source_custom_freshness_at_table(self, _: Any) -> None:
        block = self.file_block_for(SOURCE_FIELD_AT_SOURCE_CUSTOM_FRESHNESS_AT_TABLE, 'test_one.yml')
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        unpatched_src_default = self.parser.manifest.sources['source.snowplow.my_source.my_table']
        src_default = self.source_patcher.parse_source(unpatched_src_default)
        assert src_default.loaded_at_query == 'select 1 as id'

    @mock.patch('dbt.parser.sources.get_adapter')
    def test_parse_source_field_at_custom_freshness_both_at_table_fails(self, _: Any) -> None:
        block = self.file_block_for(SOURCE_FIELD_AT_CUSTOM_FRESHNESS_BOTH_AT_TABLE, 'test_one.yml')
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        unpatched_src_default = self.parser.manifest.sources['source.snowplow.my_source.my_table']
        with self.assertRaises(ParsingError):
            self.source_patcher.parse_source(unpatched_src_default)

    @mock.patch('dbt.parser.sources.get_adapter')
    def test_parse_source_field_at_custom_freshness_both_at_source_fails(self, _: Any) -> None:
        block = self.file_block_for(SOURCE_FIELD_AT_CUSTOM_FRESHNESS_BOTH_AT_SOURCE, 'test_one.yml')
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        unpatched_src_default = self.parser.manifest.sources['source.snowplow.my_source.my_table']
        with self.assertRaises(ParsingError):
            self.source_patcher.parse_source(unpatched_src_default)

    def test__parse_basic_source(self) -> None:
        block = self.file_block_for(SINGLE_TABLE_SOURCE, 'test_one.yml')
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        self.assert_has_manifest_lengths(self.parser.manifest, sources=1)
        src = list(self.parser.manifest.sources.values())[0]
        assert isinstance(src, UnpatchedSourceDefinition)
        assert src.package_name == 'snowplow'
        assert src.source.name == 'my_source'
        assert src.table.name == 'my_table'
        assert src.resource_type == NodeType.Source
        assert src.fqn == ['snowplow', 'my_source', 'my_table']

    @mock.patch('dbt.parser.sources.get_adapter')
    def test__parse_basic_source_meta(self, mock_get_adapter: Any) -> None:
        block = self.file_block_for(MULTIPLE_TABLE_SOURCE_META, 'test_one.yml')
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        self.assert_has_manifest_lengths(self.parser.manifest, sources=2)
        unpatched_src_default = self.parser.manifest.sources['source.snowplow.my_source.my_table_shared_field_default']
        src_default = self.source_patcher.parse_source(unpatched_src_default)
        assert src_default.meta == {
            'source_field': 'source_value',
            'shared_field': 'shared_field_default',
            'table_field': 'table_value',
        }
        assert src_default.source_meta == {'source_field': 'source_value', 'shared_field': 'shared_field_default'}
        unpatched_src_override = self.parser.manifest.sources['source.snowplow.my_source.my_table_shared_field_override']
        src_override = self.source_patcher.parse_source(unpatched_src_override)
        assert src_override.meta == {
            'source_field': 'source_value',
            'shared_field': 'shared_field_table_override',
            'table_field': 'table_value',
        }
        assert src_override.source_meta == {'source_field': 'source_value', 'shared_field': 'shared_field_default'}

    def test__read_basic_source_tests(self) -> None:
        block = self.yaml_block_for(SINGLE_TABLE_SOURCE_TESTS, 'test_one.yml')
        analysis_tests = AnalysisPatchParser(self.parser, block, 'analyses').parse().test_blocks
        model_tests = ModelPatchParser(self.parser, block, 'models').parse().test_blocks
        source_tests = SourceParser(self.parser, block, 'sources').parse().test_blocks
        macro_tests = MacroPatchParser(self.parser, block, 'macros').parse().test_blocks
        self.assertEqual(len(analysis_tests), 0)
        self.assertEqual(len(model_tests), 0)
        self.assertEqual(len(source_tests), 0)
        self.assertEqual(len(macro_tests), 0)
        self.assertEqual(len(list(self.parser.manifest.nodes)), 0)
        self.assertEqual(len(list(self.parser.manifest.source_patches)), 0)
        source_values = list(self.parser.manifest.sources.values())
        self.assertEqual(len(source_values), 1)
        self.assertEqual(source_values[0].source.name, 'my_source')
        self.assertEqual(source_values[0].table.name, 'my_table')
        self.assertEqual(source_values[0].table.description, 'A description of my table')
        self.assertEqual(len(source_values[0].table.columns), 1)

    def test__parse_basic_source_tests(self) -> None:
        block = self.file_block_for(SINGLE_TABLE_SOURCE_TESTS, 'test_one.yml')
        self.parser.manifest.files[block.file.file_id] = block.file
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        self.assertEqual(len(self.parser.manifest.nodes), 0)
        self.assertEqual(len(self.parser.manifest.sources), 1)
        src = list(self.parser.manifest.sources.values())[0]
        self.assertEqual(src.source.name, 'my_source')
        self.assertEqual(src.source.schema, None)
        self.assertEqual(src.table.name, 'my_table')
        self.assertEqual(src.table.description, 'A description of my table')
        tests = [self.source_patcher.parse_source_test(src, test, col) for test, col in src.get_tests()]
        tests.sort(key=lambda n: n.unique_id)
        self.assertEqual(tests[0].config.severity, 'ERROR')
        self.assertEqual(tests[0].tags, [])
        self.assertEqual(tests[0].sources, [['my_source', 'my_table']])
        self.assertEqual(tests[0].column_name, 'color')
        self.assertEqual(tests[0].fqn, ['snowplow', tests[0].name])
        self.assertEqual(tests[1].config.severity, 'WARN')
        self.assertEqual(tests[1].tags, [])
        self.assertEqual(tests[1].sources, [['my_source', 'my_table']])
        self.assertEqual(tests[1].column_name, 'color')
        self.assertEqual(tests[1].fqn, ['snowplow', tests[1].name])
        file_id = 'snowplow://' + normalize('models/test_one.yml')
        self.assertIn(file_id, self.parser.manifest.files)
        self.assertEqual(self.parser.manifest.files[file_id].data_tests, {})
        self.assertEqual(self.parser.manifest.files[file_id].sources, ['source.snowplow.my_source.my_table'])
        self.assertEqual(self.parser.manifest.files[file_id].source_patches, [])

    def test__read_source_patch(self) -> None:
        block = self.yaml_block_for(SINGLE_TABLE_SOURCE_PATCH, 'test_one.yml')
        analysis_tests = AnalysisPatchParser(self.parser, block, 'analyses').parse().test_blocks
        model_tests = TestablePatchParser(self.parser, block, 'models').parse().test_blocks
        source_tests = SourceParser(self.parser, block, 'sources').parse().test_blocks
        macro_tests = MacroPatchParser(self.parser, block, 'macros').parse().test_blocks
        self.assertEqual(len(analysis_tests), 0)
        self.assertEqual(len(model_tests), 0)
        self.assertEqual(len(source_tests), 0)
        self.assertEqual(len(macro_tests), 0)
        self.assertEqual(len(list(self.parser.manifest.nodes)), 0)
        self.assertEqual(len(list(self.parser.manifest.sources)), 0)
        source_patches = list(self.parser.manifest.source_patches.values())
        self.assertEqual(len(source_patches), 1)
        self.assertEqual(source_patches[0].name, 'my_source')
        self.assertEqual(source_patches[0].overrides, 'snowplow')
        self.assertIsNone(source_patches[0].description)
        self.assertEqual(len(source_patches[0].tables), 1)
        table = source_patches[0].tables[0]
        self.assertEqual(table.name, 'my_table')
        self.assertIsNone(table.description)
        self.assertEqual(len(table.columns), 1)
        self.assertEqual(len(table.columns[0].data_tests), 2)


class SchemaParserModelsTest(SchemaParserTest):
    def setUp(self) -> None:
        super().setUp()
        my_model_node = MockNode(
            package='root',
            name='my_model',
            config=mock.MagicMock(enabled=True),
            refs=[],
            sources=[],
            patch_path=None,
        )
        source_file = self.source_file_for('', 'my_model.sql', 'models')
        nodes = {my_model_node.unique_id: my_model_node}
        macros = {m.unique_id: m for m in generate_name_macros('root')}
        self.manifest = Manifest(nodes=nodes, macros=macros)
        self.manifest.files[source_file.file_id] = source_file
        self.manifest.ref_lookup
        self.parser = SchemaParser(
            project=self.snowplow_project_config, manifest=self.manifest, root_project=self.root_project_config
        )

    def test__read_basic_model_tests(self) -> None:
        block = self.yaml_block_for(SINGLE_TABLE_MODEL_TESTS, 'test_one.yml')
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        self.assertEqual(len(list(self.parser.manifest.sources)), 0)
        self.assertEqual(len(list(self.parser.manifest.nodes)), 4)

    def test__parse_model_freshness(self) -> None:
        block = self.file_block_for(SINGLE_TALBE_MODEL_FRESHNESS, 'test_one.yml')
        self.parser.manifest.files[block.file.file_id] = block.file
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        self.assert_has_manifest_lengths(self.parser.manifest, nodes=1)
        assert self.parser.manifest.nodes['model.root.my_model'].freshness.build_after == ModelBuildAfter(
            count=1, period='day', depends_on=ModelFreshnessDependsOnOptions.any
        )

    def test__parse_model_freshness_depend_on(self) -> None:
        block = self.file_block_for(SINGLE_TALBE_MODEL_FRESHNESS_ONLY_DEPEND_ON, 'test_one.yml')
        self.parser.manifest.files[block.file.file_id] = block.file
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        self.assert_has_manifest_lengths(self.parser.manifest, nodes=1)
        assert self.parser.manifest.nodes['model.root.my_model'].freshness.build_after == ModelBuildAfter(
            count=0, period='hour', depends_on=ModelFreshnessDependsOnOptions.all
        )

    def test__read_basic_model_tests_wrong_severity(self) -> None:
        block = self.yaml_block_for(SINGLE_TABLE_MODEL_TESTS_WRONG_SEVERITY, 'test_one.yml')
        dct = yaml_from_file(block.file)
        with self.assertRaisesRegex(SchemaConfigError, "Severity must be either 'warn' or 'error'. Got 'WARNING'"):
            self.parser.parse_file(block, dct)

    def test__parse_basic_model_tests(self) -> None:
        block = self.file_block_for(SINGLE_TABLE_MODEL_TESTS, 'test_one.yml')
        self.parser.manifest.files[block.file.file_id] = block.file
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        self.assert_has_manifest_lengths(self.parser.manifest, nodes=4)
        all_nodes = sorted(self.parser.manifest.nodes.values(), key=lambda n: n.unique_id)
        tests = []
        for node in all_nodes:
            if node.resource_type != NodeType.Test:
                continue
            tests.append(node)
        self.assertEqual(tests[0].config.severity, 'ERROR')
        self.assertEqual(tests[0].tags, [])
        self.assertEqual(tests[0].refs, [RefArgs(name='my_model')])
        self.assertEqual(tests[0].column_name, 'color')
        self.assertEqual(tests[0].description, 'Only primary colors are allowed in here')
        self.assertEqual(tests[0].package_name, 'snowplow')
        self.assertTrue(tests[0].name.startswith('accepted_values_'))
        self.assertEqual(tests[0].fqn, ['snowplow', tests[0].name])
        self.assertEqual(tests[0].unique_id.split('.'), ['test', 'snowplow', tests[0].name, '9d4814efde'])
        self.assertEqual(tests[0].test_metadata.name, 'accepted_values')
        self.assertIsNone(tests[0].test_metadata.namespace)
        self.assertEqual(
            tests[0].test_metadata.kwargs,
            {'column_name': 'color', 'model': "{{ get_where_subquery(ref('my_model')) }}", 'values': ['red', 'blue', 'green']},
        )
        self.assertEqual(tests[1].config.severity, 'ERROR')
        self.assertEqual(tests[1].tags, [])
        self.assertEqual(tests[1].refs, [RefArgs(name='my_model')])
        self.assertEqual(tests[1].column_name, 'color')
        self.assertEqual(tests[1].description, '')
        self.assertEqual(tests[1].fqn, ['snowplow', tests[1].name])
        self.assertTrue(tests[1].name.startswith('foreign_package_test_case_'))
        self.assertEqual(tests[1].package_name, 'snowplow')
        self.assertEqual(tests[1].unique_id.split('.'), ['test', 'snowplow', tests[1].name, '13958f62f7'])
        self.assertEqual(tests[1].test_metadata.name, 'test_case')
        self.assertEqual(tests[1].test_metadata.namespace, 'foreign_package')
        self.assertEqual(
            tests[1].test_metadata.kwargs,
            {'column_name': 'color', 'model': "{{ get_where_subquery(ref('my_model')) }}", 'arg': 100},
        )
        self.assertEqual(tests[2].config.severity, 'WARN')
        self.assertEqual(tests[2].tags, [])
        self.assertEqual(tests[2].refs, [RefArgs(name='my_model')])
        self.assertEqual(tests[2].column_name, 'color')
        self.assertEqual(tests[2].package_name, 'snowplow')
        self.assertTrue(tests[2].name.startswith('not_null_'))
        self.assertEqual(tests[2].fqn, ['snowplow', tests[2].name])
        self.assertEqual(tests[2].unique_id.split('.'), ['test', 'snowplow', tests[2].name, '2f61818750'])
        self.assertEqual(tests[2].test_metadata.name, 'not_null')
        self.assertIsNone(tests[2].test_metadata.namespace)
        self.assertEqual(
            tests[2].test_metadata.kwargs, {'column_name': 'color', 'model': "{{ get_where_subquery(ref('my_model')) }}"}
        )
        file_id = 'snowplow://' + normalize('models/test_one.yml')
        self.assertIn(file_id, self.parser.manifest.files)
        schema_file_test_ids = self.parser.manifest.files[file_id].get_all_test_ids()
        self.assertEqual(sorted(schema_file_test_ids), [t.unique_id for t in tests])
        self.assertEqual(self.parser.manifest.files[file_id].node_patches, ['model.root.my_model'])


class SchemaParserVersionedModels(SchemaParserTest):
    def setUp(self) -> None:
        super().setUp()
        my_model_v1_node = MockNode(
            package='snowplow',
            name='arbitrary_file_name',
            config=mock.MagicMock(enabled=True),
            refs=[],
            sources=[],
            patch_path=None,
            file_id='snowplow://models/arbitrary_file_name.sql',
        )
        my_model_v1_source_file = self.source_file_for('', 'arbitrary_file_name.sql', 'models')
        my_model_v2_node = MockNode(
            package='snowplow',
            name='my_model_v2',
            config=mock.MagicMock(enabled=True),
            refs=[],
            sources=[],
            patch_path=None,
            file_id='snowplow://models/my_model_v2.sql',
        )
        my_model_v2_source_file = self.source_file_for('', 'my_model_v2.sql', 'models')
        nodes = {my_model_v1_node.unique_id: my_model_v1_node, my_model_v2_node.unique_id: my_model_v2_node}
        macros = {m.unique_id: m for m in generate_name_macros('root')}
        files = {my_model_v1_source_file.file_id: my_model_v1_source_file, my_model_v2_source_file.file_id: my_model_v2_source_file}
        self.manifest = Manifest(nodes=nodes, macros=macros, files=files)
        self.manifest.ref_lookup
        self.parser = SchemaParser(
            project=self.snowplow_project_config, manifest=self.manifest, root_project=self.root_project_config
        )

    def test__read_versioned_model_tests(self) -> None:
        block = self.yaml_block_for(MULTIPLE_TABLE_VERSIONED_MODEL_TESTS, 'test_one.yml')
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        self.assertEqual(len(list(self.parser.manifest.sources)), 0)
        self.assertEqual(len(list(self.parser.manifest.nodes)), 5)

    def test__parse_versioned_model_tests(self) -> None:
        block = self.file_block_for(MULTIPLE_TABLE_VERSIONED_MODEL_TESTS, 'test_one.yml')
        self.parser.manifest.files[block.file.file_id] = block.file
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        self.assert_has_manifest_lengths(self.parser.manifest, nodes=5)
        all_nodes = sorted(self.parser.manifest.nodes.values(), key=lambda n: n.unique_id)
        tests = [node for node in all_nodes if node.resource_type == NodeType.Test]
        self.assertEqual(tests[0].config.severity, 'WARN')
        self.assertEqual(tests[0].tags, [])
        self.assertEqual(tests[0].refs, [RefArgs(name='my_model', version='1')])
        self.assertEqual(tests[0].column_name, 'color')
        self.assertEqual(tests[0].package_name, 'snowplow')
        self.assertTrue(tests[0].name.startswith('not_null'))
        self.assertEqual(tests[0].fqn, ['snowplow', tests[0].name])
        self.assertEqual(tests[0].unique_id.split('.'), ['test', 'snowplow', tests[0].name, 'b704420587'])
        self.assertEqual(tests[0].test_metadata.name, 'not_null')
        self.assertIsNone(tests[0].test_metadata.namespace)
        self.assertEqual(
            tests[0].test_metadata.kwargs, {'column_name': 'color', 'model': "{{ get_where_subquery(ref('my_model', version='1')) }}"}
        )
        self.assertEqual(tests[1].config.severity, 'WARN')
        self.assertEqual(tests[1].tags, [])
        self.assertEqual(tests[1].refs, [RefArgs(name='my_model', version='2')])
        self.assertEqual(tests[1].column_name, 'color')
        self.assertEqual(tests[1].fqn, ['snowplow', tests[1].name])
        self.assertTrue(tests[1].name.startswith('not_null'))
        self.assertEqual(tests[1].package_name, 'snowplow')
        self.assertEqual(tests[1].unique_id.split('.'), ['test', 'snowplow', tests[1].name, '3375708d04'])
        self.assertEqual(tests[1].test_metadata.name, 'not_null')
        self.assertIsNone(tests[0].test_metadata.namespace)
        self.assertEqual(
            tests[1].test_metadata.kwargs, {'column_name': 'color', 'model': "{{ get_where_subquery(ref('my_model', version='2')) }}"}
        )
        self.assertEqual(tests[2].config.severity, 'ERROR')
        self.assertEqual(tests[2].tags, [])
        self.assertEqual(tests[2].refs, [RefArgs(name='my_model', version='2')])
        self.assertIsNone(tests[2].column_name)
        self.assertEqual(tests[2].package_name, 'snowplow')
        self.assertTrue(tests[2].name.startswith('unique'))
        self.assertEqual(tests[2].fqn, ['snowplow', tests[2].name])
        self.assertEqual(tests[2].unique_id.split('.'), ['test', 'snowplow', tests[2].name, '29b09359d1'])
        self.assertEqual(tests[2].test_metadata.name, 'unique')
        self.assertIsNone(tests[2].test_metadata.namespace)
        self.assertEqual(
            tests[2].test_metadata.kwargs, {'column_name': 'color', 'model': "{{ get_where_subquery(ref('my_model', version='2')) }}"}
        )
        file_id = 'snowplow://' + normalize('models/test_one.yml')
        self.assertIn(file_id, self.parser.manifest.files)
        schema_file_test_ids = self.parser.manifest.files[file_id].get_all_test_ids()
        self.assertEqual(sorted(schema_file_test_ids), [t.unique_id for t in tests])
        self.assertEqual(self.parser.manifest.files[file_id].node_patches, ['model.snowplow.my_model.v1', 'model.snowplow.my_model.v2'])

    def test__parsed_versioned_models(self) -> None:
        block = self.file_block_for(MULTIPLE_TABLE_VERSIONED_MODEL, 'test_one.yml')
        self.parser.manifest.files[block.file.file_id] = block.file
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        self.assert_has_manifest_lengths(self.parser.manifest, nodes=2)

    def test__parsed_versioned_models_contract_enforced(self) -> None:
        block = self.file_block_for(MULTIPLE_TABLE_VERSIONED_MODEL_CONTRACT_ENFORCED, 'test_one.yml')
        self.parser.manifest.files[block.file.file_id] = block.file
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        self.assert_has_manifest_lengths(self.parser.manifest, nodes=2)
        for node in self.parser.manifest.nodes.values():
            assert node.contract.enforced
            node.build_contract_checksum.assert_called()

    def test__parsed_versioned_models_v0(self) -> None:
        block = self.file_block_for(MULTIPLE_TABLE_VERSIONED_MODEL_V0, 'test_one.yml')
        self.parser.manifest.files[block.file.file_id] = block.file
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        self.assert_has_manifest_lengths(self.parser.manifest, nodes=2)

    def test__parsed_versioned_models_v0_latest_version(self) -> None:
        block = self.file_block_for(MULTIPLE_TABLE_VERSIONED_MODEL_V0_LATEST_VERSION, 'test_one.yml')
        self.parser.manifest.files[block.file.file_id] = block.file
        dct = yaml_from_file(block.file)
        self.parser.parse_file(block, dct)
        self.assert_has_manifest_lengths(self.parser.manifest, nodes=2)


sql_model: str = '\n{{ config(materialized="table") }}\nselect 1 as id\n'
sql_model_parse_error: str = '{{ SYNTAX ERROR }}'
python_model: str = '\nimport textblob\nimport text as a\nfrom torch import b\nimport textblob.text\nimport sklearn\n\ndef model(dbt, session):\n    dbt.config(\n        materialized=\'table\',\n        packages=[\'sklearn==0.1.0\']\n    )\n    df0 = dbt.ref("a_model").to_pandas()\n    df1 = dbt.ref("my_sql_model").task.limit(2)\n    df2 = dbt.ref("my_sql_model_1")\n    df3 = dbt.ref("my_sql_model_2")\n    df4 = dbt.source("test", \'table1\').limit(max=[max(dbt.ref(\'something\'))])\n    df5 = [dbt.ref(\'test1\')]\n\n    a_dict = {\'test2\': dbt.ref(\'test2\')}\n    df5 = {\'test2\': dbt.ref(\'test3\')}\n    df6 = [dbt.ref("test4")]\n    f"{dbt.ref(\'test5\')}"\n\n    df = df0.limit(2)\n    return df\n'
python_model_config: str = '\ndef model(dbt, session):\n    dbt.config.get("param_1")\n    dbt.config.get("param_2")\n    return dbt.ref("some_model")\n'
python_model_config_with_defaults: str = '\ndef model(dbt, session):\n    dbt.config.get("param_None", None)\n    dbt.config.get("param_Str", "default")\n    dbt.config.get("param_List", [1, 2])\n    return dbt.ref("some_model")\n'
python_model_single_argument: str = '\ndef model(dbt):\n     dbt.config(materialized="table")\n     return dbt.ref("some_model")\n'
python_model_no_argument: str = '\nimport pandas as pd\n\ndef model():\n    return pd.dataframe([1, 2])\n'
python_model_incorrect_argument_name: str = '\ndef model(tbd, session):\n    tbd.config(materialized="table")\n    return tbd.ref("some_model")\n'
python_model_multiple_models: str = '\ndef model(dbt, session):\n    dbt.config(materialized=\'table\')\n    return dbt.ref("some_model")\n\ndef model(dbt, session):\n    dbt.config(materialized=\'table\')\n    return dbt.ref("some_model")\n'
python_model_incorrect_function_name: str = '\ndef model1(dbt, session):\n    dbt.config(materialized=\'table\')\n    return dbt.ref("some_model")\n'
python_model_empty_file: str = '    '
python_model_multiple_returns: str = '\ndef model(dbt, session):\n    dbt.config(materialized=\'table\')\n    return dbt.ref("some_model"), dbt.ref("some_other_model")\n'
python_model_f_string: str = '\n# my_python_model.py\nimport pandas as pd\n\ndef model(dbt, fal):\n    dbt.config(materialized="table")\n    print(f"my var: {dbt.config.get(\'my_var\')}") # Prints "my var: None"\n    df: pd.DataFrame = dbt.ref("some_model")\n    return df\n'
python_model_no_return: str = "\ndef model(dbt, session):\n    dbt.config(materialized='table')\n"
python_model_single_return: str = "\nimport pandas as pd\n\ndef model(dbt, session):\n    dbt.config(materialized='table')\n    return pd.dataframe([1, 2])\n"
python_model_incorrect_ref: str = '\ndef model(dbt, session):\n    model_names = ["orders", "customers"]\n    models = []\n\n    for model_name in model_names:\n        models.extend(dbt.ref(model_name))\n\n    return models[0]\n'
python_model_default_materialization: str = '\nimport pandas as pd\n\ndef model(dbt, session):\n    return pd.dataframe([1, 2])\n'
python_model_custom_materialization: str = '\nimport pandas as pd\n\ndef model(dbt, session):\n    dbt.config(materialized="incremental")\n    return pd.dataframe([1, 2])\n'


class ModelParserTest(BaseParserTest):
    def setUp(self) -> None:
        super().setUp()
        self.parser: ModelParser = ModelParser(
            project=self.snowplow_project_config, manifest=self.manifest, root_project=self.root_project_config
        )

    def file_block_for(self, data: str, filename: str) -> FileBlock:
        return super().file_block_for(data, filename, 'models')

    def test_basic(self) -> None:
        block = self.file_block_for(sql_model, 'nested/model_1.sql')
        self.parser.manifest.files[block.file.file_id] = block.file
        self.parser.parse_file(block)
        self.assert_has_manifest_lengths(self.parser.manifest, nodes=1)
        node = list(self.parser.manifest.nodes.values())[0]
        expected = ModelNode(
            alias='model_1',
            name='model_1',
            database='test',
            schema='analytics',
            resource_type=NodeType.Model,
            unique_id='model.snowplow.model_1',
            fqn=['snowplow', 'nested', 'model_1'],
            package_name='snowplow',
            original_file_path=normalize('models/nested/model_1.sql'),
            config=ModelConfig(materialized='table'),
            path=normalize('nested/model_1.sql'),
            language='sql',
            raw_code=sql_model,
            checksum=block.file.checksum,
            unrendered_config={'materialized': 'table'},
            config_call_dict={'materialized': 'table'},
        )
        assertEqualNodes(node, expected)
        file_id = 'snowplow://' + normalize('models/nested/model_1.sql')
        self.assertIn(file_id, self.parser.manifest.files)
        self.assertEqual(self.parser.manifest.files[file_id].nodes, ['model.snowplow.model_1'])

    def test_sql_model_parse_error(self) -> None:
        block = self.file_block_for(sql_model_parse_error, 'nested/model_1.sql')
        with self.assertRaises(CompilationError):
            self.parser.parse_file(block)

    def test_python_model_parse(self) -> None:
        block = self.file_block_for(python_model, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        self.parser.parse_file(block)
        self.assert_has_manifest_lengths(self.parser.manifest, nodes=1)
        node = list(self.parser.manifest.nodes.values())[0]
        python_packages = ['sklearn==0.1.0']
        expected = ModelNode(
            alias='py_model',
            name='py_model',
            database='test',
            schema='analytics',
            resource_type=NodeType.Model,
            unique_id='model.snowplow.py_model',
            fqn=['snowplow', 'nested', 'py_model'],
            package_name='snowplow',
            original_file_path=normalize('models/nested/py_model.py'),
            config=ModelConfig(materialized='table', packages=python_packages),
            path=normalize('nested/py_model.py'),
            language='python',
            raw_code=python_model,
            checksum=block.file.checksum,
            unrendered_config={'materialized': 'table', 'packages': python_packages},
            config_call_dict={'materialized': 'table', 'packages': python_packages},
            refs=[
                RefArgs(name='a_model'),
                RefArgs('my_sql_model'),
                RefArgs('my_sql_model_1'),
                RefArgs('my_sql_model_2'),
                RefArgs('something'),
                RefArgs('test1'),
                RefArgs('test2'),
                RefArgs('test3'),
                RefArgs('test4'),
                RefArgs('test5'),
            ],
            sources=[['test', 'table1']],
        )
        assertEqualNodes(node, expected)
        file_id = 'snowplow://' + normalize('models/nested/py_model.py')
        self.assertIn(file_id, self.parser.manifest.files)
        self.assertEqual(self.parser.manifest.files[file_id].nodes, ['model.snowplow.py_model'])

    def test_python_model_config(self) -> None:
        block = self.file_block_for(python_model_config, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        self.parser.parse_file(block)
        node = list(self.parser.manifest.nodes.values())[0]
        self.assertEqual(node.config.to_dict()['config_keys_used'], ['param_1', 'param_2'])

    def test_python_model_f_string_config(self) -> None:
        block = self.file_block_for(python_model_f_string, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        self.parser.parse_file(block)
        node = list(self.parser.manifest.nodes.values())[0]
        self.assertEqual(node.config.to_dict()['config_keys_used'], ['my_var'])

    def test_python_model_config_with_defaults(self) -> None:
        block = self.file_block_for(python_model_config_with_defaults, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        self.parser.parse_file(block)
        node = list(self.parser.manifest.nodes.values())[0]
        default_values = node.config.to_dict()['config_keys_defaults']
        self.assertIsNone(default_values[0])
        self.assertEqual(default_values[1], 'default')
        self.assertEqual(default_values[2], [1, 2])

    def test_python_model_single_argument(self) -> None:
        block = self.file_block_for(python_model_single_argument, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        with self.assertRaises(ParsingError):
            self.parser.parse_file(block)

    def test_python_model_no_argument(self) -> None:
        block = self.file_block_for(python_model_no_argument, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        with self.assertRaises(ParsingError):
            self.parser.parse_file(block)

    def test_python_model_incorrect_argument_name(self) -> None:
        block = self.file_block_for(python_model_incorrect_argument_name, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        with self.assertRaises(ParsingError):
            self.parser.parse_file(block)

    def test_python_model_multiple_models(self) -> None:
        block = self.file_block_for(python_model_multiple_models, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        with self.assertRaises(ParsingError):
            self.parser.parse_file(block)

    def test_python_model_incorrect_function_name(self) -> None:
        block = self.file_block_for(python_model_incorrect_function_name, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        with self.assertRaises(ParsingError):
            self.parser.parse_file(block)

    def test_python_model_empty_file(self) -> None:
        block = self.file_block_for(python_model_empty_file, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        self.assertIsNone(self.parser.parse_file(block))

    def test_python_model_multiple_returns(self) -> None:
        block = self.file_block_for(python_model_multiple_returns, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        with self.assertRaises(ParsingError):
            self.parser.parse_file(block)

    def test_python_model_no_return(self) -> None:
        block = self.file_block_for(python_model_no_return, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        with self.assertRaises(ParsingError):
            self.parser.parse_file(block)

    def test_python_model_single_return(self) -> None:
        block = self.file_block_for(python_model_single_return, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        self.assertIsNone(self.parser.parse_file(block))

    def test_python_model_incorrect_ref(self) -> None:
        block = self.file_block_for(python_model_incorrect_ref, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        with self.assertRaises(ParsingError):
            self.parser.parse_file(block)

    def test_python_model_default_materialization(self) -> None:
        block = self.file_block_for(python_model_default_materialization, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        self.parser.parse_file(block)
        node = list(self.parser.manifest.nodes.values())[0]
        self.assertEqual(node.get_materialization(), 'table')

    def test_python_model_custom_materialization(self) -> None:
        block = self.file_block_for(python_model_custom_materialization, 'nested/py_model.py')
        self.parser.manifest.files[block.file.file_id] = block.file
        self.parser.parse_file(block)
        node = list(self.parser.manifest.nodes.values())[0]
        self.assertEqual(node.get_materialization(), 'incremental')


class StaticModelParserTest(BaseParserTest):
    def setUp(self) -> None:
        super().setUp()
        self.parser: ModelParser = ModelParser(
            project=self.snowplow_project_config, manifest=self.manifest, root_project=self.root_project_config
        )

    def file_block_for(self, data: str, filename: str) -> FileBlock:
        return super().file_block_for(data, filename, 'models')

    def test_built_in_macro_override_detection(self) -> None:
        macro_unique_id = 'macro.root.ref'
        self.parser.manifest.macros[macro_unique_id] = Macro(
            name='ref',
            resource_type=NodeType.Macro,
            unique_id=macro_unique_id,
            package_name='root',
            original_file_path=normalize('macros/macro.sql'),
            path=normalize('macros/macro.sql'),
            macro_sql='{% macro ref(model_name) %}{% set x = raise("boom") %}{% endmacro %}',
        )
        raw_code = '{{ config(materialized="table") }}select 1 as id'
        block = self.file_block_for(raw_code, 'nested/model_1.sql')
        node = ModelNode(
            alias='model_1',
            name='model_1',
            database='test',
            schema='analytics',
            resource_type=NodeType.Model,
            unique_id='model.snowplow.model_1',
            fqn=['snowplow', 'nested', 'model_1'],
            package_name='snowplow',
            original_file_path=normalize('models/nested/model_1.sql'),
            config=ModelConfig(materialized='table'),
            path=normalize('nested/model_1.sql'),
            language='sql',
            raw_code=raw_code,
            checksum=block.file.checksum,
            unrendered_config={'materialized': 'table'},
        )
        assert self.parser._has_banned_macro(node)


class StaticModelParserUnitTest(BaseParserTest):
    def setUp(self) -> None:
        super().setUp()
        self.parser: ModelParser = ModelParser(
            project=self.snowplow_project_config, manifest=self.manifest, root_project=self.root_project_config
        )
        self.example_node: ModelNode = ModelNode(
            alias='model_1',
            name='model_1',
            database='test',
            schema='analytics',
            resource_type=NodeType.Model,
            unique_id='model.snowplow.model_1',
            fqn=['snowplow', 'nested', 'model_1'],
            package_name='snowplow',
            original_file_path=normalize('models/nested/model_1.sql'),
            config=ModelConfig(materialized='table'),
            path=normalize('nested/model_1.sql'),
            language='sql',
            raw_code='{{ config(materialized="table") }}select 1 as id',
            checksum=None,
            unrendered_config={'materialized': 'table'},
        )
        self.example_config: ContextConfig = ContextConfig(
            self.root_project_config, self.example_node.fqn, self.example_node.resource_type, self.snowplow_project_config
        )

    def file_block_for(self, data: str, filename: str) -> FileBlock:
        return super().file_block_for(data, filename, 'models')

    def test_config_shifting(self) -> None:
        static_parser_result: Dict[str, List[Tuple[str, Any]]] = {
            'configs': [('hello', 'world'), ('flag', True), ('tags', 'tag1'), ('tags', 'tag2')]
        }
        expected = {'hello': 'world', 'flag': True, 'tags': ['tag1', 'tag2']}
        got = _get_config_call_dict(static_parser_result)
        self.assertEqual(expected, got)

    def test_source_shifting(self) -> None:
        static_parser_result: Dict[str, List[Tuple[str, str]]] = {'sources': [('abc', 'def'), ('x', 'y')]}
        expected = {'sources': [['abc', 'def'], ['x', 'y']]}
        got = _shift_sources(static_parser_result)
        self.assertEqual(expected, got)

    def test_sample_results(self) -> None:
        node = deepcopy(self.example_node)
        config = deepcopy(self.example_config)
        sample_node = deepcopy(self.example_node)
        sample_config = deepcopy(self.example_config)
        sample_node.refs = []
        node.refs = ['myref']
        result = _get_sample_result(sample_node, sample_config, node, config)
        self.assertEqual([(7, 'missed_ref_value')], result)
        node = deepcopy(self.example_node)
        config = deepcopy(self.example_config)
        sample_node = deepcopy(self.example_node)
        sample_config = deepcopy(self.example_config)
        sample_node.refs = ['myref']
        node.refs = []
        result = _get_sample_result(sample_node, sample_config, node, config)
        self.assertEqual([(6, 'false_positive_ref_value')], result)
        node = deepcopy(self.example_node)
        config = deepcopy(self.example_config)
        sample_node = deepcopy(self.example_node)
        sample_config = deepcopy(self.example_config)
        sample_node.sources = []
        node.sources = [['abc', 'def']]
        result = _get_sample_result(sample_node, sample_config, node, config)
        self.assertEqual([(5, 'missed_source_value')], result)
        node = deepcopy(self.example_node)
        config = deepcopy(self.example_config)
        sample_node = deepcopy(self.example_node)
        sample_config = deepcopy(self.example_config)
        sample_node.sources = [['abc', 'def']]
        node.sources = []
        result = _get_sample_result(sample_node, sample_config, node, config)
        self.assertEqual([(4, 'false_positive_source_value')], result)
        node = deepcopy(self.example_node)
        config = deepcopy(self.example_config)
        sample_node = deepcopy(self.example_node)
        sample_config = deepcopy(self.example_config)
        sample_config._config_call_dict = {}
        config._config_call_dict = {'key': 'value'}
        result = _get_sample_result(sample_node, sample_config, node, config)
        self.assertEqual([(3, 'missed_config_value')], result)
        node = deepcopy(self.example_node)
        config = deepcopy(self.example_config)
        sample_node = deepcopy(self.example_node)
        sample_config = deepcopy(self.example_config)
        sample_config._config_call_dict = {'key': 'value'}
        config._config_call_dict = {}
        result = _get_sample_result(sample_node, sample_config, node, config)
        self.assertEqual([(2, 'false_positive_config_value')], result)

    def test_exp_sample_results(self) -> None:
        node = deepcopy(self.example_node)
        config = deepcopy(self.example_config)
        sample_node = deepcopy(self.example_node)
        sample_config = deepcopy(self.example_config)
        result = _get_exp_sample_result(sample_node, sample_config, node, config)
        self.assertEqual(['00_experimental_exact_match'], result)

    def test_stable_sample_results(self) -> None:
        node = deepcopy(self.example_node)
        config = deepcopy(self.example_config)
        sample_node = deepcopy(self.example_node)
        sample_config = deepcopy(self.example_config)
        result = _get_stable_sample_result(sample_node, sample_config, node, config)
        self.assertEqual(['80_stable_exact_match'], result)


class SnapshotParserTest(BaseParserTest):
    def setUp(self) -> None:
        super().setUp()
        self.parser: SnapshotParser = SnapshotParser(
            project=self.snowplow_project_config, manifest=self.manifest, root_project=self.root_project_config
        )

    def file_block_for(self, data: str, filename: str) -> FileBlock:
        return super().file_block_for(data, filename, 'snapshots')

    def test_parse_error(self) -> None:
        block = self.file_block_for('{% snapshot foo %}select 1 as id{%snapshot bar %}{% endsnapshot %}', 'nested/snap_1.sql')
        with self.assertRaises(CompilationError):
            self.parser.parse_file(block)

    def test_single_block(self) -> None:
        raw_code = '{{\n                config(unique_key="id", target_schema="analytics",\n                       target_database="dbt", strategy="timestamp",\n                       updated_at="last_update")\n            }}\n            select 1 as id, now() as last_update'
        full_file = '\n        {{% snapshot foo %}}{}{{% endsnapshot %}}\n        '.format(raw_code)
        block = self.file_block_for(full_file, 'nested/snap_1.sql')
        self.parser.manifest.files[block.file.file_id] = block.file
        self.parser.parse_file(block)
        self.assert_has_manifest_lengths(self.parser.manifest, nodes=1)
        node = list(self.parser.manifest.nodes.values())[0]
        expected = SnapshotNode(
            alias='foo',
            name='foo',
            database='dbt',
            schema='analytics',
            resource_type=NodeType.Snapshot,
            unique_id='snapshot.snowplow.foo',
            fqn=['snowplow', 'nested', 'snap_1', 'foo'],
            package_name='snowplow',
            original_file_path=normalize('snapshots/nested/snap_1.sql'),
            config=SnapshotConfig(
                strategy='timestamp',
                updated_at='last_update',
                target_database='dbt',
                target_schema='analytics',
                unique_key='id',
                materialized='snapshot',
            ),
            path=normalize('nested/snap_1.sql'),
            language='sql',
            raw_code=raw_code,
            checksum=block.file.checksum,
            unrendered_config={
                'unique_key': 'id',
                'target_schema': 'analytics',
                'target_database': 'dbt',
                'strategy': 'timestamp',
                'updated_at': 'last_update',
            },
            config_call_dict={
                'strategy': 'timestamp',
                'target_database': 'dbt',
                'target_schema': 'analytics',
                'unique_key': 'id',
                'updated_at': 'last_update',
            },
            unrendered_config_call_dict={},
        )
        assertEqualNodes(expected, node)
        file_id = 'snowplow://' + normalize('snapshots/nested/snap_1.sql')
        self.assertIn(file_id, self.parser.manifest.files)
        self.assertEqual(self.parser.manifest.files[file_id].nodes, ['snapshot.snowplow.foo'])

    def test_multi_block(self) -> None:
        raw_1 = '\n            {{\n                config(unique_key="id", target_schema="analytics",\n                       target_database="dbt", strategy="timestamp",\n                       updated_at="last_update")\n            }}\n            select 1 as id, now() as last_update\n        '
        raw_2 = '\n            {{\n                config(unique_key="id", target_schema="analytics",\n                       target_database="dbt", strategy="timestamp",\n                       updated_at="last_update")\n            }}\n            select 2 as id, now() as last_update\n        '
        full_file = '\n        {{% snapshot foo %}}{}{{% endsnapshot %}}\n        {{% snapshot bar %}}{}{{% endsnapshot %}}\n        '.format(raw_1, raw_2)
        block = self.file_block_for(full_file, 'nested/snap_1.sql')
        self.parser.manifest.files[block.file.file_id] = block.file
        self.parser.parse_file(block)
        self.assert_has_manifest_lengths(self.parser.manifest, nodes=2)
        nodes = sorted(self.parser.manifest.nodes.values(), key=lambda n: n.name)
        expect_foo = SnapshotNode(
            alias='foo',
            name='foo',
            database='dbt',
            schema='analytics',
            resource_type=NodeType.Snapshot,
            unique_id='snapshot.snowplow.foo',
            fqn=['snowplow', 'nested', 'snap_1', 'foo'],
            package_name='snowplow',
            original_file_path=normalize('snapshots/nested/snap_1.sql'),
            config=SnapshotConfig(
                strategy='timestamp',
                updated_at='last_update',
                target_database='dbt',
                target_schema='analytics',
                unique_key='id',
                materialized='snapshot',
            ),
            path=normalize('nested/snap_1.sql'),
            language='sql',
            raw_code=raw_1,
            checksum=block.file.checksum,
            unrendered_config={
                'unique_key': 'id',
                'target_schema': 'analytics',
                'target_database': 'dbt',
                'strategy': 'timestamp',
                'updated_at': 'last_update',
            },
            config_call_dict={
                'strategy': 'timestamp',
                'target_database': 'dbt',
                'target_schema': 'analytics',
                'unique_key': 'id',
                'updated_at': 'last_update',
            },
            unrendered_config_call_dict={},
        )
        expect_bar = SnapshotNode(
            alias='bar',
            name='bar',
            database='dbt',
            schema='analytics',
            resource_type=NodeType.Snapshot,
            unique_id='snapshot.snowplow.bar',
            fqn=['snowplow', 'nested', 'snap_1', 'bar'],
            package_name='snowplow',
            original_file_path=normalize('snapshots/nested/snap_1.sql'),
            config=SnapshotConfig(
                strategy='timestamp',
                updated_at='last_update',
                target_database='dbt',
                target_schema='analytics',
                unique_key='id',
                materialized='snapshot',
            ),
            path=normalize('nested/snap_1.sql'),
            language='sql',
            raw_code=raw_2,
            checksum=block.file.checksum,
            unrendered_config={
                'unique_key': 'id',
                'target_schema': 'analytics',
                'target_database': 'dbt',
                'strategy': 'timestamp',
                'updated_at': 'last_update',
            },
            config_call_dict={
                'strategy': 'timestamp',
                'target_database': 'dbt',
                'target_schema': 'analytics',
                'unique_key': 'id',
                'updated_at': 'last_update',
            },
            unrendered_config_call_dict={},
        )
        assertEqualNodes(nodes[0], expect_bar)
        assertEqualNodes(nodes[1], expect_foo)
        file_id = 'snowplow://' + normalize('snapshots/nested/snap_1.sql')
        self.assertIn(file_id, self.parser.manifest.files)
        self.assertEqual(sorted(self.parser.manifest.files[file_id].nodes), ['snapshot.snowplow.bar', 'snapshot.snowplow.foo'])


class MacroParserTest(BaseParserTest):
    def setUp(self) -> None:
        super().setUp()
        self.parser: MacroParser = MacroParser(project=self.snowplow_project_config, manifest=Manifest())

    def file_block_for(self, data: str, filename: str) -> FileBlock:
        return super().file_block_for(data, filename, 'macros')

    def test_single_block(self) -> None:
        raw_code = '{% macro foo(a, b) %}a ~ b{% endmacro %}'
        block = self.file_block_for(raw_code, 'macro.sql')
        self.parser.manifest.files[block.file.file_id] = block.file
        self.parser.parse_file(block)
        self.assertEqual(len(self.parser.manifest.macros), 1)
        macro = list(self.parser.manifest.macros.values())[0]
        expected = Macro(
            name='foo',
            resource_type=NodeType.Macro,
            unique_id='macro.snowplow.foo',
            package_name='snowplow',
            original_file_path=normalize('macros/macro.sql'),
            path=normalize('macros/macro.sql'),
            macro_sql=raw_code,
        )
        assertEqualNodes(macro, expected)
        file_id = 'snowplow://' + normalize('macros/macro.sql')
        self.assertIn(file_id, self.parser.manifest.files)
        self.assertEqual(self.parser.manifest.files[file_id].macros, ['macro.snowplow.foo'])

    def test_multiple_blocks(self) -> None:
        raw_code = '{% macro foo(a, b) %}a ~ b{% endmacro %}\n{% macro bar(c, d) %}c + d{% endmacro %}'
        block = self.file_block_for(raw_code, 'macro.sql')
        self.parser.manifest.files[block.file.file_id] = block.file
        self.parser.parse_file(block)
        self.assertEqual(len(self.parser.manifest.macros), 2)
        macros = sorted(self.parser.manifest.macros.values(), key=lambda m: m.name)
        expected_bar = Macro(
            name='bar',
            resource_type=NodeType.Macro,
            unique_id='macro.snowplow.bar',
            package_name='snowplow',
            original_file_path=normalize('macros/macro.sql'),
            path=normalize('macros/macro.sql'),
            macro_sql='{% macro bar(c, d) %}c + d{% endmacro %}',
        )
        expected_foo = Macro(
            name='foo',
            resource_type=NodeType.Macro,
            unique_id='macro.snowplow.foo',
            package_name='snowplow',
            original_file_path=normalize('macros/macro.sql'),
            path=normalize('macros/macro.sql'),
            macro_sql='{% macro foo(a, b) %}a ~ b{% endmacro %}',
        )
        assertEqualNodes(macros[0], expected_bar)
        assertEqualNodes(macros[1], expected_foo)
        file_id = 'snowplow://' + normalize('macros/macro.sql')
        self.assertIn(file_id, self.parser.manifest.files)
        self.assertEqual(sorted(self.parser.manifest.files[file_id].macros), ['macro.snowplow.bar', 'macro.snowplow.foo'])


class SingularTestParserTest(BaseParserTest):
    def setUp(self) -> None:
        super().setUp()
        self.parser: SingularTestParser = SingularTestParser(
            project=self.snowplow_project_config, manifest=self.manifest, root_project=self.root_project_config
        )

    def file_block_for(self, data: str, filename: str) -> FileBlock:
        return super().file_block_for(data, filename, 'tests')

    def test_basic(self) -> None:
        raw_code = 'select * from {{ ref("blah") }} limit 0'
        block = self.file_block_for(raw_code, 'test_1.sql')
        self.manifest.files[block.file.file_id] = block.file
        self.parser.parse_file(block)
        self.assert_has_manifest_lengths(self.parser.manifest, nodes=1)
        node = list(self.parser.manifest.nodes.values())[0]
        expected = SingularTestNode(
            alias='test_1',
            name='test_1',
            database='test',
            schema='dbt_test__audit',
            resource_type=NodeType.Test,
            unique_id='test.snowplow.test_1',
            fqn=['snowplow', 'test_1'],
            package_name='snowplow',
            original_file_path=normalize('tests/test_1.sql'),
            refs=[RefArgs(name='blah')],
            config=TestConfig(severity='ERROR'),
            tags=[],
            path=normalize('test_1.sql'),
            language='sql',
            raw_code=raw_code,
            checksum=block.file.checksum,
            unrendered_config={},
        )
        assertEqualNodes(node, expected)
        file_id = 'snowplow://' + normalize('tests/test_1.sql')
        self.assertIn(file_id, self.parser.manifest.files)
        self.assertEqual(self.parser.manifest.files[file_id].nodes, ['test.snowplow.test_1'])


class GenericTestParserTest(BaseParserTest):
    def setUp(self) -> None:
        super().setUp()
        self.parser: GenericTestParser = GenericTestParser(project=self.snowplow_project_config, manifest=Manifest())

    def file_block_for(self, data: str, filename: str) -> FileBlock:
        return super().file_block_for(data, filename, 'tests/generic')

    def test_basic(self) -> None:
        raw_code = '{% test not_null(model, column_name) %}select * from {{ model }} where {{ column_name }} is null {% endtest %}'
        block = self.file_block_for(raw_code, 'test_1.sql')
        self.parser.manifest.files[block.file.file_id] = block.file
        self.parser.parse_file(block)
        node = list(self.parser.manifest.macros.values())[0]
        expected = Macro(
            name='test_not_null',
            resource_type=NodeType.Macro,
            unique_id='macro.snowplow.test_not_null',
            package_name='snowplow',
            original_file_path=normalize('tests/generic/test_1.sql'),
            path=normalize('tests/generic/test_1.sql'),
            macro_sql=raw_code,
        )
        assertEqualNodes(node, expected)
        file_id = 'snowplow://' + normalize('tests/generic/test_1.sql')
        self.assertIn(file_id, self.parser.manifest.files)
        self.assertEqual(self.parser.manifest.files[file_id].macros, ['macro.snowplow.test_not_null'])


class AnalysisParserTest(BaseParserTest):
    def setUp(self) -> None:
        super().setUp()
        self.parser: AnalysisParser = AnalysisParser(
            project=self.snowplow_project_config, manifest=self.manifest, root_project=self.root_project_config
        )

    def file_block_for(self, data: str, filename: str) -> FileBlock:
        return super().file_block_for(data, filename, 'analyses')

    def test_basic(self) -> None:
        raw_code = 'select 1 as id'
        block = self.file_block_for(raw_code, 'nested/analysis_1.sql')
        self.manifest.files[block.file.file_id] = block.file
        self.parser.parse_file(block)
        self.assert_has_manifest_lengths(self.parser.manifest, nodes=1)
        node = list(self.parser.manifest.nodes.values())[0]
        expected = AnalysisNode(
            alias='analysis_1',
            name='analysis_1',
            database='test',
            schema='analytics',
            resource_type=NodeType.Analysis,
            unique_id='analysis.snowplow.analysis_1',
            fqn=['snowplow', 'analysis', 'nested', 'analysis_1'],
            package_name='snowplow',
            original_file_path=normalize('analyses/nested/analysis_1.sql'),
            depends_on=DependsOn(),
            config=NodeConfig(),
            path=normalize('analysis/nested/analysis_1.sql'),
            language='sql',
            raw_code=raw_code,
            checksum=block.file.checksum,
            unrendered_config={},
            relation_name=None,
        )
        assertEqualNodes(node, expected)
        file_id = 'snowplow://' + normalize('analyses/nested/analysis_1.sql')
        self.assertIn(file_id, self.parser.manifest.files)
        self.assertEqual(self.parser.manifest.files[file_id].nodes, ['analysis.snowplow.analysis_1'])
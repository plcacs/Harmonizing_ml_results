from typing import Any, Dict, Generator, List
import os
import pytest
import yaml
from dbt.exceptions import ParsingError
from dbt.tests.util import check_relations_equal, check_table_does_not_exist, run_dbt, update_config_file
from tests.functional.sources.common_source_setup import BaseSourcesTest
from tests.functional.sources.fixtures import (
    macros_macro_sql,
    malformed_models_descendant_model_sql,
    malformed_models_schema_yml,
    malformed_schema_tests_model_sql,
    malformed_schema_tests_schema_yml,
)


class SuccessfulSourcesTest(BaseSourcesTest):
    @pytest.fixture(scope='class', autouse=True)
    def setUp(self, project: Any) -> Generator[None, None, None]:
        self.run_dbt_with_vars(project, ['seed'])
        os.environ['DBT_ENV_CUSTOM_ENV_key'] = 'value'
        yield
        del os.environ['DBT_ENV_CUSTOM_ENV_key']

    @pytest.fixture(scope='class')
    def macros(self) -> Dict[str, Any]:
        return {'macro.sql': macros_macro_sql}

    def _create_schemas(self, project: Any) -> None:
        schema: str = self.alternative_schema(project.test_schema)
        project.run_sql(f'drop schema if exists {schema} cascade')
        project.run_sql(f'create schema {schema}')

    def alternative_schema(self, test_schema: str) -> str:
        return test_schema + '_other'

    @pytest.fixture(scope='class', autouse=True)
    def createDummyTables(self, project: Any) -> None:
        self._create_schemas(project)
        project.run_sql('create table {}.dummy_table (id int)'.format(project.test_schema))
        project.run_sql('create view {}.external_view as (select * from {}.dummy_table)'.format(
            self.alternative_schema(project.test_schema), project.test_schema))

    def run_dbt_with_vars(self, project: Any, cmd: List[str], *args: Any, **kwargs: Any) -> List[Any]:
        vars_dict: Dict[str, Any] = {
            'test_run_schema': project.test_schema,
            'test_run_alt_schema': self.alternative_schema(project.test_schema),
            'test_loaded_at': project.adapter.quote('updated_at')
        }
        cmd.extend(['--vars', yaml.safe_dump(vars_dict)])
        return run_dbt(cmd, *args, **kwargs)


class TestBasicSource(SuccessfulSourcesTest):
    def test_basic_source_def(self, project: Any) -> None:
        results: List[Any] = self.run_dbt_with_vars(project, ['run'])
        assert len(results) == 4
        check_relations_equal(project.adapter, ['source', 'descendant_model', 'nonsource_descendant'])
        check_relations_equal(project.adapter, ['expected_multi_source', 'multi_source_model'])
        results = self.run_dbt_with_vars(project, ['test'])
        assert len(results) == 8


class TestSourceSelector(SuccessfulSourcesTest):
    def test_source_selector(self, project: Any) -> None:
        results: List[Any] = self.run_dbt_with_vars(project, ['run', '--models', 'source:test_source.test_table+'])
        assert len(results) == 1
        check_relations_equal(project.adapter, ['source', 'descendant_model'])
        check_table_does_not_exist(project.adapter, 'nonsource_descendant')
        check_table_does_not_exist(project.adapter, 'multi_source_model')
        results = self.run_dbt_with_vars(project, ['run', '--models', 'tag:my_test_source_table_tag+'])
        assert len(results) == 1
        results = self.run_dbt_with_vars(project, ['test', '--models', 'source:test_source.test_table+'])
        assert len(results) == 6
        results = self.run_dbt_with_vars(project, ['test', '--models', 'tag:my_test_source_table_tag+'])
        assert len(results) == 6
        results = self.run_dbt_with_vars(project, ['test', '--models', 'tag:my_test_source_tag+'])
        assert len(results) == 8
        results = self.run_dbt_with_vars(project, ['test', '--models', 'tag:id_column'])
        assert len(results) == 4


class TestEmptySource(SuccessfulSourcesTest):
    def test_empty_source_def(self, project: Any) -> None:
        results: List[Any] = self.run_dbt_with_vars(project, ['run', '--models', 'source:test_source.test_table'])
        check_table_does_not_exist(project.adapter, 'nonsource_descendant')
        check_table_does_not_exist(project.adapter, 'multi_source_model')
        check_table_does_not_exist(project.adapter, 'descendant_model')
        assert len(results) == 0


class TestSourceDef(SuccessfulSourcesTest):
    def test_source_only_def(self, project: Any) -> None:
        results: List[Any] = self.run_dbt_with_vars(project, ['run', '--models', 'source:other_source+'])
        assert len(results) == 1
        check_relations_equal(project.adapter, ['expected_multi_source', 'multi_source_model'])
        check_table_does_not_exist(project.adapter, 'nonsource_descendant')
        check_table_does_not_exist(project.adapter, 'descendant_model')
        results = self.run_dbt_with_vars(project, ['run', '--models', 'source:test_source+'])
        assert len(results) == 2
        check_relations_equal(project.adapter, ['source', 'descendant_model'])
        check_relations_equal(project.adapter, ['expected_multi_source', 'multi_source_model'])
        check_table_does_not_exist(project.adapter, 'nonsource_descendant')


class TestSourceChildrenParents(SuccessfulSourcesTest):
    def test_source_childrens_parents(self, project: Any) -> None:
        results: List[Any] = self.run_dbt_with_vars(project, ['run', '--models', '@source:test_source'])
        assert len(results) == 2
        check_relations_equal(project.adapter, ['source', 'descendant_model'])
        check_relations_equal(project.adapter, ['expected_multi_source', 'multi_source_model'])
        check_table_does_not_exist(project.adapter, 'nonsource_descendant')


class TestSourceRunOperation(SuccessfulSourcesTest):
    def test_run_operation_source(self, project: Any) -> None:
        kwargs: str = '{"source_name": "test_source", "table_name": "test_table"}'
        self.run_dbt_with_vars(project, ['run-operation', 'vacuum_source', '--args', kwargs])


class TestMalformedSources(BaseSourcesTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, Any]:
        return {'schema.yml': malformed_models_schema_yml, 'descendant_model.sql': malformed_models_descendant_model_sql}

    def test_malformed_schema_will_break_run(self, project: Any) -> None:
        with pytest.raises(ParsingError):
            self.run_dbt_with_vars(project, ['seed'])


class TestRenderingInSourceTests(BaseSourcesTest):
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, Any]:
        return {'schema.yml': malformed_schema_tests_schema_yml, 'model.sql': malformed_schema_tests_model_sql}

    def test_render_in_source_tests(self, project: Any) -> None:
        self.run_dbt_with_vars(project, ['seed'])
        self.run_dbt_with_vars(project, ['run'])
        self.run_dbt_with_vars(project, ['test'], expect_pass=False)


class TestUnquotedSources(SuccessfulSourcesTest):
    def test_catalog(self, project: Any) -> None:
        new_quoting_config: Dict[str, Any] = {'quoting': {'identifier': False, 'schema': False, 'database': False}}
        update_config_file(new_quoting_config, project.project_root, 'dbt_project.yml')
        self.run_dbt_with_vars(project, ['run'])
        self.run_dbt_with_vars(project, ['docs', 'generate'])
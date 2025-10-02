import re
import pytest
from typing import Dict, List, Tuple, Any, Optional, Union
from dbt.tests.util import get_manifest, read_file, relation_from_name, run_dbt, run_dbt_and_capture, write_file
from tests.functional.adapter.constraints.fixtures import constrained_model_schema_yml, create_table_macro_sql, foreign_key_model_sql, incremental_foreign_key_model_raw_numbers_sql, incremental_foreign_key_model_stg_numbers_sql, incremental_foreign_key_schema_yml, model_contract_header_schema_yml, model_data_type_schema_yml, model_fk_constraint_schema_yml, model_quoted_column_schema_yml, model_schema_yml, my_incremental_model_sql, my_model_contract_sql_header_sql, my_model_data_type_sql, my_model_incremental_contract_sql_header_sql, my_model_incremental_with_nulls_sql, my_model_incremental_wrong_name_sql, my_model_incremental_wrong_order_depends_on_fk_sql, my_model_incremental_wrong_order_sql, my_model_sql, my_model_view_wrong_name_sql, my_model_view_wrong_order_sql, my_model_with_nulls_sql, my_model_with_quoted_column_name_sql, my_model_wrong_name_sql, my_model_wrong_order_depends_on_fk_sql, my_model_wrong_order_sql

class BaseConstraintsColumnsEqual:
    """
    dbt should catch these mismatches during its "preflight" checks.
    """

    @pytest.fixture
    def string_type(self) -> str:
        return 'TEXT'

    @pytest.fixture
    def int_type(self) -> str:
        return 'INT'

    @pytest.fixture
    def schema_string_type(self, string_type: str) -> str:
        return string_type

    @pytest.fixture
    def schema_int_type(self, int_type: str) -> str:
        return int_type

    @pytest.fixture
    def data_types(self, schema_int_type: str, int_type: str, string_type: str) -> List[List[str]]:
        return [['1', schema_int_type, int_type], ["'1'", string_type, string_type], ['true', 'bool', 'BOOL'], ["'2013-11-03 00:00:00-07'::timestamptz", 'timestamptz', 'DATETIMETZ'], ["'2013-11-03 00:00:00-07'::timestamp", 'timestamp', 'DATETIME'], ["ARRAY['a','b','c']", 'text[]', 'STRINGARRAY'], ['ARRAY[1,2,3]', 'int[]', 'INTEGERARRAY'], ["'1'::numeric", 'numeric', 'DECIMAL'], ['\'{"bar": "baz", "balance": 7.77, "active": false}\'::json', 'json', 'JSON']]

    def test__constraints_wrong_column_order(self, project: Any) -> None:
        run_dbt(['run', '-s', 'my_model_wrong_order'], expect_pass=True)
        manifest = get_manifest(project.project_root)
        model_id = 'model.test.my_model_wrong_order'
        my_model_config = manifest.nodes[model_id].config
        contract_actual_config = my_model_config.contract
        assert contract_actual_config.enforced is True

    def test__constraints_wrong_column_names(self, project: Any, string_type: str, int_type: str) -> None:
        _, log_output = run_dbt_and_capture(['run', '-s', 'my_model_wrong_name'], expect_pass=False)
        manifest = get_manifest(project.project_root)
        model_id = 'model.test.my_model_wrong_name'
        my_model_config = manifest.nodes[model_id].config
        contract_actual_config = my_model_config.contract
        assert contract_actual_config.enforced is True
        expected = ['id', 'error', 'missing in definition', 'missing in contract']
        assert all([exp in log_output or exp.upper() in log_output for exp in expected])

    def test__constraints_wrong_column_data_types(self, project: Any, string_type: str, int_type: str, schema_string_type: str, schema_int_type: str, data_types: List[List[str]]) -> None:
        for sql_column_value, schema_data_type, error_data_type in data_types:
            write_file(my_model_data_type_sql.format(sql_value=sql_column_value), 'models', 'my_model_data_type.sql')
            wrong_schema_data_type = schema_int_type if schema_data_type.upper() != schema_int_type.upper() else schema_string_type
            wrong_schema_error_data_type = int_type if schema_data_type.upper() != schema_int_type.upper() else string_type
            write_file(model_data_type_schema_yml.format(data_type=wrong_schema_data_type), 'models', 'constraints_schema.yml')
            results, log_output = run_dbt_and_capture(['run', '-s', 'my_model_data_type'], expect_pass=False)
            manifest = get_manifest(project.project_root)
            model_id = 'model.test.my_model_data_type'
            my_model_config = manifest.nodes[model_id].config
            contract_actual_config = my_model_config.contract
            assert contract_actual_config.enforced is True
            expected = ['wrong_data_type_column_name', error_data_type, wrong_schema_error_data_type, 'data type mismatch']
            assert all([exp in log_output or exp.upper() in log_output for exp in expected])

    def test__constraints_correct_column_data_types(self, project: Any, data_types: List[List[str]]) -> None:
        for sql_column_value, schema_data_type, _ in data_types:
            write_file(my_model_data_type_sql.format(sql_value=sql_column_value), 'models', 'my_model_data_type.sql')
            write_file(model_data_type_schema_yml.format(data_type=schema_data_type), 'models', 'constraints_schema.yml')
            run_dbt(['run', '-s', 'my_model_data_type'])
            manifest = get_manifest(project.project_root)
            model_id = 'model.test.my_model_data_type'
            my_model_config = manifest.nodes[model_id].config
            contract_actual_config = my_model_config.contract
            assert contract_actual_config.enforced is True

def _normalize_whitespace(input: str) -> str:
    subbed = re.sub('\\s+', ' ', input)
    return re.sub('\\s?([\\(\\),])\\s?', '\\1', subbed).lower().strip()

def _find_and_replace(sql: str, find: str, replace: str) -> str:
    sql_tokens = sql.split()
    for idx in [n for n, x in enumerate(sql_tokens) if find in x]:
        sql_tokens[idx] = replace
    return ' '.join(sql_tokens)

class BaseConstraintsRuntimeDdlEnforcement:
    """
    These constraints pass muster for dbt's preflight checks. Make sure they're
    passed into the DDL statement. If they don't match up with the underlying data,
    the data platform should raise an error at runtime.
    """

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'my_model.sql': my_model_wrong_order_depends_on_fk_sql, 'foreign_key_model.sql': foreign_key_model_sql, 'constraints_schema.yml': model_fk_constraint_schema_yml}

    @pytest.fixture(scope='class')
    def expected_sql(self) -> str:
        return "\ncreate table <model_identifier> (\n    id integer not null primary key check ((id > 0)) check (id >= 1) references <foreign_key_model_identifier> (id) unique,\n    color text,\n    date_day text\n) ;\ninsert into <model_identifier> (\n    id ,\n    color ,\n    date_day\n)\n(\n    select\n       id,\n       color,\n       date_day\n       from\n    (\n        -- depends_on: <foreign_key_model_identifier>\n        select\n            'blue' as color,\n            1 as id,\n            '2019-01-01' as date_day\n    ) as model_subq\n);\n"

    def test__constraints_ddl(self, project: Any, expected_sql: str) -> None:
        unformatted_constraint_schema_yml = read_file('models', 'constraints_schema.yml')
        write_file(unformatted_constraint_schema_yml.format(schema=project.test_schema), 'models', 'constraints_schema.yml')
        results = run_dbt(['run', '-s', '+my_model'])
        assert len(results) >= 1
        generated_sql = read_file('target', 'run', 'test', 'models', 'my_model.sql')
        generated_sql_generic = _find_and_replace(generated_sql, 'my_model', '<model_identifier>')
        generated_sql_generic = _find_and_replace(generated_sql_generic, 'foreign_key_model', '<foreign_key_model_identifier>')
        assert _normalize_whitespace(expected_sql) == _normalize_whitespace(generated_sql_generic)

class BaseConstraintsRollback:

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'my_model.sql': my_model_sql, 'constraints_schema.yml': model_schema_yml}

    @pytest.fixture(scope='class')
    def null_model_sql(self) -> str:
        return my_model_with_nulls_sql

    @pytest.fixture(scope='class')
    def expected_color(self) -> str:
        return 'blue'

    @pytest.fixture(scope='class')
    def expected_error_messages(self) -> List[str]:
        return ['null value in column "id"', 'violates not-null constraint']

    def assert_expected_error_messages(self, error_message: str, expected_error_messages: List[str]) -> None:
        assert all((msg in error_message for msg in expected_error_messages))

    def test__constraints_enforcement_rollback(self, project: Any, expected_color: str, expected_error_messages: List[str], null_model_sql: str) -> None:
        results = run_dbt(['run', '-s', 'my_model'])
        assert len(results) == 1
        write_file(null_model_sql, 'models', 'my_model.sql')
        failing_results = run_dbt(['run', '-s', 'my_model'], expect_pass=False)
        assert len(failing_results) == 1
        relation = relation_from_name(project.adapter, 'my_model')
        old_model_exists_sql = f'select * from {relation}'
        old_model_exists = project.run_sql(old_model_exists_sql, fetch='all')
        assert len(old_model_exists) == 1
        assert old_model_exists[0][1] == expected_color
        manifest = get_manifest(project.project_root)
        model_id = 'model.test.my_model'
        my_model_config = manifest.nodes[model_id].config
        contract_actual_config = my_model_config.contract
        assert contract_actual_config.enforced is True
        self.assert_expected_error_messages(failing_results[0].message, expected_error_messages)

class BaseTableConstraintsColumnsEqual(BaseConstraintsColumnsEqual):

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'my_model_wrong_order.sql': my_model_wrong_order_sql, 'my_model_wrong_name.sql': my_model_wrong_name_sql, 'constraints_schema.yml': model_schema_yml}

class BaseViewConstraintsColumnsEqual(BaseConstraintsColumnsEqual):

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'my_model_wrong_order.sql': my_model_view_wrong_order_sql, 'my_model_wrong_name.sql': my_model_view_wrong_name_sql, 'constraints_schema.yml': model_schema_yml}

class BaseIncrementalConstraintsColumnsEqual(BaseConstraintsColumnsEqual):

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'my_model_wrong_order.sql': my_model_incremental_wrong_order_sql, 'my_model_wrong_name.sql': my_model_incremental_wrong_name_sql, 'constraints_schema.yml': model_schema_yml}

class BaseIncrementalConstraintsRuntimeDdlEnforcement(BaseConstraintsRuntimeDdlEnforcement):

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'my_model.sql': my_model_incremental_wrong_order_depends_on_fk_sql, 'foreign_key_model.sql': foreign_key_model_sql, 'constraints_schema.yml': model_fk_constraint_schema_yml}

class BaseIncrementalConstraintsRollback(BaseConstraintsRollback):

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'my_model.sql': my_incremental_model_sql, 'constraints_schema.yml': model_schema_yml}

    @pytest.fixture(scope='class')
    def null_model_sql(self) -> str:
        return my_model_incremental_with_nulls_sql

class TestTableConstraintsColumnsEqual(BaseTableConstraintsColumnsEqual):
    pass

class TestViewConstraintsColumnsEqual(BaseViewConstraintsColumnsEqual):
    pass

class TestIncrementalConstraintsColumnsEqual(BaseIncrementalConstraintsColumnsEqual):
    pass

class TestTableConstraintsRuntimeDdlEnforcement(BaseConstraintsRuntimeDdlEnforcement):
    pass

class TestTableConstraintsRollback(BaseConstraintsRollback):
    pass

class TestIncrementalConstraintsRuntimeDdlEnforcement(BaseIncrementalConstraintsRuntimeDdlEnforcement):
    pass

class TestIncrementalConstraintsRollback(BaseIncrementalConstraintsRollback):
    pass

class BaseContractSqlHeader:
    """Tests a contracted model with a sql header dependency."""

    def test__contract_sql_header(self, project: Any) -> None:
        run_dbt(['run', '-s', 'my_model_contract_sql_header'])
        manifest = get_manifest(project.project_root)
        model_id = 'model.test.my_model_contract_sql_header'
        model_config = manifest.nodes[model_id].config
        assert model_config.contract.enforced

class BaseTableContractSqlHeader(BaseContractSqlHeader):

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'my_model_contract_sql_header.sql': my_model_contract_sql_header_sql, 'constraints_schema.yml': model_contract_header_schema_yml}

class BaseIncrementalContractSqlHeader(BaseContractSqlHeader):

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'my_model_contract_sql_header.sql': my_model_incremental_contract_sql_header_sql, 'constraints_schema.yml': model_contract_header_schema_yml}

class TestTableContractSqlHeader(BaseTableContractSqlHeader):
    pass

class TestIncrementalContractSqlHeader(BaseIncrementalContractSqlHeader):
    pass

class BaseModelConstraintsRuntimeEnforcement:
    """
    These model-level constraints pass muster for dbt's preflight checks. Make sure they're
    passed into the DDL statement. If they don't match up with the underlying data,
    the data platform should raise an error at runtime.
    """

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'my_model.sql': my_model_wrong_order_depends_on_fk_sql, 'foreign_key_model.sql': foreign_key_model_sql, 'constraints_schema.yml': constrained_model_schema_yml}

    @pytest.fixture(scope='class')
    def expected_sql(self) -> str:
        return "\ncreate table <model_identifier> (\n    id integer not null,\n    color text,\n    date_day text,\n    check ((id > 0)),\n    check (id >= 1),\n    primary key (id),\n    constraint strange_uniqueness_requirement unique (color, date_day),\n    foreign key (id) references <foreign_key_model_identifier> (id)\n) ;\ninsert into <model_identifier> (\n    id ,\n    color ,\n    date_day\n)\n(\n    select\n       id,\n       color,\n       date_day\n       from\n    (\n        -- depends_on: <foreign_key_model_identifier>\n        select\n            'blue' as color,\n            1 as id,\n            '2019-01-01' as date_day\n    ) as model_subq\n);\n"

    def test__model_constraints_ddl(self, project: Any, expected_sql: str) -> None:
        unformatted_constraint_schema_yml = read_file('models', 'constraints_schema.yml')
        write_file(unformatted_constraint_schema_yml.format(schema=project.test_schema), 'models', 'constraints_schema.yml')
        results = run_dbt(['run', '-s', '+my_model'])
        assert len(results) >= 1
        generated_sql = read_file('target', 'run', 'test', 'models', 'my_model.sql')
        generated_sql_generic = _find_and_replace(generated_sql, 'my_model', '<model_identifier>')
        generated_sql_generic = _find_and_replace(generated_sql_generic, 'foreign_key_model', '<foreign_key_model_identifier>')
        assert _normalize_whitespace(expected_sql) == _normalize_whitespace(generated_sql_generic)

class TestModelConstraintsRuntimeEnforcement(BaseModelConstraintsRuntimeEnforcement):
    pass

class BaseConstraintQuotedColumn(BaseConstraintsRuntimeDdlEnforcement):

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'my_model.sql': my_model_with_quoted_column_name_sql, 'constraints_schema.yml': model_quoted_column_schema_yml}

    @pytest.fixture(scope='class')
    def expected_sql(self) -> str:
        return '\ncreate table <model_identifier> (\n    id integer not null,\n    "from" text not null,\n    date_day text,\n    check (("from" = \'blue\'))\n) ;\ninsert into <model_identifier> (\n    id, "from", date_day\n)\n(\n    select id, "from", date_day\n    from (\n        select\n          \'blue\' as "from",\n          1 as id,\n          \'2019-01-01\' as date_day\n    ) as model_subq\n);\n'

class TestConstraintQuotedColumn(BaseConstraintQuotedColumn):
    pass

class TestIncrementalForeignKeyConstraint:

    @pytest.fixture(scope='class')
    def macros(self) -> Dict[str, str]:
        return {'create_table.sql': create_table_macro_sql}

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'schema.yml': incremental_foreign_key_schema_yml, 'raw_numbers.sql': incremental_foreign_key_model_raw_numbers_sql, 'stg_numbers.sql': incremental_foreign_key_model_stg_numbers_sql}

    def test_incremental_foreign_key_constraint(self, project: Any) -> None:
        unformatted_constraint_schema_yml = read_file('models', 'schema.yml')
        write_file(unformatted_constraint_schema_yml.format(schema=project.test_schema), 'models', 'schema.yml')
        run_dbt(['run', '--select', 'raw_numbers'])
        run_dbt(['run', '--select', 'stg_numbers'])
        run_dbt(['run', '--select', 'stg_numbers'])

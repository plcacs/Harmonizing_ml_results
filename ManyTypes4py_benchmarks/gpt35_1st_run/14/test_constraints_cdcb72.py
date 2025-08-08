import re
import pytest
from dbt.tests.util import get_manifest, read_file, relation_from_name, run_dbt, run_dbt_and_capture, write_file
from tests.functional.adapter.constraints.fixtures import constrained_model_schema_yml, create_table_macro_sql, foreign_key_model_sql, incremental_foreign_key_model_raw_numbers_sql, incremental_foreign_key_model_stg_numbers_sql, incremental_foreign_key_schema_yml, model_contract_header_schema_yml, model_data_type_schema_yml, model_fk_constraint_schema_yml, model_quoted_column_schema_yml, model_schema_yml, my_incremental_model_sql, my_model_contract_sql_header_sql, my_model_data_type_sql, my_model_incremental_contract_sql_header_sql, my_model_incremental_with_nulls_sql, my_model_incremental_wrong_name_sql, my_model_incremental_wrong_order_depends_on_fk_sql, my_model_incremental_wrong_order_sql, my_model_sql, my_model_view_wrong_name_sql, my_model_view_wrong_order_sql, my_model_with_nulls_sql, my_model_with_quoted_column_name_sql, my_model_wrong_name_sql, my_model_wrong_order_depends_on_fk_sql, my_model_wrong_order_sql

class BaseConstraintsColumnsEqual:
    def test__constraints_wrong_column_order(self, project: Any) -> None:
    def test__constraints_wrong_column_names(self, project: Any, string_type: str, int_type: str) -> None:
    def test__constraints_wrong_column_data_types(self, project: Any, string_type: str, int_type: str, schema_string_type: str, schema_int_type: str, data_types: List[List[str]]) -> None:
    def test__constraints_correct_column_data_types(self, project: Any, data_types: List[List[str]]) -> None:

class BaseConstraintsRuntimeDdlEnforcement:
    def test__constraints_ddl(self, project: Any, expected_sql: str) -> None:

class BaseConstraintsRollback:
    def assert_expected_error_messages(self, error_message: str, expected_error_messages: List[str]) -> None:
    def test__constraints_enforcement_rollback(self, project: Any, expected_color: str, expected_error_messages: List[str], null_model_sql: str) -> None:

class BaseTableConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
    pass

class BaseViewConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
    pass

class BaseIncrementalConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
    pass

class BaseIncrementalConstraintsRuntimeDdlEnforcement(BaseConstraintsRuntimeDdlEnforcement):
    pass

class BaseIncrementalConstraintsRollback(BaseConstraintsRollback):
    pass

class BaseContractSqlHeader:
    def test__contract_sql_header(self, project: Any) -> None:

class BaseTableContractSqlHeader(BaseContractSqlHeader):
    pass

class BaseIncrementalContractSqlHeader(BaseContractSqlHeader):
    pass

class BaseModelConstraintsRuntimeEnforcement:
    def test__model_constraints_ddl(self, project: Any, expected_sql: str) -> None:

class BaseConstraintQuotedColumn(BaseConstraintsRuntimeDdlEnforcement):
    pass

class TestConstraintQuotedColumn(BaseConstraintQuotedColumn):
    pass

class TestIncrementalForeignKeyConstraint:
    def test_incremental_foreign_key_constraint(self, project: Any) -> None:

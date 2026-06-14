from typing import Any, Dict, List

import re
import pytest
from dbt.tests.util import (
    get_manifest,
    read_file,
    relation_from_name,
    run_dbt,
    run_dbt_and_capture,
    write_file,
)
from tests.functional.adapter.constraints.fixtures import (
    constrained_model_schema_yml,
    create_table_macro_sql,
    foreign_key_model_sql,
    incremental_foreign_key_model_raw_numbers_sql,
    incremental_foreign_key_model_stg_numbers_sql,
    incremental_foreign_key_schema_yml,
    model_contract_header_schema_yml,
    model_data_type_schema_yml,
    model_fk_constraint_schema_yml,
    model_quoted_column_schema_yml,
    model_schema_yml,
    my_incremental_model_sql,
    my_model_contract_sql_header_sql,
    my_model_data_type_sql,
    my_model_incremental_contract_sql_header_sql,
    my_model_incremental_with_nulls_sql,
    my_model_incremental_wrong_name_sql,
    my_model_incremental_wrong_order_depends_on_fk_sql,
    my_model_incremental_wrong_order_sql,
    my_model_sql,
    my_model_view_wrong_name_sql,
    my_model_view_wrong_order_sql,
    my_model_with_nulls_sql,
    my_model_with_quoted_column_name_sql,
    my_model_wrong_name_sql,
    my_model_wrong_order_depends_on_fk_sql,
    my_model_wrong_order_sql,
)


class BaseConstraintsColumnsEqual:
    @pytest.fixture
    def string_type(self) -> str: ...
    @pytest.fixture
    def int_type(self) -> str: ...
    @pytest.fixture
    def schema_string_type(self, string_type: str) -> str: ...
    @pytest.fixture
    def schema_int_type(self, int_type: str) -> str: ...
    @pytest.fixture
    def data_types(self, schema_int_type: str, int_type: str, string_type: str) -> List[List[str]]: ...
    def test__constraints_wrong_column_order(self, project: Any) -> None: ...
    def test__constraints_wrong_column_names(self, project: Any, string_type: str, int_type: str) -> None: ...
    def test__constraints_wrong_column_data_types(
        self,
        project: Any,
        string_type: str,
        int_type: str,
        schema_string_type: str,
        schema_int_type: str,
        data_types: List[List[str]],
    ) -> None: ...
    def test__constraints_correct_column_data_types(self, project: Any, data_types: List[List[str]]) -> None: ...


def _normalize_whitespace(input: str) -> str: ...
def _find_and_replace(sql: str, find: str, replace: str) -> str: ...


class BaseConstraintsRuntimeDdlEnforcement:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...
    @pytest.fixture(scope="class")
    def expected_sql(self) -> str: ...
    def test__constraints_ddl(self, project: Any, expected_sql: str) -> None: ...


class BaseConstraintsRollback:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...
    @pytest.fixture(scope="class")
    def null_model_sql(self) -> str: ...
    @pytest.fixture(scope="class")
    def expected_color(self) -> str: ...
    @pytest.fixture(scope="class")
    def expected_error_messages(self) -> List[str]: ...
    def assert_expected_error_messages(self, error_message: str, expected_error_messages: List[str]) -> None: ...
    def test__constraints_enforcement_rollback(
        self,
        project: Any,
        expected_color: str,
        expected_error_messages: List[str],
        null_model_sql: str,
    ) -> None: ...


class BaseTableConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...


class BaseViewConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...


class BaseIncrementalConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...


class BaseIncrementalConstraintsRuntimeDdlEnforcement(BaseConstraintsRuntimeDdlEnforcement):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...


class BaseIncrementalConstraintsRollback(BaseConstraintsRollback):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...
    @pytest.fixture(scope="class")
    def null_model_sql(self) -> str: ...


class TestTableConstraintsColumnsEqual(BaseTableConstraintsColumnsEqual): ...
class TestViewConstraintsColumnsEqual(BaseViewConstraintsColumnsEqual): ...
class TestIncrementalConstraintsColumnsEqual(BaseIncrementalConstraintsColumnsEqual): ...
class TestTableConstraintsRuntimeDdlEnforcement(BaseConstraintsRuntimeDdlEnforcement): ...
class TestTableConstraintsRollback(BaseConstraintsRollback): ...
class TestIncrementalConstraintsRuntimeDdlEnforcement(BaseIncrementalConstraintsRuntimeDdlEnforcement): ...
class TestIncrementalConstraintsRollback(BaseIncrementalConstraintsRollback): ...


class BaseContractSqlHeader:
    def test__contract_sql_header(self, project: Any) -> None: ...


class BaseTableContractSqlHeader(BaseContractSqlHeader):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...


class BaseIncrementalContractSqlHeader(BaseContractSqlHeader):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...


class TestTableContractSqlHeader(BaseTableContractSqlHeader): ...
class TestIncrementalContractSqlHeader(BaseIncrementalContractSqlHeader): ...


class BaseModelConstraintsRuntimeEnforcement:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...
    @pytest.fixture(scope="class")
    def expected_sql(self) -> str: ...
    def test__model_constraints_ddl(self, project: Any, expected_sql: str) -> None: ...


class TestModelConstraintsRuntimeEnforcement(BaseModelConstraintsRuntimeEnforcement): ...


class BaseConstraintQuotedColumn(BaseConstraintsRuntimeDdlEnforcement):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...
    @pytest.fixture(scope="class")
    def expected_sql(self) -> str: ...


class TestConstraintQuotedColumn(BaseConstraintQuotedColumn): ...


class TestIncrementalForeignKeyConstraint:
    @pytest.fixture(scope="class")
    def macros(self) -> Dict[str, str]: ...
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]: ...
    def test_incremental_foreign_key_constraint(self, project: Any) -> None: ...
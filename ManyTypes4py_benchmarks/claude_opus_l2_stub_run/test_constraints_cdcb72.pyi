from typing import Any, List

import pytest


def _normalize_whitespace(input: str) -> str: ...
def _find_and_replace(sql: str, find: str, replace: str) -> str: ...


class BaseConstraintsColumnsEqual:
    """
    dbt should catch these mismatches during its "preflight" checks.
    """

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
    def test__constraints_wrong_column_data_types(self, project: Any, string_type: str, int_type: str, schema_string_type: str, schema_int_type: str, data_types: List[List[str]]) -> None: ...
    def test__constraints_correct_column_data_types(self, project: Any, data_types: List[List[str]]) -> None: ...


class BaseConstraintsRuntimeDdlEnforcement:
    """
    These constraints pass muster for dbt's preflight checks. Make sure they're
    passed into the DDL statement. If they don't match up with the underlying data,
    the data platform should raise an error at runtime.
    """

    @pytest.fixture(scope="class")
    def models(self) -> dict[str, str]: ...

    @pytest.fixture(scope="class")
    def expected_sql(self) -> str: ...

    def test__constraints_ddl(self, project: Any, expected_sql: str) -> None: ...


class BaseConstraintsRollback:

    @pytest.fixture(scope="class")
    def models(self) -> dict[str, str]: ...

    @pytest.fixture(scope="class")
    def null_model_sql(self) -> str: ...

    @pytest.fixture(scope="class")
    def expected_color(self) -> str: ...

    @pytest.fixture(scope="class")
    def expected_error_messages(self) -> List[str]: ...

    def assert_expected_error_messages(self, error_message: str, expected_error_messages: List[str]) -> None: ...
    def test__constraints_enforcement_rollback(self, project: Any, expected_color: str, expected_error_messages: List[str], null_model_sql: str) -> None: ...


class BaseTableConstraintsColumnsEqual(BaseConstraintsColumnsEqual):

    @pytest.fixture(scope="class")
    def models(self) -> dict[str, str]: ...


class BaseViewConstraintsColumnsEqual(BaseConstraintsColumnsEqual):

    @pytest.fixture(scope="class")
    def models(self) -> dict[str, str]: ...


class BaseIncrementalConstraintsColumnsEqual(BaseConstraintsColumnsEqual):

    @pytest.fixture(scope="class")
    def models(self) -> dict[str, str]: ...


class BaseIncrementalConstraintsRuntimeDdlEnforcement(BaseConstraintsRuntimeDdlEnforcement):

    @pytest.fixture(scope="class")
    def models(self) -> dict[str, str]: ...


class BaseIncrementalConstraintsRollback(BaseConstraintsRollback):

    @pytest.fixture(scope="class")
    def models(self) -> dict[str, str]: ...

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
    """Tests a contracted model with a sql header dependency."""

    def test__contract_sql_header(self, project: Any) -> None: ...


class BaseTableContractSqlHeader(BaseContractSqlHeader):

    @pytest.fixture(scope="class")
    def models(self) -> dict[str, str]: ...


class BaseIncrementalContractSqlHeader(BaseContractSqlHeader):

    @pytest.fixture(scope="class")
    def models(self) -> dict[str, str]: ...


class TestTableContractSqlHeader(BaseTableContractSqlHeader): ...
class TestIncrementalContractSqlHeader(BaseIncrementalContractSqlHeader): ...


class BaseModelConstraintsRuntimeEnforcement:
    """
    These model-level constraints pass muster for dbt's preflight checks. Make sure they're
    passed into the DDL statement. If they don't match up with the underlying data,
    the data platform should raise an error at runtime.
    """

    @pytest.fixture(scope="class")
    def models(self) -> dict[str, str]: ...

    @pytest.fixture(scope="class")
    def expected_sql(self) -> str: ...

    def test__model_constraints_ddl(self, project: Any, expected_sql: str) -> None: ...


class TestModelConstraintsRuntimeEnforcement(BaseModelConstraintsRuntimeEnforcement): ...


class BaseConstraintQuotedColumn(BaseConstraintsRuntimeDdlEnforcement):

    @pytest.fixture(scope="class")
    def models(self) -> dict[str, str]: ...

    @pytest.fixture(scope="class")
    def expected_sql(self) -> str: ...


class TestConstraintQuotedColumn(BaseConstraintQuotedColumn): ...


class TestIncrementalForeignKeyConstraint:

    @pytest.fixture(scope="class")
    def macros(self) -> dict[str, str]: ...

    @pytest.fixture(scope="class")
    def models(self) -> dict[str, str]: ...

    def test_incremental_foreign_key_constraint(self, project: Any) -> None: ...
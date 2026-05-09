import pytest
from dbt.tests.util import Project

class BaseConstraintsColumnsEqual:
    @pytest.fixture
    def string_type(self) -> str:
        ...
    
    @pytest.fixture
    def int_type(self) -> str:
        ...
    
    @pytest.fixture
    def schema_string_type(self, string_type: str) -> str:
        ...
    
    @pytest.fixture
    def schema_int_type(self, int_type: str) -> str:
        ...
    
    @pytest.fixture
    def data_types(self, schema_int_type: str, int_type: str, string_type: str) -> list[list[str]]:
        ...
    
    def test__constraints_wrong_column_order(self, project: Project) -> None:
        ...
    
    def test__constraints_wrong_column_names(self, project: Project, string_type: str, int_type: str) -> None:
        ...
    
    def test__constraints_wrong_column_data_types(self, project: Project, string_type: str, int_type: str, schema_string_type: str, schema_int_type: str, data_types: list[list[str]]) -> None:
        ...
    
    def test__constraints_correct_column_data_types(self, project: Project, data_types: list[list[str]]) -> None:
        ...

def _normalize_whitespace(input: str) -> str:
    ...

def _find_and_replace(sql: str, find: str, replace: str) -> str:
    ...

class BaseConstraintsRuntimeDdlEnforcement:
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        ...
    
    @pytest.fixture(scope='class')
    def expected_sql(self) -> str:
        ...
    
    def test__constraints_ddl(self, project: Project, expected_sql: str) -> None:
        ...

class BaseConstraintsRollback:
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        ...
    
    @pytest.fixture(scope='class')
    def null_model_sql(self) -> str:
        ...
    
    @pytest.fixture(scope='class')
    def expected_color(self) -> str:
        ...
    
    @pytest.fixture(scope='class')
    def expected_error_messages(self) -> list[str]:
        ...
    
    def assert_expected_error_messages(self, error_message: str, expected_error_messages: list[str]) -> None:
        ...
    
    def test__constraints_enforcement_rollback(self, project: Project, expected_color: str, expected_error_messages: list[str], null_model_sql: str) -> None:
        ...

class BaseTableConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        ...

class BaseViewConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        ...

class BaseIncrementalConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        ...

class BaseIncrementalConstraintsRuntimeDdlEnforcement(BaseConstraintsRuntimeDdlEnforcement):
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        ...

class BaseIncrementalConstraintsRollback(BaseConstraintsRollback):
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        ...
    
    @pytest.fixture(scope='class')
    def null_model_sql(self) -> str:
        ...

class TestTableConstraintsColumnsEqual(BaseTableConstraintsColumnsEqual):
    ...

class TestViewConstraintsColumnsEqual(BaseViewConstraintsColumnsEqual):
    ...

class TestIncrementalConstraintsColumnsEqual(BaseIncrementalConstraintsColumnsEqual):
    ...

class TestTableConstraintsRuntimeDdlEnforcement(BaseConstraintsRuntimeDdlEnforcement):
    ...

class TestTableConstraintsRollback(BaseConstraintsRollback):
    ...

class TestIncrementalConstraintsRuntimeDdlEnforcement(BaseIncrementalConstraintsRuntimeDdlEnforcement):
    ...

class TestIncrementalConstraintsRollback(BaseIncrementalConstraintsRollback):
    ...

class BaseContractSqlHeader:
    def test__contract_sql_header(self, project: Project) -> None:
        ...

class BaseTableContractSqlHeader(BaseContractSqlHeader):
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        ...

class BaseIncrementalContractSqlHeader(BaseContractSqlHeader):
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        ...

class TestTableContractSqlHeader(BaseTableContractSqlHeader):
    ...

class TestIncrementalContractSqlHeader(BaseIncrementalContractSqlHeader):
    ...

class BaseModelConstraintsRuntimeEnforcement:
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        ...
    
    @pytest.fixture(scope='class')
    def expected_sql(self) -> str:
        ...
    
    def test__model_constraints_ddl(self, project: Project, expected_sql: str) -> None:
        ...

class TestModelConstraintsRuntimeEnforcement(BaseModelConstraintsRuntimeEnforcement):
    ...

class BaseConstraintQuotedColumn(BaseConstraintsRuntimeDdlEnforcement):
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        ...
    
    @pytest.fixture(scope='class')
    def expected_sql(self) -> str:
        ...

class TestConstraintQuotedColumn(BaseConstraintQuotedColumn):
    ...

class TestIncrementalForeignKeyConstraint:
    @pytest.fixture(scope='class')
    def macros(self) -> dict[str, str]:
        ...
    
    @pytest.fixture(scope='class')
    def models(self) -> dict[str, str]:
        ...
    
    def test_incremental_foreign_key_constraint(self, project: Project) -> None:
        ...
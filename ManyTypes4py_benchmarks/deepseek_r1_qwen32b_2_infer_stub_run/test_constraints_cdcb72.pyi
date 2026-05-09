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
from typing import Any, List, Optional, Tuple, Union

class BaseConstraintsColumnsEqual:
    @pytest.fixture
    def string_type(self) -> str:
        ...
    
    @pytest.fixture
    def int_type(self) -> int:
        ...
    
    @pytest.fixture
    def schema_string_type(self, string_type: str) -> str:
        ...
    
    @pytest.fixture
    def schema_int_type(self, int_type: int) -> str:
        ...
    
    @pytest.fixture
    def data_types(
        self,
        schema_int_type: str,
        int_type: int,
        string_type: str
    ) -> List[List[str]]:
        ...
    
    def test__constraints_wrong_column_order(self, project: pytest.fixture) -> None:
        ...
    
    def test__constraints_wrong_column_names(
        self,
        project: pytest.fixture,
        string_type: str,
        int_type: int
    ) -> None:
        ...
    
    def test__constraints_wrong_column_data_types(
        self,
        project: pytest.fixture,
        string_type: str,
        int_type: int,
        schema_string_type: str,
        schema_int_type: str,
        data_types: List[List[str]]
    ) -> None:
        ...
    
    def test__constraints_correct_column_data_types(
        self,
        project: pytest.fixture,
        data_types: List[List[str]]
    ) -> None:
        ...

def _normalize_whitespace(input: str) -> str:
    ...

def _find_and_replace(sql: str, find: str, replace: str) -> str:
    ...

class BaseConstraintsRuntimeDdlEnforcement:
    @pytest.fixture(scope='class')
    def models(self) -> dict:
        ...
    
    @pytest.fixture(scope='class')
    def expected_sql(self) -> str:
        ...
    
    def test__constraints_ddl(
        self,
        project: pytest.fixture,
        expected_sql: str
    ) -> None:
        ...

class BaseConstraintsRollback:
    @pytest.fixture(scope='class')
    def models(self) -> dict:
        ...
    
    @pytest.fixture(scope='class')
    def null_model_sql(self) -> str:
        ...
    
    @pytest.fixture(scope='class')
    def expected_color(self) -> str:
        ...
    
    @pytest.fixture(scope='class')
    def expected_error_messages(self) -> List[str]:
        ...
    
    def assert_expected_error_messages(
        self,
        error_message: str,
        expected_error_messages: List[str]
    ) -> None:
        ...
    
    def test__constraints_enforcement_rollback(
        self,
        project: pytest.fixture,
        expected_color: str,
        expected_error_messages: List[str],
        null_model_sql: str
    ) -> None:
        ...

class BaseTableConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
    ...

class BaseViewConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
    ...

class BaseIncrementalConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
    ...

class BaseIncrementalConstraintsRuntimeDdlEnforcement(BaseConstraintsRuntimeDdlEnforcement):
    ...

class BaseIncrementalConstraintsRollback(BaseConstraintsRollback):
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
    def test__contract_sql_header(self, project: pytest.fixture) -> None:
        ...

class BaseTableContractSqlHeader(BaseContractSqlHeader):
    @pytest.fixture(scope='class')
    def models(self) -> dict:
        ...

class BaseIncrementalContractSqlHeader(BaseContractSqlHeader):
    @pytest.fixture(scope='class')
    def models(self) -> dict:
        ...

class TestTableContractSqlHeader(BaseTableContractSqlHeader):
    ...

class TestIncrementalContractSqlHeader(BaseIncrementalContractSqlHeader):
    ...

class BaseModelConstraintsRuntimeEnforcement:
    @pytest.fixture(scope='class')
    def models(self) -> dict:
        ...
    
    @pytest.fixture(scope='class')
    def expected_sql(self) -> str:
        ...
    
    def test__model_constraints_ddl(
        self,
        project: pytest.fixture,
        expected_sql: str
    ) -> None:
        ...

class TestModelConstraintsRuntimeEnforcement(BaseModelConstraintsRuntimeEnforcement):
    ...

class BaseConstraintQuotedColumn(BaseConstraintsRuntimeDdlEnforcement):
    @pytest.fixture(scope='class')
    def models(self) -> dict:
        ...
    
    @pytest.fixture(scope='class')
    def expected_sql(self) -> str:
        ...

class TestConstraintQuotedColumn(BaseConstraintQuotedColumn):
    ...

class TestIncrementalForeignKeyConstraint:
    @pytest.fixture(scope='class')
    def macros(self) -> dict:
        ...
    
    @pytest.fixture(scope='class')
    def models(self) -> dict:
        ...
    
    def test_incremental_foreign_key_constraint(self, project: pytest.fixture) -> None:
        ...
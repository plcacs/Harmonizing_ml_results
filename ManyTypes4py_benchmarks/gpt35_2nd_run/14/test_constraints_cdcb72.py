import re
from typing import List, Tuple
import pytest

class BaseConstraintsColumnsEqual:
    def test__constraints_wrong_column_order(self, project: Any) -> None:
    def test__constraints_wrong_column_names(self, project: Any, string_type: str, int_type: str) -> None:
    def test__constraints_wrong_column_data_types(self, project: Any, string_type: str, int_type: str, schema_string_type: str, schema_int_type: str, data_types: List[Tuple[str, str, str]]) -> None:
    def test__constraints_correct_column_data_types(self, project: Any, data_types: List[Tuple[str, str, str]]) -> None

def _normalize_whitespace(input: str) -> str:
def _find_and_replace(sql: str, find: str, replace: str) -> str:

class BaseConstraintsRuntimeDdlEnforcement:
    def test__constraints_ddl(self, project: Any, expected_sql: str) -> None:

class BaseConstraintsRollback:
    def assert_expected_error_messages(self, error_message: str, expected_error_messages: List[str]) -> None:
    def test__constraints_enforcement_rollback(self, project: Any, expected_color: str, expected_error_messages: List[str], null_model_sql: str) -> None:

class BaseTableConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
class BaseViewConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
class BaseIncrementalConstraintsColumnsEqual(BaseConstraintsColumnsEqual):
class BaseIncrementalConstraintsRuntimeDdlEnforcement(BaseConstraintsRuntimeDdlEnforcement):
class BaseIncrementalConstraintsRollback(BaseConstraintsRollback):

class BaseContractSqlHeader:
    def test__contract_sql_header(self, project: Any) -> None:

class BaseTableContractSqlHeader(BaseContractSqlHeader):
class BaseIncrementalContractSqlHeader(BaseContractSqlHeader):

class BaseModelConstraintsRuntimeEnforcement:
    def test__model_constraints_ddl(self, project: Any, expected_sql: str) -> None:

class BaseConstraintQuotedColumn(BaseConstraintsRuntimeDdlEnforcement):

class TestConstraintQuotedColumn(BaseConstraintQuotedColumn):
class TestIncrementalForeignKeyConstraint:

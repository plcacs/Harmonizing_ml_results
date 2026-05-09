from __future__ import annotations
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import Mock
from sqlalchemy.sql.elements import TextClause
from superset.models.core import Database
from superset.utils.core import AdhocMetricExpressionType, FilterOperator
from superset.utils.database import get_example_database
from flask.ctx import AppContext
from pytest_mock import MockerFixture
import pandas as pd
import numpy as np
from re import Pattern

VIRTUAL_TABLE_INT_TYPES: dict[str, Pattern[str]] = ...
VIRTUAL_TABLE_STRING_TYPES: dict[str, Pattern[str]] = ...

class FilterTestCase(NamedTuple):
    column: str
    operator: FilterOperator
    value: Union[str, List[str], datetime]
    expected: Union[str, List[str]]

class TestDatabaseModel(SupersetTestCase):
    def test_is_time_druid_time_col(self) -> None: ...
    def test_temporal_varchar(self) -> None: ...
    def test_db_column_types(self) -> None: ...
    @patch('superset.jinja_context.get_username', return_value='abc')
    def test_jinja_metrics_and_calc_columns(self, mock_username: Mock) -> None: ...
    @patch('superset.jinja_context.get_dataset_id_from_context')
    def test_jinja_metric_macro(self, mock_dataset_id_from_context: Mock) -> None: ...
    def test_adhoc_metrics_and_calc_columns(self) -> None: ...
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_where_operators(self) -> None: ...
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_boolean_type_where_operators(self) -> None: ...
    def test_incorrect_jinja_syntax_raises_correct_exception(self) -> None: ...
    def test_query_format_strip_trailing_semicolon(self) -> None: ...
    def test_multiple_sql_statements_raises_exception(self) -> None: ...
    def test_dml_statement_raises_exception(self) -> None: ...
    def test_fetch_metadata_for_updated_virtual_table(self) -> None: ...
    @patch('superset.models.core.Database.db_engine_spec', BigQueryEngineSpec)
    def test_labels_expected_on_mutated_query(self) -> None: ...

@pytest.fixture
def text_column_table(app_context: AppContext) -> SqlaTable: ...

def test_values_for_column_on_text_column(text_column_table: SqlaTable) -> None: ...
def test_values_for_column_on_text_column_with_rls(text_column_table: SqlaTable) -> None: ...
def test_values_for_column_on_text_column_with_rls_no_values(text_column_table: SqlaTable) -> None: ...
def test_filter_on_text_column(text_column_table: SqlaTable) -> None: ...

@only_postgresql
def test_should_generate_closed_and_open_time_filter_range(login_as_admin: Callable) -> None: ...

def test_none_operand_in_filter(login_as_admin: Callable, physical_dataset: Any) -> None: ...

@pytest.mark.usefixtures('app_context')
@pytest.mark.parametrize('table_name,sql,expected_cache_keys,has_extra_cache_keys', [
    ('test_has_extra_cache_keys_table', ..., {1, 'abc', 'abc@test.com'}, True),
    ('test_has_extra_cache_keys_table_with_set', ..., {1, 'abc', 'abc@test.com'}, True),
    ('test_has_extra_cache_keys_table_with_se_multiple', ..., {1, 'abc@test.com'}, True),
    ('test_has_extra_cache_keys_disabled_table', ..., [], True),
    ('test_has_no_extra_cache_keys_table', ..., [], False)
])
@patch('superset.jinja_context.get_user_id', return_value=1)
@patch('superset.jinja_context.get_username', return_value='abc')
@patch('superset.jinja_context.get_user_email', return_value='abc@test.com')
def test_extra_cache_keys(mock_user_email: Mock, mock_username: Mock, mock_user_id: Mock, table_name: str, sql: str, expected_cache_keys: List[Union[int, str]], has_extra_cache_keys: bool) -> None: ...

@pytest.mark.usefixtures('app_context')
@pytest.mark.parametrize('sql_expression,expected_cache_keys,has_extra_cache_keys', [
    ("(user != '{{ current_username() }}')", ['abc'], True),
    ("(user != 'abc')", [], False)
])
@patch('superset.jinja_context.get_user_id', return_value=1)
@patch('superset.jinja_context.get_username', return_value='abc')
@patch('superset.jinja_context.get_user_email', return_value='abc@test.com')
def test_extra_cache_keys_in_sql_expression(mock_user_email: Mock, mock_username: Mock, mock_user_id: Mock, sql_expression: str, expected_cache_keys: List[str], has_extra_cache_keys: bool) -> None: ...

@pytest.mark.usefixtures('app_context')
@pytest.mark.parametrize('sql_expression,expected_cache_keys,has_extra_cache_keys,item_type', [
    ("'{{ current_username() }}'", ['abc'], True, 'columns'),
    ("(user != 'abc')", [], False, 'columns'),
    ("{{ current_user_id() }}", [1], True, 'metrics'),
    ('COUNT(*)', [], False, 'metrics')
])
@patch('superset.jinja_context.get_user_id', return_value=1)
@patch('superset.jinja_context.get_username', return_value='abc')
def test_extra_cache_keys_in_adhoc_metrics_and_columns(mock_username: Mock, mock_user_id: Mock, sql_expression: str, expected_cache_keys: List[Union[int, str]], has_extra_cache_keys: bool, item_type: str) -> None: ...

@pytest.mark.usefixtures('app_context')
@patch('superset.jinja_context.get_user_id', return_value=1)
@patch('superset.jinja_context.get_username', return_value='abc')
def test_extra_cache_keys_in_dataset_metrics_and_columns(mock_username: Mock, mock_user_id: Mock) -> None: ...

@pytest.mark.usefixtures('app_context')
@pytest.mark.parametrize('row,dimension,result', [
    (pd.Series({'foo': 'abc'}), 'foo', 'abc'),
    (pd.Series({'bar': True}), 'bar', True),
    (pd.Series({'baz': 123}), 'baz', 123),
    (pd.Series({'baz': np.int16(123)}), 'baz', 123),
    (pd.Series({'baz': np.uint32(123)}), 'baz', 123),
    (pd.Series({'baz': np.int64(123)}), 'baz', 123),
    (pd.Series({'qux': 123.456}), 'qux', 123.456),
    (pd.Series({'qux': np.float32(123.456)}), 'qux', 123.45600128173828),
    (pd.Series({'qux': np.float64(123.456)}), 'qux', 123.456),
    (pd.Series({'quux': '2021-01-01'}), 'quux', '2021-01-01'),
    (pd.Series({'quuz': '2021-01-01T00:00:00'}), 'quuz', text("TIME_PARSE('2021-01-01T00:00:00')"))
])
def test__normalize_prequery_result_type(mocker: MockerFixture, row: pd.Series, dimension: str, result: Union[str, bool, int, float, TextClause]) -> None: ...

@pytest.mark.usefixtures('app_context')
def test__temporal_range_operator_in_adhoc_filter(physical_dataset: Any) -> None: ...
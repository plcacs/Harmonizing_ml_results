from __future__ import annotations
import re
from datetime import datetime
from typing import Any, Literal, NamedTuple, Optional, Union
from re import Pattern
from unittest.mock import Mock, patch
import pytest
import numpy as np
import pandas as pd
from flask.ctx import AppContext
from pytest_mock import MockerFixture
from sqlalchemy.sql import text
from sqlalchemy.sql.elements import TextClause
from superset import db
from superset.connectors.sqla.models import SqlaTable, TableColumn, SqlMetric
from superset.constants import EMPTY_STRING, NULL_STRING
from superset.db_engine_specs.bigquery import BigQueryEngineSpec
from superset.db_engine_specs.druid import DruidEngineSpec
from superset.exceptions import QueryObjectValidationError
from superset.models.core import Database
from superset.utils.core import AdhocMetricExpressionType, FilterOperator, GenericDataType
from superset.utils.database import get_example_database
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from .base_tests import SupersetTestCase
from .conftest import only_postgresql

VIRTUAL_TABLE_INT_TYPES: dict[str, Pattern] = {'hive': re.compile('^INT_TYPE$'), 'mysql': re.compile('^LONGLONG$'), 'postgresql': re.compile('^INTEGER$'), 'presto': re.compile('^INTEGER$'), 'sqlite': re.compile('^INT$')}
VIRTUAL_TABLE_STRING_TYPES: dict[str, Pattern] = {'hive': re.compile('^STRING_TYPE$'), 'mysql': re.compile('^VAR_STRING$'), 'postgresql': re.compile('^STRING$'), 'presto': re.compile('^VARCHAR*'), 'sqlite': re.compile('^STRING$')}

class FilterTestCase(NamedTuple):
    pass

class TestDatabaseModel(SupersetTestCase):

    def test_is_time_druid_time_col(self) -> None:
        ...

    def test_temporal_varchar(self) -> None:
        ...

    def test_db_column_types(self) -> None:
        ...

    @patch('superset.jinja_context.get_username', return_value='abc')
    def test_jinja_metrics_and_calc_columns(self, mock_username: Mock) -> None:
        ...

    @patch('superset.jinja_context.get_dataset_id_from_context')
    def test_jinja_metric_macro(self, mock_dataset_id_from_context: Mock) -> None:
        ...

    def test_adhoc_metrics_and_calc_columns(self) -> None:
        ...

    def test_where_operators(self) -> None:
        ...

    def test_boolean_type_where_operators(self) -> None:
        ...

    def test_incorrect_jinja_syntax_raises_correct_exception(self) -> None:
        ...

    def test_query_format_strip_trailing_semicolon(self) -> None:
        ...

    def test_multiple_sql_statements_raises_exception(self) -> None:
        ...

    def test_dml_statement_raises_exception(self) -> None:
        ...

    def test_fetch_metadata_for_updated_virtual_table(self) -> None:
        ...

    @patch('superset.models.core.Database.db_engine_spec', BigQueryEngineSpec)
    def test_labels_expected_on_mutated_query(self) -> None:
        ...

@pytest.fixture
def text_column_table(app_context: AppContext) -> SqlaTable:
    ...

def test_values_for_column_on_text_column(text_column_table: SqlaTable) -> None:
    ...

def test_values_for_column_on_text_column_with_rls(text_column_table: SqlaTable) -> None:
    ...

def test_values_for_column_on_text_column_with_rls_no_values(text_column_table: SqlaTable) -> None:
    ...

def test_filter_on_text_column(text_column_table: SqlaTable) -> None:
    ...

@only_postgresql
def test_should_generate_closed_and_open_time_filter_range(login_as_admin: Any) -> None:
    ...

def test_none_operand_in_filter(login_as_admin: Any, physical_dataset: SqlaTable) -> None:
    ...

@pytest.mark.usefixtures('app_context')
@pytest.mark.parametrize('table_name,sql,expected_cache_keys,has_extra_cache_keys', [...])
@patch('superset.jinja_context.get_user_id', return_value=1)
@patch('superset.jinja_context.get_username', return_value='abc')
@patch('superset.jinja_context.get_user_email', return_value='abc@test.com')
def test_extra_cache_keys(mock_user_email: Mock, mock_username: Mock, mock_user_id: Mock, table_name: str, sql: str, expected_cache_keys: set, has_extra_cache_keys: bool) -> None:
    ...

@pytest.mark.usefixtures('app_context')
@pytest.mark.parametrize('sql_expression,expected_cache_keys,has_extra_cache_keys', [...])
@patch('superset.jinja_context.get_user_id', return_value=1)
@patch('superset.jinja_context.get_username', return_value='abc')
@patch('superset.jinja_context.get_user_email', return_value='abc@test.com')
def test_extra_cache_keys_in_sql_expression(mock_user_email: Mock, mock_username: Mock, mock_user_id: Mock, sql_expression: str, expected_cache_keys: list, has_extra_cache_keys: bool) -> None:
    ...

@pytest.mark.usefixtures('app_context')
@pytest.mark.parametrize('sql_expression,expected_cache_keys,has_extra_cache_keys,item_type', [...])
@patch('superset.jinja_context.get_user_id', return_value=1)
@patch('superset.jinja_context.get_username', return_value='abc')
def test_extra_cache_keys_in_adhoc_metrics_and_columns(mock_username: Mock, mock_user_id: Mock, sql_expression: str, expected_cache_keys: list, has_extra_cache_keys: bool, item_type: str) -> None:
    ...

@pytest.mark.usefixtures('app_context')
@patch('superset.jinja_context.get_user_id', return_value=1)
@patch('superset.jinja_context.get_username', return_value='abc')
def test_extra_cache_keys_in_dataset_metrics_and_columns(mock_username: Mock, mock_user_id: Mock) -> None:
    ...

@pytest.mark.usefixtures('app_context')
@pytest.mark.parametrize('row,dimension,result', [...])
def test__normalize_prequery_result_type(mocker: Any, row: pd.Series, dimension: str, result: Any) -> None:
    ...

@pytest.mark.usefixtures('app_context')
def test__temporal_range_operator_in_adhoc_filter(physical_dataset: SqlaTable) -> None:
    ...

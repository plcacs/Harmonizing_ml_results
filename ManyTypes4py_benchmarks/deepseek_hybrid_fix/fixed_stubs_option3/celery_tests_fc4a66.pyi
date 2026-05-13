"""Unit tests for Superset Celery worker"""
import datetime
import random
import string
import time
import unittest.mock as mock
from typing import Optional, Any, Callable, Dict, List, Tuple, Union
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_data
import pytest
import flask
from flask import current_app, has_app_context, Flask
from superset import db, sql_lab
from superset.common.db_query_status import QueryStatus
from superset.result_set import SupersetResultSet
from superset.db_engine_specs.base import BaseEngineSpec
from superset.errors import ErrorLevel, SupersetErrorType
from superset.extensions import celery_app, Celery
from superset.models.sql_lab import Query
from superset.sql_parse import ParsedQuery, CtasMethod
from superset.utils.core import backend
from superset.utils.database import get_example_database
from tests.integration_tests.conftest import CTAS_SCHEMA_NAME
from tests.integration_tests.test_app import app
from flask.testing import FlaskClient

CELERY_SLEEP_TIME: int = 6
QUERY: str = 'SELECT name FROM birth_names LIMIT 1'
TEST_SYNC: str = 'test_sync'
TEST_ASYNC_LOWER_LIMIT: str = 'test_async_lower_limit'
TEST_SYNC_CTA: str = 'test_sync_cta'
TEST_ASYNC_CTA: str = 'test_async_cta'
TEST_ASYNC_CTA_CONFIG: str = 'test_async_cta_config'
TMP_TABLES: list[str] = [TEST_SYNC, TEST_SYNC_CTA, TEST_ASYNC_CTA, TEST_ASYNC_CTA_CONFIG, TEST_ASYNC_LOWER_LIMIT]

def get_query_by_id(id: int) -> Query:
    ...

@pytest.fixture(autouse=True, scope='module')
def setup_sqllab() -> Any:
    ...

def run_sql(test_client: FlaskClient, sql: str, cta: bool = ..., ctas_method: CtasMethod = ..., tmp_table: str = ..., async_: bool = ...) -> Dict[str, Any]:
    ...

def drop_table_if_exists(table_name: str, table_type: str) -> None:
    ...

def quote_f(value: Optional[str]) -> Optional[str]:
    ...

def cta_result(ctas_method: CtasMethod) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ...

def get_select_star(table: str, limit: int, schema: Optional[str] = ...) -> str:
    ...

@pytest.mark.usefixtures('login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
def test_run_sync_query_dont_exist(test_client: FlaskClient, ctas_method: CtasMethod) -> None:
    ...

@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
def test_run_sync_query_cta(test_client: FlaskClient, ctas_method: CtasMethod) -> None:
    ...

@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
def test_run_sync_query_cta_no_data(test_client: FlaskClient) -> None:
    ...

@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
@mock.patch('superset.sqllab.sqllab_execution_context.get_cta_schema_name', lambda d, u, s, sql: CTAS_SCHEMA_NAME)
def test_run_sync_query_cta_config(test_client: FlaskClient, ctas_method: CtasMethod) -> None:
    ...

@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
@mock.patch('superset.sqllab.sqllab_execution_context.get_cta_schema_name', lambda d, u, s, sql: CTAS_SCHEMA_NAME)
def test_run_async_query_cta_config(test_client: FlaskClient, ctas_method: CtasMethod) -> None:
    ...

@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
def test_run_async_cta_query(test_client: FlaskClient, ctas_method: CtasMethod) -> None:
    ...

@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
def test_run_async_cta_query_with_lower_limit(test_client: FlaskClient, ctas_method: CtasMethod) -> None:
    ...

SERIALIZATION_DATA: list[tuple[str, int, float, datetime.datetime]] = [('a', 4, 4.0, datetime.datetime(2019, 8, 18, 16, 39, 16, 660000))]
CURSOR_DESCR: tuple[tuple[str, str], tuple[str, str], tuple[str, str], tuple[str, str]] = (('a', 'string'), ('b', 'int'), ('c', 'float'), ('d', 'datetime'))

def test_default_data_serialization() -> None:
    ...

def test_new_data_serialization() -> None:
    ...

@pytest.mark.usefixtures('load_birth_names_data')
def test_default_payload_serialization() -> None:
    ...

@pytest.mark.usefixtures('load_birth_names_data')
def test_msgpack_payload_serialization() -> None:
    ...

def test_create_table_as() -> None:
    ...

def test_in_app_context() -> None:
    ...

def delete_tmp_view_or_table(name: str, db_object_type: str) -> None:
    ...

def wait_for_success(result: Dict[str, Any]) -> Query:
    ...
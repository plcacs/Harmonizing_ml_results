"""Unit tests for Superset Celery worker"""

import datetime
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pytest import fixture
from flask import TestClient
from superset.models.sql_lab import Query
from superset.sql_parse import CtasMethod
from superset.utils.core import backend

CELERY_SLEEP_TIME: int = ...
QUERY: str = ...
TEST_SYNC: str = ...
TEST_ASYNC_LOWER_LIMIT: str = ...
TEST_SYNC_CTA: str = ...
TEST_ASYNC_CTA: str = ...
TEST_ASYNC_CTA_CONFIG: str = ...
TMP_TABLES: List[str] = ...

def get_query_by_id(id: int) -> Query:
    ...

@fixture(autouse=True, scope='module')
def setup_sqllab() -> None:
    ...

def run_sql(test_client: TestClient, sql: str, cta: bool = ..., ctas_method: CtasMethod = ..., tmp_table: str = ..., async_: bool = ...) -> Dict[str, Any]:
    ...

def drop_table_if_exists(table_name: str, table_type: str) -> None:
    ...

def quote_f(value: Optional[str]) -> str:
    ...

def cta_result(ctas_method: CtasMethod) -> Tuple[List[Dict[str, int]], List[Dict[str, Any]]]:
    ...

def get_select_star(table: str, limit: int, schema: Optional[str] = ...) -> str:
    ...

@fixture
def load_birth_names_data() -> None:
    ...

@pytest.mark.usefixtures('login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
def test_run_sync_query_dont_exist(test_client: TestClient, ctas_method: CtasMethod) -> None:
    ...

@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
def test_run_sync_query_cta(test_client: TestClient, ctas_method: CtasMethod) -> None:
    ...

@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
def test_run_sync_query_cta_no_data(test_client: TestClient) -> None:
    ...

@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
@mock.patch('superset.sqllab.sqllab_execution_context.get_cta_schema_name')
def test_run_sync_query_cta_config(test_client: TestClient, ctas_method: CtasMethod) -> None:
    ...

@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
@mock.patch('superset.sqllab.sqllab_execution_context.get_cta_schema_name')
def test_run_async_query_cta_config(test_client: TestClient, ctas_method: CtasMethod) -> None:
    ...

@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
def test_run_async_cta_query(test_client: TestClient, ctas_method: CtasMethod) -> None:
    ...

@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
def test_run_async_cta_query_with_lower_limit(test_client: TestClient, ctas_method: CtasMethod) -> None:
    ...

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
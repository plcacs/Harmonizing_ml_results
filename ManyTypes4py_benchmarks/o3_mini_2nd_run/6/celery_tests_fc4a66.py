#!/usr/bin/env python3
"""Unit tests for Superset Celery worker"""
import datetime
import random
import string
import time
import unittest.mock as mock
from typing import Any, Dict, List, Optional, Tuple

import flask
import pytest
from flask import current_app, has_app_context

from superset import db, sql_lab
from superset.common.db_query_status import QueryStatus
from superset.db_engine_specs.base import BaseEngineSpec
from superset.errors import ErrorLevel, SupersetErrorType
from superset.extensions import celery_app
from superset.models.sql_lab import Query
from superset.result_set import SupersetResultSet
from superset.sql_parse import ParsedQuery, CtasMethod
from superset.utils.core import backend
from superset.utils.database import get_example_database
from tests.integration_tests.conftest import CTAS_SCHEMA_NAME
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_data
from tests.integration_tests.test_app import app

CELERY_SLEEP_TIME: int = 6
QUERY: str = 'SELECT name FROM birth_names LIMIT 1'
TEST_SYNC: str = 'test_sync'
TEST_ASYNC_LOWER_LIMIT: str = 'test_async_lower_limit'
TEST_SYNC_CTA: str = 'test_sync_cta'
TEST_ASYNC_CTA: str = 'test_async_cta'
TEST_ASYNC_CTA_CONFIG: str = 'test_async_cta_config'
TMP_TABLES: List[str] = [TEST_SYNC, TEST_SYNC_CTA, TEST_ASYNC_CTA, TEST_ASYNC_CTA_CONFIG, TEST_ASYNC_LOWER_LIMIT]


def get_query_by_id(query_id: int) -> Optional[Query]:
    db.session.commit()
    query: Optional[Query] = db.session.query(Query).filter_by(id=query_id).first()
    return query


@pytest.fixture(autouse=True, scope='module')
def setup_sqllab() -> Any:
    yield
    with app.app_context():
        db.session.query(Query).delete()
        db.session.commit()
        for tbl in TMP_TABLES:
            drop_table_if_exists(f'{tbl}_{CtasMethod.TABLE.lower()}', CtasMethod.TABLE)
            drop_table_if_exists(f'{tbl}_{CtasMethod.VIEW.lower()}', CtasMethod.VIEW)
            drop_table_if_exists(f'{CTAS_SCHEMA_NAME}.{tbl}_{CtasMethod.TABLE.lower()}', CtasMethod.TABLE)
            drop_table_if_exists(f'{CTAS_SCHEMA_NAME}.{tbl}_{CtasMethod.VIEW.lower()}', CtasMethod.VIEW)


def run_sql(
    test_client: Any,
    sql: str,
    cta: bool = False,
    ctas_method: str = CtasMethod.TABLE,
    tmp_table: str = 'tmp',
    async_: bool = False,
) -> Dict[str, Any]:
    db_id: int = get_example_database().id
    client_id: str = ''.join((random.choice(string.ascii_lowercase) for i in range(5)))
    response = test_client.post(
        '/api/v1/sqllab/execute/',
        json=dict(
            database_id=db_id,
            sql=sql,
            runAsync=async_,
            select_as_cta=cta,
            tmp_table_name=tmp_table,
            client_id=client_id,
            ctas_method=ctas_method,
        ),
    )
    return response.json


def drop_table_if_exists(table_name: str, table_type: str) -> None:
    """Drop table if it exists, works on any DB"""
    sql: str = f'DROP {table_type} IF EXISTS {table_name}'
    database = get_example_database()
    with database.get_sqla_engine() as engine:
        engine.execute(sql)


def quote_f(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    with get_example_database().get_inspector() as inspector:
        return inspector.engine.dialect.identifier_preparer.quote_identifier(value)


def cta_result(ctas_method: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if backend() != 'presto':
        return ([], [])
    if ctas_method == CtasMethod.TABLE:
        return ([{'rows': 1}], [{'name': 'rows', 'type': 'BIGINT', 'is_dttm': False}])
    return ([{'result': True}], [{'name': 'result', 'type': 'BOOLEAN', 'is_dttm': False}])


def get_select_star(table: str, limit: int, schema: Optional[str] = None) -> str:
    if backend() in {'presto', 'hive'}:
        schema = quote_f(schema) if schema else schema
        table = quote_f(table)
    if schema:
        return f'SELECT\n  *\nFROM {schema}.{table}\nLIMIT {limit}'
    return f'SELECT\n  *\nFROM {table}\nLIMIT {limit}'


@pytest.mark.usefixtures('login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
def test_run_sync_query_dont_exist(test_client: Any, ctas_method: str) -> None:
    examples_db = get_example_database()
    engine_name: str = examples_db.db_engine_spec.engine_name
    sql_dont_exist: str = 'SELECT name FROM table_dont_exist'
    result: Dict[str, Any] = run_sql(test_client, sql_dont_exist, cta=True, ctas_method=ctas_method)
    if backend() == 'sqlite' and ctas_method == CtasMethod.VIEW:
        assert QueryStatus.SUCCESS == result['status'], result
    elif backend() == 'presto':
        assert result['errors'][0]['error_type'] == SupersetErrorType.TABLE_DOES_NOT_EXIST_ERROR
        assert result['errors'][0]['level'] == ErrorLevel.ERROR
        assert result['errors'][0]['extra'] == {
            'engine_name': 'Presto',
            'issue_codes': [
                {
                    'code': 1003,
                    'message': 'Issue 1003 - There is a syntax error in the SQL query. Perhaps there was a misspelling or a typo.'
                },
                {
                    'code': 1005,
                    'message': 'Issue 1005 - The table was deleted or renamed in the database.'
                },
            ],
        }
    else:
        assert result['errors'][0]['error_type'] == SupersetErrorType.GENERIC_DB_ENGINE_ERROR
        assert result['errors'][0]['level'] == ErrorLevel.ERROR
        assert result['errors'][0]['extra'] == {
            'issue_codes': [{'code': 1002, 'message': 'Issue 1002 - The database returned an unexpected error.'}],
            'engine_name': engine_name,
        }


@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
def test_run_sync_query_cta(test_client: Any, ctas_method: str) -> None:
    tmp_table_name: str = f'{TEST_SYNC}_{ctas_method.lower()}'
    result: Dict[str, Any] = run_sql(test_client, QUERY, tmp_table=tmp_table_name, cta=True, ctas_method=ctas_method)
    assert QueryStatus.SUCCESS == result['query']['state'], result
    assert cta_result(ctas_method) == (result['data'], result['columns'])
    select_query: Optional[Query] = get_query_by_id(result['query']['serverId'])
    results: Dict[str, Any] = run_sql(test_client, select_query.select_sql)  # type: ignore
    assert QueryStatus.SUCCESS == results['status'], results
    assert len(results['data']) > 0
    delete_tmp_view_or_table(tmp_table_name, ctas_method)


@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
def test_run_sync_query_cta_no_data(test_client: Any) -> None:
    sql_empty_result: str = "SELECT * FROM birth_names WHERE name='random'"
    result: Dict[str, Any] = run_sql(test_client, sql_empty_result)
    assert QueryStatus.SUCCESS == result['query']['state']
    assert ([], []) == (result['data'], result['columns'])
    query: Optional[Query] = get_query_by_id(result['query']['serverId'])
    assert query is not None and QueryStatus.SUCCESS == query.status


@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
@mock.patch('superset.sqllab.sqllab_execution_context.get_cta_schema_name', lambda d, u, s, sql: CTAS_SCHEMA_NAME)
def test_run_sync_query_cta_config(test_client: Any, ctas_method: str) -> None:
    if backend() == 'sqlite':
        return
    tmp_table_name: str = f'{TEST_SYNC_CTA}_{ctas_method.lower()}'
    result: Dict[str, Any] = run_sql(test_client, QUERY, cta=True, ctas_method=ctas_method, tmp_table=tmp_table_name)
    assert QueryStatus.SUCCESS == result['query']['state'], result
    assert cta_result(ctas_method) == (result['data'], result['columns'])
    query: Optional[Query] = get_query_by_id(result['query']['serverId'])
    assert query is not None
    assert f'CREATE {ctas_method} {CTAS_SCHEMA_NAME}.{tmp_table_name} AS \n{QUERY}' == query.executed_sql
    assert query.select_sql == get_select_star(tmp_table_name, limit=query.limit, schema=CTAS_SCHEMA_NAME)
    results: Dict[str, Any] = run_sql(test_client, query.select_sql)
    assert QueryStatus.SUCCESS == results['status'], result
    delete_tmp_view_or_table(f'{CTAS_SCHEMA_NAME}.{tmp_table_name}', ctas_method)


@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
@mock.patch('superset.sqllab.sqllab_execution_context.get_cta_schema_name', lambda d, u, s, sql: CTAS_SCHEMA_NAME)
def test_run_async_query_cta_config(test_client: Any, ctas_method: str) -> None:
    if backend() == 'sqlite':
        return
    tmp_table_name: str = f'{TEST_ASYNC_CTA_CONFIG}_{ctas_method.lower()}'
    result: Dict[str, Any] = run_sql(
        test_client, QUERY, cta=True, ctas_method=ctas_method, async_=True, tmp_table=tmp_table_name
    )
    query: Query = wait_for_success(result)
    assert QueryStatus.SUCCESS == query.status
    assert get_select_star(tmp_table_name, limit=query.limit, schema=CTAS_SCHEMA_NAME) == query.select_sql
    assert f'CREATE {ctas_method} {CTAS_SCHEMA_NAME}.{tmp_table_name} AS \n{QUERY}' == query.executed_sql
    delete_tmp_view_or_table(f'{CTAS_SCHEMA_NAME}.{tmp_table_name}', ctas_method)


@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
def test_run_async_cta_query(test_client: Any, ctas_method: str) -> None:
    table_name: str = f'{TEST_ASYNC_CTA}_{ctas_method.lower()}'
    result: Dict[str, Any] = run_sql(
        test_client, QUERY, cta=True, ctas_method=ctas_method, async_=True, tmp_table=table_name
    )
    query: Query = wait_for_success(result)
    assert QueryStatus.SUCCESS == query.status
    assert get_select_star(table_name, query.limit) in query.select_sql
    assert f'CREATE {ctas_method} {table_name} AS \n{QUERY}' == query.executed_sql
    assert QUERY == query.sql
    if backend() == 'presto':
        expected_rows = 1
    else:
        expected_rows = 0
    assert query.rows == expected_rows
    assert query.select_as_cta
    assert query.select_as_cta_used
    delete_tmp_view_or_table(table_name, ctas_method)


@pytest.mark.usefixtures('load_birth_names_data', 'login_as_admin')
@pytest.mark.parametrize('ctas_method', [CtasMethod.TABLE, CtasMethod.VIEW])
def test_run_async_cta_query_with_lower_limit(test_client: Any, ctas_method: str) -> None:
    tmp_table: str = f'{TEST_ASYNC_LOWER_LIMIT}_{ctas_method.lower()}'
    result: Dict[str, Any] = run_sql(
        test_client, QUERY, cta=True, ctas_method=ctas_method, async_=True, tmp_table=tmp_table
    )
    query: Query = wait_for_success(result)
    assert QueryStatus.SUCCESS == query.status
    sqlite_select_sql: str = f'SELECT\n  *\nFROM {tmp_table}\nLIMIT {query.limit}\nOFFSET 0'
    if backend() == 'sqlite':
        expected_select = sqlite_select_sql
    else:
        expected_select = get_select_star(tmp_table, query.limit)
    assert query.select_sql == expected_select
    assert f'CREATE {ctas_method} {tmp_table} AS \n{QUERY}' == query.executed_sql
    assert QUERY == query.sql
    if backend() == 'presto':
        expected_rows = 1
    else:
        expected_rows = 0
    assert query.rows == expected_rows
    assert query.limit == 50000
    assert query.select_as_cta
    assert query.select_as_cta_used
    delete_tmp_view_or_table(tmp_table, ctas_method)


SERIALIZATION_DATA: List[Tuple[Any, ...]] = [
    ('a', 4, 4.0, datetime.datetime(2019, 8, 18, 16, 39, 16, 660000))
]
CURSOR_DESCR: Tuple[Tuple[str, str], ...] = (('a', 'string'), ('b', 'int'), ('c', 'float'), ('d', 'datetime'))


def test_default_data_serialization() -> None:
    db_engine_spec: BaseEngineSpec = BaseEngineSpec()
    results: SupersetResultSet = SupersetResultSet(SERIALIZATION_DATA, CURSOR_DESCR, db_engine_spec)
    with mock.patch.object(db_engine_spec, 'expand_data', wraps=db_engine_spec.expand_data) as expand_data:
        data = sql_lab._serialize_and_expand_data(results, db_engine_spec, False, True)
        expand_data.assert_called_once()
    assert isinstance(data[0], list)


def test_new_data_serialization() -> None:
    db_engine_spec: BaseEngineSpec = BaseEngineSpec()
    results: SupersetResultSet = SupersetResultSet(SERIALIZATION_DATA, CURSOR_DESCR, db_engine_spec)
    with mock.patch.object(db_engine_spec, 'expand_data', wraps=db_engine_spec.expand_data) as expand_data:
        data = sql_lab._serialize_and_expand_data(results, db_engine_spec, True)
        expand_data.assert_not_called()
    assert isinstance(data[0], bytes)


@pytest.mark.usefixtures('load_birth_names_data')
def test_default_payload_serialization() -> None:
    use_new_deserialization: bool = False
    db_engine_spec: BaseEngineSpec = BaseEngineSpec()
    results: SupersetResultSet = SupersetResultSet(SERIALIZATION_DATA, CURSOR_DESCR, db_engine_spec)
    query: Dict[str, Any] = {'database_id': 1, 'sql': 'SELECT * FROM birth_names LIMIT 100', 'status': QueryStatus.PENDING}
    serialized_data, selected_columns, all_columns, expanded_columns = sql_lab._serialize_and_expand_data(
        results, db_engine_spec, use_new_deserialization
    )
    payload: Dict[str, Any] = {
        'query_id': 1,
        'status': QueryStatus.SUCCESS,
        'state': QueryStatus.SUCCESS,
        'data': serialized_data,
        'columns': all_columns,
        'selected_columns': selected_columns,
        'expanded_columns': expanded_columns,
        'query': query,
    }
    serialized: str = sql_lab._serialize_payload(payload, use_new_deserialization)
    assert isinstance(serialized, str)


@pytest.mark.usefixtures('load_birth_names_data')
def test_msgpack_payload_serialization() -> None:
    use_new_deserialization: bool = True
    db_engine_spec: BaseEngineSpec = BaseEngineSpec()
    results: SupersetResultSet = SupersetResultSet(SERIALIZATION_DATA, CURSOR_DESCR, db_engine_spec)
    query: Dict[str, Any] = {'database_id': 1, 'sql': 'SELECT * FROM birth_names LIMIT 100', 'status': QueryStatus.PENDING}
    serialized_data, selected_columns, all_columns, expanded_columns = sql_lab._serialize_and_expand_data(
        results, db_engine_spec, use_new_deserialization
    )
    payload: Dict[str, Any] = {
        'query_id': 1,
        'status': QueryStatus.SUCCESS,
        'state': QueryStatus.SUCCESS,
        'data': serialized_data,
        'columns': all_columns,
        'selected_columns': selected_columns,
        'expanded_columns': expanded_columns,
        'query': query,
    }
    serialized: bytes = sql_lab._serialize_payload(payload, use_new_deserialization)
    assert isinstance(serialized, bytes)


def test_create_table_as() -> None:
    q: ParsedQuery = ParsedQuery('SELECT * FROM outer_space;')
    assert 'CREATE TABLE tmp AS \nSELECT * FROM outer_space' == q.as_create_table('tmp')
    assert 'DROP TABLE IF EXISTS tmp;\nCREATE TABLE tmp AS \nSELECT * FROM outer_space' == q.as_create_table('tmp', overwrite=True)
    q = ParsedQuery('SELECT * FROM outer_space')
    assert 'CREATE TABLE tmp AS \nSELECT * FROM outer_space' == q.as_create_table('tmp')
    multi_line_query: str = "SELECT * FROM planets WHERE\nLuke_Father = 'Darth Vader'"
    q = ParsedQuery(multi_line_query)
    assert "CREATE TABLE tmp AS \nSELECT * FROM planets WHERE\nLuke_Father = 'Darth Vader'" == q.as_create_table('tmp')


def test_in_app_context() -> None:

    @celery_app.task(bind=True)
    def my_task(self: Any) -> bool:
        return has_app_context()

    with app.app_context():
        result: bool = my_task.apply().get()
        assert result is True, 'Task should have access to current_app within app context'
    result = my_task.apply().get()
    assert result is True, 'Task should have access to current_app outside of app context'


def delete_tmp_view_or_table(name: str, db_object_type: str) -> None:
    db.get_engine().execute(f'DROP {db_object_type} IF EXISTS {name}')


def wait_for_success(result: Dict[str, Any]) -> Query:
    query: Optional[Query] = None
    for _ in range(CELERY_SLEEP_TIME * 2):
        time.sleep(0.5)
        query = get_query_by_id(result['query']['serverId'])
        if query is not None and QueryStatus.SUCCESS == query.status:
            break
    assert query is not None
    return query

"""Unit tests for Superset"""
import unittest
import copy
from datetime import datetime
from io import BytesIO
import time
from typing import Any, Optional
from unittest import mock
from zipfile import ZipFile
from flask import Response
from flask.ctx import AppContext
from tests.integration_tests.conftest import with_feature_flags
from superset.charts.data.api import ChartDataRestApi
from superset.models.sql_lab import Query
from tests.integration_tests.base_tests import SupersetTestCase, test_client
from tests.integration_tests.annotation_layers.fixtures import create_annotation_layers
from tests.integration_tests.constants import ADMIN_USERNAME, GAMMA_NO_CSV_USERNAME, GAMMA_USERNAME
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from tests.integration_tests.test_app import app
from tests.integration_tests.fixtures.energy_dashboard import load_energy_table_with_slice, load_energy_table_data
import pytest
from superset.models.slice import Slice
from superset.commands.chart.data.get_data_command import ChartDataCommand
from superset.connectors.sqla.models import TableColumn, SqlaTable
from superset.errors import SupersetErrorType
from superset.extensions import async_query_manager_factory, db
from superset.models.annotations import AnnotationLayer
from superset.superset_typing import AdhocColumn
from superset.utils.core import AnnotationType, backend, get_example_default_schema, AdhocMetricExpressionType, ExtraFiltersReasonType
from superset.utils import json
from superset.utils.database import get_example_database, get_main_database
from superset.common.chart_data import ChartDataResultFormat, ChartDataResultType
from tests.common.query_context_generator import ANNOTATION_LAYERS
from tests.integration_tests.fixtures.query_context import get_query_context
from tests.integration_tests.test_app import app
CHART_DATA_URI = 'api/v1/chart/data'
CHARTS_FIXTURE_COUNT: int = 10
ADHOC_COLUMN_FIXTURE: dict[str, Any] = {'hasCustomLabel': True, 'label': 'male_or_female', 'sqlExpression': "case when gender = 'boy' then 'male' when gender = 'girl' then 'female' else 'other' end"}
INCOMPATIBLE_ADHOC_COLUMN_FIXTURE: dict[str, Any] = {'hasCustomLabel': True, 'label': 'exciting_or_boring', 'sqlExpression': "case when genre = 'Action' then 'Exciting' else 'Boring' end"}

@pytest.fixture(autouse=True)
def skip_by_backend(app_context: AppContext) -> None:
    if backend() == 'hive':
        pytest.skip('Skipping tests for Hive backend')

class BaseTestChartDataApi(SupersetTestCase):
    query_context_payload_template: Optional[dict[str, Any]] = None

    def setUp(self) -> None:
        self.login(ADMIN_USERNAME)
        if self.query_context_payload_template is None:
            BaseTestChartDataApi.query_context_payload_template = get_query_context('birth_names')
        self.query_context_payload: dict[str, Any] = copy.deepcopy(self.query_context_payload_template) or {}

    def get_expected_row_count(self, client_id: str) -> int:
        start_date: datetime = datetime.now()
        start_date = start_date.replace(year=start_date.year - 100, hour=0, minute=0, second=0)
        quoted_table_name: str = self.quote_name('birth_names')
        sql: str = f"\n                            SELECT COUNT(*) AS rows_count FROM (\n                                SELECT name AS name, SUM(num) AS sum__num\n                                FROM {quoted_table_name}\n                                WHERE ds >= '{start_date.strftime('%Y-%m-%d %H:%M:%S')}'\n                                AND gender = 'boy'\n                                GROUP BY name\n                                ORDER BY sum__num DESC\n                                LIMIT 100) AS inner__query\n                        "
        resp: dict[str, Any] = self.run_sql(sql, client_id, raise_on_error=True)
        db.session.query(Query).delete()
        db.session.commit()
        return resp['data'][0]['rows_count']

    def quote_name(self, name: str) -> str:
        if get_main_database().backend in {'presto', 'hive'}:
            with get_example_database().get_inspector() as inspector:
                return inspector.engine.dialect.identifier_preparer.quote_identifier(name)
        return name

@pytest.mark.chart_data_flow
class TestPostChartDataApi(BaseTestChartDataApi):
    # ...

@pytest.mark.chart_data_flow
class TestGetChartDataApi(BaseTestChartDataApi):
    # ...

@pytest.fixture
def physical_query_context(physical_dataset: SqlaTable) -> dict[str, Any]:
    return {'datasource': {'type': physical_dataset.type, 'id': physical_dataset.id}, 'queries': [{'columns': ['col1'], 'metrics': ['count'], 'orderby': [['col1', True]]}], 'result_type': ChartDataResultType.FULL, 'force': True}

@mock.patch('superset.common.query_context_processor.config', {**app.config, 'CACHE_DEFAULT_TIMEOUT': 1234, 'DATA_CACHE_CONFIG': {**app.config['DATA_CACHE_CONFIG'], 'CACHE_DEFAULT_TIMEOUT': None}})
def test_cache_default_timeout(test_client: test_client, login_as_admin, physical_query_context: dict[str, Any]) -> None:
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.json['result'][0]['cache_timeout'] == 1234

def test_custom_cache_timeout(test_client: test_client, login_as_admin, physical_query_context: dict[str, Any]) -> None:
    physical_query_context['custom_cache_timeout'] = 5678
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.json['result'][0]['cache_timeout'] == 5678

def test_time_filter_with_grain(test_client: test_client, login_as_admin, physical_query_context: dict[str, Any]) -> None:
    physical_query_context['queries'][0]['filters'] = [{'col': 'col5', 'op': 'TEMPORAL_RANGE', 'val': 'Last quarter : ', 'grain': 'P1W'}]
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    query: str = rv.json['result'][0]['query']
    backend: str = get_example_database().backend
    if backend == 'sqlite':
        assert "DATETIME(col5, 'start of day',             -strftime('%w', col5) || ' days') >=" in query
    elif backend == 'mysql':
        assert 'DATE(DATE_SUB(col5, INTERVAL DAYOFWEEK(col5) - 1 DAY)) >=' in query
    elif backend == 'postgresql':
        assert "DATE_TRUNC('week', col5) >=" in query
    elif backend == 'presto':
        assert "date_trunc('week', CAST(col5 AS TIMESTAMP)) >=" in query

def test_force_cache_timeout(test_client: test_client, login_as_admin, physical_query_context: dict[str, Any]) -> None:
    physical_query_context['custom_cache_timeout'] = -1
    test_client.post(CHART_DATA_URI, json=physical_query_context)
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.json['result'][0]['cached_dttm'] is None
    assert rv.json['result'][0]['is_cached'] is None

@mock.patch('superset.common.query_context_processor.config', {**app.config, 'CACHE_DEFAULT_TIMEOUT': 100000, 'DATA_CACHE_CONFIG': {**app.config['DATA_CACHE_CONFIG'], 'CACHE_DEFAULT_TIMEOUT': 3456}})
def test_data_cache_default_timeout(test_client: test_client, login_as_admin, physical_query_context: dict[str, Any]) -> None:
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.json['result'][0]['cache_timeout'] == 3456

def test_chart_cache_timeout(load_energy_table_with_slice, test_client: test_client, login_as_admin, physical_query_context: dict[str, Any]) -> None:
    slice_with_cache_timeout: Slice = load_energy_table_with_slice[0]
    slice_with_cache_timeout.cache_timeout = 20
    datasource: SqlaTable = db.session.query(SqlaTable).filter(SqlaTable.id == physical_query_context['datasource']['id']).first()
    datasource.cache_timeout = 1254
    db.session.commit()
    physical_query_context['form_data'] = {'slice_id': slice_with_cache_timeout.id}
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.json['result'][0]['cache_timeout'] == 20

@mock.patch('superset.common.query_context_processor.config', {**app.config, 'DATA_CACHE_CONFIG': {**app.config['DATA_CACHE_CONFIG'], 'CACHE_DEFAULT_TIMEOUT': 1010}})
def test_chart_cache_timeout_not_present(test_client: test_client, login_as_admin, physical_query_context: dict[str, Any]) -> None:
    datasource: SqlaTable = db.session.query(SqlaTable).filter(SqlaTable.id == physical_query_context['datasource']['id']).first()
    datasource.cache_timeout = 1980
    db.session.commit()
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.json['result'][0]['cache_timeout'] == 1980

@mock.patch('superset.common.query_context_processor.config', {**app.config, 'DATA_CACHE_CONFIG': {**app.config['DATA_CACHE_CONFIG'], 'CACHE_DEFAULT_TIMEOUT': 1010}})
def test_chart_cache_timeout_chart_not_found(test_client: test_client, login_as_admin, physical_query_context: dict[str, Any]) -> None:
    physical_query_context['form_data'] = {'slice_id': 0}
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.json['result'][0]['cache_timeout'] == 1010

@pytest.mark.parametrize('status_code,extras', [(200, {'where': '1 = 1'}), (200, {'having': 'count(*) > 0'}), (403, {'where': 'col1 in (select distinct col1 from physical_dataset)'}), (403, {'having': 'count(*) > (select count(*) from physical_dataset)'})])
@with_feature_flags(ALLOW_ADHOC_SUBQUERY=False)
@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
def test_chart_data_subquery_not_allowed(test_client: test_client, login_as_admin, physical_dataset: SqlaTable, physical_query_context: dict[str, Any], status_code: int, extras: dict[str, Any]) -> None:
    physical_query_context['queries'][0]['extras'] = extras
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.status_code == status_code

@pytest.mark.parametrize('status_code,extras', [(200, {'where': '1 = 1'}), (200, {'having': 'count(*) > 0'}), (200, {'where': 'col1 in (select distinct col1 from physical_dataset)'}), (200, {'having': 'count(*) > (select count(*) from physical_dataset)'})])
@with_feature_flags(ALLOW_ADHOC_SUBQUERY=True)
@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
def test_chart_data_subquery_allowed(test_client: test_client, login_as_admin, physical_dataset: SqlaTable, physical_query_context: dict[str, Any], status_code: int, extras: dict[str, Any]) -> None:
    physical_query_context['queries'][0]['extras'] = extras
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.status_code == status_code

"""Unit tests for Superset"""
import unittest
import copy
from datetime import datetime
from io import BytesIO
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock
from zipfile import ZipFile
from flask import Response
from flask.ctx import AppContext
import pytest
from superset.charts.data.api import ChartDataRestApi
from superset.models.sql_lab import Query
from tests.integration_tests.base_tests import SupersetTestCase, test_client
from tests.integration_tests.annotation_layers.fixtures import create_annotation_layers
from tests.integration_tests.constants import ADMIN_USERNAME, GAMMA_NO_CSV_USERNAME, GAMMA_USERNAME
from tests.integration_tests.fixtures.birth_names_dashboard import (
    load_birth_names_dashboard_with_slices,
    load_birth_names_data,
)
from tests.integration_tests.test_app import app
from superset.models.slice import Slice
from superset.commands.chart.data.get_data_command import ChartDataCommand
from superset.connectors.sqla.models import TableColumn, SqlaTable
from superset.errors import SupersetErrorType
from superset.extensions import async_query_manager_factory, db
from superset.models.annotations import AnnotationLayer
from superset.superset_typing import AdhocColumn
from superset.utils.core import (
    AnnotationType,
    backend,
    get_example_default_schema,
    AdhocMetricExpressionType,
    ExtraFiltersReasonType,
)
from superset.utils import json
from superset.utils.database import get_example_database, get_main_database
from superset.common.chart_data import ChartDataResultFormat, ChartDataResultType
from tests.common.query_context_generator import ANNOTATION_LAYERS
from tests.integration_tests.fixtures.query_context import get_query_context
CHART_DATA_URI: str = 'api/v1/chart/data'
CHARTS_FIXTURE_COUNT: int = 10
ADHOC_COLUMN_FIXTURE: Dict[str, Any] = {
    'hasCustomLabel': True,
    'label': 'male_or_female',
    'sqlExpression': (
        "case when gender = 'boy' then 'male' when gender = 'girl' then 'female' else 'other' end"
    ),
}
INCOMPATIBLE_ADHOC_COLUMN_FIXTURE: Dict[str, Any] = {
    'hasCustomLabel': True,
    'label': 'exciting_or_boring',
    'sqlExpression': "case when genre = 'Action' then 'Exciting' else 'Boring' end",
}


@pytest.fixture(autouse=True)
def func_bpgt20l2(app_context: AppContext) -> None:
    if backend() == 'hive':
        pytest.skip('Skipping tests for Hive backend')


class BaseTestChartDataApi(SupersetTestCase):
    query_context_payload_template: Optional[Dict[str, Any]] = None
    query_context_payload: Optional[Dict[str, Any]] = None

    def func_pf5d4brj(self) -> None:
        self.login(ADMIN_USERNAME)
        if self.query_context_payload_template is None:
            BaseTestChartDataApi.query_context_payload_template = get_query_context('birth_names')
        self.query_context_payload = copy.deepcopy(self.query_context_payload_template) or {}

    def func_ow9mgfd4(self, client_id: str) -> Any:
        start_date: datetime = datetime.now()
        start_date = start_date.replace(year=start_date.year - 100, hour=0, minute=0, second=0)
        quoted_table_name: str = self.quote_name('birth_names')
        sql: str = f"""
            SELECT COUNT(*) AS rows_count FROM (
                SELECT name AS name, SUM(num) AS sum__num
                FROM {quoted_table_name}
                WHERE ds >= '{start_date.strftime('%Y-%m-%d %H:%M:%S')}'
                AND gender = 'boy'
                GROUP BY name
                ORDER BY sum__num DESC
                LIMIT 100) AS inner__query
        """
        resp: Dict[str, Any] = self.run_sql(sql, client_id, raise_on_error=True)
        db.session.query(Query).delete()
        db.session.commit()
        return resp['data'][0]['rows_count']

    def func_3b8cs9wc(self, name: str) -> str:
        if get_main_database().backend in {'presto', 'hive'}:
            with get_example_database().get_inspector() as inspector:
                return inspector.engine.dialect.identifier_preparer.quote_identifier(name)
        return name

    def post_assert_metric(self, uri: str, payload: Dict[str, Any], metric: str) -> Response:
        return self.client.post(uri, json=payload)

    def get_expected_row_count(self, client_id: str) -> int:
        return 100  # Placeholder implementation

    def assert_row_count(self, rv: Response, expected_row_count: int) -> None:
        self.assertEqual(rv.json['result'][0]['rowcount'], expected_row_count)

    def get_user(self, username: str) -> Any:
        return db.session.query(User).filter_by(username=username).one()

    def get_dttm(self) -> datetime:
        return datetime.now()

    def get_birth_names_dataset(self) -> SqlaTable:
        return db.session.query(SqlaTable).filter_by(table_name='birth_names').one()


@pytest.mark.chart_data_flow
class TestPostChartDataApi(BaseTestChartDataApi):

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_veyt2ywr(self) -> None:
        self.query_context_payload['datasource'] = {'id': 1, 'type': 'table'}
        response: Dict[str, Any] = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
            ChartDataRestApi, self.query_context_payload)
        assert response == {'dashboard_id': None, 'dataset_id': 1, 'slice_id': None}
        self.query_context_payload['datasource'] = '1__table'
        response = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
            ChartDataRestApi, self.query_context_payload)
        assert response == {'dashboard_id': None, 'dataset_id': None, 'slice_id': None}
        self.query_context_payload['datasource'] = None
        self.query_context_payload['form_data'] = {'slice_id': 1}
        response = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
            ChartDataRestApi, self.query_context_payload)
        assert response == {'dashboard_id': None, 'dataset_id': None, 'slice_id': 1}
        self.query_context_payload['datasource'] = None
        self.query_context_payload['form_data'] = {'foo': 1}
        response = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
            ChartDataRestApi, self.query_context_payload)
        assert response == {'dashboard_id': None, 'dataset_id': None, 'slice_id': None}
        self.query_context_payload['form_data'] = {'dashboardId': 1}
        response = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
            ChartDataRestApi, self.query_context_payload)
        assert response == {'dashboard_id': 1, 'dataset_id': None, 'slice_id': None}
        self.query_context_payload['form_data'] = {'dashboardId': 1, 'slice_id': 2}
        response = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
            ChartDataRestApi, self.query_context_payload)
        assert response == {'dashboard_id': 1, 'dataset_id': None, 'slice_id': 2}
        self.query_context_payload['datasource'] = {'id': 3, 'type': 'table'}
        self.query_context_payload['form_data'] = {'dashboardId': 1, 'slice_id': 2}
        response = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
            ChartDataRestApi, self.query_context_payload)
        assert response == {'dashboard_id': 1, 'dataset_id': 3, 'slice_id': 2}

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.utils.decorators.g')
    def func_dnffa0f9(self, mock_g: mock.Mock) -> None:
        mock_g.logs_context = {}
        expected_row_count: int = self.get_expected_row_count('client_id_1')
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        self.assert_row_count(rv, expected_row_count)
        assert isinstance(mock_g.logs_context.get('dataset_id'), int)

    @staticmethod
    def func_27stg813(rv: Response, expected_row_count: int) -> None:
        assert rv.json['result'][0]['rowcount'] == expected_row_count

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.common.query_context_factory.config', {**app.config, 'ROW_LIMIT': 7})
    def func_2fw5pcpl(self) -> None:
        expected_row_count: int = 7
        del self.query_context_payload['queries'][0]['row_limit']
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        self.assert_row_count(rv, expected_row_count)

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.common.query_context_factory.config', {**app.config, 'SAMPLES_ROW_LIMIT': 5})
    def func_gqnetgss(self) -> None:
        expected_row_count: int = 5
        app.config['SAMPLES_ROW_LIMIT'] = expected_row_count
        self.query_context_payload['result_type'] = ChartDataResultType.SAMPLES
        del self.query_context_payload['queries'][0]['row_limit']
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        self.assert_row_count(rv, expected_row_count)
        assert 'GROUP BY' not in rv.json['result'][0]['query']

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.utils.core.current_app.config', {**app.config, 'SQL_MAX_ROW': 10})
    def func_oz0labo2(self) -> None:
        expected_row_count: int = 10
        self.query_context_payload['queries'][0]['row_limit'] = 10000000
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        self.assert_row_count(rv, expected_row_count)

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.utils.core.current_app.config', {**app.config, 'SQL_MAX_ROW': 5})
    def func_l8akvok1(self) -> None:
        expected_row_count: int = app.config['SQL_MAX_ROW']
        self.query_context_payload['result_type'] = ChartDataResultType.SAMPLES
        self.query_context_payload['queries'][0]['row_limit'] = 10000000
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        self.assert_row_count(rv, expected_row_count)
        assert 'GROUP BY' not in rv.json['result'][0]['query']

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.common.query_actions.config', {**app.config, 'SAMPLES_ROW_LIMIT': 5, 'SQL_MAX_ROW': 15})
    def func_9vb33x5e(self) -> None:
        expected_row_count: int = 10
        self.query_context_payload['result_type'] = ChartDataResultType.SAMPLES
        self.query_context_payload['queries'][0]['row_limit'] = expected_row_count
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        self.assert_row_count(rv, expected_row_count)
        assert 'GROUP BY' not in rv.json['result'][0]['query']

    def func_3dghfqlj(self) -> None:
        self.query_context_payload['result_type'] = 'qwerty'
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 400

    def func_y6zjbn1a(self) -> None:
        self.query_context_payload['result_format'] = 'qwerty'
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 400

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_vsx23kht(self) -> None:
        invalid_query_context: Dict[str, Any] = {'form_data': 'NOT VALID JSON'}
        rv: Response = self.client.post(CHART_DATA_URI, data=invalid_query_context, content_type='multipart/form-data')
        assert rv.status_code == 400
        assert rv.json['message'] == 'Request is not JSON'

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_jlzqlt81(self) -> None:
        self.query_context_payload['result_type'] = ChartDataResultType.QUERY
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_5w5v2zlt(self) -> None:
        """
        Chart data API: Test empty chart data with CSV result format
        """
        self.query_context_payload['result_format'] = 'csv'
        self.query_context_payload['queries'] = []
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 400

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_6he1nxjw(self) -> None:
        """
        Chart data API: Test empty chart data with Excel result format
        """
        self.query_context_payload['result_format'] = 'xlsx'
        self.query_context_payload['queries'] = []
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 400

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_hzllp6ku(self) -> None:
        """
        Chart data API: Test chart data with CSV result format
        """
        self.query_context_payload['result_format'] = 'csv'
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        assert rv.mimetype == 'text/csv'

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_vr3ontml(self) -> None:
        """
        Chart data API: Test chart data with Excel result format
        """
        self.query_context_payload['result_format'] = 'xlsx'
        mimetype: str = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        assert rv.mimetype == mimetype

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_ti2gsmjf(self) -> None:
        """
        Chart data API: Test chart data with multi-query CSV result format
        """
        self.query_context_payload['result_format'] = 'csv'
        self.query_context_payload['queries'].append(copy.deepcopy(self.query_context_payload['queries'][0]))
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        assert rv.mimetype == 'application/zip'
        zipfile: ZipFile = ZipFile(BytesIO(rv.data), 'r')
        assert zipfile.namelist() == ['query_1.csv', 'query_2.csv']

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_5hgxacog(self) -> None:
        """
        Chart data API: Test chart data with multi-query Excel result format
        """
        self.query_context_payload['result_format'] = 'xlsx'
        self.query_context_payload['queries'].append(copy.deepcopy(self.query_context_payload['queries'][0]))
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        assert rv.mimetype == 'application/zip'
        zipfile: ZipFile = ZipFile(BytesIO(rv.data), 'r')
        assert zipfile.namelist() == ['query_1.xlsx', 'query_2.xlsx']

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_jn0sk5ey(self) -> None:
        """
        Chart data API: Test chart data with CSV result format
        """
        self.logout()
        self.login(GAMMA_NO_CSV_USERNAME)
        self.query_context_payload['result_format'] = 'csv'
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 403

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_brzjq00y(self) -> None:
        """
        Chart data API: Test chart data with Excel result format
        """
        self.logout()
        self.login(GAMMA_NO_CSV_USERNAME)
        self.query_context_payload['result_format'] = 'xlsx'
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 403

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_ulzchktn(self) -> None:
        """
        Chart data API: Test chart data query with limit and offset
        """
        self.query_context_payload['queries'][0]['row_limit'] = 5
        self.query_context_payload['queries'][0]['row_offset'] = 0
        self.query_context_payload['queries'][0]['orderby'] = [['name', True]]
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        self.assert_row_count(rv, 5)
        result: Dict[str, Any] = rv.json['result'][0]
        if get_example_database().backend == 'presto':
            return
        offset: int = 2
        expected_name: str = result['data'][offset]['name']
        self.query_context_payload['queries'][0]['row_offset'] = offset
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        result = rv.json['result'][0]
        assert result['rowcount'] == 5
        assert result['data'][0]['name'] == expected_name

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_0arz4zmb(self) -> None:
        """
        Chart data API: Test chart data query with applied time extras
        """
        self.query_context_payload['queries'][0]['applied_time_extras'] = {
            '__time_range': '100 years ago : now',
            '__time_origin': 'now',
        }
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        data: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        assert data['result'][0]['applied_filters'] == [
            {'column': 'gender'},
            {'column': 'num'},
            {'column': 'name'},
            {'column': '__time_range'},
        ]
        expected_row_count: int = self.get_expected_row_count('client_id_2')
        assert data['result'][0]['rowcount'] == expected_row_count

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_7gqojdg9(self) -> None:
        """
        Chart data API: Ensure mixed case filter operator generates valid result
        """
        expected_row_count: int = 10
        self.query_context_payload['queries'][0]['filters'][0]['op'] = 'In'
        self.query_context_payload['queries'][0]['row_limit'] = expected_row_count
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        self.assert_row_count(rv, expected_row_count)

    @unittest.skip('Failing due to timezone difference')
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_08h7j5ut(self) -> None:
        """
        Chart data API: Ensure temporal column filter converts epoch to dttm expression
        """
        table: SqlaTable = self.get_birth_names_dataset()
        if table.database.backend == 'presto':
            return
        self.query_context_payload['queries'][0]['time_range'] = ''
        dttm: datetime = self.get_dttm()
        ms_epoch: float = dttm.timestamp() * 1000
        self.query_context_payload['queries'][0]['filters'][0] = {
            'col': 'ds',
            'op': '!=',
            'val': ms_epoch,
        }
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        response_payload: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        result: Dict[str, Any] = response_payload['result'][0]
        assert str(ms_epoch) not in result['query']
        dttm_col: Optional[TableColumn] = None
        for col in table.columns:
            if col.column_name == table.main_dttm_col:
                dttm_col = col
        if dttm_col:
            dttm_expression: str = table.database.db_engine_spec.convert_dttm(dttm_col.type, dttm)
            assert dttm_expression in result['query']
        else:
            raise Exception('ds column not found')

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_bpteftew(self) -> None:
        """
        Chart data API: Ensure prophet post transformation works
        """
        if backend() == 'hive':
            return
        time_grain: str = 'P1Y'
        self.query_context_payload['queries'][0]['is_timeseries'] = True
        self.query_context_payload['queries'][0]['groupby'] = []
        self.query_context_payload['queries'][0]['extras'] = {'time_grain_sqla': time_grain}
        self.query_context_payload['queries'][0]['granularity'] = 'ds'
        self.query_context_payload['queries'][0]['post_processing'] = [{
            'operation': 'prophet',
            'options': {
                'time_grain': time_grain,
                'periods': 3,
                'confidence_interval': 0.9,
            },
        }]
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        response_payload: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        result: Dict[str, Any] = response_payload['result'][0]
        row: Dict[str, Any] = result['data'][0]
        assert '__timestamp' in row
        assert 'sum__num' in row
        assert 'sum__num__yhat' in row
        assert 'sum__num__yhat_upper' in row
        assert 'sum__num__yhat_lower' in row
        assert result['rowcount'] == 103

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_6pueb0c8(self) -> None:
        """
        Chart data API: Ensure incorrect post processing returns correct response
        """
        if backend() == 'hive':
            return
        query_context: Dict[str, Any] = self.query_context_payload
        query: Dict[str, Any] = query_context['queries'][0]
        query['columns'] = ['name', 'gender']
        query['post_processing'] = [{
            'operation': 'pivot',
            'options': {
                'drop_missing_columns': False,
                'columns': ['gender'],
                'index': ['name'],
                'aggregates': {},
            },
        }]
        rv: Response = self.post_assert_metric(CHART_DATA_URI, query_context, 'data')
        assert rv.status_code == 400
        data: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        assert data['message'] == 'Error: Pivot operation must include at least one aggregate'

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_1r8lpdb6(self) -> None:
        self.query_context_payload['queries'][0]['filters'] = [{'col': 'non_existent_filter', 'op': '==', 'val': 'foo'}]
        self.query_context_payload['result_type'] = ChartDataResultType.QUERY
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        assert 'non_existent_filter' not in rv.json['result'][0]['query']

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_xofxjem4(self) -> None:
        self.query_context_payload['queries'][0]['filters'] = [{'col': 'gender', 'op': '==', 'val': 'foo'}]
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        assert rv.json['result'][0]['data'] == []
        self.assert_row_count(rv, 0)

    def func_t9hazahb(self) -> None:
        self.query_context_payload['queries'][0]['filters'] = []
        self.query_context_payload['queries'][0]['extras']['where'] = '(gender abc def)'
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 400

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_g0zqg498(self) -> None:
        self.query_context_payload['queries'][0]['filters'] = []
        self.query_context_payload['queries'][0]['extras']['where'] = "state = 'CA') OR (state = 'NY'"
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 400

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_891up4hq(self) -> None:
        self.query_context_payload['queries'][0]['filters'] = []
        self.query_context_payload['queries'][0]['extras']['where'] = '1 = 1 -- abc'
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_0vqzl2n8(self) -> None:
        self.query_context_payload['queries'][0]['filters'] = []
        self.query_context_payload['queries'][0]['orderby'] = [[{
            'expressionType': 'SQL',
            'sqlExpression': 'sum__num; select 1, 1',
        }, True]]
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 400

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_jpe2udel(self) -> None:
        self.query_context_payload['queries'][0]['filters'] = []
        self.query_context_payload['queries'][0]['extras']['having'] = "COUNT(1) = 0) UNION ALL SELECT 'abc', 1--comment"
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 400

    def func_lq5p7h83(self) -> None:
        self.query_context_payload['datasource'] = 'abc'
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 400

    def func_9jmj5us7(self) -> None:
        """
        Chart data API: Test chart data query not allowed
        """
        self.logout()
        self.login(GAMMA_USERNAME)
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 403
        assert rv.json['errors'][0]['error_type'] == SupersetErrorType.DATASOURCE_SECURITY_ACCESS_ERROR

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_nqpakgv6(self) -> None:
        self.query_context_payload['result_type'] = ChartDataResultType.QUERY
        self.query_context_payload['queries'][0]['filters'] = [{'col': 'gender', 'op': '==', 'val': 'boy'}]
        self.query_context_payload['queries'][0]['extras']['where'] = "('boy' = '{{ filter_values('gender', 'xyz' )[0] }}')"
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        result: str = rv.json['result'][0]['query']
        if get_example_database().backend != 'presto':
            assert "('boy' = 'boy')" in result

    @unittest.skip('Extremely flaky test on MySQL')
    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_ztfxn1wx(self) -> None:
        self.logout()
        app._got_first_request = False
        async_query_manager_factory.init_app(app)
        self.login(ADMIN_USERNAME)
        time.sleep(1)
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        time.sleep(1)
        assert rv.status_code == 202
        time.sleep(1)
        data: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        keys: List[str] = list(data.keys())
        self.assertCountEqual(keys, ['channel_id', 'job_id', 'user_id', 'status', 'errors', 'result_url'])

    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_lca26tfw(self) -> None:
        """
        Chart data API: Test chart data query returns results synchronously
        when results are already cached.
        """
        app._got_first_request = False
        async_query_manager_factory.init_app(app)

        class QueryContext:
            result_format: str = ChartDataResultFormat.JSON
            result_type: str = ChartDataResultType.FULL

        cmd_run_val: Dict[str, Any] = {
            'query_context': QueryContext(),
            'queries': [{'query': 'select * from foo'}],
        }
        with mock.patch.object(ChartDataCommand, 'run', return_value=cmd_run_val) as patched_run:
            self.query_context_payload['result_type'] = ChartDataResultType.FULL
            rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
            assert rv.status_code == 200
            data: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
            patched_run.assert_called_once_with(force_cached=True)
            assert data == {'result': [{'query': 'select * from foo'}]}

    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_ivunbp5i(self) -> None:
        """
        Chart data API: Test chart data query non-JSON format (async)
        """
        app._got_first_request = False
        async_query_manager_factory.init_app(app)
        self.query_context_payload['result_type'] = 'results'
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200

    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_5mbj46py(self) -> None:
        """
        Chart data API: Test chart data query (async)
        """
        app._got_first_request = False
        async_query_manager_factory.init_app(app)
        test_client.set_cookie(app.config['GLOBAL_ASYNC_QUERIES_JWT_COOKIE_NAME'], 'foo')
        rv: Response = test_client.post(CHART_DATA_URI, json=self.query_context_payload)
        assert rv.status_code == 401

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_1pfuclkc(self) -> None:
        """
        Chart data API: Query total rows
        """
        expected_row_count: int = self.get_expected_row_count('client_id_4')
        self.query_context_payload['queries'][0]['is_rowcount'] = True
        self.query_context_payload['queries'][0]['groupby'] = ['name']
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.json['result'][0]['data'][0]['rowcount'] == expected_row_count

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_hbt2icwo(self) -> None:
        """
        Chart data API: Query timegrains and columns
        """
        self.query_context_payload['queries'] = [
            {'result_type': ChartDataResultType.TIMEGRAINS},
            {'result_type': ChartDataResultType.COLUMNS},
        ]
        result: List[Dict[str, Any]] = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data').json['result']
        timegrain_data_keys: List[str] = list(result[0]['data'][0].keys())
        column_data_keys: List[str] = list(result[1]['data'][0].keys())
        assert list(timegrain_data_keys) == ['name', 'function', 'duration']
        assert list(column_data_keys) == ['column_name', 'verbose_name', 'dtype']

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_7w7xk2x3(self) -> None:
        SERIES_LIMIT: int = 5
        self.query_context_payload['queries'][0]['columns'] = ['state', 'name']
        self.query_context_payload['queries'][0]['series_columns'] = ['name']
        self.query_context_payload['queries'][0]['series_limit'] = SERIES_LIMIT
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        data: List[Dict[str, Any]] = rv.json['result'][0]['data']
        unique_names: set = {row['name'] for row in data}
        self.maxDiff = None
        assert len(unique_names) == SERIES_LIMIT
        assert {column for column in data[0].keys()} == {'state', 'name', 'sum__num'}

    @pytest.mark.usefixtures('create_annotation_layers', 'load_birth_names_dashboard_with_slices')
    def func_tchrzw07(self) -> None:
        """
        Chart data API: Test chart data query
        """
        annotation_layers: List[Dict[str, Any]] = []
        self.query_context_payload['queries'][0]['annotation_layers'] = annotation_layers
        annotation_layers.append(ANNOTATION_LAYERS[AnnotationType.FORMULA])
        interval_layer: AnnotationLayer = db.session.query(AnnotationLayer).filter(
            AnnotationLayer.name == 'name1').one()
        interval: Dict[str, Any] = ANNOTATION_LAYERS[AnnotationType.INTERVAL]
        interval['value'] = interval_layer.id
        annotation_layers.append(interval)
        event_layer: AnnotationLayer = db.session.query(AnnotationLayer).filter(
            AnnotationLayer.name == 'name2').one()
        event: Dict[str, Any] = ANNOTATION_LAYERS[AnnotationType.EVENT]
        event['value'] = event_layer.id
        annotation_layers.append(event)
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        data: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        assert len(data['result'][0]['annotation_data']) == 2

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_ltnw8gbd(self) -> None:
        """
        Chart data API: test query with literal colon characters in query, metrics,
        where clause and filters
        """
        owner: Any = self.get_user('admin')
        table: SqlaTable = SqlaTable(
            table_name='virtual_table_1',
            schema=get_example_default_schema(),
            owners=[owner],
            database=get_example_database(),
            sql="select ':foo' as foo, ':bar:' as bar, state, num from birth_names",
        )
        db.session.add(table)
        db.session.commit()
        table.fetch_metadata()
        request_payload: Dict[str, Any] = self.query_context_payload
        request_payload['datasource'] = {'type': 'table', 'id': table.id}
        request_payload['queries'][0]['columns'] = ['foo', 'bar', 'state']
        request_payload['queries'][0]['where'] = "':abc' != ':xyz:qwerty'"
        request_payload['queries'][0]['orderby'] = None
        request_payload['queries'][0]['metrics'] = [{
            'expressionType': AdhocMetricExpressionType.SQL,
            'sqlExpression': "sum(case when state = ':asdf' then 0 else 1 end)",
            'label': 'count',
        }]
        request_payload['queries'][0]['filters'] = [{'col': 'foo', 'op': '!=', 'val': ':qwerty:'}]
        rv: Response = self.post_assert_metric(CHART_DATA_URI, request_payload, 'data')
        db.session.delete(table)
        db.session.commit()
        assert rv.status_code == 200
        result: Dict[str, Any] = rv.json['result'][0]
        data: List[Dict[str, Any]] = result['data']
        assert {col for col in data[0].keys()} == {'foo', 'bar', 'state', 'count'}
        assert {row['foo'] for row in data} == {':foo'}
        assert {row['bar'] for row in data} == {':bar:'}
        assert "':asdf'" in result['query']
        assert "':xyz:qwerty'" in result['query']
        assert "':qwerty:'" in result['query']

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_uhk0hos0(self) -> None:
        request_payload: Dict[str, Any] = self.query_context_payload
        request_payload['queries'][0]['columns'] = ['name', 'gender']
        request_payload['queries'][0]['metrics'] = None
        request_payload['queries'][0]['orderby'] = []
        rv: Response = self.post_assert_metric(CHART_DATA_URI, request_payload, 'data')
        result: Dict[str, Any] = rv.json['result'][0]
        assert rv.status_code == 200
        assert 'name' in result['colnames']
        assert 'gender' in result['colnames']
        assert 'name' in result['query']
        assert 'gender' in result['query']
        assert list(result['data'][0].keys()) == ['name', 'gender']

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_jo0i78yu(self) -> None:
        request_payload: Dict[str, Any] = self.query_context_payload
        request_payload['queries'][0]['columns'] = [{
            'label': 'num divide by 10',
            'sqlExpression': 'num/10',
            'expressionType': 'SQL',
        }, 'name']
        request_payload['queries'][0]['metrics'] = None
        request_payload['queries'][0]['orderby'] = []
        rv: Response = self.post_assert_metric(CHART_DATA_URI, request_payload, 'data')
        result: Dict[str, Any] = rv.json['result'][0]
        assert rv.status_code == 200
        assert 'num divide by 10' in result['colnames']
        assert 'name' in result['colnames']
        assert 'num divide by 10' in result['query']
        assert 'name' in result['query']
        assert list(result['data'][0].keys()) == ['name', 'num divide by 10']


@pytest.mark.chart_data_flow
class TestGetChartDataApi(BaseTestChartDataApi):

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_zjv1btt7(self) -> None:
        """
        Chart data API: Test GET endpoint when query context is null
        """
        chart: Slice = db.session.query(Slice).filter_by(slice_name='Genders').one()
        rv: Response = self.get_assert_metric(f'api/v1/chart/{chart.id}/data/', 'get_data')
        data: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        assert data == {'message': 'Chart has no query context saved. Please save the chart again.'}

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_r1zuf75c(self) -> None:
        """
        Chart data API: Test GET endpoint
        """
        chart: Slice = db.session.query(Slice).filter_by(slice_name='Genders').one()
        chart.query_context = json.dumps({
            'datasource': {'id': chart.table.id, 'type': 'table'},
            'force': False,
            'queries': [{
                'time_range': '1900-01-01T00:00:00 : 2000-01-01T00:00:00',
                'granularity': 'ds',
                'filters': [],
                'extras': {'having': '', 'where': ''},
                'applied_time_extras': {},
                'columns': ['gender'],
                'metrics': ['sum__num'],
                'orderby': [['sum__num', False]],
                'annotation_layers': [],
                'row_limit': 50000,
                'timeseries_limit': 0,
                'order_desc': True,
                'url_params': {},
                'custom_params': {},
                'custom_form_data': {},
            }],
            'result_format': 'json',
            'result_type': 'full',
        })
        rv: Response = self.get_assert_metric(f'api/v1/chart/{chart.id}/data/', 'get_data')
        assert rv.mimetype == 'application/json'
        data: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        assert data['result'][0]['status'] == 'success'
        assert data['result'][0]['rowcount'] == 2

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_tkbuo46h(self) -> None:
        """
        Chart data API: Test GET endpoint
        """
        chart: Slice = db.session.query(Slice).filter_by(slice_name='Genders').one()
        chart.query_context = json.dumps({
            'datasource': {'id': chart.table.id, 'type': 'table'},
            'force': False,
            'queries': [{
                'time_range': '1900-01-01T00:00:00 : 2000-01-01T00:00:00',
                'granularity': 'ds',
                'filters': [{'col': 'ds', 'op': 'TEMPORAL_RANGE', 'val': 'No filter'}],
                'extras': {'having': '', 'where': ''},
                'applied_time_extras': {},
                'columns': [{
                    'columnType': 'BASE_AXIS',
                    'datasourceWarning': False,
                    'expressionType': 'SQL',
                    'label': 'My column',
                    'sqlExpression': 'ds',
                    'timeGrain': 'P1W',
                }],
                'metrics': ['sum__num'],
                'orderby': [['sum__num', False]],
                'annotation_layers': [],
                'row_limit': 50000,
                'timeseries_limit': 0,
                'order_desc': True,
                'url_params': {},
                'custom_params': {},
                'custom_form_data': {},
            }],
            'form_data': {
                'x_axis': {
                    'datasourceWarning': False,
                    'expressionType': 'SQL',
                    'label': 'My column',
                    'sqlExpression': 'ds',
                },
            },
            'result_format': 'json',
            'result_type': 'full',
        })
        rv: Response = self.get_assert_metric(f'api/v1/chart/{chart.id}/data/', 'get_data')
        assert rv.mimetype == 'application/json'
        data: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        assert data['result'][0]['status'] == 'success'
        if backend() == 'presto':
            assert data['result'][0]['rowcount'] == 41
        else:
            assert data['result'][0]['rowcount'] == 40

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_lz480v28(self) -> None:
        """
        Chart data API: Test GET endpoint with force cache parameter
        """
        chart: Slice = db.session.query(Slice).filter_by(slice_name='Genders').one()
        chart.query_context = json.dumps({
            'datasource': {'id': chart.table.id, 'type': 'table'},
            'force': False,
            'queries': [{
                'time_range': '1900-01-01T00:00:00 : 2000-01-01T00:00:00',
                'granularity': 'ds',
                'filters': [],
                'extras': {'having': '', 'where': ''},
                'applied_time_extras': {},
                'columns': ['gender'],
                'metrics': ['sum__num'],
                'orderby': [['sum__num', False]],
                'annotation_layers': [],
                'row_limit': 50000,
                'timeseries_limit': 0,
                'order_desc': True,
                'url_params': {},
                'custom_params': {},
                'custom_form_data': {},
            }],
            'result_format': 'json',
            'result_type': 'full',
        })
        self.get_assert_metric(f'api/v1/chart/{chart.id}/data/?force=true', 'get_data')
        rv: Response = self.get_assert_metric(f'api/v1/chart/{chart.id}/data/?force=true', 'get_data')
        assert rv.json['result'][0]['is_cached'] is None
        rv = self.get_assert_metric(f'api/v1/chart/{chart.id}/data/', 'get_data')
        assert rv.json['result'][0]['is_cached']

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    @mock.patch('superset.charts.data.api.QueryContextCacheLoader')
    def func_op95qlwb(self, cache_loader: mock.Mock) -> None:
        """
        Chart data cache API: Test chart data async cache request
        """
        app._got_first_request = False
        async_query_manager_factory.init_app(app)
        cache_loader.load.return_value = self.query_context_payload
        orig_run = ChartDataCommand.run

        def mock_run(self_cmd: ChartDataCommand, **kwargs: Any) -> Any:
            assert kwargs['force_cached'] is True
            return orig_run(self_cmd, force_cached=False)

        with mock.patch.object(ChartDataCommand, 'run', new=mock_run):
            rv: Response = self.get_assert_metric(f'{CHART_DATA_URI}/test-cache-key', 'data_from_cache')
            data: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        expected_row_count: int = self.get_expected_row_count('client_id_3')
        assert rv.status_code == 200
        assert data['result'][0]['rowcount'] == expected_row_count

    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    @mock.patch('superset.charts.data.api.QueryContextCacheLoader')
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_so5i3t7s(self, cache_loader: mock.Mock) -> None:
        """
        Chart data cache API: Test chart data async cache request with run failure
        """
        app._got_first_request = False
        async_query_manager_factory.init_app(app)
        cache_loader.load.return_value = self.query_context_payload
        rv: Response = self.get_assert_metric(f'{CHART_DATA_URI}/test-cache-key', 'data_from_cache')
        data: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        assert data['message'] == 'Error loading data from cache'

    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    @mock.patch('superset.charts.data.api.QueryContextCacheLoader')
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_penanp2q(self, cache_loader: mock.Mock) -> None:
        """
        Chart data cache API: Test chart data async cache request (no login)
        """
        if get_example_database().backend == 'presto':
            return
        app._got_first_request = False
        async_query_manager_factory.init_app(app)
        self.logout()
        cache_loader.load.return_value = self.query_context_payload
        orig_run = ChartDataCommand.run

        def mock_run(self_cmd: ChartDataCommand, **kwargs: Any) -> Any:
            assert kwargs['force_cached'] is True
            return orig_run(self_cmd, force_cached=False)

        with mock.patch.object(ChartDataCommand, 'run', new=mock_run):
            rv: Response = self.client.get(f'{CHART_DATA_URI}/test-cache-key')
        assert rv.status_code == 401

    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    def func_r48kde9e(self) -> None:
        """
        Chart data cache API: Test chart data async cache request with invalid cache key
        """
        app._got_first_request = False
        async_query_manager_factory.init_app(app)
        rv: Response = self.get_assert_metric(f'{CHART_DATA_URI}/test-cache-key', 'data_from_cache')
        assert rv.status_code == 404

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_lx6x7c4r(self) -> None:
        """
        Chart data API: Test query with adhoc column in both select and where clause
        """
        request_payload: Dict[str, Any] = get_query_context('birth_names')
        request_payload['queries'][0]['columns'] = [ADHOC_COLUMN_FIXTURE]
        request_payload['queries'][0]['filters'] = [
            {'col': ADHOC_COLUMN_FIXTURE, 'op': 'IN', 'val': ['male', 'female']}
        ]
        rv: Response = self.post_assert_metric(CHART_DATA_URI, request_payload, 'data')
        response_payload: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        result: Dict[str, Any] = response_payload['result'][0]
        data: List[Dict[str, Any]] = result['data']
        assert {column for column in data[0].keys()} == {'male_or_female', 'sum__num'}
        unique_genders: set = {row['male_or_female'] for row in data}
        assert unique_genders == {'male', 'female'}
        assert result['applied_filters'] == [{'column': 'male_or_female'}]

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_jw95q7rm(self) -> None:
        """
        Chart data API: Test query with adhoc column that fails to run on this dataset
        """
        request_payload: Dict[str, Any] = get_query_context('birth_names')
        request_payload['queries'][0]['columns'] = [ADHOC_COLUMN_FIXTURE]
        request_payload['queries'][0]['filters'] = [
            {'col': INCOMPATIBLE_ADHOC_COLUMN_FIXTURE, 'op': 'IN', 'val': ['Exciting']},
            {'col': ADHOC_COLUMN_FIXTURE, 'op': 'IN', 'val': ['male', 'female']},
        ]
        rv: Response = self.post_assert_metric(CHART_DATA_URI, request_payload, 'data')
        response_payload: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        result: Dict[str, Any] = response_payload['result'][0]
        data: List[Dict[str, Any]] = result['data']
        assert {column for column in data[0].keys()} == {'male_or_female', 'sum__num'}
        unique_genders: set = {row['male_or_female'] for row in data}
        assert unique_genders == {'male', 'female'}
        assert result['applied_filters'] == [{'column': 'male_or_female'}]
        assert result['rejected_filters'] == [{'column': 'exciting_or_boring', 'reason': ExtraFiltersReasonType.COL_NOT_IN_DATASOURCE}]


@pytest.fixture
def func_8pxq4q62(physical_dataset: SqlaTable) -> Dict[str, Any]:
    return {
        'datasource': {'type': physical_dataset.type, 'id': physical_dataset.id},
        'queries': [{'columns': ['col1'], 'metrics': ['count'], 'orderby': [['col1', True]]}],
        'result_type': ChartDataResultType.FULL,
        'force': True,
    }


@mock.patch('superset.common.query_context_processor.config', {
    **app.config,
    'CACHE_DEFAULT_TIMEOUT': 1234,
    'DATA_CACHE_CONFIG': {**app.config['DATA_CACHE_CONFIG'], 'CACHE_DEFAULT_TIMEOUT': None},
})
def func_m36nfloq(test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.json['result'][0]['cache_timeout'] == 1234


def func_02ueb1m2(test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    physical_query_context['custom_cache_timeout'] = 5678
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.json['result'][0]['cache_timeout'] == 5678


def func_qat1filx(test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    physical_query_context['queries'][0]['filters'] = [{
        'col': 'col5',
        'op': 'TEMPORAL_RANGE',
        'val': 'Last quarter : ',
        'grain': 'P1W',
    }]
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    query: str = rv.json['result'][0]['query']
    backend_name: str = get_example_database().backend
    if backend_name == 'sqlite':
        assert "DATETIME(col5, 'start of day',             -strftime('%w', col5) || ' days') >=" in query
    elif backend_name == 'mysql':
        assert 'DATE(DATE_SUB(col5, INTERVAL DAYOFWEEK(col5) - 1 DAY)) >=' in query
    elif backend_name == 'postgresql':
        assert "DATE_TRUNC('week', col5) >=" in query
    elif backend_name == 'presto':
        assert "date_trunc('week', CAST(col5 AS TIMESTAMP)) >=" in query

def func_qgzrsjin(test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    physical_query_context['custom_cache_timeout'] = -1
    test_client.post(CHART_DATA_URI, json=physical_query_context)
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.json['result'][0]['cached_dttm'] is None
    assert rv.json['result'][0]['is_cached'] is None

@mock.patch('superset.common.query_context_processor.config', {
    **app.config,
    'CACHE_DEFAULT_TIMEOUT': 100000,
    'DATA_CACHE_CONFIG': {**app.config['DATA_CACHE_CONFIG'], 'CACHE_DEFAULT_TIMEOUT': 3456},
})
def func_x33ezeua(test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.json['result'][0]['cache_timeout'] == 3456


def func_pr8t8irc(
    load_energy_table_with_slice: List[Slice],
    test_client: Any,
    login_as_admin: Any,
    physical_query_context: Dict[str, Any],
) -> None:
    slice_with_cache_timeout: Slice = load_energy_table_with_slice[0]
    slice_with_cache_timeout.cache_timeout = 20
    datasource: Optional[SqlaTable] = db.session.query(SqlaTable).filter(SqlaTable.id == physical_query_context['datasource']['id']).first()
    if datasource:
        datasource.cache_timeout = 1254
        db.session.commit()
    physical_query_context['form_data'] = {'slice_id': slice_with_cache_timeout.id}
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.json['result'][0]['cache_timeout'] == 20


@mock.patch('superset.common.query_context_processor.config', {
    **app.config,
    'DATA_CACHE_CONFIG': {**app.config['DATA_CACHE_CONFIG'], 'CACHE_DEFAULT_TIMEOUT': 1010},
})
def func_2i7rnoqu(test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    datasource: Optional[SqlaTable] = db.session.query(SqlaTable).filter(SqlaTable.id == physical_query_context['datasource']['id']).first()
    if datasource:
        datasource.cache_timeout = 1980
        db.session.commit()
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.json['result'][0]['cache_timeout'] == 1980


@mock.patch('superset.common.query_context_processor.config', {
    **app.config,
    'DATA_CACHE_CONFIG': {**app.config['DATA_CACHE_CONFIG'], 'CACHE_DEFAULT_TIMEOUT': 1010},
})
def func_3rg6j6fm(test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    physical_query_context['form_data'] = {'slice_id': 0}
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.json['result'][0]['cache_timeout'] == 1010


@pytest.mark.parametrize(
    'status_code,extras',
    [
        (200, {'where': '1 = 1'}),
        (200, {'having': 'count(*) > 0'}),
        (403, {'where': 'col1 in (select distinct col1 from physical_dataset)'}),
        (403, {'having': 'count(*) > (select count(*) from physical_dataset)'}),
    ],
)
@with_feature_flags(ALLOW_ADHOC_SUBQUERY=False)
@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
def func_0mkm7bnk(
    test_client: Any,
    login_as_admin: Any,
    physical_dataset: SqlaTable,
    physical_query_context: Dict[str, Any],
    status_code: int,
    extras: Dict[str, Any],
) -> None:
    physical_query_context['queries'][0]['extras'] = extras
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.status_code == status_code


@pytest.mark.parametrize(
    'status_code,extras',
    [
        (200, {'where': '1 = 1'}),
        (200, {'having': 'count(*) > 0'}),
        (200, {'where': 'col1 in (select distinct col1 from physical_dataset)'}),
        (200, {'having': 'count(*) > (select count(*) from physical_dataset)'}),
    ],
)
@with_feature_flags(ALLOW_ADHOC_SUBQUERY=True)
@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
def func_so09kskz(
    test_client: Any,
    login_as_admin: Any,
    physical_dataset: SqlaTable,
    physical_query_context: Dict[str, Any],
    status_code: int,
    extras: Dict[str, Any],
) -> None:
    physical_query_context['queries'][0]['extras'] = extras
    rv: Response = test_client.post(CHART_DATA_URI, json=physical_query_context)
    assert rv.status_code == status_code

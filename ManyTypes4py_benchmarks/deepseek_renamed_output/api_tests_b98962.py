"""Unit tests for Superset"""
import unittest
import copy
from datetime import datetime
from io import BytesIO
import time
from typing import Any, Optional, Dict, List, Set, Union, Tuple, cast
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
from typing_extensions import Literal

CHART_DATA_URI: str = 'api/v1/chart/data'
CHARTS_FIXTURE_COUNT: int = 10
ADHOC_COLUMN_FIXTURE: Dict[str, Any] = {
    'hasCustomLabel': True, 
    'label': 'male_or_female',
    'sqlExpression': "case when gender = 'boy' then 'male' when gender = 'girl' then 'female' else 'other' end"
}
INCOMPATIBLE_ADHOC_COLUMN_FIXTURE: Dict[str, Any] = {
    'hasCustomLabel': True, 
    'label': 'exciting_or_boring', 
    'sqlExpression': "case when genre = 'Action' then 'Exciting' else 'Boring' end"
}

@pytest.fixture(autouse=True)
def func_be3ogacl(app_context: AppContext) -> None:
    if backend() == 'hive':
        pytest.skip('Skipping tests for Hive backend')

class BaseTestChartDataApi(SupersetTestCase):
    query_context_payload_template: Optional[Dict[str, Any]] = None

    def func_hlwax7d5(self) -> None:
        self.login(ADMIN_USERNAME)
        if self.query_context_payload_template is None:
            BaseTestChartDataApi.query_context_payload_template = (
                get_query_context('birth_names'))
        self.query_context_payload = copy.deepcopy(self.query_context_payload_template) or {}

    def func_c8uzmxta(self, client_id: str) -> int:
        start_date = datetime.now()
        start_date = start_date.replace(year=start_date.year - 100, hour=0, minute=0, second=0)
        quoted_table_name = self.quote_name('birth_names')
        sql = f"""
            SELECT COUNT(*) AS rows_count FROM (
                SELECT name AS name, SUM(num) AS sum__num
                FROM {quoted_table_name}
                WHERE ds >= '{start_date.strftime('%Y-%m-%d %H:%M:%S')}'
                AND gender = 'boy'
                GROUP BY name
                ORDER BY sum__num DESC
                LIMIT 100) AS inner__query
        """
        resp = self.run_sql(sql, client_id, raise_on_error=True)
        db.session.query(Query).delete()
        db.session.commit()
        return resp['data'][0]['rows_count']

    def func_e5oa8xxe(self, name: str) -> str:
        if get_main_database().backend in {'presto', 'hive'}:
            with get_example_database().get_inspector() as inspector:
                return inspector.engine.dialect.identifier_preparer.quote_identifier(name)
        return name

@pytest.mark.chart_data_flow
class TestPostChartDataApi(BaseTestChartDataApi):

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_vyhp7smx(self) -> None:
        self.query_context_payload['datasource'] = {'id': 1, 'type': 'table'}
        response = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
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
    def func_dzj0sgia(self, mock_g: mock.Mock) -> None:
        mock_g.logs_context = {}
        expected_row_count = self.get_expected_row_count('client_id_1')
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        self.assert_row_count(rv, expected_row_count)
        assert isinstance(mock_g.logs_context.get('dataset_id'), int)

    @staticmethod
    def func_uxo0926u(rv: Response, expected_row_count: int) -> None:
        assert rv.json['result'][0]['rowcount'] == expected_row_count

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.common.query_context_factory.config', {**app.config, 'ROW_LIMIT': 7})
    def func_eyf2z0wn(self) -> None:
        expected_row_count = 7
        del self.query_context_payload['queries'][0]['row_limit']
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        self.assert_row_count(rv, expected_row_count)

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.common.query_context_factory.config', {**app.config, 'SAMPLES_ROW_LIMIT': 5})
    def func_kps89l1v(self) -> None:
        expected_row_count = 5
        app.config['SAMPLES_ROW_LIMIT'] = expected_row_count
        self.query_context_payload['result_type'] = ChartDataResultType.SAMPLES
        del self.query_context_payload['queries'][0]['row_limit']
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        self.assert_row_count(rv, expected_row_count)
        assert 'GROUP BY' not in rv.json['result'][0]['query']

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.utils.core.current_app.config', {**app.config, 'SQL_MAX_ROW': 10})
    def func_bm67esj6(self) -> None:
        expected_row_count = 10
        self.query_context_payload['queries'][0]['row_limit'] = 10000000
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        self.assert_row_count(rv, expected_row_count)

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.utils.core.current_app.config', {**app.config, 'SQL_MAX_ROW': 5})
    def func_u3bl5h92(self) -> None:
        expected_row_count = app.config['SQL_MAX_ROW']
        self.query_context_payload['result_type'] = ChartDataResultType.SAMPLES
        self.query_context_payload['queries'][0]['row_limit'] = 10000000
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        self.assert_row_count(rv, expected_row_count)
        assert 'GROUP BY' not in rv.json['result'][0]['query']

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.common.query_actions.config', {**app.config, 'SAMPLES_ROW_LIMIT': 5, 'SQL_MAX_ROW': 15})
    def func_oo0t5je9(self) -> None:
        expected_row_count = 10
        self.query_context_payload['result_type'] = ChartDataResultType.SAMPLES
        self.query_context_payload['queries'][0]['row_limit'] = expected_row_count
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        self.assert_row_count(rv, expected_row_count)
        assert 'GROUP BY' not in rv.json['result'][0]['query']

    def func_puaqrc0r(self) -> None:
        self.query_context_payload['result_type'] = 'qwerty'
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 400

    def func_h4874qt4(self) -> None:
        self.query_context_payload['result_format'] = 'qwerty'
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 400

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_t9nc0h5f(self) -> None:
        invalid_query_context = {'form_data': 'NOT VALID JSON'}
        rv = self.client.post(CHART_DATA_URI, data=invalid_query_context,
            content_type='multipart/form-data')
        assert rv.status_code == 400
        assert rv.json['message'] == 'Request is not JSON'

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_8jt4pkm3(self) -> None:
        self.query_context_payload['result_type'] = ChartDataResultType.QUERY
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_ixstpt8f(self) -> None:
        """
        Chart data API: Test empty chart data with CSV result format
        """
        self.query_context_payload['result_format'] = 'csv'
        self.query_context_payload['queries'] = []
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 400

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_i7s3ubvc(self) -> None:
        """
        Chart data API: Test empty chart data with Excel result format
        """
        self.query_context_payload['result_format'] = 'xlsx'
        self.query_context_payload['queries'] = []
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 400

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_ih48x543(self) -> None:
        """
        Chart data API: Test chart data with CSV result format
        """
        self.query_context_payload['result_format'] = 'csv'
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        assert rv.mimetype == 'text/csv'

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_a9knrvxg(self) -> None:
        """
        Chart data API: Test chart data with Excel result format
        """
        self.query_context_payload['result_format'] = 'xlsx'
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        assert rv.mimetype == mimetype

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_vtpjgred(self) -> None:
        """
        Chart data API: Test chart data with multi-query CSV result format
        """
        self.query_context_payload['result_format'] = 'csv'
        self.query_context_payload['queries'].append(self.query_context_payload['queries'][0])
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        assert rv.mimetype == 'application/zip'
        zipfile = ZipFile(BytesIO(rv.data), 'r')
        assert zipfile.namelist() == ['query_1.csv', 'query_2.csv']

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_ytu2z0gp(self) -> None:
        """
        Chart data API: Test chart data with multi-query Excel result format
        """
        self.query_context_payload['result_format'] = 'xlsx'
        self.query_context_payload['queries'].append(self.query_context_payload['queries'][0])
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 200
        assert rv.mimetype == 'application/zip'
        zipfile = ZipFile(BytesIO(rv.data), 'r')
        assert zipfile.namelist() == ['query_1.xlsx', 'query_2.xlsx']

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_6rfgsube(self) -> None:
        """
        Chart data API: Test chart data with CSV result format
        """
        self.logout()
        self.login(GAMMA_NO_CSV_USERNAME)
        self.query_context_payload['result_format'] = 'csv'
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 403

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_9i6r84p3(self) -> None:
        """
        Chart data API: Test chart data with Excel result format
        """
        self.logout()
        self.login(GAMMA_NO_CSV_USERNAME)
        self.query_context_payload['result_format'] = 'xlsx'
        rv = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, 'data')
        assert rv.status_code == 403

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def func_e10z8sl2(self) -> None:
        """
        Chart data API: Test chart data query with limit and offset
        """
        self.query_context_p
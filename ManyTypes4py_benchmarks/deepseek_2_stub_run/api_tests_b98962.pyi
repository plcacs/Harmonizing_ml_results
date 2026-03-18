```python
from typing import Any, Optional
from unittest.mock import Mock
from flask import Response
from flask.ctx import AppContext
from tests.integration_tests.conftest import with_feature_flags
from superset.charts.data.api import ChartDataRestApi
from superset.models.sql_lab import Query
from tests.integration_tests.base_tests import SupersetTestCase
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

CHART_DATA_URI: str = ...
CHARTS_FIXTURE_COUNT: int = ...
ADHOC_COLUMN_FIXTURE: dict[str, Any] = ...
INCOMPATIBLE_ADHOC_COLUMN_FIXTURE: dict[str, Any] = ...

@pytest.fixture
def skip_by_backend(app_context: Any) -> Any: ...

class BaseTestChartDataApi(SupersetTestCase):
    query_context_payload_template: Any = ...

    def setUp(self) -> None: ...
    def get_expected_row_count(self, client_id: str) -> Any: ...
    def quote_name(self, name: str) -> str: ...

@pytest.mark.chart_data_flow
class TestPostChartDataApi(BaseTestChartDataApi):
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test__map_form_data_datasource_to_dataset_id(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.utils.decorators.g')
    def test_with_valid_qc__data_is_returned(self, mock_g: Mock) -> None: ...
    
    @staticmethod
    def assert_row_count(rv: Response, expected_row_count: int) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.common.query_context_factory.config', {**app.config, 'ROW_LIMIT': 7})
    def test_without_row_limit__row_count_as_default_row_limit(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.common.query_context_factory.config', {**app.config, 'SAMPLES_ROW_LIMIT': 5})
    def test_as_samples_without_row_limit__row_count_as_default_samples_row_limit(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.utils.core.current_app.config', {**app.config, 'SQL_MAX_ROW': 10})
    def test_with_row_limit_bigger_then_sql_max_row__rowcount_as_sql_max_row(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.utils.core.current_app.config', {**app.config, 'SQL_MAX_ROW': 5})
    def test_as_samples_with_row_limit_bigger_then_sql_max_row_rowcount_as_sql_max_row(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.common.query_actions.config', {**app.config, 'SAMPLES_ROW_LIMIT': 5, 'SQL_MAX_ROW': 15})
    def test_with_row_limit_as_samples__rowcount_as_row_limit(self) -> None: ...
    
    def test_with_incorrect_result_type__400(self) -> None: ...
    
    def test_with_incorrect_result_format__400(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_invalid_payload__400(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_query_result_type__200(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_empty_request_with_csv_result_format(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_empty_request_with_excel_result_format(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_csv_result_format(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_excel_result_format(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_multi_query_csv_result_format(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_multi_query_excel_result_format(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_csv_result_format_when_actor_not_permitted_for_csv__403(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_excel_result_format_when_actor_not_permitted_for_excel__403(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_row_limit_and_offset__row_limit_and_offset_were_applied(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_applied_time_extras(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_in_op_filter__data_is_returned(self) -> None: ...
    
    @unittest.skip('Failing due to timezone difference')
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_dttm_filter(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_prophet(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_invalid_post_processing(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_query_result_type_and_non_existent_filter__filter_omitted(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_filter_suppose_to_return_empty_data__no_data_returned(self) -> None: ...
    
    def test_with_invalid_where_parameter__400(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_invalid_where_parameter_closing_unclosed__400(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_where_parameter_including_comment___200(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_orderby_parameter_with_second_query__400(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_invalid_having_parameter_closing_and_comment__400(self) -> None: ...
    
    def test_with_invalid_datasource__400(self) -> None: ...
    
    def test_with_not_permitted_actor__403(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_when_where_parameter_is_template_and_query_result_type__query_is_templated(self) -> None: ...
    
    @unittest.skip('Extremely flaky test on MySQL')
    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_async(self) -> None: ...
    
    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_async_cached_sync_response(self) -> None: ...
    
    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_async_results_type(self) -> None: ...
    
    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_async_invalid_token(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_rowcount(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_timegrains_and_columns_result_types(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_series_limit(self) -> None: ...
    
    @pytest.mark.usefixtures('create_annotation_layers', 'load_birth_names_dashboard_with_slices')
    def test_with_annotations_layers__annotations_data_returned(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_virtual_table_with_colons_as_datasource(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_table_columns_without_metrics(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_with_adhoc_column_without_metrics(self) -> None: ...

@pytest.mark.chart_data_flow
class TestGetChartDataApi(BaseTestChartDataApi):
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_get_data_when_query_context_is_null(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_get(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_get_with_x_axis_using_custom_sql(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_get_forced(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    @mock.patch('superset.charts.data.api.QueryContextCacheLoader')
    def test_chart_data_cache(self, cache_loader: Mock) -> None: ...
    
    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    @mock.patch('superset.charts.data.api.QueryContextCacheLoader')
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_cache_run_failed(self, cache_loader: Mock) -> None: ...
    
    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    @mock.patch('superset.charts.data.api.QueryContextCacheLoader')
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_cache_no_login(self, cache_loader: Mock) -> None: ...
    
    @with_feature_flags(GLOBAL_ASYNC_QUERIES=True)
    def test_chart_data_cache_key_error(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_with_adhoc_column(self) -> None: ...
    
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_with_incompatible_adhoc_column(self) -> None: ...

@pytest.fixture
def physical_query_context(physical_dataset: Any) -> dict[str, Any]: ...

@mock.patch('superset.common.query_context_processor.config', {**app.config, 'CACHE_DEFAULT_TIMEOUT': 1234, 'DATA_CACHE_CONFIG': {**app.config['DATA_CACHE_CONFIG'], 'CACHE_DEFAULT_TIMEOUT': None}})
def test_cache_default_timeout(test_client: Any, login_as_admin: Any, physical_query_context: dict[str, Any]) -> None: ...

def test_custom_cache_timeout(test_client: Any, login_as_admin: Any, physical_query_context: dict[str, Any]) -> None: ...

def test_time_filter_with_grain(test_client: Any, login_as_admin: Any, physical_query_context: dict[str, Any]) -> None: ...

def test_force_cache_timeout(test_client: Any, login_as_admin: Any, physical_query_context: dict[str, Any]) -> None: ...

@mock.patch('superset.common.query_context_processor.config', {**app.config, 'CACHE_DEFAULT_TIMEOUT': 100000, 'DATA_CACHE_CONFIG': {**app.config['DATA_CACHE_CONFIG'], 'CACHE_DEFAULT_TIMEOUT': 3456}})
def test_data_cache_default_timeout(test_client: Any, login_as_admin: Any, physical_query_context: dict[str, Any]) -> None: ...

def test_chart_cache_timeout(load_energy_table_with_slice: Any, test_client: Any, login_as_admin: Any, physical_query_context: dict[str, Any]) -> None: ...

@mock.patch('superset.common.query_context_processor.config', {**app.config, 'DATA_CACHE_CONFIG': {**app.config['DATA_CACHE_CONFIG'], 'CACHE_DEFAULT_TIMEOUT': 1010}})
def test_chart_cache_timeout_not_present(test_client: Any, login_as_admin: Any, physical_query_context: dict[str, Any]) -> None: ...

@mock.patch('superset.common.query_context_processor.config', {**app.config, 'DATA_CACHE_CONFIG': {**app.config['DATA_CACHE_CONFIG'], 'CACHE_DEFAULT_TIMEOUT': 1010}})
def test_chart_cache_timeout_chart_not_found(test_client: Any, login_as_admin: Any, physical_query_context: dict[str, Any]) -> None: ...

@pytest.mark.parametrize('status_code,extras', [(200, {'where': '1 = 1'}), (200, {'having': 'count(*) > 0'}), (403, {'where': 'col1 in (select distinct col1 from physical_dataset)'}), (403, {'having': 'count(*) > (select count(*) from physical_dataset)'})])
@with_feature_flags(ALLOW_ADHOC_SUBQUERY=False)
@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
def test_chart_data_subquery_not_allowed(test_client: Any, login_as_admin: Any, physical_dataset: Any, physical_query_context: dict[str, Any], status_code: int, extras: dict[str, str]) -> None: ...

@pytest.mark.parametrize('status_code,extras', [(200, {'where': '1 = 1'}), (200, {'having': 'count(*) > 0'}), (200, {'where': 'col1 in (select distinct col1 from physical_dataset)'}), (200, {'having': 'count(*) > (select count(*) from physical_dataset)'})])
@with_feature_flags(ALLOW_ADHOC_SUBQUERY=True)
@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
def test_chart_data_subquery_allowed(test_client: Any, login_as_admin: Any, physical_dataset: Any, physical_query_context: dict[str, Any], status_code: int, extras: dict[str, str]) -> None: ...
```
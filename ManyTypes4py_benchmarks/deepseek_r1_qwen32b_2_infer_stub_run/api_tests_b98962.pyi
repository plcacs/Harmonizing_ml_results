"""Unit tests for Superset - Type stubs"""

import unittest
import pytest
from datetime import datetime
from typing import Any, Optional, List, Dict, Union, Tuple
from unittest import mock
from flask import Response
from pytest import fixture
from superset.models.slice import Slice
from superset.models.sql_lab import Query
from superset.models.annotations import AnnotationLayer
from superset.charts.data.api import ChartDataRestApi
from superset.commands.chart.data.get_data_command import ChartDataCommand
from superset.connectors.sqla.models import SqlaTable
from superset.superset_typing import AdhocColumn
from superset.utils.core import ExtraFiltersReasonType
from superset.utils.json import json

class BaseTestChartDataApi(SupersetTestCase):
    query_context_payload_template: Optional[Any] = ...
    def setUp(self) -> None:
        ...
    def get_expected_row_count(self, client_id: str) -> int:
        ...
    def quote_name(self, name: str) -> str:
        ...
    def assert_row_count(self, rv: Response, expected_row_count: int) -> None:
        ...

class TestPostChartDataApi(BaseTestChartDataApi):
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test__map_form_data_datasource_to_dataset_id(self) -> None:
        ...
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.utils.decorators.g')
    def test_with_valid_qc__data_is_returned(self, mock_g: mock.MagicMock) -> None:
        ...
    # ... (other test methods with inferred types)

class TestGetChartDataApi(BaseTestChartDataApi):
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_get_data_when_query_context_is_null(self) -> None:
        ...
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_chart_data_get(self) -> None:
        ...
    # ... (other test methods with inferred types)

@pytest.fixture
def physical_query_context(physical_dataset: Any) -> Dict[str, Any]:
    ...

@mock.patch('superset.common.query_context_processor.config')
def test_cache_default_timeout(test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    ...

def test_custom_cache_timeout(test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    ...

def test_time_filter_with_grain(test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    ...

def test_force_cache_timeout(test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    ...

@mock.patch('superset.common.query_context_processor.config')
def test_data_cache_default_timeout(test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    ...

def test_chart_cache_timeout(load_energy_table_with_slice: Any, test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    ...

@mock.patch('superset.common.query_context_processor.config')
def test_chart_cache_timeout_not_present(test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    ...

@mock.patch('superset.common.query_context_processor.config')
def test_chart_cache_timeout_chart_not_found(test_client: Any, login_as_admin: Any, physical_query_context: Dict[str, Any]) -> None:
    ...

@pytest.mark.parametrize('status_code,extras', [(200, {'where': '1 = 1'}), (200, {'having': 'count(*) > 0'}), (403, {'where': 'col1 in (select distinct col1 from physical_dataset)'}), (403, {'having': 'count(*) > (select count(*) from physical_dataset)'})])
@with_feature_flags(ALLOW_ADHOC_SUBQUERY=False)
@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
def test_chart_data_subquery_not_allowed(test_client: Any, login_as_admin: Any, physical_dataset: Any, physical_query_context: Dict[str, Any], status_code: int, extras: Dict[str, Any]) -> None:
    ...

@pytest.mark.parametrize('status_code,extras', [(200, {'where': '1 = 1'}), (200, {'having': 'count(*) > 0'}), (200, {'where': 'col1 in (select distinct col1 from physical_dataset)'}), (200, {'having': 'count(*) > (select count(*) from physical_dataset)'})])
@with_feature_flags(ALLOW_ADHOC_SUBQUERY=True)
@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
def test_chart_data_subquery_allowed(test_client: Any, login_as_admin: Any, physical_dataset: Any, physical_query_context: Dict[str, Any], status_code: int, extras: Dict[str, Any]) -> None:
    ...
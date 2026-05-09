from typing import Any, Tuple
from unittest import mock
from uuid import uuid4
import pytest
from celery.exceptions import SoftTimeLimitExceeded
from parameterized import parameterized
from superset.async_events.cache_backend import RedisCacheBackend, RedisSentinelCacheBackend
from superset.commands.chart.data.get_data_command import ChartDataCommand
from superset.commands.chart.exceptions import ChartDataQueryFailedError
from superset.exceptions import SupersetException
from superset.extensions import async_query_manager, security_manager
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from tests.integration_tests.fixtures.query_context import get_query_context
from tests.integration_tests.fixtures.tags import with_tagging_system_feature
from tests.integration_tests.test_app import app

@pytest.mark.usefixtures('load_birth_names_data', 'load_birth_names_dashboard_with_slices')
class TestAsyncQueries(SupersetTestCase):
    @parameterized.expand([(Tuple[str, Any],)]  # type: ignore
    @mock.patch('superset.tasks.async_queries.set_form_data')
    @mock.patch.object(async_query_manager, 'update_job')
    def test_load_chart_data_into_cache(
        self, cache_type: Tuple[str, Any], cache_backend: Any,  # type: ignore
        mock_update_job: Any,  # type: ignore
        mock_set_form_data: Any,  # type: ignore
    ) -> None:
        # ... rest of the method ...

    @parameterized.expand([(Tuple[str, Any],)]  # type: ignore
    @mock.patch.object(ChartDataCommand, 'run', side_effect=ChartDataQueryFailedError('Error: foo'))
    @mock.patch.object(async_query_manager, 'update_job')
    def test_load_chart_data_into_cache_error(
        self, cache_type: Tuple[str, Any], cache_backend: Any,  # type: ignore
        mock_update_job: Any,  # type: ignore
        mock_run_command: Any,  # type: ignore
    ) -> None:
        # ... rest of the method ...

    @parameterized.expand([(Tuple[str, Any],)]  # type: ignore
    @mock.patch.object(ChartDataCommand, 'run')
    @mock.patch.object(async_query_manager, 'update_job')
    def test_soft_timeout_load_chart_data_into_cache(
        self, cache_type: Tuple[str, Any], cache_backend: Any,  # type: ignore
        mock_update_job: Any,  # type: ignore
        mock_run_command: Any,  # type: ignore
    ) -> None:
        # ... rest of the method ...

    @parameterized.expand([(Tuple[str, Any],)]  # type: ignore
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @pytest.mark.skip(reason='This test will be changed to use the api/v1/data')
    @mock.patch.object(async_query_manager, 'update_job')
    def test_load_explore_json_into_cache(
        self, cache_type: Tuple[str, Any], cache_backend: Any,  # type: ignore
        mock_update_job: Any,  # type: ignore
    ) -> None:
        # ... rest of the method ...

    @parameterized.expand([(Tuple[str, Any],)]  # type: ignore
    @mock.patch.object(async_query_manager, 'update_job')
    @mock.patch('superset.tasks.async_queries.set_form_data')
    def test_load_explore_json_into_cache_error(
        self, cache_type: Tuple[str, Any], cache_backend: Any,  # type: ignore
        mock_set_form_data: Any,  # type: ignore
        mock_update_job: Any,  # type: ignore
    ) -> None:
        # ... rest of the method ...

    @parameterized.expand([(Tuple[str, Any],)]  # type: ignore
    @mock.patch.object(ChartDataCommand, 'run')
    @mock.patch.object(async_query_manager, 'update_job')
    def test_soft_timeout_load_explore_json_into_cache(
        self, cache_type: Tuple[str, Any], cache_backend: Any,  # type: ignore
        mock_update_job: Any,  # type: ignore
        mock_run_command: Any,  # type: ignore
    ) -> None:
        # ... rest of the method ...

from unittest import mock
from uuid import UUID
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

class TestAsyncQueries(SupersetTestCase):

    def test_load_chart_data_into_cache(self, cache_type: str, cache_backend: mock.Mock, mock_update_job: mock.Mock, mock_set_form_data: mock.Mock) -> None:

    def test_load_chart_data_into_cache_error(self, cache_type: str, cache_backend: mock.Mock, mock_update_job: mock.Mock, mock_run_command: mock.Mock) -> None:

    def test_soft_timeout_load_chart_data_into_cache(self, cache_type: str, cache_backend: mock.Mock, mock_update_job: mock.Mock, mock_run_command: mock.Mock) -> None:

    def test_load_explore_json_into_cache(self, cache_type: str, cache_backend: mock.Mock, mock_update_job: mock.Mock) -> None:

    def test_load_explore_json_into_cache_error(self, cache_type: str, cache_backend: mock.Mock, mock_set_form_data: mock.Mock, mock_update_job: mock.Mock) -> None:

    def test_soft_timeout_load_explore_json_into_cache(self, cache_type: str, cache_backend: mock.Mock, mock_update_job: mock.Mock, mock_run_command: mock.Mock) -> None:

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Generator, List, Optional, Union
from uuid import UUID
import pytest
from flask.ctx import AppContext
from unittest.mock import MagicMock
from sqlalchemy.sql import func
from superset.reports.models import ReportSchedule, ReportExecutionLog, ReportState
from superset.models.core import Database
from superset.models.slice import Slice
from superset.models.dashboard import Dashboard
from superset.key_value.models import KeyValueEntry
from superset.utils.database import get_example_database

@pytest.fixture
def create_report_email_chart() -> ReportSchedule:
    ...

@pytest.fixture
def create_report_email_chart_with_cc_and_bcc() -> ReportSchedule:
    ...

@pytest.fixture
def create_report_email_chart_alpha_owner(get_user: Callable[[str], User]) -> ReportSchedule:
    ...

@pytest.fixture
def create_report_email_chart_force_screenshot() -> ReportSchedule:
    ...

@pytest.fixture
def create_report_email_chart_with_csv() -> ReportSchedule:
    ...

@pytest.fixture
def create_report_email_chart_with_text() -> ReportSchedule:
    ...

@pytest.fixture
def create_report_email_chart_with_csv_no_query_context() -> ReportSchedule:
    ...

@pytest.fixture
def create_report_email_dashboard() -> ReportSchedule:
    ...

@pytest.fixture
def create_report_email_dashboard_force_screenshot() -> ReportSchedule:
    ...

@pytest.fixture
def create_report_slack_chart() -> ReportSchedule:
    ...

@pytest.fixture
def create_report_slack_chartv2() -> ReportSchedule:
    ...

@pytest.fixture
def create_report_slack_chart_with_csv() -> ReportSchedule:
    ...

@pytest.fixture
def create_report_slack_chart_with_text() -> ReportSchedule:
    ...

@pytest.fixture
def create_report_slack_chart_working() -> ReportSchedule:
    ...

@pytest.fixture
def create_alert_slack_chart_success() -> ReportSchedule:
    ...

@pytest.fixture
def create_alert_slack_chart_grace(request: pytest.FixtureRequest) -> ReportSchedule:
    ...

@pytest.fixture
def create_alert_email_chart(request: pytest.FixtureRequest) -> ReportSchedule:
    ...

@pytest.fixture
def create_no_alert_email_chart(request: pytest.FixtureRequest) -> ReportSchedule:
    ...

@pytest.fixture
def create_mul_alert_email_chart(request: pytest.FixtureRequest) -> ReportSchedule:
    ...

@pytest.fixture
def create_invalid_sql_alert_email_chart(request: pytest.FixtureRequest, app_context: AppContext) -> ReportSchedule:
    ...

@contextmanager
def create_test_table_context(database: Database) -> Generator[None, None, None]:
    ...

def get_target_from_report_schedule(report_schedule: ReportSchedule) -> List[str]:
    ...

def get_cctarget_from_report_schedule(report_schedule: ReportSchedule) -> List[str]:
    ...

def get_bcctarget_from_report_schedule(report_schedule: ReportSchedule) -> List[str]:
    ...

def get_error_logs_query(report_schedule: ReportSchedule) -> BaseQuery:
    ...

def get_notification_error_sent_count(report_schedule: ReportSchedule) -> int:
    ...

def assert_log(state: ReportState, error_message: Optional[str] = None) -> None:
    ...

def test_email_chart_report_schedule_with_cc_bcc(screenshot_mock: MagicMock, email_mock: MagicMock, create_report_email_chart_with_cc_and_bcc: ReportSchedule) -> None:
    ...

def test_email_chart_report_schedule(screenshot_mock: MagicMock, email_mock: MagicMock, create_report_email_chart: ReportSchedule) -> None:
    ...

def test_email_chart_report_schedule_alpha_owner(screenshot_mock: MagicMock, email_mock: MagicMock, create_report_email_chart_alpha_owner: ReportSchedule) -> None:
    ...

def test_email_chart_report_schedule_force_screenshot(screenshot_mock: MagicMock, email_mock: MagicMock, create_report_email_chart_force_screenshot: ReportSchedule) -> None:
    ...

def test_email_chart_alert_schedule(screenshot_mock: MagicMock, email_mock: MagicMock, create_alert_email_chart: ReportSchedule) -> None:
    ...

def test_email_chart_report_dry_run(screenshot_mock: MagicMock, email_mock: MagicMock, create_report_email_chart: ReportSchedule) -> None:
    ...

def test_email_chart_report_schedule_with_csv(csv_mock: MagicMock, email_mock: MagicMock, mock_open: MagicMock, mock_urlopen: MagicMock, create_report_email_chart_with_csv: ReportSchedule) -> None:
    ...

def test_email_chart_report_schedule_with_csv_no_query_context(screenshot_mock: MagicMock, csv_mock: MagicMock, email_mock: MagicMock, mock_open: MagicMock, mock_urlopen: MagicMock, create_report_email_chart_with_csv_no_query_context: ReportSchedule) -> None:
    ...

def test_email_chart_report_schedule_with_text(dataframe_mock: MagicMock, email_mock: MagicMock, mock_open: MagicMock, mock_urlopen: MagicMock, create_report_email_chart_with_text: ReportSchedule) -> None:
    ...

def test_email_dashboard_report_schedule(screenshot_mock: MagicMock, email_mock: MagicMock, create_report_email_dashboard: ReportSchedule) -> None:
    ...

def test_email_dashboard_report_schedule_with_tab_anchor(_email_mock: MagicMock, _screenshot_mock: MagicMock) -> None:
    ...

def test_email_dashboard_report_schedule_disabled_tabs(_email_mock: MagicMock, _screenshot_mock: MagicMock) -> None:
    ...

def test_email_dashboard_report_schedule_force_screenshot(screenshot_mock: MagicMock, email_mock: MagicMock, create_report_email_dashboard_force_screenshot: ReportSchedule) -> None:
    ...

def test_slack_chart_report_schedule_converts_to_v2(screenshot_mock: MagicMock, slack_client_mock: MagicMock, slack_should_use_v2_api_mock: MagicMock, get_channels_with_search_mock: MagicMock, create_report_slack_chart: ReportSchedule) -> None:
    ...

def test_slack_chart_report_schedule_v2(screenshot_mock: MagicMock, slack_client_mock: MagicMock, slack_should_use_v2_api_mock: MagicMock, get_channels_with_search_mock: MagicMock, create_report_slack_chart: ReportSchedule) -> None:
    ...

def test_slack_chart_report_schedule_with_errors(screenshot_mock: MagicMock, web_client_mock: MagicMock, create_report_slack_chart: ReportSchedule) -> None:
    ...

def test_slack_chart_report_schedule_with_csv(csv_mock: MagicMock, mock_open: MagicMock, mock_urlopen: MagicMock, slack_client_mock_class: MagicMock, slack_should_use_v2_api_mock: MagicMock, create_report_slack_chart_with_csv: ReportSchedule) -> None:
    ...

def test_slack_chart_report_schedule_with_text(dataframe_mock: MagicMock, slack_client_mock_class: MagicMock, mock_open: MagicMock, mock_urlopen: MagicMock, slack_should_use_v2_api_mock: MagicMock, create_report_slack_chart_with_text: ReportSchedule) -> None:
    ...

def test_report_schedule_not_found(create_report_slack_chart: ReportSchedule) -> None:
    ...

def test_report_schedule_working(create_report_slack_chart_working: ReportSchedule) -> None:
    ...

def test_report_schedule_working_timeout(create_report_slack_chart_working: ReportSchedule) -> None:
    ...

def test_report_schedule_success_grace(create_alert_slack_chart_success: ReportSchedule) -> None:
    ...

def test_report_schedule_success_grace_end(slack_client_mock_class: MagicMock, screenshot_mock: MagicMock, file_upload_mock: MagicMock, create_alert_slack_chart_grace: ReportSchedule) -> None:
    ...

def test_alert_limit_is_applied(screenshot_mock: MagicMock, email_mock: MagicMock, create_alert_email_chart: ReportSchedule) -> None:
    ...

def test_email_dashboard_report_fails(screenshot_mock: MagicMock, email_mock: MagicMock, create_report_email_dashboard: ReportSchedule) -> None:
    ...

def test_email_dashboard_report_fails_uncaught_exception(screenshot_mock: MagicMock, email_mock: MagicMock, create_report_email_dashboard: ReportSchedule) -> None:
    ...

def test_slack_chart_alert(screenshot_mock: MagicMock, email_mock: MagicMock, create_alert_email_chart: ReportSchedule) -> None:
    ...

def test_slack_chart_alert_no_attachment(email_mock: MagicMock, create_alert_email_chart: ReportSchedule) -> None:
    ...

def test_slack_token_callable_chart_report(screenshot_mock: MagicMock, slack_client_mock_class: MagicMock, create_report_slack_chart: ReportSchedule) -> None:
    ...

def test_email_chart_no_alert(create_no_alert_email_chart: ReportSchedule) -> None:
    ...

def test_email_mul_alert(create_mul_alert_email_chart: ReportSchedule) -> None:
    ...

def test_soft_timeout_alert(email_mock: MagicMock, create_alert_email_chart: ReportSchedule) -> None:
    ...

def test_soft_timeout_screenshot(screenshot_mock: MagicMock, email_mock: MagicMock, create_alert_email_chart: ReportSchedule) -> None:
    ...

def test_soft_timeout_csv(csv_mock: MagicMock, email_mock: MagicMock, mock_open: MagicMock, mock_urlopen: MagicMock, create_report_email_chart_with_csv: ReportSchedule) -> None:
    ...

def test_fail_csv(csv_mock: MagicMock, mock_open: MagicMock, mock_urlopen: MagicMock, email_mock: MagicMock, create_report_email_chart_with_csv: ReportSchedule) -> None:
    ...

def test_email_disable_screenshot(email_mock: MagicMock, create_alert_email_chart: ReportSchedule) -> None:
    ...

def test_invalid_sql_alert(email_mock: MagicMock, create_invalid_sql_alert_email_chart: ReportSchedule) -> None:
    ...

def test_grace_period_error(email_mock: MagicMock, create_invalid_sql_alert_email_chart: ReportSchedule) -> None:
    ...

def test_grace_period_error_flap(screenshot_mock: MagicMock, email_mock: MagicMock, create_invalid_sql_alert_email_chart: ReportSchedule) -> None:
    ...

def test_prune_log_soft_time_out(bulk_delete_logs: MagicMock, create_report_email_dashboard: ReportSchedule) -> None:
    ...

def test__send_with_client_errors(notification_mock: MagicMock, logger_mock: MagicMock) -> None:
    ...

def test__send_with_multiple_errors(notification_mock: MagicMock, logger_mock: MagicMock) -> None:
    ...

def test__send_with_server_errors(notification_mock: MagicMock, logger_mock: MagicMock) -> None:
    ...
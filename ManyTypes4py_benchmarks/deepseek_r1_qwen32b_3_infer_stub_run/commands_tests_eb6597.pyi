from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Generator,
    List,
    Optional,
    Union,
)
from uuid import UUID

from flask_sqlalchemy import BaseQuery
from sqlalchemy.orm import Session
from superset.models.core import Database
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.reports.models import (
    ReportDataFormat,
    ReportExecutionLog,
    ReportRecipientType,
    ReportSchedule,
    ReportScheduleType,
    ReportScheduleValidatorType,
    ReportState,
)
from superset.utils.database import get_example_database

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

@contextmanager
def assert_log(state: ReportState, error_message: Optional[str] = None) -> Generator[None, None, None]:
    ...

@contextmanager
def create_test_table_context(database: Database) -> Generator[Session, None, None]:
    ...

@pytest.fixture
def create_report_email_chart() -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_report_email_chart_with_cc_and_bcc() -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_report_email_chart_alpha_owner(get_user: Any) -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_report_email_chart_force_screenshot() -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_report_email_chart_with_csv() -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_report_email_chart_with_text() -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_report_email_chart_with_csv_no_query_context() -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_report_email_dashboard() -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_report_email_dashboard_force_screenshot() -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_report_slack_chart() -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_report_slack_chartv2() -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_report_slack_chart_with_csv() -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_report_slack_chart_with_text() -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_report_slack_chart_working() -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_alert_slack_chart_success() -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_alert_slack_chart_grace(request: Any) -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_alert_email_chart(request: Any) -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_no_alert_email_chart(request: Any) -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_mul_alert_email_chart(request: Any) -> Generator[ReportSchedule, None, None]:
    ...

@pytest.fixture
def create_invalid_sql_alert_email_chart(request: Any, app_context: Any) -> Generator[ReportSchedule, None, None]:
    ...

def test_email_chart_report_schedule_with_cc_bcc(screenshot_mock: Any, email_mock: Any, create_report_email_chart_with_cc_and_bcc: ReportSchedule) -> None:
    ...

def test_email_chart_report_schedule(screenshot_mock: Any, email_mock: Any, create_report_email_chart: ReportSchedule) -> None:
    ...

def test_email_chart_report_schedule_alpha_owner(screenshot_mock: Any, email_mock: Any, create_report_email_chart_alpha_owner: ReportSchedule) -> None:
    ...

def test_email_chart_report_schedule_force_screenshot(screenshot_mock: Any, email_mock: Any, create_report_email_chart_force_screenshot: ReportSchedule) -> None:
    ...

def test_email_chart_alert_schedule(screenshot_mock: Any, email_mock: Any, create_alert_email_chart: ReportSchedule) -> None:
    ...

def test_email_chart_report_dry_run(screenshot_mock: Any, email_mock: Any, create_report_email_chart: ReportSchedule) -> None:
    ...

def test_email_chart_report_schedule_with_csv(csv_mock: Any, email_mock: Any, mock_open: Any, mock_urlopen: Any, create_report_email_chart_with_csv: ReportSchedule) -> None:
    ...

def test_email_chart_report_schedule_with_csv_no_query_context(screenshot_mock: Any, csv_mock: Any, email_mock: Any, mock_open: Any, mock_urlopen: Any, create_report_email_chart_with_csv_no_query_context: ReportSchedule) -> None:
    ...

def test_email_chart_report_schedule_with_text(dataframe_mock: Any, email_mock: Any, mock_open: Any, mock_urlopen: Any, create_report_email_chart_with_text: ReportSchedule) -> None:
    ...

def test_email_dashboard_report_schedule(screenshot_mock: Any, email_mock: Any, create_report_email_dashboard: ReportSchedule) -> None:
    ...

def test_email_dashboard_report_schedule_with_tab_anchor(_email_mock: Any, _screenshot_mock: Any) -> None:
    ...

def test_email_dashboard_report_schedule_disabled_tabs(_email_mock: Any, _screenshot_mock: Any) -> None:
    ...

def test_email_dashboard_report_schedule_force_screenshot(screenshot_mock: Any, email_mock: Any, create_report_email_dashboard_force_screenshot: ReportSchedule) -> None:
    ...

def test_slack_chart_report_schedule_converts_to_v2(screenshot_mock: Any, slack_client_mock: Any, slack_should_use_v2_api_mock: Any, get_channels_with_search_mock: Any, create_report_slack_chart: ReportSchedule) -> None:
    ...

def test_slack_chart_report_schedule_v2(screenshot_mock: Any, slack_client_mock: Any, slack_should_use_v2_api_mock: Any, get_channels_with_search_mock: Any, create_report_slack_chart: ReportSchedule) -> None:
    ...

def test_slack_chart_report_schedule_with_errors(screenshot_mock: Any, web_client_mock: Any, create_report_slack_chart: ReportSchedule) -> None:
    ...

def test_slack_chart_report_schedule_with_csv(csv_mock: Any, mock_open: Any, mock_urlopen: Any, slack_client_mock_class: Any, slack_should_use_v2_api_mock: Any, create_report_slack_chart_with_csv: ReportSchedule) -> None:
    ...

def test_slack_chart_report_schedule_with_text(dataframe_mock: Any, slack_client_mock_class: Any, mock_open: Any, mock_urlopen: Any, slack_should_use_v2_api_mock: Any, create_report_slack_chart_with_text: ReportSchedule) -> None:
    ...

def test_report_schedule_not_found(create_report_slack_chart: ReportSchedule) -> None:
    ...

def test_report_schedule_working(create_report_slack_chart_working: ReportSchedule) -> None:
    ...

def test_report_schedule_working_timeout(create_report_slack_chart_working: ReportSchedule) -> None:
    ...

def test_report_schedule_success_grace(create_alert_slack_chart_success: ReportSchedule) -> None:
    ...

def test_report_schedule_success_grace_end(slack_client_mock_class: Any, screenshot_mock: Any, file_upload_mock: Any, create_alert_slack_chart_grace: ReportSchedule) -> None:
    ...

def test_alert_limit_is_applied(screenshot_mock: Any, email_mock: Any, create_alert_email_chart: ReportSchedule) -> None:
    ...

def test_email_dashboard_report_fails(screenshot_mock: Any, email_mock: Any, create_report_email_dashboard: ReportSchedule) -> None:
    ...

def test_email_dashboard_report_fails_uncaught_exception(screenshot_mock: Any, email_mock: Any, create_report_email_dashboard: ReportSchedule) -> None:
    ...

def test_slack_chart_alert(screenshot_mock: Any, email_mock: Any, create_alert_email_chart: ReportSchedule) -> None:
    ...

def test_slack_chart_alert_no_attachment(email_mock: Any, create_alert_email_chart: ReportSchedule) -> None:
    ...

def test_slack_token_callable_chart_report(screenshot_mock: Any, slack_client_mock_class: Any, create_report_slack_chart: ReportSchedule) -> None:
    ...

def test_email_chart_no_alert(create_no_alert_email_chart: ReportSchedule) -> None:
    ...

def test_email_mul_alert(create_mul_alert_email_chart: ReportSchedule) -> None:
    ...

def test_soft_timeout_alert(email_mock: Any, create_alert_email_chart: ReportSchedule) -> None:
    ...

def test_soft_timeout_screenshot(screenshot_mock: Any, email_mock: Any, create_alert_email_chart: ReportSchedule) -> None:
    ...

def test_soft_timeout_csv(csv_mock: Any, email_mock: Any, mock_open: Any, mock_urlopen: Any, create_report_email_chart_with_csv: ReportSchedule) -> None:
    ...

def test_fail_screenshot(screenshot_mock: Any, email_mock: Any, create_report_email_chart: ReportSchedule) -> None:
    ...

def test_fail_csv(csv_mock: Any, mock_open: Any, mock_urlopen: Any, email_mock: Any, create_report_email_chart_with_csv: ReportSchedule) -> None:
    ...

def test_email_disable_screenshot(email_mock: Any, create_alert_email_chart: ReportSchedule) -> None:
    ...

def test_invalid_sql_alert(email_mock: Any, create_invalid_sql_alert_email_chart: ReportSchedule) -> None:
    ...

def test_grace_period_error(email_mock: Any, create_invalid_sql_alert_email_chart: ReportSchedule) -> None:
    ...

def test_grace_period_error_flap(screenshot_mock: Any, email_mock: Any, create_invalid_sql_alert_email_chart: ReportSchedule) -> None:
    ...

def test_prune_log_soft_time_out(bulk_delete_logs: Any, create_report_email_dashboard: ReportSchedule) -> None:
    ...

def test__send_with_client_errors(notification_mock: Any, logger_mock: Any) -> None:
    ...

def test__send_with_multiple_errors(notification_mock: Any, logger_mock: Any) -> None:
    ...

def test__send_with_server_errors(notification_mock: Any, logger_mock: Any) -> None:
    ...
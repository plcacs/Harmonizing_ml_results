from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generator, List, Optional, Union
from unittest.mock import Mock
from uuid import UUID
import pytest
from flask import Flask
from flask.ctx import AppContext
from flask_appbuilder.security.sqla.models import User
from flask_sqlalchemy import BaseQuery
from freezegun import freeze_time
from slack_sdk.errors import (
    BotUserAccessError,
    SlackApiError,
    SlackClientConfigurationError,
    SlackClientError,
    SlackClientNotConnectedError,
    SlackObjectFormationError,
    SlackRequestError,
    SlackTokenRotationError,
)
from sqlalchemy.sql import func
from superset import db
from superset.commands.report.exceptions import (
    AlertQueryError,
    AlertQueryInvalidTypeError,
    AlertQueryMultipleColumnsError,
    AlertQueryMultipleRowsError,
    ReportScheduleClientErrorsException,
    ReportScheduleCsvFailedError,
    ReportScheduleCsvTimeout,
    ReportScheduleNotFoundError,
    ReportSchedulePreviousWorkingError,
    ReportScheduleScreenshotFailedError,
    ReportScheduleScreenshotTimeout,
    ReportScheduleSystemErrorsException,
    ReportScheduleWorkingTimeoutError,
)
from superset.commands.report.execute import AsyncExecuteReportScheduleCommand, BaseReportState
from superset.commands.report.log_prune import AsyncPruneReportScheduleLogCommand
from superset.exceptions import SupersetException
from superset.key_value.models import KeyValueEntry
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
from superset.reports.notifications.exceptions import NotificationError, NotificationParamException
from superset.tasks.types import ExecutorType
from superset.utils import json
from superset.utils.database import get_example_database
from tests.integration_tests.fixtures.birth_names_dashboard import (
    load_birth_names_dashboard_with_slices,
    load_birth_names_data,
)
from tests.integration_tests.fixtures.tabbed_dashboard import tabbed_dashboard
from tests.integration_tests.fixtures.world_bank_dashboard import (
    load_world_bank_dashboard_with_slices_module_scope,
    load_world_bank_data,
)
from tests.integration_tests.reports.utils import (
    cleanup_report_schedule,
    create_report_notification,
    CSV_FILE,
    DEFAULT_OWNER_EMAIL,
    reset_key_values,
    SCREENSHOT_FILE,
    TEST_ID,
)
from tests.integration_tests.test_app import app

pytestmark: Any = ...

def get_target_from_report_schedule(report_schedule: ReportSchedule) -> List[str]: ...
def get_cctarget_from_report_schedule(report_schedule: ReportSchedule) -> List[str]: ...
def get_bcctarget_from_report_schedule(report_schedule: ReportSchedule) -> List[str]: ...
def get_error_logs_query(report_schedule: ReportSchedule) -> BaseQuery: ...
def get_notification_error_sent_count(report_schedule: ReportSchedule) -> int: ...
def assert_log(state: ReportState, error_message: Optional[str] = ...) -> None: ...

@contextmanager
def create_test_table_context(database: Database) -> Generator[Any, None, None]: ...

@pytest.fixture
def create_report_email_chart() -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture
def create_report_email_chart_with_cc_and_bcc() -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture
def create_report_email_chart_alpha_owner(get_user: Any) -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture
def create_report_email_chart_force_screenshot() -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture
def create_report_email_chart_with_csv() -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture
def create_report_email_chart_with_text() -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture
def create_report_email_chart_with_csv_no_query_context() -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture
def create_report_email_dashboard() -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture
def create_report_email_dashboard_force_screenshot() -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture
def create_report_slack_chart() -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture
def create_report_slack_chartv2() -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture
def create_report_slack_chart_with_csv() -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture
def create_report_slack_chart_with_text() -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture
def create_report_slack_chart_working() -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture
def create_alert_slack_chart_success() -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture(params=["alert1"])
def create_alert_slack_chart_grace(request: Any) -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture(params=["alert1", "alert2", "alert3", "alert4", "alert5", "alert6", "alert7", "alert8"])
def create_alert_email_chart(request: Any) -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture(params=["alert1", "alert2", "alert3", "alert4", "alert5", "alert6", "alert7", "alert8", "alert9"])
def create_no_alert_email_chart(request: Any) -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture(params=["alert1", "alert2"])
def create_mul_alert_email_chart(request: Any) -> Generator[ReportSchedule, None, None]: ...
@pytest.fixture(params=["alert1", "alert2"])
def create_invalid_sql_alert_email_chart(request: Any, app_context: Any) -> Generator[ReportSchedule, None, None]: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_email_chart_with_cc_and_bcc")
@patch("superset.reports.notifications.email.send_email_smtp")
@patch("superset.utils.screenshots.ChartScreenshot.get_screenshot")
def test_email_chart_report_schedule_with_cc_bcc(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_chart_with_cc_and_bcc: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_email_chart")
@patch("superset.reports.notifications.email.send_email_smtp")
@patch("superset.utils.screenshots.ChartScreenshot.get_screenshot")
def test_email_chart_report_schedule(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_chart: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_email_chart_alpha_owner")
@patch("superset.reports.notifications.email.send_email_smtp")
@patch("superset.utils.screenshots.ChartScreenshot.get_screenshot")
def test_email_chart_report_schedule_alpha_owner(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_chart_alpha_owner: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_email_chart_force_screenshot")
@patch("superset.reports.notifications.email.send_email_smtp")
@patch("superset.utils.screenshots.ChartScreenshot.get_screenshot")
def test_email_chart_report_schedule_force_screenshot(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_chart_force_screenshot: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_alert_email_chart")
@patch("superset.reports.notifications.email.send_email_smtp")
@patch("superset.utils.screenshots.ChartScreenshot.get_screenshot")
def test_email_chart_alert_schedule(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_alert_email_chart: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_email_chart")
@patch("superset.reports.notifications.email.send_email_smtp")
@patch("superset.utils.screenshots.ChartScreenshot.get_screenshot")
def test_email_chart_report_dry_run(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_chart: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_email_chart_with_csv")
@patch("superset.utils.csv.urllib.request.urlopen")
@patch("superset.utils.csv.urllib.request.OpenerDirector.open")
@patch("superset.reports.notifications.email.send_email_smtp")
@patch("superset.utils.csv.get_chart_csv_data")
def test_email_chart_report_schedule_with_csv(
    csv_mock: Mock,
    email_mock: Mock,
    mock_open: Mock,
    mock_urlopen: Mock,
    create_report_email_chart_with_csv: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_email_chart_with_csv_no_query_context")
@patch("superset.utils.csv.urllib.request.urlopen")
@patch("superset.utils.csv.urllib.request.OpenerDirector.open")
@patch("superset.reports.notifications.email.send_email_smtp")
@patch("superset.utils.csv.get_chart_csv_data")
@patch("superset.utils.screenshots.ChartScreenshot.get_screenshot")
def test_email_chart_report_schedule_with_csv_no_query_context(
    screenshot_mock: Mock,
    csv_mock: Mock,
    email_mock: Mock,
    mock_open: Mock,
    mock_urlopen: Mock,
    create_report_email_chart_with_csv_no_query_context: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_email_chart_with_text")
@patch("superset.utils.csv.urllib.request.urlopen")
@patch("superset.utils.csv.urllib.request.OpenerDirector.open")
@patch("superset.reports.notifications.email.send_email_smtp")
@patch("superset.utils.csv.get_chart_dataframe")
def test_email_chart_report_schedule_with_text(
    dataframe_mock: Mock,
    email_mock: Mock,
    mock_open: Mock,
    mock_urlopen: Mock,
    create_report_email_chart_with_text: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_email_dashboard")
@patch("superset.reports.notifications.email.send_email_smtp")
@patch("superset.utils.screenshots.DashboardScreenshot.get_screenshot")
def test_email_dashboard_report_schedule(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_dashboard: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("tabbed_dashboard")
@patch("superset.utils.screenshots.DashboardScreenshot.get_screenshot")
@patch("superset.reports.notifications.email.send_email_smtp")
@patch.dict("superset.extensions.feature_flag_manager._feature_flags", ALERT_REPORT_TABS=True)
def test_email_dashboard_report_schedule_with_tab_anchor(
    _email_mock: Mock,
    _screenshot_mock: Mock,
) -> None: ...

@pytest.mark.usefixtures("tabbed_dashboard")
@patch("superset.utils.screenshots.DashboardScreenshot.get_screenshot")
@patch("superset.reports.notifications.email.send_email_smtp")
@patch.dict("superset.extensions.feature_flag_manager._feature_flags", ALERT_REPORT_TABS=False)
def test_email_dashboard_report_schedule_disabled_tabs(
    _email_mock: Mock,
    _screenshot_mock: Mock,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_email_dashboard_force_screenshot")
@patch("superset.reports.notifications.email.send_email_smtp")
@patch("superset.utils.screenshots.DashboardScreenshot.get_screenshot")
def test_email_dashboard_report_schedule_force_screenshot(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_report_email_dashboard_force_screenshot: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_slack_chart")
@patch("superset.commands.report.execute.get_channels_with_search")
@patch("superset.reports.notifications.slack.should_use_v2_api", return_value=True)
@patch("superset.reports.notifications.slackv2.get_slack_client")
@patch("superset.utils.screenshots.ChartScreenshot.get_screenshot")
def test_slack_chart_report_schedule_converts_to_v2(
    screenshot_mock: Mock,
    slack_client_mock: Mock,
    slack_should_use_v2_api_mock: Mock,
    get_channels_with_search_mock: Mock,
    create_report_slack_chart: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_slack_chartv2")
@patch("superset.commands.report.execute.get_channels_with_search")
@patch("superset.reports.notifications.slack.should_use_v2_api", return_value=True)
@patch("superset.reports.notifications.slackv2.get_slack_client")
@patch("superset.utils.screenshots.ChartScreenshot.get_screenshot")
def test_slack_chart_report_schedule_v2(
    screenshot_mock: Mock,
    slack_client_mock: Mock,
    slack_should_use_v2_api_mock: Mock,
    get_channels_with_search_mock: Mock,
    create_report_slack_chart: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_slack_chart")
@patch("superset.utils.slack.get_slack_client")
@patch("superset.utils.screenshots.ChartScreenshot.get_screenshot")
def test_slack_chart_report_schedule_with_errors(
    screenshot_mock: Mock,
    web_client_mock: Mock,
    create_report_slack_chart: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_slack_chart_with_csv")
@patch("superset.reports.notifications.slack.should_use_v2_api", return_value=False)
@patch("superset.reports.notifications.slack.get_slack_client")
@patch("superset.utils.csv.urllib.request.urlopen")
@patch("superset.utils.csv.urllib.request.OpenerDirector.open")
@patch("superset.utils.csv.get_chart_csv_data")
def test_slack_chart_report_schedule_with_csv(
    csv_mock: Mock,
    mock_open: Mock,
    mock_urlopen: Mock,
    slack_client_mock_class: Mock,
    slack_should_use_v2_api_mock: Mock,
    create_report_slack_chart_with_csv: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_slack_chart_with_text")
@patch("superset.reports.notifications.slack.should_use_v2_api", return_value=False)
@patch("superset.utils.csv.urllib.request.urlopen")
@patch("superset.utils.csv.urllib.request.OpenerDirector.open")
@patch("superset.reports.notifications.slack.get_slack_client")
@patch("superset.utils.csv.get_chart_dataframe")
def test_slack_chart_report_schedule_with_text(
    dataframe_mock: Mock,
    slack_client_mock_class: Mock,
    mock_open: Mock,
    mock_urlopen: Mock,
    slack_should_use_v2_api_mock: Mock,
    create_report_slack_chart_with_text: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("create_report_slack_chart")
def test_report_schedule_not_found(create_report_slack_chart: ReportSchedule) -> None: ...

@pytest.mark.usefixtures("create_report_slack_chart_working")
def test_report_schedule_working(create_report_slack_chart_working: ReportSchedule) -> None: ...

@pytest.mark.usefixtures("create_report_slack_chart_working")
def test_report_schedule_working_timeout(create_report_slack_chart_working: ReportSchedule) -> None: ...

@pytest.mark.usefixtures("create_alert_slack_chart_success")
def test_report_schedule_success_grace(create_alert_slack_chart_success: ReportSchedule) -> None: ...

@pytest.mark.usefixtures("create_alert_slack_chart_grace")
@patch("superset.utils.slack.WebClient.files_upload")
@patch("superset.utils.screenshots.ChartScreenshot.get_screenshot")
@patch("superset.reports.notifications.slack.get_slack_client")
def test_report_schedule_success_grace_end(
    slack_client_mock_class: Mock,
    screenshot_mock: Mock,
    file_upload_mock: Mock,
    create_alert_slack_chart_grace: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("create_alert_email_chart")
@patch("superset.reports.notifications.email.send_email_smtp")
@patch("superset.utils.screenshots.ChartScreenshot.get_screenshot")
def test_alert_limit_is_applied(
    screenshot_mock: Mock,
    email_mock: Mock,
    create_alert_email_chart: ReportSchedule,
) -> None: ...

@pytest.mark.usefixtures("load_birth_names_dashboard_with_slices", "create_report_email_dashboard")
@patch("
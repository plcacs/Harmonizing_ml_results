from superset.reports.models import ReportRecipientType, ReportSchedule, ReportScheduleType, ReportSourceFormat
from superset.utils.core import HeaderDataType
from superset.utils.screenshots import ChartScreenshot
from superset.commands.report.execute import BaseReportState
from superset.dashboards.permalink.types import DashboardPermalinkState
from tests.integration_tests.conftest import with_feature_flags
from pytest_mock import MockerFixture
from superset.app import SupersetApp
from unittest.mock import patch
from uuid import UUID
import pytest
import json
from datetime import datetime

def test_log_data_with_chart(mocker: MockerFixture) -> None:
def test_log_data_with_dashboard(mocker: MockerFixture) -> None:
def test_log_data_with_email_recipients(mocker: MockerFixture) -> None:
def test_log_data_with_slack_recipients(mocker: MockerFixture) -> None:
def test_log_data_no_owners(mocker: MockerFixture) -> None:
def test_log_data_with_missing_values(mocker: MockerFixture) -> None:
def test_get_dashboard_urls_with_multiple_tabs(mock_run: patch, mocker: MockerFixture) -> None:
def test_get_dashboard_urls_with_exporting_dashboard_only(mock_run: patch, mocker: MockerFixture) -> None:
def test_get_tab_urls(mock_run: patch, mocker: MockerFixture) -> None:
def test_get_tab_url(mock_run: patch, mocker: MockerFixture) -> None:
def create_report_schedule(mocker: MockerFixture, custom_width=None, custom_height=None) -> ReportSchedule:
def test_screenshot_width_calculation(app, mocker: MockerFixture, test_id: str, custom_width: int, max_width: int, window_width: int, expected_width: int) -> None:

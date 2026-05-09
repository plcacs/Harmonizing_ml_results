import json
from datetime import datetime
from unittest.mock import patch
from uuid import UUID
import pytest
from pytest_mock import MockerFixture
from superset.app import SupersetApp
from superset.commands.report.execute import BaseReportState
from superset.dashboards.permalink.types import DashboardPermalinkState
from superset.reports.models import ReportRecipientType, ReportSchedule, ReportScheduleType, ReportSourceFormat
from superset.utils.core import HeaderDataType
from superset.utils.screenshots import ChartScreenshot
from tests.integration_tests.conftest import with_feature_flags

def test_log_data_with_chart(mocker: MockerFixture) -> None:
    # ...

def test_log_data_with_dashboard(mocker: MockerFixture) -> None:
    # ...

def test_log_data_with_email_recipients(mocker: MockerFixture) -> None:
    # ...

def test_log_data_with_slack_recipients(mocker: MockerFixture) -> None:
    # ...

def test_log_data_no_owners(mocker: MockerFixture) -> None:
    # ...

def test_log_data_with_missing_values(mocker: MockerFixture) -> None:
    # ...

@pytest.mark.parametrize('anchors, permalink_side_effect, expected_uris', [(['mock_tab_anchor_1', 'mock_tab_anchor_2'], ['url1', 'url2'], ['http://0.0.0.0:8080/superset/dashboard/p/url1/', 'http://0.0.0.0:8080/superset/dashboard/p/url2/']), ('mock_tab_anchor_1', ['url1'], ['http://0.0.0.0:8080/superset/dashboard/p/url1/'])])
@patch('superset.commands.dashboard.permalink.create.CreateDashboardPermalinkCommand.run')
@with_feature_flags(ALERT_REPORT_TABS=True)
def test_get_dashboard_urls_with_multiple_tabs(mock_run: Any, mocker: MockerFixture, anchors: list, permalink_side_effect: list, expected_uris: list) -> None:
    # ...

@patch('superset.commands.dashboard.permalink.create.CreateDashboardPermalinkCommand.run')
@with_feature_flags(ALERT_REPORT_TABS=True)
def test_get_dashboard_urls_with_exporting_dashboard_only(mock_run: Any, mocker: MockerFixture) -> None:
    # ...

@patch('superset.commands.dashboard.permalink.create.CreateDashboardPermalinkCommand.run')
def test_get_tab_urls(mock_run: Any, mocker: MockerFixture) -> None:
    # ...

@patch('superset.commands.dashboard.permalink.create.CreateDashboardPermalinkCommand.run')
def test_get_tab_url(mock_run: Any, mocker: MockerFixture) -> None:
    # ...

def create_report_schedule(mocker: MockerFixture, custom_width: int = None, custom_height: int = None) -> ReportSchedule:
    # ...

@pytest.mark.parametrize('test_id,custom_width,max_width,window_width,expected_width', [('exceeds_max', 2000, 1600, 800, 1600), ('under_max', 1200, 1600, 800, 1200), ('no_custom', None, 1600, 800, 800), ('equals_max', 1600, 1600, 800, 1600)])
def test_screenshot_width_calculation(app: SupersetApp, mocker: MockerFixture, test_id: str, custom_width: int, max_width: int, window_width: int, expected_width: int) -> None:
    # ...

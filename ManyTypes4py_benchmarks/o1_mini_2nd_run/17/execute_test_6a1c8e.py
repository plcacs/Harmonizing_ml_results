import json
from datetime import datetime
from unittest.mock import patch
from uuid import UUID
from typing import Any, List, Optional, Union
import pytest
from pytest_mock import MockerFixture
from superset.app import SupersetApp
from superset.commands.report.execute import BaseReportState
from superset.dashboards.permalink.types import DashboardPermalinkState
from superset.reports.models import (
    ReportRecipientType,
    ReportSchedule,
    ReportScheduleType,
    ReportSourceFormat,
)
from superset.utils.core import HeaderDataType
from superset.utils.screenshots import ChartScreenshot
from tests.integration_tests.conftest import with_feature_flags


def test_log_data_with_chart(mocker: MockerFixture) -> None:
    mock_report_schedule = mocker.Mock(spec=ReportSchedule)
    mock_report_schedule.chart = True
    mock_report_schedule.chart_id = 123
    mock_report_schedule.dashboard_id = None
    mock_report_schedule.type = 'report_type'
    mock_report_schedule.report_format = 'report_format'
    mock_report_schedule.owners = [1, 2]
    mock_report_schedule.recipients = []
    class_instance = BaseReportState(mock_report_schedule, 'January 1, 2021', 'execution_id_example')
    class_instance._report_schedule = mock_report_schedule
    result = class_instance._get_log_data()
    expected_result = {
        'notification_type': 'report_type',
        'notification_source': ReportSourceFormat.CHART,
        'notification_format': 'report_format',
        'chart_id': 123,
        'dashboard_id': None,
        'owners': [1, 2],
        'slack_channels': None,
    }
    assert result == expected_result


def test_log_data_with_dashboard(mocker: MockerFixture) -> None:
    mock_report_schedule = mocker.Mock(spec=ReportSchedule)
    mock_report_schedule.chart = False
    mock_report_schedule.chart_id = None
    mock_report_schedule.dashboard_id = 123
    mock_report_schedule.type = 'report_type'
    mock_report_schedule.report_format = 'report_format'
    mock_report_schedule.owners = [1, 2]
    mock_report_schedule.recipients = []
    class_instance = BaseReportState(mock_report_schedule, 'January 1, 2021', 'execution_id_example')
    class_instance._report_schedule = mock_report_schedule
    result = class_instance._get_log_data()
    expected_result = {
        'notification_type': 'report_type',
        'notification_source': ReportSourceFormat.DASHBOARD,
        'notification_format': 'report_format',
        'chart_id': None,
        'dashboard_id': 123,
        'owners': [1, 2],
        'slack_channels': None,
    }
    assert result == expected_result


def test_log_data_with_email_recipients(mocker: MockerFixture) -> None:
    mock_report_schedule = mocker.Mock(spec=ReportSchedule)
    mock_report_schedule.chart = False
    mock_report_schedule.chart_id = None
    mock_report_schedule.dashboard_id = 123
    mock_report_schedule.type = 'report_type'
    mock_report_schedule.report_format = 'report_format'
    mock_report_schedule.owners = [1, 2]
    mock_report_schedule.recipients = []
    mock_report_schedule.recipients = [
        mocker.Mock(type=ReportRecipientType.EMAIL, recipient_config_json='email_1'),
        mocker.Mock(type=ReportRecipientType.EMAIL, recipient_config_json='email_2'),
    ]
    class_instance = BaseReportState(mock_report_schedule, 'January 1, 2021', 'execution_id_example')
    class_instance._report_schedule = mock_report_schedule
    result = class_instance._get_log_data()
    expected_result = {
        'notification_type': 'report_type',
        'notification_source': ReportSourceFormat.DASHBOARD,
        'notification_format': 'report_format',
        'chart_id': None,
        'dashboard_id': 123,
        'owners': [1, 2],
        'slack_channels': [],
    }
    assert result == expected_result


def test_log_data_with_slack_recipients(mocker: MockerFixture) -> None:
    mock_report_schedule = mocker.Mock(spec=ReportSchedule)
    mock_report_schedule.chart = False
    mock_report_schedule.chart_id = None
    mock_report_schedule.dashboard_id = 123
    mock_report_schedule.type = 'report_type'
    mock_report_schedule.report_format = 'report_format'
    mock_report_schedule.owners = [1, 2]
    mock_report_schedule.recipients = []
    mock_report_schedule.recipients = [
        mocker.Mock(type=ReportRecipientType.SLACK, recipient_config_json='channel_1'),
        mocker.Mock(type=ReportRecipientType.SLACK, recipient_config_json='channel_2'),
    ]
    class_instance = BaseReportState(mock_report_schedule, 'January 1, 2021', 'execution_id_example')
    class_instance._report_schedule = mock_report_schedule
    result = class_instance._get_log_data()
    expected_result = {
        'notification_type': 'report_type',
        'notification_source': ReportSourceFormat.DASHBOARD,
        'notification_format': 'report_format',
        'chart_id': None,
        'dashboard_id': 123,
        'owners': [1, 2],
        'slack_channels': ['channel_1', 'channel_2'],
    }
    assert result == expected_result


def test_log_data_no_owners(mocker: MockerFixture) -> None:
    mock_report_schedule = mocker.Mock(spec=ReportSchedule)
    mock_report_schedule.chart = False
    mock_report_schedule.chart_id = None
    mock_report_schedule.dashboard_id = 123
    mock_report_schedule.type = 'report_type'
    mock_report_schedule.report_format = 'report_format'
    mock_report_schedule.owners = []
    mock_report_schedule.recipients = [
        mocker.Mock(type=ReportRecipientType.SLACK, recipient_config_json='channel_1'),
        mocker.Mock(type=ReportRecipientType.SLACK, recipient_config_json='channel_2'),
    ]
    class_instance = BaseReportState(mock_report_schedule, 'January 1, 2021', 'execution_id_example')
    class_instance._report_schedule = mock_report_schedule
    result = class_instance._get_log_data()
    expected_result = {
        'notification_type': 'report_type',
        'notification_source': ReportSourceFormat.DASHBOARD,
        'notification_format': 'report_format',
        'chart_id': None,
        'dashboard_id': 123,
        'owners': [],
        'slack_channels': ['channel_1', 'channel_2'],
    }
    assert result == expected_result


def test_log_data_with_missing_values(mocker: MockerFixture) -> None:
    mock_report_schedule = mocker.Mock(spec=ReportSchedule)
    mock_report_schedule.chart = None
    mock_report_schedule.chart_id = None
    mock_report_schedule.dashboard_id = None
    mock_report_schedule.type = 'report_type'
    mock_report_schedule.report_format = 'report_format'
    mock_report_schedule.owners = [1, 2]
    mock_report_schedule.recipients = [
        mocker.Mock(type=ReportRecipientType.SLACK, recipient_config_json='channel_1'),
        mocker.Mock(type=ReportRecipientType.SLACKV2, recipient_config_json='channel_2'),
    ]
    class_instance = BaseReportState(mock_report_schedule, 'January 1, 2021', 'execution_id_example')
    class_instance._report_schedule = mock_report_schedule
    result = class_instance._get_log_data()
    expected_result = {
        'notification_type': 'report_type',
        'notification_source': ReportSourceFormat.DASHBOARD,
        'notification_format': 'report_format',
        'chart_id': None,
        'dashboard_id': None,
        'owners': [1, 2],
        'slack_channels': ['channel_1', 'channel_2'],
    }
    assert result == expected_result


@pytest.mark.parametrize(
    'anchors, permalink_side_effect, expected_uris',
    [
        (
            ['mock_tab_anchor_1', 'mock_tab_anchor_2'],
            ['url1', 'url2'],
            [
                'http://0.0.0.0:8080/superset/dashboard/p/url1/',
                'http://0.0.0.0:8080/superset/dashboard/p/url2/',
            ],
        ),
        (
            'mock_tab_anchor_1',
            ['url1'],
            ['http://0.0.0.0:8080/superset/dashboard/p/url1/'],
        ),
    ],
)
@patch('superset.commands.dashboard.permalink.create.CreateDashboardPermalinkCommand.run')
@with_feature_flags(ALERT_REPORT_TABS=True)
def test_get_dashboard_urls_with_multiple_tabs(
    mock_run: Any,
    mocker: MockerFixture,
    anchors: Union[List[str], str],
    permalink_side_effect: List[str],
    expected_uris: List[str],
) -> None:
    mock_report_schedule = mocker.Mock(spec=ReportSchedule)
    mock_report_schedule.chart = False
    mock_report_schedule.chart_id = None
    mock_report_schedule.dashboard_id = 123
    mock_report_schedule.type = 'report_type'
    mock_report_schedule.report_format = 'report_format'
    mock_report_schedule.owners = [1, 2]
    mock_report_schedule.recipients = []
    mock_report_schedule.extra = {
        'dashboard': {
            'anchor': json.dumps(anchors) if isinstance(anchors, list) else anchors,
            'dataMask': None,
            'activeTabs': None,
            'urlParams': None,
        }
    }
    class_instance = BaseReportState(
        mock_report_schedule, 'January 1, 2021', 'execution_id_example'
    )
    class_instance._report_schedule = mock_report_schedule
    mock_run.side_effect = permalink_side_effect
    result = class_instance.get_dashboard_urls()
    assert result == expected_uris


@patch('superset.commands.dashboard.permalink.create.CreateDashboardPermalinkCommand.run')
@with_feature_flags(ALERT_REPORT_TABS=True)
def test_get_dashboard_urls_with_exporting_dashboard_only(
    mock_run: Any, mocker: MockerFixture
) -> None:
    mock_report_schedule = mocker.Mock(spec=ReportSchedule)
    mock_report_schedule.chart = False
    mock_report_schedule.chart_id = None
    mock_report_schedule.dashboard_id = 123
    mock_report_schedule.type = 'report_type'
    mock_report_schedule.report_format = 'report_format'
    mock_report_schedule.owners = [1, 2]
    mock_report_schedule.recipients = []
    mock_report_schedule.extra = {
        'dashboard': {
            'anchor': '',
            'dataMask': None,
            'activeTabs': None,
            'urlParams': None,
        }
    }
    mock_run.return_value = 'url1'
    class_instance = BaseReportState(
        mock_report_schedule, 'January 1, 2021', 'execution_id_example'
    )
    class_instance._report_schedule = mock_report_schedule
    result = class_instance.get_dashboard_urls()
    assert 'http://0.0.0.0:8080/superset/dashboard/p/url1/' == result[0]


@patch('superset.commands.dashboard.permalink.create.CreateDashboardPermalinkCommand.run')
def test_get_tab_urls(
    mock_run: Any, mocker: MockerFixture
) -> None:
    mock_report_schedule = mocker.Mock(spec=ReportSchedule)
    mock_report_schedule.dashboard_id = 123
    class_instance = BaseReportState(
        mock_report_schedule, 'January 1, 2021', 'execution_id_example'
    )
    class_instance._report_schedule = mock_report_schedule
    mock_run.side_effect = ['uri1', 'uri2']
    tab_anchors = ['1', '2']
    result = class_instance._get_tabs_urls(tab_anchors)
    assert result == [
        'http://0.0.0.0:8080/superset/dashboard/p/uri1/',
        'http://0.0.0.0:8080/superset/dashboard/p/uri2/',
    ]


@patch('superset.commands.dashboard.permalink.create.CreateDashboardPermalinkCommand.run')
def test_get_tab_url(
    mock_run: Any, mocker: MockerFixture
) -> None:
    mock_report_schedule = mocker.Mock(spec=ReportSchedule)
    mock_report_schedule.dashboard_id = 123
    class_instance = BaseReportState(
        mock_report_schedule, 'January 1, 2021', 'execution_id_example'
    )
    class_instance._report_schedule = mock_report_schedule
    mock_run.return_value = 'uri'
    dashboard_state = DashboardPermalinkState(
        anchor='1', dataMask=None, activeTabs=None, urlParams=None
    )
    result = class_instance._get_tab_url(dashboard_state)
    assert result == 'http://0.0.0.0:8080/superset/dashboard/p/uri/'


def create_report_schedule(
    mocker: MockerFixture,
    custom_width: Optional[int] = None,
    custom_height: Optional[int] = None,
) -> ReportSchedule:
    """Helper function to create a ReportSchedule instance with specified dimensions."""
    schedule = ReportSchedule()
    schedule.type = ReportScheduleType.REPORT
    schedule.name = 'Test Report'
    schedule.description = 'Test Description'
    schedule.chart = mocker.MagicMock()
    schedule.chart.id = 1
    schedule.dashboard = None
    schedule.database = None
    schedule.custom_width = custom_width
    schedule.custom_height = custom_height
    return schedule


@pytest.mark.parametrize(
    'test_id,custom_width,max_width,window_width,expected_width',
    [
        ('exceeds_max', 2000, 1600, 800, 1600),
        ('under_max', 1200, 1600, 800, 1200),
        ('no_custom', None, 1600, 800, 800),
        ('equals_max', 1600, 1600, 800, 1600),
    ],
)
def test_screenshot_width_calculation(
    app: SupersetApp,
    mocker: MockerFixture,
    test_id: str,
    custom_width: Optional[int],
    max_width: int,
    window_width: int,
    expected_width: int,
) -> None:
    """
    Test that screenshot width is correctly calculated.

    The width should be:
    - Limited by max_width when custom_width exceeds it
    - Equal to custom_width when it's less than max_width
    - Equal to window_width when custom_width is None
    """
    from superset.commands.report.execute import BaseReportState

    app.config.update({
        'ALERT_REPORTS_MAX_CUSTOM_SCREENSHOT_WIDTH': max_width,
        'WEBDRIVER_WINDOW': {
            'slice': (window_width, 600),
            'dashboard': (window_width, 600),
        },
        'ALERT_REPORTS_EXECUTORS': {},
    })
    report_schedule = create_report_schedule(mocker, custom_width=custom_width)
    report_state = BaseReportState(
        report_schedule=report_schedule,
        scheduled_dttm=datetime.now(),
        execution_id=UUID('084e7ee6-5557-4ecd-9632-b7f39c9ec524'),
    )
    with patch('superset.commands.report.execute.security_manager') as mock_security_manager, \
         patch('superset.utils.screenshots.ChartScreenshot.get_screenshot') as mock_get_screenshot:
        mock_user = mocker.MagicMock()
        mock_security_manager.find_user.return_value = mock_user
        mock_get_screenshot.return_value = b'screenshot bytes'
        with patch('superset.commands.report.execute.get_executor') as mock_get_executor:
            mock_get_executor.return_value = ('executor', 'username')
            with patch('superset.commands.report.execute.ChartScreenshot', wraps=ChartScreenshot) as mock_chart_screenshot:
                report_state._get_screenshots()
                mock_chart_screenshot.assert_called_once()
                _, kwargs = mock_chart_screenshot.call_args
                assert (
                    kwargs['window_size'][0] == expected_width
                ), f"Test {test_id}: Expected width {expected_width}, but got {kwargs['window_size'][0]}"

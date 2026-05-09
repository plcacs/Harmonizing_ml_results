import uuid
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from slack_sdk.errors import SlackApiError
from superset.reports.notifications.slackv2 import SlackV2Notification
from superset.utils.core import HeaderDataType

@pytest.fixture
def mock_header_data() -> dict:
    return {'notification_format': 'PNG', 'notification_type': 'Alert', 'owners': [1], 'notification_source': None, 'chart_id': None, 'dashboard_id': None, 'slack_channels': ['some_channel']}

def test_get_channel_with_multi_recipients(mock_header_data: dict) -> None:
    # ... (rest of the function remains the same)

def test_valid_recipient_config_json_slackv2(mock_header_data: dict) -> None:
    # ... (rest of the function remains the same)

def test_get_inline_files_with_screenshots(mock_header_data: dict) -> tuple:
    # ... (rest of the function remains the same)

def test_get_inline_files_with_no_screenshots_or_csv(mock_header_data: dict) -> tuple:
    # ... (rest of the function remains the same)

@patch('superset.reports.notifications.slackv2.g')
@patch('superset.reports.notifications.slackv2.logger')
@patch('superset.reports.notifications.slackv2.get_slack_client')
def test_send_slackv2(slack_client_mock: MagicMock, logger_mock: MagicMock, flask_global_mock: MagicMock, mock_header_data: dict) -> None:
    # ... (rest of the function remains the same)

@patch('superset.reports.notifications.slack.g')
@patch('superset.reports.notifications.slack.logger')
@patch('superset.utils.slack.get_slack_client')
@patch('superset.reports.notifications.slack.get_slack_client')
def test_send_slack(slack_client_mock: MagicMock, slack_client_mock_util: MagicMock, logger_mock: MagicMock, flask_global_mock: MagicMock, mock_header_data: dict) -> None:
    # ... (rest of the function remains the same)

@patch('superset.reports.notifications.slack.g')
@patch('superset.reports.notifications.slack.logger')
@patch('superset.utils.slack.get_slack_client')
@patch('superset.reports.notifications.slack.get_slack_client')
def test_send_slack_no_feature_flag(slack_client_mock: MagicMock, slack_client_mock_util: MagicMock, logger_mock: MagicMock, flask_global_mock: MagicMock, mock_header_data: dict) -> None:
    # ... (rest of the function remains the same)

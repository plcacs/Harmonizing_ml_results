from typing import Any, Dict, List, Optional, Tuple
import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from slack_sdk.errors import SlackApiError
from superset.reports.notifications.slackv2 import SlackV2Notification
from superset.utils.core import HeaderDataType


@pytest.fixture
def mock_header_data() -> Dict[str, Any]:
    return {
        'notification_format': 'PNG',
        'notification_type': 'Alert',
        'owners': [1],
        'notification_source': None,
        'chart_id': None,
        'dashboard_id': None,
        'slack_channels': ['some_channel'],
    }


def test_get_channel_with_multi_recipients(mock_header_data: Dict[str, Any]) -> None:
    """
    Test the _get_channel function to ensure it will return a string
    with recipients separated by commas without interstitial spacing
    """
    from superset.reports.models import ReportRecipients, ReportRecipientType
    from superset.reports.notifications.base import NotificationContent
    from superset.reports.notifications.slack import SlackNotification

    content = NotificationContent(
        name='test alert',
        header_data=mock_header_data,
        embedded_data=pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['111', '222', '<a href="http://www.example.com">333</a>']
        }),
        description='<p>This is <a href="#">a test</a> alert</p><br />'
    )
    slack_notification = SlackNotification(
        recipient=ReportRecipients(
            type=ReportRecipientType.SLACK,
            recipient_config_json='{"target": "some_channel; second_channel, third_channel"}'
        ),
        content=content
    )
    result: str = slack_notification._get_channel()  # type: ignore
    assert result == 'some_channel,second_channel,third_channel'


def test_valid_recipient_config_json_slackv2(mock_header_data: Dict[str, Any]) -> None:
    """
    Test if the recipient configuration JSON is valid when using a SlackV2 recipient type
    """
    from superset.reports.models import ReportRecipients, ReportRecipientType
    from superset.reports.notifications.base import NotificationContent
    from superset.reports.notifications.slack import SlackNotification

    content = NotificationContent(
        name='test alert',
        header_data=mock_header_data,
        embedded_data=pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['111', '222', '<a href="http://www.example.com">333</a>']
        }),
        description='<p>This is <a href="#">a test</a> alert</p><br />'
    )
    slack_notification = SlackNotification(
        recipient=ReportRecipients(
            type=ReportRecipientType.SLACKV2,
            recipient_config_json='{"target": "some_channel"}'
        ),
        content=content
    )
    result: str = slack_notification._recipient.recipient_config_json  # type: ignore
    assert result == '{"target": "some_channel"}'


def test_get_inline_files_with_screenshots(mock_header_data: Dict[str, Any]) -> None:
    """
    Test the _get_inline_files function to ensure it will return the correct tuple
    when content has screenshots
    """
    from superset.reports.models import ReportRecipients, ReportRecipientType
    from superset.reports.notifications.base import NotificationContent
    from superset.reports.notifications.slack import SlackNotification

    content = NotificationContent(
        name='test alert',
        header_data=mock_header_data,
        embedded_data=pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['111', '222', '<a href="http://www.example.com">333</a>']
        }),
        description='<p>This is <a href="#">a test</a> alert</p><br />',
        screenshots=[b'screenshot1', b'screenshot2']
    )
    slack_notification = SlackNotification(
        recipient=ReportRecipients(
            type=ReportRecipientType.SLACK,
            recipient_config_json='{"target": "some_channel"}'
        ),
        content=content
    )
    result: Tuple[Optional[str], List[bytes]] = slack_notification._get_inline_files()  # type: ignore
    assert result == ('png', [b'screenshot1', b'screenshot2'])


def test_get_inline_files_with_no_screenshots_or_csv(mock_header_data: Dict[str, Any]) -> None:
    """
    Test the _get_inline_files function to ensure it will return None
    when content has no screenshots or csv
    """
    from superset.reports.models import ReportRecipients, ReportRecipientType
    from superset.reports.notifications.base import NotificationContent
    from superset.reports.notifications.slack import SlackNotification

    content = NotificationContent(
        name='test alert',
        header_data=mock_header_data,
        embedded_data=pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['111', '222', '<a href="http://www.example.com">333</a>']
        }),
        description='<p>This is <a href="#">a test</a> alert</p><br />'
    )
    slack_notification = SlackNotification(
        recipient=ReportRecipients(
            type=ReportRecipientType.SLACK,
            recipient_config_json='{"target": "some_channel"}'
        ),
        content=content
    )
    result: Tuple[Optional[str], List[bytes]] = slack_notification._get_inline_files()  # type: ignore
    assert result == (None, [])


@patch('superset.reports.notifications.slackv2.g')
@patch('superset.reports.notifications.slackv2.logger')
@patch('superset.reports.notifications.slackv2.get_slack_client')
def test_send_slackv2(
    slack_client_mock: MagicMock,
    logger_mock: MagicMock,
    flask_global_mock: MagicMock,
    mock_header_data: Dict[str, Any],
) -> None:
    from superset.reports.models import ReportRecipients, ReportRecipientType
    from superset.reports.notifications.base import NotificationContent

    execution_id: uuid.UUID = uuid.uuid4()
    flask_global_mock.logs_context = {'execution_id': execution_id}
    slack_client_mock.return_value.chat_postMessage.return_value = {'ok': True}

    content = NotificationContent(
        name='test alert',
        header_data=mock_header_data,
        embedded_data=pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['111', '222', '<a href="http://www.example.com">333</a>']
        }),
        description='<p>This is <a href="#">a test</a> alert</p><br />'
    )
    from superset.reports.models import ReportRecipients, ReportRecipientType  # re-import if needed
    notification = SlackV2Notification(
        recipient=ReportRecipients(
            type=ReportRecipientType.SLACKV2,
            recipient_config_json='{"target": "some_channel"}'
        ),
        content=content
    )
    notification.send()
    logger_mock.info.assert_called_with('Report sent to slack', extra={'execution_id': execution_id})
    slack_client_mock.return_value.chat_postMessage.assert_called_with(
        channel='some_channel',
        text='*test alert*\n\n<p>This is <a href="#">a test</a> alert</p><br />\n\n<None|Explore in Superset>\n\n```\n|    |   A |   B | C                                        |\n|---:|----:|----:|:-----------------------------------------|\n|  0 |   1 |   4 | 111                                      |\n|  1 |   2 |   5 | 222                                      |\n|  2 |   3 |   6 | <a href="http://www.example.com">333</a> |\n```\n'
    )


@patch('superset.reports.notifications.slack.g')
@patch('superset.reports.notifications.slack.logger')
@patch('superset.utils.slack.get_slack_client')
@patch('superset.reports.notifications.slack.get_slack_client')
def test_send_slack(
    slack_client_mock: MagicMock,
    slack_client_mock_util: MagicMock,
    logger_mock: MagicMock,
    flask_global_mock: MagicMock,
    mock_header_data: Dict[str, Any],
) -> None:
    from superset.reports.models import ReportRecipients, ReportRecipientType
    from superset.reports.notifications.base import NotificationContent
    from superset.reports.notifications.slack import SlackNotification

    execution_id: uuid.UUID = uuid.uuid4()
    flask_global_mock.logs_context = {'execution_id': execution_id}
    slack_client_mock.return_value.chat_postMessage.return_value = {'ok': True}
    slack_client_mock_util.return_value.conversations_list.side_effect = SlackApiError('scope not found', 'error')

    content = NotificationContent(
        name='test alert',
        header_data=mock_header_data,
        embedded_data=pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['111', '222', '<a href="http://www.example.com">333</a>']
        }),
        description='<p>This is <a href="#">a test</a> alert</p><br />'
    )
    notification = SlackNotification(
        recipient=ReportRecipients(
            type=ReportRecipientType.SLACKV2,
            recipient_config_json='{"target": "some_channel"}'
        ),
        content=content
    )
    notification.send()
    logger_mock.info.assert_called_with('Report sent to slack', extra={'execution_id': execution_id})
    slack_client_mock.return_value.chat_postMessage.assert_called_with(
        channel='some_channel',
        text='*test alert*\n\n<p>This is <a href="#">a test</a> alert</p><br />\n\n<None|Explore in Superset>\n\n```\n|    |   A |   B | C                                        |\n|---:|----:|----:|:-----------------------------------------|\n|  0 |   1 |   4 | 111                                      |\n|  1 |   2 |   5 | 222                                      |\n|  2 |   3 |   6 | <a href="http://www.example.com">333</a> |\n```\n'
    )


@patch('superset.reports.notifications.slack.g')
@patch('superset.reports.notifications.slack.logger')
@patch('superset.utils.slack.get_slack_client')
@patch('superset.reports.notifications.slack.get_slack_client')
def test_send_slack_no_feature_flag(
    slack_client_mock: MagicMock,
    slack_client_mock_util: MagicMock,
    logger_mock: MagicMock,
    flask_global_mock: MagicMock,
    mock_header_data: Dict[str, Any],
) -> None:
    from superset.reports.models import ReportRecipients, ReportRecipientType
    from superset.reports.notifications.base import NotificationContent
    from superset.reports.notifications.slack import SlackNotification

    execution_id: uuid.UUID = uuid.uuid4()
    flask_global_mock.logs_context = {'execution_id': execution_id}
    slack_client_mock.return_value.chat_postMessage.return_value = {'ok': True}
    slack_client_mock_util.return_value.conversations_list.return_value = {
        'channels': [{'id': 'foo', 'name': 'bar'}]
    }

    content = NotificationContent(
        name='test alert',
        header_data=mock_header_data,
        embedded_data=pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['111', '222', '<a href="http://www.example.com">333</a>']
        }),
        description='<p>This is <a href="#">a test</a> alert</p><br />'
    )
    notification = SlackNotification(
        recipient=ReportRecipients(
            type=ReportRecipientType.SLACKV2,
            recipient_config_json='{"target": "some_channel"}'
        ),
        content=content
    )
    notification.send()
    logger_mock.info.assert_called_with('Report sent to slack', extra={'execution_id': execution_id})
    slack_client_mock.return_value.chat_postMessage.assert_called_with(
        channel='some_channel',
        text='*test alert*\n\n<p>This is <a href="#">a test</a> alert</p><br />\n\n<None|Explore in Superset>\n\n```\n|    |   A |   B | C                                        |\n|---:|----:|----:|:-----------------------------------------|\n|  0 |   1 |   4 | 111                                      |\n|  1 |   2 |   5 | 222                                      |\n|  2 |   3 |   6 | <a href="http://www.example.com">333</a> |\n```\n'
    )
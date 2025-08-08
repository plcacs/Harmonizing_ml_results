from typing import Dict, Any, Tuple, Optional

def test_get_channel_with_multi_recipients(mock_header_data: Dict[str, Any]) -> None:
def test_valid_recipient_config_json_slackv2(mock_header_data: Dict[str, Any]) -> None:
def test_get_inline_files_with_screenshots(mock_header_data: Dict[str, Any]) -> None:
def test_get_inline_files_with_no_screenshots_or_csv(mock_header_data: Dict[str, Any]) -> None:
def test_send_slackv2(slack_client_mock: MagicMock, logger_mock: MagicMock, flask_global_mock: MagicMock, mock_header_data: Dict[str, Any]) -> None:
def test_send_slack(slack_client_mock: MagicMock, slack_client_mock_util: MagicMock, logger_mock: MagicMock, flask_global_mock: MagicMock, mock_header_data: Dict[str, Any]) -> None:
def test_send_slack_no_feature_flag(slack_client_mock: MagicMock, slack_client_mock_util: MagicMock, logger_mock: MagicMock, flask_global_mock: MagicMock, mock_header_data: Dict[str, Any]) -> None:

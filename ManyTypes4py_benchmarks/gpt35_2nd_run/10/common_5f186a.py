from typing import Annotated, Any

MISSING_EVENT_HEADER_MESSAGE: Annotated[str, Any] = "Hi there!  Your bot {bot_name} just sent an HTTP request to {request_path} that\nis missing the HTTP {header_name} header.  Because this header is how\n{integration_name} indicates the event type, this usually indicates a configuration\nissue, where you either entered the URL for a different integration, or are running\nan older version of the third-party service that doesn't provide that header.\nContact {support_email} if you need help debugging!\n"
INVALID_JSON_MESSAGE: Annotated[str, Any] = "\nHi there! It looks like you tried to set up the Zulip {webhook_name} integration,\nbut didn't correctly configure the webhook to send data in the JSON format\nthat this integration expects!\n"
SETUP_MESSAGE_TEMPLATE: Annotated[str, Any] = '{integration} webhook has been successfully configured'
SETUP_MESSAGE_USER_PART: Annotated[str, Any] = ' by {user_name}'
OptionalUserSpecifiedTopicStr: Annotated[str | None, Any] = Annotated[str | None, ApiParamConfig('topic')]

def get_setup_webhook_message(integration: str, user_name: str = None) -> str:
    content: str = SETUP_MESSAGE_TEMPLATE.format(integration=integration)
    if user_name:
        content += SETUP_MESSAGE_USER_PART.format(user_name=user_name)
    content = f'{content}.'
    return content

def notify_bot_owner_about_invalid_json(user_profile: UserProfile, webhook_client_name: str) -> None:
    send_rate_limited_pm_notification_to_bot_owner(user_profile, user_profile.realm, INVALID_JSON_MESSAGE.format(webhook_name=webhook_client_name).strip()

class MissingHTTPEventHeaderError(AnomalousWebhookPayloadError):
    code: ErrorCode = ErrorCode.MISSING_HTTP_EVENT_HEADER
    data_fields: Annotated[list[str], Any] = ['header']

    def __init__(self, header: str) -> None:
        self.header = header

    @staticmethod
    @override
    def msg_format() -> str:
        return _("Missing the HTTP event header '{header}'")

def check_send_webhook_message(request: HttpRequest, user_profile: UserProfile, topic: str, body: str, complete_event_type: str = None, *, stream: str = None, user_specified_topic: OptionalUserSpecifiedTopicStr = None, only_events: list[str] = None, exclude_events: list[str] = None, unquote_url_parameters: bool = False) -> None:
    pass

def standardize_headers(input_headers: dict[str, str]) -> dict[str, str]:
    pass

def validate_extract_webhook_http_header(request: HttpRequest, header: str, integration_name: str) -> str:
    pass

def get_fixture_http_headers(integration_name: str, fixture_name: str) -> dict[str, str]:
    pass

def get_http_headers_from_filename(http_header_key: str) -> callable:
    pass

def unix_milliseconds_to_timestamp(milliseconds: int, webhook: str) -> datetime:
    pass

def parse_multipart_string(body: str) -> dict[str, str]:
    pass

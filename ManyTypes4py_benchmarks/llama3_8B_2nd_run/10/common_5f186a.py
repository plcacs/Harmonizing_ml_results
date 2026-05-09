from typing import Annotated, Any, TypeAlias
from zerver.actions.message_send import check_send_private_message, check_send_stream_message, check_send_stream_message_by_id, send_rate_limited_pm_notification_to_bot_owner
from zerver.lib.exceptions import AnomalousWebhookPayloadError, ErrorCode, JsonableError, StreamDoesNotExistError
from zerver.lib.request import RequestNotes
from zerver.lib.send_email import FromAddress
from zerver.lib.timestamp import timestamp_to_datetime
from zerver.lib.typed_endpoint import ApiParamConfig, typed_endpoint
from zerver.models import UserProfile
from zerver.webhooks import MissingHTTPEventHeaderError

OptionalUserSpecifiedTopicStr: TypeAlias = Annotated[str | None, ApiParamConfig('topic')]

@dataclass
class WebhookConfigOption:
    pass

def get_setup_webhook_message(integration: str, user_name: str | None = None) -> str:
    content = SETUP_MESSAGE_TEMPLATE.format(integration=integration)
    if user_name:
        content += SETUP_MESSAGE_USER_PART.format(user_name=user_name)
    content = f'{content}.'
    return content

def notify_bot_owner_about_invalid_json(user_profile: UserProfile, webhook_client_name: str) -> None:
    send_rate_limited_pm_notification_to_bot_owner(user_profile, user_profile.realm, INVALID_JSON_MESSAGE.format(webhook_name=webhook_client_name).strip())

class MissingHTTPEventHeaderError(AnomalousWebhookPayloadError):
    code: ErrorCode = ErrorCode.MISSING_HTTP_EVENT_HEADER
    data_fields: list[str] = ['header']

    def __init__(self, header: str):
        self.header = header

    @staticmethod
    @override
    def msg_format() -> str:
        return _("Missing the HTTP event header '{header}'")

@typed_endpoint
def check_send_webhook_message(request: HttpRequest, user_profile: UserProfile, topic: OptionalUserSpecifiedTopicStr, body: str, complete_event_type: str | None = None, *, stream: str | None, user_specified_topic: str | None = None, only_events: list[str] | None = None, exclude_events: list[str] | None = None, unquote_url_parameters: bool = False) -> None:
    ...

def standardize_headers(input_headers: dict[str, str]) -> dict[str, str]:
    ...

def validate_extract_webhook_http_header(request: HttpRequest, header: str, integration_name: str) -> str:
    ...

def get_fixture_http_headers(integration_name: str, fixture_name: str) -> dict[str, str]:
    ...

def get_http_headers_from_filename(http_header_key: str) -> dict[str, str]:
    ...

def unix_milliseconds_to_timestamp(milliseconds: int, webhook: str) -> datetime:
    ...

def parse_multipart_string(body: str) -> dict[str, str]:
    ...

from typing import Annotated, Any

MISSING_EVENT_HEADER_MESSAGE: Annotated[str, Any] = """Hi there!  Your bot {bot_name} just sent an HTTP request to {request_path} that
is missing the HTTP {header_name} header.  Because this header is how
{integration_name} indicates the event type, this usually indicates a configuration
issue, where you either entered the URL for a different integration, or are running
an older version of the third-party service that doesn't provide that header.
Contact {support_email} if you need help debugging!
"""
INVALID_JSON_MESSAGE: Annotated[str, Any] = """
Hi there! It looks like you tried to set up the Zulip {webhook_name} integration,
but didn't correctly configure the webhook to send data in the JSON format
that this integration expects!
"""
SETUP_MESSAGE_TEMPLATE: Annotated[str, Any] = (
    '{integration} webhook has been successfully configured')
SETUP_MESSAGE_USER_PART: Annotated[str, Any] = ' by {user_name}'
OptionalUserSpecifiedTopicStr: Annotated[str | None, Any] = Annotated[str | None, ApiParamConfig('topic')]

def func_hnfe1pzi(integration: str, user_name: str = None) -> str:
    ...

def func_uc8cenjy(user_profile: UserProfile, webhook_client_name: str) -> None:
    ...

class MissingHTTPEventHeaderError(AnomalousWebhookPayloadError):
    ...

def func_ni3jazws(request: HttpRequest, user_profile: UserProfile, topic: str, body: str, complete_event_type: str = None, *,
    stream: str = None, user_specified_topic: OptionalUserSpecifiedTopicStr = None, only_events: list[str] = None,
    exclude_events: list[str] = None, unquote_url_parameters: bool = False) -> None:
    ...

def func_zv0lk43y(input_headers: dict[str, str]) -> dict[str, str]:
    ...

def func_w1ljyct3(request: HttpRequest, header: str, integration_name: str) -> str:
    ...

def func_u5tm5qif(integration_name: str, fixture_name: str) -> dict[str, str]:
    ...

def func_ac3ms6d6(http_header_key: str) -> callable:
    ...

def func_ncavq3fi(milliseconds: int, webhook: str) -> datetime:
    ...

def func_1wi4xmff(body: str) -> dict[str, str]:
    ...

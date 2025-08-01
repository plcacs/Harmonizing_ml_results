import fnmatch
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Optional, List, Dict
from urllib.parse import unquote

from django.http import HttpRequest
from django.utils.translation import gettext as _
from pydantic import Json
from typing_extensions import override

from zerver.actions.message_send import (
    check_send_private_message,
    check_send_stream_message,
    check_send_stream_message_by_id,
    send_rate_limited_pm_notification_to_bot_owner,
)
from zerver.lib.exceptions import (
    AnomalousWebhookPayloadError,
    ErrorCode,
    JsonableError,
    StreamDoesNotExistError,
)
from zerver.lib.request import RequestNotes
from zerver.lib.send_email import FromAddress
from zerver.lib.timestamp import timestamp_to_datetime
from zerver.lib.typed_endpoint import ApiParamConfig, typed_endpoint
from zerver.models import UserProfile

MISSING_EVENT_HEADER_MESSAGE: str = (
    "Hi there!  Your bot {bot_name} just sent an HTTP request to {request_path} that\n"
    "is missing the HTTP {header_name} header.  Because this header is how\n"
    "{integration_name} indicates the event type, this usually indicates a configuration\n"
    "issue, where you either entered the URL for a different integration, or are running\n"
    "an older version of the third-party service that doesn't provide that header.\n"
    "Contact {support_email} if you need help debugging!\n"
)
INVALID_JSON_MESSAGE: str = (
    "\nHi there! It looks like you tried to set up the Zulip {webhook_name} integration,\n"
    "but didn't correctly configure the webhook to send data in the JSON format\n"
    "that this integration expects!\n"
)
SETUP_MESSAGE_TEMPLATE: str = '{integration} webhook has been successfully configured'
SETUP_MESSAGE_USER_PART: str = ' by {user_name}'
OptionalUserSpecifiedTopicStr = Annotated[str | None, ApiParamConfig('topic')]


@dataclass
class WebhookConfigOption:
    pass


def get_setup_webhook_message(integration: str, user_name: Optional[str] = None) -> str:
    content: str = SETUP_MESSAGE_TEMPLATE.format(integration=integration)
    if user_name:
        content += SETUP_MESSAGE_USER_PART.format(user_name=user_name)
    content = f'{content}.'
    return content


def notify_bot_owner_about_invalid_json(user_profile: UserProfile, webhook_client_name: str) -> None:
    send_rate_limited_pm_notification_to_bot_owner(
        user_profile,
        user_profile.realm,
        INVALID_JSON_MESSAGE.format(webhook_name=webhook_client_name).strip(),
    )


class MissingHTTPEventHeaderError(AnomalousWebhookPayloadError):
    code = ErrorCode.MISSING_HTTP_EVENT_HEADER
    data_fields = ['header']

    def __init__(self, header: str) -> None:
        self.header = header

    @staticmethod
    @override
    def msg_format() -> str:
        return _("Missing the HTTP event header '{header}'")


@typed_endpoint
def check_send_webhook_message(
    request: HttpRequest,
    user_profile: UserProfile,
    topic: str,
    body: str,
    complete_event_type: Optional[str] = None,
    *,
    stream: Optional[str] = None,
    user_specified_topic: Optional[str] = None,
    only_events: Optional[List[str]] = None,
    exclude_events: Optional[List[str]] = None,
    unquote_url_parameters: bool = False,
) -> None:
    if complete_event_type is not None and (
        only_events is not None
        and all((not fnmatch.fnmatch(complete_event_type, pattern) for pattern in only_events))
        or (
            exclude_events is not None
            and any((fnmatch.fnmatch(complete_event_type, pattern) for pattern in exclude_events))
        )
    ):
        return
    client = RequestNotes.get_notes(request).client
    assert client is not None
    if stream is None:
        assert user_profile.bot_owner is not None
        check_send_private_message(user_profile, client, user_profile.bot_owner, body)
    else:
        if unquote_url_parameters:
            stream = unquote(stream)
        if user_specified_topic is not None:
            topic = user_specified_topic
            if unquote_url_parameters:
                topic = unquote(topic)
        try:
            if stream.isdecimal():
                check_send_stream_message_by_id(user_profile, client, int(stream), topic, body)
            else:
                check_send_stream_message(user_profile, client, stream, topic, body)
        except StreamDoesNotExistError:
            pass


def standardize_headers(input_headers: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    This method can be used to standardize a dictionary of headers with
    the standard format that Django expects. For reference, refer to:
    https://docs.djangoproject.com/en/5.0/ref/request-response/#django.http.HttpRequest.headers

    NOTE: Historically, Django's headers were not case-insensitive. We're still
    capitalizing our headers to make it easier to compare/search later if required.
    """
    canonical_headers: Dict[str, str] = {}
    if not input_headers:
        return {}
    for raw_header in input_headers:
        polished_header: str = raw_header.upper().replace('-', '_')
        if (
            polished_header not in ['CONTENT_TYPE', 'CONTENT_LENGTH']
            and not polished_header.startswith('HTTP_')
        ):
            polished_header = 'HTTP_' + polished_header
        canonical_headers[polished_header] = str(input_headers[raw_header])
    return canonical_headers


def validate_extract_webhook_http_header(
    request: HttpRequest, header: str, integration_name: str
) -> str:
    assert request.user.is_authenticated
    extracted_header: Optional[str] = request.headers.get(header)
    if extracted_header is None:
        message_body: str = MISSING_EVENT_HEADER_MESSAGE.format(
            bot_name=request.user.full_name,
            request_path=request.path,
            header_name=header,
            integration_name=integration_name,
            support_email=FromAddress.SUPPORT,
        )
        send_rate_limited_pm_notification_to_bot_owner(request.user, request.user.realm, message_body)
        raise MissingHTTPEventHeaderError(header)
    return extracted_header


def get_fixture_http_headers(integration_name: str, fixture_name: str) -> Dict[str, str]:
    """
    For integrations that require custom HTTP headers for some (or all)
    of their test fixtures, this method will call a specially named
    function from the target integration module to determine what set
    of HTTP headers goes with the given test fixture.
    """
    view_module_name: str = f'zerver.webhooks.{integration_name}.view'
    try:
        view_module = importlib.import_module(view_module_name)
        fixture_to_headers: Callable[[str], Dict[str, str]] = view_module.fixture_to_headers
    except (ImportError, AttributeError):
        return {}
    return fixture_to_headers(fixture_name)


def get_http_headers_from_filename(http_header_key: str) -> Callable[[str], Dict[str, str]]:
    """
    If an integration requires an event type kind of HTTP header which can
    be easily (statically) determined, then name the fixtures in the format
    of "header_value__other_details" or even "header_value" and then use this
    method in the headers.py file for the integration.
    """

    def fixture_to_headers(filename: str) -> Dict[str, str]:
        if '__' in filename:
            event_type: str = filename.split('__')[0]
        else:
            event_type = filename
        return {http_header_key: event_type}

    return fixture_to_headers


def unix_milliseconds_to_timestamp(milliseconds: float, webhook: str) -> datetime:
    """
    If an integration requires time input in unix milliseconds, this helper
    checks to ensure correct type and will catch any errors related to type or
    value and raise a JsonableError.
    Returns a datetime representing the time.
    """
    try:
        seconds: float = milliseconds / 1000
        return timestamp_to_datetime(seconds)
    except (ValueError, TypeError):
        raise JsonableError(
            _('The {webhook} webhook expects time in milliseconds.').format(webhook=webhook)
        )


def parse_multipart_string(body: str) -> Dict[str, str]:
    """
    Converts multipart/form-data string (fixture) to dict
    """
    boundary: str = body.split('\n')[0][2:]
    parts: List[str] = body.split(f'--{boundary}')
    data: Dict[str, str] = {}
    for part in parts:
        if part.strip() in ['', '--']:
            continue
        headers_part, body_part = part.split('\n\n', 1)
        body_part = body_part.removesuffix('\n--')
        content_disposition: str = next(
            (line for line in headers_part.splitlines() if 'Content-Disposition' in line), ''
        )
        field_name: str = content_disposition.split('name="')[1].split('"')[0]
        data[field_name] = body_part
    return data
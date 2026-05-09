import logging
import re
import secrets
from email.headerregistry import Address, AddressHeader
from email.message import EmailMessage
from typing import Any, Callable, Dict, Iterable, List, Match, Optional, Set, Tuple, Union
from django.conf import settings
from django.utils.translation import gettext as _
from zerver.actions.message_send import (
    check_send_message,
    internal_send_group_direct_message,
    internal_send_private_message,
    internal_send_stream_message,
)
from zerver.lib.display_recipient import get_display_recipient
from zerver.lib.email_mirror_helpers import (
    ZulipEmailForwardError,
    ZulipEmailForwardUserError,
    decode_email_address,
    get_email_gateway_message_string_from_address,
)
from zerver.lib.email_notifications import convert_html_to_markdown
from zerver.lib.exceptions import JsonableError, RateLimitedError
from zerver.lib.message import normalize_body, truncate_content, truncate_topic
from zerver.lib.queue import queue_json_publish_rollback_unsafe
from zerver.lib.rate_limiter import RateLimitedObject
from zerver.lib.send_email import FromAddress
from zerver.lib.streams import access_stream_for_send_message
from zerver.lib.string_validation import is_character_printable
from zerver.lib.upload import upload_message_attachment
from zerver.models import (
    ChannelEmailAddress,
    Message,
    MissedMessageEmailAddress,
    Realm,
    Recipient,
    Stream,
    UserProfile,
)
from zerver.models.clients import get_client
from zerver.models.streams import get_stream_by_id_in_realm
from zerver.models.users import get_system_bot, get_user_profile_by_id
from zproject.backends import is_user_active

logger: logging.Logger = ...

def redact_email_address(error_message: str) -> str: ...

def log_error(email_message: dict, error_message: str, to: str | None) -> None: ...

def generate_missed_message_token() -> str: ...

def is_missed_message_address(address: str) -> bool: ...

def is_mm_32_format(msg_string: str | None) -> bool: ...

def get_missed_message_token_from_address(address: str) -> str: ...

def get_usable_missed_message_address(address: str) -> MissedMessageEmailAddress: ...

def create_missed_message_address(user_profile: UserProfile, message: Message) -> str: ...

def construct_zulip_body(
    message: EmailMessage,
    realm: Realm,
    sender: UserProfile,
    show_sender: bool = ...,
    include_quotes: bool = ...,
    include_footer: bool = ...,
    prefer_text: bool = ...,
) -> str: ...

def send_zulip(sender: UserProfile, stream: Stream, topic_name: str, content: str) -> None: ...

def send_mm_reply_to_stream(user_profile: UserProfile, stream: Stream, topic_name: str, body: str) -> None: ...

def get_message_part_by_type(message: EmailMessage, content_type: str) -> str | None: ...

def extract_body(
    message: EmailMessage,
    include_quotes: bool = ...,
    prefer_text: bool = ...,
) -> str: ...

def extract_plaintext_body(message: EmailMessage, include_quotes: bool = ...) -> str | None: ...

def extract_html_body(message: EmailMessage, include_quotes: bool = ...) -> str | None: ...

def filter_footer(text: str) -> str: ...

def extract_and_upload_attachments(message: EmailMessage, realm: Realm, sender: UserProfile) -> str: ...

def decode_stream_email_address(email: str) -> Tuple[ChannelEmailAddress, dict]: ...

def find_emailgateway_recipient(message: EmailMessage) -> str: ...

def strip_from_subject(subject: str) -> str: ...

def is_forwarded(subject: str) -> bool: ...

def process_stream_message(to: str, message: EmailMessage) -> None: ...

def process_missed_message(to: str, message: EmailMessage) -> None: ...

def process_message(message: EmailMessage, rcpt_to: str | None = ...) -> None: ...

def validate_to_address(rcpt_to: str) -> None: ...

def mirror_email_message(rcpt_to: str, msg_base64: str) -> Dict[str, str]: ...

class RateLimitedRealmMirror(RateLimitedObject):
    def __init__(self, realm: Realm) -> None: ...
    
    def key(self) -> str: ...
    
    def rules(self) -> List[Tuple[int, int]]: ...

def rate_limit_mirror_by_realm(recipient_realm: Realm) -> None: ...
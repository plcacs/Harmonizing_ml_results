import logging
import re
import secrets
from email.headerregistry import Address, AddressHeader
from email.message import EmailMessage
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Match,
    Optional,
    Set,
    Tuple,
    Union,
)
from typing_extensions import override

from django.conf import settings
from django.utils.translation import gettext as _
from zerver.actions.message_send import (
    check_send_message,
    internal_send_group_direct_message,
    internal_send_private_message,
    internal_send_stream_message,
)
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

logger = logging.getLogger(__name__)


def redact_email_address(error_message: str) -> str:
    ...


def log_error(email_message: Dict[str, str], error_message: str, to: Optional[str]) -> None:
    ...


def generate_missed_message_token() -> str:
    ...


def is_missed_message_address(address: str) -> bool:
    ...


def is_mm_32_format(msg_string: str) -> bool:
    ...


def get_missed_message_token_from_address(address: str) -> str:
    ...


def get_usable_missed_message_address(address: str) -> MissedMessageEmailAddress:
    ...


def create_missed_message_address(user_profile: UserProfile, message: Message) -> str:
    ...


def construct_zulip_body(
    message: EmailMessage,
    realm: Realm,
    sender: UserProfile,
    show_sender: bool = ...,
    include_quotes: bool = ...,
    include_footer: bool = ...,
    prefer_text: bool = ...,
) -> str:
    ...


def send_zulip(sender: UserProfile, stream: Stream, topic_name: str, content: str) -> None:
    ...


def send_mm_reply_to_stream(
    user_profile: UserProfile, stream: Stream, topic_name: str, body: str
) -> None:
    ...


def get_message_part_by_type(message: EmailMessage, content_type: str) -> Optional[str]:
    ...


def extract_body(
    message: EmailMessage,
    include_quotes: bool = ...,
    prefer_text: bool = ...,
) -> str:
    ...


def extract_plaintext_body(message: EmailMessage, include_quotes: bool = ...) -> Optional[str]:
    ...


def extract_html_body(message: EmailMessage, include_quotes: bool = ...) -> Optional[str]:
    ...


def filter_footer(text: str) -> str:
    ...


def extract_and_upload_attachments(
    message: EmailMessage, realm: Realm, sender: UserProfile
) -> str:
    ...


def decode_stream_email_address(email: str) -> Tuple[ChannelEmailAddress, Dict[str, Any]]:
    ...


def find_emailgateway_recipient(message: EmailMessage) -> str:
    ...


def strip_from_subject(subject: str) -> str:
    ...


def is_forwarded(subject: str) -> bool:
    ...


def process_stream_message(to: str, message: EmailMessage) -> None:
    ...


def process_missed_message(to: str, message: EmailMessage) -> None:
    ...


def process_message(message: EmailMessage, rcpt_to: Optional[str] = ...) -> None:
    ...


def validate_to_address(rcpt_to: str) -> None:
    ...


def mirror_email_message(rcpt_to: str, msg_base64: str) -> Dict[str, Union[str, Dict]]:
    ...


class RateLimitedRealmMirror:
    def __init__(self, realm: Realm) -> None:
        ...

    @override
    def key(self) -> str:
        ...

    @override
    def rules(self) -> List[Tuple[int, int]]:
        ...


def rate_limit_mirror_by_realm(recipient_realm: Realm) -> None:
    ...
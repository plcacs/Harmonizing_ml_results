```pyi
from collections.abc import Callable
from datetime import datetime
from typing import Any

from django.db.models import QuerySet
from zerver.lib.types import EditHistoryEvent, StreamMessageEditRequest
from zerver.models import Message, UserProfile

ORIG_TOPIC: str
TOPIC_NAME: str
TOPIC_LINKS: str
MATCH_TOPIC: str
RESOLVED_TOPIC_PREFIX: str
EXPORT_TOPIC_NAME: str
DB_TOPIC_NAME: str
MESSAGE__TOPIC: str

def get_topic_from_message_info(message_info: dict[str, Any]) -> str: ...
def filter_by_topic_name_via_message(query: QuerySet[Any], topic_name: str) -> QuerySet[Any]: ...
def messages_for_topic(realm_id: int, stream_recipient_id: int, topic_name: str) -> QuerySet[Message]: ...
def get_first_message_for_user_in_topic(
    realm_id: int,
    user_profile: UserProfile | None,
    recipient_id: int,
    topic_name: str,
    history_public_to_subscribers: bool,
    acting_user_has_channel_content_access: bool = False,
) -> int | None: ...
def save_message_for_edit_use_case(message: Message) -> None: ...
def user_message_exists_for_topic(user_profile: UserProfile, recipient_id: int, topic_name: str) -> bool: ...
def update_edit_history(message: Message, last_edit_time: datetime, edit_history_event: EditHistoryEvent) -> None: ...
def update_messages_for_topic_edit(
    acting_user: UserProfile,
    edited_message: Message,
    message_edit_request: StreamMessageEditRequest,
    edit_history_event: EditHistoryEvent,
    last_edit_time: datetime,
) -> tuple[QuerySet[Message], Callable[[], QuerySet[Message]]]: ...
def generate_topic_history_from_db_rows(
    rows: list[tuple[str, int]], allow_empty_topic_name: bool
) -> list[dict[str, Any]]: ...
def get_topic_history_for_public_stream(
    realm_id: int, recipient_id: int, allow_empty_topic_name: bool
) -> list[dict[str, Any]]: ...
def get_topic_history_for_stream(
    user_profile: UserProfile, recipient_id: int, public_history: bool, allow_empty_topic_name: bool
) -> list[dict[str, Any]]: ...
def get_topic_resolution_and_bare_name(stored_name: str) -> tuple[bool, str]: ...
def participants_for_topic(realm_id: int, recipient_id: int, topic_name: str) -> set[int]: ...
def maybe_rename_general_chat_to_empty_topic(topic_name: str) -> str: ...
def maybe_rename_empty_topic_to_general_chat(
    topic_name: str, is_channel_message: bool, allow_empty_topic_name: bool
) -> str: ...
def get_topic_display_name(topic_name: str, language: str) -> str: ...
```
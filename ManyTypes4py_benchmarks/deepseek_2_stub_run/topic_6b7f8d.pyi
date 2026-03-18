```python
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional, Union, Iterable, Sequence, Collection, Set, Dict, List, Tuple
import orjson
from django.db import connection
from django.db.models import F, Func, JSONField, Q, QuerySet, Subquery, TextField, Value
from django.db.models.functions import Cast
from django.utils.translation import override as override_language
from zerver.lib.types import EditHistoryEvent, StreamMessageEditRequest
from zerver.lib.utils import assert_is_not_none
from zerver.models import Message, Reaction, UserMessage, UserProfile

ORIG_TOPIC: str
TOPIC_NAME: str
TOPIC_LINKS: str
MATCH_TOPIC: str
RESOLVED_TOPIC_PREFIX: str
EXPORT_TOPIC_NAME: str
DB_TOPIC_NAME: str
MESSAGE__TOPIC: str

def get_topic_from_message_info(message_info: Any) -> Any: ...

def filter_by_topic_name_via_message(query: Any, topic_name: str) -> Any: ...

def messages_for_topic(realm_id: Any, stream_recipient_id: Any, topic_name: str) -> QuerySet[Message]: ...

def get_first_message_for_user_in_topic(
    realm_id: Any,
    user_profile: Optional[UserProfile],
    recipient_id: Any,
    topic_name: str,
    history_public_to_subscribers: bool,
    acting_user_has_channel_content_access: bool = ...
) -> Optional[Any]: ...

def save_message_for_edit_use_case(message: Message) -> None: ...

def user_message_exists_for_topic(user_profile: UserProfile, recipient_id: Any, topic_name: str) -> bool: ...

def update_edit_history(message: Message, last_edit_time: datetime, edit_history_event: EditHistoryEvent) -> None: ...

def update_messages_for_topic_edit(
    acting_user: UserProfile,
    edited_message: Message,
    message_edit_request: StreamMessageEditRequest,
    edit_history_event: EditHistoryEvent,
    last_edit_time: datetime
) -> Tuple[Any, Callable[[], QuerySet[Message]]]: ...

def generate_topic_history_from_db_rows(rows: Any, allow_empty_topic_name: bool) -> List[Dict[str, Any]]: ...

def get_topic_history_for_public_stream(realm_id: Any, recipient_id: Any, allow_empty_topic_name: bool) -> List[Dict[str, Any]]: ...

def get_topic_history_for_stream(
    user_profile: UserProfile,
    recipient_id: Any,
    public_history: bool,
    allow_empty_topic_name: bool
) -> List[Dict[str, Any]]: ...

def get_topic_resolution_and_bare_name(stored_name: str) -> Tuple[bool, str]: ...

def participants_for_topic(realm_id: Any, recipient_id: Any, topic_name: str) -> Set[Any]: ...

def maybe_rename_general_chat_to_empty_topic(topic_name: str) -> str: ...

def maybe_rename_empty_topic_to_general_chat(
    topic_name: str,
    is_channel_message: bool,
    allow_empty_topic_name: bool
) -> str: ...

def get_topic_display_name(topic_name: str, language: str) -> str: ...
```
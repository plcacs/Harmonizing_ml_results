from typing import Any, List, Set, Dict, Tuple, Optional

class MessageDetailsDict(TypedDict, total=False):
    pass

class RawUnreadStreamDict(TypedDict):
    pass

class RawUnreadDirectMessageDict(TypedDict):
    pass

class RawUnreadDirectMessageGroupDict(TypedDict):
    pass

class RawUnreadMessagesResult(TypedDict):
    pass

class UnreadStreamInfo(TypedDict):
    pass

class UnreadDirectMessageInfo(TypedDict):
    pass

class UnreadDirectMessageGroupInfo(TypedDict):
    pass

class UnreadMessagesResult(TypedDict):
    pass

@dataclass
class SendMessageRequest:
    submessages: List = field(default_factory=list)
    deliver_at: Optional[datetime] = None
    delivery_type: Optional[Any] = None
    limit_unread_user_ids: Optional[Any] = None
    service_queue_events: Optional[Any] = None
    disable_external_notifications: bool = False
    automatic_new_visibility_policy: Optional[Any] = None
    recipients_for_user_creation_events: Optional[Any] = None

def truncate_content(content: str, max_length: int, truncation_message: str) -> str:
    if len(content) > max_length:
        content = content[:max_length - len(truncation_message)] + truncation_message
    return content

def normalize_body(body: str) -> str:
    body = body.rstrip().lstrip('\n')
    if len(body) == 0:
        raise JsonableError(_('Message must not be empty'))
    if '\x00' in body:
        raise JsonableError(_('Message must not contain null bytes'))
    return truncate_content(body, settings.MAX_MESSAGE_LENGTH, '\n[message truncated]')

def normalize_body_for_import(body: str) -> str:
    if '\x00' in body:
        body = re.sub('\\x00', '', body)
    return truncate_content(body, settings.MAX_MESSAGE_LENGTH, '\n[message truncated]')

def truncate_topic(topic_name: str) -> str:
    return truncate_content(topic_name, MAX_TOPIC_NAME_LENGTH, '...')

def messages_for_ids(message_ids: List[int], user_message_flags: Dict[int, Any], search_fields: Dict[int, Any], apply_markdown: Any, client_gravatar: Any, allow_empty_topic_name: Any, allow_edit_history: Any, user_profile: Any, realm: Any) -> List[Dict[str, Any]]:
    ...

def access_message(user_profile: Any, message_id: int, lock_message: bool = False) -> Any:
    ...

def access_message_and_usermessage(user_profile: Any, message_id: int, lock_message: bool = False) -> Tuple[Any, Any]:
    ...

def access_web_public_message(realm: Any, message_id: int) -> Any:
    ...

def has_message_access(user_profile: Any, message: Any, *, has_user_message: Callable[[], bool], stream: Any = None, is_subscribed: Any = None) -> bool:
    ...

def event_recipient_ids_for_action_on_messages(messages: List[Any], *, channel: Any = None, exclude_long_term_idle_users: bool = True) -> Set[int]:
    ...

def bulk_access_messages(user_profile: Any, messages: List[Any], *, stream: Any = None) -> List[Any]:
    ...

def bulk_access_stream_messages_query(user_profile: Any, messages: Any, stream: Any) -> Any:
    ...

def get_messages_with_usermessage_rows_for_user(user_profile_id: int, message_ids: List[int]) -> List[int]:
    ...

def direct_message_group_users(recipient_id: int) -> str:
    ...

def get_inactive_recipient_ids(user_profile: Any) -> List[int]:
    ...

def get_muted_stream_ids(user_profile: Any) -> Set[int]:
    ...

def get_starred_message_ids(user_profile: Any) -> List[int]:
    ...

def get_raw_unread_data(user_profile: Any, message_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    ...

def extract_unread_data_from_um_rows(rows: List[Dict[str, Any]], user_profile: Any) -> Dict[str, Any]:
    ...

def aggregate_streams(*, input_dict: Dict[int, Dict[str, Any]], allow_empty_topic_name: Any) -> List[Dict[str, Any]]:
    ...

def aggregate_pms(*, input_dict: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    ...

def aggregate_direct_message_groups(*, input_dict: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    ...

def aggregate_unread_data(raw_data: Dict[str, Any], allow_empty_topic_name: Any) -> Dict[str, Any]:
    ...

def apply_unread_message_event(user_profile: Any, state: Dict[str, Any], message: Dict[str, Any], flags: Any) -> None:
    ...

def remove_message_id_from_unread_mgs(state: Dict[str, Any], message_id: int) -> None:
    ...

def format_unread_message_details(my_user_id: int, raw_unread_data: Dict[str, Any]) -> Dict[str, Any]:
    ...

def add_message_to_unread_msgs(my_user_id: int, state: Dict[str, Any], message_id: int, message_details: Dict[str, Any]) -> None:
    ...

def estimate_recent_messages(realm: Any, hours: int) -> int:
    ...

def get_first_visible_message_id(realm: Any) -> int:
    ...

def maybe_update_first_visible_message_id(realm: Any, lookback_hours: int) -> None:
    ...

def update_first_visible_message_id(realm: Any) -> None:
    ...

def get_last_message_id() -> int:
    ...

def get_recent_conversations_recipient_id(user_profile: Any, recipient_id: int, sender_id: int) -> int:
    ...

def get_recent_private_conversations(user_profile: Any) -> Dict[int, Dict[str, Any]]:
    ...

def can_mention_many_users(sender: Any) -> bool:
    ...

def topic_wildcard_mention_allowed(sender: Any, topic_participant_count: int, realm: Any) -> bool:
    ...

def stream_wildcard_mention_allowed(sender: Any, stream: Any, realm: Any) -> bool:
    ...

def check_user_group_mention_allowed(sender: Any, user_group_ids: List[int]) -> None:
    ...

def parse_message_time_limit_setting(value: Any, special_values_map: Dict[str, int], *, setting_name: str) -> int:
    ...

def visibility_policy_for_participation(sender: Any, is_stream_muted: bool) -> Optional[int]:
    ...

def visibility_policy_for_send(sender: Any, is_stream_muted: bool) -> Optional[int]:
    ...

def visibility_policy_for_send_message(sender: Any, message: Any, stream: Any, is_stream_muted: bool, current_visibility_policy: int) -> Optional[int]:
    ...

def should_change_visibility_policy(new_visibility_policy: int, sender: Any, stream_id: int, topic_name: str) -> bool:
    ...

def set_visibility_policy_possible(user_profile: Any, message: Any) -> bool:
    ...

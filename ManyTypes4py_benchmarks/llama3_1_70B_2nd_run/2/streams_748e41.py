from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Optional, Set, Any

class StreamDict(TypedDict, total=False):
    name: str
    invite_only: bool
    is_web_public: bool
    history_public_to_subscribers: bool
    description: str
    message_retention_days: Optional[int]
    can_add_subscribers_group: Optional[int]
    can_administer_channel_group: Optional[int]
    can_send_message_group: Optional[int]
    can_remove_subscribers_group: Optional[int]

class APIStreamDict(TypedDict):
    is_archived: bool
    can_add_subscribers_group: int
    can_administer_channel_group: int
    can_send_message_group: int
    can_remove_subscribers_group: int
    creator_id: int
    date_created: int
    description: str
    first_message_id: int
    is_recently_active: bool
    history_public_to_subscribers: bool
    invite_only: bool
    is_web_public: bool
    message_retention_days: Optional[int]
    name: str
    rendered_description: str
    stream_id: int
    stream_post_policy: int
    is_announcement_only: bool
    stream_weekly_traffic: Optional[int]

class AnonymousSettingGroupDict(TypedDict):
    direct_members: List[int]
    direct_subgroups: List[int]

def get_stream_permission_policy_name(*, invite_only: Optional[bool] = None, history_public_to_subscribers: Optional[bool] = None, is_web_public: Optional[bool] = None) -> str:
    ...

def get_default_value_for_history_public_to_subscribers(realm: Any, invite_only: bool, history_public_to_subscribers: Optional[bool]) -> bool:
    ...

def render_stream_description(text: str, realm: Any, *, acting_user: Optional[Any] = None) -> str:
    ...

def send_stream_creation_event(realm: Any, stream: Any, user_ids: List[int], recent_traffic: Optional[Dict[int, int]] = None, setting_groups_dict: Optional[Dict[int, Any]] = None) -> None:
    ...

def get_stream_permission_default_group(setting_name: str, system_groups_name_dict: Dict[str, Any], creator: Optional[Any] = None) -> Any:
    ...

def get_default_values_for_stream_permission_group_settings(realm: Any, creator: Optional[Any] = None) -> Dict[str, Any]:
    ...

def get_user_ids_with_metadata_access_via_permission_groups(stream: Any) -> Set[int]:
    ...

@transaction.atomic(savepoint=False)
def create_stream_if_needed(realm: Any, stream_name: str, *, invite_only: bool = False, is_web_public: bool = False, history_public_to_subscribers: Optional[bool] = None, stream_description: str = '', message_retention_days: Optional[int] = None, can_add_subscribers_group: Optional[Any] = None, can_administer_channel_group: Optional[Any] = None, can_send_message_group: Optional[Any] = None, can_remove_subscribers_group: Optional[Any] = None, acting_user: Optional[Any] = None, setting_groups_dict: Optional[Dict[int, Any]] = None) -> tuple:
    ...

def create_streams_if_needed(realm: Any, stream_dicts: List[StreamDict], acting_user: Optional[Any] = None, setting_groups_dict: Optional[Dict[int, Any]] = None) -> tuple:
    ...

def subscribed_to_stream(user_profile: Any, stream_id: int) -> bool:
    ...

def is_user_in_can_administer_channel_group(stream: Any, user_recursive_group_ids: Set[int]) -> bool:
    ...

def is_user_in_can_add_subscribers_group(stream: Any, user_recursive_group_ids: Set[int]) -> bool:
    ...

def is_user_in_can_remove_subscribers_group(stream: Any, user_recursive_group_ids: Set[int]) -> bool:
    ...

def check_stream_access_based_on_can_send_message_group(sender: Any, stream: Any) -> None:
    ...

def access_stream_for_send_message(sender: Any, stream: Any, forwarder_user_profile: Optional[Any], archived_channel_notice: bool = False) -> None:
    ...

def get_public_streams_queryset(realm: Any) -> QuerySet:
    ...

def get_web_public_streams_queryset(realm: Any) -> QuerySet:
    ...

def check_stream_name_available(realm: Any, name: str) -> None:
    ...

def access_stream_by_name(user_profile: Any, stream_name: str, require_content_access: bool = True) -> tuple:
    ...

def access_web_public_stream(stream_id: int, realm: Any) -> Any:
    ...

def access_stream_to_remove_visibility_policy_by_name(user_profile: Any, stream_name: str, error: str) -> Any:
    ...

def access_stream_to_remove_visibility_policy_by_id(user_profile: Any, stream_id: int, error: str) -> Any:
    ...

def private_stream_user_ids(stream_id: int) -> Set[int]:
    ...

def public_stream_user_ids(stream: Any) -> Set[int]:
    ...

def can_access_stream_metadata_user_ids(stream: Any) -> Set[int]:
    ...

def can_access_stream_history(user_profile: Any, stream: Any) -> bool:
    ...

def can_access_stream_history_by_name(user_profile: Any, stream_name: str) -> bool:
    ...

def can_access_stream_history_by_id(user_profile: Any, stream_id: int) -> bool:
    ...

def bulk_can_remove_subscribers_from_streams(streams: List[Any], user_profile: Any) -> bool:
    ...

def get_streams_to_which_user_cannot_add_subscribers(streams: List[Any], user_profile: Any, *, allow_default_streams: bool = False) -> List[Any]:
    ...

def can_administer_accessible_channel(channel: Any, user_profile: Any) -> bool:
    ...

@dataclass
class UserGroupMembershipDetails:
    pass

def user_has_content_access(user_profile: Any, stream: Any, user_group_membership_details: UserGroupMembershipDetails, *, is_subscribed: bool) -> bool:
    ...

def check_stream_access_for_delete_or_update_requiring_metadata_access(user_profile: Any, stream: Any, sub: Optional[Any] = None) -> None:
    ...

def access_stream_for_delete_or_update_requiring_metadata_access(user_profile: Any, stream_id: int) -> tuple:
    ...

def has_metadata_access_to_channel_via_groups(user_profile: Any, user_recursive_group_ids: Set[int], can_administer_channel_group_id: int, can_add_subscribers_group_id: int) -> bool:
    ...

def check_basic_stream_access(user_profile: Any, stream: Any, *, is_subscribed: bool, require_content_access: bool = True) -> bool:
    ...

def access_stream_common(user_profile: Any, stream: Any, error: str, require_active: bool = True, require_content_access: bool = True) -> Optional[Any]:
    ...

def access_stream_by_id(user_profile: Any, stream_id: int, require_active: bool = True, require_content_access: bool = True) -> tuple:
    ...

def access_stream_by_id_for_message(user_profile: Any, stream_id: int, require_active: bool = True, require_content_access: bool = True) -> tuple:
    ...

def get_stream_by_narrow_operand_access_unchecked(operand: Any, realm: Any) -> Any:
    ...

def ensure_stream(realm: Any, stream_name: str, invite_only: bool = False, stream_description: str = '', *, acting_user: Any) -> Any:
    ...

def get_occupied_streams(realm: Any) -> QuerySet:
    ...

def get_stream_post_policy_value_based_on_group_setting(setting_group: Any) -> int:
    ...

def stream_to_dict(stream: Any, recent_traffic: Optional[Dict[int, int]] = None, setting_groups_dict: Optional[Dict[int, Any]] = None) -> APIStreamDict:
    ...

def get_web_public_streams(realm: Any) -> List[APIStreamDict]:
    ...

def get_streams_for_user(user_profile: Any, include_public: bool = True, include_web_public: bool = False, include_subscribed: bool = True, exclude_archived: bool = True, include_all_active: bool = False, include_owner_subscribed: bool = False) -> List[Any]:
    ...

def do_get_streams(user_profile: Any, include_public: bool = True, include_web_public: bool = False, include_subscribed: bool = True, exclude_archived: bool = True, include_all_active: bool = False, include_default: bool = False, include_owner_subscribed: bool = False) -> List[APIStreamDict]:
    ...

def get_subscribed_private_streams_for_user(user_profile: Any) -> QuerySet:
    ...

def notify_stream_is_recently_active_update(stream: Any, value: bool) -> None:
    ...

@transaction.atomic(durable=True)
def update_stream_active_status_for_realm(realm: Any, date_days_ago: datetime) -> int:
    ...

def check_update_all_streams_active_status(days: int = 30) -> int:
    ...

def send_stream_deletion_event(realm: Any, user_ids: List[int], streams: List[Any]) -> None:
    ...

def list_to_streams(streams_raw: List[StreamDict], user_profile: Any, autocreate: bool = False, unsubscribing_others: bool = False, is_default_stream: bool = False, setting_groups_dict: Optional[Dict[int, Any]] = None) -> tuple:
    ...

def access_default_stream_group_by_id(realm: Any, group_id: int) -> Any:
    ...

def get_group_setting_value_dict_for_streams(streams: List[Any]) -> Dict[int, Any]:
    ...

def get_setting_values_for_group_settings(group_ids: List[int]) -> Dict[int, Any]:
    ...

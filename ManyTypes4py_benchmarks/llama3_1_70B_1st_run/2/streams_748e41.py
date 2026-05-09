from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypedDict, List, Tuple, Dict, Optional, Set, Any
from django.db import transaction
from django.db.models import Exists, OuterRef, Q, QuerySet, Value
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from zerver.lib.default_streams import get_default_stream_ids_for_realm
from zerver.lib.exceptions import CannotAdministerChannelError, IncompatibleParametersError, JsonableError, OrganizationOwnerRequiredError
from zerver.lib.stream_subscription import get_active_subscriptions_for_stream_id, get_subscribed_stream_ids_for_user
from zerver.lib.stream_traffic import get_average_weekly_stream_traffic, get_streams_traffic
from zerver.lib.string_validation import check_stream_name
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.types import AnonymousSettingGroupDict, APIStreamDict
from zerver.lib.user_groups import get_recursive_group_members, get_recursive_group_members_union_for_groups, get_recursive_membership_groups, get_role_based_system_groups_dict, user_has_permission_for_group_setting
from zerver.models import DefaultStreamGroup, GroupGroupMembership, Message, NamedUserGroup, Realm, RealmAuditLog, Recipient, Stream, Subscription, UserGroup, UserGroupMembership, UserProfile
from zerver.models.groups import SystemGroups
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.streams import bulk_get_streams, get_realm_stream, get_stream, get_stream_by_id_for_sending_message, get_stream_by_id_in_realm
from zerver.models.users import active_non_guest_user_ids, active_user_ids, is_cross_realm_bot_email
from zerver.tornado.django_api import send_event_on_commit

class StreamDict(TypedDict, total=False):
    pass

def get_stream_permission_policy_name(*, invite_only: Optional[bool] = None, history_public_to_subscribers: Optional[bool] = None, is_web_public: Optional[bool] = None) -> str:
    pass

def get_default_value_for_history_public_to_subscribers(realm: Realm, invite_only: bool, history_public_to_subscribers: Optional[bool]) -> bool:
    pass

def render_stream_description(text: str, realm: Realm, *, acting_user: Optional[UserProfile] = None) -> str:
    pass

def send_stream_creation_event(realm: Realm, stream: Stream, user_ids: List[int], recent_traffic: Optional[Dict[int, int]] = None, setting_groups_dict: Optional[Dict[int, Any]] = None) -> None:
    pass

def get_stream_permission_default_group(setting_name: str, system_groups_name_dict: Dict[str, UserGroup], creator: Optional[UserProfile] = None) -> UserGroup:
    pass

def get_default_values_for_stream_permission_group_settings(realm: Realm, creator: Optional[UserProfile] = None) -> Dict[str, UserGroup]:
    pass

def get_user_ids_with_metadata_access_via_permission_groups(stream: Stream) -> Set[int]:
    pass

@transaction.atomic(savepoint=False)
def create_stream_if_needed(realm: Realm, stream_name: str, *, invite_only: bool = False, is_web_public: bool = False, history_public_to_subscribers: Optional[bool] = None, stream_description: str = '', message_retention_days: Optional[int] = None, can_add_subscribers_group: Optional[UserGroup] = None, can_administer_channel_group: Optional[UserGroup] = None, can_send_message_group: Optional[UserGroup] = None, can_remove_subscribers_group: Optional[UserGroup] = None, acting_user: Optional[UserProfile] = None, setting_groups_dict: Optional[Dict[int, Any]] = None) -> Tuple[Stream, bool]:
    pass

def create_streams_if_needed(realm: Realm, stream_dicts: List[StreamDict], acting_user: Optional[UserProfile] = None, setting_groups_dict: Optional[Dict[int, Any]] = None) -> Tuple[List[Stream], List[Stream]]:
    pass

def subscribed_to_stream(user_profile: UserProfile, stream_id: int) -> bool:
    pass

def is_user_in_can_administer_channel_group(stream: Stream, user_recursive_group_ids: Set[int]) -> bool:
    pass

def is_user_in_can_add_subscribers_group(stream: Stream, user_recursive_group_ids: Set[int]) -> bool:
    pass

def is_user_in_can_remove_subscribers_group(stream: Stream, user_recursive_group_ids: Set[int]) -> bool:
    pass

def check_stream_access_based_on_can_send_message_group(sender: UserProfile, stream: Stream) -> None:
    pass

def access_stream_for_send_message(sender: UserProfile, stream: Stream, forwarder_user_profile: Optional[UserProfile], archived_channel_notice: bool = False) -> None:
    pass

def check_for_exactly_one_stream_arg(stream_id: Optional[int], stream: Optional[Stream]) -> None:
    pass

@dataclass
class UserGroupMembershipDetails:
    pass

def user_has_content_access(user_profile: UserProfile, stream: Stream, user_group_membership_details: UserGroupMembershipDetails, *, is_subscribed: bool) -> bool:
    pass

def check_stream_access_for_delete_or_update_requiring_metadata_access(user_profile: UserProfile, stream: Stream, sub: Optional[Subscription]) -> None:
    pass

def access_stream_for_delete_or_update_requiring_metadata_access(user_profile: UserProfile, stream_id: int) -> Tuple[Stream, Optional[Subscription]]:
    pass

def has_metadata_access_to_channel_via_groups(user_profile: UserProfile, user_recursive_group_ids: Set[int], can_administer_channel_group_id: int, can_add_subscribers_group_id: int) -> bool:
    pass

def check_basic_stream_access(user_profile: UserProfile, stream: Stream, *, is_subscribed: bool, require_content_access: bool) -> bool:
    pass

def access_stream_common(user_profile: UserProfile, stream: Stream, error: str, require_active: bool = True, require_content_access: bool = True) -> Optional[Subscription]:
    pass

def access_stream_by_id(user_profile: UserProfile, stream_id: int, require_active: bool = True, require_content_access: bool = True) -> Tuple[Stream, Optional[Subscription]]:
    pass

def access_stream_by_id_for_message(user_profile: UserProfile, stream_id: int, require_active: bool = True, require_content_access: bool = True) -> Tuple[Stream, Optional[Subscription]]:
    pass

def get_public_streams_queryset(realm: Realm) -> QuerySet[Stream]:
    pass

def get_web_public_streams_queryset(realm: Realm) -> QuerySet[Stream]:
    pass

def check_stream_name_available(realm: Realm, name: str) -> None:
    pass

def access_stream_by_name(user_profile: UserProfile, stream_name: str, require_content_access: bool = True) -> Tuple[Stream, Optional[Subscription]]:
    pass

def access_web_public_stream(stream_id: int, realm: Realm) -> Stream:
    pass

def access_stream_to_remove_visibility_policy_by_name(user_profile: UserProfile, stream_name: str, error: str) -> Stream:
    pass

def access_stream_to_remove_visibility_policy_by_id(user_profile: UserProfile, stream_id: int, error: str) -> Stream:
    pass

def private_stream_user_ids(stream_id: int) -> Set[int]:
    pass

def public_stream_user_ids(stream: Stream) -> Set[int]:
    pass

def can_access_stream_metadata_user_ids(stream: Stream) -> Set[int]:
    pass

def can_access_stream_history(user_profile: UserProfile, stream: Stream) -> bool:
    pass

def can_access_stream_history_by_name(user_profile: UserProfile, stream_name: str) -> bool:
    pass

def can_access_stream_history_by_id(user_profile: UserProfile, stream_id: int) -> bool:
    pass

def bulk_can_remove_subscribers_from_streams(streams: List[Stream], user_profile: UserProfile) -> bool:
    pass

def get_streams_to_which_user_cannot_add_subscribers(streams: List[Stream], user_profile: UserProfile, *, allow_default_streams: bool = False) -> List[Stream]:
    pass

def can_administer_accessible_channel(channel: Stream, user_profile: UserProfile) -> bool:
    pass

@dataclass
class StreamsCategorizedByPermissions:
    pass

def filter_stream_authorization(user_profile: UserProfile, streams: List[Stream], is_subscribing_other_users: bool = False) -> StreamsCategorizedByPermissions:
    pass

def list_to_streams(streams_raw: List[StreamDict], user_profile: UserProfile, autocreate: bool = False, unsubscribing_others: bool = False, is_default_stream: bool = False, setting_groups_dict: Optional[Dict[int, Any]] = None) -> Tuple[List[Stream], List[Stream]]:
    pass

def access_default_stream_group_by_id(realm: Realm, group_id: int) -> DefaultStreamGroup:
    pass

def get_stream_by_narrow_operand_access_unchecked(operand: Union[str, int], realm: Realm) -> Stream:
    pass

def ensure_stream(realm: Realm, stream_name: str, invite_only: bool = False, stream_description: str = '', *, acting_user: UserProfile) -> Stream:
    pass

def get_occupied_streams(realm: Realm) -> QuerySet[Stream]:
    pass

def get_stream_post_policy_value_based_on_group_setting(setting_group: UserGroup) -> int:
    pass

def stream_to_dict(stream: Stream, recent_traffic: Optional[Dict[int, int]] = None, setting_groups_dict: Optional[Dict[int, Any]] = None) -> APIStreamDict:
    pass

def get_web_public_streams(realm: Realm) -> List[APIStreamDict]:
    pass

def get_streams_for_user(user_profile: UserProfile, include_public: bool = True, include_web_public: bool = False, include_subscribed: bool = True, exclude_archived: bool = True, include_all_active: bool = False, include_owner_subscribed: bool = False) -> List[Stream]:
    pass

def do_get_streams(user_profile: UserProfile, include_public: bool = True, include_web_public: bool = False, include_subscribed: bool = True, exclude_archived: bool = True, include_all_active: bool = False, include_default: bool = False, include_owner_subscribed: bool = False) -> List[APIStreamDict]:
    pass

def get_subscribed_private_streams_for_user(user_profile: UserProfile) -> QuerySet[Stream]:
    pass

def notify_stream_is_recently_active_update(stream: Stream, value: bool) -> None:
    pass

@transaction.atomic(durable=True)
def update_stream_active_status_for_realm(realm: Realm, date_days_ago: datetime) -> int:
    pass

def check_update_all_streams_active_status(days: int = Stream.LAST_ACTIVITY_DAYS_BEFORE_FOR_ACTIVE) -> int:
    pass

def send_stream_deletion_event(realm: Realm, user_ids: List[int], streams: List[Stream]) -> None:
    pass

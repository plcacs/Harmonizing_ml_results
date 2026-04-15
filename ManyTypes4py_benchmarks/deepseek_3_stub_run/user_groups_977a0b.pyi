from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TypedDict, Any, Optional, Union
import django.db.utils
from django.db import transaction
from django.utils.timezone import datetime as datetime_with_tz
from django.utils.translation import gettext as _
from zerver.lib.exceptions import JsonableError
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.types import AnonymousSettingGroupDict
from zerver.lib.user_groups import get_group_setting_value_for_api, get_group_setting_value_for_audit_log_data, get_role_based_system_groups_dict, set_defaults_for_group_settings
from zerver.models import GroupGroupMembership, NamedUserGroup, Realm, RealmAuditLog, UserGroup, UserGroupMembership, UserProfile
from zerver.models.groups import SystemGroups
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.users import active_user_ids
from zerver.tornado.django_api import send_event_on_commit

class MemberGroupUserDict(TypedDict):
    pass

@transaction.atomic(savepoint=False)
def create_user_group_in_database(
    name: str,
    members: Sequence[UserProfile],
    realm: Realm,
    *,
    acting_user: Optional[UserProfile],
    description: str = "",
    group_settings_map: Mapping[str, Any] = ...,
    is_system_group: bool = False
) -> NamedUserGroup: ...

@transaction.atomic(savepoint=False)
def update_users_in_full_members_system_group(
    realm: Realm,
    affected_user_ids: Sequence[int] = ...,
    *,
    acting_user: Optional[UserProfile]
) -> None: ...

def promote_new_full_members() -> None: ...

def do_send_create_user_group_event(
    user_group: NamedUserGroup,
    members: Sequence[UserProfile],
    direct_subgroups: Sequence[NamedUserGroup] = ...
) -> None: ...

def check_add_user_group(
    realm: Realm,
    name: str,
    initial_members: Sequence[UserProfile],
    description: str = "",
    group_settings_map: Mapping[str, Any] = ...,
    *,
    acting_user: Optional[UserProfile]
) -> NamedUserGroup: ...

def do_send_user_group_update_event(
    user_group: NamedUserGroup,
    data: dict[str, Any]
) -> None: ...

@transaction.atomic(savepoint=False)
def do_update_user_group_name(
    user_group: NamedUserGroup,
    name: str,
    *,
    acting_user: Optional[UserProfile]
) -> None: ...

@transaction.atomic(savepoint=False)
def do_update_user_group_description(
    user_group: NamedUserGroup,
    description: str,
    *,
    acting_user: Optional[UserProfile]
) -> None: ...

def do_send_user_group_members_update_event(
    event_name: str,
    user_group: NamedUserGroup,
    user_ids: Sequence[int]
) -> None: ...

@transaction.atomic(savepoint=False)
def bulk_add_members_to_user_groups(
    user_groups: Sequence[NamedUserGroup],
    user_profile_ids: Sequence[int],
    *,
    acting_user: Optional[UserProfile]
) -> None: ...

@transaction.atomic(savepoint=False)
def bulk_remove_members_from_user_groups(
    user_groups: Sequence[NamedUserGroup],
    user_profile_ids: Sequence[int],
    *,
    acting_user: Optional[UserProfile]
) -> None: ...

def do_send_subgroups_update_event(
    event_name: str,
    user_group: NamedUserGroup,
    subgroup_ids: Sequence[int]
) -> None: ...

@transaction.atomic(savepoint=False)
def add_subgroups_to_user_group(
    user_group: NamedUserGroup,
    subgroups: Sequence[NamedUserGroup],
    *,
    acting_user: Optional[UserProfile]
) -> None: ...

@transaction.atomic(savepoint=False)
def remove_subgroups_from_user_group(
    user_group: NamedUserGroup,
    subgroups: Sequence[NamedUserGroup],
    *,
    acting_user: Optional[UserProfile]
) -> None: ...

@transaction.atomic(savepoint=False)
def do_deactivate_user_group(
    user_group: NamedUserGroup,
    *,
    acting_user: Optional[UserProfile]
) -> None: ...

@transaction.atomic(savepoint=False)
def do_change_user_group_permission_setting(
    user_group: NamedUserGroup,
    setting_name: str,
    setting_value_group: Any,
    *,
    old_setting_api_value: Optional[Any] = None,
    acting_user: Optional[UserProfile]
) -> None: ...
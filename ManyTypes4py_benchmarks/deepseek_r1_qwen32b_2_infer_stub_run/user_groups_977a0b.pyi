from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from django.db import models
from django.utils.translation import gettext
from zerver.lib.exceptions import JsonableError
from zerver.lib.types import AnonymousSettingGroupDict
from zerver.models import (
    GroupGroupMembership,
    NamedUserGroup,
    Realm,
    RealmAuditLog,
    UserGroup,
    UserGroupMembership,
    UserProfile,
)
from zerver.models.groups import SystemGroups
from zerver.models.realm_audit_logs import AuditLogEventType

class MemberGroupUserDict(TypedDict):
    pass

@transaction.atomic(savepoint=False)
def create_user_group_in_database(
    name: str,
    members: List[UserProfile],
    realm: Realm,
    *,
    acting_user: UserProfile,
    description: str = '',
    group_settings_map: Dict[str, Any] = {},
    is_system_group: bool = False
) -> NamedUserGroup:
    ...

def promote_new_full_members() -> None:
    ...

def do_send_create_user_group_event(
    user_group: NamedUserGroup,
    members: List[UserProfile],
    direct_subgroups: List[UserGroup] = []
) -> None:
    ...

def check_add_user_group(
    realm: Realm,
    name: str,
    initial_members: List[UserProfile],
    description: str = '',
    group_settings_map: Dict[str, Any] = {},
    *,
    acting_user: UserProfile
) -> NamedUserGroup:
    ...

def do_send_user_group_update_event(
    user_group: NamedUserGroup,
    data: Dict[str, Any]
) -> None:
    ...

@transaction.atomic(savepoint=False)
def do_update_user_group_name(
    user_group: NamedUserGroup,
    name: str,
    *,
    acting_user: UserProfile
) -> None:
    ...

@transaction.atomic(savepoint=False)
def do_update_user_group_description(
    user_group: NamedUserGroup,
    description: str,
    *,
    acting_user: UserProfile
) -> None:
    ...

def do_send_user_group_members_update_event(
    event_name: str,
    user_group: NamedUserGroup,
    user_ids: List[int]
) -> None:
    ...

@transaction.atomic(savepoint=False)
def bulk_add_members_to_user_groups(
    user_groups: List[UserGroup],
    user_profile_ids: List[int],
    *,
    acting_user: UserProfile
) -> None:
    ...

@transaction.atomic(savepoint=False)
def bulk_remove_members_from_user_groups(
    user_groups: List[UserGroup],
    user_profile_ids: List[int],
    *,
    acting_user: UserProfile
) -> None:
    ...

def do_send_subgroups_update_event(
    event_name: str,
    user_group: NamedUserGroup,
    subgroup_ids: List[int]
) -> None:
    ...

@transaction.atomic(savepoint=False)
def add_subgroups_to_user_group(
    user_group: NamedUserGroup,
    subgroups: List[UserGroup],
    *,
    acting_user: UserProfile
) -> None:
    ...

@transaction.atomic(savepoint=False)
def remove_subgroups_from_user_group(
    user_group: NamedUserGroup,
    subgroups: List[UserGroup],
    *,
    acting_user: UserProfile
) -> None:
    ...

@transaction.atomic(savepoint=False)
def do_deactivate_user_group(
    user_group: NamedUserGroup,
    *,
    acting_user: UserProfile
) -> None:
    ...

@transaction.atomic(savepoint=False)
def do_change_user_group_permission_setting(
    user_group: NamedUserGroup,
    setting_name: str,
    setting_value_group: Any,
    *,
    old_setting_api_value: Optional[Any] = None,
    acting_user: UserProfile
) -> None:
    ...
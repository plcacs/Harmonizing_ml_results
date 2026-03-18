```python
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, TypedDict
import django.db.utils
from django.db import transaction
from django.utils.timezone import datetime as datetime
from zerver.lib.exceptions import JsonableError
from zerver.lib.types import AnonymousSettingGroupDict
from zerver.models import GroupGroupMembership, NamedUserGroup, Realm, RealmAuditLog, UserGroup, UserGroupMembership, UserProfile
from zerver.models.groups import SystemGroups
from zerver.models.realm_audit_logs import AuditLogEventType

class MemberGroupUserDict(TypedDict):
    pass

def create_user_group_in_database(
    name: Any,
    members: Any,
    realm: Any,
    *,
    acting_user: Any,
    description: str = ...,
    group_settings_map: dict[Any, Any] = ...,
    is_system_group: bool = ...
) -> Any: ...

def update_users_in_full_members_system_group(
    realm: Any,
    affected_user_ids: list[Any] = ...,
    *,
    acting_user: Any
) -> None: ...

def promote_new_full_members() -> None: ...

def do_send_create_user_group_event(
    user_group: Any,
    members: Any,
    direct_subgroups: list[Any] = ...
) -> None: ...

def check_add_user_group(
    realm: Any,
    name: Any,
    initial_members: Any,
    description: str = ...,
    group_settings_map: dict[Any, Any] = ...,
    *,
    acting_user: Any
) -> Any: ...

def do_send_user_group_update_event(
    user_group: Any,
    data: Any
) -> None: ...

def do_update_user_group_name(
    user_group: Any,
    name: Any,
    *,
    acting_user: Any
) -> None: ...

def do_update_user_group_description(
    user_group: Any,
    description: Any,
    *,
    acting_user: Any
) -> None: ...

def do_send_user_group_members_update_event(
    event_name: Any,
    user_group: Any,
    user_ids: Any
) -> None: ...

def bulk_add_members_to_user_groups(
    user_groups: Any,
    user_profile_ids: Any,
    *,
    acting_user: Any
) -> None: ...

def bulk_remove_members_from_user_groups(
    user_groups: Any,
    user_profile_ids: Any,
    *,
    acting_user: Any
) -> None: ...

def do_send_subgroups_update_event(
    event_name: Any,
    user_group: Any,
    subgroup_ids: Any
) -> None: ...

def add_subgroups_to_user_group(
    user_group: Any,
    subgroups: Any,
    *,
    acting_user: Any
) -> None: ...

def remove_subgroups_from_user_group(
    user_group: Any,
    subgroups: Any,
    *,
    acting_user: Any
) -> None: ...

def do_deactivate_user_group(
    user_group: Any,
    *,
    acting_user: Any
) -> None: ...

def do_change_user_group_permission_setting(
    user_group: Any,
    setting_name: Any,
    setting_value_group: Any,
    *,
    old_setting_api_value: Any = ...,
    acting_user: Any
) -> None: ...
```
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, TypedDict

@dataclass
class GroupSettingChangeRequest:
    old: Any  # type: ignore

class UserGroupDict(TypedDict):
    pass

@dataclass
class LockedUserGroupContext:
    """User groups in this dataclass are guaranteeed to be locked until the
    end of the current transaction.

    supergroup: the user group to have subgroups added or removed;
    direct_subgroups: user groups that are recursively queried for subgroups;
    recursive_subgroups: include direct_subgroups and their descendants.
    """

direct_subgroups: List[NamedUserGroup]
recursive_subgroups: List[NamedUserGroup]
supergroup: NamedUserGroup

@contextmanager
def lock_subgroups_with_respect_to_supergroup(potential_subgroup_ids: List[int], potential_supergroup_id: int, acting_user: UserProfile, *, permission_setting: str, creating_group: bool) -> LockedUserGroupContext:
    ...

def has_user_group_access_for_subgroup(user_group: NamedUserGroup, user_profile: UserProfile, *, allow_deactivated: bool = False) -> bool:
    ...

def get_user_group_by_id_in_realm(user_group_id: int, realm: Realm, *, for_read: bool, for_setting: bool = False, allow_deactivated: bool = False) -> NamedUserGroup:
    ...

def access_user_group_to_read_membership(user_group_id: int, realm: Realm) -> NamedUserGroup:
    ...

def access_user_group_for_update(user_group_id: int, user_profile: UserProfile, *, permission_setting: str) -> NamedUserGroup:
    ...

def check_setting_configuration_for_system_groups(setting_group: NamedUserGroup, setting_name: str, permission_configuration: Any) -> None:
    ...

def get_group_setting_value_for_api(setting_value_group: Any) -> Any:
    ...

def get_setting_value_for_user_group_object(setting_value_group: Any, direct_members_dict: Dict[int, List[int]], direct_subgroups_dict: Dict[int, List[int]]) -> Any:
    ...

def user_groups_in_realm_serialized(realm: Realm, *, include_deactivated_groups: bool) -> List[Dict[str, Any]]:
    ...

def get_direct_user_groups(user_profile: UserProfile) -> List[NamedUserGroup]:
    ...

def get_user_group_direct_member_ids(user_group: NamedUserGroup) -> List[int]:
    ...

def get_user_group_direct_members(user_group: NamedUserGroup) -> List[UserProfile]:
    ...

def get_direct_memberships_of_users(user_group: NamedUserGroup, members: List[UserProfile], *, direct_member_only: bool = False) -> List[int]:
    ...

def get_subgroup_ids(user_group: NamedUserGroup, *, direct_subgroup_only: bool = False) -> List[int]:
    ...

def get_recursive_subgroups_for_groups(user_group_ids: List[int], realm: Realm) -> With:
    ...

def get_root_id_annotated_recursive_subgroups_for_groups(user_group_ids: List[int], realm_id: int) -> With:
    ...

def create_system_user_groups_for_realm(realm: Realm) -> Dict[str, NamedUserGroup]:
    ...

def get_system_user_group_for_user(user_profile: UserProfile) -> NamedUserGroup:
    ...

def get_server_supported_permission_settings() -> ServerSupportedPermissionSettings:
    ...

def parse_group_setting_value(setting_value: Any) -> Any:
    ...

def are_both_group_setting_values_equal(first_setting_value: Any, second_setting_value: Any) -> bool:
    ...

def validate_group_setting_value_change(current_setting_api_value: Any, new_setting_value: Any, expected_current_setting_value: Any) -> bool:
    ...

def get_group_setting_value_for_audit_log_data(setting_value: Any) -> Any:
    ...

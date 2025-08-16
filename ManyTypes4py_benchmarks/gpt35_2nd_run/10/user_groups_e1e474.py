from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, TypedDict
from django.db import connection, transaction
from django.db.models import F, Q, QuerySet
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django_cte import With
from psycopg2.sql import SQL, Literal
from zerver.lib.exceptions import CannotDeactivateGroupInUseError, JsonableError, PreviousSettingValueMismatchedError, SystemGroupRequiredError
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.types import AnonymousSettingGroupDict, GroupPermissionSetting, ServerSupportedPermissionSettings
from zerver.models import GroupGroupMembership, NamedUserGroup, Realm, RealmAuditLog, Stream, UserGroup, UserGroupMembership, UserProfile
from zerver.models.groups import SystemGroups
from zerver.models.realm_audit_logs import AuditLogEventType

@dataclass
class GroupSettingChangeRequest:
    old: Any = None

class UserGroupDict(TypedDict):
    pass

@dataclass
class LockedUserGroupContext:
    supergroup: Any
    direct_subgroups: Any
    recursive_subgroups: Any

def has_user_group_access_for_subgroup(user_group, user_profile, *, allow_deactivated: bool = False) -> bool:
    ...

def get_user_group_by_id_in_realm(user_group_id, realm, *, for_read: bool, for_setting: bool = False, allow_deactivated: bool = False) -> Any:
    ...

def access_user_group_to_read_membership(user_group_id, realm) -> Any:
    ...

def access_user_group_for_update(user_group_id, user_profile, *, permission_setting, allow_deactivated: bool = False) -> Any:
    ...

def access_user_group_for_deactivation(user_group_id, user_profile) -> Any:
    ...

@contextmanager
def lock_subgroups_with_respect_to_supergroup(potential_subgroup_ids, potential_supergroup_id, acting_user, *, permission_setting, creating_group: bool = False) -> Any:
    ...

def check_setting_configuration_for_system_groups(setting_group, setting_name, permission_configuration) -> None:
    ...

def update_or_create_user_group_for_setting(user_profile, direct_members, direct_subgroups, current_setting_value) -> Any:
    ...

def access_user_group_for_setting(setting_user_group, user_profile, *, setting_name, permission_configuration, current_setting_value=None) -> Any:
    ...

def check_user_group_name(group_name) -> str:
    ...

def get_group_setting_value_for_api(setting_value_group) -> Any:
    ...

def get_setting_value_for_user_group_object(setting_value_group, direct_members_dict, direct_subgroups_dict) -> Any:
    ...

def user_groups_in_realm_serialized(realm, *, include_deactivated_groups: bool) -> Any:
    ...

def get_direct_user_groups(user_profile) -> Any:
    ...

def get_user_group_direct_member_ids(user_group) -> Any:
    ...

def get_user_group_direct_members(user_group) -> Any:
    ...

def get_direct_memberships_of_users(user_group, members) -> Any:
    ...

def get_recursive_subgroups_union_for_groups(user_group_ids) -> Any:
    ...

def get_recursive_subgroups(user_group_id) -> Any:
    ...

def get_recursive_strict_subgroups(user_group) -> Any:
    ...

def get_recursive_group_members(user_group_id) -> Any:
    ...

def get_recursive_group_members_union_for_groups(user_group_ids) -> Any:
    ...

def get_recursive_membership_groups(user_profile) -> Any:
    ...

def user_has_permission_for_group_setting(user_group, user, setting_config, *, direct_member_only: bool = False) -> bool:
    ...

def is_user_in_group(user_group, user, *, direct_member_only: bool = False) -> bool:
    ...

def is_any_user_in_group(user_group, user_ids, *, direct_member_only: bool = False) -> bool:
    ...

def get_user_group_member_ids(user_group, *, direct_member_only: bool = False) -> Any:
    ...

def get_subgroup_ids(user_group, *, direct_subgroup_only: bool = False) -> Any:
    ...

def get_recursive_subgroups_for_groups(user_group_ids, realm) -> Any:
    ...

def get_root_id_annotated_recursive_subgroups_for_groups(user_group_ids, realm_id) -> Any:
    ...

def get_role_based_system_groups_dict(realm) -> Any:
    ...

def set_defaults_for_group_settings(user_group, group_settings_map, system_groups_name_dict) -> Any:
    ...

def bulk_create_system_user_groups(groups, realm) -> Any:
    ...

@transaction.atomic(savepoint=False)
def create_system_user_groups_for_realm(realm) -> Any:
    ...

def get_system_user_group_for_user(user_profile) -> Any:
    ...

def get_server_supported_permission_settings() -> Any:
    ...

def parse_group_setting_value(setting_value) -> Any:
    ...

def are_both_group_setting_values_equal(first_setting_value, second_setting_value) -> bool:
    ...

def validate_group_setting_value_change(current_setting_api_value, new_setting_value, expected_current_setting_value) -> bool:
    ...

def get_group_setting_value_for_audit_log_data(setting_value) -> Any:
    ...

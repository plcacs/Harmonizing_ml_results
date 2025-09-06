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
    supergroup: NamedUserGroup
    direct_subgroups: QuerySet
    recursive_subgroups: QuerySet

def func_acoiwghj(user_group: NamedUserGroup, user_profile: UserProfile, *, allow_deactivated: bool = False) -> bool:
    ...

def func_35hdvdls(user_group_id: int, realm: Realm, *, for_read: bool, for_setting: bool = False, allow_deactivated: bool = False) -> NamedUserGroup:
    ...

def func_80xq8zch(user_group_id: int, realm: Realm) -> NamedUserGroup:
    ...

def func_apt8j5iv(user_group_id: int, user_profile: UserProfile, *, permission_setting: str, allow_deactivated: bool = False) -> NamedUserGroup:
    ...

def func_fq19x782(user_group_id: int, user_profile: UserProfile) -> NamedUserGroup:
    ...

@contextmanager
def func_8dalgx6v(potential_subgroup_ids: List[int], potential_supergroup_id: int, acting_user: UserProfile, *, permission_setting: str, creating_group: bool = False) -> Generator[LockedUserGroupContext, None, None]:
    ...

def func_essaikr5(setting_group: NamedUserGroup, setting_name: str, permission_configuration: GroupPermissionSetting) -> UserGroup:
    ...

def func_9obtmtu9(user_profile: UserProfile, direct_members: List[int], direct_subgroups: List[int], current_setting_value: Any) -> UserGroup:
    ...

def func_cczlrf1t(setting_user_group: Union[int, AnonymousSettingGroupDict], user_profile: UserProfile, *, setting_name: str, permission_configuration: GroupPermissionSetting, current_setting_value: Any = None) -> UserGroup:
    ...

def func_jt8vvz1g(group_name: str) -> str:
    ...

def func_rfmuiusl(setting_value_group: Union[int, AnonymousSettingGroupDict]) -> int:
    ...

def func_zgbm7heh(setting_value_group: Union[int, AnonymousSettingGroupDict], direct_members_dict: dict, direct_subgroups_dict: dict) -> int:
    ...

def func_k1nxxzcj(realm: Realm, *, include_deactivated_groups: bool) -> List[dict]:
    ...

def func_gyp6emsf(user_profile: UserProfile) -> List[NamedUserGroup]:
    ...

def func_as4pp7uj(user_group: NamedUserGroup) -> List[int]:
    ...

def func_vax0ybqm(user_group: NamedUserGroup) -> QuerySet:
    ...

def func_ljcy7u2t(user_group: NamedUserGroup, members: List[int]) -> List[int]:
    ...

def func_nw8v8hut(user_group_ids: List[int]) -> QuerySet:
    ...

def func_9nyb67ia(user_group_id: int) -> QuerySet:
    ...

def func_dg87kkwb(user_group: NamedUserGroup) -> QuerySet:
    ...

def func_vieqau8j(user_group_id: int) -> QuerySet:
    ...

def func_uukv9us5(user_group_ids: List[int]) -> QuerySet:
    ...

def func_u56905te(user_profile: UserProfile) -> QuerySet:
    ...

def func_n21fwrpi(user_group: NamedUserGroup, user: UserProfile, setting_config: ServerSupportedPermissionSettings, *, direct_member_only: bool = False) -> bool:
    ...

def func_53mxi1ug(user_group: NamedUserGroup, user: UserProfile, *, direct_member_only: bool = False) -> bool:
    ...

def func_p865uv9y(user_group: NamedUserGroup, user_ids: List[int], *, direct_member_only: bool = False) -> bool:
    ...

def func_qvpxjkm7(user_group: NamedUserGroup, *, direct_member_only: bool = False) -> List[int]:
    ...

def func_naigeoxi(user_group: NamedUserGroup, *, direct_subgroup_only: bool = False) -> List[int]:
    ...

def func_e6fhflnz(user_group_ids: List[int], realm: Realm) -> QuerySet:
    ...

def func_eddb4le6(user_group_ids: List[int], realm_id: int) -> QuerySet:
    ...

def func_epzw5q1x(realm: Realm) -> dict:
    ...

def func_kawu5agi(user_group: NamedUserGroup, group_settings_map: dict, system_groups_name_dict: dict) -> NamedUserGroup:
    ...

def func_qj98xqyu(groups: List[dict], realm: Realm) -> None:
    ...

@transaction.atomic(savepoint=False)
def func_91bnr75r(realm: Realm) -> dict:
    ...

def func_sz8r2inb(user_profile: UserProfile) -> NamedUserGroup:
    ...

def func_5fbs4r06() -> ServerSupportedPermissionSettings:
    ...

def func_1p1iq62p(setting_value: Union[int, AnonymousSettingGroupDict]) -> int:
    ...

def func_f9c3d2qx(first_setting_value: Union[int, AnonymousSettingGroupDict], second_setting_value: Union[int, AnonymousSettingGroupDict]) -> bool:
    ...

def func_3ve3898r(current_setting_api_value: Union[int, AnonymousSettingGroupDict], new_setting_value: Union[int, AnonymousSettingGroupDict], expected_current_setting_value: Union[int, AnonymousSettingGroupDict]) -> bool:
    ...

def func_pd17r54t(setting_value: Union[int, AnonymousSettingGroupDict]) -> int:
    ...


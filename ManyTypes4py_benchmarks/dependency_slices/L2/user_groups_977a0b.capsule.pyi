from typing import Any

# === Third-party dependency: django.db ===
# Used symbols: transaction

# === Third-party dependency: django.db.utils ===
class IntegrityError(DatabaseError): ...

# === Third-party dependency: django.utils.timezone ===
def now() -> Any: ...

# === Third-party dependency: django.utils.translation ===
def gettext(message) -> Any: ...

# === Internal dependency: zerver.lib.exceptions ===
class JsonableError(Exception): ...

# === Internal dependency: zerver.lib.timestamp ===
def datetime_to_timestamp(dt: datetime) -> int: ...

# === Internal dependency: zerver.lib.types ===
class AnonymousSettingGroupDict: ...

# === Internal dependency: zerver.lib.user_groups ===
def get_group_setting_value_for_api(setting_value_group: UserGroup) -> int | AnonymousSettingGroupDict: ...
def get_role_based_system_groups_dict(realm: Realm) -> dict[str, NamedUserGroup]: ...
def set_defaults_for_group_settings(user_group: NamedUserGroup, group_settings_map: Mapping[str, UserGroup], system_groups_name_dict: dict[str, NamedUserGroup]) -> NamedUserGroup: ...
def get_group_setting_value_for_audit_log_data(setting_value: int | AnonymousSettingGroupDict) -> int | dict[str, list[int]]: ...

# === Internal dependency: zerver.models ===
# re-export: from zerver.models.groups import GroupGroupMembership as GroupGroupMembership
# re-export: from zerver.models.groups import NamedUserGroup as NamedUserGroup
# re-export: from zerver.models.groups import UserGroupMembership as UserGroupMembership
# re-export: from zerver.models.realm_audit_logs import RealmAuditLog as RealmAuditLog
# re-export: from zerver.models.realms import Realm as Realm
# re-export: from zerver.models.users import UserProfile as UserProfile

# === Internal dependency: zerver.models.groups ===
class SystemGroups: ...

# === Internal dependency: zerver.models.realm_audit_logs ===
class AuditLogEventType(IntEnum): ...

# === Internal dependency: zerver.models.users ===
def active_user_ids(realm_id: int) -> list[int]: ...

# === Unresolved dependency: zerver.tornado.django_api ===
# Used unresolved symbols: send_event_on_commit
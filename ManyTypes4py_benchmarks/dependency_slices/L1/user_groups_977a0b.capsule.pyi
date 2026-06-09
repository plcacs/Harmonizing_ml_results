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
def datetime_to_timestamp(dt): ...

# === Internal dependency: zerver.lib.types ===
class AnonymousSettingGroupDict: ...

# === Internal dependency: zerver.lib.user_groups ===
def get_group_setting_value_for_api(setting_value_group): ...
def get_role_based_system_groups_dict(realm): ...
def set_defaults_for_group_settings(user_group, group_settings_map, system_groups_name_dict): ...
def get_group_setting_value_for_audit_log_data(setting_value): ...

# === Internal dependency: zerver.models ===
from zerver.models.groups import GroupGroupMembership as GroupGroupMembership
from zerver.models.groups import NamedUserGroup as NamedUserGroup
from zerver.models.groups import UserGroupMembership as UserGroupMembership
from zerver.models.realm_audit_logs import RealmAuditLog as RealmAuditLog
from zerver.models.realms import Realm as Realm
from zerver.models.users import UserProfile as UserProfile

# === Internal dependency: zerver.models.groups ===
class SystemGroups: ...

# === Internal dependency: zerver.models.realm_audit_logs ===
class AuditLogEventType(IntEnum): ...

# === Internal dependency: zerver.models.users ===
def active_user_ids(realm_id): ...

# === Unresolved dependency: zerver.tornado.django_api ===
# Used unresolved symbols: send_event_on_commit
from typing import Any

# === Internal dependency: confirmation.models ===
def generate_key() -> str: ...
def create_confirmation_link(obj: ConfirmationObjT, confirmation_type: int, *, validity_in_minutes: int | None | Unset = ..., url_args: Mapping[str, str] = ..., no_associated_realm_object: bool = ...) -> str: ...
class Confirmation(models.Model): ...

# === Internal dependency: corporate.lib.stripe ===
class RealmBillingSession(BillingSession):
    def __init__(self, user: UserProfile | None = ..., realm: Realm | None = ..., *, support_session: bool = ...) -> None: ...

# === Internal dependency: corporate.models ===
class CustomerPlan(AbstractCustomerPlan):
    ...

# === Third-party dependency: django.conf ===
settings: LazySettings

# === Third-party dependency: django.db ===
# Used symbols: transaction

# === Third-party dependency: django.utils.timezone ===
def get_current_timezone_name() -> Any: ...
def now() -> Any: ...

# === Third-party dependency: django.utils.translation ===
def gettext(message) -> Any: ...

# === Internal dependency: zerver.actions.custom_profile_fields ===
def do_remove_realm_custom_profile_fields(realm: Realm) -> None: ...

# === Internal dependency: zerver.actions.message_delete ===
def do_delete_messages_by_sender(user: UserProfile) -> None: ...

# === Internal dependency: zerver.actions.user_groups ===
def update_users_in_full_members_system_group(realm: Realm, affected_user_ids: Sequence[int] = ..., *, acting_user: UserProfile | None) -> None: ...

# === Internal dependency: zerver.actions.user_settings ===
def do_delete_avatar_image(user: UserProfile, *, acting_user: UserProfile | None) -> None: ...

# === Internal dependency: zerver.lib.exceptions ===
class JsonableError(Exception): ...

# === Internal dependency: zerver.lib.message ===
def update_first_visible_message_id(realm: Realm) -> None: ...
def parse_message_time_limit_setting(value: int | str, special_values_map: Mapping[str, int | None], *, setting_name: str) -> int | None: ...

# === Internal dependency: zerver.lib.queue ===
def queue_json_publish_rollback_unsafe(queue_name: str, event: dict[str, Any], processor: Callable[[Any], None] | None = ...) -> None: ...

# === Internal dependency: zerver.lib.remote_server ===
def maybe_enqueue_audit_log_upload(realm: Realm) -> None: ...

# === Internal dependency: zerver.lib.retention ===
def move_messages_to_archive(message_ids: list[int], realm: Realm | None = ..., chunk_size: int = ...) -> None: ...

# === Internal dependency: zerver.lib.send_email ===
class FromAddress: ...
def send_email(template_prefix: str, to_user_ids: list[int] | None = ..., to_emails: list[str] | None = ..., from_name: str | None = ..., from_address: str | None = ..., reply_to_email: str | None = ..., language: str | None = ..., context: Mapping[str, Any] = ..., realm: Realm | None = ..., connection: BaseEmailBackend | None = ..., dry_run: bool = ..., request: HttpRequest | None = ...) -> None: ...
def send_email_to_admins(template_prefix: str, realm: Realm, from_name: str | None = ..., from_address: str | None = ..., language: str | None = ..., context: Mapping[str, Any] = ...) -> None: ...

# === Internal dependency: zerver.lib.sessions ===
def delete_realm_user_sessions(realm: Realm) -> None: ...

# === Internal dependency: zerver.lib.timestamp ===
def timestamp_to_datetime(timestamp: float) -> datetime: ...
def datetime_to_timestamp(dt: datetime) -> int: ...

# === Internal dependency: zerver.lib.timezone ===
def canonicalize_timezone(key: str) -> str: ...

# === Internal dependency: zerver.lib.upload ===
def delete_message_attachments(path_ids: list[str]) -> None: ...

# === Internal dependency: zerver.lib.user_counts ===
def realm_user_count_by_role(realm: Realm) -> dict[str, Any]: ...

# === Internal dependency: zerver.lib.user_groups ===
def get_group_setting_value_for_api(setting_value_group: UserGroup) -> int | AnonymousSettingGroupDict: ...
def get_group_setting_value_for_audit_log_data(setting_value: int | AnonymousSettingGroupDict) -> int | dict[str, list[int]]: ...

# === Internal dependency: zerver.lib.utils ===
def optional_bytes_to_mib(value: int | None) -> int | None: ...

# === Internal dependency: zerver.models ===
# re-export: from zerver.models.groups import NamedUserGroup as NamedUserGroup
# re-export: from zerver.models.messages import ArchivedAttachment as ArchivedAttachment
# re-export: from zerver.models.messages import Attachment as Attachment
# re-export: from zerver.models.messages import Message as Message
# re-export: from zerver.models.prereg_users import RealmReactivationStatus as RealmReactivationStatus
# re-export: from zerver.models.realm_audit_logs import RealmAuditLog as RealmAuditLog
# re-export: from zerver.models.realms import Realm as Realm
# re-export: from zerver.models.realms import RealmAuthenticationMethod as RealmAuthenticationMethod
# re-export: from zerver.models.recipients import Recipient as Recipient
# re-export: from zerver.models.scheduled_jobs import ScheduledEmail as ScheduledEmail
# re-export: from zerver.models.streams import Stream as Stream
# re-export: from zerver.models.streams import Subscription as Subscription
# re-export: from zerver.models.users import UserProfile as UserProfile

# === Internal dependency: zerver.models.groups ===
class SystemGroups: ...

# === Internal dependency: zerver.models.realm_audit_logs ===
class AuditLogEventType(IntEnum): ...

# === Internal dependency: zerver.models.realms ===
def get_realm(string_id: str) -> Realm: ...
def get_default_max_invites_for_realm_plan_type(plan_type: int) -> int | None: ...

# === Internal dependency: zerver.models.users ===
def active_user_ids(realm_id: int) -> list[int]: ...

# === Unresolved dependency: zerver.tornado.django_api ===
# Used unresolved symbols: send_event_on_commit

# === Unresolved dependency: zproject.backends ===
# Used unresolved symbols: AUTH_BACKEND_NAME_MAP
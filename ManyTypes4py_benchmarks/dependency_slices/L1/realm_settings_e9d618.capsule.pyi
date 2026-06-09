from typing import Any

# === Internal dependency: confirmation.models ===
def generate_key(): ...
def create_confirmation_link(obj, confirmation_type, *, validity_in_minutes=..., url_args=..., no_associated_realm_object=...): ...
class Confirmation(models.Model): ...

# === Internal dependency: corporate.lib.stripe ===
class RealmBillingSession(BillingSession):
    def __init__(self, user=..., realm=..., *, support_session=...): ...

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
def do_remove_realm_custom_profile_fields(realm): ...

# === Internal dependency: zerver.actions.message_delete ===
def do_delete_messages_by_sender(user): ...

# === Internal dependency: zerver.actions.user_groups ===
def update_users_in_full_members_system_group(realm, affected_user_ids=..., *, acting_user): ...

# === Internal dependency: zerver.actions.user_settings ===
def do_delete_avatar_image(user, *, acting_user): ...

# === Internal dependency: zerver.lib.exceptions ===
class JsonableError(Exception): ...

# === Internal dependency: zerver.lib.message ===
def update_first_visible_message_id(realm): ...
def parse_message_time_limit_setting(value, special_values_map, *, setting_name): ...

# === Internal dependency: zerver.lib.queue ===
def queue_json_publish_rollback_unsafe(queue_name, event, processor=...): ...

# === Internal dependency: zerver.lib.remote_server ===
def maybe_enqueue_audit_log_upload(realm): ...

# === Internal dependency: zerver.lib.retention ===
def move_messages_to_archive(message_ids, realm=..., chunk_size=...): ...

# === Internal dependency: zerver.lib.send_email ===
class FromAddress: ...
def send_email(template_prefix, to_user_ids=..., to_emails=..., from_name=..., from_address=..., reply_to_email=..., language=..., context=..., realm=..., connection=..., dry_run=..., request=...): ...
def send_email_to_admins(template_prefix, realm, from_name=..., from_address=..., language=..., context=...): ...

# === Internal dependency: zerver.lib.sessions ===
def delete_realm_user_sessions(realm): ...

# === Internal dependency: zerver.lib.timestamp ===
def timestamp_to_datetime(timestamp): ...
def datetime_to_timestamp(dt): ...

# === Internal dependency: zerver.lib.timezone ===
def canonicalize_timezone(key): ...

# === Internal dependency: zerver.lib.upload ===
def delete_message_attachments(path_ids): ...

# === Internal dependency: zerver.lib.user_counts ===
def realm_user_count_by_role(realm): ...

# === Internal dependency: zerver.lib.user_groups ===
def get_group_setting_value_for_api(setting_value_group): ...
def get_group_setting_value_for_audit_log_data(setting_value): ...

# === Internal dependency: zerver.lib.utils ===
def optional_bytes_to_mib(value): ...

# === Internal dependency: zerver.models ===
from zerver.models.groups import NamedUserGroup as NamedUserGroup
from zerver.models.messages import ArchivedAttachment as ArchivedAttachment
from zerver.models.messages import Attachment as Attachment
from zerver.models.messages import Message as Message
from zerver.models.prereg_users import RealmReactivationStatus as RealmReactivationStatus
from zerver.models.realm_audit_logs import RealmAuditLog as RealmAuditLog
from zerver.models.realms import Realm as Realm
from zerver.models.realms import RealmAuthenticationMethod as RealmAuthenticationMethod
from zerver.models.recipients import Recipient as Recipient
from zerver.models.scheduled_jobs import ScheduledEmail as ScheduledEmail
from zerver.models.streams import Stream as Stream
from zerver.models.streams import Subscription as Subscription
from zerver.models.users import UserProfile as UserProfile

# === Internal dependency: zerver.models.groups ===
class SystemGroups: ...

# === Internal dependency: zerver.models.realm_audit_logs ===
class AuditLogEventType(IntEnum): ...

# === Internal dependency: zerver.models.realms ===
def get_realm(string_id): ...
def get_default_max_invites_for_realm_plan_type(plan_type): ...

# === Internal dependency: zerver.models.users ===
def active_user_ids(realm_id): ...

# === Unresolved dependency: zerver.tornado.django_api ===
# Used unresolved symbols: send_event_on_commit

# === Unresolved dependency: zproject.backends ===
# Used unresolved symbols: AUTH_BACKEND_NAME_MAP
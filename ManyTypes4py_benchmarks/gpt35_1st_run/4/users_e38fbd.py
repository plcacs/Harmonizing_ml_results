import secrets
from collections import defaultdict
from email.headerregistry import Address
from typing import Any
from django.conf import settings
from django.contrib.auth.tokens import PasswordResetTokenGenerator, default_token_generator
from django.db import transaction
from django.db.models import Q
from django.http import HttpRequest
from django.urls import reverse
from django.utils.http import urlsafe_base64_encode
from django.utils.timezone import now as timezone_now
from django.utils.translation import get_language
from zerver.actions.user_groups import do_send_user_group_members_update_event, update_users_in_full_members_system_group
from zerver.lib.avatar import get_avatar_field
from zerver.lib.bot_config import ConfigError, get_bot_config, get_bot_configs, set_bot_config
from zerver.lib.cache import bot_dict_fields
from zerver.lib.create_user import create_user
from zerver.lib.invites import revoke_invites_generated_by_user
from zerver.lib.remote_server import maybe_enqueue_audit_log_upload
from zerver.lib.send_email import FromAddress, clear_scheduled_emails, send_email
from zerver.lib.sessions import delete_user_sessions
from zerver.lib.soft_deactivation import queue_soft_reactivation
from zerver.lib.stream_traffic import get_streams_traffic
from zerver.lib.streams import get_group_setting_value_dict_for_streams, get_streams_for_user, send_stream_deletion_event, stream_to_dict
from zerver.lib.subscription_info import bulk_get_subscriber_peer_info
from zerver.lib.types import AnonymousSettingGroupDict
from zerver.lib.user_counts import realm_user_count_by_role
from zerver.lib.user_groups import get_system_user_group_for_user
from zerver.lib.users import get_active_bots_owned_by_user, get_user_ids_who_can_access_user, get_users_involved_in_dms_with_target_users, user_access_restricted_in_realm
from zerver.models import GroupGroupMembership, Message, NamedUserGroup, Realm, RealmAuditLog, Recipient, Service, Stream, Subscription, UserGroup, UserGroupMembership, UserProfile
from zerver.models.bots import get_bot_services
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.realms import get_fake_email_domain
from zerver.models.users import active_non_guest_user_ids, active_user_ids, bot_owner_user_ids, get_bot_dicts_in_realm, get_user_profile_by_id
from zerver.tornado.django_api import send_event_on_commit

def do_delete_user(user_profile: UserProfile, *, acting_user: UserProfile) -> None:
    ...

def do_delete_user_preserving_messages(user_profile: UserProfile) -> None:
    ...

def change_user_is_active(user_profile: UserProfile, value: bool) -> None:
    ...

def send_group_update_event_for_anonymous_group_setting(setting_group: UserGroup, group_members_dict: dict, group_subgroups_dict: dict, named_group: NamedUserGroup, notify_user_ids: list) -> None:
    ...

def send_realm_update_event_for_anonymous_group_setting(setting_group: UserGroup, group_members_dict: dict, group_subgroups_dict: dict, notify_user_ids: list) -> None:
    ...

def send_update_events_for_anonymous_group_settings(setting_groups: list, realm: Realm, notify_user_ids: list) -> None:
    ...

def send_events_for_user_deactivation(user_profile: UserProfile) -> None:
    ...

def do_deactivate_user(user_profile: UserProfile, _cascade: bool = True, *, acting_user: UserProfile) -> None:
    ...

def send_stream_events_for_role_update(user_profile: UserProfile, old_accessible_streams: list) -> None:
    ...

def do_change_user_role(user_profile: UserProfile, value: int, *, acting_user: UserProfile) -> None:
    ...

def do_change_is_billing_admin(user_profile: UserProfile, value: bool) -> None:
    ...

def do_change_can_forge_sender(user_profile: UserProfile, value: bool) -> None:
    ...

def do_change_can_create_users(user_profile: UserProfile, value: bool) -> None:
    ...

def do_change_can_change_user_emails(user_profile: UserProfile, value: bool) -> None:
    ...

def do_update_outgoing_webhook_service(bot_profile: UserProfile, service_interface: str, service_payload_url: str) -> None:
    ...

def do_update_bot_config_data(bot_profile: UserProfile, config_data: dict) -> None:
    ...

def get_service_dicts_for_bot(user_profile_id: int) -> list:
    ...

def get_service_dicts_for_bots(bot_dicts: list, realm: Realm) -> dict:
    ...

def get_owned_bot_dicts(user_profile: UserProfile, include_all_realm_bots_if_admin: bool = True) -> list:
    ...

def generate_password_reset_url(user_profile: UserProfile, token_generator: PasswordResetTokenGenerator) -> str:
    ...

def do_send_password_reset_email(email: str, realm: Realm, user_profile: UserProfile, *, token_generator: PasswordResetTokenGenerator, request: HttpRequest) -> None:
    ...

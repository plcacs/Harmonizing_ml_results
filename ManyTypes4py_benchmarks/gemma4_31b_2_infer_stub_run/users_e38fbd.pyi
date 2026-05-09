import secrets
from collections import defaultdict
from email.headerregistry import Address
from typing import Any, Dict, List, Optional, Set, Union, Iterable
from django.http import HttpRequest
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from zerver.models import (
    GroupGroupMembership,
    Message,
    NamedUserGroup,
    Realm,
    RealmAuditLog,
    Recipient,
    Service,
    Stream,
    Subscription,
    UserGroup,
    UserGroupMembership,
    UserProfile,
)
from zerver.lib.types import AnonymousSettingGroupDict

def do_delete_user(user_profile: UserProfile, *, acting_user: UserProfile) -> None: ...

def do_delete_user_preserving_messages(user_profile: UserProfile) -> None: ...

def change_user_is_active(user_profile: UserProfile, value: bool) -> None: ...

def send_group_update_event_for_anonymous_group_setting(
    setting_group: UserGroup,
    group_members_dict: Dict[int, List[int]],
    group_subgroups_dict: Dict[int, List[int]],
    named_group: NamedUserGroup,
    notify_user_ids: List[int],
) -> None: ...

def send_realm_update_event_for_anonymous_group_setting(
    setting_group: UserGroup,
    group_members_dict: Dict[int, List[int]],
    group_subgroups_dict: Dict[int, List[int]],
    notify_user_ids: List[int],
) -> None: ...

def send_update_events_for_anonymous_group_settings(
    setting_groups: Iterable[UserGroup],
    realm: Realm,
    notify_user_ids: List[int],
) -> None: ...

def send_events_for_user_deactivation(user_profile: UserProfile) -> None: ...

def do_deactivate_user(user_profile: UserProfile, _cascade: bool = True, *, acting_user: UserProfile) -> None: ...

def send_stream_events_for_role_update(user_profile: UserProfile, old_accessible_streams: Iterable[Stream]) -> None: ...

def do_change_user_role(user_profile: UserProfile, value: str, *, acting_user: UserProfile) -> None: ...

def do_change_is_billing_admin(user_profile: UserProfile, value: bool) -> None: ...

def do_change_can_forge_sender(user_profile: UserProfile, value: bool) -> None: ...

def do_change_can_create_users(user_profile: UserProfile, value: bool) -> None: ...

def do_change_can_change_user_emails(user_profile: UserProfile, value: bool) -> None: ...

def do_update_outgoing_webhook_service(
    bot_profile: UserProfile,
    service_interface: str,
    service_payload_url: str,
) -> None: ...

def do_update_bot_config_data(bot_profile: UserProfile, config_data: Dict[str, Any]) -> None: ...

def get_service_dicts_for_bot(user_profile_id: int) -> List[Dict[str, Any]]: ...

def get_service_dicts_for_bots(bot_dicts: Iterable[Dict[str, Any]], realm: Realm) -> Dict[int, List[Dict[str, Any]]]: ...

def get_owned_bot_dicts(user_profile: UserProfile, include_all_realm_bots_if_admin: bool = True) -> List[Dict[str, Any]]: ...

def generate_password_reset_url(user_profile: UserProfile, token_generator: PasswordResetTokenGenerator) -> str: ...

def do_send_password_reset_email(
    email: str,
    realm: Realm,
    user_profile: Optional[UserProfile],
    *,
    token_generator: PasswordResetTokenGenerator = ...,
    request: Optional[HttpRequest] = None,
) -> None: ...
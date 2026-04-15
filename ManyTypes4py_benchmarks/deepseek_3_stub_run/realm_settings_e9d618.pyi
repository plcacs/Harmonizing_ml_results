import datetime
from email.headerregistry import Address
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from confirmation.models import Confirmation
from django.conf import settings
from django.db import transaction
from django.utils.timezone import datetime as datetime_type
from zerver.lib.exceptions import JsonableError
from zerver.lib.types import AnonymousSettingGroupDict
from zerver.models import (
    ArchivedAttachment,
    Attachment,
    Message,
    NamedUserGroup,
    Realm,
    RealmAuditLog,
    RealmReactivationStatus,
    RealmUserDefault,
    Stream,
    Subscription,
    UserGroup,
    UserProfile,
)
from zerver.models.groups import SystemGroups
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.tornado.django_api import send_event_on_commit

RealmDeactivationReasonType = Literal[
    "owner_request",
    "tos_violation",
    "inactivity",
    "self_hosting_migration",
    "subdomain_change",
]


@transaction.atomic(savepoint=False)
def do_set_realm_property(
    realm: Realm,
    name: str,
    value: Any,
    *,
    acting_user: Optional[UserProfile],
) -> None: ...


@transaction.atomic(durable=True)
def do_set_push_notifications_enabled_end_timestamp(
    realm: Realm,
    value: Optional[int],
    *,
    acting_user: Optional[UserProfile],
) -> None: ...


@transaction.atomic(savepoint=False)
def do_change_realm_permission_group_setting(
    realm: Realm,
    setting_name: str,
    user_group: UserGroup,
    old_setting_api_value: Optional[AnonymousSettingGroupDict] = None,
    *,
    acting_user: Optional[UserProfile],
) -> None: ...


def parse_and_set_setting_value_if_required(
    realm: Realm,
    setting_name: str,
    value: Any,
    *,
    acting_user: Optional[UserProfile],
) -> Tuple[Optional[int], bool]: ...


def get_realm_authentication_methods_for_page_params_api(
    realm: Realm,
    authentication_methods: Dict[str, bool],
) -> Dict[str, Dict[str, Union[bool, str]]]: ...


def validate_authentication_methods_dict_from_api(
    realm: Realm,
    authentication_methods: Dict[str, bool],
) -> None: ...


def validate_plan_for_authentication_methods(
    realm: Realm,
    authentication_methods: Dict[str, bool],
) -> None: ...


@transaction.atomic(savepoint=False)
def do_set_realm_authentication_methods(
    realm: Realm,
    authentication_methods: Dict[str, bool],
    *,
    acting_user: Optional[UserProfile],
) -> None: ...


def do_set_realm_stream(
    realm: Realm,
    field: str,
    stream: Optional[Stream],
    stream_id: Optional[int],
    *,
    acting_user: Optional[UserProfile],
) -> None: ...


def do_set_realm_moderation_request_channel(
    realm: Realm,
    stream: Optional[Stream],
    stream_id: Optional[int],
    *,
    acting_user: Optional[UserProfile],
) -> None: ...


def do_set_realm_new_stream_announcements_stream(
    realm: Realm,
    stream: Optional[Stream],
    stream_id: Optional[int],
    *,
    acting_user: Optional[UserProfile],
) -> None: ...


def do_set_realm_signup_announcements_stream(
    realm: Realm,
    stream: Optional[Stream],
    stream_id: Optional[int],
    *,
    acting_user: Optional[UserProfile],
) -> None: ...


def do_set_realm_zulip_update_announcements_stream(
    realm: Realm,
    stream: Optional[Stream],
    stream_id: Optional[int],
    *,
    acting_user: Optional[UserProfile],
) -> None: ...


@transaction.atomic(durable=True)
def do_set_realm_user_default_setting(
    realm_user_default: RealmUserDefault,
    name: str,
    value: Any,
    *,
    acting_user: Optional[UserProfile],
) -> None: ...


def do_deactivate_realm(
    realm: Realm,
    *,
    acting_user: Optional[UserProfile],
    deactivation_reason: RealmDeactivationReasonType,
    deletion_delay_days: Optional[int] = None,
    email_owners: bool,
) -> None: ...


def do_reactivate_realm(realm: Realm) -> None: ...


def do_add_deactivated_redirect(realm: Realm, redirect_url: str) -> None: ...


def do_delete_all_realm_attachments(realm: Realm, *, batch_size: int = 1000) -> None: ...


@transaction.atomic(durable=True)
def do_scrub_realm(realm: Realm, *, acting_user: Optional[UserProfile]) -> None: ...


def scrub_deactivated_realm(realm_to_scrub: Realm) -> None: ...


def clean_deactivated_realm_data() -> None: ...


@transaction.atomic(durable=True)
def do_change_realm_org_type(
    realm: Realm,
    org_type: int,
    acting_user: Optional[UserProfile],
) -> None: ...


@transaction.atomic(durable=True)
def do_change_realm_max_invites(
    realm: Realm,
    max_invites: int,
    acting_user: Optional[UserProfile],
) -> None: ...


@transaction.atomic(savepoint=False)
def do_change_realm_plan_type(
    realm: Realm,
    plan_type: int,
    *,
    acting_user: Optional[UserProfile],
) -> None: ...


def do_send_realm_reactivation_email(
    realm: Realm,
    *,
    acting_user: Optional[UserProfile],
) -> None: ...


def do_send_realm_deactivation_email(
    realm: Realm,
    acting_user: Optional[UserProfile],
    deletion_delay_days: Optional[int],
) -> None: ...
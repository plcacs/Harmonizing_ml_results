import datetime
import logging
import zoneinfo
from email.headerregistry import Address
from typing import Any, Dict, List, Literal, Optional, Tuple
from django.conf import settings
from django.db import transaction
from django.utils.timezone import (
    get_current_timezone_name as timezone_get_current_timezone_name,
    now as timezone_now,
)
from django.utils.translation import gettext as _
from confirmation.models import Confirmation, create_confirmation_link, generate_key
from zerver.actions.custom_profile_fields import do_remove_realm_custom_profile_fields
from zerver.actions.message_delete import do_delete_messages_by_sender
from zerver.actions.user_groups import update_users_in_full_members_system_group
from zerver.actions.user_settings import do_delete_avatar_image
from zerver.lib.exceptions import JsonableError
from zerver.lib.message import (
    parse_message_time_limit_setting,
    update_first_visible_message_id,
)
from zerver.lib.queue import queue_json_publish_rollback_unsafe
from zerver.lib.retention import move_messages_to_archive
from zerver.lib.send_email import FromAddress, send_email, send_email_to_admins
from zerver.lib.sessions import delete_realm_user_sessions
from zerver.lib.timestamp import datetime_to_timestamp, timestamp_to_datetime
from zerver.lib.timezone import canonicalize_timezone
from zerver.lib.types import AnonymousSettingGroupDict
from zerver.lib.upload import delete_message_attachments
from zerver.lib.user_counts import realm_user_count_by_role
from zerver.lib.user_groups import (
    get_group_setting_value_for_api,
    get_group_setting_value_for_audit_log_data,
)
from zerver.lib.utils import optional_bytes_to_mib
from zerver.models import (
    ArchivedAttachment,
    Attachment,
    Message,
    NamedUserGroup,
    Realm,
    RealmAuditLog,
    RealmAuthenticationMethod,
    RealmReactivationStatus,
    RealmUserDefault,
    Recipient,
    ScheduledEmail,
    Stream,
    Subscription,
    UserGroup,
    UserProfile,
)
from zerver.models.groups import SystemGroups
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.realms import (
    get_default_max_invites_for_realm_plan_type,
    get_realm,
)
from zerver.models.users import active_user_ids
from zerver.tornado.django_api import send_event_on_commit


@transaction.atomic(savepoint=False)
def do_set_realm_property(
    realm: Realm, name: str, value: Any, *, acting_user: UserProfile
) -> None:
    """Takes in a realm object, the name of an attribute to update, the
    value to update and the user who initiated the update.
    """
    property_type = Realm.property_types[name]
    assert isinstance(
        value, property_type
    ), f'Cannot update {name}: {value} is not an instance of {property_type}'
    old_value = getattr(realm, name)
    if old_value == value:
        return
    setattr(realm, name, value)
    realm.save(update_fields=[name])
    event: Dict[str, Any] = dict(type="realm", op="update", property=name, value=value)
    message_edit_settings = [
        "allow_message_editing",
        "message_content_edit_limit_seconds",
    ]
    if name in message_edit_settings:
        event = dict(
            type="realm", op="update_dict", property="default", data={name: value}
        )
    send_event_on_commit(realm, event, active_user_ids(realm.id))
    event_time = timezone_now()
    RealmAuditLog.objects.create(
        realm=realm,
        event_type=AuditLogEventType.REALM_PROPERTY_CHANGED,
        event_time=event_time,
        acting_user=acting_user,
        extra_data={
            RealmAuditLog.OLD_VALUE: old_value,
            RealmAuditLog.NEW_VALUE: value,
            "property": name,
        },
    )
    if name == "waiting_period_threshold":
        update_users_in_full_members_system_group(
            realm, acting_user=acting_user
        )


@transaction.atomic(durable=True)
def do_set_push_notifications_enabled_end_timestamp(
    realm: Realm, value: Optional[int], *, acting_user: UserProfile
) -> None:
    name = "push_notifications_enabled_end_timestamp"
    old_timestamp: Optional[int] = None
    old_datetime: Optional[datetime.datetime] = getattr(realm, name)
    if old_datetime is not None:
        old_timestamp = datetime_to_timestamp(old_datetime)
    if old_timestamp == value:
        return
    new_datetime: Optional[datetime.datetime] = None
    if value is not None:
        new_datetime = timestamp_to_datetime(value)
    setattr(realm, name, new_datetime)
    realm.save(update_fields=[name])
    event_time = timezone_now()
    RealmAuditLog.objects.create(
        realm=realm,
        event_type=AuditLogEventType.REALM_PROPERTY_CHANGED,
        event_time=event_time,
        acting_user=acting_user,
        extra_data={
            RealmAuditLog.OLD_VALUE: old_timestamp,
            RealmAuditLog.NEW_VALUE: value,
            "property": name,
        },
    )
    event: Dict[str, Any] = dict(type="realm", op="update", property=name, value=value)
    send_event_on_commit(realm, event, active_user_ids(realm.id))


@transaction.atomic(savepoint=False)
def do_change_realm_permission_group_setting(
    realm: Realm,
    setting_name: str,
    user_group: NamedUserGroup,
    old_setting_api_value: Optional[Any] = None,
    *,
    acting_user: UserProfile,
) -> None:
    """Takes in a realm object, the name of an attribute to update, the
    user_group to update and the user who initiated the update.
    """
    assert setting_name in Realm.REALM_PERMISSION_GROUP_SETTINGS
    old_value: Any = getattr(realm, setting_name)
    setattr(realm, setting_name, user_group)
    realm.save(update_fields=[setting_name])
    if old_setting_api_value is None:
        old_setting_api_value = get_group_setting_value_for_api(old_value)
    new_setting_api_value = get_group_setting_value_for_api(user_group)
    if not hasattr(old_value, "named_user_group") and hasattr(user_group, "named_user_group"):
        old_value.delete()
    event: Dict[str, Any] = dict(
        type="realm", op="update_dict", property="default", data={setting_name: new_setting_api_value}
    )
    send_event_on_commit(realm, event, active_user_ids(realm.id))
    event_time = timezone_now()
    RealmAuditLog.objects.create(
        realm=realm,
        event_type=AuditLogEventType.REALM_PROPERTY_CHANGED,
        event_time=event_time,
        acting_user=acting_user,
        extra_data={
            RealmAuditLog.OLD_VALUE: get_group_setting_value_for_audit_log_data(
                old_setting_api_value
            ),
            RealmAuditLog.NEW_VALUE: get_group_setting_value_for_audit_log_data(
                new_setting_api_value
            ),
            "property": setting_name,
        },
    )


def parse_and_set_setting_value_if_required(
    realm: Realm, setting_name: str, value: Any, *, acting_user: UserProfile
) -> Tuple[Optional[Any], bool]:
    parsed_value = parse_message_time_limit_setting(
        value, Realm.MESSAGE_TIME_LIMIT_SETTING_SPECIAL_VALUES_MAP, setting_name=setting_name
    )
    setting_value_changed = False
    if parsed_value is None and getattr(realm, setting_name) is not None:
        do_set_realm_property(realm, setting_name, parsed_value, acting_user=acting_user)
        setting_value_changed = True
    return (parsed_value, setting_value_changed)


def get_realm_authentication_methods_for_page_params_api(
    realm: Realm, authentication_methods: Dict[str, bool]
) -> Dict[str, Dict[str, Any]]:
    from zproject.backends import AUTH_BACKEND_NAME_MAP

    result_dict: Dict[str, Dict[str, Any]] = {
        backend_name: {"enabled": enabled, "available": True}
        for backend_name, enabled in authentication_methods.items()
    }
    if not settings.BILLING_ENABLED:
        return result_dict
    from corporate.models import CustomerPlan

    for backend_name, backend_result in result_dict.items():
        available_for = AUTH_BACKEND_NAME_MAP[backend_name].available_for_cloud_plans
        if available_for is not None and realm.plan_type not in available_for:
            backend_result["available"] = False
            required_upgrade_plan_number = min(
                set(available_for).intersection({Realm.PLAN_TYPE_STANDARD, Realm.PLAN_TYPE_PLUS})
            )
            if required_upgrade_plan_number == Realm.PLAN_TYPE_STANDARD:
                required_upgrade_plan_name = CustomerPlan.name_from_tier(
                    CustomerPlan.TIER_CLOUD_STANDARD
                )
            else:
                assert required_upgrade_plan_number == Realm.PLAN_TYPE_PLUS
                required_upgrade_plan_name = CustomerPlan.name_from_tier(
                    CustomerPlan.TIER_CLOUD_PLUS
                )
            backend_result["unavailable_reason"] = _(
                "You need to upgrade to the {required_upgrade_plan_name} plan to use this authentication method."
            ).format(required_upgrade_plan_name=required_upgrade_plan_name)
        else:
            backend_result["available"] = True
    return result_dict


def validate_authentication_methods_dict_from_api(
    realm: Realm, authentication_methods: Dict[str, bool]
) -> None:
    current_authentication_methods = realm.authentication_methods_dict()
    for name in authentication_methods:
        if name not in current_authentication_methods:
            raise JsonableError(
                _(
                    "Invalid authentication method: {name}. Valid methods are: {methods}"
                ).format(name=name, methods=sorted(current_authentication_methods.keys()))
            )
    if settings.BILLING_ENABLED:
        validate_plan_for_authentication_methods(realm, authentication_methods)


def validate_plan_for_authentication_methods(
    realm: Realm, authentication_methods: Dict[str, bool]
) -> None:
    from zproject.backends import AUTH_BACKEND_NAME_MAP

    old_authentication_methods = realm.authentication_methods_dict()
    newly_enabled_authentication_methods: set[str] = {
        name
        for name, enabled in authentication_methods.items()
        if enabled and not old_authentication_methods.get(name, False)
    }
    for name in newly_enabled_authentication_methods:
        available_for = AUTH_BACKEND_NAME_MAP[name].available_for_cloud_plans
        if available_for is not None and realm.plan_type not in available_for:
            raise JsonableError(
                _(
                    "Authentication method {name} is not available on your current plan."
                ).format(name=name)
            )


@transaction.atomic(savepoint=False)
def do_set_realm_authentication_methods(
    realm: Realm, authentication_methods: Dict[str, bool], *, acting_user: UserProfile
) -> None:
    old_value = realm.authentication_methods_dict()
    for key, value in authentication_methods.items():
        if value:
            RealmAuthenticationMethod.objects.get_or_create(realm=realm, name=key)
        else:
            RealmAuthenticationMethod.objects.filter(realm=realm, name=key).delete()
    updated_value = realm.authentication_methods_dict()
    RealmAuditLog.objects.create(
        realm=realm,
        event_type=AuditLogEventType.REALM_PROPERTY_CHANGED,
        event_time=timezone_now(),
        acting_user=acting_user,
        extra_data={
            RealmAuditLog.OLD_VALUE: old_value,
            RealmAuditLog.NEW_VALUE: updated_value,
            "property": "authentication_methods",
        },
    )
    event_data: Dict[str, Any] = dict(
        authentication_methods=get_realm_authentication_methods_for_page_params_api(
            realm, updated_value
        )
    )
    event: Dict[str, Any] = dict(
        type="realm", op="update_dict", property="default", data=event_data
    )
    send_event_on_commit(realm, event, active_user_ids(realm.id))


def do_set_realm_stream(
    realm: Realm,
    field: str,
    stream: Optional[Stream],
    stream_id: Optional[int],
    *,
    acting_user: UserProfile,
) -> None:
    if field == "moderation_request_channel":
        old_value = realm.moderation_request_channel_id
        realm.moderation_request_channel = stream
        property = "moderation_request_channel_id"
    elif field == "new_stream_announcements_stream":
        old_value = realm.new_stream_announcements_stream_id
        realm.new_stream_announcements_stream = stream
        property = "new_stream_announcements_stream_id"
    elif field == "signup_announcements_stream":
        old_value = realm.signup_announcements_stream_id
        realm.signup_announcements_stream = stream
        property = "signup_announcements_stream_id"
    elif field == "zulip_update_announcements_stream":
        old_value = realm.zulip_update_announcements_stream_id
        realm.zulip_update_announcements_stream = stream
        property = "zulip_update_announcements_stream_id"
    else:
        raise AssertionError("Invalid realm stream field.")
    with transaction.atomic(durable=True):
        realm.save(update_fields=[field])
        event_time = timezone_now()
        RealmAuditLog.objects.create(
            realm=realm,
            event_type=AuditLogEventType.REALM_PROPERTY_CHANGED,
            event_time=event_time,
            acting_user=acting_user,
            extra_data={
                RealmAuditLog.OLD_VALUE: old_value,
                RealmAuditLog.NEW_VALUE: stream_id,
                "property": field,
            },
        )
        event: Dict[str, Any] = dict(type="realm", op="update", property=property, value=stream_id)
        send_event_on_commit(realm, event, active_user_ids(realm.id))


def do_set_realm_moderation_request_channel(
    realm: Realm, stream: Optional[Stream], stream_id: Optional[int], *, acting_user: UserProfile
) -> None:
    if stream is not None and stream.is_public():
        raise JsonableError(_("Moderation request channel must be private."))
    do_set_realm_stream(
        realm, "moderation_request_channel", stream, stream_id, acting_user=acting_user
    )


def do_set_realm_new_stream_announcements_stream(
    realm: Realm, stream: Optional[Stream], stream_id: Optional[int], *, acting_user: UserProfile
) -> None:
    do_set_realm_stream(
        realm, "new_stream_announcements_stream", stream, stream_id, acting_user=acting_user
    )


def do_set_realm_signup_announcements_stream(
    realm: Realm, stream: Optional[Stream], stream_id: Optional[int], *, acting_user: UserProfile
) -> None:
    do_set_realm_stream(
        realm, "signup_announcements_stream", stream, stream_id, acting_user=acting_user
    )


def do_set_realm_zulip_update_announcements_stream(
    realm: Realm, stream: Optional[Stream], stream_id: Optional[int], *, acting_user: UserProfile
) -> None:
    do_set_realm_stream(
        realm,
        "zulip_update_announcements_stream",
        stream,
        stream_id,
        acting_user=acting_user,
    )


@transaction.atomic(durable=True)
def do_set_realm_user_default_setting(
    realm_user_default: RealmUserDefault,
    name: str,
    value: Any,
    *,
    acting_user: UserProfile,
) -> None:
    old_value = getattr(realm_user_default, name)
    realm = realm_user_default.realm
    event_time = timezone_now()
    setattr(realm_user_default, name, value)
    realm_user_default.save(update_fields=[name])
    RealmAuditLog.objects.create(
        realm=realm,
        event_type=AuditLogEventType.REALM_DEFAULT_USER_SETTINGS_CHANGED,
        event_time=event_time,
        acting_user=acting_user,
        extra_data={
            RealmAuditLog.OLD_VALUE: old_value,
            RealmAuditLog.NEW_VALUE: value,
            "property": name,
        },
    )
    event: Dict[str, Any] = dict(
        type="realm_user_settings_defaults", op="update", property=name, value=value
    )
    send_event_on_commit(realm, event, active_user_ids(realm.id))


RealmDeactivationReasonType = Literal[
    "owner_request",
    "tos_violation",
    "inactivity",
    "self_hosting_migration",
    "subdomain_change",
]


def do_deactivate_realm(
    realm: Realm,
    *,
    acting_user: UserProfile,
    deactivation_reason: RealmDeactivationReasonType,
    deletion_delay_days: Optional[int] = None,
    email_owners: bool,
) -> None:
    """
    Deactivate this realm. Do NOT deactivate the users -- we need to be able to
    tell the difference between users that were intentionally deactivated,
    e.g. by a realm admin, and users who can't currently use Zulip because their
    realm has been deactivated.
    """
    if realm.deactivated:
        return
    if settings.BILLING_ENABLED:
        from corporate.lib.stripe import RealmBillingSession
    with transaction.atomic(durable=True):
        realm.deactivated = True
        if deletion_delay_days is None:
            realm.save(update_fields=["deactivated"])
        else:
            realm.scheduled_deletion_date = timezone_now() + datetime.timedelta(
                days=deletion_delay_days
            )
            realm.save(update_fields=["scheduled_deletion_date", "deactivated"])
        if settings.BILLING_ENABLED:
            billing_session = RealmBillingSession(user=acting_user, realm=realm)
            billing_session.downgrade_now_without_creating_additional_invoices()
        event_time = timezone_now()
        RealmAuditLog.objects.create(
            realm=realm,
            event_type=AuditLogEventType.REALM_DEACTIVATED,
            event_time=event_time,
            acting_user=acting_user,
            extra_data={
                RealmAuditLog.ROLE_COUNT: realm_user_count_by_role(realm),
                "deactivation_reason": deactivation_reason,
            },
        )
        from zerver.lib.remote_server import maybe_enqueue_audit_log_upload

        maybe_enqueue_audit_log_upload(realm)
        ScheduledEmail.objects.filter(realm=realm).delete()
        event: Dict[str, Any] = dict(type="realm", op="deactivated", realm_id=realm.id)
        send_event_on_commit(realm, event, active_user_ids(realm.id))
        if deletion_delay_days == 0:
            event = {"type": "scrub_deactivated_realm", "realm_id": realm.id}
            queue_json_publish_rollback_unsafe("deferred_work", event)
    delete_realm_user_sessions(realm)
    if email_owners:
        do_send_realm_deactivation_email(realm, acting_user, deletion_delay_days)


def do_reactivate_realm(realm: Realm) -> None:
    if not realm.deactivated:
        logging.warning(
            "Realm %s cannot be reactivated because it is already active.", realm.id
        )
        return
    realm.deactivated = False
    realm.scheduled_deletion_date = None
    with transaction.atomic(durable=True):
        realm.save(update_fields=["deactivated", "scheduled_deletion_date"])
        event_time = timezone_now()
        RealmAuditLog.objects.create(
            acting_user=None,
            realm=realm,
            event_type=AuditLogEventType.REALM_REACTIVATED,
            event_time=event_time,
            extra_data={
                RealmAuditLog.ROLE_COUNT: realm_user_count_by_role(realm)
            },
        )
        from zerver.lib.remote_server import maybe_enqueue_audit_log_upload

        maybe_enqueue_audit_log_upload(realm)


def do_add_deactivated_redirect(realm: Realm, redirect_url: str) -> None:
    realm.deactivated_redirect = redirect_url
    realm.save(update_fields=["deactivated_redirect"])


def do_delete_all_realm_attachments(realm: Realm, *, batch_size: int = 1000) -> None:
    for obj_class in (Attachment, ArchivedAttachment):
        last_id = 0
        while True:
            to_delete: List[Tuple[int, int]] = list(
                obj_class._default_manager.filter(realm_id=realm.id, pk__gt=last_id)
                .order_by("pk")
                .values_list("pk", "path_id")[:batch_size]
            )
            if len(to_delete) > 0:
                delete_message_attachments([row[1] for row in to_delete])
                last_id = to_delete[-1][0]
            if len(to_delete) < batch_size:
                break
        obj_class._default_manager.filter(realm=realm).delete()


@transaction.atomic(durable=True)
def do_scrub_realm(
    realm: Realm, *, acting_user: Optional[UserProfile]
) -> None:
    if settings.BILLING_ENABLED:
        from corporate.lib.stripe import RealmBillingSession

        billing_session = RealmBillingSession(user=acting_user, realm=realm)
        billing_session.downgrade_now_without_creating_additional_invoices()
    users: List[UserProfile] = list(UserProfile.objects.filter(realm=realm))
    for user in users:
        do_delete_messages_by_sender(user)
        do_delete_avatar_image(user, acting_user=acting_user)
        user.full_name = f"Scrubbed {generate_key()[:15]}"
        scrubbed_email = Address(
            username=f"scrubbed-{generate_key()[:15]}", domain=realm.host
        ).addr_spec
        user.email = scrubbed_email
        user.delivery_email = scrubbed_email
        user.save(update_fields=["full_name", "email", "delivery_email"])
    internal_realm = get_realm(settings.SYSTEM_BOT_REALM)
    all_recipient_ids_in_realm: List[int] = list(
        Stream.objects.filter(realm=realm)
        .values_list("recipient_id", flat=True)
    ) + list(
        UserProfile.objects.filter(realm=realm)
        .values_list("recipient_id", flat=True)
    ) + list(
        Subscription.objects.filter(
            recipient__type=Recipient.DIRECT_MESSAGE_GROUP, user_profile__realm=realm
        )
        .values_list("recipient_id", flat=True)
    )
    cross_realm_bot_message_ids: List[int] = list(
        Message.objects.filter(
            sender__realm=internal_realm, recipient_id__in=all_recipient_ids_in_realm, realm=realm
        )
        .values_list("id", flat=True)
    )
    move_messages_to_archive(cross_realm_bot_message_ids)
    do_remove_realm_custom_profile_fields(realm)
    do_delete_all_realm_attachments(realm)
    RealmAuditLog.objects.create(
        realm=realm,
        event_time=timezone_now(),
        acting_user=acting_user,
        event_type=AuditLogEventType.REALM_SCRUBBED,
    )
    realm.scheduled_deletion_date = None
    realm.save()


def scrub_deactivated_realm(realm_to_scrub: Realm) -> None:
    if realm_to_scrub.scheduled_deletion_date is not None and realm_to_scrub.scheduled_deletion_date <= timezone_now():
        assert realm_to_scrub.deactivated, "Non-deactivated realm unexpectedly scheduled for deletion."
        do_scrub_realm(realm_to_scrub, acting_user=None)
        logging.info("Scrubbed realm %s", realm_to_scrub.id)


def clean_deactivated_realm_data() -> None:
    realms_to_scrub: List[Realm] = list(
        Realm.objects.filter(deactivated=True, scheduled_deletion_date__lte=timezone_now())
    )
    for realm in realms_to_scrub:
        scrub_deactivated_realm(realm)


@transaction.atomic(durable=True)
def do_change_realm_org_type(
    realm: Realm, org_type: str, acting_user: UserProfile
) -> None:
    old_value = realm.org_type
    realm.org_type = org_type
    realm.save(update_fields=["org_type"])
    RealmAuditLog.objects.create(
        event_type=AuditLogEventType.REALM_ORG_TYPE_CHANGED,
        realm=realm,
        event_time=timezone_now(),
        acting_user=acting_user,
        extra_data={"old_value": old_value, "new_value": org_type},
    )
    event: Dict[str, Any] = dict(
        type="realm", op="update", property="org_type", value=org_type
    )
    send_event_on_commit(realm, event, active_user_ids(realm.id))


@transaction.atomic(durable=True)
def do_change_realm_max_invites(
    realm: Realm, max_invites: int, acting_user: UserProfile
) -> None:
    old_value = realm.max_invites
    if max_invites == 0:
        new_max = get_default_max_invites_for_realm_plan_type(realm.plan_type)
    else:
        new_max = max_invites
    realm.max_invites = new_max
    realm.save(update_fields=["_max_invites"])
    RealmAuditLog.objects.create(
        event_type=AuditLogEventType.REALM_PROPERTY_CHANGED,
        realm=realm,
        event_time=timezone_now(),
        acting_user=acting_user,
        extra_data={
            "old_value": old_value,
            "new_value": new_max,
            "property": "max_invites",
        },
    )


@transaction.atomic(savepoint=False)
def do_change_realm_plan_type(
    realm: Realm, plan_type: str, *, acting_user: UserProfile
) -> None:
    from zproject.backends import AUTH_BACKEND_NAME_MAP

    old_value = realm.plan_type
    if plan_type not in Realm.ALL_PLAN_TYPES:
        raise AssertionError("Invalid plan type")
    if plan_type == Realm.PLAN_TYPE_LIMITED:
        do_set_realm_property(
            realm, "enable_spectator_access", False, acting_user=acting_user
        )
    if old_value in [Realm.PLAN_TYPE_PLUS, Realm.PLAN_TYPE_SELF_HOSTED] and plan_type not in [
        Realm.PLAN_TYPE_PLUS,
        Realm.PLAN_TYPE_SELF_HOSTED,
    ]:
        everyone_system_group = NamedUserGroup.objects.get(
            name=SystemGroups.EVERYONE, realm=realm, is_system_group=True
        )
        if realm.can_access_all_users_group_id != everyone_system_group.id:
            do_change_realm_permission_group_setting(
                realm,
                "can_access_all_users_group",
                everyone_system_group,
                acting_user=acting_user,
            )
    if settings.BILLING_ENABLED:
        realm_authentication_methods = realm.authentication_methods_dict()
        for backend_name, enabled in realm_authentication_methods.items():
            if enabled and plan_type < old_value:
                available_for = AUTH_BACKEND_NAME_MAP[backend_name].available_for_cloud_plans
                if available_for is not None and plan_type not in available_for:
                    realm_authentication_methods[backend_name] = False
        if realm_authentication_methods != realm.authentication_methods_dict():
            do_set_realm_authentication_methods(
                realm, realm_authentication_methods, acting_user=acting_user
            )
    realm.plan_type = plan_type
    realm.save(update_fields=["plan_type"])
    RealmAuditLog.objects.create(
        event_type=AuditLogEventType.REALM_PLAN_TYPE_CHANGED,
        realm=realm,
        event_time=timezone_now(),
        acting_user=acting_user,
        extra_data={"old_value": old_value, "new_value": plan_type},
    )
    realm.max_invites = get_default_max_invites_for_realm_plan_type(plan_type)
    if plan_type == Realm.PLAN_TYPE_LIMITED:
        realm.message_visibility_limit = Realm.MESSAGE_VISIBILITY_LIMITED
    else:
        realm.message_visibility_limit = None
    update_first_visible_message_id(realm)
    realm.save(update_fields=["_max_invites", "message_visibility_limit"])
    event: Dict[str, Any] = dict(
        type="realm",
        op="update_dict",
        property="default",
        data={
            "plan_type": plan_type,
            "upload_quota_mib": optional_bytes_to_mib(realm.upload_quota_bytes()),
            "max_file_upload_size_mib": realm.get_max_file_upload_size_mebibytes(),
        },
    )
    send_event_on_commit(realm, event, active_user_ids(realm.id))


def do_send_realm_reactivation_email(
    realm: Realm, *, acting_user: UserProfile
) -> None:
    obj = RealmReactivationStatus.objects.create(realm=realm)
    url = create_confirmation_link(obj, Confirmation.REALM_REACTIVATION)
    RealmAuditLog.objects.create(
        realm=realm,
        acting_user=acting_user,
        event_type=AuditLogEventType.REALM_REACTIVATION_EMAIL_SENT,
        event_time=timezone_now(),
    )
    context = {
        "confirmation_url": url,
        "realm_url": realm.url,
        "realm_name": realm.name,
        "corporate_enabled": settings.CORPORATE_ENABLED,
    }
    language = realm.default_language
    send_email_to_admins(
        "zerver/emails/realm_reactivation",
        realm,
        from_address=FromAddress.tokenized_no_reply_address(),
        from_name=FromAddress.security_email_from_name(language=language),
        language=language,
        context=context,
    )


def do_send_realm_deactivation_email(
    realm: Realm, acting_user: Optional[UserProfile], deletion_delay_days: Optional[int]
) -> None:
    shared_context = {"realm_name": realm.name}
    deactivation_time = timezone_now()
    owners: set[UserProfile] = set(realm.get_human_owner_users())
    anonymous_deactivation = False
    data_deleted = False
    scheduled_data_deletion: Optional[datetime.datetime] = None
    if acting_user is None:
        anonymous_deactivation = True
    if acting_user is not None and acting_user not in owners:
        anonymous_deactivation = True
    if deletion_delay_days is not None:
        if deletion_delay_days == 0:
            data_deleted = True
        else:
            scheduled_data_deletion = realm.scheduled_deletion_date
    for owner in owners:
        owner_tz = owner.timezone
        if owner_tz == "":
            owner_tz = timezone_get_current_timezone_name()
        local_date = deactivation_time.astimezone(
            zoneinfo.ZoneInfo(canonicalize_timezone(owner_tz))
        ).date()
        if scheduled_data_deletion:
            data_deletion_date = scheduled_data_deletion.astimezone(
                zoneinfo.ZoneInfo(canonicalize_timezone(owner_tz))
            ).date()
        else:
            data_deletion_date = None
        if anonymous_deactivation:
            context = dict(
                acting_user=False,
                initiated_deactivation=False,
                event_date=local_date,
                data_already_deleted=data_deleted,
                scheduled_deletion_date=data_deletion_date,
                **shared_context,
            )
        else:
            assert acting_user is not None
            if owner == acting_user:
                context = dict(
                    acting_user=True,
                    initiated_deactivation=True,
                    event_date=local_date,
                    data_already_deleted=data_deleted,
                    scheduled_deletion_date=data_deletion_date,
                    **shared_context,
                )
            else:
                context = dict(
                    acting_user=True,
                    initiated_deactivation=False,
                    deactivating_owner=acting_user.full_name,
                    event_date=local_date,
                    data_already_deleted=data_deleted,
                    scheduled_deletion_date=data_deletion_date,
                    **shared_context,
                )
        send_email(
            "zerver/emails/realm_deactivated",
            to_emails=[owner.delivery_email],
            from_name=FromAddress.security_email_from_name(
                language=owner.default_language
            ),
            from_address=FromAddress.SUPPORT,
            language=owner.default_language,
            context=context,
            realm=realm,
        )



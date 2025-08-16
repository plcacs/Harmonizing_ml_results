import datetime
import logging
import zoneinfo
from email.headerregistry import Address
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from django.conf import settings
from django.db import transaction
from django.utils.timezone import get_current_timezone_name as timezone_get_current_timezone_name
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _

from confirmation.models import Confirmation, create_confirmation_link, generate_key
from zerver.actions.custom_profile_fields import do_remove_realm_custom_profile_fields
from zerver.actions.message_delete import do_delete_messages_by_sender
from zerver.actions.user_groups import update_users_in_full_members_system_group
from zerver.actions.user_settings import do_delete_avatar_image
from zerver.lib.exceptions import JsonableError
from zerver.lib.message import parse_message_time_limit_setting, update_first_visible_message_id
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
from zerver.models.realms import get_default_max_invites_for_realm_plan_type, get_realm
from zerver.models.users import active_user_ids
from zerver.tornado.django_api import send_event_on_commit


@transaction.atomic(savepoint=False)
def do_set_realm_property(
    realm: Realm, name: str, value: Any, *, acting_user: Optional[UserProfile]
) -> None:
    """Takes in a realm object, the name of an attribute to update, the
    value to update and the user who initiated the update.
    """
    property_type = Realm.property_types[name]
    assert isinstance(value, property_type), (
        f"Cannot update {name}: {value} is not an instance of {property_type}"
    )

    old_value = getattr(realm, name)
    if old_value == value:
        return

    setattr(realm, name, value)
    realm.save(update_fields=[name])

    event = dict(
        type="realm",
        op="update",
        property=name,
        value=value,
    )

    message_edit_settings = [
        "allow_message_editing",
        "message_content_edit_limit_seconds",
    ]
    if name in message_edit_settings:
        event = dict(
            type="realm",
            op="update_dict",
            property="default",
            data={name: value},
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
        update_users_in_full_members_system_group(realm, acting_user=acting_user)


@transaction.atomic(durable=True)
def do_set_push_notifications_enabled_end_timestamp(
    realm: Realm, value: Optional[int], *, acting_user: Optional[UserProfile]
) -> None:
    name = "push_notifications_enabled_end_timestamp"
    old_timestamp: Optional[int] = None
    old_datetime = getattr(realm, name)
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

    event = dict(
        type="realm",
        op="update",
        property=name,
        value=value,
    )
    send_event_on_commit(realm, event, active_user_ids(realm.id))


@transaction.atomic(savepoint=False)
def do_change_realm_permission_group_setting(
    realm: Realm,
    setting_name: str,
    user_group: UserGroup,
    old_setting_api_value: Union[int, AnonymousSettingGroupDict, None] = None,
    *,
    acting_user: Optional[UserProfile],
) -> None:
    assert setting_name in Realm.REALM_PERMISSION_GROUP_SETTINGS
    old_value = getattr(realm, setting_name)

    setattr(realm, setting_name, user_group)
    realm.save(update_fields=[setting_name])

    if old_setting_api_value is None:
        old_setting_api_value = get_group_setting_value_for_api(old_value)
    new_setting_api_value = get_group_setting_value_for_api(user_group)

    if not hasattr(old_value, "named_user_group") and hasattr(user_group, "named_user_group"):
        old_value.delete()

    event = dict(
        type="realm",
        op="update_dict",
        property="default",
        data={setting_name: new_setting_api_value},
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
    realm: Realm, setting_name: str, value: Union[int, str], *, acting_user: Optional[UserProfile]
) -> Tuple[Optional[int], bool]:
    parsed_value = parse_message_time_limit_setting(
        value,
        Realm.MESSAGE_TIME_LIMIT_SETTING_SPECIAL_VALUES_MAP,
        setting_name=setting_name,
    )

    setting_value_changed = False
    if parsed_value is None and getattr(realm, setting_name) is not None:
        do_set_realm_property(
            realm,
            setting_name,
            parsed_value,
            acting_user=acting_user,
        )
        setting_value_changed = True

    return parsed_value, setting_value_changed


def get_realm_authentication_methods_for_page_params_api(
    realm: Realm, authentication_methods: Dict[str, bool]
) -> Dict[str, Dict[str, Union[str, bool]]]:
    result_dict: Dict[str, Dict[str, Union[str, bool]]] = {
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
                _("Invalid authentication method: {name}. Valid methods are: {methods}").format(
                    name=name, methods=sorted(current_authentication_methods.keys())
                )
            )

    if settings.BILLING_ENABLED:
        validate_plan_for_authentication_methods(realm, authentication_methods)


def validate_plan_for_authentication_methods(
    realm: Realm, authentication_methods: Dict[str, bool]
) -> None:
    from zproject.backends import AUTH_BACKEND_NAME_MAP

    old_authentication_methods = realm.authentication_methods_dict()
    newly_enabled_authentication_methods = {
        name
        for name, enabled in authentication_methods.items()
        if enabled and not old_authentication_methods.get(name, False)
    }
    for name in newly_enabled_authentication_methods:
        available_for = AUTH_BACKEND_NAME_MAP[name].available_for_cloud_plans
        if available_for is not None and realm.plan_type not in available_for:
            raise JsonableError(
                _("Authentication method {name} is not available on your current plan.").format(
                    name=name
                )
            )


@transaction.atomic(savepoint=False)
def do_set_realm_authentication_methods(
    realm: Realm, authentication_methods: Dict[str, bool], *, acting_user: Optional[UserProfile]
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

    event_data = dict(
        authentication_methods=get_realm_authentication_methods_for_page_params_api(
            realm, updated_value
        )
    )
    event = dict(
        type="realm",
        op="update_dict",
        property="default",
        data=event_data,
    )
    send_event_on_commit(realm, event, active_user_ids(realm.id))


def do_set_realm_stream(
    realm: Realm,
    field: Literal[
        "moderation_request_channel",
        "new_stream_announcements_stream",
        "signup_announcements_stream",
        "zulip_update_announcements_stream",
    ],
    stream: Optional[Stream],
    stream_id: int,
    *,
    acting_user: Optional[UserProfile],
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

        event = dict(
            type="realm",
            op="update",
            property=property,
            value=stream_id,
        )
        send_event_on_commit(realm, event, active_user_ids(realm.id))


def do_set_realm_moderation_request_channel(
    realm: Realm, stream: Optional[Stream], stream_id: int, *, acting_user: Optional[UserProfile]
) -> None:
    if stream is not None and stream.is_public():
        raise JsonableError(_("Moderation request channel must be private."))
    do_set_realm_stream(
        realm, "moderation_request_channel", stream, stream_id, acting_user=acting_user
    )


def do_set_realm_new_stream_announcements_stream(
    realm: Realm, stream: Optional[Stream], stream_id: int, *, acting_user: Optional[UserProfile]
) -> None:
    do_set_realm_stream(
        realm, "new_stream_announcements_stream", stream, stream_id, acting_user=acting_user
    )


def do_set_realm_signup_announcements_stream(
    realm: Realm, stream: Optional[Stream], stream_id: int, *, acting_user: Optional[UserProfile]
) -> None:
    do_set_realm_stream(
        realm, "signup_announcements_stream", stream, stream_id, acting_user=acting_user
    )


def do_set_realm_zulip_update_announcements_stream(
    realm: Realm, stream: Optional[Stream], stream_id: int, *, acting_user: Optional[UserProfile]
) -> None:
    do_set_realm_stream(
        realm, "zulip_update_announcements_stream", stream, stream_id, acting_user=acting_user
    )


@transaction.atomic(durable=True)
def do_set_realm_user_default_setting(
    realm_user_default: RealmUserDefault,
    name: str,
    value: Any,
    *,
    acting_user: Optional[UserProfile],
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

    event = dict(
        type="realm_user_settings_defaults",
        op="update",
        property=name,
        value=value,
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
    acting_user: Optional[UserProfile],
    deactivation_reason: RealmDeactivationReasonType,
    deletion_delay_days: Optional[int] = None,
    email_owners: bool,
) -> None:
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

        event
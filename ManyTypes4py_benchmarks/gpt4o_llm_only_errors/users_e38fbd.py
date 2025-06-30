import secrets
from collections import defaultdict
from email.headerregistry import Address
from typing import Any, Dict, List, Optional, Union
from django.conf import settings
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.db import transaction
from django.db.models import Q, QuerySet
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

def do_delete_user(user_profile: UserProfile, *, acting_user: Optional[UserProfile]) -> None:
    if user_profile.realm.is_zephyr_mirror_realm:
        raise AssertionError('Deleting zephyr mirror users is not supported')
    do_deactivate_user(user_profile, acting_user=acting_user)
    to_resubscribe_recipient_ids = set(Subscription.objects.filter(user_profile=user_profile, recipient__type=Recipient.DIRECT_MESSAGE_GROUP).values_list('recipient_id', flat=True))
    user_id = user_profile.id
    realm = user_profile.realm
    date_joined = user_profile.date_joined
    personal_recipient = user_profile.recipient
    with transaction.atomic(durable=True):
        user_profile.delete()
        assert personal_recipient is not None
        personal_recipient.delete()
        replacement_user = create_user(force_id=user_id, email=Address(username=f'deleteduser{user_id}', domain=get_fake_email_domain(realm.host)).addr_spec, password=None, realm=realm, full_name=f'Deleted User {user_id}', active=False, is_mirror_dummy=True, force_date_joined=date_joined)
        subs_to_recreate = [Subscription(user_profile=replacement_user, recipient=recipient, is_user_active=replacement_user.is_active) for recipient in Recipient.objects.filter(id__in=to_resubscribe_recipient_ids)]
        Subscription.objects.bulk_create(subs_to_recreate)
        RealmAuditLog.objects.create(realm=replacement_user.realm, modified_user=replacement_user, acting_user=acting_user, event_type=AuditLogEventType.USER_DELETED, event_time=timezone_now())

def do_delete_user_preserving_messages(user_profile: UserProfile) -> None:
    if user_profile.realm.is_zephyr_mirror_realm:
        raise AssertionError('Deleting zephyr mirror users is not supported')
    do_deactivate_user(user_profile, acting_user=None)
    user_id = user_profile.id
    personal_recipient = user_profile.recipient
    realm = user_profile.realm
    date_joined = user_profile.date_joined
    with transaction.atomic(durable=True):
        random_token = secrets.token_hex(16)
        temp_replacement_user = create_user(email=Address(username=f'temp_deleteduser{random_token}', domain=get_fake_email_domain(realm.host)).addr_spec, password=None, realm=realm, full_name=f'Deleted User {user_id} (temp)', active=False, is_mirror_dummy=True, force_date_joined=date_joined, create_personal_recipient=False)
        Message.objects.filter(realm_id=realm.id, sender=user_profile).update(sender=temp_replacement_user)
        Subscription.objects.filter(user_profile=user_profile, recipient__type=Recipient.DIRECT_MESSAGE_GROUP).update(user_profile=temp_replacement_user)
        user_profile.delete()
        replacement_user = create_user(force_id=user_id, email=Address(username=f'deleteduser{user_id}', domain=get_fake_email_domain(realm.host)).addr_spec, password=None, realm=realm, full_name=f'Deleted User {user_id}', active=False, is_mirror_dummy=True, force_date_joined=date_joined, create_personal_recipient=False)
        replacement_user.recipient = personal_recipient
        replacement_user.save(update_fields=['recipient'])
        Message.objects.filter(realm_id=realm.id, sender=temp_replacement_user).update(sender=replacement_user)
        Subscription.objects.filter(user_profile=temp_replacement_user, recipient__type=Recipient.DIRECT_MESSAGE_GROUP).update(user_profile=replacement_user, is_user_active=replacement_user.is_active)
        temp_replacement_user.delete()
        RealmAuditLog.objects.create(realm=replacement_user.realm, modified_user=replacement_user, acting_user=None, event_type=AuditLogEventType.USER_DELETED_PRESERVING_MESSAGES, event_time=timezone_now())

def change_user_is_active(user_profile: UserProfile, value: bool) -> None:
    with transaction.atomic(savepoint=False):
        user_profile.is_active = value
        user_profile.save(update_fields=['is_active'])
        Subscription.objects.filter(user_profile=user_profile).update(is_user_active=value)

def send_group_update_event_for_anonymous_group_setting(setting_group: UserGroup, group_members_dict: Dict[int, List[int]], group_subgroups_dict: Dict[int, List[int]], named_group: NamedUserGroup, notify_user_ids: List[int]) -> None:
    realm = setting_group.realm
    for setting_name in NamedUserGroup.GROUP_PERMISSION_SETTINGS:
        if getattr(named_group, setting_name + '_id') == setting_group.id:
            new_setting_value = AnonymousSettingGroupDict(direct_members=group_members_dict[setting_group.id], direct_subgroups=group_subgroups_dict[setting_group.id])
            event = dict(type='user_group', op='update', group_id=named_group.id, data={setting_name: new_setting_value})
            send_event_on_commit(realm, event, notify_user_ids)
            return

def send_realm_update_event_for_anonymous_group_setting(setting_group: UserGroup, group_members_dict: Dict[int, List[int]], group_subgroups_dict: Dict[int, List[int]], notify_user_ids: List[int]) -> None:
    realm = setting_group.realm
    for setting_name in Realm.REALM_PERMISSION_GROUP_SETTINGS:
        if getattr(realm, setting_name + '_id') == setting_group.id:
            new_setting_value = AnonymousSettingGroupDict(direct_members=group_members_dict[setting_group.id], direct_subgroups=group_subgroups_dict[setting_group.id])
            event = dict(type='realm', op='update_dict', property='default', data={setting_name: new_setting_value})
            send_event_on_commit(realm, event, notify_user_ids)
            return

def send_update_events_for_anonymous_group_settings(setting_groups: List[UserGroup], realm: Realm, notify_user_ids: List[int]) -> None:
    setting_group_ids = [group.id for group in setting_groups]
    membership = UserGroupMembership.objects.filter(user_group_id__in=setting_group_ids).exclude(user_profile__is_active=False).values_list('user_group_id', 'user_profile_id')
    group_membership = GroupGroupMembership.objects.filter(supergroup_id__in=setting_group_ids).values_list('subgroup_id', 'supergroup_id')
    group_members = defaultdict(list)
    for user_group_id, user_profile_id in membership:
        group_members[user_group_id].append(user_profile_id)
    group_subgroups = defaultdict(list)
    for subgroup_id, supergroup_id in group_membership:
        group_subgroups[supergroup_id].append(subgroup_id)
    group_setting_query = Q()
    for setting_name in NamedUserGroup.GROUP_PERMISSION_SETTINGS:
        group_setting_query |= Q(**{f'{setting_name}__in': setting_group_ids})
    named_groups_using_setting_groups_dict = {}
    named_groups_using_setting_groups = NamedUserGroup.objects.filter(realm=realm).filter(group_setting_query)
    for group in named_groups_using_setting_groups:
        for setting_name in NamedUserGroup.GROUP_PERMISSION_SETTINGS:
            setting_value_id = getattr(group, setting_name + '_id')
            if setting_value_id in setting_group_ids:
                named_groups_using_setting_groups_dict[setting_value_id] = group
    for setting_group in setting_groups:
        if setting_group.id in named_groups_using_setting_groups_dict:
            named_group = named_groups_using_setting_groups_dict[setting_group.id]
            send_group_update_event_for_anonymous_group_setting(setting_group, group_members, group_subgroups, named_group, notify_user_ids)
        else:
            send_realm_update_event_for_anonymous_group_setting(setting_group, group_members, group_subgroups, notify_user_ids)

def send_events_for_user_deactivation(user_profile: UserProfile) -> None:
    event_deactivate_user = dict(type='realm_user', op='update', person=dict(user_id=user_profile.id, is_active=False))
    realm = user_profile.realm
    if not user_access_restricted_in_realm(user_profile):
        send_event_on_commit(realm, event_deactivate_user, active_user_ids(realm.id))
        return
    non_guest_user_ids = active_non_guest_user_ids(realm.id)
    users_involved_in_dms_dict = get_users_involved_in_dms_with_target_users([user_profile], realm)
    deactivated_user_subs = Subscription.objects.filter(user_profile=user_profile, recipient__type__in=[Recipient.STREAM, Recipient.DIRECT_MESSAGE_GROUP], active=True).values_list('recipient_id', flat=True)
    subscribers_in_deactivated_user_subs = Subscription.objects.filter(recipient_id__in=list(deactivated_user_subs), recipient__type__in=[Recipient.STREAM, Recipient.DIRECT_MESSAGE_GROUP], is_user_active=True, active=True).values_list('recipient__type', 'user_profile_id')
    peer_stream_subscribers = set()
    peer_direct_message_group_subscribers = set()
    for recipient_type, user_id in subscribers_in_deactivated_user_subs:
        if recipient_type == Recipient.DIRECT_MESSAGE_GROUP:
            peer_direct_message_group_subscribers.add(user_id)
        else:
            peer_stream_subscribers.add(user_id)
    users_with_access_to_deactivated_user = set(non_guest_user_ids) | users_involved_in_dms_dict[user_profile.id] | peer_direct_message_group_subscribers
    if users_with_access_to_deactivated_user:
        send_event_on_commit(realm, event_deactivate_user, list(users_with_access_to_deactivated_user))
    all_active_user_ids = active_user_ids(realm.id)
    users_without_access_to_deactivated_user = set(all_active_user_ids) - users_with_access_to_deactivated_user
    if users_without_access_to_deactivated_user:
        deactivated_user_groups = user_profile.direct_groups.select_related('named_user_group').order_by('id')
        deactivated_user_named_groups = []
        deactivated_user_setting_groups = []
        for group in deactivated_user_groups:
            if not hasattr(group, 'named_user_group'):
                deactivated_user_setting_groups.append(group)
            else:
                deactivated_user_named_groups.append(group)
        for user_group in deactivated_user_named_groups:
            event = dict(type='user_group', op='remove_members', group_id=user_group.id, user_ids=[user_profile.id])
            send_event_on_commit(user_group.realm, event, list(users_without_access_to_deactivated_user))
        if deactivated_user_setting_groups:
            send_update_events_for_anonymous_group_settings(deactivated_user_setting_groups, user_profile.realm, list(users_without_access_to_deactivated_user))
    users_losing_access_to_deactivated_user = peer_stream_subscribers - users_with_access_to_deactivated_user
    if users_losing_access_to_deactivated_user:
        event_remove_user = dict(type='realm_user', op='remove', person=dict(user_id=user_profile.id, full_name=str(UserProfile.INACCESSIBLE_USER_NAME)))
        send_event_on_commit(realm, event_remove_user, list(users_losing_access_to_deactivated_user))

def do_deactivate_user(user_profile: UserProfile, _cascade: bool = True, *, acting_user: Optional[UserProfile]) -> None:
    if not user_profile.is_active:
        return
    if settings.BILLING_ENABLED:
        from corporate.lib.stripe import RealmBillingSession
    if _cascade:
        bot_profiles = get_active_bots_owned_by_user(user_profile)
        for profile in bot_profiles:
            do_deactivate_user(profile, _cascade=False, acting_user=acting_user)
    with transaction.atomic(savepoint=False):
        if user_profile.realm.is_zephyr_mirror_realm:
            user_profile.is_mirror_dummy = True
            user_profile.save(update_fields=['is_mirror_dummy'])
        change_user_is_active(user_profile, False)
        clear_scheduled_emails(user_profile.id)
        revoke_invites_generated_by_user(user_profile)
        event_time = timezone_now()
        RealmAuditLog.objects.create(realm=user_profile.realm, modified_user=user_profile, acting_user=acting_user, event_type=AuditLogEventType.USER_DEACTIVATED, event_time=event_time, extra_data={RealmAuditLog.ROLE_COUNT: realm_user_count_by_role(user_profile.realm)})
        maybe_enqueue_audit_log_upload(user_profile.realm)
        if settings.BILLING_ENABLED:
            billing_session = RealmBillingSession(user=user_profile, realm=user_profile.realm)
            billing_session.update_license_ledger_if_needed(event_time)
        transaction.on_commit(lambda: delete_user_sessions(user_profile))
        send_events_for_user_deactivation(user_profile)
        if user_profile.is_bot:
            event_deactivate_bot = dict(type='realm_bot', op='update', bot=dict(user_id=user_profile.id, is_active=False))
            send_event_on_commit(user_profile.realm, event_deactivate_bot, bot_owner_user_ids(user_profile))

def send_stream_events_for_role_update(user_profile: UserProfile, old_accessible_streams: List[Stream]) -> None:
    current_accessible_streams = get_streams_for_user(user_profile, include_all_active=user_profile.is_realm_admin, include_web_public=True)
    old_accessible_stream_ids = {stream.id for stream in old_accessible_streams}
    current_accessible_stream_ids = {stream.id for stream in current_accessible_streams}
    now_accessible_stream_ids = current_accessible_stream_ids - old_accessible_stream_ids
    if now_accessible_stream_ids:
        recent_traffic = get_streams_traffic(now_accessible_stream_ids, user_profile.realm)
        now_accessible_streams = [stream for stream in current_accessible_streams if stream.id in now_accessible_stream_ids]
        setting_groups_dict = get_group_setting_value_dict_for_streams(now_accessible_streams)
        event = dict(type='stream', op='create', streams=[stream_to_dict(stream, recent_traffic, setting_groups_dict) for stream in now_accessible_streams])
        send_event_on_commit(user_profile.realm, event, [user_profile.id])
        subscriber_peer_info = bulk_get_subscriber_peer_info(user_profile.realm, now_accessible_streams)
        for stream_id, stream_subscriber_set in subscriber_peer_info.subscribed_ids.items():
            peer_add_event = dict(type='subscription', op='peer_add', stream_ids=[stream_id], user_ids=sorted(stream_subscriber_set))
            send_event_on_commit(user_profile.realm, peer_add_event, [user_profile.id])
    now_inaccessible_stream_ids = old_accessible_stream_ids - current_accessible_stream_ids
    if now_inaccessible_stream_ids:
        now_inaccessible_streams = [stream for stream in old_accessible_streams if stream.id in now_inaccessible_stream_ids]
        send_stream_deletion_event(user_profile.realm, [user_profile.id], now_inaccessible_streams)

@transaction.atomic(savepoint=False)
def do_change_user_role(user_profile: UserProfile, value: int, *, acting_user: Optional[UserProfile]) -> None:
    UserProfile.objects.select_for_update().get(id=user_profile.id)
    user_profile.refresh_from_db()
    old_value = user_profile.role
    if old_value == value:
        return
    old_system_group = get_system_user_group_for_user(user_profile)
    previously_accessible_streams = get_streams_for_user(user_profile, include_web_public=True, include_all_active=user_profile.is_realm_admin)
    user_profile.role = value
    user_profile.save(update_fields=['role'])
    RealmAuditLog.objects.create(realm=user_profile.realm, modified_user=user_profile, acting_user=acting_user, event_type=AuditLogEventType.USER_ROLE_CHANGED, event_time=timezone_now(), extra_data={RealmAuditLog.OLD_VALUE: old_value, RealmAuditLog.NEW_VALUE: value, RealmAuditLog.ROLE_COUNT: realm_user_count_by_role(user_profile.realm)})
    maybe_enqueue_audit_log_upload(user_profile.realm)
    if settings.BILLING_ENABLED and UserProfile.ROLE_GUEST in [old_value, value]:
        from corporate.lib.stripe import RealmBillingSession
        billing_session = RealmBillingSession(user=user_profile, realm=user_profile.realm)
        billing_session.update_license_ledger_if_needed(timezone_now())
    event = dict(type='realm_user', op='update', person=dict(user_id=user_profile.id, role=user_profile.role))
    send_event_on_commit(user_profile.realm, event, get_user_ids_who_can_access_user(user_profile))
    UserGroupMembership.objects.filter(user_profile=user_profile, user_group=old_system_group).delete()
    system_group = get_system_user_group_for_user(user_profile)
    now = timezone_now()
    UserGroupMembership.objects.create(user_profile=user_profile, user_group=system_group)
    RealmAuditLog.objects.bulk_create([RealmAuditLog(realm=user_profile.realm, modified_user=user_profile, modified_user_group=old_system_group, event_type=AuditLogEventType.USER_GROUP_DIRECT_USER_MEMBERSHIP_REMOVED, event_time=now, acting_user=acting_user), RealmAuditLog(realm=user_profile.realm, modified_user=user_profile, modified_user_group=system_group, event_type=AuditLogEventType.USER_GROUP_DIRECT_USER_MEMBERSHIP_ADDED, event_time=now, acting_user=acting_user)])
    do_send_user_group_members_update_event('remove_members', old_system_group, [user_profile.id])
    do_send_user_group_members_update_event('add_members', system_group, [user_profile.id])
    if UserProfile.ROLE_MEMBER in [old_value, value]:
        update_users_in_full_members_system_group(user_profile.realm, [user_profile.id], acting_user=acting_user)
    send_stream_events_for_role_update(user_profile, previously_accessible_streams)

@transaction.atomic(savepoint=False)
def do_change_is_billing_admin(user_profile: UserProfile, value: bool) -> None:
    event_time = timezone_now()
    old_value = user_profile.is_billing_admin
    user_profile.is_billing_admin = value
    user_profile.save(update_fields=['is_billing_admin'])
    RealmAuditLog.objects.create(realm=user_profile.realm, event_type=AuditLogEventType.USER_SPECIAL_PERMISSION_CHANGED, event_time=event_time, acting_user=None, modified_user=user_profile, extra_data={RealmAuditLog.OLD_VALUE: old_value, RealmAuditLog.NEW_VALUE: value, 'property': 'is_billing_admin'})
    event = dict(type='realm_user', op='update', person=dict(user_id=user_profile.id, is_billing_admin=value))
    send_event_on_commit(user_profile.realm, event, get_user_ids_who_can_access_user(user_profile))

@transaction.atomic(savepoint=False)
def do_change_can_forge_sender(user_profile: UserProfile, value: bool) -> None:
    event_time = timezone_now()
    old_value = user_profile.can_forge_sender
    user_profile.can_forge_sender = value
    user_profile.save(update_fields=['can_forge_sender'])
    RealmAuditLog.objects.create(realm=user_profile.realm, event_type=AuditLogEventType.USER_SPECIAL_PERMISSION_CHANGED, event_time=event_time, acting_user=None, modified_user=user_profile, extra_data={RealmAuditLog.OLD_VALUE: old_value, RealmAuditLog.NEW_VALUE: value, 'property': 'can_forge_sender'})

@transaction.atomic(savepoint=False)
def do_change_can_create_users(user_profile: UserProfile, value: bool) -> None:
    event_time = timezone_now()
    old_value = user_profile.can_create_users
    user_profile.can_create_users = value
    user_profile.save(update_fields=['can_create_users'])
    RealmAuditLog.objects.create(realm=user_profile.realm, event_type=AuditLogEventType.USER_SPECIAL_PERMISSION_CHANGED, event_time=event_time, acting_user=None, modified_user=user_profile, extra_data={RealmAuditLog.OLD_VALUE: old_value, RealmAuditLog.NEW_VALUE: value, 'property': 'can_create_users'})

@transaction.atomic(savepoint=False)
def do_change_can_change_user_emails(user_profile: UserProfile, value: bool) -> None:
    event_time = timezone_now()
    old_value = user_profile.can_change_user_emails
    user_profile.can_change_user_emails = value
    user_profile.save(update_fields=['can_change_user_emails'])
    RealmAuditLog.objects.create(realm=user_profile.realm, event_type=AuditLogEventType.USER_SPECIAL_PERMISSION_CHANGED, event_time=event_time, acting_user=None, modified_user=user_profile, extra_data={RealmAuditLog.OLD_VALUE: old_value, RealmAuditLog.NEW_VALUE: value, 'property': 'can_change_user_emails'})

@transaction.atomic(durable=True)
def do_update_outgoing_webhook_service(bot_profile: UserProfile, service_interface: int, service_payload_url: str) -> None:
    service = get_bot_services(bot_profile.id)[0]
    service.base_url = service_payload_url
    service.interface = service_interface
    service.save()
    send_event_on_commit(bot_profile.realm, dict(type='realm_bot', op='update', bot=dict(user_id=bot_profile.id, services=[dict(base_url=service.base_url, interface=service.interface, token=service.token)])), bot_owner_user_ids(bot_profile))

@transaction.atomic(durable=True)
def do_update_bot_config_data(bot_profile: UserProfile, config_data: Dict[str, str]) -> None:
    for key, value in config_data.items():
        set_bot_config(bot_profile, key, value)
    updated_config_data = get_bot_config(bot_profile)
    send_event_on_commit(bot_profile.realm, dict(type='realm_bot', op='update', bot=dict(user_id=bot_profile.id, services=[dict(config_data=updated_config_data)])), bot_owner_user_ids(bot_profile))

def get_service_dicts_for_bot(user_profile_id: int) -> List[Dict[str, Any]]:
    user_profile = get_user_profile_by_id(user_profile_id)
    services = get_bot_services(user_profile_id)
    if user_profile.bot_type == UserProfile.OUTGOING_WEBHOOK_BOT:
        return [{'base_url': service.base_url, 'interface': service.interface, 'token': service.token} for service in services]
    elif user_profile.bot_type == UserProfile.EMBEDDED_BOT:
        try:
            return [{'config_data': get_bot_config(user_profile), 'service_name': services[0].name}]
        except ConfigError:
            return []
    else:
        return []

def get_service_dicts_for_bots(bot_dicts: List[Dict[str, Any]], realm: Realm) -> Dict[int, List[Dict[str, Any]]]:
    bot_profile_ids = [bot_dict['id'] for bot_dict in bot_dicts]
    bot_services_by_uid = defaultdict(list)
    for service in Service.objects.filter(user_profile_id__in=bot_profile_ids):
        bot_services_by_uid[service.user_profile_id].append(service)
    embedded_bot_ids = [bot_dict['id'] for bot_dict in bot_dicts if bot_dict['bot_type'] == UserProfile.EMBEDDED_BOT]
    embedded_bot_configs = get_bot_configs(embedded_bot_ids)
    service_dicts_by_uid = {}
    for bot_dict in bot_dicts:
        bot_profile_id = bot_dict['id']
        bot_type = bot_dict['bot_type']
        services = bot_services_by_uid[bot_profile_id]
        service_dicts = []
        if bot_type == UserProfile.OUTGOING_WEBHOOK_BOT:
            service_dicts = [{'base_url': service.base_url, 'interface': service.interface, 'token': service.token} for service in services]
        elif bot_type == UserProfile.EMBEDDED_BOT and bot_profile_id in embedded_bot_configs:
            bot_config = embedded_bot_configs[bot_profile_id]
            service_dicts = [{'config_data': bot_config, 'service_name': services[0].name}]
        service_dicts_by_uid[bot_profile_id] = service_dicts
    return service_dicts_by_uid

def get_owned_bot_dicts(user_profile: UserProfile, include_all_realm_bots_if_admin: bool = True) -> List[Dict[str, Any]]:
    if user_profile.is_realm_admin and include_all_realm_bots_if_admin:
        result = get_bot_dicts_in_realm(user_profile.realm)
    else:
        result = UserProfile.objects.filter(realm=user_profile.realm, is_bot=True, bot_owner=user_profile).values(*bot_dict_fields)
    services_by_ids = get_service_dicts_for_bots(result, user_profile.realm)
    return [{'email': botdict['email'], 'user_id': botdict['id'], 'full_name': botdict['full_name'], 'bot_type': botdict['bot_type'], 'is_active': botdict['is_active'], 'api_key': botdict['api_key'], 'default_sending_stream': botdict['default_sending_stream__name'], 'default_events_register_stream': botdict['default_events_register_stream__name'], 'default_all_public_streams': botdict['default_all_public_streams'], 'owner_id': botdict['bot_owner_id'], 'avatar_url': get_avatar_field(user_id=botdict['id'], realm_id=botdict['realm_id'], email=botdict['email'], avatar_source=botdict['avatar_source'], avatar_version=botdict['avatar_version'], medium=False, client_gravatar=False), 'services': services_by_ids[botdict['id']]} for botdict in result]

def generate_password_reset_url(user_profile: UserProfile, token_generator: PasswordResetTokenGenerator) -> str:
    token = token_generator.make_token(user_profile)
    uid = urlsafe_base64_encode(str(user_profile.id).encode())
    endpoint = reverse('password_reset_confirm', kwargs=dict(uidb64=uid, token=token))
    return f'{user_profile.realm.url}{endpoint}'

def do_send_password_reset_email(email: str, realm: Realm, user_profile: Optional[UserProfile], *, token_generator: PasswordResetTokenGenerator = default_token_generator, request: Optional[HttpRequest] = None) -> None:
    context = {'email': email, 'realm_url': realm.url, 'realm_name': realm.name}
    if user_profile is not None and (not user_profile.is_active):
        context['user_deactivated'] = True
        user_profile = None
    if user_profile is not None:
        queue_soft_reactivation(user_profile.id)
        context['active_account_in_realm'] = True
        context['reset_url'] = generate_password_reset_url(user_profile, token_generator)
        send_email('zerver/emails/password_reset', to_user_ids=[user_profile.id], from_name=FromAddress.security_email_from_name(user_profile=user_profile), from_address=FromAddress.tokenized_no_reply_address(), context=context, realm=realm, request=request)
    else:
        context['active_account_in_realm'] = False
        active_accounts_in_other_realms = UserProfile.objects.filter(delivery_email__iexact=email, is_active=True)
        if active_accounts_in_other_realms:
            context['active_accounts_in_other_realms'] = active_accounts_in_other_realms
        language = get_language()
        send_email('zerver/emails/password_reset', to_emails=[email], from_name=FromAddress.security_email_from_name(language=language), from_address=FromAddress.tokenized_no_reply_address(), language=language, context=context, realm=realm, request=request)

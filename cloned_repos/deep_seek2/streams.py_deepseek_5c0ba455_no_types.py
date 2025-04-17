from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypedDict, Optional, Union, List, Dict, Set, Tuple
from django.db import transaction
from django.db.models import Exists, OuterRef, Q, QuerySet, Value
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from zerver.lib.default_streams import get_default_stream_ids_for_realm
from zerver.lib.exceptions import CannotAdministerChannelError, IncompatibleParametersError, JsonableError, OrganizationOwnerRequiredError
from zerver.lib.stream_subscription import get_active_subscriptions_for_stream_id, get_subscribed_stream_ids_for_user
from zerver.lib.stream_traffic import get_average_weekly_stream_traffic, get_streams_traffic
from zerver.lib.string_validation import check_stream_name
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.types import AnonymousSettingGroupDict, APIStreamDict
from zerver.lib.user_groups import get_recursive_group_members, get_recursive_group_members_union_for_groups, get_recursive_membership_groups, get_role_based_system_groups_dict, user_has_permission_for_group_setting
from zerver.models import DefaultStreamGroup, GroupGroupMembership, Message, NamedUserGroup, Realm, RealmAuditLog, Recipient, Stream, Subscription, UserGroup, UserGroupMembership, UserProfile
from zerver.models.groups import SystemGroups
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.streams import bulk_get_streams, get_realm_stream, get_stream, get_stream_by_id_for_sending_message, get_stream_by_id_in_realm
from zerver.models.users import active_non_guest_user_ids, active_user_ids, is_cross_realm_bot_email
from zerver.tornado.django_api import send_event_on_commit

class StreamDict(TypedDict, total=False):
    name: str
    description: str
    invite_only: bool
    is_web_public: bool
    stream_post_policy: int
    history_public_to_subscribers: Optional[bool]
    message_retention_days: Optional[int]
    can_add_subscribers_group: Optional[UserGroup]
    can_administer_channel_group: Optional[UserGroup]
    can_send_message_group: Optional[UserGroup]
    can_remove_subscribers_group: Optional[UserGroup]

def get_stream_permission_policy_name(*, invite_only: Optional[bool]=None, history_public_to_subscribers: Optional[bool]=None, is_web_public: Optional[bool]=None):
    policy_name = None
    for permission_dict in Stream.PERMISSION_POLICIES.values():
        if permission_dict['invite_only'] == invite_only and permission_dict['history_public_to_subscribers'] == history_public_to_subscribers and (permission_dict['is_web_public'] == is_web_public):
            policy_name = permission_dict['policy_name']
            break
    assert policy_name is not None
    return policy_name

def get_default_value_for_history_public_to_subscribers(realm, invite_only, history_public_to_subscribers):
    if invite_only:
        if history_public_to_subscribers is None:
            history_public_to_subscribers = False
    else:
        history_public_to_subscribers = True
    if realm.is_zephyr_mirror_realm:
        history_public_to_subscribers = False
    return history_public_to_subscribers

def render_stream_description(text, realm, *, acting_user: Optional[UserProfile]=None):
    from zerver.lib.markdown import markdown_convert
    return markdown_convert(text, message_realm=realm, no_previews=True, acting_user=acting_user).rendered_content

def send_stream_creation_event(realm, stream, user_ids, recent_traffic=None, setting_groups_dict=None):
    event = dict(type='stream', op='create', streams=[stream_to_dict(stream, recent_traffic, setting_groups_dict)])
    send_event_on_commit(realm, event, user_ids)

def get_stream_permission_default_group(setting_name, system_groups_name_dict, creator=None):
    setting_default_name = Stream.stream_permission_group_settings[setting_name].default_group_name
    if setting_default_name == 'stream_creator_or_nobody':
        if creator:
            default_group = UserGroup(realm=creator.realm)
            default_group.save()
            UserGroupMembership.objects.create(user_profile=creator, user_group=default_group)
            return default_group
        else:
            return system_groups_name_dict[SystemGroups.NOBODY]
    return system_groups_name_dict[setting_default_name]

def get_default_values_for_stream_permission_group_settings(realm, creator=None):
    group_setting_values = {}
    system_groups_name_dict = get_role_based_system_groups_dict(realm)
    for setting_name in Stream.stream_permission_group_settings:
        group_setting_values[setting_name] = get_stream_permission_default_group(setting_name, system_groups_name_dict, creator)
    return group_setting_values

def get_user_ids_with_metadata_access_via_permission_groups(stream):
    return set(get_recursive_group_members_union_for_groups([stream.can_add_subscribers_group_id, stream.can_administer_channel_group_id]).exclude(role=UserProfile.ROLE_GUEST).values_list('id', flat=True))

@transaction.atomic(savepoint=False)
def create_stream_if_needed(realm, stream_name, *, invite_only: bool=False, is_web_public: bool=False, history_public_to_subscribers: Optional[bool]=None, stream_description: str='', message_retention_days: Optional[int]=None, can_add_subscribers_group: Optional[UserGroup]=None, can_administer_channel_group: Optional[UserGroup]=None, can_send_message_group: Optional[UserGroup]=None, can_remove_subscribers_group: Optional[UserGroup]=None, acting_user: Optional[UserProfile]=None, setting_groups_dict: Optional[Dict[int, Union[int, AnonymousSettingGroupDict]]]=None):
    history_public_to_subscribers = get_default_value_for_history_public_to_subscribers(realm, invite_only, history_public_to_subscribers)
    group_setting_values = {}
    request_settings_dict = locals()
    system_groups_name_dict = None
    for setting_name in Stream.stream_permission_group_settings:
        if setting_name not in request_settings_dict:
            continue
        if request_settings_dict[setting_name] is None:
            if system_groups_name_dict is None:
                system_groups_name_dict = get_role_based_system_groups_dict(realm)
            group_setting_values[setting_name] = get_stream_permission_default_group(setting_name, system_groups_name_dict, creator=acting_user)
        else:
            group_setting_values[setting_name] = request_settings_dict[setting_name]
    stream_name = stream_name.strip()
    stream, created = Stream.objects.get_or_create(realm=realm, name__iexact=stream_name, defaults=dict(name=stream_name, creator=acting_user, description=stream_description, invite_only=invite_only, is_web_public=is_web_public, history_public_to_subscribers=history_public_to_subscribers, is_in_zephyr_realm=realm.is_zephyr_mirror_realm, message_retention_days=message_retention_days, **group_setting_values))
    if created:
        recipient = Recipient.objects.create(type_id=stream.id, type=Recipient.STREAM)
        stream.recipient = recipient
        stream.rendered_description = render_stream_description(stream_description, realm, acting_user=acting_user)
        stream.save(update_fields=['recipient', 'rendered_description'])
        event_time = timezone_now()
        RealmAuditLog.objects.create(realm=realm, acting_user=acting_user, modified_stream=stream, event_type=AuditLogEventType.CHANNEL_CREATED, event_time=event_time)
        if setting_groups_dict is None:
            setting_groups_dict = get_group_setting_value_dict_for_streams([stream])
        if stream.is_public():
            if stream.is_web_public:
                notify_user_ids = active_user_ids(stream.realm_id)
            else:
                notify_user_ids = active_non_guest_user_ids(stream.realm_id)
            send_stream_creation_event(realm, stream, notify_user_ids, setting_groups_dict=setting_groups_dict)
        else:
            realm_admin_ids = {user.id for user in stream.realm.get_admin_users_and_bots()}
            send_stream_creation_event(realm, stream, list(realm_admin_ids | get_user_ids_with_metadata_access_via_permission_groups(stream)), setting_groups_dict=setting_groups_dict)
    return (stream, created)

def create_streams_if_needed(realm, stream_dicts, acting_user=None, setting_groups_dict=None):
    added_streams: List[Stream] = []
    existing_streams: List[Stream] = []
    for stream_dict in stream_dicts:
        invite_only = stream_dict.get('invite_only', False)
        stream, created = create_stream_if_needed(realm, stream_dict['name'], invite_only=invite_only, is_web_public=stream_dict.get('is_web_public', False), history_public_to_subscribers=stream_dict.get('history_public_to_subscribers'), stream_description=stream_dict.get('description', ''), message_retention_days=stream_dict.get('message_retention_days', None), can_add_subscribers_group=stream_dict.get('can_add_subscribers_group', None), can_administer_channel_group=stream_dict.get('can_administer_channel_group', None), can_send_message_group=stream_dict.get('can_send_message_group', None), can_remove_subscribers_group=stream_dict.get('can_remove_subscribers_group', None), acting_user=acting_user, setting_groups_dict=setting_groups_dict)
        if created:
            added_streams.append(stream)
        else:
            existing_streams.append(stream)
    return (added_streams, existing_streams)

def subscribed_to_stream(user_profile, stream_id):
    return Subscription.objects.filter(user_profile=user_profile, active=True, recipient__type=Recipient.STREAM, recipient__type_id=stream_id).exists()

def is_user_in_can_administer_channel_group(stream, user_recursive_group_ids):
    group_allowed_to_administer_channel_id = stream.can_administer_channel_group_id
    assert group_allowed_to_administer_channel_id is not None
    return group_allowed_to_administer_channel_id in user_recursive_group_ids

def is_user_in_can_add_subscribers_group(stream, user_recursive_group_ids):
    group_allowed_to_add_subscribers_id = stream.can_add_subscribers_group_id
    assert group_allowed_to_add_subscribers_id is not None
    return group_allowed_to_add_subscribers_id in user_recursive_group_ids

def is_user_in_can_remove_subscribers_group(stream, user_recursive_group_ids):
    group_allowed_to_remove_subscribers_id = stream.can_remove_subscribers_group_id
    assert group_allowed_to_remove_subscribers_id is not None
    return group_allowed_to_remove_subscribers_id in user_recursive_group_ids

def check_stream_access_based_on_can_send_message_group(sender, stream):
    if is_cross_realm_bot_email(sender.delivery_email):
        return
    can_send_message_group = stream.can_send_message_group
    if hasattr(can_send_message_group, 'named_user_group'):
        if can_send_message_group.named_user_group.name == SystemGroups.EVERYONE:
            return
        if can_send_message_group.named_user_group.name == SystemGroups.NOBODY:
            raise JsonableError(_('You do not have permission to post in this channel.'))
    if not user_has_permission_for_group_setting(stream.can_send_message_group, sender, Stream.stream_permission_group_settings['can_send_message_group'], direct_member_only=False):
        raise JsonableError(_('You do not have permission to post in this channel.'))

def access_stream_for_send_message(sender, stream, forwarder_user_profile, archived_channel_notice=False):
    try:
        check_stream_access_based_on_can_send_message_group(sender, stream)
    except JsonableError as e:
        if sender.is_bot and sender.bot_owner is not None:
            check_stream_access_based_on_can_send_message_group(sender.bot_owner, stream)
        else:
            raise JsonableError(e.msg)
    if forwarder_user_profile is not None and forwarder_user_profile != sender:
        if forwarder_user_profile.can_forge_sender and forwarder_user_profile.realm_id == sender.realm_id and (sender.realm_id == stream.realm_id):
            return
        else:
            raise JsonableError(_('User not authorized for this query'))
    if stream.deactivated:
        if archived_channel_notice:
            return
        raise JsonableError(_("Not authorized to send to channel '{channel_name}'").format(channel_name=stream.name))
    if is_cross_realm_bot_email(sender.delivery_email):
        return
    if stream.realm_id != sender.realm_id:
        raise JsonableError(_('User not authorized for this query'))
    if stream.is_web_public:
        return
    if not (stream.invite_only or sender.is_guest):
        return
    if subscribed_to_stream(sender, stream.id):
        return
    if sender.can_forge_sender:
        return
    if sender.is_bot and (sender.bot_owner is not None and subscribed_to_stream(sender.bot_owner, stream.id)):
        return
    raise JsonableError(_("Not authorized to send to channel '{channel_name}'").format(channel_name=stream.name))

def check_for_exactly_one_stream_arg(stream_id, stream):
    if stream_id is None and stream is None:
        error = _("Missing '{var_name}' argument").format(var_name='stream_id')
        raise JsonableError(error)
    if stream_id is not None and stream is not None:
        raise IncompatibleParametersError(['stream_id', 'stream'])

@dataclass
class UserGroupMembershipDetails:
    user_recursive_group_ids: Optional[Set[int]]

def user_has_content_access(user_profile, stream, user_group_membership_details, *, is_subscribed: bool):
    if stream.is_web_public:
        return True
    if is_subscribed:
        return True
    if user_profile.is_guest:
        return False
    if stream.is_public():
        return True
    if user_group_membership_details.user_recursive_group_ids is None:
        user_group_membership_details.user_recursive_group_ids = set(get_recursive_membership_groups(user_profile).values_list('id', flat=True))
    if is_user_in_can_add_subscribers_group(stream, user_group_membership_details.user_recursive_group_ids):
        return True
    return False

def check_stream_access_for_delete_or_update_requiring_metadata_access(user_profile, stream, sub=None):
    error = _('Invalid channel ID')
    if stream.realm_id != user_profile.realm_id:
        raise JsonableError(error)
    if user_profile.is_realm_admin:
        return
    if can_administer_accessible_channel(stream, user_profile):
        return
    user_group_membership_details = UserGroupMembershipDetails(user_recursive_group_ids=None)
    if user_has_content_access(user_profile, stream, user_group_membership_details, is_subscribed=sub is not None):
        raise CannotAdministerChannelError
    raise JsonableError(error)

def access_stream_for_delete_or_update_requiring_metadata_access(user_profile, stream_id):
    try:
        stream = Stream.objects.get(id=stream_id)
    except Stream.DoesNotExist:
        raise JsonableError(_('Invalid channel ID'))
    try:
        sub = Subscription.objects.get(user_profile=user_profile, recipient=stream.recipient, active=True)
    except Subscription.DoesNotExist:
        sub = None
    check_stream_access_for_delete_or_update_requiring_metadata_access(user_profile, stream, sub)
    return (stream, sub)

def has_metadata_access_to_channel_via_groups(user_profile, user_recursive_group_ids, can_administer_channel_group_id, can_add_subscribers_group_id):
    for setting_name in Stream.stream_permission_group_settings_granting_metadata_access:
        permission_configuration = Stream.stream_permission_group_settings[setting_name]
        if not permission_configuration.allow_everyone_group and user_profile.is_guest:
            return False
    return can_administer_channel_group_id in user_recursive_group_ids or can_add_subscribers_group_id in user_recursive_group_ids

def check_basic_stream_access(user_profile, stream, *, is_subscribed: bool, require_content_access: bool=True):
    user_group_membership_details = UserGroupMembershipDetails(user_recursive_group_ids=None)
    if user_has_content_access(user_profile, stream, user_group_membership_details, is_subscribed=is_subscribed):
        return True
    if not require_content_access:
        if user_profile.is_realm_admin:
            return True
        if user_group_membership_details.user_recursive_group_ids is None:
            user_group_membership_details.user_recursive_group_ids = set(get_recursive_membership_groups(user_profile).values_list('id', flat=True))
        if has_metadata_access_to_channel_via_groups(user_profile, user_group_membership_details.user_recursive_group_ids, stream.can_administer_channel_group_id, stream.can_add_subscribers_group_id):
            return True
    return False

def access_stream_common(user_profile, stream, error, require_active=True, require_content_access=True):
    if stream.realm_id != user_profile.realm_id:
        raise AssertionError("user_profile and stream realms don't match")
    try:
        assert stream.recipient_id is not None
        sub = Subscription.objects.get(user_profile=user_profile, recipient_id=stream.recipient_id, active=require_active)
    except Subscription.DoesNotExist:
        sub = None
    if not stream.deactivated and check_basic_stream_access(user_profile, stream, is_subscribed=sub is not None, require_content_access=require_content_access):
        return sub
    raise JsonableError(error)

def access_stream_by_id(user_profile, stream_id, require_active=True, require_content_access=True):
    error = _('Invalid channel ID')
    try:
        stream = get_stream_by_id_in_realm(stream_id, user_profile.realm)
    except Stream.DoesNotExist:
        raise JsonableError(error)
    sub = access_stream_common(user_profile, stream, error, require_active=require_active, require_content_access=require_content_access)
    return (stream, sub)

def access_stream_by_id_for_message(user_profile, stream_id, require_active=True, require_content_access=True):
    error = _('Invalid channel ID')
    try:
        stream = get_stream_by_id_for_sending_message(stream_id, user_profile.realm)
    except Stream.DoesNotExist:
        raise JsonableError(error)
    sub = access_stream_common(user_profile, stream, error, require_active=require_active, require_content_access=require_content_access)
    return (stream, sub)

def get_public_streams_queryset(realm):
    return Stream.objects.filter(realm=realm, invite_only=False, history_public_to_subscribers=True)

def get_web_public_streams_queryset(realm):
    return Stream.objects.filter(realm=realm, is_web_public=True, deactivated=False, invite_only=False, history_public_to_subscribers=True).select_related('can_send_message_group', 'can_send_message_group__named_user_group')

def check_stream_name_available(realm, name):
    check_stream_name(name)
    try:
        get_stream(name, realm)
        raise JsonableError(_('Channel name already in use.'))
    except Stream.DoesNotExist:
        pass

def access_stream_by_name(user_profile, stream_name, require_content_access=True):
    error = _("Invalid channel name '{channel_name}'").format(channel_name=stream_name)
    try:
        stream = get_realm_stream(stream_name, user_profile.realm_id)
    except Stream.DoesNotExist:
        raise JsonableError(error)
    sub = access_stream_common(user_profile, stream, error, require_content_access=require_content_access)
    return (stream, sub)

def access_web_public_stream(stream_id, realm):
    error = _('Invalid channel ID')
    try:
        stream = get_stream_by_id_in_realm(stream_id, realm)
    except Stream.DoesNotExist:
        raise JsonableError(error)
    if not stream.is_web_public:
        raise JsonableError(error)
    return stream

def access_stream_to_remove_visibility_policy_by_name(user_profile, stream_name, error):
    try:
        stream = get_stream(stream_name, user_profile.realm)
    except Stream.DoesNotExist:
        raise JsonableError(error)
    return stream

def access_stream_to_remove_visibility_policy_by_id(user_profile, stream_id, error):
    try:
        stream = Stream.objects.get(id=stream_id, realm_id=user_profile.realm_id)
    except Stream.DoesNotExist:
        raise JsonableError(error)
    return stream

def private_stream_user_ids(stream_id):
    subscriptions = get_active_subscriptions_for_stream_id(stream_id, include_deactivated_users=False)
    return {sub['user_profile_id'] for sub in subscriptions.values('user_profile_id')}

def public_stream_user_ids(stream):
    guest_subscriptions = get_active_subscriptions_for_stream_id(stream.id, include_deactivated_users=False).filter(user_profile__role=UserProfile.ROLE_GUEST)
    guest_subscriptions_ids = {sub['user_profile_id'] for sub in guest_subscriptions.values('user_profile_id')}
    can_add_subscribers_group_user_ids = set(get_recursive_group_members(stream.can_add_subscribers_group_id).exclude(role=UserProfile.ROLE_GUEST).values_list('id', flat=True))
    return set(active_non_guest_user_ids(stream.realm_id)) | guest_subscriptions_ids | can_add_subscribers_group_user_ids

def can_access_stream_metadata_user_ids(stream):
    if stream.is_public():
        return public_stream_user_ids(stream)
    else:
        return private_stream_user_ids(stream.id) | {user.id for user in stream.realm.get_admin_users_and_bots()} | get_user_ids_with_metadata_access_via_permission_groups(stream)

def can_access_stream_history(user_profile, stream):
    if user_profile.realm_id != stream.realm_id:
        raise AssertionError("user_profile and stream realms don't match")
    if stream.is_web_public:
        return True
    if stream.is_history_realm_public() and (not user_profile.is_guest):
        return True
    if stream.is_history_public_to_subscribers():
        error = _("Invalid channel name '{channel_name}'").format(channel_name=stream.name)
        try:
            access_stream_common(user_profile, stream, error)
        except JsonableError:
            return False
        return True
    return False

def can_access_stream_history_by_name(user_profile, stream_name):
    try:
        stream = get_stream(stream_name, user_profile.realm)
    except Stream.DoesNotExist:
        return False
    return can_access_stream_history(user_profile, stream)

def can_access_stream_history_by_id(user_profile, stream_id):
    try:
        stream = get_stream_by_id_in_realm(stream_id, user_profile.realm)
    except Stream.DoesNotExist:
        return False
    return can_access_stream_history(user_profile, stream)

def bulk_can_remove_subscribers_from_streams(streams, user_profile):
    if user_profile.is_realm_admin:
        return True
    if user_profile.is_guest:
        return False
    user_recursive_group_ids = set(get_recursive_membership_groups(user_profile).values_list('id', flat=True))
    permission_failure_streams: Set[int] = set()
    for stream in streams:
        if not is_user_in_can_administer_channel_group(stream, user_recursive_group_ids):
            permission_failure_streams.add(stream.id)
    if not bool(permission_failure_streams):
        return True
    existing_recipient_ids = [stream.recipient_id for stream in streams]
    sub_recipient_ids = Subscription.objects.filter(user_profile=user_profile, recipient_id__in=existing_recipient_ids, active=True).values_list('recipient_id', flat=True)
    for stream in streams:
        assert stream.recipient_id is not None
        is_subscribed = stream.recipient_id in sub_recipient_ids
        if not check_basic_stream_access(user_profile, stream, is_subscribed=is_subscribed, require_content_access=False):
            return False
    for stream in streams:
        if not is_user_in_can_remove_subscribers_group(stream, user_recursive_group_ids):
            return False
    return True

def get_streams_to_which_user_cannot_add_subscribers(streams, user_profile, *, allow_default_streams: bool=False):
    result: List[Stream] = []
    if user_profile.can_subscribe_others_to_all_accessible_streams():
        return []
    if user_profile.is_realm_admin:
        return []
    user_recursive_group_ids = set(get_recursive_membership_groups(user_profile).values_list('id', flat=True))
    if allow_default_streams:
        default_stream_ids = get_default_stream_ids_for_realm(user_profile.realm_id)
    for stream in streams:
        if user_profile.is_guest:
            result.append(stream)
            continue
        if allow_default_streams and stream.id in default_stream_ids:
            continue
        if is_user_in_can_administer_channel_group(stream, user_recursive_group_ids):
            continue
        if not is_user_in_can_add_subscribers_group(stream, user_recursive_group_ids):
            result.append(stream)
    return result

def can_administer_accessible_channel(channel, user_profile):
    group_allowed_to_administer_channel = channel.can_administer_channel_group
    assert group_allowed_to_administer_channel is not None
    return user_has_permission_for_group_setting(group_allowed_to_administer_channel, user_profile, Stream.stream_permission_group_settings['can_administer_channel_group'])

@dataclass
class StreamsCategorizedByPermissions:
    authorized_streams: List[Stream]
    unauthorized_streams: List[Stream]
    streams_to_which_user_cannot_add_subscribers: List[Stream]

def filter_stream_authorization(user_profile, streams, is_subscribing_other_users=False):
    if len(streams) == 0:
        return StreamsCategorizedByPermissions(authorized_streams=[], unauthorized_streams=[], streams_to_which_user_cannot_add_subscribers=[])
    recipient_ids = [stream.recipient_id for stream in streams]
    subscribed_recipient_ids = set(Subscription.objects.filter(user_profile=user_profile, recipient_id__in=recipient_ids, active=True).values_list('recipient_id', flat=True))
    unauthorized_streams: List[Stream] = []
    streams_to_which_user_cannot_add_subscribers: List[Stream] = []
    if is_subscribing_other_users:
        streams_to_which_user_cannot_add_subscribers = get_streams_to_which_user_cannot_add_subscribers(list(streams), user_profile)
    for stream in streams:
        if stream.deactivated:
            unauthorized_streams.append(stream)
            continue
        if stream.recipient_id in subscribed_recipient_ids:
            continue
        if stream.is_web_public:
            continue
        if user_profile.is_guest:
            unauthorized_streams.append(stream)
            continue
        if not stream.invite_only:
            continue
        if stream.invite_only:
            user_recursive_group_ids = set(get_recursive_membership_groups(user_profile).values_list('id', flat=True))
            if is_user_in_can_add_subscribers_group(stream, user_recursive_group_ids):
                continue
        unauthorized_streams.append(stream)
    authorized_streams = [stream for stream in streams if stream.id not in {stream.id for stream in unauthorized_streams} and stream.id not in {stream.id for stream in streams_to_which_user_cannot_add_subscribers}]
    return StreamsCategorizedByPermissions(authorized_streams=authorized_streams, unauthorized_streams=unauthorized_streams, streams_to_which_user_cannot_add_subscribers=streams_to_which_user_cannot_add_subscribers)

def list_to_streams(streams_raw, user_profile, autocreate=False, unsubscribing_others=False, is_default_stream=False, setting_groups_dict=None):
    stream_set = {stream_dict['name'] for stream_dict in streams_raw}
    for stream_name in stream_set:
        assert stream_name == stream_name.strip()
        check_stream_name(stream_name)
    existing_streams: List[Stream] = []
    missing_stream_dicts: List[StreamDict] = []
    existing_stream_map = bulk_get_streams(user_profile.realm, stream_set)
    if unsubscribing_others and (not bulk_can_remove_subscribers_from_streams(list(existing_stream_map.values()), user_profile)):
        raise JsonableError(_('Insufficient permission'))
    message_retention_days_not_none = False
    web_public_stream_requested = False
    for stream_dict in streams_raw:
        stream_name = stream_dict['name']
        stream = existing_stream_map.get(stream_name.lower())
        if stream is None:
            if stream_dict.get('message_retention_days', None) is not None:
                message_retention_days_not_none = True
            missing_stream_dicts.append(stream_dict)
            if autocreate and stream_dict['is_web_public']:
                web_public_stream_requested = True
        else:
            existing_streams.append(stream)
    if len(missing_stream_dicts) == 0:
        created_streams: List[Stream] = []
    else:
        for stream_dict in missing_stream_dicts:
            invite_only = stream_dict.get('invite_only', False)
            if invite_only and (not user_profile.can_create_private_streams()):
                raise JsonableError(_('Insufficient permission'))
            if not invite_only and (not user_profile.can_create_public_streams()):
                raise JsonableError(_('Insufficient permission'))
            if is_default_stream and (not user_profile.is_realm_admin):
                raise JsonableError(_('Insufficient permission'))
            if invite_only and is_default_stream:
                raise JsonableError(_('A default channel cannot be private.'))
        if not autocreate:
            raise JsonableError(_('Channel(s) ({channel_names}) do not exist').format(channel_names=', '.join((stream_dict['name'] for stream_dict in missing_stream_dicts))))
        if web_public_stream_requested:
            if not user_profile.realm.web_public_streams_enabled():
                raise JsonableError(_('Web-public channels are not enabled.'))
            if not user_profile.can_create_web_public_streams():
                raise JsonableError(_('Insufficient permission'))
        if message_retention_days_not_none:
            if not user_profile.is_realm_owner:
                raise OrganizationOwnerRequiredError
            user_profile
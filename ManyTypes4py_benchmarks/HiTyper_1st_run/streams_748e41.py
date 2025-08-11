from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypedDict
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
    """
    This type ultimately gets used in two places:

        - we use it to create a stream
        - we use it to specify a stream


    It's possible we want a smaller type to use
    for removing streams, but it would complicate
    how we write the types for list_to_stream.

    Note that these fields are just a subset of
    the fields in the Stream model.
    """

def get_stream_permission_policy_name(*, invite_only: Union[None, bool, typing.Iterable[str], django.http.HttpRequest]=None, history_public_to_subscribers: Union[None, bool, typing.Iterable[str], django.http.HttpRequest]=None, is_web_public: Union[None, bool, typing.Iterable[str], django.http.HttpRequest]=None) -> None:
    policy_name = None
    for permission_dict in Stream.PERMISSION_POLICIES.values():
        if permission_dict['invite_only'] == invite_only and permission_dict['history_public_to_subscribers'] == history_public_to_subscribers and (permission_dict['is_web_public'] == is_web_public):
            policy_name = permission_dict['policy_name']
            break
    assert policy_name is not None
    return policy_name

def get_default_value_for_history_public_to_subscribers(realm: Union[zerver.models.Realm, bool, grouper.models.base.session.Session], invite_only: Union[bool, grouper.models.base.session.Session, list[str]], history_public_to_subscribers: Union[bool, zerver.models.Stream, None]) -> Union[bool, str, dict[int, typing.Any]]:
    if invite_only:
        if history_public_to_subscribers is None:
            history_public_to_subscribers = False
    else:
        history_public_to_subscribers = True
    if realm.is_zephyr_mirror_realm:
        history_public_to_subscribers = False
    return history_public_to_subscribers

def render_stream_description(text: Union[zerver.models.Realm, zerver.models.UserProfile, zerver.models.RealmEmoji], realm: Union[zerver.models.Realm, zerver.models.UserProfile, zerver.models.RealmEmoji], *, acting_user: Union[None, zerver.models.Realm, zerver.models.UserProfile, zerver.models.RealmEmoji]=None):
    from zerver.lib.markdown import markdown_convert
    return markdown_convert(text, message_realm=realm, no_previews=True, acting_user=acting_user).rendered_content

def send_stream_creation_event(realm: Union[int, set[int], zerver.models.Realm, None], stream: Union[dict[str, typing.Any], bool, dict[int, zam_repondeur.models.Amendement]], user_ids: Union[int, set[int], zerver.models.Realm, None], recent_traffic: Union[None, dict[str, typing.Any], bool, dict[int, zam_repondeur.models.Amendement]]=None, setting_groups_dict: Union[None, dict[str, typing.Any], bool, dict[int, zam_repondeur.models.Amendement]]=None) -> None:
    event = dict(type='stream', op='create', streams=[stream_to_dict(stream, recent_traffic, setting_groups_dict)])
    send_event_on_commit(realm, event, user_ids)

def get_stream_permission_default_group(setting_name: Union[str, None, zerver.models.Realm], system_groups_name_dict: str, creator: Union[None, str, zerver.models.UserProfile, users.models.JustfixUser]=None) -> Union[UserGroup, str]:
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

def get_default_values_for_stream_permission_group_settings(realm: Union[zerver.models.Realm, zerver.models.Subscription, zerver.models.Stream], creator: Union[None, str, zerver.models.UserProfile, Realm]=None) -> dict:
    group_setting_values = {}
    system_groups_name_dict = get_role_based_system_groups_dict(realm)
    for setting_name in Stream.stream_permission_group_settings:
        group_setting_values[setting_name] = get_stream_permission_default_group(setting_name, system_groups_name_dict, creator)
    return group_setting_values

def get_user_ids_with_metadata_access_via_permission_groups(stream: Union[zerver.models.Stream, list[dict[str, typing.Any]]]) -> set:
    return set(get_recursive_group_members_union_for_groups([stream.can_add_subscribers_group_id, stream.can_administer_channel_group_id]).exclude(role=UserProfile.ROLE_GUEST).values_list('id', flat=True))

@transaction.atomic(savepoint=False)
def create_stream_if_needed(realm: Union[bool, str, django.http.HttpRequest], stream_name: str, *, invite_only: bool=False, is_web_public: bool=False, history_public_to_subscribers: Union[None, str, bool, zerver.models.Realm]=None, stream_description: typing.Text='', message_retention_days: Union[None, str, bool]=None, can_add_subscribers_group: Union[None, bool, django.http.HttpRequest]=None, can_administer_channel_group: Union[None, bool, django.http.HttpRequest]=None, can_send_message_group: Union[None, bool, django.http.HttpRequest]=None, can_remove_subscribers_group: Union[None, bool, django.http.HttpRequest]=None, acting_user: Union[None, str, bool, zerver.models.Realm]=None, setting_groups_dict: Union[None, zerver.models.UserProfile, zerver.models.Realm, bool]=None) -> tuple[typing.Union[typing.BinaryIO,zerver.models.Stream,dcpquery.db.models.File,datetime.datetime,datetime.date,str]]:
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

def create_streams_if_needed(realm: Union[zerver.models.Attachment, zerver.models.UserProfile, zilencer.models.RemoteZulipServer, bool], stream_dicts: zerver.models.UserProfile, acting_user: Union[None, zerver.models.Attachment, zerver.models.UserProfile, zilencer.models.RemoteZulipServer, bool]=None, setting_groups_dict: Union[None, zerver.models.Attachment, zerver.models.UserProfile, zilencer.models.RemoteZulipServer, bool]=None) -> tuple[list]:
    """Note that stream_dict["name"] is assumed to already be stripped of
    whitespace"""
    added_streams = []
    existing_streams = []
    for stream_dict in stream_dicts:
        invite_only = stream_dict.get('invite_only', False)
        stream, created = create_stream_if_needed(realm, stream_dict['name'], invite_only=invite_only, is_web_public=stream_dict.get('is_web_public', False), history_public_to_subscribers=stream_dict.get('history_public_to_subscribers'), stream_description=stream_dict.get('description', ''), message_retention_days=stream_dict.get('message_retention_days', None), can_add_subscribers_group=stream_dict.get('can_add_subscribers_group', None), can_administer_channel_group=stream_dict.get('can_administer_channel_group', None), can_send_message_group=stream_dict.get('can_send_message_group', None), can_remove_subscribers_group=stream_dict.get('can_remove_subscribers_group', None), acting_user=acting_user, setting_groups_dict=setting_groups_dict)
        if created:
            added_streams.append(stream)
        else:
            existing_streams.append(stream)
    return (added_streams, existing_streams)

def subscribed_to_stream(user_profile: Union[zerver.models.UserProfile, int, Realm], stream_id: Union[zerver.models.UserProfile, int, Realm]) -> Union[bool, set[str]]:
    return Subscription.objects.filter(user_profile=user_profile, active=True, recipient__type=Recipient.STREAM, recipient__type_id=stream_id).exists()

def is_user_in_can_administer_channel_group(stream: Union[zerver.models.UserProfile, int, zerver.models.Recipient], user_recursive_group_ids: Union[Realm, zerver.models.UserProfile, str]) -> bool:
    group_allowed_to_administer_channel_id = stream.can_administer_channel_group_id
    assert group_allowed_to_administer_channel_id is not None
    return group_allowed_to_administer_channel_id in user_recursive_group_ids

def is_user_in_can_add_subscribers_group(stream: Union[str, zerver.models.UserProfile, int], user_recursive_group_ids: Union[str, zerver.models.Realm, daylighdb.models.User]) -> bool:
    group_allowed_to_add_subscribers_id = stream.can_add_subscribers_group_id
    assert group_allowed_to_add_subscribers_id is not None
    return group_allowed_to_add_subscribers_id in user_recursive_group_ids

def is_user_in_can_remove_subscribers_group(stream: Union[str, zerver.models.UserProfile, Realm], user_recursive_group_ids: Union[str, zerver.models.Realm, None]) -> bool:
    group_allowed_to_remove_subscribers_id = stream.can_remove_subscribers_group_id
    assert group_allowed_to_remove_subscribers_id is not None
    return group_allowed_to_remove_subscribers_id in user_recursive_group_ids

def check_stream_access_based_on_can_send_message_group(sender: Union[zerver.models.Realm, zerver.models.UserProfile], stream: Union[zerver.models.Stream, tracim.models.auth.User, list[zerver.data_imporimport_util.ZerverFieldsT]]) -> None:
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

def access_stream_for_send_message(sender: Union[zerver.models.UserProfile, str, asgard.models.accounAccount], stream: Union[zerver.models.UserProfile, str, zerver.models.Stream], forwarder_user_profile: Union[zerver.models.UserProfile, None, accounts.models.Account], archived_channel_notice: bool=False) -> None:
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

def check_for_exactly_one_stream_arg(stream_id: Union[str, dict[str, typing.Any], None, bytes], stream: Union[str, dict[str, typing.Any], None, bytes]) -> None:
    if stream_id is None and stream is None:
        error = _("Missing '{var_name}' argument").format(var_name='stream_id')
        raise JsonableError(error)
    if stream_id is not None and stream is not None:
        raise IncompatibleParametersError(['stream_id', 'stream'])

@dataclass
class UserGroupMembershipDetails:
    pass

def user_has_content_access(user_profile: Union[bool, zerver.models.UserProfile, None], stream: Union[bool, zerver.models.UserProfile, zerver.models.Stream], user_group_membership_details: Union[bool, zerver.models.UserProfile, None, zerver.models.Stream], *, is_subscribed: Union[bool, list[zerver.models.DefaultStreamGroup]]) -> bool:
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

def check_stream_access_for_delete_or_update_requiring_metadata_access(user_profile: Union[zerver.models.UserProfile, zerver.models.Realm, int, None], stream: Union[zerver.models.UserProfile, zerver.models.Realm], sub: Union[None, zerver.models.UserProfile, zerver.models.Realm]=None) -> None:
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

def access_stream_for_delete_or_update_requiring_metadata_access(user_profile: Union[zerver.models.UserProfile, bool, zerver.models.Realm], stream_id: Union[int, zerver.models.UserProfile, Realm]) -> tuple[typing.Union[list,str,qutebrowser.utils.objreg.ObjectRegistry,dict[str, typing.Any],None]]:
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

def has_metadata_access_to_channel_via_groups(user_profile: Union[int, zerver.models.UserProfile, None], user_recursive_group_ids: zerver.models.UserProfile, can_administer_channel_group_id: Union[bool, zerver.models.UserProfile, None], can_add_subscribers_group_id: Union[bool, zerver.models.UserProfile, None]) -> bool:
    for setting_name in Stream.stream_permission_group_settings_granting_metadata_access:
        permission_configuration = Stream.stream_permission_group_settings[setting_name]
        if not permission_configuration.allow_everyone_group and user_profile.is_guest:
            return False
    return can_administer_channel_group_id in user_recursive_group_ids or can_add_subscribers_group_id in user_recursive_group_ids

def check_basic_stream_access(user_profile: Union[bool, str, zerver.models.Realm, None], stream: Union[bool, str, zerver.models.Realm, None], *, is_subscribed: Union[bool, zerver.models.Realm, None], require_content_access: bool=True) -> bool:
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

def access_stream_common(user_profile: Union[bool, zerver.models.UserProfile, str], stream: Union[bool, list[dict[str, typing.Any]], zerver.models.UserProfile, None], error: Union[str, bool, list[str]], require_active: bool=True, require_content_access: bool=True) -> Union[frozenset, tuple[int], list[tuple[typing.Any]], None]:
    """Common function for backend code where the target use attempts to
    access the target stream, returning all the data fetched along the
    way.  If that user does not have permission to access that stream,
    we throw an exception.  A design goal is that the error message is
    the same for streams you can't access and streams that don't exist."""
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

def access_stream_by_id(user_profile: Union[bool, str, zerver.models.UserProfile, None], stream_id: Union[bool, str], require_active: bool=True, require_content_access: bool=True) -> tuple:
    error = _('Invalid channel ID')
    try:
        stream = get_stream_by_id_in_realm(stream_id, user_profile.realm)
    except Stream.DoesNotExist:
        raise JsonableError(error)
    sub = access_stream_common(user_profile, stream, error, require_active=require_active, require_content_access=require_content_access)
    return (stream, sub)

def access_stream_by_id_for_message(user_profile: Union[bool, str], stream_id: Union[bool, str], require_active: bool=True, require_content_access: bool=True) -> tuple:
    """
    Variant of access_stream_by_id that uses get_stream_by_id_for_sending_message
    to ensure we do a select_related("can_send_message_group").
    """
    error = _('Invalid channel ID')
    try:
        stream = get_stream_by_id_for_sending_message(stream_id, user_profile.realm)
    except Stream.DoesNotExist:
        raise JsonableError(error)
    sub = access_stream_common(user_profile, stream, error, require_active=require_active, require_content_access=require_content_access)
    return (stream, sub)

def get_public_streams_queryset(realm: Union[zerver.models.Realm, None]) -> Union[cmk.utils.type_defs.UserId, None, bool, django.http.HttpRequest]:
    return Stream.objects.filter(realm=realm, invite_only=False, history_public_to_subscribers=True)

def get_web_public_streams_queryset(realm: Union[zerver.models.Realm, None]) -> Union[zerver.models.Realm, None, bool, str]:
    return Stream.objects.filter(realm=realm, is_web_public=True, deactivated=False, invite_only=False, history_public_to_subscribers=True).select_related('can_send_message_group', 'can_send_message_group__named_user_group')

def check_stream_name_available(realm: Union[zerver.models.Realm, str, None], name: Union[str, zerver.models.Realm, cmk.utils.type_defs.UserId]) -> None:
    check_stream_name(name)
    try:
        get_stream(name, realm)
        raise JsonableError(_('Channel name already in use.'))
    except Stream.DoesNotExist:
        pass

def access_stream_by_name(user_profile: Union[str, zerver.models.UserProfile, zerver.models.Realm, None], stream_name: Union[str, zerver.models.Stream, None, bool], require_content_access: bool=True) -> tuple:
    error = _("Invalid channel name '{channel_name}'").format(channel_name=stream_name)
    try:
        stream = get_realm_stream(stream_name, user_profile.realm_id)
    except Stream.DoesNotExist:
        raise JsonableError(error)
    sub = access_stream_common(user_profile, stream, error, require_content_access=require_content_access)
    return (stream, sub)

def access_web_public_stream(stream_id: Union[int, Realm, None], realm: Union[int, Realm, None]) -> Union[typing.TextIO, list, str]:
    error = _('Invalid channel ID')
    try:
        stream = get_stream_by_id_in_realm(stream_id, realm)
    except Stream.DoesNotExist:
        raise JsonableError(error)
    if not stream.is_web_public:
        raise JsonableError(error)
    return stream

def access_stream_to_remove_visibility_policy_by_name(user_profile: Union[str, None], stream_name: Union[str, None], error: Union[str, None, dict[str, str]]):
    """
    It may seem a little silly to have this helper function for unmuting
    topics, but it gets around a linter warning, and it helps to be able
    to review all security-related stuff in one place.

    Our policy for accessing streams when you unmute a topic is that you
    don't necessarily need to have an active subscription or even "legal"
    access to the stream.  Instead, we just verify the stream_id has been
    muted in the past (not here, but in the caller).

    Long term, we'll probably have folks just pass us in the id of the
    UserTopic row to unmute topics.
    """
    try:
        stream = get_stream(stream_name, user_profile.realm)
    except Stream.DoesNotExist:
        raise JsonableError(error)
    return stream

def access_stream_to_remove_visibility_policy_by_id(user_profile: Union[int, zerver.models.UserProfile, str], stream_id: Union[int, zerver.models.UserProfile, str], error: Union[str, zerver.models.UserProfile, dict[str, typing.Any]]) -> Union[str, typing.TextIO, int]:
    try:
        stream = Stream.objects.get(id=stream_id, realm_id=user_profile.realm_id)
    except Stream.DoesNotExist:
        raise JsonableError(error)
    return stream

def private_stream_user_ids(stream_id: Union[int, None]) -> set:
    subscriptions = get_active_subscriptions_for_stream_id(stream_id, include_deactivated_users=False)
    return {sub['user_profile_id'] for sub in subscriptions.values('user_profile_id')}

def public_stream_user_ids(stream: Union[zerver.models.Stream, bool, dict]) -> Union[set, int]:
    guest_subscriptions = get_active_subscriptions_for_stream_id(stream.id, include_deactivated_users=False).filter(user_profile__role=UserProfile.ROLE_GUEST)
    guest_subscriptions_ids = {sub['user_profile_id'] for sub in guest_subscriptions.values('user_profile_id')}
    can_add_subscribers_group_user_ids = set(get_recursive_group_members(stream.can_add_subscribers_group_id).exclude(role=UserProfile.ROLE_GUEST).values_list('id', flat=True))
    return set(active_non_guest_user_ids(stream.realm_id)) | guest_subscriptions_ids | can_add_subscribers_group_user_ids

def can_access_stream_metadata_user_ids(stream: Union[zerver.models.Stream, str, typing.IO]) -> Union[str, bytes, bool]:
    if stream.is_public():
        return public_stream_user_ids(stream)
    else:
        return private_stream_user_ids(stream.id) | {user.id for user in stream.realm.get_admin_users_and_bots()} | get_user_ids_with_metadata_access_via_permission_groups(stream)

def can_access_stream_history(user_profile: Union[zerver.models.UserProfile, bool, None], stream: Union[zerver.models.UserProfile, bool, zerver.models.Message]) -> bool:
    """Determine whether the provided user is allowed to access the
    history of the target stream.

    This is used by the caller to determine whether this user can get
    historical messages before they joined for a narrowing search.

    Because of the way our search is currently structured,
    we may be passed an invalid stream here.  We return
    False in that situation, and subsequent code will do
    validation and raise the appropriate JsonableError.

    Note that this function should only be used in contexts where
    access_stream is being called elsewhere to confirm that the user
    can actually see this stream.
    """
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

def can_access_stream_history_by_name(user_profile: Union[str, zerver.models.UserProfile, Realm], stream_name: Union[str, zerver.models.UserProfile, Realm]) -> bool:
    try:
        stream = get_stream(stream_name, user_profile.realm)
    except Stream.DoesNotExist:
        return False
    return can_access_stream_history(user_profile, stream)

def can_access_stream_history_by_id(user_profile: Union[int, zerver.models.UserProfile, Realm], stream_id: Union[int, zerver.models.UserProfile, Realm]) -> bool:
    try:
        stream = get_stream_by_id_in_realm(stream_id, user_profile.realm)
    except Stream.DoesNotExist:
        return False
    return can_access_stream_history(user_profile, stream)

def bulk_can_remove_subscribers_from_streams(streams: Union[zerver.models.Stream, zerver.models.Realm, zerver.models.UserProfile], user_profile: Union[zerver.models.UserProfile, list[zerver.models.UserProfile]]) -> bool:
    if user_profile.is_realm_admin:
        return True
    if user_profile.is_guest:
        return False
    user_recursive_group_ids = set(get_recursive_membership_groups(user_profile).values_list('id', flat=True))
    permission_failure_streams = set()
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

def get_streams_to_which_user_cannot_add_subscribers(streams: Union[zerver.models.Client, set[str]], user_profile: Union[zerver.models.UserProfile, bool, zerver.models.Realm, None], *, allow_default_streams: bool=False) -> Union[list, list[str]]:
    result = []
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

def can_administer_accessible_channel(channel: Union[zerver.models.UserProfile, bool, zerver.models.UserGroup], user_profile: Union[zerver.models.UserProfile, accounts.models.Account, django.contrib.auth.models.AbstractUser]):
    group_allowed_to_administer_channel = channel.can_administer_channel_group
    assert group_allowed_to_administer_channel is not None
    return user_has_permission_for_group_setting(group_allowed_to_administer_channel, user_profile, Stream.stream_permission_group_settings['can_administer_channel_group'])

@dataclass
class StreamsCategorizedByPermissions:
    pass

def filter_stream_authorization(user_profile: Union[bool, zerver.models.Realm, None, django.db.models.query.QuerySet], streams: Union[django.contrib.admin.ModelAdmin, list[dict[str, typing.Any]], hackathon.utils.utils.ResultsMessage], is_subscribing_other_users: bool=False) -> StreamsCategorizedByPermissions:
    if len(streams) == 0:
        return StreamsCategorizedByPermissions(authorized_streams=[], unauthorized_streams=[], streams_to_which_user_cannot_add_subscribers=[])
    recipient_ids = [stream.recipient_id for stream in streams]
    subscribed_recipient_ids = set(Subscription.objects.filter(user_profile=user_profile, recipient_id__in=recipient_ids, active=True).values_list('recipient_id', flat=True))
    unauthorized_streams = []
    streams_to_which_user_cannot_add_subscribers = []
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

def list_to_streams(streams_raw: zerver.models.UserProfile, user_profile: Union[bool, zerver.models.Realm, None, zerver.models.UserProfile], autocreate: bool=False, unsubscribing_others: bool=False, is_default_stream: bool=False, setting_groups_dict: Union[None, bool, zerver.models.Realm]=None) -> tuple[list[typing.Union[str,list,None]]]:
    """Converts list of dicts to a list of Streams, validating input in the process

    For each stream name, we validate it to ensure it meets our
    requirements for a proper stream name using check_stream_name.

    This function in autocreate mode should be atomic: either an exception will be raised
    during a precheck, or all the streams specified will have been created if applicable.

    @param streams_raw The list of stream dictionaries to process;
      names should already be stripped of whitespace by the caller.
    @param user_profile The user for whom we are retrieving the streams
    @param autocreate Whether we should create streams if they don't already exist
    """
    stream_set = {stream_dict['name'] for stream_dict in streams_raw}
    for stream_name in stream_set:
        assert stream_name == stream_name.strip()
        check_stream_name(stream_name)
    existing_streams = []
    missing_stream_dicts = []
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
        created_streams = []
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
            user_profile.realm.ensure_not_on_limited_plan()
        created_streams, dup_streams = create_streams_if_needed(realm=user_profile.realm, stream_dicts=missing_stream_dicts, acting_user=user_profile, setting_groups_dict=setting_groups_dict)
        existing_streams += dup_streams
    return (existing_streams, created_streams)

def access_default_stream_group_by_id(realm: Union[zerver.models.Realm, int, set[int]], group_id: Union[int, str, zerver.models.Realm]) -> Union[users.models.User, None, zerver.models.Realm, dict[str, typing.Any]]:
    try:
        return DefaultStreamGroup.objects.get(realm=realm, id=group_id)
    except DefaultStreamGroup.DoesNotExist:
        raise JsonableError(_("Default channel group with id '{group_id}' does not exist.").format(group_id=group_id))

def get_stream_by_narrow_operand_access_unchecked(operand: Union[str, zerver.models.Realm], realm: Union[str, zerver.models.Recipient, zerver.models.UserProfile]) -> Union[str, bool, tildes.models.user.User]:
    """This is required over access_stream_* in certain cases where
    we need the stream data only to prepare a response that user can access
    and not send it out to unauthorized recipients.
    """
    if isinstance(operand, str):
        return get_stream(operand, realm)
    return get_stream_by_id_in_realm(operand, realm)

def ensure_stream(realm: Union[str, bool, zerver.models.Realm], stream_name: Union[str, bool, zerver.models.Realm], invite_only: bool=False, stream_description: typing.Text='', *, acting_user: Union[str, bool, zerver.models.Realm]) -> Union[str, dict, int]:
    return create_stream_if_needed(realm, stream_name, invite_only=invite_only, stream_description=stream_description, acting_user=acting_user)[0]

def get_occupied_streams(realm: Union[zerver.models.Realm, UserProfile]) -> Union[list[str], str, typing.Callable[None,None,None,None, typing.Any]]:
    """Get streams with subscribers"""
    exists_expression = Exists(Subscription.objects.filter(active=True, is_user_active=True, user_profile__realm=realm, recipient_id=OuterRef('recipient_id')))
    occupied_streams = Stream.objects.filter(realm=realm, deactivated=False).alias(occupied=exists_expression).filter(occupied=True)
    return occupied_streams

def get_stream_post_policy_value_based_on_group_setting(setting_group: Union[models.Organization, str]):
    if hasattr(setting_group, 'named_user_group') and setting_group.named_user_group.is_system_group:
        group_name = setting_group.named_user_group.name
        if group_name in Stream.SYSTEM_GROUPS_ENUM_MAP:
            return Stream.SYSTEM_GROUPS_ENUM_MAP[group_name]
    return Stream.STREAM_POST_POLICY_EVERYONE

def stream_to_dict(stream: Union[typing.DefaultDict, dict, int], recent_traffic: Union[None, bool]=None, setting_groups_dict: Union[None, str, dict, zerver.models.Realm]=None) -> APIStreamDict:
    if recent_traffic is not None:
        stream_weekly_traffic = get_average_weekly_stream_traffic(stream.id, stream.date_created, recent_traffic)
    else:
        stream_weekly_traffic = None
    assert setting_groups_dict is not None
    can_add_subscribers_group = setting_groups_dict[stream.can_add_subscribers_group_id]
    can_administer_channel_group = setting_groups_dict[stream.can_administer_channel_group_id]
    can_send_message_group = setting_groups_dict[stream.can_send_message_group_id]
    can_remove_subscribers_group = setting_groups_dict[stream.can_remove_subscribers_group_id]
    stream_post_policy = get_stream_post_policy_value_based_on_group_setting(stream.can_send_message_group)
    return APIStreamDict(is_archived=stream.deactivated, can_add_subscribers_group=can_add_subscribers_group, can_administer_channel_group=can_administer_channel_group, can_send_message_group=can_send_message_group, can_remove_subscribers_group=can_remove_subscribers_group, creator_id=stream.creator_id, date_created=datetime_to_timestamp(stream.date_created), description=stream.description, first_message_id=stream.first_message_id, is_recently_active=stream.is_recently_active, history_public_to_subscribers=stream.history_public_to_subscribers, invite_only=stream.invite_only, is_web_public=stream.is_web_public, message_retention_days=stream.message_retention_days, name=stream.name, rendered_description=stream.rendered_description, stream_id=stream.id, stream_post_policy=stream_post_policy, is_announcement_only=stream_post_policy == Stream.STREAM_POST_POLICY_ADMINS, stream_weekly_traffic=stream_weekly_traffic)

def get_web_public_streams(realm: Union[zerver.models.Realm, UserProfile]) -> list:
    query = get_web_public_streams_queryset(realm)
    streams = query.only(*Stream.API_FIELDS)
    setting_groups_dict = get_group_setting_value_dict_for_streams(list(streams))
    stream_dicts = [stream_to_dict(stream, None, setting_groups_dict) for stream in streams]
    return stream_dicts

def get_streams_for_user(user_profile: Union[bool, zerver.models.UserProfile, zerver.models.Realm, None], include_public: bool=True, include_web_public: bool=False, include_subscribed: bool=True, exclude_archived: bool=True, include_all_active: bool=False, include_owner_subscribed: bool=False) -> list:
    if include_all_active and (not user_profile.is_realm_admin):
        raise JsonableError(_('User not authorized for this query'))
    include_public = include_public and user_profile.can_access_public_streams()
    query = Stream.objects.select_related('can_send_message_group', 'can_send_message_group__named_user_group').filter(realm=user_profile.realm)
    if exclude_archived:
        query = query.filter(deactivated=False)
    if include_all_active:
        streams = query.only(*Stream.API_FIELDS, 'can_send_message_group', 'can_send_message_group__named_user_group')
    else:
        query_filter = None

        def add_filter_option(option: Any) -> None:
            nonlocal query_filter
            if query_filter is None:
                query_filter = option
            else:
                query_filter |= option
        if include_subscribed:
            subscribed_stream_ids = get_subscribed_stream_ids_for_user(user_profile)
            recipient_check = Q(id__in=set(subscribed_stream_ids))
            add_filter_option(recipient_check)
        if include_public:
            invite_only_check = Q(invite_only=False)
            add_filter_option(invite_only_check)
        if include_web_public:
            web_public_check = Q(is_web_public=True, invite_only=False, history_public_to_subscribers=True, deactivated=False)
            add_filter_option(web_public_check)
        if include_owner_subscribed and user_profile.is_bot:
            bot_owner = user_profile.bot_owner
            assert bot_owner is not None
            owner_stream_ids = get_subscribed_stream_ids_for_user(bot_owner)
            owner_subscribed_check = Q(id__in=set(owner_stream_ids))
            add_filter_option(owner_subscribed_check)
        if query_filter is not None:
            query = query.filter(query_filter)
            streams = query.only(*Stream.API_FIELDS)
        else:
            return []
    return list(streams)

def get_group_setting_value_dict_for_streams(streams: Union[types.streams.StreamT, typing.IO, str]) -> Union[str, list[str]]:
    setting_group_ids = set()
    for stream in streams:
        for setting_name in Stream.stream_permission_group_settings:
            setting_group_ids.add(getattr(stream, setting_name + '_id'))
    return get_setting_values_for_group_settings(list(setting_group_ids))

def get_setting_values_for_group_settings(group_ids: int) -> dict:
    user_groups = UserGroup.objects.filter(id__in=group_ids).select_related('named_user_group')
    setting_groups_dict = dict()
    anonymous_group_ids = []
    for group in user_groups:
        if hasattr(group, 'named_user_group'):
            setting_groups_dict[group.id] = group.id
        else:
            anonymous_group_ids.append(group.id)
    if len(anonymous_group_ids) == 0:
        return setting_groups_dict
    user_members = UserGroupMembership.objects.filter(user_group_id__in=anonymous_group_ids).annotate(member_type=Value('user')).values_list('member_type', 'user_group_id', 'user_profile_id')
    group_subgroups = GroupGroupMembership.objects.filter(supergroup_id__in=anonymous_group_ids).annotate(member_type=Value('group')).values_list('member_type', 'supergroup_id', 'subgroup_id')
    all_members = user_members.union(group_subgroups)
    for member_type, group_id, member_id in all_members:
        if group_id not in setting_groups_dict:
            setting_groups_dict[group_id] = AnonymousSettingGroupDict(direct_members=[], direct_subgroups=[])
        anonymous_group_dict = setting_groups_dict[group_id]
        assert isinstance(anonymous_group_dict, AnonymousSettingGroupDict)
        if member_type == 'user':
            anonymous_group_dict.direct_members.append(member_id)
        else:
            anonymous_group_dict.direct_subgroups.append(member_id)
    return setting_groups_dict

def do_get_streams(user_profile: Union[bool, typing.Iterable[str], None, zerver.models.Realm], include_public: bool=True, include_web_public: bool=False, include_subscribed: bool=True, exclude_archived: bool=True, include_all_active: bool=False, include_default: bool=False, include_owner_subscribed: bool=False) -> list:
    streams = get_streams_for_user(user_profile, include_public, include_web_public, include_subscribed, exclude_archived, include_all_active, include_owner_subscribed)
    stream_ids = {stream.id for stream in streams}
    recent_traffic = get_streams_traffic(stream_ids, user_profile.realm)
    setting_groups_dict = get_group_setting_value_dict_for_streams(streams)
    stream_dicts = sorted((stream_to_dict(stream, recent_traffic, setting_groups_dict) for stream in streams), key=lambda elt: elt['name'])
    if include_default:
        default_stream_ids = get_default_stream_ids_for_realm(user_profile.realm_id)
        for stream_dict in stream_dicts:
            stream_dict['is_default'] = stream_dict['stream_id'] in default_stream_ids
    return stream_dicts

def get_subscribed_private_streams_for_user(user_profile: Union[UserProfile, zerver.models.Realm]) -> Union[list[typing.Any], str, dict[str, int]]:
    exists_expression = Exists(Subscription.objects.filter(user_profile=user_profile, active=True, is_user_active=True, recipient_id=OuterRef('recipient_id')))
    subscribed_private_streams = Stream.objects.filter(realm=user_profile.realm, invite_only=True, deactivated=False).alias(subscribed=exists_expression).filter(subscribed=True)
    return subscribed_private_streams

def notify_stream_is_recently_active_update(stream: Union[dict, zerver.models.Stream, T], value: Union[dict, zerver.models.Stream, audiopyle.lib.models.file_meta.CompressedAudioFileMeta]) -> None:
    event = dict(type='stream', op='update', property='is_recently_active', value=value, stream_id=stream.id, name=stream.name)
    send_event_on_commit(stream.realm, event, can_access_stream_metadata_user_ids(stream))

@transaction.atomic(durable=True)
def update_stream_active_status_for_realm(realm: Union[int, zerver.models.Realm, zerver.models.UserProfile], date_days_ago: Union[int, zerver.models.Realm, zerver.models.UserProfile]):
    recent_messages_subquery = Message.objects.filter(date_sent__gte=date_days_ago, realm=realm, recipient__type=Recipient.STREAM, recipient__type_id=OuterRef('id'))
    streams_to_mark_inactive = Stream.objects.filter(~Exists(recent_messages_subquery), is_recently_active=True, realm=realm)
    for stream in streams_to_mark_inactive:
        notify_stream_is_recently_active_update(stream, False)
    count = streams_to_mark_inactive.update(is_recently_active=False)
    return count

def check_update_all_streams_active_status(days: Any=Stream.LAST_ACTIVITY_DAYS_BEFORE_FOR_ACTIVE) -> int:
    date_days_ago = timezone_now() - timedelta(days=days)
    count = 0
    for realm in Realm.objects.filter(deactivated=False):
        count += update_stream_active_status_for_realm(realm, date_days_ago)
    return count

def send_stream_deletion_event(realm: Union[int, zerver.models.UserProfile, set[int]], user_ids: Union[int, zerver.models.UserProfile, set[int]], streams: Union[set[int], zerver.models.UserProfile]) -> None:
    stream_deletion_event = dict(type='stream', op='delete', streams=[dict(stream_id=stream.id) for stream in streams], stream_ids=[stream.id for stream in streams])
    send_event_on_commit(realm, stream_deletion_event, user_ids)
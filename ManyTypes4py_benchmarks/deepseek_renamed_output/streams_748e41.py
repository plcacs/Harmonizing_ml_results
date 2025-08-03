from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypedDict, Optional, List, Dict, Set, Tuple, Union, Any, cast
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
    invite_only: bool
    is_web_public: bool
    history_public_to_subscribers: Optional[bool]
    description: str
    message_retention_days: Optional[int]
    can_add_subscribers_group: Optional[UserGroup]
    can_administer_channel_group: Optional[UserGroup]
    can_send_message_group: Optional[UserGroup]
    can_remove_subscribers_group: Optional[UserGroup]


def func_rqk89ff1(*, invite_only: bool, history_public_to_subscribers: bool,
    is_web_public: bool) -> str:
    policy_name = None
    for permission_dict in Stream.PERMISSION_POLICIES.values():
        if permission_dict['invite_only'] == invite_only and permission_dict[
            'history_public_to_subscribers'
            ] == history_public_to_subscribers and permission_dict[
            'is_web_public'] == is_web_public:
            policy_name = permission_dict['policy_name']
            break
    assert policy_name is not None
    return policy_name


def func_yszqg9ul(realm: Realm, invite_only: bool, history_public_to_subscribers: Optional[bool]) -> bool:
    if invite_only:
        if history_public_to_subscribers is None:
            history_public_to_subscribers = False
    else:
        history_public_to_subscribers = True
    if realm.is_zephyr_mirror_realm:
        history_public_to_subscribers = False
    return history_public_to_subscribers


def func_r4io5gyy(text: str, realm: Realm, *, acting_user: Optional[UserProfile] = None) -> str:
    from zerver.lib.markdown import markdown_convert
    return markdown_convert(text, message_realm=realm, no_previews=True,
        acting_user=acting_user).rendered_content


def func_lgnk2uge(realm: Realm, stream: Stream, user_ids: Collection[int], recent_traffic: Optional[Dict[int, int]] = None,
    setting_groups_dict: Optional[Dict[int, Union[int, AnonymousSettingGroupDict]]] = None) -> None:
    event = dict(type='stream', op='create', streams=[stream_to_dict(stream,
        recent_traffic, setting_groups_dict)])
    send_event_on_commit(realm, event, user_ids)


def func_fdxyczmc(setting_name: str, system_groups_name_dict: Dict[str, UserGroup], creator: Optional[UserProfile] = None) -> UserGroup:
    setting_default_name = Stream.stream_permission_group_settings[setting_name
        ].default_group_name
    if setting_default_name == 'stream_creator_or_nobody':
        if creator:
            default_group = UserGroup(realm=creator.realm)
            default_group.save()
            UserGroupMembership.objects.create(user_profile=creator,
                user_group=default_group)
            return default_group
        else:
            return system_groups_name_dict[SystemGroups.NOBODY]
    return system_groups_name_dict[setting_default_name]


def func_l3yim3gq(realm: Realm, creator: Optional[UserProfile] = None) -> Dict[str, UserGroup]:
    group_setting_values: Dict[str, UserGroup] = {}
    system_groups_name_dict = get_role_based_system_groups_dict(realm)
    for setting_name in Stream.stream_permission_group_settings:
        group_setting_values[setting_name] = func_fdxyczmc(setting_name,
            system_groups_name_dict, creator)
    return group_setting_values


def func_m92gjue2(stream: Stream) -> Set[int]:
    return set(get_recursive_group_members_union_for_groups([stream.
        can_add_subscribers_group_id, stream.
        can_administer_channel_group_id]).exclude(role=UserProfile.
        ROLE_GUEST).values_list('id', flat=True))


@transaction.atomic(savepoint=False)
def func_i2dgbetu(realm: Realm, stream_name: str, *, invite_only: bool = False, is_web_public: bool = False, 
    history_public_to_subscribers: Optional[bool] = None, stream_description: str = '',
    message_retention_days: Optional[int] = None, can_add_subscribers_group: Optional[UserGroup] = None,
    can_administer_channel_group: Optional[UserGroup] = None, can_send_message_group: Optional[UserGroup] = None,
    can_remove_subscribers_group: Optional[UserGroup] = None, acting_user: Optional[UserProfile] = None,
    setting_groups_dict: Optional[Dict[int, Union[int, AnonymousSettingGroupDict]]] = None) -> Tuple[Stream, bool]:
    history_public_to_subscribers = func_yszqg9ul(realm, invite_only,
        history_public_to_subscribers)
    group_setting_values: Dict[str, UserGroup] = {}
    request_settings_dict = locals()
    system_groups_name_dict: Optional[Dict[str, UserGroup]] = None
    for setting_name in Stream.stream_permission_group_settings:
        if setting_name not in request_settings_dict:
            continue
        if request_settings_dict[setting_name] is None:
            if system_groups_name_dict is None:
                system_groups_name_dict = get_role_based_system_groups_dict(
                    realm)
            group_setting_values[setting_name] = func_fdxyczmc(setting_name,
                system_groups_name_dict, creator=acting_user)
        else:
            group_setting_values[setting_name] = request_settings_dict[
                setting_name]
    stream_name = stream_name.strip()
    stream, created = Stream.objects.get_or_create(realm=realm,
        name__iexact=stream_name, defaults=dict(name=stream_name, creator=
        acting_user, description=stream_description, invite_only=
        invite_only, is_web_public=is_web_public,
        history_public_to_subscribers=history_public_to_subscribers,
        is_in_zephyr_realm=realm.is_zephyr_mirror_realm,
        message_retention_days=message_retention_days, **group_setting_values))
    if created:
        recipient = Recipient.objects.create(type_id=stream.id, type=
            Recipient.STREAM)
        stream.recipient = recipient
        stream.rendered_description = func_r4io5gyy(stream_description,
            realm, acting_user=acting_user)
        stream.save(update_fields=['recipient', 'rendered_description'])
        event_time = timezone_now()
        RealmAuditLog.objects.create(realm=realm, acting_user=acting_user,
            modified_stream=stream, event_type=AuditLogEventType.
            CHANNEL_CREATED, event_time=event_time)
        if setting_groups_dict is None:
            setting_groups_dict = get_group_setting_value_dict_for_streams([
                stream])
        if stream.is_public():
            if stream.is_web_public:
                notify_user_ids = active_user_ids(stream.realm_id)
            else:
                notify_user_ids = active_non_guest_user_ids(stream.realm_id)
            func_lgnk2uge(realm, stream, notify_user_ids,
                setting_groups_dict=setting_groups_dict)
        else:
            realm_admin_ids = {user.id for user in stream.realm.
                get_admin_users_and_bots()}
            func_lgnk2uge(realm, stream, list(realm_admin_ids |
                func_m92gjue2(stream)), setting_groups_dict=setting_groups_dict
                )
    return stream, created


def func_yb4ey3p9(realm: Realm, stream_dicts: List[StreamDict], acting_user: Optional[UserProfile] = None,
    setting_groups_dict: Optional[Dict[int, Union[int, AnonymousSettingGroupDict]]] = None) -> Tuple[List[Stream], List[Stream]]:
    added_streams: List[Stream] = []
    existing_streams: List[Stream] = []
    for stream_dict in stream_dicts:
        invite_only = stream_dict.get('invite_only', False)
        stream, created = func_i2dgbetu(realm, stream_dict['name'],
            invite_only=invite_only, is_web_public=stream_dict.get(
            'is_web_public', False), history_public_to_subscribers=
            stream_dict.get('history_public_to_subscribers'),
            stream_description=stream_dict.get('description', ''),
            message_retention_days=stream_dict.get('message_retention_days',
            None), can_add_subscribers_group=stream_dict.get(
            'can_add_subscribers_group', None),
            can_administer_channel_group=stream_dict.get(
            'can_administer_channel_group', None), can_send_message_group=
            stream_dict.get('can_send_message_group', None),
            can_remove_subscribers_group=stream_dict.get(
            'can_remove_subscribers_group', None), acting_user=acting_user,
            setting_groups_dict=setting_groups_dict)
        if created:
            added_streams.append(stream)
        else:
            existing_streams.append(stream)
    return added_streams, existing_streams


def func_js765ira(user_profile: UserProfile, stream_id: int) -> bool:
    return Subscription.objects.filter(user_profile=user_profile, active=
        True, recipient__type=Recipient.STREAM, recipient__type_id=stream_id
        ).exists()


def func_841no0pw(stream: Stream, user_recursive_group_ids: Set[int]) -> bool:
    group_allowed_to_administer_channel_id = (stream.
        can_administer_channel_group_id)
    assert group_allowed_to_administer_channel_id is not None
    return group_allowed_to_administer_channel_id in user_recursive_group_ids


def func_fha7fd24(stream: Stream, user_recursive_group_ids: Set[int]) -> bool:
    group_allowed_to_add_subscribers_id = stream.can_add_subscribers_group_id
    assert group_allowed_to_add_subscribers_id is not None
    return group_allowed_to_add_subscribers_id in user_recursive_group_ids


def func_sxtzkov4(stream: Stream, user_recursive_group_ids: Set[int]) -> bool:
    group_allowed_to_remove_subscribers_id = (stream.
        can_remove_subscribers_group_id)
    assert group_allowed_to_remove_subscribers_id is not None
    return group_allowed_to_remove_subscribers_id in user_recursive_group_ids


def func_4vre5hjy(sender: UserProfile, stream: Stream) -> None:
    if is_cross_realm_bot_email(sender.delivery_email):
        return
    can_send_message_group = stream.can_send_message_group
    if hasattr(can_send_message_group, 'named_user_group'):
        if (can_send_message_group.named_user_group.name == SystemGroups.
            EVERYONE):
            return
        if can_send_message_group.named_user_group.name == SystemGroups.NOBODY:
            raise JsonableError(_(
                'You do not have permission to post in this channel.'))
    if not user_has_permission_for_group_setting(stream.
        can_send_message_group, sender, Stream.
        stream_permission_group_settings['can_send_message_group'],
        direct_member_only=False):
        raise JsonableError(_(
            'You do not have permission to post in this channel.'))


def func_avivedbb(sender: UserProfile, stream: Stream, forwarder_user_profile: Optional[UserProfile],
    archived_channel_notice: bool = False) -> None:
    try:
        func_4vre5hjy(sender, stream)
    except JsonableError as e:
        if sender.is_bot and sender.bot_owner is not None:
            func_4vre5hjy(sender.bot_owner, stream)
        else:
            raise JsonableError(e.msg)
    if forwarder_user_profile is not None and forwarder_user_profile != sender:
        if (forwarder_user_profile.can_forge_sender and 
            forwarder_user_profile.realm_id == sender.realm_id and sender.
            realm_id == stream.realm_id):
            return
        else:
            raise JsonableError(_('User not authorized for this query'))
    if stream.deactivated:
        if archived_channel_notice:
            return
        raise JsonableError(_(
            "Not authorized to send to channel '{channel_name}'").format(
            channel_name=stream.name))
    if is_cross_realm_bot_email(sender.delivery_email):
        return
    if stream.realm_id != sender.realm_id:
        raise JsonableError(_('User not authorized for this query'))
    if stream.is_web_public:
        return
    if not (stream.invite_only or sender.is_guest):
        return
    if func_js765ira(sender, stream.id):
        return
    if sender.can_forge_sender:
        return
    if sender.is_bot and (sender.bot_owner is not None and func_js765ira(
        sender.bot_owner, stream.id)):
        return
    raise JsonableError(_(
        "Not authorized to send to channel '{channel_name}'").format(
        channel_name=stream.name))


def func_0jy7xwbq(stream_id: Optional[int], stream: Optional[Stream]) -> None:
    if stream_id is None and stream is None:
        error = _("Missing '{var_name}' argument").format(var_name='stream_id')
        raise JsonableError(error)
    if stream_id is not None and stream is not None:
        raise IncompatibleParametersError(['stream_id', 'stream'])


@dataclass
class UserGroupMembershipDetails:
    user_recursive_group_ids: Optional[Set[int]] = None


def func_tmm6zz12(user_profile: UserProfile, stream: Stream, user_group_membership_details: UserGroupMembershipDetails, *,
    is_subscribed: bool) -> bool:
    if stream.is_web_public:
        return True
    if is_subscribed:
        return True
    if user_profile.is_guest:
        return False
    if stream.is_public():
        return True
    if user_group_membership_details.user_recursive_group_ids is None:
        user_group_membership_details.user_recursive_group_ids = set(
            get_recursive_membership_groups(user_profile).values_list('id',
            flat=True))
    if func_fha7fd24(stream, user_group_membership_details.
        user_recursive_group_ids):
        return True
    return False


def func_dlyimpqc(user_profile: UserProfile, stream: Stream, sub: Optional[Subscription] = None) -> None:
    error = _('Invalid channel ID')
    if stream.realm_id != user_profile.realm_id:
        raise JsonableError(error)
    if user_profile.is_realm_admin:
        return
    if can_administer_accessible_channel(stream, user_profile):
        return
    user_group_membership_details = UserGroupMembershipDetails(
        user_recursive_group_ids=None)
    if func_tmm6zz12(user_profile, stream, user_group_membership_details,
        is_subscribed=sub is not None):
        raise CannotAdministerChannelError
    raise JsonableError(error)


def func_hnyahj4n(user_profile: UserProfile, stream_id: int) -> Tuple[Stream, Optional[Subscription]]:
    try:
        stream = Stream.objects.get(id=stream_id)
    except Stream.DoesNotExist:
        raise JsonableError(_('Invalid channel ID'))
    try:
        sub = Subscription.objects.get(user_profile=user_profile, recipient
            =stream.recipient, active=True)
    except Subscription.DoesNotExist:
        sub = None
    func_dlyimpqc(user_profile, stream, sub)
    return stream, sub


def func_cqc80cdx(user_profile: UserProfile, user_recursive_group_ids: Set[int],
    can_administer_channel_group_id: int, can_add_subscribers_group_id: int) -> bool:
    for setting_name in Stream.stream_permission_group_settings_granting_metadata_access:
        permission_configuration = Stream.stream_permission_group_settings[
            setting_name]
        if (not permission_configuration.allow_everyone_group and
            user_profile.is_guest):
            return False
    return (can_administer_channel_group_id in user_recursive_group_ids or 
        can_add_subscribers_group_id in user_recursive_group_ids)


def func_5v9ps1do(user_profile: UserProfile, stream: Stream, *, is_subscribed: bool,

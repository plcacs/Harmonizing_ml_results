from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Literal
from django.conf import settings
from django.db import transaction
from django.db.models import Q, QuerySet
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import override as override_language
from zerver.actions.default_streams import do_remove_default_stream, do_remove_streams_from_default_stream_group
from zerver.actions.message_send import internal_send_stream_message
from zerver.lib.cache import cache_delete_many, cache_set, display_recipient_cache_key, to_dict_cache_key_id
from zerver.lib.exceptions import JsonableError
from zerver.lib.mention import silent_mention_syntax_for_user, silent_mention_syntax_for_user_group
from zerver.lib.message import get_last_message_id
from zerver.lib.queue import queue_event_on_commit
from zerver.lib.stream_color import pick_colors
from zerver.lib.stream_subscription import SubInfo, SubscriberPeerInfo, get_active_subscriptions_for_stream_id, get_bulk_stream_subscriber_info, get_used_colors_for_user_ids, get_user_ids_for_streams, get_users_for_streams
from zerver.lib.stream_traffic import get_streams_traffic
from zerver.lib.streams import (can_access_stream_metadata_user_ids, check_basic_stream_access, get_group_setting_value_dict_for_streams,
                                get_occupied_streams, get_stream_permission_policy_name, get_stream_post_policy_value_based_on_group_setting,
                                get_user_ids_with_metadata_access_via_permission_groups, render_stream_description, send_stream_creation_event,
                                send_stream_deletion_event, stream_to_dict)
from zerver.lib.subscription_info import bulk_get_subscriber_peer_info, get_subscribers_query
from zerver.lib.types import AnonymousSettingGroupDict, APISubscriptionDict
from zerver.lib.user_groups import get_group_setting_value_for_api, get_group_setting_value_for_audit_log_data
from zerver.lib.users import get_subscribers_of_target_user_subscriptions, get_users_involved_in_dms_with_target_users
from zerver.models import ArchivedAttachment, Attachment, ChannelEmailAddress, DefaultStream, DefaultStreamGroup, Message, Realm, RealmAuditLog, Recipient, Stream, Subscription, UserGroup, UserProfile
from zerver.models.groups import NamedUserGroup, SystemGroups
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.realms import get_realm_by_id
from zerver.models.users import active_non_guest_user_ids, active_user_ids, get_system_bot
from zerver.tornado.django_api import send_event_on_commit

SubT = tuple[List[SubInfo], List[SubInfo]]
SubAndRemovedT = tuple[List[Tuple[UserProfile, Stream]], List[Tuple[UserProfile, Stream]]]

def maybe_set_moderation_or_announcement_channels_none(stream: Stream) -> None:
    realm: Realm = get_realm_by_id(realm_id=stream.realm_id)
    realm_moderation_or_announcement_channels: Tuple[str, ...] = (
        'moderation_request_channel_id',
        'new_stream_announcements_stream_id',
        'signup_announcements_stream_id',
        'zulip_update_announcements_stream_id',
    )
    update_realm_moderation_or_announcement_channels: List[str] = []
    for field in realm_moderation_or_announcement_channels:
        if getattr(realm, field) == stream.id:
            setattr(realm, field, None)
            update_realm_moderation_or_announcement_channels.append(field)
    if update_realm_moderation_or_announcement_channels:
        realm.save(update_fields=update_realm_moderation_or_announcement_channels)
        event_data: Dict[str, int] = {}
        for field in update_realm_moderation_or_announcement_channels:
            event_data[field] = -1
        event: Dict[str, Any] = dict(type='realm', op='update_dict', property='default', data=event_data)
        send_event_on_commit(realm, event, active_user_ids(realm.id))

@transaction.atomic(savepoint=False)
def do_deactivate_stream(stream: Stream, *, acting_user: UserProfile) -> None:
    if stream.deactivated is True:
        raise JsonableError(_('Channel is already deactivated'))
    affected_user_ids: Set[int] = can_access_stream_metadata_user_ids(stream)
    was_public: bool = stream.is_public()
    was_web_public: bool = stream.is_web_public
    stream.deactivated = True
    stream.save(update_fields=['deactivated'])
    ChannelEmailAddress.objects.filter(realm=stream.realm, channel=stream).update(deactivated=True)
    maybe_set_moderation_or_announcement_channels_none(stream)
    assert stream.recipient_id is not None
    if was_web_public:
        assert was_public
        Attachment.objects.filter(messages__recipient_id=stream.recipient_id).update(is_web_public=None, is_realm_public=None)
        ArchivedAttachment.objects.filter(messages__recipient_id=stream.recipient_id).update(is_web_public=None, is_realm_public=None)
    elif was_public:
        Attachment.objects.filter(messages__recipient_id=stream.recipient_id).update(is_realm_public=None)
        ArchivedAttachment.objects.filter(messages__recipient_id=stream.recipient_id).update(is_realm_public=None)
    if DefaultStream.objects.filter(realm_id=stream.realm_id, stream_id=stream.id).exists():
        do_remove_default_stream(stream)
    default_stream_groups_for_stream: QuerySet[DefaultStreamGroup] = DefaultStreamGroup.objects.filter(streams__id=stream.id)
    for group in default_stream_groups_for_stream:
        do_remove_streams_from_default_stream_group(stream.realm, group, [stream])
    send_stream_deletion_event(stream.realm, affected_user_ids, [stream])
    event_time = timezone_now()
    RealmAuditLog.objects.create(
        realm=stream.realm,
        acting_user=acting_user,
        modified_stream=stream,
        event_type=AuditLogEventType.CHANNEL_DEACTIVATED,
        event_time=event_time,
    )
    sender: UserProfile = get_system_bot(settings.NOTIFICATION_BOT, stream.realm_id)
    with override_language(stream.realm.default_language):
        internal_send_stream_message(
            sender,
            stream,
            topic_name=str(Realm.STREAM_EVENTS_NOTIFICATION_TOPIC_NAME),
            content=_('Channel {channel_name} has been archived.').format(channel_name=stream.name),
            archived_channel_notice=True,
            limit_unread_user_ids=set(),
        )

def deactivated_streams_by_old_name(realm: Realm, stream_name: str) -> QuerySet[Stream]:
    fixed_length_prefix: str = '.......!DEACTIVATED:'
    truncated_name: str = stream_name[0:Stream.MAX_NAME_LENGTH - len(fixed_length_prefix)]
    old_names: List[str] = [
        (('!' * bang_length) + 'DEACTIVATED:' + stream_name)[:Stream.MAX_NAME_LENGTH]
        for bang_length in range(1, 21)
    ]
    possible_streams: QuerySet[Stream] = Stream.objects.filter(realm=realm, deactivated=True).filter(
        Q(name=stream_name)
        | Q(name__regex=f'^{fixed_length_prefix}{truncated_name}')
        | Q(name__in=old_names)
    )
    return possible_streams

@transaction.atomic(savepoint=False)
def do_unarchive_stream(stream: Stream, new_name: str, *, acting_user: UserProfile) -> None:
    realm: Realm = stream.realm
    stream_subscribers = get_active_subscriptions_for_stream_id(stream.id, include_deactivated_users=True).select_related('user_profile')
    if not stream.deactivated:
        raise JsonableError(_('Channel is not currently deactivated'))
    if stream.name != new_name and Stream.objects.filter(realm=realm, name=new_name).exists():
        raise JsonableError(_('Channel named {channel_name} already exists').format(channel_name=new_name))
    if stream.invite_only and (not stream_subscribers):
        raise JsonableError(_('Channel is private and have no subscribers'))
    assert stream.recipient_id is not None
    stream.deactivated = False
    stream.name = new_name
    if stream.invite_only and stream.is_web_public:
        stream.is_web_public = False
    stream.save(update_fields=['name', 'deactivated', 'is_web_public'])
    ChannelEmailAddress.objects.filter(realm=realm, channel=stream).update(deactivated=False)
    cache_set(display_recipient_cache_key(stream.recipient_id), new_name)
    messages = Message.objects.filter(realm_id=realm.id, recipient_id=stream.recipient_id).only('id')
    cache_delete_many((to_dict_cache_key_id(message.id) for message in messages))
    Attachment.objects.filter(messages__recipient_id=stream.recipient_id).update(is_web_public=None, is_realm_public=None)
    ArchivedAttachment.objects.filter(messages__recipient_id=stream.recipient_id).update(is_web_public=None, is_realm_public=None)
    RealmAuditLog.objects.create(
        realm=realm,
        acting_user=acting_user,
        modified_stream=stream,
        event_type=AuditLogEventType.CHANNEL_REACTIVATED,
        event_time=timezone_now(),
    )
    recent_traffic = get_streams_traffic({stream.id}, realm)
    notify_user_ids: List[int] = list(can_access_stream_metadata_user_ids(stream))
    setting_groups_dict = get_group_setting_value_dict_for_streams([stream])
    send_stream_creation_event(realm, stream, notify_user_ids, recent_traffic, setting_groups_dict)
    sender: UserProfile = get_system_bot(settings.NOTIFICATION_BOT, stream.realm_id)
    with override_language(stream.realm.default_language):
        internal_send_stream_message(
            sender,
            stream,
            str(Realm.STREAM_EVENTS_NOTIFICATION_TOPIC_NAME),
            _('Channel {channel_name} un-archived.').format(channel_name=new_name)
        )

def bulk_delete_cache_keys(message_ids_to_clear: List[int]) -> None:
    while len(message_ids_to_clear) > 0:
        batch: List[int] = message_ids_to_clear[0:5000]
        keys_to_delete: List[str] = [to_dict_cache_key_id(message_id) for message_id in batch]
        cache_delete_many(keys_to_delete)
        message_ids_to_clear = message_ids_to_clear[5000:]

def merge_streams(realm: Realm, stream_to_keep: Stream, stream_to_destroy: Stream) -> Tuple[int, int, int]:
    recipient_to_destroy = stream_to_destroy.recipient
    recipient_to_keep = stream_to_keep.recipient
    assert recipient_to_keep is not None
    assert recipient_to_destroy is not None
    if recipient_to_destroy.id == recipient_to_keep.id:
        return (0, 0, 0)
    existing_subs = Subscription.objects.filter(recipient=recipient_to_keep)
    users_already_subscribed: Dict[int, bool] = {sub.user_profile_id: sub.active for sub in existing_subs}
    subs_to_deactivate = Subscription.objects.filter(recipient=recipient_to_destroy, active=True)
    users_to_activate: List[UserProfile] = [sub.user_profile for sub in subs_to_deactivate if not users_already_subscribed.get(sub.user_profile_id, False)]
    if len(users_to_activate) > 0:
        bulk_add_subscriptions(realm, [stream_to_keep], users_to_activate, acting_user=None)  # type: ignore
    message_ids_to_clear: List[int] = list(Message.objects.filter(realm_id=realm.id, recipient=recipient_to_destroy).values_list('id', flat=True))
    count: int = Message.objects.filter(realm_id=realm.id, recipient=recipient_to_destroy).update(recipient=recipient_to_keep)
    bulk_delete_cache_keys(message_ids_to_clear)
    if len(subs_to_deactivate) > 0:
        bulk_remove_subscriptions(realm, [sub.user_profile for sub in subs_to_deactivate], [stream_to_destroy], acting_user=None)  # type: ignore
    do_deactivate_stream(stream_to_destroy, acting_user=None)  # type: ignore
    return (len(users_to_activate), count, len(subs_to_deactivate))

def get_subscriber_ids(stream: Stream, requesting_user: Optional[UserProfile] = None) -> QuerySet[int]:
    subscriptions_query = get_subscribers_query(stream, requesting_user)
    return subscriptions_query.values_list('user_profile_id', flat=True)

def send_subscription_add_events(realm: Realm, sub_info_list: List[SubInfo], subscriber_dict: Dict[int, Set[int]]) -> None:
    info_by_user: Dict[int, List[SubInfo]] = defaultdict(list)
    for sub_info in sub_info_list:
        info_by_user[sub_info.user.id].append(sub_info)
    stream_ids: Set[int] = {sub_info.stream.id for sub_info in sub_info_list}
    recent_traffic = get_streams_traffic(stream_ids=stream_ids, realm=realm)
    stream_subscribers_dict: Dict[int, List[int]] = {}
    for sub_info in sub_info_list:
        stream = sub_info.stream
        if stream.id not in stream_subscribers_dict:
            if stream.is_in_zephyr_realm and (not stream.invite_only):
                subscribers: List[int] = []
            else:
                subscribers = list(subscriber_dict[stream.id])
            stream_subscribers_dict[stream.id] = subscribers
    streams: List[Stream] = [sub_info.stream for sub_info in sub_info_list]
    setting_groups_dict = get_group_setting_value_dict_for_streams(streams)
    for user_id, sub_infos in info_by_user.items():
        sub_dicts: List[APISubscriptionDict] = []
        for sub_info in sub_infos:
            stream = sub_info.stream
            stream_subscribers = stream_subscribers_dict[stream.id]
            subscription = sub_info.sub
            stream_dict = stream_to_dict(stream, recent_traffic, setting_groups_dict)
            sub_dict: APISubscriptionDict = APISubscriptionDict(
                audible_notifications=subscription.audible_notifications,
                color=subscription.color,
                desktop_notifications=subscription.desktop_notifications,
                email_notifications=subscription.email_notifications,
                is_muted=subscription.is_muted,
                pin_to_top=subscription.pin_to_top,
                push_notifications=subscription.push_notifications,
                wildcard_mentions_notify=subscription.wildcard_mentions_notify,
                in_home_view=not subscription.is_muted,
                stream_weekly_traffic=stream_dict['stream_weekly_traffic'],
                subscribers=stream_subscribers,
                is_archived=stream_dict['is_archived'],
                can_add_subscribers_group=stream_dict['can_add_subscribers_group'],
                can_administer_channel_group=stream_dict['can_administer_channel_group'],
                can_send_message_group=stream_dict['can_send_message_group'],
                can_remove_subscribers_group=stream_dict['can_remove_subscribers_group'],
                creator_id=stream_dict['creator_id'],
                date_created=stream_dict['date_created'],
                description=stream_dict['description'],
                first_message_id=stream_dict['first_message_id'],
                is_recently_active=stream_dict['is_recently_active'],
                history_public_to_subscribers=stream_dict['history_public_to_subscribers'],
                invite_only=stream_dict['invite_only'],
                is_web_public=stream_dict['is_web_public'],
                message_retention_days=stream_dict['message_retention_days'],
                name=stream_dict['name'],
                rendered_description=stream_dict['rendered_description'],
                stream_id=stream_dict['stream_id'],
                stream_post_policy=stream_dict['stream_post_policy'],
                is_announcement_only=stream_dict['is_announcement_only'],
            )
            sub_dicts.append(sub_dict)
        event: Dict[str, Any] = dict(type='subscription', op='add', subscriptions=sub_dicts)
        send_event_on_commit(realm, event, [user_id])

@transaction.atomic(savepoint=False)
def bulk_add_subs_to_db_with_logging(realm: Realm, acting_user: UserProfile, subs_to_add: List[SubInfo], subs_to_activate: List[SubInfo]) -> None:
    Subscription.objects.bulk_create((info.sub for info in subs_to_add))
    sub_ids = [info.sub.id for info in subs_to_activate]
    Subscription.objects.filter(id__in=sub_ids).update(active=True)
    event_time = timezone_now()
    event_last_message_id = get_last_message_id()
    all_subscription_logs = [
        RealmAuditLog(
            realm=realm,
            acting_user=acting_user,
            modified_user=sub_info.user,
            modified_stream=sub_info.stream,
            event_last_message_id=event_last_message_id,
            event_type=event_type,
            event_time=event_time,
        )
        for event_type, subs in [(AuditLogEventType.SUBSCRIPTION_CREATED, subs_to_add), (AuditLogEventType.SUBSCRIPTION_ACTIVATED, subs_to_activate)]
        for sub_info in subs
    ]
    RealmAuditLog.objects.bulk_create(all_subscription_logs)

def send_stream_creation_events_for_previously_inaccessible_streams(
    realm: Realm,
    stream_dict: Dict[int, Stream],
    altered_user_dict: Dict[int, Set[int]],
    altered_guests: Set[int],
) -> None:
    stream_ids: Set[int] = set(altered_user_dict.keys())
    recent_traffic = get_streams_traffic(stream_ids, realm)
    streams: List[Stream] = [stream_dict[stream_id] for stream_id in stream_ids]
    setting_groups_dict: Optional[Any] = None
    for stream_id, stream_users_ids in altered_user_dict.items():
        stream = stream_dict[stream_id]
        notify_user_ids: List[int] = []
        if not stream.is_public():
            realm_admin_ids: Set[int] = {user.id for user in realm.get_admin_users_and_bots()}
            user_ids_with_metadata_access_via_permission_groups = get_user_ids_with_metadata_access_via_permission_groups(stream)
            notify_user_ids = list(stream_users_ids - realm_admin_ids - user_ids_with_metadata_access_via_permission_groups)
        elif not stream.is_web_public:
            notify_user_ids = list(stream_users_ids & altered_guests)
        if notify_user_ids:
            if setting_groups_dict is None:
                setting_groups_dict = get_group_setting_value_dict_for_streams(streams)
            send_stream_creation_event(realm, stream, notify_user_ids, recent_traffic, setting_groups_dict)

def send_peer_subscriber_events(
    op: Literal['peer_add', 'peer_remove'],
    realm: Realm,
    stream_dict: Dict[int, Stream],
    altered_user_dict: Dict[int, Set[int]],
    subscriber_peer_info: SubscriberPeerInfo,
) -> None:
    assert op in ['peer_add', 'peer_remove']
    private_stream_ids: List[int] = [stream_id for stream_id in altered_user_dict if stream_dict[stream_id].invite_only]
    private_peer_dict = subscriber_peer_info.private_peer_dict
    for stream_id in private_stream_ids:
        altered_user_ids = altered_user_dict[stream_id]
        peer_user_ids = private_peer_dict[stream_id] - altered_user_ids
        if peer_user_ids and altered_user_ids:
            event = dict(type='subscription', op=op, stream_ids=[stream_id], user_ids=sorted(altered_user_ids))
            send_event_on_commit(realm, event, peer_user_ids)
    public_stream_ids: List[int] = [stream_id for stream_id in altered_user_dict if not stream_dict[stream_id].invite_only and (not stream_dict[stream_id].is_in_zephyr_realm)]
    web_public_stream_ids: List[int] = [stream_id for stream_id in public_stream_ids if stream_dict[stream_id].is_web_public]
    subscriber_dict = subscriber_peer_info.subscribed_ids
    if public_stream_ids:
        user_streams: Dict[int, Set[int]] = defaultdict(set)
        non_guest_user_ids: Set[int] = set(active_non_guest_user_ids(realm.id))
        if web_public_stream_ids:
            web_public_peer_ids: Set[int] = set(active_user_ids(realm.id))
        for stream_id in public_stream_ids:
            altered_user_ids = altered_user_dict[stream_id]
            if stream_id in web_public_stream_ids:
                peer_user_ids = web_public_peer_ids - altered_user_ids
            else:
                peer_user_ids = (non_guest_user_ids | subscriber_dict[stream_id]) - altered_user_ids
            if peer_user_ids and altered_user_ids:
                if len(altered_user_ids) == 1:
                    [altered_user_id] = altered_user_ids
                    user_streams[altered_user_id].add(stream_id)
                else:
                    event = dict(type='subscription', op=op, stream_ids=[stream_id], user_ids=sorted(altered_user_ids))
                    send_event_on_commit(realm, event, peer_user_ids)
        for user_id, stream_ids in user_streams.items():
            if stream_ids & set(web_public_stream_ids):
                peer_user_ids = web_public_peer_ids - altered_user_ids
            else:
                peer_user_ids = (non_guest_user_ids | subscriber_dict[next(iter(stream_ids))]) - altered_user_ids
            event = dict(type='subscription', op=op, stream_ids=sorted(stream_ids), user_ids=[user_id])
            send_event_on_commit(realm, event, peer_user_ids)

def send_user_creation_events_on_adding_subscriptions(
    realm: Realm,
    altered_user_dict: Dict[int, Set[int]],
    altered_streams_dict: Dict[UserProfile, Set[int]],
    subscribers_of_altered_user_subscriptions: Dict[int, Set[int]],
) -> None:
    altered_users: List[UserProfile] = list(altered_streams_dict.keys())
    non_guest_user_ids: Set[int] = active_non_guest_user_ids(realm.id)
    users_involved_in_dms = get_users_involved_in_dms_with_target_users(altered_users, realm)
    altered_stream_ids = altered_user_dict.keys()
    subscribers_dict = get_users_for_streams(set(altered_stream_ids))
    subscribers_user_id_map: Dict[int, UserProfile] = {}
    subscriber_ids_dict: Dict[int, Set[int]] = defaultdict(set)
    for stream_id, subscribers in subscribers_dict.items():
        for user in subscribers:
            subscriber_ids_dict[stream_id].add(user.id)
            subscribers_user_id_map[user.id] = user
    from zerver.actions.create_user import notify_created_user
    for user in altered_users:
        streams_for_user: Set[int] = altered_streams_dict[user]
        subscribers_in_altered_streams: Set[int] = set()
        for stream_id in streams_for_user:
            subscribers_in_altered_streams |= subscriber_ids_dict[stream_id]
        users_already_with_access_to_altered_user: Set[int] = set(non_guest_user_ids) | subscribers_of_altered_user_subscriptions[user.id] | users_involved_in_dms[user.id] | {user.id}
        users_to_receive_creation_event: Set[int] = subscribers_in_altered_streams - users_already_with_access_to_altered_user
        if users_to_receive_creation_event:
            notify_created_user(user, list(users_to_receive_creation_event))
        if user.is_guest:
            users_already_accessible_to_altered_user = subscribers_of_altered_user_subscriptions[user.id] | users_involved_in_dms[user.id] | {user.id}
            new_accessible_user_ids = subscribers_in_altered_streams - users_already_accessible_to_altered_user
            for accessible_user_id in new_accessible_user_ids:
                accessible_user = subscribers_user_id_map[accessible_user_id]
                notify_created_user(accessible_user, [user.id])

@transaction.atomic(savepoint=False)
def bulk_add_subscriptions(
    realm: Realm,
    streams: List[Stream],
    users: List[UserProfile],
    color_map: Dict[str, Any] = {},
    from_user_creation: bool = False,
    *,
    acting_user: Optional[UserProfile],
) -> SubT:
    users = list(users)
    user_ids: List[int] = [user.id for user in users]
    for stream in streams:
        assert stream.realm_id == realm.id
    for user in users:
        assert user.realm_id == realm.id
    recipient_ids: List[Optional[int]] = [stream.recipient_id for stream in streams]  # type: ignore
    recipient_id_to_stream: Dict[int, Stream] = {stream.recipient_id: stream for stream in streams if stream.recipient_id is not None}
    recipient_color_map: Dict[int, Any] = {}
    recipient_ids_set: Set[int] = set()
    for stream in streams:
        assert stream.recipient_id is not None
        recipient_ids_set.add(stream.recipient_id)
        color = color_map.get(stream.name, None)
        if color is not None:
            recipient_color_map[stream.recipient_id] = color
    used_colors_for_user_ids = get_used_colors_for_user_ids(user_ids)
    existing_subs = Subscription.objects.filter(user_profile_id__in=user_ids, recipient__type=Recipient.STREAM, recipient_id__in=recipient_ids)
    subs_by_user: Dict[int, List[Subscription]] = defaultdict(list)
    for sub in existing_subs:
        subs_by_user[sub.user_profile_id].append(sub)
    already_subscribed: List[SubInfo] = []
    subs_to_activate: List[SubInfo] = []
    subs_to_add: List[SubInfo] = []
    for user_profile in users:
        my_subs = subs_by_user[user_profile.id]
        new_recipient_ids = recipient_ids_set.copy()
        for sub in my_subs:
            if sub.recipient_id in new_recipient_ids:
                new_recipient_ids.remove(sub.recipient_id)
                stream = recipient_id_to_stream[sub.recipient_id]
                sub_info = SubInfo(user_profile, sub, stream)
                if sub.active:
                    already_subscribed.append(sub_info)
                else:
                    subs_to_activate.append(sub_info)
        used_colors = used_colors_for_user_ids.get(user_profile.id, set())
        user_color_map = pick_colors(used_colors, recipient_color_map, list(new_recipient_ids))
        for recipient_id in new_recipient_ids:
            stream = recipient_id_to_stream[recipient_id]
            color = user_color_map[recipient_id]
            sub = Subscription(user_profile=user_profile, is_user_active=user_profile.is_active, active=True, color=color, recipient_id=recipient_id)
            sub_info = SubInfo(user_profile, sub, stream)
            subs_to_add.append(sub_info)
    if len(subs_to_add) == 0 and len(subs_to_activate) == 0:
        return ([], already_subscribed)
    altered_user_dict: Dict[int, Set[int]] = defaultdict(set)
    altered_guests: Set[int] = set()
    altered_streams_dict: Dict[UserProfile, Set[int]] = defaultdict(set)
    for sub_info in subs_to_add + subs_to_activate:
        altered_user_dict[sub_info.stream.id].add(sub_info.user.id)
        altered_streams_dict[sub_info.user].add(sub_info.stream.id)
        if sub_info.user.is_guest:
            altered_guests.add(sub_info.user.id)
    if realm.can_access_all_users_group.named_user_group.name != SystemGroups.EVERYONE:
        altered_users = list(altered_streams_dict.keys())
        subscribers_of_altered_user_subscriptions = get_subscribers_of_target_user_subscriptions(altered_users)
    else:
        subscribers_of_altered_user_subscriptions = {}
    bulk_add_subs_to_db_with_logging(realm=realm, acting_user=acting_user, subs_to_add=subs_to_add, subs_to_activate=subs_to_activate)
    stream_dict: Dict[int, Stream] = {stream.id: stream for stream in streams}
    new_streams: List[Stream] = [stream_dict[stream_id] for stream_id in altered_user_dict]
    subscriber_peer_info: SubscriberPeerInfo = bulk_get_subscriber_peer_info(realm=realm, streams=new_streams)
    if not from_user_creation:
        send_stream_creation_events_for_previously_inaccessible_streams(realm=realm, stream_dict=stream_dict, altered_user_dict=altered_user_dict, altered_guests=altered_guests)
        send_subscription_add_events(realm=realm, sub_info_list=subs_to_add + subs_to_activate, subscriber_dict=subscriber_peer_info.subscribed_ids)
    if realm.can_access_all_users_group.named_user_group.name != SystemGroups.EVERYONE:
        send_user_creation_events_on_adding_subscriptions(realm, altered_user_dict, altered_streams_dict, subscribers_of_altered_user_subscriptions)  # type: ignore
    send_peer_subscriber_events(op='peer_add', realm=realm, altered_user_dict=altered_user_dict, stream_dict=stream_dict, subscriber_peer_info=subscriber_peer_info)
    return (subs_to_add + subs_to_activate, already_subscribed)

def send_peer_remove_events(realm: Realm, streams: List[Stream], altered_user_dict: Dict[int, Set[int]]) -> None:
    subscriber_peer_info: SubscriberPeerInfo = bulk_get_subscriber_peer_info(realm=realm, streams=streams)
    stream_dict: Dict[int, Stream] = {stream.id: stream for stream in streams}
    send_peer_subscriber_events(op='peer_remove', realm=realm, stream_dict=stream_dict, altered_user_dict=altered_user_dict, subscriber_peer_info=subscriber_peer_info)

def notify_subscriptions_removed(realm: Realm, user_profile: UserProfile, streams: List[Stream]) -> None:
    payload = [dict(name=stream.name, stream_id=stream.id) for stream in streams]
    event: Dict[str, Any] = dict(type='subscription', op='remove', subscriptions=payload)
    send_event_on_commit(realm, event, [user_profile.id])

def send_subscription_remove_events(
    realm: Realm,
    users: List[UserProfile],
    streams: List[Stream],
    removed_subs: List[Tuple[UserProfile, Stream]],
) -> None:
    altered_user_dict: Dict[int, Set[int]] = defaultdict(set)
    streams_by_user: Dict[int, List[Stream]] = defaultdict(list)
    for user, stream in removed_subs:
        streams_by_user[user.id].append(stream)
        altered_user_dict[stream.id].add(user.id)
    for user_profile in users:
        if len(streams_by_user[user_profile.id]) == 0:
            continue
        notify_subscriptions_removed(realm, user_profile, streams_by_user[user_profile.id])
        event = dict(type='mark_stream_messages_as_read', user_profile_id=user_profile.id, stream_recipient_ids=[stream.recipient_id for stream in streams_by_user[user_profile.id]])
        queue_event_on_commit('deferred_work', event)
        if not user_profile.is_realm_admin:
            inaccessible_streams = [stream for stream in streams_by_user[user_profile.id] if not check_basic_stream_access(user_profile, stream, is_subscribed=False, require_content_access=False)]
            if inaccessible_streams:
                send_stream_deletion_event(realm, [user_profile.id], inaccessible_streams)

def send_user_remove_events_on_removing_subscriptions(realm: Realm, altered_user_dict: Dict[UserProfile, Set[int]]) -> None:
    altered_stream_ids: Set[int] = set()
    altered_users: List[UserProfile] = list(altered_user_dict.keys())
    for stream_ids in altered_user_dict.values():
        altered_stream_ids |= stream_ids
    users_involved_in_dms = get_users_involved_in_dms_with_target_users(altered_users, realm)
    subscribers_of_altered_user_subscriptions = get_subscribers_of_target_user_subscriptions(altered_users)
    non_guest_user_ids: Set[int] = set(active_non_guest_user_ids(realm.id))
    subscribers_dict = get_user_ids_for_streams(altered_stream_ids)
    for user in altered_users:
        users_in_unsubscribed_streams: Set[int] = set()
        for stream_id in altered_user_dict[user]:
            users_in_unsubscribed_streams |= subscribers_dict[stream_id]
        users_who_can_access_altered_user = set(non_guest_user_ids) | subscribers_of_altered_user_subscriptions[user.id] | users_involved_in_dms[user.id] | {user.id}
        subscribers_without_access_to_altered_user = users_in_unsubscribed_streams - users_who_can_access_altered_user
        if subscribers_without_access_to_altered_user:
            event_remove_user = dict(type='realm_user', op='remove', person=dict(user_id=user.id, full_name=str(UserProfile.INACCESSIBLE_USER_NAME)))
            send_event_on_commit(realm, event_remove_user, list(subscribers_without_access_to_altered_user))
        if user.is_guest:
            users_inaccessible_to_altered_user = users_in_unsubscribed_streams - (subscribers_of_altered_user_subscriptions[user.id] | users_involved_in_dms[user.id] | {user.id})
            for user_id in users_inaccessible_to_altered_user:
                event_remove_user = dict(type='realm_user', op='remove', person=dict(user_id=user_id, full_name=str(UserProfile.INACCESSIBLE_USER_NAME)))
                send_event_on_commit(realm, event_remove_user, [user.id])

def bulk_remove_subscriptions(
    realm: Realm,
    users: List[UserProfile],
    streams: List[Stream],
    *,
    acting_user: UserProfile,
) -> SubAndRemovedT:
    users = list(users)
    streams = list(streams)
    for stream in streams:
        assert stream.realm_id == realm.id
    for user in users:
        assert user.realm_id == realm.id
    stream_dict: Dict[int, Stream] = {stream.id: stream for stream in streams}
    existing_subs_by_user = get_bulk_stream_subscriber_info(users, streams)

    def get_non_subscribed_subs() -> List[Tuple[UserProfile, Stream]]:
        stream_ids: Set[int] = {stream.id for stream in streams}
        not_subscribed: List[Tuple[UserProfile, Stream]] = []
        for user_profile in users:
            user_sub_stream_info = existing_subs_by_user[user_profile.id]
            subscribed_stream_ids = {sub_info.stream.id for sub_info in user_sub_stream_info}
            not_subscribed_stream_ids = stream_ids - subscribed_stream_ids
            not_subscribed.extend(((user_profile, stream_dict[stream_id]) for stream_id in not_subscribed_stream_ids))
        return not_subscribed

    not_subscribed = get_non_subscribed_subs()
    subs_to_deactivate: List[SubInfo] = [sub_info for sub_infos in existing_subs_by_user.values() for sub_info in sub_infos]
    if len(subs_to_deactivate) == 0:
        return ([], not_subscribed)
    sub_ids_to_deactivate = [sub_info.sub.id for sub_info in subs_to_deactivate]
    streams_to_unsubscribe = [sub_info.stream for sub_info in subs_to_deactivate]
    with transaction.atomic(savepoint=False):
        Subscription.objects.filter(id__in=sub_ids_to_deactivate).update(active=False)
        occupied_streams_after = list(get_occupied_streams(realm))
        event_time = timezone_now()
        event_last_message_id = get_last_message_id()
        all_subscription_logs = [
            RealmAuditLog(
                realm=sub_info.user.realm,
                acting_user=acting_user,
                modified_user=sub_info.user,
                modified_stream=sub_info.stream,
                event_last_message_id=event_last_message_id,
                event_type=AuditLogEventType.SUBSCRIPTION_DEACTIVATED,
                event_time=event_time,
            )
            for sub_info in subs_to_deactivate
        ]
        RealmAuditLog.objects.bulk_create(all_subscription_logs)
    removed_sub_tuples: List[Tuple[UserProfile, Stream]] = [(sub_info.user, sub_info.stream) for sub_info in subs_to_deactivate]
    send_subscription_remove_events(realm, users, streams, removed_sub_tuples)
    if realm.can_access_all_users_group.named_user_group.name != SystemGroups.EVERYONE:
        altered_user_dict: Dict[UserProfile, Set[int]] = defaultdict(set)
        for user, stream in removed_sub_tuples:
            altered_user_dict[user].add(stream.id)
        send_user_remove_events_on_removing_subscriptions(realm, altered_user_dict)
    new_vacant_streams = set(streams_to_unsubscribe) - set(occupied_streams_after)
    new_vacant_private_streams = [stream for stream in new_vacant_streams if stream.invite_only]
    if new_vacant_private_streams:
        for stream in new_vacant_private_streams:
            do_deactivate_stream(stream, acting_user=acting_user)
    return (removed_sub_tuples, not_subscribed)

@transaction.atomic(durable=True)
def do_change_subscription_property(
    user_profile: UserProfile,
    sub: Subscription,
    stream: Stream,
    property_name: str,
    value: Any,
    *,
    acting_user: UserProfile,
) -> None:
    database_property_name: str = property_name
    database_value: Any = value
    if property_name == 'in_home_view':
        database_property_name = 'is_muted'
        database_value = not value
    old_value: Any = getattr(sub, database_property_name)
    setattr(sub, database_property_name, database_value)
    sub.save(update_fields=[database_property_name])
    event_time = timezone_now()
    RealmAuditLog.objects.create(
        realm=user_profile.realm,
        event_type=AuditLogEventType.SUBSCRIPTION_PROPERTY_CHANGED,
        event_time=event_time,
        modified_user=user_profile,
        acting_user=acting_user,
        modified_stream=stream,
        extra_data={RealmAuditLog.OLD_VALUE: old_value, RealmAuditLog.NEW_VALUE: database_value, 'property': database_property_name},
    )
    if database_property_name == 'is_muted':
        event_value = not database_value
        in_home_view_event = dict(type='subscription', op='update', property='in_home_view', value=event_value, stream_id=stream.id)
        send_event_on_commit(user_profile.realm, in_home_view_event, [user_profile.id])
    event = dict(type='subscription', op='update', property=database_property_name, value=database_value, stream_id=stream.id)
    send_event_on_commit(user_profile.realm, event, [user_profile.id])

def send_change_stream_permission_notification(
    stream: Stream,
    *,
    old_policy_name: str,
    new_policy_name: str,
    acting_user: UserProfile,
) -> None:
    sender: UserProfile = get_system_bot(settings.NOTIFICATION_BOT, acting_user.realm_id)
    user_mention: str = silent_mention_syntax_for_user(acting_user)
    with override_language(stream.realm.default_language):
        notification_string = _('{user} changed the [access permissions]({help_link}) for this channel from **{old_policy}** to **{new_policy}**.')
        notification_string = notification_string.format(user=user_mention, help_link='/help/channel-permissions', old_policy=old_policy_name, new_policy=new_policy_name)
        internal_send_stream_message(sender, stream, str(Realm.STREAM_EVENTS_NOTIFICATION_TOPIC_NAME), notification_string)

@transaction.atomic(savepoint=False)
def do_change_stream_permission(
    stream: Stream,
    *,
    invite_only: bool,
    history_public_to_subscribers: bool,
    is_web_public: bool,
    acting_user: UserProfile,
) -> None:
    old_invite_only_value: bool = stream.invite_only
    old_history_public_to_subscribers_value: bool = stream.history_public_to_subscribers
    old_is_web_public_value: bool = stream.is_web_public
    stream.is_web_public = is_web_public
    stream.invite_only = invite_only
    stream.history_public_to_subscribers = history_public_to_subscribers
    stream.save(update_fields=['invite_only', 'history_public_to_subscribers', 'is_web_public'])
    realm: Realm = stream.realm
    event_time = timezone_now()
    if old_invite_only_value != stream.invite_only:
        assert stream.recipient_id is not None
        Attachment.objects.filter(messages__recipient_id=stream.recipient_id).update(is_realm_public=None)
        ArchivedAttachment.objects.filter(messages__recipient_id=stream.recipient_id).update(is_realm_public=None)
        RealmAuditLog.objects.create(
            realm=realm,
            acting_user=acting_user,
            modified_stream=stream,
            event_type=AuditLogEventType.CHANNEL_PROPERTY_CHANGED,
            event_time=event_time,
            extra_data={RealmAuditLog.OLD_VALUE: old_invite_only_value, RealmAuditLog.NEW_VALUE: stream.invite_only, 'property': 'invite_only'},
        )
    if old_history_public_to_subscribers_value != stream.history_public_to_subscribers:
        RealmAuditLog.objects.create(
            realm=realm,
            acting_user=acting_user,
            modified_stream=stream,
            event_type=AuditLogEventType.CHANNEL_PROPERTY_CHANGED,
            event_time=event_time,
            extra_data={RealmAuditLog.OLD_VALUE: old_history_public_to_subscribers_value, RealmAuditLog.NEW_VALUE: stream.history_public_to_subscribers, 'property': 'history_public_to_subscribers'},
        )
    if old_is_web_public_value != stream.is_web_public:
        assert stream.recipient_id is not None
        Attachment.objects.filter(messages__recipient_id=stream.recipient_id).update(is_web_public=None)
        ArchivedAttachment.objects.filter(messages__recipient_id=stream.recipient_id).update(is_web_public=None)
        RealmAuditLog.objects.create(
            realm=realm,
            acting_user=acting_user,
            modified_stream=stream,
            event_type=AuditLogEventType.CHANNEL_PROPERTY_CHANGED,
            event_time=event_time,
            extra_data={RealmAuditLog.OLD_VALUE: old_is_web_public_value, RealmAuditLog.NEW_VALUE: stream.is_web_public, 'property': 'is_web_public'},
        )
    notify_stream_creation_ids: Set[int] = set()
    if old_invite_only_value and (not stream.invite_only):
        stream_subscriber_user_ids = get_active_subscriptions_for_stream_id(stream.id, include_deactivated_users=False).values_list('user_profile_id', flat=True)
        old_can_access_stream_metadata_user_ids: Set[int] = set(stream_subscriber_user_ids) | {user.id for user in stream.realm.get_admin_users_and_bots()}
        user_ids_with_metadata_access_via_permission_groups = get_user_ids_with_metadata_access_via_permission_groups(stream)
        non_guest_user_ids: Set[int] = set(active_non_guest_user_ids(stream.realm_id))
        notify_stream_creation_ids = non_guest_user_ids - old_can_access_stream_metadata_user_ids - user_ids_with_metadata_access_via_permission_groups
        recent_traffic = get_streams_traffic({stream.id}, realm)
        setting_groups_dict = get_group_setting_value_dict_for_streams([stream])
        send_stream_creation_event(realm, stream, list(notify_stream_creation_ids), recent_traffic, setting_groups_dict)
        old_subscribers_access_user_ids = set(stream_subscriber_user_ids) | {user.id for user in stream.realm.get_admin_users_and_bots()}
        peer_notify_user_ids = non_guest_user_ids - old_subscribers_access_user_ids - user_ids_with_metadata_access_via_permission_groups
        peer_add_event = dict(type='subscription', op='peer_add', stream_ids=[stream.id], user_ids=sorted(stream_subscriber_user_ids))
        send_event_on_commit(stream.realm, peer_add_event, peer_notify_user_ids)
    event = dict(op='update', type='stream', property='invite_only', value=stream.invite_only, history_public_to_subscribers=stream.history_public_to_subscribers, is_web_public=stream.is_web_public, stream_id=stream.id, name=stream.name)
    notify_stream_update_ids = can_access_stream_metadata_user_ids(stream) - notify_stream_creation_ids
    send_event_on_commit(stream.realm, event, notify_stream_update_ids)
    old_policy_name: str = get_stream_permission_policy_name(invite_only=old_invite_only_value, history_public_to_subscribers=old_history_public_to_subscribers_value, is_web_public=old_is_web_public_value)
    new_policy_name: str = get_stream_permission_policy_name(invite_only=stream.invite_only, history_public_to_subscribers=stream.history_public_to_subscribers, is_web_public=stream.is_web_public)
    send_change_stream_permission_notification(stream, old_policy_name=old_policy_name, new_policy_name=new_policy_name, acting_user=acting_user)

def get_users_string_with_permission(setting_value: Any) -> str:
    if isinstance(setting_value, int):
        setting_group = NamedUserGroup.objects.get(id=setting_value)
        return silent_mention_syntax_for_user_group(setting_group)
    groups_with_permission = NamedUserGroup.objects.filter(id__in=setting_value.direct_subgroups).order_by('id')
    group_name_syntax_list: List[str] = [silent_mention_syntax_for_user_group(group) for group in groups_with_permission]
    users_with_permission = UserProfile.objects.filter(id__in=setting_value.direct_members).order_by('id')
    user_name_syntax_list: List[str] = [silent_mention_syntax_for_user(user) for user in users_with_permission]
    return ', '.join(group_name_syntax_list + user_name_syntax_list)

def send_stream_posting_permission_update_notification(
    stream: Stream,
    *,
    old_setting_value: Any,
    new_setting_value: Any,
    acting_user: UserProfile,
) -> None:
    sender: UserProfile = get_system_bot(settings.NOTIFICATION_BOT, acting_user.realm_id)
    user_mention: str = silent_mention_syntax_for_user(acting_user)
    old_setting_description: str = get_users_string_with_permission(old_setting_value)
    new_setting_description: str = get_users_string_with_permission(new_setting_value)
    with override_language(stream.realm.default_language):
        notification_string = _('{user} changed the [posting permissions]({help_link}) for this channel:\n\n* **Old**: {old_setting_description}\n* **New**: {new_setting_description}\n')
        notification_string = notification_string.format(user=user_mention, help_link='/help/channel-posting-policy', old_setting_description=old_setting_description, new_setting_description=new_setting_description)
        internal_send_stream_message(sender, stream, str(Realm.STREAM_EVENTS_NOTIFICATION_TOPIC_NAME), notification_string)

@transaction.atomic(durable=True)
def do_rename_stream(stream: Stream, new_name: str, user_profile: UserProfile) -> None:
    old_name: str = stream.name
    stream.name = new_name
    stream.save(update_fields=['name'])
    RealmAuditLog.objects.create(
        realm=stream.realm,
        acting_user=user_profile,
        modified_stream=stream,
        event_type=AuditLogEventType.CHANNEL_NAME_CHANGED,
        event_time=timezone_now(),
        extra_data={RealmAuditLog.OLD_VALUE: old_name, RealmAuditLog.NEW_VALUE: new_name},
    )
    assert stream.recipient_id is not None
    recipient_id: int = stream.recipient_id
    messages = Message.objects.filter(realm_id=stream.realm_id, recipient_id=recipient_id).only('id')
    cache_set(display_recipient_cache_key(recipient_id), stream.name)
    cache_delete_many((to_dict_cache_key_id(message.id) for message in messages))
    event = dict(op='update', type='stream', property='name', value=new_name, stream_id=stream.id, name=old_name)
    send_event_on_commit(stream.realm, event, can_access_stream_metadata_user_ids(stream))
    sender: UserProfile = get_system_bot(settings.NOTIFICATION_BOT, stream.realm_id)
    with override_language(stream.realm.default_language):
        internal_send_stream_message(
            sender,
            stream,
            str(Realm.STREAM_EVENTS_NOTIFICATION_TOPIC_NAME),
            _('{user_name} renamed channel {old_channel_name} to {new_channel_name}.').format(
                user_name=silent_mention_syntax_for_user(user_profile),
                old_channel_name=f'**{old_name}**',
                new_channel_name=f'**{new_name}**'
            )
        )

def send_change_stream_description_notification(
    stream: Stream,
    *,
    old_description: str,
    new_description: str,
    acting_user: UserProfile,
) -> None:
    sender: UserProfile = get_system_bot(settings.NOTIFICATION_BOT, acting_user.realm_id)
    user_mention: str = silent_mention_syntax_for_user(acting_user)
    with override_language(stream.realm.default_language):
        if new_description == '':
            new_description = '*' + _('No description.') + '*'
        if old_description == '':
            old_description = '*' + _('No description.') + '*'
        notification_string = _('{user} changed the description for this channel.').format(user=user_mention) + '\n\n* **' + _('Old description') + ':**' + f'\n```` quote\n{old_description}\n````\n' + '* **' + _('New description') + ':**' + f'\n```` quote\n{new_description}\n````'
        internal_send_stream_message(sender, stream, str(Realm.STREAM_EVENTS_NOTIFICATION_TOPIC_NAME), notification_string)

@transaction.atomic(durable=True)
def do_change_stream_description(stream: Stream, new_description: str, *, acting_user: UserProfile) -> None:
    old_description: str = stream.description
    stream.description = new_description
    stream.rendered_description = render_stream_description(new_description, stream.realm, acting_user=acting_user)
    stream.save(update_fields=['description', 'rendered_description'])
    RealmAuditLog.objects.create(
        realm=stream.realm,
        acting_user=acting_user,
        modified_stream=stream,
        event_type=AuditLogEventType.CHANNEL_PROPERTY_CHANGED,
        event_time=timezone_now(),
        extra_data={RealmAuditLog.OLD_VALUE: old_description, RealmAuditLog.NEW_VALUE: new_description, 'property': 'description'},
    )
    event = dict(
        type='stream',
        op='update',
        property='description',
        name=stream.name,
        stream_id=stream.id,
        value=new_description,
        rendered_description=stream.rendered_description,
    )
    send_event_on_commit(stream.realm, event, can_access_stream_metadata_user_ids(stream))
    send_change_stream_description_notification(stream, old_description=old_description, new_description=new_description, acting_user=acting_user)

def send_change_stream_message_retention_days_notification(user_profile: UserProfile, stream: Stream, old_value: Optional[int], new_value: Optional[int]) -> None:
    sender: UserProfile = get_system_bot(settings.NOTIFICATION_BOT, user_profile.realm_id)
    user_mention: str = silent_mention_syntax_for_user(user_profile)
    if old_value is None:
        old_value = stream.realm.message_retention_days
    if new_value is None:
        new_value = stream.realm.message_retention_days
    with override_language(stream.realm.default_language):
        if old_value == Stream.MESSAGE_RETENTION_SPECIAL_VALUES_MAP['unlimited']:
            old_retention_period = _('Forever')
            new_retention_period = _('{number_of_days} days').format(number_of_days=new_value)
            summary_line = _('Messages in this channel will now be automatically deleted {number_of_days} days after they are sent.').format(number_of_days=new_value)
        elif new_value == Stream.MESSAGE_RETENTION_SPECIAL_VALUES_MAP['unlimited']:
            old_retention_period = _('{number_of_days} days').format(number_of_days=old_value)
            new_retention_period = _('Forever')
            summary_line = _('Messages in this channel will now be retained forever.')
        else:
            old_retention_period = _('{number_of_days} days').format(number_of_days=old_value)
            new_retention_period = _('{number_of_days} days').format(number_of_days=new_value)
            summary_line = _('Messages in this channel will now be automatically deleted {number_of_days} days after they are sent.').format(number_of_days=new_value)
        notification_string = _('{user} has changed the [message retention period]({help_link}) for this channel:\n* **Old retention period**: {old_retention_period}\n* **New retention period**: {new_retention_period}\n\n{summary_line}')
        notification_string = notification_string.format(user=user_mention, help_link='/help/message-retention-policy', old_retention_period=old_retention_period, new_retention_period=new_retention_period, summary_line=summary_line)
        internal_send_stream_message(sender, stream, str(Realm.STREAM_EVENTS_NOTIFICATION_TOPIC_NAME), notification_string)

@transaction.atomic(durable=True)
def do_change_stream_message_retention_days(stream: Stream, acting_user: UserProfile, message_retention_days: Optional[int] = None) -> None:
    old_message_retention_days_value: Optional[int] = stream.message_retention_days
    stream.message_retention_days = message_retention_days
    stream.save(update_fields=['message_retention_days'])
    RealmAuditLog.objects.create(
        realm=stream.realm,
        acting_user=acting_user,
        modified_stream=stream,
        event_type=AuditLogEventType.CHANNEL_MESSAGE_RETENTION_DAYS_CHANGED,
        event_time=timezone_now(),
        extra_data={RealmAuditLog.OLD_VALUE: old_message_retention_days_value, RealmAuditLog.NEW_VALUE: message_retention_days},
    )
    event = dict(op='update', type='stream', property='message_retention_days', value=message_retention_days, stream_id=stream.id, name=stream.name)
    send_event_on_commit(stream.realm, event, can_access_stream_metadata_user_ids(stream))
    send_change_stream_message_retention_days_notification(user_profile=acting_user, stream=stream, old_value=old_message_retention_days_value, new_value=message_retention_days)

def do_change_stream_group_based_setting(
    stream: Stream,
    setting_name: str,
    user_group: Any,
    *,
    old_setting_api_value: Optional[Any] = None,
    acting_user: Optional[UserProfile] = None,
) -> None:
    old_user_group = getattr(stream, setting_name)
    setattr(stream, setting_name, user_group)
    stream.save()
    if old_setting_api_value is None:
        old_setting_api_value = get_group_setting_value_for_api(old_user_group)
    new_setting_api_value = get_group_setting_value_for_api(user_group)
    RealmAuditLog.objects.create(
        realm=stream.realm,
        acting_user=acting_user,
        modified_stream=stream,
        event_type=AuditLogEventType.CHANNEL_GROUP_BASED_SETTING_CHANGED,
        event_time=timezone_now(),
        extra_data={RealmAuditLog.OLD_VALUE: get_group_setting_value_for_audit_log_data(old_setting_api_value), RealmAuditLog.NEW_VALUE: get_group_setting_value_for_audit_log_data(new_setting_api_value), 'property': setting_name},
    )
    event = dict(op='update', type='stream', property=setting_name, value=new_setting_api_value, stream_id=stream.id, name=stream.name)
    send_event_on_commit(stream.realm, event, can_access_stream_metadata_user_ids(stream))
    if setting_name == 'can_send_message_group':
        old_stream_post_policy = get_stream_post_policy_value_based_on_group_setting(old_user_group)
        stream_post_policy = get_stream_post_policy_value_based_on_group_setting(user_group)
        if old_stream_post_policy != stream_post_policy:
            event = dict(op='update', type='stream', property='stream_post_policy', value=stream_post_policy, stream_id=stream.id, name=stream.name)
            send_event_on_commit(stream.realm, event, can_access_stream_metadata_user_ids(stream))
            event = dict(op='update', type='stream', property='is_announcement_only', value=stream_post_policy == Stream.STREAM_POST_POLICY_ADMINS, stream_id=stream.id, name=stream.name)
            send_event_on_commit(stream.realm, event, can_access_stream_metadata_user_ids(stream))
        assert acting_user is not None
        send_stream_posting_permission_update_notification(stream, old_setting_value=old_setting_api_value, new_setting_value=new_setting_api_value, acting_user=acting_user)
    if not hasattr(old_user_group, 'named_user_group') and hasattr(user_group, 'named_user_group'):
        old_user_group.delete()
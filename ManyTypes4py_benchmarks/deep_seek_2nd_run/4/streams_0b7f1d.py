from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from typing import Any, TypeAlias, Dict, List, Set, Tuple, Optional, Union
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
from zerver.lib.streams import can_access_stream_metadata_user_ids, check_basic_stream_access, get_group_setting_value_dict_for_streams, get_occupied_streams, get_stream_permission_policy_name, get_stream_post_policy_value_based_on_group_setting, get_user_ids_with_metadata_access_via_permission_groups, render_stream_description, send_stream_creation_event, send_stream_deletion_event, stream_to_dict
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

def maybe_set_moderation_or_announcement_channels_none(stream: Stream) -> None:
    realm = get_realm_by_id(realm_id=stream.realm_id)
    realm_moderation_or_announcement_channels = ('moderation_request_channel_id', 'new_stream_announcements_stream_id', 'signup_announcements_stream_id', 'zulip_update_announcements_stream_id')
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
        event = dict(type='realm', op='update_dict', property='default', data=event_data)
        send_event_on_commit(realm, event, active_user_ids(realm.id))

@transaction.atomic(savepoint=False)
def do_deactivate_stream(stream: Stream, *, acting_user: UserProfile) -> None:
    if stream.deactivated is True:
        raise JsonableError(_('Channel is already deactivated'))
    affected_user_ids = can_access_stream_metadata_user_ids(stream)
    was_public = stream.is_public()
    was_web_public = stream.is_web_public
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
    default_stream_groups_for_stream = DefaultStreamGroup.objects.filter(streams__id=stream.id)
    for group in default_stream_groups_for_stream:
        do_remove_streams_from_default_stream_group(stream.realm, group, [stream])
    send_stream_deletion_event(stream.realm, affected_user_ids, [stream])
    event_time = timezone_now()
    RealmAuditLog.objects.create(realm=stream.realm, acting_user=acting_user, modified_stream=stream, event_type=AuditLogEventType.CHANNEL_DEACTIVATED, event_time=event_time)
    sender = get_system_bot(settings.NOTIFICATION_BOT, stream.realm_id)
    with override_language(stream.realm.default_language):
        internal_send_stream_message(sender, stream, topic_name=str(Realm.STREAM_EVENTS_NOTIFICATION_TOPIC_NAME), content=_('Channel {channel_name} has been archived.').format(channel_name=stream.name), archived_channel_notice=True, limit_unread_user_ids=set())

def deactivated_streams_by_old_name(realm: Realm, stream_name: str) -> QuerySet[Stream]:
    fixed_length_prefix = '.......!DEACTIVATED:'
    truncated_name = stream_name[0:Stream.MAX_NAME_LENGTH - len(fixed_length_prefix)]
    old_names = [('!' * bang_length + 'DEACTIVATED:' + stream_name)[:Stream.MAX_NAME_LENGTH] for bang_length in range(1, 21)]
    possible_streams = Stream.objects.filter(realm=realm, deactivated=True).filter(Q(name=stream_name) | Q(name__regex=f'^{fixed_length_prefix}{truncated_name}') | Q(name__in=old_names))
    return possible_streams

@transaction.atomic(savepoint=False)
def do_unarchive_stream(stream: Stream, new_name: str, *, acting_user: UserProfile) -> None:
    realm = stream.realm
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
    RealmAuditLog.objects.create(realm=realm, acting_user=acting_user, modified_stream=stream, event_type=AuditLogEventType.CHANNEL_REACTIVATED, event_time=timezone_now())
    recent_traffic = get_streams_traffic({stream.id}, realm)
    notify_user_ids = list(can_access_stream_metadata_user_ids(stream))
    setting_groups_dict = get_group_setting_value_dict_for_streams([stream])
    send_stream_creation_event(realm, stream, notify_user_ids, recent_traffic, setting_groups_dict)
    sender = get_system_bot(settings.NOTIFICATION_BOT, stream.realm_id)
    with override_language(stream.realm.default_language):
        internal_send_stream_message(sender, stream, str(Realm.STREAM_EVENTS_NOTIFICATION_TOPIC_NAME), _('Channel {channel_name} un-archived.').format(channel_name=new_name))

def bulk_delete_cache_keys(message_ids_to_clear: List[int]) -> None:
    while len(message_ids_to_clear) > 0:
        batch = message_ids_to_clear[0:5000]
        keys_to_delete = [to_dict_cache_key_id(message_id) for message_id in batch]
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
    users_already_subscribed = {sub.user_profile_id: sub.active for sub in existing_subs}
    subs_to_deactivate = Subscription.objects.filter(recipient=recipient_to_destroy, active=True)
    users_to_activate = [sub.user_profile for sub in subs_to_deactivate if not users_already_subscribed.get(sub.user_profile_id, False)]
    if len(users_to_activate) > 0:
        bulk_add_subscriptions(realm, [stream_to_keep], users_to_activate, acting_user=None)
    message_ids_to_clear = list(Message.objects.filter(realm_id=realm.id, recipient=recipient_to_destroy).values_list('id', flat=True))
    count = Message.objects.filter(realm_id=realm.id, recipient=recipient_to_destroy).update(recipient=recipient_to_keep)
    bulk_delete_cache_keys(message_ids_to_clear)
    if len(subs_to_deactivate) > 0:
        bulk_remove_subscriptions(realm, [sub.user_profile for sub in subs_to_deactivate], [stream_to_destroy], acting_user=None)
    do_deactivate_stream(stream_to_destroy, acting_user=None)
    return (len(users_to_activate), count, len(subs_to_deactivate))

def get_subscriber_ids(stream: Stream, requesting_user: Optional[UserProfile] = None) -> QuerySet[int]:
    subscriptions_query = get_subscribers_query(stream, requesting_user)
    return subscriptions_query.values_list('user_profile_id', flat=True)

def send_subscription_add_events(realm: Realm, sub_info_list: List[SubInfo], subscriber_dict: Dict[int, List[int]]) -> None:
    info_by_user: Dict[int, List[SubInfo]] = defaultdict(list)
    for sub_info in sub_info_list:
        info_by_user[sub_info.user.id].append(sub_info)
    stream_ids = {sub_info.stream.id for sub_info in sub_info_list}
    recent_traffic = get_streams_traffic(stream_ids=stream_ids, realm=realm)
    stream_subscribers_dict: Dict[int, List[int]] = {}
    for sub_info in sub_info_list:
        stream = sub_info.stream
        if stream.id not in stream_subscribers_dict:
            if stream.is_in_zephyr_realm and (not stream.invite_only):
                subscribers = []
            else:
                subscribers = list(subscriber_dict[stream.id])
            stream_subscribers_dict[stream.id] = subscribers
    streams = [sub_info.stream for sub_info in sub_info_list]
    setting_groups_dict = get_group_setting_value_dict_for_streams(streams)
    for user_id, sub_infos in info_by_user.items():
        sub_dicts = []
        for sub_info in sub_infos:
            stream = sub_info.stream
            stream_subscribers = stream_subscribers_dict[stream.id]
            subscription = sub_info.sub
            stream_dict = stream_to_dict(stream, recent_traffic, setting_groups_dict)
            sub_dict = APISubscriptionDict(audible_notifications=subscription.audible_notifications, color=subscription.color, desktop_notifications=subscription.desktop_notifications, email_notifications=subscription.email_notifications, is_muted=subscription.is_muted, pin_to_top=subscription.pin_to_top, push_notifications=subscription.push_notifications, wildcard_mentions_notify=subscription.wildcard_mentions_notify, in_home_view=not subscription.is_muted, stream_weekly_traffic=stream_dict['stream_weekly_traffic'], subscribers=stream_subscribers, is_archived=stream_dict['is_archived'], can_add_subscribers_group=stream_dict['can_add_subscribers_group'], can_administer_channel_group=stream_dict['can_administer_channel_group'], can_send_message_group=stream_dict['can_send_message_group'], can_remove_subscribers_group=stream_dict['can_remove_subscribers_group'], creator_id=stream_dict['creator_id'], date_created=stream_dict['date_created'], description=stream_dict['description'], first_message_id=stream_dict['first_message_id'], is_recently_active=stream_dict['is_recently_active'], history_public_to_subscribers=stream_dict['history_public_to_subscribers'], invite_only=stream_dict['invite_only'], is_web_public=stream_dict['is_web_public'], message_retention_days=stream_dict['message_retention_days'], name=stream_dict['name'], rendered_description=stream_dict['rendered_description'], stream_id=stream_dict['stream_id'], stream_post_policy=stream_dict['stream_post_policy'], is_announcement_only=stream_dict['is_announcement_only'])
            sub_dicts.append(sub_dict)
        event = dict(type='subscription', op='add', subscriptions=sub_dicts)
        send_event_on_commit(realm, event, [user_id])

@transaction.atomic(savepoint=False)
def bulk_add_subs_to_db_with_logging(realm: Realm, acting_user: Optional[UserProfile], subs_to_add: List[SubInfo], subs_to_activate: List[SubInfo]) -> None:
    Subscription.objects.bulk_create((info.sub for info in subs_to_add))
    sub_ids = [info.sub.id for info in subs_to_activate]
    Subscription.objects.filter(id__in=sub_ids).update(active=True)
    event_time = timezone_now()
    event_last_message_id = get_last_message_id()
    all_subscription_logs = [RealmAuditLog(realm=realm, acting_user=acting_user, modified_user=sub_info.user, modified_stream=sub_info.stream, event_last_message_id=event_last_message_id, event_type=event_type, event_time=event_time) for event_type, subs in [(AuditLogEventType.SUBSCRIPTION_CREATED, subs_to_add), (AuditLogEventType.SUBSCRIPTION_ACTIVATED, subs_to_activate)] for sub_info in subs]
    RealmAuditLog.objects.bulk_create(all_subscription_logs)

def send_stream_creation_events_for_previously_inaccessible_streams(realm: Realm, stream_dict: Dict[int, Stream], altered_user_dict: Dict[int, Set[int]], altered_guests: Set[int]) -> None:
    stream_ids = set(altered_user_dict.keys())
    recent_traffic = get_streams_traffic(stream_ids, realm)
    streams = [stream_dict[stream_id] for stream_id in stream_ids]
    setting_groups_dict = None
    for stream_id, stream_users_ids in altered_user_dict.items():
        stream = stream_dict[stream_id]
        notify_user_ids = []
        if not stream.is_public():
            realm_admin_ids = {user.id for user in realm.get_admin_users_and_bots()}
            user_ids_with_metadata_access_via_permission_groups = get_user_ids_with_metadata_access_via_permission_groups(stream)
            notify_user_ids = list(stream_users_ids - realm_admin_ids - user_ids_with_metadata_access_via_permission_groups)
        elif not stream.is_web_public:
            notify_user_ids = list(stream_users_ids & altered_guests)
        if notify_user_ids:
            if setting_groups_dict is None:
                setting_groups_dict = get_group_setting_value_dict_for_streams(streams)
            send_stream_creation_event(realm, stream, notify_user_ids, recent_traffic, setting_groups_dict)

def send_peer_subscriber_events(op: str, realm: Realm, stream_dict: Dict[int, Stream], altered_user_dict: Dict[int, Set[int]], subscriber_peer_info: SubscriberPeerInfo) -> None:
    assert op in ['peer_add', 'peer_remove']
    private_stream_ids = [stream_id for stream_id in altered_user_dict if stream_dict[stream_id].inv
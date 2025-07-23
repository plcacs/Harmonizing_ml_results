import itertools
from collections import defaultdict
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Dict, List, Set, Union
from django.db.models import Q, QuerySet
from zerver.models import AlertWord, Recipient, Stream, Subscription, UserProfile, UserTopic

@dataclass
class SubInfo:
    user: UserProfile
    sub: Subscription
    stream: Stream

@dataclass
class SubscriberPeerInfo:
    pass

def get_active_subscriptions_for_stream_id(stream_id: int, *, include_deactivated_users: bool) -> QuerySet:
    query = Subscription.objects.filter(recipient__type=Recipient.STREAM, recipient__type_id=stream_id, active=True)
    if not include_deactivated_users:
        query = query.filter(is_user_active=True)
    return query

def get_active_subscriptions_for_stream_ids(stream_ids: List[int]) -> QuerySet:
    return Subscription.objects.filter(recipient__type=Recipient.STREAM, recipient__type_id__in=stream_ids, active=True, is_user_active=True)

def get_subscribed_stream_ids_for_user(user_profile: UserProfile) -> QuerySet:
    return Subscription.objects.filter(user_profile_id=user_profile, recipient__type=Recipient.STREAM, active=True).values_list('recipient__type_id', flat=True)

def get_subscribed_stream_recipient_ids_for_user(user_profile: UserProfile) -> QuerySet:
    return Subscription.objects.filter(user_profile_id=user_profile, recipient__type=Recipient.STREAM, active=True).values_list('recipient_id', flat=True)

def get_stream_subscriptions_for_user(user_profile: UserProfile) -> QuerySet:
    return Subscription.objects.filter(user_profile=user_profile, recipient__type=Recipient.STREAM)

def get_used_colors_for_user_ids(user_ids: List[int]) -> Dict[int, Set[str]]:
    query = Subscription.objects.filter(user_profile_id__in=user_ids, recipient__type=Recipient.STREAM).values('user_profile_id', 'color').distinct()
    result = defaultdict(set)
    for row in query:
        assert row['color'] is not None
        result[row['user_profile_id']].add(row['color'])
    return result

def get_bulk_stream_subscriber_info(users: List[UserProfile], streams: List[Stream]) -> Dict[int, List[SubInfo]]:
    stream_ids = {stream.id for stream in streams}
    subs = Subscription.objects.filter(user_profile__in=users, recipient__type=Recipient.STREAM, recipient__type_id__in=stream_ids, active=True).only('user_profile_id', 'recipient_id')
    stream_map = {stream.recipient_id: stream for stream in streams}
    user_map = {user.id: user for user in users}
    result = {user.id: [] for user in users}
    for sub in subs:
        user_id = sub.user_profile_id
        user = user_map[user_id]
        recipient_id = sub.recipient_id
        stream = stream_map[recipient_id]
        sub_info = SubInfo(user=user, sub=sub, stream=stream)
        result[user_id].append(sub_info)
    return result

def num_subscribers_for_stream_id(stream_id: int) -> int:
    return get_active_subscriptions_for_stream_id(stream_id, include_deactivated_users=False).count()

def get_user_ids_for_streams(stream_ids: List[int]) -> Dict[int, Set[int]]:
    all_subs = get_active_subscriptions_for_stream_ids(stream_ids).values('recipient__type_id', 'user_profile_id').order_by('recipient__type_id')
    get_stream_id = itemgetter('recipient__type_id')
    result = defaultdict(set)
    for stream_id, rows in itertools.groupby(all_subs, get_stream_id):
        user_ids = {row['user_profile_id'] for row in rows}
        result[stream_id] = user_ids
    return result

def get_users_for_streams(stream_ids: List[int]) -> Dict[int, Set[UserProfile]]:
    all_subs = get_active_subscriptions_for_stream_ids(stream_ids).select_related('user_profile', 'recipient').order_by('recipient__type_id')
    result = defaultdict(set)
    for stream_id, rows in itertools.groupby(all_subs, key=lambda obj: obj.recipient.type_id):
        users = {row.user_profile for row in rows}
        result[stream_id] = users
    return result

def handle_stream_notifications_compatibility(user_profile: Union[UserProfile, None], stream_dict: Dict[str, Any], notification_settings_null: bool) -> None:
    assert not notification_settings_null
    for notification_type in ['desktop_notifications', 'audible_notifications', 'push_notifications', 'email_notifications']:
        if stream_dict[notification_type] is not None:
            continue
        target_attr = 'enable_stream_' + notification_type
        stream_dict[notification_type] = False if user_profile is None else getattr(user_profile, target_attr)

def subscriber_ids_with_stream_history_access(stream: Stream) -> Set[int]:
    if not stream.is_history_public_to_subscribers():
        return set()
    return set(get_active_subscriptions_for_stream_id(stream.id, include_deactivated_users=False).values_list('user_profile_id', flat=True))

def get_subscriptions_for_send_message(*, realm_id: int, stream_id: int, topic_name: str, possible_stream_wildcard_mention: bool, topic_participant_user_ids: Set[int], possibly_mentioned_user_ids: Set[int]) -> QuerySet:
    query = get_active_subscriptions_for_stream_id(stream_id, include_deactivated_users=False)
    if possible_stream_wildcard_mention:
        return query
    query = query.filter(
        Q(user_profile__long_term_idle=False) |
        Q(push_notifications=True) |
        Q(push_notifications=None) & Q(user_profile__enable_stream_push_notifications=True) |
        Q(email_notifications=True) |
        Q(email_notifications=None) & Q(user_profile__enable_stream_email_notifications=True) |
        Q(user_profile_id__in=possibly_mentioned_user_ids) |
        Q(user_profile_id__in=topic_participant_user_ids) |
        Q(user_profile_id__in=AlertWord.objects.filter(realm_id=realm_id).values_list('user_profile_id')) |
        Q(user_profile_id__in=UserTopic.objects.filter(stream_id=stream_id, topic_name__iexact=topic_name, visibility_policy=UserTopic.VisibilityPolicy.FOLLOWED).values_list('user_profile_id'))
    )
    return query

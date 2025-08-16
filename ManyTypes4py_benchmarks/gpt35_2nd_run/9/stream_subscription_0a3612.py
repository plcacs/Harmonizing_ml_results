from typing import Any, Dict, List, Set

from django.db.models import QuerySet
from zerver.models import AlertWord, Recipient, Stream, Subscription, UserProfile, UserTopic

@dataclass
class SubInfo:
    user: UserProfile
    sub: Subscription
    stream: Stream

@dataclass
class SubscriberPeerInfo:
    pass

def get_active_subscriptions_for_stream_id(stream_id: int, *, include_deactivated_users: bool) -> QuerySet[Subscription]:
    ...

def get_active_subscriptions_for_stream_ids(stream_ids: List[int]) -> QuerySet[Subscription]:
    ...

def get_subscribed_stream_ids_for_user(user_profile: UserProfile) -> List[int]:
    ...

def get_subscribed_stream_recipient_ids_for_user(user_profile: UserProfile) -> List[int]:
    ...

def get_stream_subscriptions_for_user(user_profile: UserProfile) -> QuerySet[Subscription]:
    ...

def get_used_colors_for_user_ids(user_ids: List[int]) -> Dict[int, Set[str]]:
    ...

def get_bulk_stream_subscriber_info(users: List[UserProfile], streams: List[Stream]) -> Dict[int, List[SubInfo]]:
    ...

def num_subscribers_for_stream_id(stream_id: int) -> int:
    ...

def get_user_ids_for_streams(stream_ids: List[int]) -> Dict[int, Set[int]]:
    ...

def get_users_for_streams(stream_ids: List[int]) -> Dict[int, Set[UserProfile]]:
    ...

def handle_stream_notifications_compatibility(user_profile: UserProfile, stream_dict: Dict[str, Any], notification_settings_null: bool) -> None:
    ...

def subscriber_ids_with_stream_history_access(stream: Stream) -> Set[int]:
    ...

def get_subscriptions_for_send_message(*, realm_id: int, stream_id: int, topic_name: str, possible_stream_wildcard_mention: bool, topic_participant_user_ids: List[int], possibly_mentioned_user_ids: List[int]) -> QuerySet[Subscription]:
    ...

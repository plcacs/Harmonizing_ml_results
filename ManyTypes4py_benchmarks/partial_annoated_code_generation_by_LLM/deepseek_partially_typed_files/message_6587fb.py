import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, TypedDict, Optional, Union, List, Set, Dict, Tuple, cast
from django.conf import settings
from django.db import connection
from django.db.models import Exists, Max, OuterRef, QuerySet, Sum
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from psycopg2.sql import SQL
from analytics.lib.counts import COUNT_STATS
from analytics.models import RealmCount
from zerver.lib.cache import generic_bulk_cached_fetch, to_dict_cache_key_id
from zerver.lib.display_recipient import get_display_recipient_by_id
from zerver.lib.exceptions import JsonableError, MissingAuthenticationError
from zerver.lib.markdown import MessageRenderingResult
from zerver.lib.mention import MentionData
from zerver.lib.message_cache import MessageDict, extract_message_dict, stringify_message_dict
from zerver.lib.partial import partial
from zerver.lib.request import RequestVariableConversionError
from zerver.lib.stream_subscription import get_active_subscriptions_for_stream_id, get_stream_subscriptions_for_user, get_subscribed_stream_recipient_ids_for_user, num_subscribers_for_stream_id
from zerver.lib.streams import can_access_stream_history, get_web_public_streams_queryset
from zerver.lib.topic import MESSAGE__TOPIC, TOPIC_NAME, maybe_rename_general_chat_to_empty_topic, messages_for_topic
from zerver.lib.types import UserDisplayRecipient
from zerver.lib.user_groups import user_has_permission_for_group_setting
from zerver.lib.user_topics import build_get_topic_visibility_policy, get_topic_visibility_policy
from zerver.lib.users import get_inaccessible_user_ids
from zerver.models import Message, NamedUserGroup, Realm, Recipient, Stream, Subscription, UserMessage, UserProfile, UserTopic
from zerver.models.constants import MAX_TOPIC_NAME_LENGTH
from zerver.models.groups import SystemGroups
from zerver.models.messages import get_usermessage_by_message_id
from zerver.models.users import is_cross_realm_bot_email

class MessageDetailsDict(TypedDict, total=False):
    type: str
    mentioned: bool
    user_ids: List[int]
    stream_id: int
    topic: str
    unmuted_stream_msg: bool

class RawUnreadStreamDict(TypedDict):
    stream_id: int
    topic: str

class RawUnreadDirectMessageDict(TypedDict):
    other_user_id: int

class RawUnreadDirectMessageGroupDict(TypedDict):
    user_ids_string: str

class RawUnreadMessagesResult(TypedDict):
    pm_dict: Dict[int, RawUnreadDirectMessageDict]
    stream_dict: Dict[int, RawUnreadStreamDict]
    huddle_dict: Dict[int, RawUnreadDirectMessageGroupDict]
    mentions: Set[int]
    muted_stream_ids: Set[int]
    unmuted_stream_msgs: Set[int]
    old_unreads_missing: bool

class UnreadStreamInfo(TypedDict):
    stream_id: int
    topic: str
    unread_message_ids: List[int]

class UnreadDirectMessageInfo(TypedDict):
    other_user_id: int
    sender_id: int
    unread_message_ids: List[int]

class UnreadDirectMessageGroupInfo(TypedDict):
    user_ids_string: str
    unread_message_ids: List[int]

class UnreadMessagesResult(TypedDict):
    pms: List[UnreadDirectMessageInfo]
    streams: List[UnreadStreamInfo]
    huddles: List[UnreadDirectMessageGroupInfo]
    mentions: List[int]
    count: int
    old_unreads_missing: bool

@dataclass
class SendMessageRequest:
    message: Message
    rendering_result: MessageRenderingResult
    stream: Optional[Stream]
    sender_muted_stream: Optional[bool]
    local_id: Optional[str]
    sender_queue_id: Optional[str]
    realm: Realm
    mention_data: MentionData
    mentioned_user_groups_map: Dict[int, int]
    active_user_ids: Set[int]
    online_push_user_ids: Set[int]
    dm_mention_push_disabled_user_ids: Set[int]
    dm_mention_email_disabled_user_ids: Set[int]
    stream_push_user_ids: Set[int]
    stream_email_user_ids: Set[int]
    followed_topic_push_user_ids: Set[int]
    followed_topic_email_user_ids: Set[int]
    muted_sender_user_ids: Set[int]
    um_eligible_user_ids: Set[int]
    long_term_idle_user_ids: Set[int]
    default_bot_user_ids: Set[int]
    service_bot_tuples: List[Tuple[int, int]]
    all_bot_user_ids: Set[int]
    topic_wildcard_mention_user_ids: Set[int]
    stream_wildcard_mention_user_ids: Set[int]
    topic_wildcard_mention_in_followed_topic_user_ids: Set[int]
    stream_wildcard_mention_in_followed_topic_user_ids: Set[int]
    topic_participant_user_ids: Set[int]
    links_for_embed: Set[str]
    widget_content: Optional[Dict[str, Any]]
    submessages: List[Any] = field(default_factory=list)
    deliver_at: Optional[Any] = None
    delivery_type: Optional[Any] = None
    limit_unread_user_ids: Optional[Any] = None
    service_queue_events: Optional[Any] = None
    disable_external_notifications: bool = False
    automatic_new_visibility_policy: Optional[Any] = None
    recipients_for_user_creation_events: Optional[Any] = None

MAX_UNREAD_MESSAGES: int = 50000

def truncate_content(content: str, max_length: int, truncation_message: str) -> str:
    if len(content) > max_length:
        content = content[:max_length - len(truncation_message)] + truncation_message
    return content

def normalize_body(body: str) -> str:
    body = body.rstrip().lstrip('\n')
    if len(body) == 0:
        raise JsonableError(_('Message must not be empty'))
    if '\x00' in body:
        raise JsonableError(_('Message must not contain null bytes'))
    return truncate_content(body, settings.MAX_MESSAGE_LENGTH, '\n[message truncated]')

def normalize_body_for_import(body: str) -> str:
    if '\x00' in body:
        body = re.sub('\\x00', '', body)
    return truncate_content(body, settings.MAX_MESSAGE_LENGTH, '\n[message truncated]')

def truncate_topic(topic_name: str) -> str:
    return truncate_content(topic_name, MAX_TOPIC_NAME_LENGTH, '...')

def messages_for_ids(message_ids: List[int], user_message_flags: Dict[int, List[str]], search_fields: Dict[int, Any], apply_markdown: bool, client_gravatar: bool, allow_empty_topic_name: bool, allow_edit_history: bool, user_profile: Optional[UserProfile], realm: Realm) -> List[Dict[str, Any]]:
    def id_fetcher(row: Dict[str, Any]) -> int:
        return row['id']
    message_dicts = generic_bulk_cached_fetch(to_dict_cache_key_id, MessageDict.ids_to_dict, message_ids, id_fetcher=id_fetcher, cache_transformer=lambda obj: obj, extractor=extract_message_dict, setter=stringify_message_dict)
    message_list: List[Dict[str, Any]] = []
    sender_ids = [message_dicts[message_id]['sender_id'] for message_id in message_ids]
    inaccessible_sender_ids = get_inaccessible_user_ids(sender_ids, user_profile) if user_profile is not None else set()
    for message_id in message_ids:
        msg_dict = message_dicts[message_id]
        flags = user_message_flags.get(message_id, [])
        if 'stream_wildcard_mentioned' in flags or 'topic_wildcard_mentioned' in flags:
            flags.append('wildcard_mentioned')
        msg_dict.update(flags=flags)
        if message_id in search_fields:
            msg_dict.update(search_fields[message_id])
        if 'edit_history' in msg_dict and (not allow_edit_history):
            del msg_dict['edit_history']
        msg_dict['can_access_sender'] = msg_dict['sender_id'] not in inaccessible_sender_ids
        message_list.append(msg_dict)
    MessageDict.post_process_dicts(message_list, apply_markdown=apply_markdown, client_gravatar=client_gravatar, allow_empty_topic_name=allow_empty_topic_name, realm=realm, user_recipient_id=None if user_profile is None else user_profile.recipient_id)
    return message_list

def access_message(user_profile: UserProfile, message_id: int, lock_message: bool = False) -> Message:
    try:
        base_query = Message.objects.select_related(*Message.DEFAULT_SELECT_RELATED)
        if lock_message:
            base_query = base_query.select_for_update(of=('self',))
        message = base_query.get(id=message_id)
    except Message.DoesNotExist:
        raise JsonableError(_('Invalid message(s)'))
    def has_user_message() -> bool:
        return UserMessage.objects.filter(user_profile=user_profile, message_id=message_id).exists()
    if has_message_access(user_profile, message, has_user_message=has_user_message):
        return message
    raise JsonableError(_('Invalid message(s)'))

def access_message_and_usermessage(user_profile: UserProfile, message_id: int, lock_message: bool = False) -> Tuple[Message, Optional[UserMessage]]:
    try:
        base_query = Message.objects.select_related(*Message.DEFAULT_SELECT_RELATED)
        if lock_message:
            base_query = base_query.select_for_update(of=('self',))
        message = base_query.get(id=message_id)
    except Message.DoesNotExist:
        raise JsonableError(_('Invalid message(s)'))
    user_message = get_usermessage_by_message_id(user_profile, message_id)
    def has_user_message() -> bool:
        return user_message is not None
    if has_message_access(user_profile, message, has_user_message=has_user_message):
        return (message, user_message)
    raise JsonableError(_('Invalid message(s)'))

def access_web_public_message(realm: Realm, message_id: int) -> Message:
    if not realm.web_public_streams_enabled():
        raise MissingAuthenticationError
    try:
        message = Message.objects.select_related(*Message.DEFAULT_SELECT_RELATED).get(id=message_id)
    except Message.DoesNotExist:
        raise MissingAuthenticationError
    if not message.is_stream_message():
        raise MissingAuthenticationError
    queryset = get_web_public_streams_queryset(realm)
    try:
        stream = queryset.get(id=message.recipient.type_id)
    except Stream.DoesNotExist:
        raise MissingAuthenticationError
    assert stream.is_web_public
    assert not stream.deactivated
    assert not stream.invite_only
    assert stream.history_public_to_subscribers
    return message

def has_message_access(user_profile: UserProfile, message: Message, *, has_user_message: Callable[[], bool], stream: Optional[Stream] = None, is_subscribed: Optional[bool] = None) -> bool:
    if message.recipient.type != Recipient.STREAM:
        return has_user_message()
    if stream is None:
        stream = Stream.objects.get(id=message.recipient.type_id)
    else:
        assert stream.recipient_id == message.recipient_id
    if stream.realm_id != user_profile.realm_id:
        return False
    if stream.deactivated:
        return False

    def is_subscribed_helper() -> bool:
        if is_subscribed is not None:
            return is_subscribed
        return Subscription.objects.filter(user_profile=user_profile, active=True, recipient=message.recipient).exists()
    if stream.is_public() and user_profile.can_access_public_streams():
        return True
    if not stream.is_history_public_to_subscribers():
        return has_user_message() and is_subscribed_helper()
    return is_subscribed_helper()

def event_recipient_ids_for_action_on_messages(messages: List[Message], *, channel: Optional[Stream] = None, exclude_long_term_idle_users: bool = True) -> Set[int]:
    assert len(messages) > 0
    message_ids = [message.id for message in messages]

    def get_user_ids_having_usermessage_row_for_messages(message_ids: List[int]) -> Set[int]:
        usermessages = UserMessage.objects.filter(message_id__in=message_ids)
        if exclude_long_term_idle_users:
            usermessages = usermessages.exclude(user_profile__long_term_idle=True)
        return set(usermessages.values_list('user_profile_id', flat=True))
    sample_message = messages[0]
    if not sample_message.is_stream_message():
        return get_user_ids_having_usermessage_row_for_messages(message_ids)
    channel_id = sample_message.recipient.type_id
    if channel is None:
        channel = Stream.objects.get(id=channel_id)
    subscriptions = get_active_subscriptions_for_stream_id(channel_id, include_deactivated_users=False)
    if exclude_long_term_idle_users:
        subscriptions = subscriptions.exclude(user_profile__long_term_idle=True)
    subscriber_ids = set(subscriptions.values_list('user_profile_id', flat=True))
    if not channel.is_history_public_to_subscribers():
        assert not channel.is_public()
        user_ids_with_usermessage_row = get_user_ids_having_usermessage_row_for_messages(message_ids)
        return user_ids_with_usermessage_row & subscriber_ids
    if not channel.is_public():
        return subscriber_ids
    usermessage_rows = UserMessage.objects.filter(message_id__in=message_ids).exclude(user_profile__role=UserProfile.ROLE_GUEST)
    if exclude_long_term_idle_users:
        usermessage_rows = usermessage_rows.exclude(user_profile__long_term_idle=True)
    user_ids_with_usermessage_row_and_channel_access = set(usermessage_rows.values_list('user_profile_id', flat=True))
    return user_ids_with_usermessage_row_and_channel_access | subscriber_ids

def bulk_access_messages(user_profile: UserProfile, messages: List[Message], *, stream: Optional[Stream] = None) -> List[Message]:
    filtered_messages: List[Message] = []
    user_message_set = set(get_messages_with_usermessage_rows_for_user(user_profile.id, [message.id for message in messages]))
    streams: Dict[int, Stream] = {}
    if stream is None:
        streams = {stream.recipient_id: stream for stream in Stream.objects.filter(id__in={message.recipient.type_id for message in messages if message.recipient.type == Recipient.STREAM})}
    subscribed_recipient_ids = set(get_subscribed_stream_recipient_ids_for_user(user_profile))
    for message in messages:
        is_subscribed = message.recipient_id in subscribed_recipient_ids
        message_stream = streams.get(message.recipient_id) if stream is None else stream
        if has_message_access(user_profile, message, has_user_message=lambda: message.id in user_message_set, stream=message_stream, is_subscribed=is_subscribed):
            filtered_messages.append(message)
    return filtered_messages

def bulk_access_stream_messages_query(user_profile: UserProfile, messages: QuerySet[Message], stream: Stream) -> QuerySet[Message]:
    assert stream.recipient_id is not None
    messages = messages.filter(realm_id=user_profile.realm_id, recipient_id=stream.recipient_id)
    if stream.is_public() and user_profile.can_access_public_streams():
        return messages
    if not Subscription.objects.filter(user_profile=user_profile, active=True, recipient=stream.recipient).exists():
        return Message.objects.none()
    if not stream.is_history_public_to_subscribers():
        messages = messages.alias(has_usermessage=Exists(UserMessage.objects.filter(user_profile_id=user_profile.id, message_id=OuterRef('id')))).filter(has_usermessage=True)
    return messages

def get_messages_with_usermessage_rows_for_user(user_profile_id: int, message_ids: Sequence[int]) -> QuerySet:
    return UserMessage.objects.filter(user_profile_id=user_profile_id, message_id__in=message_ids).values_list('message_id', flat=True)

def direct_message_group_users(recipient_id: int) -> str:
    display_recipient: List[UserDisplayRecipient] = get_display_recipient_by_id(recipient_id, Recipient.DIRECT_MESSAGE_GROUP, None)
    user_ids: List[int] = [obj['id'] for obj in display_recipient]
    user_ids = sorted(user_ids)
    return ','.join((str(uid) for uid in user_ids))

def get_inactive_recipient_ids(user_profile: UserProfile) -> List[int]:
    rows = get_stream_subscriptions_for_user(user_profile).filter(active=False).values('recipient_id')
    inactive_recipient_ids = [row['recipient_id'] for row in rows]
    return inactive_recipient_ids

def get_muted_stream_ids(user_profile: UserProfile) -> Set[int]:
    rows = get_stream_subscriptions_for_user(user_profile).filter(active=True, is_muted=True).values('recipient__type_id')
    muted_stream_ids = {row['recipient__type_id'] for row in rows}
    return muted_stream_ids

def get_starred_message_ids(user_profile: UserProfile) -> List[int]:
    return list(UserMessage.objects.filter(user_profile=user_profile).extra(where=[UserMessage.where_starred()]).order_by('message_id').values_list('message_id', flat=True)[0:10000])

def get_raw_unread_data(user_profile: UserProfile, message_ids: Optional[List[int]] = None) -> RawUnreadMessagesResult:
    excluded_recipient_ids = get_inactive_recipient_ids(user_profile)
    first_visible_message_id = get_first_visible_message_id(user_profile.realm)
    user_msgs = UserMessage.objects.filter(user_profile=user_profile, message_id__gte=first_visible_message_id).exclude(message__recipient_id__in=excluded_recipient_ids).values('message_id', 'message__sender_id', MESSAGE__TOPIC, 'message__recipient
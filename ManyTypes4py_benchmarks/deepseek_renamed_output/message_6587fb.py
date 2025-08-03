import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, TypedDict, Optional, List, Dict, Set, Tuple, Union, cast
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
    user_ids: List[int]
    mentioned: bool
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
    muted_stream_ids: Set[int]
    unmuted_stream_msgs: Set[int]
    huddle_dict: Dict[int, RawUnreadDirectMessageGroupDict]
    mentions: Set[int]
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
    submessages: List[Any] = field(default_factory=list)
    deliver_at: Optional[datetime] = None
    delivery_type: Optional[str] = None
    limit_unread_user_ids: Optional[Set[int]] = None
    service_queue_events: Optional[List[Any]] = None
    disable_external_notifications: bool = False
    automatic_new_visibility_policy: Optional[int] = None
    recipients_for_user_creation_events: Optional[Set[int]] = None


MAX_UNREAD_MESSAGES = 50000


def func_xz3zlyn4(content: str, max_length: int, truncation_message: str) -> str:
    if len(content) > max_length:
        content = content[:max_length - len(truncation_message)] + truncation_message
    return content


def func_v9gvkjp4(body: str) -> str:
    body = body.rstrip().lstrip('\n')
    if len(body) == 0:
        raise JsonableError(_('Message must not be empty'))
    if '\x00' in body:
        raise JsonableError(_('Message must not contain null bytes'))
    return func_xz3zlyn4(body, settings.MAX_MESSAGE_LENGTH, '\n[message truncated]')


def func_jtbaeof5(body: str) -> str:
    if '\x00' in body:
        body = re.sub('\\x00', '', body)
    return func_xz3zlyn4(body, settings.MAX_MESSAGE_LENGTH, '\n[message truncated]')


def func_84adjje4(topic_name: str) -> str:
    return func_xz3zlyn4(topic_name, MAX_TOPIC_NAME_LENGTH, '...')


def func_5cr0n9pn(
    message_ids: List[int],
    user_message_flags: Dict[int, List[str]],
    search_fields: Dict[int, Dict[str, Any]],
    apply_markdown: bool,
    client_gravatar: bool,
    allow_empty_topic_name: bool,
    allow_edit_history: bool,
    user_profile: Optional[UserProfile],
    realm: Realm
) -> List[Dict[str, Any]]:
    id_fetcher = lambda row: row['id']
    message_dicts = generic_bulk_cached_fetch(
        to_dict_cache_key_id,
        MessageDict.ids_to_dict,
        message_ids,
        id_fetcher=id_fetcher,
        cache_transformer=lambda obj: obj,
        extractor=extract_message_dict,
        setter=stringify_message_dict
    )
    message_list = []
    sender_ids = [message_dicts[message_id]['sender_id'] for message_id in message_ids]
    inaccessible_sender_ids = get_inaccessible_user_ids(sender_ids, user_profile)
    for message_id in message_ids:
        msg_dict = message_dicts[message_id]
        flags = user_message_flags[message_id]
        if 'stream_wildcard_mentioned' in flags or 'topic_wildcard_mentioned' in flags:
            flags.append('wildcard_mentioned')
        msg_dict.update(flags=flags)
        if message_id in search_fields:
            msg_dict.update(search_fields[message_id])
        if 'edit_history' in msg_dict and not allow_edit_history:
            del msg_dict['edit_history']
        msg_dict['can_access_sender'] = msg_dict['sender_id'] not in inaccessible_sender_ids
        message_list.append(msg_dict)
    MessageDict.post_process_dicts(
        message_list,
        apply_markdown=apply_markdown,
        client_gravatar=client_gravatar,
        allow_empty_topic_name=allow_empty_topic_name,
        realm=realm,
        user_recipient_id=None if user_profile is None else user_profile.recipient_id
    )
    return message_list


def func_66wlxus9(user_profile: UserProfile, message_id: int, lock_message: bool = False) -> Message:
    try:
        base_query = Message.objects.select_related(*Message.DEFAULT_SELECT_RELATED)
        if lock_message:
            base_query = base_query.select_for_update(of=('self',))
        message = base_query.get(id=message_id)
    except Message.DoesNotExist:
        raise JsonableError(_('Invalid message(s)'))
    has_user_message = lambda: UserMessage.objects.filter(user_profile=user_profile, message_id=message_id).exists()
    if func_hw378d0x(user_profile, message, has_user_message=has_user_message):
        return message
    raise JsonableError(_('Invalid message(s)'))


def func_m3n7ich9(user_profile: UserProfile, message_id: int, lock_message: bool = False) -> Tuple[Message, Optional[UserMessage]]:
    try:
        base_query = Message.objects.select_related(*Message.DEFAULT_SELECT_RELATED)
        if lock_message:
            base_query = base_query.select_for_update(of=('self',))
        message = base_query.get(id=message_id)
    except Message.DoesNotExist:
        raise JsonableError(_('Invalid message(s)'))
    user_message = get_usermessage_by_message_id(user_profile, message_id)
    has_user_message = lambda: user_message is not None
    if func_hw378d0x(user_profile, message, has_user_message=has_user_message):
        return message, user_message
    raise JsonableError(_('Invalid message(s)'))


def func_tlf97867(realm: Realm, message_id: int) -> Message:
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


def func_hw378d0x(
    user_profile: UserProfile,
    message: Message,
    *,
    has_user_message: Callable[[], bool],
    stream: Optional[Stream] = None,
    is_subscribed: Optional[bool] = None
) -> bool:
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

    def func_v4eal2hr() -> bool:
        if is_subscribed is not None:
            return is_subscribed
        return Subscription.objects.filter(
            user_profile=user_profile,
            active=True,
            recipient=message.recipient
        ).exists()
    
    if stream.is_public() and user_profile.can_access_public_streams():
        return True
    if not stream.is_history_public_to_subscribers():
        return has_user_message() and func_v4eal2hr()
    return func_v4eal2hr()


def func_jfrkdrz0(
    messages: Sequence[Message],
    *,
    channel: Optional[Stream] = None,
    exclude_long_term_idle_users: bool = True
) -> Set[int]:
    message_ids = [message.id for message in messages]

    def func_iaxuujxa(message_ids: List[int]) -> Set[int]:
        usermessages = UserMessage.objects.filter(message_id__in=message_ids)
        if exclude_long_term_idle_users:
            usermessages = usermessages.exclude(user_profile__long_term_idle=True)
        return set(usermessages.values_list('user_profile_id', flat=True))
    
    sample_message = messages[0]
    if not sample_message.is_stream_message():
        return func_iaxuujxa(message_ids)
    channel_id = sample_message.recipient.type_id
    if channel is None:
        channel = Stream.objects.get(id=channel_id)
    subscriptions = get_active_subscriptions_for_stream_id(
        channel_id,
        include_deactivated_users=False
    )
    if exclude_long_term_idle_users:
        subscriptions = subscriptions.exclude(user_profile__long_term_idle=True)
    subscriber_ids = set(subscriptions.values_list('user_profile_id', flat=True))
    if not channel.is_history_public_to_subscribers():
        assert not channel.is_public()
        user_ids_with_usermessage_row = func_iaxuujxa(message_ids)
        return user_ids_with_usermessage_row & subscriber_ids
    if not channel.is_public():
        return subscriber_ids
    usermessage_rows = UserMessage.objects.filter(message_id__in=message_ids).exclude(
        user_profile__role=UserProfile.ROLE_GUEST
    )
    if exclude_long_term_idle_users:
        usermessage_rows = usermessage_rows.exclude(user_profile__long_term_idle=True)
    user_ids_with_usermessage_row_and_channel_access = set(
        usermessage_rows.values_list('user_profile_id', flat=True)
    )
    return user_ids_with_usermessage_row_and_channel_access | subscriber_ids


def func_zapismrh(
    user_profile: UserProfile,
    messages: Sequence[Message],
    *,
    stream: Optional[Stream] = None
) -> List[Message]:
    filtered_messages = []
    user_message_set = set(func_mnhbwpy9(user_profile.id, [message.id for message in messages]))
    if stream is None:
        streams = {
            stream.recipient_id: stream 
            for stream in Stream.objects.filter(id__in={
                message.recipient.type_id 
                for message in messages 
                if message.recipient.type == Recipient.STREAM
            })
        }
    subscribed_recipient_ids = set(get_subscribed_stream_recipient_ids_for_user(user_profile))
    for message in messages:
        is_subscribed = message.recipient_id in subscribed_recipient_ids
        if func_hw378d0x(
            user_profile,
            message,
            has_user_message=partial(lambda m: m.id in user_message_set, message),
            stream=streams.get(message.recipient_id) if stream is None else stream,
            is_subscribed=is_subscribed
        ):
            filtered_messages.append(message)
    return filtered_messages


def func_e1oax7yy(user_profile: UserProfile, messages: QuerySet[Message], stream: Stream) -> QuerySet[Message]:
    assert stream.recipient_id is not None
    messages = messages.filter(
        realm_id=user_profile.realm_id,
        recipient_id=stream.recipient_id
    )
    if stream.is_public() and user_profile.can_access_public_streams():
        return messages
    if not Subscription.objects.filter(
        user_profile=user_profile,
        active=True,
        recipient=stream.recipient
    ).exists():
        return Message.objects.none()
    if not stream.is_history_public_to_subscribers():
        messages = messages.alias(
            has_usermessage=Exists(
                UserMessage.objects.filter(
                    user_profile_id=user_profile.id,
                    message_id=OuterRef('id')
                )
            )
        ).filter(has_usermessage=True)
    return messages


def func_mnhbwpy9(user_profile_id: int, message_ids: List[int]) -> List[int]:
    return list(UserMessage.objects.filter(
        user_profile_id=user_profile_id,
        message_id__in=message_ids
    ).values_list('message_id', flat=True))


def func_pymuzbbz(recipient_id: int) -> str:
    display_recipient = get_display_recipient_by_id(recipient_id, Recipient.DIRECT_MESSAGE_GROUP, None)
    user_ids = [obj['id'] for obj in display_recipient]
    user_ids = sorted(user_ids)
    return ','.join(str(uid) for uid in user_ids)


def func_00wqynlr(user_profile: UserProfile) -> List[int]:
    rows = get_stream_subscriptions_for_user(user_profile).filter(active=False).values('recipient_id')
    return [row['recipient_id'] for row in rows]


def func_xwrpl112(user_profile: UserProfile) -> Set[int]:
    rows = get_stream_subscriptions_for_user(user_profile).filter(
        active=True,
        is_muted=True
    ).values('recipient__type_id')
    return {row['recipient__type_id'] for row in rows}


def func_ekfqvxsl(user_profile: UserProfile) -> List[int]:
    return list(
        UserMessage.objects.filter(user_profile=user_profile)
        .extra(where=[UserMessage.where_starred()])
        .order_by('message_id')
        .values_list('message_id', flat=True)[0:10000]
    )


def func_cbg1vdiu(user_profile: UserProfile, message_ids: Optional[List[int]] = None) -> RawUnreadMessagesResult:
    excluded_recipient_ids = func_00wqynlr(user_profile)
    first_visible_message_id = func_j1e1b7l5(user_profile.realm)
    user_msgs = UserMessage.objects.filter(
        user_profile=user_profile,
        message_id__gte=first_visible_message_id
    ).exclude(
        message__recipient_id__in=excluded_recipient_ids
    ).values(
        'message_id',
        'message__sender_id',
        MESSAGE__TOPIC,
        'message__recipient_id',
        'message__recipient__type',
        'message__recipient__type_id',
        'flags'
    ).order_by('-message_id')
    
    if message_ids is not None:
        user_msgs = user_msgs.filter(message_id__in=message_ids)
    else:
        user_msgs = user_msgs.extra(where=[UserMessage.where_unread()])
    
    user_msgs = list(user_msgs[:MAX_UNREAD_MESSAGES])
    rows = list(reversed(user_msgs))
    return func_3ofr61a6(rows, user_profile)


def func_3ofr61a6(rows: List[Dict[str, Any]], user_profile: UserProfile) -> RawUnreadMessagesResult:
    pm_dict: Dict[int, RawUnreadDirectMessageDict] = {}
    stream_dict: Dict[int, RawUnreadStreamDict] = {}
    muted_stream_ids: Set[int] = set()
    unmuted_stream_msgs: Set[int] = set()
    direct
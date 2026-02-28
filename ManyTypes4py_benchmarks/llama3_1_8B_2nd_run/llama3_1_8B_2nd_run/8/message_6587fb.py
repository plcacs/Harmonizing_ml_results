import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, TypedDict
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
    pass

class RawUnreadStreamDict(TypedDict):
    pass

class RawUnreadDirectMessageDict(TypedDict):
    pass

class RawUnreadDirectMessageGroupDict(TypedDict):
    pass

class RawUnreadMessagesResult(TypedDict):
    pass

class UnreadStreamInfo(TypedDict):
    pass

class UnreadDirectMessageInfo(TypedDict):
    pass

class UnreadDirectMessageGroupInfo(TypedDict):
    pass

class UnreadMessagesResult(TypedDict):
    pass

@dataclass
class SendMessageRequest:
    submessages: list = field(default_factory=list)
    deliver_at: Any = None
    delivery_type: Any = None
    limit_unread_user_ids: Any = None
    service_queue_events: Any = None
    disable_external_notifications: bool = False
    automatic_new_visibility_policy: Any = None
    recipients_for_user_creation_events: Any = None

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

def messages_for_ids(message_ids: Sequence[int], user_message_flags: Mapping[int, Any], search_fields: Mapping[int, Any], apply_markdown: bool, client_gravatar: bool, allow_empty_topic_name: bool, allow_edit_history: bool, user_profile: UserProfile, realm: Realm) -> Sequence[MessageDict]:
    id_fetcher = lambda row: row['id']
    message_dicts = generic_bulk_cached_fetch(to_dict_cache_key_id, MessageDict.ids_to_dict, message_ids, id_fetcher=id_fetcher, cache_transformer=lambda obj: obj, extractor=extract_message_dict, setter=stringify_message_dict)
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
        if 'edit_history' in msg_dict and (not allow_edit_history):
            del msg_dict['edit_history']
        msg_dict['can_access_sender'] = msg_dict['sender_id'] not in inaccessible_sender_ids
        message_list.append(msg_dict)
    MessageDict.post_process_dicts(message_list, apply_markdown=apply_markdown, client_gravatar=client_gravatar, allow_empty_topic_name=allow_empty_topic_name, realm=realm, user_recipient_id=None if user_profile is None else user_profile.recipient_id)
    return message_list

def access_message(user_profile: UserProfile, message_id: int, lock_message: bool = False) -> Message:
    """You can access a message by ID in our APIs that either:
    (1) You received or have previously accessed via starring
        (aka have a UserMessage row for).
    (2) Was sent to a public stream in your realm.

    We produce consistent, boring error messages to avoid leaking any
    information from a security perspective.

    The lock_message parameter should be passed by callers that are
    planning to modify the Message object. This will use the SQL
    `SELECT FOR UPDATE` feature to ensure that other processes cannot
    delete the message during the current transaction, which is
    important to prevent rare race conditions. Callers must only
    pass lock_message when inside a @transaction.atomic block.
    """
    try:
        base_query = Message.objects.select_related(*Message.DEFAULT_SELECT_RELATED)
        if lock_message:
            base_query = base_query.select_for_update(of=('self',))
        message = base_query.get(id=message_id)
    except Message.DoesNotExist:
        raise JsonableError(_('Invalid message(s)'))
    has_user_message = lambda: UserMessage.objects.filter(user_profile=user_profile, message_id=message_id).exists()
    if has_message_access(user_profile, message, has_user_message=has_user_message):
        return message
    raise JsonableError(_('Invalid message(s)'))

def access_message_and_usermessage(user_profile: UserProfile, message_id: int, lock_message: bool = False) -> tuple[Message, UserMessage]:
    """As access_message, but also returns the usermessage, if any."""
    try:
        base_query = Message.objects.select_related(*Message.DEFAULT_SELECT_RELATED)
        if lock_message:
            base_query = base_query.select_for_update(of=('self',))
        message = base_query.get(id=message_id)
    except Message.DoesNotExist:
        raise JsonableError(_('Invalid message(s)'))
    user_message = get_usermessage_by_message_id(user_profile, message_id)
    has_user_message = lambda: user_message is not None
    if has_message_access(user_profile, message, has_user_message=has_user_message):
        return (message, user_message)
    raise JsonableError(_('Invalid message(s)'))

def access_web_public_message(realm: Realm, message_id: int) -> Message:
    """Access control method for unauthenticated requests interacting
    with a message in web-public streams.
    """
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

def has_message_access(user_profile: UserProfile, message: Message, *, has_user_message: Callable[[], bool], stream: Stream = None, is_subscribed: Callable[[], bool] = None) -> bool:
    """
    Returns whether a user has access to a given message.

    * The user_message parameter must be provided if the user has a UserMessage
      row for the target message.
    * The optional stream parameter is validated; is_subscribed is not.
    """
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
            return is_subscribed()
        return Subscription.objects.filter(user_profile=user_profile, active=True, recipient=message.recipient).exists()
    if stream.is_public() and user_profile.can_access_public_streams():
        return True
    if not stream.is_history_public_to_subscribers():
        return has_user_message() and is_subscribed_helper()
    return is_subscribed_helper()

def event_recipient_ids_for_action_on_messages(messages: Sequence[Message], *, channel: Stream = None, exclude_long_term_idle_users: bool = True) -> Sequence[int]:
    """Returns IDs of users who should receive events when an action
    (delete, react, etc) is performed on given set of messages, which
    are expected to all be in a single conversation.

    This function aligns with the 'has_message_access' above to ensure
    that events reach only those users who have access to the messages.

    Notably, for performance reasons, we do not send live-update
    events to everyone who could potentially have a cached copy of a
    message because they fetched messages in a public channel to which
    they are not subscribed. Such events are limited to those messages
    where the user has a UserMessage row (including `historical` rows).
    """
    assert len(messages) > 0
    message_ids = [message.id for message in messages]

    def get_user_ids_having_usermessage_row_for_messages(message_ids: Sequence[int]) -> set[int]:
        """Returns the IDs of users who actually received the messages."""
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

def bulk_access_messages(user_profile: UserProfile, messages: Sequence[Message], *, stream: Stream = None) -> Sequence[Message]:
    """This function does the full has_message_access check for each
    message.  If stream is provided, it is used to avoid unnecessary
    database queries, and will use exactly 2 bulk queries instead.

    Throws AssertionError if stream is passed and any of the messages
    were not sent to that stream.

    """
    filtered_messages = []
    user_message_set = set(get_messages_with_usermessage_rows_for_user(user_profile.id, [message.id for message in messages]))
    if stream is None:
        streams = {stream.recipient_id: stream for stream in Stream.objects.filter(id__in={message.recipient.type_id for message in messages if message.recipient.type == Recipient.STREAM})}
    subscribed_recipient_ids = set(get_subscribed_stream_recipient_ids_for_user(user_profile))
    for message in messages:
        is_subscribed = message.recipient_id in subscribed_recipient_ids
        if has_message_access(user_profile, message, has_user_message=partial(lambda m: m.id in user_message_set, message), stream=streams.get(message.recipient_id) if stream is None else stream, is_subscribed=is_subscribed):
            filtered_messages.append(message)
    return filtered_messages

def bulk_access_stream_messages_query(user_profile: UserProfile, messages: QuerySet[Message], stream: Stream) -> QuerySet[Message]:
    """This function mirrors bulk_access_messages, above, but applies the
    limits to a QuerySet and returns a new QuerySet which only
    contains messages in the given stream which the user can access.
    Note that this only works with streams.  It may return an empty
    QuerySet if the user has access to no messages (for instance, for
    a private stream which the user is not subscribed to).

    """
    assert stream.recipient_id is not None
    messages = messages.filter(realm_id=user_profile.realm_id, recipient_id=stream.recipient_id)
    if stream.is_public() and user_profile.can_access_public_streams():
        return messages
    if not Subscription.objects.filter(user_profile=user_profile, active=True, recipient=stream.recipient).exists():
        return Message.objects.none()
    if not stream.is_history_public_to_subscribers():
        messages = messages.alias(has_usermessage=Exists(UserMessage.objects.filter(user_profile_id=user_profile.id, message_id=OuterRef('id')))).filter(has_usermessage=True)
    return messages

def get_messages_with_usermessage_rows_for_user(user_profile_id: int, message_ids: Sequence[int]) -> Sequence[int]:
    """
    Returns a subset of `message_ids` containing only messages the
    user has a UserMessage for.  Makes O(1) database queries.
    Note that this is not sufficient for access verification for
    stream messages.

    See `access_message`, `bulk_access_messages` for proper message access
    checks that follow our security model.
    """
    return UserMessage.objects.filter(user_profile_id=user_profile_id, message_id__in=message_ids).values_list('message_id', flat=True)

def direct_message_group_users(recipient_id: int) -> str:
    display_recipient = get_display_recipient_by_id(recipient_id, Recipient.DIRECT_MESSAGE_GROUP, None)
    user_ids = [obj['id'] for obj in display_recipient]
    user_ids = sorted(user_ids)
    return ','.join((str(uid) for uid in user_ids))

def get_inactive_recipient_ids(user_profile: UserProfile) -> Sequence[int]:
    rows = get_stream_subscriptions_for_user(user_profile).filter(active=False).values('recipient_id')
    inactive_recipient_ids = [row['recipient_id'] for row in rows]
    return inactive_recipient_ids

def get_muted_stream_ids(user_profile: UserProfile) -> set[int]:
    rows = get_stream_subscriptions_for_user(user_profile).filter(active=True, is_muted=True).values('recipient__type_id')
    muted_stream_ids = {row['recipient__type_id'] for row in rows}
    return muted_stream_ids

def get_starred_message_ids(user_profile: UserProfile) -> Sequence[int]:
    return list(UserMessage.objects.filter(user_profile=user_profile).extra(where=[UserMessage.where_starred()]).order_by('message_id').values_list('message_id', flat=True)[0:10000])

def get_raw_unread_data(user_profile: UserProfile, message_ids: Sequence[int] = None) -> RawUnreadMessagesResult:
    excluded_recipient_ids = get_inactive_recipient_ids(user_profile)
    first_visible_message_id = get_first_visible_message_id(user_profile.realm)
    user_msgs = UserMessage.objects.filter(user_profile=user_profile, message_id__gte=first_visible_message_id).exclude(message__recipient_id__in=excluded_recipient_ids).values('message_id', 'message__sender_id', MESSAGE__TOPIC, 'message__recipient_id', 'message__recipient__type', 'message__recipient__type_id', 'flags').order_by('-message_id')
    if message_ids is not None:
        user_msgs = user_msgs.filter(message_id__in=message_ids)
    else:
        user_msgs = user_msgs.extra(where=[UserMessage.where_unread()])
    user_msgs = list(user_msgs[:MAX_UNREAD_MESSAGES])
    rows = list(reversed(user_msgs))
    return extract_unread_data_from_um_rows(rows, user_profile)

def extract_unread_data_from_um_rows(rows: Sequence[dict[str, Any]], user_profile: UserProfile) -> RawUnreadMessagesResult:
    pm_dict = {}
    stream_dict = {}
    muted_stream_ids = set()
    unmuted_stream_msgs = set()
    direct_message_group_dict = {}
    mentions = set()
    total_unreads = 0
    raw_unread_messages = dict(pm_dict=pm_dict, stream_dict=stream_dict, muted_stream_ids=muted_stream_ids, unmuted_stream_msgs=unmuted_stream_msgs, huddle_dict=direct_message_group_dict, mentions=mentions, old_unreads_missing=False)
    if user_profile is None:
        return raw_unread_messages
    muted_stream_ids = get_muted_stream_ids(user_profile)
    raw_unread_messages['muted_stream_ids'] = muted_stream_ids
    get_topic_visibility_policy = build_get_topic_visibility_policy(user_profile)

    def is_row_muted(stream_id: int, recipient_id: int, topic_name: str) -> bool:
        stream_muted = stream_id in muted_stream_ids
        visibility_policy = get_topic_visibility_policy(recipient_id, topic_name)
        if stream_muted and visibility_policy in [UserTopic.VisibilityPolicy.UNMUTED, UserTopic.VisibilityPolicy.FOLLOWED]:
            return False
        if stream_muted:
            return True
        if visibility_policy == UserTopic.VisibilityPolicy.MUTED:
            return True
        return False
    direct_message_group_cache = {}

    def get_direct_message_group_users(recipient_id: int) -> str:
        if recipient_id in direct_message_group_cache:
            return direct_message_group_cache[recipient_id]
        user_ids_string = direct_message_group_users(recipient_id)
        direct_message_group_cache[recipient_id] = user_ids_string
        return user_ids_string
    for row in rows:
        total_unreads += 1
        message_id = row['message_id']
        msg_type = row['message__recipient__type']
        recipient_id = row['message__recipient_id']
        sender_id = row['message__sender_id']
        if msg_type == Recipient.STREAM:
            stream_id = row['message__recipient__type_id']
            topic_name = row[MESSAGE__TOPIC]
            stream_dict[message_id] = dict(stream_id=stream_id, topic=topic_name)
            if not is_row_muted(stream_id, recipient_id, topic_name):
                unmuted_stream_msgs.add(message_id)
        elif msg_type == Recipient.PERSONAL:
            if sender_id == user_profile.id:
                other_user_id = row['message__recipient__type_id']
            else:
                other_user_id = sender_id
            pm_dict[message_id] = dict(other_user_id=other_user_id)
        elif msg_type == Recipient.DIRECT_MESSAGE_GROUP:
            user_ids_string = get_direct_message_group_users(recipient_id)
            direct_message_group_dict[message_id] = dict(user_ids_string=user_ids_string)
        is_mentioned = row['flags'] & UserMessage.flags.mentioned != 0
        is_stream_wildcard_mentioned = row['flags'] & UserMessage.flags.stream_wildcard_mentioned != 0
        is_topic_wildcard_mentioned = row['flags'] & UserMessage.flags.topic_wildcard_mentioned != 0
        if is_mentioned:
            mentions.add(message_id)
        if is_stream_wildcard_mentioned or is_topic_wildcard_mentioned:
            if msg_type == Recipient.STREAM:
                stream_id = row['message__recipient__type_id']
                topic_name = row[MESSAGE__TOPIC]
                if not is_row_muted(stream_id, recipient_id, topic_name):
                    mentions.add(message_id)
            else:
                mentions.add(message_id)
    raw_unread_messages['old_unreads_missing'] = total_unreads == MAX_UNREAD_MESSAGES
    return raw_unread_messages

def aggregate_streams(*, input_dict: Mapping[int, Any], allow_empty_topic_name: bool) -> Sequence[UnreadStreamInfo]:
    lookup_dict = {}
    for message_id, attribute_dict in input_dict.items():
        stream_id = attribute_dict['stream_id']
        topic_name = attribute_dict['topic']
        if topic_name == '' and (not allow_empty_topic_name):
            topic_name = Message.EMPTY_TOPIC_FALLBACK_NAME
        lookup_key = (stream_id, topic_name.lower())
        if lookup_key not in lookup_dict:
            obj = UnreadStreamInfo(stream_id=stream_id, topic=topic_name, unread_message_ids=[])
            lookup_dict[lookup_key] = obj
        bucket = lookup_dict[lookup_key]
        bucket['unread_message_ids'].append(message_id)
    for dct in lookup_dict.values():
        dct['unread_message_ids'].sort()
    sorted_keys = sorted(lookup_dict.keys())
    return [lookup_dict[k] for k in sorted_keys]

def aggregate_pms(*, input_dict: Mapping[int, Any]) -> Sequence[UnreadDirectMessageInfo]:
    lookup_dict = {}
    for message_id, attribute_dict in input_dict.items():
        other_user_id = attribute_dict['other_user_id']
        if other_user_id not in lookup_dict:
            obj = UnreadDirectMessageInfo(other_user_id=other_user_id, sender_id=other_user_id, unread_message_ids=[])
            lookup_dict[other_user_id] = obj
        bucket = lookup_dict[other_user_id]
        bucket['unread_message_ids'].append(message_id)
    for dct in lookup_dict.values():
        dct['unread_message_ids'].sort()
    sorted_keys = sorted(lookup_dict.keys())
    return [lookup_dict[k] for k in sorted_keys]

def aggregate_direct_message_groups(*, input_dict: Mapping[int, Any]) -> Sequence[UnreadDirectMessageGroupInfo]:
    lookup_dict = {}
    for message_id, attribute_dict in input_dict.items():
        user_ids_string = attribute_dict['user_ids_string']
        if user_ids_string not in lookup_dict:
            obj = UnreadDirectMessageGroupInfo(user_ids_string=user_ids_string, unread_message_ids=[])
            lookup_dict[user_ids_string] = obj
        bucket = lookup_dict[user_ids_string]
        bucket['unread_message_ids'].append(message_id)
    for dct in lookup_dict.values():
        dct['unread_message_ids'].sort()
    sorted_keys = sorted(lookup_dict.keys())
    return [lookup_dict[k] for k in sorted_keys]

def aggregate_unread_data(raw_data: RawUnreadMessagesResult, allow_empty_topic_name: bool) -> UnreadMessagesResult:
    pm_dict = raw_data['pm_dict']
    stream_dict = raw_data['stream_dict']
    unmuted_stream_msgs = raw_data['unmuted_stream_msgs']
    direct_message_group_dict = raw_data['huddle_dict']
    mentions = list(raw_data['mentions'])
    count = len(pm_dict) + len(unmuted_stream_msgs) + len(direct_message_group_dict)
    pm_objects = aggregate_pms(input_dict=pm_dict)
    stream_objects = aggregate_streams(input_dict=stream_dict, allow_empty_topic_name=allow_empty_topic_name)
    direct_message_groups = aggregate_direct_message_groups(input_dict=direct_message_group_dict)
    result = dict(pms=pm_objects, streams=stream_objects, huddles=direct_message_groups, mentions=mentions, count=count, old_unreads_missing=raw_data['old_unreads_missing'])
    return result

def apply_unread_message_event(user_profile: UserProfile, state: RawUnreadMessagesResult, message: Message, flags: Any) -> None:
    message_id = message['id']
    if message['type'] == 'stream':
        recipient_type = 'stream'
    elif message['type'] == 'private':
        others = [recip for recip in message['display_recipient'] if recip['id'] != user_profile.id]
        if len(others) <= 1:
            recipient_type = 'private'
        else:
            recipient_type = 'huddle'
    else:
        raise AssertionError('Invalid message type {}'.format(message['type']))
    if recipient_type == 'stream':
        stream_id = message['stream_id']
        topic_name = message[TOPIC_NAME]
        state['stream_dict'][message_id] = RawUnreadStreamDict(stream_id=stream_id, topic=topic_name)
        stream_muted = stream_id in state['muted_stream_ids']
        visibility_policy = get_topic_visibility_policy(user_profile, stream_id, topic_name=maybe_rename_general_chat_to_empty_topic(topic_name))
        if not stream_muted and visibility_policy != UserTopic.VisibilityPolicy.MUTED or (stream_muted and visibility_policy in [UserTopic.VisibilityPolicy.UNMUTED, UserTopic.VisibilityPolicy.FOLLOWED]):
            state['unmuted_stream_msgs'].add(message_id)
    elif recipient_type == 'private':
        if len(others) == 1:
            other_user_id = others[0]['id']
        else:
            other_user_id = user_profile.id
        state['pm_dict'][message_id] = RawUnreadDirectMessageDict(other_user_id=other_user_id)
    else:
        display_recipient = message['display_recipient']
        user_ids = [obj['id'] for obj in display_recipient]
        user_ids = sorted(user_ids)
        user_ids_string = ','.join((str(uid) for uid in user_ids))
        state['huddle_dict'][message_id] = RawUnreadDirectMessageGroupDict(user_ids_string=user_ids_string)
    if 'mentioned' in flags:
        state['mentions'].add(message_id)
    if ('stream_wildcard_mentioned' in flags or 'topic_wildcard_mentioned' in flags) and message_id in state['unmuted_stream_msgs']:
        state['mentions'].add(message_id)

def remove_message_id_from_unread_mgs(state: RawUnreadMessagesResult, message_id: int) -> None:
    state['pm_dict'].pop(message_id, None)
    state['stream_dict'].pop(message_id, None)
    state['huddle_dict'].pop(message_id, None)
    state['unmuted_stream_msgs'].discard(message_id)
    state['mentions'].discard(message_id)

def format_unread_message_details(my_user_id: int, raw_unread_data: RawUnreadMessagesResult) -> dict[str, MessageDetailsDict]:
    unread_data = {}
    for message_id, private_message_details in raw_unread_data['pm_dict'].items():
        other_user_id = private_message_details['other_user_id']
        if other_user_id == my_user_id:
            user_ids = []
        else:
            user_ids = [other_user_id]
        message_details = MessageDetailsDict(type='private', user_ids=user_ids)
        if message_id in raw_unread_data['mentions']:
            message_details['mentioned'] = True
        unread_data[str(message_id)] = message_details
    for message_id, stream_message_details in raw_unread_data['stream_dict'].items():
        unmuted_stream_msg = message_id in raw_unread_data['unmuted_stream_msgs']
        message_details = MessageDetailsDict(type='stream', stream_id=stream_message_details['stream_id'], topic=stream_message_details['topic'], unmuted_stream_msg=unmuted_stream_msg)
        if message_id in raw_unread_data['mentions']:
            message_details['mentioned'] = True
        unread_data[str(message_id)] = message_details
    for message_id, huddle_message_details in raw_unread_data['huddle_dict'].items():
        user_ids = sorted((user_id for s in huddle_message_details['user_ids_string'].split(',') if (user_id := int(s)) != my_user_id))
        message_details = MessageDetailsDict(type='private', user_ids=user_ids)
        if message_id in raw_unread_data['mentions']:
            message_details['mentioned'] = True
        unread_data[str(message_id)] = message_details
    return unread_data

def add_message_to_unread_msgs(my_user_id: int, state: RawUnreadMessagesResult, message_id: int, message_details: MessageDetailsDict) -> None:
    if message_details.get('mentioned'):
        state['mentions'].add(message_id)
    if message_details['type'] == 'private':
        user_ids = message_details['user_ids']
        user_ids = [user_id for user_id in user_ids if user_id != my_user_id]
        if user_ids == []:
            state['pm_dict'][message_id] = RawUnreadDirectMessageDict(other_user_id=my_user_id)
        elif len(user_ids) == 1:
            state['pm_dict'][message_id] = RawUnreadDirectMessageDict(other_user_id=user_ids[0])
        else:
            user_ids.append(my_user_id)
            user_ids_string = ','.join((str(user_id) for user_id in sorted(user_ids)))
            state['huddle_dict'][message_id] = RawUnreadDirectMessageGroupDict(user_ids_string=user_ids_string)
    elif message_details['type'] == 'stream':
        state['stream_dict'][message_id] = RawUnreadStreamDict(stream_id=message_details['stream_id'], topic=message_details['topic'])
        if message_details['unmuted_stream_msg']:
            state['unmuted_stream_msgs'].add(message_id)

def estimate_recent_messages(realm: Realm, hours: int) -> int:
    stat = COUNT_STATS['messages_sent:is_bot:hour']
    d = timezone_now() - timedelta(hours=hours)
    return RealmCount.objects.filter(property=stat.property, end_time__gt=d, realm=realm).aggregate(Sum('value'))['value__sum'] or 0

def get_first_visible_message_id(realm: Realm) -> int:
    return realm.first_visible_message_id

def maybe_update_first_visible_message_id(realm: Realm, lookback_hours: int) -> None:
    recent_messages_count = estimate_recent_messages(realm, lookback_hours)
    if realm.message_visibility_limit is not None and recent_messages_count > 0:
        update_first_visible_message_id(realm)

def update_first_visible_message_id(realm: Realm) -> None:
    if realm.message_visibility_limit is None:
        realm.first_visible_message_id = 0
    else:
        try:
            first_visible_message_id = Message.objects.filter(realm=realm).values('id').order_by('-id')[realm.message_visibility_limit - 1]['id']
        except IndexError:
            first_visible_message_id = 0
        realm.first_visible_message_id = first_visible_message_id
    realm.save(update_fields=['first_visible_message_id'])

def get_last_message_id() -> int:
    last_id = Message.objects.aggregate(Max('id'))['id__max']
    if last_id is None:
        last_id = -1
    return last_id

def get_recent_conversations_recipient_id(user_profile: UserProfile, recipient_id: int, sender_id: int) -> int:
    """Helper for doing lookups of the recipient_id that
    get_recent_private_conversations would have used to record that
    message in its data structure.
    """
    my_recipient_id = user_profile.recipient_id
    if recipient_id == my_recipient_id:
        return UserProfile.objects.values_list('recipient_id', flat=True).get(id=sender_id)
    return recipient_id

def get_recent_private_conversations(user_profile: UserProfile) -> dict[int, dict[str, Any]]:
    """This function uses some carefully optimized SQL queries, designed
    to use the UserMessage index on private_messages.  It is
    somewhat complicated by the fact that for 1:1 direct
    messages, we store the message against a recipient_id of whichever
    user was the recipient, and thus for 1:1 direct messages sent
    directly to us, we need to look up the other user from the
    sender_id on those messages.  You'll see that pattern repeated
    both here and also in zerver/lib/events.py.

    It may be possible to write this query directly in Django, however
    it is made much easier by using CTEs, which Django does not
    natively support.

    We return a dictionary structure for convenient modification
    below; this structure is converted into its final form by
    post_process.

    """
    RECENT_CONVERSATIONS_LIMIT = 1000
    recipient_map = {}
    my_recipient_id = user_profile.recipient_id
    query = SQL('\n        WITH personals AS (\n            SELECT   um.message_id AS message_id\n            FROM     zerver_usermessage um\n            WHERE    um.user_profile_id = %(user_profile_id)s\n            AND      um.flags & 2048 <> 0\n            ORDER BY message_id DESC limit %(conversation_limit)s\n        ),\n        message AS (\n            SELECT message_id,\n                   CASE\n                          WHEN m.recipient_id = %(my_recipient_id)s\n                          THEN m.sender_id\n                          ELSE NULL\n                   END AS sender_id,\n                   CASE\n                          WHEN m.recipient_id <> %(my_recipient_id)s\n                          THEN m.recipient_id\n                          ELSE NULL\n                   END AS outgoing_recipient_id\n            FROM   personals\n            JOIN   zerver_message m\n            ON     personals.message_id = m.id\n        ),\n        unified AS (\n            SELECT    message_id,\n                      COALESCE(zerver_userprofile.recipient_id, outgoing_recipient_id) AS other_recipient_id\n            FROM      message\n            LEFT JOIN zerver_userprofile\n            ON        zerver_userprofile.id = sender_id\n        )\n        SELECT   other_recipient_id,\n                 MAX(message_id)\n        FROM     unified\n        GROUP BY other_recipient_id\n    ')
    with connection.cursor() as cursor:
        cursor.execute(query, {'user_profile_id': user_profile.id, 'conversation_limit': RECENT_CONVERSATIONS_LIMIT, 'my_recipient_id': my_recipient_id})
        rows = cursor.fetchall()
    for recipient_id, max_message_id in rows:
        recipient_map[recipient_id] = dict(max_message_id=max_message_id, user_ids=[])
    for recipient_id, user_profile_id in Subscription.objects.filter(recipient_id__in=recipient_map.keys()).exclude(user_profile_id=user_profile.id).values_list('recipient_id', 'user_profile_id'):
        recipient_map[recipient_id]['user_ids'].append(user_profile_id)
    for rec in recipient_map.values():
        rec['user_ids'].sort()
    return recipient_map

def can_mention_many_users(sender: UserProfile) -> bool:
    """Helper function for 'topic_wildcard_mention_allowed' and
    'stream_wildcard_mention_allowed' to check if the sender is allowed to use
    wildcard mentions based on the 'can_mention_many_users_group' setting of that realm.
    This check is used only if the participants count in the topic or the subscribers
    count in the stream is greater than 'Realm.WILDCARD_MENTION_THRESHOLD'.
    """
    return sender.has_permission('can_mention_many_users_group')

def topic_wildcard_mention_allowed(sender: UserProfile, topic_participant_count: int, realm: Realm) -> bool:
    if topic_participant_count <= Realm.WILDCARD_MENTION_THRESHOLD:
        return True
    return can_mention_many_users(sender)

def stream_wildcard_mention_allowed(sender: UserProfile, stream: Stream, realm: Realm) -> bool:
    if num_subscribers_for_stream_id(stream.id) <= Realm.WILDCARD_MENTION_THRESHOLD:
        return True
    return can_mention_many_users(sender)

def check_user_group_mention_allowed(sender: UserProfile, user_group_ids: Sequence[int]) -> None:
    user_groups = NamedUserGroup.objects.filter(id__in=user_group_ids).select_related('can_mention_group', 'can_mention_group__named_user_group')
    sender_is_system_bot = is_cross_realm_bot_email(sender.delivery_email)
    for group in user_groups:
        can_mention_group = group.can_mention_group
        if hasattr(can_mention_group, 'named_user_group') and can_mention_group.named_user_group.name == SystemGroups.EVERYONE:
            continue
        if sender_is_system_bot:
            raise JsonableError(_("You are not allowed to mention user group '{user_group_name}'.").format(user_group_name=group.name))
        if not user_has_permission_for_group_setting(can_mention_group, sender, NamedUserGroup.GROUP_PERMISSION_SETTINGS['can_mention_group'], direct_member_only=False):
            raise JsonableError(_("You are not allowed to mention user group '{user_group_name}'.").format(user_group_name=group.name))

def parse_message_time_limit_setting(value: Any, special_values_map: Mapping[str, int], *, setting_name: str) -> int:
    if isinstance(value, str) and value in special_values_map:
        return special_values_map[value]
    if isinstance(value, str) or value <= 0:
        raise RequestVariableConversionError(setting_name, value)
    assert isinstance(value, int)
    return value

def visibility_policy_for_participation(sender: UserProfile, is_stream_muted: bool) -> UserTopic.VisibilityPolicy:
    """
    This function determines the visibility policy to set when a user
    participates in a topic, depending on the 'automatically_follow_topics_policy'
    and 'automatically_unmute_topics_in_muted_streams_policy' settings.
    """
    if sender.automatically_follow_topics_policy == UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_PARTICIPATION:
        return UserTopic.VisibilityPolicy.FOLLOWED
    if is_stream_muted and sender.automatically_unmute_topics_in_muted_streams_policy == UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_PARTICIPATION:
        return UserTopic.VisibilityPolicy.UNMUTED
    return None

def visibility_policy_for_send(sender: UserProfile, is_stream_muted: bool) -> UserTopic.VisibilityPolicy:
    if sender.automatically_follow_topics_policy == UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_SEND:
        return UserTopic.VisibilityPolicy.FOLLOWED
    if is_stream_muted and sender.automatically_unmute_topics_in_muted_streams_policy == UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_SEND:
        return UserTopic.VisibilityPolicy.UNMUTED
    return None

def visibility_policy_for_send_message(sender: UserProfile, message: Message, stream: Stream, is_stream_muted: bool, current_visibility_policy: UserTopic.VisibilityPolicy) -> UserTopic.VisibilityPolicy:
    """
    This function determines the visibility policy to set when a message
    is sent to a topic, depending on the 'automatically_follow_topics_policy'
    and 'automatically_unmute_topics_in_muted_streams_policy' settings.

    It returns None when the policies can't make it more visible than the
    current visibility policy.
    """
    visibility_policy = None
    if current_visibility_policy == UserTopic.VisibilityPolicy.FOLLOWED:
        return visibility_policy
    visibility_policy_participation = visibility_policy_for_participation(sender, is_stream_muted)
    visibility_policy_send = visibility_policy_for_send(sender, is_stream_muted)
    if UserTopic.VisibilityPolicy.FOLLOWED in (visibility_policy_participation, visibility_policy_send):
        return UserTopic.VisibilityPolicy.FOLLOWED
    if UserTopic.VisibilityPolicy.UNMUTED in (visibility_policy_participation, visibility_policy_send):
        visibility_policy = UserTopic.VisibilityPolicy.UNMUTED
    if current_visibility_policy != UserTopic.VisibilityPolicy.INHERIT:
        if visibility_policy and current_visibility_policy == visibility_policy:
            return None
        return visibility_policy
    if can_access_stream_history(sender, stream):
        old_accessible_messages_in_topic = messages_for_topic(realm_id=sender.realm_id, stream_recipient_id=message.recipient_id, topic_name=message.topic_name()).exclude(id=message.id)
    else:
        old_accessible_messages_in_topic = UserMessage.objects.filter(user_profile=sender, message__recipient_id=message.recipient_id, message__subject__iexact=message.topic_name()).exclude(message_id=message.id)
    if sender.automatically_follow_topics_policy == UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION and (not old_accessible_messages_in_topic.exists()):
        return UserTopic.VisibilityPolicy.FOLLOWED
    if is_stream_muted and sender.automatically_unmute_topics_in_muted_streams_policy == UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_ON_INITIATION and (not old_accessible_messages_in_topic.exists()):
        visibility_policy = UserTopic.VisibilityPolicy.UNMUTED
    return visibility_policy

def should_change_visibility_policy(new_visibility_policy: UserTopic.VisibilityPolicy, sender: UserProfile, stream_id: int, topic_name: str) -> bool:
    try:
        user_topic = UserTopic.objects.get(user_profile=sender, stream_id=stream_id, topic_name__iexact=topic_name)
    except UserTopic.DoesNotExist:
        return True
    current_visibility_policy = user_topic.visibility_policy
    if new_visibility_policy == current_visibility_policy:
        return False
    if current_visibility_policy == UserTopic.VisibilityPolicy.FOLLOWED:
        return False
    return True

def set_visibility_policy_possible(user_profile: UserProfile, message: Message) -> bool:
    """If the user can set a visibility policy."""
    if not message.is_stream_message():
        return False
    if user_profile.is_bot:
        return False
    if user_profile.realm != message.get_realm():
        return False
    return True

def remove_single_newlines(content: str) -> str:
    content = content.strip('\n')
    return re.sub('(?<!\\n)\\n(?!\\n|[-*] |[0-9]+\\. )', ' ', content)

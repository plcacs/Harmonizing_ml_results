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
from zerver.lib.stream_subscription import (
    get_active_subscriptions_for_stream_id,
    get_stream_subscriptions_for_user,
    get_subscribed_stream_recipient_ids_for_user,
    num_subscribers_for_stream_id,
)
from zerver.lib.streams import can_access_stream_history, get_web_public_streams_queryset
from zerver.lib.topic import (
    MESSAGE__TOPIC,
    TOPIC_NAME,
    maybe_rename_general_chat_to_empty_topic,
    messages_for_topic,
)
from zerver.lib.types import UserDisplayRecipient
from zerver.lib.user_groups import user_has_permission_for_group_setting
from zerver.lib.user_topics import build_get_topic_visibility_policy, get_topic_visibility_policy
from zerver.lib.users import get_inaccessible_user_ids
from zerver.models import (
    Message,
    NamedUserGroup,
    Realm,
    Recipient,
    Stream,
    Subscription,
    UserMessage,
    UserProfile,
    UserTopic,
)
from zerver.models.constants import MAX_TOPIC_NAME_LENGTH
from zerver.models.groups import SystemGroups
from zerver.models.messages import get_usermessage_by_message_id
from zerver.models.users import is_cross_realm_bot_email


class MessageDetailsDict(TypedDict, total=False):
    type: str
    mentioned: bool
    user_ids: list[int]
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
    pm_dict: dict[int, RawUnreadDirectMessageDict]
    stream_dict: dict[int, RawUnreadStreamDict]
    huddle_dict: dict[int, RawUnreadDirectMessageGroupDict]
    mentions: set[int]
    muted_stream_ids: set[int]
    unmuted_stream_msgs: set[int]
    old_unreads_missing: bool


class UnreadStreamInfo(TypedDict):
    stream_id: int
    topic: str
    unread_message_ids: list[int]


class UnreadDirectMessageInfo(TypedDict):
    other_user_id: int
    # Deprecated and misleading synonym for other_user_id
    sender_id: int
    unread_message_ids: list[int]


class UnreadDirectMessageGroupInfo(TypedDict):
    user_ids_string: str
    unread_message_ids: list[int]


class UnreadMessagesResult(TypedDict):
    pms: list[UnreadDirectMessageInfo]
    streams: list[UnreadStreamInfo]
    huddles: list[UnreadDirectMessageGroupInfo]
    mentions: list[int]
    count: int
    old_unreads_missing: bool


@dataclass
class SendMessageRequest:
    message: Message
    rendering_result: MessageRenderingResult
    stream: Stream | None
    sender_muted_stream: bool | None
    local_id: str | None
    sender_queue_id: str | None
    realm: Realm
    mention_data: MentionData
    mentioned_user_groups_map: dict[int, int]
    active_user_ids: set[int]
    online_push_user_ids: set[int]
    dm_mention_push_disabled_user_ids: set[int]
    dm_mention_email_disabled_user_ids: set[int]
    stream_push_user_ids: set[int]
    stream_email_user_ids: set[int]
    # IDs of users who have followed the topic the message is being sent to,
    # and have the followed topic push notifications setting ON.
    followed_topic_push_user_ids: set[int]
    # IDs of users who have followed the topic the message is being sent to,
    # and have the followed topic email notifications setting ON.
    followed_topic_email_user_ids: set[int]
    muted_sender_user_ids: set[int]
    um_eligible_user_ids: set[int]
    long_term_idle_user_ids: set[int]
    default_bot_user_ids: set[int]
    service_bot_tuples: list[tuple[int, int]]
    all_bot_user_ids: set[int]
    # IDs of topic participants who should be notified of topic wildcard mention.
    # The 'user_allows_notifications_in_StreamTopic' with 'wildcard_mentions_notify'
    # setting ON should return True.
    # A user_id can exist in either or both of the 'topic_wildcard_mention_user_ids'
    # and 'topic_wildcard_mention_in_followed_topic_user_ids' sets.
    topic_wildcard_mention_user_ids: set[int]
    # IDs of users subscribed to the stream who should be notified of
    # stream wildcard mention.
    # The 'user_allows_notifications_in_StreamTopic' with 'wildcard_mentions_notify'
    # setting ON should return True.
    # A user_id can exist in either or both of the 'stream_wildcard_mention_user_ids'
    # and 'stream_wildcard_mention_in_followed_topic_user_ids' sets.
    stream_wildcard_mention_user_ids: set[int]
    # IDs of topic participants who have followed the topic the message
    # (having topic wildcard) is being sent to, and have the
    # 'followed_topic_wildcard_mentions_notify' setting ON.
    topic_wildcard_mention_in_followed_topic_user_ids: set[int]
    # IDs of users who have followed the topic the message
    # (having stream wildcard) is being sent to, and have the
    # 'followed_topic_wildcard_mentions_notify' setting ON.
    stream_wildcard_mention_in_followed_topic_user_ids: set[int]
    # A topic participant is anyone who either sent or reacted to messages in the topic.
    topic_participant_user_ids: set[int]
    links_for_embed: set[str]
    widget_content: dict[str, Any] | None
    submessages: list[dict[str, Any]] = field(default_factory=list)
    deliver_at: datetime | None = None
    delivery_type: str | None = None
    limit_unread_user_ids: set[int] | None = None
    service_queue_events: dict[str, list[dict[str, Any]]] | None = None
    disable_external_notifications: bool = False
    automatic_new_visibility_policy: int | None = None
    recipients_for_user_creation_events: dict[UserProfile, set[int]] | None = None


# We won't try to fetch more unread message IDs from the database than
# this limit.  The limit is super high, in large part because it means
# client-side code mostly doesn't need to think about the case that a
# user has more older unread messages that were cut off.
MAX_UNREAD_MESSAGES = 50000


def truncate_content(content: str, max_length: int, truncation_message: str) -> str:
    if len(content) > max_length:
        content = content[: max_length - len(truncation_message)] + truncation_message
    return content


def normalize_body(body: str) -> str:
    body = body.rstrip().lstrip("\n")
    if len(body) == 0:
        raise JsonableError(_("Message must not be empty"))
    if "\x00" in body:
        raise JsonableError(_("Message must not contain null bytes"))
    return truncate_content(body, settings.MAX_MESSAGE_LENGTH, "\n[message truncated]")


def normalize_body_for_import(body: str) -> str:
    if "\x00" in body:
        body = re.sub(r"\x00", "", body)
    return truncate_content(body, settings.MAX_MESSAGE_LENGTH, "\n[message truncated]")


def truncate_topic(topic_name: str) -> str:
    return truncate_content(topic_name, MAX_TOPIC_NAME_LENGTH, "...")


def messages_for_ids(
    message_ids: list[int],
    user_message_flags: dict[int, list[str]],
    search_fields: dict[int, dict[str, str]],
    apply_markdown: bool,
    client_gravatar: bool,
    allow_empty_topic_name: bool,
    allow_edit_history: bool,
    user_profile: UserProfile | None,
    realm: Realm,
) -> list[dict[str, Any]]:
    id_fetcher = lambda row: row["id"]

    message_dicts = generic_bulk_cached_fetch(
        to_dict_cache_key_id,
        MessageDict.ids_to_dict,
        message_ids,
        id_fetcher=id_fetcher,
        cache_transformer=lambda obj: obj,
        extractor=extract_message_dict,
        setter=stringify_message_dict,
    )

    message_list: list[dict[str, Any]] = []

    sender_ids = [message_dicts[message_id]["sender_id"] for message_id in message_ids]
    inaccessible_sender_ids = get_inaccessible_user_ids(sender_ids, user_profile)

    for message_id in message_ids:
        msg_dict = message_dicts[message_id]
        flags = user_message_flags[message_id]
        # TODO/compatibility: The `wildcard_mentioned` flag was deprecated in favor of
        # the `stream_wildcard_mentioned` and `topic_wildcard_mentioned` flags.  The
        # `wildcard_mentioned` flag exists for backwards-compatibility with older
        # clients.  Remove this when we no longer support legacy clients that have not
        # been updated to access `stream_wildcard_mentioned`.
        if "stream_wildcard_mentioned" in flags or "topic_wildcard_mentioned" in flags:
            flags.append("wildcard_mentioned")
        msg_dict.update(flags=flags)
        if message_id in search_fields:
            msg_dict.update(search_fields[message_id])
        # Make sure that we never send message edit history to clients
        # in realms with allow_edit_history disabled.
        if "edit_history" in msg_dict and not allow_edit_history:
            del msg_dict["edit_history"]
        msg_dict["can_access_sender"] = msg_dict["sender_id"] not in inaccessible_sender_ids
        message_list.append(msg_dict)

    MessageDict.post_process_dicts(
        message_list,
        apply_markdown=apply_markdown,
        client_gravatar=client_gravatar,
        allow_empty_topic_name=allow_empty_topic_name,
        realm=realm,
        user_recipient_id=None if user_profile is None else user_profile.recipient_id,
    )

    return message_list


def access_message(
    user_profile: UserProfile,
    message_id: int,
    lock_message: bool = False,
) -> Message:
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
            # We want to lock only the `Message` row, and not the related fields
            # because the `Message` row only has a possibility of races.
            base_query = base_query.select_for_update(of=("self",))
        message = base_query.get(id=message_id)
    except Message.DoesNotExist:
        raise JsonableError(_("Invalid message(s)"))

    has_user_message = lambda: UserMessage.objects.filter(
        user_profile=user_profile, message_id=message_id
    ).exists()

    if has_message_access(user_profile, message, has_user_message=has_user_message):
        return message
    raise JsonableError(_("Invalid message(s)"))


def access_message_and_usermessage(
    user_profile: UserProfile,
    message_id: int,
    lock_message: bool = False,
) -> tuple[Message, UserMessage | None]:
    """As access_message, but also returns the usermessage, if any."""
    try:
        base_query = Message.objects.select_related(*Message.DEFAULT_SELECT_RELATED)
        if lock_message:
            # We want to lock only the `Message` row, and not the related fields
            # because the `Message` row only has a possibility of races.
            base_query = base_query.select_for_update(of=("self",))
        message = base_query.get(id=message_id)
    except Message.DoesNotExist:
        raise JsonableError(_("Invalid message(s)"))

    user_message = get_usermessage_by_message_id(user_profile, message_id)
    has_user_message = lambda: user_message is not None

    if has_message_access(user_profile, message, has_user_message=has_user_message):
        return (message, user_message)
    raise JsonableError(_("Invalid message(s)"))


def access_web_public_message(
    realm: Realm,
    message_id: int,
) -> Message:
    """Access control method for unauthenticated requests interacting
    with a message in web-public streams.
    """

    # We throw a MissingAuthenticationError for all errors in this
    # code path, to avoid potentially leaking information on whether a
    # message with the provided ID exists on the server if the client
    # shouldn't have access to it.
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

    # These should all have been enforced by the code in
    # get_web_public_streams_queryset
    assert stream.is_web_public
    assert not stream.deactivated
    assert not stream.invite_only
    assert stream.history_public_to_subscribers

    # Now that we've confirmed this message was sent to the target
    # web-public stream, we can return it as having been successfully
    # accessed.
    return message


def has_message_access(
    user_profile: UserProfile,
    message: Message,
    *,
    has_user_message: Callable[[], bool],
    stream: Stream | None = None,
    is_subscribed: bool | None = None,
) -> bool:
    """
    Returns whether a user has access to a given message.

    * The user_message parameter must be provided if the user has a UserMessage
      row for the target message.
    * The optional stream parameter is validated; is_subscribed is not.
    """

    if message.recipient.type != Recipient.STREAM:
        # You can only access direct messages you received
        return has_user_message()

    if stream is None:
        stream = Stream.objects.get(id=message.recipient.type_id)
    else:
        assert stream.recipient_id == message.recipient_id

    if stream.realm_id != user_profile.realm_id:
        # You can't access public stream messages in other realms
        return False

    if stream.deactivated:
        # You can't access messages in deactivated streams
        return False

    def is_subscribed_helper() -> bool:
        if is_subscribed is not None:
            return is_subscribed

        return Subscription.objects.filter(
            user_profile=user_profile, active=True, recipient=message.recipient
        ).exists()

    if stream.is_public() and user_profile.can_access_public_streams():
        return True

    if not stream.is_history_public_to_subscribers():
        # Unless history is public to subscribers, you need to both:
        # (1) Have directly received the message.
        # AND
        # (2) Be subscribed to the stream.
        return has_user_message() and is_subscribed_helper()

    # is_history_public_to_subscribers, so check if you're subscribed
    return is_subscribed_helper()


def event_recipient_ids_for_action_on_messages(
    messages: Collection[Message],
    *,
    channel: Stream | None = None,
    exclude_long_term_idle_users: bool = True,
) -> set[int]:
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

    def get_user
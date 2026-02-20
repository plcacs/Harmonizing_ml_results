import logging
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import timedelta
from email.headerregistry import Address
from typing import Any, TypedDict, Optional, Union

import orjson
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.db.models import F, Q, QuerySet
from django.utils.html import escape
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import override as override_language

from zerver.actions.uploads import do_claim_attachments
from zerver.actions.user_topics import (
    bulk_do_set_user_topic_visibility_policy,
    do_set_user_topic_visibility_policy,
)
from zerver.lib.addressee import Addressee
from zerver.lib.alert_words import get_alert_word_automaton
from zerver.lib.cache import cache_with_key, user_profile_delivery_email_cache_key
from zerver.lib.create_user import create_user
from zerver.lib.exceptions import (
    DirectMessageInitiationError,
    DirectMessagePermissionError,
    JsonableError,
    MarkdownRenderingError,
    StreamDoesNotExistError,
    StreamWildcardMentionNotAllowedError,
    StreamWithIDDoesNotExistError,
    TopicWildcardMentionNotAllowedError,
    ZephyrMessageAlreadySentError,
)
from zerver.lib.markdown import MessageRenderingResult, render_message_markdown
from zerver.lib.markdown import version as markdown_version
from zerver.lib.mention import MentionBackend, MentionData
from zerver.lib.message import (
    SendMessageRequest,
    check_user_group_mention_allowed,
    normalize_body,
    set_visibility_policy_possible,
    stream_wildcard_mention_allowed,
    topic_wildcard_mention_allowed,
    truncate_topic,
    visibility_policy_for_send_message,
)
from zerver.lib.message_cache import MessageDict
from zerver.lib.muted_users import get_muting_users
from zerver.lib.notification_data import (
    UserMessageNotificationsData,
    get_user_group_mentions_data,
    user_allows_notifications_in_StreamTopic,
)
from zerver.lib.query_helpers import query_for_ids
from zerver.lib.queue import queue_event_on_commit
from zerver.lib.recipient_users import recipient_for_user_profiles
from zerver.lib.stream_subscription import (
    get_subscriptions_for_send_message,
    num_subscribers_for_stream_id,
)
from zerver.lib.stream_topic import StreamTopicTarget
from zerver.lib.streams import (
    access_stream_for_send_message,
    ensure_stream,
    notify_stream_is_recently_active_update,
    subscribed_to_stream,
)
from zerver.lib.string_validation import check_stream_name
from zerver.lib.thumbnail import get_user_upload_previews, rewrite_thumbnailed_images
from zerver.lib.timestamp import timestamp_to_datetime
from zerver.lib.topic import participants_for_topic
from zerver.lib.url_preview.types import UrlEmbedData
from zerver.lib.user_groups import is_any_user_in_group, is_user_in_group
from zerver.lib.user_message import UserMessageLite, bulk_insert_ums
from zerver.lib.users import (
    check_can_access_user,
    get_inaccessible_user_ids,
    get_subscribers_of_target_user_subscriptions,
    get_user_ids_who_can_access_user,
    get_users_involved_in_dms_with_target_users,
    user_access_restricted_in_realm,
)
from zerver.lib.validator import check_widget_content
from zerver.lib.widget import do_widget_post_save_actions
from zerver.models import (
    Client,
    Message,
    Realm,
    Recipient,
    Stream,
    UserMessage,
    UserPresence,
    UserProfile,
    UserTopic,
)
from zerver.models.clients import get_client
from zerver.models.groups import SystemGroups
from zerver.models.recipients import get_direct_message_group_user_ids
from zerver.models.scheduled_jobs import NotificationTriggers
from zerver.models.streams import (
    get_stream_by_id_for_sending_message,
    get_stream_by_name_for_sending_message,
)
from zerver.models.users import get_system_bot, get_user_by_delivery_email, is_cross_realm_bot_email
from zerver.tornado.django_api import send_event_on_commit


def compute_irc_user_fullname(email: str) -> str:
    return Address(addr_spec=email).username + " (IRC)"


def compute_jabber_user_fullname(email: str) -> str:
    return Address(addr_spec=email).username + " (XMPP)"


def get_user_profile_delivery_email_cache_key(
    realm: Realm, email: str, email_to_fullname: Callable[[str], str]
) -> str:
    return user_profile_delivery_email_cache_key(email, realm.id)


@cache_with_key(
    get_user_profile_delivery_email_cache_key,
    timeout=3600 * 24 * 7,
)
def create_mirror_user_if_needed(
    realm: Realm, email: str, email_to_fullname: Callable[[str], str]
) -> UserProfile:
    try:
        return get_user_by_delivery_email(email, realm)
    except UserProfile.DoesNotExist:
        try:
            # Forge a user for this person
            return create_user(
                email=email,
                password=None,
                realm=realm,
                full_name=email_to_fullname(email),
                active=False,
                is_mirror_dummy=True,
            )
        except IntegrityError:
            return get_user_by_delivery_email(email, realm)


def render_incoming_message(
    message: Message,
    content: str,
    realm: Realm,
    mention_data: Optional[MentionData] = None,
    url_embed_data: Optional[dict[str, UrlEmbedData]] = None,
    email_gateway: bool = False,
    acting_user: Optional[UserProfile] = None,
) -> MessageRenderingResult:
    realm_alert_words_automaton = get_alert_word_automaton(realm)
    try:
        rendering_result = render_message_markdown(
            message=message,
            content=content,
            realm=realm,
            realm_alert_words_automaton=realm_alert_words_automaton,
            mention_data=mention_data,
            url_embed_data=url_embed_data,
            email_gateway=email_gateway,
            acting_user=acting_user,
        )
    except MarkdownRenderingError:
        raise JsonableError(_("Unable to render message"))
    return rendering_result


@dataclass
class RecipientInfoResult:
    active_user_ids: set[int]
    online_push_user_ids: set[int]
    dm_mention_email_disabled_user_ids: set[int]
    dm_mention_push_disabled_user_ids: set[int]
    stream_email_user_ids: set[int]
    stream_push_user_ids: set[int]
    topic_wildcard_mention_user_ids: set[int]
    stream_wildcard_mention_user_ids: set[int]
    followed_topic_email_user_ids: set[int]
    followed_topic_push_user_ids: set[int]
    topic_wildcard_mention_in_followed_topic_user_ids: set[int]
    stream_wildcard_mention_in_followed_topic_user_ids: set[int]
    muted_sender_user_ids: set[int]
    um_eligible_user_ids: set[int]
    long_term_idle_user_ids: set[int]
    default_bot_user_ids: set[int]
    service_bot_tuples: list[tuple[int, int]]
    all_bot_user_ids: set[int]
    topic_participant_user_ids: set[int]
    sender_muted_stream: Optional[bool]


class ActiveUserDict(TypedDict):
    id: int
    enable_online_push_notifications: bool
    enable_offline_email_notifications: bool
    enable_offline_push_notifications: bool
    long_term_idle: bool
    is_bot: bool
    bot_type: Optional[int]


@dataclass
class SentMessageResult:
    message_id: int
    automatic_new_visibility_policy: Optional[int] = None


def get_recipient_info(
    *,
    realm_id: int,
    recipient: Recipient,
    sender_id: int,
    stream_topic: Optional[StreamTopicTarget],
    possibly_mentioned_user_ids: set[int] = set(),
    possible_topic_wildcard_mention: bool = True,
    possible_stream_wildcard_mention: bool = True,
) -> RecipientInfoResult:
    stream_push_user_ids: set[int] = set()
    stream_email_user_ids: set[int] = set()
    topic_wildcard_mention_user_ids: set[int] = set()
    stream_wildcard_mention_user_ids: set[int] = set()
    followed_topic_push_user_ids: set[int] = set()
    followed_topic_email_user_ids: set[int] = set()
    topic_wildcard_mention_in_followed_topic_user_ids: set[int] = set()
    stream_wildcard_mention_in_followed_topic_user_ids: set[int] = set()
    muted_sender_user_ids: set[int] = get_muting_users(sender_id)
    topic_participant_user_ids: set[int] = set()
    sender_muted_stream: Optional[bool] = None

    if recipient.type == Recipient.PERSONAL:
        # The sender and recipient may be the same id, so
        # de-duplicate using a set.
        message_to_user_id_set = {recipient.type_id, sender_id}
        assert len(message_to_user_id_set) in [1, 2]

    elif recipient.type == Recipient.STREAM:
        # Anybody calling us w/r/t a stream message needs to supply
        # stream_topic.  We may eventually want to have different versions
        # of this function for different message types.
        assert stream_topic is not None

        if possible_topic_wildcard_mention:
            # A topic participant is anyone who either sent or reacted to messages in the topic.
            # It is expensive to call `participants_for_topic` if the topic has a large number
            # of messages. But it is fine to call it here, as this gets called only if the message
            # has syntax that might be a @topic mention without having confirmed the syntax isn't, say,
            # in a code block.
            topic_participant_user_ids = participants_for_topic(
                realm_id, recipient.id, stream_topic.topic_name
            )
            # We explicitly include the sender as a topic participant because the message will
            # be actually sent at a later stage in this codepath, so `participants_for_topic`
            # misses this sender. This is useful when the sender is sending their first message
            # in the topic.
            topic_participant_user_ids.add(sender_id)
        subscription_rows = (
            get_subscriptions_for_send_message(
                realm_id=realm_id,
                stream_id=stream_topic.stream_id,
                topic_name=stream_topic.topic_name,
                possible_stream_wildcard_mention=possible_stream_wildcard_mention,
                topic_participant_user_ids=topic_participant_user_ids,
                possibly_mentioned_user_ids=possibly_mentioned_user_ids,
            )
            .annotate(
                user_profile_email_notifications=F(
                    "user_profile__enable_stream_email_notifications"
                ),
                user_profile_push_notifications=F("user_profile__enable_stream_push_notifications"),
                user_profile_wildcard_mentions_notify=F("user_profile__wildcard_mentions_notify"),
                followed_topic_email_notifications=F(
                    "user_profile__enable_followed_topic_email_notifications"
                ),
                followed_topic_push_notifications=F(
                    "user_profile__enable_followed_topic_push_notifications"
                ),
                followed_topic_wildcard_mentions_notify=F(
                    "user_profile__enable_followed_topic_wildcard_mentions_notify"
                ),
            )
            .values(
                "user_profile_id",
                "push_notifications",
                "email_notifications",
                "wildcard_mentions_notify",
                "followed_topic_push_notifications",
                "followed_topic_email_notifications",
                "followed_topic_wildcard_mentions_notify",
                "user_profile_email_notifications",
                "user_profile_push_notifications",
                "user_profile_wildcard_mentions_notify",
                "is_muted",
            )
            .order_by("user_profile_id")
        )

        message_to_user_id_set = set()
        for row in subscription_rows:
            message_to_user_id_set.add(row["user_profile_id"])
            # We store the 'sender_muted_stream' information here to avoid db query at
            # a later stage when we perform automatically unmute topic in muted stream operation.
            if row["user_profile_id"] == sender_id:
                sender_muted_stream = row["is_muted"]

        user_id_to_visibility_policy = stream_topic.user_id_to_visibility_policy_dict()

        def notification_recipients(setting: str) -> set[int]:
            return {
                row["user_profile_id"]
                for row in subscription_rows
                if user_allows_notifications_in_StreamTopic(
                    row["is_muted"],
                    user_id_to_visibility_policy.get(
                        row["user_profile_id"], UserTopic.VisibilityPolicy.INHERIT
                    ),
                    row[setting],
                    row["user_profile_" + setting],
                )
            }

        stream_push_user_ids = notification_recipients("push_notifications")
        stream_email_user_ids = notification_recipients("email_notifications")

        def followed_topic_notification_recipients(setting: str) -> set[int]:
            return {
                row["user_profile_id"]
                for row in subscription_rows
                if user_id_to_visibility_policy.get(
                    row["user_profile_id"], UserTopic.VisibilityPolicy.INHERIT
                )
                == UserTopic.VisibilityPolicy.FOLLOWED
                and row["followed_topic_" + setting]
            }

        followed_topic_email_user_ids = followed_topic_notification_recipients(
            "email_notifications"
        )
        followed_topic_push_user_ids = followed_topic_notification_recipients("push_notifications")

        if possible_stream_wildcard_mention or possible_topic_wildcard_mention:
            # We calculate `wildcard_mentions_notify_user_ids` and `followed_topic_wildcard_mentions_notify_user_ids`
            # only if there's a possible stream or topic wildcard mention in the message.
            # This is important so as to avoid unnecessarily sending huge user ID lists with
            # thousands of elements to the event queue (which can happen because these settings
            # are `True` by default for new users.)
            wildcard_mentions_notify_user_ids = notification_recipients("wildcard_mentions_notify")
            followed_topic_wildcard_mentions_notify_user_ids = (
                followed_topic_notification_recipients("wildcard_mentions_notify")
            )

        if possible_stream_wildcard_mention:
            stream_wildcard_mention_user_ids = wildcard_mentions_notify_user_ids
            stream_wildcard_mention_in_followed_topic_user_ids = (
                followed_topic_wildcard_mentions_notify_user_ids
            )

        if possible_topic_wildcard_mention:
            topic_wildcard_mention_user_ids = topic_participant_user_ids.intersection(
                wildcard_mentions_notify_user_ids
            )
            topic_wildcard_mention_in_followed_topic_user_ids = (
                topic_participant_user_ids.intersection(
                    followed_topic_wildcard_mentions_notify_user_ids
                )
            )

    elif recipient.type == Recipient.DIRECT_MESSAGE_GROUP:
        message_to_user_id_set = set(get_direct_message_group_user_ids(recipient))

    else:
        raise ValueError("Bad recipient type")

    # Important note: Because we haven't rendered Markdown yet, we
    # don't yet know which of these possibly-mentioned users was
    # actually mentioned in the message (in other words, the
    # mention syntax might have been in a code block or otherwise
    # escaped).  `get_ids_for` will filter these extra user rows
    # for our data structures not related to bots
    user_ids = message_to_user_id_set | possibly_mentioned_user_ids

    if user_ids:
        query: QuerySet[UserProfile, ActiveUserDict] = UserProfile.objects.filter(
            is_active=True
        ).values(
            "id",
            "enable_online_push_notifications",
            "enable_offline_email_notifications",
            "enable_offline_push_notifications",
            "is_bot",
            "bot_type",
            "long_term_idle",
        )

        # query_for_ids is fast highly optimized for large queries, and we
        # need this codepath to be fast (it's part of sending messages)
        query = query_for_ids(
            query=query,
            user_ids=sorted(user_ids),
            field="id",
        )
        rows = list(query)
    else:
        # TODO: We should always have at least one user_id as a recipient
        #       of any message we send.  Right now the exception to this
        #       rule is `notify_new_user`, which, at least in a possibly
        #       contrived test scenario, can attempt to send messages
        #       to an inactive bot.  When we plug that hole, we can avoid
        #       this `else` clause and just `assert(user_ids)`.
        #
        # UPDATE: It's February 2020 (and a couple years after the above
        #         comment was written).  We have simplified notify_new_user
        #         so that it should be a little easier to reason about.
        #         There is currently some cleanup to how we handle cross
        #         realm bots that is still under development.  Once that
        #         effort is complete, we should be able to address this
        #         to-do.
        rows = []

    def get_ids_for(f: Callable[[ActiveUserDict], bool]) -> set[int]:
        """Only includes users on the explicit message to line"""
        return {row["id"] for row in rows if f(row)} & message_to_user_id_set

    active_user_ids = get_ids_for(lambda r: True)
    online_push_user_ids = get_ids_for(
        lambda r: r["enable_online_push_notifications"],
    )

   
from typing import Optional, List, Set, Dict, Tuple, Sequence, Callable, Collection, Any
from django.db.models import QuerySet
from django.core.exceptions import ValidationError
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import override as override_language
from django.db import IntegrityError, transaction
from django.conf import settings
from django.db.models import F, Q
from django.utils.html import escape
from email.headerregistry import Address
from datetime import timedelta
from collections import defaultdict
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TypedDict
import logging
import orjson

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
    url_embed_data: Optional[Dict[str, Optional[UrlEmbedData]]] = None,
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
    active_user_ids: Set[int]
    online_push_user_ids: Set[int]
    dm_mention_email_disabled_user_ids: Set[int]
    dm_mention_push_disabled_user_ids: Set[int]
    stream_email_user_ids: Set[int]
    stream_push_user_ids: Set[int]
    topic_wildcard_mention_user_ids: Set[int]
    stream_wildcard_mention_user_ids: Set[int]
    followed_topic_email_user_ids: Set[int]
    followed_topic_push_user_ids: Set[int]
    topic_wildcard_mention_in_followed_topic_user_ids: Set[int]
    stream_wildcard_mention_in_followed_topic_user_ids: Set[int]
    muted_sender_user_ids: Set[int]
    um_eligible_user_ids: Set[int]
    long_term_idle_user_ids: Set[int]
    default_bot_user_ids: Set[int]
    service_bot_tuples: List[Tuple[int, int]]
    all_bot_user_ids: Set[int]
    topic_participant_user_ids: Set[int]
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
    possibly_mentioned_user_ids: AbstractSet[int] = set(),
    possible_topic_wildcard_mention: bool = True,
    possible_stream_wildcard_mention: bool = True,
) -> RecipientInfoResult:
    stream_push_user_ids: Set[int] = set()
    stream_email_user_ids: Set[int] = set()
    topic_wildcard_mention_user_ids: Set[int] = set()
    stream_wildcard_mention_user_ids: Set[int] = set()
    followed_topic_push_user_ids: Set[int] = set()
    followed_topic_email_user_ids: Set[int] = set()
    topic_wildcard_mention_in_followed_topic_user_ids: Set[int] = set()
    stream_wildcard_mention_in_followed_topic_user_ids: Set[int] = set()
    muted_sender_user_ids: Set[int] = get_muting_users(sender_id)
    topic_participant_user_ids: Set[int] = set()
    sender_muted_stream: Optional[bool] = None

    if recipient.type == Recipient.PERSONAL:
        message_to_user_id_set = {recipient.type_id, sender_id}
        assert len(message_to_user_id_set) in [1, 2]

    elif recipient.type == Recipient.STREAM:
        assert stream_topic is not None

        if possible_topic_wildcard_mention:
            topic_participant_user_ids = participants_for_topic(
                realm_id, recipient.id, stream_topic.topic_name
            )
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
            if row["user_profile_id"] == sender_id:
                sender_muted_stream = row["is_muted"]

        user_id_to_visibility_policy = stream_topic.user_id_to_visibility_policy_dict()

        def notification_recipients(setting: str) -> Set[int]:
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

        def followed_topic_notification_recipients(setting: str) -> Set[int]:
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

        query = query_for_ids(
            query=query,
            user_ids=sorted(user_ids),
            field="id",
        )
        rows = list(query)
    else:
        rows = []

    def get_ids_for(f: Callable[[ActiveUserDict], bool]) -> Set[int]:
        return {row["id"] for row in rows if f(row)} & message_to_user_id_set

    active_user_ids = get_ids_for(lambda r: True)
    online_push_user_ids = get_ids_for(
        lambda r: r["enable_online_push_notifications"],
    )

    dm_mention_email_disabled_user_ids = get_ids_for(
        lambda r: not r["enable_offline_email_notifications"]
    )
    dm_mention_push_disabled_user_ids = get_ids_for(
        lambda r: not r["enable_offline_push_notifications"]
    )

    um_eligible_user_ids = get_ids_for(lambda r: True)

    long_term_idle_user_ids = get_ids_for(
        lambda r: r["long_term_idle"],
    )

    default_bot_user_ids = {
        row["id"] for row in rows if row["is_bot"] and row["bot_type"] == UserProfile.DEFAULT_BOT
    }

    service_bot_tuples = [
        (row["id"], row["bot_type"])
        for row in rows
        if row["is_bot"] and row["bot_type"] in UserProfile.SERVICE_BOT_TYPES
    ]

    all_bot_user_ids = {row["id"] for row in rows if row["is_bot"]}

    return RecipientInfoResult(
        active_user_ids=active_user_ids,
        online_push_user_ids=online_push_user_ids,
        dm_mention_email_disabled_user_ids=dm_mention_email_disabled_user_ids,
        dm_mention_push_disabled_user_ids=dm_mention_push_disabled_user_ids,
        stream_push_user_ids=stream_push_user_ids,
        stream_email_user_ids=stream_email_user_ids,
        topic_wildcard_mention_user_ids=topic_wildcard_mention_user_ids,
        stream_wildcard_mention_user_ids=stream_wildcard_mention_user_ids,
        followed_topic_push_user_ids=followed_topic_push_user_ids,
        followed_topic_email_user_ids=followed_topic_email_user_ids,
        topic_wildcard_mention_in_followed_topic_user_ids=topic_wildcard_mention_in_followed_topic_user_ids,
        stream_wildcard_mention_in_followed_topic_user_ids=stream_wildcard_mention_in_followed_topic_user_ids,
        muted_sender_user_ids=muted_sender_user_ids,
        um_eligible_user_ids=um_eligible_user_ids,
        long_term_idle_user_ids=long_term_idle_user_ids,
        default_bot_user_ids=default_bot_user_ids,
        service_bot_tuples=service_bot_tuples,
        all_bot_user_ids=all_bot_user_ids,
        topic_participant_user_ids=topic_participant_user_ids,
        sender_muted_stream=sender_muted_stream,
    )

def get_service_bot_events(
    sender: UserProfile,
    service_bot_tuples: List[Tuple[int, int]],
    mentioned_user_ids: Set[int],
    active_user_ids: Set[int],
    recipient_type: int,
) -> Dict[str, List[Dict[str, Any]]]:
    event_dict: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    if sender.is_bot:
        return event_dict

    def maybe_add_event(user_profile_id: int, bot_type: int) -> None:
        if bot_type == UserProfile.OUTGOING_WEBHOOK_BOT:
            queue_name = "outgoing_webhooks"
        elif bot_type == UserProfile.EMBEDDED_BOT:
            queue_name = "embedded_bots"
        else:
            logging.error(
                "Unexpected bot_type for Service bot id=%s: %s",
                user_profile_id,
                bot_type,
            )
            return

        is_stream = recipient_type == Recipient.STREAM

        if user_profile_id not in mentioned_user_ids and user_profile_id not in active_user_ids:
            return

        if is_stream and user_profile_id in mentioned_user_ids:
            trigger = "mention"
        elif not is_stream and user_profile_id in active_user_ids:
            trigger = NotificationTriggers.DIRECT_MESSAGE
        else:
            return

        event_dict[queue_name].append(
            {
                "trigger": trigger,
                "user_profile_id": user_profile_id,
            }
        )

    for user_profile_id, bot_type in service_bot_tuples:
        maybe_add_event(
            user_profile_id=user_profile_id,
            bot_type=bot_type,
        )

    return event_dict

def build_message_send_dict(
    message: Message,
    stream: Optional[Stream] = None,
    local_id: Optional[str] = None,
    sender_queue_id: Optional[str] = None,
    widget_content_dict: Optional[Dict[str, Any]] = None,
    email_gateway: bool = False,
    mention_backend: Optional[MentionBackend] = None,
    limit_unread_user_ids: Optional[Set[int]] = None,
    disable_external_notifications: bool = False,
    recipients_for_user_creation_events: Optional[Dict[UserProfile, Set[int]]] = None,
    acting_user: Optional[UserProfile] = None,
) -> SendMessageRequest:
    realm = message.realm

    if mention_backend is None:
        mention_backend = MentionBackend(realm.id)

    mention_data = MentionData(
        mention_backend=mention_backend,
        content=message.content,
        message_sender=message.sender,
    )

    if message.is_stream_message():
        stream_id = message.recipient.type_id
        stream_topic: Optional[StreamTopicTarget] = StreamTopicTarget(
            stream_id=stream_id,
            topic_name=message.topic_name(),
        )
    else:
        stream_topic = None

    info = get_recipient_info(
        realm_id=realm.id,
        recipient=message.recipient,
        sender_id=message.sender_id,
        stream_topic=stream_topic,
        possibly_mentioned_user_ids=mention_data.get_user_ids(),
        possible_topic_wildcard_mention=mention_data.message_has_topic_wildcards(),
        possible_stream_wildcard_mention=mention_data.message_has_stream_wildcards(),
    )

    assert message.rendered_content is None

    rendering_result = render_incoming_message(
        message,
        message.content,
        realm,
        mention_data=mention_data,
        email_gateway=email_gateway,
        acting_user=acting_user,
    )
    message.rendered_content = rendering_result.rendered_content
    message.rendered_content_version = markdown_version
    links_for_embed = rendering_result.links_for_preview

    mentioned_user_groups_map = get_user_group_mentions_data(
        mentioned_user_ids=rendering_result.mentions_user_ids,
        mentioned_user_group_ids=list(rendering_result.mentions_user_group_ids),
        mention_data=mention_data,
    )

    for group_id in rendering_result.mentions_user_group_ids:
        members = mention_data.get_group_members(group_id)
        rendering_result.mentions_user_ids.update(members)

    if rendering_result.mentions_stream_wildcard:
        stream_wildcard_mention_user_ids = info.stream_wildcard_mention_user_ids
        stream_wildcard_mention_in_followed_topic_user_ids = (
            info.stream_wildcard_mention_in_followed_topic_user_ids
        )
    else:
        stream_wildcard_mention_user_ids = set()
        stream_wildcard_mention_in_followed_topic_user_ids = set()

    if rendering_result.mentions_topic_wildcard:
        topic_wildcard_mention_user_ids = info.topic_wildcard_mention_user_ids
        topic_wildcard_mention_in_followed_topic_user_ids = (
            info.topic_wildcard_mention_in_followed_topic_user_ids
        )
        topic_participant_user_ids = info.topic_participant_user_ids
    else:
        topic_wildcard_mention_user_ids = set()
        topic_wildcard_mention_in_followed_topic_user_ids = set()
        topic_participant_user_ids = set()

    mentioned_user_ids = rendering_result.mentions_user_ids
    default_bot_user_ids = info.default_bot_user_ids
    mentioned_bot_user_ids = default_bot_user_ids & mentioned_user_ids
    info.um_eligible_user_ids |= mentioned_bot_user_ids

    message_send_dict = SendMessageRequest(
        stream=stream,
        sender_muted_stream=info.sender_muted_stream,
        local_id=local_id,
        sender_queue_id=sender_queue_id,
        realm=realm,
        mention_data=mention_data,
        mentioned_user_groups_map=mentioned_user_groups_map,
        message=message,
        rendering_result=rendering_result,
        active_user_ids=info.active_user_ids,
        online_push_user_ids=info.online_push_user_ids,
        dm_mention_email_disabled_user_ids=info.dm_mention_email_disabled_user_ids,
        dm_mention_push_disabled_user_ids=info.dm_mention_push_disabled_user_ids,
        stream_push_user_ids=info.stream_push_user_ids,
        stream_email_user_ids=info.stream_email_user_ids,
        followed_topic_push_user_ids=info.followed_topic_push_user_ids,
        followed_topic_email_user_ids=info.followed_topic_email_user_ids,
        muted_sender_user_ids=info.muted_sender_user_ids,
        um_eligible_user_ids=info.um_eligible_user_ids,
        long_term_idle_user_ids=info.long_term_idle_user_ids,
        default_bot_user_ids=info.default_bot_user_ids,
        service_bot_tuples=info.service_bot_tuples,
        all_bot_user_ids=info.all_bot_user_ids,
        topic_wildcard_mention_user_ids=topic_wildcard_mention_user_ids,
        stream_wildcard_mention_user_ids=stream_wildcard_mention_user_ids,
        topic_wildcard_mention_in_followed_topic_user_ids=topic_wildcard_mention_in_followed_topic_user_ids,
        stream_wildcard_mention_in_followed_topic_user_ids=stream_wildcard_mention_in_followed_topic_user_ids,
        links_for_embed=links_for_embed,
        widget_content=widget_content_dict,
        limit_unread_user_ids=limit_unread_user_ids,
        disable_external_notifications=disable_external_notifications,
        topic_participant_user_ids=topic_participant_user_ids,
        recipients_for_user_creation_events=recipients_for_user_creation_events,
    )

    return message_send_dict

def create_user_messages(
    message: Message,
    rendering_result: MessageRenderingResult,
    um_eligible_user_ids: AbstractSet[int],
    long_term_idle_user_ids: AbstractSet[int],
    stream_push_user_ids: AbstractSet[int],
    stream_email_user_ids: AbstractSet[int],
    mentioned_user_ids: AbstractSet[int],
    followed_topic_push_user_ids: AbstractSet[int],
    followed_topic_email_user_ids: AbstractSet[int],
    mark_as_read_user_ids: Set[int],
    limit_unread_user_ids: Optional[Set[int]],
    topic_participant_user_ids: Set[int],
) -> List[UserMessageLite]:
    ids_with_alert_words = rendering_result.user_ids_with_alert_words
    is_stream_message = message.is_stream_message()

    base_flags = 0
    if rendering_result.mentions_stream_wildcard:
        base_flags |= UserMessage.flags.stream_wildcard_mentioned
    if message.recipient.type in [Recipient.DIRECT_MESSAGE_GROUP, Recipient.PERSONAL]:
        base_flags |= UserMessage.flags.is_private

    user_messages = []
    for user_profile_id in um_eligible_user_ids:
        flags = base_flags
        if user_profile_id in mark_as_read_user_ids or (
            limit_unread_user_ids is not None and user_profile_id not in limit_unread_user_ids
        ):
            flags |= UserMessage.flags.read
        if user_profile_id in mentioned_user_ids:
            flags |= UserMessage.flags.mentioned
        if user_profile_id in ids_with_alert_words:
            flags |= UserMessage.flags.has_alert_word
        if (
            rendering_result.mentions_topic_wildcard
            and user_profile_id in topic_participant_user_ids
        ):
            flags |= UserMessage.flags.topic_wildcard_mentioned

        if (
            user_profile_id in long_term_idle_user_ids
            and user_profile_id not in stream_push_user_ids
            and user_profile_id not in stream_email_user_ids
            and user_profile_id not in followed_topic_push_user_ids
            and user_profile_id not in followed_topic_email_user_ids
            and is_stream_message
            and int(flags) == 0
        ):
            continue

        um = UserMessageLite(
            user_profile_id=user_profile_id,
            message_id=message.id,
            flags=flags,
        )
        user_messages.append(um)

    return user_messages

def filter_presence_idle_user_ids(user_ids: Set[int]) -> List[int]:
    if not user_ids:
        return []

    recent = timezone_now() - timedelta(seconds=settings.OFFLINE_THRESHOLD_SECS)
    rows = UserPresence.objects.filter(
        user_profile_id__in=user_ids,
        last_active_time__gte=recent,
    ).values("user_profile_id")
    active_user_ids = {row["user_profile_id"] for row in rows}
    idle_user_ids = user_ids - active_user_ids
    return sorted(idle_user_ids)

def get_active_presence_idle_user_ids(
    realm: Realm,
    sender_id: int,
    user_notifications_data_list: List[UserMessageNotificationsData],
) -> List[int]:
    if realm.presence_disabled:
        return []

    user_ids = set()
    for user_notifications_data in user_notifications_data_list:
        if user_notifications_data.is_notifiable(sender_id, idle=True):
            user_ids.add(user_notifications_data.user_id)

    return filter_presence_idle_user_ids(user_ids)

@transaction.atomic(savepoint=False)
def do_send_messages(
    send_message_requests_maybe_none: Sequence[Optional[SendMessageRequest]],
    *,
    mark_as_read: Sequence[int] = [],
) -> List[SentMessageResult]:
    send_message_requests = [
        send_request
        for send_request in send_message_requests_maybe_none
        if send_request is not None
    ]

    user_message_flags: Dict[int, Dict[int, List[str]]] = defaultdict(dict)

    Message.objects.bulk_create(send_request.message for send_request in send_message_requests)

    for send_request in send_message_requests:
        if do_claim_attachments(
            send_request.message, send_request.rendering_result.potential_attachment_path_ids
        ):
            send_request.message.has_attachment = True
            update_fields = ["has_attachment"]

            assert send_request.message.rendered_content is not None
            if send_request.rendering_result.thumbnail_spinners:
                previews = get_user_upload_previews(
                    send_request.message.realm_id,
                    send_request.message.content,
                    lock=True,
                    enqueue=False,
                    path_ids=list(send_request.rendering_result.thumbnail_spinners),
                )
                new_rendered_content = rewrite_thumbnailed_images(
                    send_request.message.rendered_content, previews
                )[0]
                if new_rendered_content is not None:
                    send_request.message.rendered_content = new_rendered_content
                    update_fields.append("rendered_content")

            send_request.message.save(update_fields=update_fields)

    ums: List[UserMessageLite] = []
    for send_request in send_message_requests:
        mentioned_user_ids = send_request.rendering_result.mentions_user_ids

        mark_as_read_user_ids = send_request.muted_sender_user_ids
        mark_as_read_user_ids.update(mark_as_read)

        user_messages = create_user_messages(
            message=send_request.message,
            rendering_result=send_request.rendering_result,
            um_eligible_user_ids=send_request.um_eligible_user_ids,
            long_term_idle_user_ids=send_request.long_term_idle_user_ids,
            stream_push_user_ids=send_request.stream_push_user_ids,
            stream_email_user_ids=send_request.stream_email_user_ids,
            mentioned_user_ids=mentioned_user_ids,
            followed_topic_push_user_ids=send_request.followed_topic_push_user_ids,
            followed_topic_email_user_ids=send_request.followed_topic_email_user_ids,
            mark_as_read_user_ids=mark_as_read_user_ids,
            limit_unread_user_ids=send_request.limit_unread_user_ids,
            topic_participant_user_ids=send_request.topic_participant_user_ids,
        )

        for um in user_messages:
            user_message_flags[send_request.message.id][um.user_profile_id] = um.flags_list()

        ums.extend(user_messages)

        send_request.service_queue_events = get_service_bot_events(
            sender=send_request.message.sender,
            service_bot_tuples=send_request.service_bot_tuples,
            mentioned_user_ids=mentioned_user_ids,
            active_user_ids=send_request.active_user_ids,
            recipient_type=send_request.message.recipient.type,
        )

    bulk_insert_ums(ums)

    for send_request in send_message_requests:
        do_widget_post_save_actions(send_request)

    for send_request in send_message_requests:
        realm_id: Optional[int] = None
        if send_request.message.is_stream_message():
            if send_request.stream is None:
                stream_id = send_request.message.recipient.type_id
                send_request.stream = Stream.objects.get(id=stream_id)
            assert send_request.stream is not None
            realm_id = send_request.stream.realm_id
            sender = send_request.message.sender

            if set_visibility_policy_possible(sender, send_request.message) and not (
                sender.automatically_follow_topics_policy
                == sender.automatically_unmute_topics_in_muted_streams_policy
                == UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_NEVER
            ):
                try:
                    user_topic = UserTopic.objects.get(
                        user_profile=sender,
                        stream_id=send_request.stream.id,
                        topic_name__iexact=send_request.message.topic_name(),
                    )
                    visibility_policy = user_topic.visibility_policy
                except UserTopic.DoesNotExist:
                    visibility_policy = UserTopic.VisibilityPolicy.INHERIT

                new_visibility_policy = visibility_policy_for_send_message(
                    sender,
                    send_request.message,
                    send_request.stream,
                    send_request.sender_muted_stream,
                    visibility_policy,
                )
                if new_visibility_policy:
                    do_set_user_topic_visibility_policy(
                        user_profile=sender,
                        stream=send_request.stream,
                        topic_name=send_request.message.topic_name(),
                        visibility_policy=new_visibility_policy,
                    )
                    send_request.automatic_new_visibility_policy = new_visibility_policy

            human_user_personal_mentions = send_request.rendering_result.mentions_user_ids & (
                send_request.active_user_ids - send_request.all_bot_user_ids
            )
            expect_follow_user_profiles: Set[UserProfile] = set()

            if len(human_user_personal_mentions) > 0:
                expect_follow_user_profiles = set(
                    UserProfile.objects.filter(
                        realm_id=realm_id,
                        id__in=human_user_personal_mentions,
                        automatically_follow_topics_where_mentioned=True,
                    )
                )
            if len(expect_follow_user_profiles) > 0:
                user_topics_query_set = UserTopic.objects.filter(
                    user_profile__in=expect_follow_user_profiles,
                    stream_id=send_request.stream.id,
                    topic_name__iexact=send_request.message.topic_name(),
                    visibility_policy__in=[
                        UserTopic.VisibilityPolicy.MUTED,
                        UserTopic.VisibilityPolicy.FOLLOWED,
                    ],
                )
                skip_follow_users = {
                    user_topic.user_profile for user_topic in user_topics_query_set
                }

                to_follow_users = list(expect_follow_user_profiles - skip_follow_users)

                if to_follow_users:
                    bulk_do_set_user_topic_visibility_policy(
                        user_profiles=to_follow_users,
                        stream=send_request.stream,
                        topic_name=send_request.message.topic_name(),
                        visibility_policy=UserTopic.VisibilityPolicy.FOLLOWED,
                    )

        wide_message_dict = MessageDict.wide_dict(send_request.message, realm_id)

        user_flags = user_message_flags.get(send_request.message.id, {})

        user_ids = send_request.active_user_ids | set(user_flags.keys())
        sender_id = send_request.message.sender_id

        if sender_id in user_ids:
            user_list = [sender_id, *user_ids - {sender_id}]
        else:
            user_list = list(user_ids)

        class UserData(TypedDict):
            id: int
            flags: List[str]
            mentioned_user_group_id: Optional[int]

        users: List[UserData] = []
        for user_id in user_list:
            flags = user_flags.get(user_id, [])
            if "stream_wildcard_mentioned" in flags or "topic_wildcard_mentioned" in flags:
                flags.append("wildcard_mentioned")
            user_data: UserData = dict(id=user_id, flags=flags, mentioned_user_group_id=None)

            if user_id in send_request.mentioned_user_groups_map:
                user_data["mentioned_user_group
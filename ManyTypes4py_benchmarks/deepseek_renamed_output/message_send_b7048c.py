import logging
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence, Set as AbstractSet
from dataclasses import dataclass
from datetime import timedelta
from email.headerregistry import Address
from typing import Any, TypedDict, Optional, List, Dict, Set, Tuple, Union
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
from zerver.actions.user_topics import bulk_do_set_user_topic_visibility_policy, do_set_user_topic_visibility_policy
from zerver.lib.addressee import Addressee
from zerver.lib.alert_words import get_alert_word_automaton
from zerver.lib.cache import cache_with_key, user_profile_delivery_email_cache_key
from zerver.lib.create_user import create_user
from zerver.lib.exceptions import DirectMessageInitiationError, DirectMessagePermissionError, JsonableError, MarkdownRenderingError, StreamDoesNotExistError, StreamWildcardMentionNotAllowedError, StreamWithIDDoesNotExistError, TopicWildcardMentionNotAllowedError, ZephyrMessageAlreadySentError
from zerver.lib.markdown import MessageRenderingResult, render_message_markdown
from zerver.lib.markdown import version as markdown_version
from zerver.lib.mention import MentionBackend, MentionData
from zerver.lib.message import SendMessageRequest, check_user_group_mention_allowed, normalize_body, set_visibility_policy_possible, stream_wildcard_mention_allowed, topic_wildcard_mention_allowed, truncate_topic, visibility_policy_for_send_message
from zerver.lib.message_cache import MessageDict
from zerver.lib.muted_users import get_muting_users
from zerver.lib.notification_data import UserMessageNotificationsData, get_user_group_mentions_data, user_allows_notifications_in_StreamTopic
from zerver.lib.query_helpers import query_for_ids
from zerver.lib.queue import queue_event_on_commit
from zerver.lib.recipient_users import recipient_for_user_profiles
from zerver.lib.stream_subscription import get_subscriptions_for_send_message, num_subscribers_for_stream_id
from zerver.lib.stream_topic import StreamTopicTarget
from zerver.lib.streams import access_stream_for_send_message, ensure_stream, notify_stream_is_recently_active_update, subscribed_to_stream
from zerver.lib.string_validation import check_stream_name
from zerver.lib.thumbnail import get_user_upload_previews, rewrite_thumbnailed_images
from zerver.lib.timestamp import timestamp_to_datetime
from zerver.lib.topic import participants_for_topic
from zerver.lib.url_preview.types import UrlEmbedData
from zerver.lib.user_groups import is_any_user_in_group, is_user_in_group
from zerver.lib.user_message import UserMessageLite, bulk_insert_ums
from zerver.lib.users import check_can_access_user, get_inaccessible_user_ids, get_subscribers_of_target_user_subscriptions, get_user_ids_who_can_access_user, get_users_involved_in_dms_with_target_users, user_access_restricted_in_realm
from zerver.lib.validator import check_widget_content
from zerver.lib.widget import do_widget_post_save_actions
from zerver.models import Client, Message, Realm, Recipient, Stream, UserMessage, UserPresence, UserProfile, UserTopic
from zerver.models.clients import get_client
from zerver.models.groups import SystemGroups
from zerver.models.recipients import get_direct_message_group_user_ids
from zerver.models.scheduled_jobs import NotificationTriggers
from zerver.models.streams import get_stream_by_id_for_sending_message, get_stream_by_name_for_sending_message
from zerver.models.users import get_system_bot, get_user_by_delivery_email, is_cross_realm_bot_email
from zerver.tornado.django_api import send_event_on_commit


def func_x0nyp3vz(email: str) -> str:
    return Address(addr_spec=email).username + ' (IRC)'


def func_2gjdefja(email: str) -> str:
    return Address(addr_spec=email).username + ' (XMPP)'


def func_jcr9sbj2(realm: Realm, email: str, email_to_fullname: Callable[[str], str]) -> str:
    return user_profile_delivery_email_cache_key(email, realm.id)


@cache_with_key(get_user_profile_delivery_email_cache_key, timeout=3600 * 24 * 7)
def func_fr3vumxr(realm: Realm, email: str, email_to_fullname: Callable[[str], str]) -> UserProfile:
    try:
        return get_user_by_delivery_email(email, realm)
    except UserProfile.DoesNotExist:
        try:
            return create_user(email=email, password=None, realm=realm,
                full_name=email_to_fullname(email), active=False,
                is_mirror_dummy=True)
        except IntegrityError:
            return get_user_by_delivery_email(email, realm)


def func_gu1eb6n0(message: Message, content: str, realm: Realm, mention_data: Optional[MentionData] = None,
    url_embed_data: Optional[UrlEmbedData] = None, email_gateway: bool = False, acting_user: Optional[UserProfile] = None) -> MessageRenderingResult:
    realm_alert_words_automaton = get_alert_word_automaton(realm)
    try:
        rendering_result = render_message_markdown(message=message, content
            =content, realm=realm, realm_alert_words_automaton=
            realm_alert_words_automaton, mention_data=mention_data,
            url_embed_data=url_embed_data, email_gateway=email_gateway,
            acting_user=acting_user)
    except MarkdownRenderingError:
        raise JsonableError(_('Unable to render message'))
    return rendering_result


@dataclass
class RecipientInfoResult:
    active_user_ids: Set[int]
    online_push_user_ids: Set[int]
    dm_mention_email_disabled_user_ids: Set[int]
    dm_mention_push_disabled_user_ids: Set[int]
    stream_push_user_ids: Set[int]
    stream_email_user_ids: Set[int]
    topic_wildcard_mention_user_ids: Set[int]
    stream_wildcard_mention_user_ids: Set[int]
    followed_topic_push_user_ids: Set[int]
    followed_topic_email_user_ids: Set[int]
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
    pass


@dataclass
class SentMessageResult:
    message_id: int
    automatic_new_visibility_policy: Optional[int] = None


def func_o4182qvp(*, realm_id: int, recipient: Recipient, sender_id: int, stream_topic: Optional[StreamTopicTarget],
    possibly_mentioned_user_ids: Set[int] = set(), possible_topic_wildcard_mention: bool = True,
    possible_stream_wildcard_mention: bool = True) -> RecipientInfoResult:
    stream_push_user_ids = set()
    stream_email_user_ids = set()
    topic_wildcard_mention_user_ids = set()
    stream_wildcard_mention_user_ids = set()
    followed_topic_push_user_ids = set()
    followed_topic_email_user_ids = set()
    topic_wildcard_mention_in_followed_topic_user_ids = set()
    stream_wildcard_mention_in_followed_topic_user_ids = set()
    muted_sender_user_ids = get_muting_users(sender_id)
    topic_participant_user_ids = set()
    sender_muted_stream = None
    if recipient.type == Recipient.PERSONAL:
        message_to_user_id_set = {recipient.type_id, sender_id}
        assert len(message_to_user_id_set) in [1, 2]
    elif recipient.type == Recipient.STREAM:
        assert stream_topic is not None
        if possible_topic_wildcard_mention:
            topic_participant_user_ids = participants_for_topic(realm_id,
                recipient.id, stream_topic.topic_name)
            topic_participant_user_ids.add(sender_id)
        subscription_rows = get_subscriptions_for_send_message(realm_id=
            realm_id, stream_id=stream_topic.stream_id, topic_name=
            stream_topic.topic_name, possible_stream_wildcard_mention=
            possible_stream_wildcard_mention, topic_participant_user_ids=
            topic_participant_user_ids, possibly_mentioned_user_ids=
            possibly_mentioned_user_ids).annotate(
            user_profile_email_notifications=F(
            'user_profile__enable_stream_email_notifications'),
            user_profile_push_notifications=F(
            'user_profile__enable_stream_push_notifications'),
            user_profile_wildcard_mentions_notify=F(
            'user_profile__wildcard_mentions_notify'),
            followed_topic_email_notifications=F(
            'user_profile__enable_followed_topic_email_notifications'),
            followed_topic_push_notifications=F(
            'user_profile__enable_followed_topic_push_notifications'),
            followed_topic_wildcard_mentions_notify=F(
            'user_profile__enable_followed_topic_wildcard_mentions_notify')
            ).values('user_profile_id', 'push_notifications',
            'email_notifications', 'wildcard_mentions_notify',
            'followed_topic_push_notifications',
            'followed_topic_email_notifications',
            'followed_topic_wildcard_mentions_notify',
            'user_profile_email_notifications',
            'user_profile_push_notifications',
            'user_profile_wildcard_mentions_notify', 'is_muted').order_by(
            'user_profile_id')
        message_to_user_id_set = set()
        for row in subscription_rows:
            message_to_user_id_set.add(row['user_profile_id'])
            if row['user_profile_id'] == sender_id:
                sender_muted_stream = row['is_muted']
        user_id_to_visibility_policy = (stream_topic.
            user_id_to_visibility_policy_dict())

        def func_wx9cbnmf(setting: str) -> Set[int]:
            return {row['user_profile_id'] for row in subscription_rows if
                user_allows_notifications_in_StreamTopic(row['is_muted'],
                user_id_to_visibility_policy.get(row['user_profile_id'],
                UserTopic.VisibilityPolicy.INHERIT), row[setting], row[
                'user_profile_' + setting])}
        stream_push_user_ids = func_wx9cbnmf('push_notifications')
        stream_email_user_ids = func_wx9cbnmf('email_notifications')

        def func_yn6rh265(setting: str) -> Set[int]:
            return {row['user_profile_id'] for row in subscription_rows if 
                user_id_to_visibility_policy.get(row['user_profile_id'],
                UserTopic.VisibilityPolicy.INHERIT) == UserTopic.
                VisibilityPolicy.FOLLOWED and row['followed_topic_' + setting]}
        followed_topic_email_user_ids = func_yn6rh265('email_notifications')
        followed_topic_push_user_ids = func_yn6rh265('push_notifications')
        if possible_stream_wildcard_mention or possible_topic_wildcard_mention:
            wildcard_mentions_notify_user_ids = func_wx9cbnmf(
                'wildcard_mentions_notify')
            followed_topic_wildcard_mentions_notify_user_ids = func_yn6rh265(
                'wildcard_mentions_notify')
        if possible_stream_wildcard_mention:
            stream_wildcard_mention_user_ids = (
                wildcard_mentions_notify_user_ids)
            stream_wildcard_mention_in_followed_topic_user_ids = (
                followed_topic_wildcard_mentions_notify_user_ids)
        if possible_topic_wildcard_mention:
            topic_wildcard_mention_user_ids = (topic_participant_user_ids.
                intersection(wildcard_mentions_notify_user_ids))
            topic_wildcard_mention_in_followed_topic_user_ids = (
                topic_participant_user_ids.intersection(
                followed_topic_wildcard_mentions_notify_user_ids))
    elif recipient.type == Recipient.DIRECT_MESSAGE_GROUP:
        message_to_user_id_set = set(get_direct_message_group_user_ids(
            recipient))
    else:
        raise ValueError('Bad recipient type')
    user_ids = message_to_user_id_set | possibly_mentioned_user_ids
    if user_ids:
        query = UserProfile.objects.filter(is_active=True).values('id',
            'enable_online_push_notifications',
            'enable_offline_email_notifications',
            'enable_offline_push_notifications', 'is_bot', 'bot_type',
            'long_term_idle')
        query = query_for_ids(query=query, user_ids=sorted(user_ids), field
            ='id')
        rows = list(query)
    else:
        rows = []

    def func_d0w8gm8d(f: Callable[[Dict[str, Any]], bool]) -> Set[int]:
        """Only includes users on the explicit message to line"""
        return {row['id'] for row in rows if f(row)} & message_to_user_id_set
    active_user_ids = func_d0w8gm8d(lambda r: True)
    online_push_user_ids = func_d0w8gm8d(lambda r: r[
        'enable_online_push_notifications'])
    dm_mention_email_disabled_user_ids = func_d0w8gm8d(lambda r: not r[
        'enable_offline_email_notifications'])
    dm_mention_push_disabled_user_ids = func_d0w8gm8d(lambda r: not r[
        'enable_offline_push_notifications'])
    um_eligible_user_ids = func_d0w8gm8d(lambda r: True)
    long_term_idle_user_ids = func_d0w8gm8d(lambda r: r['long_term_idle'])
    default_bot_user_ids = {row['id'] for row in rows if row['is_bot'] and 
        row['bot_type'] == UserProfile.DEFAULT_BOT}
    service_bot_tuples = [(row['id'], row['bot_type']) for row in rows if 
        row['is_bot'] and row['bot_type'] in UserProfile.SERVICE_BOT_TYPES]
    all_bot_user_ids = {row['id'] for row in rows if row['is_bot']}
    return RecipientInfoResult(active_user_ids=active_user_ids,
        online_push_user_ids=online_push_user_ids,
        dm_mention_email_disabled_user_ids=
        dm_mention_email_disabled_user_ids,
        dm_mention_push_disabled_user_ids=dm_mention_push_disabled_user_ids,
        stream_push_user_ids=stream_push_user_ids, stream_email_user_ids=
        stream_email_user_ids, topic_wildcard_mention_user_ids=
        topic_wildcard_mention_user_ids, stream_wildcard_mention_user_ids=
        stream_wildcard_mention_user_ids, followed_topic_push_user_ids=
        followed_topic_push_user_ids, followed_topic_email_user_ids=
        followed_topic_email_user_ids,
        topic_wildcard_mention_in_followed_topic_user_ids=
        topic_wildcard_mention_in_followed_topic_user_ids,
        stream_wildcard_mention_in_followed_topic_user_ids=
        stream_wildcard_mention_in_followed_topic_user_ids,
        muted_sender_user_ids=muted_sender_user_ids, um_eligible_user_ids=
        um_eligible_user_ids, long_term_idle_user_ids=
        long_term_idle_user_ids, default_bot_user_ids=default_bot_user_ids,
        service_bot_tuples=service_bot_tuples, all_bot_user_ids=
        all_bot_user_ids, topic_participant_user_ids=
        topic_participant_user_ids, sender_muted_stream=sender_muted_stream)


def func_06sas9a9(sender: UserProfile, service_bot_tuples: List[Tuple[int, int]], mentioned_user_ids: Set[int],
    active_user_ids: Set[int], recipient_type: int) -> Dict[str, List[Dict[str, Any]]]:
    event_dict = defaultdict(list)
    if sender.is_bot:
        return event_dict

    def func_mw3uyztg(user_profile_id: int, bot_type: int) -> None:
        if bot_type == UserProfile.OUTGOING_WEBHOOK_BOT:
            queue_name = 'outgoing_webhooks'
        elif bot_type == UserProfile.EMBEDDED_BOT:
            queue_name = 'embedded_bots'
        else:
            logging.error('Unexpected bot_type for Service bot id=%s: %s',
                user_profile_id, bot_type)
            return
        is_stream = recipient_type == Recipient.STREAM
        if (user_profile_id not in mentioned_user_ids and user_profile_id
             not in active_user_ids):
            return
        if is_stream and user_profile_id in mentioned_user_ids:
            trigger = 'mention'
        elif not is_stream and user_profile_id in active_user_ids:
            trigger = NotificationTriggers.DIRECT_MESSAGE
        else:
            return
        event_dict[queue_name].append({'trigger': trigger,
            'user_profile_id': user_profile_id})
    for user_profile_id, bot_type in service_bot_tuples:
        func_mw3uyztg(user_profile_id=user_profile_id, bot_type=bot_type)
    return event_dict


def func_3a0cti7l(message: Message, stream: Optional[Stream] = None, local_id: Optional[str] = None, sender_queue_id: Optional[str] = None,
    widget_content_dict
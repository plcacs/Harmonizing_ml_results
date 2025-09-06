import logging
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import timedelta
from email.headerregistry import Address
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TypedDict
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
from zerver.lib.cache import cache_with_key, user_profile_delivery_email_cache_key, get_user_profile_delivery_email_cache_key
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

def func_ilype0c3(email: str) -> str:
    return Address(addr_spec=email).username + ' (IRC)'

def func_g8retj1u(email: str) -> str:
    return Address(addr_spec=email).username + ' (XMPP)'

def func_krd2o6nw(realm: Realm, email: str, email_to_fullname: Callable[[str], str]) -> str:
    return user_profile_delivery_email_cache_key(email, realm.id)

@cache_with_key(get_user_profile_delivery_email_cache_key, timeout=3600 * 24 * 7)
def func_ryejyagq(realm: Realm, email: str, email_to_fullname: Callable[[str], str]) -> UserProfile:
    try:
        return get_user_by_delivery_email(email, realm)
    except UserProfile.DoesNotExist:
        try:
            return create_user(email=email, password=None, realm=realm,
                               full_name=email_to_fullname(email), active=False,
                               is_mirror_dummy=True)
        except IntegrityError:
            return get_user_by_delivery_email(email, realm)

def func_h7a287w4(message: Message, content: str, realm: Realm, 
                  mention_data: Optional[MentionData] = None,
                  url_embed_data: Optional[UrlEmbedData] = None, 
                  email_gateway: bool = False, acting_user: Optional[UserProfile] = None) -> MessageRenderingResult:
    realm_alert_words_automaton = get_alert_word_automaton(realm)
    try:
        rendering_result = render_message_markdown(message=message, content=content, realm=realm, 
                                                   realm_alert_words_automaton=realm_alert_words_automaton, 
                                                   mention_data=mention_data, url_embed_data=url_embed_data, 
                                                   email_gateway=email_gateway, acting_user=acting_user)
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
    service_bot_tuples: List[Tuple[int, Any]]
    all_bot_user_ids: Set[int]
    topic_participant_user_ids: Set[int]
    sender_muted_stream: Optional[bool]

class ActiveUserDict(TypedDict):
    ...

@dataclass
class SentMessageResult:
    message_id: int
    automatic_new_visibility_policy: Optional[Any] = None

def func_4avgk8r7(*, realm_id: int, recipient: Recipient, sender_id: int, 
                   stream_topic: Optional[StreamTopicTarget],
                   possibly_mentioned_user_ids: Set[int] = set(), 
                   possible_topic_wildcard_mention: bool = True,
                   possible_stream_wildcard_mention: bool = True) -> RecipientInfoResult:
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
        message_to_user_id_set: Set[int] = {recipient.type_id, sender_id}
        assert len(message_to_user_id_set) in [1, 2]
    elif recipient.type == Recipient.STREAM:
        assert stream_topic is not None
        if possible_topic_wildcard_mention:
            topic_participant_user_ids = participants_for_topic(realm_id,
                                                                 recipient.id, stream_topic.topic_name)
            topic_participant_user_ids.add(sender_id)
        subscription_rows = get_subscriptions_for_send_message(realm_id=realm_id, stream_id=stream_topic.stream_id, topic_name=stream_topic.topic_name, possible_stream_wildcard_mention=possible_stream_wildcard_mention, topic_participant_user_ids=topic_participant_user_ids, possibly_mentioned_user_ids=possibly_mentioned_user_ids).annotate(
            user_profile_email_notifications=F('user_profile__enable_stream_email_notifications'),
            user_profile_push_notifications=F('user_profile__enable_stream_push_notifications'),
            user_profile_wildcard_mentions_notify=F('user_profile__wildcard_mentions_notify'),
            followed_topic_email_notifications=F('user_profile__enable_followed_topic_email_notifications'),
            followed_topic_push_notifications=F('user_profile__enable_followed_topic_push_notifications'),
            followed_topic_wildcard_mentions_notify=F('user_profile__enable_followed_topic_wildcard_mentions_notify')
        ).values('user_profile_id', 'push_notifications', 'email_notifications', 'wildcard_mentions_notify',
                 'followed_topic_push_notifications', 'followed_topic_email_notifications',
                 'followed_topic_wildcard_mentions_notify', 'user_profile_email_notifications',
                 'user_profile_push_notifications', 'user_profile_wildcard_mentions_notify', 'is_muted').order_by('user_profile_id')
        message_to_user_id_set: Set[int] = set()
        for row in subscription_rows:
            message_to_user_id_set.add(row['user_profile_id'])
            if row['user_profile_id'] == sender_id:
                sender_muted_stream = row['is_muted']
        user_id_to_visibility_policy = stream_topic.user_id_to_visibility_policy_dict()

        def func_40l5p6c8(setting: str) -> Set[int]:
            return {row['user_profile_id'] for row in subscription_rows if
                    user_allows_notifications_in_StreamTopic(row['is_muted'],
                                                             user_id_to_visibility_policy.get(row['user_profile_id'],
                                                                                             UserTopic.VisibilityPolicy.INHERIT),
                                                             row[setting], row['user_profile_' + setting])}
        stream_push_user_ids = func_40l5p6c8('push_notifications')
        stream_email_user_ids = func_40l5p6c8('email_notifications')

        def func_pe61uv3f(setting: str) -> Set[int]:
            return {row['user_profile_id'] for row in subscription_rows if 
                    user_id_to_visibility_policy.get(row['user_profile_id'],
                                                     UserTopic.VisibilityPolicy.INHERIT) == UserTopic.VisibilityPolicy.FOLLOWED and row['followed_topic_' + setting]}
        followed_topic_email_user_ids = func_pe61uv3f('email_notifications')
        followed_topic_push_user_ids = func_pe61uv3f('push_notifications')
        if possible_stream_wildcard_mention or possible_topic_wildcard_mention:
            wildcard_mentions_notify_user_ids: Set[int] = func_40l5p6c8('wildcard_mentions_notify')
            followed_topic_wildcard_mentions_notify_user_ids: Set[int] = func_pe61uv3f('wildcard_mentions_notify')
        if possible_stream_wildcard_mention:
            stream_wildcard_mention_user_ids = wildcard_mentions_notify_user_ids
            stream_wildcard_mention_in_followed_topic_user_ids = followed_topic_wildcard_mentions_notify_user_ids
        if possible_topic_wildcard_mention:
            topic_wildcard_mention_user_ids = topic_participant_user_ids.intersection(wildcard_mentions_notify_user_ids)
            topic_wildcard_mention_in_followed_topic_user_ids = topic_participant_user_ids.intersection(followed_topic_wildcard_mentions_notify_user_ids)
    elif recipient.type == Recipient.DIRECT_MESSAGE_GROUP:
        message_to_user_id_set = set(get_direct_message_group_user_ids(recipient))
    else:
        raise ValueError('Bad recipient type')
    user_ids: Set[int] = message_to_user_id_set | possibly_mentioned_user_ids
    if user_ids:
        query: QuerySet = UserProfile.objects.filter(is_active=True).values('id',
                                                                           'enable_online_push_notifications',
                                                                           'enable_offline_email_notifications',
                                                                           'enable_offline_push_notifications', 'is_bot', 'bot_type',
                                                                           'long_term_idle')
        query = query_for_ids(query=query, user_ids=sorted(user_ids), field='id')
        rows: List[Dict[str, Any]] = list(query)
    else:
        rows = []

    def func_yonflixy(f: Callable[[Dict[str, Any]], bool]) -> Set[int]:
        """Only includes users on the explicit message to line"""
        return {row['id'] for row in rows if f(row)} & message_to_user_id_set
    active_user_ids: Set[int] = func_yonflixy(lambda r: True)
    online_push_user_ids: Set[int] = func_yonflixy(lambda r: r['enable_online_push_notifications'])
    dm_mention_email_disabled_user_ids: Set[int] = func_yonflixy(lambda r: not r['enable_offline_email_notifications'])
    dm_mention_push_disabled_user_ids: Set[int] = func_yonflixy(lambda r: not r['enable_offline_push_notifications'])
    um_eligible_user_ids: Set[int] = func_yonflixy(lambda r: True)
    long_term_idle_user_ids: Set[int] = func_yonflixy(lambda r: r['long_term_idle'])
    default_bot_user_ids: Set[int] = {row['id'] for row in rows if row['is_bot'] and row['bot_type'] == UserProfile.DEFAULT_BOT}
    service_bot_tuples: List[Tuple[int, Any]] = [(row['id'], row['bot_type']) for row in rows if row['is_bot'] and row['bot_type'] in UserProfile.SERVICE_BOT_TYPES]
    all_bot_user_ids: Set[int] = {row['id'] for row in rows if row['is_bot']}
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
        sender_muted_stream=sender_muted_stream
    )

def func_nnfs507g(sender: UserProfile, service_bot_tuples: List[Tuple[int, Any]], 
                  mentioned_user_ids: Set[int], active_user_ids: Set[int], recipient_type: int) -> Dict[str, List[Dict[str, Any]]]:
    event_dict: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if sender.is_bot:
        return event_dict

    def func_gubsck7g(user_profile_id: int, bot_type: Any) -> None:
        if bot_type == UserProfile.OUTGOING_WEBHOOK_BOT:
            queue_name = 'outgoing_webhooks'
        elif bot_type == UserProfile.EMBEDDED_BOT:
            queue_name = 'embedded_bots'
        else:
            logging.error('Unexpected bot_type for Service bot id=%s: %s', user_profile_id, bot_type)
            return
        is_stream: bool = recipient_type == Recipient.STREAM
        if (user_profile_id not in mentioned_user_ids and user_profile_id not in active_user_ids):
            return
        if is_stream and user_profile_id in mentioned_user_ids:
            trigger = 'mention'
        elif not is_stream and user_profile_id in active_user_ids:
            trigger = NotificationTriggers.DIRECT_MESSAGE
        else:
            return
        event_dict[queue_name].append({'trigger': trigger, 'user_profile_id': user_profile_id})
    for user_profile_id, bot_type in service_bot_tuples:
        func_gubsck7g(user_profile_id=user_profile_id, bot_type=bot_type)
    return event_dict

def func_o3krieek(message: Message, stream: Optional[Stream] = None, local_id: Optional[int] = None, 
                   sender_queue_id: Optional[int] = None, widget_content_dict: Optional[Dict[str, Any]] = None, 
                   email_gateway: bool = False, mention_backend: Optional[MentionBackend] = None,
                   limit_unread_user_ids: Optional[Set[int]] = None, disable_external_notifications: bool = False,
                   recipients_for_user_creation_events: Optional[Dict[Any, Set[int]]] = None, 
                   acting_user: Optional[UserProfile] = None) -> SendMessageRequest:
    realm: Realm = message.realm
    if mention_backend is None:
        mention_backend = MentionBackend(realm.id)
    mention_data: MentionData = MentionData(mention_backend=mention_backend, content=message.content, message_sender=message.sender)
    if message.is_stream_message():
        stream_id: int = message.recipient.type_id
        stream_topic: StreamTopicTarget = StreamTopicTarget(stream_id=stream_id, topic_name=message.topic_name())
    else:
        stream_topic = None
    info: RecipientInfoResult = func_4avgk8r7(realm_id=realm.id, recipient=message.recipient,
                                               sender_id=message.sender_id, stream_topic=stream_topic,
                                               possibly_mentioned_user_ids=mention_data.get_user_ids(),
                                               possible_topic_wildcard_mention=mention_data.message_has_topic_wildcards(),
                                               possible_stream_wildcard_mention=mention_data.message_has_stream_wildcards())
    assert message.rendered_content is None
    rendering_result: MessageRenderingResult = func_h7a287w4(message, message.content, realm,
                                                               mention_data=mention_data, email_gateway=email_gateway, acting_user=acting_user)
    message.rendered_content = rendering_result.rendered_content
    message.rendered_content_version = markdown_version
    links_for_embed: Any = rendering_result.links_for_preview
    mentioned_user_groups_map: Dict[int, Any] = get_user_group_mentions_data(mentioned_user_ids=rendering_result.mentions_user_ids, 
                                                                             mentioned_user_group_ids=list(rendering_result.mentions_user_group_ids), 
                                                                             mention_data=mention_data)
    for group_id in rendering_result.mentions_user_group_ids:
        members = mention_data.get_group_members(group_id)
        rendering_result.mentions_user_ids.update(members)
    if rendering_result.mentions_stream_wildcard:
        stream_wildcard_mention_user_ids: Set[int] = info.stream_wildcard_mention_user_ids
        stream_wildcard_mention_in_followed_topic_user_ids: Set[int] = info.stream_wildcard_mention_in_followed_topic_user_ids
    else:
        stream_wildcard_mention_user_ids = set()
        stream_wildcard_mention_in_followed_topic_user_ids = set()
    if rendering_result.mentions_topic_wildcard:
        topic_wildcard_mention_user_ids: Set[int] = info.topic_wildcard_mention_user_ids
        topic_wildcard_mention_in_followed_topic_user_ids: Set[int] = info.topic_wildcard_mention_in_followed_topic_user_ids
        topic_participant_user_ids: Set[int] = info.topic_participant_user_ids
    else:
        topic_wildcard_mention_user_ids = set()
        topic_wildcard_mention_in_followed_topic_user_ids = set()
        topic_participant_user_ids = set()
    mentioned_user_ids: Set[int] = rendering_result.mentions_user_ids
    default_bot_user_ids: Set[int] = info.default_bot_user_ids
    mentioned_bot_user_ids: Set[int] = default_bot_user_ids & mentioned_user_ids
    info.um_eligible_user_ids |= mentioned_bot_user_ids
    message_send_dict: SendMessageRequest = SendMessageRequest(
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
        recipients_for_user_creation_events=recipients_for_user_creation_events
    )
    return message_send_dict

def func_8og22tns(message: Message, rendering_result: MessageRenderingResult, um_eligible_user_ids: Set[int],
                  long_term_idle_user_ids: Set[int], stream_push_user_ids: Set[int], stream_email_user_ids: Set[int],
                  mentioned_user_ids: Set[int], followed_topic_push_user_ids: Set[int],
                  followed_topic_email_user_ids: Set[int], mark_as_read_user_ids: Set[int],
                  limit_unread_user_ids: Optional[Set[int]], topic_participant_user_ids: Set[int]) -> List[UserMessageLite]:
    ids_with_alert_words: Set[int] = rendering_result.user_ids_with_alert_words
    is_stream_message: bool = message.is_stream_message()
    base_flags: int = 0
    if rendering_result.mentions_stream_wildcard:
        base_flags |= UserMessage.flags.stream_wildcard_mentioned
    if message.recipient.type in [Recipient.DIRECT_MESSAGE_GROUP, Recipient.PERSONAL]:
        base_flags |= UserMessage.flags.is_private
    user_messages: List[UserMessageLite] = []
    for user_profile_id in um_eligible_user_ids:
        flags = base_flags
        if (user_profile_id in mark_as_read_user_ids or (limit_unread_user_ids is not None and user_profile_id not in limit_unread_user_ids)):
            flags |= UserMessage.flags.read
        if user_profile_id in mentioned_user_ids:
            flags |= UserMessage.flags.mentioned
        if user_profile_id in ids_with_alert_words:
            flags |= UserMessage.flags.has_alert_word
        if (rendering_result.mentions_topic_wildcard and user_profile_id in topic_participant_user_ids):
            flags |= UserMessage.flags.topic_wildcard_mentioned
        if (user_profile_id in long_term_idle_user_ids and user_profile_id not in stream_push_user_ids and user_profile_id not in stream_email_user_ids and user_profile_id not in followed_topic_push_user_ids and user_profile_id not in followed_topic_email_user_ids and is_stream_message and int(flags) == 0):
            continue
        um = UserMessageLite(user_profile_id=user_profile_id, message_id=message.id, flags=flags)
        user_messages.append(um)
    return user_messages

def func_i4k0rime(user_ids: Set[int]) -> List[int]:
    if not user_ids:
        return []
    recent = timezone_now() - timedelta(seconds=settings.OFFLINE_THRESHOLD_SECS)
    rows = UserPresence.objects.filter(user_profile_id__in=user_ids, last_active_time__gte=recent).values('user_profile_id')
    active_user_ids: Set[int] = {row['user_profile_id'] for row in rows}
    idle_user_ids: Set[int] = user_ids - active_user_ids
    return sorted(idle_user_ids)

def func_zics59t9(realm: Realm, sender_id: int, user_notifications_data_list: List[UserMessageNotificationsData]) -> List[int]:
    if realm.presence_disabled:
        return []
    user_ids: Set[int] = set()
    for user_notifications_data in user_notifications_data_list:
        if user_notifications_data.is_notifiable(sender_id, idle=True):
            user_ids.add(user_notifications_data.user_id)
    return func_i4k0rime(user_ids)

@transaction.atomic(savepoint=False)
def func_5if1xban(send_message_requests_maybe_none: Sequence[Optional[SendMessageRequest]], *, mark_as_read: List[int] = []) -> List[SentMessageResult]:
    send_message_requests: List[SendMessageRequest] = [send_request for send_request in send_message_requests_maybe_none if send_request is not None]
    user_message_flags: Dict[int, Dict[int, List[str]]] = defaultdict(dict)
    Message.objects.bulk_create(send_request.message for send_request in send_message_requests)
    for send_request in send_message_requests:
        if do_claim_attachments(send_request.message, send_request.rendering_result.potential_attachment_path_ids):
            send_request.message.has_attachment = True
            update_fields: List[str] = ['has_attachment']
            assert send_request.message.rendered_content is not None
            if send_request.rendering_result.thumbnail_spinners:
                previews = get_user_upload_previews(send_request.message.realm_id, send_request.message.content, lock=True, enqueue=False, path_ids=list(send_request.rendering_result.thumbnail_spinners))
                new_rendered_content = rewrite_thumbnailed_images(send_request.message.rendered_content, previews)[0]
                if new_rendered_content is not None:
                    send_request.message.rendered_content = new_rendered_content
                    update_fields.append('rendered_content')
            send_request.message.save(update_fields=update_fields)
    ums: List[UserMessageLite] = []
    for send_request in send_message_requests:
        mentioned_user_ids: Set[int] = send_request.rendering_result.mentions_user_ids
        mark_as_read_user_ids: Set[int] = send_request.muted_sender_user_ids
        mark_as_read_user_ids.update(mark_as_read)
        user_messages: List[UserMessageLite] = func_8og22tns(message=send_request.message,
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
                                                              topic_participant_user_ids=send_request.topic_participant_user_ids)
        for um in user_messages:
            user_message_flags[send_request.message.id][um.user_profile_id] = um.flags_list()
        ums.extend(user_messages)
        send_request.service_queue_events = func_nnfs507g(sender=send_request.message.sender, service_bot_tuples=send_request.service_bot_tuples, mentioned_user_ids=mentioned_user_ids, active_user_ids=send_request.active_user_ids, recipient_type=send_request.message.recipient.type)
    bulk_insert_ums(ums)
    for send_request in send_message_requests:
        do_widget_post_save_actions(send_request)
    for send_request in send_message_requests:
        realm_id: Optional[int] = None
        if send_request.message.is_stream_message():
            if send_request.stream is None:
                stream_id: int = send_request.message.recipient.type_id
                send_request.stream = Stream.objects.get(id=stream_id)
            assert send_request.stream is not None
            realm_id = send_request.stream.realm_id
            sender = send_request.message.sender
            if (set_visibility_policy_possible(sender, send_request.message) and not sender.automatically_follow_topics_policy == sender.automatically_unmute_topics_in_muted_streams_policy == UserProfile.AUTOMATICALLY_CHANGE_VISIBILITY_POLICY_NEVER):
                try:
                    user_topic = UserTopic.objects.get(user_profile=sender, stream_id=send_request.stream.id, topic_name__iexact=send_request.message.topic_name())
                    visibility_policy = user_topic.visibility_policy
                except UserTopic.DoesNotExist:
                    visibility_policy = UserTopic.VisibilityPolicy.INHERIT
                new_visibility_policy = visibility_policy_for_send_message(sender, send_request.message, send_request.stream, send_request.sender_muted_stream, visibility_policy)
                if new_visibility_policy:
                    do_set_user_topic_visibility_policy(user_profile=sender, stream=send_request.stream, topic_name=send_request.message.topic_name(), visibility_policy=new_visibility_policy)
                    send_request.automatic_new_visibility_policy = new_visibility_policy
            human_user_personal_mentions: Set[int] = (send_request.rendering_result.mentions_user_ids & send_request.active_user_ids - send_request.all_bot_user_ids)
            expect_follow_user_profiles: Set[UserProfile] = set()
            if len(human_user_personal_mentions) > 0:
                expect_follow_user_profiles = set(UserProfile.objects.filter(realm_id=realm_id, id__in=human_user_personal_mentions, automatically_follow_topics_where_mentioned=True))
            if len(expect_follow_user_profiles) > 0:
                user_topics_query_set = UserTopic.objects.filter(user_profile__in=expect_follow_user_profiles, stream_id=send_request.stream.id, topic_name__iexact=send_request.message.topic_name(), visibility_policy__in=[UserTopic.VisibilityPolicy.MUTED, UserTopic.VisibilityPolicy.FOLLOWED])
                skip_follow_users: Set[UserProfile] = {user_topic.user_profile for user_topic in user_topics_query_set}
                to_follow_users: List[UserProfile] = list(expect_follow_user_profiles - skip_follow_users)
                if to_follow_users:
                    bulk_do_set_user_topic_visibility_policy(user_profiles=to_follow_users, stream=send_request.stream, topic_name=send_request.message.topic_name(), visibility_policy=UserTopic.VisibilityPolicy.FOLLOWED)
        wide_message_dict: Dict[str, Any] = MessageDict.wide_dict(send_request.message, realm_id)
        user_flags: Dict[int, List[str]] = user_message_flags.get(send_request.message.id, {})
        user_ids: Set[int] = send_request.active_user_ids | set(user_flags.keys())
        sender_id: int = send_request.message.sender_id
        if sender_id in user_ids:
            user_list: List[int] = [sender_id, *(user_ids - {sender_id})]
        else:
            user_list = list(user_ids)

        class UserData(TypedDict):
            id: int
            flags: List[str]
            mentioned_user_group_id: Optional[Any]

        users: List[UserData] = []
        for user_id in user_list:
            flags = user_flags.get(user_id, [])
            if ('stream_wildcard_mentioned' in flags or 'topic_wildcard_mentioned' in flags):
                flags.append('wildcard_mentioned')
            user_data: UserData = dict(id=user_id, flags=flags, mentioned_user_group_id=None)
            if user_id in send_request.mentioned_user_groups_map:
                user_data['mentioned_user_group_id'] = send_request.mentioned_user_groups_map[user_id]
            users.append(user_data)
        sender = send_request.message.sender
        recipient_type = wide_message_dict['type']
        user_notifications_data_list: List[UserMessageNotificationsData] = [UserMessageNotificationsData.from_user_id_sets(
            user_id=user_id, 
            flags=user_flags.get(user_id, []), 
            private_message=(recipient_type == 'private'),
            disable_external_notifications=send_request.disable_external_notifications, 
            online_push_user_ids=send_request.online_push_user_ids,
            dm_mention_push_disabled_user_ids=send_request.dm_mention_push_disabled_user_ids,
            dm_mention_email_disabled_user_ids=send_request.dm_mention_email_disabled_user_ids, 
            stream_push_user_ids=send_request.stream_push_user_ids, 
            stream_email_user_ids=send_request.stream_email_user_ids,
            topic_wildcard_mention_user_ids=send_request.topic_wildcard_mention_user_ids,
            stream_wildcard_mention_user_ids=send_request.stream_wildcard_mention_user_ids, 
            followed_topic_push_user_ids=send_request.followed_topic_push_user_ids,
            followed_topic_email_user_ids=send_request.followed_topic_email_user_ids,
            topic_wildcard_mention_in_followed_topic_user_ids=send_request.topic_wildcard_mention_in_followed_topic_user_ids,
            stream_wildcard_mention_in_followed_topic_user_ids=send_request.stream_wildcard_mention_in_followed_topic_user_ids,
            muted_sender_user_ids=send_request.muted_sender_user_ids,
            all_bot_user_ids=send_request.all_bot_user_ids) for user_id in send_request.active_user_ids]
        presence_idle_user_ids: List[int] = func_zics59t9(realm=send_request.realm, sender_id=sender.id, user_notifications_data_list=user_notifications_data_list)
        if send_request.recipients_for_user_creation_events is not None:
            from zerver.actions.create_user import notify_created_user
            for new_accessible_user, notify_user_ids in send_request.recipients_for_user_creation_events.items():
                notify_created_user(new_accessible_user, list(notify_user_ids))
        event: Dict[str, Any] = dict(
            type='message',
            message=send_request.message.id,
            message_dict=wide_message_dict,
            presence_idle_user_ids=presence_idle_user_ids,
            online_push_user_ids=list(send_request.online_push_user_ids),
            dm_mention_push_disabled_user_ids=list(send_request.dm_mention_push_disabled_user_ids),
            dm_mention_email_disabled_user_ids=list(send_request.dm_mention_email_disabled_user_ids),
            stream_push_user_ids=list(send_request.stream_push_user_ids),
            stream_email_user_ids=list(send_request.stream_email_user_ids),
            topic_wildcard_mention_user_ids=list(send_request.topic_wildcard_mention_user_ids),
            stream_wildcard_mention_user_ids=list(send_request.stream_wildcard_mention_user_ids),
            followed_topic_push_user_ids=list(send_request.followed_topic_push_user_ids),
            followed_topic_email_user_ids=list(send_request.followed_topic_email_user_ids),
            topic_wildcard_mention_in_followed_topic_user_ids=list(send_request.topic_wildcard_mention_in_followed_topic_user_ids),
            stream_wildcard_mention_in_followed_topic_user_ids=list(send_request.stream_wildcard_mention_in_followed_topic_user_ids),
            muted_sender_user_ids=list(send_request.muted_sender_user_ids),
            all_bot_user_ids=list(send_request.all_bot_user_ids),
            disable_external_notifications=send_request.disable_external_notifications,
            realm_host=send_request.realm.host
        )
        if send_request.message.is_stream_message():
            assert send_request.stream is not None
            stream_update_fields: List[str] = []
            if send_request.stream.is_public():
                event['realm_id'] = send_request.stream.realm_id
                event['stream_name'] = send_request.stream.name
            if send_request.stream.invite_only:
                event['invite_only'] = True
            if send_request.stream.first_message_id is None:
                send_request.stream.first_message_id = send_request.message.id
                stream_update_fields.append('first_message_id')
            if not send_request.stream.is_recently_active:
                send_request.stream.is_recently_active = True
                stream_update_fields.append('is_recently_active')
                notify_stream_is_recently_active_update(send_request.stream, True)
            if len(stream_update_fields) > 0:
                send_request.stream.save(update_fields=stream_update_fields)
            if user_access_restricted_in_realm(send_request.message.sender) and not subscribed_to_stream(send_request.message.sender, send_request.stream.id):
                user_ids_who_can_access_sender = get_user_ids_who_can_access_user(send_request.message.sender)
                user_ids_receiving_event: Set[int] = {user['id'] for user in users}
                user_ids_without_access_to_sender: Set[int] = user_ids_receiving_event - set(user_ids_who_can_access_sender)
                event['user_ids_without_access_to_sender'] = user_ids_without_access_to_sender
        if send_request.local_id is not None:
            event['local_id'] = send_request.local_id
        if send_request.sender_queue_id is not None:
            event['sender_queue_id'] = send_request.sender_queue_id
        send_event_on_commit(send_request.realm, event, users)
        if send_request.links_for_embed:
            event_data: Dict[str, Any] = {'message_id': send_request.message.id,
                                          'message_content': send_request.message.content,
                                          'message_realm_id': send_request.realm.id,
                                          'urls': list(send_request.links_for_embed)}
            queue_event_on_commit('embed_links', event_data)
        if send_request.message.recipient.type == Recipient.PERSONAL:
            welcome_bot_id: int = get_system_bot(settings.WELCOME_BOT, send_request.realm.id).id
            if (welcome_bot_id in send_request.active_user_ids and welcome_bot_id != send_request.message.sender_id):
                from zerver.lib.onboarding import send_welcome_bot_response
                send_welcome_bot_response(send_request)
        assert send_request.service_queue_events is not None
        for queue_name, events in send_request.service_queue_events.items():
            for event_item in events:
                queue_event_on_commit(queue_name, {'message': wide_message_dict, 'trigger': event_item['trigger'], 'user_profile_id': event_item['user_profile_id']})
    sent_message_results: List[SentMessageResult] = [SentMessageResult(message_id=send_request.message.id, automatic_new_visibility_policy=send_request.automatic_new_visibility_policy) for send_request in send_message_requests]
    return sent_message_results

def func_3xks5lef(message: Message) -> Optional[int]:
    if message.recipient.type == Recipient.DIRECT_MESSAGE_GROUP:
        time_window = timedelta(seconds=10)
    else:
        time_window = timedelta(seconds=0)
    messages = Message.objects.filter(realm_id=message.realm_id, sender=message.sender, recipient=message.recipient, subject=message.topic_name(), content=message.content, sending_client=message.sending_client, date_sent__gte=message.date_sent - time_window, date_sent__lte=message.date_sent + time_window)
    if messages.exists():
        return messages[0].id
    return None

def func_cnxomsk1(s: str) -> Union[str, int]:
    try:
        data = orjson.loads(s)
    except orjson.JSONDecodeError:
        return s
    if isinstance(data, list):
        if len(data) != 1:
            raise JsonableError(_('Expected exactly one channel'))
        data = data[0]
    if isinstance(data, str):
        return data
    if isinstance(data, int):
        return data
    raise JsonableError(_('Invalid data type for channel'))

def func_md922ntt(s: str) -> Union[List[str], List[int]]:
    try:
        data = orjson.loads(s)
    except orjson.JSONDecodeError:
        data = s
    if isinstance(data, str):
        data = data.split(',')
    if not isinstance(data, list):
        raise JsonableError(_('Invalid data type for recipients'))
    if not data:
        return data
    if isinstance(data[0], str):
        return get_validated_emails(data)  # type: ignore
    if not isinstance(data[0], int):
        raise JsonableError(_('Invalid data type for recipients'))
    return get_validated_user_ids(data)  # type: ignore

def func_wpj12jq5(user_ids: Sequence[Any]) -> List[int]:
    for user_id in user_ids:
        if not isinstance(user_id, int):
            raise JsonableError(_('Recipient lists may contain emails or user IDs, but not both.'))
    return list(set(user_ids))  # type: ignore

def func_xipn445r(emails: Sequence[str]) -> List[str]:
    return list(filter(bool, {email.strip() for email in emails}))

def func_sm168tyh(sender: UserProfile, client: Client, stream_name: str, topic_name: str, body: str, *, realm: Optional[Realm] = None, read_by_sender: bool = False) -> int:
    addressee: Addressee = Addressee.for_stream_name(stream_name, topic_name)
    message: SendMessageRequest = check_message(sender, client, addressee, body, realm)  # type: ignore
    sent_message_result: SentMessageResult = func_5if1xban([message], mark_as_read=[sender.id] if read_by_sender else [])[0]
    return sent_message_result.message_id

def func_4eb4mler(sender: UserProfile, client: Client, stream_id: int, topic_name: str, body: str, realm: Optional[Realm] = None) -> int:
    addressee: Addressee = Addressee.for_stream_id(stream_id, topic_name)
    message: SendMessageRequest = check_message(sender, client, addressee, body, realm)  # type: ignore
    sent_message_result: SentMessageResult = func_5if1xban([message])[0]
    return sent_message_result.message_id

def func_k3fkr32t(sender: UserProfile, client: Client, receiving_user: UserProfile, body: str) -> int:
    addressee: Addressee = Addressee.for_user_profile(receiving_user)
    message: SendMessageRequest = check_message(sender, client, addressee, body)  # type: ignore
    sent_message_result: SentMessageResult = func_5if1xban([message])[0]
    return sent_message_result.message_id

def func_76a4b7bh(sender: UserProfile, client: Client, recipient_type_name: str, message_to: Any, topic_name: str, message_content: str, realm: Optional[Realm] = None, forged: bool = False, forged_timestamp: Optional[int] = None, forwarder_user_profile: Optional[UserProfile] = None, local_id: Optional[int] = None, sender_queue_id: Optional[int] = None, widget_content: Optional[str] = None, *, skip_stream_access_check: bool = False, read_by_sender: bool = False) -> SentMessageResult:
    addressee: Addressee = Addressee.legacy_build(sender, recipient_type_name, message_to, topic_name)
    try:
        message: SendMessageRequest = check_message(sender, client, addressee, message_content, realm, forged, forged_timestamp, forwarder_user_profile, local_id, sender_queue_id, widget_content, skip_stream_access_check=skip_stream_access_check)  # type: ignore
    except ZephyrMessageAlreadySentError as e:
        return SentMessageResult(message_id=e.message_id)
    return func_5if1xban([message], mark_as_read=[sender.id] if read_by_sender else [])[0]

def func_ed79vci3(sender: UserProfile, realm: Realm, content: str) -> None:
    if sender.realm.is_zephyr_mirror_realm or sender.realm.deactivated:
        return
    if not sender.is_bot or sender.bot_owner is None:
        return
    if sender.realm != realm:
        return
    last_reminder = sender.last_reminder
    waitperiod = timedelta(minutes=UserProfile.BOT_OWNER_STREAM_ALERT_WAITPERIOD)
    if last_reminder and timezone_now() - last_reminder <= waitperiod:
        return
    internal_send_private_message(get_system_bot(settings.NOTIFICATION_BOT, sender.bot_owner.realm_id), sender.bot_owner, content)
    sender.last_reminder = timezone_now()
    sender.save(update_fields=['last_reminder'])

def func_kvvbksp6(stream: Optional[Stream], realm: Realm, sender: UserProfile, stream_name: Optional[str] = None, stream_id: Optional[int] = None) -> None:
    if not sender.is_bot or sender.bot_owner is None:
        return
    if sender.bot_owner is not None:
        with override_language(sender.bot_owner.default_language):
            arg_dict: Dict[str, Any] = {'bot_identity': f'`{sender.delivery_email}`'}
            if stream is None:
                if stream_id is not None:
                    arg_dict = {**arg_dict, 'channel_id': stream_id}
                    content = _('Your bot {bot_identity} tried to send a message to channel ID {channel_id}, but there is no channel with that ID.').format(**arg_dict)
                else:
                    assert stream_name is not None
                    arg_dict = {**arg_dict, 'channel_name': f'#**{stream_name}**', 'new_channel_link': '#channels/new'}
                    content = _('Your bot {bot_identity} tried to send a message to channel {channel_name}, but that channel does not exist. Click [here]({new_channel_link}) to create it.').format(**arg_dict)
            else:
                if num_subscribers_for_stream_id(stream.id) > 0:
                    return
                arg_dict = {**arg_dict, 'channel_name': f'#**{stream.name}**'}
                content = _('Your bot {bot_identity} tried to send a message to channel {channel_name}. The channel exists but does not have any subscribers.').format(**arg_dict)
        func_ed79vci3(sender, realm, content)

def func_53hjqnkk(stream_name: str, realm: Realm, sender: UserProfile) -> Stream:
    stream_name = stream_name.strip()
    check_stream_name(stream_name)
    try:
        stream = get_stream_by_name_for_sending_message(stream_name, realm)
        func_kvvbksp6(stream, realm, sender)
    except Stream.DoesNotExist:
        func_kvvbksp6(None, realm, sender, stream_name=stream_name)
        raise StreamDoesNotExistError(escape(stream_name))
    return stream

def func_9bi5kqrs(stream_id: int, realm: Realm, sender: UserProfile) -> Stream:
    try:
        stream = get_stream_by_id_for_sending_message(stream_id, realm)
        func_kvvbksp6(stream, realm, sender)
    except Stream.DoesNotExist:
        func_kvvbksp6(None, realm, sender, stream_id=stream_id)
        raise StreamWithIDDoesNotExistError(stream_id)
    return stream

def func_bpcs6tkp(realm: Realm, sender: UserProfile, recipient_users: Sequence[UserProfile], recipient: Recipient) -> None:
    if sender.is_bot:
        return
    if all(user_profile.is_bot or user_profile.id == sender.id for user_profile in recipient_users):
        return
    direct_message_permission_group = realm.direct_message_permission_group
    if (not hasattr(direct_message_permission_group, 'named_user_group') or direct_message_permission_group.named_user_group.name != SystemGroups.EVERYONE):
        user_ids = [recipient_user.id for recipient_user in recipient_users] + [sender.id]
        if not is_any_user_in_group(direct_message_permission_group, user_ids):
            is_nobody_group = (hasattr(direct_message_permission_group, 'named_user_group') and direct_message_permission_group.named_user_group.name == SystemGroups.NOBODY)
            raise DirectMessagePermissionError(is_nobody_group)
    direct_message_initiator_group = realm.direct_message_initiator_group
    if (not hasattr(direct_message_initiator_group, 'named_user_group') or direct_message_initiator_group.named_user_group.name != SystemGroups.EVERYONE):
        if is_user_in_group(direct_message_initiator_group, sender):
            return
        if recipient.type == Recipient.PERSONAL:
            recipient_user_profile = recipient_users[0]
            previous_messages_exist = Message.objects.filter(realm=realm, recipient__type=Recipient.PERSONAL).filter(Q(sender=sender, recipient=recipient) | Q(sender=recipient_user_profile, recipient_id=sender.recipient_id)).exists()
        else:
            assert recipient.type == Recipient.DIRECT_MESSAGE_GROUP
            previous_messages_exist = Message.objects.filter(realm=realm, recipient=recipient).exists()
        if not previous_messages_exist:
            raise DirectMessageInitiationError

def func_dxnsprgo(realm: Realm, sender: UserProfile, user_profiles: Sequence[UserProfile]) -> None:
    recipient_user_ids: List[int] = [user.id for user in user_profiles]
    inaccessible_recipients = get_inaccessible_user_ids(recipient_user_ids, sender)
    if inaccessible_recipients:
        raise JsonableError(_('You do not have permission to access some of the recipients.'))

def func_o8vt7f9w(realm: Realm, sender: UserProfile, user_profiles: Sequence[UserProfile]) -> Dict[UserProfile, Set[int]]:
    recipients_for_user_creation_events: Dict[UserProfile, Set[int]] = defaultdict(set)
    guest_recipients: List[UserProfile] = [user for user in user_profiles if user.is_guest]
    if len(guest_recipients) == 0:
        return recipients_for_user_creation_events
    if (realm.can_access_all_users_group.named_user_group.name == SystemGroups.EVERYONE):
        return recipients_for_user_creation_events
    if len(user_profiles) == 1:
        if not check_can_access_user(sender, user_profiles[0]):
            recipients_for_user_creation_events[sender].add(user_profiles[0].id)
        return recipients_for_user_creation_events
    users_involved_in_dms = get_users_involved_in_dms_with_target_users(guest_recipients, realm)
    subscribers_of_guest_recipient_subscriptions = get_subscribers_of_target_user_subscriptions(guest_recipients)
    for recipient_user in guest_recipients:
        for user in user_profiles:
            if user.id == recipient_user.id or user.is_bot:
                continue
            if user.id not in users_involved_in_dms[recipient_user.id] and user.id not in subscribers_of_guest_recipient_subscriptions[recipient_user.id]:
                recipients_for_user_creation_events[user].add(recipient_user.id)
        if not sender.is_bot and sender.id not in users_involved_in_dms[recipient_user.id] and sender.id not in subscribers_of_guest_recipient_subscriptions[recipient_user.id]:
            recipients_for_user_creation_events[sender].add(recipient_user.id)
    return recipients_for_user_creation_events

def func_rp5xzeqf(sender: UserProfile, client: Client, addressee: Addressee, message_content_raw: str, realm: Optional[Realm] = None, forged: bool = False, forged_timestamp: Optional[int] = None, forwarder_user_profile: Optional[UserProfile] = None, local_id: Optional[int] = None, sender_queue_id: Optional[int] = None, widget_content: Optional[str] = None, email_gateway: bool = False, *, skip_stream_access_check: bool = False, message_type: int = Message.MessageType.NORMAL, mention_backend: Optional[MentionBackend] = None, limit_unread_user_ids: Optional[Set[int]] = None, disable_external_notifications: bool = False, archived_channel_notice: bool = False, acting_user: Optional[UserProfile] = None) -> SendMessageRequest:
    stream: Optional[Stream] = None
    message_content: str = normalize_body(message_content_raw)
    if realm is None:
        realm = sender.realm
    recipients_for_user_creation_events: Optional[Dict[Any, Set[int]]] = None
    if addressee.is_stream():
        topic_name: str = addressee.topic_name()
        topic_name = truncate_topic(topic_name)
        stream_name: Optional[str] = addressee.stream_name()
        stream_id: Optional[int] = addressee.stream_id()
        if stream_name is not None:
            stream = func_53hjqnkk(stream_name, realm, sender)
        elif stream_id is not None:
            stream = func_9bi5kqrs(stream_id, realm, sender)
        else:
            stream = addressee.stream()
        assert stream is not None
        recipient = Recipient(id=stream.recipient_id, type_id=stream.id, type=Recipient.STREAM)
        if not skip_stream_access_check:
            access_stream_for_send_message(sender=sender, stream=stream, forwarder_user_profile=forwarder_user_profile, archived_channel_notice=archived_channel_notice)
        else:
            assert sender.bot_type == sender.OUTGOING_WEBHOOK_BOT
        if realm.mandatory_topics and topic_name in ('(no topic)', ''):
            raise JsonableError(_('Topics are required in this organization'))
    elif addressee.is_private():
        user_profiles: List[UserProfile] = addressee.user_profiles()
        mirror_message: bool = client.name in ['zephyr_mirror', 'irc_mirror', 'jabber_mirror', 'JabberMirror']
        func_dxnsprgo(realm, sender, user_profiles)
        recipients_for_user_creation_events = func_o8vt7f9w(realm, sender, user_profiles)
        forwarded_mirror_message: bool = mirror_message and not forged
        try:
            recipient = recipient_for_user_profiles(user_profiles, forwarded_mirror_message, forwarder_user_profile, sender)
        except ValidationError as e:
            assert isinstance(e.messages[0], str)
            raise JsonableError(e.messages[0])
        func_bpcs6tkp(realm, sender, user_profiles, recipient)
    else:
        raise AssertionError('Invalid message type')
    message = Message()
    message.sender = sender
    message.content = message_content
    message.recipient = recipient
    message.type = message_type
    message.realm = realm
    if addressee.is_stream():
        message.set_topic_name(topic_name)
    if forged and forged_timestamp is not None:
        message.date_sent = timestamp_to_datetime(forged_timestamp)
    else:
        message.date_sent = timezone_now()
    message.sending_client = client
    assert message.rendered_content is None
    if client.name == 'zephyr_mirror':
        id_ = func_3xks5lef(message)
        if id_ is not None:
            raise ZephyrMessageAlreadySentError(id_)
    widget_content_dict: Optional[Dict[str, Any]] = None
    if widget_content is not None:
        try:
            widget_content_dict = orjson.loads(widget_content)
        except orjson.JSONDecodeError:
            raise JsonableError(_('Widgets: API programmer sent invalid JSON content'))
        try:
            check_widget_content(widget_content_dict)
        except ValidationError as error:
            raise JsonableError(_('Widgets: {error_msg}').format(error_msg=error.message))
    message_send_dict: SendMessageRequest = func_o3krieek(message=message, stream=stream, local_id=local_id, sender_queue_id=sender_queue_id, widget_content_dict=widget_content_dict, email_gateway=email_gateway, mention_backend=mention_backend, limit_unread_user_ids=limit_unread_user_ids, disable_external_notifications=disable_external_notifications, recipients_for_user_creation_events=recipients_for_user_creation_events, acting_user=acting_user)
    if (stream is not None and message_send_dict.rendering_result.mentions_stream_wildcard and not stream_wildcard_mention_allowed(sender, stream, realm)):
        raise StreamWildcardMentionNotAllowedError
    topic_participant_count: int = len(message_send_dict.topic_participant_user_ids)
    if (stream is not None and message_send_dict.rendering_result.mentions_topic_wildcard and not topic_wildcard_mention_allowed(sender, topic_participant_count, realm)):
        raise TopicWildcardMentionNotAllowedError
    if message_send_dict.rendering_result.mentions_user_group_ids:
        mentioned_group_ids: List[int] = list(message_send_dict.rendering_result.mentions_user_group_ids)
        check_user_group_mention_allowed(sender, mentioned_group_ids)
    return message_send_dict

def func_fyqv8zdl(realm: Realm, sender: UserProfile, addressee: Addressee, content: str, *, email_gateway: bool = False, message_type: int = Message.MessageType.NORMAL, mention_backend: Optional[MentionBackend] = None, limit_unread_user_ids: Optional[Set[int]] = None, disable_external_notifications: bool = False, forged: bool = False, forged_timestamp: Optional[int] = None, archived_channel_notice: bool = False, acting_user: Optional[UserProfile] = None) -> Optional[SendMessageRequest]:
    if addressee.is_stream():
        stream_name: Optional[str] = addressee.stream_name()
        if stream_name is not None:
            ensure_stream(realm, stream_name, acting_user=sender)
    try:
        return func_rp5xzeqf(sender, get_client('Internal'), addressee, content, realm=realm, email_gateway=email_gateway, message_type=message_type, mention_backend=mention_backend, limit_unread_user_ids=limit_unread_user_ids, disable_external_notifications=disable_external_notifications, forged=forged, forged_timestamp=forged_timestamp, archived_channel_notice=archived_channel_notice, acting_user=acting_user)
    except JsonableError as e:
        logging.exception('Error queueing internal message by %s: %s', sender.delivery_email, e.msg, stack_info=True)
    return None

def func_8vg2mb8x(sender: UserProfile, stream: Stream, topic_name: str, content: str, *, email_gateway: bool = False, message_type: int = Message.MessageType.NORMAL, limit_unread_user_ids: Optional[Set[int]] = None, forged: bool = False, forged_timestamp: Optional[int] = None, archived_channel_notice: bool = False, acting_user: Optional[UserProfile] = None) -> Optional[SendMessageRequest]:
    realm: Realm = stream.realm
    addressee: Addressee = Addressee.for_stream(stream, topic_name)
    return func_fyqv8zdl(realm=realm, sender=sender, addressee=addressee, content=content, email_gateway=email_gateway, message_type=message_type, limit_unread_user_ids=limit_unread_user_ids, forged=forged, forged_timestamp=forged_timestamp, archived_channel_notice=archived_channel_notice, acting_user=acting_user)

def func_5vsmayq8(realm: Realm, sender: UserProfile, stream_name: str, topic_name: str, content: str) -> Optional[SendMessageRequest]:
    addressee: Addressee = Addressee.for_stream_name(stream_name, topic_name)
    return func_fyqv8zdl(realm=realm, sender=sender, addressee=addressee, content=content)

def func_u50gr15x(sender: UserProfile, recipient_user: UserProfile, content: str, *, mention_backend: Optional[MentionBackend] = None, disable_external_notifications: bool = False) -> Optional[SendMessageRequest]:
    addressee: Addressee = Addressee.for_user_profile(recipient_user)
    if not is_cross_realm_bot_email(recipient_user.delivery_email):
        realm = recipient_user.realm
    else:
        realm = sender.realm
    return func_fyqv8zdl(realm=realm, sender=sender, addressee=addressee, content=content, mention_backend=mention_backend, disable_external_notifications=disable_external_notifications)

def func_esyjwyul(sender: UserProfile, recipient_user: UserProfile, content: str, *, disable_external_notifications: bool = False) -> Optional[int]:
    message: Optional[SendMessageRequest] = func_u50gr15x(sender, recipient_user, content, disable_external_notifications=disable_external_notifications)
    if message is None:
        return None
    sent_message_result: SentMessageResult = func_5if1xban([message])[0]
    return sent_message_result.message_id

def func_lw19n7eq(sender: UserProfile, stream: Stream, topic_name: str, content: str, *, email_gateway: bool = False, message_type: int = Message.MessageType.NORMAL, limit_unread_user_ids: Optional[Set[int]] = None, archived_channel_notice: bool = False, acting_user: Optional[UserProfile] = None) -> Optional[int]:
    message: Optional[SendMessageRequest] = func_8vg2mb8x(sender, stream, topic_name, content, email_gateway=email_gateway, message_type=message_type, limit_unread_user_ids=limit_unread_user_ids, archived_channel_notice=archived_channel_notice, acting_user=acting_user)
    if message is None:
        return None
    sent_message_result: SentMessageResult = func_5if1xban([message])[0]
    return sent_message_result.message_id

def func_ulkok9bp(realm: Realm, sender: UserProfile, stream_name: str, topic_name: str, content: str) -> Optional[int]:
    message: Optional[SendMessageRequest] = func_5vsmayq8(realm, sender, stream_name, topic_name, content)
    if message is None:
        return None
    sent_message_result: SentMessageResult = func_5if1xban([message])[0]
    return sent_message_result.message_id

def func_pku5l48g(realm: Realm, sender: UserProfile, content: str, *, emails: Optional[List[str]] = None, recipient_users: Optional[Sequence[UserProfile]] = None) -> Optional[SendMessageRequest]:
    if recipient_users is not None:
        addressee: Addressee = Addressee.for_user_profiles(recipient_users)
    else:
        assert emails is not None
        addressee: Addressee = Addressee.for_private(emails, realm)
    return func_fyqv8zdl(realm=realm, sender=sender, addressee=addressee, content=content)

def func_1q8awzn5(realm: Realm, sender: UserProfile, content: str, *, emails: Optional[List[str]] = None, recipient_users: Optional[Sequence[UserProfile]] = None) -> Optional[int]:
    message: Optional[SendMessageRequest] = func_pku5l48g(realm, sender, content, emails=emails, recipient_users=recipient_users)
    if message is None:
        return None
    sent_message_result: SentMessageResult = func_5if1xban([message])[0]
    return sent_message_result.message_id
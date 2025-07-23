import itertools
from collections import defaultdict
from collections.abc import Iterable, Set as AbstractSet
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from django.conf import settings
from django.db import transaction
from django.db.models import Q, QuerySet
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy
from django.utils.translation import override as override_language
from django_stubs_ext import StrPromise
from zerver.actions.message_delete import DeleteMessagesEvent, do_delete_messages
from zerver.actions.message_flags import do_update_mobile_push_notification
from zerver.actions.message_send import filter_presence_idle_user_ids, get_recipient_info, internal_send_stream_message, render_incoming_message
from zerver.actions.uploads import AttachmentChangeResult, check_attachment_reference_change
from zerver.actions.user_topics import bulk_do_set_user_topic_visibility_policy
from zerver.lib.exceptions import JsonableError, MessageMoveError, StreamWildcardMentionNotAllowedError, TopicWildcardMentionNotAllowedError
from zerver.lib.markdown import MessageRenderingResult, topic_links
from zerver.lib.markdown import version as markdown_version
from zerver.lib.mention import MentionBackend, MentionData, silent_mention_syntax_for_user
from zerver.lib.message import access_message, bulk_access_stream_messages_query, check_user_group_mention_allowed, event_recipient_ids_for_action_on_messages, normalize_body, stream_wildcard_mention_allowed, topic_wildcard_mention_allowed, truncate_topic
from zerver.lib.message_cache import update_message_cache
from zerver.lib.queue import queue_event_on_commit
from zerver.lib.stream_subscription import get_active_subscriptions_for_stream_id
from zerver.lib.stream_topic import StreamTopicTarget
from zerver.lib.streams import access_stream_by_id, access_stream_by_id_for_message, can_access_stream_history, check_stream_access_based_on_can_send_message_group, notify_stream_is_recently_active_update
from zerver.lib.string_validation import check_stream_topic
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.topic import ORIG_TOPIC, RESOLVED_TOPIC_PREFIX, TOPIC_LINKS, TOPIC_NAME, maybe_rename_general_chat_to_empty_topic, messages_for_topic, participants_for_topic, save_message_for_edit_use_case, update_edit_history, update_messages_for_topic_edit
from zerver.lib.types import DirectMessageEditRequest, EditHistoryEvent, StreamMessageEditRequest
from zerver.lib.url_encoding import near_stream_message_url
from zerver.lib.user_message import bulk_insert_all_ums
from zerver.lib.user_topics import get_users_with_user_topic_visibility_policy
from zerver.lib.widget import is_widget_message
from zerver.models import ArchivedAttachment, Attachment, Message, Reaction, Recipient, Stream, Subscription, UserMessage, UserProfile, UserTopic
from zerver.models.streams import get_stream_by_id_in_realm
from zerver.models.users import get_system_bot
from zerver.tornado.django_api import send_event_on_commit

@dataclass
class UpdateMessageResult:
    changed_messages_count: int
    detached_attachments: List[Attachment]

def subscriber_info(user_id: int) -> Dict[str, Any]:
    return {'id': user_id, 'flags': ['read']}

def validate_message_edit_payload(message: Message, stream_id: Optional[int], topic_name: Optional[str], propagate_mode: str, content: Optional[str]) -> None:
    """
    Checks that the data sent is well-formed. Does not handle editability, permissions etc.
    """
    if topic_name is None and content is None and (stream_id is None):
        raise JsonableError(_('Nothing to change'))
    if not message.is_stream_message():
        if stream_id is not None:
            raise JsonableError(_('Direct messages cannot be moved to channels.'))
        if topic_name is not None:
            raise JsonableError(_('Direct messages cannot have topics.'))
    if propagate_mode != 'change_one' and topic_name is None and (stream_id is None):
        raise JsonableError(_('Invalid propagate_mode without topic edit'))
    if message.realm.mandatory_topics and topic_name in ('(no topic)', ''):
        raise JsonableError(_('Topics are required in this organization.'))
    if topic_name in {RESOLVED_TOPIC_PREFIX.strip(), f'{RESOLVED_TOPIC_PREFIX}{Message.EMPTY_TOPIC_FALLBACK_NAME}'}:
        raise JsonableError(_('General chat cannot be marked as resolved'))
    if topic_name is not None:
        check_stream_topic(topic_name)
    if stream_id is not None and content is not None:
        raise JsonableError(_('Cannot change message content while changing channel'))
    if content is not None and is_widget_message(message):
        raise JsonableError(_('Widgets cannot be edited.'))

def validate_user_can_edit_message(user_profile: UserProfile, message: Message, edit_limit_buffer: int) -> None:
    """
    Checks if the user has the permission to edit the message.
    """
    if not user_profile.realm.allow_message_editing:
        raise JsonableError(_('Your organization has turned off message editing'))
    if message.sender_id != user_profile.id:
        raise JsonableError(_("You don't have permission to edit this message"))
    if user_profile.realm.message_content_edit_limit_seconds is not None:
        deadline_seconds = user_profile.realm.message_content_edit_limit_seconds + edit_limit_buffer
        if timezone_now() - message.date_sent > timedelta(seconds=deadline_seconds):
            raise JsonableError(_('The time limit for editing this message has passed'))

def maybe_send_resolve_topic_notifications(*, user_profile: UserProfile, message_edit_request: StreamMessageEditRequest, changed_messages: QuerySet[Message]) -> Tuple[Optional[int], bool]:
    """Returns resolved_topic_message_id if resolve topic notifications were in fact sent."""
    topic_resolved = message_edit_request.topic_resolved
    topic_unresolved = message_edit_request.topic_unresolved
    if not topic_resolved and (not topic_unresolved):
        return (None, False)
    stream = message_edit_request.orig_stream
    if maybe_delete_previous_resolve_topic_notification(user_profile, stream, message_edit_request.target_topic_name):
        return (None, True)
    affected_participant_ids = set(changed_messages.values_list('sender_id', flat=True).union(Reaction.objects.filter(message__in=changed_messages).values_list('user_profile_id', flat=True)))
    sender = get_system_bot(settings.NOTIFICATION_BOT, user_profile.realm_id)
    user_mention = silent_mention_syntax_for_user(user_profile)
    with override_language(stream.realm.default_language):
        if topic_resolved:
            notification_string = _('{user} has marked this topic as resolved.')
        elif topic_unresolved:
            notification_string = _('{user} has marked this topic as unresolved.')
        resolved_topic_message_id = internal_send_stream_message(sender, stream, message_edit_request.target_topic_name, notification_string.format(user=user_mention), message_type=Message.MessageType.RESOLVE_TOPIC_NOTIFICATION, limit_unread_user_ids=affected_participant_ids, acting_user=user_profile)
    return (resolved_topic_message_id, False)

def maybe_delete_previous_resolve_topic_notification(user_profile: UserProfile, stream: Stream, topic: str) -> bool:
    assert stream.recipient_id is not None
    last_message = messages_for_topic(stream.realm_id, stream.recipient_id, topic).last()
    if last_message is None:
        return False
    if last_message.type != Message.MessageType.RESOLVE_TOPIC_NOTIFICATION:
        return False
    current_time = timezone_now()
    time_difference = (current_time - last_message.date_sent).total_seconds()
    if time_difference > settings.RESOLVE_TOPIC_UNDO_GRACE_PERIOD_SECONDS:
        return False
    do_delete_messages(stream.realm, [last_message], acting_user=user_profile)
    return True

def send_message_moved_breadcrumbs(target_message: Message, user_profile: UserProfile, message_edit_request: StreamMessageEditRequest, old_thread_notification_string: Optional[str], new_thread_notification_string: Optional[str], changed_messages_count: int) -> None:
    old_stream = message_edit_request.orig_stream
    sender = get_system_bot(settings.NOTIFICATION_BOT, old_stream.realm_id)
    user_mention = silent_mention_syntax_for_user(user_profile)
    old_topic_name = message_edit_request.orig_topic_name
    new_stream = message_edit_request.target_stream
    new_topic_name = message_edit_request.target_topic_name
    old_topic_link = f'#**{old_stream.name}>{old_topic_name}**'
    new_topic_link = f'#**{new_stream.name}>{new_topic_name}**'
    message = {'id': target_message.id, 'stream_id': new_stream.id, 'display_recipient': new_stream.name, 'topic': new_topic_name}
    moved_message_link = near_stream_message_url(target_message.realm, message)
    if new_thread_notification_string is not None:
        with override_language(new_stream.realm.default_language):
            internal_send_stream_message(sender, new_stream, new_topic_name, new_thread_notification_string.format(message_link=moved_message_link, old_location=old_topic_link, user=user_mention, changed_messages_count=changed_messages_count), acting_user=user_profile)
    if old_thread_notification_string is not None:
        with override_language(old_stream.realm.default_language):
            internal_send_stream_message(sender, old_stream, old_topic_name, old_thread_notification_string.format(user=user_mention, new_location=new_topic_link, changed_messages_count=changed_messages_count), acting_user=user_profile)

def get_mentions_for_message_updates(message: Message) -> Set[int]:
    mentioned_user_ids = UserMessage.objects.filter(message=message.id, flags=~UserMessage.flags.historical).filter(Q(flags__andnz=UserMessage.flags.mentioned | UserMessage.flags.stream_wildcard_mentioned | UserMessage.flags.topic_wildcard_mentioned | UserMessage.flags.group_mentioned)).values_list('user_profile_id', flat=True)
    user_ids_having_message_access = event_recipient_ids_for_action_on_messages([message])
    return set(mentioned_user_ids) & user_ids_having_message_access

def update_user_message_flags(rendering_result: MessageRenderingResult, ums: Iterable[UserMessage], topic_participant_user_ids: Set[int] = set()) -> None:
    mentioned_ids = rendering_result.mentions_user_ids
    ids_with_alert_words = rendering_result.user_ids_with_alert_words
    changed_ums = set()

    def update_flag(um: UserMessage, should_set: bool, flag: int) -> None:
        if should_set:
            if not um.flags & flag:
                um.flags |= flag
                changed_ums.add(um)
        elif um.flags & flag:
            um.flags &= ~flag
            changed_ums.add(um)
    for um in ums:
        has_alert_word = um.user_profile_id in ids_with_alert_words
        update_flag(um, has_alert_word, UserMessage.flags.has_alert_word)
        mentioned = um.user_profile_id in mentioned_ids
        update_flag(um, mentioned, UserMessage.flags.mentioned)
        if rendering_result.mentions_stream_wildcard:
            update_flag(um, True, UserMessage.flags.stream_wildcard_mentioned)
        elif rendering_result.mentions_topic_wildcard:
            topic_wildcard_mentioned = um.user_profile_id in topic_participant_user_ids
            update_flag(um, topic_wildcard_mentioned, UserMessage.flags.topic_wildcard_mentioned)
    for um in changed_ums:
        um.save(update_fields=['flags'])

def do_update_embedded_data(user_profile: UserProfile, message: Message, rendered_content: Union[str, MessageRenderingResult]) -> None:
    ums = UserMessage.objects.filter(message=message.id)
    update_fields = ['rendered_content']
    if isinstance(rendered_content, MessageRenderingResult):
        update_user_message_flags(rendered_content, ums)
        message.rendered_content = rendered_content.rendered_content
        message.rendered_content_version = markdown_version
        update_fields.append('rendered_content_version')
    else:
        message.rendered_content = rendered_content
    message.save(update_fields=update_fields)
    update_message_cache([message])
    event = {'type': 'update_message', 'user_id': None, 'edit_timestamp': datetime_to_timestamp(timezone_now()), 'message_id': message.id, 'message_ids': [message.id], 'content': message.content, 'rendered_content': message.rendered_content, 'rendering_only': True}
    users_to_notify = event_recipient_ids_for_action_on_messages([message])
    filtered_ums = [um for um in ums if um.user_profile_id in users_to_notify]

    def user_info(um: UserMessage) -> Dict[str, Any]:
        return {'id': um.user_profile_id, 'flags': um.flags_list()}
    send_event_on_commit(user_profile.realm, event, list(map(user_info, filtered_ums)))

def get_visibility_policy_after_merge(orig_topic_visibility_policy: int, target_topic_visibility_policy: int) -> int:
    if orig_topic_visibility_policy == target_topic_visibility_policy:
        return orig_topic_visibility_policy
    elif UserTopic.VisibilityPolicy.UNMUTED in (orig_topic_visibility_policy, target_topic_visibility_policy):
        return UserTopic.VisibilityPolicy.UNMUTED
    return UserTopic.VisibilityPolicy.INHERIT

def update_message_content(user_profile: UserProfile, target_message: Message, content: str, rendering_result: MessageRenderingResult, prior_mention_user_ids: Set[int], mention_data: MentionData, event: Dict[str, Any], edit_history_event: Dict[str, Any], stream_topic: Optional[StreamTopicTarget]) -> None:
    realm = user_profile.realm
    ums = UserMessage.objects.filter(message=target_message.id)
    for group_id in rendering_result.mentions_user_group_ids:
        members = mention_data.get_group_members(group_id)
        rendering_result.mentions_user_ids.update(members)
    edit_history_event['prev_content'] = target_message.content
    edit_history_event['prev_rendered_content'] = target_message.rendered_content
    edit_history_event['prev_rendered_content_version'] = target_message.rendered_content_version
    event['orig_content'] = target_message.content
    event['orig_rendered_content'] = target_message.rendered_content
    event['content'] = content
    event['rendered_content'] = rendering_result.rendered_content
    event['is_me_message'] = Message.is_status_message(content, rendering_result.rendered_content)
    target_message.content = content
    target_message.rendered_content = rendering_result.rendered_content
    target_message.rendered_content_version = markdown_version
    info = get_recipient_info(realm_id=realm.id, recipient=target_message.recipient, sender_id=target_message.sender_id, stream_topic=stream_topic, possible_topic_wildcard_mention=mention_data.message_has_topic_wildcards(), possible_stream_wildcard_mention=mention_data.message_has_stream_wildcards())
    event['online_push_user_ids'] = list(info.online_push_user_ids)
    event['dm_mention_push_disabled_user_ids'] = list(info.dm_mention_push_disabled_user_ids)
    event['dm_mention_email_disabled_user_ids'] = list(info.dm_mention_email_disabled_user_ids)
    event['stream_push_user_ids'] = list(info.stream_push_user_ids)
    event['stream_email_user_ids'] = list(info.stream_email_user_ids)
    event['followed_topic_push_user_ids'] = list(info.followed_topic_push_user_ids)
    event['followed_topic_email_user_ids'] = list(info.followed_topic_email_user_ids)
    event['muted_sender_user_ids'] = list(info.muted_sender_user_ids)
    event['prior_mention_user_ids'] = list(prior_mention_user_ids)
    event['presence_idle_user_ids'] = filter_presence_idle_user_ids(info.active_user_ids)
    event['all_bot_user_ids'] = list(info.all_bot_user_ids)
    if rendering_result.mentions_stream_wildcard:
        event['stream_wildcard_mention_user_ids'] = list(info.stream_wildcard_mention_user_ids)
        event['stream_wildcard_mention_in_followed_topic_user_ids'] = list(info.stream_wildcard_mention_in_followed_topic_user_ids)
    else:
        event['stream_wildcard_mention_user_ids'] = []
        event['stream_wildcard_mention_in_followed_topic_user_ids'] = []
    if rendering_result.mentions_topic_wildcard:
        event['topic_wildcard_mention_user_ids'] = list(info.topic_wildcard_mention_user_ids)
        event['topic_wildcard_mention_in_followed_topic_user_ids'] = list(info.topic_wildcard_mention_in_followed_topic_user_ids)
        topic_participant_user_ids = info.topic_participant_user_ids
    else:
        event['topic_wildcard_mention_user_ids'] = []
        event['topic_wildcard_mention_in_followed_topic_user_ids'] = []
        topic_participant_user_ids = set()
    update_user_message_flags(rendering_result, ums, topic_participant_user_ids)
    do_update_mobile_push_notification(target_message, prior_mention_user_ids, rendering_result.mentions_user_ids, info.stream_push_user_ids)

@transaction.atomic(savepoint=False)
def do_update_message(user_profile
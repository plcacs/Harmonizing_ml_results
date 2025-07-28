import itertools
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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
    changed_messages_count: int = 0
    detached_attachments: List[Attachment] = None

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

def send_message_moved_breadcrumbs(target_message: Message, user_profile: UserProfile, message_edit_request: StreamMessageEditRequest, old_thread_notification_string: Optional[StrPromise], new_thread_notification_string: Optional[StrPromise], changed_messages_count: int) -> None:
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

def update_user_message_flags(rendering_result: MessageRenderingResult, ums: List[UserMessage], topic_participant_user_ids: Set[int] = set()) -> None:
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
def do_update_message(user_profile: UserProfile, target_message: Message, message_edit_request: Union[StreamMessageEditRequest, DirectMessageEditRequest], send_notification_to_old_thread: bool, send_notification_to_new_thread: bool, rendering_result: Optional[MessageRenderingResult], prior_mention_user_ids: Set[int], mention_data: Optional[MentionData] = None) -> UpdateMessageResult:
    """
    The main function for message editing.  A message edit event can
    modify:
    * the message's content (in which case the caller will have set
      both content and rendered_content in message_edit_request object),
    * the topic, in which case the caller will have set target_topic_name
      field with the new topic name in message_edit_request object
    * or both message's content and the topic
    * or stream and/or topic, in which case the caller will have set
      target_stream and/or target_topic_name to their new values in
      message_edit_request object.

    With topic edits, propagate_mode field in message_edit_request
    determines whether other message also have their topics edited.
    """
    timestamp = timezone_now()
    target_message.last_edit_time = timestamp
    event: Dict[str, Any] = {'type': 'update_message', 'user_id': user_profile.id, 'edit_timestamp': datetime_to_timestamp(timestamp), 'message_id': target_message.id, 'rendering_only': False}
    edit_history_event: Dict[str, Any] = {'user_id': user_profile.id, 'timestamp': event['edit_timestamp']}
    realm = user_profile.realm
    attachment_reference_change = AttachmentChangeResult(False, [])
    ums = UserMessage.objects.filter(message=target_message.id)

    def user_info(um: UserMessage) -> Dict[str, Any]:
        return {'id': um.user_profile_id, 'flags': um.flags_list()}
    if message_edit_request.is_content_edited:
        assert rendering_result is not None
        assert mention_data is not None
        if isinstance(message_edit_request, StreamMessageEditRequest):
            stream_topic = StreamTopicTarget(stream_id=message_edit_request.orig_stream.id, topic_name=message_edit_request.target_topic_name)
        else:
            stream_topic = None
        update_message_content(user_profile, target_message, message_edit_request.content, rendering_result, prior_mention_user_ids, mention_data, event, edit_history_event, stream_topic)
        attachment_reference_change = check_attachment_reference_change(target_message, rendering_result)
        target_message.has_attachment = attachment_reference_change.did_attachment_change
        if isinstance(message_edit_request, DirectMessageEditRequest):
            update_edit_history(target_message, timestamp, edit_history_event)
            save_message_for_edit_use_case(message=target_message)
            event['message_ids'] = update_message_cache([target_message])
            users_to_be_notified = list(map(user_info, ums))
            send_event_on_commit(user_profile.realm, event, users_to_be_notified)
            changed_messages_count = 1
            return UpdateMessageResult(changed_messages_count, attachment_reference_change.detached_attachments)
    assert isinstance(message_edit_request, StreamMessageEditRequest)
    stream_being_edited = message_edit_request.orig_stream
    orig_topic_name = message_edit_request.orig_topic_name
    event['stream_name'] = stream_being_edited.name
    event['stream_id'] = stream_being_edited.id
    if message_edit_request.is_message_moved:
        event['propagate_mode'] = message_edit_request.propagate_mode
    users_losing_access: QuerySet[UserProfile] = UserProfile.objects.none()
    user_ids_gaining_usermessages: List[int] = []
    if message_edit_request.is_stream_edited:
        new_stream = message_edit_request.target_stream
        edit_history_event['prev_stream'] = stream_being_edited.id
        edit_history_event['stream'] = new_stream.id
        event[ORIG_TOPIC] = orig_topic_name
        assert new_stream.recipient_id is not None
        target_message.recipient_id = new_stream.recipient_id
        event['new_stream_id'] = new_stream.id
        event['propagate_mode'] = message_edit_request.propagate_mode
        old_stream_all_users = UserProfile.objects.filter(id__in=Subscription.objects.filter(recipient__type=Recipient.STREAM, recipient__type_id=stream_being_edited.id).values_list('user_profile_id')).only('id')
        new_stream_current_users = UserProfile.objects.filter(id__in=get_active_subscriptions_for_stream_id(new_stream.id, include_deactivated_users=True).values_list('user_profile_id')).only('id')
        users_losing_usermessages = old_stream_all_users.difference(new_stream_current_users)
        if new_stream.is_public():
            users_losing_access = old_stream_all_users.filter(role=UserProfile.ROLE_GUEST).difference(new_stream_current_users)
        else:
            users_losing_access = users_losing_usermessages
        unmodified_user_messages = ums.exclude(user_profile__in=users_losing_usermessages)
        if not new_stream.is_history_public_to_subscribers():
            user_ids_gaining_usermessages = list(new_stream_current_users.values_list('id', flat=True))
    else:
        unmodified_user_messages = ums
    if message_edit_request.is_topic_edited:
        topic_name = message_edit_request.target_topic_name
        target_message.set_topic_name(topic_name)
        event[ORIG_TOPIC] = orig_topic_name
        event[TOPIC_NAME] = topic_name
        event[TOPIC_LINKS] = topic_links(target_message.realm_id, topic_name)
        edit_history_event['prev_topic'] = orig_topic_name
        edit_history_event['topic'] = topic_name
    update_edit_history(target_message, timestamp, edit_history_event)
    if message_edit_request.is_message_moved:
        target_stream = message_edit_request.target_stream
        target_topic_name = message_edit_request.target_topic_name
        assert target_stream.recipient_id is not None
        target_topic_has_messages = messages_for_topic(realm.id, target_stream.recipient_id, target_topic_name).exists()
    changed_messages = Message.objects.filter(id=target_message.id)
    changed_message_ids = [target_message.id]
    changed_messages_count = 1
    save_changes_for_propagation_mode = lambda: Message.objects.filter(id=target_message.id).select_related(*Message.DEFAULT_SELECT_RELATED)
    if message_edit_request.propagate_mode in ['change_later', 'change_all']:
        topic_only_edit_history_event: Dict[str, Any] = {'user_id': edit_history_event['user_id'], 'timestamp': edit_history_event['timestamp']}
        if message_edit_request.is_topic_edited:
            topic_only_edit_history_event['prev_topic'] = edit_history_event['prev_topic']
            topic_only_edit_history_event['topic'] = edit_history_event['topic']
        if message_edit_request.is_stream_edited:
            topic_only_edit_history_event['prev_stream'] = edit_history_event['prev_stream']
            topic_only_edit_history_event['stream'] = edit_history_event['stream']
        later_messages, save_changes_for_propagation_mode = update_messages_for_topic_edit(acting_user=user_profile, edited_message=target_message, message_edit_request=message_edit_request, edit_history_event=topic_only_edit_history_event, last_edit_time=timestamp)
        changed_messages |= later_messages
        changed_message_ids = list(changed_messages.values_list('id', flat=True))
        changed_messages_count = len(changed_message_ids)
    if message_edit_request.is_stream_edited:
        bulk_insert_all_ums(user_ids_gaining_usermessages, changed_message_ids, UserMessage.flags.read)
        UserMessage.objects.filter(user_profile__in=users_losing_usermessages, message__in=changed_messages).delete()
        delete_event: Dict[str, Any] = {'type': 'delete_message', 'message_ids': changed_message_ids, 'message_type': 'stream', 'stream_id': stream_being_edited.id, 'topic': orig_topic_name}
        send_event_on_commit(user_profile.realm, delete_event, [user.id for user in users_losing_access])
        if message_edit_request.target_stream.invite_only != stream_being_edited.invite_only:
            Attachment.objects.filter(messages__in=changed_messages.values('id')).update(is_realm_public=None)
            ArchivedAttachment.objects.filter(messages__in=changed_messages.values('id')).update(is_realm_public=None)
        if message_edit_request.target_stream.is_web_public != stream_being_edited.is_web_public:
            Attachment.objects.filter(messages__in=changed_messages.values('id')).update(is_web_public=None)
            ArchivedAttachment.objects.filter(messages__in=changed_messages.values('id')).update(is_web_public=None)
    save_message_for_edit_use_case(message=target_message)
    changed_messages = save_changes_for_propagation_mode()
    realm_id = target_message.realm_id
    event['message_ids'] = update_message_cache(changed_messages, realm_id)
    users_to_be_notified = list(map(user_info, unmodified_user_messages))
    if stream_being_edited.is_history_public_to_subscribers():
        subscriptions = get_active_subscriptions_for_stream_id(message_edit_request.target_stream.id, include_deactivated_users=False)
        subscriptions = subscriptions.exclude(user_profile__long_term_idle=True)
        subscriptions = subscriptions.exclude(user_profile_id__in=[um.user_profile_id for um in unmodified_user_messages])
        if message_edit_request.is_stream_edited:
            subscriptions = subscriptions.exclude(user_profile__in=users_losing_access)
            old_stream_current_users = UserProfile.objects.filter(id__in=get_active_subscriptions_for_stream_id(stream_being_edited.id, include_deactivated_users=True).values_list('user_profile_id', flat=True)).only('id')
            subscriptions = subscriptions.exclude(user_profile__in=new_stream_current_users.filter(role=UserProfile.ROLE_GUEST).difference(old_stream_current_users))
        subscriber_ids = set(subscriptions.values_list('user_profile_id', flat=True))
        users_to_be_notified += map(subscriber_info, sorted(subscriber_ids))
    moved_all_visible_messages = False
    if message_edit_request.is_message_moved:
        if message_edit_request.propagate_mode == 'change_all':
            moved_all_visible_messages = True
        else:
            assert stream_being_edited.recipient_id is not None
            unmoved_messages = messages_for_topic(realm.id, stream_being_edited.recipient_id, orig_topic_name)
            visible_unmoved_messages = bulk_access_stream_messages_query(user_profile, unmoved_messages, stream_being_edited)
            moved_all_visible_messages = not visible_unmoved_messages.exists()
    if moved_all_visible_messages:
        stream_inaccessible_to_user_profiles: List[UserProfile] = []
        orig_topic_user_profile_to_visibility_policy: Dict[UserProfile, int] = {}
        target_topic_user_profile_to_visibility_policy: Dict[UserProfile, int] = {}
        user_ids_losing_access = {user.id for user in users_losing_access}
        for user_topic in get_users_with_user_topic_visibility_policy(stream_being_edited.id, orig_topic_name):
            if message_edit_request.is_stream_edited and user_topic.user_profile_id in user_ids_losing_access:
                stream_inaccessible_to_user_profiles.append(user_topic.user_profile)
            else:
                orig_topic_user_profile_to_visibility_policy[user_topic.user_profile] = user_topic.visibility_policy
        for user_topic in get_users_with_user_topic_visibility_policy(target_stream.id, target_topic_name):
            target_topic_user_profile_to_visibility_policy[user_topic.user_profile] = user_topic.visibility_policy
        user_profiles_having_visibility_policy = set(itertools.chain(orig_topic_user_profile_to_visibility_policy.keys(), target_topic_user_profile_to_visibility_policy.keys()))
        user_profiles_for_visibility_policy_pair: Dict[Tuple[int, int], List[UserProfile]] = defaultdict(list)
        for user_profile_with_policy in user_profiles_having_visibility_policy:
            if user_profile_with_policy not in target_topic_user_profile_to_visibility_policy:
                target_topic_user_profile_to_visibility_policy[user_profile_with_policy] = UserTopic.VisibilityPolicy.INHERIT
            elif user_profile_with_policy not in orig_topic_user_profile_to_visibility_policy:
                orig_topic_user_profile_to_visibility_policy[user_profile_with_policy] = UserTopic.VisibilityPolicy.INHERIT
            orig_topic_visibility_policy = orig_topic_user_profile_to_visibility_policy[user_profile_with_policy]
            target_topic_visibility_policy = target_topic_user_profile_to_visibility_policy[user_profile_with_policy]
            user_profiles_for_visibility_policy_pair[orig_topic_visibility_policy, target_topic_visibility_policy].append(user_profile_with_policy)
        bulk_do_set_user_topic_visibility_policy(stream_inaccessible_to_user_profiles, stream_being_edited, orig_topic_name, visibility_policy=UserTopic.VisibilityPolicy.INHERIT)
        for visibility_policy_pair, user_profiles in user_profiles_for_visibility_policy_pair.items():
            orig_topic_visibility_policy, target_topic_visibility_policy = visibility_policy_pair
            if orig_topic_visibility_policy != UserTopic.VisibilityPolicy.INHERIT:
                bulk_do_set_user_topic_visibility_policy(user_profiles, stream_being_edited, orig_topic_name, visibility_policy=UserTopic.VisibilityPolicy.INHERIT, skip_muted_topics_event=True)
            new_visibility_policy = orig_topic_visibility_policy
            if target_topic_has_messages:
                new_visibility_policy = get_visibility_policy_after_merge(orig_topic_visibility_policy, target_topic_visibility_policy)
                if new_visibility_policy == target_topic_visibility_policy:
                    continue
                bulk_do_set_user_topic_visibility_policy(user_profiles, target_stream, target_topic_name, visibility_policy=new_visibility_policy)
            else:
                if new_visibility_policy == target_topic_visibility_policy:
                    continue
                bulk_do_set_user_topic_visibility_policy(user_profiles, target_stream, target_topic_name, visibility_policy=new_visibility_policy)
    send_event_on_commit(user_profile.realm, event, users_to_be_notified)
    resolved_topic_message_id = None
    resolved_topic_message_deleted = False
    if message_edit_request.is_topic_edited and (not message_edit_request.is_content_edited) and (not message_edit_request.is_stream_edited):
        resolved_topic_message_id, resolved_topic_message_deleted = maybe_send_resolve_topic_notifications(user_profile=user_profile, message_edit_request=message_edit_request, changed_messages=changed_messages)
    if message_edit_request.is_message_moved:
        old_thread_notification_string = None
        if send_notification_to_old_thread:
            if moved_all_visible_messages:
                old_thread_notification_string = gettext_lazy('This topic was moved to {new_location} by {user}.')
            elif changed_messages_count == 1:
                old_thread_notification_string = gettext_lazy('A message was moved from this topic to {new_location} by {user}.')
            else:
                old_thread_notification_string = gettext_lazy('{changed_messages_count} messages were moved from this topic to {new_location} by {user}.')
        new_thread_notification_string = None
        if send_notification_to_new_thread and (message_edit_request.is_stream_edited or (message_edit_request.is_topic_edited and (not message_edit_request.topic_resolved) and (not message_edit_request.topic_unresolved))):
            stream_for_new_topic = message_edit_request.target_stream
            assert stream_for_new_topic.recipient_id is not None
            new_topic_name = message_edit_request.target_topic_name
            preexisting_topic_messages = messages_for_topic(realm.id, stream_for_new_topic.recipient_id, new_topic_name).exclude(id__in=[*changed_message_ids, resolved_topic_message_id])
            visible_preexisting_messages = bulk_access_stream_messages_query(user_profile, preexisting_topic_messages, stream_for_new_topic)
            no_visible_preexisting_messages = not visible_preexisting_messages.exists()
            if no_visible_preexisting_messages and moved_all_visible_messages:
                new_thread_notification_string = gettext_lazy('This topic was moved here from {old_location} by {user}.')
            elif changed_messages_count == 1:
                new_thread_notification_string = gettext_lazy('[A message]({message_link}) was moved here from {old_location} by {user}.')
            else:
                new_thread_notification_string = gettext_lazy('{changed_messages_count} messages were moved here from {old_location} by {user}.')
        send_message_moved_breadcrumbs(target_message, user_profile, message_edit_request, old_thread_notification_string, new_thread_notification_string, changed_messages_count)
    return UpdateMessageResult(changed_messages_count, attachment_reference_change.detached_attachments)

def check_time_limit_for_change_all_propagate_mode(message: Message, user_profile: UserProfile, topic_name: Optional[str] = None, stream_id: Optional[int] = None) -> None:
    realm = user_profile.realm
    message_move_limit_buffer = 20
    topic_edit_deadline_seconds = None
    if topic_name is not None and realm.move_messages_within_stream_limit_seconds is not None:
        topic_edit_deadline_seconds = realm.move_messages_within_stream_limit_seconds + message_move_limit_buffer
    stream_edit_deadline_seconds = None
    if stream_id is not None and realm.move_messages_between_streams_limit_seconds is not None:
        stream_edit_deadline_seconds = realm.move_messages_between_streams_limit_seconds + message_move_limit_buffer
    if topic_edit_deadline_seconds is not None and stream_edit_deadline_seconds is not None:
        message_move_deadline_seconds = min(topic_edit_deadline_seconds, stream_edit_deadline_seconds)
    elif topic_edit_deadline_seconds is not None:
        message_move_deadline_seconds = topic_edit_deadline_seconds
    elif stream_edit_deadline_seconds is not None:
        message_move_deadline_seconds = stream_edit_deadline_seconds
    else:
        return
    stream = get_stream_by_id_in_realm(message.recipient.type_id, realm)
    if not can_access_stream_history(user_profile, stream):
        accessible_messages_in_topic = UserMessage.objects.filter(user_profile=user_profile, message__recipient_id=message.recipient_id, message__subject__iexact=message.topic_name()).values_list('message_id', flat=True)
        messages_allowed_to_move = list(Message.objects.filter(id__in=accessible_messages_in_topic, date_sent__gt=timezone_now() - timedelta(seconds=message_move_deadline_seconds)).order_by('date_sent').values_list('id', flat=True))
        total_messages_requested_to_move = len(accessible_messages_in_topic)
    else:
        all_messages_in_topic = messages_for_topic(message.realm_id, message.recipient_id, message.topic_name()).order_by('id').values_list('id', 'date_sent')
        oldest_allowed_message_date = timezone_now() - timedelta(seconds=message_move_deadline_seconds)
        messages_allowed_to_move = [message[0] for message in all_messages_in_topic if message[1] > oldest_allowed_message_date]
        total_messages_requested_to_move = len(all_messages_in_topic)
    if total_messages_requested_to_move == len(messages_allowed_to_move):
        return
    raise MessageMoveError(first_message_id_allowed_to_move=messages_allowed_to_move[0], total_messages_in_topic=total_messages_requested_to_move, total_messages_allowed_to_move=len(messages_allowed_to_move))

def build_message_edit_request(*, message: Message, user_profile: UserProfile, propagate_mode: str, stream_id: Optional[int] = None, topic_name: Optional[str] = None, content: Optional[str] = None) -> Union[StreamMessageEditRequest, DirectMessageEditRequest]:
    is_content_edited = False
    new_content = message.content
    if content is not None:
        is_content_edited = True
        if content.rstrip() == '':
            content = '(deleted)'
        new_content = normalize_body(content)
    if not message.is_stream_message():
        return DirectMessageEditRequest(content=new_content, orig_content=message.content, is_content_edited=True)
    is_topic_edited = False
    topic_resolved = False
    topic_unresolved = False
    old_topic_name = message.topic_name()
    target_topic_name = old_topic_name
    if topic_name is not None:
        is_topic_edited = True
        pre_truncation_target_topic_name = topic_name
        target_topic_name = truncate_topic(topic_name)
        resolved_prefix_len = len(RESOLVED_TOPIC_PREFIX)
        topic_resolved = target_topic_name.startswith(RESOLVED_TOPIC_PREFIX) and (not old_topic_name.startswith(RESOLVED_TOPIC_PREFIX)) and (pre_truncation_target_topic_name[resolved_prefix_len:] == old_topic_name)
        topic_unresolved = old_topic_name.startswith(RESOLVED_TOPIC_PREFIX) and (not target_topic_name.startswith(RESOLVED_TOPIC_PREFIX)) and (old_topic_name.lstrip(RESOLVED_TOPIC_PREFIX) == target_topic_name)
    orig_stream_id = message.recipient.type_id
    orig_stream = get_stream_by_id_in_realm(orig_stream_id, message.realm)
    is_stream_edited = False
    target_stream = orig_stream
    if stream_id is not None:
        target_stream = access_stream_by_id_for_message(user_profile, stream_id, require_active=True)[0]
        is_stream_edited = True
    return StreamMessageEditRequest(is_content_edited=is_content_edited, content=new_content, is_topic_edited=is_topic_edited, target_topic_name=target_topic_name, is_stream_edited=is_stream_edited, topic_resolved=topic_resolved, topic_unresolved=topic_unresolved, orig_content=message.content, orig_topic_name=old_topic_name, orig_stream=orig_stream, propagate_mode=propagate_mode, target_stream=target_stream, is_message_moved=is_stream_edited or is_topic_edited)

@transaction.atomic(durable=True)
def check_update_message(user_profile: UserProfile, message_id: int, stream_id: Optional[int] = None, topic_name: Optional[str] = None, propagate_mode: str = 'change_one', send_notification_to_old_thread: bool = True, send_notification_to_new_thread: bool = True, content: Optional[str] = None) -> UpdateMessageResult:
    """This will update a message given the message id and user profile.
    It checks whether the user profile has the permission to edit the message
    and raises a JsonableError if otherwise.
    It returns the number changed.
    """
    message = access_message(user_profile, message_id, lock_message=True)
    edit_limit_buffer = 20
    if content is not None:
        validate_user_can_edit_message(user_profile, message, edit_limit_buffer)
    if topic_name is not None:
        topic_name = topic_name.strip()
        topic_name = maybe_rename_general_chat_to_empty_topic(topic_name)
        if topic_name == message.topic_name():
            topic_name = None
    validate_message_edit_payload(message, stream_id, topic_name, propagate_mode, content)
    message_edit_request = build_message_edit_request(message=message, user_profile=user_profile, propagate_mode=propagate_mode, stream_id=stream_id, topic_name=topic_name, content=content)
    if isinstance(message_edit_request, StreamMessageEditRequest) and message_edit_request.is_topic_edited:
        if not user_profile.can_move_messages_to_another_topic():
            raise JsonableError(_("You don't have permission to edit this message"))
        if user_profile.realm.move_messages_within_stream_limit_seconds is not None and (not user_profile.is_realm_admin) and (not user_profile.is_moderator):
            deadline_seconds = user_profile.realm.move_messages_within_stream_limit_seconds + edit_limit_buffer
            if timezone_now() - message.date_sent > timedelta(seconds=deadline_seconds):
                raise JsonableError(_("The time limit for editing this message's topic has passed."))
    rendering_result = None
    links_for_embed: Set[str] = set()
    prior_mention_user_ids: Set[int] = set()
    mention_data = None
    if message_edit_request.is_content_edited:
        mention_backend = MentionBackend(user_profile.realm_id)
        mention_data = MentionData(mention_backend=mention_backend, content=message_edit_request.content, message_sender=message.sender)
        prior_mention_user_ids = get_mentions_for_message_updates(message)
        rendering_result = render_incoming_message(message, message_edit_request.content, user_profile.realm, mention_data=mention_data)
        links_for_embed |= rendering_result.links_for_preview
        if message.is_stream_message() and rendering_result.mentions_stream_wildcard:
            stream = access_stream_by_id(user_profile, message.recipient.type_id)[0]
            if not stream_wildcard_mention_allowed(message.sender, stream, message.realm):
                raise StreamWildcardMentionNotAllowedError
        if message.is_stream_message() and rendering_result.mentions_topic_wildcard:
            topic_participant_count = len(participants_for_topic(message.realm.id, message.recipient.id, message.topic_name()))
            if not topic_wildcard_mention_allowed(message.sender, topic_participant_count, message.realm):
                raise TopicWildcardMentionNotAllowedError
        if rendering_result.mentions_user_group_ids:
            mentioned_group_ids = list(rendering_result.mentions_user_group_ids)
            check_user_group_mention_allowed(user_profile, mentioned_group_ids)
    if isinstance(message_edit_request, StreamMessageEditRequest):
        if message_edit_request.is_stream_edited:
            assert message.is_stream_message()
            if not user_profile.can_move_messages_between_streams():
                raise JsonableError(_("You don't have permission to move this message"))
            check_stream_access_based_on_can_send_message_group(user_profile, message_edit_request.target_stream)
            if user_profile.realm.move_messages_between_streams_limit_seconds is not None and (not user_profile.is_realm_admin) and (not user_profile.is_moderator):
                deadline_seconds = user_profile.realm.move_messages_between_streams_limit_seconds + edit_limit_buffer
                if timezone_now() - message.date_sent > timedelta(seconds=deadline_seconds):
                    raise JsonableError(_("The time limit for editing this message's channel has passed"))
        if propagate_mode == 'change_all' and (not user_profile.is_realm_admin) and (not user_profile.is_moderator) and message_edit_request.is_message_moved:
            check_time_limit_for_change_all_propagate_mode(message, user_profile, topic_name, stream_id)
    updated_message_result = do_update_message(user_profile, message, message_edit_request, send_notification_to_old_thread, send_notification_to_new_thread, rendering_result, prior_mention_user_ids, mention_data)
    if links_for_embed:
        event_data = {'message_id': message.id, 'message_content': message.content, 'message_realm_id': user_profile.realm_id, 'urls': list(links_for_embed)}
        queue_event_on_commit('embed_links', event_data)
    if isinstance(message_edit_request, StreamMessageEditRequest) and message_edit_request.is_stream_edited and (not message_edit_request.target_stream.is_recently_active):
        date_days_ago = timezone_now() - timedelta(days=Stream.LAST_ACTIVITY_DAYS_BEFORE_FOR_ACTIVE)
        new_stream = message_edit_request.target_stream
        is_stream_active = Message.objects.filter(date_sent__gte=date_days_ago, recipient__type=Recipient.STREAM, realm=user_profile.realm, recipient__type_id=new_stream.id).exists()
        if is_stream_active != new_stream.is_recently_active:
            new_stream.is_recently_active = is_stream_active
            new_stream.save(update_fields=['is_recently_active'])
            notify_stream_is_recently_active_update(new_stream, is_stream_active)
    return updated_message_result

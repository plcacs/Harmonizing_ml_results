import itertools
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

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
from zerver.actions.message_send import (
    filter_presence_idle_user_ids,
    get_recipient_info,
    internal_send_stream_message,
    render_incoming_message,
)
from zerver.actions.uploads import AttachmentChangeResult, check_attachment_reference_change
from zerver.actions.user_topics import bulk_do_set_user_topic_visibility_policy
from zerver.lib.exceptions import (
    JsonableError,
    MessageMoveError,
    StreamWildcardMentionNotAllowedError,
    TopicWildcardMentionNotAllowedError,
)
from zerver.lib.markdown import MessageRenderingResult, topic_links
from zerver.lib.markdown import version as markdown_version
from zerver.lib.mention import MentionBackend, MentionData, silent_mention_syntax_for_user
from zerver.lib.message import (
    access_message,
    bulk_access_stream_messages_query,
    check_user_group_mention_allowed,
    event_recipient_ids_for_action_on_messages,
    normalize_body,
    stream_wildcard_mention_allowed,
    topic_wildcard_mention_allowed,
    truncate_topic,
)
from zerver.lib.message_cache import update_message_cache
from zerver.lib.queue import queue_event_on_commit
from zerver.lib.stream_subscription import get_active_subscriptions_for_stream_id
from zerver.lib.stream_topic import StreamTopicTarget
from zerver.lib.streams import (
    access_stream_by_id,
    access_stream_by_id_for_message,
    can_access_stream_history,
    check_stream_access_based_on_can_send_message_group,
    notify_stream_is_recently_active_update,
)
from zerver.lib.string_validation import check_stream_topic
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.topic import (
    ORIG_TOPIC,
    RESOLVED_TOPIC_PREFIX,
    TOPIC_LINKS,
    TOPIC_NAME,
    maybe_rename_general_chat_to_empty_topic,
    messages_for_topic,
    participants_for_topic,
    save_message_for_edit_use_case,
    update_edit_history,
    update_messages_for_topic_edit,
)
from zerver.lib.types import DirectMessageEditRequest, EditHistoryEvent, StreamMessageEditRequest
from zerver.lib.url_encoding import near_stream_message_url
from zerver.lib.user_message import bulk_insert_all_ums
from zerver.lib.user_topics import get_users_with_user_topic_visibility_policy
from zerver.lib.widget import is_widget_message
from zerver.models import (
    ArchivedAttachment,
    Attachment,
    Message,
    Reaction,
    Recipient,
    Stream,
    Subscription,
    UserMessage,
    UserProfile,
    UserTopic,
)
from zerver.models.streams import get_stream_by_id_in_realm
from zerver.models.users import get_system_bot
from zerver.tornado.django_api import send_event_on_commit


@dataclass
class UpdateMessageResult:
    changed_message_count: int
    detached_uploads: list[dict[str, Any]]


def subscriber_info(user_id: int) -> dict[str, Any]:
    return {"id": user_id, "flags": ["read"]}


def validate_message_edit_payload(
    message: Message,
    stream_id: int | None,
    topic_name: str | None,
    propagate_mode: str | None,
    content: str | None,
) -> None:
    """
    Checks that the data sent is well-formed. Does not handle editability, permissions etc.
    """
    if topic_name is None and content is None and stream_id is None:
        raise JsonableError(_("Nothing to change"))

    if not message.is_stream_message():
        if stream_id is not None:
            raise JsonableError(_("Direct messages cannot be moved to channels."))
        if topic_name is not None:
            raise JsonableError(_("Direct messages cannot have topics."))

    if propagate_mode != "change_one" and topic_name is None and stream_id is None:
        raise JsonableError(_("Invalid propagate_mode without topic edit"))

    if message.realm.mandatory_topics and topic_name in ("(no topic)", ""):
        raise JsonableError(_("Topics are required in this organization."))

    if topic_name in {
        RESOLVED_TOPIC_PREFIX.strip(),
        f"{RESOLVED_TOPIC_PREFIX}{Message.EMPTY_TOPIC_FALLBACK_NAME}",
    }:
        raise JsonableError(_("General chat cannot be marked as resolved"))

    if topic_name is not None:
        check_stream_topic(topic_name)

    if stream_id is not None and content is not None:
        raise JsonableError(_("Cannot change message content while changing channel"))

    # Right now, we prevent users from editing widgets.
    if content is not None and is_widget_message(message):
        raise JsonableError(_("Widgets cannot be edited."))


def validate_user_can_edit_message(
    user_profile: UserProfile,
    message: Message,
    edit_limit_buffer: int,
) -> None:
    """
    Checks if the user has the permission to edit the message.
    """
    if not user_profile.realm.allow_message_editing:
        raise JsonableError(_("Your organization has turned off message editing"))

    # You cannot edit the content of message sent by someone else.
    if message.sender_id != user_profile.id:
        raise JsonableError(_("You don't have permission to edit this message"))

    if user_profile.realm.message_content_edit_limit_seconds is not None:
        deadline_seconds = user_profile.realm.message_content_edit_limit_seconds + edit_limit_buffer
        if (timezone_now() - message.date_sent) > timedelta(seconds=deadline_seconds):
            raise JsonableError(_("The time limit for editing this message has passed"))


def maybe_send_resolve_topic_notifications(
    *,
    user_profile: UserProfile,
    message_edit_request: StreamMessageEditRequest,
    changed_messages: QuerySet[Message],
) -> tuple[int | None, bool]:
    """Returns resolved_topic_message_id if resolve topic notifications were in fact sent."""
    # Note that topics will have already been stripped in check_update_message.
    topic_resolved = message_edit_request.topic_resolved
    topic_unresolved = message_edit_request.topic_unresolved
    if not topic_resolved and not topic_unresolved:
        # If there's some other weird topic that does not toggle the
        # state of "topic starts with RESOLVED_TOPIC_PREFIX", we do
        # nothing. Any other logic could result in cases where we send
        # these notifications in a non-alternating fashion.
        #
        # Note that it is still possible for an individual topic to
        # have multiple "This topic was marked as resolved"
        # notifications in a row: one can send new messages to the
        # pre-resolve topic and then resolve the topic created that
        # way to get multiple in the resolved topic. And then an
        # administrator can delete the messages in between. We consider this
        # to be a fundamental risk of irresponsible message deletion,
        # not a bug with the "resolve topics" feature.
        return None, False

    stream = message_edit_request.orig_stream
    # Sometimes a user might accidentally resolve a topic, and then
    # have to undo the action. We don't want to spam "resolved",
    # "unresolved" messages one after another in such a situation.
    # For that reason, we apply a short grace period during which
    # such an undo action will just delete the previous notification
    # message instead.
    if maybe_delete_previous_resolve_topic_notification(
        user_profile, stream, message_edit_request.target_topic_name
    ):
        return None, True

    # Compute the users who either sent or reacted to messages that
    # were moved via the "resolve topic' action. Only those users
    # should be eligible for this message being managed as unread.
    affected_participant_ids = set(
        changed_messages.values_list("sender_id", flat=True).union(
            Reaction.objects.filter(message__in=changed_messages).values_list(
                "user_profile_id", flat=True
            )
        )
    )
    sender = get_system_bot(settings.NOTIFICATION_BOT, user_profile.realm_id)
    user_mention = silent_mention_syntax_for_user(user_profile)
    with override_language(stream.realm.default_language):
        if topic_resolved:
            notification_string = _("{user} has marked this topic as resolved.")
        elif topic_unresolved:
            notification_string = _("{user} has marked this topic as unresolved.")

        resolved_topic_message_id = internal_send_stream_message(
            sender,
            stream,
            message_edit_request.target_topic_name,
            notification_string.format(
                user=user_mention,
            ),
            message_type=Message.MessageType.RESOLVE_TOPIC_NOTIFICATION,
            limit_unread_user_ids=affected_participant_ids,
            acting_user=user_profile,
        )

    return resolved_topic_message_id, False


def maybe_delete_previous_resolve_topic_notification(
    user_profile: UserProfile,
    stream: Stream,
    topic: str,
) -> bool:
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


def send_message_moved_breadcrumbs(
    target_message: Message,
    user_profile: UserProfile,
    message_edit_request: StreamMessageEditRequest,
    old_thread_notification_string: StrPromise | None,
    new_thread_notification_string: StrPromise | None,
    changed_messages_count: int,
) -> None:
    # Since moving content between streams is highly disruptive,
    # it's worth adding a couple tombstone messages showing what
    # happened.
    old_stream = message_edit_request.orig_stream
    sender = get_system_bot(settings.NOTIFICATION_BOT, old_stream.realm_id)

    user_mention = silent_mention_syntax_for_user(user_profile)
    old_topic_name = message_edit_request.orig_topic_name
    new_stream = message_edit_request.target_stream
    new_topic_name = message_edit_request.target_topic_name
    old_topic_link = f"#**{old_stream.name}>{old_topic_name}**"
    new_topic_link = f"#**{new_stream.name}>{new_topic_name}**"
    message = {
        "id": target_message.id,
        "stream_id": new_stream.id,
        "display_recipient": new_stream.name,
        "topic": new_topic_name,
    }
    moved_message_link = near_stream_message_url(target_message.realm, message)

    if new_thread_notification_string is not None:
        with override_language(new_stream.realm.default_language):
            internal_send_stream_message(
                sender,
                new_stream,
                new_topic_name,
                new_thread_notification_string.format(
                    message_link=moved_message_link,
                    old_location=old_topic_link,
                    user=user_mention,
                    changed_messages_count=changed_messages_count,
                ),
                acting_user=user_profile,
            )

    if old_thread_notification_string is not None:
        with override_language(old_stream.realm.default_language):
            # Send a notification to the old stream that the topic was moved.
            internal_send_stream_message(
                sender,
                old_stream,
                old_topic_name,
                old_thread_notification_string.format(
                    user=user_mention,
                    new_location=new_topic_link,
                    changed_messages_count=changed_messages_count,
                ),
                acting_user=user_profile,
            )


def get_mentions_for_message_updates(message: Message) -> set[int]:
    # We exclude UserMessage.flags.historical rows since those
    # users did not receive the message originally, and thus
    # probably are not relevant for reprocessed alert_words,
    # mentions and similar rendering features.  This may be a
    # decision we change in the future.
    mentioned_user_ids = (
        UserMessage.objects.filter(
            message=message.id,
            flags=~UserMessage.flags.historical,
        )
        .filter(
            Q(
                flags__andnz=UserMessage.flags.mentioned
                | UserMessage.flags.stream_wildcard_mentioned
                | UserMessage.flags.topic_wildcard_mentioned
                | UserMessage.flags.group_mentioned
            )
        )
        .values_list("user_profile_id", flat=True)
    )

    user_ids_having_message_access = event_recipient_ids_for_action_on_messages([message])

    return set(mentioned_user_ids) & user_ids_having_message_access


def update_user_message_flags(
    rendering_result: MessageRenderingResult,
    ums: Iterable[UserMessage],
    topic_participant_user_ids: set[int] = set(),
) -> None:
    mentioned_ids = rendering_result.mentions_user_ids
    ids_with_alert_words = rendering_result.user_ids_with_alert_words
    changed_ums: set[UserMessage] = set()

    def update_flag(um: UserMessage, should_set: bool, flag: int) -> None:
        if should_set:
            if not (um.flags & flag):
                um.flags |= flag
                changed_ums.add(um)
        else:
            if um.flags & flag:
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
        um.save(update_fields=["flags"])


def do_update_embedded_data(
    user_profile: UserProfile,
    message: Message,
    rendered_content: str | MessageRenderingResult,
) -> None:
    ums = UserMessage.objects.filter(message=message.id)
    update_fields = ["rendered_content"]
    if isinstance(rendered_content, MessageRenderingResult):
        update_user_message_flags(rendered_content, ums)
        message.rendered_content = rendered_content.rendered_content
        message.rendered_content_version = markdown_version
        update_fields.append("rendered_content_version")
    else:
        message.rendered_content = rendered_content
    message.save(update_fields=update_fields)

    update_message_cache([message])
    event: dict[str, Any] = {
        "type": "update_message",
        "user_id": None,
        "edit_timestamp": datetime_to_timestamp(timezone_now()),
        "message_id": message.id,
        "message_ids": [message.id],
        "content": message.content,
        "rendered_content": message.rendered_content,
        "rendering_only": True,
    }

    users_to_notify = event_recipient_ids_for_action_on_messages([message])
    filtered_ums = [um for um in ums if um.user_profile_id in users_to_notify]

    def user_info(um: UserMessage) -> dict[str, Any]:
        return {
            "id": um.user_profile_id,
            "flags": um.flags_list(),
        }

    send_event_on_commit(user_profile.realm, event, list(map(user_info, filtered_ums)))


def get_visibility_policy_after_merge(
    orig_topic_visibility_policy: int,
    target_topic_visibility_policy: int,
) -> int:
    # This function determines the final visibility_policy after the merge
    # operation, based on the visibility policies of the original and target
    # topics.
    #
    # The algorithm to decide is based on:
    # Whichever of the two policies is most visible is what we keep.
    # The general motivation is to err on the side of showing messages
    # rather than hiding them.
    if orig_topic_visibility_policy == target_topic_visibility_policy:
        return orig_topic_visibility_policy
    elif UserTopic.VisibilityPolicy.UNMUTED in (
        orig_topic_visibility_policy,
        target_topic_visibility_policy,
    ):
        return UserTopic.VisibilityPolicy.UNMUTED
    return UserTopic.VisibilityPolicy.INHERIT


def update_message_content(
    user_profile: UserProfile,
    target_message: Message,
    content: str,
    rendering_result: MessageRenderingResult,
    prior_mention_user_ids: set[int],
    mention_data: MentionData,
    event: dict[str, Any],
    edit_history_event: EditHistoryEvent,
    stream_topic: StreamTopicTarget | None,
) -> None:
    realm = user_profile.realm

    ums = UserMessage.objects.filter(message=target_message.id)

    # add data from group mentions to mentions_user_ids.
    for group_id in rendering_result.mentions_user_group_ids:
        members = mention_data.get_group_members(group_id)
        rendering_result.mentions_user_ids.update(members)

    # One could imagine checking realm.allow_edit_history here and
    # modifying the events based on that setting, but doing
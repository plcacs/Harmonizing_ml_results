from collections.abc import Callable
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import orjson
from django.db import connection
from django.db.models import F, Func, JSONField, Q, QuerySet, Subquery, TextField, Value
from django.db.models.functions import Cast
from django.utils.translation import gettext as _
from django.utils.translation import override as override_language
from zerver.lib.types import EditHistoryEvent, StreamMessageEditRequest
from zerver.lib.utils import assert_is_not_none
from zerver.models import Message, Reaction, UserMessage, UserProfile, Realm, Recipient, Stream

ORIG_TOPIC = 'orig_subject'
TOPIC_NAME = 'subject'
TOPIC_LINKS = 'topic_links'
MATCH_TOPIC = 'match_subject'
RESOLVED_TOPIC_PREFIX = '✔ '
EXPORT_TOPIC_NAME = 'subject'

def get_topic_from_message_info(message_info: Dict[str, Any]) -> str:
    if 'topic' in message_info:
        return message_info['topic']
    return message_info['subject']

DB_TOPIC_NAME = 'subject'
MESSAGE__TOPIC = 'message__subject'

def filter_by_topic_name_via_message(query: QuerySet, topic_name: str) -> QuerySet:
    return query.filter(message__subject__iexact=topic_name)

def messages_for_topic(realm_id: int, stream_recipient_id: int, topic_name: str) -> QuerySet[Message]:
    return Message.objects.filter(realm_id=realm_id, recipient_id=stream_recipient_id, subject__iexact=topic_name)

def get_first_message_for_user_in_topic(
    realm_id: int,
    user_profile: Optional[UserProfile],
    recipient_id: int,
    topic_name: str,
    history_public_to_subscribers: bool,
    acting_user_has_channel_content_access: bool = False
) -> Optional[int]:
    assert acting_user_has_channel_content_access
    if history_public_to_subscribers:
        return messages_for_topic(realm_id, recipient_id, topic_name).values_list('id', flat=True).first()
    elif user_profile is not None:
        return UserMessage.objects.filter(
            user_profile=user_profile,
            message__recipient_id=recipient_id,
            message__subject__iexact=topic_name
        ).values_list('message_id', flat=True).first()
    return None

def save_message_for_edit_use_case(message: Message) -> None:
    message.save(update_fields=[TOPIC_NAME, 'content', 'rendered_content', 'rendered_content_version', 'last_edit_time', 'edit_history', 'has_attachment', 'has_image', 'has_link', 'recipient_id'])

def user_message_exists_for_topic(user_profile: UserProfile, recipient_id: int, topic_name: str) -> bool:
    return UserMessage.objects.filter(
        user_profile=user_profile,
        message__recipient_id=recipient_id,
        message__subject__iexact=topic_name
    ).exists()

def update_edit_history(message: Message, last_edit_time: datetime, edit_history_event: EditHistoryEvent) -> None:
    message.last_edit_time = last_edit_time
    if message.edit_history is not None:
        edit_history = orjson.loads(message.edit_history)
        edit_history.insert(0, edit_history_event)
    else:
        edit_history = [edit_history_event]
    message.edit_history = orjson.dumps(edit_history).decode()

def update_messages_for_topic_edit(
    acting_user: UserProfile,
    edited_message: Message,
    message_edit_request: StreamMessageEditRequest,
    edit_history_event: EditHistoryEvent,
    last_edit_time: datetime
) -> Tuple[QuerySet[Message], Callable[[], QuerySet[Message]]]:
    old_stream = message_edit_request.orig_stream
    messages = Message.objects.filter(
        realm_id=old_stream.realm_id,
        recipient_id=assert_is_not_none(old_stream.recipient_id),
        subject__iexact=message_edit_request.orig_topic_name
    )
    if message_edit_request.propagate_mode == 'change_all':
        messages = messages.exclude(id=edited_message.id)
    if message_edit_request.propagate_mode == 'change_later':
        messages = messages.filter(id__gt=edited_message.id)
    if message_edit_request.is_stream_edited:
        from zerver.lib.message import bulk_access_stream_messages_query
        messages = bulk_access_stream_messages_query(acting_user, messages, old_stream)
    update_fields = {
        'last_edit_time': last_edit_time,
        'edit_history': Cast(
            Func(
                Cast(Value(orjson.dumps([edit_history_event]).decode()), JSONField()),
                Cast(Func(F('edit_history'), Value('[]'), function='COALESCE'), JSONField()),
                function='',
                arg_joiner=' || '
            ), TextField()
        )
    }
    if message_edit_request.is_stream_edited:
        update_fields['recipient'] = message_edit_request.target_stream.recipient
    if message_edit_request.is_topic_edited:
        update_fields['subject'] = message_edit_request.target_topic_name
    message_ids = [edited_message.id, *messages.values_list('id', flat=True)]

    def propagate() -> QuerySet[Message]:
        messages.update(**update_fields)
        return Message.objects.filter(id__in=message_ids).select_related(*Message.DEFAULT_SELECT_RELATED)
    return (messages, propagate)

def generate_topic_history_from_db_rows(rows: List[Tuple[str, int]], allow_empty_topic_name: bool) -> List[Dict[str, Any]]:
    canonical_topic_names = {}
    rows = sorted(rows, key=lambda tup: tup[1])
    for topic_name, max_message_id in rows:
        canonical_name = topic_name.lower()
        canonical_topic_names[canonical_name] = (max_message_id, topic_name)
    history = []
    for max_message_id, topic_name in canonical_topic_names.values():
        if topic_name == '' and (not allow_empty_topic_name):
            topic_name = Message.EMPTY_TOPIC_FALLBACK_NAME
        history.append(dict(name=topic_name, max_id=max_message_id))
    return sorted(history, key=lambda x: -x['max_id'])

def get_topic_history_for_public_stream(realm_id: int, recipient_id: int, allow_empty_topic_name: bool) -> List[Dict[str, Any]]:
    cursor = connection.cursor()
    query = """
    SELECT
        "zerver_message"."subject" as topic,
        max("zerver_message".id) as max_message_id
    FROM "zerver_message"
    WHERE (
        "zerver_message"."realm_id" = %s AND
        "zerver_message"."recipient_id" = %s
    )
    GROUP BY (
        "zerver_message"."subject"
    )
    ORDER BY max("zerver_message".id) DESC
    """
    cursor.execute(query, [realm_id, recipient_id])
    rows = cursor.fetchall()
    cursor.close()
    return generate_topic_history_from_db_rows(rows, allow_empty_topic_name)

def get_topic_history_for_stream(
    user_profile: UserProfile,
    recipient_id: int,
    public_history: bool,
    allow_empty_topic_name: bool
) -> List[Dict[str, Any]]:
    if public_history:
        return get_topic_history_for_public_stream(user_profile.realm_id, recipient_id, allow_empty_topic_name)
    cursor = connection.cursor()
    query = """
    SELECT
        "zerver_message"."subject" as topic,
        max("zerver_message".id) as max_message_id
    FROM "zerver_message"
    INNER JOIN "zerver_usermessage" ON (
        "zerver_usermessage"."message_id" = "zerver_message"."id"
    )
    WHERE (
        "zerver_usermessage"."user_profile_id" = %s AND
        "zerver_message"."realm_id" = %s AND
        "zerver_message"."recipient_id" = %s
    )
    GROUP BY (
        "zerver_message"."subject"
    )
    ORDER BY max("zerver_message".id) DESC
    """
    cursor.execute(query, [user_profile.id, user_profile.realm_id, recipient_id])
    rows = cursor.fetchall()
    cursor.close()
    return generate_topic_history_from_db_rows(rows, allow_empty_topic_name)

def get_topic_resolution_and_bare_name(stored_name: str) -> Tuple[bool, str]:
    if stored_name.startswith(RESOLVED_TOPIC_PREFIX):
        return (True, stored_name.removeprefix(RESOLVED_TOPIC_PREFIX))
    return (False, stored_name)

def participants_for_topic(realm_id: int, recipient_id: int, topic_name: str) -> Set[int]:
    messages = Message.objects.filter(realm_id=realm_id, recipient_id=recipient_id, subject__iexact=topic_name)
    participants = set(UserProfile.objects.filter(
        Q(id__in=Subquery(messages.values('sender_id'))) | 
        Q(id__in=Subquery(Reaction.objects.filter(message__in=messages).values('user_profile_id')))
    ).values_list('id', flat=True))
    return participants

def maybe_rename_general_chat_to_empty_topic(topic_name: str) -> str:
    if topic_name == Message.EMPTY_TOPIC_FALLBACK_NAME:
        topic_name = ''
    return topic_name

def maybe_rename_empty_topic_to_general_chat(topic_name: str, is_channel_message: bool, allow_empty_topic_name: bool) -> str:
    if is_channel_message and topic_name == '' and (not allow_empty_topic_name):
        return Message.EMPTY_TOPIC_FALLBACK_NAME
    return topic_name

def get_topic_display_name(topic_name: str, language: str) -> str:
    if topic_name == '':
        with override_language(language):
            return _(Message.EMPTY_TOPIC_FALLBACK_NAME)
    return topic_name

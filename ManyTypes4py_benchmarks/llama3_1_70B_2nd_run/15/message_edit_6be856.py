from typing import Optional, Set, List
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from django.db import transaction
from django.db.models import QuerySet
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext_lazy
from django_stubs_ext import StrPromise
from zerver.models import Message, UserProfile, Stream

@dataclass
class UpdateMessageResult:
    changed_messages_count: int
    detached_attachments: List[str]

def subscriber_info(user_id: int) -> dict:
    return {'id': user_id, 'flags': ['read']}

def validate_message_edit_payload(message: Message, stream_id: Optional[int], topic_name: Optional[str], propagate_mode: str, content: Optional[str]) -> None:
    ...

def validate_user_can_edit_message(user_profile: UserProfile, message: Message, edit_limit_buffer: int) -> None:
    ...

def maybe_send_resolve_topic_notifications(*, user_profile: UserProfile, message_edit_request: object, changed_messages: QuerySet[Message]) -> tuple:
    ...

def maybe_delete_previous_resolve_topic_notification(user_profile: UserProfile, stream: Stream, topic: str) -> bool:
    ...

def send_message_moved_breadcrumbs(target_message: Message, user_profile: UserProfile, message_edit_request: object, old_thread_notification_string: Optional[str], new_thread_notification_string: Optional[str], changed_messages_count: int) -> None:
    ...

def get_mentions_for_message_updates(message: Message) -> Set[int]:
    ...

def update_user_message_flags(rendering_result: object, ums: QuerySet[object], topic_participant_user_ids: Set[int]) -> None:
    ...

def do_update_embedded_data(user_profile: UserProfile, message: Message, rendered_content: object) -> None:
    ...

def get_visibility_policy_after_merge(orig_topic_visibility_policy: str, target_topic_visibility_policy: str) -> str:
    ...

def update_message_content(user_profile: UserProfile, target_message: Message, content: str, rendering_result: object, prior_mention_user_ids: Set[int], mention_data: object, event: dict, edit_history_event: dict, stream_topic: object) -> None:
    ...

@transaction.atomic(savepoint=False)
def do_update_message(user_profile: UserProfile, target_message: Message, message_edit_request: object, send_notification_to_old_thread: bool, send_notification_to_new_thread: bool, rendering_result: Optional[object], prior_mention_user_ids: Set[int], mention_data: Optional[object] = None) -> UpdateMessageResult:
    ...

def check_time_limit_for_change_all_propagate_mode(message: Message, user_profile: UserProfile, topic_name: Optional[str] = None, stream_id: Optional[int] = None) -> None:
    ...

def build_message_edit_request(*, message: Message, user_profile: UserProfile, propagate_mode: str, stream_id: Optional[int] = None, topic_name: Optional[str] = None, content: Optional[str] = None) -> object:
    ...

@transaction.atomic(durable=True)
def check_update_message(user_profile: UserProfile, message_id: int, stream_id: Optional[int] = None, topic_name: Optional[str] = None, propagate_mode: str = 'change_one', send_notification_to_old_thread: bool = True, send_notification_to_new_thread: bool = True, content: Optional[str] = None) -> UpdateMessageResult:
    ...

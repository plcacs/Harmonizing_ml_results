from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple

import logging

from django.conf import settings
from django.db import transaction
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import override as override_language

from zerver.actions.message_send import check_message, do_send_messages, internal_send_private_message
from zerver.actions.uploads import check_attachment_reference_change, do_claim_attachments
from zerver.lib.addressee import Addressee
from zerver.lib.display_recipient import get_recipient_ids
from zerver.lib.exceptions import JsonableError, RealmDeactivatedError, UserDeactivatedError
from zerver.lib.markdown import render_message_markdown
from zerver.lib.message import SendMessageRequest, truncate_topic
from zerver.lib.recipient_parsing import extract_direct_message_recipient_ids, extract_stream_id
from zerver.lib.scheduled_messages import access_scheduled_message
from zerver.lib.string_validation import check_stream_topic
from zerver.models import Client, Realm, ScheduledMessage, Subscription, UserProfile
from zerver.models.users import get_system_bot
from zerver.tornado.django_api import send_event_on_commit

SCHEDULED_MESSAGE_LATE_CUTOFF_MINUTES: int = 10

def check_schedule_message(
    sender: UserProfile,
    client: Client,
    recipient_type_name: str,
    message_to: Any,
    topic_name: str,
    message_content: str,
    deliver_at: datetime,
    realm: Optional[Realm] = None,
    *,
    forwarder_user_profile: Optional[UserProfile] = None,
    read_by_sender: Optional[bool] = None,
) -> int:
    addressee: Addressee = Addressee.legacy_build(sender, recipient_type_name, message_to, topic_name)
    send_request: SendMessageRequest = check_message(sender, client, addressee, message_content, realm=realm, forwarder_user_profile=forwarder_user_profile)
    send_request.deliver_at = deliver_at
    if read_by_sender is None:
        read_by_sender = client.default_read_by_sender() and send_request.message.recipient != sender.recipient
    scheduled_ids: List[int] = do_schedule_messages([send_request], sender, read_by_sender=read_by_sender)
    return scheduled_ids[0]

def do_schedule_messages(
    send_message_requests: List[SendMessageRequest],
    sender: UserProfile,
    *,
    read_by_sender: bool = False
) -> List[int]:
    scheduled_messages: List[Tuple[ScheduledMessage, SendMessageRequest]] = []
    for send_request in send_message_requests:
        scheduled_message: ScheduledMessage = ScheduledMessage()
        scheduled_message.sender = send_request.message.sender
        scheduled_message.recipient = send_request.message.recipient
        topic_name_val: str = send_request.message.topic_name()
        scheduled_message.set_topic_name(topic_name=topic_name_val)
        rendering_result = render_message_markdown(send_request.message, send_request.message.content, send_request.realm)
        scheduled_message.content = send_request.message.content
        scheduled_message.rendered_content = rendering_result.rendered_content
        scheduled_message.sending_client = send_request.message.sending_client
        scheduled_message.stream = send_request.stream
        scheduled_message.realm = send_request.realm
        assert send_request.deliver_at is not None
        scheduled_message.scheduled_timestamp = send_request.deliver_at
        scheduled_message.read_by_sender = read_by_sender
        scheduled_message.delivery_type = ScheduledMessage.SEND_LATER
        scheduled_messages.append((scheduled_message, send_request))
    with transaction.atomic(durable=True):
        ScheduledMessage.objects.bulk_create([sm for sm, _ in scheduled_messages])
        for scheduled_message, send_request in scheduled_messages:
            if do_claim_attachments(scheduled_message, send_request.rendering_result.potential_attachment_path_ids):
                scheduled_message.has_attachment = True
                scheduled_message.save(update_fields=['has_attachment'])
        event = {
            'type': 'scheduled_messages',
            'op': 'add',
            'scheduled_messages': [sm.to_dict() for sm, _ in scheduled_messages],
        }
        send_event_on_commit(sender.realm, event, [sender.id])
    return [sm.id for sm, _ in scheduled_messages]

def notify_update_scheduled_message(user_profile: UserProfile, scheduled_message: ScheduledMessage) -> None:
    event = {
        'type': 'scheduled_messages',
        'op': 'update',
        'scheduled_message': scheduled_message.to_dict(),
    }
    send_event_on_commit(user_profile.realm, event, [user_profile.id])

@transaction.atomic(durable=True)
def edit_scheduled_message(
    sender: UserProfile,
    client: Client,
    scheduled_message_id: int,
    recipient_type_name: Optional[str],
    message_to: Optional[Any],
    topic_name: Optional[str],
    message_content: Optional[str],
    deliver_at: Optional[datetime],
    realm: Realm,
) -> None:
    scheduled_message_object: ScheduledMessage = access_scheduled_message(sender, scheduled_message_id)
    if scheduled_message_object.delivered is True:
        raise JsonableError(_('Scheduled message was already sent'))
    if scheduled_message_object.failed and deliver_at is None:
        raise JsonableError(_('Scheduled delivery time must be in the future.'))

    existing_recipient, existing_recipient_type_name = get_recipient_ids(scheduled_message_object.recipient, sender.id)
    if recipient_type_name is not None or message_to is not None or message_content is not None:
        if recipient_type_name is not None:
            updated_recipient_type_name: str = recipient_type_name
        else:
            updated_recipient_type_name = existing_recipient_type_name
        if message_to is not None:
            if updated_recipient_type_name == 'stream':
                stream_id: int = extract_stream_id(message_to)
                updated_recipient = [stream_id]
            else:
                updated_recipient = extract_direct_message_recipient_ids(message_to)
        else:
            updated_recipient = existing_recipient
        if topic_name is not None:
            updated_topic_name: str = topic_name
        else:
            updated_topic_name = scheduled_message_object.topic_name()
        if message_content is not None:
            updated_content: str = message_content
        else:
            updated_content = scheduled_message_object.content
        addressee: Addressee = Addressee.legacy_build(sender, updated_recipient_type_name, updated_recipient, updated_topic_name)
        send_request: SendMessageRequest = check_message(sender, client, addressee, updated_content, realm=realm, forwarder_user_profile=sender)
    if recipient_type_name is not None or message_to is not None:
        scheduled_message_object.recipient = send_request.message.recipient
        scheduled_message_object.stream = send_request.stream
        new_topic_name: str = send_request.message.topic_name()
        scheduled_message_object.set_topic_name(topic_name=new_topic_name)
    elif topic_name is not None and existing_recipient_type_name == 'stream':
        check_stream_topic(topic_name)
        new_topic_name = truncate_topic(topic_name)
        scheduled_message_object.set_topic_name(topic_name=new_topic_name)
    if message_content is not None:
        rendering_result = render_message_markdown(send_request.message, send_request.message.content, send_request.realm)
        scheduled_message_object.content = send_request.message.content
        scheduled_message_object.rendered_content = rendering_result.rendered_content
        attachment_reference_change = check_attachment_reference_change(scheduled_message_object, rendering_result)
        scheduled_message_object.has_attachment = attachment_reference_change.did_attachment_change
    if deliver_at is not None:
        scheduled_message_object.scheduled_timestamp = deliver_at
    scheduled_message_object.sending_client = client
    if scheduled_message_object.failed:
        scheduled_message_object.failed = False
        scheduled_message_object.failure_message = None
    scheduled_message_object.save()
    notify_update_scheduled_message(sender, scheduled_message_object)

def notify_remove_scheduled_message(user_profile: UserProfile, scheduled_message_id: int) -> None:
    event = {
        'type': 'scheduled_messages',
        'op': 'remove',
        'scheduled_message_id': scheduled_message_id,
    }
    send_event_on_commit(user_profile.realm, event, [user_profile.id])

@transaction.atomic(durable=True)
def delete_scheduled_message(user_profile: UserProfile, scheduled_message_id: int) -> None:
    scheduled_message_object: ScheduledMessage = access_scheduled_message(user_profile, scheduled_message_id)
    scheduled_message_id_val: int = scheduled_message_object.id
    scheduled_message_object.delete()
    notify_remove_scheduled_message(user_profile, scheduled_message_id_val)

def send_scheduled_message(scheduled_message: ScheduledMessage) -> None:
    assert not scheduled_message.delivered
    assert not scheduled_message.failed
    assert scheduled_message.delivery_type == ScheduledMessage.SEND_LATER
    if scheduled_message.realm.deactivated:
        raise RealmDeactivatedError
    if not scheduled_message.sender.is_active:
        raise UserDeactivatedError
    latest_send_time: datetime = scheduled_message.scheduled_timestamp + timedelta(minutes=SCHEDULED_MESSAGE_LATE_CUTOFF_MINUTES)
    if timezone_now() > latest_send_time:
        raise JsonableError(_('Message could not be sent at the scheduled time.'))
    if scheduled_message.stream is not None:
        addressee: Addressee = Addressee.for_stream(scheduled_message.stream, scheduled_message.topic_name())
    else:
        subscriber_ids: List[int] = list(
            Subscription.objects.filter(recipient=scheduled_message.recipient).values_list('user_profile_id', flat=True)
        )
        addressee = Addressee.for_user_ids(subscriber_ids, scheduled_message.realm)
    send_request: SendMessageRequest = check_message(scheduled_message.sender, scheduled_message.sending_client, addressee, scheduled_message.content, scheduled_message.realm)
    sent_message_result = do_send_messages([send_request], mark_as_read=[scheduled_message.sender_id] if scheduled_message.read_by_sender else [])[0]
    scheduled_message.delivered_message_id = sent_message_result.message_id
    scheduled_message.delivered = True
    scheduled_message.save(update_fields=['delivered', 'delivered_message_id'])
    notify_remove_scheduled_message(scheduled_message.sender, scheduled_message.id)

def send_failed_scheduled_message_notification(user_profile: UserProfile, scheduled_message_id: int) -> None:
    scheduled_message: ScheduledMessage = access_scheduled_message(user_profile, scheduled_message_id)
    delivery_datetime_string: str = str(scheduled_message.scheduled_timestamp)
    with override_language(user_profile.default_language):
        error_string: str = scheduled_message.failure_message
        delivery_time_markdown: str = f'<time:{delivery_datetime_string}> '
        content: str = ''.join([
            _('The message you scheduled for {delivery_datetime} was not sent because of the following error:'),
            '\n\n',
            '> {error_message}',
            '\n\n',
            _('[View scheduled messages](#scheduled)'),
            '\n\n'
        ])
    content = content.format(delivery_datetime=delivery_time_markdown, error_message=error_string)
    internal_send_private_message(get_system_bot(settings.NOTIFICATION_BOT, user_profile.realm_id), user_profile, content)

@transaction.atomic(durable=True)
def try_deliver_one_scheduled_message() -> bool:
    scheduled_message: Optional[ScheduledMessage] = ScheduledMessage.objects.filter(
        scheduled_timestamp__lte=timezone_now(), delivered=False, failed=False
    ).select_for_update().first()
    if scheduled_message is None:
        return False
    logging.info(
        'Sending scheduled message %s with date %s (sender: %s)',
        scheduled_message.id,
        scheduled_message.scheduled_timestamp,
        scheduled_message.sender_id,
    )
    with override_language(scheduled_message.sender.default_language):
        try:
            send_scheduled_message(scheduled_message)
        except Exception as e:
            scheduled_message.refresh_from_db()
            was_delivered: bool = scheduled_message.delivered
            scheduled_message.failed = True
            if isinstance(e, JsonableError):
                scheduled_message.failure_message = e.msg
                logging.info('Failed with message: %s', e.msg)
            else:
                scheduled_message.failure_message = _('Internal server error')
                logging.exception('Unexpected error sending scheduled message %s (sent: %s)', scheduled_message.id, was_delivered, stack_info=True)
            scheduled_message.save(update_fields=['failed', 'failure_message'])
            if not was_delivered and (not isinstance(e, RealmDeactivatedError)) and (not isinstance(e, UserDeactivatedError)):
                notify_update_scheduled_message(scheduled_message.sender, scheduled_message)
                send_failed_scheduled_message_notification(scheduled_message.sender, scheduled_message.id)
    return True
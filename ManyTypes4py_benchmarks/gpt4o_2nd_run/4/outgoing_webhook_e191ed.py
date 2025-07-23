import abc
import json
import logging
from contextlib import suppress
from time import perf_counter
from typing import Any, Dict, Optional, Type
import requests
from django.conf import settings
from django.utils.translation import gettext as _
from requests import Response
from typing_extensions import override
from version import ZULIP_VERSION
from zerver.actions.message_send import check_send_message
from zerver.lib.exceptions import JsonableError, StreamDoesNotExistError
from zerver.lib.message_cache import MessageDict
from zerver.lib.outgoing_http import OutgoingSession
from zerver.lib.queue import retry_event
from zerver.lib.topic import get_topic_from_message_info
from zerver.lib.url_encoding import near_message_url
from zerver.lib.users import check_can_access_user, check_user_can_access_all_users
from zerver.models import Realm, Service, UserProfile
from zerver.models.bots import GENERIC_INTERFACE, SLACK_INTERFACE
from zerver.models.clients import get_client
from zerver.models.users import get_user_profile_by_id

class OutgoingWebhookServiceInterface(abc.ABC):

    def __init__(self, token: str, user_profile: UserProfile, service_name: str) -> None:
        self.token = token
        self.user_profile = user_profile
        self.service_name = service_name
        self.session = OutgoingSession(
            role='webhook',
            timeout=settings.OUTGOING_WEBHOOK_TIMEOUT_SECONDS,
            headers={'User-Agent': 'ZulipOutgoingWebhook/' + ZULIP_VERSION}
        )

    @abc.abstractmethod
    def make_request(self, base_url: str, event: Dict[str, Any], realm: Realm) -> Optional[Response]:
        raise NotImplementedError

    @abc.abstractmethod
    def process_success(self, response_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

class GenericOutgoingWebhookService(OutgoingWebhookServiceInterface):

    @override
    def make_request(self, base_url: str, event: Dict[str, Any], realm: Realm) -> Optional[Response]:
        message_dict = MessageDict.finalize_payload(
            event['message'],
            apply_markdown=False,
            client_gravatar=False,
            allow_empty_topic_name=True,
            keep_rendered_content=True,
            can_access_sender=check_user_can_access_all_users(self.user_profile) or check_can_access_user(
                get_user_profile_by_id(event['message']['sender_id']), self.user_profile),
            realm_host=realm.host,
            is_incoming_1_to_1=event['message']['recipient_id'] == self.user_profile.recipient_id
        )
        request_data = {
            'data': event['command'],
            'message': message_dict,
            'bot_email': self.user_profile.email,
            'bot_full_name': self.user_profile.full_name,
            'token': self.token,
            'trigger': event['trigger']
        }
        return self.session.post(base_url, json=request_data)

    @override
    def process_success(self, response_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if response_json.get('response_not_required', False):
            return None
        if 'response_string' in response_json:
            content = str(response_json['response_string'])
            success_data = dict(content=content)
            return success_data
        if 'content' in response_json:
            content = str(response_json['content'])
            success_data = dict(content=content)
            if 'widget_content' in response_json:
                success_data['widget_content'] = response_json['widget_content']
            return success_data
        return None

class SlackOutgoingWebhookService(OutgoingWebhookServiceInterface):

    @override
    def make_request(self, base_url: str, event: Dict[str, Any], realm: Realm) -> Optional[Response]:
        if event['message']['type'] == 'private':
            failure_message = "Slack outgoing webhooks don't support direct messages."
            fail_with_message(event, failure_message)
            return None
        request_data = [
            ('token', self.token),
            ('team_id', f'T{realm.id}'),
            ('team_domain', realm.host),
            ('channel_id', f'C{event["message"]["stream_id"]}'),
            ('channel_name', event['message']['display_recipient']),
            ('thread_ts', event['message']['timestamp']),
            ('timestamp', event['message']['timestamp']),
            ('user_id', f'U{event["message"]["sender_id"]}'),
            ('user_name', event['message']['sender_full_name']),
            ('text', event['command']),
            ('trigger_word', event['trigger']),
            ('service_id', event['user_profile_id'])
        ]
        return self.session.post(base_url, data=request_data)

    @override
    def process_success(self, response_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if 'text' in response_json:
            content = response_json['text']
            success_data = dict(content=content)
            return success_data
        return None

AVAILABLE_OUTGOING_WEBHOOK_INTERFACES: Dict[str, Type[OutgoingWebhookServiceInterface]] = {
    GENERIC_INTERFACE: GenericOutgoingWebhookService,
    SLACK_INTERFACE: SlackOutgoingWebhookService
}

def get_service_interface_class(interface: str) -> Type[OutgoingWebhookServiceInterface]:
    if interface not in AVAILABLE_OUTGOING_WEBHOOK_INTERFACES:
        return AVAILABLE_OUTGOING_WEBHOOK_INTERFACES[GENERIC_INTERFACE]
    else:
        return AVAILABLE_OUTGOING_WEBHOOK_INTERFACES[interface]

def get_outgoing_webhook_service_handler(service: Service) -> OutgoingWebhookServiceInterface:
    service_interface_class = get_service_interface_class(service.interface_name())
    service_interface = service_interface_class(
        token=service.token,
        user_profile=service.user_profile,
        service_name=service.name
    )
    return service_interface

def send_response_message(bot_id: int, message_info: Dict[str, Any], response_data: Dict[str, Any]) -> None:
    recipient_type_name = message_info['type']
    display_recipient = message_info['display_recipient']
    try:
        topic_name = get_topic_from_message_info(message_info)
    except KeyError:
        topic_name = None
    bot_user = get_user_profile_by_id(bot_id)
    realm = bot_user.realm
    client = get_client('OutgoingWebhookResponse')
    content = response_data.get('content')
    assert content
    widget_content = response_data.get('widget_content')
    if recipient_type_name == 'stream':
        message_to = [display_recipient]
    elif recipient_type_name == 'private':
        message_to = [recipient['email'] for recipient in display_recipient]
    else:
        raise JsonableError(_('Invalid message type'))
    check_send_message(
        sender=bot_user,
        client=client,
        recipient_type_name=recipient_type_name,
        message_to=message_to,
        topic_name=topic_name,
        message_content=content,
        widget_content=widget_content,
        realm=realm,
        skip_stream_access_check=True
    )

def fail_with_message(event: Dict[str, Any], failure_message: str) -> None:
    bot_id = event['user_profile_id']
    message_info = event['message']
    content = 'Failure! ' + failure_message
    response_data = dict(content=content)
    with suppress(StreamDoesNotExistError):
        send_response_message(bot_id=bot_id, message_info=message_info, response_data=response_data)

def get_message_url(event: Dict[str, Any]) -> str:
    bot_user = get_user_profile_by_id(event['user_profile_id'])
    message = event['message']
    realm = bot_user.realm
    return near_message_url(realm=realm, message=message)

def notify_bot_owner(event: Dict[str, Any], status_code: Optional[int] = None, response_content: Optional[str] = None, failure_message: Optional[str] = None, exception: Optional[Exception] = None) -> None:
    message_url = get_message_url(event)
    bot_id = event['user_profile_id']
    bot = get_user_profile_by_id(bot_id)
    bot_owner = bot.bot_owner
    assert bot_owner is not None
    notification_message = f'[A message]({message_url}) to your bot @_**{bot.full_name}** triggered an outgoing webhook.'
    if exception:
        notification_message += f'\nWhen trying to send a request to the webhook service, an exception of type {type(exception).__name__} occurred:\n
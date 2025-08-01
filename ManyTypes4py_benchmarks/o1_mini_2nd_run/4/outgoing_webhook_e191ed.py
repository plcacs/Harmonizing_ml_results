import abc
import json
import logging
from contextlib import suppress
from time import perf_counter
from typing import Any, AnyStr, Callable, Dict, Optional, Type
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
from zerver.lib.users import (
    check_can_access_user,
    check_user_can_access_all_users,
    get_user_profile_by_id,
)
from zerver.models import Realm, Service, UserProfile
from zerver.models.bots import GENERIC_INTERFACE, SLACK_INTERFACE
from zerver.models.clients import get_client


class OutgoingWebhookServiceInterface(abc.ABC):
    def __init__(
        self,
        token: str,
        user_profile: UserProfile,
        service_name: str,
    ) -> None:
        self.token = token
        self.user_profile = user_profile
        self.service_name = service_name
        self.session = OutgoingSession(
            role="webhook",
            timeout=settings.OUTGOING_WEBHOOK_TIMEOUT_SECONDS,
            headers={"User-Agent": "ZulipOutgoingWebhook/" + ZULIP_VERSION},
        )

    @abc.abstractmethod
    def make_request(
        self, base_url: str, event: Dict[str, Any], realm: Realm
    ) -> Optional[Response]:
        raise NotImplementedError

    @abc.abstractmethod
    def process_success(self, response_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class GenericOutgoingWebhookService(OutgoingWebhookServiceInterface):
    @override
    def make_request(
        self, base_url: str, event: Dict[str, Any], realm: Realm
    ) -> Optional[Response]:
        """
        We send a simple version of the message to outgoing
        webhooks, since most of them really only need
        `content` and a few other fields.  We may eventually
        allow certain bots to get more information, but
        that's not a high priority.  We do send the gravatar
        info to the clients (so they don't have to compute
        it themselves).
        """
        message_dict = MessageDict.finalize_payload(
            event["message"],
            apply_markdown=False,
            client_gravatar=False,
            allow_empty_topic_name=True,
            keep_rendered_content=True,
            can_access_sender=check_user_can_access_all_users(self.user_profile)
            or check_can_access_user(
                get_user_profile_by_id(event["message"]["sender_id"]),
                self.user_profile,
            ),
            realm_host=realm.host,
            is_incoming_1_to_1=event["message"]["recipient_id"]
            == self.user_profile.recipient_id,
        )
        request_data: Dict[str, Any] = {
            "data": event["command"],
            "message": message_dict,
            "bot_email": self.user_profile.email,
            "bot_full_name": self.user_profile.full_name,
            "token": self.token,
            "trigger": event["trigger"],
        }
        return self.session.post(base_url, json=request_data)

    @override
    def process_success(
        self, response_json: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if response_json.get("response_not_required", False):
            return None
        if "response_string" in response_json:
            content = str(response_json["response_string"])
            success_data: Dict[str, Any] = dict(content=content)
            return success_data
        if "content" in response_json:
            content = str(response_json["content"])
            success_data: Dict[str, Any] = dict(content=content)
            if "widget_content" in response_json:
                success_data["widget_content"] = response_json["widget_content"]
            return success_data
        return None


class SlackOutgoingWebhookService(OutgoingWebhookServiceInterface):
    @override
    def make_request(
        self, base_url: str, event: Dict[str, Any], realm: Realm
    ) -> Optional[Response]:
        if event["message"]["type"] == "private":
            failure_message = "Slack outgoing webhooks don't support direct messages."
            fail_with_message(event, failure_message)
            return None
        request_data: list = [
            ("token", self.token),
            ("team_id", f"T{realm.id}"),
            ("team_domain", realm.host),
            ("channel_id", f"C{event['message']['stream_id']}"),
            ("channel_name", event["message"]["display_recipient"]),
            ("thread_ts", event["message"]["timestamp"]),
            ("timestamp", event["message"]["timestamp"]),
            ("user_id", f"U{event['message']['sender_id']}"),
            ("user_name", event["message"]["sender_full_name"]),
            ("text", event["command"]),
            ("trigger_word", event["trigger"]),
            ("service_id", event["user_profile_id"]),
        ]
        return self.session.post(base_url, data=request_data)

    @override
    def process_success(
        self, response_json: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if "text" in response_json:
            content = response_json["text"]
            success_data: Dict[str, Any] = dict(content=content)
            return success_data
        return None


AVAILABLE_OUTGOING_WEBHOOK_INTERFACES: Dict[str, Type[OutgoingWebhookServiceInterface]] = {
    GENERIC_INTERFACE: GenericOutgoingWebhookService,
    SLACK_INTERFACE: SlackOutgoingWebhookService,
}


def get_service_interface_class(
    interface: str,
) -> Type[OutgoingWebhookServiceInterface]:
    if interface not in AVAILABLE_OUTGOING_WEBHOOK_INTERFACES:
        return AVAILABLE_OUTGOING_WEBHOOK_INTERFACES[GENERIC_INTERFACE]
    else:
        return AVAILABLE_OUTGOING_WEBHOOK_INTERFACES[interface]


def get_outgoing_webhook_service_handler(service: Service) -> OutgoingWebhookServiceInterface:
    service_interface_class = get_service_interface_class(service.interface_name())
    service_interface = service_interface_class(
        token=service.token, user_profile=service.user_profile, service_name=service.name
    )
    return service_interface


def send_response_message(
    bot_id: int, message_info: Dict[str, Any], response_data: Dict[str, Any]
) -> None:
    """
    bot_id is the user_id of the bot sending the response

    message_info is used to address the message and should have these fields:
        type - "stream" or "private"
        display_recipient - like we have in other message events
        topic - see get_topic_from_message_info

    response_data is what the bot wants to send back and has these fields:
        content - raw Markdown content for Zulip to render

    WARNING: This function sends messages bypassing the stream access check
    for the bot - so use with caution to not call this in codepaths
    that might let someone send arbitrary messages to any stream through this.
    """
    recipient_type_name: str = message_info["type"]
    display_recipient: Any = message_info["display_recipient"]
    try:
        topic_name: Optional[str] = get_topic_from_message_info(message_info)
    except KeyError:
        topic_name = None
    bot_user: UserProfile = get_user_profile_by_id(bot_id)
    realm: Realm = bot_user.realm
    client = get_client("OutgoingWebhookResponse")
    content: str = response_data.get("content")  # type: ignore
    assert content
    widget_content: Optional[Any] = response_data.get("widget_content")
    if recipient_type_name == "stream":
        message_to: Any = [display_recipient]
    elif recipient_type_name == "private":
        message_to = [recipient["email"] for recipient in display_recipient]
    else:
        raise JsonableError(_("Invalid message type"))
    check_send_message(
        sender=bot_user,
        client=client,
        recipient_type_name=recipient_type_name,
        message_to=message_to,
        topic_name=topic_name,
        message_content=content,
        widget_content=widget_content,
        realm=realm,
        skip_stream_access_check=True,
    )


def fail_with_message(event: Dict[str, Any], failure_message: str) -> None:
    bot_id: int = event["user_profile_id"]
    message_info: Dict[str, Any] = event["message"]
    content: str = "Failure! " + failure_message
    response_data: Dict[str, Any] = dict(content=content)
    with suppress(StreamDoesNotExistError):
        send_response_message(bot_id=bot_id, message_info=message_info, response_data=response_data)


def get_message_url(event: Dict[str, Any]) -> str:
    bot_user: UserProfile = get_user_profile_by_id(event["user_profile_id"])
    message: Dict[str, Any] = event["message"]
    realm: Realm = bot_user.realm
    return near_message_url(realm=realm, message=message)


def notify_bot_owner(
    event: Dict[str, Any],
    status_code: Optional[int] = None,
    response_content: Optional[Any] = None,
    failure_message: Optional[str] = None,
    exception: Optional[Exception] = None,
) -> None:
    message_url: str = get_message_url(event)
    bot_id: int = event["user_profile_id"]
    bot: UserProfile = get_user_profile_by_id(bot_id)
    bot_owner: Optional[UserProfile] = bot.bot_owner
    assert bot_owner is not None
    notification_message: str = f'[A message]({message_url}) to your bot @_**{bot.full_name}** triggered an outgoing webhook.'
    if exception:
        notification_message += (
            f"\nWhen trying to send a request to the webhook service, an exception of type {type(exception).__name__} "
            "occurred:\n
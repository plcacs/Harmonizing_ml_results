from __future__ import annotations
import asyncio
import io
from ipaddress import ip_network
import logging
from typing import Any
import httpx
from telegram import Bot, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, ReplyKeyboardMarkup, ReplyKeyboardRemove, Update, User
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import CallbackContext, filters
from telegram.request import HTTPXRequest
import voluptuous as vol
from homeassistant.const import ATTR_COMMAND, ATTR_LATITUDE, ATTR_LONGITUDE, CONF_API_KEY, CONF_PLATFORM, CONF_URL, HTTP_BEARER_AUTHENTICATION, HTTP_DIGEST_AUTHENTICATION
from homeassistant.core import Context, HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.helpers import config_validation as cv, issue_registry as ir
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import async_get_loaded_integration
from homeassistant.util.ssl import get_default_context, get_default_no_verify_context

_LOGGER: logging.Logger

ATTR_DATA: str = 'data'
ATTR_MESSAGE: str = 'message'
ATTR_TITLE: str = 'title'
ATTR_ARGS: str = 'args'
ATTR_AUTHENTICATION: str = 'authentication'
ATTR_CALLBACK_QUERY: str = 'callback_query'
ATTR_CALLBACK_QUERY_ID: str = 'callback_query_id'
ATTR_CAPTION: str = 'caption'
ATTR_CHAT_ID: str = 'chat_id'
ATTR_CHAT_INSTANCE: str = 'chat_instance'
ATTR_DATE: str = 'date'
ATTR_DISABLE_NOTIF: str = 'disable_notification'
ATTR_DISABLE_WEB_PREV: str = 'disable_web_page_preview'
ATTR_EDITED_MSG: str = 'edited_message'
ATTR_FILE: str = 'file'
ATTR_FROM_FIRST: str = 'from_first'
ATTR_FROM_LAST: str = 'from_last'
ATTR_KEYBOARD: str = 'keyboard'
ATTR_RESIZE_KEYBOARD: str = 'resize_keyboard'
ATTR_ONE_TIME_KEYBOARD: str = 'one_time_keyboard'
ATTR_KEYBOARD_INLINE: str = 'inline_keyboard'
ATTR_MESSAGEID: str = 'message_id'
ATTR_MSG: str = 'message'
ATTR_MSGID: str = 'id'
ATTR_PARSER: str = 'parse_mode'
ATTR_PASSWORD: str = 'password'
ATTR_REPLY_TO_MSGID: str = 'reply_to_message_id'
ATTR_REPLYMARKUP: str = 'reply_markup'
ATTR_SHOW_ALERT: str = 'show_alert'
ATTR_STICKER_ID: str = 'sticker_id'
ATTR_TARGET: str = 'target'
ATTR_TEXT: str = 'text'
ATTR_URL: str = 'url'
ATTR_USER_ID: str = 'user_id'
ATTR_USERNAME: str = 'username'
ATTR_VERIFY_SSL: str = 'verify_ssl'
ATTR_TIMEOUT: str = 'timeout'
ATTR_MESSAGE_TAG: str = 'message_tag'
ATTR_CHANNEL_POST: str = 'channel_post'
ATTR_QUESTION: str = 'question'
ATTR_OPTIONS: str = 'options'
ATTR_ANSWERS: str = 'answers'
ATTR_OPEN_PERIOD: str = 'open_period'
ATTR_IS_ANONYMOUS: str = 'is_anonymous'
ATTR_ALLOWS_MULTIPLE_ANSWERS: str = 'allows_multiple_answers'
ATTR_MESSAGE_THREAD_ID: str = 'message_thread_id'
CONF_ALLOWED_CHAT_IDS: str = 'allowed_chat_ids'
CONF_PROXY_URL: str = 'proxy_url'
CONF_PROXY_PARAMS: str = 'proxy_params'
CONF_TRUSTED_NETWORKS: str = 'trusted_networks'
DOMAIN: str = 'telegram_bot'
SERVICE_SEND_MESSAGE: str = 'send_message'
SERVICE_SEND_PHOTO: str = 'send_photo'
SERVICE_SEND_STICKER: str = 'send_sticker'
SERVICE_SEND_ANIMATION: str = 'send_animation'
SERVICE_SEND_VIDEO: str = 'send_video'
SERVICE_SEND_VOICE: str = 'send_voice'
SERVICE_SEND_DOCUMENT: str = 'send_document'
SERVICE_SEND_LOCATION: str = 'send_location'
SERVICE_SEND_POLL: str = 'send_poll'
SERVICE_EDIT_MESSAGE: str = 'edit_message'
SERVICE_EDIT_CAPTION: str = 'edit_caption'
SERVICE_EDIT_REPLYMARKUP: str = 'edit_replymarkup'
SERVICE_ANSWER_CALLBACK_QUERY: str = 'answer_callback_query'
SERVICE_DELETE_MESSAGE: str = 'delete_message'
SERVICE_LEAVE_CHAT: str = 'leave_chat'
EVENT_TELEGRAM_CALLBACK: str = 'telegram_callback'
EVENT_TELEGRAM_COMMAND: str = 'telegram_command'
EVENT_TELEGRAM_TEXT: str = 'telegram_text'
EVENT_TELEGRAM_SENT: str = 'telegram_sent'
PARSER_HTML: str = 'html'
PARSER_MD: str = 'markdown'
PARSER_MD2: str = 'markdownv2'
PARSER_PLAIN_TEXT: str = 'plain_text'
DEFAULT_TRUSTED_NETWORKS: list[ip_network] = [ip_network('149.154.160.0/20'), ip_network('91.108.4.0/22')]
CONFIG_SCHEMA: vol.Schema
BASE_SERVICE_SCHEMA: vol.Schema
SERVICE_SCHEMA_SEND_MESSAGE: vol.Schema
SERVICE_SCHEMA_SEND_FILE: vol.Schema
SERVICE_SCHEMA_SEND_STICKER: vol.Schema
SERVICE_SCHEMA_SEND_LOCATION: vol.Schema
SERVICE_SCHEMA_SEND_POLL: vol.Schema
SERVICE_SCHEMA_EDIT_MESSAGE: vol.Schema
SERVICE_SCHEMA_EDIT_CAPTION: vol.Schema
SERVICE_SCHEMA_EDIT_REPLYMARKUP: vol.Schema
SERVICE_SCHEMA_ANSWER_CALLBACK_QUERY: vol.Schema
SERVICE_SCHEMA_DELETE_MESSAGE: vol.Schema
SERVICE_SCHEMA_LEAVE_CHAT: vol.Schema
SERVICE_MAP: dict[str, vol.Schema]

def _read_file_as_bytesio(file_path: str) -> io.BytesIO: ...

async def load_data(hass: HomeAssistant, url: str = None, filepath: str = None, username: str = None, password: str = None, authentication: str = None, num_retries: int = 5, verify_ssl: bool = None) -> io.BytesIO: ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool: ...

def initialize_bot(hass: HomeAssistant, p_config: ConfigType) -> Bot: ...

class TelegramNotificationService:
    def __init__(self, hass: HomeAssistant, bot: Bot, allowed_chat_ids: list[int], parser: str): ...

    def _get_msg_ids(self, msg_data: dict, chat_id: int) -> tuple: ...

    def _get_target_chat_ids(self, target: Any) -> list[int]: ...

    def _get_msg_kwargs(self, data: dict) -> dict: ...

    async def _send_msg(self, func_send: Any, msg_error: str, message_tag: str, *args_msg: Any, context: Any = None, **kwargs_msg: Any) -> Any: ...

    async def send_message(self, message: str = '', target: Any = None, context: Any = None, **kwargs: Any) -> dict[int, int]: ...

    async def delete_message(self, chat_id: int = None, context: Any = None, **kwargs: Any) -> Any: ...

    async def edit_message(self, type_edit: str, chat_id: int = None, context: Any = None, **kwargs: Any) -> Any: ...

    async def answer_callback_query(self, message: str, callback_query_id: int, show_alert: bool = False, context: Any = None, **kwargs: Any) -> None: ...

    async def send_file(self, file_type: str = SERVICE_SEND_PHOTO, target: Any = None, context: Any = None, **kwargs: Any) -> dict[int, int]: ...

    async def send_sticker(self, target: Any = None, context: Any = None, **kwargs: Any) -> dict[int, int]: ...

    async def send_location(self, latitude: str, longitude: str, target: Any = None, context: Any = None, **kwargs: Any) -> dict[int, int]: ...

    async def send_poll(self, question: str, options: list[str], is_anonymous: bool, allows_multiple_answers: bool, target: Any = None, context: Any = None, **kwargs: Any) -> dict[int, int]: ...

    async def leave_chat(self, chat_id: int = None, context: Any = None) -> Any: ...

class BaseTelegramBotEntity:
    def __init__(self, hass: HomeAssistant, config: ConfigType): ...

    async def handle_update(self, update: Update, context: Any) -> bool: ...

    @staticmethod
    def _get_command_event_data(command_text: str) -> dict: ...

    def _get_message_event_data(self, message: Message) -> tuple: ...

    def _get_user_event_data(self, user: User) -> dict: ...

    def _get_callback_query_event_data(self, callback_query: CallbackQuery) -> tuple: ...

    def authorize_update(self, update: Update) -> bool: ...

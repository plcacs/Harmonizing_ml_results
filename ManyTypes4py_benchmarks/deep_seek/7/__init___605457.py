"""Support to send and receive Telegram messages."""
from __future__ import annotations
import asyncio
import io
from ipaddress import IPv4Network, IPv6Network
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast
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
from ssl import SSLContext

_LOGGER = logging.getLogger(__name__)

ATTR_DATA = 'data'
ATTR_MESSAGE = 'message'
ATTR_TITLE = 'title'
ATTR_ARGS = 'args'
ATTR_AUTHENTICATION = 'authentication'
ATTR_CALLBACK_QUERY = 'callback_query'
ATTR_CALLBACK_QUERY_ID = 'callback_query_id'
ATTR_CAPTION = 'caption'
ATTR_CHAT_ID = 'chat_id'
ATTR_CHAT_INSTANCE = 'chat_instance'
ATTR_DATE = 'date'
ATTR_DISABLE_NOTIF = 'disable_notification'
ATTR_DISABLE_WEB_PREV = 'disable_web_page_preview'
ATTR_EDITED_MSG = 'edited_message'
ATTR_FILE = 'file'
ATTR_FROM_FIRST = 'from_first'
ATTR_FROM_LAST = 'from_last'
ATTR_KEYBOARD = 'keyboard'
ATTR_RESIZE_KEYBOARD = 'resize_keyboard'
ATTR_ONE_TIME_KEYBOARD = 'one_time_keyboard'
ATTR_KEYBOARD_INLINE = 'inline_keyboard'
ATTR_MESSAGEID = 'message_id'
ATTR_MSG = 'message'
ATTR_MSGID = 'id'
ATTR_PARSER = 'parse_mode'
ATTR_PASSWORD = 'password'
ATTR_REPLY_TO_MSGID = 'reply_to_message_id'
ATTR_REPLYMARKUP = 'reply_markup'
ATTR_SHOW_ALERT = 'show_alert'
ATTR_STICKER_ID = 'sticker_id'
ATTR_TARGET = 'target'
ATTR_TEXT = 'text'
ATTR_URL = 'url'
ATTR_USER_ID = 'user_id'
ATTR_USERNAME = 'username'
ATTR_VERIFY_SSL = 'verify_ssl'
ATTR_TIMEOUT = 'timeout'
ATTR_MESSAGE_TAG = 'message_tag'
ATTR_CHANNEL_POST = 'channel_post'
ATTR_QUESTION = 'question'
ATTR_OPTIONS = 'options'
ATTR_ANSWERS = 'answers'
ATTR_OPEN_PERIOD = 'open_period'
ATTR_IS_ANONYMOUS = 'is_anonymous'
ATTR_ALLOWS_MULTIPLE_ANSWERS = 'allows_multiple_answers'
ATTR_MESSAGE_THREAD_ID = 'message_thread_id'
CONF_ALLOWED_CHAT_IDS = 'allowed_chat_ids'
CONF_PROXY_URL = 'proxy_url'
CONF_PROXY_PARAMS = 'proxy_params'
CONF_TRUSTED_NETWORKS = 'trusted_networks'
DOMAIN = 'telegram_bot'
SERVICE_SEND_MESSAGE = 'send_message'
SERVICE_SEND_PHOTO = 'send_photo'
SERVICE_SEND_STICKER = 'send_sticker'
SERVICE_SEND_ANIMATION = 'send_animation'
SERVICE_SEND_VIDEO = 'send_video'
SERVICE_SEND_VOICE = 'send_voice'
SERVICE_SEND_DOCUMENT = 'send_document'
SERVICE_SEND_LOCATION = 'send_location'
SERVICE_SEND_POLL = 'send_poll'
SERVICE_EDIT_MESSAGE = 'edit_message'
SERVICE_EDIT_CAPTION = 'edit_caption'
SERVICE_EDIT_REPLYMARKUP = 'edit_replymarkup'
SERVICE_ANSWER_CALLBACK_QUERY = 'answer_callback_query'
SERVICE_DELETE_MESSAGE = 'delete_message'
SERVICE_LEAVE_CHAT = 'leave_chat'
EVENT_TELEGRAM_CALLBACK = 'telegram_callback'
EVENT_TELEGRAM_COMMAND = 'telegram_command'
EVENT_TELEGRAM_TEXT = 'telegram_text'
EVENT_TELEGRAM_SENT = 'telegram_sent'
PARSER_HTML = 'html'
PARSER_MD = 'markdown'
PARSER_MD2 = 'markdownv2'
PARSER_PLAIN_TEXT = 'plain_text'
DEFAULT_TRUSTED_NETWORKS: List[Union[IPv4Network, IPv6Network]] = [ip_network('149.154.160.0/20'), ip_network('91.108.4.0/22')]
CONFIG_SCHEMA = vol.Schema({DOMAIN: vol.All(cv.ensure_list, [vol.Schema({vol.Required(CONF_PLATFORM): vol.In(('broadcast', 'polling', 'webhooks')), vol.Required(CONF_API_KEY): cv.string, vol.Required(CONF_ALLOWED_CHAT_IDS): vol.All(cv.ensure_list, [vol.Coerce(int)]), vol.Optional(ATTR_PARSER, default=PARSER_MD): cv.string, vol.Optional(CONF_PROXY_URL): cv.string, vol.Optional(CONF_PROXY_PARAMS): dict, vol.Optional(CONF_URL): cv.url, vol.Optional(CONF_TRUSTED_NETWORKS, default=DEFAULT_TRUSTED_NETWORKS): vol.All(cv.ensure_list, [ip_network])})])}, extra=vol.ALLOW_EXTRA)
BASE_SERVICE_SCHEMA = vol.Schema({vol.Optional(ATTR_TARGET): vol.All(cv.ensure_list, [vol.Coerce(int)]), vol.Optional(ATTR_PARSER): cv.string, vol.Optional(ATTR_DISABLE_NOTIF): cv.boolean, vol.Optional(ATTR_DISABLE_WEB_PREV): cv.boolean, vol.Optional(ATTR_RESIZE_KEYBOARD): cv.boolean, vol.Optional(ATTR_ONE_TIME_KEYBOARD): cv.boolean, vol.Optional(ATTR_KEYBOARD): vol.All(cv.ensure_list, [cv.string]), vol.Optional(ATTR_KEYBOARD_INLINE): cv.ensure_list, vol.Optional(ATTR_TIMEOUT): cv.positive_int, vol.Optional(ATTR_MESSAGE_TAG): cv.string, vol.Optional(ATTR_MESSAGE_THREAD_ID): vol.Coerce(int)}, extra=vol.ALLOW_EXTRA)
SERVICE_SCHEMA_SEND_MESSAGE = BASE_SERVICE_SCHEMA.extend({vol.Required(ATTR_MESSAGE): cv.string, vol.Optional(ATTR_TITLE): cv.string})
SERVICE_SCHEMA_SEND_FILE = BASE_SERVICE_SCHEMA.extend({vol.Optional(ATTR_URL): cv.string, vol.Optional(ATTR_FILE): cv.string, vol.Optional(ATTR_CAPTION): cv.string, vol.Optional(ATTR_USERNAME): cv.string, vol.Optional(ATTR_PASSWORD): cv.string, vol.Optional(ATTR_AUTHENTICATION): cv.string, vol.Optional(ATTR_VERIFY_SSL): cv.boolean})
SERVICE_SCHEMA_SEND_STICKER = SERVICE_SCHEMA_SEND_FILE.extend({vol.Optional(ATTR_STICKER_ID): cv.string})
SERVICE_SCHEMA_SEND_LOCATION = BASE_SERVICE_SCHEMA.extend({vol.Required(ATTR_LONGITUDE): cv.string, vol.Required(ATTR_LATITUDE): cv.string})
SERVICE_SCHEMA_SEND_POLL = vol.Schema({vol.Optional(ATTR_TARGET): vol.All(cv.ensure_list, [vol.Coerce(int)]), vol.Required(ATTR_QUESTION): cv.string, vol.Required(ATTR_OPTIONS): vol.All(cv.ensure_list, [cv.string]), vol.Optional(ATTR_OPEN_PERIOD): cv.positive_int, vol.Optional(ATTR_IS_ANONYMOUS, default=True): cv.boolean, vol.Optional(ATTR_ALLOWS_MULTIPLE_ANSWERS, default=False): cv.boolean, vol.Optional(ATTR_DISABLE_NOTIF): cv.boolean, vol.Optional(ATTR_TIMEOUT): cv.positive_int, vol.Optional(ATTR_MESSAGE_THREAD_ID): vol.Coerce(int)})
SERVICE_SCHEMA_EDIT_MESSAGE = SERVICE_SCHEMA_SEND_MESSAGE.extend({vol.Required(ATTR_MESSAGEID): vol.Any(cv.positive_int, vol.All(cv.string, 'last')), vol.Required(ATTR_CHAT_ID): vol.Coerce(int)})
SERVICE_SCHEMA_EDIT_CAPTION = vol.Schema({vol.Required(ATTR_MESSAGEID): vol.Any(cv.positive_int, vol.All(cv.string, 'last')), vol.Required(ATTR_CHAT_ID): vol.Coerce(int), vol.Required(ATTR_CAPTION): cv.string, vol.Optional(ATTR_KEYBOARD_INLINE): cv.ensure_list}, extra=vol.ALLOW_EXTRA)
SERVICE_SCHEMA_EDIT_REPLYMARKUP = vol.Schema({vol.Required(ATTR_MESSAGEID): vol.Any(cv.positive_int, vol.All(cv.string, 'last')), vol.Required(ATTR_CHAT_ID): vol.Coerce(int), vol.Required(ATTR_KEYBOARD_INLINE): cv.ensure_list}, extra=vol.ALLOW_EXTRA)
SERVICE_SCHEMA_ANSWER_CALLBACK_QUERY = vol.Schema({vol.Required(ATTR_MESSAGE): cv.string, vol.Required(ATTR_CALLBACK_QUERY_ID): vol.Coerce(int), vol.Optional(ATTR_SHOW_ALERT): cv.boolean}, extra=vol.ALLOW_EXTRA)
SERVICE_SCHEMA_DELETE_MESSAGE = vol.Schema({vol.Required(ATTR_CHAT_ID): vol.Coerce(int), vol.Required(ATTR_MESSAGEID): vol.Any(cv.positive_int, vol.All(cv.string, 'last'))}, extra=vol.ALLOW_EXTRA)
SERVICE_SCHEMA_LEAVE_CHAT = vol.Schema({vol.Required(ATTR_CHAT_ID): vol.Coerce(int)})
SERVICE_MAP = {SERVICE_SEND_MESSAGE: SERVICE_SCHEMA_SEND_MESSAGE, SERVICE_SEND_PHOTO: SERVICE_SCHEMA_SEND_FILE, SERVICE_SEND_STICKER: SERVICE_SCHEMA_SEND_STICKER, SERVICE_SEND_ANIMATION: SERVICE_SCHEMA_SEND_FILE, SERVICE_SEND_VIDEO: SERVICE_SCHEMA_SEND_FILE, SERVICE_SEND_VOICE: SERVICE_SCHEMA_SEND_FILE, SERVICE_SEND_DOCUMENT: SERVICE_SCHEMA_SEND_FILE, SERVICE_SEND_LOCATION: SERVICE_SCHEMA_SEND_LOCATION, SERVICE_SEND_POLL: SERVICE_SCHEMA_SEND_POLL, SERVICE_EDIT_MESSAGE: SERVICE_SCHEMA_EDIT_MESSAGE, SERVICE_EDIT_CAPTION: SERVICE_SCHEMA_EDIT_CAPTION, SERVICE_EDIT_REPLYMARKUP: SERVICE_SCHEMA_EDIT_REPLYMARKUP, SERVICE_ANSWER_CALLBACK_QUERY: SERVICE_SCHEMA_ANSWER_CALLBACK_QUERY, SERVICE_DELETE_MESSAGE: SERVICE_SCHEMA_DELETE_MESSAGE, SERVICE_LEAVE_CHAT: SERVICE_SCHEMA_LEAVE_CHAT}

def _read_file_as_bytesio(file_path: str) -> io.BytesIO:
    """Read a file and return it as a BytesIO object."""
    with open(file_path, 'rb') as file:
        data = io.BytesIO(file.read())
        data.name = file_path
        return data

async def load_data(hass: HomeAssistant, url: Optional[str] = None, filepath: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None, authentication: Optional[str] = None, num_retries: int = 5, verify_ssl: Optional[bool] = None) -> Optional[io.BytesIO]:
    """Load data into ByteIO/File container from a source."""
    try:
        if url is not None:
            params: Dict[str, Any] = {}
            headers: Dict[str, str] = {}
            if authentication == HTTP_BEARER_AUTHENTICATION and password is not None:
                headers = {'Authorization': f'Bearer {password}'}
            elif username is not None and password is not None:
                if authentication == HTTP_DIGEST_AUTHENTICATION:
                    params['auth'] = httpx.DigestAuth(username, password)
                else:
                    params['auth'] = httpx.BasicAuth(username, password)
            if verify_ssl is not None:
                params['verify'] = verify_ssl
            retry_num = 0
            async with httpx.AsyncClient(timeout=15, headers=headers, **params) as client:
                while retry_num < num_retries:
                    req = await client.get(url)
                    if req.status_code != 200:
                        _LOGGER.warning('Status code %s (retry #%s) loading %s', req.status_code, retry_num + 1, url)
                    else:
                        data = io.BytesIO(req.content)
                        if data.read():
                            data.seek(0)
                            data.name = url
                            return data
                        _LOGGER.warning('Empty data (retry #%s) in %s)', retry_num + 1, url)
                    retry_num += 1
                    if retry_num < num_retries:
                        await asyncio.sleep(1)
                _LOGGER.warning("Can't load data in %s after %s retries", url, retry_num)
        elif filepath is not None:
            if hass.config.is_allowed_path(filepath):
                return await hass.async_add_executor_job(_read_file_as_bytesio, filepath)
            _LOGGER.warning("'%s' are not secure to load data from!", filepath)
        else:
            _LOGGER.warning("Can't load data. No data found in params!")
    except (OSError, TypeError) as error:
        _LOGGER.error("Can't load data into ByteIO: %s", error)
    return None

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Telegram bot component."""
    domain_config = config[DOMAIN]
    if not domain_config:
        return False
    platforms = await async_get_loaded_integration(hass, DOMAIN).async_get_platforms({p_config[CONF_PLATFORM] for p_config in domain_config})
    for p_config in domain_config:
        bot = await hass.async_add_executor_job(initialize_bot, hass, p_config)
        p_type = p_config[CONF_PLATFORM]
        platform = platforms[p_type]
        _LOGGER.debug('Setting up %s.%s', DOMAIN, p_type)
        try:
            receiver_service = await platform.async_setup_platform(hass, bot, p_config)
            if receiver_service is False:
                _LOGGER.error('Failed to initialize Telegram bot %s', p_type)
                return False
        except Exception:
            _LOGGER.exception('Error setting up platform %s', p_type)
            return False
        notify_service = TelegramNotificationService(hass, bot, p_config.get(CONF_ALLOWED_CHAT_IDS), p_config.get(ATTR_PARSER))

    async def async_send_telegram_message(service: ServiceCall) -> Optional[ServiceResponse]:
        """Handle sending Telegram Bot message service calls."""
        msgtype = service.service
        kwargs = dict(service.data)
        _LOGGER.debug('New telegram message %s: %s', msgtype, kwargs)
        messages = None
        if msgtype == SERVICE_SEND_MESSAGE:
            messages = await notify_service.send_message(context=service.context, **kwargs)
        elif msgtype in [SERVICE_SEND_PHOTO, SERVICE_SEND_ANIMATION, SERVICE_SEND_VIDEO, SERVICE_SEND_VOICE, SERVICE_SEND_DOCUMENT]:
            messages = await notify_service.send_file(msgtype, context=service.context, **kwargs)
        elif msgtype == SERVICE_SEND_STICKER:
            messages = await notify_service.send_sticker(context=service.context, **kwargs)
        elif msgtype == SERVICE_SEND_LOCATION:
            messages = await notify_service.send_location(context=service.context, **kwargs)
        elif msgtype == SERVICE_SEND_POLL:
            messages = await notify_service.send_poll(context=service.context, **kwargs)
        elif msgtype == SERVICE_ANSWER_CALLBACK_QUERY:
            await notify_service.answer_callback_query(context=service.context, **kwargs)
        elif msgtype == SERVICE_DELETE_MESSAGE:
            await notify_service.delete_message(context=service.context, **kwargs)
        else:
            await notify_service.edit_message(msgtype, context=service.context, **kwargs)
        if service.return_response and messages:
            return {'chats': [{'chat_id': cid, 'message_id': mid} for cid, mid in messages.items()]}
        return None
    for service_notif, schema in SERVICE_MAP.items():
        supports_response = SupportsResponse.NONE
        if service_notif in [SERVICE_SEND_MESSAGE, SERVICE_SEND_PHOTO, SERVICE_SEND_ANIMATION, SERVICE_SEND_VIDEO, SERVICE_SEND_VOICE, SERVICE_SEND_DOCUMENT, SERVICE_SEND_STICKER, SERVICE_SEND_LOCATION, SERVICE_SEND_POLL]:
            supports_response = SupportsResponse.OPTIONAL
       
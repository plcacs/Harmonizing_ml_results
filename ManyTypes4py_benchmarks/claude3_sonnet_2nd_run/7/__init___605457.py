"""Support to send and receive Telegram messages."""
from __future__ import annotations
import asyncio
import io
from ipaddress import ip_network
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
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
DEFAULT_TRUSTED_NETWORKS = [ip_network('149.154.160.0/20'), ip_network('91.108.4.0/22')]
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

async def load_data(
    hass: HomeAssistant, 
    url: Optional[str] = None, 
    filepath: Optional[str] = None, 
    username: Optional[str] = None, 
    password: Optional[str] = None, 
    authentication: Optional[str] = None, 
    num_retries: int = 5, 
    verify_ssl: Optional[bool] = None
) -> Optional[io.BytesIO]:
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
        hass.services.async_register(DOMAIN, service_notif, async_send_telegram_message, schema=schema, supports_response=supports_response)
    return True

def initialize_bot(hass: HomeAssistant, p_config: Dict[str, Any]) -> Bot:
    """Initialize telegram bot with proxy support."""
    api_key = p_config[CONF_API_KEY]
    proxy_url = p_config.get(CONF_PROXY_URL)
    proxy_params = p_config.get(CONF_PROXY_PARAMS)
    if proxy_url is not None:
        auth = None
        if proxy_params is None:
            proxy_params = {}
        elif 'username' in proxy_params and 'password' in proxy_params:
            auth = (proxy_params.pop('username'), proxy_params.pop('password'))
            ir.create_issue(hass, DOMAIN, 'proxy_params_auth_deprecation', breaks_in_ha_version='2024.10.0', is_persistent=False, is_fixable=False, severity=ir.IssueSeverity.WARNING, translation_placeholders={'proxy_params': CONF_PROXY_PARAMS, 'proxy_url': CONF_PROXY_URL, 'telegram_bot': 'Telegram bot'}, translation_key='proxy_params_auth_deprecation', learn_more_url='https://github.com/home-assistant/core/pull/112778')
        else:
            ir.create_issue(hass, DOMAIN, 'proxy_params_deprecation', breaks_in_ha_version='2024.10.0', is_persistent=False, is_fixable=False, severity=ir.IssueSeverity.WARNING, translation_placeholders={'proxy_params': CONF_PROXY_PARAMS, 'proxy_url': CONF_PROXY_URL, 'httpx': 'httpx', 'telegram_bot': 'Telegram bot'}, translation_key='proxy_params_deprecation', learn_more_url='https://github.com/home-assistant/core/pull/112778')
        proxy = httpx.Proxy(proxy_url, auth=auth, **proxy_params)
        request = HTTPXRequest(connection_pool_size=8, proxy=proxy)
    else:
        request = HTTPXRequest(connection_pool_size=8)
    return Bot(token=api_key, request=request)

class TelegramNotificationService:
    """Implement the notification services for the Telegram Bot domain."""

    def __init__(
        self, 
        hass: HomeAssistant, 
        bot: Bot, 
        allowed_chat_ids: List[int], 
        parser: Optional[str]
    ) -> None:
        """Initialize the service."""
        self.allowed_chat_ids = allowed_chat_ids
        self._default_user = self.allowed_chat_ids[0]
        self._last_message_id: Dict[int, Optional[int]] = {user: None for user in self.allowed_chat_ids}
        self._parsers = {PARSER_HTML: ParseMode.HTML, PARSER_MD: ParseMode.MARKDOWN, PARSER_MD2: ParseMode.MARKDOWN_V2, PARSER_PLAIN_TEXT: None}
        self._parse_mode = self._parsers.get(parser)
        self.bot = bot
        self.hass = hass

    def _get_msg_ids(self, msg_data: Dict[str, Any], chat_id: int) -> Tuple[Optional[int], Optional[str]]:
        """Get the message id to edit.

        This can be one of (message_id, inline_message_id) from a msg dict,
        returning a tuple.
        **You can use 'last' as message_id** to edit
        the message last sent in the chat_id.
        """
        message_id = inline_message_id = None
        if ATTR_MESSAGEID in msg_data:
            message_id = msg_data[ATTR_MESSAGEID]
            if isinstance(message_id, str) and message_id == 'last' and (self._last_message_id[chat_id] is not None):
                message_id = self._last_message_id[chat_id]
        else:
            inline_message_id = msg_data['inline_message_id']
        return (message_id, inline_message_id)

    def _get_target_chat_ids(self, target: Optional[Union[int, List[int]]]) -> List[int]:
        """Validate chat_id targets or return default target (first).

        :param target: optional list of integers ([12234, -12345])
        :return list of chat_id targets (integers)
        """
        if target is not None:
            if isinstance(target, int):
                target = [target]
            chat_ids = [t for t in target if t in self.allowed_chat_ids]
            if chat_ids:
                return chat_ids
            _LOGGER.warning('Disallowed targets: %s, using default: %s', target, self._default_user)
        return [self._default_user]

    def _get_msg_kwargs(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get parameters in message data kwargs."""

        def _make_row_inline_keyboard(row_keyboard: Union[str, List[Tuple[str, str]]]) -> List[InlineKeyboardButton]:
            """Make a list of InlineKeyboardButtons.

            It can accept:
              - a list of tuples like:
                `[(text_b1, data_callback_b1),
                (text_b2, data_callback_b2), ...]
              - a string like: `/cmd1, /cmd2, /cmd3`
              - or a string like: `text_b1:/cmd1, text_b2:/cmd2`
              - also supports urls instead of callback commands
            """
            buttons: List[InlineKeyboardButton] = []
            if isinstance(row_keyboard, str):
                for key in row_keyboard.split(','):
                    if ':/' in key:
                        if key.startswith('https://'):
                            label = key.split(',')[0]
                            url = key[len(label) + 1:]
                            buttons.append(InlineKeyboardButton(label, url=url))
                        else:
                            label = key.split(':/')[0]
                            command = key[len(label) + 1:]
                            buttons.append(InlineKeyboardButton(label, callback_data=command))
                    else:
                        label = key.strip()[1:].upper()
                        buttons.append(InlineKeyboardButton(label, callback_data=key))
            elif isinstance(row_keyboard, list):
                for entry in row_keyboard:
                    text_btn, data_btn = entry
                    if data_btn.startswith('https://'):
                        buttons.append(InlineKeyboardButton(text_btn, url=data_btn))
                    else:
                        buttons.append(InlineKeyboardButton(text_btn, callback_data=data_btn))
            else:
                raise TypeError(str(row_keyboard))
            return buttons
        params: Dict[str, Any] = {ATTR_PARSER: self._parse_mode, ATTR_DISABLE_NOTIF: False, ATTR_DISABLE_WEB_PREV: None, ATTR_REPLY_TO_MSGID: None, ATTR_REPLYMARKUP: None, ATTR_TIMEOUT: None, ATTR_MESSAGE_TAG: None, ATTR_MESSAGE_THREAD_ID: None}
        if data is not None:
            if ATTR_PARSER in data:
                params[ATTR_PARSER] = self._parsers.get(data[ATTR_PARSER], self._parse_mode)
            if ATTR_TIMEOUT in data:
                params[ATTR_TIMEOUT] = data[ATTR_TIMEOUT]
            if ATTR_DISABLE_NOTIF in data:
                params[ATTR_DISABLE_NOTIF] = data[ATTR_DISABLE_NOTIF]
            if ATTR_DISABLE_WEB_PREV in data:
                params[ATTR_DISABLE_WEB_PREV] = data[ATTR_DISABLE_WEB_PREV]
            if ATTR_REPLY_TO_MSGID in data:
                params[ATTR_REPLY_TO_MSGID] = data[ATTR_REPLY_TO_MSGID]
            if ATTR_MESSAGE_TAG in data:
                params[ATTR_MESSAGE_TAG] = data[ATTR_MESSAGE_TAG]
            if ATTR_MESSAGE_THREAD_ID in data:
                params[ATTR_MESSAGE_THREAD_ID] = data[ATTR_MESSAGE_THREAD_ID]
            if ATTR_KEYBOARD in data:
                keys = data.get(ATTR_KEYBOARD)
                keys = keys if isinstance(keys, list) else [keys]
                if keys:
                    params[ATTR_REPLYMARKUP] = ReplyKeyboardMarkup([[key.strip() for key in row.split(',')] for row in keys], resize_keyboard=data.get(ATTR_RESIZE_KEYBOARD, False), one_time_keyboard=data.get(ATTR_ONE_TIME_KEYBOARD, False))
                else:
                    params[ATTR_REPLYMARKUP] = ReplyKeyboardRemove(True)
            elif ATTR_KEYBOARD_INLINE in data:
                keys = data.get(ATTR_KEYBOARD_INLINE)
                keys = keys if isinstance(keys, list) else [keys]
                params[ATTR_REPLYMARKUP] = InlineKeyboardMarkup([_make_row_inline_keyboard(row) for row in keys])
        return params

    async def _send_msg(
        self, 
        func_send: Any, 
        msg_error: str, 
        message_tag: Optional[str], 
        *args_msg: Any, 
        context: Optional[Context] = None, 
        **kwargs_msg: Any
    ) -> Any:
        """Send one message."""
        try:
            out = await func_send(*args_msg, **kwargs_msg)
            if not isinstance(out, bool) and hasattr(out, ATTR_MESSAGEID):
                chat_id = out.chat_id
                message_id = out[ATTR_MESSAGEID]
                self._last_message_id[chat_id] = message_id
                _LOGGER.debug('Last message ID: %s (from chat_id %s)', self._last_message_id, chat_id)
                event_data: Dict[str, Any] = {ATTR_CHAT_ID: chat_id, ATTR_MESSAGEID: message_id}
                if message_tag is not None:
                    event_data[ATTR_MESSAGE_TAG] = message_tag
                if kwargs_msg.get(ATTR_MESSAGE_THREAD_ID) is not None:
                    event_data[ATTR_MESSAGE_THREAD_ID] = kwargs_msg[ATTR_MESSAGE_THREAD_ID]
                self.hass.bus.async_fire(EVENT_TELEGRAM_SENT, event_data, context=context)
            elif not isinstance(out, bool):
                _LOGGER.warning('Update last message: out_type:%s, out=%s', type(out), out)
        except TelegramError as exc:
            _LOGGER.error('%s: %s. Args: %s, kwargs: %s', msg_error, exc, args_msg, kwargs_msg)
            return None
        return out

    async def send_message(
        self, 
        message: str = '', 
        target: Optional[Union[int, List[int]]] = None, 
        context: Optional[Context] = None, 
        **kwargs: Any
    ) -> Dict[int, int]:
        """Send a message to one or multiple pre-allowed chat IDs."""
        title = kwargs.get(ATTR_TITLE)
        text = f'{title}\n{message}' if title else message
        params = self._get_msg_kwargs(kwargs)
        msg_ids: Dict[int, int] = {}
        for chat_id in self._get_target_chat_ids(target):
            _LOGGER.debug('Send message in chat ID %s with params: %s', chat_id, params)
            msg = await self._send_msg(self.bot.send_message, 'Error sending message', params[ATTR_MESSAGE_TAG], chat_id, text, parse_mode=params[ATTR_PARSER], disable_web_page_preview=params[ATTR_DISABLE_WEB_PREV], disable_notification=params[ATTR_DISABLE_NOTIF], reply_to_message_id=params[ATTR_REPLY_TO_MSGID], reply_markup=params[ATTR_REPLYMARKUP], read_timeout=params[ATTR_TIMEOUT], message_thread_id=params[ATTR_MESSAGE_THREAD_ID], context=context)
            if msg is not None:
                msg_ids[chat_id] = msg.id
        return msg_ids

    async def delete_message(
        self, 
        chat_id: Optional[Union[int, List[int]]] = None, 
        context: Optional[Context] = None, 
        **kwargs: Any
    ) -> Optional[bool]:
        """Delete a previously sent message."""
        chat_id = self._get_target_chat_ids(chat_id)[0]
        message_id, _ = self._get_msg_ids(kwargs, chat_id)
        _LOGGER.debug('Delete message %s in chat ID %s', message_id, chat_id)
        deleted = await self._send_msg(self.bot.delete_message, 'Error deleting message', None, chat_id, message_id, context=context)
        if self._last_message_id[chat_id] is not None:
            self._last_message_id[chat_id] -= 1
        return deleted

    async def edit_message(
        self, 
        type_edit: str, 
        chat_id: Optional[Union[int, List[int]]] = None, 
        context: Optional[Context] = None, 
        **kwargs: Any
    ) -> Optional[Message]:
        """Edit a previously sent message."""
        chat_id = self._get_target_chat_ids(chat_id)[0]
        message_id, inline_message_id = self._get_msg_ids(kwargs, chat_id)
        params = self._get_msg_kwargs(kwargs)
        _LOGGER.debug('Edit message %s in chat ID %s with params: %s', message_id or inline_message_id, chat_id, params)
        if type_edit == SERVICE_EDIT_MESSAGE:
            message = kwargs.get(ATTR_MESSAGE)
            title = kwargs.get(ATTR_TITLE)
            text = f'{title}\n{message}' if title else message
            _LOGGER.debug('Editing message with ID %s', message_id or inline_message_id)
            return await self._send_msg(self.bot.edit_message_text, 'Error editing text message', params[ATTR_MESSAGE_TAG], text, chat_id=chat_id, message_id=message_id, inline_message_id=inline_message_id, parse_mode=params[ATTR_PARSER], disable_web_page_preview=params[ATTR_DISABLE_WEB_PREV], reply_markup=params[ATTR_REPLYMARKUP], read_timeout=params[ATTR_TIMEOUT], context=context)
        if type_edit == SERVICE_EDIT_CAPTION:
            return await self._send_msg(self.bot.edit_message_caption, 'Error editing message attributes', params[ATTR_MESSAGE_TAG], chat_id=chat_id, message_id=message_id, inline_message_id=inline_message_id, caption=kwargs.get(ATTR_CAPTION), reply_markup=params[ATTR_REPLYMARKUP], read_timeout=params[ATTR_TIMEOUT], parse_mode=params[ATTR_PARSER], context=context)
        return await self._send_msg(self.bot.edit_message_reply_markup, 'Error editing message attributes', params[ATTR_MESSAGE_TAG], chat_id=chat_id, message_id=message_id, inline_message_id=inline_message_id, reply_markup=params[ATTR_REPLYMARKUP], read_timeout=params[ATTR_TIMEOUT], context=context)

    async def answer_callback_query(
        self, 
        message: str, 
        callback_query_id: int, 
        show_alert: bool = False, 
        context: Optional[Context] = None, 
        **kwargs: Any
    ) -> None:
        """Answer a callback originated with a press in an inline keyboard."""
        params = self._get_msg_kwargs(kwargs)
        _LOGGER.debug('Answer callback query with callback ID %s: %s, alert: %s', callback_query_id, message, show_alert)
        await self._send_msg(self.bot.answer_callback_query, 'Error sending answer callback query', params[ATTR_MESSAGE_TAG], callback_query_id, text=message, show_alert=show_alert, read_timeout=params[ATTR_TIMEOUT], context=context)

    async def send_file(
        self, 
        file_type: str = SERVICE_SEND_PHOTO, 
        target: Optional[Union[int, List[int]]] = None, 
        context: Optional[Context] = None, 
        **kwargs: Any
    ) -> Dict[int, int]:
        """Send a photo, sticker, video, or document."""
        params = self._get_msg_kwargs(kwargs)
        file_content = await load_data(self.hass, url=kwargs.get(ATTR_URL), filepath=kwargs.get(ATTR_FILE), username=kwargs.get(ATTR_USERNAME), password=kwargs.get(ATTR_PASSWORD), authentication=kwargs.get(ATTR_AUTHENTICATION), verify_ssl=get_default_context() if kwargs.get(ATTR_VERIFY_SSL, False) else get_default_no_verify_context())
        msg_ids: Dict[int, int] = {}
        if file_content:
            for chat_id in self._get_target_chat_ids(target):
                _LOGGER.debug('Sending file to chat ID %s', chat_id)
                if file_type == SERVICE_SEND_PHOTO:
                    msg = await self._send_msg(self.bot.send_photo, 'Error sending photo', params[ATTR_MESSAGE_TAG], chat_id=chat_id, photo=file_content, caption=kwargs.get(ATTR_CAPTION), disable_notification=params[ATTR_DISABLE_NOTIF], reply_to_message_id=params[ATTR_REPLY_TO_MSGID], reply_markup=params[ATTR_REPLYMARKUP], read_timeout=params[ATTR_TIMEOUT], parse_mode=params[ATTR_PARSER], message_thread_id=params[ATTR_MESSAGE_THREAD_ID], context=context)
                elif file_type == SERVICE_SEND_STICKER:
                    msg = await self._send_msg(self.bot.send_sticker, 'Error sending sticker', params[ATTR_MESSAGE_TAG], chat_id=chat_id, sticker=file_content, disable_notification=params[ATTR_DISABLE_NOTIF], reply_to_message_id=params[ATTR_REPLY_TO_MSGID], reply_markup=params[ATTR_REPLYMARKUP], read_timeout=params[ATTR_TIMEOUT], message_thread_id=params[ATTR_MESSAGE_THREAD_ID], context=context)
                elif file_type == SERVICE_SEND_VIDEO:
                    msg = await self._send_msg(self.bot.send_video, 'Error sending video', params[ATTR_MESSAGE_TAG], chat_id=chat_id, video=file_content, caption=kwargs.get(ATTR_CAPTION), disable_notification=params[ATTR_DISABLE_NOTIF], reply_to_message_id=params[ATTR_REPLY_TO_MSGID], reply_markup=params[ATTR_REPLYMARKUP], read_timeout=params[ATTR_TIMEOUT], parse_mode=params[ATTR_PARSER], message_thread_id=params[ATTR_MESSAGE_THREAD_ID], context=context)
                elif file_type == SERVICE_SEND_DOCUMENT:
                    msg = await self._send_msg(self.bot.send_document, 'Error sending document', params[ATTR_MESSAGE_TAG], chat_id=chat_id, document=file_content, caption=kwargs.get(ATTR_CAPTION), disable_notification=params[ATTR_DISABLE_NOTIF], reply_to_message_id=params[ATTR_REPLY_TO_MSGID], reply_markup=params[ATTR_REPLYMARKUP], read_timeout=params[ATTR_TIMEOUT], parse_mode=params[ATTR_PARSER], message_thread_id=params[ATTR_MESSAGE_THREAD_ID], context=context)
                elif file_type == SERVICE_SEND_VOICE:
                    msg = await self._send_msg(self.bot.send_voice, 'Error sending voice', params[ATTR_MESSAGE_TAG], chat_id=chat_id, voice=file_content, caption=kwargs.get(ATTR_CAPTION), disable_notification=params[ATTR_DISABLE_NOTIF], reply_to_message_id=params[ATTR_REPLY_TO_MSGID], reply_markup=params[ATTR_REPLYMARKUP], read_timeout=params[ATTR_TIMEOUT], message_thread_id=params[ATTR_MESSAGE_THREAD_ID], context=context)
                elif file_type == SERVICE_SEND_ANIMATION:
                    msg = await self._send_msg(self.bot.send_animation, 'Error sending animation', params[ATTR_MESSAGE_TAG], chat_id=chat_id, animation=file_content, caption=kwargs.get(ATTR_CAPTION), disable_notification=params[ATTR_DISABLE_NOTIF], reply_to_message_id=params[ATTR_REPLY_TO_MSGID], reply_markup=params[ATTR_REPLYMARKUP], read_timeout=params[ATTR_TIMEOUT], parse_mode=params[ATTR_PARSER], message_thread_id=params[ATTR_MESSAGE_THREAD_ID], context=context)
                msg_ids[chat_id] = msg.id
                file_content.seek(0)
        else:
            _LOGGER.error("Can't send file with kwargs: %s", kwargs)
        return msg_ids

    async def send_sticker(
        self, 
        target: Optional[Union[int, List[int]]] = None, 
        context: Optional[Context] = None, 
        **kwargs: Any
    ) -> Dict[int, int]:
        """Send a sticker from a telegram sticker pack."""
        params = self._get_msg_kwargs(kwargs)
        stickerid = kwargs.get(ATTR_STICKER_ID)
        msg_ids: Dict[int, int] = {}
        if stickerid:
            for chat_id in self._get_target_chat_ids(target):
                msg = await self._send_msg(self.bot.send_sticker, 'Error sending sticker', params[ATTR_MESSAGE_TAG], chat_id=chat_id, sticker=stickerid, disable_notification=params[ATTR_DISABLE_NOTIF], reply_to_message_id=params[ATTR_REPLY_TO_MSGID], reply_markup=params[ATTR_REPLYMARKUP], read_timeout=params[ATTR_TIMEOUT], message_thread_id=params[ATTR_MESSAGE_THREAD_ID], context=context)
                msg_ids[chat_id] = msg.id
            return msg_ids
        return await self.send_file(SERVICE_SEND_STICKER, target, context=context, **kwargs)

    async def send_location(
        self, 
        latitude: str, 
        longitude: str, 
        target: Optional[Union[int, List[int]]] = None, 
        context: Optional[Context] = None, 
        **kwargs: Any
    ) -> Dict[int, int]:
        """Send a location."""
        latitude_float = float(latitude)
        longitude_float = float(longitude)
        params = self._get_msg_kwargs(kwargs)
        msg_ids: Dict[int, int] = {}
        for chat_id in self._get_target_chat_ids(target):
            _LOGGER.debug('Send location %s/%s to chat ID %s', latitude, longitude, chat_id)
            msg = await self._send_msg(self.bot.send_location, 'Error sending location', params[ATTR_MESSAGE_TAG], chat_id=chat_id, latitude=latitude_float, longitude=longitude_float, disable_notification=params[ATTR_DISABLE_NOTIF], reply_to_message_id=params[ATTR_REPLY_TO_MSGID], read_timeout=params[ATTR_TIMEOUT], message_thread_id=params[ATTR_MESSAGE_THREAD_ID], context=context)
            msg_ids[chat_id] = msg.id
        return msg_ids

    async def send_poll(
        self, 
        question: str, 
        options: List[str], 
        is_anonymous: bool, 
        allows_multiple_answers: bool, 
        target: Optional[Union[int, List[int]]] = None, 
        context: Optional[Context] = None, 
        **kwargs: Any
    ) -> Dict[int, int]:
        """Send a poll."""
        params = self._get_msg_kwargs(kwargs)
        openperiod = kwargs.get(ATTR_OPEN_PERIOD)
        msg_ids: Dict[int, int] = {}
        for chat_id in self._get_target_chat_ids(target):
            _LOGGER.debug("Send poll '%s' to chat ID %s", question, chat_id)
            msg = await self._send_msg(self.bot.send_poll, 'Error sending poll', params[ATTR_MESSAGE_TAG], chat_id=chat_id, question=question, options=options, is_anonymous=is_anonymous, allows_multiple_answers=allows_multiple_answers, open_period=openperiod, disable_notification=params[ATTR_DISABLE_NOTIF], reply_to_message_id=params[ATTR_REPLY_TO_MSGID], read_timeout=params[ATTR_TIMEOUT], message_thread_id=params[ATTR_MESSAGE_THREAD_ID], context=context)
            msg_ids[chat_id] = msg.id
        return msg_ids

    async def leave_chat(
        self, 
        chat_id: Optional[Union[int, List[int]]] = None, 
        context: Optional[Context] = None
    ) -> Optional[bool]:
        """Remove bot from chat."""
        chat_id = self._get_target_chat_ids(chat_id)[0]
        _LOGGER.debug('Leave from chat ID %s', chat_id)
        return await self._send_msg(self.bot.leave_chat, 'Error leaving chat', None, chat_id, context=context)

class BaseTelegramBotEntity:
    """The base class for the telegram bot."""

    def __init__(self, hass: HomeAssistant, config: Dict[str, Any]) -> None:
        """Initialize the bot base class."""
        self.allowed_chat_ids = config[CONF_ALLOWED_CHAT_IDS]
        self.hass = hass

    async def handle_update(self, update: Update, context: CallbackContext) -> bool:
        """Handle updates from bot application set up by the respective platform."""
        _LOGGER.debug('Handling update %s', update)
        if not self.authorize_update(update):
            return False
        if update.callback_query:
            event_type, event_data = self._get_callback_query_event_data(update.callback_query)
        elif update.effective_message:
            event_type, event_data = self._get_message_event_data(update.effective_message)
        else:
            _LOGGER.warning('Unhandled update: %s', update)
            return True
        event_context = Context()
        _LOGGER.debug('Firing event %s: %s', event_type, event_data)
        self.hass.bus.async_fire(event_type, event_data, context=event_context)
        return True

    @staticmethod
    def _get_command_event_data(command_text: Optional[str]) -> Dict[str, Any]:
        if not command_text or not command_text.startswith('/'):
            return {}
        command_parts = command_text.split()
        command = command_parts[0]
        args = command_parts[1:]
        return {ATTR_COMMAND: command, ATTR_ARGS: args}

    def _get_message_event_data(self, message: Message) -> Tuple[str, Dict[str, Any]]:
        event_data: Dict[str, Any] = {ATTR_MSGID: message.message_id, ATTR_CHAT_ID: message.chat.id, ATTR_DATE: message.date, ATTR_MESSAGE_THREAD_ID: message.message_thread_id}
        if filters.COMMAND.filter(message):
            event_type = EVENT_TELEGRAM_COMMAND
            event_data.update(self._get_command_event_data(message.text))
        else:
            event_type = EVENT_TELEGRAM_TEXT
            event_data[ATTR_TEXT] = message.text
        if message.from_user:
            event_data.update(self._get_user_event_data(message.from_user))
        return (event_type, event_data)

    def _get_user_event_data(self, user: User) -> Dict[str, Any]:
        return {ATTR_USER_ID: user.id, ATTR_FROM_FIRST: user.first_name, ATTR_FROM_LAST: user.last_name}

    def _get_callback_query_event_data(self, callback_query: CallbackQuery) -> Tuple[str, Dict[str, Any]]:
        event_type = EVENT_TELEGRAM_CALLBACK
        event_data: Dict[str, Any] = {ATTR_MSGID: callback_query.id, ATTR_CHAT_INSTANCE: callback_query.chat_instance, ATTR_DATA: callback_query.data, ATTR_MSG: None, ATTR_CHAT_ID: None}
        if callback_query.message:
            event_data[ATTR_MSG] = callback_query.message.to_dict()
            event_data[ATTR_CHAT_ID] = callback_query.message.chat.id
        if callback_query.from_user:
            event_data.update(self._get_user_event_data(callback_query.from_user))
        event_data.update(self._get_command_event_data(callback_query.data))
        return (event_type, event_data)

    def authorize_update(self, update: Update) -> bool:
        """Make sure either user or chat is in allowed_chat_ids."""
        from_user = update.effective_user.id if update.effective_user else None
        from_chat = update.effective_chat.id if update.effective_chat else None
        if from_user in self.allowed_chat_ids or from_chat in self.allowed_chat_ids:
            return True
        _LOGGER.error('Unauthorized update - neither user id %s nor chat id %s is in allowed chats: %s', from_user, from_chat, self.allowed_chat_ids)
        return False

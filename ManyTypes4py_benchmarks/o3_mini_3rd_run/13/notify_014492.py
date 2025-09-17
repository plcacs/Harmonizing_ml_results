from __future__ import annotations
from contextlib import suppress
from datetime import datetime, timedelta
from functools import partial
from http import HTTPStatus
import json
import logging
import time
from typing import Any, Dict, Optional, List, Union
from urllib.parse import urlparse
import uuid

from aiohttp.web import Request, Response
from aiohttp.hdrs import AUTHORIZATION
import jwt
from py_vapid import Vapid
from pywebpush import WebPusher
import voluptuous as vol
from voluptuous.humanize import humanize_error

from homeassistant.components import websocket_api
from homeassistant.components.http import KEY_HASS, HomeAssistantView
from homeassistant.components.notify import ATTR_DATA, ATTR_TARGET, ATTR_TITLE, ATTR_TITLE_DEFAULT, PLATFORM_SCHEMA as NOTIFY_PLATFORM_SCHEMA, BaseNotificationService
from homeassistant.config_entries import SOURCE_IMPORT
from homeassistant.const import ATTR_NAME, URL_ROOT
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.json import save_json
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import ensure_unique_string
from homeassistant.util.json import JsonObjectType, load_json_object

from .const import ATTR_VAPID_EMAIL, ATTR_VAPID_PRV_KEY, ATTR_VAPID_PUB_KEY, DOMAIN, SERVICE_DISMISS
from .issues import async_create_html5_issue

_LOGGER = logging.getLogger(__name__)

REGISTRATIONS_FILE: str = 'html5_push_registrations.conf'

PLATFORM_SCHEMA = NOTIFY_PLATFORM_SCHEMA.extend({
    vol.Optional('gcm_sender_id'): cv.string,
    vol.Optional('gcm_api_key'): cv.string,
    vol.Required(ATTR_VAPID_PUB_KEY): cv.string,
    vol.Required(ATTR_VAPID_PRV_KEY): cv.string,
    vol.Required(ATTR_VAPID_EMAIL): cv.string
})

ATTR_SUBSCRIPTION = 'subscription'
ATTR_BROWSER = 'browser'
ATTR_ENDPOINT = 'endpoint'
ATTR_KEYS = 'keys'
ATTR_AUTH = 'auth'
ATTR_P256DH = 'p256dh'
ATTR_EXPIRATIONTIME = 'expirationTime'
ATTR_TAG = 'tag'
ATTR_ACTION = 'action'
ATTR_ACTIONS = 'actions'
ATTR_TYPE = 'type'
ATTR_URL = 'url'
ATTR_DISMISS = 'dismiss'
ATTR_PRIORITY = 'priority'
DEFAULT_PRIORITY = 'normal'
ATTR_TTL = 'ttl'
DEFAULT_TTL = 86400
ATTR_JWT = 'jwt'
WS_TYPE_APPKEY = 'notify/html5/appkey'
SCHEMA_WS_APPKEY = websocket_api.BASE_COMMAND_MESSAGE_SCHEMA.extend({vol.Required('type'): WS_TYPE_APPKEY})
JWT_VALID_DAYS = 7
VAPID_CLAIM_VALID_HOURS = 12
KEYS_SCHEMA = vol.All(dict, vol.Schema({vol.Required(ATTR_AUTH): cv.string, vol.Required(ATTR_P256DH): cv.string}))
SUBSCRIPTION_SCHEMA = vol.All(dict, vol.Schema({
    vol.Required(ATTR_ENDPOINT): vol.Url(),
    vol.Required(ATTR_KEYS): KEYS_SCHEMA,
    vol.Optional(ATTR_EXPIRATIONTIME): vol.Any(None, cv.positive_int)
}))
DISMISS_SERVICE_SCHEMA = vol.Schema({
    vol.Optional(ATTR_TARGET): vol.All(cv.ensure_list, [cv.string]),
    vol.Optional(ATTR_DATA): dict
})
REGISTER_SCHEMA = vol.Schema({
    vol.Required(ATTR_SUBSCRIPTION): SUBSCRIPTION_SCHEMA,
    vol.Required(ATTR_BROWSER): vol.In(['chrome', 'firefox']),
    vol.Optional(ATTR_NAME): cv.string
})
CALLBACK_EVENT_PAYLOAD_SCHEMA = vol.Schema({
    vol.Required(ATTR_TAG): cv.string,
    vol.Required(ATTR_TYPE): vol.In(['received', 'clicked', 'closed']),
    vol.Required(ATTR_TARGET): cv.string,
    vol.Optional(ATTR_ACTION): cv.string,
    vol.Optional(ATTR_DATA): dict
})
NOTIFY_CALLBACK_EVENT = 'html5_notification'
HTML5_SHOWNOTIFICATION_PARAMETERS = (
    'actions', 'badge', 'body', 'dir', 'icon', 'image', 'lang',
    'renotify', 'requireInteraction', 'tag', 'timestamp', 'vibrate', 'silent'
)


async def async_get_service(
    hass: HomeAssistant, config: Optional[ConfigType], discovery_info: Optional[DiscoveryInfoType] = None
) -> Optional[BaseNotificationService]:
    """Get the HTML5 push notification service."""
    if config:
        existing_config_entry = hass.config_entries.async_entries(DOMAIN)
        if existing_config_entry:
            async_create_html5_issue(hass, True)
            return None
        hass.async_create_task(hass.config_entries.flow.async_init(DOMAIN, context={'source': SOURCE_IMPORT}, data=config))
        return None
    if discovery_info is None:
        return None
    json_path: str = hass.config.path(REGISTRATIONS_FILE)
    registrations: Dict[str, Any] = await hass.async_add_executor_job(_load_config, json_path)
    vapid_pub_key: str = discovery_info[ATTR_VAPID_PUB_KEY]
    vapid_prv_key: str = discovery_info[ATTR_VAPID_PRV_KEY]
    vapid_email: str = discovery_info[ATTR_VAPID_EMAIL]

    def websocket_appkey(_hass: HomeAssistant, connection: Any, msg: Dict[str, Any]) -> None:
        connection.send_message(websocket_api.result_message(msg['id'], vapid_pub_key))

    websocket_api.async_register_command(hass, WS_TYPE_APPKEY, websocket_appkey, SCHEMA_WS_APPKEY)
    hass.http.register_view(HTML5PushRegistrationView(registrations, json_path))
    hass.http.register_view(HTML5PushCallbackView(registrations))
    return HTML5NotificationService(hass, vapid_prv_key, vapid_email, registrations, json_path)


def _load_config(filename: str) -> Dict[str, Any]:
    """Load configuration."""
    with suppress(HomeAssistantError):
        return load_json_object(filename)
    return {}


class HTML5PushRegistrationView(HomeAssistantView):
    """Accepts push registrations from a browser."""
    url: str = '/api/notify.html5'
    name: str = 'api:notify.html5'

    def __init__(self, registrations: Dict[str, Any], json_path: str) -> None:
        """Init HTML5PushRegistrationView."""
        self.registrations: Dict[str, Any] = registrations
        self.json_path: str = json_path

    async def post(self, request: Request) -> Response:
        """Accept the POST request for push registrations from a browser."""
        try:
            data: Dict[str, Any] = await request.json()
        except ValueError:
            return self.json_message('Invalid JSON', HTTPStatus.BAD_REQUEST)
        try:
            data = REGISTER_SCHEMA(data)
        except vol.Invalid as ex:
            return self.json_message(humanize_error(data, ex), HTTPStatus.BAD_REQUEST)
        devname: Optional[str] = data.get(ATTR_NAME)
        data.pop(ATTR_NAME, None)
        name: str = self.find_registration_name(data, devname)
        previous_registration: Optional[Dict[str, Any]] = self.registrations.get(name)
        self.registrations[name] = data
        try:
            hass: HomeAssistant = request.app[KEY_HASS]
            await hass.async_add_executor_job(save_json, self.json_path, self.registrations)
            return self.json_message('Push notification subscriber registered.')
        except HomeAssistantError:
            if previous_registration is not None:
                self.registrations[name] = previous_registration
            else:
                self.registrations.pop(name)
            return self.json_message('Error saving registration.', HTTPStatus.INTERNAL_SERVER_ERROR)

    def find_registration_name(self, data: Dict[str, Any], suggested: Optional[str]) -> str:
        """Find a registration name matching data or generate a unique one."""
        endpoint: str = data.get(ATTR_SUBSCRIPTION).get(ATTR_ENDPOINT)
        for key, registration in self.registrations.items():
            subscription = registration.get(ATTR_SUBSCRIPTION)
            if subscription.get(ATTR_ENDPOINT) == endpoint:
                return key
        return ensure_unique_string(suggested or 'unnamed device', self.registrations)

    async def delete(self, request: Request) -> Response:
        """Delete a registration."""
        try:
            data: Dict[str, Any] = await request.json()
        except ValueError:
            return self.json_message('Invalid JSON', HTTPStatus.BAD_REQUEST)
        subscription = data.get(ATTR_SUBSCRIPTION)
        found: Optional[str] = None
        for key, registration in self.registrations.items():
            if registration.get(ATTR_SUBSCRIPTION) == subscription:
                found = key
                break
        if not found:
            return self.json_message('Registration not found.')
        reg: Dict[str, Any] = self.registrations.pop(found)
        try:
            hass: HomeAssistant = request.app[KEY_HASS]
            await hass.async_add_executor_job(save_json, self.json_path, self.registrations)
        except HomeAssistantError:
            self.registrations[found] = reg
            return self.json_message('Error saving registration.', HTTPStatus.INTERNAL_SERVER_ERROR)
        return self.json_message('Push notification subscriber unregistered.')


class HTML5PushCallbackView(HomeAssistantView):
    """Accepts push callback requests from a browser."""
    requires_auth: bool = False
    url: str = '/api/notify.html5/callback'
    name: str = 'api:notify.html5/callback'

    def __init__(self, registrations: Dict[str, Any]) -> None:
        """Init HTML5PushCallbackView."""
        self.registrations: Dict[str, Any] = registrations

    def decode_jwt(self, token: str) -> Any:
        """Find the registration that signed this JWT and return it."""
        target_check: Dict[str, Any] = jwt.decode(token, algorithms=['ES256', 'HS256'], options={'verify_signature': False})
        if target_check.get(ATTR_TARGET) in self.registrations:
            possible_target: Dict[str, Any] = self.registrations[target_check[ATTR_TARGET]]
            key: str = possible_target[ATTR_SUBSCRIPTION][ATTR_KEYS][ATTR_AUTH]
            with suppress(jwt.exceptions.DecodeError):
                return jwt.decode(token, key, algorithms=['ES256', 'HS256'])
        return self.json_message('No target found in JWT', status_code=HTTPStatus.UNAUTHORIZED)

    def check_authorization_header(self, request: Request) -> Any:
        """Check the authorization header."""
        auth: Optional[str] = request.headers.get(AUTHORIZATION)
        if not auth:
            return self.json_message('Authorization header is expected', status_code=HTTPStatus.UNAUTHORIZED)
        parts: List[str] = auth.split()
        if parts[0].lower() != 'bearer':
            return self.json_message('Authorization header must start with Bearer', status_code=HTTPStatus.UNAUTHORIZED)
        if len(parts) != 2:
            return self.json_message('Authorization header must be Bearer token', status_code=HTTPStatus.UNAUTHORIZED)
        token: str = parts[1]
        try:
            payload: Any = self.decode_jwt(token)
        except jwt.exceptions.InvalidTokenError:
            return self.json_message('token is invalid', status_code=HTTPStatus.UNAUTHORIZED)
        return payload

    async def post(self, request: Request) -> Response:
        """Accept the POST request for push registrations event callback."""
        auth_check: Any = self.check_authorization_header(request)
        if not isinstance(auth_check, dict):
            return auth_check
        try:
            data: Dict[str, Any] = await request.json()
        except ValueError:
            return self.json_message('Invalid JSON', HTTPStatus.BAD_REQUEST)
        event_payload: Dict[str, Any] = {
            ATTR_TAG: data.get(ATTR_TAG),
            ATTR_TYPE: data[ATTR_TYPE],
            ATTR_TARGET: auth_check[ATTR_TARGET]
        }
        if data.get(ATTR_ACTION) is not None:
            event_payload[ATTR_ACTION] = data.get(ATTR_ACTION)
        if data.get(ATTR_DATA) is not None:
            event_payload[ATTR_DATA] = data.get(ATTR_DATA)
        try:
            event_payload = CALLBACK_EVENT_PAYLOAD_SCHEMA(event_payload)
        except vol.Invalid as ex:
            _LOGGER.warning('Callback event payload is not valid: %s', humanize_error(event_payload, ex))
        event_name: str = f'{NOTIFY_CALLBACK_EVENT}.{event_payload[ATTR_TYPE]}'
        request.app[KEY_HASS].bus.fire(event_name, event_payload)
        return self.json({'status': 'ok', 'event': event_payload[ATTR_TYPE]})


class HTML5NotificationService(BaseNotificationService):
    """Implement the notification service for HTML5."""

    def __init__(
        self,
        hass: HomeAssistant,
        vapid_prv: str,
        vapid_email: str,
        registrations: Dict[str, Any],
        json_path: str
    ) -> None:
        """Initialize the service."""
        self.hass: HomeAssistant = hass
        self._vapid_prv: str = vapid_prv
        self._vapid_email: str = vapid_email
        self.registrations: Dict[str, Any] = registrations
        self.registrations_json_path: str = json_path

        async def async_dismiss_message(service: ServiceCall) -> None:
            """Handle dismissing notification message service calls."""
            kwargs: Dict[str, Any] = {}
            if self.targets is not None:
                kwargs[ATTR_TARGET] = self.targets
            elif service.data.get(ATTR_TARGET) is not None:
                kwargs[ATTR_TARGET] = service.data.get(ATTR_TARGET)
            kwargs[ATTR_DATA] = service.data.get(ATTR_DATA)
            await self.async_dismiss(**kwargs)

        hass.services.async_register(DOMAIN, SERVICE_DISMISS, async_dismiss_message, schema=DISMISS_SERVICE_SCHEMA)

    @property
    def targets(self) -> Dict[str, str]:
        """Return a dictionary of registered targets."""
        return {registration: registration for registration in self.registrations}

    def dismiss(self, **kwargs: Any) -> None:
        """Dismisses a notification."""
        data: Optional[Dict[str, Any]] = kwargs.get(ATTR_DATA)
        tag: str = data.get(ATTR_TAG) if data else ''
        payload: Dict[str, Any] = {ATTR_TAG: tag, ATTR_DISMISS: True, ATTR_DATA: {}}
        self._push_message(payload, **kwargs)

    async def async_dismiss(self, **kwargs: Any) -> None:
        """Dismisses a notification.

        This method must be run in the event loop.
        """
        await self.hass.async_add_executor_job(partial(self.dismiss, **kwargs))

    def send_message(self, message: str = '', **kwargs: Any) -> None:
        """Send a message to a user."""
        tag: str = str(uuid.uuid4())
        payload: Dict[str, Any] = {
            'badge': '/static/images/notification-badge.png',
            'body': message,
            ATTR_DATA: {},
            'icon': '/static/icons/favicon-192x192.png',
            ATTR_TAG: tag,
            ATTR_TITLE: kwargs.get(ATTR_TITLE, ATTR_TITLE_DEFAULT)
        }
        if (data := kwargs.get(ATTR_DATA)):
            data_tmp: Dict[str, Any] = {}
            for key, val in data.items():
                if key in HTML5_SHOWNOTIFICATION_PARAMETERS:
                    payload[key] = val
                else:
                    data_tmp[key] = val
            payload[ATTR_DATA] = data_tmp
        if payload[ATTR_DATA].get(ATTR_URL) is None and payload.get(ATTR_ACTIONS) is None:
            payload[ATTR_DATA][ATTR_URL] = URL_ROOT
        self._push_message(payload, **kwargs)

    def _push_message(self, payload: Dict[str, Any], **kwargs: Any) -> None:
        """Send the message."""
        timestamp: int = int(time.time())
        ttl: int = int(kwargs.get(ATTR_TTL, DEFAULT_TTL))
        priority: str = kwargs.get(ATTR_PRIORITY, DEFAULT_PRIORITY)
        if priority not in ['normal', 'high']:
            priority = DEFAULT_PRIORITY
        payload['timestamp'] = timestamp * 1000
        targets: Optional[Any] = kwargs.get(ATTR_TARGET)
        if not targets:
            targets = self.registrations.keys()
        for target in list(targets):
            info: Any = self.registrations.get(target)
            try:
                info = REGISTER_SCHEMA(info)
            except vol.Invalid:
                _LOGGER.error('%s is not a valid HTML5 push notification target', target)
                continue
            subscription: Dict[str, Any] = info[ATTR_SUBSCRIPTION]
            payload[ATTR_DATA][ATTR_JWT] = add_jwt(timestamp, target, payload[ATTR_TAG], subscription[ATTR_KEYS][ATTR_AUTH])
            webpusher: WebPusher = WebPusher(info[ATTR_SUBSCRIPTION])
            endpoint = urlparse(subscription[ATTR_ENDPOINT])
            vapid_claims: Dict[str, Any] = {
                'sub': f'mailto:{self._vapid_email}',
                'aud': f'{endpoint.scheme}://{endpoint.netloc}',
                'exp': timestamp + VAPID_CLAIM_VALID_HOURS * 60 * 60
            }
            vapid_headers: Dict[str, Any] = Vapid.from_string(self._vapid_prv).sign(vapid_claims)
            vapid_headers.update({'urgency': priority, 'priority': priority})
            response = webpusher.send(data=json.dumps(payload), headers=vapid_headers, ttl=ttl)
            if response.status_code == 410:
                _LOGGER.info('Notification channel has expired')
                reg: Dict[str, Any] = self.registrations.pop(target)
                try:
                    save_json(self.registrations_json_path, self.registrations)
                except HomeAssistantError:
                    self.registrations[target] = reg
                    _LOGGER.error('Error saving registration')
                else:
                    _LOGGER.info('Configuration saved')
            elif response.status_code > 399:
                _LOGGER.error('There was an issue sending the notification %s: %s', response.status_code, response.text)


def add_jwt(timestamp: int, target: str, tag: str, jwt_secret: str) -> str:
    """Create JWT json to put into payload."""
    jwt_exp: datetime = datetime.fromtimestamp(timestamp) + timedelta(days=JWT_VALID_DAYS)
    jwt_claims: Dict[str, Any] = {'exp': jwt_exp, 'nbf': timestamp, 'iat': timestamp, ATTR_TARGET: target, ATTR_TAG: tag}
    return jwt.encode(jwt_claims, jwt_secret)
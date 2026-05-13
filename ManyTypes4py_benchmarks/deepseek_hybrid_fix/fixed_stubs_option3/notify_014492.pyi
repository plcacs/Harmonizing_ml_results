from __future__ import annotations
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import datetime, timedelta
from http import HTTPStatus
import json
import logging
import time
from typing import Any, Optional, Union
from urllib.parse import ParseResult, urlparse
import uuid
from aiohttp import web
from aiohttp.hdrs import AUTHORIZATION
from homeassistant.components.http import HomeAssistantView
from homeassistant.components.notify import BaseNotificationService
from homeassistant.components import websocket_api
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.json import save_json
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import ensure_unique_string
from homeassistant.util.json import JsonObjectType, load_json_object
from jwt import PyJWT
import jwt
from py_vapid import Vapid
from pywebpush import WebPusher
import voluptuous as vol
from voluptuous.humanize import humanize_error
from .const import ATTR_VAPID_EMAIL, ATTR_VAPID_PRV_KEY, ATTR_VAPID_PUB_KEY, DOMAIN, SERVICE_DISMISS
from .issues import async_create_html5_issue

ATTR_SUBSCRIPTION: str
ATTR_BROWSER: str
ATTR_ENDPOINT: str
ATTR_KEYS: str
ATTR_AUTH: str
ATTR_P256DH: str
ATTR_EXPIRATIONTIME: str
ATTR_TAG: str
ATTR_ACTION: str
ATTR_ACTIONS: str
ATTR_TYPE: str
ATTR_URL: str
ATTR_DISMISS: str
ATTR_PRIORITY: str
ATTR_TTL: str
ATTR_JWT: str
ATTR_NAME: str
ATTR_TARGET: str
ATTR_TITLE: str
ATTR_DATA: str
ATTR_TITLE_DEFAULT: str
DEFAULT_PRIORITY: str
DEFAULT_TTL: int
WS_TYPE_APPKEY: str
SCHEMA_WS_APPKEY: Any
JWT_VALID_DAYS: int
VAPID_CLAIM_VALID_HOURS: int
KEYS_SCHEMA: Any
SUBSCRIPTION_SCHEMA: Any
DISMISS_SERVICE_SCHEMA: Any
REGISTER_SCHEMA: Any
CALLBACK_EVENT_PAYLOAD_SCHEMA: Any
NOTIFY_CALLBACK_EVENT: str
HTML5_SHOWNOTIFICATION_PARAMETERS: tuple[str, ...]
REGISTRATIONS_FILE: str
PLATFORM_SCHEMA: Any
NOTIFY_PLATFORM_SCHEMA: Any
KEY_HASS: str
URL_ROOT: str
_LOGGER: logging.Logger

async def async_get_service(hass: HomeAssistant, config: ConfigType, discovery_info: DiscoveryInfoType | None = None) -> HTML5NotificationService | None: ...

def _load_config(filename: str) -> JsonObjectType: ...

class HTML5PushRegistrationView(HomeAssistantView):
    url: str = '/api/notify.html5'
    name: str = 'api:notify.html5'
    registrations: dict[str, Any]
    json_path: str
    def __init__(self, registrations: dict[str, Any], json_path: str) -> None: ...
    async def post(self, request: web.Request) -> web.Response: ...
    def find_registration_name(self, data: dict[str, Any], suggested: str | None = None) -> str: ...
    async def delete(self, request: web.Request) -> web.Response: ...

class HTML5PushCallbackView(HomeAssistantView):
    requires_auth: bool = False
    url: str = '/api/notify.html5/callback'
    name: str = 'api:notify.html5/callback'
    registrations: dict[str, Any]
    def __init__(self, registrations: dict[str, Any]) -> None: ...
    def decode_jwt(self, token: str) -> dict[str, Any] | web.Response: ...
    def check_authorization_header(self, request: web.Request) -> dict[str, Any] | web.Response: ...
    async def post(self, request: web.Request) -> web.Response: ...

class HTML5NotificationService(BaseNotificationService):
    _vapid_prv: str
    _vapid_email: str
    registrations: dict[str, Any]
    registrations_json_path: str
    def __init__(self, hass: HomeAssistant, vapid_prv: str, vapid_email: str, registrations: dict[str, Any], json_path: str) -> None: ...
    @property
    def targets(self) -> dict[str, str]: ...
    def dismiss(self, **kwargs: Any) -> None: ...
    async def async_dismiss(self, **kwargs: Any) -> None: ...
    def send_message(self, message: str = '', **kwargs: Any) -> None: ...
    def _push_message(self, payload: dict[str, Any], **kwargs: Any) -> None: ...

def add_jwt(timestamp: int, target: str, tag: str, jwt_secret: str) -> str: ...
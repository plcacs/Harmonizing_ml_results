from __future__ import annotations

from datetime import datetime, timedelta
from http import HTTPStatus
from typing import Any

import voluptuous as vol
from aiohttp import web

from homeassistant.components.http import HomeAssistantView
from homeassistant.components.notify import BaseNotificationService
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util.json import JsonObjectType

_LOGGER: logging.Logger
REGISTRATIONS_FILE: str
PLATFORM_SCHEMA: vol.Schema
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
DEFAULT_PRIORITY: str
ATTR_TTL: str
DEFAULT_TTL: int
ATTR_JWT: str
WS_TYPE_APPKEY: str
SCHEMA_WS_APPKEY: vol.Schema
JWT_VALID_DAYS: int
VAPID_CLAIM_VALID_HOURS: int
KEYS_SCHEMA: vol.Schema
SUBSCRIPTION_SCHEMA: vol.Schema
DISMISS_SERVICE_SCHEMA: vol.Schema
REGISTER_SCHEMA: vol.Schema
CALLBACK_EVENT_PAYLOAD_SCHEMA: vol.Schema
NOTIFY_CALLBACK_EVENT: str
HTML5_SHOWNOTIFICATION_PARAMETERS: tuple[str, ...]

import logging

async def async_get_service(
    hass: HomeAssistant,
    config: ConfigType,
    discovery_info: DiscoveryInfoType | None = ...,
) -> HTML5NotificationService | None: ...

def _load_config(filename: str) -> JsonObjectType: ...

class HTML5PushRegistrationView(HomeAssistantView):
    url: str
    name: str
    registrations: dict[str, Any]
    json_path: str

    def __init__(self, registrations: dict[str, Any], json_path: str) -> None: ...
    async def post(self, request: web.Request) -> web.Response: ...
    def find_registration_name(self, data: dict[str, Any], suggested: str | None = ...) -> str: ...
    async def delete(self, request: web.Request) -> web.Response: ...

class HTML5PushCallbackView(HomeAssistantView):
    requires_auth: bool
    url: str
    name: str
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

    def __init__(
        self,
        hass: HomeAssistant,
        vapid_prv: str,
        vapid_email: str,
        registrations: dict[str, Any],
        json_path: str,
    ) -> None: ...

    @property
    def targets(self) -> dict[str, str]: ...
    def dismiss(self, **kwargs: Any) -> None: ...
    async def async_dismiss(self, **kwargs: Any) -> None: ...
    def send_message(self, message: str = ..., **kwargs: Any) -> None: ...
    def _push_message(self, payload: dict[str, Any], **kwargs: Any) -> None: ...

def add_jwt(timestamp: int, target: str, tag: str, jwt_secret: str) -> str: ...
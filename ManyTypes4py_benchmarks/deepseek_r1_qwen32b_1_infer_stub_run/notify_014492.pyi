"""HTML5 Push Messaging notification service."""

from __future__ import annotations
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)
import web
import voluptuous
import jwt
from homeassistant.core import HomeAssistant
from homeassistant.components.http import HomeAssistantView
from homeassistant.components.notify import BaseNotificationService
from pywebpush import WebPusher

REGISTRATIONS_FILE: str = ...
_LOGGER: logging.Logger = ...

class HTML5PushRegistrationView(HomeAssistantView):
    """Accepts push registrations from a browser."""
    url: str = ...
    name: str = ...

    def __init__(self, registrations: Dict[str, Any], json_path: str) -> None:
        ...

    async def post(self, request: web.Request) -> Dict[str, Any]:
        ...

    async def delete(self, request: web.Request) -> Dict[str, Any]:
        ...

    def find_registration_name(self, data: Dict[str, Any], suggested: Optional[str]) -> str:
        ...

class HTML5PushCallbackView(HomeAssistantView):
    """Accepts push registrations from a browser."""
    requires_auth: bool = ...
    url: str = ...
    name: str = ...

    def __init__(self, registrations: Dict[str, Any]) -> None:
        ...

    def decode_jwt(self, token: str) -> Union[Dict[str, Any], str]:
        ...

    def check_authorization_header(self, request: web.Request) -> Union[Dict[str, Any], str]:
        ...

    async def post(self, request: web.Request) -> Dict[str, Any]:
        ...

class HTML5NotificationService(BaseNotificationService):
    """Implement the notification service for HTML5."""

    def __init__(self, hass: HomeAssistant, vapid_prv: str, vapid_email: str, registrations: Dict[str, Any], json_path: str) -> None:
        ...

    @property
    def targets(self) -> Dict[str, str]:
        ...

    def dismiss(self, **kwargs: Any) -> None:
        ...

    async def async_dismiss(self, **kwargs: Any) -> None:
        ...

    def send_message(self, message: str, **kwargs: Any) -> None:
        ...

    def _push_message(self, payload: Dict[str, Any], **kwargs: Any) -> None:
        ...

def async_get_service(hass: HomeAssistant, config: Dict[str, Any], discovery_info: Optional[Dict[str, Any]]) -> Optional[HTML5NotificationService]:
    ...

def add_jwt(timestamp: int, target: str, tag: str, jwt_secret: str) -> str:
    ...
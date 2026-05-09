"""HTML5 Push Messaging notification service."""

from typing import Any, Optional, Union, Dict, List, Tuple, overload
from datetime import datetime
from aiohttp import web
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.components.http import HomeAssistantView
from homeassistant.components.notify import BaseNotificationService
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

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
SCHEMA_WS_APPKEY: Any
JWT_VALID_DAYS: int
VAPID_CLAIM_VALID_HOURS: int
KEYS_SCHEMA: Any
SUBSCRIPTION_SCHEMA: Any
DISMISS_SERVICE_SCHEMA: Any
REGISTER_SCHEMA: Any
CALLBACK_EVENT_PAYLOAD_SCHEMA: Any
NOTIFY_CALLBACK_EVENT: str
HTML5_SHOWNOTIFICATION_PARAMETERS: Tuple[str, ...]

async def async_get_service(
    hass: HomeAssistant, 
    config: Optional[ConfigType], 
    discovery_info: Optional[DiscoveryInfoType] = None
) -> Optional[HTML5NotificationService]:
    """Get the HTML5 push notification service."""
    ...

def _load_config(filename: str) -> Dict[str, Any]:
    """Load configuration."""
    ...

class HTML5PushRegistrationView(HomeAssistantView):
    """Accepts push registrations from a browser."""
    url: str
    name: str

    def __init__(self, registrations: Dict[str, Any], json_path: str) -> None:
        """Init HTML5PushRegistrationView."""
        ...

    async def post(self, request: web.Request) -> web.Response:
        """Accept the POST request for push registrations from a browser."""
        ...

    def find_registration_name(self, data: Dict[str, Any], suggested: Optional[str] = None) -> str:
        """Find a registration name matching data or generate a unique one."""
        ...

    async def delete(self, request: web.Request) -> web.Response:
        """Delete a registration."""
        ...

class HTML5PushCallbackView(HomeAssistantView):
    """Accepts push registrations from a browser."""
    requires_auth: bool
    url: str
    name: str

    def __init__(self, registrations: Dict[str, Any]) -> None:
        """Init HTML5PushCallbackView."""
        ...

    def decode_jwt(self, token: str) -> Union[Dict[str, Any], web.Response]:
        """Find the registration that signed this JWT and return it."""
        ...

    def check_authorization_header(self, request: web.Request) -> Union[Dict[str, Any], web.Response]:
        """Check the authorization header."""
        ...

    async def post(self, request: web.Request) -> web.Response:
        """Accept the POST request for push registrations event callback."""
        ...

class HTML5NotificationService(BaseNotificationService):
    """Implement the notification service for HTML5."""

    def __init__(self, hass: HomeAssistant, vapid_prv: str, vapid_email: str, registrations: Dict[str, Any], json_path: str) -> None:
        """Initialize the service."""
        ...

    @property
    def targets(self) -> Dict[str, Any]:
        """Return a dictionary of registered targets."""
        ...

    def dismiss(self, **kwargs: Any) -> None:
        """Dismisses a notification."""
        ...

    async def async_dismiss(self, **kwargs: Any) -> None:
        """Dismisses a notification.

        This method must be run in the event loop.
        """
        ...

    def send_message(self, message: str = '', **kwargs: Any) -> None:
        """Send a message to a user."""
        ...

    def _push_message(self, payload: Dict[str, Any], **kwargs: Any) -> None:
        """Send the message."""
        ...

def add_jwt(timestamp: int, target: str, tag: str, jwt_secret: str) -> str:
    """Create JWT json to put into payload."""
    ...
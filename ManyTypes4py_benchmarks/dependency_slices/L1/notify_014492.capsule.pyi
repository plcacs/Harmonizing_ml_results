from typing import Any

# === Third-party dependency: aiohttp.hdrs ===
AUTHORIZATION: Final[istr]

# === Internal dependency: homeassistant.components.html5.const ===
DOMAIN = 'html5'
SERVICE_DISMISS = 'dismiss'

# === Internal dependency: homeassistant.components.http ===
from homeassistant.helpers.http import KEY_HASS
from homeassistant.helpers.http import HomeAssistantView

# === Internal dependency: homeassistant.components.notify ===
from .const import ATTR_DATA
from .const import ATTR_TARGET
from .const import ATTR_TITLE
from .legacy import BaseNotificationService
ATTR_TITLE_DEFAULT = 'Home Assistant'
PLATFORM_SCHEMA = vol.Schema(...)

# === Internal dependency: homeassistant.components.websocket_api ===
def async_register_command(hass, command_or_handler, handler=..., schema=...): ...
from .messages import BASE_COMMAND_MESSAGE_SCHEMA
from .messages import result_message

# === Internal dependency: homeassistant.const ===
ATTR_NAME = 'name'
URL_ROOT = '/'

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def ensure_list(value): ...
def string(value): ...
positive_int = vol.All(...)

# === Internal dependency: homeassistant.helpers.json ===
def save_json(filename, data, private=..., *, encoder=..., atomic_writes=...): ...

# === Internal dependency: homeassistant.util ===
def ensure_unique_string(preferred_string, current_strings): ...

# === Internal dependency: homeassistant.util.json ===
def load_json_object(filename, default=...): ...

# === Third-party dependency: jwt ===
# Used symbols: decode, encode, exceptions

# === Third-party dependency: py_vapid ===
class Vapid02(Vapid01): ...
Vapid = Vapid02

# === Unresolved dependency: pywebpush ===
# Used unresolved symbols: WebPusher

# === Third-party dependency: voluptuous ===
# Used symbols: All, Any, In, Invalid, Optional, Required, Schema, Url

# === Third-party dependency: voluptuous.humanize ===
def humanize_error(data, validation_error, max_sub_error_length = ...) -> Any: ...
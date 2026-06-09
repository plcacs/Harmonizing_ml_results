from typing import Any

# === Third-party dependency: aiohttp.hdrs ===
AUTHORIZATION: Final[istr]

# === Internal dependency: homeassistant.components.html5.const ===
DOMAIN: str
SERVICE_DISMISS: str

# === Internal dependency: homeassistant.components.http ===
# re-export: from homeassistant.helpers.http import KEY_HASS
# re-export: from homeassistant.helpers.http import HomeAssistantView

# === Internal dependency: homeassistant.components.notify ===
# re-export: from .const import ATTR_DATA
# re-export: from .const import ATTR_TARGET
# re-export: from .const import ATTR_TITLE
# re-export: from .legacy import BaseNotificationService
ATTR_TITLE_DEFAULT: str
PLATFORM_SCHEMA: Schema

# === Internal dependency: homeassistant.components.websocket_api ===
def async_register_command(hass: HomeAssistant, command_or_handler: str | const.WebSocketCommandHandler, handler: const.WebSocketCommandHandler | None = ..., schema: VolSchemaType | None = ...) -> None: ...
# re-export: from .messages import BASE_COMMAND_MESSAGE_SCHEMA
# re-export: from .messages import result_message

# === Internal dependency: homeassistant.const ===
ATTR_NAME: Final
URL_ROOT: Final

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def ensure_list(value: None) -> list[Any]: ...
def ensure_list(value: list[_T]) -> list[_T]: ...
def ensure_list(value: list[_T] | _T) -> list[_T]: ...
def ensure_list(value: _T | None) -> list[_T] | list[Any]: ...
def string(value: Any) -> str: ...
positive_int: All

# === Internal dependency: homeassistant.helpers.json ===
def save_json(filename: str, data: list | dict, private: bool = ..., *, encoder: type[json.JSONEncoder] | None = ..., atomic_writes: bool = ...) -> None: ...

# === Internal dependency: homeassistant.util ===
def ensure_unique_string(preferred_string: str, current_strings: Iterable[str] | KeysView[str]) -> str: ...

# === Internal dependency: homeassistant.util.json ===
def load_json_object(filename: str | PathLike[str], default: JsonObjectType = ...) -> JsonObjectType: ...

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
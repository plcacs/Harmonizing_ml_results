from typing import Any

# === Third-party dependency: aiohttp ===
# Used symbols: hdrs, web, web_urldispatcher

# === Unresolved dependency: hass_frontend ===
# Used unresolved symbols: where

# === Internal dependency: homeassistant.components.frontend.storage ===
async def async_setup_frontend_storage(hass: HomeAssistant) -> None: ...

# === Internal dependency: homeassistant.components.http ===
class StaticPathConfig: ...
# re-export: from homeassistant.helpers.http import KEY_HASS
# re-export: from homeassistant.helpers.http import HomeAssistantView

# === Internal dependency: homeassistant.components.onboarding ===
def async_is_onboarded(hass: HomeAssistant) -> bool: ...

# === Internal dependency: homeassistant.components.websocket_api ===
def async_register_command(hass: HomeAssistant, command_or_handler: str | const.WebSocketCommandHandler, handler: const.WebSocketCommandHandler | None = ..., schema: VolSchemaType | None = ...) -> None: ...
# re-export: from .connection import ActiveConnection
# re-export: from .decorators import async_response
# re-export: from .decorators import websocket_command
# re-export: from .messages import event_message
# re-export: from .messages import result_message

# === Internal dependency: homeassistant.config ===
async def async_hass_config_yaml(hass: HomeAssistant) -> dict: ...

# === Internal dependency: homeassistant.const ===
CONF_MODE: Final
CONF_NAME: Final
EVENT_THEMES_UPDATED: Final
EVENT_PANELS_UPDATED: Final

# === Internal dependency: homeassistant.core ===
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def isdir(value: Any) -> str: ...
def ensure_list(value: None) -> list[Any]: ...
def ensure_list(value: list[_T]) -> list[_T]: ...
def ensure_list(value: list[_T] | _T) -> list[_T]: ...
def ensure_list(value: _T | None) -> list[_T] | list[Any]: ...
def match_all(value: _T) -> _T: ...
def string(value: Any) -> str: ...

# === Internal dependency: homeassistant.helpers.icon ===
async def async_get_icons(hass: HomeAssistant, category: str, integrations: Iterable[str] | None = ...) -> dict[str, Any]: ...

# === Internal dependency: homeassistant.helpers.json ===
def json_dumps_sorted(data: Any) -> str: ...

# === Internal dependency: homeassistant.helpers.service ===
def async_register_admin_service(hass: HomeAssistant, domain: str, service: str, service_func: Callable[[ServiceCall], Awaitable[None] | None], schema: VolSchemaType = ...) -> None: ...

# === Internal dependency: homeassistant.helpers.storage ===
class Store:
    def __init__(self, hass: HomeAssistant, version: int, key: str, private: bool = ..., *, atomic_writes: bool = ..., encoder: type[JSONEncoder] | None = ..., minor_version: int = ..., read_only: bool = ...) -> None: ...
    async def async_load(self) -> _T | None: ...
    def async_delay_save(self, data_func: Callable[[], _T], delay: float = ...) -> None: ...

# === Internal dependency: homeassistant.helpers.translation ===
async def async_get_translations(hass: HomeAssistant, language: str, category: str, integrations: Iterable[str] | None = ..., config_flow: bool | None = ...) -> dict[str, str]: ...

# === Internal dependency: homeassistant.loader ===
async def async_get_integration(hass: HomeAssistant, domain: str) -> Integration: ...
def bind_hass(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.util.hass_dict ===
class HassKey(_Key[_T]):
    ...

# === Third-party dependency: jinja2 ===
# Used symbols: Template

# === Third-party dependency: voluptuous ===
# Used symbols: ALLOW_EXTRA, All, Any, In, Optional, Required, Schema

# === Third-party dependency: yarl ===
# Used symbols: URL
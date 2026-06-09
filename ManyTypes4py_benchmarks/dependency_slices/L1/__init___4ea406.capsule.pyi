from typing import Any

# === Third-party dependency: aiohttp ===
# Used symbols: hdrs, web, web_urldispatcher

# === Unresolved dependency: hass_frontend ===
# Used unresolved symbols: where

# === Internal dependency: homeassistant.components.frontend.storage ===
async def async_setup_frontend_storage(hass): ...

# === Internal dependency: homeassistant.components.http ===
class StaticPathConfig: ...
from homeassistant.helpers.http import KEY_HASS
from homeassistant.helpers.http import HomeAssistantView

# === Internal dependency: homeassistant.components.onboarding ===
def async_is_onboarded(hass): ...

# === Internal dependency: homeassistant.components.websocket_api ===
def async_register_command(hass, command_or_handler, handler=..., schema=...): ...
from .connection import ActiveConnection
from .decorators import async_response
from .decorators import websocket_command
from .messages import event_message
from .messages import result_message

# === Internal dependency: homeassistant.config ===
async def async_hass_config_yaml(hass): ...

# === Internal dependency: homeassistant.const ===
CONF_MODE = 'mode'
CONF_NAME = 'name'
EVENT_THEMES_UPDATED = 'themes_updated'
EVENT_PANELS_UPDATED = 'panels_updated'

# === Internal dependency: homeassistant.core ===
def callback(func): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def isdir(value): ...
def ensure_list(value): ...
def match_all(value): ...
def string(value): ...

# === Internal dependency: homeassistant.helpers.icon ===
async def async_get_icons(hass, category, integrations=...): ...

# === Internal dependency: homeassistant.helpers.json ===
def json_dumps_sorted(data): ...

# === Internal dependency: homeassistant.helpers.service ===
def async_register_admin_service(hass, domain, service, service_func, schema=...): ...

# === Internal dependency: homeassistant.helpers.storage ===
class Store:
    def __init__(self, hass, version, key, private=..., *, atomic_writes=..., encoder=..., minor_version=..., read_only=...): ...
    async def async_load(self): ...
    def async_delay_save(self, data_func, delay=...): ...

# === Internal dependency: homeassistant.helpers.translation ===
async def async_get_translations(hass, language, category, integrations=..., config_flow=...): ...

# === Internal dependency: homeassistant.loader ===
async def async_get_integration(hass, domain): ...
def bind_hass(func): ...

# === Internal dependency: homeassistant.util.hass_dict ===
class HassKey(_Key[_T]):
    ...

# === Third-party dependency: jinja2 ===
# Used symbols: Template

# === Third-party dependency: voluptuous ===
# Used symbols: ALLOW_EXTRA, All, Any, In, Optional, Required, Schema

# === Third-party dependency: yarl ===
# Used symbols: URL
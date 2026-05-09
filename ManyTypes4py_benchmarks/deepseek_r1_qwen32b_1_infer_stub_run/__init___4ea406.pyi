"""Handle the frontend for Home Assistant."""
from __future__ import annotations
from collections.abc import Callable, Iterator
from functools import lru_cache, partial
import logging
import os
import pathlib
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from aiohttp import hdrs, web, web_urldispatcher
import jinja2
from propcache.api import cached_property
import voluptuous as vol
from yarl import URL
from homeassistant.components import onboarding, websocket_api
from homeassistant.components.http import (
    KEY_HASS,
    HomeAssistantView,
    StaticPathConfig,
)
from homeassistant.components.websocket_api import ActiveConnection
from homeassistant.config import async_hass_config_yaml
from homeassistant.const import (
    CONF_MODE,
    CONF_NAME,
    EVENT_PANELS_UPDATED,
    EVENT_THEMES_UPDATED,
)
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.helpers import (
    config_validation as cv,
    service,
    icon as async_get_icons,
    json as json_dumps_sorted,
    storage as Store,
    translation as async_get_translations,
)
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import async_get_integration, bind_hass
from homeassistant.util.hass_dict import HassKey

DOMAIN: str = ...
CONF_THEMES: str = ...
CONF_THEMES_MODES: str = ...
CONF_THEMES_LIGHT: str = ...
CONF_THEMES_DARK: str = ...
CONF_EXTRA_HTML_URL: str = ...
CONF_EXTRA_HTML_URL_ES5: str = ...
CONF_EXTRA_MODULE_URL: str = ...
CONF_EXTRA_JS_URL_ES5: str = ...
CONF_FRONTEND_REPO: str = ...
CONF_JS_VERSION: str = ...
DEFAULT_THEME_COLOR: str = ...
DATA_PANELS: str = ...
DATA_JS_VERSION: str = ...
DATA_EXTRA_MODULE_URL: str = ...
DATA_EXTRA_JS_URL_ES5: str = ...
DATA_WS_SUBSCRIBERS: HassKey = ...
THEMES_STORAGE_KEY: str = ...
THEMES_STORAGE_VERSION: int = ...
THEMES_SAVE_DELAY: int = ...
DATA_THEMES_STORE: str = ...
DATA_THEMES: str = ...
DATA_DEFAULT_THEME: str = ...
DATA_DEFAULT_DARK_THEME: str = ...
DEFAULT_THEME: str = ...
VALUE_NO_THEME: str = ...
PRIMARY_COLOR: str = ...
_LOGGER: logging.Logger = ...
EXTENDED_THEME_SCHEMA: vol.Schema = ...
THEME_SCHEMA: vol.Schema = ...
CONFIG_SCHEMA: vol.Schema = ...
SERVICE_SET_THEME: str = ...
SERVICE_RELOAD_THEMES: str = ...

class Manifest:
    """Manage the manifest.json contents."""

    def __init__(self, data: dict) -> None:
        ...

    def __getitem__(self, key: str) -> Any:
        ...

    @property
    def json(self) -> str:
        ...

    def update_key(self, key: str, val: Any) -> None:
        ...

class UrlManager:
    """Manage urls to be used on the frontend."""

    def __init__(self, on_change: Callable[[str, str], None], urls: Iterable[str]) -> None:
        ...

    def add(self, url: str) -> None:
        ...

    def remove(self, url: str) -> None:
        ...

class Panel:
    """Abstract class for panels."""

    sidebar_icon: Optional[str] = ...
    sidebar_title: Optional[str] = ...
    frontend_url_path: Optional[str] = ...
    config: dict = ...
    require_admin: bool = ...
    config_panel_domain: Optional[str] = ...

    def __init__(
        self,
        component_name: str,
        sidebar_title: Optional[str],
        sidebar_icon: Optional[str],
        frontend_url_path: Optional[str],
        config: dict,
        require_admin: bool,
        config_panel_domain: Optional[str],
    ) -> None:
        ...

    @callback
    def to_response(self) -> dict:
        ...

@bind_hass
@callback
def async_register_built_in_panel(
    hass: HomeAssistant,
    component_name: str,
    sidebar_title: Optional[str] = None,
    sidebar_icon: Optional[str] = None,
    frontend_url_path: Optional[str] = None,
    config: Optional[dict] = None,
    require_admin: bool = False,
    update: bool = False,
    config_panel_domain: Optional[str] = None,
) -> None:
    ...

@bind_hass
@callback
def async_remove_panel(
    hass: HomeAssistant,
    frontend_url_path: str,
    warn_if_unknown: bool = True,
) -> None:
    ...

def add_extra_js_url(hass: HomeAssistant, url: str, es5: bool = False) -> None:
    ...

def remove_extra_js_url(hass: HomeAssistant, url: str, es5: bool = False) -> None:
    ...

def add_manifest_json_key(key: str, val: Any) -> None:
    ...

def _frontend_root(dev_repo_path: Optional[str]) -> pathlib.Path:
    ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    ...

async def _async_setup_themes(hass: HomeAssistant, themes: Optional[dict]) -> None:
    ...

@callback
@lru_cache(maxsize=1)
def _async_render_index_cached(template: jinja2.Template, **kwargs: Any) -> str:
    ...

class IndexView(web_urldispatcher.AbstractResource):
    """Serve the frontend."""

    def __init__(self, repo_path: Optional[str], hass: HomeAssistant) -> None:
        ...

    @cached_property
    def canonical(self) -> str:
        ...

    @cached_property
    def _route(self) -> web_urldispatcher.ResourceRoute:
        ...

    def url_for(self, **kwargs: Any) -> URL:
        ...

    async def resolve(
        self, request: web.Request
    ) -> Tuple[Optional[web_urldispatcher.UrlMappingMatchInfo], Set[str]]:
        ...

    def add_prefix(self, prefix: str) -> None:
        ...

    def get_info(self) -> dict:
        ...

    def raw_match(self, path: str) -> bool:
        ...

    def get_template(self) -> jinja2.Template:
        ...

    async def get(self, request: web.Request) -> web.Response:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[web_urldispatcher.ResourceRoute]:
        ...

class ManifestJSONView(HomeAssistantView):
    """View to return a manifest.json."""

    requires_auth: bool = ...
    url: str = ...
    name: str = ...

    @callback
    def get(self, request: web.Request) -> web.Response:
        ...

@websocket_api.websocket_command(
    {
        'type': 'frontend/get_icons',
        vol.Required('category'): vol.In({'entity', 'entity_component', 'services'}),
        vol.Optional('integration'): vol.All(cv.ensure_list, [str]),
    }
)
@websocket_api.async_response
async def websocket_get_icons(
    hass: HomeAssistant, connection: ActiveConnection, msg: dict
) -> None:
    ...

@callback
@websocket_api.websocket_command({'type': 'get_panels'})
def websocket_get_panels(
    hass: HomeAssistant, connection: ActiveConnection, msg: dict
) -> None:
    ...

@callback
@websocket_api.websocket_command({'type': 'frontend/get_themes'})
def websocket_get_themes(
    hass: HomeAssistant, connection: ActiveConnection, msg: dict
) -> None:
    ...

@websocket_api.websocket_command(
    {
        'type': 'frontend/get_translations',
        vol.Required('language'): str,
        vol.Required('category'): str,
        vol.Optional('integration'): vol.All(cv.ensure_list, [str]),
        vol.Optional('config_flow'): bool,
    }
)
@websocket_api.async_response
async def websocket_get_translations(
    hass: HomeAssistant, connection: ActiveConnection, msg: dict
) -> None:
    ...

@websocket_api.websocket_command({'type': 'frontend/get_version'})
@websocket_api.async_response
async def websocket_get_version(
    hass: HomeAssistant, connection: ActiveConnection, msg: dict
) -> None:
    ...

@callback
@websocket_api.websocket_command({'type': 'frontend/subscribe_extra_js'})
def websocket_subscribe_extra_js(
    hass: HomeAssistant, connection: ActiveConnection, msg: dict
) -> None:
    ...

class PanelRespons(TypedDict):
    """Represent the panel response type."""
    component_name: str
    icon: Optional[str]
    title: str
    config: dict
    url_path: str
    require_admin: bool
    config_panel_domain: Optional[str]
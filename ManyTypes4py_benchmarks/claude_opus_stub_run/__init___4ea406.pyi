from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import lru_cache, partial
import pathlib
from typing import Any, TypedDict

from aiohttp import web, web_urldispatcher
import jinja2
import voluptuous as vol
from yarl import URL

from homeassistant.components.http import HomeAssistantView
from homeassistant.components.websocket_api import ActiveConnection
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.helpers.typing import ConfigType
from homeassistant.util.hass_dict import HassKey

DOMAIN: str
CONF_THEMES: str
CONF_THEMES_MODES: str
CONF_THEMES_LIGHT: str
CONF_THEMES_DARK: str
CONF_EXTRA_HTML_URL: str
CONF_EXTRA_HTML_URL_ES5: str
CONF_EXTRA_MODULE_URL: str
CONF_EXTRA_JS_URL_ES5: str
CONF_FRONTEND_REPO: str
CONF_JS_VERSION: str
DEFAULT_THEME_COLOR: str
DATA_PANELS: str
DATA_JS_VERSION: str
DATA_EXTRA_MODULE_URL: str
DATA_EXTRA_JS_URL_ES5: str
DATA_WS_SUBSCRIBERS: HassKey[set[tuple[ActiveConnection, int]]]
THEMES_STORAGE_KEY: str
THEMES_STORAGE_VERSION: int
THEMES_SAVE_DELAY: int
DATA_THEMES_STORE: str
DATA_THEMES: str
DATA_DEFAULT_THEME: str
DATA_DEFAULT_DARK_THEME: str
DEFAULT_THEME: str
VALUE_NO_THEME: str
PRIMARY_COLOR: str

EXTENDED_THEME_SCHEMA: vol.Schema
THEME_SCHEMA: vol.Schema
CONFIG_SCHEMA: vol.Schema
SERVICE_SET_THEME: str
SERVICE_RELOAD_THEMES: str

class Manifest:
    manifest: dict[str, Any]
    def __init__(self, data: dict[str, Any]) -> None: ...
    def __getitem__(self, key: str) -> Any: ...
    @property
    def json(self) -> str: ...
    def _serialize(self) -> None: ...
    def update_key(self, key: str, val: Any) -> None: ...

MANIFEST_JSON: Manifest

class UrlManager:
    urls: frozenset[str]
    def __init__(self, on_change: Callable[[str, str], None], urls: list[str] | frozenset[str]) -> None: ...
    def add(self, url: str) -> None: ...
    def remove(self, url: str) -> None: ...

class Panel:
    component_name: str
    sidebar_icon: str | None
    sidebar_title: str | None
    frontend_url_path: str | None
    config: dict[str, Any] | None
    require_admin: bool
    config_panel_domain: str | None
    def __init__(
        self,
        component_name: str,
        sidebar_title: str | None,
        sidebar_icon: str | None,
        frontend_url_path: str | None,
        config: dict[str, Any] | None,
        require_admin: bool,
        config_panel_domain: str | None,
    ) -> None: ...
    @callback
    def to_response(self) -> PanelRespons: ...

@callback
def async_register_built_in_panel(
    hass: HomeAssistant,
    component_name: str,
    sidebar_title: str | None = ...,
    sidebar_icon: str | None = ...,
    frontend_url_path: str | None = ...,
    config: dict[str, Any] | None = ...,
    require_admin: bool = ...,
    *,
    update: bool = ...,
    config_panel_domain: str | None = ...,
) -> None: ...

@callback
def async_remove_panel(
    hass: HomeAssistant,
    frontend_url_path: str,
    *,
    warn_if_unknown: bool = ...,
) -> None: ...

def add_extra_js_url(hass: HomeAssistant, url: str, es5: bool = ...) -> None: ...
def remove_extra_js_url(hass: HomeAssistant, url: str, es5: bool = ...) -> None: ...
def add_manifest_json_key(key: str, val: Any) -> None: ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool: ...

class IndexView(web_urldispatcher.AbstractResource):
    repo_path: str | None
    hass: HomeAssistant
    def __init__(self, repo_path: str | None, hass: HomeAssistant) -> None: ...
    @property
    def canonical(self) -> str: ...
    def url_for(self, **kwargs: Any) -> URL: ...
    async def resolve(self, request: web.Request) -> tuple[web_urldispatcher.UrlMappingMatchInfo | None, set[str]]: ...
    def add_prefix(self, prefix: str) -> None: ...
    def get_info(self) -> dict[str, Any]: ...
    def raw_match(self, path: str) -> bool: ...
    def get_template(self) -> jinja2.Template: ...
    async def get(self, request: web.Request) -> web.Response: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[web_urldispatcher.ResourceRoute]: ...

class ManifestJSONView(HomeAssistantView):
    requires_auth: bool
    url: str
    name: str
    @callback
    def get(self, request: web.Request) -> web.Response: ...

async def websocket_get_icons(
    hass: HomeAssistant, connection: ActiveConnection, msg: dict[str, Any]
) -> None: ...

def websocket_get_panels(
    hass: HomeAssistant, connection: ActiveConnection, msg: dict[str, Any]
) -> None: ...

def websocket_get_themes(
    hass: HomeAssistant, connection: ActiveConnection, msg: dict[str, Any]
) -> None: ...

async def websocket_get_translations(
    hass: HomeAssistant, connection: ActiveConnection, msg: dict[str, Any]
) -> None: ...

async def websocket_get_version(
    hass: HomeAssistant, connection: ActiveConnection, msg: dict[str, Any]
) -> None: ...

def websocket_subscribe_extra_js(
    hass: HomeAssistant, connection: ActiveConnection, msg: dict[str, Any]
) -> None: ...

class PanelRespons(TypedDict):
    component_name: str
    icon: str | None
    title: str | None
    config: dict[str, Any] | None
    url_path: str | None
    require_admin: bool
    config_panel_domain: str | None
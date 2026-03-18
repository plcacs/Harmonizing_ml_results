from typing import Any, Iterator, TypedDict
import logging
import pathlib
from aiohttp import web, web_urldispatcher
from jinja2 import Template
from yarl import URL
from homeassistant.components.http import HomeAssistantView
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
DATA_WS_SUBSCRIBERS: HassKey[Any]
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
_LOGGER: logging.Logger
EXTENDED_THEME_SCHEMA: Any
THEME_SCHEMA: Any
CONFIG_SCHEMA: Any
SERVICE_SET_THEME: str
SERVICE_RELOAD_THEMES: str

class Manifest:
    def __init__(self, data: Any) -> None: ...
    def __getitem__(self, key: Any) -> Any: ...
    @property
    def json(self) -> str: ...
    def update_key(self, key: Any, val: Any) -> None: ...

MANIFEST_JSON: Manifest

class UrlManager:
    def __init__(self, on_change: Any, urls: Any) -> None: ...
    urls: frozenset[str]
    def add(self, url: str) -> None: ...
    def remove(self, url: str) -> None: ...

class Panel:
    sidebar_icon: Any | None
    sidebar_title: Any | None
    frontend_url_path: Any | None
    config: Any
    require_admin: bool
    config_panel_domain: Any | None
    component_name: Any
    def __init__(
        self,
        component_name: Any,
        sidebar_title: Any | None,
        sidebar_icon: Any | None,
        frontend_url_path: Any | None,
        config: Any,
        require_admin: bool,
        config_panel_domain: Any | None,
    ) -> None: ...
    def to_response(self) -> dict[str, Any]: ...

def async_register_built_in_panel(
    hass: Any,
    component_name: Any,
    sidebar_title: Any | None = ...,
    sidebar_icon: Any | None = ...,
    frontend_url_path: Any | None = ...,
    config: Any = ...,
    require_admin: bool = ...,
    *,
    update: bool = ...,
    config_panel_domain: Any | None = ...,
) -> None: ...

def async_remove_panel(hass: Any, frontend_url_path: Any, *, warn_if_unknown: bool = ...) -> None: ...
def add_extra_js_url(hass: Any, url: str, es5: bool = ...) -> None: ...
def remove_extra_js_url(hass: Any, url: str, es5: bool = ...) -> None: ...
def add_manifest_json_key(key: Any, val: Any) -> None: ...
def _frontend_root(dev_repo_path: Any) -> pathlib.Path: ...
async def async_setup(hass: Any, config: Any) -> bool: ...
async def _async_setup_themes(hass: Any, themes: Any) -> None: ...
def _async_render_index_cached(template: Any, **kwargs: Any) -> str: ...

class IndexView(web_urldispatcher.AbstractResource):
    repo_path: Any
    hass: Any
    _template_cache: Any
    def __init__(self, repo_path: Any, hass: Any) -> None: ...
    @property
    def canonical(self) -> str: ...
    @property
    def _route(self) -> web_urldispatcher.ResourceRoute: ...
    def url_for(self, **kwargs: Any) -> URL: ...
    async def resolve(self, request: web.Request) -> tuple[Any, set[str]]: ...
    def add_prefix(self, prefix: str) -> None: ...
    def get_info(self) -> dict[str, Any]: ...
    def raw_match(self, path: str) -> bool: ...
    def get_template(self) -> Template: ...
    async def get(self, request: web.Request) -> web.Response: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[web_urldispatcher.ResourceRoute]: ...

class ManifestJSONView(HomeAssistantView):
    requires_auth: bool
    url: str
    name: str
    def get(self, request: web.Request) -> web.Response: ...

async def websocket_get_icons(hass: Any, connection: Any, msg: Any) -> None: ...
def websocket_get_panels(hass: Any, connection: Any, msg: Any) -> None: ...
def websocket_get_themes(hass: Any, connection: Any, msg: Any) -> None: ...
async def websocket_get_translations(hass: Any, connection: Any, msg: Any) -> None: ...
async def websocket_get_version(hass: Any, connection: Any, msg: Any) -> None: ...
def websocket_subscribe_extra_js(hass: Any, connection: Any, msg: Any) -> None: ...

class PanelRespons(TypedDict):
    ...
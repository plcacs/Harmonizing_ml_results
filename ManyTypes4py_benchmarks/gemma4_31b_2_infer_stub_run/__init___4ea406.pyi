from __future__ import annotations
from collections.abc import Callable, Iterator
from typing import Any, Optional, Union, TypedDict, FrozenSet
from aiohttp import web, web_urldispatcher
from yarl import URL
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.components.websocket_api import ActiveConnection
from homeassistant.components.http import HomeAssistantView

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

class Manifest:
    """Manage the manifest.json contents."""
    def __init__(self, data: dict[str, Any]) -> None: ...
    def __getitem__(self, key: str) -> Any: ...
    @property
    def json(self) -> str: ...
    def _serialize(self) -> None: ...
    def update_key(self, key: str, val: Any) -> None: ...

MANIFEST_JSON: Manifest

class UrlManager:
    """Manage urls to be used on the frontend."""
    def __init__(self, on_change: Callable[[str, str], None], urls: Union[list[str], FrozenSet[str]]) -> None: ...
    urls: FrozenSet[str]
    def add(self, url: str) -> None: ...
    def remove(self, url: str) -> None: ...

class Panel:
    """Abstract class for panels."""
    sidebar_icon: Optional[str]
    sidebar_title: Optional[str]
    frontend_url_path: Optional[str]
    config: Optional[Any]
    require_admin: bool
    config_panel_domain: Optional[str]

    def __init__(
        self,
        component_name: str,
        sidebar_title: Optional[str],
        sidebar_icon: Optional[str],
        frontend_url_path: Optional[str],
        config: Optional[Any],
        require_admin: bool,
        config_panel_domain: Optional[str],
    ) -> None: ...

    def to_response(self) -> dict[str, Any]: ...

def async_register_built_in_panel(
    hass: HomeAssistant,
    component_name: str,
    sidebar_title: Optional[str] = None,
    sidebar_icon: Optional[str] = None,
    frontend_url_path: Optional[str] = None,
    config: Optional[Any] = None,
    require_admin: bool = False,
    *,
    update: bool = False,
    config_panel_domain: Optional[str] = None,
) -> None: ...

def async_remove_panel(hass: HomeAssistant, frontend_url_path: str, *, warn_if_unknown: bool = True) -> None: ...

def add_extra_js_url(hass: HomeAssistant, url: str, es5: bool = False) -> None: ...

def remove_extra_js_url(hass: HomeAssistant, url: str, es5: bool = False) -> None: ...

def add_manifest_json_key(key: str, val: Any) -> None: ...

async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool: ...

class IndexView(web_urldispatcher.AbstractResource):
    """Serve the frontend."""
    def __init__(self, repo_path: Optional[str], hass: HomeAssistant) -> None: ...
    @property
    def canonical(self) -> str: ...
    @property
    def _route(self) -> web_urldispatcher.ResourceRoute: ...
    def url_for(self, **kwargs: Any) -> URL: ...
    async def resolve(self, request: web.Request) -> tuple[Optional[web_urldispatcher.UrlMappingMatchInfo], set[str]]: ...
    def add_prefix(self, prefix: str) -> None: ...
    def get_info(self) -> dict[str, list[str]]: ...
    def raw_match(self, path: str) -> bool: ...
    def get_template(self) -> Any: ...
    async def get(self, request: web.Request) -> web.Response: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[web_urldispatcher.ResourceRoute]: ...

class ManifestJSONView(HomeAssistantView):
    """View to return a manifest.json."""
    requires_auth: bool
    url: str
    name: str
    def get(self, request: web.Request) -> web.Response: ...

async def websocket_get_icons(hass: HomeAssistant, connection: ActiveConnection, msg: dict[str, Any]) -> None: ...

def websocket_get_panels(hass: HomeAssistant, connection: ActiveConnection, msg: dict[str, Any]) -> None: ...

def websocket_get_themes(hass: HomeAssistant, connection: ActiveConnection, msg: dict[str, Any]) -> None: ...

async def websocket_get_translations(hass: HomeAssistant, connection: ActiveConnection, msg: dict[str, Any]) -> None: ...

async def websocket_get_version(hass: HomeAssistant, connection: ActiveConnection, msg: dict[str, Any]) -> None: ...

def websocket_subscribe_extra_js(hass: HomeAssistant, connection: ActiveConnection, msg: dict[str, Any]) -> None: ...

class PanelRespons(TypedDict):
    """Represent the panel response type."""
    component_name: str
    icon: Optional[str]
    title: Optional[str]
    config: Optional[Any]
    url_path: str
    require_admin: bool
    config_panel_domain: Optional[str]
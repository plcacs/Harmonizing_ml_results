"""Handle the frontend for Home Assistant."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import lru_cache, partial
import logging
import os
import pathlib
from typing import Any, TypedDict

from aiohttp import hdrs, web, web_urldispatcher
import jinja2
from propcache.api import cached_property
import voluptuous as vol
from yarl import URL

from homeassistant.components import onboarding, websocket_api
from homeassistant.components.http import KEY_HASS, HomeAssistantView, StaticPathConfig
from homeassistant.components.websocket_api import ActiveConnection
from homeassistant.config import async_hass_config_yaml
from homeassistant.const import (
    CONF_MODE,
    CONF_NAME,
    EVENT_PANELS_UPDATED,
    EVENT_THEMES_UPDATED,
)
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.helpers import config_validation as cv, service
from homeassistant.helpers.icon import async_get_icons
from homeassistant.helpers.json import json_dumps_sorted
from homeassistant.helpers.storage import Store
from homeassistant.helpers.translation import async_get_translations
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import async_get_integration, bind_hass
from homeassistant.util.hass_dict import HassKey

from .storage import async_setup_frontend_storage

DOMAIN = "frontend"
CONF_THEMES = "themes"
CONF_THEMES_MODES = "modes"
CONF_THEMES_LIGHT = "light"
CONF_THEMES_DARK = "dark"
CONF_EXTRA_HTML_URL = "extra_html_url"
CONF_EXTRA_HTML_URL_ES5 = "extra_html_url_es5"
CONF_EXTRA_MODULE_URL = "extra_module_url"
CONF_EXTRA_JS_URL_ES5 = "extra_js_url_es5"
CONF_FRONTEND_REPO = "development_repo"
CONF_JS_VERSION = "javascript_version"

DEFAULT_THEME_COLOR = "#03A9F4"


DATA_PANELS = "frontend_panels"
DATA_JS_VERSION = "frontend_js_version"
DATA_EXTRA_MODULE_URL = "frontend_extra_module_url"
DATA_EXTRA_JS_URL_ES5 = "frontend_extra_js_url_es5"

DATA_WS_SUBSCRIBERS: HassKey[set[tuple[websocket_api.ActiveConnection, int]]] = HassKey(
    "frontend_ws_subscribers"
)

THEMES_STORAGE_KEY = f"{DOMAIN}_theme"
THEMES_STORAGE_VERSION = 1
THEMES_SAVE_DELAY = 60
DATA_THEMES_STORE = "frontend_themes_store"
DATA_THEMES = "frontend_themes"
DATA_DEFAULT_THEME = "frontend_default_theme"
DATA_DEFAULT_DARK_THEME = "frontend_default_dark_theme"
DEFAULT_THEME = "default"
VALUE_NO_THEME = "none"

PRIMARY_COLOR = "primary-color"

_LOGGER = logging.getLogger(__name__)

EXTENDED_THEME_SCHEMA = vol.Schema(
    {
        # Theme variables that apply to all modes
        cv.string: cv.string,
        # Mode specific theme variables
        vol.Optional(CONF_THEMES_MODES): vol.Schema(
            {
                vol.Optional(CONF_THEMES_LIGHT): vol.Schema({cv.string: cv.string}),
                vol.Optional(CONF_THEMES_DARK): vol.Schema({cv.string: cv.string}),
            }
        ),
    }
)

THEME_SCHEMA = vol.Schema(
    {
        cv.string: (
            vol.Any(
                # Legacy theme scheme
                {cv.string: cv.string},
                # New extended schema with mode support
                EXTENDED_THEME_SCHEMA,
            )
        )
    }
)

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.Schema(
            {
                vol.Optional(CONF_FRONTEND_REPO): cv.isdir,
                vol.Optional(CONF_THEMES): THEME_SCHEMA,
                vol.Optional(CONF_EXTRA_MODULE_URL): vol.All(
                    cv.ensure_list, [cv.string]
                ),
                vol.Optional(CONF_EXTRA_JS_URL_ES5): vol.All(
                    cv.ensure_list, [cv.string]
                ),
                # We no longer use these options.
                vol.Optional(CONF_EXTRA_HTML_URL): cv.match_all,
                vol.Optional(CONF_EXTRA_HTML_URL_ES5): cv.match_all,
                vol.Optional(CONF_JS_VERSION): cv.match_all,
            },
        )
    },
    extra=vol.ALLOW_EXTRA,
)

SERVICE_SET_THEME = "set_theme"
SERVICE_RELOAD_THEMES = "reload_themes"


class Manifest:
    """Manage the manifest.json contents."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Init the manifest manager."""
        self.manifest = data
        self._serialize()

    def __getitem__(self, key: str) -> Any:
        """Return an item in the manifest."""
        return self.manifest[key]

    @property
    def json(self) -> str:
        """Return the serialized manifest."""
        return self._serialized

    def _serialize(self) -> None:
        self._serialized = json_dumps_sorted(self.manifest)

    def update_key(self, key: str, val: str) -> None:
        """Add a keyval to the manifest.json."""
        self.manifest[key] = val
        self._serialize()


MANIFEST_JSON = Manifest(
    {
        "background_color": "#FFFFFF",
        "description": (
            "Home automation platform that puts local control and privacy first."
        ),
        "dir": "ltr",
        "display": "standalone",
        "icons": [
            {
                "src": f"/static/icons/favicon-{size}x{size}.png",
                "sizes": f"{size}x{size}",
                "type": "image/png",
                "purpose": "any",
            }
            for size in (192, 384, 512, 1024)
        ]
        + [
            {
                "src": f"/static/icons/maskable_icon-{size}x{size}.png",
                "sizes": f"{size}x{size}",
                "type": "image/png",
                "purpose": "maskable",
            }
            for size in (48, 72, 96, 128, 192, 384, 512)
        ],
        "screenshots": [
            {
                "src": "/static/images/screenshots/screenshot-1.png",
                "sizes": "413x792",
                "type": "image/png",
            }
        ],
        "lang": "en-US",
        "name": "Home Assistant",
        "short_name": "Home Assistant",
        "start_url": "/?homescreen=1",
        "id": "/?homescreen=1",
        "theme_color": DEFAULT_THEME_COLOR,
        "prefer_related_applications": True,
        "related_applications": [
            {"platform": "play", "id": "io.homeassistant.companion.android"}
        ],
    }
)


class UrlManager:
    """Manage urls to be used on the frontend.

    This is abstracted into a class because
    some integrations add a remove these directly
    on hass.data
    """

    def __init__(
        self,
        on_change: Callable[[str, str], None],
        urls: list[str],
    ) -> None:
        """Init the url manager."""
        self._on_change = on_change
        self.urls = frozenset(urls)

    def add(self, url: str) -> None:
        """Add a url to the set."""
        self.urls = frozenset([*self.urls, url])
        self._on_change("added", url)

    def remove(self, url: str) -> None:
        """Remove a url from the set."""
        self.urls = self.urls - {url}
        self._on_change("removed", url)


class Panel:
    """Abstract class for panels."""

    # Name of the webcomponent
    component_name: str

    # Icon to show in the sidebar
    sidebar_icon: str | None = None

    # Title to show in the sidebar
    sidebar_title: str | None = None

    # Url to show the panel in the frontend
    frontend_url_path: str | None = None

    # Config to pass to the webcomponent
    config: dict[str, Any] | None = None

    # If the panel should only be visible to admins
    require_admin = False

    # If the panel is a configuration panel for a integration
    config_panel_domain: str | None = None

    def __init__(
        self,
        component_name: str,
        sidebar_title: str | None,
        sidebar_icon: str | None,
        frontend_url_path: str | None,
        config: dict[str, Any] | None,
        require_admin: bool,
        config_panel_domain: str | None,
    ) -> None:
        """Initialize a built-in panel."""
        self.component_name = component_name
        self.sidebar_title = sidebar_title
        self.sidebar_icon = sidebar_icon
        self.frontend_url_path = frontend_url_path or component_name
        self.config = config
        self.require_admin = require_admin
        self.config_panel_domain = config_panel_domain

    @callback
    def to_response(self) -> PanelRespons:
        """Panel as dictionary."""
        return {
            "component_name": self.component_name,
            "icon": self.sidebar_icon,
            "title": self.sidebar_title,
            "config": self.config,
            "url_path": self.frontend_url_path,
            "require_admin": self.require_admin,
            "config_panel_domain": self.config_panel_domain,
        }


@bind_hass
@callback
def async_register_built_in_panel(
    hass: HomeAssistant,
    component_name: str,
    sidebar_title: str | None = None,
    sidebar_icon: str | None = None,
    frontend_url_path: str | None = None,
    config: dict[str, Any] | None = None,
    require_admin: bool = False,
    *,
    update: bool = False,
    config_panel_domain: str | None = None,
) -> None:
    """Register a built-in panel."""
    panel = Panel(
        component_name,
        sidebar_title,
        sidebar_icon,
        frontend_url_path,
        config,
        require_admin,
        config_panel_domain,
    )

    panels = hass.data.setdefault(DATA_PANELS, {})

    if not update and panel.frontend_url_path in panels:
        raise ValueError(f"Overwriting panel {panel.frontend_url_path}")

    panels[panel.frontend_url_path] = panel

    hass.bus.async_fire(EVENT_PANELS_UPDATED)


@bind_hass
@callback
def async_remove_panel(
    hass: HomeAssistant, frontend_url_path: str, *, warn_if_unknown: bool = True
) -> None:
    """Remove a built-in panel."""
    panel = hass.data.get(DATA_PANELS, {}).pop(frontend_url_path, None)

    if panel is None:
        if warn_if_unknown:
            _LOGGER.warning("Removing unknown panel %s", frontend_url_path)
        return

    hass.bus.async_fire(EVENT_PANELS_UPDATED)


def add_extra_js_url(hass: HomeAssistant, url: str, es5: bool = False) -> None:
    """Register extra js or module url to load.

    This function allows custom integrations to register extra js or module.
    """
    key = DATA_EXTRA_JS_URL_ES5 if es5 else DATA_EXTRA_MODULE_URL
    hass.data[key].add(url)


def remove_extra_js_url(hass: HomeAssistant, url: str, es5: bool = False) -> None:
    """Remove extra js or module url to load.

    This function allows custom integrations to remove extra js or module.
    """
    key = DATA_EXTRA_JS_URL_ES5 if es5 else DATA_EXTRA_MODULE_URL
    hass.data[key].remove(url)


def add_manifest_json_key(key: str, val: str) -> None:
    """Add a keyval to the manifest.json."""
    MANIFEST_JSON.update_key(key, val)


def _frontend_root(dev_repo_path: str | None) -> pathlib.Path:
    """Return root path to the frontend files."""
    if dev_repo_path is not None:
        return pathlib.Path(dev_repo_path) / "hass_frontend"
    # Keep import here so that we can import frontend without installing reqs
    # pylint: disable-next=import-outside-toplevel
    import hass_frontend

    return hass_frontend.where()


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the serving of the frontend."""
    await async_setup_frontend_storage(hass)
    websocket_api.async_register_command(hass, websocket_get_icons)
    websocket_api.async_register_command(hass, websocket_get_panels)
    websocket_api.async_register_command(hass, websocket_get_themes)
    websocket_api.async_register_command(hass, websocket_get_translations)
    websocket_api.async_register_command(hass, websocket_get_version)
    websocket_api.async_register_command(hass, websocket_subscribe_extra_js)
    hass.http.register_view(ManifestJSONView())

    conf = config.get(DOMAIN, {})

    for key in (CONF_EXTRA_HTML_URL, CONF_EXTRA_HTML_URL_ES5, CONF_JS_VERSION):
        if key in conf:
            _LOGGER.error(
                "Please remove %s from your frontend config. It is no longer supported",
                key,
            )

    repo_path = conf.get(CONF_FRONTEND_REPO)
    is_dev = repo_path is not None
    root_path = _frontend_root(repo_path)

    static_paths_configs: list[StaticPathConfig] = []

    for path, should_cache in (
        ("service_worker.js", False),
        ("sw-modern.js", False),
        ("sw-modern.js.map", False),
        ("sw-legacy.js", False),
        ("sw-legacy.js.map", False),
        ("robots.txt", False),
        ("onboarding.html", not is_dev),
        ("static", not is_dev),
        ("frontend_latest", not is_dev),
        ("frontend_es5", not is_dev),
    ):
        static_paths_configs.append(
            StaticPathConfig(f"/{path}", str(root_path / path), should_cache)
        )

    static_paths_configs.append(
        StaticPathConfig("/auth/authorize", str(root_path / "authorize.html"), False)
    )
    # https://wicg.github.io/change-password-url/
    hass.http.register_redirect(
        "/.well-known/change-password", "/profile", redirect_exc=web.HTTPFound
    )

    local = hass.config.path("www")
    if await hass.async_add_executor_job(os.path.isdir, local):
        static_paths_configs.append(StaticPathConfig("/local", local, not is_dev))

    await hass.http.async_register_static_paths(static_paths_configs)
    # Shopping list panel was replaced by todo panel in 2023.11
    hass.http.register_redirect("/shopping-list", "/todo")

    hass.http.app.router.register_resource(IndexView(repo_path, hass))

    async_register_built_in_panel(hass, "profile")

    async_register_built_in_panel(
        hass,
        "developer-tools",
        require_admin=True,
        sidebar_title="developer_tools",
        sidebar_icon="hass:hammer",
    )

    @callback
    def async_change_listener(
        resource_type: str,
        change_type: str,
        url: str,
    ) -> None:
        subscribers = hass.data[DATA_WS_SUBSCRIBERS]
        json_msg = {
            "change_type": change_type,
            "item": {"type": resource_type, "url": url},
        }
        for connection, msg_id in subscribers:
            connection.send_message(websocket_api.event_message(msg_id, json_msg))

    hass.data[DATA_EXTRA_MODULE_URL] = UrlManager(
        partial(async_change_listener, "module"), conf.get(CONF_EXTRA_MODULE_URL, [])
    )
    hass.data[DATA_EXTRA_JS_URL_ES5] = UrlManager(
        partial(async_change_listener, "es5"), conf.get(CONF_EXTRA_JS_URL_ES5, [])
    )
    hass.data[DATA_WS_SUBSCRIBERS] = set()

    await _async_setup_themes(hass, conf.get(CONF_THEMES))

    return True


async def _async_setup_themes(
    hass: HomeAssistant,
    themes: dict[str, Any] | None,
) -> None:
    """Set up themes data and services."""
    hass.data[DATA_THEMES] = themes or {}

    store = hass.data[DATA_THEMES_STORE] = Store(
        hass, THEMES_STORAGE_VERSION, THEMES_STORAGE_KEY
    )

    if not (theme_data := await store.async_load()) or not isinstance(theme_data, dict):
        theme_data = {}
    theme_name = theme_data.get(DATA_DEFAULT_THEME, DEFAULT_THEME)
    dark_theme_name = theme_data.get(DATA_DEFAULT_DARK_THEME)

    if theme_name == DEFAULT_THEME or theme_name in hass.data[DATA_THEMES]:
        hass.data[DATA_DEFAULT_THEME] = theme_name
    else:
        hass.data[DATA_DEFAULT_THEME] = DEFAULT_THEME

    if dark_theme_name ==
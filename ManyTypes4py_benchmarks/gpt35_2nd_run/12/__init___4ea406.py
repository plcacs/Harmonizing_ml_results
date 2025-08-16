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
from homeassistant.const import CONF_MODE, CONF_NAME, EVENT_PANELS_UPDATED, EVENT_THEMES_UPDATED
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

DOMAIN: str = 'frontend'
CONF_THEMES: str = 'themes'
CONF_THEMES_MODES: str = 'modes'
CONF_THEMES_LIGHT: str = 'light'
CONF_THEMES_DARK: str = 'dark'
CONF_EXTRA_HTML_URL: str = 'extra_html_url'
CONF_EXTRA_HTML_URL_ES5: str = 'extra_html_url_es5'
CONF_EXTRA_MODULE_URL: str = 'extra_module_url'
CONF_EXTRA_JS_URL_ES5: str = 'extra_js_url_es5'
CONF_FRONTEND_REPO: str = 'development_repo'
CONF_JS_VERSION: str = 'javascript_version'
DEFAULT_THEME_COLOR: str = '#03A9F4'
DATA_PANELS: str = 'frontend_panels'
DATA_JS_VERSION: str = 'frontend_js_version'
DATA_EXTRA_MODULE_URL: str = 'frontend_extra_module_url'
DATA_EXTRA_JS_URL_ES5: str = 'frontend_extra_js_url_es5'
DATA_WS_SUBSCRIBERS: HassKey = HassKey('frontend_ws_subscribers')
THEMES_STORAGE_KEY: str = f'{DOMAIN}_theme'
THEMES_STORAGE_VERSION: int = 1
THEMES_SAVE_DELAY: int = 60
DATA_THEMES_STORE: str = 'frontend_themes_store'
DATA_THEMES: str = 'frontend_themes'
DATA_DEFAULT_THEME: str = 'frontend_default_theme'
DATA_DEFAULT_DARK_THEME: str = 'frontend_default_dark_theme'
DEFAULT_THEME: str = 'default'
VALUE_NO_THEME: str = 'none'
PRIMARY_COLOR: str = 'primary-color'
_LOGGER: logging.Logger = logging.getLogger(__name__)

class Manifest(TypedDict):
    background_color: str
    description: str
    dir: str
    display: str
    icons: list[dict[str, str]]
    screenshots: list[dict[str, str]]
    lang: str
    name: str
    short_name: str
    start_url: str
    id: str
    theme_color: str
    prefer_related_applications: bool
    related_applications: list[dict[str, str]]

MANIFEST_JSON: Manifest = Manifest({
    'background_color': '#FFFFFF',
    'description': 'Home automation platform that puts local control and privacy first.',
    'dir': 'ltr',
    'display': 'standalone',
    'icons': [{'src': f'/static/icons/favicon-{size}x{size}.png', 'sizes': f'{size}x{size}', 'type': 'image/png', 'purpose': 'any'} for size in (192, 384, 512, 1024)] + [{'src': f'/static/icons/maskable_icon-{size}x{size}.png', 'sizes': f'{size}x{size}', 'type': 'image/png', 'purpose': 'maskable'} for size in (48, 72, 96, 128, 192, 384, 512)],
    'screenshots': [{'src': '/static/images/screenshots/screenshot-1.png', 'sizes': '413x792', 'type': 'image/png'}],
    'lang': 'en-US',
    'name': 'Home Assistant',
    'short_name': 'Home Assistant',
    'start_url': '/?homescreen=1',
    'id': '/?homescreen=1',
    'theme_color': DEFAULT_THEME_COLOR,
    'prefer_related_applications': True,
    'related_applications': [{'platform': 'play', 'id': 'io.homeassistant.companion.android'}]
})

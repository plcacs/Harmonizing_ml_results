#!/usr/bin/env python3
"""Handle the frontend for Home Assistant."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Iterable
from functools import lru_cache, partial
import logging
import os
import pathlib
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

from aiohttp import hdrs, web, web_urldispatcher
import jinja2
from propcache.api import cached_property
import voluptuous as vol
from yarl import URL

from homeassistant.components import onboarding, websocket_api
from homeassistant.components.http import KEY_HASS, HomeAssistantView, StaticPathConfig
from homeassistant.components.websocket_api import ActiveConnection
from homeassistant.config import async_hass_config_yaml
from homeassistant.const import CONF_MODE, CONF_NAME, EVENT_PANELS_UPDATED, EVENT_THEMES_UPDATED, DEFAULT_THEME_COLOR
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
_LOGGER = logging.getLogger(__name__)

EXTENDED_THEME_SCHEMA = vol.Schema({
    cv.string: cv.string,
    vol.Optional(CONF_THEMES_MODES): vol.Schema({
        vol.Optional(CONF_THEMES_LIGHT): vol.Schema({cv.string: cv.string}),
        vol.Optional(CONF_THEMES_DARK): vol.Schema({cv.string: cv.string})
    })
})
THEME_SCHEMA = vol.Schema({cv.string: vol.Any({cv.string: cv.string}, EXTENDED_THEME_SCHEMA)})
CONFIG_SCHEMA = vol.Schema({
    DOMAIN: vol.Schema({
        vol.Optional(CONF_FRONTEND_REPO): cv.isdir,
        vol.Optional(CONF_THEMES): THEME_SCHEMA,
        vol.Optional(CONF_EXTRA_MODULE_URL): vol.All(cv.ensure_list, [cv.string]),
        vol.Optional(CONF_EXTRA_JS_URL_ES5): vol.All(cv.ensure_list, [cv.string]),
        vol.Optional(CONF_EXTRA_HTML_URL): cv.match_all,
        vol.Optional(CONF_EXTRA_HTML_URL_ES5): cv.match_all,
        vol.Optional(CONF_JS_VERSION): cv.match_all
    })
}, extra=vol.ALLOW_EXTRA)

SERVICE_SET_THEME: str = 'set_theme'
SERVICE_RELOAD_THEMES: str = 'reload_themes'


class Manifest:
    """Manage the manifest.json contents."""

    def __init__(self, data: Dict[str, Any]) -> None:
        """Init the manifest manager."""
        self.manifest: Dict[str, Any] = data
        self._serialize()

    def __getitem__(self, key: str) -> Any:
        """Return an item in the manifest."""
        return self.manifest[key]

    @property
    def json(self) -> str:
        """Return the serialized manifest."""
        return self._serialized

    def _serialize(self) -> None:
        self._serialized: str = json_dumps_sorted(self.manifest)

    def update_key(self, key: str, val: Any) -> None:
        """Add a keyval to the manifest.json."""
        self.manifest[key] = val
        self._serialize()


MANIFEST_JSON: Manifest = Manifest({
    'background_color': '#FFFFFF',
    'description': 'Home automation platform that puts local control and privacy first.',
    'dir': 'ltr',
    'display': 'standalone',
    'icons': [
        {
            'src': f'/static/icons/favicon-{size}x{size}.png',
            'sizes': f'{size}x{size}',
            'type': 'image/png',
            'purpose': 'any'
        } for size in (192, 384, 512, 1024)
    ] + [
        {
            'src': f'/static/icons/maskable_icon-{size}x{size}.png',
            'sizes': f'{size}x{size}',
            'type': 'image/png',
            'purpose': 'maskable'
        } for size in (48, 72, 96, 128, 192, 384, 512)
    ],
    'screenshots': [{
        'src': '/static/images/screenshots/screenshot-1.png',
        'sizes': '413x792',
        'type': 'image/png'
    }],
    'lang': 'en-US',
    'name': 'Home Assistant',
    'short_name': 'Home Assistant',
    'start_url': '/?homescreen=1',
    'id': '/?homescreen=1',
    'theme_color': DEFAULT_THEME_COLOR,
    'prefer_related_applications': True,
    'related_applications': [{
        'platform': 'play',
        'id': 'io.homeassistant.companion.android'
    }]
})


class UrlManager:
    """Manage urls to be used on the frontend.

    This is abstracted into a class because
    some integrations add a remove these directly on hass.data
    """

    def __init__(self, on_change: Callable[[str, str, str], None], urls: Iterable[str]) -> None:
        """Init the url manager."""
        self._on_change: Callable[[str, str, str], None] = on_change
        self.urls: frozenset[str] = frozenset(urls)

    def add(self, url: str) -> None:
        """Add a url to the set."""
        self.urls = frozenset([*self.urls, url])
        self._on_change('added', url, url)

    def remove(self, url: str) -> None:
        """Remove a url from the set."""
        self.urls = self.urls - {url}
        self._on_change('removed', url, url)


class Panel:
    """Abstract class for panels."""
    sidebar_icon: Optional[str] = None
    sidebar_title: Optional[str] = None
    frontend_url_path: Optional[str] = None
    config: Any = None
    require_admin: bool = False
    config_panel_domain: Optional[str] = None

    def __init__(
        self,
        component_name: str,
        sidebar_title: Optional[str],
        sidebar_icon: Optional[str],
        frontend_url_path: Optional[str],
        config: Any,
        require_admin: bool,
        config_panel_domain: Optional[str]
    ) -> None:
        """Initialize a built-in panel."""
        self.component_name: str = component_name
        self.sidebar_title = sidebar_title
        self.sidebar_icon = sidebar_icon
        self.frontend_url_path = frontend_url_path or component_name
        self.config = config
        self.require_admin = require_admin
        self.config_panel_domain = config_panel_domain

    @callback
    def to_response(self) -> Dict[str, Any]:
        """Panel as dictionary."""
        return {
            'component_name': self.component_name,
            'icon': self.sidebar_icon,
            'title': self.sidebar_title,
            'config': self.config,
            'url_path': self.frontend_url_path,
            'require_admin': self.require_admin,
            'config_panel_domain': self.config_panel_domain
        }


@bind_hass
@callback
def async_register_built_in_panel(
    hass: HomeAssistant,
    component_name: str,
    sidebar_title: Optional[str] = None,
    sidebar_icon: Optional[str] = None,
    frontend_url_path: Optional[str] = None,
    config: Any = None,
    require_admin: bool = False,
    *,
    update: bool = False,
    config_panel_domain: Optional[str] = None
) -> None:
    """Register a built-in panel."""
    panel = Panel(component_name, sidebar_title, sidebar_icon, frontend_url_path, config, require_admin, config_panel_domain)
    panels: Dict[str, Panel] = hass.data.setdefault(DATA_PANELS, {})
    if not update and panel.frontend_url_path in panels:
        raise ValueError(f'Overwriting panel {panel.frontend_url_path}')
    panels[panel.frontend_url_path] = panel
    hass.bus.async_fire(EVENT_PANELS_UPDATED)


@bind_hass
@callback
def async_remove_panel(
    hass: HomeAssistant,
    frontend_url_path: str,
    *,
    warn_if_unknown: bool = True
) -> None:
    """Remove a built-in panel."""
    panel: Optional[Panel] = hass.data.get(DATA_PANELS, {}).pop(frontend_url_path, None)
    if panel is None:
        if warn_if_unknown:
            _LOGGER.warning('Removing unknown panel %s', frontend_url_path)
        return
    hass.bus.async_fire(EVENT_PANELS_UPDATED)


def add_extra_js_url(hass: HomeAssistant, url: str, es5: bool = False) -> None:
    """Register extra js or module url to load.

    This function allows custom integrations to register extra js or module.
    """
    key: str = DATA_EXTRA_JS_URL_ES5 if es5 else DATA_EXTRA_MODULE_URL
    hass.data[key].add(url)


def remove_extra_js_url(hass: HomeAssistant, url: str, es5: bool = False) -> None:
    """Remove extra js or module url to load.

    This function allows custom integrations to remove extra js or module.
    """
    key: str = DATA_EXTRA_JS_URL_ES5 if es5 else DATA_EXTRA_MODULE_URL
    hass.data[key].remove(url)


def add_manifest_json_key(key: str, val: Any) -> None:
    """Add a keyval to the manifest.json."""
    MANIFEST_JSON.update_key(key, val)


def _frontend_root(dev_repo_path: Optional[str]) -> pathlib.Path:
    """Return root path to the frontend files."""
    if dev_repo_path is not None:
        return pathlib.Path(dev_repo_path) / 'hass_frontend'
    import hass_frontend  # type: ignore
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
    conf: Dict[str, Any] = config.get(DOMAIN, {})
    for key in (CONF_EXTRA_HTML_URL, CONF_EXTRA_HTML_URL_ES5, CONF_JS_VERSION):
        if key in conf:
            _LOGGER.error('Please remove %s from your frontend config. It is no longer supported', key)
    repo_path: Optional[str] = conf.get(CONF_FRONTEND_REPO)
    is_dev: bool = repo_path is not None
    root_path: pathlib.Path = _frontend_root(repo_path)
    static_paths_configs: List[StaticPathConfig] = []
    for path, should_cache in (
        ('service_worker.js', False),
        ('sw-modern.js', False),
        ('sw-modern.js.map', False),
        ('sw-legacy.js', False),
        ('sw-legacy.js.map', False),
        ('robots.txt', False),
        ('onboarding.html', not is_dev),
        ('static', not is_dev),
        ('frontend_latest', not is_dev),
        ('frontend_es5', not is_dev)
    ):
        static_paths_configs.append(StaticPathConfig(f'/{path}', str(root_path / path), should_cache))
    static_paths_configs.append(StaticPathConfig('/auth/authorize', str(root_path / 'authorize.html'), False))
    hass.http.register_redirect('/.well-known/change-password', '/profile', redirect_exc=web.HTTPFound)
    local: str = hass.config.path('www')
    if await hass.async_add_executor_job(os.path.isdir, local):
        static_paths_configs.append(StaticPathConfig('/local', local, not is_dev))
    await hass.http.async_register_static_paths(static_paths_configs)
    hass.http.register_redirect('/shopping-list', '/todo')
    hass.http.app.router.register_resource(IndexView(repo_path, hass))
    async_register_built_in_panel(hass, 'profile')
    async_register_built_in_panel(hass, 'developer-tools', require_admin=True, sidebar_title='developer_tools', sidebar_icon='hass:hammer')

    @callback
    def async_change_listener(resource_type: str, change_type: str, url: str) -> None:
        subscribers: Set[Tuple[ActiveConnection, Any]] = hass.data[DATA_WS_SUBSCRIBERS]
        json_msg: Dict[str, Any] = {'change_type': change_type, 'item': {'type': resource_type, 'url': url}}
        for connection, msg_id in subscribers:
            connection.send_message(websocket_api.event_message(msg_id, json_msg))

    hass.data[DATA_EXTRA_MODULE_URL] = UrlManager(
        partial(async_change_listener, 'module'),
        conf.get(CONF_EXTRA_MODULE_URL, [])
    )
    hass.data[DATA_EXTRA_JS_URL_ES5] = UrlManager(
        partial(async_change_listener, 'es5'),
        conf.get(CONF_EXTRA_JS_URL_ES5, [])
    )
    hass.data[DATA_WS_SUBSCRIBERS] = set()  # type: Set[Tuple[ActiveConnection, Any]]
    await _async_setup_themes(hass, conf.get(CONF_THEMES))
    return True


async def _async_setup_themes(hass: HomeAssistant, themes: Optional[Dict[str, Any]]) -> None:
    """Set up themes data and services."""
    hass.data[DATA_THEMES] = themes or {}
    store: Store = hass.data[DATA_THEMES_STORE] = Store(hass, THEMES_STORAGE_VERSION, THEMES_STORAGE_KEY)
    theme_data: Optional[Dict[str, Any]] = await store.async_load()
    if not (theme_data) or not isinstance(theme_data, dict):
        theme_data = {}
    theme_name: str = theme_data.get(DATA_DEFAULT_THEME, DEFAULT_THEME)
    dark_theme_name: Optional[str] = theme_data.get(DATA_DEFAULT_DARK_THEME)
    if theme_name == DEFAULT_THEME or theme_name in hass.data[DATA_THEMES]:
        hass.data[DATA_DEFAULT_THEME] = theme_name
    else:
        hass.data[DATA_DEFAULT_THEME] = DEFAULT_THEME
    if dark_theme_name == DEFAULT_THEME or (dark_theme_name and dark_theme_name in hass.data[DATA_THEMES]):
        hass.data[DATA_DEFAULT_DARK_THEME] = dark_theme_name

    @callback
    def update_theme_and_fire_event() -> None:
        """Update theme_color in manifest."""
        name: str = hass.data[DATA_DEFAULT_THEME]
        themes_dict: Dict[str, Any] = hass.data[DATA_THEMES]
        if name != DEFAULT_THEME:
            MANIFEST_JSON.update_key(
                'theme_color',
                themes_dict[name].get('app-header-background-color', themes_dict[name].get(PRIMARY_COLOR, DEFAULT_THEME_COLOR))
            )
        else:
            MANIFEST_JSON.update_key('theme_color', DEFAULT_THEME_COLOR)
        hass.bus.async_fire(EVENT_THEMES_UPDATED)

    @callback
    def set_theme(call: ServiceCall) -> None:
        """Set backend-preferred theme."""
        name: str = call.data[CONF_NAME]
        mode: str = call.data.get('mode', 'light')
        if name not in (DEFAULT_THEME, VALUE_NO_THEME) and name not in hass.data[DATA_THEMES]:
            _LOGGER.warning('Theme %s not found', name)
            return
        light_mode: bool = mode == 'light'
        theme_key: str = DATA_DEFAULT_THEME if light_mode else DATA_DEFAULT_DARK_THEME
        if name == VALUE_NO_THEME:
            to_set: Optional[str] = DEFAULT_THEME if light_mode else None
        else:
            _LOGGER.info('Theme %s set as default %s theme', name, mode)
            to_set = name
        hass.data[theme_key] = to_set
        store.async_delay_save(
            lambda: {
                DATA_DEFAULT_THEME: hass.data[DATA_DEFAULT_THEME],
                DATA_DEFAULT_DARK_THEME: hass.data.get(DATA_DEFAULT_DARK_THEME)
            },
            THEMES_SAVE_DELAY
        )
        update_theme_and_fire_event()

    async def reload_themes(_: Any) -> None:
        """Reload themes."""
        config_yaml: Dict[str, Any] = await async_hass_config_yaml(hass)
        new_themes: Dict[str, Any] = config_yaml.get(DOMAIN, {}).get(CONF_THEMES, {})
        hass.data[DATA_THEMES] = new_themes
        if hass.data[DATA_DEFAULT_THEME] not in new_themes:
            hass.data[DATA_DEFAULT_THEME] = DEFAULT_THEME
        if hass.data.get(DATA_DEFAULT_DARK_THEME) and hass.data.get(DATA_DEFAULT_DARK_THEME) not in new_themes:
            hass.data[DATA_DEFAULT_DARK_THEME] = None
        update_theme_and_fire_event()

    service.async_register_admin_service(
        hass, DOMAIN, SERVICE_SET_THEME, set_theme,
        vol.Schema({vol.Required(CONF_NAME): cv.string, vol.Optional(CONF_MODE): vol.Any('dark', 'light')})
    )
    service.async_register_admin_service(
        hass, DOMAIN, SERVICE_RELOAD_THEMES, reload_themes
    )


@callback
@lru_cache(maxsize=1)
def _async_render_index_cached(template: jinja2.Template, **kwargs: Any) -> str:
    return template.render(**kwargs)


class IndexView(web_urldispatcher.AbstractResource):
    """Serve the frontend."""

    def __init__(self, repo_path: Optional[str], hass: HomeAssistant) -> None:
        """Initialize the frontend view."""
        super().__init__(name='frontend:index')
        self.repo_path: Optional[str] = repo_path
        self.hass: HomeAssistant = hass
        self._template_cache: Optional[jinja2.Template] = None

    @cached_property
    def canonical(self) -> str:
        """Return resource's canonical path."""
        return '/'

    @cached_property
    def _route(self) -> web_urldispatcher.ResourceRoute:
        """Return the index route."""
        return web_urldispatcher.ResourceRoute('GET', self.get, self)

    def url_for(self, **kwargs: Any) -> URL:
        """Construct url for resource with additional params."""
        return URL('/')

    async def resolve(self, request: web.Request) -> Tuple[Optional[web_urldispatcher.UrlMappingMatchInfo], Set[str]]:
        """Resolve resource.

        Return (UrlMappingMatchInfo, allowed_methods) pair.
        """
        if request.path != '/' and (parts := request.rel_url.parts) and (len(parts) > 1) and (parts[1] not in self.hass.data[DATA_PANELS]):
            return (None, set())
        if request.method != hdrs.METH_GET:
            return (None, {'GET'})
        return (web_urldispatcher.UrlMappingMatchInfo({}, self._route), {'GET'})

    def get_info(self) -> Dict[str, Any]:
        """Return a dict with additional info useful for introspection."""
        return {'panels': list(self.hass.data[DATA_PANELS])}

    def raw_match(self, path: str) -> bool:
        """Perform a raw match against path."""
        return False

    def get_template(self) -> jinja2.Template:
        """Get template."""
        if (tpl := self._template_cache) is None:
            with (_frontend_root(self.repo_path) / 'index.html').open(encoding='utf8') as file:
                tpl = jinja2.Template(file.read())
            if self.repo_path is None:
                self._template_cache = tpl
        return tpl

    async def get(self, request: web.Request) -> web.Response:
        """Serve the index page for panel pages."""
        hass: HomeAssistant = request.app[KEY_HASS]
        if not onboarding.async_is_onboarded(hass):
            return web.Response(status=302, headers={'location': '/onboarding.html'})
        template: jinja2.Template = self._template_cache or await hass.async_add_executor_job(self.get_template)
        if hass.config.safe_mode:
            extra_modules: frozenset[str] = frozenset()
            extra_js_es5: frozenset[str] = frozenset()
        else:
            extra_modules = hass.data[DATA_EXTRA_MODULE_URL].urls
            extra_js_es5 = hass.data[DATA_EXTRA_JS_URL_ES5].urls
        response: web.Response = web.Response(
            text=_async_render_index_cached(
                template,
                theme_color=MANIFEST_JSON['theme_color'],
                extra_modules=extra_modules,
                extra_js_es5=extra_js_es5
            ),
            content_type='text/html'
        )
        response.enable_compression()
        return response

    def __len__(self) -> int:
        """Return length of resource."""
        return 1

    def __iter__(self) -> Iterator[web_urldispatcher.ResourceRoute]:
        """Iterate over routes."""
        return iter([self._route])


class ManifestJSONView(HomeAssistantView):
    """View to return a manifest.json."""
    requires_auth: bool = False
    url: str = '/manifest.json'
    name: str = 'manifestjson'

    @callback
    def get(self, request: web.Request) -> web.Response:
        """Return the manifest.json."""
        response: web.Response = web.Response(text=MANIFEST_JSON.json, content_type='application/manifest+json')
        response.enable_compression()
        return response


@websocket_api.websocket_command({
    'type': 'frontend/get_icons',
    vol.Required('category'): vol.In({'entity', 'entity_component', 'services'}),
    vol.Optional('integration'): vol.All(cv.ensure_list, [str])
})
@websocket_api.async_response
async def websocket_get_icons(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle get icons command."""
    resources: Any = await async_get_icons(hass, msg['category'], msg.get('integration'))
    connection.send_message(websocket_api.result_message(msg['id'], {'resources': resources}))


@callback
@websocket_api.websocket_command({'type': 'get_panels'})
def websocket_get_panels(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle get panels command."""
    user_is_admin: bool = connection.user.is_admin
    panels: Dict[str, Any] = {
        panel_key: panel.to_response() for panel_key, panel in connection.hass.data[DATA_PANELS].items()
        if user_is_admin or not panel.require_admin
    }
    connection.send_message(websocket_api.result_message(msg['id'], panels))


@callback
@websocket_api.websocket_command({'type': 'frontend/get_themes'})
def websocket_get_themes(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle get themes command."""
    if hass.config.recovery_mode or hass.config.safe_mode:
        connection.send_message(websocket_api.result_message(msg['id'], {'themes': {}, 'default_theme': 'default'}))
        return
    connection.send_message(websocket_api.result_message(msg['id'], {
        'themes': hass.data[DATA_THEMES],
        'default_theme': hass.data[DATA_DEFAULT_THEME],
        'default_dark_theme': hass.data.get(DATA_DEFAULT_DARK_THEME)
    }))


@websocket_api.websocket_command({
    'type': 'frontend/get_translations',
    vol.Required('language'): str,
    vol.Required('category'): str,
    vol.Optional('integration'): vol.All(cv.ensure_list, [str]),
    vol.Optional('config_flow'): bool
})
@websocket_api.async_response
async def websocket_get_translations(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle get translations command."""
    resources: Any = await async_get_translations(
        hass, msg['language'], msg['category'], msg.get('integration'), msg.get('config_flow')
    )
    connection.send_message(websocket_api.result_message(msg['id'], {'resources': resources}))


@websocket_api.websocket_command({'type': 'frontend/get_version'})
@websocket_api.async_response
async def websocket_get_version(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Handle get version command."""
    integration = await async_get_integration(hass, 'frontend')
    frontend: Optional[str] = None
    for req in integration.requirements:
        if req.startswith('home-assistant-frontend=='):
            frontend = req.removeprefix('home-assistant-frontend==')
    if frontend is None:
        connection.send_error(msg['id'], 'unknown_version', 'Version not found')
    else:
        connection.send_result(msg['id'], {'version': frontend})


@callback
@websocket_api.websocket_command({'type': 'frontend/subscribe_extra_js'})
def websocket_subscribe_extra_js(hass: HomeAssistant, connection: ActiveConnection, msg: Dict[str, Any]) -> None:
    """Subscribe to URL manager updates."""
    subscribers: Set[Tuple[ActiveConnection, Any]] = hass.data[DATA_WS_SUBSCRIBERS]
    subscribers.add((connection, msg['id']))

    @callback
    def cancel_subscription() -> None:
        subscribers.remove((connection, msg['id']))

    connection.subscriptions[msg['id']] = cancel_subscription
    connection.send_message(websocket_api.result_message(msg['id']))


class PanelRespons(TypedDict):
    """Represent the panel response type."""
    component_name: str
    icon: Optional[str]
    title: Optional[str]
    config: Any
    url_path: str
    require_admin: bool
    config_panel_domain: Optional[str]
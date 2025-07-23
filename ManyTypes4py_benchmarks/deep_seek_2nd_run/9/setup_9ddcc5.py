"""All methods needed to bootstrap a Home Assistant instance."""
from __future__ import annotations
import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable, Generator, Mapping, Iterable
import contextlib
import contextvars
from enum import StrEnum
from functools import partial
import logging.handlers
import time
from types import ModuleType
from typing import Any, Final, TypedDict, TypeVar, cast, overload
from . import config as conf_util, core, loader, requirements
from .const import BASE_PLATFORMS, EVENT_COMPONENT_LOADED, EVENT_HOMEASSISTANT_START, PLATFORM_FORMAT
from .core import CALLBACK_TYPE, DOMAIN as HOMEASSISTANT_DOMAIN, Event, HomeAssistant, callback
from .exceptions import DependencyError, HomeAssistantError
from .helpers import issue_registry as ir, singleton, translation
from .helpers.issue_registry import IssueSeverity, async_create_issue
from .helpers.typing import ConfigType
from .util.async_ import create_eager_task
from .util.hass_dict import HassKey

_T = TypeVar('_T')

current_setup_group: contextvars.ContextVar[tuple[str | None, str | None] | None] = contextvars.ContextVar('current_setup_group', default=None)
_LOGGER: Final = logging.getLogger(__name__)
ATTR_COMPONENT: Final = 'component'
DATA_SETUP: Final[HassKey[dict[str, asyncio.Future[bool]]]] = HassKey('setup_tasks')
DATA_SETUP_DONE: Final[HassKey[dict[str, asyncio.Future[bool]]]] = HassKey('setup_done')
DATA_SETUP_STARTED: Final[HassKey[dict[tuple[str, str | None], float]]] = HassKey('setup_started')
DATA_SETUP_TIME: Final[HassKey[defaultdict[str, defaultdict[str | None, defaultdict[str, float]]]]] = HassKey('setup_time')
DATA_DEPS_REQS: Final[HassKey[set[str]]] = HassKey('deps_reqs_processed')
DATA_PERSISTENT_ERRORS: Final[HassKey[dict[str, str | None]]] = HassKey('bootstrap_persistent_errors')
NOTIFY_FOR_TRANSLATION_KEYS: Final[list[str]] = ['config_validation_err', 'platform_config_validation_err']
SLOW_SETUP_WARNING: Final = 10
SLOW_SETUP_MAX_WAIT: Final = 300

class EventComponentLoaded(TypedDict):
    """EventComponentLoaded data."""
    component: str

@callback
def async_notify_setup_error(hass: HomeAssistant, component: str, display_link: str | None = None) -> None:
    """Print a persistent notification.

    This method must be run in the event loop.
    """
    from .components import persistent_notification
    if (errors := hass.data.get(DATA_PERSISTENT_ERRORS)) is None:
        errors = hass.data[DATA_PERSISTENT_ERRORS] = {}
    errors[component] = errors.get(component) or display_link
    message = 'The following integrations and platforms could not be set up:\n\n'
    for name, link in errors.items():
        show_logs = f'[Show logs](/config/logs?filter={name})'
        part = f'[{name}]({link})' if link else name
        message += f' - {part} ({show_logs})\n'
    message += '\nPlease check your config and [logs](/config/logs).'
    persistent_notification.async_create(hass, message, 'Invalid config', 'invalid_config')

@core.callback
def async_set_domains_to_be_loaded(hass: HomeAssistant, domains: set[str]) -> None:
    """Set domains that are going to be loaded from the config.

    This allow us to:
     - Properly handle after_dependencies.
     - Keep track of domains which will load but have not yet finished loading
    """
    setup_done_futures = hass.data.setdefault(DATA_SETUP_DONE, {})
    setup_futures = hass.data.setdefault(DATA_SETUP, {})
    old_domains = set(setup_futures) | set(setup_done_futures) | hass.config.components
    if (overlap := (old_domains & domains)):
        _LOGGER.debug('Domains to be loaded %s already loaded or pending', overlap)
    setup_done_futures.update({domain: hass.loop.create_future() for domain in domains - old_domains})

def setup_component(hass: HomeAssistant, domain: str, config: ConfigType) -> bool:
    """Set up a component and all its dependencies."""
    return asyncio.run_coroutine_threadsafe(async_setup_component(hass, domain, config), hass.loop).result()

async def async_setup_component(hass: HomeAssistant, domain: str, config: ConfigType) -> bool:
    """Set up a component and all its dependencies.

    This method is a coroutine.
    """
    if domain in hass.config.components:
        return True
    setup_futures = hass.data.setdefault(DATA_SETUP, {})
    setup_done_futures = hass.data.setdefault(DATA_SETUP_DONE, {})
    if (existing_setup_future := setup_futures.get(domain)):
        return await existing_setup_future
    setup_future = hass.loop.create_future()
    setup_futures[domain] = setup_future
    try:
        result = await _async_setup_component(hass, domain, config)
        setup_future.set_result(result)
        if (setup_done_future := setup_done_futures.pop(domain, None)):
            setup_done_future.set_result(result)
    except BaseException as err:
        futures = [setup_future]
        if (setup_done_future := setup_done_futures.pop(domain, None)):
            futures.append(setup_done_future)
        for future in futures:
            future.set_exception(err)
            with contextlib.suppress(BaseException):
                await future
        raise
    return result

async def _async_process_dependencies(hass: HomeAssistant, config: ConfigType, integration: loader.Integration) -> list[str]:
    """Ensure all dependencies are set up.

    Returns a list of dependencies which failed to set up.
    """
    setup_futures = hass.data.setdefault(DATA_SETUP, {})
    dependencies_tasks = {dep: setup_futures.get(dep) or create_eager_task(async_setup_component(hass, dep, config), name=f'setup {dep} as dependency of {integration.domain}', loop=hass.loop) for dep in integration.dependencies if dep not in hass.config.components}
    after_dependencies_tasks = {}
    to_be_loaded = hass.data.get(DATA_SETUP_DONE, {})
    for dep in integration.after_dependencies:
        if dep not in dependencies_tasks and dep in to_be_loaded and (dep not in hass.config.components):
            after_dependencies_tasks[dep] = to_be_loaded[dep]
    if not dependencies_tasks and (not after_dependencies_tasks):
        return []
    if dependencies_tasks:
        _LOGGER.debug('Dependency %s will wait for dependencies %s', integration.domain, dependencies_tasks.keys())
    if after_dependencies_tasks:
        _LOGGER.debug('Dependency %s will wait for after dependencies %s', integration.domain, after_dependencies_tasks.keys())
    async with hass.timeout.async_freeze(integration.domain):
        results = await asyncio.gather(*dependencies_tasks.values(), *after_dependencies_tasks.values())
    failed = [domain for idx, domain in enumerate(dependencies_tasks) if not results[idx]]
    if failed:
        _LOGGER.error("Unable to set up dependencies of '%s'. Setup failed for dependencies: %s", integration.domain, failed)
    return failed

def _log_error_setup_error(hass: HomeAssistant, domain: str, integration: loader.Integration | None, msg: str, exc_info: BaseException | None = None) -> None:
    """Log helper."""
    if integration is None:
        custom = ''
        link = None
    else:
        custom = '' if integration.is_built_in else 'custom integration '
        link = integration.documentation
    _LOGGER.error("Setup failed for %s'%s': %s", custom, domain, msg, exc_info=exc_info)
    async_notify_setup_error(hass, domain, link)

async def _async_setup_component(hass: HomeAssistant, domain: str, config: ConfigType) -> bool:
    """Set up a component for Home Assistant.

    This method is a coroutine.
    """
    try:
        integration = await loader.async_get_integration(hass, domain)
    except loader.IntegrationNotFound:
        _log_error_setup_error(hass, domain, None, 'Integration not found.')
        if not hass.config.safe_mode and hass.config_entries.async_entries(domain):
            ir.async_create_issue(hass, HOMEASSISTANT_DOMAIN, f'integration_not_found.{domain}', is_fixable=True, issue_domain=HOMEASSISTANT_DOMAIN, severity=IssueSeverity.ERROR, translation_key='integration_not_found', translation_placeholders={'domain': domain}, data={'domain': domain})
        return False
    log_error = partial(_log_error_setup_error, hass, domain, integration)
    if integration.disabled:
        log_error(f'Dependency is disabled - {integration.disabled}')
        return False
    integration_set = {domain}
    load_translations_task = None
    if integration.has_translations and (not translation.async_translations_loaded(hass, integration_set)):
        load_translations_task = create_eager_task(translation.async_load_integrations(hass, integration_set), loop=hass.loop)
    if not await integration.resolve_dependencies():
        return False
    try:
        await async_process_deps_reqs(hass, config, integration)
    except HomeAssistantError as err:
        log_error(str(err))
        return False
    try:
        component = await integration.async_get_component()
    except ImportError as err:
        log_error(f'Unable to import component: {err}', err)
        return False
    integration_config_info = await conf_util.async_process_component_config(hass, config, integration, component)
    conf_util.async_handle_component_errors(hass, integration_config_info, integration)
    processed_config = conf_util.async_drop_config_annotations(integration_config_info, integration)
    for platform_exception in integration_config_info.exception_info_list:
        if platform_exception.translation_key not in NOTIFY_FOR_TRANSLATION_KEYS:
            continue
        async_notify_setup_error(hass, platform_exception.platform_path, platform_exception.integration_link)
    if processed_config is None:
        log_error('Invalid config.')
        return False
    if domain in processed_config and (not hasattr(component, 'async_setup')) and (not hasattr(component, 'setup')) and (not hasattr(component, 'CONFIG_SCHEMA')):
        _LOGGER.error("The '%s' integration does not support YAML setup, please remove it from your configuration", domain)
        async_create_issue(hass, HOMEASSISTANT_DOMAIN, f'config_entry_only_{domain}', is_fixable=False, severity=IssueSeverity.ERROR, issue_domain=domain, translation_key='config_entry_only', translation_placeholders={'domain': domain, 'add_integration': f'/config/integrations/dashboard/add?domain={domain}'})
    _LOGGER.info('Setting up %s', domain)
    with async_start_setup(hass, integration=domain, phase=SetupPhases.SETUP):
        if hasattr(component, 'PLATFORM_SCHEMA'):
            warn_task = None
        else:
            warn_task = hass.loop.call_later(SLOW_SETUP_WARNING, _LOGGER.warning, 'Setup of %s is taking over %s seconds.', domain, SLOW_SETUP_WARNING)
        task = None
        result = True
        try:
            if hasattr(component, 'async_setup'):
                task = component.async_setup(hass, processed_config)
            elif hasattr(component, 'setup'):
                task = hass.loop.run_in_executor(None, component.setup, hass, processed_config)
            elif not hasattr(component, 'async_setup_entry'):
                log_error('No setup or config entry setup function defined.')
                return False
            if task:
                async with hass.timeout.async_timeout(SLOW_SETUP_MAX_WAIT, domain):
                    result = await task
        except TimeoutError:
            _LOGGER.error("Setup of '%s' is taking longer than %s seconds. Startup will proceed without waiting any longer", domain, SLOW_SETUP_MAX_WAIT)
            return False
        except (asyncio.CancelledError, SystemExit, Exception) as exc:
            _LOGGER.exception('Error during setup of component %s: %s', domain, exc)
            async_notify_setup_error(hass, domain, integration.documentation)
            return False
        finally:
            if warn_task:
                warn_task.cancel()
        if result is False:
            log_error('Integration failed to initialize.')
            return False
        if result is not True:
            log_error(f'Integration {domain!r} did not return boolean if setup was successful. Disabling component.')
            return False
        if load_translations_task:
            await load_translations_task
    if integration.platforms_exists(('config_flow',)):
        await hass.config_entries.flow.async_wait_import_flow_initialized(domain)
    hass.config.components.add(domain)
    if (entries := hass.config_entries.async_entries(domain, include_ignore=False, include_disabled=False)):
        await asyncio.gather(*(create_eager_task(entry.async_setup_locked(hass, integration=integration), name=f'config entry setup {entry.title} {entry.domain} {entry.entry_id}', loop=hass.loop) for entry in entries))
    hass.data[DATA_SETUP].pop(domain, None)
    hass.bus.async_fire_internal(EVENT_COMPONENT_LOADED, EventComponentLoaded(component=domain))
    return True

async def async_prepare_setup_platform(hass: HomeAssistant, hass_config: ConfigType, domain: str, platform_name: str) -> Any | None:
    """Load a platform and makes sure dependencies are setup.

    This method is a coroutine.
    """
    platform_path = PLATFORM_FORMAT.format(domain=domain, platform=platform_name)

    def log_error(msg: str) -> None:
        """Log helper."""
        _LOGGER.error("Unable to prepare setup for platform '%s': %s", platform_path, msg)
        async_notify_setup_error(hass, platform_path)
    try:
        integration = await loader.async_get_integration(hass, platform_name)
    except loader.IntegrationNotFound:
        log_error('Integration not found')
        return None
    if (load_top_level_component := (integration.domain not in hass.config.components)):
        try:
            await async_process_deps_reqs(hass, hass_config, integration)
        except HomeAssistantError as err:
            log_error(str(err))
            return None
        try:
            component = await integration.async_get_component()
        except ImportError as exc:
            log_error(f'Unable to import the component ({exc}).')
            return None
    if not integration.platforms_exists((domain,)):
        log_error(f"Platform not found (No module named '{integration.pkg_path}.{domain}')")
        return None
    try:
        platform = await integration.async_get_platform(domain)
    except ImportError as exc:
        log_error(f'Platform not found ({exc}).')
        return None
    if platform_path in hass.config.components:
        return platform
    if load_top_level_component:
        if (hasattr(component, 'setup') or hasattr(component, 'async_setup')) and (not await async_setup_component(hass, integration.domain, hass_config)):
            log_error('Unable to set up component.')
            return None
    return platform

async def async_process_deps_reqs(hass: HomeAssistant, config: ConfigType, integration: loader.Integration) -> None:
    """Process all dependencies and requirements for a module.

    Module is a Python module of either a component or platform.
    """
    if (processed := hass.data.get(DATA_DEPS_REQS)) is None:
        processed = hass.data[DATA_DEPS_REQS] = set()
    elif integration.domain in processed:
        return
    if (failed_deps := (await _async_process_dependencies(hass, config, integration))):
        raise DependencyError(failed_deps)
    async with hass.timeout.async_freeze(integration.domain):
        await requirements.async_get_integration_with_requirements(hass, integration.domain)
    processed.add(integration.domain)

@core.callback
def async_when_setup(hass: HomeAssistant, component: str, when_setup_cb: Callable[[HomeAssistant, str], Awaitable[None]]) -> None:
    """Call a method when a component is setup."""
    _async_when_setup(hass, component, when_setup_cb, False)

@core.callback
def async_when_setup_or_start(hass: HomeAssistant, component: str, when_setup_cb: Callable[[HomeAssistant, str], Awaitable[None]]) -> None:
    """Call a method when a component is setup or state is fired."""
    _async_when_setup(hass, component, when_setup_cb, True)

@core.callback
def _async_when_setup(hass: HomeAssistant, component: str, when_setup_cb: Callable[[HomeAssistant, str], Awaitable[None]], start_event: bool) -> None:
    """Call a method when a component is setup or the start event fires."""

    async def when_setup() -> None:
        """Call the callback."""
        try:
            await when_setup_cb(hass, component)
        except Exception:
            _LOGGER.exception('Error handling when_setup callback for %s', component)
    if component in hass.config.com
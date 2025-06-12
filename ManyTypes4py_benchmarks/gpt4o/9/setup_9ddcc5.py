"""All methods needed to bootstrap a Home Assistant instance."""
from __future__ import annotations
import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable, Generator, Mapping
import contextlib
import contextvars
from enum import StrEnum
from functools import partial
import logging.handlers
import time
from types import ModuleType
from typing import Any, Final, TypedDict, Optional, Union
from . import config as conf_util, core, loader, requirements
from .const import BASE_PLATFORMS, EVENT_COMPONENT_LOADED, EVENT_HOMEASSISTANT_START, PLATFORM_FORMAT
from .core import CALLBACK_TYPE, DOMAIN as HOMEASSISTANT_DOMAIN, Event, HomeAssistant, callback
from .exceptions import DependencyError, HomeAssistantError
from .helpers import issue_registry as ir, singleton, translation
from .helpers.issue_registry import IssueSeverity, async_create_issue
from .helpers.typing import ConfigType
from .util.async_ import create_eager_task
from .util.hass_dict import HassKey

current_setup_group: contextvars.ContextVar[Optional[tuple[str, Optional[str]]]] = contextvars.ContextVar('current_setup_group', default=None)
_LOGGER = logging.getLogger(__name__)

ATTR_COMPONENT: Final = 'component'
DATA_SETUP: Final = HassKey('setup_tasks')
DATA_SETUP_DONE: Final = HassKey('setup_done')
DATA_SETUP_STARTED: Final = HassKey('setup_started')
DATA_SETUP_TIME: Final = HassKey('setup_time')
DATA_DEPS_REQS: Final = HassKey('deps_reqs_processed')
DATA_PERSISTENT_ERRORS: Final = HassKey('bootstrap_persistent_errors')
NOTIFY_FOR_TRANSLATION_KEYS: Final = ['config_validation_err', 'platform_config_validation_err']
SLOW_SETUP_WARNING: Final = 10
SLOW_SETUP_MAX_WAIT: Final = 300

class EventComponentLoaded(TypedDict):
    """EventComponentLoaded data."""
    component: str

@callback
def async_notify_setup_error(hass: HomeAssistant, component: str, display_link: Optional[str] = None) -> None:
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

def _log_error_setup_error(hass: HomeAssistant, domain: str, integration: Optional[loader.Integration], msg: str, exc_info: Optional[BaseException] = None) -> None:
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
        task: Optional[Union[Awaitable[bool], asyncio.Future[bool]]] = None
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

async def async_prepare_setup_platform(hass: HomeAssistant, hass_config: ConfigType, domain: str, platform_name: str) -> Optional[ModuleType]:
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
    if component in hass.config.components:
        hass.async_create_task_internal(when_setup(), f'when setup {component}', eager_start=True)
        return
    listeners: list[Callable[[], None]] = []

    async def _matched_event(event: Event) -> None:
        """Call the callback when we matched an event."""
        for listener in listeners:
            listener()
        await when_setup()

    @callback
    def _async_is_component_filter(event_data: dict[str, Any]) -> bool:
        """Check if the event is for the component."""
        return event_data[ATTR_COMPONENT] == component
    listeners.append(hass.bus.async_listen(EVENT_COMPONENT_LOADED, _matched_event, event_filter=_async_is_component_filter))
    if start_event:
        listeners.append(hass.bus.async_listen(EVENT_HOMEASSISTANT_START, _matched_event))

@core.callback
def async_get_loaded_integrations(hass: HomeAssistant) -> set[str]:
    """Return the complete list of loaded integrations."""
    return hass.config.all_components

class SetupPhases(StrEnum):
    """Constants for setup time measurements."""
    SETUP = 'setup'
    'Set up of a component in __init__.py.'
    CONFIG_ENTRY_SETUP = 'config_entry_setup'
    'Set up of a config entry in __init__.py.'
    PLATFORM_SETUP = 'platform_setup'
    'Set up of a platform integration.\n\n    ex async_setup_platform or setup_platform or\n    a legacy platform like device_tracker.legacy\n    '
    CONFIG_ENTRY_PLATFORM_SETUP = 'config_entry_platform_setup'
    'Set up of a platform in a config entry after the config entry is setup.\n\n    This is only for platforms that are not awaited in async_setup_entry.\n    '
    WAIT_BASE_PLATFORM_SETUP = 'wait_base_component'
    'Wait time for the base component to be setup.'
    WAIT_IMPORT_PLATFORMS = 'wait_import_platforms'
    'Wait time for the platforms to import.'
    WAIT_IMPORT_PACKAGES = 'wait_import_packages'
    'Wait time for the packages to import.'

@singleton.singleton(DATA_SETUP_STARTED)
def _setup_started(hass: HomeAssistant) -> dict[tuple[str, Optional[str]], float]:
    """Return the setup started dict."""
    return {}

@contextlib.contextmanager
def async_pause_setup(hass: HomeAssistant, phase: SetupPhases) -> Generator[None, None, None]:
    """Keep track of time we are blocked waiting for other operations.

    We want to count the time we wait for importing and
    setting up the base components so we can subtract it
    from the total setup time.
    """
    if not (running := current_setup_group.get()) or running not in _setup_started(hass):
        yield
        return
    started = time.monotonic()
    try:
        yield
    finally:
        time_taken = time.monotonic() - started
        integration, group = running
        _setup_times(hass)[integration][group][phase] = -time_taken
        _LOGGER.debug('Adding wait for %s for %s (%s) of %.2f', phase, integration, group, time_taken)

@singleton.singleton(DATA_SETUP_TIME)
def _setup_times(hass: HomeAssistant) -> defaultdict[str, defaultdict[Optional[str], defaultdict[SetupPhases, float]]]:
    """Return the setup timings default dict."""
    return defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

@contextlib.contextmanager
def async_start_setup(hass: HomeAssistant, integration: str, phase: SetupPhases, group: Optional[str] = None) -> Generator[None, None, None]:
    """Keep track of when setup starts and finishes.

    :param hass: Home Assistant instance
    :param integration: The integration that is being setup
    :param phase: The phase of setup
    :param group: The group (config entry/platform instance) that is being setup

      A group is a group of setups that run in parallel.

    """
    if hass.is_stopping or hass.state is core.CoreState.running:
        yield
        return
    setup_started = _setup_started(hass)
    current = (integration, group)
    if current in setup_started:
        yield
        return
    started = time.monotonic()
    current_setup_group.set(current)
    setup_started[current] = started
    try:
        yield
    finally:
        time_taken = time.monotonic() - started
        del setup_started[current]
        group_setup_times = _setup_times(hass)[integration][group]
        group_setup_times[phase] = max(group_setup_times[phase], time_taken)
        if group is None:
            _LOGGER.info('Setup of domain %s took %.2f seconds', integration, time_taken)
        elif _LOGGER.isEnabledFor(logging.DEBUG):
            wait_time = -sum((value for value in group_setup_times.values() if value < 0))
            calculated_time = time_taken - wait_time
            _LOGGER.debug('Phase %s for %s (%s) took %.2fs (elapsed=%.2fs) (wait_time=%.2fs)', phase, integration, group, calculated_time, time_taken, wait_time)

@callback
def async_get_setup_timings(hass: HomeAssistant) -> dict[str, float]:
    """Return timing data for each integration."""
    setup_time = _setup_times(hass)
    domain_timings: dict[str, float] = {}
    for domain, timings in setup_time.items():
        top_level_timings = timings.get(None, {})
        total_top_level = sum(top_level_timings.values())
        group_totals = {group: sum(group_timings.values()) for group, group_timings in timings.items() if group is not None}
        group_max = max(group_totals.values(), default=0)
        domain_timings[domain] = total_top_level + group_max
    return domain_timings

@callback
def async_get_domain_setup_times(hass: HomeAssistant, domain: str) -> defaultdict[Optional[str], defaultdict[SetupPhases, float]]:
    """Return timing data for each integration."""
    return _setup_times(hass).get(domain, {})

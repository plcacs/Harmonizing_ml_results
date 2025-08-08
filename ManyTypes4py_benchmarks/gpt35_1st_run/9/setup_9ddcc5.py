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
from typing import Any, Final, TypedDict

from . import config as conf_util, core, loader, requirements
from .const import BASE_PLATFORMS, EVENT_COMPONENT_LOADED, EVENT_HOMEASSISTANT_START, PLATFORM_FORMAT
from .core import CALLBACK_TYPE, DOMAIN as HOMEASSISTANT_DOMAIN, Event, HomeAssistant, callback
from .exceptions import DependencyError, HomeAssistantError
from .helpers import issue_registry as ir, singleton, translation
from .helpers.issue_registry import IssueSeverity, async_create_issue
from .helpers.typing import ConfigType
from .util.async_ import create_eager_task
from .util.hass_dict import HassKey

current_setup_group: contextvars.ContextVar = contextvars.ContextVar('current_setup_group', default=None)
_LOGGER: logging.Logger = logging.getLogger(__name__)
ATTR_COMPONENT: Final[str] = 'component'
DATA_SETUP: HassKey = HassKey('setup_tasks')
DATA_SETUP_DONE: HassKey = HassKey('setup_done')
DATA_SETUP_STARTED: HassKey = HassKey('setup_started')
DATA_SETUP_TIME: HassKey = HassKey('setup_time')
DATA_DEPS_REQS: HassKey = HassKey('deps_reqs_processed')
DATA_PERSISTENT_ERRORS: HassKey = HassKey('bootstrap_persistent_errors')
NOTIFY_FOR_TRANSLATION_KEYS: list[str] = ['config_validation_err', 'platform_config_validation_err']
SLOW_SETUP_WARNING: int = 10
SLOW_SETUP_MAX_WAIT: int = 300

class EventComponentLoaded(TypedDict):
    """EventComponentLoaded data."""

@callback
def async_notify_setup_error(hass: HomeAssistant, component: str, display_link: str = None) -> None:
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

async def _async_process_dependencies(hass: HomeAssistant, config: ConfigType, integration: Integration) -> list[str]:
    """Ensure all dependencies are set up.

    Returns a list of dependencies which failed to set up.
    """

def _log_error_setup_error(hass: HomeAssistant, domain: str, integration: Integration, msg: str, exc_info: Exception = None) -> None:
    """Log helper."""

async def _async_setup_component(hass: HomeAssistant, domain: str, config: ConfigType) -> bool:
    """Set up a component for Home Assistant.

    This method is a coroutine.
    """

async def async_prepare_setup_platform(hass: HomeAssistant, hass_config: ConfigType, domain: str, platform_name: str) -> Platform | None:
    """Load a platform and makes sure dependencies are setup.

    This method is a coroutine.
    """

async def async_process_deps_reqs(hass: HomeAssistant, config: ConfigType, integration: Integration) -> None:
    """Process all dependencies and requirements for a module.

    Module is a Python module of either a component or platform.
    """

@core.callback
def async_when_setup(hass: HomeAssistant, component: str, when_setup_cb: Callable) -> None:
    """Call a method when a component is setup."""

@core.callback
def async_when_setup_or_start(hass: HomeAssistant, component: str, when_setup_cb: Callable) -> None:
    """Call a method when a component is setup or state is fired."""

@core.callback
def _async_when_setup(hass: HomeAssistant, component: str, when_setup_cb: Callable, start_event: bool) -> None:
    """Call a method when a component is setup or the start event fires."""

@core.callback
def async_get_loaded_integrations(hass: HomeAssistant) -> set[str]:
    """Return the complete list of loaded integrations."""

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
def _setup_started(hass: HomeAssistant) -> dict[tuple[str, str], float]:
    """Return the setup started dict."""

@contextlib.contextmanager
def async_pause_setup(hass: HomeAssistant, phase: SetupPhases) -> None:
    """Keep track of time we are blocked waiting for other operations.

    We want to count the time we wait for importing and
    setting up the base components so we can subtract it
    from the total setup time.
    """

@singleton.singleton(DATA_SETUP_TIME)
def _setup_times(hass: HomeAssistant) -> dict[str, dict[str, dict[str, float]]]:
    """Return the setup timings default dict."""

@contextlib.contextmanager
def async_start_setup(hass: HomeAssistant, integration: str, phase: SetupPhases, group: str = None) -> None:
    """Keep track of when setup starts and finishes.

    :param hass: Home Assistant instance
    :param integration: The integration that is being setup
    :param phase: The phase of setup
    :param group: The group (config entry/platform instance) that is being setup

      A group is a group of setups that run in parallel.

    """

@callback
def async_get_setup_timings(hass: HomeAssistant) -> dict[str, float]:
    """Return timing data for each integration."""

@callback
def async_get_domain_setup_times(hass: HomeAssistant, domain: str) -> dict[str, float]:
    """Return timing data for each integration."""

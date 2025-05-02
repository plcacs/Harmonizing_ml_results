"""Class to manage the entities for a single platform."""
from __future__ import annotations
import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from contextvars import ContextVar
from datetime import timedelta
from logging import Logger, getLogger
from typing import TYPE_CHECKING, Any, Protocol, Optional, Union, List, Dict, Tuple
from homeassistant import config_entries
from homeassistant.const import ATTR_RESTORED, DEVICE_DEFAULT_NAME, EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import CALLBACK_TYPE, DOMAIN as HOMEASSISTANT_DOMAIN, CoreState, HomeAssistant, ServiceCall, SupportsResponse, callback, split_entity_id, valid_entity_id
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryError, ConfigEntryNotReady, HomeAssistantError, PlatformNotReady
from homeassistant.generated import languages
from homeassistant.setup import SetupPhases, async_start_setup
from homeassistant.util.async_ import create_eager_task
from homeassistant.util.hass_dict import HassKey
from . import device_registry as dev_reg, entity_registry as ent_reg, service, translation
from .entity_registry import EntityRegistry, RegistryEntryDisabler, RegistryEntryHider
from .event import async_call_later
from .issue_registry import IssueSeverity, async_create_issue
from .typing import UNDEFINED, ConfigType, DiscoveryInfoType, VolDictType, VolSchemaType

if TYPE_CHECKING:
    from .entity import Entity

SLOW_SETUP_WARNING = 10
SLOW_SETUP_MAX_WAIT = 60
SLOW_ADD_ENTITY_MAX_WAIT = 15
SLOW_ADD_MIN_TIMEOUT = 500
PLATFORM_NOT_READY_RETRIES = 10
DATA_ENTITY_PLATFORM = HassKey('entity_platform')
DATA_DOMAIN_ENTITIES = HassKey('domain_entities')
DATA_DOMAIN_PLATFORM_ENTITIES = HassKey('domain_platform_entities')
PLATFORM_NOT_READY_BASE_WAIT_TIME = 30
_LOGGER = getLogger(__name__)

class AddEntitiesCallback(Protocol):
    """Protocol type for EntityPlatform.add_entities callback."""

    def __call__(self, new_entities: Iterable[Entity], update_before_add: bool = False) -> None:
        """Define add_entities type."""

class AddConfigEntryEntitiesCallback(Protocol):
    """Protocol type for EntityPlatform.add_entities callback."""

    def __call__(self, new_entities: Iterable[Entity], update_before_add: bool = False, *, config_subentry_id: Optional[str] = None) -> None:
        """Define add_entities type.

        :param config_subentry_id: subentry which the entities should be added to
        """

class EntityPlatformModule(Protocol):
    """Protocol type for entity platform modules."""

    async def async_setup_platform(self, hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: Optional[DiscoveryInfoType] = None) -> None:
        """Set up an integration platform async."""

    def setup_platform(self, hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: Optional[DiscoveryInfoType] = None) -> None:
        """Set up an integration platform."""

    async def async_setup_entry(self, hass: HomeAssistant, entry: config_entries.ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
        """Set up an integration platform from a config entry."""

class EntityPlatform:
    """Manage the entities for a single platform.

    An example of an entity platform is 'hue.light', which is managed by
    the entity component 'light'.
    """

    def __init__(self, *, hass: HomeAssistant, logger: Logger, domain: str, platform_name: str, platform: Optional[EntityPlatformModule], scan_interval: timedelta, entity_namespace: Optional[str]) -> None:
        """Initialize the entity platform."""
        self.hass = hass
        self.logger = logger
        self.domain = domain
        self.platform_name = platform_name
        self.platform = platform
        self.scan_interval = scan_interval
        self.scan_interval_seconds = scan_interval.total_seconds()
        self.entity_namespace = entity_namespace
        self.config_entry: Optional[config_entries.ConfigEntry] = None
        self.entities: Dict[str, Entity] = {}
        self.component_translations: Dict[str, str] = {}
        self.platform_translations: Dict[str, str] = {}
        self.object_id_component_translations: Dict[str, str] = {}
        self.object_id_platform_translations: Dict[str, str] = {}
        self.default_language_platform_translations: Dict[str, str] = {}
        self._tasks: List[asyncio.Task] = []
        self._setup_complete = False
        self._async_polling_timer: Optional[asyncio.TimerHandle] = None
        self._async_cancel_retry_setup: Optional[Callable[[], None]] = None
        self._process_updates: Optional[asyncio.Lock] = None
        self.parallel_updates: Optional[asyncio.Semaphore] = None
        self._update_in_sequence = False
        self.parallel_updates_created = platform is None
        self.domain_entities = hass.data.setdefault(DATA_DOMAIN_ENTITIES, {}).setdefault(domain, {})
        key = (domain, platform_name)
        self.domain_platform_entities = hass.data.setdefault(DATA_DOMAIN_PLATFORM_ENTITIES, {}).setdefault(key, {})

    def __repr__(self) -> str:
        """Represent an EntityPlatform."""
        return f'<EntityPlatform domain={self.domain} platform_name={self.platform_name} config_entry={self.config_entry}>'

    @callback
    def _get_parallel_updates_semaphore(self, entity_has_sync_update: bool) -> Optional[asyncio.Semaphore]:
        """Get or create a semaphore for parallel updates.

        Semaphore will be created on demand because we base it off if update
        method is async or not.

        - If parallel updates is set to 0, we skip the semaphore.
        - If parallel updates is set to a number, we initialize the semaphore
          to that number.

        The default value for parallel requests is decided based on the first
        entity of the platform which is added to Home Assistant. It's 1 if the
        entity implements the update method, else it's 0.
        """
        if self.parallel_updates_created:
            return self.parallel_updates
        self.parallel_updates_created = True
        parallel_updates = getattr(self.platform, 'PARALLEL_UPDATES', None)
        if parallel_updates is None and entity_has_sync_update:
            parallel_updates = 1
        if parallel_updates == 0:
            parallel_updates = None
        if parallel_updates is not None:
            self.parallel_updates = asyncio.Semaphore(parallel_updates)
            self._update_in_sequence = parallel_updates == 1
        return self.parallel_updates

    async def async_setup(self, platform_config: ConfigType, discovery_info: Optional[DiscoveryInfoType] = None) -> None:
        """Set up the platform from a config file."""
        platform = self.platform
        hass = self.hass
        if not hasattr(platform, 'async_setup_platform') and (not hasattr(platform, 'setup_platform')):
            self.logger.error('The %s platform for the %s integration does not support platform setup. Please remove it from your config.', self.platform_name, self.domain)
            learn_more_url = None
            if self.platform and 'custom_components' not in self.platform.__file__:
                learn_more_url = f'https://www.home-assistant.io/integrations/{self.platform_name}/'
            platform_key = f'platform: {self.platform_name}'
            yaml_example = f'
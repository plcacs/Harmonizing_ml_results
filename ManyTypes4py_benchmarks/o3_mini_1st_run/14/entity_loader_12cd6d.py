from __future__ import annotations
import asyncio
from collections.abc import Callable, Coroutine, Sequence
from datetime import timedelta
from functools import partial
from typing import Any, Set, Tuple
from aiounifi.interfaces.api_handlers import ItemEvent
from homeassistant.const import Platform
from homeassistant.core import callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from ..const import LOGGER, UNIFI_WIRELESS_CLIENTS
from ..entity import UnifiEntity, UnifiEntityDescription
if TYPE_CHECKING:
    from .hub import UnifiHub

CHECK_HEARTBEAT_INTERVAL = timedelta(seconds=1)
POLL_INTERVAL = timedelta(seconds=10)

class UnifiEntityLoader:
    """UniFi Network integration handling platforms for entity registration."""

    def __init__(self, hub: UnifiHub) -> None:
        """Initialize the UniFi entity loader."""
        self.hub: UnifiHub = hub
        self.api_updaters: Tuple[Callable[[], Coroutine[Any, Any, Any]], ...] = (
            hub.api.clients.update,
            hub.api.clients_all.update,
            hub.api.devices.update,
            hub.api.dpi_apps.update,
            hub.api.dpi_groups.update,
            hub.api.port_forwarding.update,
            hub.api.sites.update,
            hub.api.system_information.update,
            hub.api.traffic_rules.update,
            hub.api.traffic_routes.update,
            hub.api.wlans.update,
        )
        self.polling_api_updaters: Tuple[Callable[[], Coroutine[Any, Any, Any]], ...] = (
            hub.api.traffic_rules.update,
            hub.api.traffic_routes.update,
        )
        self.wireless_clients = hub.hass.data[UNIFI_WIRELESS_CLIENTS]
        self._dataUpdateCoordinator: DataUpdateCoordinator[Any] = DataUpdateCoordinator(
            hub.hass, LOGGER, name='Unifi entity poller', update_method=self._update_pollable_api_data, update_interval=POLL_INTERVAL
        )
        self._update_listener = self._dataUpdateCoordinator.async_add_listener(update_callback=lambda: None)
        self.platforms: list[Tuple[AddEntitiesCallback, type[UnifiEntity], Sequence[UnifiEntityDescription], bool]] = []
        self.known_objects: Set[Tuple[str, str]] = set()
        # Tuples of entity description key and object ID of loaded entities.

    async def initialize(self) -> None:
        """Initialize API data and extra client support."""
        await self._refresh_api_data()
        self._restore_inactive_clients()
        self.wireless_clients.update_clients(set(self.hub.api.clients.values()))

    async def _refresh_data(self, updaters: Sequence[Callable[[], Coroutine[Any, Any, Any]]]) -> None:
        results = await asyncio.gather(*[update() for update in updaters], return_exceptions=True)
        for result in results:
            if result is not None:
                LOGGER.warning('Exception on update %s', result)

    async def _update_pollable_api_data(self) -> None:
        """Refresh API data for pollable updaters."""
        await self._refresh_data(self.polling_api_updaters)

    async def _refresh_api_data(self) -> None:
        """Refresh API data from network application."""
        await self._refresh_data(self.api_updaters)

    @callback
    def _restore_inactive_clients(self) -> None:
        """Restore inactive clients.

        Provide inactive clients to device tracker and switch platform.
        """
        config = self.hub.config
        entity_registry = er.async_get(self.hub.hass)
        macs = [
            entry.unique_id.split('-', 1)[1]
            for entry in er.async_entries_for_config_entry(entity_registry, config.entry.entry_id)
            if entry.domain == Platform.DEVICE_TRACKER and '-' in entry.unique_id
        ]
        api = self.hub.api
        for mac in config.option_supported_clients + config.option_block_clients + macs:
            if mac not in api.clients and mac in api.clients_all:
                api.clients.process_raw([dict(api.clients_all[mac].raw)])

    @callback
    def register_platform(
        self,
        async_add_entities: AddEntitiesCallback,
        entity_class: type[UnifiEntity],
        descriptions: Sequence[UnifiEntityDescription],
        requires_admin: bool = False,
    ) -> None:
        """Register UniFi entity platforms."""
        self.platforms.append((async_add_entities, entity_class, descriptions, requires_admin))

    @callback
    def load_entities(self) -> None:
        """Load entities into the registered UniFi platforms."""
        for async_add_entities, entity_class, descriptions, requires_admin in self.platforms:
            if requires_admin and (not self.hub.is_admin):
                continue
            self._load_entities(entity_class, descriptions, async_add_entities)

    @callback
    def _should_add_entity(self, description: UnifiEntityDescription, obj_id: str) -> bool:
        """Validate if entity is allowed and supported before creating it."""
        return bool(
            (description.key, obj_id) not in self.known_objects
            and description.allowed_fn(self.hub, obj_id)
            and description.supported_fn(self.hub, obj_id)
        )

    @callback
    def _load_entities(
        self,
        unifi_platform_entity: type[UnifiEntity],
        descriptions: Sequence[UnifiEntityDescription],
        async_add_entities: AddEntitiesCallback
    ) -> None:
        """Load entities and subscribe for future entities."""

        @callback
        def add_unifi_entities() -> None:
            """Add currently known UniFi entities."""
            async_add_entities(
                (
                    unifi_platform_entity(obj_id, self.hub, description)
                    for description in descriptions
                    for obj_id in description.api_handler_fn(self.hub.api)
                    if self._should_add_entity(description, obj_id)
                )
            )
        add_unifi_entities()
        self.hub.config.entry.async_on_unload(
            async_dispatcher_connect(self.hub.hass, self.hub.signal_options_update, add_unifi_entities)
        )

        @callback
        def create_unifi_entity(
            description: UnifiEntityDescription, event: ItemEvent, obj_id: str
        ) -> None:
            """Create new UniFi entity on event."""
            if self._should_add_entity(description, obj_id):
                async_add_entities([unifi_platform_entity(obj_id, self.hub, description)])

        for description in descriptions:
            description.api_handler_fn(self.hub.api).subscribe(partial(create_unifi_entity, description), ItemEvent.ADDED)
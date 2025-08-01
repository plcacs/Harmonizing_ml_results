"""Support for Ruckus devices."""
from __future__ import annotations
import logging
from typing import Any, List, Set, Optional, Dict
from homeassistant.components.device_tracker import ScannerEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import API_CLIENT_HOSTNAME, API_CLIENT_IP, COORDINATOR, DOMAIN, KEY_SYS_CLIENTS, UNDO_UPDATE_LISTENERS
from .coordinator import RuckusDataUpdateCoordinator

_LOGGER = logging.getLogger(__package__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up device tracker for Ruckus component."""
    coordinator: RuckusDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id][COORDINATOR]
    tracked: Set[str] = set()

    @callback
    def router_update() -> None:
        """Update the values of the router."""
        add_new_entities(coordinator, async_add_entities, tracked)

    router_update()
    hass.data[DOMAIN][entry.entry_id][UNDO_UPDATE_LISTENERS].append(coordinator.async_add_listener(router_update))
    registry: er.EntityRegistry = er.async_get(hass)
    restore_entities(registry, coordinator, entry, async_add_entities, tracked)


@callback
def add_new_entities(
    coordinator: RuckusDataUpdateCoordinator,
    async_add_entities: AddConfigEntryEntitiesCallback,
    tracked: Set[str],
) -> None:
    """Add new tracker entities from the router."""
    new_tracked: List[RuckusDevice] = []
    for mac in coordinator.data[KEY_SYS_CLIENTS]:
        if mac in tracked:
            continue
        device: Dict[str, Any] = coordinator.data[KEY_SYS_CLIENTS][mac]
        _LOGGER.debug("adding new device: [%s] %s", mac, device[API_CLIENT_HOSTNAME])
        new_tracked.append(RuckusDevice(coordinator, mac, device[API_CLIENT_HOSTNAME]))
        tracked.add(mac)
    async_add_entities(new_tracked)


@callback
def restore_entities(
    registry: er.EntityRegistry,
    coordinator: RuckusDataUpdateCoordinator,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
    tracked: Set[str],
) -> None:
    """Restore clients that are not a part of active clients list."""
    missing: List[RuckusDevice] = []
    for entity in registry.entities.get_entries_for_config_entry_id(entry.entry_id).values():
        if entity.platform == DOMAIN and entity.unique_id not in coordinator.data[KEY_SYS_CLIENTS]:
            missing.append(RuckusDevice(coordinator, entity.unique_id, entity.original_name))
            tracked.add(entity.unique_id)
    _LOGGER.debug("added %d missing devices", len(missing))
    async_add_entities(missing)


class RuckusDevice(CoordinatorEntity, ScannerEntity):
    """Representation of a Ruckus client."""

    def __init__(self, coordinator: RuckusDataUpdateCoordinator, mac: str, name: str) -> None:
        """Initialize a Ruckus client."""
        super().__init__(coordinator)
        self._mac: str = mac
        self._name: str = name

    @property
    def mac_address(self) -> str:
        """Return a mac address."""
        return self._mac

    @property
    def name(self) -> str:
        """Return the name."""
        if not self.is_connected:
            return self._name
        return self.coordinator.data[KEY_SYS_CLIENTS][self._mac][API_CLIENT_HOSTNAME]

    @property
    def ip_address(self) -> Optional[str]:
        """Return the ip address."""
        if not self.is_connected:
            return None
        return self.coordinator.data[KEY_SYS_CLIENTS][self._mac][API_CLIENT_IP]

    @property
    def is_connected(self) -> bool:
        """Return true if the device is connected to the network."""
        return self._mac in self.coordinator.data[KEY_SYS_CLIENTS]
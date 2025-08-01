"""Support for the Dynalite devices as entities."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Optional
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity
from .bridge import DynaliteBridge, DynaliteConfigEntry
from .const import DOMAIN, LOGGER

def async_setup_entry_base(
    hass: HomeAssistant,
    config_entry: DynaliteConfigEntry,
    async_add_entities: AddEntitiesCallback,
    platform: str,
    entity_from_device: Callable[[Any, DynaliteBridge], DynaliteBase],
) -> None:
    """Record the async_add_entities function to add them later when received from Dynalite."""
    LOGGER.debug("Setting up %s entry = %s", platform, config_entry.data)
    bridge: DynaliteBridge = config_entry.runtime_data

    @callback
    def async_add_entities_platform(devices: list[Any]) -> None:
        added_entities = [entity_from_device(device, bridge) for device in devices]
        async_add_entities(added_entities)

    bridge.register_add_devices(platform, async_add_entities_platform)

class DynaliteBase(RestoreEntity, ABC):
    """Base class for the Dynalite entities."""
    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, device: Any, bridge: DynaliteBridge) -> None:
        """Initialize the base class."""
        self._device: Any = device
        self._bridge: DynaliteBridge = bridge
        self._unsub_dispatchers: list[Callable[[], None]] = []

    @property
    def unique_id(self) -> str:
        """Return the unique ID of the entity."""
        return self._device.unique_id

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return self._device.available

    @property
    def device_info(self) -> DeviceInfo:
        """Device info for this entity."""
        return DeviceInfo(
            identifiers={(DOMAIN, self._device.unique_id)},
            manufacturer="Dynalite",
            name=self._device.name,
        )

    async def async_added_to_hass(self) -> None:
        """Handle addition to hass: restore state and register to dispatch."""
        await super().async_added_to_hass()
        cur_state: Optional[Any] = await self.async_get_last_state()
        if cur_state:
            self.initialize_state(cur_state)
        else:
            LOGGER.warning("Restore state not available for %s", self.entity_id)
        self._unsub_dispatchers.append(
            async_dispatcher_connect(
                self.hass,
                self._bridge.update_signal(self._device),
                self.async_schedule_update_ha_state,
            )
        )
        self._unsub_dispatchers.append(
            async_dispatcher_connect(
                self.hass,
                self._bridge.update_signal(),
                self.async_schedule_update_ha_state,
            )
        )

    @abstractmethod
    def initialize_state(self, state: Any) -> None:
        """Initialize the state from cache."""

    async def async_will_remove_from_hass(self) -> None:
        """Unregister signal dispatch listeners when being removed."""
        for unsub in self._unsub_dispatchers:
            unsub()
        self._unsub_dispatchers = []
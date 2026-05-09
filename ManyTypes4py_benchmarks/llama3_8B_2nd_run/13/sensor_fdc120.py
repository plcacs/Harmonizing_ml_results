from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from pylaunches.types import Event, Launch
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME, PERCENTAGE
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity, DataUpdateCoordinator
from homeassistant.util.dt import parse_datetime
from . import LaunchLibraryData
from .const import DOMAIN

@dataclass(frozen=True, kw_only=True)
class LaunchLibrarySensorEntityDescription(SensorEntityDescription):
    """Describes a Next Launch sensor entity."""
    key: str
    icon: str
    translation_key: str
    value_fn: Callable[[Any], Optional[str]]
    attributes_fn: Callable[[Any], Dict[str, Any]]

SENSOR_DESCRIPTIONS: List[LaunchLibrarySensorEntityDescription] = ...

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the sensor platform."""
    name: str = entry.data.get(CONF_NAME, DEFAULT_NEXT_LAUNCH_NAME)
    coordinator: DataUpdateCoordinator[LaunchLibraryData] = hass.data[DOMAIN]
    async_add_entities((LaunchLibrarySensor(coordinator=coordinator, entry_id=entry.entry_id, description=description, name=name) for description in SENSOR_DESCRIPTIONS))

class LaunchLibrarySensor(CoordinatorEntity[DataUpdateCoordinator[LaunchLibraryData]], SensorEntity):
    """Representation of the next launch sensors."""
    _attr_attribution: str
    _attr_has_entity_name: bool
    _next_event: Optional[Event]
    _attr_device_info: DeviceInfo

    def __init__(self, coordinator: DataUpdateCoordinator[LaunchLibraryData], entry_id: str, description: LaunchLibrarySensorEntityDescription, name: str) -> None:
        """Initialize a Launch Library sensor."""
        super().__init__(coordinator)
        self._attr_unique_id: str = f'{entry_id}_{description.key}'
        self.entity_description: LaunchLibrarySensorEntityDescription = description
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, entry_id)}, entry_type=DeviceEntryType.SERVICE, name=name)

    @property
    def native_value(self) -> Optional[str]:
        """Return the state of the sensor."""
        if self._next_event is None:
            return None
        return self.entity_description.value_fn(self._next_event)

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        """Return the attributes of the sensor."""
        if self._next_event is None:
            return None
        return self.entity_description.attributes_fn(self._next_event)

    @property
    def available(self) -> bool:
        """Return if the sensor is available."""
        return super().available and self._next_event is not None

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        if self.entity_description.key == 'starship_launch':
            events: List[Event] = self.coordinator.data['starship_events']['upcoming']['launches']
        elif self.entity_description.key == 'starship_event':
            events: List[Event] = self.coordinator.data['starship_events']['upcoming']['events']
        else:
            events: List[Launch] = self.coordinator.data['upcoming_launches']
        self._next_event = next((event for event in events), None)
        super()._handle_coordinator_update()

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        self._handle_coordinator_update()

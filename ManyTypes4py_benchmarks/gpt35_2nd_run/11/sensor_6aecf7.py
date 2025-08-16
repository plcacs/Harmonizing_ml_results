from __future__ import annotations
from collections.abc import Mapping
from datetime import timedelta
from typing import Any, List, Optional, Tuple
from homeassistant.components.sensor import RestoreSensor, SensorDeviceClass, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ATTRIBUTION, ATTR_LATITUDE, ATTR_LONGITUDE, CONF_MODE, CONF_NAME, UnitOfLength, UnitOfTime
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import ATTR_DESTINATION, ATTR_DESTINATION_NAME, ATTR_DISTANCE, ATTR_DURATION, ATTR_DURATION_IN_TRAFFIC, ATTR_ORIGIN, ATTR_ORIGIN_NAME, DOMAIN, ICON_CAR, ICONS
from .coordinator import HERERoutingDataUpdateCoordinator, HERETransitDataUpdateCoordinator

def sensor_descriptions(travel_mode: str) -> Tuple[SensorEntityDescription, SensorEntityDescription, SensorEntityDescription]:
    ...

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class HERETravelTimeSensor(CoordinatorEntity[HERERoutingDataUpdateCoordinator | HERETransitDataUpdateCoordinator], RestoreSensor):
    ...

    def __init__(self, unique_id_prefix: str, name: str, sensor_description: SensorEntityDescription, coordinator: HERERoutingDataUpdateCoordinator | HERETransitDataUpdateCoordinator) -> None:
        ...

    async def _async_restore_state(self) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...

    @property
    def attribution(self) -> Optional[str]:
        ...

class OriginSensor(HERETravelTimeSensor):
    ...

    def __init__(self, unique_id_prefix: str, name: str, coordinator: HERERoutingDataUpdateCoordinator | HERETransitDataUpdateCoordinator) -> None:
        ...

    @property
    def extra_state_attributes(self) -> Optional[Mapping[str, Any]]:
        ...

class DestinationSensor(HERETravelTimeSensor):
    ...

    def __init__(self, unique_id_prefix: str, name: str, coordinator: HERERoutingDataUpdateCoordinator | HERETransitDataUpdateCoordinator) -> None:
        ...

    @property
    def extra_state_attributes(self) -> Optional[Mapping[str, Any]]:
        ...

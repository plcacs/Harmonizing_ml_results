"""Support for HERE travel time sensors."""
from __future__ import annotations
from collections.abc import Mapping
from datetime import timedelta
from typing import Any, Optional, Tuple
from homeassistant.components.sensor import (
    RestoreSensor,
    SensorDeviceClass,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_ATTRIBUTION,
    ATTR_LATITUDE,
    ATTR_LONGITUDE,
    CONF_MODE,
    CONF_NAME,
    UnitOfLength,
    UnitOfTime,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import (
    ATTR_DESTINATION,
    ATTR_DESTINATION_NAME,
    ATTR_DISTANCE,
    ATTR_DURATION,
    ATTR_DURATION_IN_TRAFFIC,
    ATTR_ORIGIN,
    ATTR_ORIGIN_NAME,
    DOMAIN,
    ICON_CAR,
    ICONS,
)
from .coordinator import HERERoutingDataUpdateCoordinator, HERETransitDataUpdateCoordinator

SCAN_INTERVAL: timedelta = timedelta(minutes=5)

def sensor_descriptions(travel_mode: str) -> Tuple[SensorEntityDescription, SensorEntityDescription, SensorEntityDescription]:
    """Construct SensorEntityDescriptions."""
    return (
        SensorEntityDescription(
            translation_key='duration',
            icon=ICONS.get(travel_mode, ICON_CAR),
            key=ATTR_DURATION,
            state_class=SensorStateClass.MEASUREMENT,
            native_unit_of_measurement=UnitOfTime.MINUTES,
        ),
        SensorEntityDescription(
            translation_key='duration_in_traffic',
            icon=ICONS.get(travel_mode, ICON_CAR),
            key=ATTR_DURATION_IN_TRAFFIC,
            state_class=SensorStateClass.MEASUREMENT,
            native_unit_of_measurement=UnitOfTime.MINUTES,
        ),
        SensorEntityDescription(
            translation_key='distance',
            icon=ICONS.get(travel_mode, ICON_CAR),
            key=ATTR_DISTANCE,
            state_class=SensorStateClass.MEASUREMENT,
            device_class=SensorDeviceClass.DISTANCE,
            native_unit_of_measurement=UnitOfLength.KILOMETERS,
        ),
    )

async def async_setup_entry(
    hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Add HERE travel time entities from a config_entry."""
    entry_id: str = config_entry.entry_id
    name: str = config_entry.data[CONF_NAME]
    coordinator: HERERoutingDataUpdateCoordinator | HERETransitDataUpdateCoordinator = hass.data[DOMAIN][entry_id]

    sensors = [
        HERETravelTimeSensor(entry_id, name, sensor_description, coordinator)
        for sensor_description in sensor_descriptions(config_entry.data[CONF_MODE])
    ]
    sensors.append(OriginSensor(entry_id, name, coordinator))
    sensors.append(DestinationSensor(entry_id, name, coordinator))
    async_add_entities(sensors)

class HERETravelTimeSensor(CoordinatorEntity[HERERoutingDataUpdateCoordinator | HERETransitDataUpdateCoordinator], RestoreSensor):
    """Representation of a HERE travel time sensor."""
    _attr_has_entity_name: bool = True

    def __init__(
        self,
        unique_id_prefix: str,
        name: str,
        sensor_description: SensorEntityDescription,
        coordinator: HERERoutingDataUpdateCoordinator | HERETransitDataUpdateCoordinator,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self.entity_description: SensorEntityDescription = sensor_description
        self._attr_unique_id: str = f"{unique_id_prefix}_{sensor_description.key}"
        self._attr_device_info: DeviceInfo = DeviceInfo(
            identifiers={(DOMAIN, unique_id_prefix)},
            entry_type=DeviceEntryType.SERVICE,
            name=name,
            manufacturer="HERE Technologies",
        )

    async def _async_restore_state(self) -> None:
        """Restore state."""
        restored_data = await self.async_get_last_sensor_data()
        if restored_data:
            self._attr_native_value = restored_data.native_value

    async def async_added_to_hass(self) -> None:
        """Wait for start so origin and destination entities can be resolved."""
        await self._async_restore_state()
        await super().async_added_to_hass()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        if self.coordinator.data is not None:
            self._attr_native_value = self.coordinator.data.get(self.entity_description.key)
            self.async_write_ha_state()

    @property
    def attribution(self) -> Optional[str]:
        """Return the attribution."""
        if self.coordinator.data is not None:
            attribution_value = self.coordinator.data.get(ATTR_ATTRIBUTION)
            if attribution_value is not None:
                return str(attribution_value)
        return None

class OriginSensor(HERETravelTimeSensor):
    """Sensor holding information about the route origin."""

    def __init__(
        self,
        unique_id_prefix: str,
        name: str,
        coordinator: HERERoutingDataUpdateCoordinator | HERETransitDataUpdateCoordinator,
    ) -> None:
        """Initialize the sensor."""
        sensor_description = SensorEntityDescription(
            translation_key='origin',
            icon='mdi:store-marker',
            key=ATTR_ORIGIN_NAME,
        )
        super().__init__(unique_id_prefix, name, sensor_description, coordinator)

    @property
    def extra_state_attributes(self) -> Optional[Mapping[str, Any]]:
        """GPS coordinates."""
        if self.coordinator.data is not None:
            origin: str = self.coordinator.data[ATTR_ORIGIN]
            parts = origin.split(',')
            return {ATTR_LATITUDE: parts[0], ATTR_LONGITUDE: parts[1]}
        return None

class DestinationSensor(HERETravelTimeSensor):
    """Sensor holding information about the route destination."""

    def __init__(
        self,
        unique_id_prefix: str,
        name: str,
        coordinator: HERERoutingDataUpdateCoordinator | HERETransitDataUpdateCoordinator,
    ) -> None:
        """Initialize the sensor."""
        sensor_description = SensorEntityDescription(
            translation_key='destination',
            icon='mdi:store-marker',
            key=ATTR_DESTINATION_NAME,
        )
        super().__init__(unique_id_prefix, name, sensor_description, coordinator)

    @property
    def extra_state_attributes(self) -> Optional[Mapping[str, Any]]:
        """GPS coordinates."""
        if self.coordinator.data is not None:
            destination: str = self.coordinator.data[ATTR_DESTINATION]
            parts = destination.split(',')
            return {ATTR_LATITUDE: parts[0], ATTR_LONGITUDE: parts[1]}
        return None

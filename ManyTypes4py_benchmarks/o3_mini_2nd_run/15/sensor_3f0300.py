"""Support for the Airzone sensors."""
from __future__ import annotations

from typing import Any, Callable, Dict, Set, List
from aioairzone.const import AZD_HOT_WATER, AZD_HUMIDITY, AZD_TEMP, AZD_TEMP_UNIT, AZD_WEBSERVER, AZD_WIFI_RSSI, AZD_ZONES
from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, SIGNAL_STRENGTH_DECIBELS_MILLIWATT, EntityCategory, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import TEMP_UNIT_LIB_TO_HASS
from .coordinator import AirzoneUpdateCoordinator
from .entity import AirzoneEntity, AirzoneHotWaterEntity, AirzoneWebServerEntity, AirzoneZoneEntity

HOT_WATER_SENSOR_TYPES: tuple[SensorEntityDescription, ...] = (
    SensorEntityDescription(
        device_class=SensorDeviceClass.TEMPERATURE,
        key=AZD_TEMP,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
    ),
)
WEBSERVER_SENSOR_TYPES: tuple[SensorEntityDescription, ...] = (
    SensorEntityDescription(
        device_class=SensorDeviceClass.SIGNAL_STRENGTH,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
        key=AZD_WIFI_RSSI,
        translation_key="rssi",
        native_unit_of_measurement=SIGNAL_STRENGTH_DECIBELS_MILLIWATT,
        state_class=SensorStateClass.MEASUREMENT,
    ),
)
ZONE_SENSOR_TYPES: tuple[SensorEntityDescription, ...] = (
    SensorEntityDescription(
        device_class=SensorDeviceClass.TEMPERATURE,
        key=AZD_TEMP,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        device_class=SensorDeviceClass.HUMIDITY,
        key=AZD_HUMIDITY,
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Add Airzone sensors from a config_entry."""
    coordinator: AirzoneUpdateCoordinator = entry.runtime_data
    added_zones: Set[str] = set()

    def _async_entity_listener() -> None:
        """Handle additions of sensors."""
        entities: List[SensorEntity] = []
        zones_data: Dict[str, Any] = coordinator.data.get(AZD_ZONES, {})
        received_zones: Set[str] = set(zones_data)
        new_zones: Set[str] = received_zones - added_zones
        if new_zones:
            for system_zone_id in new_zones:
                zone_info = zones_data.get(system_zone_id)
                if not isinstance(zone_info, dict):
                    continue
                for description in ZONE_SENSOR_TYPES:
                    if description.key in zone_info:
                        entities.append(
                            AirzoneZoneSensor(coordinator, description, entry, system_zone_id, zone_info)
                        )
            added_zones.update(new_zones)
        async_add_entities(entities)

    entities: List[SensorEntity] = []
    if AZD_HOT_WATER in coordinator.data:
        for description in HOT_WATER_SENSOR_TYPES:
            if description.key in coordinator.data[AZD_HOT_WATER]:
                entities.append(AirzoneHotWaterSensor(coordinator, description, entry))
    if AZD_WEBSERVER in coordinator.data:
        for description in WEBSERVER_SENSOR_TYPES:
            if description.key in coordinator.data[AZD_WEBSERVER]:
                entities.append(AirzoneWebServerSensor(coordinator, description, entry))
    async_add_entities(entities)
    entry.async_on_unload(coordinator.async_add_listener(_async_entity_listener))
    _async_entity_listener()


class AirzoneSensor(AirzoneEntity, SensorEntity):
    """Define an Airzone sensor."""

    @callback
    def _handle_coordinator_update(self) -> None:
        """Update attributes when the coordinator updates."""
        self._async_update_attrs()
        super()._handle_coordinator_update()

    @callback
    def _async_update_attrs(self) -> None:
        """Update sensor attributes."""
        self._attr_native_value = self.get_airzone_value(self.entity_description.key)


class AirzoneHotWaterSensor(AirzoneHotWaterEntity, AirzoneSensor):
    """Define an Airzone Hot Water sensor."""

    def __init__(
        self,
        coordinator: AirzoneUpdateCoordinator,
        description: SensorEntityDescription,
        entry: ConfigEntry,
    ) -> None:
        """Initialize."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{self._attr_unique_id}_dhw_{description.key}"
        self.entity_description = description
        self._attr_native_unit_of_measurement = TEMP_UNIT_LIB_TO_HASS.get(
            self.get_airzone_value(AZD_TEMP_UNIT)
        )
        self._async_update_attrs()


class AirzoneWebServerSensor(AirzoneWebServerEntity, AirzoneSensor):
    """Define an Airzone WebServer sensor."""

    def __init__(
        self,
        coordinator: AirzoneUpdateCoordinator,
        description: SensorEntityDescription,
        entry: ConfigEntry,
    ) -> None:
        """Initialize."""
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{self._attr_unique_id}_ws_{description.key}"
        self.entity_description = description
        self._async_update_attrs()


class AirzoneZoneSensor(AirzoneZoneEntity, AirzoneSensor):
    """Define an Airzone Zone sensor."""

    def __init__(
        self,
        coordinator: AirzoneUpdateCoordinator,
        description: SensorEntityDescription,
        entry: ConfigEntry,
        system_zone_id: str,
        zone_data: Dict[str, Any],
    ) -> None:
        """Initialize."""
        super().__init__(coordinator, entry, system_zone_id, zone_data)
        self._attr_unique_id = f"{self._attr_unique_id}_{system_zone_id}_{description.key}"
        self.entity_description = description
        if description.key == AZD_TEMP:
            self._attr_native_unit_of_measurement = TEMP_UNIT_LIB_TO_HASS.get(
                self.get_airzone_value(AZD_TEMP_UNIT)
            )
        self._async_update_attrs()
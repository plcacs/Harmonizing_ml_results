"""Support for monitoring a Sense energy sensor."""
from __future__ import annotations
from datetime import datetime
from typing import Any, List, Optional
from sense_energy import ASyncSenseable, Scale
from sense_energy.sense_api import SenseDevice
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.const import PERCENTAGE, UnitOfElectricPotential, UnitOfEnergy, UnitOfPower
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import SenseConfigEntry
from .const import (
    ACTIVE_TYPE,
    CONSUMPTION_ID,
    CONSUMPTION_NAME,
    FROM_GRID_ID,
    FROM_GRID_NAME,
    NET_PRODUCTION_ID,
    NET_PRODUCTION_NAME,
    PRODUCTION_ID,
    PRODUCTION_NAME,
    PRODUCTION_PCT_ID,
    PRODUCTION_PCT_NAME,
    SOLAR_POWERED_ID,
    SOLAR_POWERED_NAME,
    TO_GRID_ID,
    TO_GRID_NAME,
)
from .coordinator import SenseRealtimeCoordinator, SenseTrendCoordinator
from .entity import SenseDeviceEntity, SenseEntity

TRENDS_SENSOR_TYPES: dict[Scale, str] = {
    Scale.DAY: "Daily",
    Scale.WEEK: "Weekly",
    Scale.MONTH: "Monthly",
    Scale.YEAR: "Yearly",
    Scale.CYCLE: "Bill",
}
SENSOR_VARIANTS: list[tuple[str, str]] = [(PRODUCTION_ID, PRODUCTION_NAME), (CONSUMPTION_ID, CONSUMPTION_NAME)]
TREND_SENSOR_VARIANTS: list[tuple[str, str]] = [
    *SENSOR_VARIANTS,
    (PRODUCTION_PCT_ID, PRODUCTION_PCT_NAME),
    (NET_PRODUCTION_ID, NET_PRODUCTION_NAME),
    (FROM_GRID_ID, FROM_GRID_NAME),
    (TO_GRID_ID, TO_GRID_NAME),
    (SOLAR_POWERED_ID, SOLAR_POWERED_NAME),
]


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: SenseConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Sense sensor."""
    data: ASyncSenseable = config_entry.runtime_data.data
    trends_coordinator: SenseTrendCoordinator = config_entry.runtime_data.trends
    realtime_coordinator: SenseRealtimeCoordinator = config_entry.runtime_data.rt
    await trends_coordinator.async_request_refresh()
    sense_monitor_id: str = data.sense_monitor_id
    entities: List[SensorEntity] = []
    for device in config_entry.runtime_data.data.devices:
        entities.append(SenseDevicePowerSensor(device, sense_monitor_id, realtime_coordinator))
        entities.extend(
            (
                SenseDeviceEnergySensor(device, scale, realtime_coordinator, sense_monitor_id)
                for scale in Scale
            )
        )
    for variant_id, variant_name in SENSOR_VARIANTS:
        entities.append(
            SensePowerSensor(data, sense_monitor_id, variant_id, variant_name, realtime_coordinator)
        )
    entities.extend(
        (
            SenseVoltageSensor(data, i, sense_monitor_id, realtime_coordinator)
            for i in range(len(data.active_voltage))
        )
    )
    for scale in Scale:
        for variant_id, variant_name in TREND_SENSOR_VARIANTS:
            entities.append(
                SenseTrendsSensor(data, scale, variant_id, variant_name, trends_coordinator, sense_monitor_id)
            )
    async_add_entities(entities)


class SensePowerSensor(SenseEntity, SensorEntity):
    """Implementation of a Sense energy sensor."""
    _attr_device_class = SensorDeviceClass.POWER
    _attr_native_unit_of_measurement = UnitOfPower.WATT
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(
        self,
        gateway: ASyncSenseable,
        sense_monitor_id: str,
        variant_id: str,
        variant_name: str,
        realtime_coordinator: SenseRealtimeCoordinator,
    ) -> None:
        """Initialize the Sense sensor."""
        super().__init__(gateway, realtime_coordinator, sense_monitor_id, f"{ACTIVE_TYPE}-{variant_id}")
        self._attr_name = variant_name
        self._variant_id = variant_id

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        value: float = (
            self._gateway.active_solar_power
            if self._variant_id == PRODUCTION_ID
            else self._gateway.active_power
        )
        return round(value)


class SenseVoltageSensor(SenseEntity, SensorEntity):
    """Implementation of a Sense energy voltage sensor."""
    _attr_device_class = SensorDeviceClass.VOLTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfElectricPotential.VOLT

    def __init__(
        self,
        gateway: ASyncSenseable,
        index: int,
        sense_monitor_id: str,
        realtime_coordinator: SenseRealtimeCoordinator,
    ) -> None:
        """Initialize the Sense sensor."""
        super().__init__(gateway, realtime_coordinator, sense_monitor_id, f"L{index + 1}")
        self._attr_name = f"L{index + 1} Voltage"
        self._voltage_index: int = index

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        return round(self._gateway.active_voltage[self._voltage_index], 1)


class SenseTrendsSensor(SenseEntity, SensorEntity):
    """Implementation of a Sense energy sensor."""

    def __init__(
        self,
        gateway: ASyncSenseable,
        scale: Scale,
        variant_id: str,
        variant_name: str,
        trends_coordinator: SenseTrendCoordinator,
        sense_monitor_id: str,
    ) -> None:
        """Initialize the Sense sensor."""
        super().__init__(gateway, trends_coordinator, sense_monitor_id, f"{TRENDS_SENSOR_TYPES[scale].lower()}-{variant_id}")
        self._attr_name = f"{TRENDS_SENSOR_TYPES[scale]} {variant_name}"
        self._scale: Scale = scale
        self._variant_id: str = variant_id
        self._had_any_update: bool = False
        if variant_id in [PRODUCTION_PCT_ID, SOLAR_POWERED_ID]:
            self._attr_native_unit_of_measurement = PERCENTAGE
            self._attr_entity_registry_enabled_default = False
            self._attr_state_class = None
            self._attr_device_class = None
        else:
            self._attr_device_class = SensorDeviceClass.ENERGY
            self._attr_state_class = SensorStateClass.TOTAL
            self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        return round(self._gateway.get_stat(self._scale, self._variant_id), 1)

    @property
    def last_reset(self) -> Optional[datetime]:
        """Return the time when the sensor was last reset, if any."""
        if self._attr_state_class == SensorStateClass.TOTAL:
            return self._gateway.trend_start(self._scale)
        return None


class SenseDevicePowerSensor(SenseDeviceEntity, SensorEntity):
    """Implementation of a Sense energy device."""
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfPower.WATT
    _attr_device_class = SensorDeviceClass.POWER

    def __init__(
        self,
        device: SenseDevice,
        sense_monitor_id: str,
        coordinator: SenseRealtimeCoordinator,
    ) -> None:
        """Initialize the Sense device sensor."""
        super().__init__(device, coordinator, sense_monitor_id, f"{device.id}-{CONSUMPTION_ID}")

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        return self._device.power_w


class SenseDeviceEnergySensor(SenseDeviceEntity, SensorEntity):
    """Implementation of a Sense device energy sensor."""
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_device_class = SensorDeviceClass.ENERGY

    def __init__(
        self,
        device: SenseDevice,
        scale: Scale,
        coordinator: SenseRealtimeCoordinator,
        sense_monitor_id: str,
    ) -> None:
        """Initialize the Sense device sensor."""
        super().__init__(device, coordinator, sense_monitor_id, f"{device.id}-{TRENDS_SENSOR_TYPES[scale].lower()}-energy")
        self._attr_translation_key = f"{TRENDS_SENSOR_TYPES[scale].lower()}_energy"
        self._attr_suggested_display_precision = 2
        self._scale: Scale = scale
        self._device: SenseDevice = device

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        return self._device.energy_kwh[self._scale]
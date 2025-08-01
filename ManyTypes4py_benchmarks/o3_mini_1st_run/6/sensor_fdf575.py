"""Sensor platform for Sensibo integration."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Union
from pysensibo.model import MotionSensor, PureAQI, SensiboDevice
from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import (
    CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
    CONCENTRATION_PARTS_PER_BILLION,
    CONCENTRATION_PARTS_PER_MILLION,
    PERCENTAGE,
    SIGNAL_STRENGTH_DECIBELS_MILLIWATT,
    EntityCategory,
    UnitOfElectricPotential,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from . import SensiboConfigEntry
from .coordinator import SensiboDataUpdateCoordinator
from .entity import SensiboDeviceBaseEntity, SensiboMotionBaseEntity

PARALLEL_UPDATES: int = 0

def _smart_type_name(_type: Optional[str]) -> str:
    """Return a lowercase name of smart type."""
    if _type and _type == 'feelsLike':
        return 'feelslike'
    return _type  # type: ignore

@dataclass(frozen=True, kw_only=True)
class SensiboMotionSensorEntityDescription(SensorEntityDescription):
    """Describes Sensibo Motion sensor entity."""

@dataclass(frozen=True, kw_only=True)
class SensiboDeviceSensorEntityDescription(SensorEntityDescription):
    """Describes Sensibo Device sensor entity."""

FILTER_LAST_RESET_DESCRIPTION: SensiboDeviceSensorEntityDescription = SensiboDeviceSensorEntityDescription(
    key='filter_last_reset',
    translation_key='filter_last_reset',
    device_class=SensorDeviceClass.TIMESTAMP,
    value_fn=lambda data: data.filter_last_reset,
    extra_fn=None,
)

MOTION_SENSOR_TYPES: tuple[SensiboMotionSensorEntityDescription, ...] = (
    SensiboMotionSensorEntityDescription(
        key='rssi',
        translation_key='rssi',
        device_class=SensorDeviceClass.SIGNAL_STRENGTH,
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=SIGNAL_STRENGTH_DECIBELS_MILLIWATT,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.rssi,
        entity_registry_enabled_default=False,
    ),
    SensiboMotionSensorEntityDescription(
        key='battery_voltage',
        translation_key='battery_voltage',
        device_class=SensorDeviceClass.VOLTAGE,
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfElectricPotential.VOLT,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.battery_voltage,
    ),
    SensiboMotionSensorEntityDescription(
        key='humidity',
        device_class=SensorDeviceClass.HUMIDITY,
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.humidity,
    ),
    SensiboMotionSensorEntityDescription(
        key='temperature',
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.temperature,
    ),
)

PURE_SENSOR_TYPES: tuple[SensiboDeviceSensorEntityDescription, ...] = (
    SensiboDeviceSensorEntityDescription(
        key='pm25',
        translation_key='pm25_pure',
        device_class=SensorDeviceClass.ENUM,
        value_fn=lambda data: data.pm25_pure.name.lower() if data.pm25_pure else None,
        extra_fn=None,
        options=[aqi.name.lower() for aqi in PureAQI],
    ),
    SensiboDeviceSensorEntityDescription(
        key='pure_sensitivity',
        translation_key='sensitivity',
        value_fn=lambda data: data.pure_sensitivity,
        extra_fn=None,
    ),
    FILTER_LAST_RESET_DESCRIPTION,
)

DEVICE_SENSOR_TYPES: tuple[SensiboDeviceSensorEntityDescription, ...] = (
    SensiboDeviceSensorEntityDescription(
        key='timer_time',
        translation_key='timer_time',
        device_class=SensorDeviceClass.TIMESTAMP,
        value_fn=lambda data: data.timer_time,
        extra_fn=lambda data: {'id': data.timer_id, 'turn_on': data.timer_state_on},
    ),
    SensiboDeviceSensorEntityDescription(
        key='feels_like',
        translation_key='feels_like',
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.feelslike,
        extra_fn=None,
        entity_registry_enabled_default=False,
    ),
    SensiboDeviceSensorEntityDescription(
        key='climate_react_low',
        translation_key='climate_react_low',
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.smart_low_temp_threshold,
        extra_fn=lambda data: data.smart_low_state,
        entity_registry_enabled_default=False,
    ),
    SensiboDeviceSensorEntityDescription(
        key='climate_react_high',
        translation_key='climate_react_high',
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.smart_high_temp_threshold,
        extra_fn=lambda data: data.smart_high_state,
        entity_registry_enabled_default=False,
    ),
    SensiboDeviceSensorEntityDescription(
        key='climate_react_type',
        translation_key='smart_type',
        value_fn=lambda data: _smart_type_name(data.smart_type),
        extra_fn=None,
        entity_registry_enabled_default=False,
    ),
    FILTER_LAST_RESET_DESCRIPTION,
)

AIRQ_SENSOR_TYPES: tuple[SensiboDeviceSensorEntityDescription, ...] = (
    SensiboDeviceSensorEntityDescription(
        key='airq_tvoc',
        translation_key='airq_tvoc',
        native_unit_of_measurement=CONCENTRATION_PARTS_PER_BILLION,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.tvoc,
        extra_fn=None,
    ),
    SensiboDeviceSensorEntityDescription(
        key='airq_co2',
        translation_key='airq_co2',
        device_class=SensorDeviceClass.CO2,
        native_unit_of_measurement=CONCENTRATION_PARTS_PER_MILLION,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.co2,
        extra_fn=None,
    ),
    *DEVICE_SENSOR_TYPES,
)

ELEMENT_SENSOR_TYPES: tuple[SensiboDeviceSensorEntityDescription, ...] = (
    SensiboDeviceSensorEntityDescription(
        key='pm25',
        device_class=SensorDeviceClass.PM25,
        native_unit_of_measurement=CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.pm25,
        extra_fn=None,
    ),
    SensiboDeviceSensorEntityDescription(
        key='tvoc',
        translation_key='tvoc',
        native_unit_of_measurement=CONCENTRATION_PARTS_PER_BILLION,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.tvoc,
        extra_fn=None,
    ),
    SensiboDeviceSensorEntityDescription(
        key='co2',
        device_class=SensorDeviceClass.CO2,
        native_unit_of_measurement=CONCENTRATION_PARTS_PER_MILLION,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.co2,
        extra_fn=None,
    ),
    SensiboDeviceSensorEntityDescription(
        key='ethanol',
        native_unit_of_measurement=CONCENTRATION_PARTS_PER_MILLION,
        state_class=SensorStateClass.MEASUREMENT,
        translation_key='ethanol',
        value_fn=lambda data: data.etoh,
        extra_fn=None,
    ),
    SensiboDeviceSensorEntityDescription(
        key='iaq',
        device_class=SensorDeviceClass.AQI,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.iaq,
        extra_fn=None,
    ),
)

DESCRIPTION_BY_MODELS: dict[str, tuple[SensiboDeviceSensorEntityDescription, ...]] = {
    'pure': PURE_SENSOR_TYPES,
    'airq': AIRQ_SENSOR_TYPES,
    'elements': ELEMENT_SENSOR_TYPES,
}

async def async_setup_entry(
    hass: HomeAssistant,
    entry: SensiboConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Sensibo sensor platform."""
    coordinator: SensiboDataUpdateCoordinator = entry.runtime_data
    added_devices: set[str] = set()

    def _add_remove_devices() -> None:
        """Handle additions of devices and sensors."""
        entities: list[SensorEntity] = []
        nonlocal added_devices
        new_devices, remove_devices, added_devices = coordinator.get_devices(added_devices)
        if new_devices:
            for device_id, device_data in coordinator.data.parsed.items():
                if device_data.motion_sensors:
                    for sensor_id, sensor_data in device_data.motion_sensors.items():
                        if sensor_id in new_devices:
                            for description in MOTION_SENSOR_TYPES:
                                entities.append(
                                    SensiboMotionSensor(coordinator, device_id, sensor_id, sensor_data, description)
                                )
            for device_id, device_data in coordinator.data.parsed.items():
                if device_id in new_devices:
                    for description in DESCRIPTION_BY_MODELS.get(device_data.model, DEVICE_SENSOR_TYPES):
                        entities.append(SensiboDeviceSensor(coordinator, device_id, description))
            async_add_entities(entities)

    entry.async_on_unload(coordinator.async_add_listener(_add_remove_devices))
    _add_remove_devices()

class SensiboMotionSensor(SensiboMotionBaseEntity, SensorEntity):
    """Representation of a Sensibo Motion Sensor."""

    def __init__(
        self,
        coordinator: SensiboDataUpdateCoordinator,
        device_id: str,
        sensor_id: str,
        sensor_data: MotionSensor,
        entity_description: SensiboMotionSensorEntityDescription,
    ) -> None:
        """Initiate Sensibo Motion Sensor."""
        super().__init__(coordinator, device_id, sensor_id, sensor_data)
        self.entity_description: SensiboMotionSensorEntityDescription = entity_description
        self._attr_unique_id: str = f'{sensor_id}-{entity_description.key}'

    @property
    def native_value(self) -> StateType:
        """Return value of sensor."""
        if False:  # for TYPE_CHECKING
            assert self.sensor_data  # type: ignore
        return self.entity_description.value_fn(self.sensor_data)

class SensiboDeviceSensor(SensiboDeviceBaseEntity, SensorEntity):
    """Representation of a Sensibo Device Sensor."""

    def __init__(
        self,
        coordinator: SensiboDataUpdateCoordinator,
        device_id: str,
        entity_description: SensiboDeviceSensorEntityDescription,
    ) -> None:
        """Initiate Sensibo Device Sensor."""
        super().__init__(coordinator, device_id)
        self.entity_description: SensiboDeviceSensorEntityDescription = entity_description
        self._attr_unique_id: str = f'{device_id}-{entity_description.key}'

    @property
    def native_value(self) -> StateType:
        """Return value of sensor."""
        return self.entity_description.value_fn(self.device_data)

    @property
    def extra_state_attributes(self) -> Optional[Mapping[str, Any]]:
        """Return additional attributes."""
        if self.entity_description.extra_fn is not None:
            return self.entity_description.extra_fn(self.device_data)
        return None

"""Representation of Venstar sensors."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, List, Dict
from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONCENTRATION_PARTS_PER_MILLION, PERCENTAGE, UnitOfTemperature, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import DOMAIN
from .coordinator import VenstarDataUpdateCoordinator
from .entity import VenstarEntity

RUNTIME_HEAT1: str = 'heat1'
RUNTIME_HEAT2: str = 'heat2'
RUNTIME_COOL1: str = 'cool1'
RUNTIME_COOL2: str = 'cool2'
RUNTIME_AUX1: str = 'aux1'
RUNTIME_AUX2: str = 'aux2'
RUNTIME_FC: str = 'fc'
RUNTIME_OV: str = 'ov'
RUNTIME_DEVICES: List[str] = [RUNTIME_HEAT1, RUNTIME_HEAT2, RUNTIME_COOL1, RUNTIME_COOL2, RUNTIME_AUX1, RUNTIME_AUX2, RUNTIME_FC, RUNTIME_OV]
RUNTIME_ATTRIBUTES: Dict[str, str] = {
    RUNTIME_HEAT1: 'Heating Stage 1',
    RUNTIME_HEAT2: 'Heating Stage 2',
    RUNTIME_COOL1: 'Cooling Stage 1',
    RUNTIME_COOL2: 'Cooling Stage 2',
    RUNTIME_AUX1: 'Aux Stage 1',
    RUNTIME_AUX2: 'Aux Stage 2',
    RUNTIME_FC: 'Free Cooling',
    RUNTIME_OV: 'Override'
}
SCHEDULE_PARTS: Dict[int, str] = {0: 'morning', 1: 'day', 2: 'evening', 3: 'night', 255: 'inactive'}
STAGES: Dict[int, str] = {0: 'idle', 1: 'first_stage', 2: 'second_stage'}

@dataclass(frozen=True, kw_only=True)
class VenstarSensorEntityDescription(SensorEntityDescription):
    """Base description of a Sensor entity."""
    # Additional callable functions for name, value and uom
    name_fn: Optional[Callable[[str], str]] = None
    value_fn: Callable[[VenstarDataUpdateCoordinator, str], Any] = lambda coordinator, sensor_name: None
    uom_fn: Callable[[VenstarDataUpdateCoordinator], Any] = lambda coordinator: None

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up Venstar device sensors based on a config entry."""
    coordinator: VenstarDataUpdateCoordinator = hass.data[DOMAIN][config_entry.entry_id]
    entities: List[VenstarSensor] = []
    sensors = coordinator.client.get_sensor_list()
    if sensors:
        for sensor_name in sensors:
            entities.extend([
                VenstarSensor(coordinator, config_entry, description, sensor_name)
                for description in SENSOR_ENTITIES
                if coordinator.client.get_sensor(sensor_name, description.key) is not None
            ])
        runtimes: Dict[str, Any] = coordinator.runtimes[-1]
        for sensor_name in runtimes:
            if sensor_name in RUNTIME_DEVICES:
                entities.append(VenstarSensor(coordinator, config_entry, RUNTIME_ENTITY, sensor_name))
            entities.extend([
                VenstarSensor(coordinator, config_entry, description, sensor_name)
                for description in CONSUMABLE_ENTITIES
                if description.key == sensor_name
            ])
    for description in INFO_ENTITIES:
        try:
            coordinator.client.get_info(description.key)
        except KeyError:
            continue
        entities.append(VenstarSensor(coordinator, config_entry, description, description.key))
    if entities:
        async_add_entities(entities)

def temperature_unit(coordinator: VenstarDataUpdateCoordinator) -> str:
    """Return the correct unit for temperature."""
    unit: str = UnitOfTemperature.CELSIUS
    if coordinator.client.tempunits == coordinator.client.TEMPUNITS_F:
        unit = UnitOfTemperature.FAHRENHEIT
    return unit

class VenstarSensor(VenstarEntity, SensorEntity):
    """Base class for a Venstar sensor."""

    def __init__(
        self,
        coordinator: VenstarDataUpdateCoordinator,
        config: ConfigEntry,
        entity_description: VenstarSensorEntityDescription,
        sensor_name: str
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, config)
        self.entity_description: VenstarSensorEntityDescription = entity_description
        self.sensor_name: str = sensor_name
        if entity_description.name_fn:
            self._attr_name: str = entity_description.name_fn(sensor_name)
        self._config: ConfigEntry = config

    @property
    def unique_id(self) -> str:
        """Return the unique id."""
        return f"{self._config.entry_id}_{self.sensor_name.replace(' ', '_')}_{self.entity_description.key}"

    @property
    def native_value(self) -> Any:
        """Return state of the sensor."""
        return self.entity_description.value_fn(self.coordinator, self.sensor_name)

    @property
    def native_unit_of_measurement(self) -> Optional[str]:
        """Return unit of measurement the value is expressed in."""
        return self.entity_description.uom_fn(self.coordinator)

SENSOR_ENTITIES: tuple[VenstarSensorEntityDescription, ...] = (
    VenstarSensorEntityDescription(
        key='hum',
        device_class=SensorDeviceClass.HUMIDITY,
        state_class=SensorStateClass.MEASUREMENT,
        uom_fn=lambda _: PERCENTAGE,
        value_fn=lambda coordinator, sensor_name: coordinator.client.get_sensor(sensor_name, 'hum'),
        name_fn=lambda sensor_name: f'{sensor_name} Humidity'
    ),
    VenstarSensorEntityDescription(
        key='temp',
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        uom_fn=temperature_unit,
        value_fn=lambda coordinator, sensor_name: round(float(coordinator.client.get_sensor(sensor_name, 'temp')), 1),
        name_fn=lambda sensor_name: f"{sensor_name.replace(' Temp', '')} Temperature"
    ),
    VenstarSensorEntityDescription(
        key='co2',
        device_class=SensorDeviceClass.CO2,
        state_class=SensorStateClass.MEASUREMENT,
        uom_fn=lambda _: CONCENTRATION_PARTS_PER_MILLION,
        value_fn=lambda coordinator, sensor_name: coordinator.client.get_sensor(sensor_name, 'co2'),
        name_fn=lambda sensor_name: f'{sensor_name} CO2'
    ),
    VenstarSensorEntityDescription(
        key='iaq',
        device_class=SensorDeviceClass.AQI,
        state_class=SensorStateClass.MEASUREMENT,
        uom_fn=lambda _: None,
        value_fn=lambda coordinator, sensor_name: coordinator.client.get_sensor(sensor_name, 'iaq'),
        name_fn=lambda sensor_name: f'{sensor_name} IAQ'
    ),
    VenstarSensorEntityDescription(
        key='battery',
        device_class=SensorDeviceClass.BATTERY,
        state_class=SensorStateClass.MEASUREMENT,
        uom_fn=lambda _: PERCENTAGE,
        value_fn=lambda coordinator, sensor_name: coordinator.client.get_sensor(sensor_name, 'battery'),
        name_fn=lambda sensor_name: f'{sensor_name} Battery'
    )
)

RUNTIME_ENTITY: VenstarSensorEntityDescription = VenstarSensorEntityDescription(
    key='runtime',
    state_class=SensorStateClass.MEASUREMENT,
    uom_fn=lambda _: UnitOfTime.MINUTES,
    value_fn=lambda coordinator, sensor_name: coordinator.runtimes[-1][sensor_name],
    name_fn=lambda sensor_name: f'{RUNTIME_ATTRIBUTES[sensor_name]} Runtime'
)

CONSUMABLE_ENTITIES: tuple[VenstarSensorEntityDescription, ...] = (
    VenstarSensorEntityDescription(
        key='filterHours',
        state_class=SensorStateClass.MEASUREMENT,
        uom_fn=lambda _: UnitOfTime.HOURS,
        value_fn=lambda coordinator, sensor_name: coordinator.runtimes[-1][sensor_name] / 100,
        name_fn=None,
        translation_key='filter_install_time'
    ),
    VenstarSensorEntityDescription(
        key='filterDays',
        state_class=SensorStateClass.MEASUREMENT,
        uom_fn=lambda _: UnitOfTime.DAYS,
        value_fn=lambda coordinator, sensor_name: coordinator.runtimes[-1][sensor_name],
        name_fn=None,
        translation_key='filter_usage'
    )
)

INFO_ENTITIES: tuple[VenstarSensorEntityDescription, ...] = (
    VenstarSensorEntityDescription(
        key='schedulepart',
        device_class=SensorDeviceClass.ENUM,
        options=list(SCHEDULE_PARTS.values()),
        translation_key='schedule_part',
        uom_fn=lambda _: None,
        value_fn=lambda coordinator, sensor_name: SCHEDULE_PARTS[coordinator.client.get_info(sensor_name)],
        name_fn=None
    ),
    VenstarSensorEntityDescription(
        key='activestage',
        device_class=SensorDeviceClass.ENUM,
        options=list(STAGES.values()),
        translation_key='active_stage',
        uom_fn=lambda _: None,
        value_fn=lambda coordinator, sensor_name: STAGES[coordinator.client.get_info(sensor_name)],
        name_fn=None
    )
)
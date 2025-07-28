from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, List
from aiolyric import Lyric
from aiolyric.objects.device import LyricDevice
from aiolyric.objects.location import LyricLocation
from aiolyric.objects.priority import LyricAccessory, LyricRoom
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from .const import DOMAIN, PRESET_HOLD_UNTIL, PRESET_NO_HOLD, PRESET_PERMANENT_HOLD, PRESET_TEMPORARY_HOLD, PRESET_VACATION_HOLD
from .entity import LyricAccessoryEntity, LyricDeviceEntity

LYRIC_SETPOINT_STATUS_NAMES: dict[str, str] = {
    PRESET_NO_HOLD: 'Following Schedule',
    PRESET_PERMANENT_HOLD: 'Held Permanently',
    PRESET_TEMPORARY_HOLD: 'Held Temporarily',
    PRESET_VACATION_HOLD: 'Holiday'
}

@dataclass(frozen=True, kw_only=True)
class LyricSensorEntityDescription(SensorEntityDescription):
    value_fn: Callable[[LyricDevice], StateType]
    suitable_fn: Callable[[LyricDevice], bool]

@dataclass(frozen=True, kw_only=True)
class LyricSensorAccessoryEntityDescription(SensorEntityDescription):
    value_fn: Callable[[LyricRoom, LyricAccessory], StateType]
    suitable_fn: Callable[[LyricRoom, LyricAccessory], bool]

DEVICE_SENSORS: List[LyricSensorEntityDescription] = [
    LyricSensorEntityDescription(
        key='indoor_temperature',
        translation_key='indoor_temperature',
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda device: device.indoor_temperature,
        suitable_fn=lambda device: device.indoor_temperature is not None
    ),
    LyricSensorEntityDescription(
        key='indoor_humidity',
        translation_key='indoor_humidity',
        device_class=SensorDeviceClass.HUMIDITY,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        value_fn=lambda device: device.indoor_humidity,
        suitable_fn=lambda device: device.indoor_humidity is not None
    ),
    LyricSensorEntityDescription(
        key='outdoor_temperature',
        translation_key='outdoor_temperature',
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda device: device.outdoor_temperature,
        suitable_fn=lambda device: device.outdoor_temperature is not None
    ),
    LyricSensorEntityDescription(
        key='outdoor_humidity',
        translation_key='outdoor_humidity',
        device_class=SensorDeviceClass.HUMIDITY,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        value_fn=lambda device: device.displayed_outdoor_humidity,
        suitable_fn=lambda device: device.displayed_outdoor_humidity is not None
    ),
    LyricSensorEntityDescription(
        key='next_period_time',
        translation_key='next_period_time',
        device_class=SensorDeviceClass.TIMESTAMP,
        value_fn=lambda device: get_datetime_from_future_time(device.changeable_values.next_period_time),  # type: ignore
        suitable_fn=lambda device: bool(device.changeable_values and device.changeable_values.next_period_time)
    ),
    LyricSensorEntityDescription(
        key='setpoint_status',
        translation_key='setpoint_status',
        value_fn=lambda device: get_setpoint_status(device.changeable_values.thermostat_setpoint_status, device.changeable_values.next_period_time),  # type: ignore
        suitable_fn=lambda device: bool(device.changeable_values and device.changeable_values.thermostat_setpoint_status)
    )
]

ACCESSORY_SENSORS: List[LyricSensorAccessoryEntityDescription] = [
    LyricSensorAccessoryEntityDescription(
        key='room_temperature',
        translation_key='room_temperature',
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda room, accessory: accessory.temperature,  # type: ignore
        suitable_fn=lambda room, accessory: accessory.type == 'IndoorAirSensor'
    ),
    LyricSensorAccessoryEntityDescription(
        key='room_humidity',
        translation_key='room_humidity',
        device_class=SensorDeviceClass.HUMIDITY,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        value_fn=lambda room, accessory: room.room_avg_humidity,  # type: ignore
        suitable_fn=lambda room, accessory: accessory.type == 'IndoorAirSensor'
    )
]

def get_setpoint_status(status: str, time: str) -> Optional[str]:
    if status == PRESET_HOLD_UNTIL:
        return f'Held until {time}'
    return LYRIC_SETPOINT_STATUS_NAMES.get(status)

def get_datetime_from_future_time(time_str: str) -> datetime:
    time_obj = dt_util.parse_time(time_str)
    if time_obj is None:
        raise ValueError(f'Unable to parse time {time_str}')
    now = dt_util.utcnow()
    if time_obj <= now.time():
        now = now + timedelta(days=1)
    return dt_util.as_utc(datetime.combine(now.date(), time_obj))

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    coordinator: DataUpdateCoordinator[Any] = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        (
            LyricSensor(coordinator, device_sensor, location, device)
            for location in coordinator.data.locations
            for device in location.devices
            for device_sensor in DEVICE_SENSORS
            if device_sensor.suitable_fn(device)
        )
    )
    async_add_entities(
        (
            LyricAccessorySensor(coordinator, accessory_sensor, location, device, room, accessory)
            for location in coordinator.data.locations
            for device in location.devices
            for room in coordinator.data.rooms_dict.get(device.mac_id, {}).values()
            for accessory in room.accessories
            for accessory_sensor in ACCESSORY_SENSORS
            if accessory_sensor.suitable_fn(room, accessory)
        )
    )

class LyricSensor(LyricDeviceEntity, SensorEntity):
    def __init__(
        self,
        coordinator: DataUpdateCoordinator[Any],
        description: LyricSensorEntityDescription,
        location: LyricLocation,
        device: LyricDevice
    ) -> None:
        super().__init__(coordinator, location, device, f'{device.mac_id}_{description.key}')
        self.entity_description: LyricSensorEntityDescription = description
        if description.device_class == SensorDeviceClass.TEMPERATURE:
            if device.units == 'Fahrenheit':
                self._attr_native_unit_of_measurement = UnitOfTemperature.FAHRENHEIT
            else:
                self._attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS

    @property
    def native_value(self) -> StateType:
        return self.entity_description.value_fn(self.device)

class LyricAccessorySensor(LyricAccessoryEntity, SensorEntity):
    def __init__(
        self,
        coordinator: DataUpdateCoordinator[Any],
        description: LyricSensorAccessoryEntityDescription,
        location: LyricLocation,
        parentDevice: LyricDevice,
        room: LyricRoom,
        accessory: LyricAccessory
    ) -> None:
        super().__init__(coordinator, location, parentDevice, room, accessory, f'{parentDevice.mac_id}_room{room.id}_acc{accessory.id}_{description.key}')
        self.entity_description: LyricSensorAccessoryEntityDescription = description
        if description.device_class == SensorDeviceClass.TEMPERATURE:
            if parentDevice.units == 'Fahrenheit':
                self._attr_native_unit_of_measurement = UnitOfTemperature.FAHRENHEIT
            else:
                self._attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS

    @property
    def native_value(self) -> StateType:
        return self.entity_description.value_fn(self.room, self.accessory)
"""Support for Honeywell Lyric sensor platform."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from aiolyric import Lyric
from aiolyric.objects.device import LyricDevice
from aiolyric.objects.location import LyricLocation
from aiolyric.objects.priority import LyricAccessory, LyricRoom
from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    PRESET_HOLD_UNTIL,
    PRESET_NO_HOLD,
    PRESET_PERMANENT_HOLD,
    PRESET_TEMPORARY_HOLD,
    PRESET_VACATION_HOLD,
)
from .entity import LyricAccessoryEntity, LyricDeviceEntity

LYRIC_SETPOINT_STATUS_NAMES: Dict[str, str] = {
    PRESET_NO_HOLD: "Following Schedule",
    PRESET_PERMANENT_HOLD: "Held Permanently",
    PRESET_TEMPORARY_HOLD: "Held Temporarily",
    PRESET_VACATION_HOLD: "Holiday",
}


@dataclass(frozen=True, kw_only=True)
class LyricSensorEntityDescription(SensorEntityDescription):
    """Class describing Honeywell Lyric sensor entities."""

    value_fn: Callable[[LyricDevice], Optional[StateType]]
    suitable_fn: Callable[[LyricDevice], bool]


@dataclass(frozen=True, kw_only=True)
class LyricSensorAccessoryEntityDescription(SensorEntityDescription):
    """Class describing Honeywell Lyric room sensor entities."""

    value_fn: Callable[[LyricRoom, LyricAccessory], Optional[StateType]]
    suitable_fn: Callable[[LyricRoom, LyricAccessory], bool]


DEVICE_SENSORS: List[LyricSensorEntityDescription] = [
    LyricSensorEntityDescription(
        key="indoor_temperature",
        translation_key="indoor_temperature",
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda device: device.indoor_temperature,
        suitable_fn=lambda device: device.indoor_temperature is not None,
    ),
    LyricSensorEntityDescription(
        key="indoor_humidity",
        translation_key="indoor_humidity",
        device_class=SensorDeviceClass.HUMIDITY,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        value_fn=lambda device: device.indoor_humidity,
        suitable_fn=lambda device: device.indoor_humidity is not None,
    ),
    LyricSensorEntityDescription(
        key="outdoor_temperature",
        translation_key="outdoor_temperature",
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda device: device.outdoor_temperature,
        suitable_fn=lambda device: device.outdoor_temperature is not None,
    ),
    LyricSensorEntityDescription(
        key="outdoor_humidity",
        translation_key="outdoor_humidity",
        device_class=SensorDeviceClass.HUMIDITY,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        value_fn=lambda device: device.displayed_outdoor_humidity,
        suitable_fn=lambda device: device.displayed_outdoor_humidity is not None,
    ),
    LyricSensorEntityDescription(
        key="next_period_time",
        translation_key="next_period_time",
        device_class=SensorDeviceClass.TIMESTAMP,
        value_fn=lambda device: get_datetime_from_future_time(device.changeable_values.next_period_time)
        if device.changeable_values and device.changeable_values.next_period_time
        else None,
        suitable_fn=lambda device: device.changeable_values is not None
        and device.changeable_values.next_period_time is not None,
    ),
    LyricSensorEntityDescription(
        key="setpoint_status",
        translation_key="setpoint_status",
        value_fn=lambda device: get_setpoint_status(
            device.changeable_values.thermostat_setpoint_status, device.changeable_values.next_period_time
        )
        if device.changeable_values
        else None,
        suitable_fn=lambda device: device.changeable_values is not None
        and device.changeable_values.thermostat_setpoint_status is not None,
    ),
]

ACCESSORY_SENSORS: List[LyricSensorAccessoryEntityDescription] = [
    LyricSensorAccessoryEntityDescription(
        key="room_temperature",
        translation_key="room_temperature",
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda room, accessory: accessory.temperature,
        suitable_fn=lambda room, accessory: accessory.type == "IndoorAirSensor",
    ),
    LyricSensorAccessoryEntityDescription(
        key="room_humidity",
        translation_key="room_humidity",
        device_class=SensorDeviceClass.HUMIDITY,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        value_fn=lambda room, _: room.room_avg_humidity,
        suitable_fn=lambda room, accessory: accessory.type == "IndoorAirSensor",
    ),
]


def get_setpoint_status(status: str, time: str) -> Optional[str]:
    """Get status of the setpoint."""
    if status == PRESET_HOLD_UNTIL:
        return f"Held until {time}"
    return LYRIC_SETPOINT_STATUS_NAMES.get(status)


def get_datetime_from_future_time(time_str: str) -> datetime:
    """Get datetime from future time provided."""
    time_obj = dt_util.parse_time(time_str)
    if time_obj is None:
        raise ValueError(f"Unable to parse time {time_str}")
    now = dt_util.utcnow()
    if time_obj <= now.time():
        now = now + timedelta(days=1)
    return dt_util.as_utc(datetime.combine(now.date(), time_obj))


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Honeywell Lyric sensor platform based on a config entry."""
    coordinator: DataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]
    devices: List[LyricDevice] = [
        device for location in coordinator.data.locations for device in location.devices
    ]
    sensor_entities: List[LyricSensor] = [
        LyricSensor(coordinator, device_sensor, location, device)
        for location in coordinator.data.locations
        for device in location.devices
        for device_sensor in DEVICE_SENSORS
        if device_sensor.suitable_fn(device)
    ]
    accessory_entities: List[LyricAccessorySensor] = [
        LyricAccessorySensor(coordinator, accessory_sensor, location, device, room, accessory)
        for location in coordinator.data.locations
        for device in location.devices
        for room in coordinator.data.rooms_dict.get(device.mac_id, {}).values()
        for accessory in room.accessories
        for accessory_sensor in ACCESSORY_SENSORS
        if accessory_sensor.suitable_fn(room, accessory)
    ]
    async_add_entities(sensor_entities + accessory_entities)


class LyricSensor(LyricDeviceEntity, SensorEntity):
    """Define a Honeywell Lyric sensor."""

    entity_description: LyricSensorEntityDescription
    _attr_native_unit_of_measurement: Optional[str]

    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        description: LyricSensorEntityDescription,
        location: LyricLocation,
        device: LyricDevice,
    ) -> None:
        """Initialize."""
        unique_id = f"{device.mac_id}_{description.key}"
        super().__init__(coordinator, location, device, unique_id)
        self.entity_description = description
        if description.device_class == SensorDeviceClass.TEMPERATURE:
            if device.units == "Fahrenheit":
                self._attr_native_unit_of_measurement = UnitOfTemperature.FAHRENHEIT
            else:
                self._attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS

    @property
    def native_value(self) -> Optional[StateType]:
        """Return the state."""
        return self.entity_description.value_fn(self.device)


class LyricAccessorySensor(LyricAccessoryEntity, SensorEntity):
    """Define a Honeywell Lyric sensor."""

    entity_description: LyricSensorAccessoryEntityDescription
    _attr_native_unit_of_measurement: Optional[str]

    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        description: LyricSensorAccessoryEntityDescription,
        location: LyricLocation,
        parentDevice: LyricDevice,
        room: LyricRoom,
        accessory: LyricAccessory,
    ) -> None:
        """Initialize."""
        unique_id = f"{parentDevice.mac_id}_room{room.id}_acc{accessory.id}_{description.key}"
        super().__init__(coordinator, location, parentDevice, room, accessory, unique_id)
        self.entity_description = description
        if description.device_class == SensorDeviceClass.TEMPERATURE:
            if parentDevice.units == "Fahrenheit":
                self._attr_native_unit_of_measurement = UnitOfTemperature.FAHRENHEIT
            else:
                self._attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS

    @property
    def native_value(self) -> Optional[StateType]:
        """Return the state."""
        return self.entity_description.value_fn(self.room, self.accessory)

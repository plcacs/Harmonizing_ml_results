from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
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

LYRIC_SETPOINT_STATUS_NAMES: dict[str, str] = {PRESET_NO_HOLD: 'Following Schedule', PRESET_PERMANENT_HOLD: 'Held Permanently', PRESET_TEMPORARY_HOLD: 'Held Temporarily', PRESET_VACATION_HOLD: 'Holiday'}

@dataclass(frozen=True, kw_only=True)
class LyricSensorEntityDescription(SensorEntityDescription):
    """Class describing Honeywell Lyric sensor entities."""

@dataclass(frozen=True, kw_only=True)
class LyricSensorAccessoryEntityDescription(SensorEntityDescription):
    """Class describing Honeywell Lyric room sensor entities."""

DEVICE_SENSORS: list[LyricSensorEntityDescription] = [...]
ACCESSORY_SENSORS: list[LyricSensorAccessoryEntityDescription] = [...]

def get_setpoint_status(status: str, time: str) -> str:
    """Get status of the setpoint."""
    ...

def get_datetime_from_future_time(time_str: str) -> datetime:
    """Get datetime from future time provided."""
    ...

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the Honeywell Lyric sensor platform based on a config entry."""
    ...

class LyricSensor(LyricDeviceEntity, SensorEntity):
    """Define a Honeywell Lyric sensor."""

    def __init__(self, coordinator: DataUpdateCoordinator, description: LyricSensorEntityDescription, location: LyricLocation, device: LyricDevice) -> None:
        """Initialize."""
        ...

    @property
    def native_value(self) -> StateType:
        """Return the state."""
        ...

class LyricAccessorySensor(LyricAccessoryEntity, SensorEntity):
    """Define a Honeywell Lyric sensor."""

    def __init__(self, coordinator: DataUpdateCoordinator, description: LyricSensorAccessoryEntityDescription, location: LyricLocation, parentDevice: LyricDevice, room: LyricRoom, accessory: LyricAccessory) -> None:
        """Initialize."""
        ...

    @property
    def native_value(self) -> StateType:
        """Return the state."""
        ...

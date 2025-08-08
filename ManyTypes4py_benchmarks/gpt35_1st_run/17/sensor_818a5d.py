from __future__ import annotations
from collections.abc import Callable, Mapping
import logging
import math
from typing import TYPE_CHECKING, Any, Optional, Union

import voluptuous as vol
from homeassistant import util
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_UNIT_OF_MEASUREMENT, CONF_NAME, CONF_UNIQUE_ID, PERCENTAGE, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTemperature
from homeassistant.core import CALLBACK_TYPE, Event, EventStateChangedData, HomeAssistant, State, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.device import async_device_info_to_link_from_entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util.unit_conversion import TemperatureConverter
from homeassistant.util.unit_system import METRIC_SYSTEM
from .const import CONF_CALIBRATION_FACTOR, CONF_INDOOR_HUMIDITY, CONF_INDOOR_TEMP, CONF_OUTDOOR_TEMP, DEFAULT_NAME

_LOGGER: logging.Logger

ATTR_CRITICAL_TEMP: str = 'estimated_critical_temp'
ATTR_DEWPOINT: str = 'dewpoint'
MAGNUS_K2: float = 17.62
MAGNUS_K3: float = 243.12

PLATFORM_SCHEMA: vol.Schema

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:

class MoldIndicator(SensorEntity):

    def __init__(self, hass: HomeAssistant, name: str, is_metric: bool, indoor_temp_sensor: str, outdoor_temp_sensor: str, indoor_humidity_sensor: str, calib_factor: Optional[float], unique_id: Optional[str]) -> None:

    async def async_added_to_hass(self) -> None:

    @callback
    def async_start_preview(self, preview_callback: Callable[[str, dict[str, Any]], None]) -> Union[CALLBACK_TYPE, None]:

    async def async_update(self) -> None:

    @property
    def extra_state_attributes(self) -> dict[str, Union[float, None]]:

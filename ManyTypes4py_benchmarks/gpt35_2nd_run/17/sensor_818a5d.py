from __future__ import annotations
from collections.abc import Callable, Mapping
import logging
import math
from typing import TYPE_CHECKING, Any
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

ATTR_CRITICAL_TEMP: str
ATTR_DEWPOINT: str
MAGNUS_K2: float
MAGNUS_K3: float
PLATFORM_SCHEMA: vol.Schema

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType) -> None:

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:

class MoldIndicator(SensorEntity):

    def __init__(self, hass: HomeAssistant, name: str, is_metric: bool, indoor_temp_sensor: str, outdoor_temp_sensor: str, indoor_humidity_sensor: str, calib_factor: float, unique_id: str) -> None:

    @callback
    def async_start_preview(self, preview_callback: Callable) -> Any:

    async def async_added_to_hass(self) -> None:

    @callback
    def _async_setup_sensor(self) -> None:

    def _update_sensor(self, entity: str, old_state: State, new_state: State) -> bool:

    @staticmethod
    def _update_temp_sensor(state: State) -> float:

    @staticmethod
    def _update_hum_sensor(state: State) -> float:

    async def async_update(self) -> None:

    def _calc_dewpoint(self) -> None:

    def _calc_moldindicator(self) -> None:

    @property
    def extra_state_attributes(self) -> Mapping[str, Any]:

from __future__ import annotations
from datetime import datetime
import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_UNIT_OF_MEASUREMENT, CONF_NAME, CONF_TYPE, CONF_UNIQUE_ID, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import Event, HomeAssistant
from homeassistant.helpers import config_validation as cv, entity_registry as er
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType, StateType
from . import PLATFORMS
from .const import CONF_ENTITY_IDS, CONF_ROUND_DIGITS, DOMAIN

_LOGGER: logging.Logger

ATTR_MIN_VALUE: str
ATTR_MIN_ENTITY_ID: str
ATTR_MAX_VALUE: str
ATTR_MAX_ENTITY_ID: str
ATTR_MEAN: str
ATTR_MEDIAN: str
ATTR_LAST: str
ATTR_LAST_ENTITY_ID: str
ATTR_RANGE: str
ATTR_SUM: str
ICON: str
SENSOR_TYPES: Dict[str, str]
SENSOR_TYPE_TO_ATTR: Dict[str, str]
PLATFORM_SCHEMA: vol.Schema

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: Optional[DiscoveryInfoType] = None) -> None:

def calc_min(sensor_values: List[Tuple[str, StateType]]) -> Tuple[Optional[str], Optional[StateType]]:

def calc_max(sensor_values: List[Tuple[str, StateType]]) -> Tuple[Optional[str], Optional[StateType]]:

def calc_mean(sensor_values: List[Tuple[str, StateType]], round_digits: int) -> Optional[float]:

def calc_median(sensor_values: List[Tuple[str, StateType]], round_digits: int) -> Optional[float]:

def calc_range(sensor_values: List[Tuple[str, StateType]], round_digits: int) -> Optional[float]:

def calc_sum(sensor_values: List[Tuple[str, StateType]], round_digits: int) -> Optional[float]:

class MinMaxSensor(SensorEntity):

    def __init__(self, entity_ids: List[str], name: Optional[str], sensor_type: str, round_digits: int, unique_id: Optional[str]) -> None:

    async def async_added_to_hass(self) -> None:

    @property
    def native_value(self) -> Optional[StateType]:

    @property
    def native_unit_of_measurement(self) -> str:

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:

    def _async_min_max_sensor_state_listener(self, event: Event, update_state: bool = True) -> None:

    def _calc_values(self) -> None:

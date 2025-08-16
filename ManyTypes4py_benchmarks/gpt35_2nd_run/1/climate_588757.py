from __future__ import annotations
import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta
import logging
import math
from typing import Any, Dict, List, Optional
import voluptuous as vol
from homeassistant.components.climate import ATTR_PRESET_MODE, PLATFORM_SCHEMA as CLIMATE_PLATFORM_SCHEMA, PRESET_NONE, ClimateEntity, ClimateEntityFeature, HVACAction, HVACMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, ATTR_TEMPERATURE, CONF_NAME, CONF_UNIQUE_ID, EVENT_HOMEASSISTANT_START, PRECISION_HALVES, PRECISION_TENTHS, PRECISION_WHOLE, SERVICE_TURN_OFF, SERVICE_TURN_ON, STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTemperature
from homeassistant.core import DOMAIN as HOMEASSISTANT_DOMAIN, CoreState, Event, EventStateChangedData, HomeAssistant, State, callback
from homeassistant.exceptions import ConditionError
from homeassistant.helpers import condition, config_validation as cv
from homeassistant.helpers.device import async_device_info_to_link_from_entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType, VolDictType
from .const import CONF_AC_MODE, CONF_COLD_TOLERANCE, CONF_HEATER, CONF_HOT_TOLERANCE, CONF_MAX_TEMP, CONF_MIN_DUR, CONF_MIN_TEMP, CONF_PRESETS, CONF_SENSOR, DEFAULT_TOLERANCE, DOMAIN, PLATFORMS

_LOGGER: logging.Logger

DEFAULT_NAME: str

CONF_INITIAL_HVAC_MODE: str
CONF_KEEP_ALIVE: str
CONF_PRECISION: str
CONF_TARGET_TEMP: str
CONF_TEMP_STEP: str

PRESETS_SCHEMA: Dict[str, vol.Coerce[float]]
PLATFORM_SCHEMA_COMMON: vol.Schema
PLATFORM_SCHEMA: vol.Schema

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

async def _async_setup_config(hass: HomeAssistant, config: ConfigType, unique_id: str, async_add_entities: AddEntitiesCallback) -> None:

class GenericThermostat(ClimateEntity, RestoreEntity):

    def __init__(self, hass: HomeAssistant, name: str, heater_entity_id: str, sensor_entity_id: str, min_temp: Optional[float], max_temp: Optional[float], target_temp: Optional[float], ac_mode: Optional[bool], min_cycle_duration: Optional[timedelta], cold_tolerance: float, hot_tolerance: float, keep_alive: Optional[timedelta], initial_hvac_mode: Optional[str], presets: Dict[str, float], precision: Optional[float], target_temperature_step: Optional[float], unit: UnitOfTemperature, unique_id: str) -> None:

    async def async_added_to_hass(self) -> None:

    @property
    def precision(self) -> float:

    @property
    def target_temperature_step(self) -> float:

    @property
    def current_temperature(self) -> Optional[float]:

    @property
    def hvac_mode(self) -> str:

    @property
    def hvac_action(self) -> str:

    @property
    def target_temperature(self) -> Optional[float]:

    async def async_set_hvac_mode(self, hvac_mode: str) -> None:

    async def async_set_temperature(self, **kwargs: Any) -> None:

    @property
    def min_temp(self) -> float:

    @property
    def max_temp(self) -> float:

    async def _async_sensor_changed(self, event: Event) -> None:

    async def _check_switch_initial_state(self) -> None:

    def _async_switch_changed(self, event: Event) -> None:

    def _async_update_temp(self, state: State) -> None:

    async def _async_control_heating(self, time: Optional[datetime] = None, force: bool = False) -> None:

    @property
    def _is_device_active(self) -> Optional[bool]:

    async def _async_heater_turn_on(self) -> None:

    async def _async_heater_turn_off(self) -> None:

    async def async_set_preset_mode(self, preset_mode: str) -> None:

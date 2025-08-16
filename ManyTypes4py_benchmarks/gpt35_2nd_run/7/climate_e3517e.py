from __future__ import annotations
import asyncio
import enum
import logging
from typing import Any
from aiolyric import Lyric
from aiolyric.objects.device import LyricDevice
from aiolyric.objects.location import LyricLocation
import voluptuous as vol
from homeassistant.components.climate import ATTR_TARGET_TEMP_HIGH, ATTR_TARGET_TEMP_LOW, FAN_AUTO, FAN_DIFFUSE, FAN_ON, ClimateEntity, ClimateEntityDescription, ClimateEntityFeature, HVACAction, HVACMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_TEMPERATURE, PRECISION_HALVES, PRECISION_WHOLE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, entity_platform
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import VolDictType
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from .const import DOMAIN, LYRIC_EXCEPTIONS, PRESET_HOLD_UNTIL, PRESET_NO_HOLD, PRESET_PERMANENT_HOLD, PRESET_TEMPORARY_HOLD, PRESET_VACATION_HOLD
from .entity import LyricDeviceEntity

SUPPORT_FLAGS_LCC: int = ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.PRESET_MODE | ClimateEntityFeature.TARGET_TEMPERATURE_RANGE
SUPPORT_FLAGS_TCC: int = ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.TARGET_TEMPERATURE_RANGE

LYRIC_HVAC_ACTION_OFF: str = 'EquipmentOff'
LYRIC_HVAC_ACTION_HEAT: str = 'Heat'
LYRIC_HVAC_ACTION_COOL: str = 'Cool'
LYRIC_HVAC_MODE_OFF: str = 'Off'
LYRIC_HVAC_MODE_HEAT: str = 'Heat'
LYRIC_HVAC_MODE_COOL: str = 'Cool'
LYRIC_HVAC_MODE_HEAT_COOL: str = 'Auto'
LYRIC_FAN_MODE_ON: str = 'On'
LYRIC_FAN_MODE_AUTO: str = 'Auto'
LYRIC_FAN_MODE_DIFFUSE: str = 'Circulate'

LYRIC_HVAC_MODES: dict[HVACMode, str] = {HVACMode.OFF: LYRIC_HVAC_MODE_OFF, HVACMode.HEAT: LYRIC_HVAC_MODE_HEAT, HVACMode.COOL: LYRIC_HVAC_MODE_COOL, HVACMode.HEAT_COOL: LYRIC_HVAC_MODE_HEAT_COOL}
HVAC_MODES: dict[str, HVACMode] = {LYRIC_HVAC_MODE_OFF: HVACMode.OFF, LYRIC_HVAC_MODE_HEAT: HVACMode.HEAT, LYRIC_HVAC_MODE_COOL: HVACMode.COOL, LYRIC_HVAC_MODE_HEAT_COOL: HVACMode.HEAT_COOL}
LYRIC_FAN_MODES: dict[str, str] = {FAN_ON: LYRIC_FAN_MODE_ON, FAN_AUTO: LYRIC_FAN_MODE_AUTO, FAN_DIFFUSE: LYRIC_FAN_MODE_DIFFUSE}
FAN_MODES: dict[str, str] = {LYRIC_FAN_MODE_ON: FAN_ON, LYRIC_FAN_MODE_AUTO: FAN_AUTO, LYRIC_FAN_MODE_DIFFUSE: FAN_DIFFUSE}
HVAC_ACTIONS: dict[str, HVACAction] = {LYRIC_HVAC_ACTION_OFF: HVACAction.OFF, LYRIC_HVAC_ACTION_HEAT: HVACAction.HEATING, LYRIC_HVAC_ACTION_COOL: HVACAction.COOLING}

SERVICE_HOLD_TIME: str = 'set_hold_time'
ATTR_TIME_PERIOD: str = 'time_period'
SCHEMA_HOLD_TIME: dict = {vol.Required(ATTR_TIME_PERIOD, default='01:00:00'): vol.All(cv.time_period, cv.positive_timedelta, lambda td: strftime('%H:%M:%S', localtime(time() + td.total_seconds())))}

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    coordinator: DataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities((LyricClimate(coordinator, ClimateEntityDescription(key=f'{device.mac_id}_thermostat', name=device.name), location, device) for location in coordinator.data.locations for device in location.devices, True)
    platform: entity_platform.EntityPlatform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(SERVICE_HOLD_TIME, SCHEMA_HOLD_TIME, 'async_set_hold_time')

class LyricThermostatType(enum.Enum):
    TCC = enum.auto()
    LCC = enum.auto()

class LyricClimate(LyricDeviceEntity, ClimateEntity):
    _attr_name: str = None
    _attr_preset_modes: list[str] = [PRESET_NO_HOLD, PRESET_HOLD_UNTIL, PRESET_PERMANENT_HOLD, PRESET_TEMPORARY_HOLD, PRESET_VACATION_HOLD]

    def __init__(self, coordinator: DataUpdateCoordinator, description: ClimateEntityDescription, location: LyricLocation, device: LyricDevice) -> None:
        ...

    @property
    def current_temperature(self) -> Any:
        ...

    @property
    def hvac_action(self) -> HVACAction:
        ...

    @property
    def hvac_mode(self) -> HVACMode:
        ...

    @property
    def target_temperature(self) -> Any:
        ...

    @property
    def target_temperature_high(self) -> Any:
        ...

    @property
    def target_temperature_low(self) -> Any:
        ...

    @property
    def preset_mode(self) -> Any:
        ...

    @property
    def min_temp(self) -> Any:
        ...

    @property
    def max_temp(self) -> Any:
        ...

    @property
    def fan_mode(self) -> Any:
        ...

    async def async_set_temperature(self, **kwargs: Any) -> None:
        ...

    async def async_set_hvac_mode(self, hvac_mode: str) -> None:
        ...

    async def _async_set_hvac_mode_tcc(self, hvac_mode: str) -> None:
        ...

    async def _async_set_hvac_mode_lcc(self, hvac_mode: str) -> None:
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        ...

    async def async_set_hold_time(self, time_period: str) -> None:
        ...

    async def async_set_fan_mode(self, fan_mode: str) -> None:
        ...

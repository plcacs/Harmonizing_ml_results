from __future__ import annotations
import datetime
import logging
from typing import Any, List, Optional, Union
import voluptuous as vol
from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_COLOR_TEMP_KELVIN, ATTR_RGB_COLOR, ATTR_TRANSITION, ATTR_XY_COLOR, DOMAIN as LIGHT_DOMAIN, VALID_TRANSITION, is_on
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN, SwitchEntity
from homeassistant.const import ATTR_ENTITY_ID, CONF_BRIGHTNESS, CONF_LIGHTS, CONF_MODE, CONF_NAME, CONF_PLATFORM, SERVICE_TURN_ON, STATE_ON, SUN_EVENT_SUNRISE, SUN_EVENT_SUNSET
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv, event
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.sun import get_astral_event_date
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import slugify
from homeassistant.util.color import color_RGB_to_xy_brightness, color_temperature_to_rgb
from homeassistant.util.dt import as_local, utcnow as dt_utcnow

_LOGGER: logging.Logger

ATTR_UNIQUE_ID: str
CONF_START_TIME: str
CONF_STOP_TIME: str
CONF_START_CT: str
CONF_SUNSET_CT: str
CONF_STOP_CT: str
CONF_DISABLE_BRIGHTNESS_ADJUST: str
CONF_INTERVAL: str
MODE_XY: str
MODE_MIRED: str
MODE_RGB: str
DEFAULT_MODE: str
PLATFORM_SCHEMA: vol.Schema

async def async_set_lights_xy(hass: HomeAssistant, lights: List[str], x_val: Optional[float], y_val: Optional[float], brightness: Optional[int], transition: Optional[int]) -> None

async def async_set_lights_temp(hass: HomeAssistant, lights: List[str], kelvin: Optional[int], brightness: Optional[int], transition: Optional[int]) -> None

async def async_set_lights_rgb(hass: HomeAssistant, lights: List[str], rgb: Optional[List[int]], transition: Optional[int]) -> None

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None

class FluxSwitch(SwitchEntity, RestoreEntity):
    def __init__(self, name: str, hass: HomeAssistant, lights: List[str], start_time: Optional[datetime.time], stop_time: Optional[datetime.time], start_colortemp: int, sunset_colortemp: int, stop_colortemp: int, brightness: Optional[int], disable_brightness_adjust: bool, mode: str, interval: int, transition: int, unique_id: str) -> None

    @property
    def name(self) -> str

    @property
    def is_on(self) -> bool

    async def async_added_to_hass(self) -> None

    async def async_will_remove_from_hass(self) -> None

    async def async_turn_on(self, **kwargs: Any) -> None

    async def async_turn_off(self, **kwargs: Any) -> None

    async def async_flux_update(self, utcnow: Optional[datetime.datetime] = None) -> None

    def find_start_time(self, now: datetime.datetime) -> datetime.datetime

    def find_stop_time(self, now: datetime.datetime) -> datetime.datetime

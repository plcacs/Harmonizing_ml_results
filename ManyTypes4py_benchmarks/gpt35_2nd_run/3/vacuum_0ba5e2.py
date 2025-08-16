from __future__ import annotations
from datetime import timedelta
import logging
from typing import Any, Dict, List, Optional
import voluptuous as vol
from homeassistant.components.vacuum import ATTR_STATUS, StateVacuumEntity, VacuumActivity, VacuumEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_MODE
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, entity_platform
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import ACTION, ALERTS, ERRORS, MODE, NEATO_LOGIN, NEATO_MAP_DATA, NEATO_PERSISTENT_MAPS, NEATO_ROBOTS, SCAN_INTERVAL_MINUTES
from .entity import NeatoEntity
from .hub import NeatoHub

_LOGGER: logging.Logger

SCAN_INTERVAL: timedelta

ATTR_CLEAN_START: str
ATTR_CLEAN_STOP: str
ATTR_CLEAN_AREA: str
ATTR_CLEAN_BATTERY_START: str
ATTR_CLEAN_BATTERY_END: str
ATTR_CLEAN_SUSP_COUNT: str
ATTR_CLEAN_SUSP_TIME: str
ATTR_CLEAN_PAUSE_TIME: str
ATTR_CLEAN_ERROR_TIME: str
ATTR_LAUNCHED_FROM: str
ATTR_NAVIGATION: str
ATTR_CATEGORY: str
ATTR_ZONE: str

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class NeatoConnectedVacuum(NeatoEntity, StateVacuumEntity):

    _attr_supported_features: int
    _attr_name: Optional[str]

    def __init__(self, neato: NeatoHub, robot: Robot, mapdata: Dict[str, Any], persistent_maps: Dict[str, List[Dict[str, Any]]]) -> None:

    def update(self) -> None:

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:

    @property
    def device_info(self) -> DeviceInfo:

    def start(self) -> None:

    def pause(self) -> None:

    def return_to_base(self, **kwargs: Any) -> None:

    def stop(self, **kwargs: Any) -> None:

    def locate(self, **kwargs: Any) -> None:

    def clean_spot(self, **kwargs: Any) -> None:

    def neato_custom_cleaning(self, mode: int, navigation: int, category: int, zone: Optional[str] = None) -> None:

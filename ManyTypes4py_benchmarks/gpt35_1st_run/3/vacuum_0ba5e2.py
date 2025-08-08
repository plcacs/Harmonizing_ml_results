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

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    neato: NeatoHub = hass.data[NEATO_LOGIN]
    mapdata: Optional[Dict[str, Any]] = hass.data.get(NEATO_MAP_DATA)
    persistent_maps: Optional[Dict[str, List[Dict[str, Any]]]] = hass.data.get(NEATO_PERSISTENT_MAPS)
    dev: List[NeatoConnectedVacuum] = [NeatoConnectedVacuum(neato, robot, mapdata, persistent_maps) for robot in hass.data[NEATO_ROBOTS]]
    if not dev:
        return
    _LOGGER.debug('Adding vacuums %s', dev)
    async_add_entities(dev, True)
    platform: entity_platform.EntityPlatform = entity_platform.async_get_current_platform()
    assert platform is not None
    platform.async_register_entity_service('custom_cleaning', {vol.Optional(ATTR_MODE, default=2): cv.positive_int, vol.Optional(ATTR_NAVIGATION, default=1): cv.positive_int, vol.Optional(ATTR_CATEGORY, default=4): cv.positive_int, vol.Optional(ATTR_ZONE): cv.string}, 'neato_custom_cleaning')

class NeatoConnectedVacuum(NeatoEntity, StateVacuumEntity):
    _attr_supported_features: int = VacuumEntityFeature.BATTERY | VacuumEntityFeature.PAUSE | VacuumEntityFeature.RETURN_HOME | VacuumEntityFeature.STOP | VacuumEntityFeature.START | VacuumEntityFeature.CLEAN_SPOT | VacuumEntityFeature.STATE | VacuumEntityFeature.MAP | VacuumEntityFeature.LOCATE
    _attr_name: Optional[str] = None

    def __init__(self, neato: NeatoHub, robot: Robot, mapdata: Optional[Dict[str, Any]], persistent_maps: Optional[Dict[str, List[Dict[str, Any]]]) -> None:
        super().__init__(robot)
        self._attr_available: bool = neato is not None
        self._mapdata: Optional[Dict[str, Any]] = mapdata
        self._robot_has_map: bool = self.robot.has_persistent_maps
        self._robot_maps: Optional[Dict[str, List[Dict[str, Any]]]] = persistent_maps
        self._robot_serial: str = self.robot.serial
        self._attr_unique_id: str = self.robot.serial
        self._status_state: Optional[str] = None
        self._state: Optional[Dict[str, Any]] = None
        self._clean_time_start: Optional[str] = None
        self._clean_time_stop: Optional[str] = None
        self._clean_area: Optional[float] = None
        self._clean_battery_start: Optional[int] = None
        self._clean_battery_end: Optional[int] = None
        self._clean_susp_charge_count: Optional[int] = None
        self._clean_susp_time: Optional[int] = None
        self._clean_pause_time: Optional[int] = None
        self._clean_error_time: Optional[int] = None
        self._launched_from: Optional[str] = None
        self._robot_boundaries: List[Dict[str, Any]] = []
        self._robot_stats: Optional[Dict[str, Any]] = None

    def update(self) -> None:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

    @property
    def device_info(self) -> Dict[str, Any]:
        ...

    def start(self) -> None:
        ...

    def pause(self) -> None:
        ...

    def return_to_base(self, **kwargs: Any) -> None:
        ...

    def stop(self, **kwargs: Any) -> None:
        ...

    def locate(self, **kwargs: Any) -> None:
        ...

    def clean_spot(self, **kwargs: Any) -> None:
        ...

    def neato_custom_cleaning(self, mode: int, navigation: int, category: int, zone: Optional[str] = None) -> None:
        ...

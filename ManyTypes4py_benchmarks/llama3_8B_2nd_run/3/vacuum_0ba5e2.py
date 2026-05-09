from __future__ import annotations
from datetime import timedelta
import logging
from typing import Any, Optional

from pybotvac import Robot
from pybotvac.exceptions import NeatoRobotException
import voluptuous as vol
from homeassistant.components.vacuum import ATTR_STATUS, StateVacuumEntity, VacuumActivity, VacuumEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_MODE
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, entity_platform
from homeassistant.helpers.device_registry import DeviceInfo
from .const import ACTION, ALERTS, ERRORS, MODE, NEATO_LOGIN, NEATO_MAP_DATA, NEATO_PERSISTENT_MAPS, NEATO_ROBOTS, SCAN_INTERVAL_MINUTES
from .entity import NeatoEntity
from .hub import NeatoHub

_LOGGER = logging.getLogger(__name__)
SCAN_INTERVAL = timedelta(minutes=SCAN_INTERVAL_MINUTES)

class NeatoConnectedVacuum(NeatoEntity, StateVacuumEntity):
    """Representation of a Neato Connected Vacuum."""

    _attr_supported_features: VacuumEntityFeature = VacuumEntityFeature.BATTERY | VacuumEntityFeature.PAUSE | VacuumEntityFeature.RETURN_HOME | VacuumEntityFeature.STOP | VacuumEntityFeature.START | VacuumEntityFeature.CLEAN_SPOT | VacuumEntityFeature.STATE | VacuumEntityFeature.MAP | VacuumEntityFeature.LOCATE
    _attr_name: str
    _attr_available: bool
    _mapdata: Optional[dict]
    _robot_has_map: bool
    _robot_maps: Optional[dict]
    _robot_serial: str
    _attr_unique_id: str
    _status_state: str
    _state: Optional[dict]
    _clean_time_start: Optional[timedelta]
    _clean_time_stop: Optional[timedelta]
    _clean_area: Optional[float]
    _clean_battery_start: Optional[float]
    _clean_battery_end: Optional[float]
    _clean_susp_charge_count: Optional[int]
    _clean_susp_time: Optional[timedelta]
    _clean_pause_time: Optional[timedelta]
    _clean_error_time: Optional[timedelta]
    _launched_from: Optional[str]
    _robot_boundaries: Optional[list]
    _robot_stats: Optional[dict]

    def __init__(self, neato: NeatoHub, robot: Robot, mapdata: dict, persistent_maps: dict):
        """Initialize the Neato Connected Vacuum."""
        super().__init__(robot)
        self._attr_available = neato is not None
        self._mapdata = mapdata
        self._robot_has_map = robot.has_persistent_maps
        self._robot_maps = persistent_maps
        self._robot_serial = robot.serial
        self._attr_unique_id = robot.serial
        self._status_state = None
        self._state = None
        self._clean_time_start = None
        self._clean_time_stop = None
        self._clean_area = None
        self._clean_battery_start = None
        self._clean_battery_end = None
        self._clean_susp_charge_count = None
        self._clean_susp_time = None
        self._clean_pause_time = None
        self._clean_error_time = None
        self._launched_from = None
        self._robot_boundaries = []
        self._robot_stats = None

    @property
    def extra_state_attributes(self) -> dict:
        """Return the state attributes of the vacuum cleaner."""
        data = {}
        if self._status_state is not None:
            data[ATTR_STATUS] = self._status_state
        if self._clean_time_start is not None:
            data[ATTR_CLEAN_START] = self._clean_time_start
        if self._clean_time_stop is not None:
            data[ATTR_CLEAN_STOP] = self._clean_time_stop
        if self._clean_area is not None:
            data[ATTR_CLEAN_AREA] = self._clean_area
        if self._clean_susp_charge_count is not None:
            data[ATTR_CLEAN_SUSP_COUNT] = self._clean_susp_charge_count
        if self._clean_susp_time is not None:
            data[ATTR_CLEAN_SUSP_TIME] = self._clean_susp_time
        if self._clean_pause_time is not None:
            data[ATTR_CLEAN_PAUSE_TIME] = self._clean_pause_time
        if self._clean_error_time is not None:
            data[ATTR_CLEAN_ERROR_TIME] = self._clean_error_time
        if self._clean_battery_start is not None:
            data[ATTR_CLEAN_BATTERY_START] = self._clean_battery_start
        if self._clean_battery_end is not None:
            data[ATTR_CLEAN_BATTERY_END] = self._clean_battery_end
        if self._launched_from is not None:
            data[ATTR_LAUNCHED_FROM] = self._launched_from
        return data

    @property
    def device_info(self) -> DeviceInfo:
        """Device info for neato robot."""
        device_info = self._attr_device_info
        if self._robot_stats:
            device_info['manufacturer'] = self._robot_stats['battery']['vendor']
            device_info['model'] = self._robot_stats['model']
            device_info['sw_version'] = self._robot_stats['firmware']
        return device_info

    def start(self) -> None:
        """Start cleaning or resume cleaning."""
        if self._state:
            try:
                if self._state['state'] == 1:
                    self.robot.start_cleaning()
                elif self._state['state'] == 3:
                    self.robot.resume_cleaning()
            except NeatoRobotException as ex:
                _LOGGER.error("Neato vacuum connection error for '%s': %s", self.entity_id, ex)

    def pause(self) -> None:
        """Pause the vacuum."""
        try:
            self.robot.pause_cleaning()
        except NeatoRobotException as ex:
            _LOGGER.error("Neato vacuum connection error for '%s': %s", self.entity_id, ex)

    def return_to_base(self, **kwargs) -> None:
        """Set the vacuum cleaner to return to the dock."""
        try:
            if self._attr_activity == VacuumActivity.CLEANING:
                self.robot.pause_cleaning()
            self._attr_activity = VacuumActivity.RETURNING
            self.robot.send_to_base()
        except NeatoRobotException as ex:
            _LOGGER.error("Neato vacuum connection error for '%s': %s", self.entity_id, ex)

    def stop(self, **kwargs) -> None:
        """Stop the vacuum cleaner."""
        try:
            self.robot.stop_cleaning()
        except NeatoRobotException as ex:
            _LOGGER.error("Neato vacuum connection error for '%s': %s", self.entity_id, ex)

    def locate(self, **kwargs) -> None:
        """Locate the robot by making it emit a sound."""
        try:
            self.robot.locate()
        except NeatoRobotException as ex:
            _LOGGER.error("Neato vacuum connection error for '%s': %s", self.entity_id, ex)

    def clean_spot(self, **kwargs) -> None:
        """Run a spot cleaning starting from the base."""
        try:
            self.robot.start_spot_cleaning()
        except NeatoRobotException as ex:
            _LOGGER.error("Neato vacuum connection error for '%s': %s", self.entity_id, ex)

    def neato_custom_cleaning(self, mode: int, navigation: int, category: int, zone: Optional[str] = None) -> None:
        """Zone cleaning service call."""
        boundary_id = None
        if zone is not None:
            for boundary in self._robot_boundaries:
                if zone in boundary['name']:
                    boundary_id = boundary['id']
            if boundary_id is None:
                _LOGGER.error("Zone '%s' was not found for the robot '%s'", zone, self.entity_id)
                return
            _LOGGER.debug("Start cleaning zone '%s' with robot %s", zone, self.entity_id)
        self._attr_activity = VacuumActivity.CLEANING
        try:
            self.robot.start_cleaning(mode, navigation, category, boundary_id)
        except NeatoRobotException as ex:
            _LOGGER.error("Neato vacuum connection error for '%s': %s", self.entity_id, ex)

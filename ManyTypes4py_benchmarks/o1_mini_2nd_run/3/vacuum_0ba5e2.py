"""Support for Neato Connected Vacuums."""
from __future__ import annotations
from datetime import timedelta
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from pybotvac import Robot
from pybotvac.exceptions import NeatoRobotException
import voluptuous as vol
from homeassistant.components.vacuum import (
    ATTR_STATUS,
    StateVacuumEntity,
    VacuumActivity,
    VacuumEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_MODE
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, entity_platform
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import (
    ACTION,
    ALERTS,
    ERRORS,
    MODE,
    NEATO_LOGIN,
    NEATO_MAP_DATA,
    NEATO_PERSISTENT_MAPS,
    NEATO_ROBOTS,
    SCAN_INTERVAL_MINUTES,
)
from .entity import NeatoEntity
from .hub import NeatoHub

_LOGGER: logging.Logger = logging.getLogger(__name__)

SCAN_INTERVAL: timedelta = timedelta(minutes=SCAN_INTERVAL_MINUTES)

ATTR_CLEAN_START: str = "clean_start"
ATTR_CLEAN_STOP: str = "clean_stop"
ATTR_CLEAN_AREA: str = "clean_area"
ATTR_CLEAN_BATTERY_START: str = "battery_level_at_clean_start"
ATTR_CLEAN_BATTERY_END: str = "battery_level_at_clean_end"
ATTR_CLEAN_SUSP_COUNT: str = "clean_suspension_count"
ATTR_CLEAN_SUSP_TIME: str = "clean_suspension_time"
ATTR_CLEAN_PAUSE_TIME: str = "clean_pause_time"
ATTR_CLEAN_ERROR_TIME: str = "clean_error_time"
ATTR_LAUNCHED_FROM: str = "launched_from"
ATTR_NAVIGATION: str = "navigation"
ATTR_CATEGORY: str = "category"
ATTR_ZONE: str = "zone"


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Neato vacuum with config entry."""
    neato: Optional[NeatoHub] = hass.data.get(NEATO_LOGIN)
    mapdata: Optional[Dict[str, Any]] = hass.data.get(NEATO_MAP_DATA)
    persistent_maps: Optional[Dict[str, Any]] = hass.data.get(NEATO_PERSISTENT_MAPS)
    robots: Optional[List[Robot]] = hass.data.get(NEATO_ROBOTS)
    dev: List[NeatoConnectedVacuum] = [
        NeatoConnectedVacuum(neato, robot, mapdata, persistent_maps) for robot in robots or []
    ]
    if not dev:
        return
    _LOGGER.debug("Adding vacuums %s", dev)
    async_add_entities(dev, True)
    platform = entity_platform.async_get_current_platform()
    assert platform is not None
    platform.async_register_entity_service(
        "custom_cleaning",
        {
            vol.Optional(ATTR_MODE, default=2): cv.positive_int,
            vol.Optional(ATTR_NAVIGATION, default=1): cv.positive_int,
            vol.Optional(ATTR_CATEGORY, default=4): cv.positive_int,
            vol.Optional(ATTR_ZONE): cv.string,
        },
        "neato_custom_cleaning",
    )


class NeatoConnectedVacuum(NeatoEntity, StateVacuumEntity):
    """Representation of a Neato Connected Vacuum."""

    _attr_supported_features: int = (
        VacuumEntityFeature.BATTERY
        | VacuumEntityFeature.PAUSE
        | VacuumEntityFeature.RETURN_HOME
        | VacuumEntityFeature.STOP
        | VacuumEntityFeature.START
        | VacuumEntityFeature.CLEAN_SPOT
        | VacuumEntityFeature.STATE
        | VacuumEntityFeature.MAP
        | VacuumEntityFeature.LOCATE
    )
    _attr_name: Optional[str] = None

    def __init__(
        self,
        neato: Optional[NeatoHub],
        robot: Robot,
        mapdata: Optional[Dict[str, Any]],
        persistent_maps: Optional[Dict[str, Any]],
    ) -> None:
        """Initialize the Neato Connected Vacuum."""
        super().__init__(robot)
        self._attr_available: bool = neato is not None
        self._mapdata: Optional[Dict[str, Any]] = mapdata
        self._robot_has_map: bool = self.robot.has_persistent_maps
        self._robot_maps: Optional[Dict[str, Any]] = persistent_maps
        self._robot_serial: str = self.robot.serial
        self._attr_unique_id: str = self.robot.serial
        self._status_state: Optional[str] = None
        self._state: Optional[Dict[str, Any]] = None
        self._clean_time_start: Optional[str] = None
        self._clean_time_stop: Optional[str] = None
        self._clean_area: Optional[float] = None
        self._clean_battery_start: Optional[float] = None
        self._clean_battery_end: Optional[float] = None
        self._clean_susp_charge_count: Optional[int] = None
        self._clean_susp_time: Optional[int] = None
        self._clean_pause_time: Optional[int] = None
        self._clean_error_time: Optional[int] = None
        self._launched_from: Optional[str] = None
        self._robot_boundaries: List[Dict[str, Any]] = []
        self._robot_stats: Optional[Dict[str, Any]] = None

    def update(self) -> None:
        """Update the states of Neato Vacuums."""
        _LOGGER.debug("Running Neato Vacuums update for '%s'", self.entity_id)
        try:
            if self._robot_stats is None:
                general_info = self.robot.get_general_info().json()
                self._robot_stats = general_info.get("data")
        except NeatoRobotException:
            _LOGGER.warning("Couldn't fetch robot information of %s", self.entity_id)
        try:
            self._state = self.robot.state
        except NeatoRobotException as ex:
            if self._attr_available:
                _LOGGER.error(
                    "Neato vacuum connection error for '%s': %s", self.entity_id, ex
                )
            self._state = None
            self._attr_available = False
            return
        if self._state is None:
            return
        self._attr_available = True
        _LOGGER.debug("self._state=%s", self._state)
        if "alert" in self._state:
            robot_alert: Optional[str] = ALERTS.get(self._state["alert"])
        else:
            robot_alert = None
        state_value: int = self._state.get("state")
        if state_value == 1:
            if self._state["details"].get("isCharging"):
                self._attr_activity = VacuumActivity.DOCKED
                self._status_state = "Charging"
            elif self._state["details"].get("isDocked") and not self._state["details"].get(
                "isCharging"
            ):
                self._attr_activity = VacuumActivity.DOCKED
                self._status_state = "Docked"
            else:
                self._attr_activity = VacuumActivity.IDLE
                self._status_state = "Stopped"
            if robot_alert is not None:
                self._status_state = robot_alert
        elif state_value == 2:
            if robot_alert is None:
                self._attr_activity = VacuumActivity.CLEANING
                mode_str: str = MODE.get(self._state["cleaning"]["mode"], "Unknown Mode")
                action_str: str = ACTION.get(self._state["action"], "Unknown Action")
                self._status_state = f"{mode_str} {action_str}"
                if (
                    "boundary" in self._state["cleaning"]
                    and "name" in self._state["cleaning"]["boundary"]
                ):
                    boundary_name: str = self._state["cleaning"]["boundary"]["name"]
                    self._status_state += f" {boundary_name}"
            else:
                self._status_state = robot_alert
        elif state_value == 3:
            self._attr_activity = VacuumActivity.PAUSED
            self._status_state = "Paused"
        elif state_value == 4:
            self._attr_activity = VacuumActivity.ERROR
            self._status_state = ERRORS.get(self._state.get("error"), "Unknown Error")
        self._attr_battery_level = self._state["details"].get("charge")

        if (
            self._mapdata is None
            or not self._mapdata.get(self._robot_serial, {}).get("maps", [])
        ):
            return
        mapdata: Dict[str, Any] = self._mapdata[self._robot_serial]["maps"][0]
        self._clean_time_start = mapdata.get("start_at")
        self._clean_time_stop = mapdata.get("end_at")
        self._clean_area = mapdata.get("cleaned_area")
        self._clean_susp_charge_count = mapdata.get("suspended_cleaning_charging_count")
        self._clean_susp_time = mapdata.get("time_in_suspended_cleaning")
        self._clean_pause_time = mapdata.get("time_in_pause")
        self._clean_error_time = mapdata.get("time_in_error")
        self._clean_battery_start = mapdata.get("run_charge_at_start")
        self._clean_battery_end = mapdata.get("run_charge_at_end")
        self._launched_from = mapdata.get("launched_from")
        if (
            self._robot_has_map
            and self._state
            and self._state.get("availableServices", {}).get("maps") != "basic-1"
            and self._robot_maps
        ):
            allmaps: List[Dict[str, Any]] = self._robot_maps.get(self._robot_serial, [])
            _LOGGER.debug(
                "Found the following maps for '%s': %s", self.entity_id, allmaps
            )
            self._robot_boundaries = []
            for maps in allmaps:
                try:
                    robot_boundaries: Dict[str, Any] = self.robot.get_map_boundaries(
                        maps["id"]
                    ).json()
                except NeatoRobotException as ex:
                    _LOGGER.error(
                        "Could not fetch map boundaries for '%s': %s",
                        self.entity_id,
                        ex,
                    )
                    return
                _LOGGER.debug(
                    "Boundaries for robot '%s' in map '%s': %s",
                    self.entity_id,
                    maps.get("name", "Unknown"),
                    robot_boundaries,
                )
                if "boundaries" in robot_boundaries.get("data", {}):
                    self._robot_boundaries += robot_boundaries["data"]["boundaries"]
                    _LOGGER.debug(
                        "List of boundaries for '%s': %s",
                        self.entity_id,
                        self._robot_boundaries,
                    )

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes of the vacuum cleaner."""
        data: Dict[str, Any] = {}
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
        device_info: DeviceInfo = self._attr_device_info
        if self._robot_stats:
            device_info["manufacturer"] = self._robot_stats["battery"].get("vendor")
            device_info["model"] = self._robot_stats.get("model")
            device_info["sw_version"] = self._robot_stats.get("firmware")
        return device_info

    def start(self) -> None:
        """Start cleaning or resume cleaning."""
        if self._state:
            try:
                if self._state.get("state") == 1:
                    self.robot.start_cleaning()
                elif self._state.get("state") == 3:
                    self.robot.resume_cleaning()
            except NeatoRobotException as ex:
                _LOGGER.error(
                    "Neato vacuum connection error for '%s': %s",
                    self.entity_id,
                    ex,
                )

    def pause(self) -> None:
        """Pause the vacuum."""
        try:
            self.robot.pause_cleaning()
        except NeatoRobotException as ex:
            _LOGGER.error(
                "Neato vacuum connection error for '%s': %s",
                self.entity_id,
                ex,
            )

    def return_to_base(self, **kwargs: Any) -> None:
        """Set the vacuum cleaner to return to the dock."""
        try:
            if self._attr_activity == VacuumActivity.CLEANING:
                self.robot.pause_cleaning()
            self._attr_activity = VacuumActivity.RETURNING
            self.robot.send_to_base()
        except NeatoRobotException as ex:
            _LOGGER.error(
                "Neato vacuum connection error for '%s': %s",
                self.entity_id,
                ex,
            )

    def stop(self, **kwargs: Any) -> None:
        """Stop the vacuum cleaner."""
        try:
            self.robot.stop_cleaning()
        except NeatoRobotException as ex:
            _LOGGER.error(
                "Neato vacuum connection error for '%s': %s",
                self.entity_id,
                ex,
            )

    def locate(self, **kwargs: Any) -> None:
        """Locate the robot by making it emit a sound."""
        try:
            self.robot.locate()
        except NeatoRobotException as ex:
            _LOGGER.error(
                "Neato vacuum connection error for '%s': %s",
                self.entity_id,
                ex,
            )

    def clean_spot(self, **kwargs: Any) -> None:
        """Run a spot cleaning starting from the base."""
        try:
            self.robot.start_spot_cleaning()
        except NeatoRobotException as ex:
            _LOGGER.error(
                "Neato vacuum connection error for '%s': %s",
                self.entity_id,
                ex,
            )

    def neato_custom_cleaning(
        self,
        mode: int,
        navigation: int,
        category: int,
        zone: Optional[str] = None,
    ) -> None:
        """Zone cleaning service call."""
        boundary_id: Optional[str] = None
        if zone is not None:
            for boundary in self._robot_boundaries:
                if zone in boundary.get("name", ""):
                    boundary_id = boundary.get("id")
                    break
            if boundary_id is None:
                _LOGGER.error(
                    "Zone '%s' was not found for the robot '%s'", zone, self.entity_id
                )
                return
            _LOGGER.debug(
                "Start cleaning zone '%s' with robot %s", zone, self.entity_id
            )
        self._attr_activity = VacuumActivity.CLEANING
        try:
            self.robot.start_cleaning(mode, navigation, category, boundary_id)
        except NeatoRobotException as ex:
            _LOGGER.error(
                "Neato vacuum connection error for '%s': %s",
                self.entity_id,
                ex,
            )

"""Support for Tado thermostats."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any, Optional, Union, Dict, List, Tuple

import PyTado
import voluptuous as vol

from homeassistant.components.climate import (
    FAN_AUTO,
    PRESET_AWAY,
    PRESET_HOME,
    SWING_BOTH,
    SWING_HORIZONTAL,
    SWING_OFF,
    SWING_ON,
    SWING_VERTICAL,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.const import ATTR_TEMPERATURE, PRECISION_TENTHS, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, entity_platform
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, VolDictType

from . import TadoConfigEntry
from .const import (
    CONST_EXCLUSIVE_OVERLAY_GROUP,
    CONST_FAN_AUTO,
    CONST_FAN_OFF,
    CONST_MODE_AUTO,
    CONST_MODE_COOL,
    CONST_MODE_HEAT,
    CONST_MODE_OFF,
    CONST_MODE_SMART_SCHEDULE,
    CONST_OVERLAY_MANUAL,
    CONST_OVERLAY_TADO_OPTIONS,
    DOMAIN,
    HA_TERMINATION_DURATION,
    HA_TERMINATION_TYPE,
    HA_TO_TADO_FAN_MODE_MAP,
    HA_TO_TADO_FAN_MODE_MAP_LEGACY,
    HA_TO_TADO_HVAC_MODE_MAP,
    ORDERED_KNOWN_TADO_MODES,
    PRESET_AUTO,
    SUPPORT_PRESET_AUTO,
    SUPPORT_PRESET_MANUAL,
    TADO_DEFAULT_MAX_TEMP,
    TADO_DEFAULT_MIN_TEMP,
    TADO_FANLEVEL_SETTING,
    TADO_FANSPEED_SETTING,
    TADO_HORIZONTAL_SWING_SETTING,
    TADO_HVAC_ACTION_TO_HA_HVAC_ACTION,
    TADO_MODES_WITH_NO_TEMP_SETTING,
    TADO_SWING_OFF,
    TADO_SWING_ON,
    TADO_SWING_SETTING,
    TADO_TO_HA_FAN_MODE_MAP,
    TADO_TO_HA_FAN_MODE_MAP_LEGACY,
    TADO_TO_HA_HVAC_MODE_MAP,
    TADO_TO_HA_OFFSET_MAP,
    TADO_TO_HA_SWING_MODE_MAP,
    TADO_VERTICAL_SWING_SETTING,
    TEMP_OFFSET,
    TYPE_AIR_CONDITIONING,
    TYPE_HEATING,
)
from .coordinator import TadoDataUpdateCoordinator
from .entity import TadoZoneEntity
from .helper import decide_duration, decide_overlay_mode, generate_supported_fanmodes

_LOGGER = logging.getLogger(__name__)

SERVICE_CLIMATE_TIMER = "set_climate_timer"
ATTR_TIME_PERIOD = "time_period"
ATTR_REQUESTED_OVERLAY = "requested_overlay"

CLIMATE_TIMER_SCHEMA: VolDictType = {
    vol.Required(ATTR_TEMPERATURE): vol.Coerce(float),
    vol.Exclusive(ATTR_TIME_PERIOD, CONST_EXCLUSIVE_OVERLAY_GROUP): vol.All(
        cv.time_period, cv.positive_timedelta, lambda td: td.total_seconds()
    ),
    vol.Exclusive(ATTR_REQUESTED_OVERLAY, CONST_EXCLUSIVE_OVERLAY_GROUP): vol.In(
        CONST_OVERLAY_TADO_OPTIONS
    ),
}

SERVICE_TEMP_OFFSET = "set_climate_temperature_offset"
ATTR_OFFSET = "offset"

CLIMATE_TEMP_OFFSET_SCHEMA: VolDictType = {
    vol.Required(ATTR_OFFSET, default=0): vol.Coerce(float),
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: TadoConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Tado climate platform."""
    tado = entry.runtime_data.coordinator
    entities = await _generate_entities(tado)

    platform = entity_platform.async_get_current_platform()

    platform.async_register_entity_service(
        SERVICE_CLIMATE_TIMER,
        CLIMATE_TIMER_SCHEMA,
        "set_timer",
    )

    platform.async_register_entity_service(
        SERVICE_TEMP_OFFSET,
        CLIMATE_TEMP_OFFSET_SCHEMA,
        "set_temp_offset",
    )

    async_add_entities(entities, True)


async def _generate_entities(tado: TadoDataUpdateCoordinator) -> List[TadoClimate]:
    """Create all climate entities."""
    entities = []
    for zone in tado.zones:
        if zone["type"] in [TYPE_HEATING, TYPE_AIR_CONDITIONING]:
            entity = await create_climate_entity(
                tado, zone["name"], zone["id"], zone["devices"][0]
            )
            if entity:
                entities.append(entity)
    return entities


async def create_climate_entity(
    tado: TadoDataUpdateCoordinator, name: str, zone_id: int, device_info: Dict[str, Any]
) -> Optional[TadoClimate]:
    """Create a Tado climate entity."""
    capabilities = await tado.get_capabilities(zone_id)
    _LOGGER.debug("Capabilities for zone %s: %s", zone_id, capabilities)

    zone_type = capabilities["type"]
    support_flags = (
        ClimateEntityFeature.PRESET_MODE
        | ClimateEntityFeature.TARGET_TEMPERATURE
        | ClimateEntityFeature.TURN_OFF
        | ClimateEntityFeature.TURN_ON
    )
    supported_hvac_modes = [
        TADO_TO_HA_HVAC_MODE_MAP[CONST_MODE_OFF],
        TADO_TO_HA_HVAC_MODE_MAP[CONST_MODE_SMART_SCHEDULE],
    ]
    supported_fan_modes: Optional[List[str]] = None
    supported_swing_modes: Optional[List[str]] = None
    heat_temperatures: Optional[Dict[str, Any]] = None
    cool_temperatures: Optional[Dict[str, Any]] = None

    if zone_type == TYPE_AIR_CONDITIONING:
        for mode in ORDERED_KNOWN_TADO_MODES:
            if mode not in capabilities:
                continue

            supported_hvac_modes.append(TADO_TO_HA_HVAC_MODE_MAP[mode])
            if (
                TADO_SWING_SETTING in capabilities[mode]
                or TADO_VERTICAL_SWING_SETTING in capabilities[mode]
                or TADO_VERTICAL_SWING_SETTING in capabilities[mode]
            ):
                support_flags |= ClimateEntityFeature.SWING_MODE
                supported_swing_modes = []
                if TADO_SWING_SETTING in capabilities[mode]:
                    supported_swing_modes.append(
                        TADO_TO_HA_SWING_MODE_MAP[TADO_SWING_ON]
                    )
                if TADO_VERTICAL_SWING_SETTING in capabilities[mode]:
                    supported_swing_modes.append(SWING_VERTICAL)
                if TADO_HORIZONTAL_SWING_SETTING in capabilities[mode]:
                    supported_swing_modes.append(SWING_HORIZONTAL)
                if (
                    SWING_HORIZONTAL in supported_swing_modes
                    and SWING_VERTICAL in supported_swing_modes
                ):
                    supported_swing_modes.append(SWING_BOTH)
                supported_swing_modes.append(TADO_TO_HA_SWING_MODE_MAP[TADO_SWING_OFF])

            if (
                TADO_FANSPEED_SETTING not in capabilities[mode]
                and TADO_FANLEVEL_SETTING not in capabilities[mode]
            ):
                continue

            support_flags |= ClimateEntityFeature.FAN_MODE

            if supported_fan_modes:
                continue

            if TADO_FANSPEED_SETTING in capabilities[mode]:
                supported_fan_modes = generate_supported_fanmodes(
                    TADO_TO_HA_FAN_MODE_MAP_LEGACY,
                    capabilities[mode][TADO_FANSPEED_SETTING],
                )
            else:
                supported_fan_modes = generate_supported_fanmodes(
                    TADO_TO_HA_FAN_MODE_MAP, capabilities[mode][TADO_FANLEVEL_SETTING]
                )

        cool_temperatures = capabilities[CONST_MODE_COOL]["temperatures"]
    else:
        supported_hvac_modes.append(HVACMode.HEAT)

    if CONST_MODE_HEAT in capabilities:
        heat_temperatures = capabilities[CONST_MODE_HEAT]["temperatures"]

    if heat_temperatures is None and "temperatures" in capabilities:
        heat_temperatures = capabilities["temperatures"]

    if cool_temperatures is None and heat_temperatures is None:
        _LOGGER.debug("Not adding zone %s since it has no temperatures", name)
        return None

    heat_min_temp: Optional[float] = None
    heat_max_temp: Optional[float] = None
    heat_step: Optional[float] = None
    cool_min_temp: Optional[float] = None
    cool_max_temp: Optional[float] = None
    cool_step: Optional[float] = None

    if heat_temperatures is not None:
        heat_min_temp = float(heat_temperatures["celsius"]["min"])
        heat_max_temp = float(heat_temperatures["celsius"]["max"])
        heat_step = heat_temperatures["celsius"].get("step", PRECISION_TENTHS)

    if cool_temperatures is not None:
        cool_min_temp = float(cool_temperatures["celsius"]["min"])
        cool_max_temp = float(cool_temperatures["celsius"]["max"])
        cool_step = cool_temperatures["celsius"].get("step", PRECISION_TENTHS)

    auto_geofencing_supported = await tado.get_auto_geofencing_supported()

    return TadoClimate(
        tado,
        name,
        zone_id,
        zone_type,
        supported_hvac_modes,
        support_flags,
        device_info,
        capabilities,
        auto_geofencing_supported,
        heat_min_temp,
        heat_max_temp,
        heat_step,
        cool_min_temp,
        cool_max_temp,
        cool_step,
        supported_fan_modes,
        supported_swing_modes,
    )


class TadoClimate(TadoZoneEntity, ClimateEntity):
    """Representation of a Tado climate entity."""

    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _attr_name = None
    _attr_translation_key = DOMAIN
    _available = False

    def __init__(
        self,
        coordinator: TadoDataUpdateCoordinator,
        zone_name: str,
        zone_id: int,
        zone_type: str,
        supported_hvac_modes: List[HVACMode],
        support_flags: ClimateEntityFeature,
        device_info: Dict[str, str],
        capabilities: Dict[str, Any],
        auto_geofencing_supported: bool,
        heat_min_temp: Optional[float] = None,
        heat_max_temp: Optional[float] = None,
        heat_step: Optional[float] = None,
        cool_min_temp: Optional[float] = None,
        cool_max_temp: Optional[float] = None,
        cool_step: Optional[float] = None,
        supported_fan_modes: Optional[List[str]] = None,
        supported_swing_modes: Optional[List[str]] = None,
    ) -> None:
        """Initialize of Tado climate entity."""
        self._tado = coordinator
        super().__init__(zone_name, coordinator.home_id, zone_id, coordinator)

        self.zone_id = zone_id
        self.zone_type = zone_type

        self._attr_unique_id = f"{zone_type} {zone_id} {coordinator.home_id}"

        self._device_info = device_info
        self._device_id = self._device_info["shortSerialNo"]

        self._ac_device = zone_type == TYPE_AIR_CONDITIONING
        self._attr_hvac_modes = supported_hvac_modes
        self._attr_fan_modes = supported_fan_modes
        self._attr_supported_features = support_flags

        self._cur_temp: Optional[float] = None
        self._cur_humidity: Optional[int] = None
        self._attr_swing_modes = supported_swing_modes

        self._heat_min_temp = heat_min_temp
        self._heat_max_temp = heat_max_temp
        self._heat_step = heat_step

        self._cool_min_temp = cool_min_temp
        self._cool_max_temp = cool_max_temp
        self._cool_step = cool_step

        self._target_temp: Optional[float] = None

        self._current_tado_fan_speed = CONST_FAN_OFF
        self._current_tado_fan_level = CONST_FAN_OFF
        self._current_tado_hvac_mode = CONST_MODE_OFF
        self._current_tado_hvac_action = HVACAction.OFF
        self._current_tado_swing_mode = TADO_SWING_OFF
        self._current_tado_vertical_swing = TADO_SWING_OFF
        self._current_tado_horizontal_swing = TADO_SWING_OFF

        self._current_tado_capabilities = capabilities
        self._auto_geofencing_supported = auto_geofencing_supported

        self._tado_zone_data: Dict[str, Any] = {}
        self._tado_geofence_data: Optional[Dict[str, str]] = None

        self._tado_zone_temp_offset: Dict[str, Any] = {}

        self._async_update_zone_data()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self._async_update_zone_data()
        super()._handle_coordinator_update()

    @callback
    def _async_update_zone_data(self) -> None:
        """Load tado data into zone."""
        self._tado_geofence_data = self._tado.data["geofence"]
        self._tado_zone_data = self._tado.data["zone"][self.zone_id]

        for offset_key, attr in TADO_TO_HA_OFFSET_MAP.items():
            if (
                self._device_id in self._tado.data["device"]
                and offset_key
                in self._tado.data["device"][self._device_id][TEMP_OFFSET]
            ):
                self._tado_zone_temp_offset[attr] = self._tado.data["device"][
                    self._device_id
                ][TEMP_OFFSET][offset_key]

        self._current_tado_hvac_mode = self._tado_zone_data.current_hvac_mode
        self._current_tado_hvac_action = self._tado_zone_data.current_hvac_action

        if self._is_valid_setting_for_hvac_mode(TADO_FANLEVEL_SETTING):
            self._current_tado_fan_level = self._tado_zone_data.current_fan_level
        if self._is_valid_setting_for_hvac_mode(TADO_FANSPEED_SETTING):
            self._current_tado_fan_speed = self._tado_zone_data.current_fan_speed
        if self._is_valid_setting_for_hvac_mode(TADO_SWING_SETTING):
            self._current_tado_swing_mode = self._tado_zone_data.current_swing_mode
        if self._is_valid_setting_for_hvac_mode(TADO_VERTICAL_SWING_SETTING):
            self._current_tado_vertical_swing = (
                self._tado_zone_data.current_vertical_swing_mode
            )
        if self._is_valid_setting_for_hvac_mode(TADO_HORIZONTAL_SWING_SETTING):
            self._current_tado_horizontal_swing = (
                self._tado_zone_data.current_horizontal_swing_mode
            )

    @callback
    def _async_update_zone_callback(self) -> None:
        """Load tado data and update state."""
        self._async_update_zone_data()

    @property
    def current_humidity(self) -> Optional[int]:
        """Return the current humidity."""
        return self._tado_zone_data.current_humidity

    @property
    def current_temperature(self) -> Optional[float]:
        """Return the sensor temperature."""
        return self._tado_zone_data.current_temp

    @property
    def hvac_mode(self) -> HVACMode:
        """Return hvac operation ie. heat, cool mode."""
        return TADO_TO_HA_HVAC_MODE_MAP.get(self._current_tado_hvac_mode, HVACMode.OFF)

    @property
    def hvac_action(self) -> HVACAction:
        """Return the current running hvac operation if supported."""
        return TADO_HVAC_ACTION_TO_HA_HVAC_ACTION.get(
            self._tado_zone_data.current_hvac_action, HVACAction.OFF
        )

    @property
    def fan_mode(self) -> Optional[str]:
        """Return the fan setting."""
        if self._ac_device:
            if self._is_valid_setting_for_hvac_mode(TADO_FANSPEED_SETTING):
                return TADO_TO_HA_FAN_MODE_MAP_LEGACY.get(
                    self._current_tado_fan_speed, FAN_AUTO
                )
            if self._is_valid_setting_for_hvac_mode(TADO_FANLEVEL_SETTING):
                return TADO_TO_HA_FAN_MODE_MAP.get(
                    self._current_tado_fan_level, FAN_AUTO
                )
            return FAN_AUTO
        return None

    async def async_set_fan_mode(self, fan_mode: str) -> None:
        """Turn fan on/off."""
        if self._is_valid_setting_for_hvac_mode(TADO_FANSPEED_SETTING):
            await self._control_hvac
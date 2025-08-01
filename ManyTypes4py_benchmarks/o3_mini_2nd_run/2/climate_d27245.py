from __future__ import annotations
from collections.abc import Mapping
import logging
from typing import Any, Optional, List, Dict, Callable, Coroutine
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
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import VolDictType
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

SERVICE_CLIMATE_TIMER = 'set_climate_timer'
ATTR_TIME_PERIOD = 'time_period'
ATTR_REQUESTED_OVERLAY = 'requested_overlay'
CLIMATE_TIMER_SCHEMA: Dict[str, Any] = {
    vol.Required(ATTR_TEMPERATURE): vol.Coerce(float),
    vol.Exclusive(ATTR_TIME_PERIOD, CONST_EXCLUSIVE_OVERLAY_GROUP): vol.All(
        cv.time_period, cv.positive_timedelta, lambda td: td.total_seconds()
    ),
    vol.Exclusive(ATTR_REQUESTED_OVERLAY, CONST_EXCLUSIVE_OVERLAY_GROUP): vol.In(CONST_OVERLAY_TADO_OPTIONS),
}
SERVICE_TEMP_OFFSET = 'set_climate_temperature_offset'
ATTR_OFFSET = 'offset'
CLIMATE_TEMP_OFFSET_SCHEMA: Dict[str, Any] = {vol.Required(ATTR_OFFSET, default=0): vol.Coerce(float)}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: TadoConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Tado climate platform."""
    tado: TadoDataUpdateCoordinator = entry.runtime_data.coordinator
    entities: List[ClimateEntity] = await _generate_entities(tado)
    platform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(SERVICE_CLIMATE_TIMER, CLIMATE_TIMER_SCHEMA, 'set_timer')
    platform.async_register_entity_service(SERVICE_TEMP_OFFSET, CLIMATE_TEMP_OFFSET_SCHEMA, 'set_temp_offset')
    async_add_entities(entities, True)


async def _generate_entities(tado: TadoDataUpdateCoordinator) -> List[ClimateEntity]:
    """Create all climate entities."""
    entities: List[ClimateEntity] = []
    for zone in tado.zones:
        if zone['type'] in [TYPE_HEATING, TYPE_AIR_CONDITIONING]:
            entity = await create_climate_entity(tado, zone['name'], zone['id'], zone['devices'][0])
            if entity:
                entities.append(entity)
    return entities


async def create_climate_entity(
    tado: TadoDataUpdateCoordinator,
    name: str,
    zone_id: Any,
    device_info: Dict[str, Any],
) -> Optional[TadoClimate]:
    """Create a Tado climate entity."""
    capabilities: Dict[str, Any] = await tado.get_capabilities(zone_id)
    _LOGGER.debug('Capabilities for zone %s: %s', zone_id, capabilities)
    zone_type: str = capabilities['type']
    support_flags: int = ClimateEntityFeature.PRESET_MODE | ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.TURN_OFF | ClimateEntityFeature.TURN_ON
    supported_hvac_modes: List[str] = [TADO_TO_HA_HVAC_MODE_MAP[CONST_MODE_OFF], TADO_TO_HA_HVAC_MODE_MAP[CONST_MODE_SMART_SCHEDULE]]
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
                or TADO_HORIZONTAL_SWING_SETTING in capabilities[mode]
            ):
                support_flags |= ClimateEntityFeature.SWING_MODE
                supported_swing_modes = []
                if TADO_SWING_SETTING in capabilities[mode]:
                    supported_swing_modes.append(TADO_TO_HA_SWING_MODE_MAP[TADO_SWING_ON])
                if TADO_VERTICAL_SWING_SETTING in capabilities[mode]:
                    supported_swing_modes.append(SWING_VERTICAL)
                if TADO_HORIZONTAL_SWING_SETTING in capabilities[mode]:
                    supported_swing_modes.append(SWING_HORIZONTAL)
                if SWING_HORIZONTAL in supported_swing_modes and SWING_VERTICAL in supported_swing_modes:
                    supported_swing_modes.append(SWING_BOTH)
                supported_swing_modes.append(TADO_TO_HA_SWING_MODE_MAP[TADO_SWING_OFF])
            if TADO_FANSPEED_SETTING not in capabilities[mode] and TADO_FANLEVEL_SETTING not in capabilities[mode]:
                continue
            support_flags |= ClimateEntityFeature.FAN_MODE
            if supported_fan_modes:
                continue
            if TADO_FANSPEED_SETTING in capabilities[mode]:
                supported_fan_modes = generate_supported_fanmodes(TADO_TO_HA_FAN_MODE_MAP_LEGACY, capabilities[mode][TADO_FANSPEED_SETTING])
            else:
                supported_fan_modes = generate_supported_fanmodes(TADO_TO_HA_FAN_MODE_MAP, capabilities[mode][TADO_FANLEVEL_SETTING])
        cool_temperatures = capabilities[CONST_MODE_COOL]['temperatures']
    else:
        supported_hvac_modes.append(HVACMode.HEAT)
    if CONST_MODE_HEAT in capabilities:
        heat_temperatures = capabilities[CONST_MODE_HEAT]['temperatures']
    if heat_temperatures is None and 'temperatures' in capabilities:
        heat_temperatures = capabilities['temperatures']
    if cool_temperatures is None and heat_temperatures is None:
        _LOGGER.debug('Not adding zone %s since it has no temperatures', name)
        return None
    heat_min_temp: Optional[float] = None
    heat_max_temp: Optional[float] = None
    heat_step: Optional[float] = None
    cool_min_temp: Optional[float] = None
    cool_max_temp: Optional[float] = None
    cool_step: Optional[float] = None
    if heat_temperatures is not None:
        heat_min_temp = float(heat_temperatures['celsius']['min'])
        heat_max_temp = float(heat_temperatures['celsius']['max'])
        heat_step = heat_temperatures['celsius'].get('step', PRECISION_TENTHS)
    if cool_temperatures is not None:
        cool_min_temp = float(cool_temperatures['celsius']['min'])
        cool_max_temp = float(cool_temperatures['celsius']['max'])
        cool_step = cool_temperatures['celsius'].get('step', PRECISION_TENTHS)
    auto_geofencing_supported: bool = await tado.get_auto_geofencing_supported()
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
    _attr_name: Optional[str] = None
    _attr_translation_key: str = DOMAIN
    _available: bool = False

    def __init__(
        self,
        coordinator: TadoDataUpdateCoordinator,
        zone_name: str,
        zone_id: Any,
        zone_type: str,
        supported_hvac_modes: List[str],
        support_flags: int,
        device_info: Dict[str, Any],
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
        self._tado: TadoDataUpdateCoordinator = coordinator
        super().__init__(zone_name, coordinator.home_id, zone_id, coordinator)
        self.zone_id: Any = zone_id
        self.zone_type: str = zone_type
        self._attr_unique_id: str = f'{zone_type} {zone_id} {coordinator.home_id}'
        self._device_info: Dict[str, Any] = device_info
        self._device_id: str = self._device_info['shortSerialNo']
        self._ac_device: bool = zone_type == TYPE_AIR_CONDITIONING
        self._attr_hvac_modes: List[str] = supported_hvac_modes
        self._attr_fan_modes: Optional[List[str]] = supported_fan_modes
        self._attr_supported_features: int = support_flags
        self._cur_temp: Optional[float] = None
        self._cur_humidity: Optional[float] = None
        self._attr_swing_modes: Optional[List[str]] = supported_swing_modes
        self._heat_min_temp: Optional[float] = heat_min_temp
        self._heat_max_temp: Optional[float] = heat_max_temp
        self._heat_step: Optional[float] = heat_step
        self._cool_min_temp: Optional[float] = cool_min_temp
        self._cool_max_temp: Optional[float] = cool_max_temp
        self._cool_step: Optional[float] = cool_step
        self._target_temp: Optional[float] = None
        self._current_tado_fan_speed: str = CONST_FAN_OFF
        self._current_tado_fan_level: str = CONST_FAN_OFF
        self._current_tado_hvac_mode: str = CONST_MODE_OFF
        self._current_tado_hvac_action: HVACAction = HVACAction.OFF
        self._current_tado_swing_mode: str = TADO_SWING_OFF
        self._current_tado_vertical_swing: str = TADO_SWING_OFF
        self._current_tado_horizontal_swing: str = TADO_SWING_OFF
        self._current_tado_capabilities: Dict[str, Any] = capabilities
        self._auto_geofencing_supported: bool = auto_geofencing_supported
        self._tado_zone_data: Dict[str, Any] = {}
        self._tado_geofence_data: Optional[Dict[str, Any]] = None
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
        self._tado_geofence_data = self._tado.data['geofence']
        self._tado_zone_data = self._tado.data['zone'][self.zone_id]
        for offset_key, attr in TADO_TO_HA_OFFSET_MAP.items():
            if (
                self._device_id in self._tado.data['device']
                and offset_key in self._tado.data['device'][self._device_id][TEMP_OFFSET]
            ):
                self._tado_zone_temp_offset[attr] = self._tado.data['device'][self._device_id][TEMP_OFFSET][offset_key]
        self._current_tado_hvac_mode = self._tado_zone_data.current_hvac_mode
        self._current_tado_hvac_action = self._tado_zone_data.current_hvac_action
        if self._is_valid_setting_for_hvac_mode(TADO_FANLEVEL_SETTING):
            self._current_tado_fan_level = self._tado_zone_data.current_fan_level
        if self._is_valid_setting_for_hvac_mode(TADO_FANSPEED_SETTING):
            self._current_tado_fan_speed = self._tado_zone_data.current_fan_speed
        if self._is_valid_setting_for_hvac_mode(TADO_SWING_SETTING):
            self._current_tado_swing_mode = self._tado_zone_data.current_swing_mode
        if self._is_valid_setting_for_hvac_mode(TADO_VERTICAL_SWING_SETTING):
            self._current_tado_vertical_swing = self._tado_zone_data.current_vertical_swing_mode
        if self._is_valid_setting_for_hvac_mode(TADO_HORIZONTAL_SWING_SETTING):
            self._current_tado_horizontal_swing = self._tado_zone_data.current_horizontal_swing_mode

    @callback
    def _async_update_zone_callback(self) -> None:
        """Load tado data and update state."""
        self._async_update_zone_data()

    @property
    def current_humidity(self) -> Optional[float]:
        """Return the current humidity."""
        return self._tado_zone_data.current_humidity

    @property
    def current_temperature(self) -> Optional[float]:
        """Return the sensor temperature."""
        return self._tado_zone_data.current_temp

    @property
    def hvac_mode(self) -> str:
        """Return hvac operation ie. heat, cool mode.

        Need to be one of HVAC_MODE_*.
        """
        return TADO_TO_HA_HVAC_MODE_MAP.get(self._current_tado_hvac_mode, HVACMode.OFF)

    @property
    def hvac_action(self) -> str:
        """Return the current running hvac operation if supported.

        Need to be one of CURRENT_HVAC_*.
        """
        return TADO_HVAC_ACTION_TO_HA_HVAC_ACTION.get(self._tado_zone_data.current_hvac_action, HVACAction.OFF)

    @property
    def fan_mode(self) -> Optional[str]:
        """Return the fan setting."""
        if self._ac_device:
            if self._is_valid_setting_for_hvac_mode(TADO_FANSPEED_SETTING):
                return TADO_TO_HA_FAN_MODE_MAP_LEGACY.get(self._current_tado_fan_speed, FAN_AUTO)
            if self._is_valid_setting_for_hvac_mode(TADO_FANLEVEL_SETTING):
                return TADO_TO_HA_FAN_MODE_MAP.get(self._current_tado_fan_level, FAN_AUTO)
            return FAN_AUTO
        return None

    async def async_set_fan_mode(self, fan_mode: str) -> None:
        """Turn fan on/off."""
        if self._is_valid_setting_for_hvac_mode(TADO_FANSPEED_SETTING):
            await self._control_hvac(fan_mode=HA_TO_TADO_FAN_MODE_MAP_LEGACY[fan_mode])
        elif self._is_valid_setting_for_hvac_mode(TADO_FANLEVEL_SETTING):
            await self._control_hvac(fan_mode=HA_TO_TADO_FAN_MODE_MAP[fan_mode])
        await self.coordinator.async_request_refresh()

    @property
    def preset_mode(self) -> str:
        """Return the current preset mode (home, away or auto)."""
        if self._tado_geofence_data is not None and 'presenceLocked' in self._tado_geofence_data:
            if not self._tado_geofence_data['presenceLocked']:
                return PRESET_AUTO
        if self._tado_zone_data.is_away:
            return PRESET_AWAY
        return PRESET_HOME

    @property
    def preset_modes(self) -> List[str]:
        """Return a list of available preset modes."""
        if self._auto_geofencing_supported:
            return SUPPORT_PRESET_AUTO
        return SUPPORT_PRESET_MANUAL

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set new preset mode."""
        await self._tado.set_presence(preset_mode)
        await self.coordinator.async_request_refresh()

    @property
    def target_temperature_step(self) -> Optional[float]:
        """Return the supported step of target temperature."""
        if self._tado_zone_data.current_hvac_mode == CONST_MODE_COOL:
            return self._cool_step or self._heat_step
        return self._heat_step or self._cool_step

    @property
    def target_temperature(self) -> Optional[float]:
        """Return the temperature we try to reach."""
        return self._tado_zone_data.target_temp or self._tado_zone_data.current_temp

    async def set_timer(self, temperature: float, time_period: Optional[float] = None, requested_overlay: Optional[str] = None) -> None:
        """Set the timer on the entity, and temperature if supported."""
        await self._control_hvac(
            hvac_mode=CONST_MODE_HEAT,
            target_temp=temperature,
            duration=time_period,
            overlay_mode=requested_overlay,
        )
        await self.coordinator.async_request_refresh()

    async def set_temp_offset(self, offset: float) -> None:
        """Set offset on the entity."""
        _LOGGER.debug('Setting temperature offset for device %s setting to (%.1f)', self._device_id, offset)
        await self._tado.set_temperature_offset(self._device_id, offset)
        await self.coordinator.async_request_refresh()

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        temperature: Optional[float] = kwargs.get(ATTR_TEMPERATURE)
        if temperature is None:
            return
        if self._current_tado_hvac_mode not in (CONST_MODE_OFF, CONST_MODE_AUTO, CONST_MODE_SMART_SCHEDULE):
            await self._control_hvac(target_temp=temperature)
            await self.coordinator.async_request_refresh()
            return
        new_hvac_mode: str = CONST_MODE_COOL if self._ac_device else CONST_MODE_HEAT
        await self._control_hvac(target_temp=temperature, hvac_mode=new_hvac_mode)
        await self.coordinator.async_request_refresh()

    async def async_set_hvac_mode(self, hvac_mode: str) -> None:
        """Set new target hvac mode."""
        _LOGGER.debug('Setting new hvac mode for device %s to %s', self._device_id, hvac_mode)
        await self._control_hvac(hvac_mode=HA_TO_TADO_HVAC_MODE_MAP[hvac_mode])
        await self.coordinator.async_request_refresh()

    @property
    def available(self) -> bool:
        """Return if the device is available."""
        return self._tado_zone_data.available

    @property
    def min_temp(self) -> float:
        """Return the minimum temperature."""
        if self._current_tado_hvac_mode == CONST_MODE_COOL and self._cool_min_temp is not None:
            return self._cool_min_temp
        if self._heat_min_temp is not None:
            return self._heat_min_temp
        return TADO_DEFAULT_MIN_TEMP

    @property
    def max_temp(self) -> float:
        """Return the maximum temperature."""
        if self._current_tado_hvac_mode == CONST_MODE_HEAT and self._heat_max_temp is not None:
            return self._heat_max_temp
        if self._heat_max_temp is not None:
            return self._heat_max_temp
        return TADO_DEFAULT_MAX_TEMP

    @property
    def swing_mode(self) -> str:
        """Active swing mode for the device."""
        swing_modes_tuple = (
            self._current_tado_swing_mode,
            self._current_tado_vertical_swing,
            self._current_tado_horizontal_swing,
        )
        if swing_modes_tuple == (TADO_SWING_OFF, TADO_SWING_OFF, TADO_SWING_OFF):
            return TADO_TO_HA_SWING_MODE_MAP[TADO_SWING_OFF]
        if swing_modes_tuple == (TADO_SWING_ON, TADO_SWING_OFF, TADO_SWING_OFF):
            return TADO_TO_HA_SWING_MODE_MAP[TADO_SWING_ON]
        if swing_modes_tuple == (TADO_SWING_OFF, TADO_SWING_ON, TADO_SWING_OFF):
            return SWING_VERTICAL
        if swing_modes_tuple == (TADO_SWING_OFF, TADO_SWING_OFF, TADO_SWING_ON):
            return SWING_HORIZONTAL
        if swing_modes_tuple == (TADO_SWING_OFF, TADO_SWING_ON, TADO_SWING_ON):
            return SWING_BOTH
        return TADO_TO_HA_SWING_MODE_MAP[TADO_SWING_OFF]

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return temperature offset."""
        state_attr: Dict[str, Any] = self._tado_zone_temp_offset
        state_attr[HA_TERMINATION_TYPE] = self._tado_zone_data.default_overlay_termination_type
        state_attr[HA_TERMINATION_DURATION] = self._tado_zone_data.default_overlay_termination_duration
        return state_attr

    async def async_set_swing_mode(self, swing_mode: str) -> None:
        """Set swing modes for the device."""
        vertical_swing: Optional[str] = None
        horizontal_swing: Optional[str] = None
        swing: Optional[str] = None
        if self._attr_swing_modes is None:
            return
        if swing_mode == SWING_OFF:
            if self._is_valid_setting_for_hvac_mode(TADO_SWING_SETTING):
                swing = TADO_SWING_OFF
            if self._is_valid_setting_for_hvac_mode(TADO_HORIZONTAL_SWING_SETTING):
                horizontal_swing = TADO_SWING_OFF
            if self._is_valid_setting_for_hvac_mode(TADO_VERTICAL_SWING_SETTING):
                vertical_swing = TADO_SWING_OFF
        if swing_mode == SWING_ON:
            swing = TADO_SWING_ON
        if swing_mode == SWING_VERTICAL:
            if self._is_valid_setting_for_hvac_mode(TADO_VERTICAL_SWING_SETTING):
                vertical_swing = TADO_SWING_ON
            if self._is_valid_setting_for_hvac_mode(TADO_HORIZONTAL_SWING_SETTING):
                horizontal_swing = TADO_SWING_OFF
        if swing_mode == SWING_HORIZONTAL:
            if self._is_valid_setting_for_hvac_mode(TADO_VERTICAL_SWING_SETTING):
                vertical_swing = TADO_SWING_OFF
            if self._is_valid_setting_for_hvac_mode(TADO_HORIZONTAL_SWING_SETTING):
                horizontal_swing = TADO_SWING_ON
        if swing_mode == SWING_BOTH:
            if self._is_valid_setting_for_hvac_mode(TADO_VERTICAL_SWING_SETTING):
                vertical_swing = TADO_SWING_ON
            if self._is_valid_setting_for_hvac_mode(TADO_HORIZONTAL_SWING_SETTING):
                horizontal_swing = TADO_SWING_ON
        await self._control_hvac(swing_mode=swing, vertical_swing=vertical_swing, horizontal_swing=horizontal_swing)
        await self.coordinator.async_request_refresh()

    def _normalize_target_temp_for_hvac_mode(self) -> None:
        def adjust_temp(min_temp: Optional[float], max_temp: Optional[float]) -> float:
            if max_temp is not None and self._target_temp is not None and self._target_temp > max_temp:
                return max_temp
            if min_temp is not None and self._target_temp is not None and self._target_temp < min_temp:
                return min_temp
            return self._target_temp  # type: ignore

        if self._target_temp is None:
            self._target_temp = self._tado_zone_data.current_temp
        elif self._current_tado_hvac_mode == CONST_MODE_COOL:
            self._target_temp = adjust_temp(self._cool_min_temp, self._cool_max_temp)
        elif self._current_tado_hvac_mode == CONST_MODE_HEAT:
            self._target_temp = adjust_temp(self._heat_min_temp, self._heat_max_temp)

    async def _control_hvac(
        self,
        hvac_mode: Optional[str] = None,
        target_temp: Optional[float] = None,
        fan_mode: Optional[str] = None,
        swing_mode: Optional[str] = None,
        duration: Optional[float] = None,
        overlay_mode: Optional[str] = None,
        vertical_swing: Optional[str] = None,
        horizontal_swing: Optional[str] = None,
    ) -> None:
        """Send new target temperature to Tado."""
        if hvac_mode:
            self._current_tado_hvac_mode = hvac_mode
        if target_temp:
            self._target_temp = target_temp
        if fan_mode:
            if self._is_valid_setting_for_hvac_mode(TADO_FANSPEED_SETTING):
                self._current_tado_fan_speed = fan_mode
            if self._is_valid_setting_for_hvac_mode(TADO_FANLEVEL_SETTING):
                self._current_tado_fan_level = fan_mode
        if swing_mode:
            self._current_tado_swing_mode = swing_mode
        if vertical_swing:
            self._current_tado_vertical_swing = vertical_swing
        if horizontal_swing:
            self._current_tado_horizontal_swing = horizontal_swing
        self._normalize_target_temp_for_hvac_mode()
        if self._current_tado_fan_speed == CONST_FAN_OFF and self._current_tado_hvac_mode != CONST_MODE_OFF:
            self._current_tado_fan_speed = CONST_FAN_AUTO
        if self._current_tado_hvac_mode == CONST_MODE_OFF:
            _LOGGER.debug('Switching to OFF for zone %s (%d)', self.zone_name, self.zone_id)
            await self._tado.set_zone_off(self.zone_id, CONST_OVERLAY_MANUAL, self.zone_type)
            return
        if self._current_tado_hvac_mode == CONST_MODE_SMART_SCHEDULE:
            _LOGGER.debug('Switching to SMART_SCHEDULE for zone %s (%d)', self.zone_name, self.zone_id)
            await self._tado.reset_zone_overlay(self.zone_id)
            return
        overlay_mode = decide_overlay_mode(coordinator=self._tado, duration=duration, overlay_mode=overlay_mode, zone_id=self.zone_id)
        duration = decide_duration(coordinator=self._tado, duration=duration, zone_id=self.zone_id, overlay_mode=overlay_mode)
        _LOGGER.debug(
            'Switching to %s for zone %s (%d) with temperature %s °C and duration %s using overlay %s',
            self._current_tado_hvac_mode,
            self.zone_name,
            self.zone_id,
            self._target_temp,
            duration,
            overlay_mode,
        )
        temperature_to_send: Optional[float] = self._target_temp
        if self._current_tado_hvac_mode in TADO_MODES_WITH_NO_TEMP_SETTING:
            temperature_to_send = None
        fan_speed: Optional[str] = None
        fan_level: Optional[str] = None
        if self.supported_features & ClimateEntityFeature.FAN_MODE:
            if self._is_current_setting_supported_by_current_hvac_mode(TADO_FANSPEED_SETTING, self._current_tado_fan_speed):
                fan_speed = self._current_tado_fan_speed
            if self._is_current_setting_supported_by_current_hvac_mode(TADO_FANLEVEL_SETTING, self._current_tado_fan_level):
                fan_level = self._current_tado_fan_level
        swing: Optional[str] = None
        vertical_swing_out: Optional[str] = None
        horizontal_swing_out: Optional[str] = None
        if self.supported_features & ClimateEntityFeature.SWING_MODE and self._attr_swing_modes is not None:
            if self._is_current_setting_supported_by_current_hvac_mode(TADO_VERTICAL_SWING_SETTING, self._current_tado_vertical_swing):
                vertical_swing_out = self._current_tado_vertical_swing
            if self._is_current_setting_supported_by_current_hvac_mode(TADO_HORIZONTAL_SWING_SETTING, self._current_tado_horizontal_swing):
                horizontal_swing_out = self._current_tado_horizontal_swing
            if self._is_current_setting_supported_by_current_hvac_mode(TADO_SWING_SETTING, self._current_tado_swing_mode):
                swing = self._current_tado_swing_mode
        await self._tado.set_zone_overlay(
            zone_id=self.zone_id,
            overlay_mode=overlay_mode,
            temperature=temperature_to_send,
            duration=duration,
            device_type=self.zone_type,
            mode=self._current_tado_hvac_mode,
            fan_speed=fan_speed,
            swing=swing,
            fan_level=fan_level,
            vertical_swing=vertical_swing_out,
            horizontal_swing=horizontal_swing_out,
        )

    def _is_valid_setting_for_hvac_mode(self, setting: str) -> bool:
        """Determine if a setting is valid for the current HVAC mode."""
        capabilities: Any = self._current_tado_capabilities.get(self._current_tado_hvac_mode, {})
        if isinstance(capabilities, dict):
            return capabilities.get(setting) is not None
        return False

    def _is_current_setting_supported_by_current_hvac_mode(self, setting: str, current_state: str) -> bool:
        """Determine if the current setting is supported by the current HVAC mode."""
        capabilities: Any = self._current_tado_capabilities.get(self._current_tado_hvac_mode, {})
        if isinstance(capabilities, dict) and self._is_valid_setting_for_hvac_mode(setting):
            return current_state in capabilities.get(setting, [])
        return False

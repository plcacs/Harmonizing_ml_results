from __future__ import annotations
import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta
import logging
import math
from typing import Any, Optional, cast

import voluptuous as vol

from homeassistant.components.climate import (
    ATTR_PRESET_MODE,
    PLATFORM_SCHEMA as CLIMATE_PLATFORM_SCHEMA,
    PRESET_NONE,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_TEMPERATURE,
    CONF_NAME,
    CONF_UNIQUE_ID,
    EVENT_HOMEASSISTANT_START,
    PRECISION_HALVES,
    PRECISION_TENTHS,
    PRECISION_WHOLE,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfTemperature,
)
from homeassistant.core import (
    DOMAIN as HOMEASSISTANT_DOMAIN,
    CoreState,
    Event,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.exceptions import ConditionError
from homeassistant.helpers import condition, config_validation as cv
from homeassistant.helpers.device import async_device_info_to_link_from_entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType, VolDictType

from .const import CONF_AC_MODE, CONF_COLD_TOLERANCE, CONF_HEATER, CONF_HOT_TOLERANCE, CONF_MAX_TEMP, CONF_MIN_DUR, CONF_MIN_TEMP, CONF_PRESETS, CONF_SENSOR, DEFAULT_TOLERANCE, DOMAIN, PLATFORMS

_LOGGER = logging.getLogger(__name__)

DEFAULT_NAME = 'Generic Thermostat'
CONF_INITIAL_HVAC_MODE = 'initial_hvac_mode'
CONF_KEEP_ALIVE = 'keep_alive'
CONF_PRECISION = 'precision'
CONF_TARGET_TEMP = 'target_temp'
CONF_TEMP_STEP = 'target_temp_step'
PRESETS_SCHEMA = {vol.Optional(v): vol.Coerce(float) for v in CONF_PRESETS.values()}
PLATFORM_SCHEMA_COMMON = vol.Schema({
    vol.Required(CONF_HEATER): cv.entity_id,
    vol.Required(CONF_SENSOR): cv.entity_id,
    vol.Optional(CONF_AC_MODE): cv.boolean,
    vol.Optional(CONF_MAX_TEMP): vol.Coerce(float),
    vol.Optional(CONF_MIN_DUR): cv.positive_time_period,
    vol.Optional(CONF_MIN_TEMP): vol.Coerce(float),
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_COLD_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
    vol.Optional(CONF_HOT_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
    vol.Optional(CONF_TARGET_TEMP): vol.Coerce(float),
    vol.Optional(CONF_KEEP_ALIVE): cv.positive_time_period,
    vol.Optional(CONF_INITIAL_HVAC_MODE): vol.In([HVACMode.COOL, HVACMode.HEAT, HVACMode.OFF]),
    vol.Optional(CONF_PRECISION): vol.All(vol.Coerce(float), vol.In([PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE])),
    vol.Optional(CONF_TEMP_STEP): vol.All(vol.In([PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE])),
    vol.Optional(CONF_UNIQUE_ID): cv.string,
    **PRESETS_SCHEMA
})
PLATFORM_SCHEMA = CLIMATE_PLATFORM_SCHEMA.extend(PLATFORM_SCHEMA_COMMON.schema)


async def async_setup_entry(
    hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    await _async_setup_config(hass, PLATFORM_SCHEMA_COMMON(dict(config_entry.options)), config_entry.entry_id, async_add_entities)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)
    await _async_setup_config(hass, config, config.get(CONF_UNIQUE_ID), async_add_entities)


async def _async_setup_config(
    hass: HomeAssistant, config: VolDictType, unique_id: Optional[str], async_add_entities: AddEntitiesCallback
) -> None:
    name: str = config[CONF_NAME]
    heater_entity_id: str = config[CONF_HEATER]
    sensor_entity_id: str = config[CONF_SENSOR]
    min_temp: Optional[float] = config.get(CONF_MIN_TEMP)
    max_temp: Optional[float] = config.get(CONF_MAX_TEMP)
    target_temp: Optional[float] = config.get(CONF_TARGET_TEMP)
    ac_mode: Optional[bool] = config.get(CONF_AC_MODE)
    min_cycle_duration: Optional[timedelta] = config.get(CONF_MIN_DUR)
    cold_tolerance: float = config[CONF_COLD_TOLERANCE]
    hot_tolerance: float = config[CONF_HOT_TOLERANCE]
    keep_alive: Optional[timedelta] = config.get(CONF_KEEP_ALIVE)
    initial_hvac_mode: Optional[HVACMode] = config.get(CONF_INITIAL_HVAC_MODE)
    presets: Mapping[str, float] = {key: config[value] for key, value in CONF_PRESETS.items() if value in config}
    precision: Optional[float] = config.get(CONF_PRECISION)
    target_temperature_step: Optional[float] = config.get(CONF_TEMP_STEP)
    unit: UnitOfTemperature = hass.config.units.temperature_unit
    async_add_entities([GenericThermostat(
        hass,
        name,
        heater_entity_id,
        sensor_entity_id,
        min_temp,
        max_temp,
        target_temp,
        ac_mode,
        min_cycle_duration,
        cold_tolerance,
        hot_tolerance,
        keep_alive,
        initial_hvac_mode,
        presets,
        precision,
        target_temperature_step,
        unit,
        unique_id
    )])


class GenericThermostat(ClimateEntity, RestoreEntity):
    _attr_should_poll = False

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        heater_entity_id: str,
        sensor_entity_id: str,
        min_temp: Optional[float],
        max_temp: Optional[float],
        target_temp: Optional[float],
        ac_mode: Optional[bool],
        min_cycle_duration: Optional[timedelta],
        cold_tolerance: float,
        hot_tolerance: float,
        keep_alive: Optional[timedelta],
        initial_hvac_mode: Optional[HVACMode],
        presets: Mapping[str, float],
        precision: Optional[float],
        target_temperature_step: Optional[float],
        unit: UnitOfTemperature,
        unique_id: Optional[str]
    ) -> None:
        self._attr_name: str = name
        self.heater_entity_id: str = heater_entity_id
        self.sensor_entity_id: str = sensor_entity_id
        self._attr_device_info = async_device_info_to_link_from_entity(hass, heater_entity_id)
        self.ac_mode: Optional[bool] = ac_mode
        self.min_cycle_duration: Optional[timedelta] = min_cycle_duration
        self._cold_tolerance: float = cold_tolerance
        self._hot_tolerance: float = hot_tolerance
        self._keep_alive: Optional[timedelta] = keep_alive
        self._hvac_mode: Optional[HVACMode] = initial_hvac_mode
        self._saved_target_temp: Optional[float] = target_temp or next(iter(presets.values()), None)
        self._temp_precision: Optional[float] = precision
        self._temp_target_temperature_step: Optional[float] = target_temperature_step
        self._min_temp: Optional[float] = min_temp
        self._max_temp: Optional[float] = max_temp
        self._active: bool = False
        self._cur_temp: Optional[float] = None
        self._temp_lock: asyncio.Lock = asyncio.Lock()
        self._attr_preset_mode: str = PRESET_NONE
        self._target_temp: Optional[float] = target_temp
        self._attr_temperature_unit: UnitOfTemperature = unit
        self._attr_unique_id: Optional[str] = unique_id
        self._attr_supported_features: int = (
            ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.TURN_OFF | ClimateEntityFeature.TURN_ON
        )
        if len(presets):
            self._attr_supported_features |= ClimateEntityFeature.PRESET_MODE
            self._attr_preset_modes: list[str] = [PRESET_NONE, *presets.keys()]
        else:
            self._attr_preset_modes = [PRESET_NONE]
        self._presets: Mapping[str, float] = presets
        self._presets_inv: dict[float, str] = {v: k for k, v in presets.items()}

        if self.ac_mode:
            self._attr_hvac_modes: list[HVACMode] = [HVACMode.COOL, HVACMode.OFF]
        else:
            self._attr_hvac_modes = [HVACMode.HEAT, HVACMode.OFF]

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(async_track_state_change_event(self.hass, [self.sensor_entity_id], self._async_sensor_changed))
        self.async_on_remove(async_track_state_change_event(self.hass, [self.heater_entity_id], self._async_switch_changed))
        if self._keep_alive:
            self.async_on_remove(async_track_time_interval(self.hass, self._async_control_heating, self._keep_alive))

        @callback
        def _async_startup(_: Any = None) -> None:
            sensor_state: Optional[State] = self.hass.states.get(self.sensor_entity_id)
            if sensor_state and sensor_state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                self._async_update_temp(sensor_state)
                self.async_write_ha_state()
            switch_state: Optional[State] = self.hass.states.get(self.heater_entity_id)
            if switch_state and switch_state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                self.hass.async_create_task(self._check_switch_initial_state(), eager_start=True)
        if self.hass.state is CoreState.running:
            _async_startup()
        else:
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, _async_startup)
        old_state: Optional[State] = (await self.async_get_last_state())
        if old_state is not None:
            if self._target_temp is None:
                if old_state.attributes.get(ATTR_TEMPERATURE) is None:
                    if self.ac_mode:
                        self._target_temp = self._max_temp  # type: ignore
                    else:
                        self._target_temp = self._min_temp  # type: ignore
                    _LOGGER.warning('Undefined target temperature, falling back to %s', self._target_temp)
                else:
                    self._target_temp = float(old_state.attributes[ATTR_TEMPERATURE])
            if self.preset_modes and old_state.attributes.get(ATTR_PRESET_MODE) in self.preset_modes:
                self._attr_preset_mode = cast(str, old_state.attributes.get(ATTR_PRESET_MODE))
            if not self._hvac_mode and old_state.state:
                self._hvac_mode = HVACMode(old_state.state)
        else:
            if self._target_temp is None:
                if self.ac_mode:
                    self._target_temp = self._max_temp  # type: ignore
                else:
                    self._target_temp = self._min_temp  # type: ignore
            _LOGGER.warning('No previously saved temperature, setting to %s', self._target_temp)
        if not self._hvac_mode:
            self._hvac_mode = HVACMode.OFF

    @property
    def precision(self) -> float:
        if self._temp_precision is not None:
            return self._temp_precision
        return cast(float, super().precision)

    @property
    def target_temperature_step(self) -> float:
        if self._temp_target_temperature_step is not None:
            return self._temp_target_temperature_step
        return self.precision

    @property
    def current_temperature(self) -> Optional[float]:
        return self._cur_temp

    @property
    def hvac_mode(self) -> HVACMode:
        return cast(HVACMode, self._hvac_mode)

    @property
    def hvac_action(self) -> HVACAction:
        if self._hvac_mode == HVACMode.OFF:
            return HVACAction.OFF
        if not self._is_device_active:
            return HVACAction.IDLE
        if self.ac_mode:
            return HVACAction.COOLING
        return HVACAction.HEATING

    @property
    def target_temperature(self) -> Optional[float]:
        return self._target_temp

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        if hvac_mode == HVACMode.HEAT:
            self._hvac_mode = HVACMode.HEAT
            await self._async_control_heating(force=True)
        elif hvac_mode == HVACMode.COOL:
            self._hvac_mode = HVACMode.COOL
            await self._async_control_heating(force=True)
        elif hvac_mode == HVACMode.OFF:
            self._hvac_mode = HVACMode.OFF
            if self._is_device_active:
                await self._async_heater_turn_off()
        else:
            _LOGGER.error('Unrecognized hvac mode: %s', hvac_mode)
            return
        self.async_write_ha_state()

    async def async_set_temperature(self, **kwargs: Any) -> None:
        if (temperature := kwargs.get(ATTR_TEMPERATURE)) is None:
            return
        self._attr_preset_mode = self._presets_inv.get(temperature, PRESET_NONE)
        self._target_temp = temperature
        await self._async_control_heating(force=True)
        self.async_write_ha_state()

    @property
    def min_temp(self) -> Optional[float]:
        if self._min_temp is not None:
            return self._min_temp
        return cast(Optional[float], super().min_temp)

    @property
    def max_temp(self) -> Optional[float]:
        if self._max_temp is not None:
            return self._max_temp
        return cast(Optional[float], super().max_temp)

    async def _async_sensor_changed(self, event: Event) -> None:
        new_state: Optional[State] = event.data['new_state']
        if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return
        self._async_update_temp(new_state)
        await self._async_control_heating()
        self.async_write_ha_state()

    async def _check_switch_initial_state(self) -> None:
        if self._hvac_mode == HVACMode.OFF and self._is_device_active:
            _LOGGER.warning('The climate mode is OFF, but the switch device is ON. Turning off device %s', self.heater_entity_id)
            await self._async_heater_turn_off()

    @callback
    def _async_switch_changed(self, event: Event) -> None:
        new_state: Optional[State] = event.data['new_state']
        old_state: Optional[State] = event.data['old_state']
        if new_state is None:
            return
        if old_state is None:
            self.hass.async_create_task(self._check_switch_initial_state(), eager_start=True)
        self.async_write_ha_state()

    @callback
    def _async_update_temp(self, state: State) -> None:
        try:
            cur_temp: float = float(state.state)
            if not math.isfinite(cur_temp):
                raise ValueError(f'Sensor has illegal state {state.state}')
            self._cur_temp = cur_temp
        except ValueError as ex:
            _LOGGER.error('Unable to update from sensor: %s', ex)

    async def _async_control_heating(self, time: Optional[datetime] = None, force: bool = False) -> None:
        async with self._temp_lock:
            if not self._active and None not in (self._cur_temp, self._target_temp):
                self._active = True
                _LOGGER.debug('Obtained current and target temperature. Generic thermostat active. %s, %s', self._cur_temp, self._target_temp)
            if not self._active or self._hvac_mode == HVACMode.OFF:
                return
            if not force and time is None and self.min_cycle_duration:
                let current_state: str = STATE_ON if self._is_device_active else HVACMode.OFF  # type: ignore
                try:
                    long_enough: bool = condition.state(self.hass, self.heater_entity_id, current_state, self.min_cycle_duration)
                except ConditionError:
                    long_enough = False
                if not long_enough:
                    return
            assert self._cur_temp is not None and self._target_temp is not None
            too_cold: bool = self._target_temp >= self._cur_temp + self._cold_tolerance
            too_hot: bool = self._cur_temp >= self._target_temp + self._hot_tolerance
            if self._is_device_active:
                if self.ac_mode and too_cold or (not self.ac_mode and too_hot):
                    _LOGGER.debug('Turning off heater %s', self.heater_entity_id)
                    await self._async_heater_turn_off()
                elif time is not None:
                    _LOGGER.debug('Keep-alive - Turning on heater heater %s', self.heater_entity_id)
                    await self._async_heater_turn_on()
            elif self.ac_mode and too_hot or (not self.ac_mode and too_cold):
                _LOGGER.debug('Turning on heater %s', self.heater_entity_id)
                await self._async_heater_turn_on()
            elif time is not None:
                _LOGGER.debug('Keep-alive - Turning off heater %s', self.heater_entity_id)
                await self._async_heater_turn_off()

    @property
    def _is_device_active(self) -> Optional[bool]:
        if not self.hass.states.get(self.heater_entity_id):
            return None
        return self.hass.states.is_state(self.heater_entity_id, STATE_ON)

    async def _async_heater_turn_on(self) -> None:
        data: dict[str, Any] = {ATTR_ENTITY_ID: self.heater_entity_id}
        await self.hass.services.async_call(HOMEASSISTANT_DOMAIN, SERVICE_TURN_ON, data, context=self._context)

    async def _async_heater_turn_off(self) -> None:
        data: dict[str, Any] = {ATTR_ENTITY_ID: self.heater_entity_id}
        await self.hass.services.async_call(HOMEASSISTANT_DOMAIN, SERVICE_TURN_OFF, data, context=self._context)

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        if preset_mode not in (self.preset_modes or []):
            raise ValueError(f'Got unsupported preset_mode {preset_mode}. Must be one of {self.preset_modes}')
        if preset_mode == self._attr_preset_mode:
            return
        if preset_mode == PRESET_NONE:
            self._attr_preset_mode = PRESET_NONE
            self._target_temp = self._saved_target_temp
            await self._async_control_heating(force=True)
        else:
            if self._attr_preset_mode == PRESET_NONE:
                self._saved_target_temp = self._target_temp
            self._attr_preset_mode = preset_mode
            self._target_temp = self._presets[preset_mode]
            await self._async_control_heating(force=True)
        self.async_write_ha_state()
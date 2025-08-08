from __future__ import annotations
import asyncio
import enum
import logging
from time import localtime, strftime, time
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
_LOGGER = logging.getLogger(__name__)
SUPPORT_FLAGS_LCC: ClimateEntityFeature = ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.PRESET_MODE | ClimateEntityFeature.TARGET_TEMPERATURE_RANGE
SUPPORT_FLAGS_TCC: ClimateEntityFeature = ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.TARGET_TEMPERATURE_RANGE
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
SCHEMA_HOLD_TIME: dict[str, Any] = {vol.Required(ATTR_TIME_PERIOD, default='01:00:00'): vol.All(cv.time_period, cv.positive_timedelta, lambda td: strftime('%H:%M:%S', localtime(time() + td.total_seconds())))}

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the Honeywell Lyric climate platform based on a config entry."""
    coordinator: DataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities((LyricClimate(coordinator, ClimateEntityDescription(key=f'{device.mac_id}_thermostat', name=device.name), location, device) for location in coordinator.data.locations for device in location.devices), True)
    platform: entity_platform.EntityPlatform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(SERVICE_HOLD_TIME, SCHEMA_HOLD_TIME, 'async_set_hold_time')

class LyricThermostatType(enum.Enum):
    """Lyric thermostats are classified as TCC or LCC devices."""
    TCC = enum.auto()
    LCC = enum.auto()

class LyricClimate(LyricDeviceEntity, ClimateEntity):
    """Defines a Honeywell Lyric climate entity."""
    _attr_name: str = None
    _attr_preset_modes: list[str] = [PRESET_NO_HOLD, PRESET_HOLD_UNTIL, PRESET_PERMANENT_HOLD, PRESET_TEMPORARY_HOLD, PRESET_VACATION_HOLD]

    def __init__(self, coordinator: DataUpdateCoordinator, description: ClimateEntityDescription, location: LyricLocation, device: LyricDevice) -> None:
        """Initialize Honeywell Lyric climate entity."""
        if device.changeable_values.thermostat_setpoint_status:
            self._attr_thermostat_type: LyricThermostatType = LyricThermostatType.LCC
        else:
            self._attr_thermostat_type: LyricThermostatType = LyricThermostatType.TCC
        if device.units == 'Fahrenheit':
            self._attr_temperature_unit: UnitOfTemperature = UnitOfTemperature.FAHRENHEIT
            self._attr_precision: int = PRECISION_WHOLE
        else:
            self._attr_temperature_unit: UnitOfTemperature = UnitOfTemperature.CELSIUS
            self._attr_precision: int = PRECISION_HALVES
        self._attr_hvac_modes: list[HVACMode] = [HVACMode.OFF]
        if LYRIC_HVAC_MODE_HEAT in device.allowed_modes:
            self._attr_hvac_modes.append(HVACMode.HEAT)
        if LYRIC_HVAC_MODE_COOL in device.allowed_modes:
            self._attr_hvac_modes.append(HVACMode.COOL)
        if LYRIC_HVAC_MODE_HEAT_COOL in device.allowed_modes or (self._attr_thermostat_type is LyricThermostatType.TCC and LYRIC_HVAC_MODE_HEAT in device.allowed_modes and (LYRIC_HVAC_MODE_COOL in device.allowed_modes)):
            self._attr_hvac_modes.append(HVACMode.HEAT_COOL)
        if self._attr_thermostat_type is LyricThermostatType.LCC:
            self._attr_supported_features: ClimateEntityFeature = SUPPORT_FLAGS_LCC
        else:
            self._attr_supported_features: ClimateEntityFeature = SUPPORT_FLAGS_TCC
        if (device_fan_modes := device.settings.attributes.get('fan', {}).get('allowedModes')):
            self._attr_fan_modes: list[str] = [FAN_MODES[device_fan_mode] for device_fan_mode in device_fan_modes if device_fan_mode in FAN_MODES]
            self._attr_supported_features: ClimateEntityFeature = self._attr_supported_features | ClimateEntityFeature.FAN_MODE
        if len(self.hvac_modes) > 1:
            self._attr_supported_features |= ClimateEntityFeature.TURN_OFF | ClimateEntityFeature.TURN_ON
        super().__init__(coordinator, location, device, f'{device.mac_id}_thermostat')
        self.entity_description = description

    @property
    def current_temperature(self) -> float:
        """Return the current temperature."""
        return self.device.indoor_temperature

    @property
    def hvac_action(self) -> HVACAction:
        """Return the current hvac action."""
        action: HVACAction = HVAC_ACTIONS.get(self.device.operation_status.mode, None)
        if action == HVACAction.OFF and self.hvac_mode != HVACMode.OFF:
            action = HVACAction.IDLE
        return action

    @property
    def hvac_mode(self) -> HVACMode:
        """Return the hvac mode."""
        return HVAC_MODES[self.device.changeable_values.mode]

    @property
    def target_temperature(self) -> float:
        """Return the temperature we try to reach."""
        device = self.device
        if device.changeable_values.auto_changeover_active or HVAC_MODES[device.changeable_values.mode] == HVACMode.OFF:
            return None
        if self.hvac_mode == HVACMode.COOL:
            return device.changeable_values.cool_setpoint
        return device.changeable_values.heat_setpoint

    @property
    def target_temperature_high(self) -> float:
        """Return the highbound target temperature we try to reach."""
        device = self.device
        if not device.changeable_values.auto_changeover_active or HVAC_MODES[device.changeable_values.mode] == HVACMode.OFF:
            return None
        return device.changeable_values.cool_setpoint

    @property
    def target_temperature_low(self) -> float:
        """Return the lowbound target temperature we try to reach."""
        device = self.device
        if not device.changeable_values.auto_changeover_active or HVAC_MODES[device.changeable_values.mode] == HVACMode.OFF:
            return None
        return device.changeable_values.heat_setpoint

    @property
    def preset_mode(self) -> str:
        """Return current preset mode."""
        return self.device.changeable_values.thermostat_setpoint_status

    @property
    def min_temp(self) -> float:
        """Identify min_temp in Lyric API or defaults if not available."""
        device = self.device
        if LYRIC_HVAC_MODE_COOL in device.allowed_modes:
            return device.min_cool_setpoint
        return device.min_heat_setpoint

    @property
    def max_temp(self) -> float:
        """Identify max_temp in Lyric API or defaults if not available."""
        device = self.device
        if LYRIC_HVAC_MODE_HEAT in device.allowed_modes:
            return device.max_heat_setpoint
        return device.max_cool_setpoint

    @property
    def fan_mode(self) -> str:
        """Return current fan mode."""
        device = self.device
        return FAN_MODES.get(device.settings.attributes.get('fan', {}).get('changeableValues', {}).get('mode'))

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        if self.hvac_mode == HVACMode.OFF:
            return
        device = self.device
        target_temp_low = kwargs.get(ATTR_TARGET_TEMP_LOW)
        target_temp_high = kwargs.get(ATTR_TARGET_TEMP_HIGH)
        if device.changeable_values.mode == LYRIC_HVAC_MODE_HEAT_COOL:
            if target_temp_low is None or target_temp_high is None:
                raise HomeAssistantError('Could not find target_temp_low and/or target_temp_high in arguments')
            if self._attr_thermostat_type is LyricThermostatType.TCC:
                mode = HVAC_MODES[device.changeable_values.heat_cool_mode]
            else:
                mode = None
            _LOGGER.debug('Set temperature: %s - %s', target_temp_low, target_temp_high)
            try:
                await self._update_thermostat(self.location, device, cool_setpoint=target_temp_high, heat_setpoint=target_temp_low, mode=mode)
            except LYRIC_EXCEPTIONS as exception:
                _LOGGER.error(exception)
            await self.coordinator.async_refresh()
        else:
            temp = kwargs.get(ATTR_TEMPERATURE)
            _LOGGER.debug('Set temperature: %s', temp)
            try:
                if self.hvac_mode == HVACMode.COOL:
                    await self._update_thermostat(self.location, device, cool_setpoint=temp)
                else:
                    await self._update_thermostat(self.location, device, heat_setpoint=temp)
            except LYRIC_EXCEPTIONS as exception:
                _LOGGER.error(exception)
            await self.coordinator.async_refresh()

    async def async_set_hvac_mode(self, hvac_mode: str) -> None:
        """Set hvac mode."""
        _LOGGER.debug('HVAC mode: %s', hvac_mode)
        try:
            match self._attr_thermostat_type:
                case LyricThermostatType.TCC:
                    await self._async_set_hvac_mode_tcc(hvac_mode)
                case LyricThermostatType.LCC:
                    await self._async_set_hvac_mode_lcc(hvac_mode)
        except LYRIC_EXCEPTIONS as exception:
            _LOGGER.error(exception)
        await self.coordinator.async_refresh()

    async def _async_set_hvac_mode_tcc(self, hvac_mode: str) -> None:
        """Set hvac mode for TCC devices (e.g., Lyric round)."""
        if LYRIC_HVAC_MODES[hvac_mode] == LYRIC_HVAC_MODE_HEAT_COOL:
            if HVAC_MODES[self.device.changeable_values.mode] == HVACMode.OFF:
                _LOGGER.debug('HVAC mode passed to lyric: %s', HVAC_MODES[LYRIC_HVAC_MODE_COOL])
                await self._update_thermostat(self.location, self.device, mode=HVAC_MODES[LYRIC_HVAC_MODE_HEAT], auto_changeover_active=False)
                await asyncio.sleep(3)
                _LOGGER.debug('HVAC mode passed to lyric: %s', HVAC_MODES[LYRIC_HVAC_MODE_HEAT])
                await self._update_thermostat(self.location, self.device, mode=HVAC_MODES[LYRIC_HVAC_MODE_HEAT], auto_changeover_active=True)
            else:
                _LOGGER.debug('HVAC mode passed to lyric: %s', HVAC_MODES[self.device.changeable_values.mode])
                await self._update_thermostat(self.location, self.device, auto_changeover_active=True)
        else:
            _LOGGER.debug('HVAC mode passed to lyric: %s', LYRIC_HVAC_MODES[hvac_mode])
            await self._update_thermostat(self.location, self.device, mode=LYRIC_HVAC_MODES[hvac_mode], auto_changeover_active=False)

    async def _async_set_hvac_mode_lcc(self, hvac_mode: str) -> None:
        """Set hvac mode for LCC devices (e.g., T5,6)."""
        _LOGGER.debug('HVAC mode passed to lyric: %s', LYRIC_HVAC_MODES[hvac_mode])
        if LYRIC_HVAC_MODES[hvac_mode] == LYRIC_HVAC_MODE_HEAT_COOL and (not self.device.changeable_values.auto_changeover_active):
            auto_changeover = True
        else:
            auto_changeover = None
        await self._update_thermostat(self.location, self.device, mode=LYRIC_HVAC_MODES[hvac_mode], auto_changeover_active=auto_changeover)

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set preset (PermanentHold, HoldUntil, NoHold, VacationHold) mode."""
        _LOGGER.debug('Set preset mode: %s', preset_mode)
        try:
            await self._update_thermostat(self.location, self.device, thermostat_setpoint_status=preset_mode)
        except LYRIC_EXCEPTIONS as exception:
            _LOGGER.error(exception)
        await self.coordinator.async_refresh()

    async def async_set_hold_time(self, time_period: str) -> None:
        """Set the time to hold until."""
        _LOGGER.debug('set_hold_time: %s', time_period)
        try:
            await self._update_thermostat(self.location, self.device, thermostat_setpoint_status=PRESET_HOLD_UNTIL, next_period_time=time_period)
        except LYRIC_EXCEPTIONS as exception:
            _LOGGER.error(exception)
        await self.coordinator.async_refresh()

    async def async_set_fan_mode(self, fan_mode: str) -> None:
        """Set fan mode."""
        _LOGGER.debug('Set fan mode: %s', fan_mode)
        try:
            _LOGGER.debug('Fan mode passed to lyric: %s', LYRIC_FAN_MODES[fan_mode])
            await self._update_fan(self.location, self.device, mode=LYRIC_FAN_MODES[fan_mode])
        except LYRIC_EXCEPTIONS as exception:
            _LOGGER.error(exception)
        except KeyError:
            _LOGGER.error('The fan mode requested does not have a corresponding mode in lyric: %s', fan_mode)
        await self.coordinator.async_refresh()

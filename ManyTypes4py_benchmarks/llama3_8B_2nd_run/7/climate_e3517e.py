from __future__ import annotations
import asyncio
import enum
import logging
from time import localtime, strftime, time
from typing import Any, Dict, List

class LyricThermostatType(enum.Enum):
    """Lyric thermostats are classified as TCC or LCC devices."""
    TCC = enum.auto()
    LCC = enum.auto()

class LyricClimate(LyricDeviceEntity, ClimateEntity):
    """Defines a Honeywell Lyric climate entity."""

    def __init__(self, 
                 coordinator: DataUpdateCoordinator, 
                 description: ClimateEntityDescription, 
                 location: LyricLocation, 
                 device: LyricDevice) -> None:
        """Initialize Honeywell Lyric climate entity."""
        # ... (rest of the code remains the same)

    @property
    def current_temperature(self) -> float:
        """Return the current temperature."""
        return self.device.indoor_temperature

    @property
    def hvac_action(self) -> HVACAction:
        """Return the current hvac action."""
        action = HVAC_ACTIONS.get(self.device.operation_status.mode, None)
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

    async def async_set_temperature(self, **kwargs: Dict[str, Any]) -> None:
        """Set new target temperature."""
        # ... (rest of the code remains the same)

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set hvac mode."""
        # ... (rest of the code remains the same)

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set preset (PermanentHold, HoldUntil, NoHold, VacationHold) mode."""
        # ... (rest of the code remains the same)

    async def async_set_hold_time(self, time_period: str) -> None:
        """Set the time to hold until."""
        # ... (rest of the code remains the same)

    async def async_set_fan_mode(self, fan_mode: str) -> None:
        """Set fan mode."""
        # ... (rest of the code remains the same)

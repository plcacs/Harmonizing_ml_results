"""Demo platform that offers a fake climate device."""
from __future__ import annotations
from typing import Any, List, Optional, Set

from homeassistant.components.climate import ATTR_HVAC_MODE, ATTR_TARGET_TEMP_HIGH, ATTR_TARGET_TEMP_LOW, ClimateEntity, ClimateEntityFeature, HVACAction, HVACMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import DOMAIN

SUPPORT_FLAGS = ClimateEntityFeature(0)

async def async_setup_entry(
    hass: HomeAssistant, 
    config_entry: ConfigEntry, 
    async_add_entities: AddEntitiesCallback
) -> None:
    """Set up the demo climate platform."""
    async_add_entities([
        DemoClimate(
            unique_id='climate_1',
            device_name='HeatPump',
            target_temperature=68,
            unit_of_measurement=UnitOfTemperature.FAHRENHEIT,
            preset=None,
            current_temperature=77,
            fan_mode=None,
            target_humidity=None,
            current_humidity=None,
            swing_mode=None,
            swing_horizontal_mode=None,
            hvac_mode=HVACMode.HEAT,
            hvac_action=HVACAction.HEATING,
            target_temp_high=None,
            target_temp_low=None,
            hvac_modes=[HVACMode.HEAT, HVACMode.OFF]
        ),
        DemoClimate(
            unique_id='climate_2',
            device_name='Hvac',
            target_temperature=21,
            unit_of_measurement=UnitOfTemperature.CELSIUS,
            preset=None,
            current_temperature=22,
            fan_mode='on_high',
            target_humidity=67.4,
            current_humidity=54.2,
            swing_mode='off',
            swing_horizontal_mode='auto',
            hvac_mode=HVACMode.COOL,
            hvac_action=HVACAction.COOLING,
            target_temp_high=None,
            target_temp_low=None,
            hvac_modes=[cls for cls in HVACMode if cls != HVACMode.HEAT_COOL]
        ),
        DemoClimate(
            unique_id='climate_3',
            device_name='Ecobee',
            target_temperature=None,
            unit_of_measurement=UnitOfTemperature.CELSIUS,
            preset='home',
            preset_modes=['home', 'eco', 'away'],
            current_temperature=23,
            fan_mode='auto_low',
            target_humidity=None,
            current_humidity=None,
            swing_mode='auto',
            swing_horizontal_mode=None,
            hvac_mode=HVACMode.HEAT_COOL,
            hvac_action=None,
            target_temp_high=24,
            target_temp_low=21,
            hvac_modes=[cls for cls in HVACMode if cls != HVACMode.HEAT]
        )
    ])

class DemoClimate(ClimateEntity):
    """Representation of a demo climate device."""
    _attr_has_entity_name: bool = True
    _attr_name: Optional[str] = None
    _attr_should_poll: bool = False
    _attr_translation_key: str = 'ubercool'

    def __init__(
        self,
        unique_id: str,
        device_name: str,
        target_temperature: Optional[float],
        unit_of_measurement: str,
        preset: Optional[str],
        current_temperature: Optional[float],
        fan_mode: Optional[str],
        target_humidity: Optional[float],
        current_humidity: Optional[float],
        swing_mode: Optional[str],
        swing_horizontal_mode: Optional[str],
        hvac_mode: HVACMode,
        hvac_action: Optional[HVACAction],
        target_temp_high: Optional[float],
        target_temp_low: Optional[float],
        hvac_modes: List[HVACMode],
        preset_modes: Optional[List[str]] = None,
    ) -> None:
        """Initialize the climate device."""
        self._unique_id: str = unique_id
        self._attr_supported_features: ClimateEntityFeature = SUPPORT_FLAGS
        if target_temperature is not None:
            self._attr_supported_features |= ClimateEntityFeature.TARGET_TEMPERATURE
        if preset is not None:
            self._attr_supported_features |= ClimateEntityFeature.PRESET_MODE
        if fan_mode is not None:
            self._attr_supported_features |= ClimateEntityFeature.FAN_MODE
        if target_humidity is not None:
            self._attr_supported_features |= ClimateEntityFeature.TARGET_HUMIDITY
        if swing_mode is not None:
            self._attr_supported_features |= ClimateEntityFeature.SWING_MODE
        if swing_horizontal_mode is not None:
            self._attr_supported_features |= ClimateEntityFeature.SWING_HORIZONTAL_MODE
        if HVACMode.HEAT_COOL in hvac_modes or HVACMode.AUTO in hvac_modes:
            self._attr_supported_features |= ClimateEntityFeature.TARGET_TEMPERATURE_RANGE
        self._attr_supported_features |= ClimateEntityFeature.TURN_OFF | ClimateEntityFeature.TURN_ON
        self._target_temperature: Optional[float] = target_temperature
        self._target_humidity: Optional[float] = target_humidity
        self._unit_of_measurement: str = unit_of_measurement
        self._preset: Optional[str] = preset
        self._preset_modes: Optional[List[str]] = preset_modes
        self._current_temperature: Optional[float] = current_temperature
        self._current_humidity: Optional[float] = current_humidity
        self._current_fan_mode: Optional[str] = fan_mode
        self._hvac_action: Optional[HVACAction] = hvac_action
        self._hvac_mode: HVACMode = hvac_mode
        self._current_swing_mode: Optional[str] = swing_mode
        self._current_swing_horizontal_mode: Optional[str] = swing_horizontal_mode
        self._fan_modes: List[str] = ['on_low', 'on_high', 'auto_low', 'auto_high', 'off']
        self._hvac_modes: List[HVACMode] = hvac_modes
        self._swing_modes: List[str] = ['auto', '1', '2', '3', 'off']
        self._swing_horizontal_modes: List[str] = ['auto', 'rangefull', 'off']
        self._target_temperature_high: Optional[float] = target_temp_high
        self._target_temperature_low: Optional[float] = target_temp_low
        self._attr_device_info: DeviceInfo = DeviceInfo(
            identifiers={(DOMAIN, unique_id)},
            name=device_name
        )

    @property
    def unique_id(self) -> str:
        """Return the unique id."""
        return self._unique_id

    @property
    def temperature_unit(self) -> str:
        """Return the unit of measurement."""
        return self._unit_of_measurement

    @property
    def current_temperature(self) -> Optional[float]:
        """Return the current temperature."""
        return self._current_temperature

    @property
    def target_temperature(self) -> Optional[float]:
        """Return the temperature we try to reach."""
        return self._target_temperature

    @property
    def target_temperature_high(self) -> Optional[float]:
        """Return the highbound target temperature we try to reach."""
        return self._target_temperature_high

    @property
    def target_temperature_low(self) -> Optional[float]:
        """Return the lowbound target temperature we try to reach."""
        return self._target_temperature_low

    @property
    def current_humidity(self) -> Optional[float]:
        """Return the current humidity."""
        return self._current_humidity

    @property
    def target_humidity(self) -> Optional[float]:
        """Return the humidity we try to reach."""
        return self._target_humidity

    @property
    def hvac_action(self) -> Optional[HVACAction]:
        """Return current operation ie. heat, cool, idle."""
        return self._hvac_action

    @property
    def hvac_mode(self) -> HVACMode:
        """Return hvac target hvac state."""
        return self._hvac_mode

    @property
    def hvac_modes(self) -> List[HVACMode]:
        """Return the list of available operation modes."""
        return self._hvac_modes

    @property
    def preset_mode(self) -> Optional[str]:
        """Return preset mode."""
        return self._preset

    @property
    def preset_modes(self) -> Optional[List[str]]:
        """Return preset modes."""
        return self._preset_modes

    @property
    def fan_mode(self) -> Optional[str]:
        """Return the fan setting."""
        return self._current_fan_mode

    @property
    def fan_modes(self) -> List[str]:
        """Return the list of available fan modes."""
        return self._fan_modes

    @property
    def swing_mode(self) -> Optional[str]:
        """Return the swing setting."""
        return self._current_swing_mode

    @property
    def swing_modes(self) -> List[str]:
        """List of available swing modes."""
        return self._swing_modes

    @property
    def swing_horizontal_mode(self) -> Optional[str]:
        """Return the swing setting."""
        return self._current_swing_horizontal_mode

    @property
    def swing_horizontal_modes(self) -> List[str]:
        """List of available swing modes."""
        return self._swing_horizontal_modes

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperatures."""
        if kwargs.get(ATTR_TEMPERATURE) is not None:
            self._target_temperature = kwargs.get(ATTR_TEMPERATURE)
        if kwargs.get(ATTR_TARGET_TEMP_HIGH) is not None and kwargs.get(ATTR_TARGET_TEMP_LOW) is not None:
            self._target_temperature_high = kwargs.get(ATTR_TARGET_TEMP_HIGH)
            self._target_temperature_low = kwargs.get(ATTR_TARGET_TEMP_LOW)
        if (hvac_mode := kwargs.get(ATTR_HVAC_MODE)) is not None:
            self._hvac_mode = hvac_mode
        self.async_write_ha_state()

    async def async_set_humidity(self, humidity: float) -> None:
        """Set new humidity level."""
        self._target_humidity = humidity
        self.async_write_ha_state()

    async def async_set_swing_mode(self, swing_mode: str) -> None:
        """Set new swing mode."""
        self._current_swing_mode = swing_mode
        self.async_write_ha_state()

    async def async_set_swing_horizontal_mode(self, swing_horizontal_mode: str) -> None:
        """Set new swing mode."""
        self._current_swing_horizontal_mode = swing_horizontal_mode
        self.async_write_ha_state()

    async def async_set_fan_mode(self, fan_mode: str) -> None:
        """Set new fan mode."""
        self._current_fan_mode = fan_mode
        self.async_write_ha_state()

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set new operation mode."""
        self._hvac_mode = hvac_mode
        self.async_write_ha_state()

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Update preset_mode on."""
        self._preset = preset_mode
        self.async_write_ha_state()

from __future__ import annotations
from typing import Any, Optional, List
from homeassistant.components.fan import FanEntity, FanEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

PRESET_MODE_AUTO: str = 'auto'
PRESET_MODE_SMART: str = 'smart'
PRESET_MODE_SLEEP: str = 'sleep'
PRESET_MODE_ON: str = 'on'
FULL_SUPPORT: int = FanEntityFeature.SET_SPEED | FanEntityFeature.OSCILLATE | FanEntityFeature.DIRECTION | FanEntityFeature.TURN_OFF | FanEntityFeature.TURN_ON
LIMITED_SUPPORT: int = FanEntityFeature.SET_SPEED | FanEntityFeature.TURN_OFF | FanEntityFeature.TURN_ON

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    async_add_entities([
        DemoPercentageFan(hass, 'fan1', 'Living Room Fan', FULL_SUPPORT, [PRESET_MODE_AUTO, PRESET_MODE_SMART, PRESET_MODE_SLEEP, PRESET_MODE_ON]),
        DemoPercentageFan(hass, 'fan2', 'Ceiling Fan', LIMITED_SUPPORT, None),
        AsyncDemoPercentageFan(hass, 'fan3', 'Percentage Full Fan', FULL_SUPPORT, [PRESET_MODE_AUTO, PRESET_MODE_SMART, PRESET_MODE_SLEEP, PRESET_MODE_ON]),
        DemoPercentageFan(hass, 'fan4', 'Percentage Limited Fan', LIMITED_SUPPORT, [PRESET_MODE_AUTO, PRESET_MODE_SMART, PRESET_MODE_SLEEP, PRESET_MODE_ON]),
        AsyncDemoPercentageFan(hass, 'fan5', 'Preset Only Limited Fan', FanEntityFeature.PRESET_MODE | FanEntityFeature.TURN_OFF | FanEntityFeature.TURN_ON, [PRESET_MODE_AUTO, PRESET_MODE_SMART, PRESET_MODE_SLEEP, PRESET_MODE_ON])
    ])

class BaseDemoFan(FanEntity):
    _attr_should_poll: bool = False
    _attr_translation_key: str = 'demo'

    def __init__(
        self,
        hass: HomeAssistant,
        unique_id: str,
        name: str,
        supported_features: int,
        preset_modes: Optional[List[str]],
    ) -> None:
        self.hass: HomeAssistant = hass
        self._unique_id: str = unique_id
        self._attr_supported_features: int = supported_features
        self._percentage: Optional[int] = None
        self._preset_modes: Optional[List[str]] = preset_modes
        self._preset_mode: Optional[str] = None
        self._oscillating: Optional[bool] = None
        self._direction: Optional[str] = None
        self._attr_name: str = name
        if supported_features & FanEntityFeature.OSCILLATE:
            self._oscillating = False
        if supported_features & FanEntityFeature.DIRECTION:
            self._direction = 'forward'

    @property
    def unique_id(self) -> str:
        return self._unique_id

    @property
    def current_direction(self) -> Optional[str]:
        return self._direction

    @property
    def oscillating(self) -> Optional[bool]:
        return self._oscillating

class DemoPercentageFan(BaseDemoFan, FanEntity):
    @property
    def percentage(self) -> Optional[int]:
        return self._percentage

    @property
    def speed_count(self) -> int:
        return 3

    def set_percentage(self, percentage: int) -> None:
        self._percentage = percentage
        self._preset_mode = None
        self.schedule_update_ha_state()

    @property
    def preset_mode(self) -> Optional[str]:
        return self._preset_mode

    @property
    def preset_modes(self) -> Optional[List[str]]:
        return self._preset_modes

    def set_preset_mode(self, preset_mode: str) -> None:
        self._preset_mode = preset_mode
        self._percentage = None
        self.schedule_update_ha_state()

    def turn_on(self, percentage: Optional[int] = None, preset_mode: Optional[str] = None, **kwargs: Any) -> None:
        if preset_mode:
            self.set_preset_mode(preset_mode)
            return
        if percentage is None:
            percentage = 67
        self.set_percentage(percentage)

    def turn_off(self, **kwargs: Any) -> None:
        self.set_percentage(0)

    def set_direction(self, direction: str) -> None:
        self._direction = direction
        self.schedule_update_ha_state()

    def oscillate(self, oscillating: bool) -> None:
        self._oscillating = oscillating
        self.schedule_update_ha_state()

class AsyncDemoPercentageFan(BaseDemoFan, FanEntity):
    @property
    def percentage(self) -> Optional[int]:
        return self._percentage

    @property
    def speed_count(self) -> int:
        return 3

    async def async_set_percentage(self, percentage: int) -> None:
        self._percentage = percentage
        self._preset_mode = None
        self.async_write_ha_state()

    @property
    def preset_mode(self) -> Optional[str]:
        return self._preset_mode

    @property
    def preset_modes(self) -> Optional[List[str]]:
        return self._preset_modes

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        self._preset_mode = preset_mode
        self._percentage = None
        self.async_write_ha_state()

    async def async_turn_on(
        self,
        percentage: Optional[int] = None,
        preset_mode: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        if preset_mode:
            await self.async_set_preset_mode(preset_mode)
            return
        if percentage is None:
            percentage = 67
        await self.async_set_percentage(percentage)

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self.async_oscillate(False)
        await self.async_set_percentage(0)

    async def async_set_direction(self, direction: str) -> None:
        self._direction = direction
        self.async_write_ha_state()

    async def async_oscillate(self, oscillating: bool) -> None:
        self._oscillating = oscillating
        self.async_write_ha_state()
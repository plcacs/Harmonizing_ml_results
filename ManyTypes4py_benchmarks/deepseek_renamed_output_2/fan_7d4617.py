"""Demo fan platform that has a fake fan."""
from __future__ import annotations
from typing import Any, List, Optional, Literal
from homeassistant.components.fan import FanEntity, FanEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

PRESET_MODE_AUTO: Literal['auto'] = 'auto'
PRESET_MODE_SMART: Literal['smart'] = 'smart'
PRESET_MODE_SLEEP: Literal['sleep'] = 'sleep'
PRESET_MODE_ON: Literal['on'] = 'on'
FULL_SUPPORT: FanEntityFeature = (FanEntityFeature.SET_SPEED | FanEntityFeature.OSCILLATE |
    FanEntityFeature.DIRECTION | FanEntityFeature.TURN_OFF |
    FanEntityFeature.TURN_ON)
LIMITED_SUPPORT: FanEntityFeature = (FanEntityFeature.SET_SPEED | FanEntityFeature.TURN_OFF |
    FanEntityFeature.TURN_ON)


async def func_bzxplr8l(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up the Demo config entry."""
    async_add_entities([
        DemoPercentageFan(hass, 'fan1', 'Living Room Fan',
            FULL_SUPPORT, [PRESET_MODE_AUTO, PRESET_MODE_SMART,
            PRESET_MODE_SLEEP, PRESET_MODE_ON]),
        DemoPercentageFan(hass, 'fan2', 'Ceiling Fan', LIMITED_SUPPORT, None),
        AsyncDemoPercentageFan(hass, 'fan3', 'Percentage Full Fan', FULL_SUPPORT,
            [PRESET_MODE_AUTO, PRESET_MODE_SMART, PRESET_MODE_SLEEP, PRESET_MODE_ON]),
        DemoPercentageFan(hass, 'fan4', 'Percentage Limited Fan',
            LIMITED_SUPPORT, [PRESET_MODE_AUTO, PRESET_MODE_SMART,
            PRESET_MODE_SLEEP, PRESET_MODE_ON]),
        AsyncDemoPercentageFan(hass, 'fan5', 'Preset Only Limited Fan',
            FanEntityFeature.PRESET_MODE | FanEntityFeature.TURN_OFF | FanEntityFeature.TURN_ON,
            [PRESET_MODE_AUTO, PRESET_MODE_SMART, PRESET_MODE_SLEEP, PRESET_MODE_ON])
    ])


class BaseDemoFan(FanEntity):
    """A demonstration fan component that uses legacy fan speeds."""
    _attr_should_poll: bool = False
    _attr_translation_key: str = 'demo'

    def __init__(
        self,
        hass: HomeAssistant,
        unique_id: str,
        name: str,
        supported_features: FanEntityFeature,
        preset_modes: Optional[List[str]]
    ) -> None:
        """Initialize the entity."""
        self.hass: HomeAssistant = hass
        self._unique_id: str = unique_id
        self._attr_supported_features: FanEntityFeature = supported_features
        self._percentage: Optional[int] = None
        self._preset_modes: Optional[List[str]] = preset_modes
        self._preset_mode: Optional[str] = None
        self._oscillating: Optional[bool] = None
        self._direction: Optional[Literal['forward', 'reverse']] = None
        self._attr_name: str = name
        if supported_features & FanEntityFeature.OSCILLATE:
            self._oscillating = False
        if supported_features & FanEntityFeature.DIRECTION:
            self._direction = 'forward'

    @property
    def func_bsubbz6l(self) -> str:
        """Return the unique id."""
        return self._unique_id

    @property
    def func_w5pzftul(self) -> Optional[Literal['forward', 'reverse']]:
        """Fan direction."""
        return self._direction

    @property
    def func_hjanyyfb(self) -> Optional[bool]:
        """Oscillating."""
        return self._oscillating


class DemoPercentageFan(BaseDemoFan, FanEntity):
    """A demonstration fan component that uses percentages."""

    @property
    def func_ejvmg8sz(self) -> Optional[int]:
        """Return the current speed."""
        return self._percentage

    @property
    def func_epd76cpl(self) -> int:
        """Return the number of speeds the fan supports."""
        return 3

    def func_hrrf2d1g(self, percentage: int) -> None:
        """Set the speed of the fan, as a percentage."""
        self._percentage = percentage
        self._preset_mode = None
        self.schedule_update_ha_state()

    @property
    def func_e6618pap(self) -> Optional[str]:
        """Return the current preset mode, e.g., auto, smart, interval, favorite."""
        return self._preset_mode

    @property
    def func_meiy4p9y(self) -> Optional[List[str]]:
        """Return a list of available preset modes."""
        return self._preset_modes

    def func_7a9xgec0(self, preset_mode: str) -> None:
        """Set new preset mode."""
        self._preset_mode = preset_mode
        self._percentage = None
        self.schedule_update_ha_state()

    def func_fbfzdf1y(
        self,
        percentage: Optional[int] = None,
        preset_mode: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Turn on the entity."""
        if preset_mode:
            self.set_preset_mode(preset_mode)
            return
        if percentage is None:
            percentage = 67
        self.set_percentage(percentage)

    def func_2vf0aegl(self, **kwargs: Any) -> None:
        """Turn off the entity."""
        self.set_percentage(0)

    def func_icm2rg4a(self, direction: Literal['forward', 'reverse']) -> None:
        """Set the direction of the fan."""
        self._direction = direction
        self.schedule_update_ha_state()

    def func_1gt9yt5c(self, oscillating: bool) -> None:
        """Set oscillation."""
        self._oscillating = oscillating
        self.schedule_update_ha_state()


class AsyncDemoPercentageFan(BaseDemoFan, FanEntity):
    """An async demonstration fan component that uses percentages."""

    @property
    def func_ejvmg8sz(self) -> Optional[int]:
        """Return the current speed."""
        return self._percentage

    @property
    def func_epd76cpl(self) -> int:
        """Return the number of speeds the fan supports."""
        return 3

    async def func_b8m01717(self, percentage: int) -> None:
        """Set the speed of the fan, as a percentage."""
        self._percentage = percentage
        self._preset_mode = None
        self.async_write_ha_state()

    @property
    def func_e6618pap(self) -> Optional[str]:
        """Return the current preset mode, e.g., auto, smart, interval, favorite."""
        return self._preset_mode

    @property
    def func_meiy4p9y(self) -> Optional[List[str]]:
        """Return a list of available preset modes."""
        return self._preset_modes

    async def func_adiugwse(self, preset_mode: str) -> None:
        """Set new preset mode."""
        self._preset_mode = preset_mode
        self._percentage = None
        self.async_write_ha_state()

    async def func_n97co5rw(
        self,
        percentage: Optional[int] = None,
        preset_mode: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Turn on the entity."""
        if preset_mode:
            await self.async_set_preset_mode(preset_mode)
            return
        if percentage is None:
            percentage = 67
        await self.async_set_percentage(percentage)

    async def func_sqf8xvjm(self, **kwargs: Any) -> None:
        """Turn off the entity."""
        await self.async_oscillate(False)
        await self.async_set_percentage(0)

    async def func_ewzdlh52(self, direction: Literal['forward', 'reverse']) -> None:
        """Set the direction of the fan."""
        self._direction = direction
        self.async_write_ha_state()

    async def func_wionu3ri(self, oscillating: bool) -> None:
        """Set oscillation."""
        self._oscillating = oscillating
        self.async_write_ha_state()

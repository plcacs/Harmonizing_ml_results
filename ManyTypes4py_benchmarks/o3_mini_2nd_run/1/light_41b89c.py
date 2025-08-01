"""Demo light platform that implements lights."""
from __future__ import annotations
import random
from typing import Any, Optional, Set, Tuple, List
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_EFFECT,
    ATTR_HS_COLOR,
    ATTR_RGBW_COLOR,
    ATTR_RGBWW_COLOR,
    ATTR_WHITE,
    DEFAULT_MAX_KELVIN,
    DEFAULT_MIN_KELVIN,
    ColorMode,
    LightEntity,
    LightEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import DOMAIN

LIGHT_COLORS: List[Tuple[int, int]] = [(56, 86), (345, 75)]
LIGHT_EFFECT_LIST: List[str] = ['rainbow', 'none']
LIGHT_TEMPS: List[int] = [4166, 2631]
SUPPORT_DEMO: Set[ColorMode] = {ColorMode.HS, ColorMode.COLOR_TEMP}
SUPPORT_DEMO_HS_WHITE: Set[ColorMode] = {ColorMode.HS, ColorMode.WHITE}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the demo light platform."""
    async_add_entities(
        [
            DemoLight(
                available=True,
                effect_list=LIGHT_EFFECT_LIST,
                effect=LIGHT_EFFECT_LIST[0],
                device_name='Bed Light',
                state=False,
                unique_id='light_1',
            ),
            DemoLight(
                available=True,
                ct=LIGHT_TEMPS[1],
                device_name='Ceiling Lights',
                state=True,
                unique_id='light_2',
            ),
            DemoLight(
                available=True,
                hs_color=LIGHT_COLORS[1],
                device_name='Kitchen Lights',
                state=True,
                unique_id='light_3',
            ),
            DemoLight(
                available=True,
                ct=LIGHT_TEMPS[1],
                device_name='Office RGBW Lights',
                rgbw_color=(255, 0, 0, 255),
                state=True,
                supported_color_modes={ColorMode.RGBW},
                unique_id='light_4',
            ),
            DemoLight(
                available=True,
                device_name='Living Room RGBWW Lights',
                rgbww_color=(255, 0, 0, 255, 0),
                state=True,
                supported_color_modes={ColorMode.RGBWW},
                unique_id='light_5',
            ),
            DemoLight(
                available=True,
                device_name='Entrance Color + White Lights',
                hs_color=LIGHT_COLORS[1],
                state=True,
                supported_color_modes=SUPPORT_DEMO_HS_WHITE,
                unique_id='light_6',
            ),
        ]
    )


class DemoLight(LightEntity):
    """Representation of a demo light."""

    _attr_has_entity_name = True
    _attr_name: Optional[str] = None
    _attr_should_poll = False
    _attr_max_color_temp_kelvin: int = DEFAULT_MAX_KELVIN
    _attr_min_color_temp_kelvin: int = DEFAULT_MIN_KELVIN

    def __init__(
        self,
        unique_id: str,
        device_name: str,
        state: bool,
        available: bool = False,
        brightness: int = 180,
        ct: Optional[int] = None,
        effect_list: Optional[List[str]] = None,
        effect: Optional[str] = None,
        hs_color: Optional[Tuple[int, int]] = None,
        rgbw_color: Optional[Tuple[int, int, int, int]] = None,
        rgbww_color: Optional[Tuple[int, int, int, int, int]] = None,
        supported_color_modes: Optional[Set[ColorMode]] = None,
    ) -> None:
        """Initialize the light."""
        self._available: bool = True
        self._brightness: int = brightness
        self._ct: int = ct or random.choice(LIGHT_TEMPS)
        self._effect: Optional[str] = effect
        self._effect_list: Optional[List[str]] = effect_list
        self._hs_color: Optional[Tuple[int, int]] = hs_color
        self._rgbw_color: Optional[Tuple[int, int, int, int]] = rgbw_color
        self._rgbww_color: Optional[Tuple[int, int, int, int, int]] = rgbww_color
        self._state: bool = state
        self._unique_id: str = unique_id

        if hs_color:
            self._color_mode: ColorMode = ColorMode.HS
        elif rgbw_color:
            self._color_mode = ColorMode.RGBW
        elif rgbww_color:
            self._color_mode = ColorMode.RGBWW
        else:
            self._color_mode = ColorMode.COLOR_TEMP

        if not supported_color_modes:
            supported_color_modes = SUPPORT_DEMO
        self._color_modes: Set[ColorMode] = supported_color_modes

        if self._effect_list is not None:
            self._attr_supported_features |= LightEntityFeature.EFFECT

        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, self.unique_id)}, name=device_name)

    @property
    def unique_id(self) -> str:
        """Return unique ID for light."""
        return self._unique_id

    @property
    def available(self) -> bool:
        """Return availability."""
        return self._available

    @property
    def brightness(self) -> int:
        """Return the brightness of this light between 0..255."""
        return self._brightness

    @property
    def color_mode(self) -> ColorMode:
        """Return the color mode of the light."""
        return self._color_mode

    @property
    def hs_color(self) -> Optional[Tuple[int, int]]:
        """Return the hs color value."""
        return self._hs_color

    @property
    def rgbw_color(self) -> Optional[Tuple[int, int, int, int]]:
        """Return the rgbw color value."""
        return self._rgbw_color

    @property
    def rgbww_color(self) -> Optional[Tuple[int, int, int, int, int]]:
        """Return the rgbww color value."""
        return self._rgbww_color

    @property
    def color_temp_kelvin(self) -> int:
        """Return the color temperature value in Kelvin."""
        return self._ct

    @property
    def effect_list(self) -> Optional[List[str]]:
        """Return the list of supported effects."""
        return self._effect_list

    @property
    def effect(self) -> Optional[str]:
        """Return the current effect."""
        return self._effect

    @property
    def is_on(self) -> bool:
        """Return true if light is on."""
        return self._state

    @property
    def supported_color_modes(self) -> Set[ColorMode]:
        """Flag supported color modes."""
        return self._color_modes

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the light on."""
        self._state = True
        if ATTR_BRIGHTNESS in kwargs:
            self._brightness = kwargs[ATTR_BRIGHTNESS]
        if ATTR_COLOR_TEMP_KELVIN in kwargs:
            self._color_mode = ColorMode.COLOR_TEMP
            self._ct = kwargs[ATTR_COLOR_TEMP_KELVIN]
        if ATTR_EFFECT in kwargs:
            self._effect = kwargs[ATTR_EFFECT]
        if ATTR_HS_COLOR in kwargs:
            self._color_mode = ColorMode.HS
            self._hs_color = kwargs[ATTR_HS_COLOR]
        if ATTR_RGBW_COLOR in kwargs:
            self._color_mode = ColorMode.RGBW
            self._rgbw_color = kwargs[ATTR_RGBW_COLOR]
        if ATTR_RGBWW_COLOR in kwargs:
            self._color_mode = ColorMode.RGBWW
            self._rgbww_color = kwargs[ATTR_RGBWW_COLOR]
        if ATTR_WHITE in kwargs:
            self._color_mode = ColorMode.WHITE
            self._brightness = kwargs[ATTR_WHITE]
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the light off."""
        self._state = False
        self.async_write_ha_state()
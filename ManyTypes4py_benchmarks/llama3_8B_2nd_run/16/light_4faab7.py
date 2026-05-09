"""Support for Z-Wave lights."""
from __future__ import annotations
from typing import Any, cast
from zwave_js_server.client import Client as ZwaveClient
from zwave_js_server.const import TARGET_VALUE_PROPERTY, TRANSITION_DURATION_OPTION, CommandClass
from zwave_js_server.const.command_class.color_switch import COLOR_SWITCH_COMBINED_AMBER, COLOR_SWITCH_COMBINED_BLUE, COLOR_SWITCH_COMBINED_COLD_WHITE, COLOR_SWITCH_COMBINED_CYAN, COLOR_SWITCH_COMBINED_GREEN, COLOR_SWITCH_COMBINED_PURPLE, COLOR_SWITCH_COMBINED_RED, COLOR_SWITCH_COMBINED_WARM_WHITE, CURRENT_COLOR_PROPERTY, TARGET_COLOR_PROPERTY, ColorComponent
from zwave_js_server.const.command_class.multilevel_switch import SET_TO_PREVIOUS_VALUE
from zwave_js_server.model.driver import Driver
from zwave_js_server.model.value import Value
from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_COLOR_TEMP_KELVIN, ATTR_HS_COLOR, ATTR_RGBW_COLOR, ATTR_TRANSITION, DOMAIN as LIGHT_DOMAIN, ColorMode, LightEntity, LightEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import color as color_util
from .const import DATA_CLIENT, DOMAIN
from .discovery import ZwaveDiscoveryInfo
from .entity import ZWaveBaseEntity
PARALLEL_UPDATES = 0
MULTI_COLOR_MAP = {ColorComponent.WARM_WHITE: COLOR_SWITCH_COMBINED_WARM_WHITE, ColorComponent.COLD_WHITE: COLOR_SWITCH_COMBINED_COLD_WHITE, ColorComponent.RED: COLOR_SWITCH_COMBINED_RED, ColorComponent.GREEN: COLOR_SWITCH_COMBINED_GREEN, ColorComponent.BLUE: COLOR_SWITCH_COMBINED_BLUE, ColorComponent.AMBER: COLOR_SWITCH_COMBINED_AMBER, ColorComponent.CYAN: COLOR_SWITCH_COMBINED_CYAN, ColorComponent.PURPLE: COLOR_SWITCH_COMBINED_PURPLE}
MIN_MIREDS = 153
MAX_MIREDS = 370

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up Z-Wave Light from Config Entry."""
    client = config_entry.runtime_data[DATA_CLIENT]

    @callback
    def async_add_light(info: ZwaveDiscoveryInfo) -> None:
        """Add Z-Wave Light."""
        driver: Driver = client.driver
        assert driver is not None
        if info.platform_hint == 'color_onoff':
            async_add_entities([ZwaveColorOnOffLight(config_entry, driver, info)])
        else:
            async_add_entities([ZwaveLight(config_entry, driver, info)])
    config_entry.async_on_unload(async_dispatcher_connect(hass, f'{DOMAIN}_{config_entry.entry_id}_add_{LIGHT_DOMAIN}', async_add_light))

def byte_to_zwave_brightness(value: int) -> int:
    """Convert brightness in 0-255 scale to 0-99 scale.

    value -- (int) Brightness byte value from 0-255.
    """
    if value > 0:
        return max(1, round(value / 255 * 99))
    return 0

class ZwaveLight(ZWaveBaseEntity, LightEntity):
    """Representation of a Z-Wave light."""
    _attr_min_color_temp_kelvin: int
    _attr_max_color_temp_kelvin: int

    def __init__(self, config_entry: ConfigEntry, driver: Driver, info: ZwaveDiscoveryInfo) -> None:
        """Initialize the light."""
        super().__init__(config_entry, driver, info)
        self._supports_color: bool
        self._supports_rgbw: bool
        self._supports_color_temp: bool
        self._color_mode: ColorMode | None
        self._hs_color: tuple[float, float] | None
        self._rgbw_color: tuple[int, int, int, int] | None
        self._color_temp_kelvin: int | None
        self._warm_white: Value | None
        self._cold_white: Value | None
        self._target_brightness: Value | None
        self._supported_color_modes: set[ColorMode]
        self._calculate_color_support()
        self._calculate_color_values()

    @property
    def brightness(self) -> int | None:
        """Return the brightness of this light between 0..255.

        Z-Wave multilevel switches use a range of [0, 99] to control brightness.
        """
        if self.info.primary_value.value is None:
            return None
        return round(cast(int, self.info.primary_value.value) / 99 * 255)

    @property
    def color_mode(self) -> ColorMode | None:
        """Return the color mode of the light."""
        return self._color_mode

    @property
    def hs_color(self) -> tuple[float, float] | None:
        """Return the hs color."""
        return self._hs_color

    @property
    def rgbw_color(self) -> tuple[int, int, int, int] | None:
        """Return the RGBW color."""
        return self._rgbw_color

    @property
    def color_temp_kelvin(self) -> int | None:
        """Return the color temperature value in Kelvin."""
        return self._color_temp_kelvin

    @property
    def supported_color_modes(self) -> set[ColorMode]:
        """Flag supported features."""
        return self._supported_color_modes

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the device on."""
        transition: int | None = kwargs.get(ATTR_TRANSITION)
        brightness: int | None = kwargs.get(ATTR_BRIGHTNESS)
        hs_color: tuple[float, float] | None = kwargs.get(ATTR_HS_COLOR)
        color_temp_kelvin: int | None = kwargs.get(ATTR_COLOR_TEMP_KELVIN)
        rgbw_color: tuple[int, int, int, int] | None = kwargs.get(ATTR_RGBW_COLOR)
        new_colors = self._get_new_colors(hs_color, color_temp_kelvin, rgbw_color)
        if new_colors is not None:
            await self._async_set_colors(new_colors, transition)
        await self._async_set_brightness(brightness, transition)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the light off."""
        await self._async_set_brightness(0, kwargs.get(ATTR_TRANSITION))

    def _get_new_colors(self, hs_color: tuple[float, float] | None, color_temp_kelvin: int | None, rgbw_color: tuple[int, int, int, int] | None) -> dict[ColorComponent, int] | None:
        ...

    async def _async_set_colors(self, colors: dict[ColorComponent, int], transition: int | None) -> None:
        ...

    async def _async_set_brightness(self, brightness: int | None, transition: int | None) -> None:
        ...

    @callback
    def _calculate_color_support(self) -> None:
        ...

    @callback
    def _calculate_color_values(self) -> None:
        ...

class ZwaveColorOnOffLight(ZwaveLight):
    """Representation of a colored Z-Wave light with an optional binary switch to turn on/off.

    Dimming for RGB lights is realized by scaling the color channels.
    """

    def __init__(self, config_entry: ConfigEntry, driver: Driver, info: ZwaveDiscoveryInfo) -> None:
        """Initialize the light."""
        super().__init__(config_entry, driver, info)
        self._last_on_color: dict[ColorComponent, int] | None
        self._last_brightness: int | None

    @property
    def brightness(self) -> int | None:
        """Return the brightness of this light between 0..255.

        Z-Wave multilevel switches use a range of [0, 99] to control brightness.
        """
        if self.info.primary_value.value is None:
            return None
        if self._target_brightness and self.info.primary_value.value is False:
            return 0
        color_values = [v.value for v in self._get_color_values() if v is not None and v.value is not None]
        return max(color_values) if color_values else 0

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the device on."""
        transition: int | None = kwargs.get(ATTR_TRANSITION)
        brightness: int | None = kwargs.get(ATTR_BRIGHTNESS)
        hs_color: tuple[float, float] | None = kwargs.get(ATTR_HS_COLOR)
        new_colors = None
        scale: float | None = None
        if brightness is None and hs_color is None:
            if self._last_on_color is not None:
                if self._target_brightness:
                    await self._async_set_brightness(None, transition)
                    return
                new_colors = self._last_on_color
            elif self._supports_color:
                new_colors = {ColorComponent.RED: 255, ColorComponent.GREEN: 255, ColorComponent.BLUE: 255}
        elif brightness is not None:
            if self.color_mode == ColorMode.HS:
                scale = brightness / 255
            if self._last_on_color is not None and None not in self._last_on_color.values():
                old_brightness = max(self._last_on_color.values())
                new_scale = brightness / old_brightness
                scale = new_scale
                new_colors = {}
                for color, value in self._last_on_color.items():
                    new_colors[color] = round(value * new_scale)
            elif hs_color is None and self._color_mode == ColorMode.HS:
                hs_color = self._hs_color
        elif hs_color is not None and brightness is None:
            current_brightness = self.brightness
            if current_brightness == 0 and self._last_brightness is not None:
                scale = self._last_brightness / 255
            elif current_brightness is not None:
                scale = current_brightness / 255
        self._last_on_color = None
        if new_colors is None:
            new_colors = self._get_new_colors(hs_color, None, None, brightness_scale=scale)
        if new_colors is not None:
            await self._async_set_colors(new_colors, transition)
        await self._async_set_brightness(brightness, transition)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the light off."""
        self._last_brightness = self.brightness
        if self._current_color and isinstance(self._current_color.value, dict):
            red = self._current_color.value.get(COLOR_SWITCH_COMBINED_RED)
            green = self._current_color.value.get(COLOR_SWITCH_COMBINED_GREEN)
            blue = self._current_color.value.get(COLOR_SWITCH_COMBINED_BLUE)
            last_color = {}
            if red is not None:
                last_color[ColorComponent.RED] = red
            if green is not None:
                last_color[ColorComponent.GREEN] = green
            if blue is not None:
                last_color[ColorComponent.BLUE] = blue
            if last_color:
                self._last_on_color = last_color
        if self._target_brightness:
            await self._async_set_brightness(0, kwargs.get(ATTR_TRANSITION))
        else:
            colors = {ColorComponent.RED: 0, ColorComponent.GREEN: 0, ColorComponent.BLUE: 0}
            await self._async_set_colors(colors, kwargs.get(ATTR_TRANSITION))
"""Support for Z-Wave lights."""
from __future__ import annotations
from typing import Any, Optional, Dict, Union, Set, Tuple, cast
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
MULTI_COLOR_MAP: Dict[ColorComponent, str] = {
    ColorComponent.WARM_WHITE: COLOR_SWITCH_COMBINED_WARM_WHITE,
    ColorComponent.COLD_WHITE: COLOR_SWITCH_COMBINED_COLD_WHITE,
    ColorComponent.RED: COLOR_SWITCH_COMBINED_RED,
    ColorComponent.GREEN: COLOR_SWITCH_COMBINED_GREEN,
    ColorComponent.BLUE: COLOR_SWITCH_COMBINED_BLUE,
    ColorComponent.AMBER: COLOR_SWITCH_COMBINED_AMBER,
    ColorComponent.CYAN: COLOR_SWITCH_COMBINED_CYAN,
    ColorComponent.PURPLE: COLOR_SWITCH_COMBINED_PURPLE,
}
MIN_MIREDS = 153
MAX_MIREDS = 370

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up Z-Wave Light from Config Entry."""
    client: ZwaveClient = config_entry.runtime_data[DATA_CLIENT]

    @callback
    def async_add_light(info: ZwaveDiscoveryInfo) -> None:
        """Add Z-Wave Light."""
        driver: Optional[Driver] = client.driver
        assert driver is not None
        if info.platform_hint == 'color_onoff':
            async_add_entities([ZwaveColorOnOffLight(config_entry, driver, info)])
        else:
            async_add_entities([ZwaveLight(config_entry, driver, info)])

    config_entry.async_on_unload(
        async_dispatcher_connect(
            hass,
            f'{DOMAIN}_{config_entry.entry_id}_add_{LIGHT_DOMAIN}',
            async_add_light
        )
    )

def byte_to_zwave_brightness(value: int) -> int:
    """Convert brightness in 0-255 scale to 0-99 scale.

    `value` -- (int) Brightness byte value from 0-255.
    """
    if value > 0:
        return max(1, round(value / 255 * 99))
    return 0

class ZwaveLight(ZWaveBaseEntity, LightEntity):
    """Representation of a Z-Wave light."""
    _attr_min_color_temp_kelvin: int = 2700
    _attr_max_color_temp_kelvin: int = 6500

    def __init__(self, config_entry: ConfigEntry, driver: Driver, info: ZwaveDiscoveryInfo) -> None:
        """Initialize the light."""
        super().__init__(config_entry, driver, info)
        self._supports_color: bool = False
        self._supports_rgbw: bool = False
        self._supports_color_temp: bool = False
        self._supports_dimming: bool = False
        self._color_mode: Optional[ColorMode] = None
        self._hs_color: Optional[Tuple[float, float]] = None
        self._rgbw_color: Optional[Tuple[int, int, int, int]] = None
        self._color_temp: Optional[int] = None
        self._warm_white: Optional[Value] = self.get_zwave_value(
            TARGET_COLOR_PROPERTY, CommandClass.SWITCH_COLOR, value_property_key=ColorComponent.WARM_WHITE
        )
        self._cold_white: Optional[Value] = self.get_zwave_value(
            TARGET_COLOR_PROPERTY, CommandClass.SWITCH_COLOR, value_property_key=ColorComponent.COLD_WHITE
        )
        self._supported_color_modes: Set[ColorMode] = set()
        self._target_brightness: Optional[Value] = None
        if self.info.primary_value.command_class == CommandClass.SWITCH_BINARY:
            self._target_brightness = self.get_zwave_value(
                TARGET_VALUE_PROPERTY, CommandClass.SWITCH_BINARY, add_to_watched_value_ids=False
            )
            self._supports_dimming = False
        elif self.info.primary_value.command_class == CommandClass.SWITCH_MULTILEVEL:
            self._target_brightness = self.get_zwave_value(
                TARGET_VALUE_PROPERTY, CommandClass.SWITCH_MULTILEVEL, add_to_watched_value_ids=False
            )
            self._supports_dimming = True
        elif self.info.primary_value.command_class == CommandClass.BASIC:
            self._attr_name = self.generate_name(include_value_name=True, alternate_value_name='Basic')
            self._target_brightness = self.get_zwave_value(
                TARGET_VALUE_PROPERTY, CommandClass.BASIC, add_to_watched_value_ids=False
            )
            self._supports_dimming = True
        self._current_color: Optional[Value] = self.get_zwave_value(
            CURRENT_COLOR_PROPERTY, CommandClass.SWITCH_COLOR, value_property_key=None
        )
        self._target_color: Optional[Value] = self.get_zwave_value(
            TARGET_COLOR_PROPERTY, CommandClass.SWITCH_COLOR, add_to_watched_value_ids=False
        )
        self._calculate_color_support()
        if self._supports_rgbw:
            self._supported_color_modes.add(ColorMode.RGBW)
        elif self._supports_color:
            self._supported_color_modes.add(ColorMode.HS)
        if self._supports_color_temp:
            self._supported_color_modes.add(ColorMode.COLOR_TEMP)
        if not self._supported_color_modes:
            self._supported_color_modes.add(ColorMode.BRIGHTNESS)
        self._calculate_color_values()
        self.supports_brightness_transition: bool = bool(
            self._target_brightness is not None and TRANSITION_DURATION_OPTION in self._target_brightness.metadata.value_change_options
        )
        self.supports_color_transition: bool = bool(
            self._target_color is not None and TRANSITION_DURATION_OPTION in self._target_color.metadata.value_change_options
        )
        if self.supports_brightness_transition or self.supports_color_transition:
            self._attr_supported_features |= LightEntityFeature.TRANSITION
        self._set_optimistic_state: bool = False

    @callback
    def on_value_update(self) -> None:
        """Call when a watched value is added or updated."""
        self._calculate_color_values()

    @property
    def brightness(self) -> Optional[int]:
        """Return the brightness of this light between 0..255.

        Z-Wave multilevel switches use a range of [0, 99] to control brightness.
        """
        if self.info.primary_value.value is None:
            return None
        return round(cast(int, self.info.primary_value.value) / 99 * 255)

    @property
    def color_mode(self) -> Optional[ColorMode]:
        """Return the color mode of the light."""
        return self._color_mode

    @property
    def is_on(self) -> Optional[bool]:
        """Return true if device is on (brightness above 0)."""
        if self._set_optimistic_state:
            self._set_optimistic_state = False
            return True
        brightness = self.brightness
        return brightness > 0 if brightness is not None else None

    @property
    def hs_color(self) -> Optional[Tuple[float, float]]:
        """Return the hs color."""
        return self._hs_color

    @property
    def rgbw_color(self) -> Optional[Tuple[int, int, int, int]]:
        """Return the RGBW color."""
        return self._rgbw_color

    @property
    def color_temp_kelvin(self) -> Optional[int]:
        """Return the color temperature value in Kelvin."""
        return self._color_temp

    @property
    def supported_color_modes(self) -> Set[ColorMode]:
        """Flag supported features."""
        return self._supported_color_modes

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the device on."""
        transition: Optional[float] = kwargs.get(ATTR_TRANSITION)
        brightness: Optional[int] = kwargs.get(ATTR_BRIGHTNESS)
        hs_color: Optional[Tuple[float, float]] = kwargs.get(ATTR_HS_COLOR)
        color_temp_k: Optional[int] = kwargs.get(ATTR_COLOR_TEMP_KELVIN)
        rgbw: Optional[Tuple[int, int, int, int]] = kwargs.get(ATTR_RGBW_COLOR)
        new_colors: Optional[Dict[ColorComponent, int]] = self._get_new_colors(hs_color, color_temp_k, rgbw)
        if new_colors is not None:
            await self._async_set_colors(new_colors, transition)
        await self._async_set_brightness(brightness, transition)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the light off."""
        await self._async_set_brightness(0, kwargs.get(ATTR_TRANSITION))

    def _get_new_colors(
        self,
        hs_color: Optional[Tuple[float, float]],
        color_temp_k: Optional[int],
        rgbw: Optional[Tuple[int, int, int, int]],
        brightness_scale: Optional[float] = None
    ) -> Optional[Dict[ColorComponent, int]]:
        """Determine the new color dict to set."""
        if hs_color is not None and self._supports_color:
            red, green, blue = color_util.color_hs_to_RGB(*hs_color)
            if brightness_scale is not None:
                red = round(red * brightness_scale)
                green = round(green * brightness_scale)
                blue = round(blue * brightness_scale)
            colors = {ColorComponent.RED: red, ColorComponent.GREEN: green, ColorComponent.BLUE: blue}
            if self._supports_color_temp:
                colors[ColorComponent.WARM_WHITE] = 0
                colors[ColorComponent.COLD_WHITE] = 0
            return colors
        if color_temp_k is not None and self._supports_color_temp:
            color_temp = color_util.color_temperature_kelvin_to_mired(color_temp_k)
            cold = max(0, min(255, round((MAX_MIREDS - color_temp) / (MAX_MIREDS - MIN_MIREDS) * 255)))
            warm = 255 - cold
            colors = {ColorComponent.WARM_WHITE: warm, ColorComponent.COLD_WHITE: cold}
            if self._supports_color:
                colors[ColorComponent.RED] = 0
                colors[ColorComponent.GREEN] = 0
                colors[ColorComponent.BLUE] = 0
            return colors
        if rgbw is not None and self._supports_rgbw:
            rgbw_channels = {ColorComponent.RED: rgbw[0], ColorComponent.GREEN: rgbw[1], ColorComponent.BLUE: rgbw[2]}
            if self._warm_white:
                rgbw_channels[ColorComponent.WARM_WHITE] = rgbw[3]
            if self._cold_white:
                rgbw_channels[ColorComponent.COLD_WHITE] = rgbw[3]
            return rgbw_channels
        return None

    async def _async_set_colors(
        self,
        colors: Dict[ColorComponent, int],
        transition: Optional[float] = None
    ) -> None:
        """Set (multiple) defined colors to given value(s)."""
        combined_color_val: Value = cast(Value, self.get_zwave_value('targetColor', CommandClass.SWITCH_COLOR, value_property_key=None))
        zwave_transition: Optional[Dict[str, Union[str, int]]] = None
        if self.supports_color_transition:
            if transition is not None:
                zwave_transition = {TRANSITION_DURATION_OPTION: f'{int(transition)}s'}
            else:
                zwave_transition = {TRANSITION_DURATION_OPTION: 'default'}
        colors_dict: Dict[str, int] = {}
        for color, value in colors.items():
            color_name = MULTI_COLOR_MAP[color]
            colors_dict[color_name] = value
        await self._async_set_value(combined_color_val, colors_dict, zwave_transition)

    async def _async_set_brightness(
        self,
        brightness: Optional[int],
        transition: Optional[float] = None
    ) -> None:
        """Set new brightness to light."""
        if not self._target_brightness:
            return
        if brightness is None:
            zwave_brightness = SET_TO_PREVIOUS_VALUE
        else:
            zwave_brightness = byte_to_zwave_brightness(brightness)
        zwave_transition: Optional[Dict[str, Union[str, int]]] = None
        if self.supports_brightness_transition:
            if transition is not None:
                zwave_transition = {TRANSITION_DURATION_OPTION: f'{int(transition)}s'}
            else:
                zwave_transition = {TRANSITION_DURATION_OPTION: 'default'}
        if self._supports_dimming:
            await self._async_set_value(self._target_brightness, zwave_brightness, zwave_transition)
        else:
            await self._async_set_value(self._target_brightness, zwave_brightness > 0, zwave_transition)
        if zwave_brightness == SET_TO_PREVIOUS_VALUE and self.info.primary_value.command_class in (CommandClass.BASIC, CommandClass.SWITCH_MULTILEVEL):
            self._set_optimistic_state = True
            self.async_write_ha_state()

    @callback
    def _get_color_values(self) -> Tuple[Optional[Value], Optional[Value], Optional[Value], Optional[Value], Optional[Value]]:
        """Get light colors."""
        red_val = self.get_zwave_value(CURRENT_COLOR_PROPERTY, CommandClass.SWITCH_COLOR, value_property_key=ColorComponent.RED.value)
        green_val = self.get_zwave_value(CURRENT_COLOR_PROPERTY, CommandClass.SWITCH_COLOR, value_property_key=ColorComponent.GREEN.value)
        blue_val = self.get_zwave_value(CURRENT_COLOR_PROPERTY, CommandClass.SWITCH_COLOR, value_property_key=ColorComponent.BLUE.value)
        ww_val = self.get_zwave_value(CURRENT_COLOR_PROPERTY, CommandClass.SWITCH_COLOR, value_property_key=ColorComponent.WARM_WHITE.value)
        cw_val = self.get_zwave_value(CURRENT_COLOR_PROPERTY, CommandClass.SWITCH_COLOR, value_property_key=ColorComponent.COLD_WHITE.value)
        return (red_val, green_val, blue_val, ww_val, cw_val)

    @callback
    def _calculate_color_support(self) -> None:
        """Calculate light colors."""
        red, green, blue, warm_white, cool_white = self._get_color_values()
        if red and green and blue:
            self._supports_color = True
        if warm_white and cool_white:
            self._supports_color_temp = True
        elif red and green and blue and warm_white or cool_white:
            self._supports_rgbw = True

    @callback
    def _calculate_color_values(self) -> None:
        """Calculate light colors."""
        red_val, green_val, blue_val, ww_val, cw_val = self._get_color_values()
        if self._current_color and isinstance(self._current_color.value, dict):
            multi_color = self._current_color.value
        else:
            multi_color = {}
        if self.supported_color_modes == {ColorMode.BRIGHTNESS}:
            self._color_mode = ColorMode.BRIGHTNESS
        else:
            self._color_mode = ColorMode.UNKNOWN
        if red_val and green_val and blue_val:
            red = multi_color.get(COLOR_SWITCH_COMBINED_RED, red_val.value)
            green = multi_color.get(COLOR_SWITCH_COMBINED_GREEN, green_val.value)
            blue = multi_color.get(COLOR_SWITCH_COMBINED_BLUE, blue_val.value)
            if None not in (red, green, blue):
                self._hs_color = color_util.color_RGB_to_hs(red, green, blue)
                self._color_mode = ColorMode.HS
        if ww_val and cw_val:
            warm_white = multi_color.get(COLOR_SWITCH_COMBINED_WARM_WHITE, ww_val.value)
            cold_white = multi_color.get(COLOR_SWITCH_COMBINED_COLD_WHITE, cw_val.value)
            if cold_white or warm_white:
                self._color_temp = color_util.color_temperature_mired_to_kelvin(
                    MAX_MIREDS - cold_white / 255 * (MAX_MIREDS - MIN_MIREDS)
                )
                self._color_mode = ColorMode.COLOR_TEMP
            else:
                self._color_temp = None
        elif red_val and green_val and blue_val and ww_val:
            white = multi_color.get(COLOR_SWITCH_COMBINED_WARM_WHITE, ww_val.value)
            self._rgbw_color = (red, green, blue, white)
            self._color_mode = ColorMode.RGBW
        elif cw_val:
            self._supports_rgbw = True
            white = multi_color.get(COLOR_SWITCH_COMBINED_COLD_WHITE, cw_val.value)
            self._rgbw_color = (red, green, blue, white)
            self._color_mode = ColorMode.RGBW

class ZwaveColorOnOffLight(ZwaveLight):
    """Representation of a colored Z-Wave light with an optional binary switch to turn on/off.

    Dimming for RGB lights is realized by scaling the color channels.
    """

    def __init__(self, config_entry: ConfigEntry, driver: Driver, info: ZwaveDiscoveryInfo) -> None:
        """Initialize the light."""
        super().__init__(config_entry, driver, info)
        self._last_on_color: Optional[Dict[ColorComponent, int]] = None
        self._last_brightness: Optional[int] = None

    @property
    def brightness(self) -> Optional[int]:
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
        if kwargs.get(ATTR_RGBW_COLOR) is not None or kwargs.get(ATTR_COLOR_TEMP_KELVIN) is not None:
            await super().async_turn_on(**kwargs)
            return
        transition: Optional[float] = kwargs.get(ATTR_TRANSITION)
        brightness: Optional[int] = kwargs.get(ATTR_BRIGHTNESS)
        hs_color: Optional[Tuple[float, float]] = kwargs.get(ATTR_HS_COLOR)
        new_colors: Optional[Dict[ColorComponent, int]] = None
        scale: Optional[float] = None
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
            new_colors = self._get_new_colors(hs_color=hs_color, color_temp_k=None, rgbw=None, brightness_scale=scale)
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
            last_color: Dict[ColorComponent, int] = {}
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

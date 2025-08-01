from __future__ import annotations
import ast
import logging
from typing import Any, Final, List, Tuple, Union, Optional, Dict, Coroutine
from flux_led.const import MultiColorEffects
from flux_led.protocol import MusicMode
from flux_led.utils import rgbcw_brightness, rgbcw_to_rgbwc, rgbw_brightness
import voluptuous as vol
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_EFFECT,
    ATTR_RGB_COLOR,
    ATTR_RGBW_COLOR,
    ATTR_RGBWW_COLOR,
    ATTR_WHITE,
    LightEntity,
    LightEntityFeature,
)
from homeassistant.const import CONF_EFFECT
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, entity_platform
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import VolDictType
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.config_entries import ConfigEntry
from .const import (
    CONF_COLORS,
    CONF_CUSTOM_EFFECT_COLORS,
    CONF_CUSTOM_EFFECT_SPEED_PCT,
    CONF_CUSTOM_EFFECT_TRANSITION,
    CONF_SPEED_PCT,
    CONF_TRANSITION,
    DEFAULT_EFFECT_SPEED,
    MIN_CCT_BRIGHTNESS,
    MIN_RGB_BRIGHTNESS,
    MULTI_BRIGHTNESS_COLOR_MODES,
    TRANSITION_GRADUAL,
    TRANSITION_JUMP,
    TRANSITION_STROBE,
)
from .coordinator import FluxLedConfigEntry, FluxLedUpdateCoordinator
from .entity import FluxOnOffEntity
from .util import (
    _effect_brightness,
    _flux_color_mode_to_hass,
    _hass_color_modes,
    _min_rgb_brightness,
    _min_rgbw_brightness,
    _min_rgbwc_brightness,
    _str_to_multi_color_effect,
)

_LOGGER: Final = logging.getLogger(__name__)

MODE_ATTRS: Final = {ATTR_EFFECT, ATTR_COLOR_TEMP_KELVIN, ATTR_RGB_COLOR, ATTR_RGBW_COLOR, ATTR_RGBWW_COLOR, ATTR_WHITE}
ATTR_FOREGROUND_COLOR: Final = 'foreground_color'
ATTR_BACKGROUND_COLOR: Final = 'background_color'
ATTR_SENSITIVITY: Final = 'sensitivity'
ATTR_LIGHT_SCREEN: Final = 'light_screen'
COLOR_TEMP_WARM_VS_COLD_WHITE_CUT_OFF: Final = 285
EFFECT_CUSTOM: Final = 'custom'
SERVICE_CUSTOM_EFFECT: Final = 'set_custom_effect'
SERVICE_SET_ZONES: Final = 'set_zones'
SERVICE_SET_MUSIC_MODE: Final = 'set_music_mode'
CUSTOM_EFFECT_DICT: Final = {
    vol.Required(CONF_COLORS): vol.All(
        cv.ensure_list,
        vol.Length(min=1, max=16),
        [vol.All(vol.Coerce(tuple), vol.ExactSequence((cv.byte, cv.byte, cv.byte)))]
    ),
    vol.Optional(CONF_SPEED_PCT, default=50): vol.All(vol.Coerce(int), vol.Range(min=0, max=100)),
    vol.Optional(CONF_TRANSITION, default=TRANSITION_GRADUAL): vol.All(cv.string, vol.In([TRANSITION_GRADUAL, TRANSITION_JUMP, TRANSITION_STROBE])),
}
SET_MUSIC_MODE_DICT: Final = {
    vol.Optional(ATTR_SENSITIVITY, default=100): vol.All(vol.Coerce(int), vol.Range(min=0, max=100)),
    vol.Optional(ATTR_BRIGHTNESS, default=100): vol.All(vol.Coerce(int), vol.Range(min=0, max=100)),
    vol.Optional(ATTR_EFFECT, default=1): vol.All(vol.Coerce(int), vol.Range(min=0, max=16)),
    vol.Optional(ATTR_LIGHT_SCREEN, default=False): bool,
    vol.Optional(ATTR_FOREGROUND_COLOR): vol.All(vol.Coerce(tuple), vol.ExactSequence((cv.byte,) * 3)),
    vol.Optional(ATTR_BACKGROUND_COLOR): vol.All(vol.Coerce(tuple), vol.ExactSequence((cv.byte,) * 3)),
}
SET_ZONES_DICT: Final = {
    vol.Required(CONF_COLORS): vol.All(
        cv.ensure_list,
        vol.Length(min=1, max=2048),
        [vol.All(vol.Coerce(tuple), vol.ExactSequence((cv.byte, cv.byte, cv.byte)))]
    ),
    vol.Optional(CONF_SPEED_PCT, default=50): vol.All(vol.Coerce(int), vol.Range(min=0, max=100)),
    vol.Optional(CONF_EFFECT, default=MultiColorEffects.STATIC.name.lower()): vol.All(cv.string, vol.In([effect.name.lower() for effect in MultiColorEffects])),
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Flux lights."""
    coordinator: FluxLedUpdateCoordinator = entry.runtime_data
    platform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(SERVICE_CUSTOM_EFFECT, CUSTOM_EFFECT_DICT, 'async_set_custom_effect')
    platform.async_register_entity_service(SERVICE_SET_ZONES, SET_ZONES_DICT, 'async_set_zones')
    platform.async_register_entity_service(SERVICE_SET_MUSIC_MODE, SET_MUSIC_MODE_DICT, 'async_set_music_mode')
    options: VolDictType = entry.options
    try:
        custom_effect_colors = ast.literal_eval(options.get(CONF_CUSTOM_EFFECT_COLORS) or '[]')
    except (ValueError, TypeError, SyntaxError, MemoryError) as ex:
        _LOGGER.warning('Could not parse custom effect colors for %s: %s', entry.unique_id, ex)
        custom_effect_colors = []
    async_add_entities(
        [
            FluxLight(
                coordinator,
                entry.unique_id or entry.entry_id,
                list(custom_effect_colors),
                options.get(CONF_CUSTOM_EFFECT_SPEED_PCT, DEFAULT_EFFECT_SPEED),
                options.get(CONF_CUSTOM_EFFECT_TRANSITION, TRANSITION_GRADUAL),
            )
        ]
    )


class FluxLight(FluxOnOffEntity, CoordinatorEntity[FluxLedUpdateCoordinator], LightEntity):
    """Representation of a Flux light."""

    _attr_name: Optional[str] = None
    _attr_supported_features: Final = LightEntityFeature.TRANSITION | LightEntityFeature.EFFECT

    def __init__(
        self,
        coordinator: FluxLedUpdateCoordinator,
        base_unique_id: str,
        custom_effect_colors: List[Tuple[int, int, int]],
        custom_effect_speed_pct: int,
        custom_effect_transition: str,
    ) -> None:
        """Initialize the light."""
        super().__init__(coordinator, base_unique_id, None)
        self._attr_min_color_temp_kelvin: int = self._device.min_temp
        self._attr_max_color_temp_kelvin: int = self._device.max_temp
        self._attr_supported_color_modes = _hass_color_modes(self._device)
        custom_effects: List[str] = []
        if custom_effect_colors:
            custom_effects.append(EFFECT_CUSTOM)
        self._attr_effect_list = [*self._device.effect_list, *custom_effects]
        self._custom_effect_colors: List[Tuple[int, int, int]] = custom_effect_colors
        self._custom_effect_speed_pct: int = custom_effect_speed_pct
        self._custom_effect_transition: str = custom_effect_transition

    @property
    def brightness(self) -> int:
        """Return the brightness of this light between 0..255."""
        return self._device.brightness

    @property
    def color_temp_kelvin(self) -> int:
        """Return the kelvin value of this light."""
        return self._device.color_temp

    @property
    def rgb_color(self) -> Tuple[int, int, int]:
        """Return the rgb color value."""
        return self._device.rgb_unscaled

    @property
    def rgbw_color(self) -> Tuple[int, int, int, int]:
        """Return the rgbw color value."""
        return self._device.rgbw

    @property
    def rgbww_color(self) -> Tuple[int, int, int, int]:
        """Return the rgbww aka rgbcw color value."""
        return self._device.rgbcw

    @property
    def color_mode(self) -> str:
        """Return the color mode of the light."""
        return _flux_color_mode_to_hass(self._device.color_mode, self._device.color_modes)

    @property
    def effect(self) -> Union[str, int]:
        """Return the current effect."""
        return self._device.effect

    async def _async_turn_on(self, **kwargs: Any) -> None:
        """Turn the specified or all lights on."""
        if self._device.requires_turn_on or not kwargs:
            if not self.is_on:
                await self._device.async_turn_on()
            if not kwargs:
                return
        if MODE_ATTRS.intersection(kwargs):
            await self._async_set_mode(**kwargs)
            return
        await self._device.async_set_brightness(self._async_brightness(**kwargs))

    async def _async_set_effect(self, effect: str, brightness: int) -> None:
        """Set an effect."""
        if effect == EFFECT_CUSTOM:
            if self._custom_effect_colors:
                await self._device.async_set_custom_pattern(self._custom_effect_colors, self._custom_effect_speed_pct, self._custom_effect_transition)
            return
        await self._device.async_set_effect(effect, self._device.speed or DEFAULT_EFFECT_SPEED, _effect_brightness(brightness))

    @callback
    def _async_brightness(self, **kwargs: Any) -> int:
        """Determine brightness from kwargs or current value."""
        brightness: Optional[int] = kwargs.get(ATTR_BRIGHTNESS)
        if brightness is None:
            brightness = self.brightness
        return max(MIN_RGB_BRIGHTNESS, brightness)

    async def _async_set_mode(self, **kwargs: Any) -> None:
        """Set an effect or color mode."""
        brightness: int = self._async_brightness(**kwargs)
        if (effect := kwargs.get(ATTR_EFFECT)):
            await self._async_set_effect(effect, brightness)
            return
        if (color_temp_kelvin := kwargs.get(ATTR_COLOR_TEMP_KELVIN)):
            if ATTR_BRIGHTNESS not in kwargs and self.color_mode in MULTI_BRIGHTNESS_COLOR_MODES:
                brightness = max(MIN_CCT_BRIGHTNESS, *self._device.rgb)
            await self._device.async_set_white_temp(color_temp_kelvin, brightness)
            return
        if (rgb := kwargs.get(ATTR_RGB_COLOR)):
            if not self._device.requires_turn_on:
                rgb = _min_rgb_brightness(rgb)
            red, green, blue = rgb
            await self._device.async_set_levels(red, green, blue, brightness=brightness)
            return
        if (rgbw := kwargs.get(ATTR_RGBW_COLOR)):
            if ATTR_BRIGHTNESS in kwargs:
                rgbw = rgbw_brightness(rgbw, brightness)
            rgbw = _min_rgbw_brightness(rgbw, self._device.rgbw)
            await self._device.async_set_levels(*rgbw)
            return
        if (rgbcw := kwargs.get(ATTR_RGBWW_COLOR)):
            if ATTR_BRIGHTNESS in kwargs:
                rgbcw = rgbcw_brightness(kwargs[ATTR_RGBWW_COLOR], brightness)
            rgbwc = rgbcw_to_rgbwc(rgbcw)
            rgbwc = _min_rgbwc_brightness(rgbwc, self._device.rgbww)
            await self._device.async_set_levels(*rgbwc)
            return
        if (white := kwargs.get(ATTR_WHITE)) is not None:
            await self._device.async_set_levels(w=white)
            return

    async def async_set_custom_effect(
        self, colors: List[Tuple[int, int, int]], speed_pct: int, transition: str
    ) -> None:
        """Set a custom effect on the bulb."""
        await self._device.async_set_custom_pattern(colors, speed_pct, transition)

    async def async_set_zones(
        self, colors: List[Tuple[int, int, int]], speed_pct: int, effect: str
    ) -> None:
        """Set a colors for zones."""
        await self._device.async_set_zones(colors, speed_pct, _str_to_multi_color_effect(effect))

    async def async_set_music_mode(
        self,
        sensitivity: int,
        brightness: int,
        effect: int,
        light_screen: bool,
        foreground_color: Optional[Tuple[int, int, int]] = None,
        background_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Configure music mode."""
        await self._async_ensure_device_on()
        await self._device.async_set_music_mode(
            sensitivity=sensitivity,
            brightness=brightness,
            mode=MusicMode.LIGHT_SCREEN.value if light_screen else None,
            effect=effect,
            foreground_color=foreground_color,
            background_color=background_color,
        )
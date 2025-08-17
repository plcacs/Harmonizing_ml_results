"""Support for Template lights."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, List, Union, Dict, cast

import voluptuous as vol

from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_EFFECT,
    ATTR_HS_COLOR,
    ATTR_RGB_COLOR,
    ATTR_RGBW_COLOR,
    ATTR_RGBWW_COLOR,
    ATTR_TRANSITION,
    DEFAULT_MAX_KELVIN,
    DEFAULT_MIN_KELVIN,
    ENTITY_ID_FORMAT,
    PLATFORM_SCHEMA as LIGHT_PLATFORM_SCHEMA,
    ColorMode,
    LightEntity,
    LightEntityFeature,
    filter_supported_color_modes,
)
from homeassistant.const import (
    CONF_ENTITY_ID,
    CONF_FRIENDLY_NAME,
    CONF_LIGHTS,
    CONF_UNIQUE_ID,
    CONF_VALUE_TEMPLATE,
    STATE_OFF,
    STATE_ON,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity import async_generate_entity_id
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.script import Script
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import color as color_util

from .const import DOMAIN
from .template_entity import (
    TEMPLATE_ENTITY_COMMON_SCHEMA_LEGACY,
    TemplateEntity,
    rewrite_common_legacy_to_modern_conf,
)

_LOGGER = logging.getLogger(__name__)
_VALID_STATES = [STATE_ON, STATE_OFF, "true", "false"]

# Legacy
CONF_COLOR_ACTION = "set_color"
CONF_COLOR_TEMPLATE = "color_template"

CONF_HS_ACTION = "set_hs"
CONF_HS_TEMPLATE = "hs_template"
CONF_RGB_ACTION = "set_rgb"
CONF_RGB_TEMPLATE = "rgb_template"
CONF_RGBW_ACTION = "set_rgbw"
CONF_RGBW_TEMPLATE = "rgbw_template"
CONF_RGBWW_ACTION = "set_rgbww"
CONF_RGBWW_TEMPLATE = "rgbww_template"
CONF_EFFECT_ACTION = "set_effect"
CONF_EFFECT_LIST_TEMPLATE = "effect_list_template"
CONF_EFFECT_TEMPLATE = "effect_template"
CONF_LEVEL_ACTION = "set_level"
CONF_LEVEL_TEMPLATE = "level_template"
CONF_MAX_MIREDS_TEMPLATE = "max_mireds_template"
CONF_MIN_MIREDS_TEMPLATE = "min_mireds_template"
CONF_OFF_ACTION = "turn_off"
CONF_ON_ACTION = "turn_on"
CONF_SUPPORTS_TRANSITION = "supports_transition_template"
CONF_TEMPERATURE_ACTION = "set_temperature"
CONF_TEMPERATURE_TEMPLATE = "temperature_template"
CONF_WHITE_VALUE_ACTION = "set_white_value"
CONF_WHITE_VALUE_TEMPLATE = "white_value_template"

DEFAULT_MIN_MIREDS = 153
DEFAULT_MAX_MIREDS = 500

LIGHT_SCHEMA = vol.All(
    cv.deprecated(CONF_ENTITY_ID),
    vol.Schema(
        {
            vol.Exclusive(CONF_COLOR_ACTION, "hs_legacy_action"): cv.SCRIPT_SCHEMA,
            vol.Exclusive(CONF_COLOR_TEMPLATE, "hs_legacy_template"): cv.template,
            vol.Exclusive(CONF_HS_ACTION, "hs_legacy_action"): cv.SCRIPT_SCHEMA,
            vol.Exclusive(CONF_HS_TEMPLATE, "hs_legacy_template"): cv.template,
            vol.Optional(CONF_RGB_ACTION): cv.SCRIPT_SCHEMA,
            vol.Optional(CONF_RGB_TEMPLATE): cv.template,
            vol.Optional(CONF_RGBW_ACTION): cv.SCRIPT_SCHEMA,
            vol.Optional(CONF_RGBW_TEMPLATE): cv.template,
            vol.Optional(CONF_RGBWW_ACTION): cv.SCRIPT_SCHEMA,
            vol.Optional(CONF_RGBWW_TEMPLATE): cv.template,
            vol.Inclusive(CONF_EFFECT_ACTION, "effect"): cv.SCRIPT_SCHEMA,
            vol.Inclusive(CONF_EFFECT_LIST_TEMPLATE, "effect"): cv.template,
            vol.Inclusive(CONF_EFFECT_TEMPLATE, "effect"): cv.template,
            vol.Optional(CONF_ENTITY_ID): cv.entity_ids,
            vol.Optional(CONF_FRIENDLY_NAME): cv.string,
            vol.Optional(CONF_LEVEL_ACTION): cv.SCRIPT_SCHEMA,
            vol.Optional(CONF_LEVEL_TEMPLATE): cv.template,
            vol.Optional(CONF_MAX_MIREDS_TEMPLATE): cv.template,
            vol.Optional(CONF_MIN_MIREDS_TEMPLATE): cv.template,
            vol.Required(CONF_OFF_ACTION): cv.SCRIPT_SCHEMA,
            vol.Required(CONF_ON_ACTION): cv.SCRIPT_SCHEMA,
            vol.Optional(CONF_SUPPORTS_TRANSITION): cv.template,
            vol.Optional(CONF_TEMPERATURE_ACTION): cv.SCRIPT_SCHEMA,
            vol.Optional(CONF_TEMPERATURE_TEMPLATE): cv.template,
            vol.Optional(CONF_UNIQUE_ID): cv.string,
            vol.Optional(CONF_VALUE_TEMPLATE): cv.template,
        }
    ).extend(TEMPLATE_ENTITY_COMMON_SCHEMA_LEGACY.schema),
)

PLATFORM_SCHEMA = vol.All(
    # CONF_WHITE_VALUE_* is deprecated, support will be removed in release 2022.9
    cv.removed(CONF_WHITE_VALUE_ACTION),
    cv.removed(CONF_WHITE_VALUE_TEMPLATE),
    LIGHT_PLATFORM_SCHEMA.extend(
        {vol.Required(CONF_LIGHTS): cv.schema_with_slug_keys(LIGHT_SCHEMA)}
    ),
)


async def _async_create_entities(hass: HomeAssistant, config: ConfigType) -> List[LightTemplate]:
    """Create the Template Lights."""
    lights: List[LightTemplate] = []

    for object_id, entity_config in config[CONF_LIGHTS].items():
        entity_config = rewrite_common_legacy_to_modern_conf(hass, entity_config)
        unique_id = entity_config.get(CONF_UNIQUE_ID)

        lights.append(
            LightTemplate(
                hass,
                object_id,
                entity_config,
                unique_id,
            )
        )

    return lights


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the template lights."""
    async_add_entities(await _async_create_entities(hass, config))


class LightTemplate(TemplateEntity, LightEntity):
    """Representation of a templated Light, including dimmable."""

    _attr_should_poll = False

    def __init__(
        self,
        hass: HomeAssistant,
        object_id: str,
        config: ConfigType,
        unique_id: Optional[str],
    ) -> None:
        """Initialize the light."""
        super().__init__(
            hass, config=config, fallback_name=object_id, unique_id=unique_id
        )
        self.entity_id = async_generate_entity_id(
            ENTITY_ID_FORMAT, object_id, hass=hass
        )
        friendly_name = self._attr_name
        self._template = config.get(CONF_VALUE_TEMPLATE)
        self._on_script = Script(hass, config[CONF_ON_ACTION], friendly_name, DOMAIN)
        self._off_script = Script(hass, config[CONF_OFF_ACTION], friendly_name, DOMAIN)
        self._level_script: Optional[Script] = None
        if (level_action := config.get(CONF_LEVEL_ACTION)) is not None:
            self._level_script = Script(hass, level_action, friendly_name, DOMAIN)
        self._level_template = config.get(CONF_LEVEL_TEMPLATE)
        self._temperature_script: Optional[Script] = None
        if (temperature_action := config.get(CONF_TEMPERATURE_ACTION)) is not None:
            self._temperature_script = Script(
                hass, temperature_action, friendly_name, DOMAIN
            )
        self._temperature_template = config.get(CONF_TEMPERATURE_TEMPLATE)
        self._color_script: Optional[Script] = None
        if (color_action := config.get(CONF_COLOR_ACTION)) is not None:
            self._color_script = Script(hass, color_action, friendly_name, DOMAIN)
        self._color_template = config.get(CONF_COLOR_TEMPLATE)
        self._hs_script: Optional[Script] = None
        if (hs_action := config.get(CONF_HS_ACTION)) is not None:
            self._hs_script = Script(hass, hs_action, friendly_name, DOMAIN)
        self._hs_template = config.get(CONF_HS_TEMPLATE)
        self._rgb_script: Optional[Script] = None
        if (rgb_action := config.get(CONF_RGB_ACTION)) is not None:
            self._rgb_script = Script(hass, rgb_action, friendly_name, DOMAIN)
        self._rgb_template = config.get(CONF_RGB_TEMPLATE)
        self._rgbw_script: Optional[Script] = None
        if (rgbw_action := config.get(CONF_RGBW_ACTION)) is not None:
            self._rgbw_script = Script(hass, rgbw_action, friendly_name, DOMAIN)
        self._rgbw_template = config.get(CONF_RGBW_TEMPLATE)
        self._rgbww_script: Optional[Script] = None
        if (rgbww_action := config.get(CONF_RGBWW_ACTION)) is not None:
            self._rgbww_script = Script(hass, rgbww_action, friendly_name, DOMAIN)
        self._rgbww_template = config.get(CONF_RGBWW_TEMPLATE)
        self._effect_script: Optional[Script] = None
        if (effect_action := config.get(CONF_EFFECT_ACTION)) is not None:
            self._effect_script = Script(hass, effect_action, friendly_name, DOMAIN)
        self._effect_list_template = config.get(CONF_EFFECT_LIST_TEMPLATE)
        self._effect_template = config.get(CONF_EFFECT_TEMPLATE)
        self._max_mireds_template = config.get(CONF_MAX_MIREDS_TEMPLATE)
        self._min_mireds_template = config.get(CONF_MIN_MIREDS_TEMPLATE)
        self._supports_transition_template = config.get(CONF_SUPPORTS_TRANSITION)

        self._state: bool = False
        self._brightness: Optional[int] = None
        self._temperature: Optional[int] = None
        self._hs_color: Optional[Tuple[float, float]] = None
        self._rgb_color: Optional[Tuple[int, int, int]] = None
        self._rgbw_color: Optional[Tuple[int, int, int, int]] = None
        self._rgbww_color: Optional[Tuple[int, int, int, int, int]] = None
        self._effect: Optional[str] = None
        self._effect_list: Optional[List[str]] = None
        self._color_mode: Optional[ColorMode] = None
        self._max_mireds: Optional[int] = None
        self._min_mireds: Optional[int] = None
        self._supports_transition: bool = False
        self._supported_color_modes: Optional[set[ColorMode]] = None

        color_modes = {ColorMode.ONOFF}
        if self._level_script is not None:
            color_modes.add(ColorMode.BRIGHTNESS)
        if self._temperature_script is not None:
            color_modes.add(ColorMode.COLOR_TEMP)
        if self._hs_script is not None:
            color_modes.add(ColorMode.HS)
        if self._color_script is not None:
            color_modes.add(ColorMode.HS)
        if self._rgb_script is not None:
            color_modes.add(ColorMode.RGB)
        if self._rgbw_script is not None:
            color_modes.add(ColorMode.RGBW)
        if self._rgbww_script is not None:
            color_modes.add(ColorMode.RGBWW)

        self._supported_color_modes = filter_supported_color_modes(color_modes)
        if len(self._supported_color_modes) > 1:
            self._color_mode = ColorMode.UNKNOWN
        if len(self._supported_color_modes) == 1:
            self._color_mode = next(iter(self._supported_color_modes))

        self._attr_supported_features = LightEntityFeature(0)
        if self._effect_script is not None:
            self._attr_supported_features |= LightEntityFeature.EFFECT
        if self._supports_transition is True:
            self._attr_supported_features |= LightEntityFeature.TRANSITION

    @property
    def brightness(self) -> Optional[int]:
        """Return the brightness of the light."""
        return self._brightness

    @property
    def color_temp_kelvin(self) -> Optional[int]:
        """Return the color temperature value in Kelvin."""
        if self._temperature is None:
            return None
        return color_util.color_temperature_mired_to_kelvin(self._temperature)

    @property
    def min_color_temp_kelvin(self) -> int:
        """Return the warmest color_temp_kelvin that this light supports."""
        if self._max_mireds is not None:
            return color_util.color_temperature_mired_to_kelvin(self._max_mireds)

        return DEFAULT_MIN_KELVIN

    @property
    def max_color_temp_kelvin(self) -> int:
        """Return the coldest color_temp_kelvin that this light supports."""
        if self._min_mireds is not None:
            return color_util.color_temperature_mired_to_kelvin(self._min_mireds)

        return DEFAULT_MAX_KELVIN

    @property
    def hs_color(self) -> Optional[Tuple[float, float]]:
        """Return the hue and saturation color value [float, float]."""
        return self._hs_color

    @property
    def rgb_color(self) -> Optional[Tuple[int, int, int]]:
        """Return the rgb color value."""
        return self._rgb_color

    @property
    def rgbw_color(self) -> Optional[Tuple[int, int, int, int]]:
        """Return the rgbw color value."""
        return self._rgbw_color

    @property
    def rgbww_color(self) -> Optional[Tuple[int, int, int, int, int]]:
        """Return the rgbww color value."""
        return self._rgbww_color

    @property
    def effect(self) -> Optional[str]:
        """Return the effect."""
        return self._effect

    @property
    def effect_list(self) -> Optional[List[str]]:
        """Return the effect list."""
        return self._effect_list

    @property
    def color_mode(self) -> Optional[ColorMode]:
        """Return current color mode."""
        return self._color_mode

    @property
    def supported_color_modes(self) -> Optional[set[ColorMode]]:
        """Flag supported color modes."""
        return self._supported_color_modes

    @property
    def is_on(self) -> bool:
        """Return true if device is on."""
        return self._state

    @callback
    def _async_setup_templates(self) -> None:
        """Set up templates."""
        if self._template:
            self.add_template_attribute(
                "_state", self._template, None, self._update_state
            )
        if self._level_template:
            self.add_template_attribute(
                "_brightness",
                self._level_template,
                None,
                self._update_brightness,
                none_on_template_error=True,
            )
        if self._max_mireds_template:
            self.add_template_attribute(
                "_max_mireds_template",
                self._max_mireds_template,
                None,
                self._update_max_mireds,
                none_on_template_error=True,
            )
        if self._min_mireds_template:
            self.add_template_attribute(
                "_min_mireds_template",
                self._min_mireds_template,
                None,
                self._update_min_mireds,
                none_on_template_error=True,
            )
        if self._temperature_template:
            self.add_template_attribute(
                "_temperature",
                self._temperature_template,
                None,
                self._update_temperature,
                none_on_template_error=True,
            )
        if self._color_template:
            self.add_template_attribute(
                "_hs_color",
                self._color_template,
                None,
                self._update_hs,
                none_on_template_error=True,
            )
        if self._hs_template:
            self.add_template_attribute(
                "_hs_color",
                self._hs_template,
                None,
                self._update_hs,
                none_on_template_error=True,
            )
        if self._rgb_template:
            self.add_template_attribute(
                "_rgb_color",
                self._rgb_template,
                None,
                self._update_rgb,
                none_on_template_error=True,
            )
        if self._rgbw_template:
            self.add_template_attribute(
                "_rgbw_color",
                self._rgbw_template,
                None,
                self._update_rgbw,
                none_on_template_error=True,
            )
        if self._rgbww_template:
            self.add_template_attribute(
                "_rgbww_color",
                self._rgbww_template,
                None,
                self._update_rgbww,
                none_on_template_error=True,
            )
        if self._effect_list_template:
            self.add_template_attribute(
                "_effect_list",
                self._effect_list_template,
                None,
                self._update_effect_list,
                none_on_template_error=True,
            )
        if self._effect_template
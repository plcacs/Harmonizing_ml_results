"""Support for MQTT lights."""
from __future__ import annotations
from collections.abc import Callable, Mapping
import logging
from typing import Any, cast, TypedDict, Optional, Union, Literal, Final, NotRequired

import voluptuous as vol
from homeassistant.components.light import (
    _DEPRECATED_ATTR_COLOR_TEMP,
    _DEPRECATED_ATTR_MAX_MIREDS,
    _DEPRECATED_ATTR_MIN_MIREDS,
    ATTR_BRIGHTNESS,
    ATTR_COLOR_MODE,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_EFFECT,
    ATTR_EFFECT_LIST,
    ATTR_HS_COLOR,
    ATTR_MAX_COLOR_TEMP_KELVIN,
    ATTR_MIN_COLOR_TEMP_KELVIN,
    ATTR_RGB_COLOR,
    ATTR_RGBW_COLOR,
    ATTR_RGBWW_COLOR,
    ATTR_SUPPORTED_COLOR_MODES,
    ATTR_WHITE,
    ATTR_XY_COLOR,
    DEFAULT_MAX_KELVIN,
    DEFAULT_MIN_KELVIN,
    ENTITY_ID_FORMAT,
    ColorMode,
    LightEntity,
    LightEntityFeature,
    valid_supported_color_modes,
)
from homeassistant.const import (
    CONF_NAME,
    CONF_OPTIMISTIC,
    CONF_PAYLOAD_OFF,
    CONF_PAYLOAD_ON,
    STATE_ON,
)
from homeassistant.core import callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.service_info.mqtt import ReceivePayloadType
from homeassistant.helpers.typing import ConfigType, VolSchemaType
from homeassistant.util import color as color_util
from .. import subscription
from ..config import MQTT_RW_SCHEMA
from ..const import (
    CONF_COLOR_TEMP_KELVIN,
    CONF_COMMAND_TOPIC,
    CONF_MAX_KELVIN,
    CONF_MIN_KELVIN,
    CONF_STATE_TOPIC,
    CONF_STATE_VALUE_TEMPLATE,
    PAYLOAD_NONE,
)
from ..entity import MqttEntity
from ..models import (
    MqttCommandTemplate,
    MqttValueTemplate,
    PayloadSentinel,
    PublishPayloadType,
    ReceiveMessage,
    TemplateVarsType,
)
from ..schemas import MQTT_ENTITY_COMMON_SCHEMA
from ..util import valid_publish_topic, valid_subscribe_topic
from .schema import MQTT_LIGHT_SCHEMA_SCHEMA

_LOGGER: Final = logging.getLogger(__name__)

CONF_BRIGHTNESS_COMMAND_TEMPLATE: Final = "brightness_command_template"
CONF_BRIGHTNESS_COMMAND_TOPIC: Final = "brightness_command_topic"
CONF_BRIGHTNESS_SCALE: Final = "brightness_scale"
CONF_BRIGHTNESS_STATE_TOPIC: Final = "brightness_state_topic"
CONF_BRIGHTNESS_VALUE_TEMPLATE: Final = "brightness_value_template"
CONF_COLOR_MODE_STATE_TOPIC: Final = "color_mode_state_topic"
CONF_COLOR_MODE_VALUE_TEMPLATE: Final = "color_mode_value_template"
CONF_COLOR_TEMP_COMMAND_TEMPLATE: Final = "color_temp_command_template"
CONF_COLOR_TEMP_COMMAND_TOPIC: Final = "color_temp_command_topic"
CONF_COLOR_TEMP_STATE_TOPIC: Final = "color_temp_state_topic"
CONF_COLOR_TEMP_VALUE_TEMPLATE: Final = "color_temp_value_template"
CONF_EFFECT_COMMAND_TEMPLATE: Final = "effect_command_template"
CONF_EFFECT_COMMAND_TOPIC: Final = "effect_command_topic"
CONF_EFFECT_LIST: Final = "effect_list"
CONF_EFFECT_STATE_TOPIC: Final = "effect_state_topic"
CONF_EFFECT_VALUE_TEMPLATE: Final = "effect_value_template"
CONF_HS_COMMAND_TEMPLATE: Final = "hs_command_template"
CONF_HS_COMMAND_TOPIC: Final = "hs_command_topic"
CONF_HS_STATE_TOPIC: Final = "hs_state_topic"
CONF_HS_VALUE_TEMPLATE: Final = "hs_value_template"
CONF_MAX_MIREDS: Final = "max_mireds"
CONF_MIN_MIREDS: Final = "min_mireds"
CONF_RGB_COMMAND_TEMPLATE: Final = "rgb_command_template"
CONF_RGB_COMMAND_TOPIC: Final = "rgb_command_topic"
CONF_RGB_STATE_TOPIC: Final = "rgb_state_topic"
CONF_RGB_VALUE_TEMPLATE: Final = "rgb_value_template"
CONF_RGBW_COMMAND_TEMPLATE: Final = "rgbw_command_template"
CONF_RGBW_COMMAND_TOPIC: Final = "rgbw_command_topic"
CONF_RGBW_STATE_TOPIC: Final = "rgbw_state_topic"
CONF_RGBW_VALUE_TEMPLATE: Final = "rgbw_value_template"
CONF_RGBWW_COMMAND_TEMPLATE: Final = "rgbww_command_template"
CONF_RGBWW_COMMAND_TOPIC: Final = "rgbww_command_topic"
CONF_RGBWW_STATE_TOPIC: Final = "rgbww_state_topic"
CONF_RGBWW_VALUE_TEMPLATE: Final = "rgbww_value_template"
CONF_XY_COMMAND_TEMPLATE: Final = "xy_command_template"
CONF_XY_COMMAND_TOPIC: Final = "xy_command_topic"
CONF_XY_STATE_TOPIC: Final = "xy_state_topic"
CONF_XY_VALUE_TEMPLATE: Final = "xy_value_template"
CONF_WHITE_COMMAND_TOPIC: Final = "white_command_topic"
CONF_WHITE_SCALE: Final = "white_scale"
CONF_ON_COMMAND_TYPE: Final = "on_command_type"

MQTT_LIGHT_ATTRIBUTES_BLOCKED: Final = frozenset(
    {
        ATTR_COLOR_MODE,
        ATTR_BRIGHTNESS,
        _DEPRECATED_ATTR_COLOR_TEMP.value,
        ATTR_COLOR_TEMP_KELVIN,
        ATTR_EFFECT,
        ATTR_EFFECT_LIST,
        ATTR_HS_COLOR,
        ATTR_MAX_COLOR_TEMP_KELVIN,
        _DEPRECATED_ATTR_MAX_MIREDS.value,
        ATTR_MIN_COLOR_TEMP_KELVIN,
        _DEPRECATED_ATTR_MIN_MIREDS.value,
        ATTR_RGB_COLOR,
        ATTR_RGBW_COLOR,
        ATTR_RGBWW_COLOR,
        ATTR_SUPPORTED_COLOR_MODES,
        ATTR_XY_COLOR,
    }
)

DEFAULT_BRIGHTNESS_SCALE: Final = 255
DEFAULT_NAME: Final = "MQTT LightEntity"
DEFAULT_PAYLOAD_OFF: Final = "OFF"
DEFAULT_PAYLOAD_ON: Final = "ON"
DEFAULT_WHITE_SCALE: Final = 255
DEFAULT_ON_COMMAND_TYPE: Final = "last"
VALUES_ON_COMMAND_TYPE: Final = ["first", "last", "brightness"]

COMMAND_TEMPLATE_KEYS: Final = [
    CONF_BRIGHTNESS_COMMAND_TEMPLATE,
    CONF_COLOR_TEMP_COMMAND_TEMPLATE,
    CONF_EFFECT_COMMAND_TEMPLATE,
    CONF_HS_COMMAND_TEMPLATE,
    CONF_RGB_COMMAND_TEMPLATE,
    CONF_RGBW_COMMAND_TEMPLATE,
    CONF_RGBWW_COMMAND_TEMPLATE,
    CONF_XY_COMMAND_TEMPLATE,
]

VALUE_TEMPLATE_KEYS: Final = [
    CONF_BRIGHTNESS_VALUE_TEMPLATE,
    CONF_COLOR_MODE_VALUE_TEMPLATE,
    CONF_COLOR_TEMP_VALUE_TEMPLATE,
    CONF_EFFECT_VALUE_TEMPLATE,
    CONF_HS_VALUE_TEMPLATE,
    CONF_RGB_VALUE_TEMPLATE,
    CONF_RGBW_VALUE_TEMPLATE,
    CONF_RGBWW_VALUE_TEMPLATE,
    CONF_STATE_VALUE_TEMPLATE,
    CONF_XY_VALUE_TEMPLATE,
]

PLATFORM_SCHEMA_MODERN_BASIC: Final = MQTT_RW_SCHEMA.extend(
    {
        vol.Optional(CONF_BRIGHTNESS_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_BRIGHTNESS_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(
            CONF_BRIGHTNESS_SCALE, default=DEFAULT_BRIGHTNESS_SCALE
        ): vol.All(vol.Coerce(int), vol.Range(min=1)),
        vol.Optional(CONF_BRIGHTNESS_STATE_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_BRIGHTNESS_VALUE_TEMPLATE): cv.template,
        vol.Optional(CONF_COLOR_MODE_STATE_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_COLOR_MODE_VALUE_TEMPLATE): cv.template,
        vol.Optional(CONF_COLOR_TEMP_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_COLOR_TEMP_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(CONF_COLOR_TEMP_STATE_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_COLOR_TEMP_VALUE_TEMPLATE): cv.template,
        vol.Optional(CONF_COLOR_TEMP_KELVIN, default=False): cv.boolean,
        vol.Optional(CONF_EFFECT_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_EFFECT_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(CONF_EFFECT_LIST): vol.All(cv.ensure_list, [cv.string]),
        vol.Optional(CONF_EFFECT_STATE_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_EFFECT_VALUE_TEMPLATE): cv.template,
        vol.Optional(CONF_HS_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_HS_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(CONF_HS_STATE_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_HS_VALUE_TEMPLATE): cv.template,
        vol.Optional(CONF_MAX_MIREDS): cv.positive_int,
        vol.Optional(CONF_MIN_MIREDS): cv.positive_int,
        vol.Optional(CONF_MAX_KELVIN): cv.positive_int,
        vol.Optional(CONF_MIN_KELVIN): cv.positive_int,
        vol.Optional(CONF_NAME): vol.Any(cv.string, None),
        vol.Optional(
            CONF_ON_COMMAND_TYPE, default=DEFAULT_ON_COMMAND_TYPE
        ): vol.In(VALUES_ON_COMMAND_TYPE),
        vol.Optional(CONF_PAYLOAD_OFF, default=DEFAULT_PAYLOAD_OFF): cv.string,
        vol.Optional(CONF_PAYLOAD_ON, default=DEFAULT_PAYLOAD_ON): cv.string,
        vol.Optional(CONF_RGB_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_RGB_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(CONF_RGB_STATE_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_RGB_VALUE_TEMPLATE): cv.template,
        vol.Optional(CONF_RGBW_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_RGBW_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(CONF_RGBW_STATE_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_RGBW_VALUE_TEMPLATE): cv.template,
        vol.Optional(CONF_RGBWW_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_RGBWW_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(CONF_RGBWW_STATE_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_RGBWW_VALUE_TEMPLATE): cv.template,
        vol.Optional(CONF_STATE_VALUE_TEMPLATE): cv.template,
        vol.Optional(CONF_WHITE_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(
            CONF_WHITE_SCALE, default=DEFAULT_WHITE_SCALE
        ): vol.All(vol.Coerce(int), vol.Range(min=1)),
        vol.Optional(CONF_XY_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_XY_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(CONF_XY_STATE_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_XY_VALUE_TEMPLATE): cv.template,
    }
).extend(MQTT_ENTITY_COMMON_SCHEMA.schema).extend(MQTT_LIGHT_SCHEMA_SCHEMA.schema)

DISCOVERY_SCHEMA_BASIC: Final = vol.All(
    PLATFORM_SCHEMA_MODERN_BASIC.extend({}, extra=vol.REMOVE_EXTRA)
)


class TopicDict(TypedDict, total=False):
    """Dictionary with MQTT topics."""

    brightness_command_topic: str | None
    brightness_state_topic: str | None
    color_mode_state_topic: str | None
    color_temp_command_topic: str | None
    color_temp_state_topic: str | None
    command_topic: str | None
    effect_command_topic: str | None
    effect_state_topic: str | None
    hs_command_topic: str | None
    hs_state_topic: str | None
    rgb_command_topic: str | None
    rgb_state_topic: str | None
    rgbw_command_topic: str | None
    rgbw_state_topic: str | None
    rgbww_command_topic: str | None
    rgbww_state_topic: str | None
    state_topic: str | None
    white_command_topic: str | None
    xy_command_topic: str | None
    xy_state_topic: str | None


class PayloadDict(TypedDict):
    """Dictionary with MQTT payloads."""

    on: str
    off: str


class MqttLight(MqttEntity, LightEntity, RestoreEntity):
    """Representation of a MQTT light."""

    _default_name: Final = DEFAULT_NAME
    _entity_id_format: Final = ENTITY_ID_FORMAT
    _attributes_extra_blocked: Final = MQTT_LIGHT_ATTRIBUTES_BLOCKED

    _color_temp_kelvin: bool
    _topic: TopicDict
    _payload: PayloadDict
    _value_templates: dict[str, Callable[[ReceivePayloadType, PayloadSentinel], str]
    _command_templates: dict[str, Callable[[str, TemplateVarsType | None], str]]
    _optimistic_color_mode: bool
    _optimistic: bool
    _optimistic_rgb_color: bool
    _optimistic_rgbw_color: bool
    _optimistic_rgbww_color: bool
    _optimistic_brightness: bool
    _optimistic_color_temp_kelvin: bool
    _optimistic_effect: bool
    _optimistic_hs_color: bool
    _optimistic_xy_color: bool

    @staticmethod
    def config_schema() -> vol.Schema:
        """Return the config schema."""
        return DISCOVERY_SCHEMA_BASIC

    def _setup_from_config(self, config: ConfigType) -> None:
        """(Re)Setup the entity."""
        self._color_temp_kelvin = config[CONF_COLOR_TEMP_KELVIN]
        self._attr_min_color_temp_kelvin = (
            color_util.color_temperature_mired_to_kelvin(max_mireds)
            if (max_mireds := config.get(CONF_MAX_MIREDS))
            else config.get(CONF_MIN_KELVIN, DEFAULT_MIN_KELVIN)
        )
        self._attr_max_color_temp_kelvin = (
            color_util.color_temperature_mired_to_kelvin(min_mireds)
            if (min_mireds := config.get(CONF_MIN_MIREDS))
            else config.get(CONF_MAX_KELVIN, DEFAULT_MAX_KELVIN)
        )
        self._attr_effect_list = config.get(CONF_EFFECT_LIST)
        topic = {
            key: config.get(key)
            for key in (
                CONF_BRIGHTNESS_COMMAND_TOPIC,
                CONF_BRIGHTNESS_STATE_TOPIC,
                CONF_COLOR_MODE_STATE_TOPIC,
                CONF_COLOR_TEMP_COMMAND_TOPIC,
                CONF_COLOR_TEMP_STATE_TOPIC,
                CONF_COMMAND_TOPIC,
                CONF_EFFECT_COMMAND_TOPIC,
                CONF_EFFECT_STATE_TOPIC,
                CONF_HS_COMMAND_TOPIC,
                CONF_HS_STATE_TOPIC,
                CONF_RGB_COMMAND_TOPIC,
                CONF_RGB_STATE_TOPIC,
                CONF_RGBW_COMMAND_TOPIC,
                CONF_RGBW_STATE_TOPIC,
                CONF_RGBWW_COMMAND_TOPIC,
                CONF_RGBWW_STATE_TOPIC,
                CONF_STATE_TOPIC,
                CONF_WHITE_COMMAND_TOPIC,
                CONF_XY_COMMAND_TOPIC,
                CONF_XY_STATE_TOPIC,
            )
        }
        self._topic = cast(TopicDict, topic)
        self._payload = {
            "on": config[CONF_PAYLOAD_ON],
            "off": config[CONF_PAYLOAD_OFF],
        }
        self._value_templates = {
            key: MqttValueTemplate(
                config.get(key), entity=self
            ).async_render_with_possible_json_value
            for key in VALUE_TEMPLATE_KEYS
        }
        self._command_templates = {
            key: MqttCommandTemplate(config.get(key), entity=self).async_render
            for key in COMMAND_TEMPLATE_KEYS
        }
        optimistic = config[CONF_OPTIMISTIC]
        self._optimistic_color_mode = (
            optimistic or topic[CONF_COLOR_MODE_STATE_TOPIC] is None
        )
        self._
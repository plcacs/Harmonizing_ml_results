from __future__ import annotations
from collections.abc import Callable
import logging
from typing import Any, cast, Dict, List, Tuple, Union
import voluptuous as vol
from homeassistant.components.light import (
    _DEPRECATED_ATTR_COLOR_TEMP, _DEPRECATED_ATTR_MAX_MIREDS, _DEPRECATED_ATTR_MIN_MIREDS,
    ATTR_BRIGHTNESS, ATTR_COLOR_MODE, ATTR_COLOR_TEMP_KELVIN, ATTR_EFFECT, ATTR_EFFECT_LIST,
    ATTR_HS_COLOR, ATTR_MAX_COLOR_TEMP_KELVIN, ATTR_MIN_COLOR_TEMP_KELVIN, ATTR_RGB_COLOR,
    ATTR_RGBW_COLOR, ATTR_RGBWW_COLOR, ATTR_SUPPORTED_COLOR_MODES, ATTR_WHITE, ATTR_XY_COLOR,
    DEFAULT_MAX_KELVIN, DEFAULT_MIN_KELVIN, ENTITY_ID_FORMAT, ColorMode, LightEntity,
    LightEntityFeature, valid_supported_color_modes
)
from homeassistant.const import CONF_NAME, CONF_OPTIMISTIC, CONF_PAYLOAD_OFF, CONF_PAYLOAD_ON, STATE_ON
from homeassistant.core import callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.service_info.mqtt import ReceivePayloadType
from homeassistant.helpers.typing import ConfigType, VolSchemaType
from homeassistant.util import color as color_util
from .. import subscription
from ..config import MQTT_RW_SCHEMA
from ..const import (
    CONF_COLOR_TEMP_KELVIN, CONF_COMMAND_TOPIC, CONF_MAX_KELVIN, CONF_MIN_KELVIN, CONF_STATE_TOPIC,
    CONF_STATE_VALUE_TEMPLATE, PAYLOAD_NONE
)
from ..entity import MqttEntity
from ..models import MqttCommandTemplate, MqttValueTemplate, PayloadSentinel, PublishPayloadType, ReceiveMessage, TemplateVarsType
from ..schemas import MQTT_ENTITY_COMMON_SCHEMA
from ..util import valid_publish_topic, valid_subscribe_topic
from .schema import MQTT_LIGHT_SCHEMA_SCHEMA

_LOGGER = logging.getLogger(__name__)

CONF_BRIGHTNESS_COMMAND_TEMPLATE: str = 'brightness_command_template'
CONF_BRIGHTNESS_COMMAND_TOPIC: str = 'brightness_command_topic'
CONF_BRIGHTNESS_SCALE: str = 'brightness_scale'
CONF_BRIGHTNESS_STATE_TOPIC: str = 'brightness_state_topic'
CONF_BRIGHTNESS_VALUE_TEMPLATE: str = 'brightness_value_template'
CONF_COLOR_MODE_STATE_TOPIC: str = 'color_mode_state_topic'
CONF_COLOR_MODE_VALUE_TEMPLATE: str = 'color_mode_value_template'
CONF_COLOR_TEMP_COMMAND_TEMPLATE: str = 'color_temp_command_template'
CONF_COLOR_TEMP_COMMAND_TOPIC: str = 'color_temp_command_topic'
CONF_COLOR_TEMP_STATE_TOPIC: str = 'color_temp_state_topic'
CONF_COLOR_TEMP_VALUE_TEMPLATE: str = 'color_temp_value_template'
CONF_EFFECT_COMMAND_TEMPLATE: str = 'effect_command_template'
CONF_EFFECT_COMMAND_TOPIC: str = 'effect_command_topic'
CONF_EFFECT_LIST: str = 'effect_list'
CONF_EFFECT_STATE_TOPIC: str = 'effect_state_topic'
CONF_EFFECT_VALUE_TEMPLATE: str = 'effect_value_template'
CONF_HS_COMMAND_TEMPLATE: str = 'hs_command_template'
CONF_HS_COMMAND_TOPIC: str = 'hs_command_topic'
CONF_HS_STATE_TOPIC: str = 'hs_state_topic'
CONF_HS_VALUE_TEMPLATE: str = 'hs_value_template'
CONF_MAX_MIREDS: str = 'max_mireds'
CONF_MIN_MIREDS: str = 'min_mireds'
CONF_RGB_COMMAND_TEMPLATE: str = 'rgb_command_template'
CONF_RGB_COMMAND_TOPIC: str = 'rgb_command_topic'
CONF_RGB_STATE_TOPIC: str = 'rgb_state_topic'
CONF_RGB_VALUE_TEMPLATE: str = 'rgb_value_template'
CONF_RGBW_COMMAND_TEMPLATE: str = 'rgbw_command_template'
CONF_RGBW_COMMAND_TOPIC: str = 'rgbw_command_topic'
CONF_RGBW_STATE_TOPIC: str = 'rgbw_state_topic'
CONF_RGBW_VALUE_TEMPLATE: str = 'rgbw_value_template'
CONF_RGBWW_COMMAND_TEMPLATE: str = 'rgbww_command_template'
CONF_RGBWW_COMMAND_TOPIC: str = 'rgbww_command_topic'
CONF_RGBWW_STATE_TOPIC: str = 'rgbww_state_topic'
CONF_RGBWW_VALUE_TEMPLATE: str = 'rgbww_value_template'
CONF_XY_COMMAND_TEMPLATE: str = 'xy_command_template'
CONF_XY_COMMAND_TOPIC: str = 'xy_command_topic'
CONF_XY_STATE_TOPIC: str = 'xy_state_topic'
CONF_XY_VALUE_TEMPLATE: str = 'xy_value_template'
CONF_WHITE_COMMAND_TOPIC: str = 'white_command_topic'
CONF_WHITE_SCALE: str = 'white_scale'
CONF_ON_COMMAND_TYPE: str = 'on_command_type'

MQTT_LIGHT_ATTRIBUTES_BLOCKED: frozenset = frozenset({
    ATTR_COLOR_MODE, ATTR_BRIGHTNESS, _DEPRECATED_ATTR_COLOR_TEMP.value, ATTR_COLOR_TEMP_KELVIN,
    ATTR_EFFECT, ATTR_EFFECT_LIST, ATTR_HS_COLOR, ATTR_MAX_COLOR_TEMP_KELVIN, _DEPRECATED_ATTR_MAX_MIREDS.value,
    ATTR_MIN_COLOR_TEMP_KELVIN, _DEPRECATED_ATTR_MIN_MIREDS.value, ATTR_RGB_COLOR, ATTR_RGBW_COLOR,
    ATTR_RGBWW_COLOR, ATTR_SUPPORTED_COLOR_MODES, ATTR_XY_COLOR
})

DEFAULT_BRIGHTNESS_SCALE: int = 255
DEFAULT_NAME: str = 'MQTT LightEntity'
DEFAULT_PAYLOAD_OFF: str = 'OFF'
DEFAULT_PAYLOAD_ON: str = 'ON'
DEFAULT_WHITE_SCALE: int = 255
DEFAULT_ON_COMMAND_TYPE: str = 'last'
VALUES_ON_COMMAND_TYPE: List[str] = ['first', 'last', 'brightness']

COMMAND_TEMPLATE_KEYS: List[str] = [
    CONF_BRIGHTNESS_COMMAND_TEMPLATE, CONF_COLOR_TEMP_COMMAND_TEMPLATE, CONF_EFFECT_COMMAND_TEMPLATE,
    CONF_HS_COMMAND_TEMPLATE, CONF_RGB_COMMAND_TEMPLATE, CONF_RGBW_COMMAND_TEMPLATE, CONF_RGBWW_COMMAND_TEMPLATE,
    CONF_XY_COMMAND_TEMPLATE
]

VALUE_TEMPLATE_KEYS: List[str] = [
    CONF_BRIGHTNESS_VALUE_TEMPLATE, CONF_COLOR_MODE_VALUE_TEMPLATE, CONF_COLOR_TEMP_VALUE_TEMPLATE,
    CONF_EFFECT_VALUE_TEMPLATE, CONF_HS_VALUE_TEMPLATE, CONF_RGB_VALUE_TEMPLATE, CONF_RGBW_VALUE_TEMPLATE,
    CONF_RGBWW_VALUE_TEMPLATE, CONF_STATE_VALUE_TEMPLATE, CONF_XY_VALUE_TEMPLATE
]

PLATFORM_SCHEMA_MODERN_BASIC: VolSchemaType = MQTT_RW_SCHEMA.extend({
    vol.Optional(CONF_BRIGHTNESS_COMMAND_TEMPLATE): cv.template,
    vol.Optional(CONF_BRIGHTNESS_COMMAND_TOPIC): valid_publish_topic,
    vol.Optional(CONF_BRIGHTNESS_SCALE, default=DEFAULT_BRIGHTNESS_SCALE): vol.All(vol.Coerce(int), vol.Range(min=1)),
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
    vol.Optional(CONF_ON_COMMAND_TYPE, default=DEFAULT_ON_COMMAND_TYPE): vol.In(VALUES_ON_COMMAND_TYPE),
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
    vol.Optional(CONF_WHITE_SCALE, default=DEFAULT_WHITE_SCALE): vol.All(vol.Coerce(int), vol.Range(min=1),
    vol.Optional(CONF_XY_COMMAND_TEMPLATE): cv.template,
    vol.Optional(CONF_XY_COMMAND_TOPIC): valid_publish_topic,
    vol.Optional(CONF_XY_STATE_TOPIC): valid_subscribe_topic,
    vol.Optional(CONF_XY_VALUE_TEMPLATE): cv.template
}).extend(MQTT_ENTITY_COMMON_SCHEMA.schema).extend(MQTT_LIGHT_SCHEMA_SCHEMA.schema)

DISCOVERY_SCHEMA_BASIC: VolSchemaType = vol.All(
    PLATFORM_SCHEMA_MODERN_BASIC.extend({}, extra=vol.REMOVE_EXTRA)
)

class MqttLight(MqttEntity, LightEntity, RestoreEntity):
    """Representation of a MQTT light."""
    _default_name: str = DEFAULT_NAME
    _entity_id_format: str = ENTITY_ID_FORMAT
    _attributes_extra_blocked: frozenset = MQTT_LIGHT_ATTRIBUTES_BLOCKED

    @staticmethod
    def config_schema() -> VolSchemaType:
        """Return the config schema."""
        return DISCOVERY_SCHEMA_BASIC

    def _setup_from_config(self, config: ConfigType) -> None:
        """(Re)Setup the entity."""
        self._color_temp_kelvin: bool = config[CONF_COLOR_TEMP_KELVIN]
        self._attr_min_color_temp_kelvin: int = color_util.color_temperature_mired_to_kelvin(max_mireds) if (max_mireds := config.get(CONF_MAX_MIREDS)) else config.get(CONF_MIN_KELVIN, DEFAULT_MIN_KELVIN)
        self._attr_max_color_temp_kelvin: int = color_util.color_temperature_mired_to_kelvin(min_mireds) if (min_mireds := config.get(CONF_MIN_MIREDS)) else config.get(CONF_MAX_KELVIN, DEFAULT_MAX_KELVIN)
        self._attr_effect_list: List[str] = config.get(CONF_EFFECT_LIST)
        topic: Dict[str, Any] = {key: config.get(key) for key in (CONF_BRIGHTNESS_COMMAND_TOPIC, CONF_BRIGHTNESS_STATE_TOPIC, CONF_COLOR_MODE_STATE_TOPIC, CONF_COLOR_TEMP_COMMAND_TOPIC, CONF_COLOR_TEMP_STATE_TOPIC, CONF_COMMAND_TOPIC, CONF_EFFECT_COMMAND_TOPIC, CONF_EFFECT_STATE_TOPIC, CONF_HS_COMMAND_TOPIC, CONF_HS_STATE_TOPIC, CONF_RGB_COMMAND_TOPIC, CONF_RGB_STATE_TOPIC, CONF_RGBW_COMMAND_TOPIC, CONF_RGBW_STATE_TOPIC, CONF_RGBWW_COMMAND_TOPIC, CONF_RGBWW_STATE_TOPIC, CONF_STATE_TOPIC, CONF_WHITE_COMMAND_TOPIC, CONF_XY_COMMAND_TOPIC, CONF_XY_STATE_TOPIC)}
        self._topic: Dict[str, Any] = topic
        self._payload: Dict[str, str] = {'on': config[CONF_PAYLOAD_ON], 'off': config[CONF_PAYLOAD_OFF]}
        self._value_templates: Dict[str, Callable] = {key: MqttValueTemplate(config.get(key), entity=self).async_render_with_possible_json_value for key in VALUE_TEMPLATE_KEYS}
        self._command_templates: Dict[str, Callable] = {key: MqttCommandTemplate(config.get(key), entity=self).async_render for key in COMMAND_TEMPLATE_KEYS}
        optimistic: bool = config[CONF_OPTIMISTIC]
        self._optimistic_color_mode: bool = optimistic or topic[CONF_COLOR_MODE_STATE_TOPIC] is None
        self._optimistic: bool = optimistic or topic[CONF_STATE_TOPIC] is None
        self._attr_assumed_state: bool = bool(self._optimistic)
        self._optimistic_rgb_color: bool = optimistic or topic[CONF_RGB_STATE_TOPIC] is None
        self._optimistic_rgbw_color: bool = optimistic or topic[CONF_RGBW_STATE_TOPIC] is None
        self._optimistic_rgbww_color: bool = optimistic or topic[CONF_RGBWW_STATE_TOPIC] is None
        self._optimistic_brightness: bool = optimistic or (topic[CONF_BRIGHTNESS_COMMAND_TOPIC] is not None and topic[CONF_BRIGHTNESS_STATE_TOPIC] is None) or (topic[CONF_BRIGHTNESS_COMMAND_TOPIC] is None and topic[CONF_RGB_STATE_TOPIC] is None)
        self._optimistic_color_temp_kelvin: bool = optimistic or topic[CONF_COLOR_TEMP_STATE_TOPIC] is None
        self._optimistic_effect: bool = optimistic or topic[CONF_EFFECT_STATE_TOPIC] is None
        self._optimistic_hs_color: bool = optimistic or topic[CONF_HS_STATE_TOPIC] is None
        self._optimistic_xy_color: bool = optimistic or topic[CONF_XY_STATE_TOPIC] is None
        supported_color_modes: set = set()
        if topic[CONF_COLOR_TEMP_COMMAND_TOPIC] is not None:
            supported_color_modes.add(ColorMode.COLOR_TEMP)
            self._attr_color_mode: ColorMode = ColorMode.COLOR_TEMP
        if topic[CONF_HS_COMMAND_TOPIC] is not None:
            supported_color_modes.add(ColorMode.HS)
            self._attr_color_mode: ColorMode = ColorMode.HS
        if topic[CONF_RGB_COMMAND_TOPIC] is not None:
            supported_color_modes.add(ColorMode.RGB)
            self._attr_color_mode: ColorMode = ColorMode.RGB
        if topic[CONF_RGBW_COMMAND_TOPIC] is not None:
            supported_color_modes.add(ColorMode.RGBW)
            self._attr_color_mode: ColorMode = ColorMode.RGBW
        if topic[CONF_RGBWW_COMMAND_TOPIC] is not None:
            supported_color_modes.add(ColorMode.RGBWW)
            self._attr_color_mode: ColorMode = ColorMode.RGBWW
        if topic[CONF_WHITE_COMMAND_TOPIC] is not None:
            supported_color_modes.add(ColorMode.WHITE)
        if topic[CONF_XY_COMMAND_TOPIC] is not None:
            supported_color_modes.add(ColorMode.XY)
            self._attr_color_mode: ColorMode = ColorMode.XY
        if len(supported_color_modes) > 1:
            self._attr_color_mode: ColorMode = ColorMode.UNKNOWN
        if not supported_color_modes:
            if topic[CONF_BRIGHTNESS_COMMAND_TOPIC] is not None:
                self._attr_color_mode: ColorMode = ColorMode.BRIGHTNESS
                supported_color_modes.add(ColorMode.BRIGHTNESS)
            else:
                self._attr_color_mode: ColorMode = ColorMode.ONOFF
                supported_color_modes.add(ColorMode.ONOFF)
        self._attr_supported_color_modes: set = valid_supported_color_modes(supported_color_modes)
        self._attr_supported_features: LightEntityFeature = LightEntityFeature(0)
        if topic[CONF_EFFECT_COMMAND_TOPIC] is not None:
            self._attr_supported_features |= LightEntityFeature.EFFECT

    def _is_optimistic(self, attribute: str) -> bool:
        """Return True if the attribute is optimistically updated."""
        attr: bool = getattr(self, f'_optimistic_{attribute}')
        return attr

    @callback
    def _state_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages."""
        payload: Union[str, None] = self._value_templates[CONF_STATE_VALUE_TEMPLATE](msg.payload, PayloadSentinel.NONE)
        if not payload:
            _LOGGER.debug("Ignoring empty state message from '%s'", msg.topic)
            return
        if payload == self._payload['on']:
            self._attr_is_on: bool = True
        elif payload == self._payload['off']:
            self._attr_is_on: bool = False
        elif payload == PAYLOAD_NONE:
            self._attr_is_on: bool = None

    @callback
    def _brightness_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages for the brightness."""
        payload: Union[str, None] = self._value_templates[CONF_BRIGHTNESS_VALUE_TEMPLATE](msg.payload, PayloadSentinel.DEFAULT)
        if payload is PayloadSentinel.DEFAULT or not payload:
            _LOGGER.debug("Ignoring empty brightness message from '%s'", msg.topic)
            return
        device_value: float = float(payload)
        if device_value == 0:
            _LOGGER.debug("Ignoring zero brightness from '%s'", msg.topic)
            return
        percent_bright: int = min(round(device_value / self._config[CONF_BRIGHTNESS_SCALE] * 255), 255)
        self._attr_brightness: int = percent_bright

    @callback
    def _rgbx_received(self, msg: ReceiveMessage, template: str, color_mode: ColorMode, convert_color: Callable) -> Union[Tuple[int, ...], None]:
        """Process MQTT messages for RGBW and RGBWW."""
        payload: Union[str, None] = self._value_templates[template](msg.payload, PayloadSentinel.DEFAULT)
        if payload is PayloadSentinel.DEFAULT or not payload:
            _LOGGER.debug("Ignoring empty %s message from '%s'", color_mode, msg.topic)
            return None
        color: Tuple[int, ...] = tuple((int(val) for val in str(payload).split(',')))
        if self._optimistic_color_mode:
            self._attr_color_mode: ColorMode = color_mode
        if self._topic[CONF_BRIGHTNESS_STATE_TOPIC] is None:
            rgb: Tuple[int, ...] = convert_color(*color)
            brightness: int = max(rgb)
            if brightness == 0:
                _LOGGER.debug("Ignoring %s message with zero rgb brightness from '%s'", color_mode, msg.topic)
                return None
            self._attr_brightness: int = brightness
            color: Tuple[int, ...] = tuple((min(round(channel / brightness * 255), 255) for channel in color))
        return color

    @callback
    def _rgb_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages for RGB."""
        rgb: Union[Tuple[int, ...], None] = self._rgbx_received(msg, CONF_RGB_VALUE_TEMPLATE, ColorMode.RGB, lambda *x: x)
        if rgb is None:
            return
        self._attr_rgb_color: Tuple[int, int, int] = cast(Tuple[int, int, int], rgb)

    @callback
    def _rgbw_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages for RGBW."""
        rgbw: Union[Tuple[int, ...], None] = self._rgbx_received(msg, CONF_RGBW_VALUE_TEMPLATE, ColorMode.RGBW, color_util.color_rgbw_to_rgb)
        if rgbw is None:
            return
        self._attr_rgbw_color: Tuple[int, int, int, int] = cast(Tuple[int, int, int, int], rgbw)

    @callback
    def _rgbww_received(self, msg:
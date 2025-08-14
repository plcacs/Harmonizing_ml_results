from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any, Optional, Tuple, Union, cast, Dict

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

_LOGGER = logging.getLogger(__name__)

CONF_BRIGHTNESS_COMMAND_TEMPLATE = "brightness_command_template"
CONF_BRIGHTNESS_COMMAND_TOPIC = "brightness_command_topic"
CONF_BRIGHTNESS_SCALE = "brightness_scale"
CONF_BRIGHTNESS_STATE_TOPIC = "brightness_state_topic"
CONF_BRIGHTNESS_VALUE_TEMPLATE = "brightness_value_template"
CONF_COLOR_MODE_STATE_TOPIC = "color_mode_state_topic"
CONF_COLOR_MODE_VALUE_TEMPLATE = "color_mode_value_template"
CONF_COLOR_TEMP_COMMAND_TEMPLATE = "color_temp_command_template"
CONF_COLOR_TEMP_COMMAND_TOPIC = "color_temp_command_topic"
CONF_COLOR_TEMP_STATE_TOPIC = "color_temp_state_topic"
CONF_COLOR_TEMP_VALUE_TEMPLATE = "color_temp_value_template"
CONF_EFFECT_COMMAND_TEMPLATE = "effect_command_template"
CONF_EFFECT_COMMAND_TOPIC = "effect_command_topic"
CONF_EFFECT_LIST = "effect_list"
CONF_EFFECT_STATE_TOPIC = "effect_state_topic"
CONF_EFFECT_VALUE_TEMPLATE = "effect_value_template"
CONF_HS_COMMAND_TEMPLATE = "hs_command_template"
CONF_HS_COMMAND_TOPIC = "hs_command_topic"
CONF_HS_STATE_TOPIC = "hs_state_topic"
CONF_HS_VALUE_TEMPLATE = "hs_value_template"
CONF_MAX_MIREDS = "max_mireds"
CONF_MIN_MIREDS = "min_mireds"
CONF_RGB_COMMAND_TEMPLATE = "rgb_command_template"
CONF_RGB_COMMAND_TOPIC = "rgb_command_topic"
CONF_RGB_STATE_TOPIC = "rgb_state_topic"
CONF_RGB_VALUE_TEMPLATE = "rgb_value_template"
CONF_RGBW_COMMAND_TEMPLATE = "rgbw_command_template"
CONF_RGBW_COMMAND_TOPIC = "rgbw_command_topic"
CONF_RGBW_STATE_TOPIC = "rgbw_state_topic"
CONF_RGBW_VALUE_TEMPLATE = "rgbw_value_template"
CONF_RGBWW_COMMAND_TEMPLATE = "rgbww_command_template"
CONF_RGBWW_COMMAND_TOPIC = "rgbww_command_topic"
CONF_RGBWW_STATE_TOPIC = "rgbww_state_topic"
CONF_RGBWW_VALUE_TEMPLATE = "rgbww_value_template"
CONF_XY_COMMAND_TEMPLATE = "xy_command_template"
CONF_XY_COMMAND_TOPIC = "xy_command_topic"
CONF_XY_STATE_TOPIC = "xy_state_topic"
CONF_XY_VALUE_TEMPLATE = "xy_value_template"
CONF_WHITE_COMMAND_TOPIC = "white_command_topic"
CONF_WHITE_SCALE = "white_scale"
CONF_ON_COMMAND_TYPE = "on_command_type"

MQTT_LIGHT_ATTRIBUTES_BLOCKED = frozenset(
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

DEFAULT_BRIGHTNESS_SCALE = 255
DEFAULT_NAME = "MQTT LightEntity"
DEFAULT_PAYLOAD_OFF = "OFF"
DEFAULT_PAYLOAD_ON = "ON"
DEFAULT_WHITE_SCALE = 255
DEFAULT_ON_COMMAND_TYPE = "last"

VALUES_ON_COMMAND_TYPE = ["first", "last", "brightness"]

COMMAND_TEMPLATE_KEYS = [
    CONF_BRIGHTNESS_COMMAND_TEMPLATE,
    CONF_COLOR_TEMP_COMMAND_TEMPLATE,
    CONF_EFFECT_COMMAND_TEMPLATE,
    CONF_HS_COMMAND_TEMPLATE,
    CONF_RGB_COMMAND_TEMPLATE,
    CONF_RGBW_COMMAND_TEMPLATE,
    CONF_RGBWW_COMMAND_TEMPLATE,
    CONF_XY_COMMAND_TEMPLATE,
]
VALUE_TEMPLATE_KEYS = [
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

PLATFORM_SCHEMA_MODERN_BASIC = (
    MQTT_RW_SCHEMA.extend(
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
            vol.Optional(CONF_ON_COMMAND_TYPE, default=DEFAULT_ON_COMMAND_TYPE): vol.In(
                VALUES_ON_COMMAND_TYPE
            ),
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
            vol.Optional(CONF_WHITE_SCALE, default=DEFAULT_WHITE_SCALE): vol.All(
                vol.Coerce(int), vol.Range(min=1)
            ),
            vol.Optional(CONF_XY_COMMAND_TEMPLATE): cv.template,
            vol.Optional(CONF_XY_COMMAND_TOPIC): valid_publish_topic,
            vol.Optional(CONF_XY_STATE_TOPIC): valid_subscribe_topic,
            vol.Optional(CONF_XY_VALUE_TEMPLATE): cv.template,
        },
    )
    .extend(MQTT_ENTITY_COMMON_SCHEMA.schema)
    .extend(MQTT_LIGHT_SCHEMA_SCHEMA.schema)
)

DISCOVERY_SCHEMA_BASIC: VolSchemaType = vol.All(
    PLATFORM_SCHEMA_MODERN_BASIC.extend({}, extra=vol.REMOVE_EXTRA),
)


class MqttLight(MqttEntity, LightEntity, RestoreEntity):
    """Representation of a MQTT light."""

    _default_name: str = DEFAULT_NAME
    _entity_id_format: str = ENTITY_ID_FORMAT
    _attributes_extra_blocked = MQTT_LIGHT_ATTRIBUTES_BLOCKED
    _topic: Dict[str, Optional[str]]
    _payload: Dict[str, str]
    _color_temp_kelvin: bool
    _command_templates: Dict[
        str, Callable[[PublishPayloadType, TemplateVarsType], PublishPayloadType]
    ]
    _value_templates: Dict[
        str, Callable[[ReceivePayloadType, ReceivePayloadType], ReceivePayloadType]
    ]
    _optimistic: bool
    _optimistic_brightness: bool
    _optimistic_color_mode: bool
    _optimistic_color_temp_kelvin: bool
    _optimistic_effect: bool
    _optimistic_hs_color: bool
    _optimistic_rgb_color: bool
    _optimistic_rgbw_color: bool
    _optimistic_rgbww_color: bool
    _optimistic_xy_color: bool

    @staticmethod
    def config_schema() -> VolSchemaType:
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

        topic: Dict[str, Optional[str]] = {
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
        self._topic = topic
        self._payload = {"on": config[CONF_PAYLOAD_ON], "off": config[CONF_PAYLOAD_OFF]}

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

        optimistic: bool = config[CONF_OPTIMISTIC]
        self._optimistic_color_mode = (
            optimistic or topic[CONF_COLOR_MODE_STATE_TOPIC] is None
        )
        self._optimistic = optimistic or topic[CONF_STATE_TOPIC] is None
        self._attr_assumed_state = bool(self._optimistic)
        self._optimistic_rgb_color = optimistic or topic[CONF_RGB_STATE_TOPIC] is None
        self._optimistic_rgbw_color = optimistic or topic[CONF_RGBW_STATE_TOPIC] is None
        self._optimistic_rgbww_color = (
            optimistic or topic[CONF_RGBWW_STATE_TOPIC] is None
        )
        self._optimistic_brightness = (
            optimistic
            or (
                topic[CONF_BRIGHTNESS_COMMAND_TOPIC] is not None
                and topic[CONF_BRIGHTNESS_STATE_TOPIC] is None
            )
            or (
                topic[CONF_BRIGHTNESS_COMMAND_TOPIC] is None
                and topic[CONF_RGB_STATE_TOPIC] is None
            )
        )
        self._optimistic_color_temp_kelvin = (
            optimistic or topic[CONF_COLOR_TEMP_STATE_TOPIC] is None
        )
        self._optimistic_effect = optimistic or topic[CONF_EFFECT_STATE_TOPIC] is None
        self._optimistic_hs_color = optimistic or topic[CONF_HS_STATE_TOPIC] is None
        self._optimistic_xy_color = optimistic or topic[CONF_XY_STATE_TOPIC] is None
        supported_color_modes: set[ColorMode] = set()
        if topic[CONF_COLOR_TEMP_COMMAND_TOPIC] is not None:
            supported_color_modes.add(ColorMode.COLOR_TEMP)
            self._attr_color_mode = ColorMode.COLOR_TEMP
        if topic[CONF_HS_COMMAND_TOPIC] is not None:
            supported_color_modes.add(ColorMode.HS)
            self._attr_color_mode = ColorMode.HS
        if topic[CONF_RGB_COMMAND_TOPIC] is not None:
            supported_color_modes.add(ColorMode.RGB)
            self._attr_color_mode = ColorMode.RGB
        if topic[CONF_RGBW_COMMAND_TOPIC] is not None:
            supported_color_modes.add(ColorMode.RGBW)
            self._attr_color_mode = ColorMode.RGBW
        if topic[CONF_RGBWW_COMMAND_TOPIC] is not None:
            supported_color_modes.add(ColorMode.RGBWW)
            self._attr_color_mode = ColorMode.RGBWW
        if topic[CONF_WHITE_COMMAND_TOPIC] is not None:
            supported_color_modes.add(ColorMode.WHITE)
        if topic[CONF_XY_COMMAND_TOPIC] is not None:
            supported_color_modes.add(ColorMode.XY)
            self._attr_color_mode = ColorMode.XY
        if len(supported_color_modes) > 1:
            self._attr_color_mode = ColorMode.UNKNOWN

        if not supported_color_modes:
            if topic[CONF_BRIGHTNESS_COMMAND_TOPIC] is not None:
                self._attr_color_mode = ColorMode.BRIGHTNESS
                supported_color_modes.add(ColorMode.BRIGHTNESS)
            else:
                self._attr_color_mode = ColorMode.ONOFF
                supported_color_modes.add(ColorMode.ONOFF)

        # Validate the color_modes configuration
        self._attr_supported_color_modes = valid_supported_color_modes(
            supported_color_modes
        )

        self._attr_supported_features = LightEntityFeature(0)
        if topic[CONF_EFFECT_COMMAND_TOPIC] is not None:
            self._attr_supported_features |= LightEntityFeature.EFFECT

    def _is_optimistic(self, attribute: str) -> bool:
        """Return True if the attribute is optimistically updated."""
        attr: bool = getattr(self, f"_optimistic_{attribute}")
        return attr

    @callback
    def _state_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages."""
        payload: Any = self._value_templates[CONF_STATE_VALUE_TEMPLATE](
            msg.payload, PayloadSentinel.NONE
        )
        if not payload:
            _LOGGER.debug("Ignoring empty state message from '%s'", msg.topic)
            return

        if payload == self._payload["on"]:
            self._attr_is_on = True
        elif payload == self._payload["off"]:
            self._attr_is_on = False
        elif payload == PAYLOAD_NONE:
            self._attr_is_on = None

    @callback
    def _brightness_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages for the brightness."""
        payload: Any = self._value_templates[CONF_BRIGHTNESS_VALUE_TEMPLATE](
            msg.payload, PayloadSentinel.DEFAULT
        )
        if payload is PayloadSentinel.DEFAULT or not payload:
            _LOGGER.debug("Ignoring empty brightness message from '%s'", msg.topic)
            return

        device_value: float = float(payload)
        if device_value == 0:
            _LOGGER.debug("Ignoring zero brightness from '%s'", msg.topic)
            return

        percent_bright: float = device_value / self._config[CONF_BRIGHTNESS_SCALE]  # type: ignore[attr-defined]
        self._attr_brightness = min(round(percent_bright * 255), 255)

    @callback
    def _rgbx_received(
        self,
        msg: ReceiveMessage,
        template: str,
        color_mode: ColorMode,
        convert_color: Callable[..., Tuple[int, ...]],
    ) -> Optional[Tuple[int, ...]]:
        """Process MQTT messages for RGBW and RGBWW."""
        payload: Any = self._value_templates[template](msg.payload, PayloadSentinel.DEFAULT)
        if payload is PayloadSentinel.DEFAULT or not payload:
            _LOGGER.debug("Ignoring empty %s message from '%s'", color_mode, msg.topic)
            return None
        color: Tuple[int, ...] = tuple(int(val) for val in str(payload).split(","))
        if self._optimistic_color_mode:
            self._attr_color_mode = color_mode
        if self._topic[CONF_BRIGHTNESS_STATE_TOPIC] is None:
            rgb: Tuple[int, ...] = convert_color(*color)
            brightness: int = max(rgb)
            if brightness == 0:
                _LOGGER.debug(
                    "Ignoring %s message with zero rgb brightness from '%s'",
                    color_mode,
                    msg.topic,
                )
                return None
            self._attr_brightness = brightness
            # Normalize the color to 100% brightness
            color = tuple(
                min(round(channel / brightness * 255), 255) for channel in color
            )
        return color

    @callback
    def _rgb_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages for RGB."""
        rgb: Optional[Tuple[int, int, int]] = self._rgbx_received(
            msg, CONF_RGB_VALUE_TEMPLATE, ColorMode.RGB, lambda *x: x  # type: ignore
        )
        if rgb is None:
            return
        self._attr_rgb_color = cast(Tuple[int, int, int], rgb)

    @callback
    def _rgbw_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages for RGBW."""
        rgbw: Optional[Tuple[int, int, int, int]] = self._rgbx_received(
            msg, CONF_RGBW_VALUE_TEMPLATE, ColorMode.RGBW, color_util.color_rgbw_to_rgb
        )
        if rgbw is None:
            return
        self._attr_rgbw_color = cast(Tuple[int, int, int, int], rgbw)

    @callback
    def _rgbww_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages for RGBWW."""

        @callback
        def _converter(
            r: int, g: int, b: int, cw: int, ww: int
        ) -> Tuple[int, int, int]:
            return color_util.color_rgbww_to_rgb(
                r, g, b, cw, ww, self.min_color_temp_kelvin, self.max_color_temp_kelvin  # type: ignore[attr-defined]
            )

        rgbww: Optional[Tuple[int, int, int, int, int]] = self._rgbx_received(
            msg,
            CONF_RGBWW_VALUE_TEMPLATE,
            ColorMode.RGBWW,
            _converter,
        )
        if rgbww is None:
            return
        self._attr_rgbww_color = cast(Tuple[int, int, int, int, int], rgbww)

    @callback
    def _color_mode_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages for color mode."""
        payload: Any = self._value_templates[CONF_COLOR_MODE_VALUE_TEMPLATE](
            msg.payload, PayloadSentinel.DEFAULT
        )
        if payload is PayloadSentinel.DEFAULT or not payload:
            _LOGGER.debug("Ignoring empty color mode message from '%s'", msg.topic)
            return

        self._attr_color_mode = ColorMode(str(payload))

    @callback
    def _color_temp_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages for color temperature."""
        payload: Any = self._value_templates[CONF_COLOR_TEMP_VALUE_TEMPLATE](
            msg.payload, PayloadSentinel.DEFAULT
        )
        if payload is PayloadSentinel.DEFAULT or not payload:
            _LOGGER.debug("Ignoring empty color temp message from '%s'", msg.topic)
            return

        if self._optimistic_color_mode:
            self._attr_color_mode = ColorMode.COLOR_TEMP
        if self._color_temp_kelvin:
            self._attr_color_temp_kelvin = int(payload)
            return
        self._attr_color_temp_kelvin = color_util.color_temperature_mired_to_kelvin(
            int(payload)
        )

    @callback
    def _effect_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages for effect."""
        payload: Any = self._value_templates[CONF_EFFECT_VALUE_TEMPLATE](
            msg.payload, PayloadSentinel.DEFAULT
        )
        if payload is PayloadSentinel.DEFAULT or not payload:
            _LOGGER.debug("Ignoring empty effect message from '%s'", msg.topic)
            return

        self._attr_effect = str(payload)

    @callback
    def _hs_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages for hs color."""
        payload: Any = self._value_templates[CONF_HS_VALUE_TEMPLATE](
            msg.payload, PayloadSentinel.DEFAULT
        )
        if payload is PayloadSentinel.DEFAULT or not payload:
            _LOGGER.debug("Ignoring empty hs message from '%s'", msg.topic)
            return
        try:
            hs_color: Tuple[float, float] = tuple(float(val) for val in str(payload).split(",", 2))
            if self._optimistic_color_mode:
                self._attr_color_mode = ColorMode.HS
            self._attr_hs_color = cast(Tuple[float, float], hs_color)
        except ValueError:
            _LOGGER.warning("Failed to parse hs state update: '%s'", payload)

    @callback
    def _xy_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages for xy color."""
        payload: Any = self._value_templates[CONF_XY_VALUE_TEMPLATE](
            msg.payload, PayloadSentinel.DEFAULT
        )
        if payload is PayloadSentinel.DEFAULT or not payload:
            _LOGGER.debug("Ignoring empty xy-color message from '%s'", msg.topic)
            return

        xy_color: Tuple[float, float] = tuple(float(val) for val in str(payload).split(",", 2))
        if self._optimistic_color_mode:
            self._attr_color_mode = ColorMode.XY
        self._attr_xy_color = cast(Tuple[float, float], xy_color)

    @callback
    def _prepare_subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""
        self.add_subscription(CONF_STATE_TOPIC, self._state_received, {"_attr_is_on"})
        self.add_subscription(
            CONF_BRIGHTNESS_STATE_TOPIC, self._brightness_received, {"_attr_brightness"}
        )
        self.add_subscription(
            CONF_RGB_STATE_TOPIC,
            self._rgb_received,
            {"_attr_brightness", "_attr_color_mode", "_attr_rgb_color"},
        )
        self.add_subscription(
            CONF_RGBW_STATE_TOPIC,
            self._rgbw_received,
            {"_attr_brightness", "_attr_color_mode", "_attr_rgbw_color"},
        )
        self.add_subscription(
            CONF_RGBWW_STATE_TOPIC,
            self._rgbww_received,
            {"_attr_brightness", "_attr_color_mode", "_attr_rgbww_color"},
        )
        self.add_subscription(
            CONF_COLOR_MODE_STATE_TOPIC, self._color_mode_received, {"_attr_color_mode"}
        )
        self.add_subscription(
            CONF_COLOR_TEMP_STATE_TOPIC,
            self._color_temp_received,
            {"_attr_color_mode", "_attr_color_temp_kelvin"},
        )
        self.add_subscription(
            CONF_EFFECT_STATE_TOPIC, self._effect_received, {"_attr_effect"}
        )
        self.add_subscription(
            CONF_HS_STATE_TOPIC,
            self._hs_received,
            {"_attr_color_mode", "_attr_hs_color"},
        )
        self.add_subscription(
            CONF_XY_STATE_TOPIC,
            self._xy_received,
            {"_attr_color_mode", "_attr_xy_color"},
        )

    async def _subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""
        subscription.async_subscribe_topics_internal(self.hass, self._sub_state)  # type: ignore[attr-defined]
        last_state = await self.async_get_last_state()

        def restore_state(
            attribute: str, condition_attribute: Optional[str] = None
        ) -> None:
            """Restore a state attribute."""
            if condition_attribute is None:
                condition_attribute = attribute
            optimistic: bool = self._is_optimistic(condition_attribute)
            if optimistic and last_state and last_state.attributes.get(attribute):
                setattr(self, f"_attr_{attribute}", last_state.attributes[attribute])

        if self._topic[CONF_STATE_TOPIC] is None and self._optimistic and last_state:
            self._attr_is_on = last_state.state == STATE_ON
        restore_state(ATTR_BRIGHTNESS)
        restore_state(ATTR_RGB_COLOR)
        restore_state(ATTR_HS_COLOR, ATTR_RGB_COLOR)
        restore_state(ATTR_RGBW_COLOR)
        restore_state(ATTR_RGBWW_COLOR)
        restore_state(ATTR_COLOR_MODE)
        restore_state(ATTR_COLOR_TEMP_KELVIN)
        restore_state(ATTR_EFFECT)
        restore_state(ATTR_HS_COLOR)
        restore_state(ATTR_XY_COLOR)
        restore_state(ATTR_HS_COLOR, ATTR_XY_COLOR)

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the device on.

        This method is a coroutine.
        """
        should_update: bool = False
        on_command_type: str = self._config[CONF_ON_COMMAND_TYPE]  # type: ignore[attr-defined]

        async def publish(topic: str, payload: PublishPayloadType) -> None:
            """Publish an MQTT message."""
            await self.async_publish_with_config(str(self._topic[topic]), payload)  # type: ignore[attr-defined]

        def scale_rgbx(
            color: Tuple[int, ...],
            brightness: Optional[int] = None,
        ) -> Tuple[int, ...]:
            """Scale RGBx for brightness."""
            if brightness is None:
                if self._topic[CONF_BRIGHTNESS_COMMAND_TOPIC] is not None:
                    brightness = 255
                else:
                    brightness = kwargs.get(ATTR_BRIGHTNESS) or self.brightness or 255  # type: ignore[attr-defined]
            return tuple(int(channel * brightness / 255) for channel in color)

        def render_rgbx(
            color: Tuple[int, ...],
            template: str,
            color_mode: ColorMode,
        ) -> PublishPayloadType:
            """Render RGBx payload."""
            rgb_color_str: str = ",".join(str(channel) for channel in color)
            keys: list[str] = ["red", "green", "blue"]
            if color_mode == ColorMode.RGBW:
                keys.append("white")
            elif color_mode == ColorMode.RGBWW:
                keys.extend(["cold_white", "warm_white"])
            variables: Dict[str, Any] = dict(zip(keys, color, strict=False))
            return self._command_templates[template](rgb_color_str, variables)

        def set_optimistic(
            attribute: str,
            value: Any,
            color_mode: Optional[ColorMode] = None,
            condition_attribute: Optional[str] = None,
        ) -> bool:
            """Optimistically update a state attribute."""
            if condition_attribute is None:
                condition_attribute = attribute
            if not self._is_optimistic(condition_attribute):
                return False
            if color_mode and self._optimistic_color_mode:
                self._attr_color_mode = color_mode
            setattr(self, f"_attr_{attribute}", value)
            return True

        if on_command_type == "first":
            await publish(CONF_COMMAND_TOPIC, self._payload["on"])
            should_update = True

        elif (
            on_command_type == "brightness"
            and ATTR_BRIGHTNESS not in kwargs
            and ATTR_WHITE not in kwargs
        ):
            kwargs[ATTR_BRIGHTNESS] = self.brightness or 255  # type: ignore[attr-defined]

        hs_color: Optional[Tuple[float, float]] = kwargs.get(ATTR_HS_COLOR)
        if hs_color and self._topic[CONF_HS_COMMAND_TOPIC] is not None:
            device_hs_payload: PublishPayloadType = self._command_templates[CONF_HS_COMMAND_TEMPLATE](
                f"{hs_color[0]},{hs_color[1]}",
                {"hue": hs_color[0], "sat": hs_color[1]},
            )
            await publish(CONF_HS_COMMAND_TOPIC, device_hs_payload)
            should_update |= set_optimistic(ATTR_HS_COLOR, hs_color, ColorMode.HS)

        rgb: Optional[Tuple[int, int, int]] = kwargs.get(ATTR_RGB_COLOR)
        if rgb and self._topic[CONF_RGB_COMMAND_TOPIC] is not None:
            scaled: Tuple[int, ...] = scale_rgbx(rgb)
            rgb_s: PublishPayloadType = render_rgbx(scaled, CONF_RGB_COMMAND_TEMPLATE, ColorMode.RGB)
            await publish(CONF_RGB_COMMAND_TOPIC, rgb_s)
            should_update |= set_optimistic(ATTR_RGB_COLOR, rgb, ColorMode.RGB)

        rgbw: Optional[Tuple[int, int, int, int]] = kwargs.get(ATTR_RGBW_COLOR)
        if rgbw and self._topic[CONF_RGBW_COMMAND_TOPIC] is not None:
            scaled = scale_rgbx(rgbw)
            rgbw_s: PublishPayloadType = render_rgbx(scaled, CONF_RGBW_COMMAND_TEMPLATE, ColorMode.RGBW)
            await publish(CONF_RGBW_COMMAND_TOPIC, rgbw_s)
            should_update |= set_optimistic(ATTR_RGBW_COLOR, rgbw, ColorMode.RGBW)

        rgbww: Optional[Tuple[int, int, int, int, int]] = kwargs.get(ATTR_RGBWW_COLOR)
        if rgbww and self._topic[CONF_RGBWW_COMMAND_TOPIC] is not None:
            scaled = scale_rgbx(rgbww)
            rgbww_s: PublishPayloadType = render_rgbx(scaled, CONF_RGBWW_COMMAND_TEMPLATE, ColorMode.RGBWW)
            await publish(CONF_RGBWW_COMMAND_TOPIC, rgbww_s)
            should_update |= set_optimistic(ATTR_RGBWW_COLOR, rgbww, ColorMode.RGBWW)

        xy_color: Optional[Tuple[float, float]] = kwargs.get(ATTR_XY_COLOR)
        if xy_color and self._topic[CONF_XY_COMMAND_TOPIC] is not None:
            device_xy_payload: PublishPayloadType = self._command_templates[CONF_XY_COMMAND_TEMPLATE](
                f"{xy_color[0]},{xy_color[1]}",
                {"x": xy_color[0], "y": xy_color[1]},
            )
            await publish(CONF_XY_COMMAND_TOPIC, device_xy_payload)
            should_update |= set_optimistic(ATTR_XY_COLOR, xy_color, ColorMode.XY)

        if (
            ATTR_BRIGHTNESS in kwargs
            and self._topic[CONF_BRIGHTNESS_COMMAND_TOPIC] is not None
        ):
            brightness_normalized: float = kwargs[ATTR_BRIGHTNESS] / 255
            brightness_scale: int = self._config[CONF_BRIGHTNESS_SCALE]  # type: ignore[attr-defined]
            device_brightness: int = min(
                round(brightness_normalized * brightness_scale), brightness_scale
            )
            device_brightness = max(device_brightness, 1)
            command_tpl: Callable[[Any, Optional[Any]], PublishPayloadType] = self._command_templates[CONF_BRIGHTNESS_COMMAND_TEMPLATE]
            device_brightness_payload: PublishPayloadType = command_tpl(device_brightness, None)
            await publish(CONF_BRIGHTNESS_COMMAND_TOPIC, device_brightness_payload)
            should_update |= set_optimistic(ATTR_BRIGHTNESS, kwargs[ATTR_BRIGHTNESS])
        elif (
            ATTR_BRIGHTNESS in kwargs
            and ATTR_RGB_COLOR not in kwargs
            and self._topic[CONF_RGB_COMMAND_TOPIC] is not None
        ):
            rgb_color = self.rgb_color or (255,) * 3  # type: ignore[attr-defined]
            rgb_scaled = scale_rgbx(rgb_color, kwargs[ATTR_BRIGHTNESS])
            rgb_s = render_rgbx(rgb_scaled, CONF_RGB_COMMAND_TEMPLATE, ColorMode.RGB)
            await publish(CONF_RGB_COMMAND_TOPIC, rgb_s)
            should_update |= set_optimistic(ATTR_BRIGHTNESS, kwargs[ATTR_BRIGHTNESS])
        elif (
            ATTR_BRIGHTNESS in kwargs
            and ATTR_RGBW_COLOR not in kwargs
            and self._topic[CONF_RGBW_COMMAND_TOPIC] is not None
        ):
            rgbw_color = self.rgbw_color or (255,) * 4  # type: ignore[attr-defined]
            rgbw_b = scale_rgbx(rgbw_color, kwargs[ATTR_BRIGHTNESS])
            rgbw_s = render_rgbx(rgbw_b, CONF_RGBW_COMMAND_TEMPLATE, ColorMode.RGBW)
            await publish(CONF_RGBW_COMMAND_TOPIC, rgbw_s)
            should_update |= set_optimistic(ATTR_BRIGHTNESS, kwargs[ATTR_BRIGHTNESS])
        elif (
            ATTR_BRIGHTNESS in kwargs
            and ATTR_RGBWW_COLOR not in kwargs
            and self._topic[CONF_RGBWW_COMMAND_TOPIC] is not None
        ):
            rgbww_color = self.rgbww_color or (255,) * 5  # type: ignore[attr-defined]
            rgbww_b = scale_rgbx(rgbww_color, kwargs[ATTR_BRIGHTNESS])
            rgbww_s = render_rgbx(rgbww_b, CONF_RGBWW_COMMAND_TEMPLATE, ColorMode.RGBWW)
            await publish(CONF_RGBWW_COMMAND_TOPIC, rgbww_s)
            should_update |= set_optimistic(ATTR_BRIGHTNESS, kwargs[ATTR_BRIGHTNESS])
        if (
            ATTR_COLOR_TEMP_KELVIN in kwargs
            and self._topic[CONF_COLOR_TEMP_COMMAND_TOPIC] is not None
        ):
            ct_command_tpl: Callable[[Any, Optional[Any]], PublishPayloadType] = self._command_templates[CONF_COLOR_TEMP_COMMAND_TEMPLATE]
            color_temp = ct_command_tpl(
                kwargs[ATTR_COLOR_TEMP_KELVIN]
                if self._color_temp_kelvin
                else color_util.color_temperature_kelvin_to_mired(
                    kwargs[ATTR_COLOR_TEMP_KELVIN]
                ),
                None,
            )
            await publish(CONF_COLOR_TEMP_COMMAND_TOPIC, color_temp)
            should_update |= set_optimistic(
                ATTR_COLOR_TEMP_KELVIN,
                kwargs[ATTR_COLOR_TEMP_KELVIN],
                ColorMode.COLOR_TEMP,
            )

        if (
            ATTR_EFFECT in kwargs
            and self._topic[CONF_EFFECT_COMMAND_TOPIC] is not None
            and CONF_EFFECT_LIST in self._config  # type: ignore[attr-defined]
        ):
            if kwargs[ATTR_EFFECT] in self._config[CONF_EFFECT_LIST]:  # type: ignore[attr-defined]
                eff_command_tpl: Callable[[Any, Optional[Any]], PublishPayloadType] = self._command_templates[CONF_EFFECT_COMMAND_TEMPLATE]
                effect = eff_command_tpl(kwargs[ATTR_EFFECT], None)
                await publish(CONF_EFFECT_COMMAND_TOPIC, effect)
                should_update |= set_optimistic(ATTR_EFFECT, kwargs[ATTR_EFFECT])
        if ATTR_WHITE in kwargs and self._topic[CONF_WHITE_COMMAND_TOPIC] is not None:
            percent_white: float = float(kwargs[ATTR_WHITE]) / 255
            white_scale: int = self._config[CONF_WHITE_SCALE]  # type: ignore[attr-defined]
            device_white_value: int = min(round(percent_white * white_scale), white_scale)
            await publish(CONF_WHITE_COMMAND_TOPIC, device_white_value)
            should_update |= set_optimistic(
                ATTR_BRIGHTNESS,
                kwargs[ATTR_WHITE],
                ColorMode.WHITE,
            )

        if on_command_type == "last":
            await publish(CONF_COMMAND_TOPIC, self._payload["on"])
            should_update = True

        if self._optimistic:
            self._attr_is_on = True
            should_update = True

        if should_update:
            self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the device off.

        This method is a coroutine.
        """
        await self.async_publish_with_config(
            str(self._topic[CONF_COMMAND_TOPIC]), self._payload["off"]
        )

        if self._optimistic:
            self._attr_is_on = False
            self.async_write_ha_state()
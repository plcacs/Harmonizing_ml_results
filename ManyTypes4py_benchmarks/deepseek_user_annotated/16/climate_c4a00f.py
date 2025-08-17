"""Support for MQTT climate devices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
import logging
from typing import Any, Final, TypedDict, cast

import voluptuous as vol

from homeassistant.components import climate
from homeassistant.components.climate import (
    ATTR_HVAC_MODE,
    ATTR_TARGET_TEMP_HIGH,
    ATTR_TARGET_TEMP_LOW,
    DEFAULT_MAX_HUMIDITY,
    DEFAULT_MIN_HUMIDITY,
    FAN_AUTO,
    FAN_HIGH,
    FAN_LOW,
    FAN_MEDIUM,
    PRESET_NONE,
    SWING_OFF,
    SWING_ON,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_TEMPERATURE,
    CONF_NAME,
    CONF_OPTIMISTIC,
    CONF_PAYLOAD_OFF,
    CONF_PAYLOAD_ON,
    CONF_TEMPERATURE_UNIT,
    CONF_VALUE_TEMPLATE,
    PRECISION_HALVES,
    PRECISION_TENTHS,
    PRECISION_WHOLE,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.service_info.mqtt import ReceivePayloadType
from homeassistant.helpers.template import Template
from homeassistant.helpers.typing import ConfigType, VolSchemaType
from homeassistant.util.unit_conversion import TemperatureConverter

from . import subscription
from .config import DEFAULT_RETAIN, MQTT_BASE_SCHEMA
from .const import (
    CONF_ACTION_TEMPLATE,
    CONF_ACTION_TOPIC,
    CONF_CURRENT_HUMIDITY_TEMPLATE,
    CONF_CURRENT_HUMIDITY_TOPIC,
    CONF_CURRENT_TEMP_TEMPLATE,
    CONF_CURRENT_TEMP_TOPIC,
    CONF_MODE_COMMAND_TEMPLATE,
    CONF_MODE_COMMAND_TOPIC,
    CONF_MODE_LIST,
    CONF_MODE_STATE_TEMPLATE,
    CONF_MODE_STATE_TOPIC,
    CONF_POWER_COMMAND_TEMPLATE,
    CONF_POWER_COMMAND_TOPIC,
    CONF_PRECISION,
    CONF_RETAIN,
    CONF_TEMP_COMMAND_TEMPLATE,
    CONF_TEMP_COMMAND_TOPIC,
    CONF_TEMP_INITIAL,
    CONF_TEMP_MAX,
    CONF_TEMP_MIN,
    CONF_TEMP_STATE_TEMPLATE,
    CONF_TEMP_STATE_TOPIC,
    DEFAULT_OPTIMISTIC,
    PAYLOAD_NONE,
)
from .entity import MqttEntity, async_setup_entity_entry_helper
from .models import (
    MqttCommandTemplate,
    MqttValueTemplate,
    PublishPayloadType,
    ReceiveMessage,
)
from .schemas import MQTT_ENTITY_COMMON_SCHEMA
from .util import valid_publish_topic, valid_subscribe_topic

_LOGGER: Final = logging.getLogger(__name__)

PARALLEL_UPDATES: Final = 0

DEFAULT_NAME: Final = "MQTT HVAC"

CONF_FAN_MODE_COMMAND_TEMPLATE: Final = "fan_mode_command_template"
CONF_FAN_MODE_COMMAND_TOPIC: Final = "fan_mode_command_topic"
CONF_FAN_MODE_LIST: Final = "fan_modes"
CONF_FAN_MODE_STATE_TEMPLATE: Final = "fan_mode_state_template"
CONF_FAN_MODE_STATE_TOPIC: Final = "fan_mode_state_topic"

CONF_HUMIDITY_COMMAND_TEMPLATE: Final = "target_humidity_command_template"
CONF_HUMIDITY_COMMAND_TOPIC: Final = "target_humidity_command_topic"
CONF_HUMIDITY_STATE_TEMPLATE: Final = "target_humidity_state_template"
CONF_HUMIDITY_STATE_TOPIC: Final = "target_humidity_state_topic"
CONF_HUMIDITY_MAX: Final = "max_humidity"
CONF_HUMIDITY_MIN: Final = "min_humidity"

CONF_PRESET_MODE_STATE_TOPIC: Final = "preset_mode_state_topic"
CONF_PRESET_MODE_COMMAND_TOPIC: Final = "preset_mode_command_topic"
CONF_PRESET_MODE_VALUE_TEMPLATE: Final = "preset_mode_value_template"
CONF_PRESET_MODE_COMMAND_TEMPLATE: Final = "preset_mode_command_template"
CONF_PRESET_MODES_LIST: Final = "preset_modes"
CONF_SWING_MODE_COMMAND_TEMPLATE: Final = "swing_mode_command_template"
CONF_SWING_MODE_COMMAND_TOPIC: Final = "swing_mode_command_topic"
CONF_SWING_MODE_LIST: Final = "swing_modes"
CONF_SWING_MODE_STATE_TEMPLATE: Final = "swing_mode_state_template"
CONF_SWING_MODE_STATE_TOPIC: Final = "swing_mode_state_topic"
CONF_TEMP_HIGH_COMMAND_TEMPLATE: Final = "temperature_high_command_template"
CONF_TEMP_HIGH_COMMAND_TOPIC: Final = "temperature_high_command_topic"
CONF_TEMP_HIGH_STATE_TEMPLATE: Final = "temperature_high_state_template"
CONF_TEMP_HIGH_STATE_TOPIC: Final = "temperature_high_state_topic"
CONF_TEMP_LOW_COMMAND_TEMPLATE: Final = "temperature_low_command_template"
CONF_TEMP_LOW_COMMAND_TOPIC: Final = "temperature_low_command_topic"
CONF_TEMP_LOW_STATE_TEMPLATE: Final = "temperature_low_state_template"
CONF_TEMP_LOW_STATE_TOPIC: Final = "temperature_low_state_topic"
CONF_TEMP_STEP: Final = "temp_step"

DEFAULT_INITIAL_TEMPERATURE: Final = 21.0

MQTT_CLIMATE_ATTRIBUTES_BLOCKED: Final = frozenset(
    {
        climate.ATTR_CURRENT_HUMIDITY,
        climate.ATTR_CURRENT_TEMPERATURE,
        climate.ATTR_FAN_MODE,
        climate.ATTR_FAN_MODES,
        climate.ATTR_HUMIDITY,
        climate.ATTR_HVAC_ACTION,
        climate.ATTR_HVAC_MODES,
        climate.ATTR_MAX_HUMIDITY,
        climate.ATTR_MAX_TEMP,
        climate.ATTR_MIN_HUMIDITY,
        climate.ATTR_MIN_TEMP,
        climate.ATTR_PRESET_MODE,
        climate.ATTR_PRESET_MODES,
        climate.ATTR_SWING_MODE,
        climate.ATTR_SWING_MODES,
        climate.ATTR_TARGET_TEMP_HIGH,
        climate.ATTR_TARGET_TEMP_LOW,
        climate.ATTR_TARGET_TEMP_STEP,
        climate.ATTR_TEMPERATURE,
    }
)

VALUE_TEMPLATE_KEYS: Final = (
    CONF_CURRENT_HUMIDITY_TEMPLATE,
    CONF_CURRENT_TEMP_TEMPLATE,
    CONF_FAN_MODE_STATE_TEMPLATE,
    CONF_HUMIDITY_STATE_TEMPLATE,
    CONF_MODE_STATE_TEMPLATE,
    CONF_ACTION_TEMPLATE,
    CONF_PRESET_MODE_VALUE_TEMPLATE,
    CONF_SWING_MODE_STATE_TEMPLATE,
    CONF_TEMP_HIGH_STATE_TEMPLATE,
    CONF_TEMP_LOW_STATE_TEMPLATE,
    CONF_TEMP_STATE_TEMPLATE,
)

COMMAND_TEMPLATE_KEYS: Final = {
    CONF_FAN_MODE_COMMAND_TEMPLATE,
    CONF_HUMIDITY_COMMAND_TEMPLATE,
    CONF_MODE_COMMAND_TEMPLATE,
    CONF_POWER_COMMAND_TEMPLATE,
    CONF_PRESET_MODE_COMMAND_TEMPLATE,
    CONF_SWING_MODE_COMMAND_TEMPLATE,
    CONF_TEMP_COMMAND_TEMPLATE,
    CONF_TEMP_HIGH_COMMAND_TEMPLATE,
    CONF_TEMP_LOW_COMMAND_TEMPLATE,
}

TOPIC_KEYS: Final = (
    CONF_ACTION_TOPIC,
    CONF_CURRENT_HUMIDITY_TOPIC,
    CONF_CURRENT_TEMP_TOPIC,
    CONF_FAN_MODE_COMMAND_TOPIC,
    CONF_FAN_MODE_STATE_TOPIC,
    CONF_HUMIDITY_COMMAND_TOPIC,
    CONF_HUMIDITY_STATE_TOPIC,
    CONF_MODE_COMMAND_TOPIC,
    CONF_MODE_STATE_TOPIC,
    CONF_POWER_COMMAND_TOPIC,
    CONF_PRESET_MODE_COMMAND_TOPIC,
    CONF_PRESET_MODE_STATE_TOPIC,
    CONF_SWING_MODE_COMMAND_TOPIC,
    CONF_SWING_MODE_STATE_TOPIC,
    CONF_TEMP_COMMAND_TOPIC,
    CONF_TEMP_HIGH_COMMAND_TOPIC,
    CONF_TEMP_HIGH_STATE_TOPIC,
    CONF_TEMP_LOW_COMMAND_TOPIC,
    CONF_TEMP_LOW_STATE_TOPIC,
    CONF_TEMP_STATE_TOPIC,
)

class ClimateConfig(TypedDict, total=False):
    """Configuration for MQTT climate."""

    current_humidity_template: Template
    current_humidity_topic: str
    current_temp_template: Template
    current_temp_topic: str
    fan_mode_command_template: Template
    fan_mode_command_topic: str
    fan_modes: list[str]
    fan_mode_state_template: Template
    fan_mode_state_topic: str
    target_humidity_command_template: Template
    target_humidity_command_topic: str
    max_humidity: float
    min_humidity: float
    target_humidity_state_template: Template
    target_humidity_state_topic: str
    mode_command_template: Template
    mode_command_topic: str
    mode_list: list[HVACMode]
    mode_state_template: Template
    mode_state_topic: str
    name: str | None
    optimistic: bool
    payload_on: str
    payload_off: str
    power_command_topic: str
    power_command_template: Template
    precision: float
    retain: bool
    action_template: Template
    action_topic: str
    preset_mode_command_topic: str
    preset_modes: list[str]
    preset_mode_command_template: Template
    preset_mode_state_topic: str
    preset_mode_value_template: Template
    swing_mode_command_template: Template
    swing_mode_command_topic: str
    swing_modes: list[str]
    swing_mode_state_template: Template
    swing_mode_state_topic: str
    temp_initial: float
    temp_min: float
    temp_max: float
    temp_step: float
    temp_command_template: Template
    temp_command_topic: str
    temperature_high_command_template: Template
    temperature_high_command_topic: str
    temperature_high_state_topic: str
    temperature_high_state_template: Template
    temperature_low_command_template: Template
    temperature_low_command_topic: str
    temperature_low_state_template: Template
    temperature_low_state_topic: str
    temp_state_template: Template
    temp_state_topic: str
    temperature_unit: str
    value_template: Template

def valid_preset_mode_configuration(config: ConfigType) -> ConfigType:
    """Validate that the preset mode reset payload is not one of the preset modes."""
    if PRESET_NONE in config[CONF_PRESET_MODES_LIST]:
        raise vol.Invalid("preset_modes must not include preset mode 'none'")
    return config

def valid_humidity_range_configuration(config: ConfigType) -> ConfigType:
    """Validate a target_humidity range configuration, throws otherwise."""
    if config[CONF_HUMIDITY_MIN] >= config[CONF_HUMIDITY_MAX]:
        raise vol.Invalid("target_humidity_max must be > target_humidity_min")
    if config[CONF_HUMIDITY_MAX] > 100:
        raise vol.Invalid("max_humidity must be <= 100")
    return config

def valid_humidity_state_configuration(config: ConfigType) -> ConfigType:
    """Validate humidity state.

    Ensure that if CONF_HUMIDITY_STATE_TOPIC is set then
    CONF_HUMIDITY_COMMAND_TOPIC is also set.
    """
    if (
        CONF_HUMIDITY_STATE_TOPIC in config
        and CONF_HUMIDITY_COMMAND_TOPIC not in config
    ):
        raise vol.Invalid(
            f"{CONF_HUMIDITY_STATE_TOPIC} cannot be used without"
            f" {CONF_HUMIDITY_COMMAND_TOPIC}"
        )
    return config

_PLATFORM_SCHEMA_BASE: Final = MQTT_BASE_SCHEMA.extend(
    {
        vol.Optional(CONF_CURRENT_HUMIDITY_TEMPLATE): cv.template,
        vol.Optional(CONF_CURRENT_HUMIDITY_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_CURRENT_TEMP_TEMPLATE): cv.template,
        vol.Optional(CONF_CURRENT_TEMP_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_FAN_MODE_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_FAN_MODE_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(
            CONF_FAN_MODE_LIST,
            default=[FAN_AUTO, FAN_LOW, FAN_MEDIUM, FAN_HIGH],
        ): cv.ensure_list,
        vol.Optional(CONF_FAN_MODE_STATE_TEMPLATE): cv.template,
        vol.Optional(CONF_FAN_MODE_STATE_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_HUMIDITY_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_HUMIDITY_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(
            CONF_HUMIDITY_MIN, default=DEFAULT_MIN_HUMIDITY
        ): cv.positive_float,
        vol.Optional(
            CONF_HUMIDITY_MAX, default=DEFAULT_MAX_HUMIDITY
        ): cv.positive_float,
        vol.Optional(CONF_HUMIDITY_STATE_TEMPLATE): cv.template,
        vol.Optional(CONF_HUMIDITY_STATE_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_MODE_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_MODE_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(
            CONF_MODE_LIST,
            default=[
                HVACMode.AUTO,
                HVACMode.OFF,
                HVACMode.COOL,
                HVACMode.HEAT,
                HVACMode.DRY,
                HVACMode.FAN_ONLY,
            ],
        ): cv.ensure_list,
        vol.Optional(CONF_MODE_STATE_TEMPLATE): cv.template,
        vol.Optional(CONF_MODE_STATE_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_NAME): vol.Any(cv.string, None),
        vol.Optional(CONF_OPTIMISTIC, default=DEFAULT_OPTIMISTIC): cv.boolean,
        vol.Optional(CONF_PAYLOAD_ON, default="ON"): cv.string,
        vol.Optional(CONF_PAYLOAD_OFF, default="OFF"): cv.string,
        vol.Optional(CONF_POWER_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(CONF_POWER_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_PRECISION): vol.In(
            [PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE]
        ),
        vol.Optional(CONF_RETAIN, default=DEFAULT_RETAIN): cv.boolean,
        vol.Optional(CONF_ACTION_TEMPLATE): cv.template,
        vol.Optional(CONF_ACTION_TOPIC): valid_subscribe_topic,
        vol.Inclusive(
            CONF_PRESET_MODE_COMMAND_TOPIC, "preset_modes"
        ): valid_publish_topic,
        vol.Inclusive(
            CONF_PRESET_MODES_LIST, "preset_modes", default=[]
        ): cv.ensure_list,
        vol.Optional(CONF_PRESET_MODE_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_PRESET_MODE_STATE_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_PRESET_MODE_VALUE_TEMPLATE): cv.template,
        vol.Optional(CONF_SWING_MODE_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_SWING_MODE_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(
            CONF_SWING_MODE_LIST, default=[SWING_ON, SWING_OFF]
        ): cv.ensure_list,
        vol.Optional(CONF_SWING_MODE_STATE_TEMPLATE): cv.template,
        vol.Optional(CONF_SWING_MODE_STATE_TOPIC): valid_subscribe_topic,
        vol.Optional(CONF_TEMP_INITIAL): vol.All(vol.Coerce(float)),
        vol.Optional(CONF_TEMP_MIN): vol.Coerce(float),
        vol.Optional(CONF_TEMP_MAX): vol.Coerce(float),
        vol.Optional(CONF_TEMP_STEP, default=1.0): vol.Coerce(float),
        vol.Optional(CONF_TEMP_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_TEMP_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(CONF_TEMP_HIGH_COMMAND_TEMPLATE): cv.template,
        vol.Optional(CONF_TEMP_HIGH_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(CONF_TEMP
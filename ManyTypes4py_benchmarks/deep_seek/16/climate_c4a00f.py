"""Support for MQTT climate devices."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
import logging
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
import voluptuous as vol
from homeassistant.components import climate
from homeassistant.components.climate import ATTR_HVAC_MODE, ATTR_TARGET_TEMP_HIGH, ATTR_TARGET_TEMP_LOW, DEFAULT_MAX_HUMIDITY, DEFAULT_MIN_HUMIDITY, FAN_AUTO, FAN_HIGH, FAN_LOW, FAN_MEDIUM, PRESET_NONE, SWING_OFF, SWING_ON, ClimateEntity, ClimateEntityFeature, HVACAction, HVACMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_TEMPERATURE, CONF_NAME, CONF_OPTIMISTIC, CONF_PAYLOAD_OFF, CONF_PAYLOAD_ON, CONF_TEMPERATURE_UNIT, CONF_VALUE_TEMPLATE, PRECISION_HALVES, PRECISION_TENTHS, PRECISION_WHOLE, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.service_info.mqtt import ReceivePayloadType
from homeassistant.helpers.template import Template
from homeassistant.helpers.typing import ConfigType, VolSchemaType
from homeassistant.util.unit_conversion import TemperatureConverter
from . import subscription
from .config import DEFAULT_RETAIN, MQTT_BASE_SCHEMA
from .const import CONF_ACTION_TEMPLATE, CONF_ACTION_TOPIC, CONF_CURRENT_HUMIDITY_TEMPLATE, CONF_CURRENT_HUMIDITY_TOPIC, CONF_CURRENT_TEMP_TEMPLATE, CONF_CURRENT_TEMP_TOPIC, CONF_MODE_COMMAND_TEMPLATE, CONF_MODE_COMMAND_TOPIC, CONF_MODE_LIST, CONF_MODE_STATE_TEMPLATE, CONF_MODE_STATE_TOPIC, CONF_POWER_COMMAND_TEMPLATE, CONF_POWER_COMMAND_TOPIC, CONF_PRECISION, CONF_RETAIN, CONF_TEMP_COMMAND_TEMPLATE, CONF_TEMP_COMMAND_TOPIC, CONF_TEMP_INITIAL, CONF_TEMP_MAX, CONF_TEMP_MIN, CONF_TEMP_STATE_TEMPLATE, CONF_TEMP_STATE_TOPIC, DEFAULT_OPTIMISTIC, PAYLOAD_NONE
from .entity import MqttEntity, async_setup_entity_entry_helper
from .models import MqttCommandTemplate, MqttValueTemplate, PublishPayloadType, ReceiveMessage
from .schemas import MQTT_ENTITY_COMMON_SCHEMA
from .util import valid_publish_topic, valid_subscribe_topic

_LOGGER: logging.Logger = logging.getLogger(__name__)
PARALLEL_UPDATES: int = 0
DEFAULT_NAME: str = 'MQTT HVAC'
CONF_FAN_MODE_COMMAND_TEMPLATE: str = 'fan_mode_command_template'
CONF_FAN_MODE_COMMAND_TOPIC: str = 'fan_mode_command_topic'
CONF_FAN_MODE_LIST: str = 'fan_modes'
CONF_FAN_MODE_STATE_TEMPLATE: str = 'fan_mode_state_template'
CONF_FAN_MODE_STATE_TOPIC: str = 'fan_mode_state_topic'
CONF_HUMIDITY_COMMAND_TEMPLATE: str = 'target_humidity_command_template'
CONF_HUMIDITY_COMMAND_TOPIC: str = 'target_humidity_command_topic'
CONF_HUMIDITY_STATE_TEMPLATE: str = 'target_humidity_state_template'
CONF_HUMIDITY_STATE_TOPIC: str = 'target_humidity_state_topic'
CONF_HUMIDITY_MAX: str = 'max_humidity'
CONF_HUMIDITY_MIN: str = 'min_humidity'
CONF_PRESET_MODE_STATE_TOPIC: str = 'preset_mode_state_topic'
CONF_PRESET_MODE_COMMAND_TOPIC: str = 'preset_mode_command_topic'
CONF_PRESET_MODE_VALUE_TEMPLATE: str = 'preset_mode_value_template'
CONF_PRESET_MODE_COMMAND_TEMPLATE: str = 'preset_mode_command_template'
CONF_PRESET_MODES_LIST: str = 'preset_modes'
CONF_SWING_MODE_COMMAND_TEMPLATE: str = 'swing_mode_command_template'
CONF_SWING_MODE_COMMAND_TOPIC: str = 'swing_mode_command_topic'
CONF_SWING_MODE_LIST: str = 'swing_modes'
CONF_SWING_MODE_STATE_TEMPLATE: str = 'swing_mode_state_template'
CONF_SWING_MODE_STATE_TOPIC: str = 'swing_mode_state_topic'
CONF_TEMP_HIGH_COMMAND_TEMPLATE: str = 'temperature_high_command_template'
CONF_TEMP_HIGH_COMMAND_TOPIC: str = 'temperature_high_command_topic'
CONF_TEMP_HIGH_STATE_TEMPLATE: str = 'temperature_high_state_template'
CONF_TEMP_HIGH_STATE_TOPIC: str = 'temperature_high_state_topic'
CONF_TEMP_LOW_COMMAND_TEMPLATE: str = 'temperature_low_command_template'
CONF_TEMP_LOW_COMMAND_TOPIC: str = 'temperature_low_command_topic'
CONF_TEMP_LOW_STATE_TEMPLATE: str = 'temperature_low_state_template'
CONF_TEMP_LOW_STATE_TOPIC: str = 'temperature_low_state_topic'
CONF_TEMP_STEP: str = 'temp_step'
DEFAULT_INITIAL_TEMPERATURE: float = 21.0
MQTT_CLIMATE_ATTRIBUTES_BLOCKED: FrozenSet[str] = frozenset({climate.ATTR_CURRENT_HUMIDITY, climate.ATTR_CURRENT_TEMPERATURE, climate.ATTR_FAN_MODE, climate.ATTR_FAN_MODES, climate.ATTR_HUMIDITY, climate.ATTR_HVAC_ACTION, climate.ATTR_HVAC_MODES, climate.ATTR_MAX_HUMIDITY, climate.ATTR_MAX_TEMP, climate.ATTR_MIN_HUMIDITY, climate.ATTR_MIN_TEMP, climate.ATTR_PRESET_MODE, climate.ATTR_PRESET_MODES, climate.ATTR_SWING_MODE, climate.ATTR_SWING_MODES, climate.ATTR_TARGET_TEMP_HIGH, climate.ATTR_TARGET_TEMP_LOW, climate.ATTR_TARGET_TEMP_STEP, climate.ATTR_TEMPERATURE})
VALUE_TEMPLATE_KEYS: Tuple[str, ...] = (CONF_CURRENT_HUMIDITY_TEMPLATE, CONF_CURRENT_TEMP_TEMPLATE, CONF_FAN_MODE_STATE_TEMPLATE, CONF_HUMIDITY_STATE_TEMPLATE, CONF_MODE_STATE_TEMPLATE, CONF_ACTION_TEMPLATE, CONF_PRESET_MODE_VALUE_TEMPLATE, CONF_SWING_MODE_STATE_TEMPLATE, CONF_TEMP_HIGH_STATE_TEMPLATE, CONF_TEMP_LOW_STATE_TEMPLATE, CONF_TEMP_STATE_TEMPLATE)
COMMAND_TEMPLATE_KEYS: Set[str] = {CONF_FAN_MODE_COMMAND_TEMPLATE, CONF_HUMIDITY_COMMAND_TEMPLATE, CONF_MODE_COMMAND_TEMPLATE, CONF_POWER_COMMAND_TEMPLATE, CONF_PRESET_MODE_COMMAND_TEMPLATE, CONF_SWING_MODE_COMMAND_TEMPLATE, CONF_TEMP_COMMAND_TEMPLATE, CONF_TEMP_HIGH_COMMAND_TEMPLATE, CONF_TEMP_LOW_COMMAND_TEMPLATE}
TOPIC_KEYS: Tuple[str, ...] = (CONF_ACTION_TOPIC, CONF_CURRENT_HUMIDITY_TOPIC, CONF_CURRENT_TEMP_TOPIC, CONF_FAN_MODE_COMMAND_TOPIC, CONF_FAN_MODE_STATE_TOPIC, CONF_HUMIDITY_COMMAND_TOPIC, CONF_HUMIDITY_STATE_TOPIC, CONF_MODE_COMMAND_TOPIC, CONF_MODE_STATE_TOPIC, CONF_POWER_COMMAND_TOPIC, CONF_PRESET_MODE_COMMAND_TOPIC, CONF_PRESET_MODE_STATE_TOPIC, CONF_SWING_MODE_COMMAND_TOPIC, CONF_SWING_MODE_STATE_TOPIC, CONF_TEMP_COMMAND_TOPIC, CONF_TEMP_HIGH_COMMAND_TOPIC, CONF_TEMP_HIGH_STATE_TOPIC, CONF_TEMP_LOW_COMMAND_TOPIC, CONF_TEMP_LOW_STATE_TOPIC, CONF_TEMP_STATE_TOPIC)

def valid_preset_mode_configuration(config: ConfigType) -> ConfigType:
    """Validate that the preset mode reset payload is not one of the preset modes."""
    if PRESET_NONE in config[CONF_PRESET_MODES_LIST]:
        raise vol.Invalid("preset_modes must not include preset mode 'none'")
    return config

def valid_humidity_range_configuration(config: ConfigType) -> ConfigType:
    """Validate a target_humidity range configuration, throws otherwise."""
    if config[CONF_HUMIDITY_MIN] >= config[CONF_HUMIDITY_MAX]:
        raise vol.Invalid('target_humidity_max must be > target_humidity_min')
    if config[CONF_HUMIDITY_MAX] > 100:
        raise vol.Invalid('max_humidity must be <= 100')
    return config

def valid_humidity_state_configuration(config: ConfigType) -> ConfigType:
    """Validate humidity state.

    Ensure that if CONF_HUMIDITY_STATE_TOPIC is set then
    CONF_HUMIDITY_COMMAND_TOPIC is also set.
    """
    if CONF_HUMIDITY_STATE_TOPIC in config and CONF_HUMIDITY_COMMAND_TOPIC not in config:
        raise vol.Invalid(f'{CONF_HUMIDITY_STATE_TOPIC} cannot be used without {CONF_HUMIDITY_COMMAND_TOPIC}')
    return config

_PLATFORM_SCHEMA_BASE: VolSchemaType = MQTT_BASE_SCHEMA.extend({vol.Optional(CONF_CURRENT_HUMIDITY_TEMPLATE): cv.template, vol.Optional(CONF_CURRENT_HUMIDITY_TOPIC): valid_subscribe_topic, vol.Optional(CONF_CURRENT_TEMP_TEMPLATE): cv.template, vol.Optional(CONF_CURRENT_TEMP_TOPIC): valid_subscribe_topic, vol.Optional(CONF_FAN_MODE_COMMAND_TEMPLATE): cv.template, vol.Optional(CONF_FAN_MODE_COMMAND_TOPIC): valid_publish_topic, vol.Optional(CONF_FAN_MODE_LIST, default=[FAN_AUTO, FAN_LOW, FAN_MEDIUM, FAN_HIGH]): cv.ensure_list, vol.Optional(CONF_FAN_MODE_STATE_TEMPLATE): cv.template, vol.Optional(CONF_FAN_MODE_STATE_TOPIC): valid_subscribe_topic, vol.Optional(CONF_HUMIDITY_COMMAND_TEMPLATE): cv.template, vol.Optional(CONF_HUMIDITY_COMMAND_TOPIC): valid_publish_topic, vol.Optional(CONF_HUMIDITY_MIN, default=DEFAULT_MIN_HUMIDITY): cv.positive_float, vol.Optional(CONF_HUMIDITY_MAX, default=DEFAULT_MAX_HUMIDITY): cv.positive_float, vol.Optional(CONF_HUMIDITY_STATE_TEMPLATE): cv.template, vol.Optional(CONF_HUMIDITY_STATE_TOPIC): valid_subscribe_topic, vol.Optional(CONF_MODE_COMMAND_TEMPLATE): cv.template, vol.Optional(CONF_MODE_COMMAND_TOPIC): valid_publish_topic, vol.Optional(CONF_MODE_LIST, default=[HVACMode.AUTO, HVACMode.OFF, HVACMode.COOL, HVACMode.HEAT, HVACMode.DRY, HVACMode.FAN_ONLY]): cv.ensure_list, vol.Optional(CONF_MODE_STATE_TEMPLATE): cv.template, vol.Optional(CONF_MODE_STATE_TOPIC): valid_subscribe_topic, vol.Optional(CONF_NAME): vol.Any(cv.string, None), vol.Optional(CONF_OPTIMISTIC, default=DEFAULT_OPTIMISTIC): cv.boolean, vol.Optional(CONF_PAYLOAD_ON, default='ON'): cv.string, vol.Optional(CONF_PAYLOAD_OFF, default='OFF'): cv.string, vol.Optional(CONF_POWER_COMMAND_TOPIC): valid_publish_topic, vol.Optional(CONF_POWER_COMMAND_TEMPLATE): cv.template, vol.Optional(CONF_PRECISION): vol.In([PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE]), vol.Optional(CONF_RETAIN, default=DEFAULT_RETAIN): cv.boolean, vol.Optional(CONF_ACTION_TEMPLATE): cv.template, vol.Optional(CONF_ACTION_TOPIC): valid_subscribe_topic, vol.Inclusive(CONF_PRESET_MODE_COMMAND_TOPIC, 'preset_modes'): valid_publish_topic, vol.Inclusive(CONF_PRESET_MODES_LIST, 'preset_modes', default=[]): cv.ensure_list, vol.Optional(CONF_PRESET_MODE_COMMAND_TEMPLATE): cv.template, vol.Optional(CONF_PRESET_MODE_STATE_TOPIC): valid_subscribe_topic, vol.Optional(CONF_PRESET_MODE_VALUE_TEMPLATE): cv.template, vol.Optional(CONF_SWING_MODE_COMMAND_TEMPLATE): cv.template, vol.Optional(CONF_SWING_MODE_COMMAND_TOPIC): valid_publish_topic, vol.Optional(CONF_SWING_MODE_LIST, default=[SWING_ON, SWING_OFF]): cv.ensure_list, vol.Optional(CONF_SWING_MODE_STATE_TEMPLATE): cv.template, vol.Optional(CONF_SWING_MODE_STATE_TOPIC): valid_subscribe_topic, vol.Optional(CONF_TEMP_INITIAL): vol.All(vol.Coerce(float)), vol.Optional(CONF_TEMP_MIN): vol.Coerce(float), vol.Optional(CONF_TEMP_MAX): vol.Coerce(float), vol.Optional(CONF_TEMP_STEP, default=1.0): vol.Coerce(float), vol.Optional(CONF_TEMP_COMMAND_TEMPLATE): cv.template, vol.Optional(CONF_TEMP_COMMAND_TOPIC): valid_publish_topic, vol.Optional(CONF_TEMP_HIGH_COMMAND_TEMPLATE): cv.template, vol.Optional(CONF_TEMP_HIGH_COMMAND_TOPIC): valid_publish_topic, vol.Optional(CONF_TEMP_HIGH_STATE_TOPIC): valid_subscribe_topic, vol.Optional(CONF_TEMP_HIGH_STATE_TEMPLATE): cv.template, vol.Optional(CONF_TEMP_LOW_COMMAND_TEMPLATE): cv.template, vol.Optional(CONF_TEMP_LOW_COMMAND_TOPIC): valid_publish_topic, vol.Optional(CONF_TEMP_LOW_STATE_TEMPLATE): cv.template, vol.Optional(CONF_TEMP_LOW_STATE_TOPIC): valid_subscribe_topic, vol.Optional(CONF_TEMP_STATE_TEMPLATE): cv.template, vol.Optional(CONF_TEMP_STATE_TOPIC): valid_subscribe_topic, vol.Optional(CONF_TEMPERATURE_UNIT): cv.temperature_unit, vol.Optional(CONF_VALUE_TEMPLATE): cv.template}).extend(MQTT_ENTITY_COMMON_SCHEMA.schema)

PLATFORM_SCHEMA_MODERN: VolSchemaType = vol.All(_PLATFORM_SCHEMA_BASE, valid_preset_mode_configuration, valid_humidity_range_configuration, valid_humidity_state_configuration)
_DISCOVERY_SCHEMA_BASE: VolSchemaType = _PLATFORM_SCHEMA_BASE.extend({}, extra=vol.REMOVE_EXTRA)
DISCOVERY_SCHEMA: VolSchemaType = vol.All(_DISCOVERY_SCHEMA_BASE, valid_preset_mode_configuration, valid_humidity_range_configuration, valid_humidity_state_configuration)

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up MQTT climate through YAML and through MQTT discovery."""
    await async_setup_entity_entry_helper(hass, config_entry, MqttClimate, climate.DOMAIN, async_add_entities, DISCOVERY_SCHEMA, PLATFORM_SCHEMA_MODERN)

class MqttTemperatureControlEntity(MqttEntity, ABC):
    """Helper entity class to control temperature.

    MqttTemperatureControlEntity supports shared methods for
    climate and water_heater platforms.
    """
    _feature_preset_mode: bool = False

    def render_template(self, msg: ReceiveMessage, template_name: str) -> Any:
        """Render a template by name."""
        template = self._value_templates[template_name]
        return template(msg.payload)

    @callback
    def handle_climate_attribute_received(self, template_name: str, attr: str, msg: ReceiveMessage) -> None:
        """Handle climate attributes coming via MQTT."""
        payload = self.render_template(msg, template_name)
        if not payload:
            _LOGGER.debug('Invalid empty payload for attribute %s, ignoring update', attr)
            return
        if payload == PAYLOAD_NONE:
            setattr(self, attr, None)
            return
        try:
            setattr(self, attr, float(payload))
        except ValueError:
            _LOGGER.error('Could not parse %s from %s', template_name, payload
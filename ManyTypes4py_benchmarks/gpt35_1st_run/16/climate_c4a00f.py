from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
import logging
from typing import Any
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

_LOGGER: logging.Logger
PARALLEL_UPDATES: int = 0
DEFAULT_NAME: str = 'MQTT HVAC'
CONF_FAN_MODE_COMMAND_TEMPLATE: str = 'fan_mode_command_template'
CONF_FAN_MODE_COMMAND_TOPIC: str = 'fan_mode_command_topic'
CONF_FAN_MODE_LIST: list[str]
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
CONF_PRESET_MODES_LIST: list[str]
CONF_SWING_MODE_COMMAND_TEMPLATE: str = 'swing_mode_command_template'
CONF_SWING_MODE_COMMAND_TOPIC: str = 'swing_mode_command_topic'
CONF_SWING_MODE_LIST: list[str]
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
MQTT_CLIMATE_ATTRIBUTES_BLOCKED: frozenset[str]
VALUE_TEMPLATE_KEYS: tuple[str]

def valid_preset_mode_configuration(config: dict) -> dict:
    ...

def valid_humidity_range_configuration(config: dict) -> dict:
    ...

def valid_humidity_state_configuration(config: dict) -> dict:
    ...

_PLATFORM_SCHEMA_BASE: VolSchemaType
PLATFORM_SCHEMA_MODERN: VolSchemaType
_DISCOVERY_SCHEMA_BASE: VolSchemaType
DISCOVERY_SCHEMA: VolSchemaType

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class MqttTemperatureControlEntity(MqttEntity, ABC):
    ...

class MqttClimate(MqttTemperatureControlEntity, ClimateEntity):
    ...


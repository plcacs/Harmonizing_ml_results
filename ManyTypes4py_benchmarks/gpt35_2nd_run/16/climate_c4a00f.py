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
PARALLEL_UPDATES: int
DEFAULT_NAME: str
CONF_FAN_MODE_COMMAND_TEMPLATE: str
CONF_FAN_MODE_COMMAND_TOPIC: str
CONF_FAN_MODE_LIST: list[str]
CONF_FAN_MODE_STATE_TEMPLATE: str
CONF_FAN_MODE_STATE_TOPIC: str
CONF_HUMIDITY_COMMAND_TEMPLATE: str
CONF_HUMIDITY_COMMAND_TOPIC: str
CONF_HUMIDITY_STATE_TEMPLATE: str
CONF_HUMIDITY_STATE_TOPIC: str
CONF_HUMIDITY_MAX: str
CONF_HUMIDITY_MIN: str
CONF_PRESET_MODE_STATE_TOPIC: str
CONF_PRESET_MODE_COMMAND_TOPIC: str
CONF_PRESET_MODE_VALUE_TEMPLATE: str
CONF_PRESET_MODE_COMMAND_TEMPLATE: str
CONF_PRESET_MODES_LIST: list[str]
CONF_SWING_MODE_COMMAND_TEMPLATE: str
CONF_SWING_MODE_COMMAND_TOPIC: str
CONF_SWING_MODE_LIST: list[str]
CONF_SWING_MODE_STATE_TEMPLATE: str
CONF_SWING_MODE_STATE_TOPIC: str
CONF_TEMP_HIGH_COMMAND_TEMPLATE: str
CONF_TEMP_HIGH_COMMAND_TOPIC: str
CONF_TEMP_HIGH_STATE_TEMPLATE: str
CONF_TEMP_HIGH_STATE_TOPIC: str
CONF_TEMP_LOW_COMMAND_TEMPLATE: str
CONF_TEMP_LOW_COMMAND_TOPIC: str
CONF_TEMP_LOW_STATE_TEMPLATE: str
CONF_TEMP_LOW_STATE_TOPIC: str
CONF_TEMP_STEP: str
DEFAULT_INITIAL_TEMPERATURE: float
MQTT_CLIMATE_ATTRIBUTES_BLOCKED: frozenset[str]
VALUE_TEMPLATE_KEYS: tuple[str, ...]
COMMAND_TEMPLATE_KEYS: set[str]
TOPIC_KEYS: tuple[str, ...]

def valid_preset_mode_configuration(config: dict) -> dict:
    ...

def valid_humidity_range_configuration(config: dict) -> dict:
    ...

def valid_humidity_state_configuration(config: dict) -> dict:
    ...

_PLATFORM_SCHEMA_BASE: vol.Schema
PLATFORM_SCHEMA_MODERN: vol.Schema
_DISCOVERY_SCHEMA_BASE: vol.Schema
DISCOVERY_SCHEMA: vol.Schema

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class MqttTemperatureControlEntity(MqttEntity, ABC):
    ...

    def render_template(self, msg: Any, template_name: str) -> Any:
        ...

    @callback
    def handle_climate_attribute_received(self, template_name: str, attr: str, msg: Any) -> None:
        ...

    @callback
    def prepare_subscribe_topics(self) -> None:
        ...

    async def _subscribe_topics(self) -> None:
        ...

    async def _publish(self, topic: str, payload: Any) -> None:
        ...

    async def _set_climate_attribute(self, temp: Any, cmnd_topic: str, cmnd_template: str, state_topic: str, attr: str) -> bool:
        ...

    @abstractmethod
    async def async_set_temperature(self, **kwargs: Any) -> None:
        ...

class MqttClimate(MqttTemperatureControlEntity, ClimateEntity):
    ...

    @staticmethod
    def config_schema() -> vol.Schema:
        ...

    def _setup_from_config(self, config: dict) -> None:
        ...

    @callback
    def _handle_action_received(self, msg: Any) -> None:
        ...

    @callback
    def _handle_mode_received(self, template_name: str, attr: str, mode_list: list[str], msg: Any) -> None:
        ...

    @callback
    def _handle_preset_mode_received(self, msg: Any) -> None:
        ...

    @callback
    def _prepare_subscribe_topics(self) -> None:
        ...

    async def async_set_temperature(self, **kwargs: Any) -> None:
        ...

    async def async_set_humidity(self, humidity: Any) -> None:
        ...

    async def async_set_swing_mode(self, swing_mode: str) -> None:
        ...

    async def async_set_fan_mode(self, fan_mode: str) -> None:
        ...

    async def async_set_hvac_mode(self, hvac_mode: str) -> None:
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        ...

    async def async_turn_on(self) -> None:
        ...

    async def async_turn_off(self) -> None:
        ...

"""Control which entities are exposed to voice assistants."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import CLOUD_NEVER_EXPOSED_ENTITIES
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.storage import Store

__all__ = [
    'AssistantPreferences',
    'ExposedEntity',
    'SerializedExposedEntities',
    'ExposedEntities',
    'async_listen_entity_updates',
    'async_get_assistant_settings',
    'async_get_entity_settings',
    'async_expose_entity',
    'async_should_expose',
    'async_set_assistant_option',
    'ws_expose_entity',
    'ws_expose_new_entities_get',
    'ws_expose_new_entities_set',
    'ws_list_exposed_entities',
]

DEFAULT_EXPOSED_DOMAINS: FrozenSet[str] = frozenset({'climate', 'cover', 'fan', 'humidifier', 'light', 'media_player', 'scene', 'switch', 'todo', 'vacuum', 'water_heater'})
DEFAULT_EXPOSED_BINARY_SENSOR_DEVICE_CLASSES: FrozenSet[BinarySensorDeviceClass] = frozenset({BinarySensorDeviceClass.DOOR, BinarySensorDeviceClass.GARAGE_DOOR, BinarySensorDeviceClass.LOCK, BinarySensorDeviceClass.MOTION, BinarySensorDeviceClass.OPENING, BinarySensorDeviceClass.PRESENCE, BinarySensorDeviceClass.WINDOW})
DEFAULT_EXPOSED_SENSOR_DEVICE_CLASSES: FrozenSet[SensorDeviceClass] = frozenset({SensorDeviceClass.AQI, SensorDeviceClass.CO, SensorDeviceClass.CO2, SensorDeviceClass.HUMIDITY, SensorDeviceClass.PM10, SensorDeviceClass.PM25, SensorDeviceClass.TEMPERATURE, SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS})
DEFAULT_EXPOSED_ASSISTANT: Dict[str, bool] = {'conversation': True}
KNOWN_ASSISTANTS: Tuple[str, ...] = ('cloud.alexa', 'cloud.google_assistant', 'conversation')

@dataclass(frozen=True)
class AssistantPreferences:
    expose_new: bool = ...

    def to_json(self) -> dict:
        ...

@dataclass(frozen=True)
class ExposedEntity:
    assistants: Dict[str, Dict[str, Any]] = ...

    def to_json(self) -> dict:
        ...

class SerializedExposedEntities(TypedDict):
    ...

class ExposedEntities:
    def __init__(self, hass: HomeAssistant) -> None:
        ...

    async def async_initialize(self) -> None:
        ...

    @callback
    def async_listen_entity_updates(self, assistant: str, listener: CALLBACK_TYPE) -> Callable[[], None]:
        ...

    @callback
    def async_set_assistant_option(self, assistant: str, entity_id: str, key: str, value: Any) -> None:
        ...

    def _async_set_legacy_assistant_option(self, assistant: str, entity_id: str, key: str, value: Any) -> None:
        ...

    @callback
    def async_get_expose_new_entities(self, assistant: str) -> bool:
        ...

    @callback
    def async_set_expose_new_entities(self, assistant: str, expose_new: bool) -> None:
        ...

    @callback
    def async_get_assistant_settings(self, assistant: str) -> Dict[str, Dict[str, Any]]:
        ...

    @callback
    def async_get_entity_settings(self, entity_id: str) -> Dict[str, Dict[str, Any]]:
        ...

    @callback
    def async_should_expose(self, assistant: str, entity_id: str) -> bool:
        ...

    def _async_should_expose_legacy_entity(self, assistant: str, entity_id: str) -> bool:
        ...

    def _is_default_exposed(self, entity_id: str, registry_entry: Any) -> bool:
        ...

    def _update_exposed_entity(self, assistant: str, entity_id: str, key: str, value: Any) -> ExposedEntity:
        ...

    def _new_exposed_entity(self, assistant: str, key: str, value: Any) -> ExposedEntity:
        ...

    async def _async_load_data(self) -> Optional[Dict[str, Any]]:
        ...

    @callback
    def _async_schedule_save(self) -> None:
        ...

    @callback
    def _data_to_save(self) -> Dict[str, Any]:
        ...

@callback
@websocket_api.require_admin
@websocket_api.websocket_command({vol.Required('type'): 'homeassistant/expose_entity', vol.Required('assistants'): [vol.In(KNOWN_ASSISTANTS)], vol.Required('entity_ids'): [str], vol.Required('should_expose'): bool})
def ws_expose_entity(hass: HomeAssistant, connection: websocket_api.WebSocketConnection, msg: Dict[str, Any]) -> None:
    ...

@callback
@websocket_api.require_admin
@websocket_api.websocket_command({vol.Required('type'): 'homeassistant/expose_entity/list'})
def ws_list_exposed_entities(hass: HomeAssistant, connection: websocket_api.WebSocketConnection, msg: Dict[str, Any]) -> None:
    ...

@callback
@websocket_api.require_admin
@websocket_api.websocket_command({vol.Required('type'): 'homeassistant/expose_new_entities/get', vol.Required('assistant'): vol.In(KNOWN_ASSISTANTS)})
def ws_expose_new_entities_get(hass: HomeAssistant, connection: websocket_api.WebSocketConnection, msg: Dict[str, Any]) -> None:
    ...

@callback
@websocket_api.require_admin
@websocket_api.websocket_command({vol.Required('type'): 'homeassistant/expose_new_entities/set', vol.Required('assistant'): vol.In(KNOWN_ASSISTANTS), vol.Required('expose_new'): bool})
def ws_expose_new_entities_set(hass: HomeAssistant, connection: websocket_api.WebSocketConnection, msg: Dict[str, Any]) -> None:
    ...

@callback
def async_listen_entity_updates(hass: HomeAssistant, assistant: str, listener: CALLBACK_TYPE) -> Callable[[], None]:
    ...

@callback
def async_get_assistant_settings(hass: HomeAssistant, assistant: str) -> Dict[str, Dict[str, Any]]:
    ...

@callback
def async_get_entity_settings(hass: HomeAssistant, entity_id: str) -> Dict[str, Dict[str, Any]]:
    ...

@callback
def async_expose_entity(hass: HomeAssistant, assistant: str, entity_id: str, should_expose: bool) -> None:
    ...

@callback
def async_should_expose(hass: HomeAssistant, assistant: str, entity_id: str) -> bool:
    ...

@callback
def async_set_assistant_option(hass: HomeAssistant, assistant: str, entity_id: str, option: str, value: Any) -> None:
    ...
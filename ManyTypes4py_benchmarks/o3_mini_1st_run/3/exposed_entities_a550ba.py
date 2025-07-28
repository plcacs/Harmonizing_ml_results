from __future__ import annotations
from collections.abc import Callable, Mapping
import dataclasses
from itertools import chain
from typing import Any, Awaitable, Callable as TypingCallable, Dict, List, Optional
import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import CLOUD_NEVER_EXPOSED_ENTITIES
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback, split_entity_id
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity import get_device_class
from homeassistant.helpers.storage import Store
from homeassistant.util.read_only_dict import ReadOnlyDict
from .const import DATA_EXPOSED_ENTITIES, DOMAIN

KNOWN_ASSISTANTS = ('cloud.alexa', 'cloud.google_assistant', 'conversation')
STORAGE_KEY = f'{DOMAIN}.exposed_entities'
STORAGE_VERSION = 1
SAVE_DELAY = 10
DEFAULT_EXPOSED_DOMAINS = {'climate', 'cover', 'fan', 'humidifier', 'light', 'media_player', 'scene', 'switch', 'todo', 'vacuum', 'water_heater'}
DEFAULT_EXPOSED_BINARY_SENSOR_DEVICE_CLASSES = {BinarySensorDeviceClass.DOOR, BinarySensorDeviceClass.GARAGE_DOOR, BinarySensorDeviceClass.LOCK, BinarySensorDeviceClass.MOTION, BinarySensorDeviceClass.OPENING, BinarySensorDeviceClass.PRESENCE, BinarySensorDeviceClass.WINDOW}
DEFAULT_EXPOSED_SENSOR_DEVICE_CLASSES = {SensorDeviceClass.AQI, SensorDeviceClass.CO, SensorDeviceClass.CO2, SensorDeviceClass.HUMIDITY, SensorDeviceClass.PM10, SensorDeviceClass.PM25, SensorDeviceClass.TEMPERATURE, SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS}
DEFAULT_EXPOSED_ASSISTANT: Dict[str, bool] = {'conversation': True}


@dataclasses.dataclass(frozen=True)
class AssistantPreferences:
    expose_new: bool

    def to_json(self) -> Dict[str, Any]:
        return {'expose_new': self.expose_new}


@dataclasses.dataclass(frozen=True)
class ExposedEntity:
    assistants: Dict[str, Dict[str, Any]]

    def to_json(self) -> Dict[str, Any]:
        return {'assistants': self.assistants}


class SerializedExposedEntities(TypedDict, total=False):
    # This is a placeholder for the SerializedExposedEntities TypedDict details.
    pass


class ExposedEntities:
    def __init__(self, hass: HomeAssistant) -> None:
        self._hass: HomeAssistant = hass
        self._listeners: Dict[str, List[CALLBACK_TYPE]] = {}
        self._store: Store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._assistants: Dict[str, AssistantPreferences] = {}
        self.entities: Dict[str, ExposedEntity] = {}

    async def async_initialize(self) -> None:
        websocket_api.async_register_command(self._hass, ws_expose_entity)
        websocket_api.async_register_command(self._hass, ws_expose_new_entities_get)
        websocket_api.async_register_command(self._hass, ws_expose_new_entities_set)
        websocket_api.async_register_command(self._hass, ws_list_exposed_entities)
        await self._async_load_data()

    @callback
    def async_listen_entity_updates(self, assistant: str, listener: CALLBACK_TYPE) -> CALLBACK_TYPE:
        def unsubscribe() -> None:
            self._listeners[assistant].remove(listener)
        self._listeners.setdefault(assistant, []).append(listener)
        return unsubscribe

    @callback
    def async_set_assistant_option(self, assistant: str, entity_id: str, key: str, value: Any) -> None:
        entity_registry = er.async_get(self._hass)
        registry_entry = entity_registry.async_get(entity_id)
        if not registry_entry:
            self._async_set_legacy_assistant_option(assistant, entity_id, key, value)
            return
        assistant_options = registry_entry.options.get(assistant, {})
        if assistant_options and assistant_options.get(key) == value:
            return
        assistant_options = {**assistant_options, key: value}  # type: Dict[str, Any]
        entity_registry.async_update_entity_options(entity_id, assistant, assistant_options)
        for listener in self._listeners.get(assistant, []):
            listener()

    @callback
    def _async_set_legacy_assistant_option(self, assistant: str, entity_id: str, key: str, value: Any) -> None:
        exposed_entity: Optional[ExposedEntity] = self.entities.get(entity_id)
        if exposed_entity:
            assistant_options = exposed_entity.assistants.get(assistant, {})
            if assistant_options and assistant_options.get(key) == value:
                return
        if exposed_entity:
            new_exposed_entity = self._update_exposed_entity(assistant, entity_id, key, value)
        else:
            new_exposed_entity = self._new_exposed_entity(assistant, key, value)
        self.entities[entity_id] = new_exposed_entity
        self._async_schedule_save()
        for listener in self._listeners.get(assistant, []):
            listener()

    @callback
    def async_get_expose_new_entities(self, assistant: str) -> bool:
        prefs: Optional[AssistantPreferences] = self._assistants.get(assistant)
        if prefs:
            return prefs.expose_new
        return DEFAULT_EXPOSED_ASSISTANT.get(assistant, False)

    @callback
    def async_set_expose_new_entities(self, assistant: str, expose_new: bool) -> None:
        self._assistants[assistant] = AssistantPreferences(expose_new=expose_new)
        self._async_schedule_save()

    @callback
    def async_get_assistant_settings(self, assistant: str) -> Dict[str, Dict[str, Any]]:
        entity_registry = er.async_get(self._hass)
        result: Dict[str, Dict[str, Any]] = {}
        for entity_id, exposed_entity in self.entities.items():
            if (options := exposed_entity.assistants.get(assistant)):
                result[entity_id] = options
        for entity_id, entry in entity_registry.entities.items():
            if (options := entry.options.get(assistant)):
                result[entity_id] = options
        return result

    @callback
    def async_get_entity_settings(self, entity_id: str) -> Dict[str, Dict[str, Any]]:
        entity_registry = er.async_get(self._hass)
        result: Dict[str, Dict[str, Any]] = {}
        registry_entry = entity_registry.async_get(entity_id)
        if registry_entry:
            assistant_settings: Dict[str, Dict[str, Any]] = registry_entry.options
        elif (exposed_entity := self.entities.get(entity_id)):
            assistant_settings = exposed_entity.assistants
        else:
            raise HomeAssistantError('Unknown entity')
        for assistant in KNOWN_ASSISTANTS:
            if (options := assistant_settings.get(assistant)):
                result[assistant] = options
        return result

    @callback
    def async_should_expose(self, assistant: str, entity_id: str) -> bool:
        if entity_id in CLOUD_NEVER_EXPOSED_ENTITIES:
            return False
        entity_registry = er.async_get(self._hass)
        registry_entry = entity_registry.async_get(entity_id)
        if not registry_entry:
            return self._async_should_expose_legacy_entity(assistant, entity_id)
        if assistant in registry_entry.options:
            if 'should_expose' in registry_entry.options[assistant]:
                should_expose: bool = registry_entry.options[assistant]['should_expose']
                return should_expose
        if self.async_get_expose_new_entities(assistant):
            should_expose = self._is_default_exposed(entity_id, registry_entry)
        else:
            should_expose = False
        assistant_options = registry_entry.options.get(assistant, {})
        assistant_options = {**assistant_options, 'should_expose': should_expose}  # type: Dict[str, Any]
        entity_registry.async_update_entity_options(entity_id, assistant, assistant_options)
        return should_expose

    def _async_should_expose_legacy_entity(self, assistant: str, entity_id: str) -> bool:
        exposed_entity: Optional[ExposedEntity] = self.entities.get(entity_id)
        if exposed_entity and assistant in exposed_entity.assistants:
            if 'should_expose' in exposed_entity.assistants[assistant]:
                should_expose: bool = exposed_entity.assistants[assistant]['should_expose']
                return should_expose
        if self.async_get_expose_new_entities(assistant):
            should_expose = self._is_default_exposed(entity_id, None)
        else:
            should_expose = False
        if exposed_entity:
            new_exposed_entity = self._update_exposed_entity(assistant, entity_id, 'should_expose', should_expose)
        else:
            new_exposed_entity = self._new_exposed_entity(assistant, 'should_expose', should_expose)
        self.entities[entity_id] = new_exposed_entity
        self._async_schedule_save()
        return should_expose

    def _is_default_exposed(self, entity_id: str, registry_entry: Optional[Any]) -> bool:
        if registry_entry and (registry_entry.entity_category is not None or registry_entry.hidden_by is not None):
            return False
        domain: str = split_entity_id(entity_id)[0]
        if domain in DEFAULT_EXPOSED_DOMAINS:
            return True
        try:
            device_class = get_device_class(self._hass, entity_id)
        except HomeAssistantError:
            return False
        if domain == 'binary_sensor' and device_class in DEFAULT_EXPOSED_BINARY_SENSOR_DEVICE_CLASSES:
            return True
        if domain == 'sensor' and device_class in DEFAULT_EXPOSED_SENSOR_DEVICE_CLASSES:
            return True
        return False

    def _update_exposed_entity(self, assistant: str, entity_id: str, key: str, value: Any) -> ExposedEntity:
        entity = self.entities[entity_id]
        assistants = dict(entity.assistants)
        old_settings = assistants.get(assistant, {})
        assistants[assistant] = {**old_settings, key: value}
        return ExposedEntity(assistants=assistants)

    def _new_exposed_entity(self, assistant: str, key: str, value: Any) -> ExposedEntity:
        return ExposedEntity(assistants={assistant: {key: value}})

    async def _async_load_data(self) -> Optional[Dict[str, Any]]:
        data: Optional[Dict[str, Any]] = await self._store.async_load()
        assistants: Dict[str, AssistantPreferences] = {}
        exposed_entities: Dict[str, ExposedEntity] = {}
        if data:
            for domain, preferences in data.get('assistants', {}).items():
                assistants[domain] = AssistantPreferences(**preferences)
            for entity_id, preferences in data.get('exposed_entities', {}).items():
                exposed_entities[entity_id] = ExposedEntity(**preferences)
        self._assistants = assistants
        self.entities = exposed_entities
        return data

    @callback
    def _async_schedule_save(self) -> None:
        self._store.async_delay_save(self._data_to_save, SAVE_DELAY)

    @callback
    def _data_to_save(self) -> Dict[str, Any]:
        return {
            'assistants': {domain: preferences.to_json() for domain, preferences in self._assistants.items()},
            'exposed_entities': {entity_id: entity.to_json() for entity_id, entity in self.entities.items()},
        }


@callback
@websocket_api.require_admin
@websocket_api.websocket_command({
    vol.Required('type'): 'homeassistant/expose_entity',
    vol.Required('assistants'): [vol.In(KNOWN_ASSISTANTS)],
    vol.Required('entity_ids'): [str],
    vol.Required('should_expose'): bool
})
def ws_expose_entity(hass: HomeAssistant, connection: Any, msg: Dict[str, Any]) -> None:
    entity_ids: List[str] = msg['entity_ids']
    blocked: Optional[str] = next((entity_id for entity_id in entity_ids if entity_id in CLOUD_NEVER_EXPOSED_ENTITIES), None)
    if blocked:
        connection.send_error(msg['id'], websocket_api.ERR_NOT_ALLOWED, f"can't expose '{blocked}'")
        return
    for entity_id in entity_ids:
        for assistant in msg['assistants']:
            async_expose_entity(hass, assistant, entity_id, msg['should_expose'])
    connection.send_result(msg['id'])


@callback
@websocket_api.require_admin
@websocket_api.websocket_command({
    vol.Required('type'): 'homeassistant/expose_entity/list'
})
def ws_list_exposed_entities(hass: HomeAssistant, connection: Any, msg: Dict[str, Any]) -> None:
    result: Dict[str, Dict[str, bool]] = {}
    exposed_entities: ExposedEntities = hass.data[DATA_EXPOSED_ENTITIES]
    entity_registry = er.async_get(hass)
    for entity_id in chain(exposed_entities.entities, entity_registry.entities):
        result[entity_id] = {}
        entity_settings = async_get_entity_settings(hass, entity_id)
        for assistant, settings in entity_settings.items():
            if 'should_expose' not in settings:
                continue
            result[entity_id][assistant] = settings['should_expose']
    connection.send_result(msg['id'], {'exposed_entities': result})


@callback
@websocket_api.require_admin
@websocket_api.websocket_command({
    vol.Required('type'): 'homeassistant/expose_new_entities/get',
    vol.Required('assistant'): vol.In(KNOWN_ASSISTANTS)
})
def ws_expose_new_entities_get(hass: HomeAssistant, connection: Any, msg: Dict[str, Any]) -> None:
    exposed_entities: ExposedEntities = hass.data[DATA_EXPOSED_ENTITIES]
    expose_new: bool = exposed_entities.async_get_expose_new_entities(msg['assistant'])
    connection.send_result(msg['id'], {'expose_new': expose_new})


@callback
@websocket_api.require_admin
@websocket_api.websocket_command({
    vol.Required('type'): 'homeassistant/expose_new_entities/set',
    vol.Required('assistant'): vol.In(KNOWN_ASSISTANTS),
    vol.Required('expose_new'): bool
})
def ws_expose_new_entities_set(hass: HomeAssistant, connection: Any, msg: Dict[str, Any]) -> None:
    exposed_entities: ExposedEntities = hass.data[DATA_EXPOSED_ENTITIES]
    exposed_entities.async_set_expose_new_entities(msg['assistant'], msg['expose_new'])
    connection.send_result(msg['id'])


@callback
def async_listen_entity_updates(hass: HomeAssistant, assistant: str, listener: CALLBACK_TYPE) -> CALLBACK_TYPE:
    exposed_entities: ExposedEntities = hass.data[DATA_EXPOSED_ENTITIES]
    return exposed_entities.async_listen_entity_updates(assistant, listener)


@callback
def async_get_assistant_settings(hass: HomeAssistant, assistant: str) -> Dict[str, Dict[str, Any]]:
    exposed_entities: ExposedEntities = hass.data[DATA_EXPOSED_ENTITIES]
    return exposed_entities.async_get_assistant_settings(assistant)


@callback
def async_get_entity_settings(hass: HomeAssistant, entity_id: str) -> Dict[str, Dict[str, Any]]:
    exposed_entities: ExposedEntities = hass.data[DATA_EXPOSED_ENTITIES]
    return exposed_entities.async_get_entity_settings(entity_id)


@callback
def async_expose_entity(hass: HomeAssistant, assistant: str, entity_id: str, should_expose: bool) -> None:
    async_set_assistant_option(hass, assistant, entity_id, 'should_expose', should_expose)


@callback
def async_should_expose(hass: HomeAssistant, assistant: str, entity_id: str) -> bool:
    exposed_entities: ExposedEntities = hass.data[DATA_EXPOSED_ENTITIES]
    return exposed_entities.async_should_expose(assistant, entity_id)


@callback
def async_set_assistant_option(hass: HomeAssistant, assistant: str, entity_id: str, option: str, value: Any) -> None:
    exposed_entities: ExposedEntities = hass.data[DATA_EXPOSED_ENTITIES]
    exposed_entities.async_set_assistant_option(assistant, entity_id, option, value)
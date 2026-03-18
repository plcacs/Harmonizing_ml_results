```pyi
from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any, TypedDict

from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util.read_only_dict import ReadOnlyDict

KNOWN_ASSISTANTS: tuple[str, str, str]
STORAGE_KEY: str
STORAGE_VERSION: int
SAVE_DELAY: int
DEFAULT_EXPOSED_DOMAINS: set[str]
DEFAULT_EXPOSED_BINARY_SENSOR_DEVICE_CLASSES: set[BinarySensorDeviceClass]
DEFAULT_EXPOSED_SENSOR_DEVICE_CLASSES: set[SensorDeviceClass]
DEFAULT_EXPOSED_ASSISTANT: dict[str, bool]

@dataclasses.dataclass(frozen=True)
class AssistantPreferences:
    expose_new: bool
    def to_json(self) -> dict[str, bool]: ...

@dataclasses.dataclass(frozen=True)
class ExposedEntity:
    assistants: dict[str, dict[str, Any]]
    def to_json(self) -> dict[str, dict[str, Any]]: ...

class SerializedExposedEntities(TypedDict):
    assistants: dict[str, dict[str, Any]]
    exposed_entities: dict[str, dict[str, Any]]

class ExposedEntities:
    _hass: HomeAssistant
    _listeners: dict[str, list[Callable[[], None]]]
    _store: Store
    _assistants: dict[str, AssistantPreferences]
    entities: dict[str, ExposedEntity]
    def __init__(self, hass: HomeAssistant) -> None: ...
    async def async_initialize(self) -> None: ...
    @callback
    def async_listen_entity_updates(
        self, assistant: str, listener: Callable[[], None]
    ) -> Callable[[], None]: ...
    @callback
    def async_set_assistant_option(
        self, assistant: str, entity_id: str, key: str, value: Any
    ) -> None: ...
    def _async_set_legacy_assistant_option(
        self, assistant: str, entity_id: str, key: str, value: Any
    ) -> None: ...
    @callback
    def async_get_expose_new_entities(self, assistant: str) -> bool: ...
    @callback
    def async_set_expose_new_entities(self, assistant: str, expose_new: bool) -> None: ...
    @callback
    def async_get_assistant_settings(self, assistant: str) -> dict[str, dict[str, Any]]: ...
    @callback
    def async_get_entity_settings(self, entity_id: str) -> dict[str, dict[str, Any]]: ...
    @callback
    def async_should_expose(self, assistant: str, entity_id: str) -> bool: ...
    def _async_should_expose_legacy_entity(self, assistant: str, entity_id: str) -> bool: ...
    def _is_default_exposed(self, entity_id: str, registry_entry: Any) -> bool: ...
    def _update_exposed_entity(
        self, assistant: str, entity_id: str, key: str, value: Any
    ) -> ExposedEntity: ...
    def _new_exposed_entity(
        self, assistant: str, key: str, value: Any
    ) -> ExposedEntity: ...
    async def _async_load_data(self) -> dict[str, Any] | None: ...
    @callback
    def _async_schedule_save(self) -> None: ...
    @callback
    def _data_to_save(self) -> dict[str, Any]: ...

@callback
def ws_expose_entity(hass: HomeAssistant, connection: Any, msg: dict[str, Any]) -> None: ...

@callback
def ws_list_exposed_entities(
    hass: HomeAssistant, connection: Any, msg: dict[str, Any]
) -> None: ...

@callback
def ws_expose_new_entities_get(
    hass: HomeAssistant, connection: Any, msg: dict[str, Any]
) -> None: ...

@callback
def ws_expose_new_entities_set(
    hass: HomeAssistant, connection: Any, msg: dict[str, Any]
) -> None: ...

@callback
def async_listen_entity_updates(
    hass: HomeAssistant, assistant: str, listener: Callable[[], None]
) -> Callable[[], None]: ...

@callback
def async_get_assistant_settings(hass: HomeAssistant, assistant: str) -> dict[str, dict[str, Any]]: ...

@callback
def async_get_entity_settings(hass: HomeAssistant, entity_id: str) -> dict[str, dict[str, Any]]: ...

@callback
def async_expose_entity(
    hass: HomeAssistant, assistant: str, entity_id: str, should_expose: bool
) -> None: ...

@callback
def async_should_expose(hass: HomeAssistant, assistant: str, entity_id: str) -> bool: ...

@callback
def async_set_assistant_option(
    hass: HomeAssistant, assistant: str, entity_id: str, option: str, value: Any
) -> None: ...
```
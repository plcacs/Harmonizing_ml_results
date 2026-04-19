from __future__ import annotations

import logging
from typing import Any, Callable, ClassVar, Optional, final

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.core import Context, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import collection, entity_registry as er
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.storage import Store
from homeassistant.helpers.typing import ConfigType
from homeassistant.util.hass_dict import HassKey

from .const import DEFAULT_NAME, DEVICE_ID, DOMAIN, EVENT_TAG_SCANNED, LOGGER, TAG_ID

LAST_SCANNED: str
LAST_SCANNED_BY_DEVICE_ID: str
STORAGE_KEY: str
STORAGE_VERSION: int
STORAGE_VERSION_MINOR: int
TAG_DATA: HassKey[TagStorageCollection]
CREATE_FIELDS: dict[Any, Any]
UPDATE_FIELDS: dict[Any, Any]
CONFIG_SCHEMA: vol.Schema

class TagIDExistsError(HomeAssistantError):
    def __init__(self, item_id: str) -> None: ...

class TagIDManager(collection.IDManager):
    def generate_id(self, suggestion: str) -> str: ...

def _create_entry(entity_registry: er.EntityRegistry, tag_id: str, name: Optional[str]) -> er.RegistryEntry: ...

class TagStore(Store[collection.SerializedStorageCollection]):
    async def _async_migrate_func(
        self,
        old_major_version: int,
        old_minor_version: int,
        old_data: collection.SerializedStorageCollection,
    ) -> collection.SerializedStorageCollection: ...

class TagStorageCollection(collection.DictStorageCollection):
    CREATE_SCHEMA: ClassVar[vol.Schema]
    UPDATE_SCHEMA: ClassVar[vol.Schema]

    def __init__(
        self,
        store: Store[collection.SerializedStorageCollection],
        id_manager: Optional[collection.IDManager] = ...,
    ) -> None: ...
    async def _process_create_data(self, data: dict[str, Any]) -> dict[str, Any]: ...
    @callback
    def _get_suggested_id(self, info: dict[str, Any]) -> str: ...
    async def _update_data(self, item: dict[str, Any], update_data: dict[str, Any]) -> dict[str, Any]: ...
    def _serialize_item(self, item_id: str, item: dict[str, Any]) -> dict[str, Any]: ...

class TagDictStorageCollectionWebsocket(collection.StorageCollectionWebsocket[TagStorageCollection]):
    def __init__(
        self,
        storage_collection: TagStorageCollection,
        api_prefix: str,
        model_name: str,
        create_schema: dict[Any, Any],
        update_schema: dict[Any, Any],
    ) -> None: ...
    @callback
    def ws_list_item(self, hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict[str, Any]) -> None: ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool: ...

async def async_scan_tag(
    hass: HomeAssistant,
    tag_id: str,
    device_id: Optional[str],
    context: Optional[Context] = ...,
) -> None: ...

class TagEntity(Entity):
    _unrecorded_attributes: ClassVar[frozenset[str]]
    _attr_should_poll: ClassVar[bool]

    def __init__(
        self,
        entity_update_handlers: dict[str, Callable[[Optional[str], Optional[str]], None]],
        name: str,
        tag_id: str,
        last_scanned: Optional[str],
        device_id: Optional[str],
    ) -> None: ...
    @callback
    def async_handle_event(self, device_id: Optional[str], last_scanned: Optional[str]) -> None: ...
    @property
    @final
    def state(self) -> Optional[str]: ...
    @property
    def extra_state_attributes(self) -> dict[str, Optional[str]]: ...
    async def async_added_to_hass(self) -> None: ...
    async def async_will_remove_from_hass(self) -> None: ...
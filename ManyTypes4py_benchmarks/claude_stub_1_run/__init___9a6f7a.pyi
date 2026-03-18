```pyi
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, final

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.const import CONF_ID, CONF_NAME
from homeassistant.core import Context, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import collection, config_validation as cv, entity_registry as er
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.storage import Store
from homeassistant.helpers.typing import ConfigType, VolDictType
from homeassistant.util import dt as dt_util, slugify
from homeassistant.util.hass_dict import HassKey

LAST_SCANNED: str
LAST_SCANNED_BY_DEVICE_ID: str
STORAGE_KEY: str
STORAGE_VERSION: int
STORAGE_VERSION_MINOR: int
TAG_DATA: HassKey[TagStorageCollection]
CREATE_FIELDS: dict[vol.Marker, Any]
UPDATE_FIELDS: dict[vol.Marker, Any]
CONFIG_SCHEMA: vol.Schema[Any]

class TagIDExistsError(HomeAssistantError):
    item_id: Any
    def __init__(self, item_id: Any) -> None: ...

class TagIDManager(collection.IDManager):
    def generate_id(self, suggestion: Any) -> Any: ...

def _create_entry(
    entity_registry: er.EntityRegistry, tag_id: str, name: str | None
) -> er.RegistryEntry: ...

class TagStore(Store[collection.SerializedStorageCollection]):
    async def _async_migrate_func(
        self, old_major_version: int, old_minor_version: int, old_data: Any
    ) -> Any: ...

class TagStorageCollection(collection.DictStorageCollection):
    CREATE_SCHEMA: vol.Schema[Any]
    UPDATE_SCHEMA: vol.Schema[Any]
    entity_registry: er.EntityRegistry
    def __init__(
        self, store: TagStore, id_manager: TagIDManager | None = None
    ) -> None: ...
    async def _process_create_data(self, data: Any) -> Any: ...
    @callback
    def _get_suggested_id(self, info: Any) -> Any: ...
    async def _update_data(self, item: Any, update_data: Any) -> Any: ...
    def _serialize_item(self, item_id: str, item: Any) -> Any: ...

class TagDictStorageCollectionWebsocket(
    collection.StorageCollectionWebsocket[TagStorageCollection]
):
    entity_registry: er.EntityRegistry
    def __init__(
        self,
        storage_collection: TagStorageCollection,
        api_prefix: str,
        model_name: str,
        create_schema: dict[vol.Marker, Any],
        update_schema: dict[vol.Marker, Any],
    ) -> None: ...
    @callback
    def ws_list_item(
        self, hass: HomeAssistant, connection: Any, msg: Any
    ) -> None: ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool: ...
async def async_scan_tag(
    hass: HomeAssistant,
    tag_id: str,
    device_id: str,
    context: Context | None = None,
) -> None: ...

class TagEntity(Entity):
    _unrecorded_attributes: frozenset[str]
    _attr_should_poll: bool
    _entity_update_handlers: dict[str, Callable[[str, Any], None]]
    _attr_name: str
    _tag_id: str
    _attr_unique_id: str
    _last_device_id: str | None
    _last_scanned: str | None
    def __init__(
        self,
        entity_update_handlers: dict[str, Callable[[str, Any], None]],
        name: str,
        tag_id: str,
        last_scanned: str | None,
        device_id: str | None,
    ) -> None: ...
    @callback
    def async_handle_event(self, device_id: str, last_scanned: str) -> None: ...
    @property
    @final
    def state(self) -> str | None: ...
    @property
    def extra_state_attributes(self) -> dict[str, Any]: ...
    async def async_added_to_hass(self) -> None: ...
    async def async_will_remove_from_hass(self) -> None: ...
```
"""The Tag integration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
from typing import TYPE_CHECKING, Any, final
import uuid

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
from homeassistant.util import dt as dt_util
from homeassistant.util.hass_dict import HassKey
from .const import DEFAULT_NAME, DEVICE_ID, DOMAIN, EVENT_TAG_SCANNED, LOGGER, TAG_ID

if TYPE_CHECKING:
    from homeassistant.helpers.entity_registry import RegistryEntry

_LOGGER: logging.Logger = ...
LAST_SCANNED: str = ...
LAST_SCANNED_BY_DEVICE_ID: str = ...
STORAGE_KEY: str = ...
STORAGE_VERSION: int = ...
STORAGE_VERSION_MINOR: int = ...
TAG_DATA: HassKey[Any] = ...
CREATE_FIELDS: VolDictType = ...
UPDATE_FIELDS: VolDictType = ...
CONFIG_SCHEMA: vol.Schema = ...

class TagIDExistsError(HomeAssistantError):
    """Raised when an item is not found."""

    def __init__(self, item_id: str) -> None: ...
    item_id: str

class TagIDManager(collection.IDManager):
    """ID manager for tags."""

    def generate_id(self, suggestion: str) -> str: ...

def _create_entry(entity_registry: er.EntityRegistry, tag_id: str, name: str | None) -> RegistryEntry: ...

class TagStore(Store[collection.SerializedStorageCollection]):
    """Store tag data."""

    async def _async_migrate_func(
        self,
        old_major_version: int,
        old_minor_version: int,
        old_data: dict[str, Any]
    ) -> dict[str, Any]: ...

class TagStorageCollection(collection.DictStorageCollection):
    """Tag collection stored in storage."""

    CREATE_SCHEMA: vol.Schema = ...
    UPDATE_SCHEMA: vol.Schema = ...
    entity_registry: er.EntityRegistry

    def __init__(
        self,
        store: TagStore,
        id_manager: TagIDManager | None = None
    ) -> None: ...

    async def _process_create_data(self, data: dict[str, Any]) -> dict[str, Any]: ...

    @callback
    def _get_suggested_id(self, info: dict[str, Any]) -> str: ...

    async def _update_data(
        self,
        item: dict[str, Any],
        update_data: dict[str, Any]
    ) -> dict[str, Any]: ...

    def _serialize_item(
        self,
        item_id: str,
        item: dict[str, Any]
    ) -> dict[str, Any]: ...

class TagDictStorageCollectionWebsocket(
    collection.StorageCollectionWebsocket[TagStorageCollection]
):
    """Class to expose tag storage collection management over websocket."""

    entity_registry: er.EntityRegistry

    def __init__(
        self,
        storage_collection: TagStorageCollection,
        api_prefix: str,
        model_name: str,
        create_schema: VolDictType,
        update_schema: VolDictType
    ) -> None: ...

    @callback
    def ws_list_item(
        self,
        hass: HomeAssistant,
        connection: websocket_api.ActiveConnection,
        msg: dict[str, Any]
    ) -> None: ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool: ...

async def async_scan_tag(
    hass: HomeAssistant,
    tag_id: str,
    device_id: str | None,
    context: Context | None = None
) -> None: ...

class TagEntity(Entity):
    """Representation of a Tag entity."""

    _unrecorded_attributes: frozenset[str] = ...
    _attr_should_poll: bool = ...
    _entity_update_handlers: dict[str, Callable[[str | None, str | None], None]]
    _attr_name: str | None
    _tag_id: str
    _attr_unique_id: str
    _last_device_id: str | None
    _last_scanned: str | None

    def __init__(
        self,
        entity_update_handlers: dict[str, Callable[[str | None, str | None], None]],
        name: str,
        tag_id: str,
        last_scanned: str | None,
        device_id: str | None
    ) -> None: ...

    @callback
    def async_handle_event(
        self,
        device_id: str | None,
        last_scanned: str | None
    ) -> None: ...

    @property
    @final
    def state(self) -> str | None: ...

    @property
    def extra_state_attributes(self) -> dict[str, str | None]: ...

    async def async_added_to_hass(self) -> None: ...

    async def async_will_remove_from_hass(self) -> None: ...
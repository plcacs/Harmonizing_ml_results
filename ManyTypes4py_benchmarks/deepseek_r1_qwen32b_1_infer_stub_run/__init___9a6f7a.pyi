"""The Tag integration."""
from __future__ import annotations
from collections.abc import Callable
from datetime import datetime
import logging
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    FrozenSet,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
)
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

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
LAST_SCANNED: Final[str] = 'last_scanned'
LAST_SCANNED_BY_DEVICE_ID: Final[str] = 'last_scanned_by_device_id'
STORAGE_KEY: Final[str] = DOMAIN
STORAGE_VERSION: Final[int] = 1
STORAGE_VERSION_MINOR: Final[int] = 3
TAG_DATA: Final[HassKey] = HassKey(DOMAIN)
CREATE_FIELDS: Final[VolDictType] = vol.Schema({vol.Optional(TAG_ID): cv.string, vol.Optional(CONF_NAME): vol.All(str, vol.Length(min=1)), vol.Optional('description'): cv.string, vol.Optional(LAST_SCANNED): cv.datetime, vol.Optional(DEVICE_ID): cv.string})
UPDATE_FIELDS: Final[VolDictType] = vol.Schema({vol.Optional(CONF_NAME): vol.All(str, vol.Length(min=1)), vol.Optional('description'): cv.string, vol.Optional(LAST_SCANNED): cv.datetime, vol.Optional(DEVICE_ID): cv.string})
CONFIG_SCHEMA: Final[vol.Schema] = cv.empty_config_schema(DOMAIN)

class TagIDExistsError(HomeAssistantError):
    """Raised when an item is not found."""

    def __init__(self, item_id: str) -> None:
        ...

class TagIDManager(collection.IDManager):
    """ID manager for tags."""

    def generate_id(self, suggestion: str) -> str:
        ...

class TagStore(Store[collection.SerializedStorageCollection]):
    """Store tag data."""

    async def _async_migrate_func(self, old_major_version: int, old_minor_version: int, old_data: Any) -> Any:
        ...

class TagStorageCollection(collection.DictStorageCollection):
    """Tag collection stored in storage."""
    CREATE_SCHEMA: ClassVar[vol.Schema] = vol.Schema(CREATE_FIELDS)
    UPDATE_SCHEMA: ClassVar[vol.Schema] = vol.Schema(UPDATE_FIELDS)

    def __init__(self, store: Store[collection.SerializedStorageCollection], id_manager: Optional[collection.IDManager] = ...) -> None:
        ...

    async def _process_create_data(self, data: dict) -> dict:
        ...

    @callback
    def _get_suggested_id(self, info: dict) -> str:
        ...

    async def _update_data(self, item: dict, update_data: dict) -> dict:
        ...

    def _serialize_item(self, item_id: str, item: dict) -> dict:
        ...

class TagDictStorageCollectionWebsocket(collection.StorageCollectionWebsocket[TagStorageCollection]):
    """Class to expose tag storage collection management over websocket."""

    def __init__(self, storage_collection: TagStorageCollection, api_prefix: str, model_name: str, create_schema: vol.Schema, update_schema: vol.Schema) -> None:
        ...

    @callback
    def ws_list_item(self, hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict) -> None:
        ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    ...

async def async_scan_tag(hass: HomeAssistant, tag_id: str, device_id: str, context: Optional[Context] = ...) -> None:
    ...

class TagEntity(Entity):
    """Representation of a Tag entity."""
    _unrecorded_attributes: ClassVar[FrozenSet[str]] = frozenset({TAG_ID})
    _attr_should_poll: ClassVar[bool] = False

    def __init__(self, entity_update_handlers: dict, name: str, tag_id: str, last_scanned: Optional[datetime], device_id: Optional[str]) -> None:
        ...

    @callback
    def async_handle_event(self, device_id: str, last_scanned: datetime) -> None:
        ...

    @property
    @final
    def state(self) -> Optional[str]:
        ...

    @property
    def extra_state_attributes(self) -> dict:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    async def async_will_remove_from_hass(self) -> None:
        ...
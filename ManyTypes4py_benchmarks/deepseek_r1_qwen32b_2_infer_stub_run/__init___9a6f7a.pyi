"""The Tag integration."""

from __future__ import annotations
from collections.abc import Callable
from datetime import datetime
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import logging
import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import collection, config_validation as cv, entity_registry as er
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.typing import ConfigType
from homeassistant.util.dt import utcnow

LOGGER = logging.Logger

TagIDExistsError = Type[HomeAssistantError]

class TagIDManager:
    def generate_id(self, suggestion: str) -> str: ...

class TagStore:
    def __init__(self, hass: HomeAssistant, major_version: int, key: str, minor_version: int) -> None: ...
    async def _async_migrate_func(self, old_major_version: int, old_minor_version: int, old_data: Dict) -> Dict: ...

class TagStorageCollection:
    def __init__(self, store: TagStore, id_manager: TagIDManager) -> None: ...
    async def _process_create_data(self, data: Dict) -> Dict: ...
    @callback
    def _get_suggested_id(self, info: Dict) -> str: ...
    async def _update_data(self, item: Dict, update_data: Dict) -> Dict: ...
    def async_items(self) -> Iterable[Dict]: ...

class TagDictStorageCollectionWebsocket:
    def __init__(self, storage_collection: TagStorageCollection, api_prefix: str, model_name: str, create_schema: vol.Schema, update_schema: vol.Schema) -> None: ...
    @callback
    def ws_list_item(self, hass: HomeAssistant, connection: websocket_api.WebSocketConnection, msg: Dict) -> None: ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool: ...

async def async_scan_tag(hass: HomeAssistant, tag_id: str, device_id: str, context: Optional[Context] = None) -> None: ...

class TagEntity(Entity):
    _unrecorded_attributes: FrozenSet[str] = ...
    _attr_should_poll: bool = ...

    def __init__(self, entity_update_handlers: Dict[str, Callable[[str, datetime], None]], name: str, tag_id: str, last_scanned: Optional[datetime], device_id: Optional[str]) -> None: ...
    @callback
    def async_handle_event(self, device_id: str, last_scanned: datetime) -> None: ...
    @property
    def state(self) -> Optional[str]: ...
    @property
    def extra_state_attributes(self) -> Dict[str, Optional[str]]: ...
    async def async_added_to_hass(self) -> None: ...
    async def async_will_remove_from_hass(self) -> None: ...
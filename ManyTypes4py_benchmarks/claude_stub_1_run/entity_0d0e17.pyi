```pyi
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Collection, Mapping
from typing import Any

from homeassistant.core import CALLBACK_TYPE, Event, EventStateChangedData, HomeAssistant, State, callback
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_component import EntityComponent

from .registry import GroupIntegrationRegistry, SingleStateType

ENTITY_ID_FORMAT: str

class GroupEntity(Entity):
    _unrecorded_attributes: frozenset[str]
    _attr_should_poll: bool
    
    @callback
    def async_start_preview(self, preview_callback: Callable[[str, dict[str, Any]], None]) -> CALLBACK_TYPE: ...
    
    async def async_added_to_hass(self) -> None: ...
    
    @callback
    def _update_at_start(self, _: Any) -> None: ...
    
    @callback
    def async_defer_or_update_ha_state(self) -> None: ...
    
    @abstractmethod
    @callback
    def async_update_group_state(self) -> None: ...
    
    @callback
    def async_update_supported_features(self, entity_id: str, new_state: State | None) -> None: ...

class Group(Entity):
    _unrecorded_attributes: frozenset[str]
    _attr_should_poll: bool
    created_by_service: bool
    mode: Callable[[Any], bool]
    
    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: list[str],
        icon: str | None,
        mode: bool,
        order: int,
    ) -> None: ...
    
    @staticmethod
    @callback
    def async_create_group_entity(
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: list[str],
        icon: str | None,
        mode: bool,
        object_id: str | None,
        order: int | None,
    ) -> Group: ...
    
    @staticmethod
    async def async_create_group(
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: list[str],
        icon: str | None,
        mode: bool,
        object_id: str | None,
        order: int | None,
    ) -> Group: ...
    
    def set_name(self, value: str) -> None: ...
    
    @property
    def state(self) -> str | None: ...
    
    def set_icon(self, value: str | None) -> None: ...
    
    @property
    def extra_state_attributes(self) -> dict[str, Any]: ...
    
    @property
    def assumed_state(self) -> bool: ...
    
    @callback
    def async_update_tracked_entity_ids(self, entity_ids: list[str]) -> None: ...
    
    def _set_tracked(self, entity_ids: list[str]) -> None: ...
    
    @callback
    def _async_deregister(self) -> None: ...
    
    @callback
    def _async_start(self, _: Any = ...) -> None: ...
    
    @callback
    def _async_start_tracking(self) -> None: ...
    
    @callback
    def _async_stop(self) -> None: ...
    
    @callback
    def async_update_group_state(self) -> None: ...
    
    async def async_added_to_hass(self) -> None: ...
    
    async def async_will_remove_from_hass(self) -> None: ...
    
    async def _async_state_changed_listener(self, event: Event[EventStateChangedData]) -> None: ...
    
    def _reset_tracked_state(self) -> None: ...
    
    def _see_state(self, new_state: State) -> None: ...
    
    @callback
    def _async_update_group_state(self, tr_state: State | None = ...) -> None: ...

def async_get_component(hass: HomeAssistant) -> EntityComponent[Group]: ...
```
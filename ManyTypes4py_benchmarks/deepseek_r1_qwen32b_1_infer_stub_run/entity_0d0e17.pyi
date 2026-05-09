"""Provide entity classes for group entities."""
from __future__ import annotations
from collections.abc import Callable, Iterable
from typing import Any, Callable, List, Optional, Union

from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
    HomeAssistant,
    State,
    EventStateChangedData,
)
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_component import EntityComponent

class GroupEntity(Entity):
    """Representation of a Group of entities."""
    _unrecorded_attributes: frozenset[str] = ...
    _attr_should_poll: bool = ...

    @callback
    def async_start_preview(self, preview_callback: Callable) -> CALLBACK_TYPE:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @callback
    def _update_at_start(self, _) -> None:
        ...

    @callback
    def async_defer_or_update_ha_state(self) -> None:
        ...

    @abstractmethod
    @callback
    def async_update_group_state(self) -> None:
        ...

    @callback
    def async_update_supported_features(self, entity_id: str, new_state: State) -> None:
        ...

class Group(GroupEntity):
    """Track a group of entity ids."""
    _unrecorded_attributes: frozenset[str] = ...
    _attr_should_poll: bool = ...

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: List[str],
        icon: str,
        mode: Callable[[Iterable[bool]], bool],
        order: int
    ) -> None:
        ...

    @staticmethod
    @callback
    def async_create_group_entity(
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: List[str],
        icon: str,
        mode: Callable[[Iterable[bool]], bool],
        object_id: str,
        order: int
    ) -> Group:
        ...

    @staticmethod
    async def async_create_group(
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: List[str],
        icon: str,
        mode: Callable[[Iterable[bool]], bool],
        object_id: str,
        order: int
    ) -> Group:
        ...

    def set_name(self, value: str) -> None:
        ...

    @property
    def state(self) -> str:
        ...

    def set_icon(self, value: str) -> None:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...

    @property
    def assumed_state(self) -> bool:
        ...

    @callback
    def async_update_tracked_entity_ids(self, entity_ids: List[str]) -> None:
        ...

    def _set_tracked(self, entity_ids: List[str]) -> None:
        ...

    @callback
    def _async_deregister(self) -> None:
        ...

    @callback
    def _async_start(self, _: Optional[Any] = None) -> None:
        ...

    @callback
    def _async_start_tracking(self) -> None:
        ...

    @callback
    def _async_stop(self) -> None:
        ...

    @callback
    def async_update_group_state(self) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    async def async_will_remove_from_hass(self) -> None:
        ...

    async def _async_state_changed_listener(self, event: Event) -> None:
        ...

    def _reset_tracked_state(self) -> None:
        ...

    def _see_state(self, new_state: State) -> None:
        ...

    @callback
    def _async_update_group_state(self, tr_state: Optional[State] = None) -> None:
        ...

def async_get_component(hass: HomeAssistant) -> EntityComponent[Group]:
    ...
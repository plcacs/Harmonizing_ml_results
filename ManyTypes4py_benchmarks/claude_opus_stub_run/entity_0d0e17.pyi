from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Collection, Mapping
import logging
from typing import Any

from homeassistant.const import ATTR_ASSUMED_STATE, ATTR_ENTITY_ID, STATE_OFF, STATE_ON
from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
    split_entity_id,
)
from homeassistant.helpers import start
from homeassistant.helpers.entity import Entity, async_generate_entity_id
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.event import async_track_state_change_event

from .const import ATTR_AUTO, ATTR_ORDER, DATA_COMPONENT, DOMAIN, GROUP_ORDER, REG_KEY
from .registry import GroupIntegrationRegistry, SingleStateType

ENTITY_ID_FORMAT: str
_PACKAGE_LOGGER: logging.Logger
_LOGGER: logging.Logger

class GroupEntity(Entity):
    _unrecorded_attributes: frozenset[str]
    _attr_should_poll: bool
    _entity_ids: Collection[str]

    @callback
    def async_start_preview(
        self,
        preview_callback: Callable[[str, Mapping[str, Any]], None],
    ) -> CALLBACK_TYPE: ...

    async def async_added_to_hass(self) -> None: ...

    @callback
    def _update_at_start(self, _: Any) -> None: ...

    @callback
    def async_defer_or_update_ha_state(self) -> None: ...

    @abstractmethod
    @callback
    def async_update_group_state(self) -> None: ...

    @callback
    def async_update_supported_features(
        self, entity_id: str, new_state: State | None
    ) -> None: ...

class Group(Entity):
    _unrecorded_attributes: frozenset[str]
    _attr_should_poll: bool
    _attr_name: str | None
    _attr_icon: str | None
    _state: str | None
    _entity_ids: Collection[str]
    _on_off: dict[str, bool]
    _assumed: dict[str, bool]
    _on_states: set[str]
    created_by_service: bool
    mode: Callable[[Any], bool]
    _order: int
    _assumed_state: bool
    _async_unsub_state_changed: CALLBACK_TYPE | None
    tracking: tuple[str, ...]
    trackable: tuple[str, ...]
    single_state_type_key: SingleStateType | None
    _registry: GroupIntegrationRegistry

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: Collection[str],
        icon: str | None,
        mode: bool | None,
        order: int,
    ) -> None: ...

    @staticmethod
    @callback
    def async_create_group_entity(
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: Collection[str],
        icon: str | None,
        mode: bool | None,
        object_id: str | None,
        order: int | None,
    ) -> Group: ...

    @staticmethod
    async def async_create_group(
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: Collection[str],
        icon: str | None,
        mode: bool | None,
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
    def async_update_tracked_entity_ids(
        self, entity_ids: Collection[str]
    ) -> None: ...

    def _set_tracked(self, entity_ids: Collection[str]) -> None: ...

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

    async def _async_state_changed_listener(
        self, event: Event[EventStateChangedData]
    ) -> None: ...

    def _reset_tracked_state(self) -> None: ...

    def _see_state(self, new_state: State) -> None: ...

    @callback
    def _async_update_group_state(
        self, tr_state: State | None = ...
    ) -> None: ...

def async_get_component(hass: HomeAssistant) -> EntityComponent[Group]: ...
"""Provide entity classes for group entities."""
from __future__ import annotations
from collections.abc import Callable, Collection, Mapping
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from homeassistant.const import ATTR_ASSUMED_STATE, ATTR_ENTITY_ID, STATE_OFF, STATE_ON
from homeassistant.core import CALLBACK_TYPE, Event, EventStateChangedData, HomeAssistant, State, callback, split_entity_id
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_component import EntityComponent
from .const import ATTR_AUTO, ATTR_ORDER, DATA_COMPONENT, DOMAIN, GROUP_ORDER, REG_KEY
from .registry import GroupIntegrationRegistry, SingleStateType

ENTITY_ID_FORMAT = DOMAIN + '.{}'
_PACKAGE_LOGGER = logging.getLogger(__package__)
_LOGGER = logging.getLogger(__name__)

class GroupEntity(Entity):
    """Representation of a Group of entities."""
    _unrecorded_attributes: frozenset[str] = ...
    _attr_should_poll: bool = ...

    @callback
    def async_start_preview(self, preview_callback: Callable[[str, Dict[str, Any]], None]) -> CALLBACK_TYPE:
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

    def __init__(self, hass: HomeAssistant, name: str, *, created_by_service: bool, entity_ids: List[str], icon: Optional[str], mode: Callable, order: int) -> None:
        ...

    @staticmethod
    @callback
    def async_create_group_entity(hass: HomeAssistant, name: str, *, created_by_service: bool, entity_ids: List[str], icon: Optional[str], mode: Callable, object_id: Optional[str], order: Optional[int]) -> Group:
        ...

    @staticmethod
    async def async_create_group(hass: HomeAssistant, name: str, *, created_by_service: bool, entity_ids: List[str], icon: Optional[str], mode: Callable, object_id: Optional[str], order: Optional[int]) -> Group:
        ...

    def set_name(self, value: str) -> None:
        ...

    @property
    def state(self) -> str:
        ...

    def set_icon(self, value: Optional[str]) -> None:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
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
    def _async_start(self, _=None) -> None:
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
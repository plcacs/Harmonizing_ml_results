from __future__ import annotations
from abc import abstractmethod
from collections.abc import Callable, Collection, Mapping
import logging
from typing import Any
from homeassistant.const import ATTR_ASSUMED_STATE, ATTR_ENTITY_ID, STATE_OFF, STATE_ON
from homeassistant.core import CALLBACK_TYPE, Event, EventStateChangedData, HomeAssistant, State, callback, split_entity_id
from homeassistant.helpers import start
from homeassistant.helpers.entity import Entity, async_generate_entity_id
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.event import async_track_state_change_event
from .const import ATTR_AUTO, ATTR_ORDER, DATA_COMPONENT, DOMAIN, GROUP_ORDER, REG_KEY
from .registry import GroupIntegrationRegistry, SingleStateType

ENTITY_ID_FORMAT: str = DOMAIN + '.{}'
_PACKAGE_LOGGER: logging.Logger = logging.getLogger(__package__)
_LOGGER: logging.Logger = logging.getLogger(__name__)

class GroupEntity(Entity):
    _unrecorded_attributes: frozenset = frozenset({ATTR_ENTITY_ID})
    _attr_should_poll: bool = False

    @callback
    def async_start_preview(self, preview_callback: Callable):
        ...

    async def async_added_to_hass(self):
        ...

    @callback
    def _update_at_start(self, _: Any):
        ...

    @callback
    def async_defer_or_update_ha_state(self):
        ...

    @abstractmethod
    @callback
    def async_update_group_state(self):
        ...

    @callback
    def async_update_supported_features(self, entity_id: str, new_state: State):
        ...

class Group(Entity):
    _unrecorded_attributes: frozenset = frozenset({ATTR_ENTITY_ID, ATTR_ORDER, ATTR_AUTO})
    _attr_should_poll: bool = False

    def __init__(self, hass: HomeAssistant, name: str, *, created_by_service: bool, entity_ids: Collection[str], icon: str, mode: Callable, order: int):
        ...

    @staticmethod
    @callback
    def async_create_group_entity(hass: HomeAssistant, name: str, *, created_by_service: bool, entity_ids: Collection[str], icon: str, mode: Callable, object_id: str, order: int):
        ...

    @staticmethod
    async def async_create_group(hass: HomeAssistant, name: str, *, created_by_service: bool, entity_ids: Collection[str], icon: str, mode: Callable, object_id: str, order: int):
        ...

    def set_name(self, value: str):
        ...

    @property
    def state(self) -> Any:
        ...

    def set_icon(self, value: str):
        ...

    @property
    def extra_state_attributes(self) -> Mapping[str, Any]:
        ...

    @property
    def assumed_state(self) -> bool:
        ...

    @callback
    def async_update_tracked_entity_ids(self, entity_ids: Collection[str]):
        ...

    @callback
    def _async_deregister(self):
        ...

    @callback
    def _async_start(self, _: Any = None):
        ...

    @callback
    def _async_start_tracking(self):
        ...

    @callback
    def _async_stop(self):
        ...

    @callback
    def async_update_group_state(self):
        ...

    async def async_added_to_hass(self):
        ...

    async def async_will_remove_from_hass(self):
        ...

    async def _async_state_changed_listener(self, event: Event):
        ...

    def _reset_tracked_state(self):
        ...

    def _see_state(self, new_state: State):
        ...

    @callback
    def _async_update_group_state(self, tr_state: State = None):
        ...

def async_get_component(hass: HomeAssistant) -> EntityComponent[Group]:
    ...

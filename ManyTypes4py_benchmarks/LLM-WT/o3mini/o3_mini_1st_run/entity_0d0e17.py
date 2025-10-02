from __future__ import annotations
from abc import abstractmethod
from collections.abc import Callable, Collection, Mapping
import logging
from typing import Any, Optional, Union
from homeassistant.const import ATTR_ASSUMED_STATE, ATTR_ENTITY_ID, STATE_OFF, STATE_ON
from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
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

ENTITY_ID_FORMAT = DOMAIN + '.{}'
_PACKAGE_LOGGER = logging.getLogger(__package__)
_LOGGER = logging.getLogger(__name__)


class GroupEntity(Entity):
    _unrecorded_attributes = frozenset({ATTR_ENTITY_ID})
    _attr_should_poll = False

    @callback
    def async_start_preview(
        self, preview_callback: Callable[[Optional[str], Mapping[str, Any]], None]
    ) -> CALLBACK_TYPE:
        for entity_id in self._entity_ids:
            state = self.hass.states.get(entity_id)
            if state is None:
                continue
            self.async_update_supported_features(entity_id, state)

        @callback
        def async_state_changed_listener(event: Optional[Event]) -> None:
            self.async_update_group_state()
            if event:
                self.async_update_supported_features(
                    event.data["entity_id"], event.data["new_state"]
                )
            calculated_state = self._async_calculate_state()  # type: ignore
            preview_callback(calculated_state.state, calculated_state.attributes)

        async_state_changed_listener(None)
        return async_track_state_change_event(self.hass, self._entity_ids, async_state_changed_listener)

    async def async_added_to_hass(self) -> None:
        for entity_id in self._entity_ids:
            state = self.hass.states.get(entity_id)
            if state is None:
                continue
            self.async_update_supported_features(entity_id, state)

        @callback
        def async_state_changed_listener(event: Event) -> None:
            self.async_set_context(event.context)
            self.async_update_supported_features(
                event.data["entity_id"], event.data["new_state"]
            )
            self.async_defer_or_update_ha_state()

        self.async_on_remove(
            async_track_state_change_event(self.hass, self._entity_ids, async_state_changed_listener)
        )
        self.async_on_remove(start.async_at_start(self.hass, self._update_at_start))

    @callback
    def _update_at_start(self, _: Any) -> None:
        self.async_update_group_state()
        self.async_write_ha_state()

    @callback
    def async_defer_or_update_ha_state(self) -> None:
        if not self.hass.is_running:
            return
        self.async_update_group_state()
        self.async_write_ha_state()

    @abstractmethod
    @callback
    def async_update_group_state(self) -> None:
        """Abstract method to update the entity."""

    @callback
    def async_update_supported_features(self, entity_id: str, new_state: State) -> None:
        """Update dictionaries with supported features."""
        pass


class Group(Entity):
    _unrecorded_attributes = frozenset({ATTR_ENTITY_ID, ATTR_ORDER, ATTR_AUTO})
    _attr_should_poll = False

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: Collection[str],
        icon: Optional[str],
        mode: bool,
        order: Optional[int] = None,
    ) -> None:
        self.hass: HomeAssistant = hass
        self._attr_name: str = name
        self._state: Optional[str] = None
        self._attr_icon: Optional[str] = icon
        self._entity_ids: Collection[str] = entity_ids
        self._on_off: dict[str, bool] = {}
        self._assumed: dict[str, bool] = {}
        self._on_states: set[Any] = set()
        self.created_by_service: bool = created_by_service
        self.mode: Callable[[Collection[bool]], bool] = any
        if mode:
            self.mode = all
        self._order: int = order if order is not None else 0
        self._assumed_state: bool = False
        self._async_unsub_state_changed: Optional[CALLBACK_TYPE] = None
        self.tracking: tuple[str, ...] = ()
        self.trackable: tuple[str, ...] = ()
        self.single_state_type_key: Optional[SingleStateType] = None

    @staticmethod
    @callback
    def async_create_group_entity(
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: Collection[str],
        icon: Optional[str],
        mode: bool,
        object_id: Optional[str],
        order: Optional[int],
    ) -> Group:
        if order is None:
            hass.data.setdefault(GROUP_ORDER, 0)
            order = hass.data[GROUP_ORDER]
            hass.data[GROUP_ORDER] += 1
        group = Group(
            hass,
            name,
            created_by_service=created_by_service,
            entity_ids=entity_ids,
            icon=icon,
            mode=mode,
            order=order,
        )
        group.entity_id = async_generate_entity_id(ENTITY_ID_FORMAT, object_id or name, hass=hass)
        return group

    @staticmethod
    async def async_create_group(
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: Collection[str],
        icon: Optional[str],
        mode: bool,
        object_id: Optional[str],
        order: Optional[int],
    ) -> Group:
        group: Group = Group.async_create_group_entity(
            hass,
            name,
            created_by_service=created_by_service,
            entity_ids=entity_ids,
            icon=icon,
            mode=mode,
            object_id=object_id,
            order=order,
        )
        await async_get_component(hass).async_add_entities([group])
        return group

    def set_name(self, value: str) -> None:
        self._attr_name = value

    @property
    def state(self) -> Optional[str]:
        return self._state

    def set_icon(self, value: Optional[str]) -> None:
        self._attr_icon = value

    @property
    def extra_state_attributes(self) -> Mapping[str, Any]:
        data: dict[str, Any] = {ATTR_ENTITY_ID: self.tracking, ATTR_ORDER: self._order}
        if self.created_by_service:
            data[ATTR_AUTO] = True
        return data

    @property
    def assumed_state(self) -> bool:
        return self._assumed_state

    @callback
    def async_update_tracked_entity_ids(self, entity_ids: Collection[str]) -> None:
        self._async_stop()
        self._set_tracked(entity_ids)
        self._reset_tracked_state()
        self._async_start()

    def _set_tracked(self, entity_ids: Collection[str]) -> None:
        if not entity_ids:
            self.tracking = ()
            self.trackable = ()
            self.single_state_type_key = None
            return
        registry: GroupIntegrationRegistry = self._registry  # type: ignore
        excluded_domains = registry.exclude_domains
        tracking: list[str] = []
        trackable: list[str] = []
        single_state_type_set: set[SingleStateType] = set()
        for ent_id in entity_ids:
            ent_id_lower = ent_id.lower()
            domain = split_entity_id(ent_id_lower)[0]
            tracking.append(ent_id_lower)
            if domain not in excluded_domains:
                trackable.append(ent_id_lower)
            if domain in registry.state_group_mapping:
                single_state_type_set.add(registry.state_group_mapping[domain])
            elif domain == DOMAIN:
                if ent_id in registry.state_group_mapping:
                    single_state_type_set.add(registry.state_group_mapping[ent_id])
            else:
                single_state_type_set.add(SingleStateType(STATE_ON, STATE_OFF))
        if len(single_state_type_set) == 1:
            self.single_state_type_key = next(iter(single_state_type_set))
            registry.state_group_mapping[self.entity_id] = self.single_state_type_key
        else:
            self.single_state_type_key = None
        self.trackable = tuple(trackable)
        self.tracking = tuple(tracking)

    @callback
    def _async_deregister(self) -> None:
        registry: GroupIntegrationRegistry = self._registry  # type: ignore
        if self.entity_id in registry.state_group_mapping:
            registry.state_group_mapping.pop(self.entity_id)

    @callback
    def _async_start(self, _: Any = None) -> None:
        self._reset_tracked_state()
        self._async_start_tracking()
        self.async_write_ha_state()

    @callback
    def _async_start_tracking(self) -> None:
        if self.trackable and self._async_unsub_state_changed is None:
            self._async_unsub_state_changed = async_track_state_change_event(
                self.hass, self.trackable, self._async_state_changed_listener
            )
        self._async_update_group_state()

    @callback
    def _async_stop(self) -> None:
        if self._async_unsub_state_changed:
            self._async_unsub_state_changed()
            self._async_unsub_state_changed = None

    @callback
    def async_update_group_state(self) -> None:
        self._state = None
        self._async_update_group_state()

    async def async_added_to_hass(self) -> None:
        self._registry = self.hass.data[REG_KEY]
        self._set_tracked(self._entity_ids)
        self.async_on_remove(start.async_at_start(self.hass, self._async_start))
        self.async_on_remove(self._async_deregister)

    async def async_will_remove_from_hass(self) -> None:
        self._async_stop()

    async def _async_state_changed_listener(self, event: Event) -> None:
        if self._async_unsub_state_changed is None:
            return
        self.async_set_context(event.context)
        new_state: Optional[State] = event.data["new_state"]
        if new_state is None:
            self._reset_tracked_state()
        self._async_update_group_state(new_state)
        self.async_write_ha_state()

    def _reset_tracked_state(self) -> None:
        self._on_off = {}
        self._assumed = {}
        self._on_states = set()
        for entity_id in self.trackable:
            state: Optional[State] = self.hass.states.get(entity_id)
            if state is not None:
                self._see_state(state)

    def _see_state(self, new_state: State) -> None:
        entity_id: str = new_state.entity_id
        domain: str = new_state.domain
        state_str: str = new_state.state
        registry: GroupIntegrationRegistry = self._registry  # type: ignore
        self._assumed[entity_id] = bool(new_state.attributes.get(ATTR_ASSUMED_STATE))
        if domain not in registry.on_states_by_domain:
            if state_str in registry.on_off_mapping:
                self._on_states.add(state_str)
            elif state_str in registry.off_on_mapping:
                self._on_states.add(registry.off_on_mapping[state_str])
            self._on_off[entity_id] = state_str in registry.on_off_mapping
        else:
            entity_on_state = registry.on_states_by_domain[domain]
            self._on_states.update(entity_on_state)
            self._on_off[entity_id] = state_str in entity_on_state

    @callback
    def _async_update_group_state(self, tr_state: Optional[State] = None) -> None:
        if tr_state:
            self._see_state(tr_state)
        if not self._on_off:
            return
        if tr_state is None or (self._assumed_state and (not tr_state.attributes.get(ATTR_ASSUMED_STATE))):
            self._assumed_state = self.mode(self._assumed.values())
        elif tr_state.attributes.get(ATTR_ASSUMED_STATE):
            self._assumed_state = True
        num_on_states: int = len(self._on_states)
        if num_on_states == 1:
            on_state = next(iter(self._on_states))
        elif num_on_states == 0:
            self._state = None
            return
        else:
            on_state = STATE_ON
        if self.single_state_type_key:
            on_state = self.single_state_type_key.on_state
        group_is_on: bool = self.mode(self._on_off.values())
        if group_is_on:
            self._state = on_state
        elif self.single_state_type_key:
            self._state = self.single_state_type_key.off_state
        else:
            self._state = STATE_OFF


def async_get_component(hass: HomeAssistant) -> EntityComponent[Group]:
    if (component := hass.data.get(DATA_COMPONENT)) is None:
        component = hass.data[DATA_COMPONENT] = EntityComponent[Group](_PACKAGE_LOGGER, DOMAIN, hass)
    return component

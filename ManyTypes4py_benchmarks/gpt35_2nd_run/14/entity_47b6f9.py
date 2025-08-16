from __future__ import annotations
from typing import Any, Dict, Optional
from homeassistant.components.homeassistant import exposed_entities
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.const import ATTR_ENTITY_ID, SERVICE_TURN_OFF, SERVICE_TURN_ON, STATE_ON, STATE_UNAVAILABLE
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity, ToggleEntity
from homeassistant.helpers.event import async_track_state_change_event
from .const import DOMAIN as SWITCH_AS_X_DOMAIN

class BaseEntity(Entity):
    _attr_should_poll: bool = False

    def __init__(self, hass: HomeAssistant, config_entry_title: str, domain: str, switch_entity_id: str, unique_id: str) -> None:
        ...

    @callback
    def async_state_changed_listener(self, event: Optional[Event] = None) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @callback
    def async_generate_entity_options(self) -> Dict[str, Any]:
        ...

class BaseToggleEntity(BaseEntity, ToggleEntity):
    async def async_turn_on(self, **kwargs: Any) -> None:
        ...

    async def async_turn_off(self, **kwargs: Any) -> None:
        ...

    @callback
    def async_state_changed_listener(self, event: Optional[Event] = None) -> None:
        ...

class BaseInvertableEntity(BaseEntity):
    def __init__(self, hass: HomeAssistant, config_entry_title: str, domain: str, invert: bool, switch_entity_id: str, unique_id: str) -> None:
        ...

    @callback
    def async_generate_entity_options(self) -> Dict[str, Any]:
        ...

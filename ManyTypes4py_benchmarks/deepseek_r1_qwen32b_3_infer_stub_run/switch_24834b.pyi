"""Integration with the Rachio Iro sprinkler system controller."""
from abc import abstractmethod
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import voluptuous as vol
from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, ServiceCall
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import DOMAIN as DOMAIN_RACHIO, SIGNAL_RACHIO_CONTROLLER_UPDATE, SIGNAL_RACHIO_RAIN_DELAY_UPDATE, SIGNAL_RACHIO_SCHEDULE_UPDATE, SIGNAL_RACHIO_ZONE_UPDATE
from .device import RachioPerson
from .entity import RachioDevice, RachioHoseTimerEntity

__all__ = [
    'async_setup_entry',
    '_create_entities',
    'RachioSwitch',
    'RachioStandbySwitch',
    'RachioRainDelay',
    'RachioZone',
    'RachioSchedule',
    'RachioValve'
]

def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None: ...

def _create_entities(hass: HomeAssistant, config_entry: ConfigEntry) -> List[RachioSwitch]: ...

class RachioSwitch(RachioDevice, SwitchEntity):
    @callback
    def _async_handle_any_update(self, *args: Any, **kwargs: Any) -> None: ...
    
    @abstractmethod
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None: ...
    
    def turn_on(self, **kwargs: Any) -> None: ...
    
    def turn_off(self, **kwargs: Any) -> None: ...
    
    async def async_added_to_hass(self) -> None: ...

class RachioStandbySwitch(RachioSwitch):
    _attr_has_entity_name: bool
    _attr_translation_key: str
    
    @property
    def unique_id(self) -> str: ...
    
    def turn_on(self, **kwargs: Any) -> None: ...
    
    def turn_off(self, **kwargs: Any) -> None: ...
    
    async def async_added_to_hass(self) -> None: ...

class RachioRainDelay(RachioSwitch):
    _attr_has_entity_name: bool
    _attr_translation_key: str
    
    def __init__(self, controller: Any) -> None: ...
    
    @property
    def unique_id(self) -> str: ...
    
    @callback
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None: ...
    
    @callback
    def _delay_expiration(self, *args: Any) -> None: ...
    
    def turn_on(self, **kwargs: Any) -> None: ...
    
    def turn_off(self, **kwargs: Any) -> None: ...
    
    async def async_added_to_hass(self) -> None: ...

class RachioZone(RachioSwitch):
    _attr_icon: str
    
    def __init__(self, person: RachioPerson, controller: Any, data: Dict[str, Any], current_schedule: Dict[str, Any]) -> None: ...
    
    def __str__(self) -> str: ...
    
    @property
    def zone_id(self) -> str: ...
    
    @property
    def zone_is_enabled(self) -> bool: ...
    
    @property
    def extra_state_attributes(self) -> Dict[str, Union[str, int, None]]: ...
    
    def turn_on(self, **kwargs: Any) -> None: ...
    
    def turn_off(self, **kwargs: Any) -> None: ...
    
    def set_moisture_percent(self, percent: int) -> None: ...
    
    @callback
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None: ...
    
    async def async_added_to_hass(self) -> None: ...

class RachioSchedule(RachioSwitch):
    def __init__(self, person: RachioPerson, controller: Any, data: Dict[str, Any], current_schedule: Dict[str, Any]) -> None: ...
    
    @property
    def icon(self) -> str: ...
    
    @property
    def extra_state_attributes(self) -> Dict[str, Union[str, bool, int]]: ...
    
    @property
    def schedule_is_enabled(self) -> bool: ...
    
    def turn_on(self, **kwargs: Any) -> None: ...
    
    def turn_off(self, **kwargs: Any) -> None: ...
    
    @callback
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None: ...
    
    async def async_added_to_hass(self) -> None: ...

class RachioValve(RachioHoseTimerEntity, SwitchEntity):
    _attr_name: Optional[str]
    
    def __init__(self, person: RachioPerson, base: Any, data: Dict[str, Any], coordinator: Any) -> None: ...
    
    def turn_on(self, **kwargs: Any) -> None: ...
    
    def turn_off(self, **kwargs: Any) -> None: ...
    
    @callback
    def _update_attr(self) -> None: ...
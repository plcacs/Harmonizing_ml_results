"""Integration with the Rachio Iro sprinkler system controller."""
from abc import abstractmethod
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Union
import voluptuous
from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, ServiceCall, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.dt import as_timestamp, now, parse_datetime, utc_from_timestamp

class RachioSwitch(RachioDevice, SwitchEntity):
    """Represent a Rachio state that can be toggled."""
    @callback
    def _async_handle_any_update(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    @abstractmethod
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None:
        ...

class RachioStandbySwitch(RachioSwitch):
    """Representation of a standby status/button."""
    _attr_has_entity_name: bool
    _attr_translation_key: str
    def turn_on(self, **kwargs: Any) -> None:
        ...
    def turn_off(self, **kwargs: Any) -> None:
        ...
    async def async_added_to_hass(self) -> None:
        ...

class RachioRainDelay(RachioSwitch):
    """Representation of a rain delay status/switch."""
    _attr_has_entity_name: bool
    _attr_translation_key: str
    def __init__(self, controller: Any) -> None:
        ...
    @callback
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None:
        ...
    @callback
    def _delay_expiration(self, *args: Any) -> None:
        ...
    def turn_on(self, **kwargs: Any) -> None:
        ...
    def turn_off(self, **kwargs: Any) -> None:
        ...
    async def async_added_to_hass(self) -> None:
        ...

class RachioZone(RachioSwitch):
    """Representation of one zone of sprinklers connected to the Rachio Iro."""
    _attr_icon: str
    def __init__(self, person: Any, controller: Any, data: Dict[str, Any], current_schedule: Dict[str, Any]) -> None:
        ...
    def turn_on(self, **kwargs: Any) -> None:
        ...
    def turn_off(self, **kwargs: Any) -> None:
        ...
    def set_moisture_percent(self, percent: int) -> None:
        ...
    @callback
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None:
        ...
    async def async_added_to_hass(self) -> None:
        ...
    @property
    def zone_id(self) -> str:
        ...
    @property
    def zone_is_enabled(self) -> bool:
        ...
    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

class RachioSchedule(RachioSwitch):
    """Representation of one fixed schedule on the Rachio Iro."""
    def __init__(self, person: Any, controller: Any, data: Dict[str, Any], current_schedule: Dict[str, Any]) -> None:
        ...
    @property
    def icon(self) -> str:
        ...
    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...
    @property
    def schedule_is_enabled(self) -> bool:
        ...
    def turn_on(self, **kwargs: Any) -> None:
        ...
    def turn_off(self, **kwargs: Any) -> None:
        ...
    @callback
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None:
        ...
    async def async_added_to_hass(self) -> None:
        ...

class RachioValve(RachioHoseTimerEntity, SwitchEntity):
    """Representation of one smart hose timer valve."""
    _attr_name: Optional[str]
    def __init__(self, person: Any, base: Any, data: Dict[str, Any], coordinator: Any) -> None:
        ...
    def turn_on(self, **kwargs: Any) -> None:
        ...
    def turn_off(self, **kwargs: Any) -> None:
        ...
    @callback
    def _update_attr(self) -> None:
        ...

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

def _create_entities(hass: HomeAssistant, config_entry: ConfigEntry) -> List[RachioSwitch]:
    ...
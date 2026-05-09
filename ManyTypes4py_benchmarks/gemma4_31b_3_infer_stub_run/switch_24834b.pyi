"""Integration with the Rachio Iro sprinkler system controller."""

from abc import abstractmethod
from datetime import timedelta
from typing import Any, Optional, Union, Callable, Awaitable
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.entity import Entity
from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from .device import RachioPerson
from .entity import RachioDevice, RachioHoseTimerEntity

ATTR_DURATION: str = 'duration'
ATTR_PERCENT: str = 'percent'
ATTR_SCHEDULE_SUMMARY: str = 'Summary'
ATTR_SCHEDULE_ENABLED: str = 'Enabled'
ATTR_SCHEDULE_DURATION: str = 'Duration'
ATTR_SCHEDULE_TYPE: str = 'Type'
ATTR_SORT_ORDER: str = 'sortOrder'
ATTR_WATERING_DURATION: str = 'Watering Duration seconds'
ATTR_ZONE_NUMBER: str = 'Zone number'
ATTR_ZONE_SHADE: str = 'Shade'
ATTR_ZONE_SLOPE: str = 'Slope'
ATTR_ZONE_SUMMARY: str = 'Summary'
ATTR_ZONE_TYPE: str = 'Type'
START_MULTIPLE_ZONES_SCHEMA: Any = ...

async def async_setup_entry(
    hass: HomeAssistant, 
    config_entry: ConfigEntry, 
    async_add_entities: Callable[[list[Entity]], Awaitable[None]]
) -> None: ...

def _create_entities(hass: HomeAssistant, config_entry: ConfigEntry) -> list[Entity]: ...

class RachioSwitch(RachioDevice, SwitchEntity):
    """Represent a Rachio state that can be toggled."""
    _controller: Any

    @callback
    def _async_handle_any_update(self, *args: Any, **kwargs: Any) -> None: ...

    @abstractmethod
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None: ...

class RachioStandbySwitch(RachioSwitch):
    """Representation of a standby status/button."""
    _attr_has_entity_name: bool
    _attr_translation_key: str

    @property
    def unique_id(self) -> str: ...

    @callback
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None: ...

    def turn_on(self, **kwargs: Any) -> None: ...

    def turn_off(self, **kwargs: Any) -> None: ...

    async def async_added_to_hass(self) -> None: ...

class RachioRainDelay(RachioSwitch):
    """Representation of a rain delay status/switch."""
    _attr_has_entity_name: bool
    _attr_translation_key: str
    _cancel_update: Optional[Callable[[], None]]

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
    """Representation of one zone of sprinklers connected to the Rachio Iro."""
    _attr_icon: str
    id: str
    _attr_name: str
    _zone_number: int
    _zone_enabled: bool
    _attr_entity_picture: Optional[str]
    _person: RachioPerson
    _shade_type: Optional[str]
    _zone_type: Optional[str]
    _slope_type: Optional[str]
    _summary: str
    _current_schedule: dict[str, Any]
    _attr_unique_id: str

    def __init__(self, person: RachioPerson, controller: Any, data: dict[str, Any], current_schedule: dict[str, Any]) -> None: ...

    def __str__(self) -> str: ...

    @property
    def zone_id(self) -> str: ...

    @property
    def zone_is_enabled(self) -> bool: ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]: ...

    def turn_on(self, **kwargs: Any) -> None: ...

    def turn_off(self, **kwargs: Any) -> None: ...

    def set_moisture_percent(self, percent: float) -> None: ...

    @callback
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None: ...

    async def async_added_to_hass(self) -> None: ...

class RachioSchedule(RachioSwitch):
    """Representation of one fixed schedule on the Rachio Iro."""
    _schedule_id: str
    _duration: int
    _schedule_enabled: bool
    _summary: str
    type: str
    _current_schedule: dict[str, Any]
    _attr_unique_id: str
    _attr_name: str

    def __init__(self, person: RachioPerson, controller: Any, data: dict[str, Any], current_schedule: dict[str, Any]) -> None: ...

    @property
    def icon(self) -> str: ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]: ...

    @property
    def schedule_is_enabled(self) -> bool: ...

    def turn_on(self, **kwargs: Any) -> None: ...

    def turn_off(self, **kwargs: Any) -> None: ...

    @callback
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None: ...

    async def async_added_to_hass(self) -> None: ...

class RachioValve(RachioHoseTimerEntity, SwitchEntity):
    """Representation of one smart hose timer valve."""
    _attr_name: Optional[str]
    _person: RachioPerson
    _base: Any
    _attr_unique_id: str

    def __init__(self, person: RachioPerson, base: Any, data: dict[str, Any], coordinator: Any) -> None: ...

    def turn_on(self, **kwargs: Any) -> None: ...

    def turn_off(self, **kwargs: Any) -> None: ...

    @callback
    def _update_attr(self) -> None: ...
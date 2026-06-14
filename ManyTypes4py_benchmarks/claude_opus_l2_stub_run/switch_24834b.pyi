from abc import abstractmethod
from datetime import timedelta
from typing import Any

import voluptuous as vol

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, ServiceCall, callback
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .device import RachioPerson
from .entity import RachioDevice, RachioHoseTimerEntity

_LOGGER: logging.Logger

ATTR_DURATION: str
ATTR_PERCENT: str
ATTR_SCHEDULE_SUMMARY: str
ATTR_SCHEDULE_ENABLED: str
ATTR_SCHEDULE_DURATION: str
ATTR_SCHEDULE_TYPE: str
ATTR_SORT_ORDER: str
ATTR_WATERING_DURATION: str
ATTR_ZONE_NUMBER: str
ATTR_ZONE_SHADE: str
ATTR_ZONE_SLOPE: str
ATTR_ZONE_SUMMARY: str
ATTR_ZONE_TYPE: str

import logging

START_MULTIPLE_ZONES_SCHEMA: vol.Schema

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None: ...

def _create_entities(
    hass: HomeAssistant, config_entry: ConfigEntry
) -> list[Entity]: ...

class RachioSwitch(RachioDevice, SwitchEntity):
    @callback
    def _async_handle_any_update(self, *args: Any, **kwargs: Any) -> None: ...
    @abstractmethod
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None: ...

class RachioStandbySwitch(RachioSwitch):
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
    _attr_has_entity_name: bool
    _attr_translation_key: str
    _cancel_update: CALLBACK_TYPE | None
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
    id: str
    _attr_name: str
    _zone_number: int
    _zone_enabled: bool
    _attr_entity_picture: str | None
    _person: RachioPerson
    _shade_type: str | None
    _zone_type: str | None
    _slope_type: str | None
    _summary: str
    _current_schedule: dict[str, Any]
    _attr_unique_id: str
    def __init__(
        self,
        person: RachioPerson,
        controller: Any,
        data: dict[str, Any],
        current_schedule: dict[str, Any],
    ) -> None: ...
    def __str__(self) -> str: ...
    @property
    def zone_id(self) -> str: ...
    @property
    def zone_is_enabled(self) -> bool: ...
    @property
    def extra_state_attributes(self) -> dict[str, Any]: ...
    def turn_on(self, **kwargs: Any) -> None: ...
    def turn_off(self, **kwargs: Any) -> None: ...
    def set_moisture_percent(self, percent: int) -> None: ...
    @callback
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None: ...
    async def async_added_to_hass(self) -> None: ...

class RachioSchedule(RachioSwitch):
    _schedule_id: str
    _duration: int
    _schedule_enabled: bool
    _summary: str
    type: str
    _current_schedule: dict[str, Any]
    _attr_unique_id: str
    _attr_name: str
    def __init__(
        self,
        person: RachioPerson,
        controller: Any,
        data: dict[str, Any],
        current_schedule: dict[str, Any],
    ) -> None: ...
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
    _attr_name: None
    _person: RachioPerson
    _base: Any
    _attr_unique_id: str
    def __init__(
        self,
        person: RachioPerson,
        base: Any,
        data: dict[str, Any],
        coordinator: Any,
    ) -> None: ...
    def turn_on(self, **kwargs: Any) -> None: ...
    def turn_off(self, **kwargs: Any) -> None: ...
    @callback
    def _update_attr(self) -> None: ...
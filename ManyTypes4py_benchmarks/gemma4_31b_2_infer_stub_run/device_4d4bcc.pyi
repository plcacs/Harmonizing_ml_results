"""Adapter to wrap the rachiopy api for home assistant."""
from __future__ import annotations

from typing import Any, Optional, Union, Sequence, Mapping, Callable, Awaitable
from rachiopy import Rachio
from homeassistant.core import HomeAssistant, ServiceCall
from voluptuous import Schema

ATTR_DEVICES: str
ATTR_DURATION: str
PERMISSION_ERROR: str
PAUSE_SERVICE_SCHEMA: Schema
RESUME_SERVICE_SCHEMA: Schema
STOP_SERVICE_SCHEMA: Schema

class RachioPerson:
    """Represent a Rachio user."""

    def __init__(self, rachio: Rachio, config_entry: Any) -> None:
        """Create an object from the provided API instance."""
        self.rachio: Rachio
        self.config_entry: Any
        self.username: Optional[str]
        self._id: Optional[str]
        self._controllers: list[RachioIro]
        self._base_stations: list[RachioBaseStation]

    async def async_setup(self, hass: HomeAssistant) -> None:
        """Create rachio devices and services."""
        ...

    def _setup(self, hass: HomeAssistant) -> None:
        """Rachio device setup."""
        ...

    @property
    def user_id(self) -> Optional[str]:
        """Get the user ID as defined by the Rachio API."""
        ...

    @property
    def controllers(self) -> list[RachioIro]:
        """Get a list of controllers managed by this account."""
        ...

    @property
    def base_stations(self) -> list[RachioBaseStation]:
        """List of smart hose timer base stations."""
        ...

    def start_multiple_zones(self, zones: list[str]) -> None:
        """Start multiple zones."""
        ...

class RachioIro:
    """Represent a Rachio Iro."""

    def __init__(self, hass: HomeAssistant, rachio: Rachio, data: Mapping[str, Any], webhooks: Any) -> None:
        """Initialize a Rachio device."""
        self.hass: HomeAssistant
        self.rachio: Rachio
        self._id: str
        self.name: str
        self.serial_number: str
        self.mac_address: str
        self.model: str
        self._zones: list[Mapping[str, Any]]
        self._schedules: list[Mapping[str, Any]]
        self._flex_schedules: list[Mapping[str, Any]]
        self._init_data: Mapping[str, Any]
        self._webhooks: Union[list[Mapping[str, Any]], Mapping[str, Any]]

    def setup(self) -> None:
        """Rachio Iro setup for webhooks."""
        ...

    def _init_webhooks(self) -> None:
        """Start getting updates from the Rachio API."""
        ...

    def __str__(self) -> str:
        """Display the controller as a string."""
        ...

    @property
    def controller_id(self) -> str:
        """Return the Rachio API controller ID."""
        ...

    @property
    def current_schedule(self) -> Any:
        """Return the schedule that the device is running right now."""
        ...

    @property
    def init_data(self) -> Mapping[str, Any]:
        """Return the information used to set up the controller."""
        ...

    def list_zones(self, include_disabled: bool = False) -> list[Mapping[str, Any]]:
        """Return a list of the zone dicts connected to the device."""
        ...

    def get_zone(self, zone_id: str) -> Optional[Mapping[str, Any]]:
        """Return the zone with the given ID."""
        ...

    def list_schedules(self) -> list[Mapping[str, Any]]:
        """Return a list of fixed schedules."""
        ...

    def list_flex_schedules(self) -> list[Mapping[str, Any]]:
        """Return a list of flex schedules."""
        ...

    def stop_watering(self) -> None:
        """Stop watering all zones connected to this controller."""
        ...

    def pause_watering(self, duration: int) -> None:
        """Pause watering on this controller."""
        ...

    def resume_watering(self) -> None:
        """Resume paused watering on this controller."""
        ...

class RachioBaseStation:
    """Represent a smart hose timer base station."""

    def __init__(self, rachio: Rachio, data: Mapping[str, Any], status_coordinator: Any, schedule_coordinator: Any) -> None:
        """Initialize a smart hose timer base station."""
        self.rachio: Rachio
        self._id: str
        self.status_coordinator: Any
        self.schedule_coordinator: Any

    def start_watering(self, valve_id: str, duration: int) -> None:
        """Start watering on this valve."""
        ...

    def stop_watering(self, valve_id: str) -> None:
        """Stop watering on this valve."""
        ...

    def create_skip(self, program_id: str, timestamp: str) -> None:
        """Create a skip for a scheduled event."""
        ...

def is_invalid_auth_code(http_status_code: int) -> bool:
    """HTTP status codes that mean invalid auth."""
    ...
from __future__ import annotations

from typing import Any

import voluptuous as vol
from rachiopy import Rachio

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .coordinator import RachioScheduleUpdateCoordinator, RachioUpdateCoordinator

ATTR_DEVICES: str
ATTR_DURATION: str
PERMISSION_ERROR: str
PAUSE_SERVICE_SCHEMA: vol.Schema
RESUME_SERVICE_SCHEMA: vol.Schema
STOP_SERVICE_SCHEMA: vol.Schema

class RachioPerson:
    rachio: Rachio
    config_entry: ConfigEntry
    username: str | None
    _id: str | None
    _controllers: list[RachioIro]
    _base_stations: list[RachioBaseStation]

    def __init__(self, rachio: Rachio, config_entry: ConfigEntry) -> None: ...
    async def async_setup(self, hass: HomeAssistant) -> None: ...
    def _setup(self, hass: HomeAssistant) -> None: ...

    @property
    def user_id(self) -> str | None: ...
    @property
    def controllers(self) -> list[RachioIro]: ...
    @property
    def base_stations(self) -> list[RachioBaseStation]: ...

    def start_multiple_zones(self, zones: list[dict[str, Any]]) -> None: ...

class RachioIro:
    hass: HomeAssistant
    rachio: Rachio
    name: str
    serial_number: str
    mac_address: str
    model: str
    _id: str
    _zones: list[dict[str, Any]]
    _schedules: list[dict[str, Any]]
    _flex_schedules: list[dict[str, Any]]
    _init_data: dict[str, Any]
    _webhooks: list[dict[str, Any]]

    def __init__(
        self,
        hass: HomeAssistant,
        rachio: Rachio,
        data: dict[str, Any],
        webhooks: list[dict[str, Any]],
    ) -> None: ...
    def setup(self) -> None: ...
    def _init_webhooks(self) -> None: ...
    def __str__(self) -> str: ...

    @property
    def controller_id(self) -> str: ...
    @property
    def current_schedule(self) -> dict[str, Any]: ...
    @property
    def init_data(self) -> dict[str, Any]: ...

    def list_zones(self, include_disabled: bool = False) -> list[dict[str, Any]]: ...
    def get_zone(self, zone_id: str) -> dict[str, Any] | None: ...
    def list_schedules(self) -> list[dict[str, Any]]: ...
    def list_flex_schedules(self) -> list[dict[str, Any]]: ...
    def stop_watering(self) -> None: ...
    def pause_watering(self, duration: int) -> None: ...
    def resume_watering(self) -> None: ...

class RachioBaseStation:
    rachio: Rachio
    _id: str
    status_coordinator: RachioUpdateCoordinator
    schedule_coordinator: RachioScheduleUpdateCoordinator

    def __init__(
        self,
        rachio: Rachio,
        data: dict[str, Any],
        status_coordinator: RachioUpdateCoordinator,
        schedule_coordinator: RachioScheduleUpdateCoordinator,
    ) -> None: ...
    def start_watering(self, valve_id: str, duration: int) -> None: ...
    def stop_watering(self, valve_id: str) -> None: ...
    def create_skip(self, program_id: str, timestamp: str) -> None: ...

def is_invalid_auth_code(http_status_code: int) -> bool: ...
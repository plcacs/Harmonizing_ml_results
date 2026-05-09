"""Adapter to wrap the rachiopy api for home assistant."""

from __future__ import annotations
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from rachiopy import Rachio
from voluptuous import Schema

from .const import (
    ATTR_DEVICES,
    ATTR_DURATION,
    DOMAIN,
    HTTPStatus,
    KEY_BASE_STATIONS,
    KEY_DEVICES,
    KEY_ENABLED,
    KEY_EXTERNAL_ID,
    KEY_FLEX_SCHEDULES,
    KEY_ID,
    KEY_MAC_ADDRESS,
    KEY_MODEL,
    KEY_NAME,
    KEY_SCHEDULES,
    KEY_SERIAL_NUMBER,
    KEY_STATUS,
    KEY_USERNAME,
    KEY_ZONES,
    LISTEN_EVENT_TYPES,
    MODEL_GENERATION_1,
    SERVICE_PAUSE_WATERING,
    SERVICE_RESUME_WATERING,
    SERVICE_STOP_WATERING,
    WEBHOOK_CONST_ID,
)

PAUSE_SERVICE_SCHEMA: Schema = ...
RESUME_SERVICE_SCHEMA: Schema = ...
STOP_SERVICE_SCHEMA: Schema = ...

class RachioPerson:
    """Represent a Rachio user."""

    def __init__(self, rachio: Rachio, config_entry: ConfigEntry) -> None:
        ...

    async def async_setup(self, hass: HomeAssistant) -> None:
        ...

    def _setup(self, hass: HomeAssistant) -> None:
        ...

    @property
    def user_id(self) -> str:
        ...

    @property
    def controllers(self) -> List[RachioIro]:
        ...

    @property
    def base_stations(self) -> List[RachioBaseStation]:
        ...

    def start_multiple_zones(self, zones: List[str]) -> None:
        ...

class RachioIro:
    """Represent a Rachio Iro."""

    def __init__(self, hass: HomeAssistant, rachio: Rachio, data: Dict[str, Any], webhooks: Dict[str, Any]) -> None:
        ...

    def setup(self) -> None:
        ...

    def _init_webhooks(self) -> None:
        ...

    def __str__(self) -> str:
        ...

    @property
    def controller_id(self) -> str:
        ...

    @property
    def current_schedule(self) -> Dict[str, Any]:
        ...

    @property
    def init_data(self) -> Dict[str, Any]:
        ...

    def list_zones(self, include_disabled: bool = False) -> List[Dict[str, Any]]:
        ...

    def get_zone(self, zone_id: str) -> Optional[Dict[str, Any]]:
        ...

    def list_schedules(self) -> List[Dict[str, Any]]:
        ...

    def list_flex_schedules(self) -> List[Dict[str, Any]]:
        ...

    def stop_watering(self) -> None:
        ...

    def pause_watering(self, duration: int) -> None:
        ...

    def resume_watering(self) -> None:
        ...

class RachioBaseStation:
    """Represent a smart hose timer base station."""

    def __init__(self, rachio: Rachio, data: Dict[str, Any], status_coordinator: Any, schedule_coordinator: Any) -> None:
        ...

    def start_watering(self, valve_id: str, duration: int) -> None:
        ...

    def stop_watering(self, valve_id: str) -> None:
        ...

    def create_skip(self, program_id: str, timestamp: str) -> None:
        ...

def is_invalid_auth_code(http_status_code: int) -> bool:
    ...
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.device_registry import DeviceRegistry, CONNECTION_NETWORK_MAC, format_mac
from aioshelly.const import MODEL_25
from typing import Any
from unittest.mock import Mock
from freezegun.api import FrozenDateTimeFactory
import pytest

async def init_integration(hass: HomeAssistant, gen: Any, model: str = MODEL_25, sleep_period: int = 0, options: dict = None, skip_setup: bool = False) -> ConfigEntry:
    ...

def mutate_rpc_device_status(monkeypatch, mock_rpc_device, top_level_key: str, key: str, value: Any) -> None:
    ...

def inject_rpc_device_event(monkeypatch, mock_rpc_device, event: Any) -> None:
    ...

async def mock_rest_update(hass: HomeAssistant, freezer: FrozenDateTimeFactory, seconds: int = REST_SENSORS_UPDATE_INTERVAL) -> None:
    ...

async def mock_polling_rpc_update(hass: HomeAssistant, freezer: FrozenDateTimeFactory) -> None:
    ...

def register_entity(hass: HomeAssistant, domain: str, object_id: str, unique_id: str, config_entry: ConfigEntry = None, capabilities: dict = None, device_id: str = None) -> str:
    ...

def get_entity(hass: HomeAssistant, domain: str, unique_id: str) -> str:
    ...

def get_entity_state(hass: HomeAssistant, entity_id: str) -> Any:
    ...

def get_entity_attribute(hass: HomeAssistant, entity_id: str, attribute: str) -> Any:
    ...

def register_device(device_registry: DeviceRegistry, config_entry: ConfigEntry) -> None:
    ...

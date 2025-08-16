from __future__ import annotations
import asyncio
from collections.abc import Mapping
import logging
from typing import Any, Final, Dict, List, Optional
from flux_led.aioscanner import AIOBulbScanner
from flux_led.const import ATTR_ID, ATTR_IPADDR, ATTR_MODEL, ATTR_MODEL_DESCRIPTION, ATTR_MODEL_INFO, ATTR_MODEL_NUM, ATTR_REMOTE_ACCESS_ENABLED, ATTR_REMOTE_ACCESS_HOST, ATTR_REMOTE_ACCESS_PORT, ATTR_VERSION_NUM
from flux_led.models_db import get_model_description
from flux_led.scanner import FluxLEDDiscovery
from homeassistant.components import network
from homeassistant.config_entries import SOURCE_INTEGRATION_DISCOVERY, ConfigEntryState
from homeassistant.const import CONF_HOST, CONF_MODEL, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, discovery_flow
from homeassistant.util.async_ import create_eager_task
from homeassistant.util.network import is_ip_address
from .const import CONF_MINOR_VERSION, CONF_MODEL_DESCRIPTION, CONF_MODEL_INFO, CONF_MODEL_NUM, CONF_REMOTE_ACCESS_ENABLED, CONF_REMOTE_ACCESS_HOST, CONF_REMOTE_ACCESS_PORT, DIRECTED_DISCOVERY_TIMEOUT, DOMAIN, FLUX_LED_DISCOVERY
from .coordinator import FluxLedConfigEntry
from .util import format_as_flux_mac, mac_matches_by_one

@callback
def async_build_cached_discovery(entry: ConfigEntryState) -> FluxLEDDiscovery:
    ...

@callback
def async_name_from_discovery(device: Dict[str, Any], model_num: Optional[str] = None) -> str:
    ...

@callback
def async_populate_data_from_discovery(current_data: Dict[str, Any], data_updates: Dict[str, Any], device: Dict[str, Any]) -> None:
    ...

@callback
def async_update_entry_from_discovery(hass: HomeAssistant, entry: ConfigEntryState, device: Dict[str, Any], model_num: Optional[str], allow_update_mac: bool) -> bool:
    ...

@callback
def async_get_discovery(hass: HomeAssistant, host: str) -> Optional[Dict[str, Any]]:
    ...

@callback
def async_clear_discovery_cache(hass: HomeAssistant, host: str) -> None:
    ...

async def async_discover_devices(hass: HomeAssistant, timeout: int, address: Optional[str] = None) -> List[Dict[str, Any]]:
    ...

async def async_discover_device(hass: HomeAssistant, host: str) -> Optional[Dict[str, Any]]:
    ...

@callback
def async_trigger_discovery(hass: HomeAssistant, discovered_devices: List[Dict[str, Any]]) -> None:
    ...

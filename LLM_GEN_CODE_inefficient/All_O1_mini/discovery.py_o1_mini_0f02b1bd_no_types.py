"""The Flux LED/MagicLight integration discovery."""
from __future__ import annotations
import asyncio
from collections.abc import Mapping
import logging
from typing import Any, Final, Dict, Optional, List, Union
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
_LOGGER: logging.Logger = logging.getLogger(__name__)
CONF_TO_DISCOVERY: Final[Dict[str, str]] = {CONF_HOST: ATTR_IPADDR, CONF_REMOTE_ACCESS_ENABLED: ATTR_REMOTE_ACCESS_ENABLED, CONF_REMOTE_ACCESS_HOST: ATTR_REMOTE_ACCESS_HOST, CONF_REMOTE_ACCESS_PORT: ATTR_REMOTE_ACCESS_PORT, CONF_MINOR_VERSION: ATTR_VERSION_NUM, CONF_MODEL: ATTR_MODEL, CONF_MODEL_NUM: ATTR_MODEL_NUM, CONF_MODEL_INFO: ATTR_MODEL_INFO, CONF_MODEL_DESCRIPTION: ATTR_MODEL_DESCRIPTION}

@callback
def async_build_cached_discovery(entry):
    """When discovery is unavailable, load it from the config entry."""
    data: Dict[str, Any] = entry.data
    return FluxLEDDiscovery(ipaddr=data[CONF_HOST], model=data.get(CONF_MODEL), id=format_as_flux_mac(entry.unique_id) if entry.unique_id else None, model_num=data.get(CONF_MODEL_NUM), version_num=data.get(CONF_MINOR_VERSION), firmware_date=None, model_info=data.get(CONF_MODEL_INFO), model_description=data.get(CONF_MODEL_DESCRIPTION), remote_access_enabled=data.get(CONF_REMOTE_ACCESS_ENABLED), remote_access_host=data.get(CONF_REMOTE_ACCESS_HOST), remote_access_port=data.get(CONF_REMOTE_ACCESS_PORT))

@callback
def async_name_from_discovery(device, model_num=None):
    """Convert a flux_led discovery to a human readable name."""
    mac_address: Optional[str] = device.get(ATTR_ID)
    if mac_address is None:
        return device.get(ATTR_IPADDR, 'Unknown')
    short_mac: str = mac_address[-6:]
    if device.get(ATTR_MODEL_DESCRIPTION):
        return f'{device[ATTR_MODEL_DESCRIPTION]} {short_mac}'
    if model_num is not None:
        return f'{get_model_description(model_num, None)} {short_mac}'
    return f'{device.get(ATTR_MODEL, 'Unknown')} {short_mac}'

@callback
def async_populate_data_from_discovery(current_data, data_updates, device):
    """Copy discovery data into config entry data."""
    for conf_key, discovery_key in CONF_TO_DISCOVERY.items():
        device_value: Any = device.get(discovery_key)
        if device_value is not None and conf_key not in data_updates and (current_data.get(conf_key) != device_value):
            data_updates[conf_key] = device_value

@callback
def async_update_entry_from_discovery(hass, entry, device, model_num, allow_update_mac):
    """Update a config entry from a flux_led discovery."""
    data_updates: Dict[str, Any] = {}
    mac_address: Optional[str] = device.get(ATTR_ID)
    assert mac_address is not None
    updates: Dict[str, Any] = {}
    formatted_mac: str = dr.format_mac(mac_address)
    if not entry.unique_id or (allow_update_mac and entry.unique_id != formatted_mac and mac_matches_by_one(formatted_mac, entry.unique_id)):
        updates['unique_id'] = formatted_mac
    if model_num and entry.data.get(CONF_MODEL_NUM) != model_num:
        data_updates[CONF_MODEL_NUM] = model_num
    async_populate_data_from_discovery(entry.data, data_updates, device)
    if is_ip_address(entry.title):
        updates['title'] = async_name_from_discovery(device, model_num)
    title_matches_name: bool = entry.title == entry.data.get(CONF_NAME)
    if data_updates or title_matches_name:
        updated_data: Dict[str, Any] = {**entry.data, **data_updates}
        if title_matches_name:
            updated_data.pop(CONF_NAME, None)
        updates['data'] = updated_data
    if updates and (not ('title' in updates and entry.state is ConfigEntryState.LOADED)):
        hass.config_entries.async_update_entry(entry, **updates)
        return True
    return False

@callback
def async_get_discovery(hass, host):
    """Check if a device was already discovered via a broadcast discovery."""
    discoveries: List[FluxLEDDiscovery] = hass.data[DOMAIN][FLUX_LED_DISCOVERY]
    for discovery in discoveries:
        if discovery.get(ATTR_IPADDR) == host:
            return discovery
    return None

@callback
def async_clear_discovery_cache(hass, host):
    """Clear the host from the discovery cache."""
    domain_data: Dict[str, Any] = hass.data[DOMAIN]
    discoveries: List[FluxLEDDiscovery] = domain_data.get(FLUX_LED_DISCOVERY, [])
    domain_data[FLUX_LED_DISCOVERY] = [discovery for discovery in discoveries if discovery.get(ATTR_IPADDR) != host]

async def async_discover_devices(hass: HomeAssistant, timeout: int, address: Optional[str]=None) -> List[FluxLEDDiscovery]:
    """Discover flux led devices."""
    if address:
        targets: List[str] = [address]
    else:
        broadcast_addresses: List[str] = [str(broadcast_address) for broadcast_address in await network.async_get_ipv4_broadcast_addresses(hass)]
        targets = broadcast_addresses
    scanner: AIOBulbScanner = AIOBulbScanner()
    scan_tasks = [create_eager_task(scanner.async_scan(timeout=timeout, address=target_address)) for target_address in targets]
    scanned_results: List[Union[FluxLEDDiscovery, Exception, None]] = await asyncio.gather(*scan_tasks, return_exceptions=True)
    for idx, discovered in enumerate(scanned_results):
        if isinstance(discovered, Exception):
            _LOGGER.debug('Scanning %s failed with error: %s', targets[idx], discovered)
            continue
    if not address:
        bulb_info: List[FluxLEDDiscovery] = scanner.getBulbInfo()
        return bulb_info
    filtered_devices: List[FluxLEDDiscovery] = [device for device in scanner.getBulbInfo() if device.get(ATTR_IPADDR) == address]
    return filtered_devices

async def async_discover_device(hass: HomeAssistant, host: str) -> Optional[FluxLEDDiscovery]:
    """Direct discovery at a single ip instead of broadcast."""
    discovered_devices: List[FluxLEDDiscovery] = await async_discover_devices(hass, DIRECTED_DISCOVERY_TIMEOUT, host)
    for device in discovered_devices:
        if device.get(ATTR_IPADDR) == host:
            return device
    return None

@callback
def async_trigger_discovery(hass, discovered_devices):
    """Trigger config flows for discovered devices."""
    for device in discovered_devices:
        device_data: Dict[str, Any] = {**device}
        discovery_flow.async_create_flow(hass, DOMAIN, context={'source': SOURCE_INTEGRATION_DISCOVERY}, data=device_data)
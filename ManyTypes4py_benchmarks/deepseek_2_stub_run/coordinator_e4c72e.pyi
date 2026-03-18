```python
from __future__ import annotations
from datetime import datetime
import logging
from typing import Any, Set, Dict, Optional, Callable
from ibeacon_ble import iBeaconAdvertisement, iBeaconParser
from homeassistant.components.bluetooth.match import BluetoothCallbackMatcher
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant
from homeassistant.helpers.device_registry import DeviceRegistry

_LOGGER: logging.Logger = ...
CONF_ALLOW_NAMELESS_UUIDS: str = ...
CONF_IGNORE_ADDRESSES: str = ...
CONF_IGNORE_UUIDS: str = ...
DOMAIN: str = ...
MAX_IDS: int = ...
MAX_IDS_PER_UUID: int = ...
MIN_SEEN_TRANSIENT_NEW: int = ...
SIGNAL_IBEACON_DEVICE_NEW: str = ...
SIGNAL_IBEACON_DEVICE_SEEN: str = ...
SIGNAL_IBEACON_DEVICE_UNAVAILABLE: str = ...
UNAVAILABLE_TIMEOUT: float = ...
UPDATE_INTERVAL: Any = ...

def signal_unavailable(unique_id: str) -> str: ...
def signal_seen(unique_id: str) -> str: ...
def make_short_address(address: str) -> str: ...

def async_name(
    service_info: Any,
    ibeacon_advertisement: iBeaconAdvertisement,
    unique_address: bool = False
) -> str: ...

def _async_dispatch_update(
    hass: HomeAssistant,
    device_id: str,
    service_info: Any,
    ibeacon_advertisement: iBeaconAdvertisement,
    new: bool,
    unique_address: bool
) -> None: ...

class IBeaconCoordinator:
    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        registry: DeviceRegistry
    ) -> None: ...
    
    @callback
    def async_device_id_seen(self, device_id: str) -> bool: ...
    
    @callback
    def _async_handle_unavailable(self, service_info: Any) -> None: ...
    
    @callback
    def _async_cancel_unavailable_tracker(self, address: str) -> None: ...
    
    @callback
    def _async_ignore_uuid(self, uuid: str) -> None: ...
    
    @callback
    def _async_ignore_address(self, address: str) -> None: ...
    
    @callback
    def _async_purge_untrackable_entities(self, unique_ids: Set[str]) -> None: ...
    
    @callback
    def _async_convert_random_mac_tracking(
        self,
        group_id: str,
        service_info: Any,
        ibeacon_advertisement: iBeaconAdvertisement
    ) -> None: ...
    
    def _async_track_ibeacon_with_unique_address(
        self,
        address: str,
        group_id: str,
        unique_id: str
    ) -> None: ...
    
    @callback
    def _async_update_ibeacon(self, service_info: Any, change: Any) -> None: ...
    
    @callback
    def _async_update_ibeacon_with_random_mac(
        self,
        group_id: str,
        service_info: Any,
        ibeacon_advertisement: iBeaconAdvertisement
    ) -> None: ...
    
    @callback
    def _async_update_ibeacon_with_unique_address(
        self,
        group_id: str,
        service_info: Any,
        ibeacon_advertisement: iBeaconAdvertisement
    ) -> None: ...
    
    @callback
    def _async_stop(self) -> None: ...
    
    @callback
    def _async_check_unavailable_groups_with_random_macs(self) -> None: ...
    
    @callback
    def _async_update_rssi_and_transients(self) -> None: ...
    
    async def async_config_entry_updated(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry
    ) -> None: ...
    
    @callback
    def _async_update(self, _now: Any) -> None: ...
    
    @callback
    def _async_restore_from_registry(self) -> None: ...
    
    async def async_start(self) -> None: ...
```
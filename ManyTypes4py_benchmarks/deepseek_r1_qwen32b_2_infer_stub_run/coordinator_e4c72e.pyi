"""Tracking for iBeacon devices."""
from __future__ import annotations
from datetime import datetime
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from homeassistant.components import bluetooth
from homeassistant.components.bluetooth.match import BluetoothCallbackMatcher
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import (
    CALLBACK_TYPE,
    HomeAssistant,
    ConfigType,
    ServiceInfo,
    BluetoothScanningMode,
    BluetoothChange,
)
from homeassistant.helpers.device_registry import DeviceRegistry
from homeassistant.helpers.dispatcher import Signal
from homeassistant.helpers.event import (
    async_track_time_interval,
    TrackTimeIntervalCallback,
)
from ibeacon_ble import (
    APPLE_MFR_ID,
    IBEACON_FIRST_BYTE,
    IBEACON_SECOND_BYTE,
    iBeaconAdvertisement,
    iBeaconParser,
)
from .const import (
    CONF_ALLOW_NAMELESS_UUIDS,
    CONF_IGNORE_ADDRESSES,
    CONF_IGNORE_UUIDS,
    DOMAIN,
    MAX_IDS,
    MAX_IDS_PER_UUID,
    MIN_SEEN_TRANSIENT_NEW,
    SIGNAL_IBEACON_DEVICE_NEW,
    SIGNAL_IBEACON_DEVICE_SEEN,
    SIGNAL_IBEACON_DEVICE_UNAVAILABLE,
    UNAVAILABLE_TIMEOUT,
    UPDATE_INTERVAL,
)

_LOGGER = logging.getLogger(__name__)
MONOTONIC_TIME = time.monotonic

def signal_unavailable(unique_id: str) -> str: ...
def signal_seen(unique_id: str) -> str: ...
def make_short_address(address: str) -> str: ...

@callback
async def async_name(service_info: Any, ibeacon_advertisement: Any, unique_address: bool = False) -> str: ...

@callback
def _async_dispatch_update(
    hass: HomeAssistant,
    device_id: str,
    service_info: Any,
    ibeacon_advertisement: Any,
    new: bool,
    unique_address: bool,
) -> None: ...

class IBeaconCoordinator:
    """Set up the iBeacon Coordinator."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        registry: DeviceRegistry,
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
        ibeacon_advertisement: Any,
    ) -> None: ...

    @callback
    def _async_track_ibeacon_with_unique_address(
        self,
        address: str,
        group_id: str,
        unique_id: str,
    ) -> None: ...

    @callback
    def _async_update_ibeacon(
        self,
        service_info: Any,
        change: BluetoothChange,
    ) -> None: ...

    @callback
    def _async_update_ibeacon_with_random_mac(
        self,
        group_id: str,
        service_info: Any,
        ibeacon_advertisement: Any,
    ) -> None: ...

    @callback
    def _async_update_ibeacon_with_unique_address(
        self,
        group_id: str,
        service_info: Any,
        ibeacon_advertisement: Any,
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
        config_entry: ConfigEntry,
    ) -> None: ...

    @callback
    def _async_update(self, _now: datetime) -> None: ...

    @callback
    def _async_restore_from_registry(self) -> None: ...

    async def async_start(self) -> None: ...

    # Attribute types inferred from usage
    hass: HomeAssistant
    _entry: ConfigEntry
    _dev_reg: DeviceRegistry
    _ibeacon_parser: iBeaconParser
    _ignore_addresses: Set[str]
    _ignore_uuids: Set[str]
    _last_ibeacon_advertisement_by_unique_id: Dict[str, iBeaconAdvertisement]
    _transient_seen_count: Dict[str, int]
    _group_ids_by_address: Dict[str, Set[str]]
    _unique_ids_by_address: Dict[str, Set[str]]
    _unique_ids_by_group_id: Dict[str, Set[str]]
    _addresses_by_group_id: Dict[str, Set[str]]
    _unavailable_trackers: Dict[str, Callable[[], None]]
    _group_ids_random_macs: Set[str]
    _last_seen_by_group_id: Dict[str, Any]
    _unavailable_group_ids: Set[str]
    _major_minor_by_uuid: Dict[str, Set[Tuple[int, int]]]
    _allow_nameless_uuids: Set[str]
    _ignored_nameless_by_uuid: Dict[str, Set[str]]
from __future__ import annotations

from datetime import datetime
import logging
from typing import Any

from ibeacon_ble import iBeaconAdvertisement, iBeaconParser
from homeassistant.components.bluetooth import BluetoothChange, BluetoothServiceInfoBleak
from homeassistant.components.bluetooth.match import BluetoothCallbackMatcher
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceRegistry

_LOGGER: logging.Logger
MONOTONIC_TIME: type[float]

def signal_unavailable(unique_id: str) -> str: ...
def signal_seen(unique_id: str) -> str: ...
def make_short_address(address: str) -> str: ...

@callback
def async_name(
    service_info: BluetoothServiceInfoBleak,
    ibeacon_advertisement: iBeaconAdvertisement,
    unique_address: bool = False,
) -> str: ...

@callback
def _async_dispatch_update(
    hass: HomeAssistant,
    device_id: str,
    service_info: BluetoothServiceInfoBleak,
    ibeacon_advertisement: iBeaconAdvertisement,
    new: bool,
    unique_address: bool,
) -> None: ...

class IBeaconCoordinator:
    hass: HomeAssistant
    _entry: ConfigEntry
    _dev_reg: DeviceRegistry
    _ibeacon_parser: iBeaconParser
    _ignore_addresses: set[str]
    _ignore_uuids: set[str]
    _last_ibeacon_advertisement_by_unique_id: dict[str, iBeaconAdvertisement]
    _transient_seen_count: dict[str, int]
    _group_ids_by_address: dict[str, set[str]]
    _unique_ids_by_address: dict[str, set[str]]
    _unique_ids_by_group_id: dict[str, set[str]]
    _addresses_by_group_id: dict[str, set[str]]
    _unavailable_trackers: dict[str, CALLBACK_TYPE]
    _group_ids_random_macs: set[str]
    _last_seen_by_group_id: dict[str, BluetoothServiceInfoBleak]
    _unavailable_group_ids: set[str]
    _major_minor_by_uuid: dict[str, set[tuple[int, int]]]
    _allow_nameless_uuids: set[str]
    _ignored_nameless_by_uuid: dict[str, set[str]]

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, registry: DeviceRegistry
    ) -> None: ...

    @callback
    def async_device_id_seen(self, device_id: str) -> bool: ...

    @callback
    def _async_handle_unavailable(
        self, service_info: BluetoothServiceInfoBleak
    ) -> None: ...

    @callback
    def _async_cancel_unavailable_tracker(self, address: str) -> None: ...

    @callback
    def _async_ignore_uuid(self, uuid: str) -> None: ...

    @callback
    def _async_ignore_address(self, address: str) -> None: ...

    @callback
    def _async_purge_untrackable_entities(self, unique_ids: set[str]) -> None: ...

    @callback
    def _async_convert_random_mac_tracking(
        self,
        group_id: str,
        service_info: BluetoothServiceInfoBleak,
        ibeacon_advertisement: iBeaconAdvertisement,
    ) -> None: ...

    def _async_track_ibeacon_with_unique_address(
        self, address: str, group_id: str, unique_id: str
    ) -> None: ...

    @callback
    def _async_update_ibeacon(
        self, service_info: BluetoothServiceInfoBleak, change: BluetoothChange
    ) -> None: ...

    @callback
    def _async_update_ibeacon_with_random_mac(
        self,
        group_id: str,
        service_info: BluetoothServiceInfoBleak,
        ibeacon_advertisement: iBeaconAdvertisement,
    ) -> None: ...

    @callback
    def _async_update_ibeacon_with_unique_address(
        self,
        group_id: str,
        service_info: BluetoothServiceInfoBleak,
        ibeacon_advertisement: iBeaconAdvertisement,
    ) -> None: ...

    @callback
    def _async_stop(self) -> None: ...

    @callback
    def _async_check_unavailable_groups_with_random_macs(self) -> None: ...

    @callback
    def _async_update_rssi_and_transients(self) -> None: ...

    async def async_config_entry_updated(
        self, hass: HomeAssistant, config_entry: ConfigEntry
    ) -> None: ...

    @callback
    def _async_update(self, _now: datetime) -> None: ...

    @callback
    def _async_restore_from_registry(self) -> None: ...

    async def async_start(self) -> None: ...
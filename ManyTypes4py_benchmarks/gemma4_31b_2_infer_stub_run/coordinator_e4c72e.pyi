"""Tracking for iBeacon devices."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Set, Tuple
from ibeacon_ble import iBeaconAdvertisement, iBeaconParser
from homeassistant.components import bluetooth
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceRegistry

def signal_unavailable(unique_id: str) -> str:
    """Signal for the unique_id going unavailable."""
    ...

def signal_seen(unique_id: str) -> str:
    """Signal for the unique_id being seen."""
    ...

def make_short_address(address: str) -> str:
    """Convert a Bluetooth address to a short address."""
    ...

def async_name(
    service_info: bluetooth.BLEDevice,
    ibeacon_advertisement: iBeaconAdvertisement,
    unique_address: bool = False,
) -> str:
    """Return a name for the device."""
    ...

def _async_dispatch_update(
    hass: HomeAssistant,
    device_id: str,
    service_info: bluetooth.BLEDevice,
    ibeacon_advertisement: iBeaconAdvertisement,
    new: bool,
    unique_address: bool,
) -> None:
    """Dispatch an update."""
    ...

class IBeaconCoordinator:
    """Set up the iBeacon Coordinator."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, registry: DeviceRegistry) -> None:
        """Initialize the Coordinator."""
        self.hass: HomeAssistant
        self._entry: ConfigEntry
        self._dev_reg: DeviceRegistry
        self._ibeacon_parser: iBeaconParser
        self._ignore_addresses: Set[str]
        self._ignore_uuids: Set[str]
        self._last_ibeacon_advertisement_by_unique_id: Dict[str, iBeaconAdvertisement]
        self._transient_seen_count: Dict[str, int]
        self._group_ids_by_address: Dict[str, Set[str]]
        self._unique_ids_by_address: Dict[str, Set[str]]
        self._unique_ids_by_group_id: Dict[str, Set[str]]
        self._addresses_by_group_id: Dict[str, Set[str]]
        self._unavailable_trackers: Dict[str, Callable[[], None]]
        self._group_ids_random_macs: Set[str]
        self._last_seen_by_group_id: Dict[str, bluetooth.BLEDevice]
        self._unavailable_group_ids: Set[str]
        self._major_minor_by_uuid: Dict[str, Set[Tuple[int, int]]]
        self._allow_nameless_uuids: Set[str]
        self._ignored_nameless_by_uuid: Dict[str, Set[str]]
        ...

    def async_device_id_seen(self, device_id: str) -> bool:
        """Return True if the device_id has been seen since boot."""
        ...

    def _async_handle_unavailable(self, service_info: bluetooth.BLEDevice) -> None:
        """Handle unavailable devices."""
        ...

    def _async_cancel_unavailable_tracker(self, address: str) -> None:
        """Cancel unavailable tracking for an address."""
        ...

    def _async_ignore_uuid(self, uuid: str) -> None:
        """Ignore an UUID that does not follow the spec and any entities created by it."""
        ...

    def _async_ignore_address(self, address: str) -> None:
        """Ignore an address that does not follow the spec and any entities created by it."""
        ...

    def _async_purge_untrackable_entities(self, unique_ids: Set[str]) -> None:
        """Remove entities that are no longer trackable."""
        ...

    def _async_convert_random_mac_tracking(
        self, group_id: str, service_info: bluetooth.BLEDevice, ibeacon_advertisement: iBeaconAdvertisement
    ) -> None:
        """Switch to random mac tracking method when a group is using rotating mac addresses."""
        ...

    def _async_track_ibeacon_with_unique_address(self, address: str, group_id: str, unique_id: str) -> None:
        """Track an iBeacon with a unique address."""
        ...

    def _async_update_ibeacon(self, service_info: bluetooth.BLEDevice, change: bluetooth.BluetoothChange) -> None:
        """Update from a bluetooth callback."""
        ...

    def _async_update_ibeacon_with_random_mac(
        self, group_id: str, service_info: bluetooth.BLEDevice, ibeacon_advertisement: iBeaconAdvertisement
    ) -> None:
        """Update iBeacons with random mac addresses."""
        ...

    def _async_update_ibeacon_with_unique_address(
        self, group_id: str, service_info: bluetooth.BLEDevice, ibeacon_advertisement: iBeaconAdvertisement
    ) -> None:
        """Update iBeacons with unique addresses."""
        ...

    def _async_stop(self) -> None:
        """Stop the Coordinator."""
        ...

    def _async_check_unavailable_groups_with_random_macs(self) -> None:
        """Check for random mac groups that have not been seen in a while and mark them as unavailable."""
        ...

    def _async_update_rssi_and_transients(self) -> None:
        """Check to see if the rssi has changed and update any devices."""
        ...

    async def async_config_entry_updated(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Restore ignored nameless beacons when the allowlist is updated."""
        ...

    def _async_update(self, _now: float) -> None:
        """Update the Coordinator."""
        ...

    def _async_restore_from_registry(self) -> None:
        """Restore the state of the Coordinator from the device registry."""
        ...

    async def async_start(self) -> None:
        """Start the Coordinator."""
        ...
```python
"""Tests for the Bluetooth integration."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
import time
from typing import Any
from unittest.mock import MagicMock

from bleak import BleakClient
from bleak.backends.scanner import AdvertisementData, BLEDevice
from bluetooth_adapters import DEFAULT_ADDRESS
from habluetooth import BaseHaScanner, get_manager
from homeassistant.components.bluetooth import (
    DOMAIN,
    MONOTONIC_TIME,
    SOURCE_LOCAL,
    BaseHaRemoteScanner,
    BluetoothServiceInfo,
    BluetoothServiceInfoBleak,
    async_get_advertisement_callback,
)
from homeassistant.components.bluetooth.manager import HomeAssistantBluetoothManager
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from tests.common import MockConfigEntry

__all__: tuple[str, ...] = (
    "MockBleakClient",
    "generate_advertisement_data",
    "generate_ble_device",
    "inject_advertisement",
    "inject_advertisement_with_source",
    "inject_advertisement_with_time_and_source",
    "inject_advertisement_with_time_and_source_connectable",
    "inject_bluetooth_service_info",
    "patch_all_discovered_devices",
    "patch_bluetooth_time",
    "patch_discovered_devices",
)

ADVERTISEMENT_DATA_DEFAULTS: dict[str, Any] = ...
BLE_DEVICE_DEFAULTS: dict[str, Any] = ...
HCI0_SOURCE_ADDRESS: str = ...
HCI1_SOURCE_ADDRESS: str = ...
NON_CONNECTABLE_REMOTE_SOURCE_ADDRESS: str = ...


@contextmanager
def patch_bluetooth_time(mock_time: float) -> Iterator[None]:
    """Patch the bluetooth time."""
    ...


def generate_advertisement_data(**kwargs: Any) -> AdvertisementData:
    """Generate advertisement data with defaults."""
    ...


def generate_ble_device(
    address: str | None = None,
    name: str | None = None,
    details: Any = None,
    rssi: int | None = None,
    **kwargs: Any,
) -> BLEDevice:
    """Generate a BLEDevice with defaults."""
    ...


def _get_manager() -> HomeAssistantBluetoothManager:
    """Return the bluetooth manager."""
    ...


def inject_advertisement(
    hass: HomeAssistant, device: BLEDevice, adv: AdvertisementData
) -> None:
    """Inject an advertisement into the manager."""
    ...


def inject_advertisement_with_source(
    hass: HomeAssistant, device: BLEDevice, adv: AdvertisementData, source: str
) -> None:
    """Inject an advertisement into the manager from a specific source."""
    ...


def inject_advertisement_with_time_and_source(
    hass: HomeAssistant,
    device: BLEDevice,
    adv: AdvertisementData,
    time: float,
    source: str,
) -> None:
    """Inject an advertisement into the manager from a specific source at a time."""
    ...


def inject_advertisement_with_time_and_source_connectable(
    hass: HomeAssistant,
    device: BLEDevice,
    adv: AdvertisementData,
    time: float,
    source: str,
    connectable: bool,
) -> None:
    """Inject an advertisement into the manager from a specific source at a time and connectable status."""
    ...


def inject_bluetooth_service_info_bleak(
    hass: HomeAssistant, info: BluetoothServiceInfoBleak
) -> None:
    """Inject an advertisement into the manager with connectable status."""
    ...


def inject_bluetooth_service_info(
    hass: HomeAssistant, info: BluetoothServiceInfo
) -> None:
    """Inject a BluetoothServiceInfo into the manager."""
    ...


@contextmanager
def patch_all_discovered_devices(mock_discovered: Iterable[BLEDevice]) -> Iterator[None]:
    """Mock all the discovered devices from all the scanners."""
    ...


@contextmanager
def patch_discovered_devices(mock_discovered: Iterable[BLEDevice]) -> Iterator[None]:
    """Mock the combined best path to discovered devices from all the scanners."""
    ...


async def async_setup_with_default_adapter(hass: HomeAssistant) -> MockConfigEntry:
    """Set up the Bluetooth integration with a default adapter."""
    ...


async def async_setup_with_one_adapter(hass: HomeAssistant) -> MockConfigEntry:
    """Set up the Bluetooth integration with one adapter."""
    ...


async def _async_setup_with_adapter(
    hass: HomeAssistant, address: str
) -> MockConfigEntry:
    """Set up the Bluetooth integration with any adapter."""
    ...


class MockBleakClient(BleakClient):
    """Mock bleak client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Mock init."""
        ...

    @property
    def is_connected(self) -> bool:
        """Mock connected."""
        ...

    async def connect(self, *args: Any, **kwargs: Any) -> bool:
        """Mock connect."""
        ...

    async def disconnect(self, *args: Any, **kwargs: Any) -> None:
        """Mock disconnect."""
        ...

    async def get_services(self, *args: Any, **kwargs: Any) -> list[Any]:
        """Mock get_services."""
        ...

    async def clear_cache(self, *args: Any, **kwargs: Any) -> bool:
        """Mock clear_cache."""
        ...


class FakeScannerMixin:
    def get_discovered_device_advertisement_data(
        self, address: str
    ) -> tuple[BLEDevice, MagicMock] | None:
        """Return the advertisement data for a discovered device."""
        ...

    @property
    def discovered_addresses(self) -> dict[str, tuple[BLEDevice, MagicMock]]:
        """Return an iterable of discovered devices."""
        ...


class FakeScanner(FakeScannerMixin, BaseHaScanner):
    """Fake scanner."""

    @property
    def discovered_devices(self) -> list[Any]:
        """Return a list of discovered devices."""
        ...

    @property
    def discovered_devices_and_advertisement_data(
        self,
    ) -> dict[str, tuple[BLEDevice, MagicMock]]:
        """Return a list of discovered devices and their advertisement data."""
        ...


class FakeRemoteScanner(BaseHaRemoteScanner):
    """Fake remote scanner."""

    def inject_advertisement(
        self,
        device: BLEDevice,
        advertisement_data: AdvertisementData,
        now: float | None = None,
    ) -> None:
        """Inject an advertisement."""
        ...
```
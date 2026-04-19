from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, Optional

from bleak import BleakClient
from bleak.backends.scanner import AdvertisementData, BLEDevice
from homeassistant.components.bluetooth import (
    BaseHaRemoteScanner,
    BluetoothServiceInfo,
    BluetoothServiceInfoBleak,
)
from homeassistant.components.bluetooth.manager import HomeAssistantBluetoothManager
from homeassistant.core import HomeAssistant
from tests.common import MockConfigEntry

__all__: tuple[str, ...] = ...

ADVERTISEMENT_DATA_DEFAULTS: dict[str, Any] = ...
BLE_DEVICE_DEFAULTS: dict[str, Any] = ...
HCI0_SOURCE_ADDRESS: str = ...
HCI1_SOURCE_ADDRESS: str = ...
NON_CONNECTABLE_REMOTE_SOURCE_ADDRESS: str = ...

@contextmanager
def patch_bluetooth_time(mock_time: float) -> Iterator[None]: ...

def generate_advertisement_data(**kwargs: Any) -> AdvertisementData: ...

def generate_ble_device(
    address: Optional[str] = ...,
    name: Optional[str] = ...,
    details: Optional[Any] = ...,
    rssi: Optional[int] = ...,
    **kwargs: Any,
) -> BLEDevice: ...

def _get_manager() -> HomeAssistantBluetoothManager: ...

def inject_advertisement(hass: HomeAssistant, device: BLEDevice, adv: AdvertisementData) -> None: ...

def inject_advertisement_with_source(
    hass: HomeAssistant, device: BLEDevice, adv: AdvertisementData, source: str
) -> None: ...

def inject_advertisement_with_time_and_source(
    hass: HomeAssistant, device: BLEDevice, adv: AdvertisementData, time: float, source: str
) -> None: ...

def inject_advertisement_with_time_and_source_connectable(
    hass: HomeAssistant,
    device: BLEDevice,
    adv: AdvertisementData,
    time: float,
    source: str,
    connectable: bool,
) -> None: ...

def inject_bluetooth_service_info_bleak(
    hass: HomeAssistant, info: BluetoothServiceInfoBleak
) -> None: ...

def inject_bluetooth_service_info(
    hass: HomeAssistant, info: BluetoothServiceInfo
) -> None: ...

@contextmanager
def patch_all_discovered_devices(mock_discovered: Iterable[BLEDevice]) -> Iterator[None]: ...

@contextmanager
def patch_discovered_devices(mock_discovered: Iterable[BLEDevice]) -> Iterator[None]: ...

async def async_setup_with_default_adapter(hass: HomeAssistant) -> MockConfigEntry: ...

async def async_setup_with_one_adapter(hass: HomeAssistant) -> MockConfigEntry: ...

async def _async_setup_with_adapter(hass: HomeAssistant, address: str) -> MockConfigEntry: ...

class MockBleakClient(BleakClient):
    _device_path: str

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @property
    def is_connected(self) -> bool: ...
    async def connect(self, *args: Any, **kwargs: Any) -> bool: ...
    async def disconnect(self, *args: Any, **kwargs: Any) -> None: ...
    async def get_services(self, *args: Any, **kwargs: Any) -> list[Any]: ...
    async def clear_cache(self, *args: Any, **kwargs: Any) -> bool: ...

class FakeScannerMixin:
    def get_discovered_device_advertisement_data(
        self, address: str
    ) -> Optional[tuple[BLEDevice, Any]]: ...
    @property
    def discovered_addresses(self) -> Iterable[str]: ...
    @property
    def discovered_devices_and_advertisement_data(self) -> dict[str, tuple[BLEDevice, Any]]: ...

class FakeScanner(FakeScannerMixin):
    @property
    def discovered_devices(self) -> list[BLEDevice]: ...
    @property
    def discovered_devices_and_advertisement_data(self) -> dict[str, tuple[BLEDevice, Any]]: ...

class FakeRemoteScanner(BaseHaRemoteScanner):
    def inject_advertisement(
        self,
        device: BLEDevice,
        advertisement_data: AdvertisementData,
        now: Optional[float] = ...,
    ) -> None: ...
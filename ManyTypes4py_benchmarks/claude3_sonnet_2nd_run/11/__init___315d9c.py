"""Tests for the Bluetooth integration."""
from collections.abc import Iterable
from contextlib import contextmanager
import itertools
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
from unittest.mock import MagicMock, patch
from bleak import BleakClient
from bleak.backends.scanner import AdvertisementData, BLEDevice
from bluetooth_adapters import DEFAULT_ADDRESS
from habluetooth import BaseHaScanner, get_manager
from homeassistant.components.bluetooth import DOMAIN, MONOTONIC_TIME, SOURCE_LOCAL, BaseHaRemoteScanner, BluetoothServiceInfo, BluetoothServiceInfoBleak, async_get_advertisement_callback
from homeassistant.components.bluetooth.manager import HomeAssistantBluetoothManager
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from tests.common import MockConfigEntry

__all__ = ('MockBleakClient', 'generate_advertisement_data', 'generate_ble_device', 'inject_advertisement', 'inject_advertisement_with_source', 'inject_advertisement_with_time_and_source', 'inject_advertisement_with_time_and_source_connectable', 'inject_bluetooth_service_info', 'patch_all_discovered_devices', 'patch_bluetooth_time', 'patch_discovered_devices')

ADVERTISEMENT_DATA_DEFAULTS: Dict[str, Any] = {'local_name': '', 'manufacturer_data': {}, 'service_data': {}, 'service_uuids': [], 'rssi': -127, 'platform_data': ((),), 'tx_power': -127}
BLE_DEVICE_DEFAULTS: Dict[str, Any] = {'name': None, 'rssi': -127, 'details': None}
HCI0_SOURCE_ADDRESS: str = 'AA:BB:CC:DD:EE:00'
HCI1_SOURCE_ADDRESS: str = 'AA:BB:CC:DD:EE:11'
NON_CONNECTABLE_REMOTE_SOURCE_ADDRESS: str = 'AA:BB:CC:DD:EE:FF'

@contextmanager
def patch_bluetooth_time(mock_time: float) -> Any:
    """Patch the bluetooth time."""
    with patch('homeassistant.components.bluetooth.MONOTONIC_TIME', return_value=mock_time), patch('habluetooth.base_scanner.monotonic_time_coarse', return_value=mock_time), patch('habluetooth.manager.monotonic_time_coarse', return_value=mock_time), patch('habluetooth.scanner.monotonic_time_coarse', return_value=mock_time):
        yield

def generate_advertisement_data(**kwargs: Any) -> AdvertisementData:
    """Generate advertisement data with defaults."""
    new: Dict[str, Any] = kwargs.copy()
    for key, value in ADVERTISEMENT_DATA_DEFAULTS.items():
        new.setdefault(key, value)
    return AdvertisementData(**new)

def generate_ble_device(address: Optional[str] = None, name: Optional[str] = None, details: Optional[Dict[str, Any]] = None, rssi: Optional[int] = None, **kwargs: Any) -> BLEDevice:
    """Generate a BLEDevice with defaults."""
    new: Dict[str, Any] = kwargs.copy()
    if address is not None:
        new['address'] = address
    if name is not None:
        new['name'] = name
    if details is not None:
        new['details'] = details
    if rssi is not None:
        new['rssi'] = rssi
    for key, value in BLE_DEVICE_DEFAULTS.items():
        new.setdefault(key, value)
    return BLEDevice(**new)

def _get_manager() -> HomeAssistantBluetoothManager:
    """Return the bluetooth manager."""
    manager = get_manager()
    return cast(HomeAssistantBluetoothManager, manager)

def inject_advertisement(hass: HomeAssistant, device: BLEDevice, adv: AdvertisementData) -> None:
    """Inject an advertisement into the manager."""
    return inject_advertisement_with_source(hass, device, adv, SOURCE_LOCAL)

def inject_advertisement_with_source(hass: HomeAssistant, device: BLEDevice, adv: AdvertisementData, source: str) -> None:
    """Inject an advertisement into the manager from a specific source."""
    inject_advertisement_with_time_and_source(hass, device, adv, time.monotonic(), source)

def inject_advertisement_with_time_and_source(hass: HomeAssistant, device: BLEDevice, adv: AdvertisementData, time: float, source: str) -> None:
    """Inject an advertisement into the manager from a specific source at a time."""
    inject_advertisement_with_time_and_source_connectable(hass, device, adv, time, source, True)

def inject_advertisement_with_time_and_source_connectable(hass: HomeAssistant, device: BLEDevice, adv: AdvertisementData, time: float, source: str, connectable: bool) -> None:
    """Inject an advertisement into the manager from a specific source at a time and connectable status."""
    async_get_advertisement_callback(hass)(BluetoothServiceInfoBleak(name=adv.local_name or device.name or device.address, address=device.address, rssi=adv.rssi, manufacturer_data=adv.manufacturer_data, service_data=adv.service_data, service_uuids=adv.service_uuids, source=source, device=device, advertisement=adv, connectable=connectable, time=time, tx_power=adv.tx_power))

def inject_bluetooth_service_info_bleak(hass: HomeAssistant, info: BluetoothServiceInfo) -> None:
    """Inject an advertisement into the manager with connectable status."""
    advertisement_data = generate_advertisement_data(local_name=None if info.name == '' else info.name, manufacturer_data=info.manufacturer_data, service_data=info.service_data, service_uuids=info.service_uuids, rssi=info.rssi)
    device = generate_ble_device(address=info.address, name=info.name, details={})
    inject_advertisement_with_time_and_source_connectable(hass, device, advertisement_data, info.time, SOURCE_LOCAL, connectable=info.connectable)

def inject_bluetooth_service_info(hass: HomeAssistant, info: BluetoothServiceInfo) -> None:
    """Inject a BluetoothServiceInfo into the manager."""
    advertisement_data = generate_advertisement_data(local_name=None if info.name == '' else info.name, manufacturer_data=info.manufacturer_data, service_data=info.service_data, service_uuids=info.service_uuids, rssi=info.rssi)
    device = generate_ble_device(address=info.address, name=info.name, details={})
    inject_advertisement(hass, device, advertisement_data)

@contextmanager
def patch_all_discovered_devices(mock_discovered: List[BLEDevice]) -> Any:
    """Mock all the discovered devices from all the scanners."""
    manager = _get_manager()
    original_history: Dict[Any, Dict[str, Tuple[BLEDevice, Any]]] = {}
    scanners = list(itertools.chain(manager._connectable_scanners, manager._non_connectable_scanners))
    for scanner in scanners:
        data = scanner.discovered_devices_and_advertisement_data
        original_history[scanner] = data.copy()
        data.clear()
    if scanners:
        data = scanners[0].discovered_devices_and_advertisement_data
        data.clear()
        data.update({device.address: (device, MagicMock()) for device in mock_discovered})
    yield
    for scanner in scanners:
        data = scanner.discovered_devices_and_advertisement_data
        data.clear()
        data.update(original_history[scanner])

@contextmanager
def patch_discovered_devices(mock_discovered: List[BLEDevice]) -> Any:
    """Mock the combined best path to discovered devices from all the scanners."""
    manager = _get_manager()
    original_all_history = manager._all_history
    original_connectable_history = manager._connectable_history
    manager._connectable_history = {}
    manager._all_history = {device.address: MagicMock(device=device) for device in mock_discovered}
    yield
    manager._all_history = original_all_history
    manager._connectable_history = original_connectable_history

async def async_setup_with_default_adapter(hass: HomeAssistant) -> MockConfigEntry:
    """Set up the Bluetooth integration with a default adapter."""
    return await _async_setup_with_adapter(hass, DEFAULT_ADDRESS)

async def async_setup_with_one_adapter(hass: HomeAssistant) -> MockConfigEntry:
    """Set up the Bluetooth integration with one adapter."""
    return await _async_setup_with_adapter(hass, '00:00:00:00:00:01')

async def _async_setup_with_adapter(hass: HomeAssistant, address: str) -> MockConfigEntry:
    """Set up the Bluetooth integration with any adapter."""
    entry = MockConfigEntry(domain='bluetooth', unique_id=address)
    entry.add_to_hass(hass)
    assert await async_setup_component(hass, DOMAIN, {DOMAIN: {}})
    await hass.async_block_till_done()
    return entry

class MockBleakClient(BleakClient):
    """Mock bleak client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Mock init."""
        super().__init__(*args, **kwargs)
        self._device_path = '/dev/test'

    @property
    def is_connected(self) -> bool:
        """Mock connected."""
        return True

    async def connect(self, *args: Any, **kwargs: Any) -> bool:
        """Mock connect."""
        return True

    async def disconnect(self, *args: Any, **kwargs: Any) -> None:
        """Mock disconnect."""

    async def get_services(self, *args: Any, **kwargs: Any) -> List[Any]:
        """Mock get_services."""
        return []

    async def clear_cache(self, *args: Any, **kwargs: Any) -> bool:
        """Mock clear_cache."""
        return True

class FakeScannerMixin:

    def get_discovered_device_advertisement_data(self, address: str) -> Optional[Tuple[BLEDevice, AdvertisementData]]:
        """Return the advertisement data for a discovered device."""
        return self.discovered_devices_and_advertisement_data.get(address)

    @property
    def discovered_addresses(self) -> Iterable[str]:
        """Return an iterable of discovered devices."""
        return self.discovered_devices_and_advertisement_data

class FakeScanner(FakeScannerMixin, BaseHaScanner):
    """Fake scanner."""

    @property
    def discovered_devices(self) -> List[BLEDevice]:
        """Return a list of discovered devices."""
        return []

    @property
    def discovered_devices_and_advertisement_data(self) -> Dict[str, Tuple[BLEDevice, AdvertisementData]]:
        """Return a list of discovered devices and their advertisement data."""
        return {}

class FakeRemoteScanner(BaseHaRemoteScanner):
    """Fake remote scanner."""

    def inject_advertisement(self, device: BLEDevice, advertisement_data: AdvertisementData, now: Optional[float] = None) -> None:
        """Inject an advertisement."""
        self._async_on_advertisement(device.address, advertisement_data.rssi, device.name, advertisement_data.service_uuids, advertisement_data.service_data, advertisement_data.manufacturer_data, advertisement_data.tx_power, {'scanner_specific_data': 'test'}, now or MONOTONIC_TIME())

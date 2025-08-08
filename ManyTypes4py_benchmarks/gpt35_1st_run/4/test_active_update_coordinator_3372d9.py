from __future__ import annotations
import asyncio
from collections.abc import Callable, Coroutine
import logging
from typing import Any, Dict
from unittest.mock import MagicMock
import pytest
from homeassistant.components.bluetooth import DOMAIN, BluetoothChange, BluetoothScanningMode, BluetoothServiceInfoBleak
from homeassistant.components.bluetooth.active_update_coordinator import ActiveBluetoothDataUpdateCoordinator
from homeassistant.core import CoreState, HomeAssistant
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.service_info.bluetooth import BluetoothServiceInfo
from homeassistant.setup import async_setup_component
from . import inject_bluetooth_service_info

_LOGGER: logging.Logger = logging.getLogger(__name__)
GENERIC_BLUETOOTH_SERVICE_INFO: BluetoothServiceInfo = BluetoothServiceInfo(name='Generic', address='aa:bb:cc:dd:ee:ff', rssi=-95, manufacturer_data={1: b'\x01\x01\x01\x01\x01\x01\x01\x01'}, service_data={}, service_uuids=[], source='local')
GENERIC_BLUETOOTH_SERVICE_INFO_2: BluetoothServiceInfo = BluetoothServiceInfo(name='Generic', address='aa:bb:cc:dd:ee:ff', rssi=-95, manufacturer_data={2: b'\x01\x01\x01\x01\x01\x01\x01\x01'}, service_data={}, service_uuids=[], source='local')

class MyCoordinator(ActiveBluetoothDataUpdateCoordinator[Dict[str, Any]]):
    """An example coordinator that subclasses ActiveBluetoothDataUpdateCoordinator."""

    def __init__(self, hass: HomeAssistant, logger: logging.Logger, *, address: str, mode: BluetoothScanningMode, needs_poll_method: Callable, poll_method: Callable = None, poll_debouncer: Debouncer = None, connectable: bool = True) -> None:
        """Initialize the coordinator."""
        self.passive_data: Dict[str, Any] = {}
        super().__init__(hass=hass, logger=logger, address=address, mode=mode, needs_poll_method=needs_poll_method, poll_method=poll_method, poll_debouncer=poll_debouncer, connectable=connectable)

    def _async_handle_bluetooth_event(self, service_info: BluetoothServiceInfo, change: BluetoothChange) -> None:
        """Handle a Bluetooth event."""
        self.passive_data = {'rssi': service_info.rssi}
        super()._async_handle_bluetooth_event(service_info, change)

async def test_basic_usage(hass: HomeAssistant) -> None:
    """Test basic usage of the ActiveBluetoothDataUpdateCoordinator."""
    await async_setup_component(hass, DOMAIN, {DOMAIN: {}})

    def _needs_poll(service_info: BluetoothServiceInfo, seconds_since_last_poll: int) -> bool:
        return True

    async def _poll_method(service_info: BluetoothServiceInfo) -> Dict[str, Any]:
        return {'fake': 'data'}
    coordinator = MyCoordinator(hass=hass, logger=_LOGGER, address='aa:bb:cc:dd:ee:ff', mode=BluetoothScanningMode.ACTIVE, needs_poll_method=_needs_poll, poll_method=_poll_method)
    assert coordinator.available is False
    mock_listener = MagicMock()
    unregister_listener = coordinator.async_add_listener(mock_listener)
    cancel = coordinator.async_start()
    inject_bluetooth_service_info(hass, GENERIC_BLUETOOTH_SERVICE_INFO)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert coordinator.passive_data == {'rssi': GENERIC_BLUETOOTH_SERVICE_INFO.rssi}
    assert coordinator.data == {'fake': 'data'}
    cancel()
    unregister_listener()

# Add type annotations for other test functions as well

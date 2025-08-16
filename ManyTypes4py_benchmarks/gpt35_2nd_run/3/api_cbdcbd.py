from __future__ import annotations
import asyncio
from asyncio import Future
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, cast, List, Dict, Any, Union
from homeassistant.core import HomeAssistant, callback as hass_callback
from homeassistant.helpers.singleton import singleton
from .const import DATA_MANAGER
from .manager import HomeAssistantBluetoothManager
from .match import BluetoothCallbackMatcher
from .models import BluetoothCallback, BluetoothChange, ProcessAdvertisementCallback
if TYPE_CHECKING:
    from bleak.backends.device import BLEDevice

@singleton(DATA_MANAGER)
def _get_manager(hass: HomeAssistant) -> HomeAssistantBluetoothManager:
    return cast(HomeAssistantBluetoothManager, get_manager())

@hass_callback
def async_get_scanner(hass: HomeAssistant) -> HaBleakScannerWrapper:
    return HaBleakScannerWrapper()

@hass_callback
def async_scanner_by_source(hass: HomeAssistant, source: str) -> Any:
    return _get_manager(hass).async_scanner_by_source(source)

@hass_callback
def async_scanner_count(hass: HomeAssistant, connectable: bool = True) -> int:
    return _get_manager(hass).async_scanner_count(connectable)

@hass_callback
def async_discovered_service_info(hass: HomeAssistant, connectable: bool = True) -> List[BluetoothServiceInfoBleak]:
    return _get_manager(hass).async_discovered_service_info(connectable)

@hass_callback
def async_last_service_info(hass: HomeAssistant, address: str, connectable: bool = True) -> BluetoothServiceInfoBleak:
    return _get_manager(hass).async_last_service_info(address, connectable)

@hass_callback
def async_ble_device_from_address(hass: HomeAssistant, address: str, connectable: bool = True) -> BLEDevice:
    return _get_manager(hass).async_ble_device_from_address(address, connectable)

@hass_callback
def async_scanner_devices_by_address(hass: HomeAssistant, address: str, connectable: bool = True) -> List[BluetoothScannerDevice]:
    return _get_manager(hass).async_scanner_devices_by_address(address, connectable)

@hass_callback
def async_address_present(hass: HomeAssistant, address: str, connectable: bool = True) -> bool:
    return _get_manager(hass).async_address_present(address, connectable)

@hass_callback
def async_register_callback(hass: HomeAssistant, callback: Callable, match_dict: Dict[str, Any], mode: str) -> BluetoothCallback:
    return _get_manager(hass).async_register_callback(callback, match_dict)

async def async_process_advertisements(hass: HomeAssistant, callback: Callable, match_dict: Dict[str, Any], mode: str, timeout: int) -> BluetoothServiceInfoBleak:
    done: Future = hass.loop.create_future()

    @hass_callback
    def _async_discovered_device(service_info: BluetoothServiceInfoBleak, change: BluetoothChange) -> None:
        if not done.done() and callback(service_info):
            done.set_result(service_info)
    unload = _get_manager(hass).async_register_callback(_async_discovered_device, match_dict)
    try:
        async with asyncio.timeout(timeout):
            return await done
    finally:
        unload()

@hass_callback
def async_track_unavailable(hass: HomeAssistant, callback: Callable, address: str, connectable: bool = True) -> Callable:
    return _get_manager(hass).async_track_unavailable(callback, address, connectable)

@hass_callback
def async_rediscover_address(hass: HomeAssistant, address: str) -> None:
    _get_manager(hass).async_rediscover_address(address)

@hass_callback
def async_register_scanner(hass: HomeAssistant, scanner: Any, connection_slots: Union[int, None] = None, source_domain: str = None, source_model: str = None, source_config_entry_id: str = None, source_device_id: str = None) -> Any:
    return _get_manager(hass).async_register_hass_scanner(scanner, connection_slots, source_domain, source_model, source_config_entry_id, source_device_id)

@hass_callback
def async_remove_scanner(hass: HomeAssistant, source: str) -> None:
    return _get_manager(hass).async_remove_scanner(source)

@hass_callback
def async_get_advertisement_callback(hass: HomeAssistant) -> Callable:
    return _get_manager(hass).scanner_adv_received

@hass_callback
def async_get_learned_advertising_interval(hass: HomeAssistant, address: str) -> int:
    return _get_manager(hass).async_get_learned_advertising_interval(address)

@hass_callback
def async_get_fallback_availability_interval(hass: HomeAssistant, address: str) -> int:
    return _get_manager(hass).async_get_fallback_availability_interval(address)

@hass_callback
def async_set_fallback_availability_interval(hass: HomeAssistant, address: str, interval: int) -> None:
    _get_manager(hass).async_set_fallback_availability_interval(address, interval)

"""The bluetooth integration apis.

These APIs are the only documented way to interact with the bluetooth integration.
"""
from __future__ import annotations
import asyncio
from asyncio import Future
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, cast
from habluetooth import BaseHaScanner, BluetoothScannerDevice, BluetoothScanningMode, HaBleakScannerWrapper, get_manager
from home_assistant_bluetooth import BluetoothServiceInfoBleak
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback as hass_callback
from homeassistant.helpers.singleton import singleton
from .const import DATA_MANAGER
from .manager import HomeAssistantBluetoothManager
from .match import BluetoothCallbackMatcher
from .models import BluetoothCallback, BluetoothChange, ProcessAdvertisementCallback
if TYPE_CHECKING:
    from bleak.backends.device import BLEDevice

@singleton(DATA_MANAGER)
def _get_manager(hass: homeassistancore.HomeAssistant) -> Union[str, None, boucanpy.core.http_server.HttpServerRepo, boucanpy.core.security.TokenPayload]:
    """Get the bluetooth manager."""
    return cast(HomeAssistantBluetoothManager, get_manager())

@hass_callback
def async_get_scanner(hass: homeassistancore.HomeAssistant) -> HaBleakScannerWrapper:
    """Return a HaBleakScannerWrapper.

    This is a wrapper around our BleakScanner singleton that allows
    multiple integrations to share the same BleakScanner.
    """
    return HaBleakScannerWrapper()

@hass_callback
def async_scanner_by_source(hass: homeassistancore.HomeAssistant, source: homeassistancore.HomeAssistant) -> Union[str, typing.Callable[str, None], bool]:
    """Return a scanner for a given source.

    This method is only intended to be used by integrations that implement
    a bluetooth client and need to interact with a scanner directly.

    It is not intended to be used by integrations that need to interact
    with a device.
    """
    return _get_manager(hass).async_scanner_by_source(source)

@hass_callback
def async_scanner_count(hass: homeassistancore.HomeAssistant, connectable: bool=True) -> Union[int, typing.Callable[float, None]]:
    """Return the number of scanners currently in use."""
    return _get_manager(hass).async_scanner_count(connectable)

@hass_callback
def async_discovered_service_info(hass: Union[homeassistancore.HomeAssistant, str], connectable: bool=True) -> Union[bool, dict[str, str], str]:
    """Return the discovered devices list."""
    return _get_manager(hass).async_discovered_service_info(connectable)

@hass_callback
def async_last_service_info(hass: Union[str, homeassistancore.HomeAssistant], address: Union[str, homeassistancore.HomeAssistant], connectable: bool=True) -> Union[dict[str, str], dict]:
    """Return the last service info for an address."""
    return _get_manager(hass).async_last_service_info(address, connectable)

@hass_callback
def async_ble_device_from_address(hass: Union[str, homeassistancore.HomeAssistant, typing.Iterable[str]], address: Union[str, homeassistancore.HomeAssistant, typing.Iterable[str]], connectable: bool=True) -> Union[str, None, dict[str, typing.Any]]:
    """Return BLEDevice for an address if its present."""
    return _get_manager(hass).async_ble_device_from_address(address, connectable)

@hass_callback
def async_scanner_devices_by_address(hass: Union[str, typing.Iterable[str], int], address: Union[str, typing.Iterable[str], int], connectable: bool=True) -> Union[str, bool, dict]:
    """Return all discovered BluetoothScannerDevice for an address."""
    return _get_manager(hass).async_scanner_devices_by_address(address, connectable)

@hass_callback
def async_address_present(hass: Union[str, homeassistancore.HomeAssistant, None], address: Union[str, homeassistancore.HomeAssistant, None], connectable: bool=True) -> str:
    """Check if an address is present in the bluetooth device list."""
    return _get_manager(hass).async_address_present(address, connectable)

@hass_callback
def async_register_callback(hass: Union[homeassistancore.HomeAssistant, str, dict[str, typing.Any]], callback: Union[homeassistancore.HomeAssistant, str, dict[str, typing.Any]], match_dict: Union[homeassistancore.HomeAssistant, str, dict[str, typing.Any]], mode: Union[dict, homeassistancore.HomeAssistant, list[dict]]) -> str:
    """Register to receive a callback on bluetooth change.

    mode is currently not used as we only support active scanning.
    Passive scanning will be available in the future. The flag
    is required to be present to avoid a future breaking change
    when we support passive scanning.

    Returns a callback that can be used to cancel the registration.
    """
    return _get_manager(hass).async_register_callback(callback, match_dict)

async def async_process_advertisements(hass, callback, match_dict, mode, timeout):
    """Process advertisements until callback returns true or timeout expires."""
    done = hass.loop.create_future()

    @hass_callback
    def _async_discovered_device(service_info: Any, change: Any) -> None:
        if not done.done() and callback(service_info):
            done.set_result(service_info)
    unload = _get_manager(hass).async_register_callback(_async_discovered_device, match_dict)
    try:
        async with asyncio.timeout(timeout):
            return await done
    finally:
        unload()

@hass_callback
def async_track_unavailable(hass: Union[str, int], callback: Union[str, int], address: Union[str, int], connectable: bool=True) -> Union[bool, typing.Callable, str]:
    """Register to receive a callback when an address is unavailable.

    Returns a callback that can be used to cancel the registration.
    """
    return _get_manager(hass).async_track_unavailable(callback, address, connectable)

@hass_callback
def async_rediscover_address(hass: Union[homeassistancore.HomeAssistant, str, typing.Iterable[str]], address: Union[homeassistancore.HomeAssistant, str, typing.Iterable[str]]) -> None:
    """Trigger discovery of devices which have already been seen."""
    _get_manager(hass).async_rediscover_address(address)

@hass_callback
def async_register_scanner(hass: Union[homeassistancore.HomeAssistant, str, dict], scanner: Union[homeassistancore.HomeAssistant, str, dict], connection_slots: Union[None, homeassistancore.HomeAssistant, str, dict]=None, source_domain: Union[None, homeassistancore.HomeAssistant, str, dict]=None, source_model: Union[None, homeassistancore.HomeAssistant, str, dict]=None, source_config_entry_id: Union[None, homeassistancore.HomeAssistant, str, dict]=None, source_device_id: Union[None, homeassistancore.HomeAssistant, str, dict]=None) -> str:
    """Register a BleakScanner."""
    return _get_manager(hass).async_register_hass_scanner(scanner, connection_slots, source_domain, source_model, source_config_entry_id, source_device_id)

@hass_callback
def async_remove_scanner(hass: homeassistancore.HomeAssistant, source: homeassistancore.HomeAssistant) -> Union[str, bool]:
    """Permanently remove a BleakScanner by source address."""
    return _get_manager(hass).async_remove_scanner(source)

@hass_callback
def async_get_advertisement_callback(hass: homeassistancore.HomeAssistant):
    """Get the advertisement callback."""
    return _get_manager(hass).scanner_adv_received

@hass_callback
def async_get_learned_advertising_interval(hass: Union[str, dict[str, typing.Any]], address: Union[str, dict[str, typing.Any]]) -> Union[bool, str]:
    """Get the learned advertising interval for a MAC address."""
    return _get_manager(hass).async_get_learned_advertising_interval(address)

@hass_callback
def async_get_fallback_availability_interval(hass: Union[str, asyncio.AbstractEventLoop, homeassistancore.HomeAssistant], address: Union[str, asyncio.AbstractEventLoop, homeassistancore.HomeAssistant]) -> Union[str, None]:
    """Get the fallback availability timeout for a MAC address."""
    return _get_manager(hass).async_get_fallback_availability_interval(address)

@hass_callback
def async_set_fallback_availability_interval(hass: Union[float, int], address: Union[float, int], interval: Union[float, int]) -> None:
    """Override the fallback availability timeout for a MAC address."""
    _get_manager(hass).async_set_fallback_availability_interval(address, interval)
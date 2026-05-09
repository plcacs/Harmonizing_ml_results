"""The bluetooth integration."""

from collections.abc import Callable, Mapping
from typing import Any, Optional, Union, overload
from bleak_retry_connector import BleakSlotManager
from bluetooth_adapters import BluetoothAdapters
from habluetooth import BaseHaRemoteScanner, BaseHaScanner, BluetoothManager
from homeassistant.core import Event, HomeAssistant
from .match import BluetoothCallbackMatcher, BluetoothCallbackMatcherIndex, BluetoothCallbackMatcherWithCallback
from .models import BluetoothCallback, BluetoothChange, BluetoothServiceInfoBleak
from .storage import BluetoothStorage

class HomeAssistantBluetoothManager(BluetoothManager):
    """Manage Bluetooth for Home Assistant."""
    _callback_index: BluetoothCallbackMatcherIndex
    _cancel_logging_listener: Optional[Callable[[], None]]
    _integration_matcher: Any  # IntegrationMatcher
    hass: HomeAssistant
    storage: BluetoothStorage
    _all_history: Mapping[str, BluetoothServiceInfoBleak]
    _connectable_history: Mapping[str, BluetoothServiceInfoBleak]
    _debug: bool

    def __init__(
        self,
        hass: HomeAssistant,
        integration_matcher: Any,
        bluetooth_adapters: BluetoothAdapters,
        storage: BluetoothStorage,
        slot_manager: BleakSlotManager,
    ) -> None: ...

    def _async_logging_changed(self, event: Optional[Event] = None) -> None: ...

    def _async_trigger_matching_discovery(self, service_info: BluetoothServiceInfoBleak) -> None: ...

    def async_rediscover_address(self, address: str) -> None: ...

    def _discover_service_info(self, service_info: BluetoothServiceInfoBleak) -> None: ...

    def _address_disappeared(self, address: str) -> None: ...

    async def async_setup(self) -> None: ...

    def async_register_callback(
        self,
        callback: BluetoothCallback,
        matcher: Optional[BluetoothCallbackMatcher],
    ) -> Callable[[], None]: ...

    def async_stop(self, event: Optional[Event] = None) -> None: ...

    def _async_save_scanner_histories(self) -> None: ...

    def _async_save_scanner_history(self, scanner: BaseHaScanner) -> None: ...

    def _async_unregister_scanner(self, scanner: BaseHaScanner, unregister: Callable[[], None]) -> None: ...

    def async_register_hass_scanner(
        self,
        scanner: BaseHaScanner,
        connection_slots: Optional[int] = None,
        source_domain: Optional[str] = None,
        source_model: Optional[str] = None,
        source_config_entry_id: Optional[str] = None,
        source_device_id: Optional[str] = None,
    ) -> Callable[[], None]: ...

    def async_register_scanner(
        self,
        scanner: BaseHaScanner,
        connection_slots: Optional[int] = None,
    ) -> Callable[[], None]: ...

    def async_remove_scanner(self, source: str) -> None: ...

    def _handle_config_entry_removed(self, entry: Any) -> None: ...
from __future__ import annotations

from typing import Optional

from bleak_retry_connector import BleakSlotManager
from bluetooth_adapters import BluetoothAdapters
from habluetooth import BaseHaRemoteScanner, BaseHaScanner, BluetoothManager
from homeassistant import config_entries
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant

from .match import (
    ADDRESS,
    CALLBACK,
    CONNECTABLE,
    BluetoothCallbackMatcher,
    BluetoothCallbackMatcherIndex,
    BluetoothCallbackMatcherWithCallback,
    IntegrationMatcher,
    ble_device_matches,
)
from .models import BluetoothCallback, BluetoothChange, BluetoothServiceInfoBleak
from .storage import BluetoothStorage


class HomeAssistantBluetoothManager(BluetoothManager):
    __slots__ = ("_callback_index", "_cancel_logging_listener", "_integration_matcher", "hass", "storage")
    hass: HomeAssistant
    storage: BluetoothStorage
    _integration_matcher: IntegrationMatcher
    _callback_index: BluetoothCallbackMatcherIndex
    _cancel_logging_listener: Optional[CALLBACK_TYPE]

    def __init__(
        self,
        hass: HomeAssistant,
        integration_matcher: IntegrationMatcher,
        bluetooth_adapters: BluetoothAdapters,
        storage: BluetoothStorage,
        slot_manager: BleakSlotManager,
    ) -> None: ...
    def _async_logging_changed(self, event: Optional[Event] = ...) -> None: ...
    def _async_trigger_matching_discovery(self, service_info: BluetoothServiceInfoBleak) -> None: ...
    def async_rediscover_address(self, address: str) -> None: ...
    def _discover_service_info(self, service_info: BluetoothServiceInfoBleak) -> None: ...
    def _address_disappeared(self, address: str) -> None: ...
    async def async_setup(self) -> None: ...
    def async_register_callback(
        self, callback: BluetoothCallback, matcher: Optional[BluetoothCallbackMatcher]
    ) -> CALLBACK_TYPE: ...
    def async_stop(self, event: Optional[Event] = ...) -> None: ...
    def _async_save_scanner_histories(self) -> None: ...
    def _async_save_scanner_history(self, scanner: BaseHaScanner) -> None: ...
    def _async_unregister_scanner(self, scanner: BaseHaScanner, unregister: CALLBACK_TYPE) -> None: ...
    def async_register_hass_scanner(
        self,
        scanner: BaseHaScanner,
        connection_slots: Optional[int] = ...,
        source_domain: Optional[str] = ...,
        source_model: Optional[str] = ...,
        source_config_entry_id: Optional[str] = ...,
        source_device_id: Optional[str] = ...,
    ) -> CALLBACK_TYPE: ...
    def async_register_scanner(self, scanner: BaseHaScanner, connection_slots: Optional[int] = ...) -> CALLBACK_TYPE: ...
    def async_remove_scanner(self, source: str) -> None: ...
    def _handle_config_entry_removed(self, entry: config_entries.ConfigEntry) -> None: ...
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from bleak_retry_connector import BleakSlotManager
from bluetooth_adapters import BluetoothAdapters
from habluetooth import BaseHaRemoteScanner, BaseHaScanner, BluetoothManager

from homeassistant import config_entries
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, callback as hass_callback

from .match import BluetoothCallbackMatcher, BluetoothCallbackMatcherIndex, IntegrationMatcher
from .models import BluetoothCallback, BluetoothChange, BluetoothServiceInfoBleak
from .storage import BluetoothStorage

_LOGGER: logging.Logger

class HomeAssistantBluetoothManager(BluetoothManager):
    _callback_index: BluetoothCallbackMatcherIndex
    _cancel_logging_listener: CALLBACK_TYPE | None
    _integration_matcher: IntegrationMatcher
    hass: HomeAssistant
    storage: BluetoothStorage

    def __init__(
        self,
        hass: HomeAssistant,
        integration_matcher: IntegrationMatcher,
        bluetooth_adapters: BluetoothAdapters,
        storage: BluetoothStorage,
        slot_manager: BleakSlotManager,
    ) -> None: ...

    def _async_logging_changed(self, event: Event | None = None) -> None: ...
    def _async_trigger_matching_discovery(self, service_info: BluetoothServiceInfoBleak) -> None: ...
    def async_rediscover_address(self, address: str) -> None: ...
    def _discover_service_info(self, service_info: BluetoothServiceInfoBleak) -> None: ...
    def _address_disappeared(self, address: str) -> None: ...
    async def async_setup(self) -> None: ...

    def async_register_callback(
        self,
        callback: BluetoothCallback,
        matcher: BluetoothCallbackMatcher | None,
    ) -> Callable[[], None]: ...

    def async_stop(self, event: Event | None = None) -> None: ...
    def _async_save_scanner_histories(self) -> None: ...
    def _async_save_scanner_history(self, scanner: BaseHaScanner) -> None: ...
    def _async_unregister_scanner(self, scanner: BaseHaScanner, unregister: Callable[[], None]) -> None: ...

    def async_register_hass_scanner(
        self,
        scanner: BaseHaScanner,
        connection_slots: int | None = None,
        source_domain: str | None = None,
        source_model: str | None = None,
        source_config_entry_id: str | None = None,
        source_device_id: str | None = None,
    ) -> CALLBACK_TYPE: ...

    def async_register_scanner(
        self,
        scanner: BaseHaScanner,
        connection_slots: int | None = None,
    ) -> CALLBACK_TYPE: ...

    def async_remove_scanner(self, source: str) -> None: ...
    def _handle_config_entry_removed(self, entry: config_entries.ConfigEntry) -> None: ...

import logging
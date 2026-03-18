```python
"""The bluetooth integration."""

from __future__ import annotations
from collections.abc import Callable, Iterable
from typing import Any

import logging
from bleak_retry_connector import BleakSlotManager
from bluetooth_adapters import BluetoothAdapters
from habluetooth import BaseHaRemoteScanner, BaseHaScanner, BluetoothManager
from homeassistant import config_entries
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from .const import CONF_SOURCE, CONF_SOURCE_CONFIG_ENTRY_ID, CONF_SOURCE_DEVICE_ID, CONF_SOURCE_DOMAIN, CONF_SOURCE_MODEL, DOMAIN
from .match import ADDRESS, CALLBACK, CONNECTABLE, BluetoothCallbackMatcher, BluetoothCallbackMatcherIndex, BluetoothCallbackMatcherWithCallback, IntegrationMatcher
from .models import BluetoothCallback, BluetoothChange, BluetoothServiceInfoBleak
from .storage import BluetoothStorage

_LOGGER: logging.Logger = ...

class HomeAssistantBluetoothManager(BluetoothManager):
    """Manage Bluetooth for Home Assistant."""
    
    __slots__ = ('_callback_index', '_cancel_logging_listener', '_integration_matcher', 'hass', 'storage')
    
    hass: HomeAssistant
    storage: BluetoothStorage
    _callback_index: BluetoothCallbackMatcherIndex
    _cancel_logging_listener: CALLBACK_TYPE | None
    _integration_matcher: IntegrationMatcher
    
    def __init__(
        self,
        hass: HomeAssistant,
        integration_matcher: IntegrationMatcher,
        bluetooth_adapters: BluetoothAdapters,
        storage: BluetoothStorage,
        slot_manager: BleakSlotManager
    ) -> None: ...
    
    @hass_callback
    def _async_logging_changed(self, event: Event | None = ...) -> None: ...
    
    def _async_trigger_matching_discovery(self, service_info: BluetoothServiceInfoBleak) -> None: ...
    
    @hass_callback
    def async_rediscover_address(self, address: str) -> None: ...
    
    def _discover_service_info(self, service_info: BluetoothServiceInfoBleak) -> None: ...
    
    def _address_disappeared(self, address: str) -> None: ...
    
    async def async_setup(self) -> None: ...
    
    def async_register_callback(
        self,
        callback: BluetoothCallback,
        matcher: BluetoothCallbackMatcher | None
    ) -> Callable[[], None]: ...
    
    @hass_callback
    def async_stop(self, event: Event | None = ...) -> None: ...
    
    def _async_save_scanner_histories(self) -> None: ...
    
    def _async_save_scanner_history(self, scanner: BaseHaScanner | BaseHaRemoteScanner) -> None: ...
    
    def _async_unregister_scanner(
        self,
        scanner: BaseHaScanner | BaseHaRemoteScanner,
        unregister: Callable[[], None]
    ) -> None: ...
    
    @hass_callback
    def async_register_hass_scanner(
        self,
        scanner: BaseHaScanner | BaseHaRemoteScanner,
        connection_slots: int | None = ...,
        source_domain: str | None = ...,
        source_model: str | None = ...,
        source_config_entry_id: str | None = ...,
        source_device_id: str | None = ...
    ) -> Callable[[], None]: ...
    
    def async_register_scanner(
        self,
        scanner: BaseHaScanner | BaseHaRemoteScanner,
        connection_slots: int | None = ...
    ) -> Callable[[], None]: ...
    
    @hass_callback
    def async_remove_scanner(self, source: str) -> None: ...
    
    @hass_callback
    def _handle_config_entry_removed(self, entry: Any) -> None: ...
```
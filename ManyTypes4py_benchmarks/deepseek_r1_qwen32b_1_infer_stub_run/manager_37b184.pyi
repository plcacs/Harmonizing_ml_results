"""The bluetooth integration."""
from __future__ import annotations
from collections.abc import Callable, Iterable
from functools import partial
import itertools
from typing import Any, Optional, Union

from bleak_retry_connector import BleakSlotManager
from bluetooth_adapters import BluetoothAdapters
from habluetooth import BaseHaRemoteScanner, BaseHaScanner
from homeassistant import config_entries
from homeassistant.const import EVENT_HOMEASSISTANT_STOP, EVENT_LOGGING_CHANGED
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, callback as hass_callback
from homeassistant.helpers import discovery_flow
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import CONF_SOURCE, CONF_SOURCE_CONFIG_ENTRY_ID, CONF_SOURCE_DEVICE_ID, CONF_SOURCE_DOMAIN, CONF_SOURCE_MODEL, DOMAIN
from .match import ADDRESS, CALLBACK, CONNECTABLE, BluetoothCallbackMatcher, BluetoothCallbackMatcherIndex, BluetoothCallbackMatcherWithCallback, IntegrationMatcher
from .models import BluetoothCallback, BluetoothChange, BluetoothServiceInfoBleak
from .storage import BluetoothStorage

class HomeAssistantBluetoothManager:
    """Manage Bluetooth for Home Assistant."""
    __slots__ = ('_callback_index', '_cancel_logging_listener', '_integration_matcher', 'hass', 'storage')

    def __init__(self, hass: HomeAssistant, integration_matcher: IntegrationMatcher, bluetooth_adapters: BluetoothAdapters, storage: BluetoothStorage, slot_manager: BleakSlotManager) -> None:
        ...

    @hass_callback
    def _async_logging_changed(self, event: Optional[Event] = None) -> None:
        ...

    def _async_trigger_matching_discovery(self, service_info: BluetoothServiceInfoBleak) -> None:
        ...

    @hass_callback
    def async_rediscover_address(self, address: str) -> None:
        ...

    def _discover_service_info(self, service_info: BluetoothServiceInfoBleak) -> None:
        ...

    def _address_disappeared(self, address: str) -> None:
        ...

    async def async_setup(self) -> None:
        ...

    def async_register_callback(self, callback: BluetoothCallback, matcher: Optional[BluetoothCallbackMatcher]) -> Callable[[], None]:
        ...

    @hass_callback
    def async_stop(self, event: Optional[Event] = None) -> None:
        ...

    def _async_save_scanner_histories(self) -> None:
        ...

    def _async_save_scanner_history(self, scanner: Union[BaseHaScanner, BaseHaRemoteScanner]) -> None:
        ...

    @hass_callback
    def async_register_hass_scanner(self, scanner: BaseHaScanner, connection_slots: Optional[int] = None, source_domain: Optional[str] = None, source_model: Optional[str] = None, source_config_entry_id: Optional[str] = None, source_device_id: Optional[str] = None) -> Callable[[], None]:
        ...

    def async_register_scanner(self, scanner: Union[BaseHaScanner, BaseHaRemoteScanner], connection_slots: Optional[int] = None) -> Callable[[], None]:
        ...

    @hass_callback
    def async_remove_scanner(self, source: str) -> None:
        ...

    @hass_callback
    def _handle_config_entry_removed(self, entry: config_entries.ConfigEntry) -> None:
        ...
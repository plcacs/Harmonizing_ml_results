"""The bluetooth integration."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
import itertools
import logging
from typing import TYPE_CHECKING, Any

from bleak_retry_connector import BleakSlotManager
from bluetooth_adapters import BluetoothAdapters
from habluetooth import BaseHaRemoteScanner, BaseHaScanner, BluetoothManager

from homeassistant import config_entries
from homeassistant.const import EVENT_HOMEASSISTANT_STOP, EVENT_LOGGING_CHANGED
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, callback as hass_callback
from homeassistant.helpers import discovery_flow
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import (
    CONF_SOURCE,
    CONF_SOURCE_CONFIG_ENTRY_ID,
    CONF_SOURCE_DEVICE_ID,
    CONF_SOURCE_DOMAIN,
    CONF_SOURCE_MODEL,
    DOMAIN,
)
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
from .util import async_load_history_from_system

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry

_LOGGER: logging.Logger


class HomeAssistantBluetoothManager(BluetoothManager):
    """Manage Bluetooth for Home Assistant."""

    __slots__ = (
        "_callback_index",
        "_cancel_logging_listener",
        "_debug",
        "_all_history",
        "_connectable_history",
        "_integration_matcher",
        "hass",
        "storage",
    )

    def __init__(
        self,
        hass: HomeAssistant,
        integration_matcher: IntegrationMatcher,
        bluetooth_adapters: BluetoothAdapters,
        storage: BluetoothStorage,
        slot_manager: BleakSlotManager,
    ) -> None: ...

    @hass_callback
    def _async_logging_changed(self, event: Event | None = None) -> None: ...

    def _async_trigger_matching_discovery(
        self, service_info: BluetoothServiceInfoBleak
    ) -> None: ...

    @hass_callback
    def async_rediscover_address(self, address: str) -> None: ...

    def _discover_service_info(
        self, service_info: BluetoothServiceInfoBleak
    ) -> None: ...

    def _address_disappeared(self, address: str) -> None: ...

    async def async_setup(self) -> None: ...

    def async_register_callback(
        self,
        callback: BluetoothCallback,
        matcher: BluetoothCallbackMatcher | None,
    ) -> Callable[[], None]: ...

    @hass_callback
    def async_stop(self, event: Event | None = None) -> None: ...

    def _async_save_scanner_histories(self) -> None: ...

    def _async_save_scanner_history(self, scanner: BaseHaScanner) -> None: ...

    def _async_unregister_scanner(
        self, scanner: BaseHaScanner, unregister: Callable[[], None]
    ) -> None: ...

    @hass_callback
    def async_register_hass_scanner(
        self,
        scanner: BaseHaScanner,
        connection_slots: int | None = None,
        source_domain: str | None = None,
        source_model: str | None = None,
        source_config_entry_id: str | None = None,
        source_device_id: str | None = None,
    ) -> Callable[[], None]: ...

    def async_register_scanner(
        self,
        scanner: BaseHaScanner,
        connection_slots: int | None = None,
    ) -> Callable[[], None]: ...

    @hass_callback
    def async_remove_scanner(self, source: str) -> None: ...

    @hass_callback
    def _handle_config_entry_removed(self, entry: ConfigEntry) -> None: ...
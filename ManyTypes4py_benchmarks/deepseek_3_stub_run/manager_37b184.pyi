"""The bluetooth integration."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
import logging
from typing import TYPE_CHECKING, Any

from bleak_retry_connector import BleakSlotManager
from bluetooth_adapters import BluetoothAdapters
from habluetooth import BaseHaRemoteScanner, BaseHaScanner, BluetoothManager
from homeassistant import config_entries
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant
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
)
from .models import BluetoothCallback, BluetoothChange, BluetoothServiceInfoBleak
from .storage import BluetoothStorage

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry

_LOGGER: logging.Logger = ...


class HomeAssistantBluetoothManager(BluetoothManager):
    """Manage Bluetooth for Home Assistant."""

    __slots__ = (
        "_callback_index",
        "_cancel_logging_listener",
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
    ) -> None:
        """Init bluetooth manager."""
        ...

    @hass_callback
    def _async_logging_changed(self, event: Event | None = ...) -> None:
        """Handle logging change."""
        ...

    def _async_trigger_matching_discovery(
        self, service_info: BluetoothServiceInfoBleak
    ) -> None:
        """Trigger discovery for matching domains."""
        ...

    @hass_callback
    def async_rediscover_address(self, address: str) -> None:
        """Trigger discovery of devices which have already been seen."""
        ...

    def _discover_service_info(self, service_info: BluetoothServiceInfoBleak) -> None:
        """Discover service info."""
        ...

    def _address_disappeared(self, address: str) -> None:
        """Dismiss all discoveries for the given address."""
        ...

    async def async_setup(self) -> None:
        """Set up the bluetooth manager."""
        ...

    def async_register_callback(
        self,
        callback: BluetoothCallback,
        matcher: BluetoothCallbackMatcher | None,
    ) -> Callable[[], None]:
        """Register a callback."""
        ...

    @hass_callback
    def async_stop(self, event: Event | None = ...) -> None:
        """Stop the Bluetooth integration at shutdown."""
        ...

    def _async_save_scanner_histories(self) -> None:
        """Save the scanner histories."""
        ...

    def _async_save_scanner_history(self, scanner: BaseHaScanner) -> None:
        """Save the scanner history."""
        ...

    def _async_unregister_scanner(
        self, scanner: BaseHaScanner, unregister: Callable[[], None]
    ) -> None:
        """Unregister a scanner."""
        ...

    @hass_callback
    def async_register_hass_scanner(
        self,
        scanner: BaseHaScanner,
        connection_slots: int | None = ...,
        source_domain: str | None = ...,
        source_model: str | None = ...,
        source_config_entry_id: str | None = ...,
        source_device_id: str | None = ...,
    ) -> Callable[[], None]:
        """Register a scanner."""
        ...

    def async_register_scanner(
        self,
        scanner: BaseHaScanner,
        connection_slots: int | None = ...,
    ) -> Callable[[], None]:
        """Register a scanner."""
        ...

    @hass_callback
    def async_remove_scanner(self, source: str) -> None:
        """Remove a scanner."""
        ...

    @hass_callback
    def _handle_config_entry_removed(self, entry: ConfigEntry) -> None:
        """Handle config entry changes."""
        ...
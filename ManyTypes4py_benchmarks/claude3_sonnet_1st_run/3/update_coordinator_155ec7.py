"""Update coordinator for the Bluetooth integration."""
from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import Callable, List, Optional
from habluetooth import BluetoothScanningMode
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from .api import async_address_present, async_last_service_info, async_register_callback, async_track_unavailable
from .match import BluetoothCallbackMatcher
from .models import BluetoothChange, BluetoothServiceInfoBleak

class BasePassiveBluetoothCoordinator(ABC):
    """Base class for passive bluetooth coordinator for bluetooth advertisements.

    The coordinator is responsible for tracking devices.
    """

    def __init__(
        self, 
        hass: HomeAssistant, 
        logger: logging.Logger, 
        address: str, 
        mode: BluetoothScanningMode, 
        connectable: bool
    ) -> None:
        """Initialize the coordinator."""
        self.hass: HomeAssistant = hass
        self.logger: logging.Logger = logger
        self.address: str = address
        self.connectable: bool = connectable
        self._on_stop: List[CALLBACK_TYPE] = []
        self.mode: BluetoothScanningMode = mode
        self._last_unavailable_time: float = 0.0
        self._last_name: str = address
        self._available: bool = async_address_present(hass, address, connectable)

    @callback
    def async_start(self) -> CALLBACK_TYPE:
        """Start the data updater."""
        self._async_start()
        return self._async_stop

    @callback
    @abstractmethod
    def _async_handle_bluetooth_event(
        self, 
        service_info: BluetoothServiceInfoBleak, 
        change: BluetoothChange
    ) -> None:
        """Handle a bluetooth event."""

    @property
    def name(self) -> str:
        """Return last known name of the device."""
        if (service_info := async_last_service_info(self.hass, self.address, self.connectable)):
            return service_info.name
        return self._last_name

    @property
    def last_seen(self) -> float:
        """Return the last time the device was seen."""
        if (service_info := async_last_service_info(self.hass, self.address, self.connectable)):
            return service_info.time
        return self._last_unavailable_time

    @callback
    def _async_start(self) -> None:
        """Start the callbacks."""
        self._on_stop.append(async_register_callback(self.hass, self._async_handle_bluetooth_event, BluetoothCallbackMatcher(address=self.address, connectable=self.connectable), self.mode))
        self._on_stop.append(async_track_unavailable(self.hass, self._async_handle_unavailable, self.address, self.connectable))

    @callback
    def _async_stop(self) -> None:
        """Stop the callbacks."""
        for unsub in self._on_stop:
            unsub()
        self._on_stop.clear()

    @callback
    def _async_handle_unavailable(self, service_info: BluetoothServiceInfoBleak) -> None:
        """Handle the device going unavailable."""
        self._last_unavailable_time = service_info.time
        self._last_name = service_info.name
        self._available = False

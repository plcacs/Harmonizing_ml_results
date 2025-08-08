from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from habluetooth import BluetoothScanningMode
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from .api import async_address_present, async_last_service_info, async_register_callback, async_track_unavailable
from .match import BluetoothCallbackMatcher
from .models import BluetoothChange, BluetoothServiceInfoBleak

class BasePassiveBluetoothCoordinator(ABC):
    def __init__(self, hass: HomeAssistant, logger: logging.Logger, address: str, mode: BluetoothScanningMode, connectable: bool) -> None:
    
    @callback
    def async_start(self) -> CALLBACK_TYPE:
    
    @callback
    @abstractmethod
    def _async_handle_bluetooth_event(self, service_info: BluetoothServiceInfoBleak, change: BluetoothChange) -> None:
    
    @property
    def name(self) -> str:
    
    @property
    def last_seen(self) -> float:
    
    @callback
    def _async_start(self) -> None:
    
    @callback
    def _async_stop(self) -> None:
    
    @callback
    def _async_handle_unavailable(self, service_info: BluetoothServiceInfoBleak) -> None:

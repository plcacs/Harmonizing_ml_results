from __future__ import annotations
import asyncio
from collections.abc import Callable
from datetime import timedelta
from typing import Any, overload

from homeassistant import config_entries
from homeassistant.components.websocket_api import ActiveConnection
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, callback as hass_callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.deprecation import DeprecatedConstant, all_with_deprecated_constants, check_if_deprecated_constant, dir_with_deprecated_constants
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.service_info.usb import UsbServiceInfo as _UsbServiceInfo
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import USBMatcher, async_get_usb

from .models import USBDevice
from .const import DOMAIN

_LOGGER: logging.Logger

PORT_EVENT_CALLBACK_TYPE: Callable[[set[USBDevice], set[USBDevice]], None]
POLLING_MONITOR_SCAN_PERIOD: timedelta
REQUEST_SCAN_COOLDOWN: int
ADD_REMOVE_SCAN_COOLDOWN: int

CONFIG_SCHEMA: cv.CONFIG_SCHEMA

class USBCallbackMatcher(USBMatcher):
    pass

def async_register_scan_request_callback(hass: HomeAssistant, callback: Callable) -> Callable[[], None]:
    ...

def async_register_initial_scan_callback(hass: HomeAssistant, callback: Callable) -> Callable[[], None]:
    ...

def async_register_port_event_callback(hass: HomeAssistant, callback: Callable) -> Callable[[], None]:
    ...

def async_is_plugged_in(hass: HomeAssistant, matcher: dict[str, str]) -> bool:
    ...

_DEPRECATED_UsbServiceInfo: DeprecatedConstant

@overload
def human_readable_device_name(device: str, serial_number: str, manufacturer: str, description: str, vid: str, pid: str) -> str:
    ...

def human_readable_device_name(device: str, serial_number: str, manufacturer: str, description: str, vid: str, pid: str) -> str:
    ...

def get_serial_by_id(dev_path: str) -> str:
    ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    ...

def _fnmatch_lower(name: str, pattern: str) -> bool:
    ...

def _is_matching(device: USBDevice, matcher: dict[str, str]) -> bool:
    ...

class USBDiscovery:
    def __init__(self, hass: HomeAssistant, usb: list[dict[str, str]]):
        ...

    async def async_setup(self) -> None:
        ...

    async def async_start(self, event: Event) -> None:
        ...

    def async_stop(self, event: Event) -> None:
        ...

    def _async_supports_monitoring(self) -> bool:
        ...

    async def _async_start_monitor(self) -> None:
        ...

    def _async_start_monitor_polling(self) -> None:
        ...

    async def _async_start_aiousbwatcher(self) -> None:
        ...

    def _async_stop_watcher(self, event: Event) -> None:
        ...

    def async_register_scan_request_callback(self, _callback: Callable) -> Callable[[], None]:
        ...

    def async_register_initial_scan_callback(self, callback: Callable) -> Callable[[], None]:
        ...

    def async_register_port_event_callback(self, callback: Callable) -> Callable[[], None]:
        ...

    async def _async_process_discovered_usb_device(self, device: USBDevice) -> None:
        ...

    async def _async_process_ports(self, ports: list[ListPortInfo]) -> None:
        ...

    def _async_delayed_add_remove_scan(self) -> None:
        ...

    async def _async_scan_serial(self) -> None:
        ...

    async def _async_scan(self) -> None:
        ...

    async def async_request_scan(self) -> None:
        ...

@websocket_api.require_admin
@websocket_api.websocket_command({vol.Required('type'): 'usb/scan'})
@websocket_api.async_response
async def websocket_usb_scan(hass: HomeAssistant, connection: ActiveConnection, msg: dict[str, Any]) -> None:
    ...

__getattr__ = partial(check_if_deprecated_constant, module_globals=globals())
__dir__ = partial(dir_with_deprecated_constants, module_globals_keys=[*globals().keys()])
__all__ = all_with_deprecated_constants(globals())

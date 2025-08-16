from __future__ import annotations
import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final

CONFIG_SCHEMA: Final = cv.empty_config_schema(DOMAIN)
HOSTNAME: Final = 'hostname'
MAC_ADDRESS: Final = 'macaddress'
IP_ADDRESS: Final = 'ip'
REGISTERED_DEVICES: Final = 'registered_devices'
SCAN_INTERVAL: Final = timedelta(minutes=60)
_LOGGER: Final = logging.getLogger(__name__)

@dataclass(slots=True)
class DhcpMatchers:
    registered_devices_domains: set
    no_oui_matchers: dict
    oui_matchers: dict

def async_index_integration_matchers(integration_matchers: list) -> DhcpMatchers:
    ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    ...

class WatcherBase:
    def __init__(self, hass: HomeAssistant, address_data: dict, integration_matchers: DhcpMatchers):
        ...

    def async_stop(self):
        ...

    def async_process_client(self, ip_address: str, hostname: str, unformatted_mac_address: str, force: bool = False):
        ...

class NetworkWatcher(WatcherBase):
    def __init__(self, hass: HomeAssistant, address_data: dict, integration_matchers: DhcpMatchers):
        ...

    def async_stop(self):
        ...

    def async_start(self):
        ...

    def async_start_discover(self, *_):
        ...

    async def async_discover(self):
        ...

class DeviceTrackerWatcher(WatcherBase):
    def async_start(self):
        ...

    def _async_process_device_event(self, event: Event):
        ...

    def _async_process_device_state(self, state: State):
        ...

class DeviceTrackerRegisteredWatcher(WatcherBase):
    def async_start(self):
        ...

    def _async_process_device_data(self, data: dict):
        ...

class DHCPWatcher(WatcherBase):
    def _async_process_dhcp_request(self, response: Any):
        ...

    async def async_start(self):
        ...

class RediscoveryWatcher(WatcherBase):
    def _handle_config_entry_removed(self, entry: Any):
        ...

    def async_start(self):
        ...

@lru_cache(maxsize=4096, typed=True)
def _compile_fnmatch(pattern: str) -> Any:
    ...

@lru_cache(maxsize=1024, typed=True)
def _memorized_fnmatch(name: str, pattern: str) -> bool:
    ...

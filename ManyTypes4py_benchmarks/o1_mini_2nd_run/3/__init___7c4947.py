"""The dhcp integration."""
from __future__ import annotations
import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import timedelta
from fnmatch import translate
from functools import lru_cache, partial
import itertools
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Final
import aiodhcpwatcher
from aiodiscover import DiscoverHosts
from aiodiscover.discovery import (
    HOSTNAME as DISCOVERY_HOSTNAME, 
    IP_ADDRESS as DISCOVERY_IP_ADDRESS, 
    MAC_ADDRESS as DISCOVERY_MAC_ADDRESS
)
from cached_ipaddress import cached_ip_addresses
from homeassistant import config_entries
from homeassistant.components.device_tracker import (
    ATTR_HOST_NAME, 
    ATTR_IP, 
    ATTR_MAC, 
    ATTR_SOURCE_TYPE, 
    CONNECTED_DEVICE_REGISTERED, 
    DOMAIN as DEVICE_TRACKER_DOMAIN, 
    SourceType
)
from homeassistant.const import (
    EVENT_HOMEASSISTANT_STARTED, 
    EVENT_HOMEASSISTANT_STOP, 
    STATE_HOME
)
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, State, callback
from homeassistant.helpers import (
    config_validation as cv, 
    device_registry as dr, 
    discovery_flow
)
from homeassistant.helpers.deprecation import (
    DeprecatedConstant, 
    all_with_deprecated_constants, 
    check_if_deprecated_constant, 
    dir_with_deprecated_constants
)
from homeassistant.helpers.device_registry import CONNECTION_NETWORK_MAC, format_mac
from homeassistant.helpers.discovery_flow import DiscoveryKey
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.event import (
    async_track_state_added_domain, 
    async_track_time_interval
)
from homeassistant.helpers.service_info.dhcp import DhcpServiceInfo as _DhcpServiceInfo
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import DHCPMatcher, async_get_dhcp
from .const import DOMAIN

CONFIG_SCHEMA: Final = cv.empty_config_schema(DOMAIN)
HOSTNAME: Final = 'hostname'
MAC_ADDRESS: Final = 'macaddress'
IP_ADDRESS: Final = 'ip'
REGISTERED_DEVICES: Final = 'registered_devices'
SCAN_INTERVAL: Final = timedelta(minutes=60)
_LOGGER: Final = logging.getLogger(__name__)
_DEPRECATED_DhcpServiceInfo: Final = DeprecatedConstant(
    _DhcpServiceInfo, 
    'homeassistant.helpers.service_info.dhcp.DhcpServiceInfo', 
    '2026.2'
)

@dataclass(slots=True)
class DhcpMatchers:
    """Prepared info from dhcp entries."""
    registered_devices_domains: Set[str]
    no_oui_matchers: Dict[str, List[Dict[str, Any]]]
    oui_matchers: Dict[str, List[Dict[str, Any]]]

def async_index_integration_matchers(
    integration_matchers: List[Dict[str, Any]]
) -> DhcpMatchers:
    """Index the integration matchers.

    We have three types of matchers:

    1. Registered devices
    2. Devices with no OUI - index by first char of lower() hostname
    3. Devices with OUI - index by OUI
    """
    registered_devices_domains: Set[str] = set()
    no_oui_matchers: Dict[str, List[Dict[str, Any]]] = {}
    oui_matchers: Dict[str, List[Dict[str, Any]]] = {}
    for matcher in integration_matchers:
        domain: str = matcher['domain']
        if REGISTERED_DEVICES in matcher:
            registered_devices_domains.add(domain)
            continue
        if (mac_address := matcher.get(MAC_ADDRESS)):
            oui_matchers.setdefault(mac_address[:6], []).append(matcher)
            continue
        if (hostname := matcher.get(HOSTNAME)):
            first_char: str = hostname[0].lower()
            no_oui_matchers.setdefault(first_char, []).append(matcher)
    return DhcpMatchers(
        registered_devices_domains=registered_devices_domains, 
        no_oui_matchers=no_oui_matchers, 
        oui_matchers=oui_matchers
    )

async def async_setup(
    hass: HomeAssistant, 
    config: ConfigType
) -> bool:
    """Set up the dhcp component."""
    watchers: List[WatcherBase] = []
    address_data: Dict[str, Dict[str, str]] = {}
    integration_matchers: DhcpMatchers = async_index_integration_matchers(
        await async_get_dhcp(hass)
    )
    device_watcher: DeviceTrackerWatcher = DeviceTrackerWatcher(
        hass, address_data, integration_matchers
    )
    device_watcher.async_start()
    watchers.append(device_watcher)
    device_tracker_registered_watcher: DeviceTrackerRegisteredWatcher = DeviceTrackerRegisteredWatcher(
        hass, address_data, integration_matchers
    )
    device_tracker_registered_watcher.async_start()
    watchers.append(device_tracker_registered_watcher)

    async def _async_initialize(event: Event) -> None:
        await aiodhcpwatcher.async_init()
        network_watcher: NetworkWatcher = NetworkWatcher(
            hass, address_data, integration_matchers
        )
        network_watcher.async_start()
        watchers.append(network_watcher)
        dhcp_watcher: DHCPWatcher = DHCPWatcher(
            hass, address_data, integration_matchers
        )
        await dhcp_watcher.async_start()
        watchers.append(dhcp_watcher)
        rediscovery_watcher: RediscoveryWatcher = RediscoveryWatcher(
            hass, address_data, integration_matchers
        )
        rediscovery_watcher.async_start()
        watchers.append(rediscovery_watcher)

        @callback
        def _async_stop(event: Event) -> None:
            for watcher in watchers:
                watcher.async_stop()
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _async_stop)
    
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _async_initialize)
    return True

class WatcherBase:
    """Base class for dhcp and device tracker watching."""

    def __init__(
        self, 
        hass: HomeAssistant, 
        address_data: Dict[str, Dict[str, str]], 
        integration_matchers: DhcpMatchers
    ) -> None:
        """Initialize class."""
        self.hass: HomeAssistant = hass
        self._integration_matchers: DhcpMatchers = integration_matchers
        self._address_data: Dict[str, Dict[str, str]] = address_data
        self._unsub: Optional[Callable[[], Any]] = None

    @callback
    def async_stop(self) -> None:
        """Stop scanning for new devices on the network."""
        if self._unsub:
            self._unsub()
            self._unsub = None

    @callback
    def async_process_client(
        self, 
        ip_address: str, 
        hostname: str, 
        unformatted_mac_address: str, 
        force: bool = False
    ) -> None:
        """Process a client."""
        if (made_ip_address := cached_ip_addresses(ip_address)) is None:
            _LOGGER.debug('Ignoring invalid IP Address: %s', ip_address)
            return
        if (
            made_ip_address.is_link_local 
            or made_ip_address.is_loopback 
            or made_ip_address.is_unspecified
        ):
            return
        formatted_mac: str = format_mac(unformatted_mac_address)
        mac_address: str = formatted_mac.replace(':', '')
        compressed_ip_address: str = made_ip_address.compressed
        data: Optional[Dict[str, str]] = self._address_data.get(mac_address)
        if (
            not force 
            and data 
            and (data[IP_ADDRESS] == compressed_ip_address) 
            and data[HOSTNAME].startswith(hostname)
        ):
            return
        data = {IP_ADDRESS: compressed_ip_address, HOSTNAME: hostname}
        self._address_data[mac_address] = data
        lowercase_hostname: str = hostname.lower()
        uppercase_mac: str = mac_address.upper()
        _LOGGER.debug(
            'Processing updated address data for %s: mac=%s hostname=%s', 
            ip_address, 
            uppercase_mac, 
            lowercase_hostname
        )
        matched_domains: Set[str] = set()
        matchers: DhcpMatchers = self._integration_matchers
        registered_devices_domains: Set[str] = matchers.registered_devices_domains
        dev_reg = dr.async_get(self.hass)
        device = dev_reg.async_get_device(
            connections={(CONNECTION_NETWORK_MAC, formatted_mac)}
        )
        if device:
            for entry_id in device.config_entries:
                entry = self.hass.config_entries.async_get_entry(entry_id)
                if entry and entry.domain in registered_devices_domains:
                    matched_domains.add(entry.domain)
        oui: str = uppercase_mac[:6]
        lowercase_hostname_first_char: str = lowercase_hostname[0] if lowercase_hostname else ''
        for matcher in itertools.chain(
            matchers.no_oui_matchers.get(lowercase_hostname_first_char, ()),
            matchers.oui_matchers.get(oui, ())
        ):
            domain: str = matcher['domain']
            matcher_hostname: Optional[str] = matcher.get(HOSTNAME)
            if matcher_hostname is not None and not _memorized_fnmatch(
                lowercase_hostname, matcher_hostname
            ):
                continue
            _LOGGER.debug('Matched %s against %s', data, matcher)
            matched_domains.add(domain)
        if not matched_domains:
            return
        discovery_key: DiscoveryKey = DiscoveryKey(
            domain=DOMAIN, key=mac_address, version=1
        )
        for domain in matched_domains:
            discovery_flow.async_create_flow(
                self.hass, 
                domain, 
                {'source': config_entries.SOURCE_DHCP}, 
                _DhcpServiceInfo(
                    ip=ip_address, 
                    hostname=lowercase_hostname, 
                    macaddress=mac_address
                ), 
                discovery_key=discovery_key
            )

class NetworkWatcher(WatcherBase):
    """Class to query ptr records routers."""

    def __init__(
        self, 
        hass: HomeAssistant, 
        address_data: Dict[str, Dict[str, str]], 
        integration_matchers: DhcpMatchers
    ) -> None:
        """Initialize class."""
        super().__init__(hass, address_data, integration_matchers)
        self._discover_hosts: Optional[DiscoverHosts] = None
        self._discover_task: Optional[asyncio.Task[Any]] = None

    @callback
    def async_stop(self) -> None:
        """Stop scanning for new devices on the network."""
        super().async_stop()
        if self._discover_task:
            self._discover_task.cancel()
            self._discover_task = None

    @callback
    def async_start(self) -> None:
        """Start scanning for new devices on the network."""
        self._discover_hosts = DiscoverHosts()
        self._unsub = async_track_time_interval(
            self.hass, 
            self.async_start_discover, 
            SCAN_INTERVAL, 
            name='DHCP network watcher'
        )
        self.async_start_discover()

    @callback
    def async_start_discover(self, _: Any = ...) -> None:
        """Start a new discovery task if one is not running."""
        if self._discover_task and not self._discover_task.done():
            return
        self._discover_task = asyncio.create_task(
            self.async_discover(), 
            name='dhcp discovery'
        )

    async def async_discover(self) -> None:
        """Process discovery."""
        assert self._discover_hosts is not None
        async for host in self._discover_hosts.async_discover():
            self.async_process_client(
                host[DISCOVERY_IP_ADDRESS], 
                host[DISCOVERY_HOSTNAME], 
                host[DISCOVERY_MAC_ADDRESS]
            )

class DeviceTrackerWatcher(WatcherBase):
    """Class to watch dhcp data from routers."""

    @callback
    def async_start(self) -> None:
        """Start watching for new device trackers."""
        self._unsub = async_track_state_added_domain(
            self.hass, 
            [DEVICE_TRACKER_DOMAIN], 
            self._async_process_device_event
        )
        for state in self.hass.states.async_all(DEVICE_TRACKER_DOMAIN):
            self._async_process_device_state(state)

    @callback
    def _async_process_device_event(self, event: Event) -> None:
        """Process a device tracker state change event."""
        state: Optional[State] = event.data.get('new_state')
        self._async_process_device_state(state)

    @callback
    def _async_process_device_state(self, state: Optional[State]) -> None:
        """Process a device tracker state."""
        if state is None or state.state != STATE_HOME:
            return
        attributes: Dict[str, Any] = state.attributes
        if attributes.get(ATTR_SOURCE_TYPE) != SourceType.ROUTER:
            return
        ip_address: Optional[str] = attributes.get(ATTR_IP)
        hostname: str = attributes.get(ATTR_HOST_NAME, '')
        mac_address: Optional[str] = attributes.get(ATTR_MAC)
        if ip_address is None or mac_address is None:
            return
        self.async_process_client(ip_address, hostname, mac_address)

class DeviceTrackerRegisteredWatcher(WatcherBase):
    """Class to watch data from device tracker registrations."""

    @callback
    def async_start(self) -> None:
        """Start watching for device tracker registrations."""
        self._unsub = async_dispatcher_connect(
            self.hass, 
            CONNECTED_DEVICE_REGISTERED, 
            self._async_process_device_data
        )

    @callback
    def _async_process_device_data(self, data: Dict[str, Any]) -> None:
        """Process a device tracker registration."""
        ip_address: Optional[str] = data.get(ATTR_IP)
        hostname: str = data.get(ATTR_HOST_NAME) or ''
        mac_address: Optional[str] = data.get(ATTR_MAC)
        if ip_address is None or mac_address is None:
            return
        self.async_process_client(ip_address, hostname, mac_address)

class DHCPWatcher(WatcherBase):
    """Class to watch dhcp requests."""

    @callback
    def _async_process_dhcp_request(self, response: aiodhcpwatcher.DhcpResponse) -> None:
        """Process a dhcp request."""
        self.async_process_client(
            response.ip_address, 
            response.hostname, 
            response.mac_address
        )

    async def async_start(self) -> None:
        """Start watching for dhcp packets."""
        self._unsub = await aiodhcpwatcher.async_start(
            self._async_process_dhcp_request
        )

class RediscoveryWatcher(WatcherBase):
    """Class to trigger rediscovery on config entry removal."""

    @callback
    def _handle_config_entry_removed(self, entry: config_entries.ConfigEntry) -> None:
        """Handle config entry removal."""
        discovery_keys: Iterable[DiscoveryKey] = entry.discovery_keys.get(DOMAIN, [])
        for discovery_key in discovery_keys:
            if discovery_key.version != 1 or not isinstance(discovery_key.key, str):
                continue
            mac_address: str = discovery_key.key
            _LOGGER.debug('Rediscover service %s', mac_address)
            if (data := self._address_data.get(mac_address)):
                self.async_process_client(
                    data[IP_ADDRESS], 
                    data[HOSTNAME], 
                    mac_address, 
                    True
                )

    @callback
    def async_start(self) -> None:
        """Start watching for config entry removals."""
        self._unsub = async_dispatcher_connect(
            self.hass, 
            config_entries.signal_discovered_config_entry_removed(DOMAIN), 
            self._handle_config_entry_removed
        )

@lru_cache(maxsize=4096, typed=True)
def _compile_fnmatch(pattern: str) -> re.Pattern:
    """Compile a fnmatch pattern."""
    return re.compile(translate(pattern))

@lru_cache(maxsize=1024, typed=True)
def _memorized_fnmatch(name: str, pattern: str) -> bool:
    """Memorized version of fnmatch that has a larger lru_cache.

    The default version of fnmatch only has a lru_cache of 256 entries.
    With many devices we quickly reach that limit and end up compiling
    the same pattern over and over again.

    DHCP has its own memorized fnmatch with its own lru_cache
    since the data is going to be relatively the same
    since the devices will not change frequently
    """
    return bool(_compile_fnmatch(pattern).match(name))

def __getattr__(name: str) -> Any:
    return check_if_deprecated_constant(name, module_globals=globals())

def __dir__() -> List[str]:
    return dir_with_deprecated_constants(module_globals_keys=[*globals().keys()])

__all__: List[str] = all_with_deprecated_constants(globals())

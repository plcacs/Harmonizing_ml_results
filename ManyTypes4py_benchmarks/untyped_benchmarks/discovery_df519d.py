"""The Thread integration."""
from __future__ import annotations
from collections.abc import Callable
import dataclasses
import logging
from typing import cast
from python_otbr_api.mdns import StateBitmap
from zeroconf import BadTypeInNameException, DNSPointer, ServiceListener, Zeroconf, instance_name_from_service_info
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf
from homeassistant.components import zeroconf
from homeassistant.core import HomeAssistant
_LOGGER = logging.getLogger(__name__)
KNOWN_BRANDS = {'Amazon': 'amazon', 'Apple Inc.': 'apple', 'Aqara': 'aqara_gateway', 'eero': 'eero', 'Google Inc.': 'google', 'HomeAssistant': 'homeassistant', 'Home Assistant': 'homeassistant', 'Nanoleaf': 'nanoleaf', 'OpenThread': 'openthread', 'Samsung': 'samsung'}
THREAD_TYPE = '_meshcop._udp.local.'
CLASS_IN = 1
TYPE_PTR = 12

@dataclasses.dataclass
class ThreadRouterDiscoveryData:
    """Thread router discovery data."""

def async_discovery_data_from_service(service, ext_addr, ext_pan_id):
    """Get a ThreadRouterDiscoveryData from an AsyncServiceInfo."""

    def try_decode(value):
        """Try decoding UTF-8."""
        if value is None:
            return None
        try:
            return value.decode()
        except UnicodeDecodeError:
            return None
    service_properties = service.properties
    border_agent_id = service_properties.get(b'id')
    model_name = try_decode(service_properties.get(b'mn'))
    network_name = try_decode(service_properties.get(b'nn'))
    server = service.server
    thread_version = try_decode(service_properties.get(b'tv'))
    vendor_name = try_decode(service_properties.get(b'vn'))
    unconfigured = None
    brand = KNOWN_BRANDS.get(vendor_name)
    if brand == 'homeassistant':
        if (state_bitmap_b := service_properties.get(b'sb')) is not None:
            try:
                state_bitmap = StateBitmap.from_bytes(state_bitmap_b)
                if not state_bitmap.is_active:
                    unconfigured = True
            except ValueError:
                _LOGGER.debug('Failed to decode state bitmap in service %s', service)
        if service_properties.get(b'at') is None:
            unconfigured = True
    return ThreadRouterDiscoveryData(instance_name=instance_name_from_service_info(service), addresses=service.parsed_addresses(), border_agent_id=border_agent_id.hex() if border_agent_id is not None else None, brand=brand, extended_address=ext_addr.hex(), extended_pan_id=ext_pan_id.hex(), model_name=model_name, network_name=network_name, server=server, thread_version=thread_version, unconfigured=unconfigured, vendor_name=vendor_name)

def async_read_zeroconf_cache(aiozc):
    """Return all meshcop records already in the zeroconf cache."""
    results = []
    records = aiozc.zeroconf.cache.async_all_by_details(THREAD_TYPE, TYPE_PTR, CLASS_IN)
    for record in records:
        record = cast(DNSPointer, record)
        try:
            info = AsyncServiceInfo(THREAD_TYPE, record.alias)
        except BadTypeInNameException as ex:
            _LOGGER.debug('Ignoring record with bad type in name: %s: %s', record.alias, ex)
            continue
        if not info.load_from_cache(aiozc.zeroconf):
            continue
        service_properties = info.properties
        if not (xa := service_properties.get(b'xa')):
            _LOGGER.debug('Ignoring record without xa %s', info)
            continue
        if not (xp := service_properties.get(b'xp')):
            _LOGGER.debug('Ignoring record without xp %s', info)
            continue
        results.append(async_discovery_data_from_service(info, xa, xp))
    return results

class ThreadRouterDiscovery:
    """mDNS based Thread router discovery."""

    class ThreadServiceListener(ServiceListener):
        """Service listener which listens for thread routers."""

        def __init__(self, hass, aiozc, router_discovered, router_removed):
            """Initialize."""
            self._aiozc = aiozc
            self._hass = hass
            self._known_routers = {}
            self._router_discovered = router_discovered
            self._router_removed = router_removed

        def add_service(self, zc, type_, name):
            """Handle service added."""
            _LOGGER.debug('add_service %s', name)
            self._hass.async_create_task(self._add_update_service(type_, name))

        def remove_service(self, zc, type_, name):
            """Handle service removed."""
            _LOGGER.debug('remove_service %s', name)
            if name not in self._known_routers:
                return
            extended_mac_address, _ = self._known_routers.pop(name)
            self._router_removed(extended_mac_address)

        def update_service(self, zc, type_, name):
            """Handle service updated."""
            _LOGGER.debug('update_service %s', name)
            self._hass.async_create_task(self._add_update_service(type_, name))

        async def _add_update_service(self, type_, name):
            """Add or update a service."""
            service = None
            tries = 0
            while service is None and tries < 4:
                service = await self._aiozc.async_get_service_info(type_, name)
                tries += 1
            if not service:
                _LOGGER.debug('_add_update_service failed to add %s, %s', type_, name)
                return
            _LOGGER.debug('_add_update_service %s %s', name, service)
            service_properties = service.properties
            if not (xa := service_properties.get(b'xa')):
                _LOGGER.info('Discovered unsupported Thread router without extended address: %s', service)
                return
            if not (xp := service_properties.get(b'xp')):
                _LOGGER.info('Discovered unsupported Thread router without extended pan ID: %s', service)
                return
            data = async_discovery_data_from_service(service, xa, xp)
            extended_mac_address = xa.hex()
            if name in self._known_routers and self._known_routers[name] == (extended_mac_address, data):
                _LOGGER.debug('_add_update_service suppressing identical update for %s', name)
                return
            self._known_routers[name] = (extended_mac_address, data)
            self._router_discovered(extended_mac_address, data)

    def __init__(self, hass, router_discovered, router_removed):
        """Initialize."""
        self._hass = hass
        self._aiozc = None
        self._router_discovered = router_discovered
        self._router_removed = router_removed
        self._service_listener = None

    async def async_start(self):
        """Start discovery."""
        self._aiozc = aiozc = await zeroconf.async_get_async_instance(self._hass)
        self._service_listener = self.ThreadServiceListener(self._hass, aiozc, self._router_discovered, self._router_removed)
        await aiozc.async_add_service_listener(THREAD_TYPE, self._service_listener)

    async def async_stop(self):
        """Stop discovery."""
        if not self._aiozc or not self._service_listener:
            return
        await self._aiozc.async_remove_service_listener(self._service_listener)
        self._service_listener = None
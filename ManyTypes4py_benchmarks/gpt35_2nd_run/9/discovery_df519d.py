from __future__ import annotations
from collections.abc import Callable
import dataclasses
import logging
from typing import cast
from zeroconf import BadTypeInNameException, DNSPointer, ServiceListener, Zeroconf, instance_name_from_service_info
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf
from homeassistant.components import zeroconf
from homeassistant.core import HomeAssistant
from python_otbr_api.mdns import StateBitmap

_LOGGER: logging.Logger = logging.getLogger(__name__)
KNOWN_BRANDS: dict[str, str] = {'Amazon': 'amazon', 'Apple Inc.': 'apple', 'Aqara': 'aqara_gateway', 'eero': 'eero', 'Google Inc.': 'google', 'HomeAssistant': 'homeassistant', 'Home Assistant': 'homeassistant', 'Nanoleaf': 'nanoleaf', 'OpenThread': 'openthread', 'Samsung': 'samsung'}
THREAD_TYPE: str = '_meshcop._udp.local.'
CLASS_IN: int = 1
TYPE_PTR: int = 12

@dataclasses.dataclass
class ThreadRouterDiscoveryData:
    """Thread router discovery data."""
    instance_name: str
    addresses: list[str]
    border_agent_id: str
    brand: str
    extended_address: str
    extended_pan_id: str
    model_name: str
    network_name: str
    server: str
    thread_version: str
    unconfigured: bool
    vendor_name: str

def async_discovery_data_from_service(service: AsyncServiceInfo, ext_addr: bytes, ext_pan_id: bytes) -> ThreadRouterDiscoveryData:
    """Get a ThreadRouterDiscoveryData from an AsyncServiceInfo."""
    ...

def async_read_zeroconf_cache(aiozc: AsyncZeroconf) -> list[ThreadRouterDiscoveryData]:
    """Return all meshcop records already in the zeroconf cache."""
    ...

class ThreadRouterDiscovery:
    """mDNS based Thread router discovery."""

    class ThreadServiceListener(ServiceListener):
        """Service listener which listens for thread routers."""
        ...

    def __init__(self, hass: HomeAssistant, router_discovered: Callable, router_removed: Callable):
        """Initialize."""
        ...

    async def async_start(self) -> None:
        """Start discovery."""
        ...

    async def async_stop(self) -> None:
        """Stop discovery."""
        ...

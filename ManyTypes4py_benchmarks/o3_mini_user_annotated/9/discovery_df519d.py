"""The Thread integration."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import logging
from typing import cast, Optional

from python_otbr_api.mdns import StateBitmap
from zeroconf import (
    BadTypeInNameException,
    DNSPointer,
    ServiceListener,
    Zeroconf,
    instance_name_from_service_info,
)
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf

from homeassistant.components import zeroconf
from homeassistant.core import HomeAssistant

_LOGGER: logging.Logger = logging.getLogger(__name__)

KNOWN_BRANDS: dict[str | None, str] = {
    "Amazon": "amazon",
    "Apple Inc.": "apple",
    "Aqara": "aqara_gateway",
    "eero": "eero",
    "Google Inc.": "google",
    "HomeAssistant": "homeassistant",
    "Home Assistant": "homeassistant",
    "Nanoleaf": "nanoleaf",
    "OpenThread": "openthread",
    "Samsung": "samsung",
}
THREAD_TYPE: str = "_meshcop._udp.local."
CLASS_IN: int = 1
TYPE_PTR: int = 12


@dataclasses.dataclass
class ThreadRouterDiscoveryData:
    """Thread router discovery data."""

    instance_name: str
    addresses: list[str]
    border_agent_id: Optional[str]
    brand: Optional[str]
    extended_address: str
    extended_pan_id: str
    model_name: Optional[str]
    network_name: Optional[str]
    server: Optional[str]
    thread_version: Optional[str]
    unconfigured: Optional[bool]
    vendor_name: Optional[str]


def async_discovery_data_from_service(
    service: AsyncServiceInfo, ext_addr: bytes, ext_pan_id: bytes
) -> ThreadRouterDiscoveryData:
    """Get a ThreadRouterDiscoveryData from an AsyncServiceInfo."""

    def try_decode(value: Optional[bytes]) -> Optional[str]:
        """Try decoding UTF-8."""
        if value is None:
            return None
        try:
            return value.decode()
        except UnicodeDecodeError:
            return None

    service_properties: dict[bytes, bytes] = service.properties
    border_agent_id: Optional[bytes] = service_properties.get(b"id")
    model_name: Optional[str] = try_decode(service_properties.get(b"mn"))
    network_name: Optional[str] = try_decode(service_properties.get(b"nn"))
    server: Optional[str] = service.server
    thread_version: Optional[str] = try_decode(service_properties.get(b"tv"))
    vendor_name: Optional[str] = try_decode(service_properties.get(b"vn"))

    unconfigured: Optional[bool] = None
    brand: Optional[str] = KNOWN_BRANDS.get(vendor_name)
    if brand == "homeassistant":
        # Attempt to detect incomplete configuration
        state_bitmap_b: Optional[bytes] = service_properties.get(b"sb")
        if state_bitmap_b is not None:
            try:
                state_bitmap = StateBitmap.from_bytes(state_bitmap_b)
                if not state_bitmap.is_active:
                    unconfigured = True
            except ValueError:
                _LOGGER.debug("Failed to decode state bitmap in service %s", service)
        if service_properties.get(b"at") is None:
            unconfigured = True

    return ThreadRouterDiscoveryData(
        instance_name=instance_name_from_service_info(service),
        addresses=service.parsed_addresses(),
        border_agent_id=border_agent_id.hex() if border_agent_id is not None else None,
        brand=brand,
        extended_address=ext_addr.hex(),
        extended_pan_id=ext_pan_id.hex(),
        model_name=model_name,
        network_name=network_name,
        server=server,
        thread_version=thread_version,
        unconfigured=unconfigured,
        vendor_name=vendor_name,
    )


def async_read_zeroconf_cache(aiozc: AsyncZeroconf) -> list[ThreadRouterDiscoveryData]:
    """Return all meshcop records already in the zeroconf cache."""
    results: list[ThreadRouterDiscoveryData] = []

    records = aiozc.zeroconf.cache.async_all_by_details(THREAD_TYPE, TYPE_PTR, CLASS_IN)
    for record in records:
        record = cast(DNSPointer, record)

        try:
            info = AsyncServiceInfo(THREAD_TYPE, record.alias)
        except BadTypeInNameException as ex:
            _LOGGER.debug(
                "Ignoring record with bad type in name: %s: %s", record.alias, ex
            )
            continue

        if not info.load_from_cache(aiozc.zeroconf):
            # data is not fully in the cache, so ignore for now
            continue

        service_properties: dict[bytes, bytes] = info.properties

        xa: Optional[bytes] = service_properties.get(b"xa")
        if not xa:
            _LOGGER.debug("Ignoring record without xa %s", info)
            continue
        xp: Optional[bytes] = service_properties.get(b"xp")
        if not xp:
            _LOGGER.debug("Ignoring record without xp %s", info)
            continue

        results.append(async_discovery_data_from_service(info, xa, xp))

    return results


class ThreadRouterDiscovery:
    """mDNS based Thread router discovery."""

    class ThreadServiceListener(ServiceListener):
        """Service listener which listens for thread routers."""

        def __init__(
            self,
            hass: HomeAssistant,
            aiozc: AsyncZeroconf,
            router_discovered: Callable[[str, ThreadRouterDiscoveryData], None],
            router_removed: Callable[[str], None],
        ) -> None:
            """Initialize."""
            self._aiozc: AsyncZeroconf = aiozc
            self._hass: HomeAssistant = hass
            self._known_routers: dict[str, tuple[str, ThreadRouterDiscoveryData]] = {}
            self._router_discovered: Callable[[str, ThreadRouterDiscoveryData], None] = router_discovered
            self._router_removed: Callable[[str], None] = router_removed

        def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            """Handle service added."""
            _LOGGER.debug("add_service %s", name)
            self._hass.async_create_task(self._add_update_service(type_, name))

        def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            """Handle service removed."""
            _LOGGER.debug("remove_service %s", name)
            if name not in self._known_routers:
                return
            extended_mac_address, _ = self._known_routers.pop(name)
            self._router_removed(extended_mac_address)

        def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            """Handle service updated."""
            _LOGGER.debug("update_service %s", name)
            self._hass.async_create_task(self._add_update_service(type_, name))

        async def _add_update_service(self, type_: str, name: str) -> None:
            """Add or update a service."""
            service: Optional[AsyncServiceInfo] = None
            tries: int = 0
            while service is None and tries < 4:
                service = await self._aiozc.async_get_service_info(type_, name)
                tries += 1

            if not service:
                _LOGGER.debug("_add_update_service failed to add %s, %s", type_, name)
                return

            _LOGGER.debug("_add_update_service %s %s", name, service)
            service_properties: dict[bytes, bytes] = service.properties

            xa: Optional[bytes] = service_properties.get(b"xa")
            if not xa:
                _LOGGER.info(
                    "Discovered unsupported Thread router without extended address: %s",
                    service,
                )
                return
            xp: Optional[bytes] = service_properties.get(b"xp")
            if not xp:
                _LOGGER.info(
                    "Discovered unsupported Thread router without extended pan ID: %s",
                    service,
                )
                return

            data: ThreadRouterDiscoveryData = async_discovery_data_from_service(service, xa, xp)
            extended_mac_address: str = xa.hex()
            if name in self._known_routers and self._known_routers[name] == (
                extended_mac_address,
                data,
            ):
                _LOGGER.debug(
                    "_add_update_service suppressing identical update for %s", name
                )
                return
            self._known_routers[name] = (extended_mac_address, data)
            self._router_discovered(extended_mac_address, data)

    def __init__(
        self,
        hass: HomeAssistant,
        router_discovered: Callable[[str, ThreadRouterDiscoveryData], None],
        router_removed: Callable[[str], None],
    ) -> None:
        """Initialize."""
        self._hass: HomeAssistant = hass
        self._aiozc: Optional[AsyncZeroconf] = None
        self._router_discovered: Callable[[str, ThreadRouterDiscoveryData], None] = router_discovered
        self._router_removed: Callable[[str], None] = router_removed
        self._service_listener: Optional[ThreadRouterDiscovery.ThreadServiceListener] = None

    async def async_start(self) -> None:
        """Start discovery."""
        self._aiozc = await zeroconf.async_get_async_instance(self._hass)
        aiozc: AsyncZeroconf = self._aiozc
        self._service_listener = self.ThreadServiceListener(
            self._hass, aiozc, self._router_discovered, self._router_removed
        )
        await aiozc.async_add_service_listener(THREAD_TYPE, self._service_listener)

    async def async_stop(self) -> None:
        """Stop discovery."""
        if not self._aiozc or not self._service_listener:
            return
        await self._aiozc.async_remove_service_listener(self._service_listener)
        self._service_listener = None

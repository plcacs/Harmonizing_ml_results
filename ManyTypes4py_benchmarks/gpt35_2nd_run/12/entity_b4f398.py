from __future__ import annotations
from aiomusiccast.capabilities import Capability
from homeassistant.const import ATTR_CONNECTIONS, ATTR_VIA_DEVICE
from homeassistant.helpers.device_registry import CONNECTION_NETWORK_MAC, DeviceInfo, format_mac
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import BRAND, DEFAULT_ZONE, DOMAIN, ENTITY_CATEGORY_MAPPING
from .coordinator import MusicCastDataUpdateCoordinator

class MusicCastEntity(CoordinatorEntity[MusicCastDataUpdateCoordinator]):
    def __init__(self, *, name: str, icon: str, coordinator: MusicCastDataUpdateCoordinator, enabled_default: bool = True) -> None:

class MusicCastDeviceEntity(MusicCastEntity):
    _zone_id: str = DEFAULT_ZONE

    @property
    def device_id(self) -> str:

    @property
    def device_name(self) -> str:

    @property
    def device_info(self) -> DeviceInfo:

    async def async_added_to_hass(self) -> None:

    async def async_will_remove_from_hass(self) -> None:

class MusicCastCapabilityEntity(MusicCastDeviceEntity):
    def __init__(self, coordinator: MusicCastDataUpdateCoordinator, capability: Capability, zone_id: str = None) -> None:

    @property
    def unique_id(self) -> str:

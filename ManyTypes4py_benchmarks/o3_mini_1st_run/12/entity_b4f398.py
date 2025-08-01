from __future__ import annotations
from typing import Optional, Set, Tuple, Any
from aiomusiccast.capabilities import Capability
from homeassistant.const import ATTR_CONNECTIONS, ATTR_VIA_DEVICE
from homeassistant.helpers.device_registry import CONNECTION_NETWORK_MAC, DeviceInfo, format_mac
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import BRAND, DEFAULT_ZONE, DOMAIN, ENTITY_CATEGORY_MAPPING
from .coordinator import MusicCastDataUpdateCoordinator


class MusicCastEntity(CoordinatorEntity[MusicCastDataUpdateCoordinator]):
    """Defines a base MusicCast entity."""

    def __init__(
        self,
        *,
        name: str,
        icon: str,
        coordinator: MusicCastDataUpdateCoordinator,
        enabled_default: bool = True,
    ) -> None:
        """Initialize the MusicCast entity."""
        super().__init__(coordinator)
        self._attr_entity_registry_enabled_default: bool = enabled_default
        self._attr_icon: str = icon
        self._attr_name: str = name


class MusicCastDeviceEntity(MusicCastEntity):
    """Defines a MusicCast device entity."""
    _zone_id: str = DEFAULT_ZONE

    @property
    def device_id(self) -> str:
        """Return the ID of the current device."""
        if self._zone_id == DEFAULT_ZONE:
            return self.coordinator.data.device_id  # type: ignore[attr-defined]
        return f'{self.coordinator.data.device_id}_{self._zone_id}'  # type: ignore[attr-defined]

    @property
    def device_name(self) -> str:
        """Return the name of the current device."""
        return self.coordinator.data.zones[self._zone_id].name  # type: ignore[attr-defined]

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information about this MusicCast device."""
        device_info = DeviceInfo(
            name=self.device_name,
            identifiers={(DOMAIN, self.device_id)},
            manufacturer=BRAND,
            model=self.coordinator.data.model_name,  # type: ignore[attr-defined]
            sw_version=self.coordinator.data.system_version,  # type: ignore[attr-defined]
        )
        if self._zone_id == DEFAULT_ZONE:
            device_info[ATTR_CONNECTIONS] = {
                (CONNECTION_NETWORK_MAC, format_mac(mac))
                for mac in self.coordinator.data.mac_addresses.values()  # type: ignore[attr-defined]
            }
        else:
            device_info[ATTR_VIA_DEVICE] = (DOMAIN, self.coordinator.data.device_id)  # type: ignore[attr-defined]
        return device_info

    async def async_added_to_hass(self) -> None:
        """Run when this Entity has been added to HA."""
        await super().async_added_to_hass()
        self.coordinator.musiccast.register_callback(self.async_write_ha_state)  # type: ignore[attr-defined]

    async def async_will_remove_from_hass(self) -> None:
        """Entity being removed from hass."""
        await super().async_will_remove_from_hass()
        self.coordinator.musiccast.remove_callback(self.async_write_ha_state)  # type: ignore[attr-defined]


class MusicCastCapabilityEntity(MusicCastDeviceEntity):
    """Base Entity type for all capabilities."""

    def __init__(
        self,
        coordinator: MusicCastDataUpdateCoordinator,
        capability: Capability,
        zone_id: Optional[str] = None,
    ) -> None:
        """Initialize a capability based entity."""
        if zone_id is not None:
            self._zone_id = zone_id
        self.capability: Capability = capability
        super().__init__(name=capability.name, icon="", coordinator=coordinator)
        self._attr_entity_category = ENTITY_CATEGORY_MAPPING.get(capability.entity_type)

    @property
    def unique_id(self) -> str:
        """Return the unique ID for this entity."""
        return f'{self.device_id}_{self.capability.id}'
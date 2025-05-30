"""The Honeywell Lyric integration."""
from __future__ import annotations
from aiolyric import Lyric
from aiolyric.objects.device import LyricDevice
from aiolyric.objects.location import LyricLocation
from aiolyric.objects.priority import LyricAccessory, LyricRoom
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity, DataUpdateCoordinator

class LyricEntity(CoordinatorEntity[DataUpdateCoordinator[Lyric]]):
    """Defines a base Honeywell Lyric entity."""
    _attr_has_entity_name = True

    def __init__(self, coordinator, location, device, key):
        """Initialize the Honeywell Lyric entity."""
        super().__init__(coordinator)
        self._key = key
        self._location = location
        self._mac_id = device.mac_id
        self._update_thermostat = coordinator.data.update_thermostat
        self._update_fan = coordinator.data.update_fan

    @property
    def unique_id(self):
        """Return the unique ID for this entity."""
        return self._key

    @property
    def location(self):
        """Get the Lyric Location."""
        return self.coordinator.data.locations_dict[self._location.location_id]

    @property
    def device(self):
        """Get the Lyric Device."""
        return self.location.devices_dict[self._mac_id]

class LyricDeviceEntity(LyricEntity):
    """Defines a Honeywell Lyric device entity."""

    @property
    def device_info(self):
        """Return device information about this Honeywell Lyric instance."""
        return DeviceInfo(identifiers={(dr.CONNECTION_NETWORK_MAC, self._mac_id)}, connections={(dr.CONNECTION_NETWORK_MAC, self._mac_id)}, manufacturer='Honeywell', model=self.device.device_model, name=f'{self.device.name} Thermostat')

class LyricAccessoryEntity(LyricDeviceEntity):
    """Defines a Honeywell Lyric accessory entity, a sub-device of a thermostat."""

    def __init__(self, coordinator, location, device, room, accessory, key):
        """Initialize the Honeywell Lyric accessory entity."""
        super().__init__(coordinator, location, device, key)
        self._room_id = room.id
        self._accessory_id = accessory.id

    @property
    def device_info(self):
        """Return device information about this Honeywell Lyric instance."""
        return DeviceInfo(identifiers={(f'{dr.CONNECTION_NETWORK_MAC}_room_accessory', f'{self._mac_id}_room{self._room_id}_accessory{self._accessory_id}')}, manufacturer='Honeywell', model='RCHTSENSOR', name=f'{self.room.room_name} Sensor', via_device=(dr.CONNECTION_NETWORK_MAC, self._mac_id))

    @property
    def room(self):
        """Get the Lyric Device."""
        return self.coordinator.data.rooms_dict[self._mac_id][self._room_id]

    @property
    def accessory(self):
        """Get the Lyric Device."""
        return next((accessory for accessory in self.room.accessories if accessory.id == self._accessory_id))
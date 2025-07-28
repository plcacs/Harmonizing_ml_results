from __future__ import annotations
from typing import Any
from aiolyric import Lyric
from aiolyric.objects.device import LyricDevice
from aiolyric.objects.location import LyricLocation
from aiolyric.objects.priority import LyricAccessory, LyricRoom
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity, DataUpdateCoordinator


class LyricEntity(CoordinatorEntity[DataUpdateCoordinator[Lyric]]):
    _attr_has_entity_name: bool = True

    def __init__(
        self,
        coordinator: DataUpdateCoordinator[Lyric],
        location: LyricLocation,
        device: LyricDevice,
        key: str,
    ) -> None:
        super().__init__(coordinator)
        self._key: str = key
        self._location: LyricLocation = location
        self._mac_id: str = device.mac_id
        self._update_thermostat: Any = coordinator.data.update_thermostat
        self._update_fan: Any = coordinator.data.update_fan

    @property
    def unique_id(self) -> str:
        return self._key

    @property
    def location(self) -> LyricLocation:
        return self.coordinator.data.locations_dict[self._location.location_id]

    @property
    def device(self) -> LyricDevice:
        return self.location.devices_dict[self._mac_id]


class LyricDeviceEntity(LyricEntity):
    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(dr.CONNECTION_NETWORK_MAC, self._mac_id)},
            connections={(dr.CONNECTION_NETWORK_MAC, self._mac_id)},
            manufacturer='Honeywell',
            model=self.device.device_model,
            name=f'{self.device.name} Thermostat'
        )


class LyricAccessoryEntity(LyricDeviceEntity):
    def __init__(
        self,
        coordinator: DataUpdateCoordinator[Lyric],
        location: LyricLocation,
        device: LyricDevice,
        room: LyricRoom,
        accessory: LyricAccessory,
        key: str,
    ) -> None:
        super().__init__(coordinator, location, device, key)
        self._room_id: str = room.id
        self._accessory_id: str = accessory.id

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={
                (f'{dr.CONNECTION_NETWORK_MAC}_room_accessory', f'{self._mac_id}_room{self._room_id}_accessory{self._accessory_id}')
            },
            manufacturer='Honeywell',
            model='RCHTSENSOR',
            name=f'{self.room.room_name} Sensor',
            via_device=(dr.CONNECTION_NETWORK_MAC, self._mac_id)
        )

    @property
    def room(self) -> LyricRoom:
        return self.coordinator.data.rooms_dict[self._mac_id][self._room_id]

    @property
    def accessory(self) -> LyricAccessory:
        return next(
            accessory
            for accessory in self.room.accessories
            if accessory.id == self._accessory_id
        )
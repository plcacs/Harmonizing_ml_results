from __future__ import annotations
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, MANUFACTURER
from .coordinator import DeviceDataUpdateCoordinator

class GoGoGate2Entity(CoordinatorEntity[DeviceDataUpdateCoordinator]):
    def __init__(self, config_entry: ConfigEntry, data_update_coordinator: DeviceDataUpdateCoordinator, door: AbstractDoor, unique_id: str) -> None:
        super().__init__(data_update_coordinator)
        self._config_entry: ConfigEntry = config_entry
        self._door: AbstractDoor = door
        self._door_id: str = door.door_id
        self._api: Any = data_update_coordinator.api
        self._attr_unique_id: str = unique_id

    @property
    def door(self) -> AbstractDoor:
        door: AbstractDoor = get_door_by_id(self._door.door_id, self.coordinator.data)
        self._door = door or self._door
        return self._door

    @property
    def door_status(self) -> Any:
        data: Any = self.coordinator.data
        door_with_statuses: Any = self._api.async_get_door_statuses_from_info(data)
        return door_with_statuses[self._door_id]

    @property
    def device_info(self) -> DeviceInfo:
        data: Any = self.coordinator.data
        if data.remoteaccessenabled:
            configuration_url: str = f'https://{data.remoteaccess}'
        else:
            configuration_url: str = f'http://{self._config_entry.data[CONF_IP_ADDRESS]}'
        return DeviceInfo(configuration_url=configuration_url, identifiers={(DOMAIN, str(self._config_entry.unique_id))}, name=self._config_entry.title, manufacturer=MANUFACTURER, model=data.model, sw_version=data.firmwareversion)

    @property
    def extra_state_attributes(self) -> dict[str, str]:
        return {'door_id': self._door_id}

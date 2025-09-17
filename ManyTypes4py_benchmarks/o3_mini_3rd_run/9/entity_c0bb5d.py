from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from aionotion.bridge.models import Bridge
from aionotion.listener.models import Listener
from homeassistant.core import callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import EntityDescription
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import DOMAIN, LOGGER
from .coordinator import NotionDataUpdateCoordinator


@dataclass(frozen=True, kw_only=True)
class NotionEntityDescription:
    pass


class NotionEntity(CoordinatorEntity[NotionDataUpdateCoordinator]):
    _attr_has_entity_name: bool = True

    def __init__(
        self,
        coordinator: NotionDataUpdateCoordinator,
        listener_id: str,
        sensor_id: str,
        bridge_id: str,
        description: NotionEntityDescription,
    ) -> None:
        super().__init__(coordinator)
        sensor = self.coordinator.data.sensors[sensor_id]
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, sensor.hardware_id)},
            manufacturer='Silicon Labs',
            model=str(sensor.hardware_revision),
            name=str(sensor.name).capitalize(),
            sw_version=sensor.firmware_version,
        )
        if (bridge := self._async_get_bridge(bridge_id)):
            self._attr_device_info['via_device'] = (DOMAIN, bridge.hardware_id)
        self._attr_extra_state_attributes = {}
        self._attr_unique_id = listener_id
        self._bridge_id: str = bridge_id
        self._listener_id: str = listener_id
        self._sensor_id: str = sensor_id
        self.entity_description: NotionEntityDescription = description

    @property
    def available(self) -> bool:
        return self.coordinator.last_update_success and self._listener_id in self.coordinator.data.listeners

    @property
    def listener(self) -> Listener:
        return self.coordinator.data.listeners[self._listener_id]

    @callback
    def _async_get_bridge(self, bridge_id: str) -> Optional[Bridge]:
        if (bridge := self.coordinator.data.bridges.get(bridge_id)) is None:
            LOGGER.debug('Entity references a non-existent bridge ID: %s', bridge_id)
            return None
        return bridge

    @callback
    def _async_update_bridge_id(self) -> None:
        sensor = self.coordinator.data.sensors[self._sensor_id]
        if self._bridge_id == sensor.bridge.id:
            return
        if (bridge := self._async_get_bridge(sensor.bridge.id)) is None:
            return
        self._bridge_id = sensor.bridge.id
        device_registry = dr.async_get(self.hass)
        this_device = device_registry.async_get_device(identifiers={(DOMAIN, sensor.hardware_id)})
        bridge_obj = self.coordinator.data.bridges[self._bridge_id]
        bridge_device = device_registry.async_get_device(identifiers={(DOMAIN, bridge_obj.hardware_id)})
        if not bridge_device or not this_device:
            return
        device_registry.async_update_device(this_device.id, via_device_id=bridge_device.id)

    @callback
    def _handle_coordinator_update(self) -> None:
        if self._listener_id in self.coordinator.data.listeners:
            self._async_update_bridge_id()
        super()._handle_coordinator_update()
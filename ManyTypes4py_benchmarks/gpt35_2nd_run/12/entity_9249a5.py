from __future__ import annotations
from pydrawise.schema import Controller, Sensor, Zone
from homeassistant.core import callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import EntityDescription
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import DOMAIN, MANUFACTURER
from .coordinator import HydrawiseDataUpdateCoordinator

class HydrawiseEntity(CoordinatorEntity[HydrawiseDataUpdateCoordinator]):
    _attr_attribution: str = 'Data provided by hydrawise.com'
    _attr_has_entity_name: bool = True

    def __init__(self, coordinator: HydrawiseDataUpdateCoordinator, description: EntityDescription, controller: Controller, *, zone_id: int = None, sensor_id: int = None) -> None:
        super().__init__(coordinator=coordinator)
        self.entity_description: EntityDescription = description
        self.controller: Controller = controller
        self.zone_id: int = zone_id
        self.sensor_id: int = sensor_id
        self._device_id: str = str(zone_id) if zone_id is not None else str(controller.id)
        self._attr_unique_id: str = f'{self._device_id}_{description.key}'
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, self._device_id)}, name=self.zone.name if zone_id is not None else controller.name, model='Zone' if zone_id is not None else controller.hardware.model.description, manufacturer=MANUFACTURER)
        if zone_id is not None or sensor_id is not None:
            self._attr_device_info['via_device'] = (DOMAIN, str(controller.id))
        self._update_attrs()

    @property
    def zone(self) -> Zone:
        assert self.zone_id is not None
        return self.coordinator.data.zones[self.zone_id]

    @property
    def sensor(self) -> Sensor:
        assert self.sensor_id is not None
        return self.coordinator.data.sensors[self.sensor_id]

    def _update_attrs(self) -> None:
        return

    @callback
    def _handle_coordinator_update(self) -> None:
        self.controller = self.coordinator.data.controllers[self.controller.id]
        self._update_attrs()
        super()._handle_coordinator_update()

    @property
    def available(self) -> bool:
        return super().available and self.controller.online

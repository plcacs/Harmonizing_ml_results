"""Support for Volvo On Call."""
from typing import Any, Dict, Optional
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import DOMAIN
from .coordinator import VolvoUpdateCoordinator

class VolvoEntity(CoordinatorEntity[VolvoUpdateCoordinator]):
    """Base class for all VOC entities."""

    def __init__(
        self,
        vin: str,
        component: str,
        attribute: str,
        slug_attr: str,
        coordinator: VolvoUpdateCoordinator,
    ) -> None:
        """Initialize the entity."""
        super().__init__(coordinator)
        self.vin: str = vin
        self.component: str = component
        self.attribute: str = attribute
        self.slug_attr: str = slug_attr

    @property
    def instrument(self) -> Any:
        """Return corresponding instrument."""
        return self.coordinator.volvo_data.instrument(self.vin, self.component, self.attribute, self.slug_attr)

    @property
    def icon(self) -> Any:
        """Return the icon."""
        return self.instrument.icon

    @property
    def vehicle(self) -> Any:
        """Return vehicle."""
        return self.instrument.vehicle

    @property
    def _entity_name(self) -> str:
        return self.instrument.name

    @property
    def _vehicle_name(self) -> str:
        return self.coordinator.volvo_data.vehicle_name(self.vehicle)

    @property
    def name(self) -> str:
        """Return full name of the entity."""
        return f"{self._vehicle_name} {self._entity_name}"

    @property
    def assumed_state(self) -> bool:
        """Return true if unable to access real state of entity."""
        return True

    @property
    def device_info(self) -> DeviceInfo:
        """Return a unique set of attributes for each vehicle."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.vehicle.vin)},
            name=self._vehicle_name,
            model=self.vehicle.vehicle_type,
            manufacturer="Volvo",
        )

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return device specific state attributes."""
        return dict(self.instrument.attributes, model=f"{self.vehicle.vehicle_type}/{self.vehicle.model_year}")

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        slug_override: str = ""
        if self.instrument.slug_override is not None:
            slug_override = f"-{self.instrument.slug_override}"
        return f"{self.vin}-{self.component}-{self.attribute}{slug_override}"
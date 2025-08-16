from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import DOMAIN
from .coordinator import VolvoUpdateCoordinator

class VolvoEntity(CoordinatorEntity[VolvoUpdateCoordinator]):
    def __init__(self, vin: str, component: str, attribute: str, slug_attr: str, coordinator: VolvoUpdateCoordinator) -> None:
        super().__init__(coordinator)
        self.vin = vin
        self.component = component
        self.attribute = attribute
        self.slug_attr = slug_attr

    @property
    def instrument(self) -> Instrument:
        return self.coordinator.volvo_data.instrument(self.vin, self.component, self.attribute, self.slug_attr)

    @property
    def icon(self) -> str:
        return self.instrument.icon

    @property
    def vehicle(self) -> Vehicle:
        return self.instrument.vehicle

    @property
    def _entity_name(self) -> str:
        return self.instrument.name

    @property
    def _vehicle_name(self) -> str:
        return self.coordinator.volvo_data.vehicle_name(self.vehicle)

    @property
    def name(self) -> str:
        return f'{self._vehicle_name} {self._entity_name}'

    @property
    def assumed_state(self) -> bool:
        return True

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(identifiers={(DOMAIN, self.vehicle.vin)}, name=self._vehicle_name, model=self.vehicle.vehicle_type, manufacturer='Volvo')

    @property
    def extra_state_attributes(self) -> dict:
        return dict(self.instrument.attributes, model=f'{self.vehicle.vehicle_type}/{self.vehicle.model_year}')

    @property
    def unique_id(self) -> str:
        slug_override = ''
        if self.instrument.slug_override is not None:
            slug_override = f'-{self.instrument.slug_override}'
        return f'{self.vin}-{self.component}-{self.attribute}{slug_override}'

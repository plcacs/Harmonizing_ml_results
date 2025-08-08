from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import DOMAIN
from .coordinator import VolvoUpdateCoordinator

class VolvoEntity(CoordinatorEntity[VolvoUpdateCoordinator]):
    def __init__(self, vin: str, component: str, attribute: str, slug_attr: str, coordinator: VolvoUpdateCoordinator) -> None:
        super().__init__(coordinator)
        self.vin: str = vin
        self.component: str = component
        self.attribute: str = attribute
        self.slug_attr: str = slug_attr

    @property
    def instrument(self) -> Instrument:
        ...

    @property
    def icon(self) -> str:
        ...

    @property
    def vehicle(self) -> Vehicle:
        ...

    @property
    def _entity_name(self) -> str:
        ...

    @property
    def _vehicle_name(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def assumed_state(self) -> bool:
        ...

    @property
    def device_info(self) -> DeviceInfo:
        ...

    @property
    def extra_state_attributes(self) -> dict:
        ...

    @property
    def unique_id(self) -> str:
        ...

"""Support for Volvo On Call."""
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import DOMAIN
from .coordinator import VolvoUpdateCoordinator

class VolvoEntity(CoordinatorEntity[VolvoUpdateCoordinator]):
    """Base class for all VOC entities."""

    def __init__(self, vin: Union[typing.Iterable[typing.Any], None, typing.Any, bool], component: Union[str, list[dict[str, typing.Any]], int, None], attribute: Union[int, None, str], slug_attr: Union[dict, int, str, None], coordinator: Union[homeassistanhelpers.update_coordinator.DataUpdateCoordinator, coordinator.NZBGetDataUpdateCoordinator]) -> None:
        """Initialize the entity."""
        super().__init__(coordinator)
        self.vin = vin
        self.component = component
        self.attribute = attribute
        self.slug_attr = slug_attr

    @property
    def instrument(self) -> str:
        """Return corresponding instrument."""
        return self.coordinator.volvo_data.instrument(self.vin, self.component, self.attribute, self.slug_attr)

    @property
    def icon(self):
        """Return the icon."""
        return self.instrument.icon

    @property
    def vehicle(self):
        """Return vehicle."""
        return self.instrument.vehicle

    @property
    def _entity_name(self):
        return self.instrument.name

    @property
    def _vehicle_name(self) -> str:
        return self.coordinator.volvo_data.vehicle_name(self.vehicle)

    @property
    def name(self) -> typing.Text:
        """Return full name of the entity."""
        return f'{self._vehicle_name} {self._entity_name}'

    @property
    def assumed_state(self) -> bool:
        """Return true if unable to access real state of entity."""
        return True

    @property
    def device_info(self) -> DeviceInfo:
        """Return a inique set of attributes for each vehicle."""
        return DeviceInfo(identifiers={(DOMAIN, self.vehicle.vin)}, name=self._vehicle_name, model=self.vehicle.vehicle_type, manufacturer='Volvo')

    @property
    def extra_state_attributes(self):
        """Return device specific state attributes."""
        return dict(self.instrument.attributes, model=f'{self.vehicle.vehicle_type}/{self.vehicle.model_year}')

    @property
    def unique_id(self) -> typing.Text:
        """Return a unique ID."""
        slug_override = ''
        if self.instrument.slug_override is not None:
            slug_override = f'-{self.instrument.slug_override}'
        return f'{self.vin}-{self.component}-{self.attribute}{slug_override}'
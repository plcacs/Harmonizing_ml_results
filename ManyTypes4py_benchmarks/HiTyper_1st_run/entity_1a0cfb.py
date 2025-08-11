"""The powerview integration base entity."""
import logging
from aiopvapi.resources.shade import BaseShade, ShadePosition
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import DOMAIN, MANUFACTURER
from .coordinator import PowerviewShadeUpdateCoordinator
from .model import PowerviewDeviceInfo
from .shade_data import PowerviewShadeData
_LOGGER = logging.getLogger(__name__)

class HDEntity(CoordinatorEntity[PowerviewShadeUpdateCoordinator]):
    """Base class for hunter douglas entities."""
    _attr_has_entity_name = True

    def __init__(self, coordinator: Union[str, homeassistanhelpers.update_coordinator.DataUpdateCoordinator], device_info: Union[str, homeassistanhelpers.update_coordinator.DataUpdateCoordinator], room_name: Union[str, homeassistanhelpers.update_coordinator.DataUpdateCoordinator], powerview_id) -> None:
        """Initialize the entity."""
        super().__init__(coordinator)
        self._room_name = room_name
        self._attr_unique_id = f'{device_info.serial_number}_{powerview_id}'
        self._device_info = device_info
        self._configuration_url = self.coordinator.hub.url

    @property
    def data(self):
        """Return the PowerviewShadeData."""
        return self.coordinator.data

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device_info of the device."""
        return DeviceInfo(connections={(dr.CONNECTION_NETWORK_MAC, self._device_info.mac_address)}, identifiers={(DOMAIN, self._device_info.serial_number)}, manufacturer=MANUFACTURER, model=self._device_info.model, name=self._device_info.name, sw_version=self._device_info.firmware, configuration_url=self._configuration_url)

class ShadeEntity(HDEntity):
    """Base class for hunter douglas shade entities."""

    def __init__(self, coordinator: Union[str, homeassistanhelpers.update_coordinator.DataUpdateCoordinator], device_info: Union[str, homeassistanhelpers.update_coordinator.DataUpdateCoordinator], room_name: Union[str, homeassistanhelpers.update_coordinator.DataUpdateCoordinator], shade: Union[str, homeassistanhelpers.update_coordinator.DataUpdateCoordinator, dict[str, str]], shade_name: Union[dict[typing.Union[int,str], int], dict[str, str], typing.Type]) -> None:
        """Initialize the shade."""
        super().__init__(coordinator, device_info, room_name, shade.id)
        self._shade_name = shade_name
        self._shade = shade
        self._is_hard_wired = not shade.is_battery_powered()
        self._configuration_url = shade.url

    @property
    def positions(self) -> Union[str, int, None, T]:
        """Return the PowerviewShadeData."""
        return self.data.get_shade_position(self._shade.id)

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device_info of the device."""
        return DeviceInfo(identifiers={(DOMAIN, self._shade.id)}, name=self._shade_name, suggested_area=self._room_name, manufacturer=MANUFACTURER, model=self._shade.type_name, sw_version=self._shade.firmware, via_device=(DOMAIN, self._device_info.serial_number), configuration_url=self._configuration_url)
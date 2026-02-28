"""Support for iBeacon device sensors."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from ibeacon_ble import iBeaconAdvertisement
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.const import SIGNAL_STRENGTH_DECIBELS_MILLIWATT, UnitOfLength
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import IBeaconConfigEntry
from .const import SIGNAL_IBEACON_DEVICE_NEW
from .coordinator import IBeaconCoordinator
from .entity import IBeaconEntity

@dataclass(frozen=True, kw_only=True)
class IBeaconSensorEntityDescription(SensorEntityDescription):
    """Describes iBeacon sensor entity."""

    device_class: SensorDeviceClass
    native_unit_of_measurement: str
    entity_registry_enabled_default: bool
    value_fn: Callable[[iBeaconAdvertisement], int | str]
    state_class: SensorStateClass

SENSOR_DESCRIPTIONS: tuple[IBeaconSensorEntityDescription, ...] = (
    IBeaconSensorEntityDescription(key='rssi', device_class=SensorDeviceClass.SIGNAL_STRENGTH, native_unit_of_measurement=SIGNAL_STRENGTH_DECIBELS_MILLIWATT, entity_registry_enabled_default=False, value_fn=lambda ibeacon_advertisement: ibeacon_advertisement.rssi, state_class=SensorStateClass.MEASUREMENT),
    IBeaconSensorEntityDescription(key='power', translation_key='power', device_class=SensorDeviceClass.SIGNAL_STRENGTH, native_unit_of_measurement=SIGNAL_STRENGTH_DECIBELS_MILLIWATT, entity_registry_enabled_default=False, value_fn=lambda ibeacon_advertisement: ibeacon_advertisement.power, state_class=SensorStateClass.MEASUREMENT),
    IBeaconSensorEntityDescription(key='estimated_distance', translation_key='estimated_distance', native_unit_of_measurement=UnitOfLength.METERS, value_fn=lambda ibeacon_advertisement: ibeacon_advertisement.distance, state_class=SensorStateClass.MEASUREMENT, device_class=SensorDeviceClass.DISTANCE),
    IBeaconSensorEntityDescription(key='vendor', translation_key='vendor', entity_registry_enabled_default=False, value_fn=lambda ibeacon_advertisement: ibeacon_advertisement.vendor)
)

async def async_setup_entry(hass: HomeAssistant, entry: IBeaconConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback):
    """Set up sensors for iBeacon Tracker component."""
    coordinator: IBeaconCoordinator = entry.runtime_data

    @callback
    def _async_device_new(unique_id: str, identifier: str, ibeacon_advertisement: iBeaconAdvertisement):
        """Signal a new device."""
        async_add_entities((IBeaconSensorEntity(coordinator, description, identifier, unique_id, ibeacon_advertisement) for description in SENSOR_DESCRIPTIONS))
    entry.async_on_unload(async_dispatcher_connect(hass, SIGNAL_IBEACON_DEVICE_NEW, _async_device_new))

class IBeaconSensorEntity(IBeaconEntity, SensorEntity):
    """An iBeacon sensor entity."""

    def __init__(self, coordinator: IBeaconCoordinator, description: IBeaconSensorEntityDescription, identifier: str, device_unique_id: str, ibeacon_advertisement: iBeaconAdvertisement):
        """Initialize an iBeacon sensor entity."""
        super().__init__(coordinator, identifier, device_unique_id, ibeacon_advertisement)
        self._attr_unique_id = f'{device_unique_id}_{description.key}'
        self.entity_description: IBeaconSensorEntityDescription = description

    @callback
    def _async_seen(self, ibeacon_advertisement: iBeaconAdvertisement):
        """Update state."""
        self._attr_available = True
        self._ibeacon_advertisement = ibeacon_advertisement
        self.async_write_ha_state()

    @callback
    def _async_unavailable(self):
        """Update state."""
        self._attr_available = False
        self.async_write_ha_state()

    @property
    def native_value(self) -> int | str:
        """Return the state of the sensor."""
        return self.entity_description.value_fn(self._ibeacon_advertisement)

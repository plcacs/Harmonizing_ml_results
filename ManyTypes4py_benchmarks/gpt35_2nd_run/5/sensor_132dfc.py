from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.const import CONCENTRATION_MICROGRAMS_PER_CUBIC_METER, CONCENTRATION_PARTS_PER_MILLION, PERCENTAGE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import EcobeeConfigEntry
from .const import DOMAIN, ECOBEE_MODEL_TO_NAME, MANUFACTURER

@dataclass(frozen=True, kw_only=True)
class EcobeeSensorEntityDescription(SensorEntityDescription):
    """Represent the ecobee sensor entity description."""
    key: str
    native_unit_of_measurement: str
    device_class: str
    state_class: str
    runtime_key: Optional[str]

async def async_setup_entry(hass: HomeAssistant, config_entry, async_add_entities: AddConfigEntryEntitiesCallback):
    """Set up ecobee sensors."""
    data = config_entry.runtime_data
    entities: List[EcobeeSensor] = [EcobeeSensor(data, sensor['name'], index, description) for index in range(len(data.ecobee.thermostats)) for sensor in data.ecobee.get_remote_sensors(index) for item in sensor['capability'] for description in SENSOR_TYPES if description.key == item['type']]

class EcobeeSensor(SensorEntity):
    """Representation of an Ecobee sensor."""
    _attr_has_entity_name: bool = True

    def __init__(self, data, sensor_name: str, sensor_index: int, description: EcobeeSensorEntityDescription):
        """Initialize the sensor."""
        self.entity_description: EcobeeSensorEntityDescription = description
        self.data = data
        self.sensor_name = sensor_name
        self.index = sensor_index
        self._state = None

    @property
    def unique_id(self) -> Optional[str]:
        """Return a unique identifier for this sensor."""
        ...

    @property
    def device_info(self) -> Optional[DeviceInfo]:
        """Return device information for this sensor."""
        ...

    @property
    def available(self) -> bool:
        """Return true if device is available."""
        ...

    @property
    def native_value(self) -> Optional[float]:
        """Return the state of the sensor."""
        ...

    async def async_update(self):
        """Get the latest state of the sensor."""
        ...

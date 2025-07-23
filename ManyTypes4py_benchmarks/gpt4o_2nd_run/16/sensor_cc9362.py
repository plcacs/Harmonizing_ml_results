"""Support for VersaSense sensor peripheral."""
from __future__ import annotations
import logging
from typing import Any, Optional
from homeassistant.components.sensor import SensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import DOMAIN
from .const import KEY_CONSUMER, KEY_IDENTIFIER, KEY_MEASUREMENT, KEY_PARENT_MAC, KEY_PARENT_NAME, KEY_UNIT

_LOGGER = logging.getLogger(__name__)

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the sensor platform."""
    if discovery_info is None:
        return
    consumer = hass.data[DOMAIN][KEY_CONSUMER]
    sensor_list = []
    for entity_info in discovery_info.values():
        peripheral = hass.data[DOMAIN][entity_info[KEY_PARENT_MAC]][entity_info[KEY_IDENTIFIER]]
        parent_name = entity_info[KEY_PARENT_NAME]
        unit = entity_info[KEY_UNIT]
        measurement = entity_info[KEY_MEASUREMENT]
        sensor_list.append(VSensor(peripheral, parent_name, unit, measurement, consumer))
    async_add_entities(sensor_list)

class VSensor(SensorEntity):
    """Representation of a Sensor."""

    def __init__(
        self,
        peripheral: Any,
        parent_name: str,
        unit: str,
        measurement: str,
        consumer: Any
    ) -> None:
        """Initialize the sensor."""
        self._state: Optional[Any] = None
        self._available: bool = True
        self._name: str = f'{parent_name} {measurement}'
        self._parent_mac: str = peripheral.parentMac
        self._identifier: str = peripheral.identifier
        self._unit: str = unit
        self._measurement: str = measurement
        self.consumer: Any = consumer

    @property
    def unique_id(self) -> str:
        """Return the unique id of the sensor."""
        return f'{self._parent_mac}/{self._identifier}/{self._measurement}'

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def native_value(self) -> Optional[Any]:
        """Return the state of the sensor."""
        return self._state

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit of measurement."""
        return self._unit

    @property
    def available(self) -> bool:
        """Return if the sensor is available."""
        return self._available

    async def async_update(self) -> None:
        """Fetch new state data for the sensor."""
        samples = await self.consumer.fetchPeripheralSample(None, self._identifier, self._parent_mac)
        if samples is not None:
            for sample in samples:
                if sample.measurement == self._measurement:
                    self._available = True
                    self._state = sample.value
                    break
        else:
            _LOGGER.error('Sample unavailable')
            self._available = False
            self._state = None

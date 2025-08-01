"""Support gathering ted5000 information."""
from __future__ import annotations
from contextlib import suppress
from datetime import timedelta
import logging
from typing import Any, Generator, Optional, Union

import requests
import voluptuous as vol
import xmltodict
from homeassistant.components.sensor import (
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PORT, UnitOfElectricPotential, UnitOfPower
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle

_LOGGER = logging.getLogger(__name__)
DEFAULT_NAME: str = 'ted'
MIN_TIME_BETWEEN_UPDATES: timedelta = timedelta(seconds=10)
PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_HOST): cv.string,
    vol.Optional(CONF_PORT, default=80): cv.port,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string
})
SENSORS: list[SensorEntityDescription] = [
    SensorEntityDescription(
        key='power',
        native_unit_of_measurement=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT
    ),
    SensorEntityDescription(
        key='voltage',
        native_unit_of_measurement=UnitOfElectricPotential.VOLT,
        device_class=SensorDeviceClass.VOLTAGE,
        state_class=SensorStateClass.MEASUREMENT
    )
]


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    """Set up the Ted5000 sensor."""
    host: str = config[CONF_HOST]
    port: int = config[CONF_PORT]
    name: str = config[CONF_NAME]
    url: str = f'http://{host}:{port}/api/LiveData.xml'
    gateway: Ted5000Gateway = Ted5000Gateway(url)
    gateway.update()
    add_entities(
        (
            Ted5000Sensor(gateway, name, mtu, description)
            for mtu in gateway.data  # type: ignore
            for description in SENSORS
        )
    )


class Ted5000Sensor(SensorEntity):
    """Implementation of a Ted5000 sensor."""

    def __init__(self, gateway: Ted5000Gateway, name: str, mtu: int, description: SensorEntityDescription) -> None:
        """Initialize the sensor."""
        self._gateway: Ted5000Gateway = gateway
        self._attr_name: str = f'{name} mtu{mtu} {description.key}'
        self._mtu: int = mtu
        self.entity_description: SensorEntityDescription = description
        self.update()

    @property
    def native_value(self) -> Optional[Union[int, float]]:
        """Return the state of the resources."""
        if (unit := self.entity_description.native_unit_of_measurement):
            with suppress(KeyError):
                return self._gateway.data[self._mtu][unit]
        return None

    def update(self) -> None:
        """Get the latest data from REST API."""
        self._gateway.update()


class Ted5000Gateway:
    """The class for handling the data retrieval."""

    def __init__(self, url: str) -> None:
        """Initialize the data object."""
        self.url: str = url
        self.data: dict[int, dict[str, Union[int, float]]] = {}

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> None:
        """Get the latest data from the Ted5000 XML API."""
        try:
            request = requests.get(self.url, timeout=10)
        except requests.exceptions.RequestException as err:
            _LOGGER.error('No connection to endpoint: %s', err)
        else:
            doc: dict[str, Any] = xmltodict.parse(request.text)
            mtus: int = int(doc['LiveData']['System']['NumberMTU'])
            for mtu in range(1, mtus + 1):
                power: int = int(doc['LiveData']['Power'][f'MTU{mtu}']['PowerNow'])
                voltage_raw: int = int(doc['LiveData']['Voltage'][f'MTU{mtu}']['VoltageNow'])
                voltage: float = voltage_raw / 10
                self.data[mtu] = {UnitOfPower.WATT: power, UnitOfElectricPotential.VOLT: voltage}
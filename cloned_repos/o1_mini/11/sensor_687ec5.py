"""Support gathering system information of hosts which are running netdata."""
from __future__ import annotations
import logging
from netdata import Netdata
from netdata.exceptions import NetdataError
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_HOST, CONF_ICON, CONF_NAME, CONF_PORT, CONF_RESOURCES, PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from typing import Any, Dict, List, Optional, Union

_LOGGER = logging.getLogger(__name__)
CONF_DATA_GROUP = 'data_group'
CONF_ELEMENT = 'element'
CONF_INVERT = 'invert'
DEFAULT_HOST = 'localhost'
DEFAULT_NAME = 'Netdata'
DEFAULT_PORT = 19999
DEFAULT_ICON = 'mdi:desktop-classic'
RESOURCE_SCHEMA = vol.Any({
    vol.Required(CONF_DATA_GROUP): cv.string,
    vol.Required(CONF_ELEMENT): cv.string,
    vol.Optional(CONF_ICON, default=DEFAULT_ICON): cv.icon,
    vol.Optional(CONF_INVERT, default=False): cv.boolean
})
PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_HOST, default=DEFAULT_HOST): cv.string,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
    vol.Required(CONF_RESOURCES): vol.Schema({cv.string: RESOURCE_SCHEMA})
})

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the Netdata sensor."""
    name: str = config[CONF_NAME]
    host: str = config[CONF_HOST]
    port: int = config[CONF_PORT]
    resources: Dict[str, Any] = config[CONF_RESOURCES]
    netdata: NetdataData = NetdataData(
        Netdata(
            host,
            port=port,
            timeout=20.0,
            httpx_client=get_async_client(hass)
        )
    )
    await netdata.async_update()
    if netdata.api.metrics is None:
        raise PlatformNotReady
    dev: List[SensorEntity] = []
    for entry, data in resources.items():
        icon: str = data[CONF_ICON]
        sensor: str = data[CONF_DATA_GROUP]
        element: str = data[CONF_ELEMENT]
        invert: bool = data[CONF_INVERT]
        sensor_name: str = entry
        try:
            resource_data: Dict[str, Any] = netdata.api.metrics[sensor]
            unit: Optional[str] = PERCENTAGE if resource_data['units'] == 'percentage' else resource_data['units']
        except KeyError:
            _LOGGER.error('Sensor is not available: %s', sensor)
            continue
        dev.append(NetdataSensor(netdata, name, sensor, sensor_name, element, icon, unit, invert))
    dev.append(NetdataAlarms(netdata, name, host, port))
    async_add_entities(dev, True)

class NetdataSensor(SensorEntity):
    """Implementation of a Netdata sensor."""

    def __init__(
        self,
        netdata: NetdataData,
        name: str,
        sensor: str,
        sensor_name: str,
        element: str,
        icon: str,
        unit: Optional[str],
        invert: bool
    ) -> None:
        """Initialize the Netdata sensor."""
        self.netdata: NetdataData = netdata
        self._state: Optional[float] = None
        self._sensor: str = sensor
        self._element: str = element
        self._sensor_name: str = self._sensor if sensor_name is None else sensor_name
        self._name: str = name
        self._icon: str = icon
        self._unit_of_measurement: Optional[str] = unit
        self._invert: bool = invert

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return f'{self._name} {self._sensor_name}'

    @property
    def native_unit_of_measurement(self) -> Optional[str]:
        """Return the unit the value is expressed in."""
        return self._unit_of_measurement

    @property
    def icon(self) -> str:
        """Return the icon to use in the frontend, if any."""
        return self._icon

    @property
    def native_value(self) -> Optional[float]:
        """Return the state of the resources."""
        return self._state

    @property
    def available(self) -> bool:
        """Could the resource be accessed during the last update call."""
        return self.netdata.available

    async def async_update(self) -> None:
        """Get the latest data from Netdata REST API."""
        await self.netdata.async_update()
        resource_data: Dict[str, Any] = self.netdata.api.metrics.get(self._sensor, {})
        dimensions: Dict[str, Any] = resource_data.get('dimensions', {})
        element_data: Dict[str, Any] = dimensions.get(self._element, {})
        value: float = element_data.get('value', 0.0)
        self._state = round(value, 2) * (-1 if self._invert else 1)

class NetdataAlarms(SensorEntity):
    """Implementation of a Netdata alarm sensor."""

    def __init__(
        self,
        netdata: NetdataData,
        name: str,
        host: str,
        port: int
    ) -> None:
        """Initialize the Netdata alarm sensor."""
        self.netdata: NetdataData = netdata
        self._state: Optional[str] = None
        self._name: str = name
        self._host: str = host
        self._port: int = port

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return f'{self._name} Alarms'

    @property
    def native_value(self) -> Optional[str]:
        """Return the state of the resources."""
        return self._state

    @property
    def icon(self) -> str:
        """Status symbol if type is symbol."""
        if self._state == 'ok':
            return 'mdi:check'
        if self._state == 'warning':
            return 'mdi:alert-outline'
        if self._state == 'critical':
            return 'mdi:alert'
        return 'mdi:crosshairs-question'

    @property
    def available(self) -> bool:
        """Could the resource be accessed during the last update call."""
        return self.netdata.available

    async def async_update(self) -> None:
        """Get the latest alarms from Netdata REST API."""
        await self.netdata.async_update()
        alarms: Dict[str, Any] = self.netdata.api.alarms.get('alarms', {})
        self._state = None
        number_of_alarms: int = len(alarms)
        number_of_relevant_alarms: int = number_of_alarms
        _LOGGER.debug('Host %s has %s alarms', self.name, number_of_alarms)
        for alarm in alarms:
            alarm_data: Dict[str, Any] = alarms[alarm]
            if alarm_data.get('recipient') == 'silent' or alarm_data.get('status') in ('CLEAR', 'UNDEFINED', 'UNINITIALIZED'):
                number_of_relevant_alarms -= 1
            elif alarm_data.get('status') == 'CRITICAL':
                self._state = 'critical'
                return
        self._state = 'ok' if number_of_relevant_alarms == 0 else 'warning'

class NetdataData:
    """The class for handling the data retrieval."""

    def __init__(self, api: Netdata) -> None:
        """Initialize the data object."""
        self.api: Netdata = api
        self.available: bool = True

    async def async_update(self) -> None:
        """Get the latest data from the Netdata REST API."""
        try:
            await self.api.get_allmetrics()
            await self.api.get_alarms()
            self.available = True
        except NetdataError:
            _LOGGER.error('Unable to retrieve data from Netdata')
            self.available = False

"""Support for Dublin RTPI information from data.dublinked.ie.

For more info on the API see :
https://data.gov.ie/dataset/real-time-passenger-information-rtpi-for-dublin-bus-bus-eireann-luas-and-irish-rail/resource/4b9f2c4f-6bf5-4958-a43a-f12dab04cf61
"""
from __future__ import annotations
from contextlib import suppress
from datetime import datetime, timedelta
from http import HTTPStatus
from typing import Any, Dict, List, Optional, TypedDict, Union
import requests
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_NAME, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util

_RESOURCE = 'https://data.dublinked.ie/cgi-bin/rtpi/realtimebusinformation'
ATTR_STOP_ID = 'Stop ID'
ATTR_ROUTE = 'Route'
ATTR_DUE_IN = 'Due in'
ATTR_DUE_AT = 'Due at'
ATTR_NEXT_UP = 'Later Bus'
CONF_STOP_ID = 'stopid'
CONF_ROUTE = 'route'
DEFAULT_NAME = 'Next Bus'
SCAN_INTERVAL = timedelta(minutes=1)
TIME_STR_FORMAT = '%H:%M'

class BusData(TypedDict):
    ATTR_DUE_AT: str
    ATTR_ROUTE: str
    ATTR_DUE_IN: str

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_STOP_ID): cv.string,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_ROUTE, default=''): cv.string
})

def due_in_minutes(timestamp: str) -> str:
    """Get the time in minutes from a timestamp.

    The timestamp should be in the format day/month/year hour/minute/second
    """
    diff = datetime.strptime(timestamp, '%d/%m/%Y %H:%M:%S') - dt_util.now().replace(tzinfo=None)
    return str(int(diff.total_seconds() / 60))

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the Dublin public transport sensor."""
    name: str = config[CONF_NAME]
    stop: str = config[CONF_STOP_ID]
    route: str = config[CONF_ROUTE]
    data: PublicTransportData = PublicTransportData(stop, route)
    add_entities([DublinPublicTransportSensor(data, stop, route, name)], True)

class DublinPublicTransportSensor(SensorEntity):
    """Implementation of an Dublin public transport sensor."""
    _attr_attribution: str = 'Data provided by data.dublinked.ie'
    _attr_icon: str = 'mdi:bus'

    def __init__(self, data: PublicTransportData, stop: str, route: str, name: str) -> None:
        """Initialize the sensor."""
        self.data: PublicTransportData = data
        self._name: str = name
        self._stop: str = stop
        self._route: str = route
        self._times: Optional[List[Dict[str, str]]] = None
        self._state: Optional[str] = None

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def native_value(self) -> Optional[str]:
        """Return the state of the sensor."""
        return self._state

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, str]]:
        """Return the state attributes."""
        if self._times is not None:
            next_up: str = 'None'
            if len(self._times) > 1:
                next_up = f'{self._times[1][ATTR_ROUTE]} in '
                next_up += self._times[1][ATTR_DUE_IN]
            return {
                ATTR_DUE_IN: self._times[0][ATTR_DUE_IN],
                ATTR_DUE_AT: self._times[0][ATTR_DUE_AT],
                ATTR_STOP_ID: self._stop,
                ATTR_ROUTE: self._times[0][ATTR_ROUTE],
                ATTR_NEXT_UP: next_up
            }
        return None

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit this state is expressed in."""
        return UnitOfTime.MINUTES

    def update(self) -> None:
        """Get the latest data from opendata.ch and update the states."""
        self.data.update()
        self._times = self.data.info
        with suppress(TypeError):
            self._state = self._times[0][ATTR_DUE_IN]

class PublicTransportData:
    """The Class for handling the data retrieval."""

    def __init__(self, stop: str, route: str) -> None:
        """Initialize the data object."""
        self.stop: str = stop
        self.route: str = route
        self.info: List[Dict[str, str]] = [{ATTR_DUE_AT: 'n/a', ATTR_ROUTE: self.route, ATTR_DUE_IN: 'n/a'}]

    def update(self) -> None:
        """Get the latest data from opendata.ch."""
        params: Dict[str, Union[str, int]] = {}
        params['stopid'] = self.stop
        if self.route:
            params['routeid'] = self.route
        params['maxresults'] = 2
        params['format'] = 'json'
        response: requests.Response = requests.get(_RESOURCE, params, timeout=10)
        if response.status_code != HTTPStatus.OK:
            self.info = [{ATTR_DUE_AT: 'n/a', ATTR_ROUTE: self.route, ATTR_DUE_IN: 'n/a'}]
            return
        result: Dict[str, Any] = response.json()
        if str(result['errorcode']) != '0':
            self.info = [{ATTR_DUE_AT: 'n/a', ATTR_ROUTE: self.route, ATTR_DUE_IN: 'n/a'}]
            return
        self.info = []
        for item in result['results']:
            due_at: Optional[str] = item.get('departuredatetime')
            route: Optional[str] = item.get('route')
            if due_at is not None and route is not None:
                bus_data: Dict[str, str] = {
                    ATTR_DUE_AT: due_at,
                    ATTR_ROUTE: route,
                    ATTR_DUE_IN: due_in_minutes(due_at)
                }
                self.info.append(bus_data)
        if not self.info:
            self.info = [{ATTR_DUE_AT: 'n/a', ATTR_ROUTE: self.route, ATTR_DUE_IN: 'n/a'}]

from __future__ import annotations
from datetime import timedelta
import logging
from operator import itemgetter
import oasatelematics
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorDeviceClass, SensorEntity
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util
_LOGGER: logging.Logger = logging.getLogger(__name__)
ATTR_STOP_ID: str = 'stop_id'
ATTR_STOP_NAME: str = 'stop_name'
ATTR_ROUTE_ID: str = 'route_id'
ATTR_ROUTE_NAME: str = 'route_name'
ATTR_NEXT_ARRIVAL: str = 'next_arrival'
ATTR_SECOND_NEXT_ARRIVAL: str = 'second_next_arrival'
ATTR_NEXT_DEPARTURE: str = 'next_departure'
CONF_STOP_ID: str = 'stop_id'
CONF_ROUTE_ID: str = 'route_id'
DEFAULT_NAME: str = 'OASA Telematics'
SCAN_INTERVAL: timedelta = timedelta(seconds=60)
PLATFORM_SCHEMA: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend({vol.Required(CONF_STOP_ID): cv.string, vol.Required(CONF_ROUTE_ID): cv.string, vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string})

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    """Set up the OASA Telematics sensor."""
    name: str = config[CONF_NAME]
    stop_id: str = config[CONF_STOP_ID]
    route_id: str = config.get(CONF_ROUTE_ID)
    data: OASATelematicsData = OASATelematicsData(stop_id, route_id)
    add_entities([OASATelematicsSensor(data, stop_id, route_id, name)], True)

class OASATelematicsSensor(SensorEntity):
    """Implementation of the OASA Telematics sensor."""
    _attr_attribution: str = 'Data retrieved from telematics.oasa.gr'
    _attr_icon: str = 'mdi:bus'

    def __init__(self, data: OASATelematicsData, stop_id: str, route_id: str, name: str) -> None:
        """Initialize the sensor."""
        self.data: OASATelematicsData = data
        self._name: str = name
        self._stop_id: str = stop_id
        self._route_id: str = route_id
        self._name_data: any = self._times: any = self._state: any = None

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def device_class(self) -> SensorDeviceClass:
        """Return the class of this sensor."""
        return SensorDeviceClass.TIMESTAMP

    @property
    def native_value(self) -> any:
        """Return the state of the sensor."""
        return self._state

    @property
    def extra_state_attributes(self) -> dict:
        """Return the state attributes."""
        params: dict = {}
        if self._times is not None:
            next_arrival_data: dict = self._times[0]
            if ATTR_NEXT_ARRIVAL in next_arrival_data:
                next_arrival: timedelta = next_arrival_data[ATTR_NEXT_ARRIVAL]
                params.update({ATTR_NEXT_ARRIVAL: next_arrival.isoformat()})
            if len(self._times) > 1:
                second_next_arrival_time: timedelta = self._times[1][ATTR_NEXT_ARRIVAL]
                if second_next_arrival_time is not None:
                    second_arrival: timedelta = second_next_arrival_time
                    params.update({ATTR_SECOND_NEXT_ARRIVAL: second_arrival.isoformat()})
            params.update({ATTR_ROUTE_ID: self._times[0][ATTR_ROUTE_ID], ATTR_STOP_ID: self._stop_id})
        params.update({ATTR_ROUTE_NAME: self._name_data[ATTR_ROUTE_NAME], ATTR_STOP_NAME: self._name_data[ATTR_STOP_NAME]})
        return {k: v for k, v in params.items() if v}

    def update(self) -> None:
        """Get the latest data from OASA API and update the states."""
        self.data.update()
        self._times: list = self.data.info
        self._name_data: dict = self.data.name_data
        next_arrival_data: dict = self._times[0]
        if ATTR_NEXT_ARRIVAL in next_arrival_data:
            self._state: timedelta = next_arrival_data[ATTR_NEXT_ARRIVAL]

class OASATelematicsData:
    """The class for handling data retrieval."""

    def __init__(self, stop_id: str, route_id: str) -> None:
        """Initialize the data object."""
        self.stop_id: str = stop_id
        self.route_id: str = route_id
        self.info: list = self.empty_result()
        self.oasa_api: any = oasatelematics
        self.name_data: dict = {ATTR_ROUTE_NAME: self.get_route_name(), ATTR_STOP_NAME: self.get_stop_name()}

    def empty_result(self) -> list:
        """Object returned when no arrivals are found."""
        return [{ATTR_ROUTE_ID: self.route_id}]

    def get_route_name(self) -> str:
        """Get the route name from the API."""
        try:
            route: list = self.oasa_api.getRouteName(self.route_id)
            if route:
                return route[0].get('route_departure_eng')
        except TypeError:
            _LOGGER.error('Cannot get route name from OASA API')
        return None

    def get_stop_name(self) -> str:
        """Get the stop name from the API."""
        try:
            name_data: list = self.oasa_api.getStopNameAndXY(self.stop_id)
            if name_data:
                return name_data[0].get('stop_descr_matrix_eng')
        except TypeError:
            _LOGGER.error('Cannot get stop name from OASA API')
        return None

    def update(self) -> None:
        """Get the latest arrival data from telematics.oasa.gr API."""
        self.info: list = []
        results: list = self.oasa_api.getStopArrivals(self.stop_id)
        if not results:
            self.info = self.empty_result()
            return
        results = [r for r in results if r.get('route_code') in self.route_id]
        current_time: datetime = dt_util.utcnow()
        for result in results:
            if (btime2 := result.get('btime2')) is not None:
                arrival_min: int = int(btime2)
                timestamp: datetime = current_time + timedelta(minutes=arrival_min)
                arrival_data: dict = {ATTR_NEXT_ARRIVAL: timestamp, ATTR_ROUTE_ID: self.route_id}
                self.info.append(arrival_data)
        if not self.info:
            _LOGGER.debug('No arrivals with given parameters')
            self.info = self.empty_result()
            return
        sort: list = sorted(self.info, key=itemgetter(ATTR_NEXT_ARRIVAL))
        self.info = sort

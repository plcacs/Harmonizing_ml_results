"""Support for OASA Telematics from telematics.oasa.gr."""
from __future__ import annotations
from datetime import datetime, timedelta
import logging
from operator import itemgetter
from typing import Any, Optional, List, Dict

import oasatelematics
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorDeviceClass, SensorEntity
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)
ATTR_STOP_ID = 'stop_id'
ATTR_STOP_NAME = 'stop_name'
ATTR_ROUTE_ID = 'route_id'
ATTR_ROUTE_NAME = 'route_name'
ATTR_NEXT_ARRIVAL = 'next_arrival'
ATTR_SECOND_NEXT_ARRIVAL = 'second_next_arrival'
ATTR_NEXT_DEPARTURE = 'next_departure'
CONF_STOP_ID = 'stop_id'
CONF_ROUTE_ID = 'route_id'
DEFAULT_NAME = 'OASA Telematics'
SCAN_INTERVAL = timedelta(seconds=60)
PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_STOP_ID): cv.string,
    vol.Required(CONF_ROUTE_ID): cv.string,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string
})


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
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
        self._name_data: Optional[Dict[str, Optional[str]]] = None
        self._times: Optional[List[Dict[str, Any]]] = None
        self._state: Optional[datetime] = None

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def device_class(self) -> SensorDeviceClass:
        """Return the class of this sensor."""
        return SensorDeviceClass.TIMESTAMP

    @property
    def native_value(self) -> Optional[datetime]:
        """Return the state of the sensor."""
        return self._state

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        params: Dict[str, Any] = {}
        if self._times is not None:
            next_arrival_data: Dict[str, Any] = self._times[0]
            if ATTR_NEXT_ARRIVAL in next_arrival_data:
                next_arrival: datetime = next_arrival_data[ATTR_NEXT_ARRIVAL]
                params.update({ATTR_NEXT_ARRIVAL: next_arrival.isoformat()})
            if len(self._times) > 1:
                second_next_arrival_time: Optional[datetime] = self._times[1].get(ATTR_NEXT_ARRIVAL)
                if second_next_arrival_time is not None:
                    second_arrival: datetime = second_next_arrival_time
                    params.update({ATTR_SECOND_NEXT_ARRIVAL: second_arrival.isoformat()})
            params.update({
                ATTR_ROUTE_ID: self._times[0][ATTR_ROUTE_ID],
                ATTR_STOP_ID: self._stop_id
            })
        if self._name_data:
            params.update({
                ATTR_ROUTE_NAME: self._name_data.get(ATTR_ROUTE_NAME),
                ATTR_STOP_NAME: self._name_data.get(ATTR_STOP_NAME)
            })
        return {k: v for k, v in params.items() if v}

    def update(self) -> None:
        """Get the latest data from OASA API and update the states."""
        self.data.update()
        self._times = self.data.info
        self._name_data = self.data.name_data
        next_arrival_data: Optional[Dict[str, Any]] = self._times[0] if self._times else None
        if next_arrival_data and ATTR_NEXT_ARRIVAL in next_arrival_data:
            self._state = next_arrival_data[ATTR_NEXT_ARRIVAL]
        else:
            self._state = None


class OASATelematicsData:
    """The class for handling data retrieval."""

    def __init__(self, stop_id: str, route_id: str) -> None:
        """Initialize the data object."""
        self.stop_id: str = stop_id
        self.route_id: str = route_id
        self.info: List[Dict[str, Any]] = self.empty_result()
        self.oasa_api = oasatelematics
        self.name_data: Dict[str, Optional[str]] = {
            ATTR_ROUTE_NAME: self.get_route_name(),
            ATTR_STOP_NAME: self.get_stop_name()
        }

    def empty_result(self) -> List[Dict[str, Any]]:
        """Object returned when no arrivals are found."""
        return [{ATTR_ROUTE_ID: self.route_id}]

    def get_route_name(self) -> Optional[str]:
        """Get the route name from the API."""
        try:
            route = self.oasa_api.getRouteName(self.route_id)
            if route:
                return route[0].get('route_departure_eng')
        except TypeError:
            _LOGGER.error('Cannot get route name from OASA API')
        return None

    def get_stop_name(self) -> Optional[str]:
        """Get the stop name from the API."""
        try:
            name_data = self.oasa_api.getStopNameAndXY(self.stop_id)
            if name_data:
                return name_data[0].get('stop_descr_matrix_eng')
        except TypeError:
            _LOGGER.error('Cannot get stop name from OASA API')
        return None

    def update(self) -> None:
        """Get the latest arrival data from telematics.oasa.gr API."""
        self.info = []
        results: Optional[List[Dict[str, Any]]] = self.oasa_api.getStopArrivals(self.stop_id)
        if not results:
            self.info = self.empty_result()
            return
        filtered_results: List[Dict[str, Any]] = [r for r in results if r.get('route_code') in self.route_id]
        current_time: datetime = dt_util.utcnow()
        for result in filtered_results:
            btime2 = result.get('btime2')
            if btime2 is not None:
                arrival_min: int = int(btime2)
                timestamp: datetime = current_time + timedelta(minutes=arrival_min)
                arrival_data: Dict[str, Any] = {
                    ATTR_NEXT_ARRIVAL: timestamp,
                    ATTR_ROUTE_ID: self.route_id
                }
                self.info.append(arrival_data)
        if not self.info:
            _LOGGER.debug('No arrivals with given parameters')
            self.info = self.empty_result()
            return
        self.info.sort(key=itemgetter(ATTR_NEXT_ARRIVAL))

from __future__ import annotations
from contextlib import suppress
from datetime import datetime, timedelta
import logging
from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional
import rjpl
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_NAME, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)
ATTR_STOP_ID = 'stop_id'
ATTR_STOP_NAME = 'stop'
ATTR_ROUTE = 'route'
ATTR_TYPE = 'type'
ATTR_DIRECTION = 'direction'
ATTR_FINAL_STOP = 'final_stop'
ATTR_DUE_IN = 'due_in'
ATTR_DUE_AT = 'due_at'
ATTR_SCHEDULED_AT = 'scheduled_at'
ATTR_REAL_TIME_AT = 'real_time_at'
ATTR_TRACK = 'track'
ATTR_NEXT_UP = 'next_departures'
CONF_STOP_ID = 'stop_id'
CONF_ROUTE = 'route'
CONF_DIRECTION = 'direction'
CONF_DEPARTURE_TYPE = 'departure_type'
DEFAULT_NAME = 'Next departure'
SCAN_INTERVAL = timedelta(minutes=1)
BUS_TYPES = ['BUS', 'EXB', 'TB']
TRAIN_TYPES = ['LET', 'S', 'REG', 'IC', 'LYN', 'TOG']
METRO_TYPES = ['M']
PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_STOP_ID): cv.string,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_ROUTE, default=[]): vol.All(cv.ensure_list, [cv.string]),
    vol.Optional(CONF_DIRECTION, default=[]): vol.All(cv.ensure_list, [cv.string]),
    vol.Optional(CONF_DEPARTURE_TYPE, default=[]): vol.All(cv.ensure_list, [vol.In([*BUS_TYPES, *TRAIN_TYPES, *METRO_TYPES])])
})

def due_in_minutes(timestamp: str) -> int:
    """Get the time in minutes from a timestamp.

    The timestamp should be in the format day.month.year hour:minute
    """
    diff = datetime.strptime(timestamp, '%d.%m.%y %H:%M') - dt_util.now().replace(tzinfo=None)
    return int(diff.total_seconds() // 60)

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_devices: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the Rejseplanen transport sensor."""
    name: str = config[CONF_NAME]
    stop_id: str = config[CONF_STOP_ID]
    route: List[str] = config.get(CONF_ROUTE)
    direction: List[str] = config.get(CONF_DIRECTION)
    departure_type: List[str] = config.get(CONF_DEPARTURE_TYPE)
    data = PublicTransportData(stop_id, route, direction, departure_type)
    add_devices([RejseplanenTransportSensor(data, stop_id, route, direction, name)], True)

class RejseplanenTransportSensor(SensorEntity):
    """Implementation of Rejseplanen transport sensor."""
    _attr_attribution: str = 'Data provided by rejseplanen.dk'
    _attr_icon: str = 'mdi:bus'

    def __init__(self, data: PublicTransportData, stop_id: str, route: List[str], direction: List[str], name: str) -> None:
        """Initialize the sensor."""
        self.data: PublicTransportData = data
        self._name: str = name
        self._stop_id: str = stop_id
        self._route: List[str] = route
        self._direction: List[str] = direction
        self._times: List[Dict[str, Any]] | None = None
        self._state: int | None = None

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def native_value(self) -> int | None:
        """Return the state of the sensor."""
        return self._state

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        if not self._times:
            return {ATTR_STOP_ID: self._stop_id}
        next_up: List[Dict[str, Any]] = []
        if len(self._times) > 1:
            next_up = self._times[1:]
        attributes: Dict[str, Any] = {ATTR_NEXT_UP: next_up, ATTR_STOP_ID: self._stop_id}
        if self._times[0] is not None:
            attributes.update(self._times[0])
        return attributes

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit this state is expressed in."""
        return UnitOfTime.MINUTES

    def update(self) -> None:
        """Get the latest data from rejseplanen.dk and update the states."""
        self.data.update()
        self._times = self.data.info
        if not self._times:
            self._state = None
        else:
            with suppress(TypeError):
                self._state = self._times[0][ATTR_DUE_IN]

class PublicTransportData:
    """The Class for handling the data retrieval."""

    def __init__(self, stop_id: str, route: List[str], direction: List[str], departure_type: List[str]) -> None:
        """Initialize the data object."""
        self.stop_id: str = stop_id
        self.route: List[str] = route
        self.direction: List[str] = direction
        self.departure_type: List[str] = departure_type
        self.info: List[Dict[str, Any]] = []

    def update(self) -> None:
        """Get the latest data from rejseplanen."""
        self.info = []

        def intersection(lst1: List[Any], lst2: List[Any]) -> List[Any]:
            """Return items contained in both lists."""
            return list(set(lst1) & set(lst2))

        all_types: bool = not bool(self.departure_type)
        use_train: bool = all_types or bool(intersection(TRAIN_TYPES, self.departure_type))
        use_bus: bool = all_types or bool(intersection(BUS_TYPES, self.departure_type))
        use_metro: bool = all_types or bool(intersection(METRO_TYPES, self.departure_type))
        try:
            results: List[Dict[str, Any]] = rjpl.departureBoard(int(self.stop_id), timeout=5, useTrain=use_train, useBus=use_bus, useMetro=use_metro)
        except rjpl.rjplAPIError as error:
            _LOGGER.debug('API returned error: %s', error)
            return
        except (rjpl.rjplConnectionError, rjpl.rjplHTTPError):
            _LOGGER.debug('Error occurred while connecting to the API')
            return
        results = [d for d in results if 'cancelled' not in d]
        if self.route:
            results = [d for d in results if d['name'] in self.route]
        if self.direction:
            results = [d for d in results if d['direction'] in self.direction]
        if self.departure_type:
            results = [d for d in results if d['type'] in self.departure_type]
        for item in results:
            route: Optional[str] = item.get('name')
            scheduled_date: Optional[str] = item.get('date')
            scheduled_time: Optional[str] = item.get('time')
            real_time_date: Optional[str] = due_at_date = item.get('rtDate')
            real_time_time: Optional[str] = due_at_time = item.get('rtTime')
            if due_at_date is None:
                due_at_date = scheduled_date
            if due_at_time is None:
                due_at_time = scheduled_time
            if due_at_date is not None and due_at_time is not None and (route is not None):
                due_at: str = f'{due_at_date} {due_at_time}'
                scheduled_at: str = f'{scheduled_date} {scheduled_time}'
                departure_data: Dict[str, Any] = {
                    ATTR_DIRECTION: item.get('direction'),
                    ATTR_DUE_IN: due_in_minutes(due_at),
                    ATTR_DUE_AT: due_at,
                    ATTR_FINAL_STOP: item.get('finalStop'),
                    ATTR_ROUTE: route,
                    ATTR_SCHEDULED_AT: scheduled_at,
                    ATTR_STOP_NAME: item.get('stop'),
                    ATTR_TYPE: item.get('type')
                }
                if real_time_date is not None and real_time_time is not None:
                    departure_data[ATTR_REAL_TIME_AT] = f'{real_time_date} {real_time_time}'
                if item.get('rtTrack') is not None:
                    departure_data[ATTR_TRACK] = item.get('rtTrack')
                self.info.append(departure_data)
        if not self.info:
            _LOGGER.debug('No departures with given parameters')
        self.info = sorted(self.info, key=itemgetter(ATTR_DUE_IN))
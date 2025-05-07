"""Support for Västtrafik public transport."""
from __future__ import annotations
from datetime import datetime, timedelta
import logging
import vasttrafik
import voluptuous as vol
from typing import Optional, List, Dict, Any, Generator
from homeassistant.components.sensor import (
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    SensorEntity,
)
from homeassistant.const import CONF_DELAY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle
from homeassistant.util.dt import now

_LOGGER = logging.getLogger(__name__)

ATTR_ACCESSIBILITY = 'accessibility'
ATTR_DIRECTION = 'direction'
ATTR_LINE = 'line'
ATTR_TRACK = 'track'
ATTR_FROM = 'from'
ATTR_TO = 'to'
ATTR_DELAY = 'delay'

CONF_DEPARTURES = 'departures'
CONF_FROM = 'from'
CONF_HEADING = 'heading'
CONF_LINES = 'lines'
CONF_KEY = 'key'
CONF_SECRET = 'secret'

DEFAULT_DELAY = 0
MIN_TIME_BETWEEN_UPDATES = timedelta(seconds=120)

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_KEY): cv.string,
    vol.Required(CONF_SECRET): cv.string,
    vol.Required(CONF_DEPARTURES): [{
        vol.Required(CONF_FROM): cv.string,
        vol.Optional(CONF_DELAY, default=DEFAULT_DELAY): cv.positive_int,
        vol.Optional(CONF_HEADING): cv.string,
        vol.Optional(CONF_LINES, default=[]): vol.All(cv.ensure_list, [cv.string]),
        vol.Optional(CONF_NAME): cv.string
    }]
})


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the departure sensor."""
    planner = vasttrafik.JournyPlanner(
        config.get(CONF_KEY),
        config.get(CONF_SECRET)
    )
    departures = config.get(CONF_DEPARTURES, [])
    sensors: Generator[VasttrafikDepartureSensor, None, None] = (
        VasttrafikDepartureSensor(
            planner,
            departure.get(CONF_NAME),
            departure.get(CONF_FROM),
            departure.get(CONF_HEADING),
            departure.get(CONF_LINES),
            departure.get(CONF_DELAY)
        )
        for departure in departures
    )
    add_entities(sensors, True)


class VasttrafikDepartureSensor(SensorEntity):
    """Implementation of a Vasttrafik Departure Sensor."""

    _attr_attribution: str = 'Data provided by Västtrafik'
    _attr_icon: str = 'mdi:train'

    def __init__(
        self,
        planner: vasttrafik.JournyPlanner,
        name: Optional[str],
        departure: str,
        heading: Optional[str],
        lines: Optional[List[str]],
        delay: int
    ) -> None:
        """Initialize the sensor."""
        self._planner: vasttrafik.JournyPlanner = planner
        self._name: str = name or departure
        self._departure: Dict[str, str] = self.get_station_id(departure)
        self._heading: Optional[Dict[str, str]] = (
            self.get_station_id(heading) if heading else None
        )
        self._lines: Optional[List[str]] = lines if lines else None
        self._delay: timedelta = timedelta(minutes=delay)
        self._departureboard: Optional[List[Dict[str, Any]]] = None
        self._state: Optional[str] = None
        self._attributes: Dict[str, Any] = {}

    def get_station_id(self, location: str) -> Dict[str, str]:
        """Get the station ID."""
        if location.isdecimal():
            station_info: Dict[str, str] = {
                'station_name': location,
                'station_id': location
            }
        else:
            location_names: List[Dict[str, Any]] = self._planner.location_name(location)
            station_id: str = location_names[0]['gid']
            station_info: Dict[str, str] = {
                'station_name': location,
                'station_id': station_id
            }
        return station_info

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        return self._attributes

    @property
    def native_value(self) -> Optional[str]:
        """Return the next departure time."""
        return self._state

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> None:
        """Get the departure board."""
        try:
            self._departureboard = self._planner.departureboard(
                self._departure['station_id'],
                direction=self._heading['station_id'] if self._heading else None,
                date=now() + self._delay
            )
        except vasttrafik.Error:
            _LOGGER.debug('Unable to read departure board, updating token')
            self._planner.update_token()

        if not self._departureboard:
            _LOGGER.debug(
                'No departures from departure station %s to destination station %s',
                self._departure['station_name'],
                self._heading['station_name'] if self._heading else 'ANY'
            )
            self._state = None
            self._attributes = {}
        else:
            for departure in self._departureboard:
                service_journey: Dict[str, Any] = departure.get('serviceJourney', {})
                line: Dict[str, Any] = service_journey.get('line', {})
                if departure.get('isCancelled'):
                    continue
                if not self._lines or line.get('shortName') in self._lines:
                    if 'estimatedOtherwisePlannedTime' in departure:
                        estimated_time: str = departure['estimatedOtherwisePlannedTime']
                        try:
                            self._state = datetime.fromisoformat(estimated_time).strftime('%H:%M')
                        except ValueError:
                            self._state = estimated_time
                    else:
                        self._state = None
                    stop_point: Dict[str, Any] = departure.get('stopPoint', {})
                    params: Dict[str, Any] = {
                        ATTR_ACCESSIBILITY: 'wheelChair' if line.get('isWheelchairAccessible') else None,
                        ATTR_DIRECTION: service_journey.get('direction'),
                        ATTR_LINE: line.get('shortName'),
                        ATTR_TRACK: stop_point.get('platform'),
                        ATTR_FROM: stop_point.get('name'),
                        ATTR_TO: self._heading['station_name'] if self._heading else 'ANY',
                        ATTR_DELAY: self._delay.seconds // 60 % 60
                    }
                    self._attributes = {k: v for k, v in params.items() if v is not None}
                    break

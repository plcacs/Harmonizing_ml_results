"""Support for Västtrafik public transport."""
from __future__ import annotations
from datetime import datetime, timedelta
import logging
import vasttrafik
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_DELAY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle
from homeassistant.util.dt import now

_LOGGER: logging.Logger = logging.getLogger(__name__)

ATTR_ACCESSIBILITY: str = 'accessibility'
ATTR_DIRECTION: str = 'direction'
ATTR_LINE: str = 'line'
ATTR_TRACK: str = 'track'
ATTR_FROM: str = 'from'
ATTR_TO: str = 'to'
ATTR_DELAY: str = 'delay'

CONF_DEPARTURES: str = 'departures'
CONF_FROM: str = 'from'
CONF_HEADING: str = 'heading'
CONF_LINES: str = 'lines'
CONF_KEY: str = 'key'
CONF_SECRET: str = 'secret'

DEFAULT_DELAY: int = 0
MIN_TIME_BETWEEN_UPDATES: timedelta = timedelta(seconds=120)

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

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType | None = None) -> None:
    """Set up the departure sensor."""
    planner = vasttrafik.JournyPlanner(config.get(CONF_KEY), config.get(CONF_SECRET))
    add_entities(
        (
            VasttrafikDepartureSensor(
                planner,
                departure.get(CONF_NAME),
                departure.get(CONF_FROM),
                departure.get(CONF_HEADING),
                departure.get(CONF_LINES),
                departure.get(CONF_DELAY)
            )
            for departure in config[CONF_DEPARTURES]
        ),
        True
    )

class VasttrafikDepartureSensor(SensorEntity):
    """Implementation of a Vasttrafik Departure Sensor."""
    _attr_attribution: str = 'Data provided by Västtrafik'
    _attr_icon: str = 'mdi:train'

    def __init__(self, planner: vasttrafik.JournyPlanner, name: str | None, departure: str, heading: str | None, lines: list[str], delay: int) -> None:
        """Initialize the sensor."""
        self._planner = planner
        self._name = name or departure
        self._departure = self.get_station_id(departure)
        self._heading = self.get_station_id(heading) if heading else None
        self._lines = lines if lines else None
        self._delay = timedelta(minutes=delay)
        self._departureboard: list[dict] | None = None
        self._state: str | None = None
        self._attributes: dict[str, str | int | None] | None = None

    def get_station_id(self, location: str) -> dict[str, str]:
        """Get the station ID."""
        if location.isdecimal():
            station_info = {'station_name': location, 'station_id': location}
        else:
            station_id = self._planner.location_name(location)[0]['gid']
            station_info = {'station_name': location, 'station_id': station_id}
        return station_info

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def extra_state_attributes(self) -> dict[str, str | int | None] | None:
        """Return the state attributes."""
        return self._attributes

    @property
    def native_value(self) -> str | None:
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
                service_journey = departure.get('serviceJourney', {})
                line = service_journey.get('line', {})
                if departure.get('isCancelled'):
                    continue
                if not self._lines or line.get('shortName') in self._lines:
                    if 'estimatedOtherwisePlannedTime' in departure:
                        try:
                            self._state = datetime.fromisoformat(departure['estimatedOtherwisePlannedTime']).strftime('%H:%M')
                        except ValueError:
                            self._state = departure['estimatedOtherwisePlannedTime']
                    else:
                        self._state = None
                    stop_point = departure.get('stopPoint', {})
                    params = {
                        ATTR_ACCESSIBILITY: 'wheelChair' if line.get('isWheelchairAccessible') else None,
                        ATTR_DIRECTION: service_journey.get('direction'),
                        ATTR_LINE: line.get('shortName'),
                        ATTR_TRACK: stop_point.get('platform'),
                        ATTR_FROM: stop_point.get('name'),
                        ATTR_TO: self._heading['station_name'] if self._heading else 'ANY',
                        ATTR_DELAY: self._delay.seconds // 60 % 60
                    }
                    self._attributes = {k: v for k, v in params.items() if v}
                    break

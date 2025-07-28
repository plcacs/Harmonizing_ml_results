from __future__ import annotations
from datetime import datetime, timedelta, time as dt_time
import logging
import ns_api
from ns_api import RequestParametersError
import requests
import voluptuous as vol
from typing import Any, List, Optional, Union
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle

_LOGGER = logging.getLogger(__name__)
CONF_ROUTES = 'routes'
CONF_FROM = 'from'
CONF_TO = 'to'
CONF_VIA = 'via'
CONF_TIME = 'time'
MIN_TIME_BETWEEN_UPDATES = timedelta(seconds=120)

ROUTE_SCHEMA = vol.Schema({
    vol.Required(CONF_NAME): cv.string,
    vol.Required(CONF_FROM): cv.string,
    vol.Required(CONF_TO): cv.string,
    vol.Optional(CONF_VIA): cv.string,
    vol.Optional(CONF_TIME): cv.time,
})
ROUTES_SCHEMA = vol.All(cv.ensure_list, [ROUTE_SCHEMA])
PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_API_KEY): cv.string,
    vol.Optional(CONF_ROUTES): ROUTES_SCHEMA
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the departure sensor."""
    nsapi: ns_api.NSAPI = ns_api.NSAPI(config[CONF_API_KEY])
    try:
        stations: List[Any] = nsapi.get_stations()
    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as error:
        _LOGGER.error('Could not connect to the internet: %s', error)
        raise PlatformNotReady from error
    except RequestParametersError as error:
        _LOGGER.error('Could not fetch stations, please check configuration: %s', error)
        return
    sensors: List[NSDepartureSensor] = []
    for departure in config.get(CONF_ROUTES, {}):
        if not valid_stations(
            stations, 
            [departure.get(CONF_FROM), departure.get(CONF_VIA), departure.get(CONF_TO)]
        ):
            continue
        sensors.append(NSDepartureSensor(
            nsapi,
            departure.get(CONF_NAME),
            departure.get(CONF_FROM),
            departure.get(CONF_TO),
            departure.get(CONF_VIA),
            departure.get(CONF_TIME)
        ))
    add_entities(sensors, True)

def valid_stations(stations: List[Any], given_stations: List[Optional[str]]) -> bool:
    """Verify the existence of the given station codes."""
    for station in given_stations:
        if station is None:
            continue
        if not any((s.code == station.upper() for s in stations)):
            _LOGGER.warning("Station '%s' is not a valid station", station)
            return False
    return True

class NSDepartureSensor(SensorEntity):
    """Implementation of a NS Departure Sensor."""
    _attr_attribution: str = 'Data provided by NS'
    _attr_icon: str = 'mdi:train'

    def __init__(
        self, 
        nsapi: ns_api.NSAPI, 
        name: str, 
        departure: str, 
        heading: str, 
        via: Optional[str], 
        time: Optional[dt_time]
    ) -> None:
        """Initialize the sensor."""
        self._nsapi: ns_api.NSAPI = nsapi
        self._name: str = name
        self._departure: str = departure
        self._via: Optional[str] = via
        self._heading: str = heading
        self._time: Optional[dt_time] = time
        self._state: Optional[str] = None
        self._trips: Optional[List[Any]] = None

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def native_value(self) -> Optional[str]:
        """Return the next departure time."""
        return self._state

    @property
    def extra_state_attributes(self) -> Optional[dict]:
        """Return the state attributes."""
        if not self._trips:
            return None
        if self._trips[0].trip_parts:
            route: List[Any] = [self._trips[0].departure]
            route.extend((k.destination for k in self._trips[0].trip_parts))
        else:
            route = []
        attributes: dict = {
            'going': self._trips[0].going,
            'departure_time_planned': None,
            'departure_time_actual': None,
            'departure_delay': False,
            'departure_platform_planned': self._trips[0].departure_platform_planned,
            'departure_platform_actual': self._trips[0].departure_platform_actual,
            'arrival_time_planned': None,
            'arrival_time_actual': None,
            'arrival_delay': False,
            'arrival_platform_planned': self._trips[0].arrival_platform_planned,
            'arrival_platform_actual': self._trips[0].arrival_platform_actual,
            'next': None,
            'status': self._trips[0].status.lower(),
            'transfers': self._trips[0].nr_transfers,
            'route': route,
            'remarks': None
        }
        if self._trips[0].departure_time_planned is not None:
            attributes['departure_time_planned'] = self._trips[0].departure_time_planned.strftime('%H:%M')
        if self._trips[0].departure_time_actual is not None:
            attributes['departure_time_actual'] = self._trips[0].departure_time_actual.strftime('%H:%M')
        if (attributes['departure_time_planned'] and attributes['departure_time_actual'] and 
            (attributes['departure_time_planned'] != attributes['departure_time_actual'])):
            attributes['departure_delay'] = True
        if self._trips[0].arrival_time_planned is not None:
            attributes['arrival_time_planned'] = self._trips[0].arrival_time_planned.strftime('%H:%M')
        if self._trips[0].arrival_time_actual is not None:
            attributes['arrival_time_actual'] = self._trips[0].arrival_time_actual.strftime('%H:%M')
        if (attributes['arrival_time_planned'] and attributes['arrival_time_actual'] and 
            (attributes['arrival_time_planned'] != attributes['arrival_time_actual'])):
            attributes['arrival_delay'] = True
        if len(self._trips) > 1:
            if self._trips[1].departure_time_actual is not None:
                attributes['next'] = self._trips[1].departure_time_actual.strftime('%H:%M')
            elif self._trips[1].departure_time_planned is not None:
                attributes['next'] = self._trips[1].departure_time_planned.strftime('%H:%M')
        return attributes

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> None:
        """Get the trip information."""
        current_time: dt_time = datetime.now().time()
        if self._time and (
            (datetime.now() + timedelta(minutes=30)).time() < self._time or 
            (datetime.now() - timedelta(minutes=30)).time() > self._time
        ):
            self._state = None
            self._trips = None
            return
        if self._time:
            trip_time: str = datetime.today().replace(
                hour=self._time.hour, minute=self._time.minute
            ).strftime('%d-%m-%Y %H:%M')
        else:
            trip_time = datetime.now().strftime('%d-%m-%Y %H:%M')
        try:
            self._trips = self._nsapi.get_trips(trip_time, self._departure, self._via, self._heading, True, 0, 2)
            if self._trips:
                if self._trips[0].departure_time_actual is None:
                    planned_time = self._trips[0].departure_time_planned
                    self._state = planned_time.strftime('%H:%M')
                else:
                    actual_time = self._trips[0].departure_time_actual
                    self._state = actual_time.strftime('%H:%M')
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as error:
            _LOGGER.error("Couldn't fetch trip info: %s", error)
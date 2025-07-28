"""Support for UK public transport data provided by transportapi.com."""
from __future__ import annotations
from datetime import datetime, timedelta
from http import HTTPStatus
import logging
import re
from typing import Any, Dict, List, Optional, cast
import requests
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_MODE, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle, dt as dt_util
_LOGGER = logging.getLogger(__name__)
ATTR_ATCOCODE = 'atcocode'
ATTR_LOCALITY = 'locality'
ATTR_STOP_NAME = 'stop_name'
ATTR_REQUEST_TIME = 'request_time'
ATTR_NEXT_BUSES = 'next_buses'
ATTR_STATION_CODE = 'station_code'
ATTR_CALLING_AT = 'calling_at'
ATTR_NEXT_TRAINS = 'next_trains'
CONF_API_APP_KEY = 'app_key'
CONF_API_APP_ID = 'app_id'
CONF_QUERIES = 'queries'
CONF_ORIGIN = 'origin'
CONF_DESTINATION = 'destination'
_QUERY_SCHEME = vol.Schema({vol.Required(CONF_MODE): vol.All(cv.ensure_list, [vol.In(['bus', 'train'])]), vol.Required(CONF_ORIGIN): cv.string, vol.Required(CONF_DESTINATION): cv.string})
PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({vol.Required(CONF_API_APP_ID): cv.string, vol.Required(CONF_API_APP_KEY): cv.string, vol.Required(CONF_QUERIES): [_QUERY_SCHEME]})

def setup_platform(
    hass: HomeAssistant, 
    config: ConfigType, 
    add_entities: AddEntitiesCallback, 
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Get the uk_transport sensor."""
    sensors: List[SensorEntity] = []
    number_sensors = len((queries := config[CONF_QUERIES]))
    interval = timedelta(seconds=87 * number_sensors)
    api_app_id: str = config[CONF_API_APP_ID]
    api_app_key: str = config[CONF_API_APP_KEY]
    for query in queries:
        if 'bus' in query.get(CONF_MODE):
            stop_atcocode: str = query.get(CONF_ORIGIN)
            bus_direction: str = query.get(CONF_DESTINATION)
            sensors.append(UkTransportLiveBusTimeSensor(api_app_id, api_app_key, stop_atcocode, bus_direction, interval))
        elif 'train' in query.get(CONF_MODE):
            station_code: str = query.get(CONF_ORIGIN)
            calling_at: str = query.get(CONF_DESTINATION)
            sensors.append(UkTransportLiveTrainTimeSensor(api_app_id, api_app_key, station_code, calling_at, interval))
    add_entities(sensors, True)

class UkTransportSensor(SensorEntity):
    """Sensor that reads the UK transport web API.

    transportapi.com provides comprehensive transport data for UK train, tube
    and bus travel across the UK via simple JSON API. Subclasses of this
    base class can be used to access specific types of information.
    """
    TRANSPORT_API_URL_BASE = 'https://transportapi.com/v3/uk/'
    _attr_icon = 'mdi:train'
    _attr_native_unit_of_measurement = UnitOfTime.MINUTES

    def __init__(self, name: str, api_app_id: str, api_app_key: str, url: str) -> None:
        """Initialize the sensor."""
        self._data: Dict[str, Any] = {}
        self._api_app_id: str = api_app_id
        self._api_app_key: str = api_app_key
        self._url: str = self.TRANSPORT_API_URL_BASE + url
        self._name: str = name
        self._state: Optional[Any] = None

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def native_value(self) -> Optional[Any]:
        """Return the state of the sensor."""
        return self._state

    def _do_api_request(self, params: Dict[str, Any]) -> None:
        """Perform an API request."""
        request_params = dict({'app_id': self._api_app_id, 'app_key': self._api_app_key}, **params)
        response = requests.get(self._url, params=request_params, timeout=10)
        if response.status_code != HTTPStatus.OK:
            _LOGGER.warning('Invalid response from API')
        elif 'error' in response.json():
            if 'exceeded' in response.json()['error']:
                self._state = 'Usage limits exceeded'
            if 'invalid' in response.json()['error']:
                self._state = 'Credentials invalid'
        else:
            self._data = response.json()

class UkTransportLiveBusTimeSensor(UkTransportSensor):
    """Live bus time sensor from UK transportapi.com."""
    _attr_icon = 'mdi:bus'

    def __init__(self, api_app_id: str, api_app_key: str, stop_atcocode: str, bus_direction: str, interval: timedelta) -> None:
        """Construct a live bus time sensor."""
        self._stop_atcocode: str = stop_atcocode
        self._bus_direction: str = bus_direction
        self._next_buses: List[Dict[str, Any]] = []
        self._destination_re = re.compile(f'{bus_direction}', re.IGNORECASE)
        sensor_name = f'Next bus to {bus_direction}'
        stop_url = f'bus/stop/{stop_atcocode}/live.json'
        UkTransportSensor.__init__(self, sensor_name, api_app_id, api_app_key, stop_url)
        self.update = Throttle(interval)(self._update)

    def _update(self) -> None:
        """Get the latest live departure data for the specified stop."""
        params: Dict[str, Any] = {'group': 'route', 'nextbuses': 'no'}
        self._do_api_request(params)
        if self._data != {}:
            self._next_buses = []
            for route, departures in self._data['departures'].items():
                for departure in departures:
                    if self._destination_re.search(departure['direction']):
                        self._next_buses.append({'route': route, 'direction': departure['direction'], 'scheduled': departure['aimed_departure_time'], 'estimated': departure['best_departure_estimate']})
            if self._next_buses:
                self._state = min((_delta_mins(bus['scheduled']) for bus in self._next_buses))
            else:
                self._state = None

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        """Return other details about the sensor state."""
        if self._data is not None:
            attrs: Dict[str, Any] = {ATTR_NEXT_BUSES: self._next_buses}
            for key in (ATTR_ATCOCODE, ATTR_LOCALITY, ATTR_STOP_NAME, ATTR_REQUEST_TIME):
                attrs[key] = self._data.get(key)
            return attrs
        return None

class UkTransportLiveTrainTimeSensor(UkTransportSensor):
    """Live train time sensor from UK transportapi.com."""
    _attr_icon = 'mdi:train'

    def __init__(self, api_app_id: str, api_app_key: str, station_code: str, calling_at: str, interval: timedelta) -> None:
        """Construct a live bus time sensor."""
        self._station_code: str = station_code
        self._calling_at: str = calling_at
        self._next_trains: List[Dict[str, Any]] = []
        sensor_name = f'Next train to {calling_at}'
        query_url = f'train/station/{station_code}/live.json'
        UkTransportSensor.__init__(self, sensor_name, api_app_id, api_app_key, query_url)
        self.update = Throttle(interval)(self._update)

    def _update(self) -> None:
        """Get the latest live departure data for the specified stop."""
        params: Dict[str, Any] = {'darwin': 'false', 'calling_at': self._calling_at, 'train_status': 'passenger'}
        self._do_api_request(params)
        self._next_trains = []
        if self._data != {}:
            if self._data['departures']['all'] == []:
                self._state = 'No departures'
            else:
                for departure in self._data['departures']['all']:
                    self._next_trains.append({'origin_name': departure['origin_name'], 'destination_name': departure['destination_name'], 'status': departure['status'], 'scheduled': departure['aimed_departure_time'], 'estimated': departure['expected_departure_time'], 'platform': departure['platform'], 'operator_name': departure['operator_name']})
                if self._next_trains:
                    self._state = min((_delta_mins(train['scheduled']) for train in self._next_trains))
                else:
                    self._state = None

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        """Return other details about the sensor state."""
        if self._data is not None:
            attrs: Dict[str, Any] = {ATTR_STATION_CODE: self._station_code, ATTR_CALLING_AT: self._calling_at}
            if self._next_trains:
                attrs[ATTR_NEXT_TRAINS] = self._next_trains
            return attrs
        return None

def _delta_mins(hhmm_time_str: str) -> float:
    """Calculate time delta in minutes to a time in hh:mm format."""
    now = dt_util.now()
    hhmm_time = datetime.strptime(hhmm_time_str, '%H:%M')
    hhmm_datetime = now.replace(hour=hhmm_time.hour, minute=hhmm_time.minute)
    if hhmm_datetime < now:
        hhmm_datetime += timedelta(days=1)
    return (hhmm_datetime - now).total_seconds() // 60

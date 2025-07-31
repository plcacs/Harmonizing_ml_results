from __future__ import annotations
import asyncio
from datetime import timedelta
import logging
from typing import Any, Dict, List, Optional, Set, Union

from RMVtransport import RMVtransport
from RMVtransport.rmvtransport import RMVtransportApiConnectionError, RMVtransportDataError
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_NAME, CONF_TIMEOUT, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle

_LOGGER = logging.getLogger(__name__)

CONF_NEXT_DEPARTURE = 'next_departure'
CONF_STATION = 'station'
CONF_DESTINATIONS = 'destinations'
CONF_DIRECTION = 'direction'
CONF_LINES = 'lines'
CONF_PRODUCTS = 'products'
CONF_TIME_OFFSET = 'time_offset'
CONF_MAX_JOURNEYS = 'max_journeys'

DEFAULT_NAME = 'RMV Journey'
VALID_PRODUCTS = ['U-Bahn', 'Tram', 'Bus', 'S', 'RB', 'RE', 'EC', 'IC', 'ICE']
ICONS: Dict[Optional[str], str] = {
    'U-Bahn': 'mdi:subway',
    'Tram': 'mdi:tram',
    'Bus': 'mdi:bus',
    'S': 'mdi:train',
    'RB': 'mdi:train',
    'RE': 'mdi:train',
    'EC': 'mdi:train',
    'IC': 'mdi:train',
    'ICE': 'mdi:train',
    'SEV': 'mdi:checkbox-blank-circle-outline',
    None: 'mdi:clock'
}
ATTRIBUTION = 'Data provided by opendata.rmv.de'
SCAN_INTERVAL = timedelta(seconds=60)

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_NEXT_DEPARTURE): [{
        vol.Required(CONF_STATION): cv.string,
        vol.Optional(CONF_DESTINATIONS, default=[]): vol.All(cv.ensure_list, [cv.string]),
        vol.Optional(CONF_DIRECTION): cv.string,
        vol.Optional(CONF_LINES, default=[]): vol.All(cv.ensure_list, [cv.positive_int, cv.string]),
        vol.Optional(CONF_PRODUCTS, default=VALID_PRODUCTS): vol.All(cv.ensure_list, [vol.In(VALID_PRODUCTS)]),
        vol.Optional(CONF_TIME_OFFSET, default=0): cv.positive_int,
        vol.Optional(CONF_MAX_JOURNEYS, default=5): cv.positive_int,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string
    }],
    vol.Optional(CONF_TIMEOUT, default=10): cv.positive_int
})


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    timeout: int = config.get(CONF_TIMEOUT)
    sensors: List[RMVDepartureSensor] = [
        RMVDepartureSensor(
            next_departure[CONF_STATION],
            next_departure.get(CONF_DESTINATIONS),
            next_departure.get(CONF_DIRECTION),
            next_departure.get(CONF_LINES),
            next_departure.get(CONF_PRODUCTS),
            next_departure.get(CONF_TIME_OFFSET),
            next_departure.get(CONF_MAX_JOURNEYS),
            next_departure.get(CONF_NAME),
            timeout
        )
        for next_departure in config[CONF_NEXT_DEPARTURE]
    ]
    tasks = [asyncio.create_task(sensor.async_update()) for sensor in sensors]
    if tasks:
        await asyncio.wait(tasks)
    if not any((sensor.data.departures for sensor in sensors)):
        raise PlatformNotReady
    async_add_entities(sensors)


class RMVDepartureSensor(SensorEntity):
    _attr_attribution: str = ATTRIBUTION

    def __init__(
        self,
        station: str,
        destinations: Optional[List[str]],
        direction: Optional[str],
        lines: Optional[List[Union[int, str]]],
        products: List[str],
        time_offset: int,
        max_journeys: int,
        name: str,
        timeout: int
    ) -> None:
        self._station: str = station
        self._name: str = name
        self._state: Optional[Any] = None
        self.data: RMVDepartureData = RMVDepartureData(
            station, destinations or [], direction, lines or [],
            products, time_offset, max_journeys, timeout
        )
        self._icon: str = ICONS[None]

    @property
    def name(self) -> str:
        return self._name

    @property
    def available(self) -> bool:
        return self._state is not None

    @property
    def native_value(self) -> Optional[Any]:
        return self._state

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        try:
            return {
                'next_departures': self.data.departures[1:],
                'direction': self.data.departures[0].get('direction'),
                'line': self.data.departures[0].get('line'),
                'minutes': self.data.departures[0].get('minutes'),
                'departure_time': self.data.departures[0].get('departure_time'),
                'product': self.data.departures[0].get('product')
            }
        except IndexError:
            return {}

    @property
    def icon(self) -> str:
        return self._icon

    @property
    def native_unit_of_measurement(self) -> str:
        return UnitOfTime.MINUTES

    async def async_update(self) -> None:
        await self.data.async_update()
        if self._name == DEFAULT_NAME:
            self._name = self.data.station or self._name
        self._station = self.data.station or self._station
        if not self.data.departures:
            self._state = None
            self._icon = ICONS[None]
            return
        self._state = self.data.departures[0].get('minutes')
        self._icon = ICONS.get(self.data.departures[0].get('product'), ICONS[None])


class RMVDepartureData:
    def __init__(
        self,
        station_id: str,
        destinations: List[str],
        direction: Optional[str],
        lines: List[Union[int, str]],
        products: List[str],
        time_offset: int,
        max_journeys: int,
        timeout: int
    ) -> None:
        self.station: Optional[str] = None
        self._station_id: str = station_id
        self._destinations: List[str] = destinations
        self._direction: Optional[str] = direction
        self._lines: List[Union[int, str]] = lines
        self._products: List[str] = products
        self._time_offset: int = time_offset
        self._max_journeys: int = max_journeys
        self.rmv: RMVtransport = RMVtransport(timeout)
        self.departures: List[Dict[str, Any]] = []
        self._error_notification: bool = False

    @Throttle(SCAN_INTERVAL)
    async def async_update(self) -> None:
        try:
            _data: Dict[str, Any] = await self.rmv.get_departures(
                self._station_id,
                products=self._products,
                direction_id=self._direction,
                max_journeys=50
            )
        except (RMVtransportApiConnectionError, RMVtransportDataError):
            self.departures = []
            _LOGGER.warning('Could not retrieve data from rmv.de')
            return

        self.station = _data.get('station')
        _deps: List[Dict[str, Any]] = []
        _deps_not_found: Set[str] = set(self._destinations)
        for journey in _data.get('journeys', []):
            _nextdep: Dict[str, Any] = {}
            if self._destinations:
                dest_found: bool = False
                for dest in self._destinations:
                    if dest in journey.get('stops', []):
                        dest_found = True
                        if dest in _deps_not_found:
                            _deps_not_found.remove(dest)
                        _nextdep['destination'] = dest
                if not dest_found:
                    continue
            if self._lines and journey.get('number') not in self._lines or journey.get('minutes', 0) < self._time_offset:
                continue
            for attr in ('direction', 'departure_time', 'product', 'minutes'):
                _nextdep[attr] = journey.get(attr, '')
            _nextdep['line'] = journey.get('number', '')
            _deps.append(_nextdep)
            if len(_deps) > self._max_journeys:
                break
        if not self._error_notification and _deps_not_found:
            self._error_notification = True
            _LOGGER.warning('Destination(s) %s not found', ', '.join(_deps_not_found))
        self.departures = _deps
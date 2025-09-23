from __future__ import annotations
from copy import deepcopy
from datetime import timedelta
import logging
from typing import Any, Dict, List, Optional, Union
import MVGLive
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_NAME, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER = logging.getLogger(__name__)

CONF_NEXT_DEPARTURE = 'nextdeparture'
CONF_STATION = 'station'
CONF_DESTINATIONS = 'destinations'
CONF_DIRECTIONS = 'directions'
CONF_LINES = 'lines'
CONF_PRODUCTS = 'products'
CONF_TIMEOFFSET = 'timeoffset'
CONF_NUMBER = 'number'

DEFAULT_PRODUCT = ['U-Bahn', 'Tram', 'Bus', 'ExpressBus', 'S-Bahn', 'Nachteule']
ICONS: Dict[str, str] = {
    'U-Bahn': 'mdi:subway',
    'Tram': 'mdi:tram',
    'Bus': 'mdi:bus',
    'ExpressBus': 'mdi:bus',
    'S-Bahn': 'mdi:train',
    'Nachteule': 'mdi:owl',
    'SEV': 'mdi:checkbox-blank-circle-outline',
    '-': 'mdi:clock'
}
ATTRIBUTION = 'Data provided by MVG-live.de'
SCAN_INTERVAL = timedelta(seconds=30)

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_NEXT_DEPARTURE): [{
        vol.Required(CONF_STATION): cv.string,
        vol.Optional(CONF_DESTINATIONS, default=['']): cv.ensure_list_csv,
        vol.Optional(CONF_DIRECTIONS, default=['']): cv.ensure_list_csv,
        vol.Optional(CONF_LINES, default=['']): cv.ensure_list_csv,
        vol.Optional(CONF_PRODUCTS, default=DEFAULT_PRODUCT): cv.ensure_list_csv,
        vol.Optional(CONF_TIMEOFFSET, default=0): cv.positive_int,
        vol.Optional(CONF_NUMBER, default=1): cv.positive_int,
        vol.Optional(CONF_NAME): cv.string
    }]
})


def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback,
                   discovery_info: Optional[DiscoveryInfoType] = None) -> None:
    add_entities(
        (
            MVGLiveSensor(
                nextdeparture.get(CONF_STATION),
                nextdeparture.get(CONF_DESTINATIONS),
                nextdeparture.get(CONF_DIRECTIONS),
                nextdeparture.get(CONF_LINES),
                nextdeparture.get(CONF_PRODUCTS),
                nextdeparture.get(CONF_TIMEOFFSET),
                nextdeparture.get(CONF_NUMBER),
                nextdeparture.get(CONF_NAME)
            )
            for nextdeparture in config[CONF_NEXT_DEPARTURE]
        ),
        True
    )


class MVGLiveSensor(SensorEntity):
    _attr_attribution: str = ATTRIBUTION

    def __init__(self, station: str, destinations: List[str], directions: List[str],
                 lines: List[str], products: List[str], timeoffset: int, number: int,
                 name: Optional[str]) -> None:
        self._station: str = station
        self._name: Optional[str] = name
        self.data: MVGLiveData = MVGLiveData(station, destinations, directions, lines, products, timeoffset, number)
        self._state: Optional[Union[int, str]] = None
        self._icon: str = ICONS['-']

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        return self._station

    @property
    def native_value(self) -> Optional[Union[int, str]]:
        return self._state

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        if not (dep := self.data.departures):
            return None
        attr: Dict[str, Any] = dep[0]
        attr['departures'] = deepcopy(dep)
        return attr

    @property
    def icon(self) -> str:
        return self._icon

    @property
    def native_unit_of_measurement(self) -> str:
        return UnitOfTime.MINUTES

    def update(self) -> None:
        self.data.update()
        if not self.data.departures:
            self._state = '-'
            self._icon = ICONS['-']
        else:
            self._state = self.data.departures[0].get('time', '-')
            self._icon = ICONS[self.data.departures[0].get('product', '-')] 


class MVGLiveData:
    def __init__(self, station: str, destinations: List[str], directions: List[str],
                 lines: List[str], products: List[str], timeoffset: int, number: int) -> None:
        self._station: str = station
        self._destinations: List[str] = destinations
        self._directions: List[str] = directions
        self._lines: List[str] = lines
        self._products: List[str] = products
        self._timeoffset: int = timeoffset
        self._number: int = number
        self._include_ubahn: bool = 'U-Bahn' in self._products
        self._include_tram: bool = 'Tram' in self._products
        self._include_bus: bool = 'Bus' in self._products
        self._include_sbahn: bool = 'S-Bahn' in self._products
        self.mvg = MVGLive.MVGLive()
        self.departures: List[Dict[str, Any]] = []

    def update(self) -> None:
        try:
            _departures: List[Dict[str, Any]] = self.mvg.getlivedata(
                station=self._station,
                timeoffset=self._timeoffset,
                ubahn=self._include_ubahn,
                tram=self._include_tram,
                bus=self._include_bus,
                sbahn=self._include_sbahn
            )
        except ValueError:
            self.departures = []
            _LOGGER.warning('Returned data not understood')
            return
        self.departures = []
        for i, _departure in enumerate(_departures):
            if '' not in self._destinations[:1] and _departure['destination'] not in self._destinations:
                continue
            if '' not in self._directions[:1] and _departure['direction'] not in self._directions:
                continue
            if '' not in self._lines[:1] and _departure['linename'] not in self._lines:
                continue
            if _departure['time'] < self._timeoffset:
                continue
            _nextdep: Dict[str, Any] = {}
            for k in ('destination', 'linename', 'time', 'direction', 'product'):
                _nextdep[k] = _departure.get(k, '')
            _nextdep['time'] = int(_nextdep['time'])
            self.departures.append(_nextdep)
            if i == self._number - 1:
                break
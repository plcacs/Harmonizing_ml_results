"""Support for departure information for public transport in Munich."""
from __future__ import annotations
from copy import deepcopy
from datetime import timedelta
import logging
from typing import Any, Dict, List, Optional, cast

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

ICONS = {
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

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the MVGLive sensor."""
    next_departures: List[Dict[str, Any]] = config[CONF_NEXT_DEPARTURE]
    sensors = [
        MVGLiveSensor(
            cast(str, nextdeparture.get(CONF_STATION)),
            cast(List[str], nextdeparture.get(CONF_DESTINATIONS)),
            cast(List[str], nextdeparture.get(CONF_DIRECTIONS)),
            cast(List[str], nextdeparture.get(CONF_LINES)),
            cast(List[str], nextdeparture.get(CONF_PRODUCTS)),
            cast(int, nextdeparture.get(CONF_TIMEOFFSET)),
            cast(int, nextdeparture.get(CONF_NUMBER)),
            cast(Optional[str], nextdeparture.get(CONF_NAME))
        )
        for nextdeparture in next_departures
    ]
    add_entities(sensors, True)

class MVGLiveSensor(SensorEntity):
    """Implementation of an MVG Live sensor."""
    _attr_attribution = ATTRIBUTION

    def __init__(
        self,
        station: str,
        destinations: List[str],
        directions: List[str],
        lines: List[str],
        products: List[str],
        timeoffset: int,
        number: int,
        name: Optional[str]
    ) -> None:
        """Initialize the sensor."""
        self._station = station
        self._name = name
        self.data = MVGLiveData(station, destinations, directions, lines, products, timeoffset, number)
        self._state: Optional[str] = None
        self._icon = ICONS['-']

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        if self._name:
            return self._name
        return self._station

    @property
    def native_value(self) -> Optional[str]:
        """Return the next departure time."""
        return self._state

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        """Return the state attributes."""
        if not (dep := self.data.departures):
            return None
        attr: Dict[str, Any] = dep[0].copy()
        attr['departures'] = deepcopy(dep)
        return attr

    @property
    def icon(self) -> str:
        """Icon to use in the frontend, if any."""
        return self._icon

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit this state is expressed in."""
        return UnitOfTime.MINUTES

    def update(self) -> None:
        """Get the latest data and update the state."""
        self.data.update()
        if not self.data.departures:
            self._state = '-'
            self._icon = ICONS['-']
        else:
            self._state = str(self.data.departures[0].get('time', '-'))
            self._icon = ICONS.get(self.data.departures[0].get('product', '-'), ICONS['-'])

class MVGLiveData:
    """Pull data from the mvg-live.de web page."""

    def __init__(
        self,
        station: str,
        destinations: List[str],
        directions: List[str],
        lines: List[str],
        products: List[str],
        timeoffset: int,
        number: int
    ) -> None:
        """Initialize the sensor."""
        self._station = station
        self._destinations = destinations
        self._directions = directions
        self._lines = lines
        self._products = products
        self._timeoffset = timeoffset
        self._number = number
        self._include_ubahn = 'U-Bahn' in self._products
        self._include_tram = 'Tram' in self._products
        self._include_bus = 'Bus' in self._products
        self._include_sbahn = 'S-Bahn' in self._products
        self.mvg = MVGLive.MVGLive()
        self.departures: List[Dict[str, Any]] = []

    def update(self) -> None:
        """Update the connection data."""
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

"""Support for departure information for public transport in Munich."""
from __future__ import annotations
from copy import deepcopy
from datetime import timedelta
import logging
import MVGLive
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_NAME, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger = logging.getLogger(__name__)

CONF_NEXT_DEPARTURE: str = 'nextdeparture'
CONF_STATION: str = 'station'
CONF_DESTINATIONS: str = 'destinations'
CONF_DIRECTIONS: str = 'directions'
CONF_LINES: str = 'lines'
CONF_PRODUCTS: str = 'products'
CONF_TIMEOFFSET: str = 'timeoffset'
CONF_NUMBER: str = 'number'

DEFAULT_PRODUCT: list[str] = ['U-Bahn', 'Tram', 'Bus', 'ExpressBus', 'S-Bahn', 'Nachteule']
ICONS: dict[str, str] = {
    'U-Bahn': 'mdi:subway',
    'Tram': 'mdi:tram',
    'Bus': 'mdi:bus',
    'ExpressBus': 'mdi:bus',
    'S-Bahn': 'mdi:train',
    'Nachteule': 'mdi:owl',
    'SEV': 'mdi:checkbox-blank-circle-outline',
    '-': 'mdi:clock'
}
ATTRIBUTION: str = 'Data provided by MVG-live.de'
SCAN_INTERVAL: timedelta = timedelta(seconds=30)

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
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    """Set up the MVGLive sensor."""
    add_entities((
        MVGLiveSensor(
            nextdeparture.get(CONF_STATION),
            nextdeparture.get(CONF_DESTINATIONS),
            nextdeparture.get(CONF_DIRECTIONS),
            nextdeparture.get(CONF_LINES),
            nextdeparture.get(CONF_PRODUCTS),
            nextdeparture.get(CONF_TIMEOFFSET),
            nextdeparture.get(CONF_NUMBER),
            nextdeparture.get(CONF_NAME)
        ) for nextdeparture in config[CONF_NEXT_DEPARTURE]
    ), True)

class MVGLiveSensor(SensorEntity):
    """Implementation of an MVG Live sensor."""
    _attr_attribution: str = ATTRIBUTION

    def __init__(
        self,
        station: str,
        destinations: list[str],
        directions: list[str],
        lines: list[str],
        products: list[str],
        timeoffset: int,
        number: int,
        name: str | None
    ) -> None:
        """Initialize the sensor."""
        self._station: str = station
        self._name: str | None = name
        self.data: MVGLiveData = MVGLiveData(station, destinations, directions, lines, products, timeoffset, number)
        self._state: str | None = None
        self._icon: str = ICONS['-']

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        if self._name:
            return self._name
        return self._station

    @property
    def native_value(self) -> str | None:
        """Return the next departure time."""
        return self._state

    @property
    def extra_state_attributes(self) -> dict | None:
        """Return the state attributes."""
        if not (dep := self.data.departures):
            return None
        attr: dict = dep[0]
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
            self._state = self.data.departures[0].get('time', '-')
            self._icon = ICONS[self.data.departures[0].get('product', '-')]

class MVGLiveData:
    """Pull data from the mvg-live.de web page."""

    def __init__(
        self,
        station: str,
        destinations: list[str],
        directions: list[str],
        lines: list[str],
        products: list[str],
        timeoffset: int,
        number: int
    ) -> None:
        """Initialize the sensor."""
        self._station: str = station
        self._destinations: list[str] = destinations
        self._directions: list[str] = directions
        self._lines: list[str] = lines
        self._products: list[str] = products
        self._timeoffset: int = timeoffset
        self._number: int = number
        self._include_ubahn: bool = 'U-Bahn' in self._products
        self._include_tram: bool = 'Tram' in self._products
        self._include_bus: bool = 'Bus' in self._products
        self._include_sbahn: bool = 'S-Bahn' in self._products
        self.mvg: MVGLive.MVGLive = MVGLive.MVGLive()
        self.departures: list[dict] = []

    def update(self) -> None:
        """Update the connection data."""
        try:
            _departures: list[dict] = self.mvg.getlivedata(
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
            _nextdep: dict = {}
            for k in ('destination', 'linename', 'time', 'direction', 'product'):
                _nextdep[k] = _departure.get(k, '')
            _nextdep['time'] = int(_nextdep['time'])
            self.departures.append(_nextdep)
            if i == self._number - 1:
                break

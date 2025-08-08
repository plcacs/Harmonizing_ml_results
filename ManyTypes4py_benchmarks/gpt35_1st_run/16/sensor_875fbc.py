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
from typing import List, Dict, Any

_LOGGER: logging.Logger

CONF_NEXT_DEPARTURE: str = 'nextdeparture'
CONF_STATION: str = 'station'
CONF_DESTINATIONS: str = 'destinations'
CONF_DIRECTIONS: str = 'directions'
CONF_LINES: str = 'lines'
CONF_PRODUCTS: str = 'products'
CONF_TIMEOFFSET: str = 'timeoffset'
CONF_NUMBER: str = 'number'
DEFAULT_PRODUCT: List[str] = ['U-Bahn', 'Tram', 'Bus', 'ExpressBus', 'S-Bahn', 'Nachteule']
ICONS: Dict[str, str] = {'U-Bahn': 'mdi:subway', 'Tram': 'mdi:tram', 'Bus': 'mdi:bus', 'ExpressBus': 'mdi:bus', 'S-Bahn': 'mdi:train', 'Nachteule': 'mdi:owl', 'SEV': 'mdi:checkbox-blank-circle-outline', '-': 'mdi:clock'}
ATTRIBUTION: str = 'Data provided by MVG-live.de'
SCAN_INTERVAL: timedelta = timedelta(seconds=30)
PLATFORM_SCHEMA: vol.Schema

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

class MVGLiveSensor(SensorEntity):

    _attr_attribution: str

    def __init__(self, station: str, destinations: List[str], directions: List[str], lines: List[str], products: List[str], timeoffset: int, number: int, name: str) -> None:

    @property
    def name(self) -> str:

    @property
    def native_value(self) -> Any:

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:

    @property
    def icon(self) -> str:

    @property
    def native_unit_of_measurement(self) -> UnitOfTime:

    def update(self) -> None:

class MVGLiveData:

    def __init__(self, station: str, destinations: List[str], directions: List[str], lines: List[str], products: List[str], timeoffset: int, number: int) -> None:

    def update(self) -> None:

"""Support for the NOAA Tides and Currents API."""
from __future__ import annotations
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Optional, List, Dict
import noaa_coops as coops
import requests
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_NAME, CONF_TIME_ZONE, CONF_UNIT_SYSTEM
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util.unit_system import METRIC_SYSTEM
from .helpers import get_station_unique_id
if TYPE_CHECKING:
    from pandas import DataFrame, Timestamp
_LOGGER = logging.getLogger(__name__)
CONF_STATION_ID = 'station_id'
DEFAULT_NAME = 'NOAA Tides'
DEFAULT_TIMEZONE = 'lst_ldt'
SCAN_INTERVAL = timedelta(minutes=60)
TIMEZONES = ['gmt', 'lst', 'lst_ldt']
UNIT_SYSTEMS = ['english', 'metric']
PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({vol.Required(CONF_STATION_ID): cv.string, vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string, vol.Optional(CONF_TIME_ZONE, default=DEFAULT_TIMEZONE): vol.In(TIMEZONES), vol.Optional(CONF_UNIT_SYSTEM): vol.In(UNIT_SYSTEMS)})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the NOAA Tides and Currents sensor."""
    station_id: str = config[CONF_STATION_ID]
    name: str = config.get(CONF_NAME, DEFAULT_NAME)
    timezone: str = config.get(CONF_TIME_ZONE, DEFAULT_TIMEZONE)
    unit_system: str
    if CONF_UNIT_SYSTEM in config:
        unit_system = config[CONF_UNIT_SYSTEM]
    elif hass.config.units is METRIC_SYSTEM:
        unit_system = UNIT_SYSTEMS[1]
    else:
        unit_system = UNIT_SYSTEMS[0]
    try:
        station: coops.Station = coops.Station(station_id, unit_system)
    except KeyError:
        _LOGGER.error('NOAA Tides Sensor station_id %s does not exist', station_id)
        return
    except requests.exceptions.ConnectionError as exception:
        _LOGGER.error('Connection error during setup in NOAA Tides Sensor for station_id: %s', station_id)
        raise PlatformNotReady from exception
    noaa_sensor: NOAATidesAndCurrentsSensor = NOAATidesAndCurrentsSensor(name, station_id, timezone, unit_system, station)
    add_entities([noaa_sensor], True)

class NOAATidesData(TypedDict):
    """Representation of a single tide."""
    time_stamp: List[Timestamp]
    hi_lo: List[str]
    predicted_wl: List[float]

class NOAATidesAndCurrentsSensor(SensorEntity):
    """Representation of a NOAA Tides and Currents sensor."""
    _attr_attribution: str = 'Data provided by NOAA'

    def __init__(
        self,
        name: str,
        station_id: str,
        timezone: str,
        unit_system: str,
        station: coops.Station
    ) -> None:
        """Initialize the sensor."""
        self._name: str = name
        self._station_id: str = station_id
        self._timezone: str = timezone
        self._unit_system: str = unit_system
        self._station: coops.Station = station
        self.data: Optional[NOAATidesData] = None
        self._attr_unique_id: str = f'{get_station_unique_id(station_id)}_summary'

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes of this device."""
        attr: Dict[str, Any] = {}
        if self.data is None:
            return attr
        if self.data['hi_lo'][1] == 'H':
            attr['high_tide_time'] = self.data['time_stamp'][1].strftime('%Y-%m-%dT%H:%M')
            attr['high_tide_height'] = self.data['predicted_wl'][1]
            attr['low_tide_time'] = self.data['time_stamp'][2].strftime('%Y-%m-%dT%H:%M')
            attr['low_tide_height'] = self.data['predicted_wl'][2]
        elif self.data['hi_lo'][1] == 'L':
            attr['low_tide_time'] = self.data['time_stamp'][1].strftime('%Y-%m-%dT%H:%M')
            attr['low_tide_height'] = self.data['predicted_wl'][1]
            attr['high_tide_time'] = self.data['time_stamp'][2].strftime('%Y-%m-%dT%H:%M')
            attr['high_tide_height'] = self.data['predicted_wl'][2]
        return attr

    @property
    def native_value(self) -> Optional[str]:
        """Return the state of the device."""
        if self.data is None:
            return None
        api_time: Timestamp = self.data['time_stamp'][0]
        if self.data['hi_lo'][0] == 'H':
            tidetime: str = api_time.strftime('%-I:%M %p')
            return f'High tide at {tidetime}'
        if self.data['hi_lo'][0] == 'L':
            tidetime: str = api_time.strftime('%-I:%M %p')
            return f'Low tide at {tidetime}'
        return None

    def update(self) -> None:
        """Get the latest data from NOAA Tides and Currents API."""
        begin: datetime = datetime.now()
        delta: timedelta = timedelta(days=2)
        end: datetime = begin + delta
        try:
            df_predictions: DataFrame = self._station.get_data(
                begin_date=begin.strftime('%Y%m%d %H:%M'),
                end_date=end.strftime('%Y%m%d %H:%M'),
                product='predictions',
                datum='MLLW',
                interval='hilo',
                units=self._unit_system,
                time_zone=self._timezone
            )
            api_data: DataFrame = df_predictions.head()
            self.data = NOAATidesData(
                time_stamp=list(api_data.index),
                hi_lo=list(api_data['type'].values),
                predicted_wl=list(api_data['v'].values)
            )
            _LOGGER.debug('Data = %s', api_data)
            _LOGGER.debug('Recent Tide data queried with start time set to %s', begin.strftime('%m-%d-%Y %H:%M'))
        except ValueError as err:
            _LOGGER.error('Check NOAA Tides and Currents: %s', err.args)
            self.data = None

from __future__ import annotations
from datetime import timedelta
from typing import Any, Dict, List, Optional
from pyirishrail.pyirishrail import IrishRailRTPI
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_NAME, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

ATTR_STATION: str = 'Station'
ATTR_ORIGIN: str = 'Origin'
ATTR_DESTINATION: str = 'Destination'
ATTR_DIRECTION: str = 'Direction'
ATTR_STOPS_AT: str = 'Stops at'
ATTR_DUE_IN: str = 'Due in'
ATTR_DUE_AT: str = 'Due at'
ATTR_EXPECT_AT: str = 'Expected at'
ATTR_NEXT_UP: str = 'Later Train'
ATTR_TRAIN_TYPE: str = 'Train type'
CONF_STATION: str = 'station'
CONF_DESTINATION: str = 'destination'
CONF_DIRECTION: str = 'direction'
CONF_STOPS_AT: str = 'stops_at'
DEFAULT_NAME: str = 'Next Train'
SCAN_INTERVAL: timedelta = timedelta(minutes=2)
TIME_STR_FORMAT: str = '%H:%M'

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_STATION): cv.string,
    vol.Optional(CONF_DIRECTION): cv.string,
    vol.Optional(CONF_DESTINATION): cv.string,
    vol.Optional(CONF_STOPS_AT): cv.string,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    station: str = config.get(CONF_STATION)
    direction: Optional[str] = config.get(CONF_DIRECTION)
    destination: Optional[str] = config.get(CONF_DESTINATION)
    stops_at: Optional[str] = config.get(CONF_STOPS_AT)
    name: str = config.get(CONF_NAME)
    irish_rail: IrishRailRTPI = IrishRailRTPI()
    data: IrishRailTransportData = IrishRailTransportData(irish_rail, station, direction, destination, stops_at)
    add_entities([IrishRailTransportSensor(data, station, direction, destination, stops_at, name)], True)

class IrishRailTransportSensor(SensorEntity):
    _attr_attribution: str = 'Data provided by Irish Rail'
    _attr_icon: str = 'mdi:train'

    def __init__(
        self,
        data: IrishRailTransportData,
        station: str,
        direction: Optional[str],
        destination: Optional[str],
        stops_at: Optional[str],
        name: str
    ) -> None:
        self.data: IrishRailTransportData = data
        self._station: str = station
        self._direction: Optional[str] = direction
        self._stops_at: Optional[str] = stops_at
        self._name: str = name
        self._state: Optional[Any] = None
        self._times: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def native_value(self) -> Optional[Any]:
        return self._state

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        if self._times:
            next_up: str = 'None'
            if len(self._times) > 1:
                next_up = f'{self._times[1][ATTR_ORIGIN]} to {self._times[1][ATTR_DESTINATION]} in {self._times[1][ATTR_DUE_IN]}'
            return {
                ATTR_STATION: self._station,
                ATTR_ORIGIN: self._times[0][ATTR_ORIGIN],
                ATTR_DESTINATION: self._times[0][ATTR_DESTINATION],
                ATTR_DUE_IN: self._times[0][ATTR_DUE_IN],
                ATTR_DUE_AT: self._times[0][ATTR_DUE_AT],
                ATTR_EXPECT_AT: self._times[0][ATTR_EXPECT_AT],
                ATTR_DIRECTION: self._times[0][ATTR_DIRECTION],
                ATTR_STOPS_AT: self._times[0][ATTR_STOPS_AT],
                ATTR_NEXT_UP: next_up,
                ATTR_TRAIN_TYPE: self._times[0][ATTR_TRAIN_TYPE]
            }
        return None

    @property
    def native_unit_of_measurement(self) -> str:
        return UnitOfTime.MINUTES

    def update(self) -> None:
        self.data.update()
        self._times = self.data.info
        if self._times:
            self._state = self._times[0][ATTR_DUE_IN]
        else:
            self._state = None

class IrishRailTransportData:
    def __init__(
        self,
        irish_rail: IrishRailRTPI,
        station: str,
        direction: Optional[str],
        destination: Optional[str],
        stops_at: Optional[str]
    ) -> None:
        self._ir_api: IrishRailRTPI = irish_rail
        self.station: str = station
        self.direction: Optional[str] = direction
        self.destination: Optional[str] = destination
        self.stops_at: Optional[str] = stops_at
        self.info: List[Dict[str, Any]] = self._empty_train_data()

    def update(self) -> None:
        trains: List[Dict[str, Any]] = self._ir_api.get_station_by_name(
            self.station,
            direction=self.direction,
            destination=self.destination,
            stops_at=self.stops_at
        )
        stops_at: str = self.stops_at if self.stops_at else ''
        self.info = []
        for train in trains:
            train_data: Dict[str, Any] = {
                ATTR_STATION: self.station,
                ATTR_ORIGIN: train.get('origin'),
                ATTR_DESTINATION: train.get('destination'),
                ATTR_DUE_IN: train.get('due_in_mins'),
                ATTR_DUE_AT: train.get('scheduled_arrival_time'),
                ATTR_EXPECT_AT: train.get('expected_departure_time'),
                ATTR_DIRECTION: train.get('direction'),
                ATTR_STOPS_AT: stops_at,
                ATTR_TRAIN_TYPE: train.get('type')
            }
            self.info.append(train_data)
        if not self.info:
            self.info = self._empty_train_data()

    def _empty_train_data(self) -> List[Dict[str, Any]]:
        dest: str = self.destination if self.destination else ''
        direction: str = self.direction if self.direction else ''
        stops_at: str = self.stops_at if self.stops_at else ''
        return [{
            ATTR_STATION: self.station,
            ATTR_ORIGIN: '',
            ATTR_DESTINATION: dest,
            ATTR_DUE_IN: None,
            ATTR_DUE_AT: None,
            ATTR_EXPECT_AT: None,
            ATTR_DIRECTION: direction,
            ATTR_STOPS_AT: stops_at,
            ATTR_TRAIN_TYPE: ''
        }]
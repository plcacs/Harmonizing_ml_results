"""Support for Irish Rail RTPI information."""
from __future__ import annotations
from datetime import timedelta
from pydantic import BaseModel
from pyirishrail.pyirishrail import IrishRailRTPI
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_NAME, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from typing import Any, Optional

ATTR_STATION = 'Station'
ATTR_ORIGIN = 'Origin'
ATTR_DESTINATION = 'Destination'
ATTR_DIRECTION = 'Direction'
ATTR_STOPS_AT = 'Stops at'
ATTR_DUE_IN = 'Due in'
ATTR_DUE_AT = 'Due at'
ATTR_EXPECT_AT = 'Expected at'
ATTR_NEXT_UP = 'Later Train'
ATTR_TRAIN_TYPE = 'Train type'
CONF_STATION = 'station'
CONF_DESTINATION = 'destination'
CONF_DIRECTION = 'direction'
CONF_STOPS_AT = 'stops_at'
DEFAULT_NAME = 'Next Train'
SCAN_INTERVAL = timedelta(minutes=2)
TIME_STR_FORMAT = '%H:%M'

class TrainData(BaseModel):
    station: str
    origin: str
    destination: str
    due_in_mins: Optional[int]
    scheduled_arrival_time: Optional[str]
    expected_departure_time: Optional[str]
    direction: str
    stops_at: str
    type: str

class IrishRailTransportSensorData:
    """The Class for handling the data retrieval."""

    def __init__(self, irish_rail: IrishRailRTPI, station: str, direction: Optional[str], destination: Optional[str], stops_at: Optional[str]):
        """Initialize the data object."""
        self._ir_api = irish_rail
        self.station = station
        self.direction = direction
        self.destination = destination
        self.stops_at = stops_at
        self.info: list[TrainData] = []

    def update(self):
        """Get the latest data from irishrail."""
        trains = self._ir_api.get_station_by_name(self.station, direction=self.direction, destination=self.destination, stops_at=self.stops_at)
        stops_at = self.stops_at if self.stops_at else ''
        self.info = []
        for train in trains:
            train_data = TrainData(
                station=self.station,
                origin=train.get('origin'),
                destination=train.get('destination'),
                due_in_mins=train.get('due_in_mins'),
                scheduled_arrival_time=train.get('scheduled_arrival_time'),
                expected_departure_time=train.get('expected_departure_time'),
                direction=train.get('direction'),
                stops_at=stops_at,
                type=train.get('type')
            )
            self.info.append(train_data)
        if not self.info:
            self.info = [TrainData(
                station=self.station,
                origin='',
                destination=self.destination if self.destination else '',
                due_in_mins=None,
                scheduled_arrival_time=None,
                expected_departure_time=None,
                direction=self.direction if self.direction else '',
                stops_at=stops_at,
                type=''
            )]

class IrishRailTransportSensor(SensorEntity):
    """Implementation of an irish rail public transport sensor."""
    _attr_attribution = 'Data provided by Irish Rail'
    _attr_icon = 'mdi:train'

    def __init__(self, data: IrishRailTransportSensorData, station: str, direction: Optional[str], destination: Optional[str], stops_at: Optional[str], name: str):
        """Initialize the sensor."""
        self.data = data
        self._station = station
        self._direction = direction
        self._stops_at = stops_at
        self._name = name
        self._state = None
        self._times: list[TrainData] = []

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def native_value(self) -> Optional[str]:
        """Return the state of the sensor."""
        return self._state

    @property
    def extra_state_attributes(self) -> Optional[dict[str, Any]]:
        """Return the state attributes."""
        if self._times:
            next_up = 'None'
            if len(self._times) > 1:
                next_up = f'{self._times[1].origin} to {self._times[1].destination} in {self._times[1].due_in_mins} minutes'
            return {
                ATTR_STATION: self._station,
                ATTR_ORIGIN: self._times[0].origin,
                ATTR_DESTINATION: self._times[0].destination,
                ATTR_DUE_IN: self._times[0].due_in_mins,
                ATTR_DUE_AT: self._times[0].scheduled_arrival_time,
                ATTR_EXPECT_AT: self._times[0].expected_departure_time,
                ATTR_DIRECTION: self._times[0].direction,
                ATTR_STOPS_AT: self._times[0].stops_at,
                ATTR_NEXT_UP: next_up,
                ATTR_TRAIN_TYPE: self._times[0].type
            }
        return None

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit this state is expressed in."""
        return UnitOfTime.MINUTES

    def update(self):
        """Get the latest data and update the states."""
        self.data.update()
        self._times = self.data.info
        if self._times:
            self._state = self._times[0].due_in_mins
        else:
            self._state = None

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({vol.Required(CONF_STATION): cv.string, vol.Optional(CONF_DIRECTION): cv.string, vol.Optional(CONF_DESTINATION): cv.string, vol.Optional(CONF_STOPS_AT): cv.string, vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string})

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: Optional[DiscoveryInfoType] = None):
    """Set up the Irish Rail transport sensor."""
    station = config.get(CONF_STATION)
    direction = config.get(CONF_DIRECTION)
    destination = config.get(CONF_DESTINATION)
    stops_at = config.get(CONF_STOPS_AT)
    name = config.get(CONF_NAME)
    irish_rail = IrishRailRTPI()
    data = IrishRailTransportSensorData(irish_rail, station, direction, destination, stops_at)
    add_entities([IrishRailTransportSensor(data, station, direction, destination, stops_at, name)], True)

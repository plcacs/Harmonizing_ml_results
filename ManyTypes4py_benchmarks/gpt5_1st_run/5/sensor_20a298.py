"""Support for Irish Rail RTPI information."""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Final, Optional, TypedDict

from pyirishrail.pyirishrail import IrishRailRTPI
import voluptuous as vol

from homeassistant.components.sensor import (
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    SensorEntity,
)
from homeassistant.const import CONF_NAME, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType


ATTR_STATION: Final[str] = "Station"
ATTR_ORIGIN: Final[str] = "Origin"
ATTR_DESTINATION: Final[str] = "Destination"
ATTR_DIRECTION: Final[str] = "Direction"
ATTR_STOPS_AT: Final[str] = "Stops at"
ATTR_DUE_IN: Final[str] = "Due in"
ATTR_DUE_AT: Final[str] = "Due at"
ATTR_EXPECT_AT: Final[str] = "Expected at"
ATTR_NEXT_UP: Final[str] = "Later Train"
ATTR_TRAIN_TYPE: Final[str] = "Train type"

CONF_STATION: Final[str] = "station"
CONF_DESTINATION: Final[str] = "destination"
CONF_DIRECTION: Final[str] = "direction"
CONF_STOPS_AT: Final[str] = "stops_at"

DEFAULT_NAME: Final[str] = "Next Train"
SCAN_INTERVAL: Final[timedelta] = timedelta(minutes=2)
TIME_STR_FORMAT: Final[str] = "%H:%M"


TrainInfo = TypedDict(
    "TrainInfo",
    {
        ATTR_STATION: str,
        ATTR_ORIGIN: str,
        ATTR_DESTINATION: str,
        ATTR_DUE_IN: Optional[int],
        ATTR_DUE_AT: Optional[str],
        ATTR_EXPECT_AT: Optional[str],
        ATTR_DIRECTION: str,
        ATTR_STOPS_AT: str,
        ATTR_TRAIN_TYPE: str,
    },
)


PLATFORM_SCHEMA: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_STATION): cv.string,
        vol.Optional(CONF_DIRECTION): cv.string,
        vol.Optional(CONF_DESTINATION): cv.string,
        vol.Optional(CONF_STOPS_AT): cv.string,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    }
)


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the Irish Rail transport sensor."""
    station: str = config.get(CONF_STATION)  # type: ignore[assignment]
    direction: Optional[str] = config.get(CONF_DIRECTION)
    destination: Optional[str] = config.get(CONF_DESTINATION)
    stops_at: Optional[str] = config.get(CONF_STOPS_AT)
    name: str = config.get(CONF_NAME)  # type: ignore[assignment]

    irish_rail: IrishRailRTPI = IrishRailRTPI()
    data = IrishRailTransportData(irish_rail, station, direction, destination, stops_at)
    add_entities(
        [IrishRailTransportSensor(data, station, direction, destination, stops_at, name)],
        True,
    )


class IrishRailTransportSensor(SensorEntity):
    """Implementation of an irish rail public transport sensor."""

    _attr_attribution: str = "Data provided by Irish Rail"
    _attr_icon: str = "mdi:train"

    def __init__(
        self,
        data: "IrishRailTransportData",
        station: str,
        direction: Optional[str],
        destination: Optional[str],
        stops_at: Optional[str],
        name: str,
    ) -> None:
        """Initialize the sensor."""
        self.data: IrishRailTransportData = data
        self._station: str = station
        self._direction: Optional[str] = direction
        self._direction: Optional[str] = direction
        self._stops_at: Optional[str] = stops_at
        self._name: str = name
        self._state: Optional[int] = None
        self._times: list[TrainInfo] = []

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def native_value(self) -> Optional[int]:
        """Return the state of the sensor."""
        return self._state

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return the state attributes."""
        if self._times:
            next_up = "None"
            if len(self._times) > 1:
                next_up = (
                    f"{self._times[1][ATTR_ORIGIN]} to "
                    f"{self._times[1][ATTR_DESTINATION]} in "
                    f"{self._times[1][ATTR_DUE_IN]}"
                )
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
                ATTR_TRAIN_TYPE: self._times[0][ATTR_TRAIN_TYPE],
            }
        return None

    @property
    def native_unit_of_measurement(self) -> str | None:
        """Return the unit this state is expressed in."""
        return UnitOfTime.MINUTES

    def update(self) -> None:
        """Get the latest data and update the states."""
        self.data.update()
        self._times = self.data.info
        if self._times:
            self._state = self._times[0][ATTR_DUE_IN]
        else:
            self._state = None


class IrishRailTransportData:
    """The Class for handling the data retrieval."""

    def __init__(
        self,
        irish_rail: IrishRailRTPI,
        station: str,
        direction: Optional[str],
        destination: Optional[str],
        stops_at: Optional[str],
    ) -> None:
        """Initialize the data object."""
        self._ir_api: IrishRailRTPI = irish_rail
        self.station: str = station
        self.direction: Optional[str] = direction
        self.destination: Optional[str] = destination
        self.stops_at: Optional[str] = stops_at
        self.info: list[TrainInfo] = self._empty_train_data()

    def update(self) -> None:
        """Get the latest data from irishrail."""
        trains: list[dict[str, Any]] = self._ir_api.get_station_by_name(
            self.station,
            direction=self.direction,
            destination=self.destination,
            stops_at=self.stops_at,
        )
        stops_at: str = self.stops_at if self.stops_at else ""
        self.info = []
        for train in trains:
            train_data: TrainInfo = {
                ATTR_STATION: self.station,
                ATTR_ORIGIN: train.get("origin", "") or "",
                ATTR_DESTINATION: train.get("destination", "") or "",
                ATTR_DUE_IN: train.get("due_in_mins"),
                ATTR_DUE_AT: train.get("scheduled_arrival_time"),
                ATTR_EXPECT_AT: train.get("expected_departure_time"),
                ATTR_DIRECTION: train.get("direction", "") or "",
                ATTR_STOPS_AT: stops_at,
                ATTR_TRAIN_TYPE: train.get("type", "") or "",
            }
            self.info.append(train_data)
        if not self.info:
            self.info = self._empty_train_data()

    def _empty_train_data(self) -> list[TrainInfo]:
        """Generate info for an empty train."""
        dest: str = self.destination if self.destination else ""
        direction: str = self.direction if self.direction else ""
        stops_at: str = self.stops_at if self.stops_at else ""
        return [
            {
                ATTR_STATION: self.station,
                ATTR_ORIGIN: "",
                ATTR_DESTINATION: dest,
                ATTR_DUE_IN: None,
                ATTR_DUE_AT: None,
                ATTR_EXPECT_AT: None,
                ATTR_DIRECTION: direction,
                ATTR_STOPS_AT: stops_at,
                ATTR_TRAIN_TYPE: "",
            }
        ]
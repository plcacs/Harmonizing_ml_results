from __future__ import annotations
from datetime import timedelta
import logging
from math import cos, pi, radians, sin
import random
from typing import Any, Callable, List, Optional
from homeassistant.components.geo_location import GeolocationEvent
from homeassistant.const import UnitOfLength
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import track_time_interval
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger
AVG_KM_PER_DEGREE: float
DEFAULT_UPDATE_INTERVAL: timedelta
MAX_RADIUS_IN_KM: int
NUMBER_OF_DEMO_DEVICES: int
EVENT_NAMES: List[str]
SOURCE: str

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: Optional[DiscoveryInfoType] = None) -> None:

class DemoManager:
    def __init__(self, hass: HomeAssistant, add_entities: AddEntitiesCallback) -> None:

    def _generate_random_event(self) -> DemoGeolocationEvent:

    def _init_regular_updates(self) -> None:

    def _update(self, count: int = 1) -> None:

class DemoGeolocationEvent(GeolocationEvent):
    _attr_should_poll: bool

    def __init__(self, name: str, distance: float, latitude: float, longitude: float, unit_of_measurement: UnitOfLength) -> None:

    @property
    def source(self) -> str:

    @property
    def distance(self) -> float:

    @property
    def latitude(self) -> float:

    @property
    def longitude(self) -> float:

    @property
    def unit_of_measurement(self) -> UnitOfLength:

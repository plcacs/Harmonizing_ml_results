"""Demo platform for the geolocation component."""
from __future__ import annotations
from datetime import timedelta
import logging
from math import cos, pi, radians, sin
import random
from typing import Optional

from homeassistant.components.geo_location import GeolocationEvent
from homeassistant.const import UnitOfLength
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import track_time_interval
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER = logging.getLogger(__name__)

AVG_KM_PER_DEGREE: float = 111.0
DEFAULT_UPDATE_INTERVAL: timedelta = timedelta(minutes=1)
MAX_RADIUS_IN_KM: float = 50
NUMBER_OF_DEMO_DEVICES: int = 5
EVENT_NAMES: list[str] = [
    'Bushfire', 'Hazard Reduction', 'Grass Fire', 'Burn off', 'Structure Fire',
    'Fire Alarm', 'Thunderstorm', 'Tornado', 'Cyclone', 'Waterspout',
    'Dust Storm', 'Blizzard', 'Ice Storm', 'Earthquake', 'Tsunami'
]
SOURCE: str = 'demo'


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the Demo geolocations."""
    DemoManager(hass, add_entities)


class DemoManager:
    """Device manager for demo geolocation events."""

    def __init__(self, hass: HomeAssistant, add_entities: AddEntitiesCallback) -> None:
        """Initialize the demo geolocation event manager."""
        self._hass: HomeAssistant = hass
        self._add_entities: AddEntitiesCallback = add_entities
        self._managed_devices: list[DemoGeolocationEvent] = []
        self._update(count=NUMBER_OF_DEMO_DEVICES)
        self._init_regular_updates()

    def _generate_random_event(self) -> DemoGeolocationEvent:
        """Generate a random event in vicinity of this HA instance."""
        home_latitude: float = self._hass.config.latitude
        home_longitude: float = self._hass.config.longitude
        radius_in_degrees: float = random.random() * MAX_RADIUS_IN_KM / AVG_KM_PER_DEGREE
        radius_in_km: float = radius_in_degrees * AVG_KM_PER_DEGREE
        angle: float = random.random() * 2 * pi
        latitude: float = home_latitude + radius_in_degrees * sin(angle)
        longitude: float = home_longitude + radius_in_degrees * cos(angle) / cos(radians(home_latitude))
        event_name: str = random.choice(EVENT_NAMES)
        return DemoGeolocationEvent(
            name=event_name,
            distance=radius_in_km,
            latitude=latitude,
            longitude=longitude,
            unit_of_measurement=UnitOfLength.KILOMETERS
        )

    def _init_regular_updates(self) -> None:
        """Schedule regular updates based on configured time interval."""
        track_time_interval(
            self._hass,
            lambda now: self._update(),
            DEFAULT_UPDATE_INTERVAL,
            cancel_on_shutdown=True
        )

    def _update(self, count: int = 1) -> None:
        """Remove events and add new random events."""
        for _ in range(count):
            if self._managed_devices:
                device: DemoGeolocationEvent = random.choice(self._managed_devices)
                if device:
                    _LOGGER.debug('Removing %s', device)
                    self._managed_devices.remove(device)
                    self._hass.add_job(device.async_remove())
        new_devices: list[DemoGeolocationEvent] = []
        for _ in range(count):
            new_device: DemoGeolocationEvent = self._generate_random_event()
            _LOGGER.debug('Adding %s', new_device)
            new_devices.append(new_device)
            self._managed_devices.append(new_device)
        self._add_entities(new_devices)


class DemoGeolocationEvent(GeolocationEvent):
    """Represents a demo geolocation event."""
    _attr_should_poll: bool = False

    def __init__(
        self,
        name: str,
        distance: float,
        latitude: float,
        longitude: float,
        unit_of_measurement: UnitOfLength
    ) -> None:
        """Initialize entity with data provided."""
        self._attr_name: str = name
        self._distance: float = distance
        self._latitude: float = latitude
        self._longitude: float = longitude
        self._unit_of_measurement: UnitOfLength = unit_of_measurement

    @property
    def source(self) -> str:
        """Return source value of this external event."""
        return SOURCE

    @property
    def distance(self) -> float:
        """Return distance value of this external event."""
        return self._distance

    @property
    def latitude(self) -> float:
        """Return latitude value of this external event."""
        return self._latitude

    @property
    def longitude(self) -> float:
        """Return longitude value of this external event."""
        return self._longitude

    @property
    def unit_of_measurement(self) -> UnitOfLength:
        """Return the unit of measurement."""
        return self._unit_of_measurement

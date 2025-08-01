"""Generic GeoRSS events service.

Retrieves current events (typically incidents or alerts) in GeoRSS format, and
shows information on events filtered by distance to the HA instance's location
and grouped by category.
"""
from __future__ import annotations
from datetime import timedelta
import logging
from typing import Any, Dict, List, Optional, Tuple
from georss_client import UPDATE_OK, UPDATE_OK_NO_DATA
from georss_generic_client import GenericFeed
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import (
    CONF_LATITUDE,
    CONF_LONGITUDE,
    CONF_NAME,
    CONF_RADIUS,
    CONF_UNIT_OF_MEASUREMENT,
    CONF_URL,
    UnitOfLength,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER = logging.getLogger(__name__)

ATTR_CATEGORY = 'category'
ATTR_DISTANCE = 'distance'
ATTR_TITLE = 'title'
CONF_CATEGORIES = 'categories'
DEFAULT_ICON = 'mdi:alert'
DEFAULT_NAME = 'Event Service'
DEFAULT_RADIUS_IN_KM = 20.0
DEFAULT_UNIT_OF_MEASUREMENT = 'Events'
DOMAIN = 'geo_rss_events'
SCAN_INTERVAL = timedelta(minutes=5)

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_URL): cv.string,
    vol.Optional(CONF_LATITUDE): cv.latitude,
    vol.Optional(CONF_LONGITUDE): cv.longitude,
    vol.Optional(CONF_RADIUS, default=DEFAULT_RADIUS_IN_KM): vol.Coerce(float),
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_CATEGORIES, default=[]): vol.All(cv.ensure_list, [cv.string]),
    vol.Optional(CONF_UNIT_OF_MEASUREMENT, default=DEFAULT_UNIT_OF_MEASUREMENT): cv.string
})


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the GeoRSS component."""
    latitude: float = config.get(CONF_LATITUDE, hass.config.latitude)
    longitude: float = config.get(CONF_LONGITUDE, hass.config.longitude)
    url: str = config.get(CONF_URL)
    radius_in_km: float = config.get(CONF_RADIUS)
    name: str = config.get(CONF_NAME)
    categories: List[str] = config.get(CONF_CATEGORIES)
    unit_of_measurement: str = config.get(CONF_UNIT_OF_MEASUREMENT)
    _LOGGER.debug('latitude=%s, longitude=%s, url=%s, radius=%s', latitude, longitude, url, radius_in_km)
    devices: List[GeoRssServiceSensor] = []
    if not categories:
        device = GeoRssServiceSensor(
            coordinates=(latitude, longitude),
            url=url,
            radius=radius_in_km,
            category=None,
            service_name=name,
            unit_of_measurement=unit_of_measurement
        )
        devices.append(device)
    else:
        for category in categories:
            device = GeoRssServiceSensor(
                coordinates=(latitude, longitude),
                url=url,
                radius=radius_in_km,
                category=category,
                service_name=name,
                unit_of_measurement=unit_of_measurement
            )
            devices.append(device)
    add_entities(devices, True)


class GeoRssServiceSensor(SensorEntity):
    """Representation of a Sensor."""

    _attr_icon: str = DEFAULT_ICON

    def __init__(
        self,
        coordinates: Tuple[float, float],
        url: str,
        radius: float,
        category: Optional[str],
        service_name: str,
        unit_of_measurement: str
    ) -> None:
        """Initialize the sensor."""
        self._category: Optional[str] = category
        self._service_name: str = service_name
        self._state: Optional[int] = None
        self._state_attributes: Optional[Dict[str, str]] = None
        self._unit_of_measurement: str = unit_of_measurement
        self._feed: GenericFeed = GenericFeed(
            coordinates,
            url,
            filter_radius=radius,
            filter_categories=None if not category else [category]
        )

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        category_display: str = 'Any' if self._category is None else self._category
        return f'{self._service_name} {category_display}'

    @property
    def native_value(self) -> Optional[int]:
        """Return the state of the sensor."""
        return self._state

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit of measurement."""
        return self._unit_of_measurement

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, str]]:
        """Return the state attributes."""
        return self._state_attributes

    def update(self) -> None:
        """Update this sensor from the GeoRSS service."""
        status: int
        feed_entries: List[Any]
        status, feed_entries = self._feed.update()
        if status == UPDATE_OK:
            _LOGGER.debug('Adding events to sensor %s: %s', self.entity_id, feed_entries)
            self._state = len(feed_entries)
            matrix: Dict[str, str] = {}
            for entry in feed_entries:
                matrix[entry.title] = f'{entry.distance_to_home:.0f}{UnitOfLength.KILOMETERS}'
            self._state_attributes = matrix
        elif status == UPDATE_OK_NO_DATA:
            _LOGGER.debug('Update successful, but no data received from %s', self._feed)
        else:
            _LOGGER.warning('Update not successful, no data received from %s', self._feed)
            self._state = 0
            self._state_attributes = {}

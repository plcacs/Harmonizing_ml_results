#!/usr/bin/env python3
"""Support for IGN Sismologia (Earthquakes) Feeds."""
from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta, datetime
import logging
from typing import Any, Optional, Dict, Tuple

from georss_ign_sismologia_client import IgnSismologiaFeedEntry, IgnSismologiaFeedManager
import voluptuous as vol

from homeassistant.components.geo_location import (
    PLATFORM_SCHEMA as GEO_LOCATION_PLATFORM_SCHEMA,
    GeolocationEvent,
)
from homeassistant.const import (
    CONF_LATITUDE,
    CONF_LONGITUDE,
    CONF_RADIUS,
    CONF_SCAN_INTERVAL,
    EVENT_HOMEASSISTANT_START,
    UnitOfLength,
)
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.dispatcher import async_dispatcher_connect, dispatcher_send
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import track_time_interval
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger = logging.getLogger(__name__)

ATTR_EXTERNAL_ID: str = 'external_id'
ATTR_IMAGE_URL: str = 'image_url'
ATTR_MAGNITUDE: str = 'magnitude'
ATTR_PUBLICATION_DATE: str = 'publication_date'
ATTR_REGION: str = 'region'
ATTR_TITLE: str = 'title'

CONF_MINIMUM_MAGNITUDE: str = 'minimum_magnitude'
DEFAULT_MINIMUM_MAGNITUDE: float = 0.0
DEFAULT_RADIUS_IN_KM: float = 50.0
SCAN_INTERVAL: timedelta = timedelta(minutes=5)
SOURCE: str = 'ign_sismologia'

PLATFORM_SCHEMA = GEO_LOCATION_PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_LATITUDE): cv.latitude,
        vol.Optional(CONF_LONGITUDE): cv.longitude,
        vol.Optional(CONF_RADIUS, default=DEFAULT_RADIUS_IN_KM): vol.Coerce(float),
        vol.Optional(CONF_MINIMUM_MAGNITUDE, default=DEFAULT_MINIMUM_MAGNITUDE): cv.positive_float,
    }
)


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the IGN Sismologia Feed platform."""
    scan_interval: timedelta = config.get(CONF_SCAN_INTERVAL, SCAN_INTERVAL)
    coordinates: Tuple[float, float] = (
        config.get(CONF_LATITUDE, hass.config.latitude),
        config.get(CONF_LONGITUDE, hass.config.longitude),
    )
    radius_in_km: float = config[CONF_RADIUS]
    minimum_magnitude: float = config[CONF_MINIMUM_MAGNITUDE]
    feed: IgnSismologiaFeedEntityManager = IgnSismologiaFeedEntityManager(
        hass, add_entities, scan_interval, coordinates, radius_in_km, minimum_magnitude
    )

    def start_feed_manager(event: Event) -> None:
        """Start feed manager."""
        feed.startup()

    hass.bus.listen_once(EVENT_HOMEASSISTANT_START, start_feed_manager)


class IgnSismologiaFeedEntityManager:
    """Feed Entity Manager for IGN Sismologia GeoRSS feed."""

    def __init__(
        self,
        hass: HomeAssistant,
        add_entities: AddEntitiesCallback,
        scan_interval: timedelta,
        coordinates: Tuple[float, float],
        radius_in_km: float,
        minimum_magnitude: float,
    ) -> None:
        """Initialize the Feed Entity Manager."""
        self._hass: HomeAssistant = hass
        self._feed_manager: IgnSismologiaFeedManager = IgnSismologiaFeedManager(
            self._generate_entity,
            self._update_entity,
            self._remove_entity,
            coordinates,
            filter_radius=radius_in_km,
            filter_minimum_magnitude=minimum_magnitude,
        )
        self._add_entities: AddEntitiesCallback = add_entities
        self._scan_interval: timedelta = scan_interval

    def startup(self) -> None:
        """Start up this manager."""
        self._feed_manager.update()
        self._init_regular_updates()

    def _init_regular_updates(self) -> None:
        """Schedule regular updates at the specified interval."""
        track_time_interval(self._hass, lambda now: self._feed_manager.update(), self._scan_interval)

    def get_entry(self, external_id: str) -> Optional[IgnSismologiaFeedEntry]:
        """Get feed entry by external id."""
        return self._feed_manager.feed_entries.get(external_id)

    def _generate_entity(self, external_id: str) -> None:
        """Generate new entity."""
        new_entity = IgnSismologiaLocationEvent(self, external_id)
        self._add_entities([new_entity], True)

    def _update_entity(self, external_id: str) -> None:
        """Update entity."""
        dispatcher_send(self._hass, f'ign_sismologia_update_{external_id}')

    def _remove_entity(self, external_id: str) -> None:
        """Remove entity."""
        dispatcher_send(self._hass, f'ign_sismologia_delete_{external_id}')


class IgnSismologiaLocationEvent(GeolocationEvent):
    """Represents an external event with IGN Sismologia feed data."""

    _attr_icon: str = 'mdi:pulse'
    _attr_should_poll: bool = False
    _attr_source: str = SOURCE
    _attr_unit_of_measurement: str = UnitOfLength.KILOMETERS

    def __init__(self, feed_manager: IgnSismologiaFeedEntityManager, external_id: str) -> None:
        """Initialize entity with data from feed entry."""
        self._feed_manager: IgnSismologiaFeedEntityManager = feed_manager
        self._external_id: str = external_id
        self._title: Optional[str] = None
        self._region: Optional[str] = None
        self._magnitude: Optional[float] = None
        self._publication_date: Optional[datetime] = None
        self._image_url: Optional[str] = None

    async def async_added_to_hass(self) -> None:
        """Call when entity is added to hass."""
        self._remove_signal_delete = async_dispatcher_connect(
            self.hass, f'ign_sismologia_delete_{self._external_id}', self._delete_callback
        )
        self._remove_signal_update = async_dispatcher_connect(
            self.hass, f'ign_sismologia_update_{self._external_id}', self._update_callback
        )

    @callback
    def _delete_callback(self) -> None:
        """Remove this entity."""
        self._remove_signal_delete()
        self._remove_signal_update()
        self.hass.async_create_task(self.async_remove(force_remove=True))

    @callback
    def _update_callback(self) -> None:
        """Call update method."""
        self.async_schedule_update_ha_state(True)

    async def async_update(self) -> None:
        """Update this entity from the data held in the feed manager."""
        _LOGGER.debug('Updating %s', self._external_id)
        feed_entry: Optional[IgnSismologiaFeedEntry] = self._feed_manager.get_entry(self._external_id)
        if feed_entry:
            self._update_from_feed(feed_entry)

    def _update_from_feed(self, feed_entry: IgnSismologiaFeedEntry) -> None:
        """Update the internal state from the provided feed entry."""
        self._title = feed_entry.title
        self._attr_distance = feed_entry.distance_to_home
        self._attr_latitude = feed_entry.coordinates[0]
        self._attr_longitude = feed_entry.coordinates[1]
        self._attr_attribution = feed_entry.attribution
        self._region = feed_entry.region
        self._magnitude = feed_entry.magnitude
        self._publication_date = feed_entry.published
        self._image_url = feed_entry.image_url

    @property
    def name(self) -> str:
        """Return the name of the entity."""
        if self._magnitude is not None and self._region:
            return f'M {self._magnitude:.1f} - {self._region}'
        if self._magnitude is not None:
            return f'M {self._magnitude:.1f}'
        if self._region:
            return self._region
        return self._title if self._title else "Unknown"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the device state attributes."""
        return {
            key: value
            for key, value in (
                (ATTR_EXTERNAL_ID, self._external_id),
                (ATTR_TITLE, self._title),
                (ATTR_REGION, self._region),
                (ATTR_MAGNITUDE, self._magnitude),
                (ATTR_PUBLICATION_DATE, self._publication_date),
                (ATTR_IMAGE_URL, self._image_url),
            )
            if value or isinstance(value, bool)
        }
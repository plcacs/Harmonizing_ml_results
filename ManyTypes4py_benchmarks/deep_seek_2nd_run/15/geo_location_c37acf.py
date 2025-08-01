"""Support for U.S. Geological Survey Earthquake Hazards Program Feeds."""
from __future__ import annotations
from collections.abc import Callable
from datetime import datetime, timedelta
import logging
from typing import Any, Final
from aio_geojson_usgs_earthquakes import UsgsEarthquakeHazardsProgramFeedManager
from aio_geojson_usgs_earthquakes.feed_entry import UsgsEarthquakeHazardsProgramFeedEntry
import voluptuous as vol
from homeassistant.components.geo_location import PLATFORM_SCHEMA as GEO_LOCATION_PLATFORM_SCHEMA, GeolocationEvent
from homeassistant.const import ATTR_TIME, CONF_LATITUDE, CONF_LONGITUDE, CONF_RADIUS, CONF_SCAN_INTERVAL, EVENT_HOMEASSISTANT_START, UnitOfLength
from homeassistant.core import HomeAssistant, Event, callback
from homeassistant.helpers import aiohttp_client, config_validation as cv
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: Final = logging.getLogger(__name__)

ATTR_ALERT: Final = 'alert'
ATTR_EXTERNAL_ID: Final = 'external_id'
ATTR_MAGNITUDE: Final = 'magnitude'
ATTR_PLACE: Final = 'place'
ATTR_STATUS: Final = 'status'
ATTR_TYPE: Final = 'type'
ATTR_UPDATED: Final = 'updated'
CONF_FEED_TYPE: Final = 'feed_type'
CONF_MINIMUM_MAGNITUDE: Final = 'minimum_magnitude'
DEFAULT_MINIMUM_MAGNITUDE: Final = 0.0
DEFAULT_RADIUS_IN_KM: Final = 50.0
DEFAULT_UNIT_OF_MEASUREMENT: Final = UnitOfLength.KILOMETERS
SCAN_INTERVAL: Final = timedelta(minutes=5)
SIGNAL_DELETE_ENTITY: Final = 'usgs_earthquakes_feed_delete_{}'
SIGNAL_UPDATE_ENTITY: Final = 'usgs_earthquakes_feed_update_{}'
SOURCE: Final = 'usgs_earthquakes_feed'
VALID_FEED_TYPES: Final = [
    'past_hour_significant_earthquakes', 'past_hour_m45_earthquakes', 'past_hour_m25_earthquakes',
    'past_hour_m10_earthquakes', 'past_hour_all_earthquakes', 'past_day_significant_earthquakes',
    'past_day_m45_earthquakes', 'past_day_m25_earthquakes', 'past_day_m10_earthquakes',
    'past_day_all_earthquakes', 'past_week_significant_earthquakes', 'past_week_m45_earthquakes',
    'past_week_m25_earthquakes', 'past_week_m10_earthquakes', 'past_week_all_earthquakes',
    'past_month_significant_earthquakes', 'past_month_m45_earthquakes', 'past_month_m25_earthquakes',
    'past_month_m10_earthquakes', 'past_month_all_earthquakes'
]

PLATFORM_SCHEMA: Final = GEO_LOCATION_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_FEED_TYPE): vol.In(VALID_FEED_TYPES),
    vol.Optional(CONF_LATITUDE): cv.latitude,
    vol.Optional(CONF_LONGITUDE): cv.longitude,
    vol.Optional(CONF_RADIUS, default=DEFAULT_RADIUS_IN_KM): vol.Coerce(float),
    vol.Optional(CONF_MINIMUM_MAGNITUDE, default=DEFAULT_MINIMUM_MAGNITUDE): cv.positive_float
})

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    """Set up the USGS Earthquake Hazards Program Feed platform."""
    scan_interval: timedelta = config.get(CONF_SCAN_INTERVAL, SCAN_INTERVAL)
    feed_type: str = config[CONF_FEED_TYPE]
    coordinates: tuple[float, float] = (
        config.get(CONF_LATITUDE, hass.config.latitude),
        config.get(CONF_LONGITUDE, hass.config.longitude)
    )
    radius_in_km: float = config[CONF_RADIUS]
    minimum_magnitude: float = config[CONF_MINIMUM_MAGNITUDE]
    manager: UsgsEarthquakesFeedEntityManager = UsgsEarthquakesFeedEntityManager(
        hass, async_add_entities, scan_interval, coordinates, feed_type, radius_in_km, minimum_magnitude
    )
    await manager.async_init()

    async def start_feed_manager(event: Event | None = None) -> None:
        """Start feed manager."""
        await manager.async_update()
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, start_feed_manager)

class UsgsEarthquakesFeedEntityManager:
    """Feed Entity Manager for USGS Earthquake Hazards Program feed."""

    def __init__(
        self,
        hass: HomeAssistant,
        async_add_entities: AddEntitiesCallback,
        scan_interval: timedelta,
        coordinates: tuple[float, float],
        feed_type: str,
        radius_in_km: float,
        minimum_magnitude: float
    ) -> None:
        """Initialize the Feed Entity Manager."""
        self._hass: HomeAssistant = hass
        websession = aiohttp_client.async_get_clientsession(hass)
        self._feed_manager: UsgsEarthquakeHazardsProgramFeedManager = UsgsEarthquakeHazardsProgramFeedManager(
            websession,
            self._generate_entity,
            self._update_entity,
            self._remove_entity,
            coordinates,
            feed_type,
            filter_radius=radius_in_km,
            filter_minimum_magnitude=minimum_magnitude
        )
        self._async_add_entities: AddEntitiesCallback = async_add_entities
        self._scan_interval: timedelta = scan_interval

    async def async_init(self) -> None:
        """Schedule initial and regular updates based on configured time interval."""

        async def update(event_time: datetime) -> None:
            """Update."""
            await self.async_update()
        async_track_time_interval(self._hass, update, self._scan_interval, cancel_on_shutdown=True)
        _LOGGER.debug('Feed entity manager initialized')

    async def async_update(self) -> None:
        """Refresh data."""
        await self._feed_manager.update()
        _LOGGER.debug('Feed entity manager updated')

    def get_entry(self, external_id: str) -> UsgsEarthquakeHazardsProgramFeedEntry | None:
        """Get feed entry by external id."""
        return self._feed_manager.feed_entries.get(external_id)

    async def _generate_entity(self, external_id: str) -> None:
        """Generate new entity."""
        new_entity: UsgsEarthquakesEvent = UsgsEarthquakesEvent(self, external_id)
        self._async_add_entities([new_entity], True)

    async def _update_entity(self, external_id: str) -> None:
        """Update entity."""
        async_dispatcher_send(self._hass, SIGNAL_UPDATE_ENTITY.format(external_id))

    async def _remove_entity(self, external_id: str) -> None:
        """Remove entity."""
        async_dispatcher_send(self._hass, SIGNAL_DELETE_ENTITY.format(external_id))

class UsgsEarthquakesEvent(GeolocationEvent):
    """Represents an external event with USGS Earthquake data."""
    _attr_icon: str = 'mdi:pulse'
    _attr_should_poll: bool = False
    _attr_source: str = SOURCE
    _attr_unit_of_measurement: str = DEFAULT_UNIT_OF_MEASUREMENT

    def __init__(self, feed_manager: UsgsEarthquakesFeedEntityManager, external_id: str) -> None:
        """Initialize entity with data from feed entry."""
        self._feed_manager: UsgsEarthquakesFeedEntityManager = feed_manager
        self._external_id: str = external_id
        self._place: str | None = None
        self._magnitude: float | None = None
        self._time: datetime | None = None
        self._updated: datetime | None = None
        self._status: str | None = None
        self._type: str | None = None
        self._alert: str | None = None

    async def async_added_to_hass(self) -> None:
        """Call when entity is added to hass."""
        self._remove_signal_delete: Callable[[], None] = async_dispatcher_connect(
            self.hass,
            SIGNAL_DELETE_ENTITY.format(self._external_id),
            self._delete_callback
        )
        self._remove_signal_update: Callable[[], None] = async_dispatcher_connect(
            self.hass,
            SIGNAL_UPDATE_ENTITY.format(self._external_id),
            self._update_callback
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
        feed_entry: UsgsEarthquakeHazardsProgramFeedEntry | None = self._feed_manager.get_entry(self._external_id)
        if feed_entry:
            self._update_from_feed(feed_entry)

    def _update_from_feed(self, feed_entry: UsgsEarthquakeHazardsProgramFeedEntry) -> None:
        """Update the internal state from the provided feed entry."""
        self._attr_name: str = feed_entry.title
        self._attr_distance: float = feed_entry.distance_to_home
        self._attr_latitude: float = feed_entry.coordinates[0]
        self._attr_longitude: float = feed_entry.coordinates[1]
        self._attr_attribution: str = feed_entry.attribution
        self._place: str = feed_entry.place
        self._magnitude: float = feed_entry.magnitude
        self._time: datetime = feed_entry.time
        self._updated: datetime = feed_entry.updated
        self._status: str = feed_entry.status
        self._type: str = feed_entry.type
        self._alert: str = feed_entry.alert

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the device state attributes."""
        return {
            key: value for key, value in (
                (ATTR_EXTERNAL_ID, self._external_id),
                (ATTR_PLACE, self._place),
                (ATTR_MAGNITUDE, self._magnitude),
                (ATTR_TIME, self._time),
                (ATTR_UPDATED, self._updated),
                (ATTR_STATUS, self._status),
                (ATTR_TYPE, self._type),
                (ATTR_ALERT, self._alert)
            ) if value or isinstance(value, bool)
        }

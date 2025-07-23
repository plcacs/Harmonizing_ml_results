"""Support for Queensland Bushfire Alert Feeds."""
from __future__ import annotations
from collections.abc import Callable
from datetime import timedelta
import logging
from typing import Any, Optional, Tuple, List, Dict
from georss_qld_bushfire_alert_client import QldBushfireAlertFeedEntry, QldBushfireAlertFeedManager
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

_LOGGER = logging.getLogger(__name__)

ATTR_CATEGORY = 'category'
ATTR_EXTERNAL_ID = 'external_id'
ATTR_PUBLICATION_DATE = 'publication_date'
ATTR_STATUS = 'status'
ATTR_UPDATED_DATE = 'updated_date'

CONF_CATEGORIES = 'categories'

DEFAULT_RADIUS_IN_KM: float = 20.0
SCAN_INTERVAL: timedelta = timedelta(minutes=5)

SIGNAL_DELETE_ENTITY: str = 'qld_bushfire_delete_{}'
SIGNAL_UPDATE_ENTITY: str = 'qld_bushfire_update_{}'

SOURCE: str = 'qld_bushfire'

VALID_CATEGORIES: List[str] = [
    'Emergency Warning',
    'Watch and Act',
    'Advice',
    'Notification',
    'Information',
]

PLATFORM_SCHEMA = GEO_LOCATION_PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_LATITUDE): cv.latitude,
    vol.Optional(CONF_LONGITUDE): cv.longitude,
    vol.Optional(CONF_RADIUS, default=DEFAULT_RADIUS_IN_KM): vol.Coerce(float),
    vol.Optional(CONF_CATEGORIES, default=[]): vol.All(cv.ensure_list, [vol.In(VALID_CATEGORIES)]),
})


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the Queensland Bushfire Alert Feed platform."""
    scan_interval: timedelta = config.get(CONF_SCAN_INTERVAL, SCAN_INTERVAL)
    coordinates: Tuple[float, float] = (
        config.get(CONF_LATITUDE, hass.config.latitude),
        config.get(CONF_LONGITUDE, hass.config.longitude),
    )
    radius_in_km: float = config[CONF_RADIUS]
    categories: List[str] = config[CONF_CATEGORIES]
    feed = QldBushfireFeedEntityManager(
        hass, add_entities, scan_interval, coordinates, radius_in_km, categories
    )

    def start_feed_manager(event: Event) -> None:
        """Start feed manager."""
        feed.startup()

    hass.bus.listen_once(EVENT_HOMEASSISTANT_START, start_feed_manager)


class QldBushfireFeedEntityManager:
    """Feed Entity Manager for Qld Bushfire Alert GeoRSS feed."""

    def __init__(
        self,
        hass: HomeAssistant,
        add_entities: AddEntitiesCallback,
        scan_interval: timedelta,
        coordinates: Tuple[float, float],
        radius_in_km: float,
        categories: List[str],
    ) -> None:
        """Initialize the Feed Entity Manager."""
        self._hass: HomeAssistant = hass
        self._feed_manager: QldBushfireAlertFeedManager = QldBushfireAlertFeedManager(
            self._generate_entity,
            self._update_entity,
            self._remove_entity,
            coordinates,
            filter_radius=radius_in_km,
            filter_categories=categories,
        )
        self._add_entities: AddEntitiesCallback = add_entities
        self._scan_interval: timedelta = scan_interval

    def startup(self) -> None:
        """Start up this manager."""
        self._feed_manager.update()
        self._init_regular_updates()

    def _init_regular_updates(self) -> None:
        """Schedule regular updates at the specified interval."""
        track_time_interval(
            self._hass,
            lambda now: self._feed_manager.update(),
            self._scan_interval,
            cancel_on_shutdown=True,
        )

    def get_entry(self, external_id: str) -> Optional[QldBushfireAlertFeedEntry]:
        """Get feed entry by external id."""
        return self._feed_manager.feed_entries.get(external_id)

    def _generate_entity(self, external_id: str) -> None:
        """Generate new entity."""
        new_entity = QldBushfireLocationEvent(self, external_id)
        self._add_entities([new_entity], True)

    def _update_entity(self, external_id: str) -> None:
        """Update entity."""
        dispatcher_send(self._hass, SIGNAL_UPDATE_ENTITY.format(external_id))

    def _remove_entity(self, external_id: str) -> None:
        """Remove entity."""
        dispatcher_send(self._hass, SIGNAL_DELETE_ENTITY.format(external_id))


class QldBushfireLocationEvent(GeolocationEvent):
    """Represents an external event with Qld Bushfire feed data."""

    _attr_icon: str = 'mdi:fire'
    _attr_should_poll: bool = False
    _attr_source: str = SOURCE
    _attr_unit_of_measurement: UnitOfLength = UnitOfLength.KILOMETERS

    def __init__(self, feed_manager: QldBushfireFeedEntityManager, external_id: str) -> None:
        """Initialize entity with data from feed entry."""
        self._feed_manager: QldBushfireFeedEntityManager = feed_manager
        self._external_id: str = external_id
        self._category: Optional[str] = None
        self._publication_date: Optional[str] = None
        self._updated_date: Optional[str] = None
        self._status: Optional[str] = None

    async def async_added_to_hass(self) -> None:
        """Call when entity is added to hass."""
        self._remove_signal_delete = async_dispatcher_connect(
            self.hass,
            SIGNAL_DELETE_ENTITY.format(self._external_id),
            self._delete_callback,
        )
        self._remove_signal_update = async_dispatcher_connect(
            self.hass,
            SIGNAL_UPDATE_ENTITY.format(self._external_id),
            self._update_callback,
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
        feed_entry: Optional[QldBushfireAlertFeedEntry] = self._feed_manager.get_entry(self._external_id)
        if feed_entry:
            self._update_from_feed(feed_entry)

    def _update_from_feed(self, feed_entry: QldBushfireAlertFeedEntry) -> None:
        """Update the internal state from the provided feed entry."""
        self._attr_name: str = feed_entry.title
        self._attr_distance: float = feed_entry.distance_to_home
        self._attr_latitude: float = feed_entry.coordinates[0]
        self._attr_longitude: float = feed_entry.coordinates[1]
        self._attr_attribution: str = feed_entry.attribution
        self._category = feed_entry.category
        self._publication_date = feed_entry.published
        self._updated_date = feed_entry.updated
        self._status = feed_entry.status

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the device state attributes."""
        return {
            key: value
            for key, value in (
                (ATTR_EXTERNAL_ID, self._external_id),
                (ATTR_CATEGORY, self._category),
                (ATTR_PUBLICATION_DATE, self._publication_date),
                (ATTR_UPDATED_DATE, self._updated_date),
                (ATTR_STATUS, self._status),
            )
            if value or isinstance(value, bool)
        }

from __future__ import annotations
import logging
from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_LATITUDE, ATTR_LONGITUDE, UnitOfLength
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import DistanceConverter
from .const import ATTR_ACTIVITY, ATTR_DISTANCE, ATTR_EXTERNAL_ID, ATTR_HAZARDS, DEFAULT_ICON, DOMAIN, FEED, IMPERIAL_UNITS
_LOGGER: logging.Logger = logging.getLogger(__name__)
ATTR_LAST_UPDATE: str = 'feed_last_update'
ATTR_LAST_UPDATE_SUCCESSFUL: str = 'feed_last_update_successful'

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the GeoNet NZ Volcano Feed platform."""
    manager = hass.data[DOMAIN][FEED][entry.entry_id]

    @callback
    def async_add_sensor(feed_manager: FeedManager, external_id: str, unit_system: str) -> None:
        """Add sensor entity from feed."""
        new_entity = GeonetnzVolcanoSensor(entry.entry_id, feed_manager, external_id, unit_system)
        _LOGGER.debug('Adding sensor %s', new_entity)
        async_add_entities([new_entity], True)
    manager.listeners.append(async_dispatcher_connect(hass, manager.async_event_new_entity(), async_add_sensor))
    hass.async_create_task(manager.async_update())
    _LOGGER.debug('Sensor setup done')

class GeonetnzVolcanoSensor(SensorEntity):
    """Represents an external event with GeoNet NZ Volcano feed data."""
    _attr_should_poll: bool = False

    def __init__(self, config_entry_id: str, feed_manager: FeedManager, external_id: str, unit_system: str) -> None:
        """Initialize entity with data from feed entry."""
        self._config_entry_id: str = config_entry_id
        self._feed_manager: FeedManager = feed_manager
        self._external_id: str = external_id
        self._attr_unique_id: str = f'{config_entry_id}_{external_id}'
        self._unit_system: str = unit_system
        self._title: str = None
        self._distance: float = None
        self._latitude: float = None
        self._longitude: float = None
        self._attribution: str = None
        self._alert_level: str = None
        self._activity: str = None
        self._hazards: str = None
        self._feed_last_update: dt_util.dt.datetime = None
        self._feed_last_update_successful: dt_util.dt.datetime = None
        self._remove_signal_update: Optional[Callable] = None

    async def async_added_to_hass(self) -> None:
        """Call when entity is added to hass."""
        self._remove_signal_update = async_dispatcher_connect(self.hass, f'geonetnz_volcano_update_{self._external_id}', self._update_callback)

    async def async_will_remove_from_hass(self) -> None:
        """Call when entity will be removed from hass."""
        if self._remove_signal_update:
            self._remove_signal_update()

    @callback
    def _update_callback(self) -> None:
        """Call update method."""
        self.async_schedule_update_ha_state(True)

    async def async_update(self) -> None:
        """Update this entity from the data held in the feed manager."""
        _LOGGER.debug('Updating %s', self._external_id)
        feed_entry = self._feed_manager.get_entry(self._external_id)
        last_update = self._feed_manager.last_update()
        last_update_successful = self._feed_manager.last_update_successful()
        if feed_entry:
            self._update_from_feed(feed_entry, last_update, last_update_successful)

    def _update_from_feed(self, feed_entry: FeedEntry, last_update: dt_util.dt.datetime, last_update_successful: dt_util.dt.datetime) -> None:
        """Update the internal state from the provided feed entry."""
        self._title: str = feed_entry.title
        if self._unit_system == IMPERIAL_UNITS:
            self._distance: float = round(DistanceConverter.convert(feed_entry.distance_to_home, UnitOfLength.KILOMETERS, UnitOfLength.MILES), 1)
        else:
            self._distance: float = round(feed_entry.distance_to_home, 1)
        self._latitude: float = round(feed_entry.coordinates[0], 5)
        self._longitude: float = round(feed_entry.coordinates[1], 5)
        self._attr_attribution: str = feed_entry.attribution
        self._alert_level: str = feed_entry.alert_level
        self._activity: str = feed_entry.activity
        self._hazards: str = feed_entry.hazards
        self._feed_last_update: dt_util.dt.datetime = dt_util.as_utc(last_update) if last_update else None
        self._feed_last_update_successful: dt_util.dt.datetime = dt_util.as_utc(last_update_successful) if last_update_successful else None

    @property
    def native_value(self) -> str:
        """Return the state of the sensor."""
        return self._alert_level

    @property
    def icon(self) -> str:
        """Return the icon to use in the frontend, if any."""
        return DEFAULT_ICON

    @property
    def name(self) -> str:
        """Return the name of the entity."""
        return f'Volcano {self._title}'

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit of measurement."""
        return 'alert level'

    @property
    def extra_state_attributes(self) -> Dict[str, Union[str, float, dt_util.dt.datetime]]:
        """Return the device state attributes."""
        return {key: value for key, value in ((ATTR_EXTERNAL_ID, self._external_id), (ATTR_ACTIVITY, self._activity), (ATTR_HAZARDS, self._hazards), (ATTR_LONGITUDE, self._longitude), (ATTR_LATITUDE, self._latitude), (ATTR_DISTANCE, self._distance), (ATTR_LAST_UPDATE, self._feed_last_update), (ATTR_LAST_UPDATE_SUCCESSFUL, self._feed_last_update_successful)) if value or isinstance(value, bool)}

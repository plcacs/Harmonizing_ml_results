"""Feed Entity Manager Sensor support for GeoNet NZ Quakes Feeds."""
from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Optional
from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import dt as dt_util
from .const import DOMAIN, FEED

_LOGGER = logging.getLogger(__name__)

ATTR_STATUS = 'status'
ATTR_LAST_UPDATE = 'last_update'
ATTR_LAST_UPDATE_SUCCESSFUL = 'last_update_successful'
ATTR_LAST_TIMESTAMP = 'last_timestamp'
ATTR_CREATED = 'created'
ATTR_UPDATED = 'updated'
ATTR_REMOVED = 'removed'
DEFAULT_ICON = 'mdi:pulse'
DEFAULT_UNIT_OF_MEASUREMENT = 'quakes'
PARALLEL_UPDATES = 0

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the GeoNet NZ Quakes Feed platform."""
    manager = hass.data[DOMAIN][FEED][entry.entry_id]
    sensor = GeonetnzQuakesSensor(entry.entry_id, entry.unique_id, entry.title, manager)
    async_add_entities([sensor])
    _LOGGER.debug('Sensor setup done')

class GeonetnzQuakesSensor(SensorEntity):
    """Status sensor for the GeoNet NZ Quakes integration."""
    _attr_should_poll = False

    def __init__(
        self,
        config_entry_id: str,
        config_unique_id: str,
        config_title: str,
        manager: Any,
    ) -> None:
        """Initialize entity."""
        self._config_entry_id = config_entry_id
        self._config_unique_id = config_unique_id
        self._config_title = config_title
        self._manager = manager
        self._status: Optional[str] = None
        self._last_update: Optional[dt_util.dt.datetime] = None
        self._last_update_successful: Optional[dt_util.dt.datetime] = None
        self._last_timestamp: Optional[dt_util.dt.datetime] = None
        self._total: Optional[int] = None
        self._created: Optional[int] = None
        self._updated: Optional[int] = None
        self._removed: Optional[int] = None
        self._remove_signal_status: Optional[Callable[[], None]] = None

    async def async_added_to_hass(self) -> None:
        """Call when entity is added to hass."""
        self._remove_signal_status = async_dispatcher_connect(
            self.hass,
            f'geonetnz_quakes_status_{self._config_entry_id}',
            self._update_status_callback,
        )
        _LOGGER.debug('Waiting for updates %s', self._config_entry_id)
        await self.async_update()

    async def async_will_remove_from_hass(self) -> None:
        """Call when entity will be removed from hass."""
        if self._remove_signal_status:
            self._remove_signal_status()

    @callback
    def _update_status_callback(self) -> None:
        """Call status update method."""
        _LOGGER.debug('Received status update for %s', self._config_entry_id)
        self.async_schedule_update_ha_state(True)

    async def async_update(self) -> None:
        """Update this entity from the data held in the feed manager."""
        _LOGGER.debug('Updating %s', self._config_entry_id)
        if self._manager:
            status_info = self._manager.status_info()
            if status_info:
                self._update_from_status_info(status_info)

    def _update_from_status_info(self, status_info: Any) -> None:
        """Update the internal state from the provided information."""
        self._status = status_info.status
        self._last_update = dt_util.as_utc(status_info.last_update) if status_info.last_update else None
        if status_info.last_update_successful:
            self._last_update_successful = dt_util.as_utc(status_info.last_update_successful)
        else:
            self._last_update_successful = None
        self._last_timestamp = status_info.last_timestamp
        self._total = status_info.total
        self._created = status_info.created
        self._updated = status_info.updated
        self._removed = status_info.removed

    @property
    def native_value(self) -> Optional[int]:
        """Return the state of the sensor."""
        return self._total

    @property
    def unique_id(self) -> str:
        """Return a unique ID containing latitude/longitude."""
        return self._config_unique_id

    @property
    def name(self) -> str:
        """Return the name of the entity."""
        return f'GeoNet NZ Quakes ({self._config_title})'

    @property
    def icon(self) -> str:
        """Return the icon to use in the frontend, if any."""
        return DEFAULT_ICON

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit of measurement."""
        return DEFAULT_UNIT_OF_MEASUREMENT

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the device state attributes."""
        return {
            key: value
            for key, value in (
                (ATTR_STATUS, self._status),
                (ATTR_LAST_UPDATE, self._last_update),
                (ATTR_LAST_UPDATE_SUCCESSFUL, self._last_update_successful),
                (ATTR_LAST_TIMESTAMP, self._last_timestamp),
                (ATTR_CREATED, self._created),
                (ATTR_UPDATED, self._updated),
                (ATTR_REMOVED, self._removed),
            )
            if value or isinstance(value, bool)
        }

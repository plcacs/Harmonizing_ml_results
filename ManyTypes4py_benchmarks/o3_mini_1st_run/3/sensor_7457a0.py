from __future__ import annotations
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, Optional

from googlemaps import Client
from googlemaps.distance_matrix import distance_matrix
from googlemaps.exceptions import ApiError, Timeout, TransportError
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_NAME, EVENT_HOMEASSISTANT_STARTED, UnitOfTime
from homeassistant.core import CoreState, HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.location import find_coordinates
from homeassistant.util import dt as dt_util
from .const import ATTRIBUTION, CONF_ARRIVAL_TIME, CONF_DEPARTURE_TIME, CONF_DESTINATION, CONF_ORIGIN, DEFAULT_NAME, DOMAIN

_LOGGER = logging.getLogger(__name__)
SCAN_INTERVAL: timedelta = timedelta(minutes=5)

def convert_time_to_utc(timestr: str) -> float:
    """Take a string like 08:00:00 and convert it to a unix timestamp."""
    combined: datetime = datetime.combine(dt_util.start_of_local_day(), dt_util.parse_time(timestr))
    if combined < datetime.now():
        combined = combined + timedelta(days=1)
    return dt_util.as_timestamp(combined)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up a Google travel time sensor entry."""
    api_key: str = config_entry.data[CONF_API_KEY]
    origin: str = config_entry.data[CONF_ORIGIN]
    destination: str = config_entry.data[CONF_DESTINATION]
    name: str = config_entry.data.get(CONF_NAME, DEFAULT_NAME)
    client: Client = Client(api_key, timeout=10)
    sensor = GoogleTravelTimeSensor(config_entry, name, api_key, origin, destination, client)
    async_add_entities([sensor], False)

class GoogleTravelTimeSensor(SensorEntity):
    """Representation of a Google travel time sensor."""
    _attr_attribution: str = ATTRIBUTION
    _attr_native_unit_of_measurement: str = UnitOfTime.MINUTES
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.DURATION
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(
        self,
        config_entry: ConfigEntry,
        name: str,
        api_key: str,
        origin: str,
        destination: str,
        client: Client,
    ) -> None:
        """Initialize the sensor."""
        self._attr_name: str = name
        self._attr_unique_id: str = config_entry.entry_id
        self._attr_device_info: DeviceInfo = DeviceInfo(
            entry_type=DeviceEntryType.SERVICE, identifiers={(DOMAIN, api_key)}, name=DOMAIN
        )
        self._config_entry: ConfigEntry = config_entry
        self._matrix: Optional[Dict[str, Any]] = None
        self._api_key: str = api_key
        self._client: Client = client
        self._origin: str = origin
        self._destination: str = destination
        self._resolved_origin: Optional[Any] = None
        self._resolved_destination: Optional[Any] = None

    async def async_added_to_hass(self) -> None:
        """Handle when entity is added."""
        if self.hass.state is not CoreState.running:
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, self.first_update)
        else:
            await self.first_update()

    @property
    def native_value(self) -> Optional[int]:
        """Return the state of the sensor."""
        if self._matrix is None:
            return None
        _data: Dict[str, Any] = self._matrix['rows'][0]['elements'][0]
        if 'duration_in_traffic' in _data:
            return round(_data['duration_in_traffic']['value'] / 60)
        if 'duration' in _data:
            return round(_data['duration']['value'] / 60)
        return None

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        """Return the state attributes."""
        if self._matrix is None:
            return None
        res: Dict[str, Any] = self._matrix.copy()
        options: Dict[str, Any] = self._config_entry.options.copy()
        res.update(options)
        del res['rows']
        _data: Dict[str, Any] = self._matrix['rows'][0]['elements'][0]
        if 'duration_in_traffic' in _data:
            res['duration_in_traffic'] = _data['duration_in_traffic']['text']
        if 'duration' in _data:
            res['duration'] = _data['duration']['text']
        if 'distance' in _data:
            res['distance'] = _data['distance']['text']
        res['origin'] = self._resolved_origin
        res['destination'] = self._resolved_destination
        return res

    async def first_update(self, _: Optional[Any] = None) -> None:
        """Run the first update and write the state."""
        await self.hass.async_add_executor_job(self.update)
        self.async_write_ha_state()

    def update(self) -> None:
        """Get the latest data from Google."""
        options_copy: Dict[str, Any] = self._config_entry.options.copy()
        dtime: Optional[Any] = options_copy.get(CONF_DEPARTURE_TIME)
        atime: Optional[Any] = options_copy.get(CONF_ARRIVAL_TIME)
        if dtime is not None and ':' in str(dtime):
            options_copy[CONF_DEPARTURE_TIME] = convert_time_to_utc(str(dtime))
        elif dtime is not None:
            options_copy[CONF_DEPARTURE_TIME] = dtime
        elif atime is None:
            options_copy[CONF_DEPARTURE_TIME] = 'now'
        if atime is not None and ':' in str(atime):
            options_copy[CONF_ARRIVAL_TIME] = convert_time_to_utc(str(atime))
        elif atime is not None:
            options_copy[CONF_ARRIVAL_TIME] = atime
        self._resolved_origin = find_coordinates(self.hass, self._origin)
        self._resolved_destination = find_coordinates(self.hass, self._destination)
        _LOGGER.debug('Getting update for origin: %s destination: %s', self._resolved_origin, self._resolved_destination)
        if self._resolved_destination is not None and self._resolved_origin is not None:
            try:
                self._matrix = distance_matrix(
                    self._client,
                    self._resolved_origin,
                    self._resolved_destination,
                    **options_copy,
                )
            except (ApiError, TransportError, Timeout) as ex:
                _LOGGER.error('Error getting travel time: %s', ex)
                self._matrix = None
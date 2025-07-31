from __future__ import annotations
import logging
from typing import Any, Optional, Dict, List, Union
from pyrail import iRail
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.config_entries import SOURCE_IMPORT, ConfigEntry
from homeassistant.const import ATTR_LATITUDE, ATTR_LONGITUDE, CONF_NAME, CONF_PLATFORM, CONF_SHOW_ON_MAP, UnitOfTime
from homeassistant.core import DOMAIN as HOMEASSISTANT_DOMAIN, HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util
from .const import CONF_EXCLUDE_VIAS, CONF_STATION_FROM, CONF_STATION_LIVE, CONF_STATION_TO, DOMAIN, PLATFORMS, find_station, find_station_by_name

_LOGGER = logging.getLogger(__name__)
API_FAILURE = -1
DEFAULT_NAME = 'NMBS'
DEFAULT_ICON = 'mdi:train'
DEFAULT_ICON_ALERT = 'mdi:alert-octagon'
PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_STATION_FROM): cv.string,
    vol.Required(CONF_STATION_TO): cv.string,
    vol.Optional(CONF_STATION_LIVE): cv.string,
    vol.Optional(CONF_EXCLUDE_VIAS, default=False): cv.boolean,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_SHOW_ON_MAP, default=False): cv.boolean
})

def get_time_until(departure_time: Optional[Union[int, float]]) -> int:
    """Calculate the time between now and a train's departure time."""
    if departure_time is None:
        return 0
    delta = dt_util.utc_from_timestamp(int(departure_time)) - dt_util.now()
    return round(delta.total_seconds() / 60)

def get_delay_in_minutes(delay: Union[int, float] = 0) -> int:
    """Get the delay in minutes from a delay in seconds."""
    return round(int(delay) / 60)

def get_ride_duration(departure_time: Union[int, float], arrival_time: Union[int, float], delay: Union[int, float] = 0) -> int:
    """Calculate the total travel time in minutes."""
    duration = dt_util.utc_from_timestamp(int(arrival_time)) - dt_util.utc_from_timestamp(int(departure_time))
    duration_time = int(round(duration.total_seconds() / 60))
    return duration_time + get_delay_in_minutes(delay)

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the NMBS sensor with iRail API."""
    if config[CONF_PLATFORM] == DOMAIN:
        if CONF_SHOW_ON_MAP not in config:
            config[CONF_SHOW_ON_MAP] = False
        if CONF_EXCLUDE_VIAS not in config:
            config[CONF_EXCLUDE_VIAS] = False
        station_types = [CONF_STATION_FROM, CONF_STATION_TO, CONF_STATION_LIVE]
        for station_type in station_types:
            station = find_station_by_name(hass, config[station_type]) if station_type in config else None
            if station is None and station_type in config:
                async_create_issue(
                    hass,
                    DOMAIN,
                    'deprecated_yaml_import_issue_station_not_found',
                    breaks_in_ha_version='2025.7.0',
                    is_fixable=False,
                    issue_domain=DOMAIN,
                    severity=IssueSeverity.WARNING,
                    translation_key='deprecated_yaml_import_issue_station_not_found',
                    translation_placeholders={'domain': DOMAIN, 'integration_title': 'NMBS', 'station_name': config[station_type], 'url': '/config/integrations/dashboard/add?domain=nmbs'}
                )
                return
        hass.async_create_task(hass.config_entries.flow.async_init(DOMAIN, context={'source': SOURCE_IMPORT}, data=config))
    async_create_issue(
        hass,
        HOMEASSISTANT_DOMAIN,
        f'deprecated_yaml_{DOMAIN}',
        breaks_in_ha_version='2025.7.0',
        is_fixable=False,
        issue_domain=DOMAIN,
        severity=IssueSeverity.WARNING,
        translation_key='deprecated_yaml',
        translation_placeholders={'domain': DOMAIN, 'integration_title': 'NMBS'}
    )

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback
) -> None:
    """Set up NMBS sensor entities based on a config entry."""
    api_client: iRail = iRail()
    name: Optional[str] = config_entry.data.get(CONF_NAME, None)
    show_on_map: bool = config_entry.data.get(CONF_SHOW_ON_MAP, False)
    excl_vias: bool = config_entry.data.get(CONF_EXCLUDE_VIAS, False)
    station_from: Dict[str, Any] = find_station(hass, config_entry.data[CONF_STATION_FROM])
    station_to: Dict[str, Any] = find_station(hass, config_entry.data[CONF_STATION_TO])
    async_add_entities([
        NMBSSensor(api_client, name, show_on_map, station_from, station_to, excl_vias),
        NMBSLiveBoard(api_client, station_from, station_from, station_to, excl_vias),
        NMBSLiveBoard(api_client, station_to, station_from, station_to, excl_vias)
    ])

class NMBSLiveBoard(SensorEntity):
    """Get the next train from a station's liveboard."""
    _attr_attribution: str = 'https://api.irail.be/'

    def __init__(self, api_client: iRail, live_station: Dict[str, Any], station_from: Dict[str, Any], station_to: Dict[str, Any], excl_vias: bool) -> None:
        """Initialize the sensor for getting liveboard data."""
        self._station: Dict[str, Any] = live_station
        self._api_client: iRail = api_client
        self._station_from: Dict[str, Any] = station_from
        self._station_to: Dict[str, Any] = station_to
        self._excl_vias: bool = excl_vias
        self._attrs: Dict[str, Any] = {}
        self._state: Optional[Any] = None
        self.entity_registry_enabled_default = False

    @property
    def name(self) -> str:
        """Return the sensor default name."""
        return f"Trains in {self._station['standardname']}"

    @property
    def unique_id(self) -> str:
        """Return the unique ID."""
        unique_id: str = f"{self._station['id']}_{self._station_from['id']}_{self._station_to['id']}"
        vias: str = '_excl_vias' if self._excl_vias else ''
        return f"nmbs_live_{unique_id}{vias}"

    @property
    def icon(self) -> str:
        """Return the default icon or an alert icon if delays."""
        if self._attrs and int(self._attrs.get('delay', 0)) > 0:
            return DEFAULT_ICON_ALERT
        return DEFAULT_ICON

    @property
    def native_value(self) -> Optional[Any]:
        """Return sensor state."""
        return self._state

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        """Return the sensor attributes if data is available."""
        if self._state is None or not self._attrs:
            return None
        delay: int = get_delay_in_minutes(self._attrs.get('delay', 0))
        departure: int = get_time_until(self._attrs.get('time'))
        attrs: Dict[str, Any] = {
            'departure': f'In {departure} minutes',
            'departure_minutes': departure,
            'extra_train': int(self._attrs.get('isExtra', 0)) > 0,
            'vehicle_id': self._attrs.get('vehicle'),
            'monitored_station': self._station.get('standardname')
        }
        if delay > 0:
            attrs['delay'] = f'{delay} minutes'
            attrs['delay_minutes'] = delay
        return attrs

    def update(self) -> None:
        """Set the state equal to the next departure."""
        liveboard: Any = self._api_client.get_liveboard(self._station['id'])
        if liveboard == API_FAILURE:
            _LOGGER.warning('API failed in NMBSLiveBoard')
            return
        departures: Any = liveboard.get('departures')
        if not departures:
            _LOGGER.warning('API returned invalid departures: %r', liveboard)
            return
        _LOGGER.debug('API returned departures: %r', departures)
        if departures.get('number') == '0':
            return
        next_departure: Dict[str, Any] = departures['departure'][0]
        self._attrs = next_departure
        self._state = f"Track {next_departure['platform']} - {next_departure['station']}"

class NMBSSensor(SensorEntity):
    """Get the total travel time for a given connection."""
    _attr_attribution: str = 'https://api.irail.be/'
    _attr_native_unit_of_measurement: str = UnitOfTime.MINUTES

    def __init__(self, api_client: iRail, name: Optional[str], show_on_map: bool, station_from: Dict[str, Any], station_to: Dict[str, Any], excl_vias: bool) -> None:
        """Initialize the NMBS connection sensor."""
        self._name: Optional[str] = name
        self._show_on_map: bool = show_on_map
        self._api_client: iRail = api_client
        self._station_from: Dict[str, Any] = station_from
        self._station_to: Dict[str, Any] = station_to
        self._excl_vias: bool = excl_vias
        self._attrs: Dict[str, Any] = {}
        self._state: Optional[Any] = None

    @property
    def unique_id(self) -> str:
        """Return the unique ID."""
        unique_id: str = f"{self._station_from['id']}_{self._station_to['id']}"
        vias: str = '_excl_vias' if self._excl_vias else ''
        return f"nmbs_connection_{unique_id}{vias}"

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        if self._name is None:
            return f"Train from {self._station_from['standardname']} to {self._station_to['standardname']}"
        return self._name

    @property
    def icon(self) -> str:
        """Return the sensor default icon or an alert icon if any delay."""
        if self._attrs:
            departure_delay: int = get_delay_in_minutes(self._attrs['departure'].get('delay', 0))
            if departure_delay > 0:
                return 'mdi:alert-octagon'
        return 'mdi:train'

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        """Return sensor attributes if data is available."""
        if self._state is None or not self._attrs:
            return None
        delay: int = get_delay_in_minutes(self._attrs['departure'].get('delay', 0))
        departure: int = get_time_until(self._attrs['departure'].get('time'))
        canceled: int = int(self._attrs['departure'].get('canceled', 0))
        attrs: Dict[str, Any] = {
            'destination': self._attrs['departure'].get('station'),
            'direction': self._attrs['departure'].get('direction', {}).get('name'),
            'platform_arriving': self._attrs['arrival'].get('platform'),
            'platform_departing': self._attrs['departure'].get('platform'),
            'vehicle_id': self._attrs['departure'].get('vehicle')
        }
        if canceled != 1:
            attrs['departure'] = f'In {departure} minutes'
            attrs['departure_minutes'] = departure
            attrs['canceled'] = False
        else:
            attrs['departure'] = None
            attrs['departure_minutes'] = None
            attrs['canceled'] = True
        if self._show_on_map and self.station_coordinates:
            attrs[ATTR_LATITUDE] = self.station_coordinates[0]
            attrs[ATTR_LONGITUDE] = self.station_coordinates[1]
        if self.is_via_connection and (not self._excl_vias):
            via: Dict[str, Any] = self._attrs['vias']['via'][0]
            attrs['via'] = via.get('station')
            attrs['via_arrival_platform'] = via.get('arrival', {}).get('platform')
            attrs['via_transfer_platform'] = via.get('departure', {}).get('platform')
            attrs['via_transfer_time'] = get_delay_in_minutes(via.get('timebetween', 0)) + get_delay_in_minutes(via.get('departure', {}).get('delay', 0))
        if delay > 0:
            attrs['delay'] = f'{delay} minutes'
            attrs['delay_minutes'] = delay
        return attrs

    @property
    def native_value(self) -> Optional[Any]:
        """Return the state of the device."""
        return self._state

    @property
    def station_coordinates(self) -> List[float]:
        """Get the lat, long coordinates for station."""
        if self._state is None or not self._attrs:
            return []
        latitude: float = float(self._attrs['departure']['stationinfo'].get('locationY', 0))
        longitude: float = float(self._attrs['departure']['stationinfo'].get('locationX', 0))
        return [latitude, longitude]

    @property
    def is_via_connection(self) -> bool:
        """Return whether the connection goes through another station."""
        if not self._attrs:
            return False
        return 'vias' in self._attrs and int(self._attrs['vias'].get('number', 0)) > 0

    def update(self) -> None:
        """Set the state to the duration of a connection."""
        connections: Any = self._api_client.get_connections(self._station_from['id'], self._station_to['id'])
        if connections == API_FAILURE:
            _LOGGER.warning('API failed in NMBSSensor')
            return
        connection_list: Any = connections.get('connection')
        if not connection_list:
            _LOGGER.warning('API returned invalid connection: %r', connections)
            return
        _LOGGER.debug('API returned connection: %r', connection_list)
        if int(connection_list[0]['departure'].get('left', 0)) > 0:
            next_connection: Dict[str, Any] = connection_list[1]
        else:
            next_connection = connection_list[0]
        self._attrs = next_connection
        if self._excl_vias and self.is_via_connection:
            _LOGGER.debug('Skipping update of NMBSSensor because this connection is a via')
            return
        duration: int = get_ride_duration(
            next_connection['departure'].get('time'),
            next_connection['arrival'].get('time'),
            next_connection['departure'].get('delay', 0)
        )
        self._state = duration

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.config_entries import SOURCE_IMPORT, ConfigEntry
from homeassistant.const import ATTR_LATITUDE, ATTR_LONGITUDE, CONF_NAME, CONF_PLATFORM, CONF_SHOW_ON_MAP, UnitOfTime
from homeassistant.core import DOMAIN as HOMEASSISTANT_DOMAIN, HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util
from .const import CONF_EXCLUDE_VIAS, CONF_STATION_FROM, CONF_STATION_LIVE, CONF_STATION_TO, DOMAIN, PLATFORMS, find_station, find_station_by_name
from pyrail import iRail

_LOGGER: logging.Logger = logging.getLogger(__name__)
API_FAILURE: int = -1
DEFAULT_NAME: str = 'NMBS'
DEFAULT_ICON: str = 'mdi:train'
DEFAULT_ICON_ALERT: str = 'mdi:alert-octagon'
PLATFORM_SCHEMA: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend({vol.Required(CONF_STATION_FROM): cv.string, vol.Required(CONF_STATION_TO): cv.string, vol.Optional(CONF_STATION_LIVE): cv.string, vol.Optional(CONF_EXCLUDE_VIAS, default=False): cv.boolean, vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string, vol.Optional(CONF_SHOW_ON_MAP, default=False): cv.boolean})

def get_time_until(departure_time: Optional[int] = None) -> int:
    if departure_time is None:
        return 0
    delta = dt_util.utc_from_timestamp(int(departure_time)) - dt_util.now()
    return round(delta.total_seconds() / 60)

def get_delay_in_minutes(delay: int = 0) -> int:
    return round(int(delay) / 60)

def get_ride_duration(departure_time: int, arrival_time: int, delay: int = 0) -> int:
    duration = dt_util.utc_from_timestamp(int(arrival_time)) - dt_util.utc_from_timestamp(int(departure_time))
    duration_time = int(round(duration.total_seconds() / 60))
    return duration_time + get_delay_in_minutes(delay)

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    if config[CONF_PLATFORM] == DOMAIN:
        if CONF_SHOW_ON_MAP not in config:
            config[CONF_SHOW_ON_MAP] = False
        if CONF_EXCLUDE_VIAS not in config:
            config[CONF_EXCLUDE_VIAS] = False
        station_types: List[str] = [CONF_STATION_FROM, CONF_STATION_TO, CONF_STATION_LIVE]
        for station_type in station_types:
            station: Optional[Dict[str, Any]] = find_station_by_name(hass, config[station_type]) if station_type in config else None
            if station is None and station_type in config:
                async_create_issue(hass, DOMAIN, 'deprecated_yaml_import_issue_station_not_found', breaks_in_ha_version='2025.7.0', is_fixable=False, issue_domain=DOMAIN, severity=IssueSeverity.WARNING, translation_key='deprecated_yaml_import_issue_station_not_found', translation_placeholders={'domain': DOMAIN, 'integration_title': 'NMBS', 'station_name': config[station_type], 'url': '/config/integrations/dashboard/add?domain=nmbs'})
                return
        hass.async_create_task(hass.config_entries.flow.async_init(DOMAIN, context={'source': SOURCE_IMPORT}, data=config))
    async_create_issue(hass, HOMEASSISTANT_DOMAIN, f'deprecated_yaml_{DOMAIN}', breaks_in_ha_version='2025.7.0', is_fixable=False, issue_domain=DOMAIN, severity=IssueSeverity.WARNING, translation_key='deprecated_yaml', translation_placeholders={'domain': DOMAIN, 'integration_title': 'NMBS'})

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    api_client: iRail = iRail()
    name: Optional[str] = config_entry.data.get(CONF_NAME, None)
    show_on_map: bool = config_entry.data.get(CONF_SHOW_ON_MAP, False)
    excl_vias: bool = config_entry.data.get(CONF_EXCLUDE_VIAS, False)
    station_from: Dict[str, Any] = find_station(hass, config_entry.data[CONF_STATION_FROM])
    station_to: Dict[str, Any] = find_station(hass, config_entry.data[CONF_STATION_TO])
    async_add_entities([NMBSSensor(api_client, name, show_on_map, station_from, station_to, excl_vias), NMBSLiveBoard(api_client, station_from, station_from, station_to, excl_vias)])

class NMBSLiveBoard(SensorEntity):
    _attr_attribution: str = 'https://api.irail.be/'

    def __init__(self, api_client: iRail, live_station: Dict[str, Any], station_from: Dict[str, Any], station_to: Dict[str, Any], excl_vias: bool):
        self._station: Dict[str, Any] = live_station
        self._api_client: iRail = api_client
        self._station_from: Dict[str, Any] = station_from
        self._station_to: Dict[str, Any] = station_to
        self._excl_vias: bool = excl_vias
        self._attrs: Dict[str, Any] = {}
        self._state: Optional[str] = None
        self.entity_registry_enabled_default: bool = False

    @property
    def name(self) -> str:
        return f'Trains in {self._station['standardname']}'

    @property
    def unique_id(self) -> str:
        unique_id: str = f'{self._station['id']}_{self._station_from['id']}_{self._station_to['id']}'
        vias: str = '_excl_vias' if self._excl_vias else ''
        return f'nmbs_live_{unique_id}{vias}'

    @property
    def icon(self) -> str:
        if self._attrs and int(self._attrs['delay']) > 0:
            return DEFAULT_ICON_ALERT
        return DEFAULT_ICON

    @property
    def native_value(self) -> Optional[str]:
        return self._state

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        if self._state is None or not self._attrs:
            return None
        delay: int = get_delay_in_minutes(self._attrs['delay'])
        departure: int = get_time_until(self._attrs['time'])
        attrs: Dict[str, Any] = {'departure': f'In {departure} minutes', 'departure_minutes': departure, 'extra_train': int(self._attrs['isExtra']) > 0, 'vehicle_id': self._attrs['vehicle'], 'monitored_station': self._station['standardname']}
        if delay > 0:
            attrs['delay'] = f'{delay} minutes'
            attrs['delay_minutes'] = delay
        return attrs

    def update(self) -> None:
        liveboard: Dict[str, Any] = self._api_client.get_liveboard(self._station['id'])
        if liveboard == API_FAILURE:
            _LOGGER.warning('API failed in NMBSLiveBoard')
            return
        if not (departures := liveboard.get('departures')):
            _LOGGER.warning('API returned invalid departures: %r', liveboard)
            return
        if departures['number'] == '0':
            return
        next_departure: Dict[str, Any] = departures['departure'][0]
        self._attrs = next_departure
        self._state = f'Track {next_departure['platform']} - {next_departure['station']}'

class NMBSSensor(SensorEntity):
    _attr_attribution: str = 'https://api.irail.be/'
    _attr_native_unit_of_measurement: UnitOfTime = UnitOfTime.MINUTES

    def __init__(self, api_client: iRail, name: Optional[str], show_on_map: bool, station_from: Dict[str, Any], station_to: Dict[str, Any], excl_vias: bool):
        self._name: Optional[str] = name
        self._show_on_map: bool = show_on_map
        self._api_client: iRail = api_client
        self._station_from: Dict[str, Any] = station_from
        self._station_to: Dict[str, Any] = station_to
        self._excl_vias: bool = excl_vias
        self._attrs: Dict[str, Any] = {}
        self._state: Optional[int] = None

    @property
    def unique_id(self) -> str:
        unique_id: str = f'{self._station_from['id']}_{self._station_to['id']}'
        vias: str = '_excl_vias' if self._excl_vias else ''
        return f'nmbs_connection_{unique_id}{vias}'

    @property
    def name(self) -> str:
        if self._name is None:
            return f'Train from {self._station_from['standardname']} to {self._station_to['standardname']}'
        return self._name

    @property
    def icon(self) -> str:
        if self._attrs:
            delay: int = get_delay_in_minutes(self._attrs['departure']['delay'])
            if delay > 0:
                return 'mdi:alert-octagon'
        return 'mdi:train'

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        if self._state is None or not self._attrs:
            return None
        delay: int = get_delay_in_minutes(self._attrs['departure']['delay'])
        departure: int = get_time_until(self._attrs['departure']['time'])
        canceled: int = int(self._attrs['departure']['canceled'])
        attrs: Dict[str, Any] = {'destination': self._attrs['departure']['station'], 'direction': self._attrs['departure']['direction']['name'], 'platform_arriving': self._attrs['arrival']['platform'], 'platform_departing': self._attrs['departure']['platform'], 'vehicle_id': self._attrs['departure']['vehicle']}
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
            attrs['via'] = via['station']
            attrs['via_arrival_platform'] = via['arrival']['platform']
            attrs['via_transfer_platform'] = via['departure']['platform']
            attrs['via_transfer_time'] = get_delay_in_minutes(via['timebetween']) + get_delay_in_minutes(via['departure']['delay'])
        if delay > 0:
            attrs['delay'] = f'{delay} minutes'
            attrs['delay_minutes'] = delay
        return attrs

    @property
    def native_value(self) -> Optional[int]:
        return self._state

    @property
    def station_coordinates(self) -> List[float]:
        if self._state is None or not self._attrs:
            return []
        latitude: float = float(self._attrs['departure']['stationinfo']['locationY'])
        longitude: float = float(self._attrs['departure']['stationinfo']['locationX'])
        return [latitude, longitude]

    @property
    def is_via_connection(self) -> bool:
        if not self._attrs:
            return False
        return 'vias' in self._attrs and int(self._attrs['vias']['number']) > 0

    def update(self) -> None:
        connections: Dict[str, Any] = self._api_client.get_connections(self._station_from['id'], self._station_to['id'])
        if connections == API_FAILURE:
            _LOGGER.warning('API failed in NMBSSensor')
            return
        if not (connection := connections.get('connection')):
            _LOGGER.warning('API returned invalid connection: %r', connections)
            return
        if int(connection[0]['departure']['left']) > 0:
            next_connection: Dict[str, Any] = connection[1]
        else:
            next_connection: Dict[str, Any] = connection[0]
        self._attrs = next_connection
        if self._excl_vias and self.is_via_connection:
            _LOGGER.debug('Skipping update of NMBSSensor because this connection is a via')
            return
        duration: int = get_ride_duration(next_connection['departure']['time'], next_connection['arrival']['time'], next_connection['departure']['delay'])
        self._state = duration

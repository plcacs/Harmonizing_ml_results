"""Real-time information about public transport departures in Norway."""
from __future__ import annotations
from datetime import datetime, timedelta
from random import randint
from enturclient import EnturPublicTransportData
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE, CONF_NAME, CONF_SHOW_ON_MAP, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle, dt as dt_util

API_CLIENT_NAME: str = 'homeassistant-{}'
CONF_STOP_IDS: str = 'stop_ids'
CONF_EXPAND_PLATFORMS: str = 'expand_platforms'
CONF_WHITELIST_LINES: str = 'line_whitelist'
CONF_OMIT_NON_BOARDING: str = 'omit_non_boarding'
CONF_NUMBER_OF_DEPARTURES: str = 'number_of_departures'
DEFAULT_NAME: str = 'Entur'
DEFAULT_ICON_KEY: str = 'bus'
ICONS: dict[str, str] = {'air': 'mdi:airplane', 'bus': 'mdi:bus', 'metro': 'mdi:subway', 'rail': 'mdi:train', 'tram': 'mdi:tram', 'water': 'mdi:ferry'}
SCAN_INTERVAL: timedelta = timedelta(seconds=45)
PLATFORM_SCHEMA: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend({vol.Required(CONF_STOP_IDS): vol.All(cv.ensure_list, [cv.string]), vol.Optional(CONF_EXPAND_PLATFORMS, default=True): cv.boolean, vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string, vol.Optional(CONF_SHOW_ON_MAP, default=False): cv.boolean, vol.Optional(CONF_WHITELIST_LINES, default=[]): cv.ensure_list, vol.Optional(CONF_OMIT_NON_BOARDING, default=True): cv.boolean, vol.Optional(CONF_NUMBER_OF_DEPARTURES, default=2): vol.All(cv.positive_int, vol.Range(min=2, max=10))})
ATTR_STOP_ID: str = 'stop_id'
ATTR_ROUTE: str = 'route'
ATTR_ROUTE_ID: str = 'route_id'
ATTR_EXPECTED_AT: str = 'due_at'
ATTR_DELAY: str = 'delay'
ATTR_REALTIME: str = 'real_time'
ATTR_NEXT_UP_IN: str = 'next_due_in'
ATTR_NEXT_UP_ROUTE: str = 'next_route'
ATTR_NEXT_UP_ROUTE_ID: str = 'next_route_id'
ATTR_NEXT_UP_AT: str = 'next_due_at'
ATTR_NEXT_UP_DELAY: str = 'next_delay'
ATTR_NEXT_UP_REALTIME: str = 'next_real_time'
ATTR_TRANSPORT_MODE: str = 'transport_mode'

def due_in_minutes(timestamp: datetime | None) -> int | None:
    """Get the time in minutes from a timestamp."""
    if timestamp is None:
        return None
    diff = timestamp - dt_util.now()
    return int(diff.total_seconds() / 60)

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType | None = None) -> None:
    """Set up the Entur public transport sensor."""
    expand: bool = config[CONF_EXPAND_PLATFORMS]
    line_whitelist: list[str] = config[CONF_WHITELIST_LINES]
    name: str = config[CONF_NAME]
    show_on_map: bool = config[CONF_SHOW_ON_MAP]
    stop_ids: list[str] = config[CONF_STOP_IDS]
    omit_non_boarding: bool = config[CONF_OMIT_NON_BOARDING]
    number_of_departures: int = config[CONF_NUMBER_OF_DEPARTURES]
    stops: list[str] = [s for s in stop_ids if 'StopPlace' in s]
    quays: list[str] = [s for s in stop_ids if 'Quay' in s]
    data: EnturPublicTransportData = EnturPublicTransportData(API_CLIENT_NAME.format(str(randint(100000, 999999))), stops=stops, quays=quays, line_whitelist=line_whitelist, omit_non_boarding=omit_non_boarding, number_of_departures=number_of_departures, web_session=async_get_clientsession(hass))
    if expand:
        await data.expand_all_quays()
    await data.update()
    proxy: EnturProxy = EnturProxy(data)
    entities: list[EnturPublicTransportSensor] = []
    for place in data.all_stop_places_quays():
        try:
            given_name: str = f'{name} {data.get_stop_info(place).name}'
        except KeyError:
            given_name: str = f'{name} {place}'
        entities.append(EnturPublicTransportSensor(proxy, given_name, place, show_on_map))
    async_add_entities(entities, True)

class EnturProxy:
    """Proxy for the Entur client.

    Ensure throttle to not hit rate limiting on the API.
    """

    def __init__(self, api: EnturPublicTransportData) -> None:
        """Initialize the proxy."""
        self._api: EnturPublicTransportData = api

    @Throttle(timedelta(seconds=15))
    async def async_update(self) -> None:
        """Update data in client."""
        await self._api.update()

    def get_stop_info(self, stop_id: str) -> dict[str, any] | None:
        """Get info about specific stop place."""
        return self._api.get_stop_info(stop_id)

class EnturPublicTransportSensor(SensorEntity):
    """Implementation of a Entur public transport sensor."""
    _attr_attribution: str = 'Data provided by entur.org under NLOD'

    def __init__(self, api: EnturProxy, name: str, stop: str, show_on_map: bool) -> None:
        """Initialize the sensor."""
        self.api: EnturProxy = api
        self._stop: str = stop
        self._show_on_map: bool = show_on_map
        self._name: str = name
        self._state: int | None = None
        self._icon: str = ICONS[DEFAULT_ICON_KEY]
        self._attributes: dict[str, any] = {}

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def native_value(self) -> int | None:
        """Return the state of the sensor."""
        return self._state

    @property
    def extra_state_attributes(self) -> dict[str, any]:
        """Return the state attributes."""
        self._attributes[ATTR_STOP_ID] = self._stop
        return self._attributes

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit this state is expressed in."""
        return UnitOfTime.MINUTES

    @property
    def icon(self) -> str:
        """Icon to use in the frontend."""
        return self._icon

    async def async_update(self) -> None:
        """Get the latest data and update the states."""
        await self.api.async_update()
        self._attributes: dict[str, any] = {}
        data: dict[str, any] = self.api.get_stop_info(self._stop)
        if data is None:
            self._state = None
            return
        if self._show_on_map and data.get('latitude') and data.get('longitude'):
            self._attributes[CONF_LATITUDE] = data['latitude']
            self._attributes[CONF_LONGITUDE] = data['longitude']
        if not (calls := data.get('estimated_calls')):
            self._state = None
            return
        self._state = due_in_minutes(calls[0].get('expected_departure_time'))
        self._icon = ICONS.get(calls[0].get('transport_mode'), ICONS[DEFAULT_ICON_KEY])
        self._attributes[ATTR_ROUTE] = calls[0].get('front_display')
        self._attributes[ATTR_ROUTE_ID] = calls[0].get('line_id')
        self._attributes[ATTR_EXPECTED_AT] = calls[0].get('expected_departure_time').strftime('%H:%M')
        self._attributes[ATTR_REALTIME] = calls[0].get('is_realtime')
        self._attributes[ATTR_DELAY] = calls[0].get('delay_in_min')
        number_of_calls: int = len(calls)
        if number_of_calls < 2:
            return
        self._attributes[ATTR_NEXT_UP_ROUTE] = calls[1].get('front_display')
        self._attributes[ATTR_NEXT_UP_ROUTE_ID] = calls[1].get('line_id')
        self._attributes[ATTR_NEXT_UP_AT] = calls[1].get('expected_departure_time').strftime('%H:%M')
        self._attributes[ATTR_NEXT_UP_IN] = f'{due_in_minutes(calls[1].get("expected_departure_time"))} min'
        self._attributes[ATTR_NEXT_UP_REALTIME] = calls[1].get('is_realtime')
        self._attributes[ATTR_NEXT_UP_DELAY] = calls[1].get('delay_in_min')
        if number_of_calls < 3:
            return
        for i, call in enumerate(calls[2:]):
            key_name: str = f'departure_#{i + 3}'
            self._attributes[key_name] = f'{("ca. " if not bool(call.get("is_realtime")) else "")}{call.get("expected_departure_time").strftime("%H:%M")} {call.get("front_display")}'

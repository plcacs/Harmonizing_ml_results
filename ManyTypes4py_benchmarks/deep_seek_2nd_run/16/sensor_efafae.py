"""Service for obtaining information about closer bus from Transport Yandex Service."""
from __future__ import annotations
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional
from aioymaps import CaptchaError, NoSessionError, YandexMapsRequester
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorDeviceClass, SensorEntity
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_create_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)
STOP_NAME = 'stop_name'
USER_AGENT = 'Home Assistant'
CONF_STOP_ID = 'stop_id'
CONF_ROUTE = 'routes'
DEFAULT_NAME = 'Yandex Transport'
SCAN_INTERVAL = timedelta(minutes=1)
PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({vol.Required(CONF_STOP_ID): cv.string, vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string, vol.Optional(CONF_ROUTE, default=[]): vol.All(cv.ensure_list, [cv.string])})

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the Yandex transport sensor."""
    stop_id: str = config[CONF_STOP_ID]
    name: str = config[CONF_NAME]
    routes: List[str] = config[CONF_ROUTE]
    client_session = async_create_clientsession(hass, requote_redirect_url=False)
    ymaps = YandexMapsRequester(user_agent=USER_AGENT, client_session=client_session)
    try:
        await ymaps.set_new_session()
    except CaptchaError as ex:
        _LOGGER.error('%s. You may need to disable the integration for some time', ex)
        return
    async_add_entities([DiscoverYandexTransport(ymaps, stop_id, routes, name)], True)

class DiscoverYandexTransport(SensorEntity):
    """Implementation of yandex_transport sensor."""
    _attr_attribution: str = 'Data provided by maps.yandex.ru'
    _attr_icon: str = 'mdi:bus'

    def __init__(
        self,
        requester: YandexMapsRequester,
        stop_id: str,
        routes: List[str],
        name: str
    ) -> None:
        """Initialize sensor."""
        self.requester: YandexMapsRequester = requester
        self._stop_id: str = stop_id
        self._routes: List[str] = routes
        self._state: Optional[datetime] = None
        self._name: str = name
        self._attrs: Optional[Dict[str, Any]] = None

    async def async_update(self, *, tries: int = 0) -> None:
        """Get the latest data from maps.yandex.ru and update the states."""
        attrs: Dict[str, Any] = {}
        closer_time: Optional[int] = None
        try:
            yandex_reply: Dict[str, Any] = await self.requester.get_stop_info(self._stop_id)
        except (CaptchaError, NoSessionError) as ex:
            _LOGGER.error('%s. You may need to disable the integration for some time', ex)
            return
        try:
            data: Dict[str, Any] = yandex_reply['data']
        except KeyError as key_error:
            _LOGGER.warning('Exception KeyError was captured, missing key is %s. Yandex returned: %s', key_error, yandex_reply)
            if tries > 0:
                return
            await self.requester.set_new_session()
            await self.async_update(tries=tries + 1)
            return
        stop_name: str = data['name']
        transport_list: List[Dict[str, Any]] = data['transports']
        for transport in transport_list:
            for thread in transport['threads']:
                if 'Events' not in thread['BriefSchedule']:
                    continue
                if thread.get('noBoarding') is True:
                    continue
                for event in thread['BriefSchedule']['Events']:
                    if 'railway' in transport['Types']:
                        route: str = ' - '.join([x['name'] for x in thread['EssentialStops']])
                    else:
                        route = transport['name']
                    if self._routes and route not in self._routes:
                        continue
                    if 'Estimated' not in event and 'Scheduled' not in event:
                        continue
                    departure: Dict[str, Any] = event.get('Estimated') or event['Scheduled']
                    posix_time_next: int = int(departure['value'])
                    if closer_time is None or closer_time > posix_time_next:
                        closer_time = posix_time_next
                    if route not in attrs:
                        attrs[route] = []
                    attrs[route].append(departure['text'])
        attrs[STOP_NAME] = stop_name
        if closer_time is None:
            self._state = None
        else:
            self._state = dt_util.utc_from_timestamp(closer_time).replace(microsecond=0)
        self._attrs = attrs

    @property
    def native_value(self) -> Optional[datetime]:
        """Return the state of the sensor."""
        return self._state

    @property
    def device_class(self) -> SensorDeviceClass:
        """Return the device class."""
        return SensorDeviceClass.TIMESTAMP

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        """Return the state attributes."""
        return self._attrs

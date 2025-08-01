from __future__ import annotations
from datetime import timedelta
from typing import Any, Dict, Optional
from TransportNSW import TransportNSW
import voluptuous as vol
from homeassistant.components.sensor import (
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import ATTR_MODE, CONF_API_KEY, CONF_NAME, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

ATTR_STOP_ID = 'stop_id'
ATTR_ROUTE = 'route'
ATTR_DUE_IN = 'due'
ATTR_DELAY = 'delay'
ATTR_REAL_TIME = 'real_time'
ATTR_DESTINATION = 'destination'
CONF_STOP_ID = 'stop_id'
CONF_ROUTE = 'route'
CONF_DESTINATION = 'destination'
DEFAULT_NAME = 'Next Bus'
ICONS: Dict[Optional[str], str] = {
    'Train': 'mdi:train',
    'Lightrail': 'mdi:tram',
    'Bus': 'mdi:bus',
    'Coach': 'mdi:bus',
    'Ferry': 'mdi:ferry',
    'Schoolbus': 'mdi:bus',
    'n/a': 'mdi:clock',
    None: 'mdi:clock'
}
SCAN_INTERVAL = timedelta(seconds=60)
PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_STOP_ID): cv.string,
    vol.Required(CONF_API_KEY): cv.string,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_ROUTE, default=''): cv.string,
    vol.Optional(CONF_DESTINATION, default=''): cv.string
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    stop_id: str = config[CONF_STOP_ID]
    api_key: str = config[CONF_API_KEY]
    route: str = config.get(CONF_ROUTE)
    destination: str = config.get(CONF_DESTINATION)
    name: str = config.get(CONF_NAME)
    data: PublicTransportData = PublicTransportData(stop_id, route, destination, api_key)
    add_entities([TransportNSWSensor(data, stop_id, name)], True)


class TransportNSWSensor(SensorEntity):
    _attr_attribution: str = 'Data provided by Transport NSW'
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.DURATION
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, data: PublicTransportData, stop_id: str, name: str) -> None:
        self.data: PublicTransportData = data
        self._name: str = name
        self._stop_id: str = stop_id
        self._times: Optional[Dict[str, Optional[Any]]] = None
        self._state: Optional[Any] = None
        self._icon: str = ICONS[None]

    @property
    def name(self) -> str:
        return self._name

    @property
    def native_value(self) -> Optional[Any]:
        return self._state

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Optional[Any]]]:
        if self._times is not None:
            return {
                ATTR_DUE_IN: self._times[ATTR_DUE_IN],
                ATTR_STOP_ID: self._stop_id,
                ATTR_ROUTE: self._times[ATTR_ROUTE],
                ATTR_DELAY: self._times[ATTR_DELAY],
                ATTR_REAL_TIME: self._times[ATTR_REAL_TIME],
                ATTR_DESTINATION: self._times[ATTR_DESTINATION],
                ATTR_MODE: self._times[ATTR_MODE]
            }
        return None

    @property
    def native_unit_of_measurement(self) -> str:
        return UnitOfTime.MINUTES

    @property
    def icon(self) -> str:
        return self._icon

    def update(self) -> None:
        self.data.update()
        self._times = self.data.info
        self._state = self._times[ATTR_DUE_IN]
        self._icon = ICONS[self._times[ATTR_MODE]]


def _get_value(value: Any) -> Optional[Any]:
    return None if value is None or value == 'n/a' else value


class PublicTransportData:
    def __init__(self, stop_id: str, route: str, destination: str, api_key: str) -> None:
        self._stop_id: str = stop_id
        self._route: str = route
        self._destination: str = destination
        self._api_key: str = api_key
        self.info: Dict[str, Optional[Any]] = {
            ATTR_ROUTE: self._route,
            ATTR_DUE_IN: None,
            ATTR_DELAY: None,
            ATTR_REAL_TIME: None,
            ATTR_DESTINATION: None,
            ATTR_MODE: None
        }
        self.tnsw: TransportNSW = TransportNSW()

    def update(self) -> None:
        _data: Dict[str, Any] = self.tnsw.get_departures(self._stop_id, self._route, self._destination, self._api_key)
        self.info = {
            ATTR_ROUTE: _get_value(_data['route']),
            ATTR_DUE_IN: _get_value(_data['due']),
            ATTR_DELAY: _get_value(_data['delay']),
            ATTR_REAL_TIME: _get_value(_data['real_time']),
            ATTR_DESTINATION: _get_value(_data['destination']),
            ATTR_MODE: _get_value(_data['mode'])
        }
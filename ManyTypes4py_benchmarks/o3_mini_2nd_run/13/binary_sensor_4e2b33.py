from __future__ import annotations
import datetime
import logging
from typing import Any, Dict, List, Optional, Union

from concord232 import client as concord232_client
import requests
import voluptuous as vol

from homeassistant.components.binary_sensor import (
    DEVICE_CLASSES_SCHEMA as BINARY_SENSOR_DEVICE_CLASSES_SCHEMA,
    PLATFORM_SCHEMA as BINARY_SENSOR_PLATFORM_SCHEMA,
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util

_LOGGER: logging.Logger = logging.getLogger(__name__)

CONF_EXCLUDE_ZONES: str = 'exclude_zones'
CONF_ZONE_TYPES: str = 'zone_types'
DEFAULT_HOST: str = 'localhost'
DEFAULT_NAME: str = 'Alarm'
DEFAULT_PORT: str = '5007'
DEFAULT_SSL: bool = False
SCAN_INTERVAL: datetime.timedelta = datetime.timedelta(seconds=10)

# ZoneType can be either a BinarySensorDeviceClass or a string like 'water'
ZoneType = Union[BinarySensorDeviceClass, str]
ZONE_TYPES_SCHEMA: vol.Schema = vol.Schema({cv.positive_int: BINARY_SENSOR_DEVICE_CLASSES_SCHEMA})

PLATFORM_SCHEMA: vol.Schema = BINARY_SENSOR_PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_EXCLUDE_ZONES, default=[]): vol.All(cv.ensure_list, [cv.positive_int]),
    vol.Optional(CONF_HOST, default=DEFAULT_HOST): cv.string,
    vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
    vol.Optional(CONF_ZONE_TYPES, default={}): ZONE_TYPES_SCHEMA,
})


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    host: str = config[CONF_HOST]
    port: int = config[CONF_PORT]
    exclude: List[int] = config[CONF_EXCLUDE_ZONES]
    zone_types: Dict[int, Any] = config[CONF_ZONE_TYPES]
    sensors: List[Concord232ZoneSensor] = []

    try:
        _LOGGER.debug('Initializing client')
        client: concord232_client.Client = concord232_client.Client(f'http://{host}:{port}')
        client.zones = client.list_zones()
        client.last_zone_update = dt_util.utcnow()
    except requests.exceptions.ConnectionError as ex:
        _LOGGER.error('Unable to connect to Concord232: %s', str(ex))
        return

    client.zones.sort(key=lambda zone: zone['number'])
    for zone in client.zones:
        _LOGGER.debug('Loading Zone found: %s', zone['name'])
        if zone['number'] not in exclude:
            zone_type: ZoneType = zone_types.get(zone['number'], get_opening_type(zone))
            sensors.append(Concord232ZoneSensor(hass, client, zone, zone_type))
    add_entities(sensors, True)


def get_opening_type(zone: Dict[str, Any]) -> ZoneType:
    if 'MOTION' in zone['name']:
        return BinarySensorDeviceClass.MOTION
    if 'KEY' in zone['name']:
        return BinarySensorDeviceClass.SAFETY
    if 'SMOKE' in zone['name']:
        return BinarySensorDeviceClass.SMOKE
    if 'WATER' in zone['name']:
        return 'water'
    return BinarySensorDeviceClass.OPENING


class Concord232ZoneSensor(BinarySensorEntity):
    def __init__(
        self,
        hass: HomeAssistant,
        client: concord232_client.Client,
        zone: Dict[str, Any],
        zone_type: ZoneType,
    ) -> None:
        self._hass: HomeAssistant = hass
        self._client: concord232_client.Client = client
        self._zone: Dict[str, Any] = zone
        self._number: int = zone['number']
        self._zone_type: ZoneType = zone_type

    @property
    def device_class(self) -> ZoneType:
        return self._zone_type

    @property
    def name(self) -> str:
        return self._zone['name']

    @property
    def is_on(self) -> bool:
        return bool(self._zone['state'] != 'Normal')

    def update(self) -> None:
        last_update: datetime.timedelta = dt_util.utcnow() - self._client.last_zone_update
        _LOGGER.debug('Zone: %s ', self._zone)
        if last_update > datetime.timedelta(seconds=1):
            self._client.zones = self._client.list_zones()
            self._client.last_zone_update = dt_util.utcnow()
            _LOGGER.debug('Updated from zone: %s', self._zone['name'])
        if hasattr(self._client, 'zones'):
            self._zone = next((x for x in self._client.zones if x['number'] == self._number), None)

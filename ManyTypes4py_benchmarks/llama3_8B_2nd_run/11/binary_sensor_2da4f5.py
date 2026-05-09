from __future__ import annotations
from datetime import timedelta
import logging
from pyhik.hikvision import HikCamera
import voluptuous as vol
from homeassistant.components.binary_sensor import PLATFORM_SCHEMA as BINARY_SENSOR_PLATFORM_SCHEMA, BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.const import ATTR_LAST_TRIP_TIME, CONF_CUSTOMIZE, CONF_DELAY, CONF_HOST, CONF_NAME, CONF_PASSWORD, CONF_PORT, CONF_SSL, CONF_USERNAME, EVENT_HOMEASSISTANT_START, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import track_point_in_utc_time
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util.dt import utcnow
_LOGGER: logging.Logger = logging.getLogger(__name__)

CONF_IGNORED: str = 'ignored'
DEFAULT_PORT: int = 80
DEFAULT_IGNORED: bool = False
DEFAULT_DELAY: int = 0
ATTR_DELAY: str = 'delay'
DEVICE_CLASS_MAP: dict[str, BinarySensorDeviceClass] = {...}
CUSTOMIZE_SCHEMA: vol.Schema = vol.Schema({vol.Optional(CONF_IGNORED, default=DEFAULT_IGNORED): cv.boolean, vol.Optional(CONF_DELAY, default=DEFAULT_DELAY): cv.positive_int})
PLATFORM_SCHEMA: vol.Schema = BINARY_SENSOR_PLATFORM_SCHEMA.extend({vol.Optional(CONF_NAME): cv.string, vol.Required(CONF_HOST): cv.string, vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port, vol.Optional(CONF_SSL, default=False): cv.boolean, vol.Required(CONF_USERNAME): cv.string, vol.Required(CONF_PASSWORD): cv.string, vol.Optional(CONF_CUSTOMIZE, default={}): CUSTOMIZE_SCHEMA})

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType | None = None) -> None:
    ...

class HikvisionData:
    """Hikvision device event stream object."""

    def __init__(self, hass: HomeAssistant, url: str, port: int, name: str | None, username: str, password: str) -> None:
        ...

class HikvisionBinarySensor(BinarySensorEntity):
    """Representation of a Hikvision binary sensor."""

    _attr_should_poll: bool = False

    def __init__(self, hass: HomeAssistant, sensor: str, channel: str, cam: HikvisionData, delay: int | None) -> None:
        ...

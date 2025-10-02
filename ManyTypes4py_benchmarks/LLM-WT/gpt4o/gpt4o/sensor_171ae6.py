"""Support for displaying details about a Gitter.im chat room."""
from __future__ import annotations
import logging
from gitterpy.client import GitterClient
from gitterpy.errors import GitterRoomError, GitterTokenError
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_API_KEY, CONF_NAME, CONF_ROOM
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from typing import Any

_LOGGER = logging.getLogger(__name__)

ATTR_MENTION = 'mention'
ATTR_ROOM = 'room'
ATTR_USERNAME = 'username'
DEFAULT_NAME = 'Gitter messages'
DEFAULT_ROOM = 'home-assistant/home-assistant'

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_API_KEY): cv.string,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_ROOM, default=DEFAULT_ROOM): cv.string
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    """Set up the Gitter sensor."""
    name: str = config.get(CONF_NAME)
    api_key: str = config.get(CONF_API_KEY)
    room: str = config.get(CONF_ROOM)
    gitter = GitterClient(api_key)
    try:
        username: str = gitter.auth.get_my_id['name']
    except GitterTokenError:
        _LOGGER.error('Token is not valid')
        return
    add_entities([GitterSensor(gitter, room, name, username)], True)

class GitterSensor(SensorEntity):
    """Representation of a Gitter sensor."""
    _attr_icon = 'mdi:message-cog'

    def __init__(self, data: GitterClient, room: str, name: str, username: str) -> None:
        """Initialize the sensor."""
        self._name: str = name
        self._data: GitterClient = data
        self._room: str = room
        self._username: str = username
        self._state: int | None = None
        self._mention: int = 0
        self._unit_of_measurement: str = 'Msg'

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def native_value(self) -> int | None:
        """Return the state of the sensor."""
        return self._state

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit the value is expressed in."""
        return self._unit_of_measurement

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""
        return {
            ATTR_USERNAME: self._username,
            ATTR_ROOM: self._room,
            ATTR_MENTION: self._mention
        }

    def update(self) -> None:
        """Get the latest data and updates the state."""
        try:
            data: dict[str, Any] = self._data.user.unread_items(self._room)
        except GitterRoomError as error:
            _LOGGER.error(error)
            return
        if 'error' not in data:
            self._mention = len(data['mention'])
            self._state = len(data['chat'])
        else:
            _LOGGER.error('Not joined: %s', self._room)

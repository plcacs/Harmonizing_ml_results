"""Pencom relay control."""
from __future__ import annotations
import logging
from typing import Any
from pencompy.pencompy import Pencompy
import voluptuous as vol
from homeassistant.components.switch import PLATFORM_SCHEMA as SWITCH_PLATFORM_SCHEMA, SwitchEntity
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
_LOGGER = logging.getLogger(__name__)
CONF_BOARDS = 'boards'
CONF_BOARD = 'board'
CONF_ADDR = 'addr'
CONF_RELAYS = 'relays'
RELAY_SCHEMA = vol.Schema({vol.Required(CONF_NAME): cv.string, vol.Required(CONF_ADDR): cv.positive_int, vol.Optional(CONF_BOARD, default=0): cv.positive_int})
PLATFORM_SCHEMA = SWITCH_PLATFORM_SCHEMA.extend({vol.Required(CONF_HOST): cv.string, vol.Required(CONF_PORT): cv.port, vol.Optional(CONF_BOARDS, default=1): cv.positive_int, vol.Required(CONF_RELAYS): vol.All(cv.ensure_list, [RELAY_SCHEMA])})

def setup_platform(hass: Union[homeassistanhelpers.ConfigType, dict, None], config: homeassistanhelpers.ConfigType, add_entities: typing.Callable[list, None], discovery_info: Union[None, homeassistanhelpers.ConfigType, dict]=None) -> None:
    """Pencom relay platform (pencompy)."""
    host = config[CONF_HOST]
    port = config[CONF_PORT]
    boards = config[CONF_BOARDS]
    try:
        hub = Pencompy(host, port, boards=boards)
    except OSError as error:
        _LOGGER.error('Could not connect to pencompy: %s', error)
        raise PlatformNotReady from error
    devs = []
    for relay in config[CONF_RELAYS]:
        name = relay[CONF_NAME]
        board = relay[CONF_BOARD]
        addr = relay[CONF_ADDR]
        devs.append(PencomRelay(hub, board, addr, name))
    add_entities(devs, True)

class PencomRelay(SwitchEntity):
    """Representation of a pencom relay."""

    def __init__(self, hub: Union[str, src.main.core.repositories.mouse_repository.MouseRepository, None], board: Union[list[str], str, int], addr: Union[str, tuple[typing.Union[str,int]]], name: str) -> None:
        """Create a relay."""
        self._hub = hub
        self._board = board
        self._addr = addr
        self._name = name
        self._state = None

    @property
    def name(self):
        """Relay name."""
        return self._name

    @property
    def is_on(self):
        """Return a relay's state."""
        return self._state

    def turn_on(self, **kwargs) -> None:
        """Turn a relay on."""
        self._hub.set(self._board, self._addr, True)

    def turn_off(self, **kwargs) -> None:
        """Turn a relay off."""
        self._hub.set(self._board, self._addr, False)

    def update(self) -> None:
        """Refresh a relay's state."""
        self._state = self._hub.get(self._board, self._addr)

    @property
    def extra_state_attributes(self) -> dict[typing.Text, ]:
        """Return supported attributes."""
        return {'board': self._board, 'addr': self._addr}
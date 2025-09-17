from __future__ import annotations
import logging
from typing import Any, Optional
from pencompy.pencompy import Pencompy
import voluptuous as vol
from homeassistant.components.switch import PLATFORM_SCHEMA as SWITCH_PLATFORM_SCHEMA, SwitchEntity
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger = logging.getLogger(__name__)

CONF_BOARDS = 'boards'
CONF_BOARD = 'board'
CONF_ADDR = 'addr'
CONF_RELAYS = 'relays'

RELAY_SCHEMA: vol.Schema = vol.Schema({
    vol.Required(CONF_NAME): cv.string,
    vol.Required(CONF_ADDR): cv.positive_int,
    vol.Optional(CONF_BOARD, default=0): cv.positive_int
})

PLATFORM_SCHEMA: vol.Schema = SWITCH_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_HOST): cv.string,
    vol.Required(CONF_PORT): cv.port,
    vol.Optional(CONF_BOARDS, default=1): cv.positive_int,
    vol.Required(CONF_RELAYS): vol.All(cv.ensure_list, [RELAY_SCHEMA])
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    host: str = config[CONF_HOST]
    port: int = config[CONF_PORT]
    boards: int = config[CONF_BOARDS]
    try:
        hub: Pencompy = Pencompy(host, port, boards=boards)
    except OSError as error:
        _LOGGER.error('Could not connect to pencompy: %s', error)
        raise PlatformNotReady from error
    devs: list[PencomRelay] = []
    for relay in config[CONF_RELAYS]:
        name: str = relay[CONF_NAME]
        board: int = relay[CONF_BOARD]
        addr: int = relay[CONF_ADDR]
        devs.append(PencomRelay(hub, board, addr, name))
    add_entities(devs, True)

class PencomRelay(SwitchEntity):
    def __init__(self, hub: Pencompy, board: int, addr: int, name: str) -> None:
        self._hub: Pencompy = hub
        self._board: int = board
        self._addr: int = addr
        self._name: str = name
        self._state: Optional[bool] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_on(self) -> Optional[bool]:
        return self._state

    def turn_on(self, **kwargs: Any) -> None:
        self._hub.set(self._board, self._addr, True)

    def turn_off(self, **kwargs: Any) -> None:
        self._hub.set(self._board, self._addr, False)

    def update(self) -> None:
        self._state = self._hub.get(self._board, self._addr)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {'board': self._board, 'addr': self._addr}
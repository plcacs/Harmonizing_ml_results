from __future__ import annotations
import logging
from typing import Any, Optional
import requests
import voluptuous as vol
from homeassistant.components.switch import PLATFORM_SCHEMA as SWITCH_PLATFORM_SCHEMA, SwitchEntity
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PASSWORD, CONF_PATH, CONF_PORT, CONF_SWITCHES, CONF_USERNAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger = logging.getLogger(__name__)
DEFAULT_PORT: int = 80
DEFAULT_PATH: str = '/cgi-bin/json.cgi'

SWITCH_SCHEMA: vol.Schema = vol.Schema({
    vol.Required(CONF_HOST): cv.string,
    vol.Optional(CONF_NAME): cv.string,
    vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
    vol.Optional(CONF_PATH, default=DEFAULT_PATH): cv.string,
    vol.Optional(CONF_USERNAME): cv.string,
    vol.Optional(CONF_PASSWORD): cv.string
})
PLATFORM_SCHEMA: vol.Schema = SWITCH_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_SWITCHES): cv.schema_with_slug_keys(SWITCH_SCHEMA)
})


def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities_callback: AddEntitiesCallback, discovery_info: Optional[DiscoveryInfoType] = None) -> None:
    """Set up Kankun Wifi switches."""
    switches: dict[str, Any] = config.get('switches', {})
    devices: list[KankunSwitch] = []
    for dev_name, properties in switches.items():
        devices.append(KankunSwitch(
            hass,
            properties.get(CONF_NAME, dev_name),
            properties.get(CONF_HOST),
            properties.get(CONF_PORT, DEFAULT_PORT),
            properties.get(CONF_PATH, DEFAULT_PATH),
            properties.get(CONF_USERNAME),
            properties.get(CONF_PASSWORD)
        ))
    add_entities_callback(devices)


class KankunSwitch(SwitchEntity):
    """Representation of a Kankun Wifi switch."""

    def __init__(self, hass: HomeAssistant, name: str, host: str, port: int, path: str, user: Optional[str], passwd: Optional[str]) -> None:
        """Initialize the device."""
        self._hass: HomeAssistant = hass
        self._name: str = name
        self._state: bool = False
        self._url: str = f'http://{host}:{port}{path}'
        if user is not None:
            self._auth: Optional[tuple[str, Optional[str]]] = (user, passwd)
        else:
            self._auth = None

    def _switch(self, newstate: str) -> Optional[bool]:
        """Switch on or off."""
        _LOGGER.debug('Switching to state: %s', newstate)
        try:
            req = requests.get(f'{self._url}?set={newstate}', auth=self._auth, timeout=5)
            return req.json()['ok']
        except requests.RequestException:
            _LOGGER.error('Switching failed')
            return None

    def _query_state(self) -> Optional[bool]:
        """Query switch state."""
        _LOGGER.debug('Querying state from: %s', self._url)
        try:
            req = requests.get(f'{self._url}?get=state', auth=self._auth, timeout=5)
            return req.json()['state'] == 'on'
        except requests.RequestException:
            _LOGGER.error('State query failed')
            return None

    @property
    def name(self) -> str:
        """Return the name of the switch."""
        return self._name

    @property
    def is_on(self) -> bool:
        """Return true if device is on."""
        return self._state

    def update(self) -> None:
        """Update device state."""
        state: Optional[bool] = self._query_state()
        if state is not None:
            self._state = state

    def turn_on(self, **kwargs: Any) -> None:
        """Turn the device on."""
        result: Optional[bool] = self._switch('on')
        if result:
            self._state = True

    def turn_off(self, **kwargs: Any) -> None:
        """Turn the device off."""
        result: Optional[bool] = self._switch('off')
        if result:
            self._state = False
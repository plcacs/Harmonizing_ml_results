"""Switch logic for loading/unloading pulseaudio loopback modules."""
from __future__ import annotations
import logging
from typing import Any, Optional
from pulsectl import Pulse, PulseError
import voluptuous as vol
from homeassistant.components.switch import PLATFORM_SCHEMA as SWITCH_PLATFORM_SCHEMA, SwitchEntity
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

DOMAIN = 'pulseaudio_loopback'
_LOGGER = logging.getLogger(__name__)
CONF_SINK_NAME = 'sink_name'
CONF_SOURCE_NAME = 'source_name'
DEFAULT_NAME = 'paloopback'
DEFAULT_PORT = 4713
IGNORED_SWITCH_WARN = 'Switch is already in the desired state. Ignoring.'

PLATFORM_SCHEMA = SWITCH_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_SINK_NAME): cv.string,
    vol.Required(CONF_SOURCE_NAME): cv.string,
    vol.Optional(CONF_HOST): cv.string,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Read in all of our configuration, and initialize the loopback switch."""
    name: str = config.get(CONF_NAME)
    sink_name: str = config.get(CONF_SINK_NAME)
    source_name: str = config.get(CONF_SOURCE_NAME)
    host: Optional[str] = config.get(CONF_HOST)
    port: int = config.get(CONF_PORT)
    hass.data.setdefault(DOMAIN, {})
    server_id: str = f'{host}:{port}'
    if host:
        connect_to_server: Optional[str] = server_id
    else:
        connect_to_server = None
    if server_id in hass.data[DOMAIN]:
        server: Pulse = hass.data[DOMAIN][server_id]
    else:
        server = Pulse(server=connect_to_server, connect=False, threading_lock=True)
        hass.data[DOMAIN][server_id] = server
    add_entities([PALoopbackSwitch(name, server, sink_name, source_name)], True)

class PALoopbackSwitch(SwitchEntity):
    """Representation the presence or absence of a PA loopback module."""

    def __init__(self, name: str, pa_server: Pulse, sink_name: str, source_name: str) -> None:
        """Initialize the Pulseaudio switch."""
        self._module_idx: Optional[int] = None
        self._name: str = name
        self._sink_name: str = sink_name
        self._source_name: str = source_name
        self._pa_svr: Pulse = pa_server

    def _get_module_idx(self) -> Optional[int]:
        try:
            self._pa_svr.connect()
            for module in self._pa_svr.module_list():
                if module.name != 'module-loopback':
                    continue
                if f'sink={self._sink_name}' not in module.argument:
                    continue
                if f'source={self._source_name}' not in module.argument:
                    continue
                return module.index
        except PulseError:
            return None
        return None

    @property
    def available(self) -> bool:
        """Return true when connected to server."""
        return self._pa_svr.connected

    @property
    def name(self) -> str:
        """Return the name of the switch."""
        return self._name

    @property
    def is_on(self) -> bool:
        """Return true if device is on."""
        return self._module_idx is not None

    def turn_on(self, **kwargs: Any) -> None:
        """Turn the device on."""
        if not self.is_on:
            self._pa_svr.module_load('module-loopback', args=f'sink={self._sink_name} source={self._source_name}')
        else:
            _LOGGER.warning(IGNORED_SWITCH_WARN)

    def turn_off(self, **kwargs: Any) -> None:
        """Turn the device off."""
        if self.is_on:
            self._pa_svr.module_unload(self._module_idx)  # type: ignore
        else:
            _LOGGER.warning(IGNORED_SWITCH_WARN)

    def update(self) -> None:
        """Refresh state in case an alternate process modified this data."""
        self._module_idx = self._get_module_idx()
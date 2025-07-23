"""Support for interface with a Ziggo Mediabox XL."""
from __future__ import annotations
import logging
import socket
import voluptuous as vol
from ziggo_mediabox_xl import ZiggoMediaboxXL
from homeassistant.components.media_player import (
    PLATFORM_SCHEMA as MEDIA_PLAYER_PLATFORM_SCHEMA,
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
)
from homeassistant.const import CONF_HOST, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER = logging.getLogger(__name__)
DATA_KNOWN_DEVICES = 'ziggo_mediabox_xl_known_devices'
PLATFORM_SCHEMA = MEDIA_PLAYER_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_HOST): cv.string,
    vol.Optional(CONF_NAME): cv.string
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    """Set up the Ziggo Mediabox XL platform."""
    if DATA_KNOWN_DEVICES not in hass.data:
        hass.data[DATA_KNOWN_DEVICES] = known_devices = set()
    else:
        known_devices = hass.data[DATA_KNOWN_DEVICES]
        
    hosts: list[ZiggoMediaboxXLDevice] = []
    connection_successful: bool = False
    host: str | None = config.get(CONF_HOST)
    name: str | None
    manual_config: bool
    if host is not None:
        name = config.get(CONF_NAME)
        manual_config = True
    elif discovery_info is not None:
        host = discovery_info.get('host')
        name = discovery_info.get('name')
        manual_config = False
    else:
        _LOGGER.error('Cannot determine device')
        return
    if host is None:
        _LOGGER.error('Host is None after parsing config')
        return
    try:
        ip_addr: str = socket.gethostbyname(host)
    except socket.gaierror as error:
        _LOGGER.error("Invalid host %s: %s", host, error)
        return
    if ip_addr not in known_devices:
        try:
            mediabox: ZiggoMediaboxXL = ZiggoMediaboxXL(ip_addr, 3)
            if mediabox.test_connection():
                connection_successful = True
            if not connection_successful and manual_config:
                _LOGGER.error("Can't connect to %s", host)
            elif not connection_successful:
                _LOGGER.error("Can't connect to %s", host)
            if manual_config or connection_successful:
                device = ZiggoMediaboxXLDevice(mediabox, host, name, connection_successful)
                hosts.append(device)
                known_devices.add(ip_addr)
        except OSError as error:
            _LOGGER.error("Can't connect to %s: %s", host, error)
    else:
        _LOGGER.warning('Ignoring duplicate Ziggo Mediabox XL %s', host)
    add_entities(hosts, True)

class ZiggoMediaboxXLDevice(MediaPlayerEntity):
    """Representation of a Ziggo Mediabox XL Device."""

    _attr_supported_features: MediaPlayerEntityFeature = (
        MediaPlayerEntityFeature.TURN_ON |
        MediaPlayerEntityFeature.TURN_OFF |
        MediaPlayerEntityFeature.NEXT_TRACK |
        MediaPlayerEntityFeature.PAUSE |
        MediaPlayerEntityFeature.PREVIOUS_TRACK |
        MediaPlayerEntityFeature.SELECT_SOURCE |
        MediaPlayerEntityFeature.PLAY
    )
    _attr_state: MediaPlayerState | None = None
    _attr_available: bool = False
    _attr_name: str | None = None

    def __init__(
        self,
        mediabox: ZiggoMediaboxXL,
        host: str,
        name: str | None,
        available: bool
    ) -> None:
        """Initialize the device."""
        self._mediabox = mediabox
        self._host = host
        self._attr_name = name
        self._attr_available = available

    def update(self) -> None:
        """Retrieve the state of the device."""
        try:
            if self._mediabox.test_connection():
                if self._mediabox.turned_on():
                    if self.state != MediaPlayerState.PAUSED:
                        self._attr_state = MediaPlayerState.PLAYING
                else:
                    self._attr_state = MediaPlayerState.OFF
                self._attr_available = True
            else:
                self._attr_available = False
        except OSError:
            _LOGGER.error("Couldn't fetch state from %s", self._host)
            self._attr_available = False

    def send_keys(self, keys: list[str]) -> None:
        """Send keys to the device and handle exceptions."""
        try:
            self._mediabox.send_keys(keys)
        except OSError:
            _LOGGER.error("Couldn't send keys to %s", self._host)

    @property
    def source_list(self) -> list[str]:
        """List of available sources (channels)."""
        return [
            self._mediabox.channels()[c]
            for c in sorted(self._mediabox.channels().keys())
        ]

    def turn_on(self) -> None:
        """Turn the media player on."""
        self.send_keys(['POWER'])

    def turn_off(self) -> None:
        """Turn off media player."""
        self.send_keys(['POWER'])

    def media_play(self) -> None:
        """Send play command."""
        self.send_keys(['PLAY'])
        self._attr_state = MediaPlayerState.PLAYING

    def media_pause(self) -> None:
        """Send pause command."""
        self.send_keys(['PAUSE'])
        self._attr_state = MediaPlayerState.PAUSED

    def media_play_pause(self) -> None:
        """Simulate play pause media player."""
        self.send_keys(['PAUSE'])
        if self.state == MediaPlayerState.PAUSED:
            self._attr_state = MediaPlayerState.PLAYING
        else:
            self._attr_state = MediaPlayerState.PAUSED

    def media_next_track(self) -> None:
        """Channel up."""
        self.send_keys(['CHAN_UP'])
        self._attr_state = MediaPlayerState.PLAYING

    def media_previous_track(self) -> None:
        """Channel down."""
        self.send_keys(['CHAN_DOWN'])
        self._attr_state = MediaPlayerState.PLAYING

    def select_source(self, source: str | int) -> None:
        """Select the channel."""
        if isinstance(source, int) or str(source).isdigit():
            digits: str = str(source)
        else:
            digits = next(
                (
                    str(key) 
                    for key, value in self._mediabox.channels().items() 
                    if value == source
                ), 
                None
            )
        if digits is None:
            return
        self.send_keys([f'NUM_{digit}' for digit in digits])
        self._attr_state = MediaPlayerState.PLAYING

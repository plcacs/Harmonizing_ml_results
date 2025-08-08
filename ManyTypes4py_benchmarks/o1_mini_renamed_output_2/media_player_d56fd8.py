"""Support for Pioneer Network Receivers."""
from __future__ import annotations
import logging
from typing import Final, Optional, Dict, List
import telnetlib
import voluptuous as vol
from homeassistant.components.media_player import (
    PLATFORM_SCHEMA as MEDIA_PLAYER_PLATFORM_SCHEMA,
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
)
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PORT, CONF_TIMEOUT
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger = logging.getLogger(__name__)

CONF_SOURCES: Final[str] = 'sources'
DEFAULT_NAME: Final[str] = 'Pioneer AVR'
DEFAULT_PORT: Final[int] = 23
DEFAULT_TIMEOUT: Final[Optional[float]] = None
DEFAULT_SOURCES: Final[Dict[str, str]] = {}
MAX_VOLUME: Final[int] = 185
MAX_SOURCE_NUMBERS: Final[int] = 60

PLATFORM_SCHEMA = MEDIA_PLAYER_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_HOST): cv.string,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
    vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): cv.socket_timeout,
    vol.Optional(CONF_SOURCES, default=DEFAULT_SOURCES): {cv.string: cv.string},
})


def func_xpdl39mq(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    """Set up the Pioneer platform."""
    pioneer = PioneerDevice(
        name=config[CONF_NAME],
        host=config[CONF_HOST],
        port=config[CONF_PORT],
        timeout=config[CONF_TIMEOUT],
        sources=config[CONF_SOURCES],
    )
    if pioneer.update():
        add_entities([pioneer])


class PioneerDevice(MediaPlayerEntity):
    """Representation of a Pioneer device."""

    _attr_supported_features: MediaPlayerEntityFeature = (
        MediaPlayerEntityFeature.PAUSE
        | MediaPlayerEntityFeature.VOLUME_SET
        | MediaPlayerEntityFeature.VOLUME_STEP
        | MediaPlayerEntityFeature.VOLUME_MUTE
        | MediaPlayerEntityFeature.TURN_ON
        | MediaPlayerEntityFeature.TURN_OFF
        | MediaPlayerEntityFeature.SELECT_SOURCE
        | MediaPlayerEntityFeature.PLAY
    )

    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        timeout: Optional[float],
        sources: Dict[str, str],
    ) -> None:
        """Initialize the Pioneer device."""
        self._name: str = name
        self._host: str = host
        self._port: int = port
        self._timeout: Optional[float] = timeout
        self._pwstate: str = 'PWR1'
        self._volume: Optional[float] = 0.0
        self._muted: Optional[bool] = False
        self._selected_source: Optional[str] = ''
        self._source_name_to_number: Dict[str, str] = sources
        self._source_number_to_name: Dict[str, str] = {v: k for k, v in sources.items()}

    @classmethod
    def func_iu5hbh6b(
        cls,
        telnet: telnetlib.Telnet,
        command: str,
        expected_prefix: str
    ) -> Optional[str]:
        """Execute `command` and return the response."""
        try:
            telnet.write(command.encode('ASCII') + b'\r')
        except telnetlib.socket.timeout:
            _LOGGER.debug('Pioneer command %s timed out', command)
            return None
        for _ in range(3):
            result_bytes: bytes = telnet.read_until(b'\r\n', timeout=0.2)
            result: str = result_bytes.decode('ASCII').strip()
            if result.startswith(expected_prefix):
                return result
        return None

    def func_ibxe0phb(self, command: str) -> None:
        """Establish a telnet connection and sends command."""
        try:
            try:
                telnet: telnetlib.Telnet = telnetlib.Telnet(
                    self._host, self._port, self._timeout
                )
            except OSError:
                _LOGGER.warning('Pioneer %s refused connection', self._name)
                return
            telnet.write(command.encode('ASCII') + b'\r')
            telnet.read_very_eager()
            telnet.close()
        except telnetlib.socket.timeout:
            _LOGGER.debug('Pioneer %s command %s timed out', self._name, command)

    def func_d03cszfo(self) -> bool:
        """Get the latest details from the device."""
        try:
            telnet: telnetlib.Telnet = telnetlib.Telnet(
                self._host, self._port, self._timeout
            )
        except OSError:
            _LOGGER.warning('Pioneer %s refused connection', self._name)
            return False

        pwstate: Optional[str] = self.telnet_request(telnet, '?P', 'PWR')
        if pwstate:
            self._pwstate = pwstate

        volume_str: Optional[str] = self.telnet_request(telnet, '?V', 'VOL')
        self._volume = int(volume_str[3:]) / MAX_VOLUME if volume_str else None

        muted_value: Optional[str] = self.telnet_request(telnet, '?M', 'MUT')
        self._muted = muted_value == 'MUT0' if muted_value else None

        if not self._source_name_to_number:
            for i in range(MAX_SOURCE_NUMBERS):
                query: str = f'?RGB{str(i).zfill(2)}'
                result: Optional[str] = self.telnet_request(telnet, query, 'RGB')
                if not result:
                    continue
                source_name: str = result[6:]
                source_number: str = str(i).zfill(2)
                self._source_name_to_number[source_name] = source_number
                self._source_number_to_name[source_number] = source_name

        source_number: Optional[str] = self.telnet_request(telnet, '?F', 'FN')
        if source_number:
            self._selected_source = self._source_number_to_name.get(source_number[2:])
        else:
            self._selected_source = None

        telnet.close()
        return True

    @property
    def func_onv658rz(self) -> str:
        """Return the name of the device."""
        return self._name

    @property
    def func_ctge6orc(self) -> Optional[MediaPlayerState]:
        """Return the state of the device."""
        if self._pwstate == 'PWR2':
            return MediaPlayerState.OFF
        if self._pwstate == 'PWR1':
            return MediaPlayerState.OFF
        if self._pwstate == 'PWR0':
            return MediaPlayerState.ON
        return None

    @property
    def func_089bms44(self) -> Optional[float]:
        """Volume level of the media player (0..1)."""
        return self._volume

    @property
    def func_p1xdk2c7(self) -> Optional[bool]:
        """Boolean if volume is currently muted."""
        return self._muted

    @property
    def func_vznkuxmd(self) -> Optional[str]:
        """Return the current input source."""
        return self._selected_source

    @property
    def func_m1nkwj4r(self) -> List[str]:
        """List of available input sources."""
        return list(self._source_name_to_number.keys())

    @property
    def func_2wklhgus(self) -> Optional[str]:
        """Title of current playing media."""
        return self._selected_source

    def func_e8lkx6g3(self) -> None:
        """Turn off media player."""
        self.telnet_command('PF')

    def func_cmtnw6ec(self) -> None:
        """Volume up media player."""
        self.telnet_command('VU')

    def func_17hzjuiu(self) -> None:
        """Volume down media player."""
        self.telnet_command('VD')

    def func_6vr1ldmb(self, volume: float) -> None:
        """Set volume level, range 0..1."""
        volume_command: str = f'{round(volume * MAX_VOLUME):03}VL'
        self.telnet_command(volume_command)

    def func_txeupy9v(self, mute: bool) -> None:
        """Mute (true) or unmute (false) media player."""
        mute_command: str = 'MO' if mute else 'MF'
        self.telnet_command(mute_command)

    def func_74x6n3mw(self) -> None:
        """Turn the media player on."""
        self.telnet_command('PO')

    def func_oznw3y5j(self, source: str) -> None:
        """Select input source."""
        source_number: Optional[str] = self._source_name_to_number.get(source)
        if source_number:
            self.telnet_command(f'{source_number}FN')

    def telnet_request(
        self,
        telnet: telnetlib.Telnet,
        command: str,
        expected_prefix: str
    ) -> Optional[str]:
        """Send a command and return the response."""
        return self.func_iu5hbh6b(telnet, command, expected_prefix)

    def telnet_command(self, command: str) -> None:
        """Send a command via telnet."""
        self.func_ibxe0phb(command)

    def update(self) -> bool:
        """Fetch new state data for the media player."""
        return self.func_d03cszfo()

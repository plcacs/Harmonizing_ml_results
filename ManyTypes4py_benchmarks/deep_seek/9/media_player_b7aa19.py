"""Support for controlling projector via the PJLink protocol."""
from __future__ import annotations
from pypjlink import MUTE_AUDIO, Projector
from pypjlink.projector import ProjectorError
import voluptuous as vol
from homeassistant.components.media_player import PLATFORM_SCHEMA as MEDIA_PLAYER_PLATFORM_SCHEMA, MediaPlayerEntity, MediaPlayerEntityFeature, MediaPlayerState
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PASSWORD, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from typing import Dict, Optional, Tuple
from .const import CONF_ENCODING, DEFAULT_ENCODING, DEFAULT_PORT, DOMAIN

ERR_PROJECTOR_UNAVAILABLE = 'projector unavailable'
PLATFORM_SCHEMA = MEDIA_PLAYER_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_HOST): cv.string,
    vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
    vol.Optional(CONF_NAME): cv.string,
    vol.Optional(CONF_ENCODING, default=DEFAULT_ENCODING): cv.string,
    vol.Optional(CONF_PASSWORD): cv.string
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the PJLink platform."""
    host: str = config.get(CONF_HOST)
    port: int = config.get(CONF_PORT)
    name: Optional[str] = config.get(CONF_NAME)
    encoding: str = config.get(CONF_ENCODING)
    password: Optional[str] = config.get(CONF_PASSWORD)
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}
    hass_data: Dict[str, PjLinkDevice] = hass.data[DOMAIN]
    device_label: str = f'{host}:{port}'
    if device_label in hass_data:
        return
    device: PjLinkDevice = PjLinkDevice(host, port, name, encoding, password)
    hass_data[device_label] = device
    add_entities([device], True)

def format_input_source(input_source_name: str, input_source_number: str) -> str:
    """Format input source for display in UI."""
    return f'{input_source_name} {input_source_number}'

class PjLinkDevice(MediaPlayerEntity):
    """Representation of a PJLink device."""
    _attr_supported_features: int = (
        MediaPlayerEntityFeature.VOLUME_MUTE |
        MediaPlayerEntityFeature.TURN_ON |
        MediaPlayerEntityFeature.TURN_OFF |
        MediaPlayerEntityFeature.SELECT_SOURCE
    )

    def __init__(
        self,
        host: str,
        port: int,
        name: Optional[str],
        encoding: str,
        password: Optional[str]
    ) -> None:
        """Initialize the PJLink device."""
        self._host: str = host
        self._port: int = port
        self._password: Optional[str] = password
        self._encoding: str = encoding
        self._source_name_mapping: Dict[str, Tuple[str, str]] = {}
        self._attr_name: Optional[str] = name
        self._attr_is_volume_muted: bool = False
        self._attr_state: MediaPlayerState = MediaPlayerState.OFF
        self._attr_source: Optional[str] = None
        self._attr_source_list: list[str] = []
        self._attr_available: bool = False

    def _force_off(self) -> None:
        self._attr_state = MediaPlayerState.OFF
        self._attr_is_volume_muted = False
        self._attr_source = None

    def _setup_projector(self) -> bool:
        try:
            with self.projector() as projector:
                if not self._attr_name:
                    self._attr_name = projector.get_name()
                inputs = projector.get_inputs()
        except ProjectorError as err:
            if str(err) == ERR_PROJECTOR_UNAVAILABLE:
                return False
            raise
        self._source_name_mapping = {format_input_source(*x): x for x in inputs}
        self._attr_source_list = sorted(self._source_name_mapping)
        return True

    def projector(self) -> Projector:
        """Create PJLink Projector instance."""
        try:
            projector: Projector = Projector.from_address(self._host, self._port)
            projector.authenticate(self._password)
        except (TimeoutError, OSError) as err:
            self._attr_available = False
            raise ProjectorError(ERR_PROJECTOR_UNAVAILABLE) from err
        return projector

    def update(self) -> None:
        """Get the latest state from the device."""
        if not self._attr_available:
            self._attr_available = self._setup_projector()
        if not self._attr_available:
            self._force_off()
            return
        try:
            with self.projector() as projector:
                pwstate: str = projector.get_power()
                if pwstate in ('on', 'warm-up'):
                    self._attr_state = MediaPlayerState.ON
                    self._attr_is_volume_muted = projector.get_mute()[1]
                    self._attr_source = format_input_source(*projector.get_input())
                else:
                    self._force_off()
        except KeyError as err:
            if str(err) == "'OK'":
                self._force_off()
            else:
                raise
        except ProjectorError as err:
            if str(err) == 'unavailable time':
                self._force_off()
            elif str(err) == ERR_PROJECTOR_UNAVAILABLE:
                self._attr_available = False
            else:
                raise

    def turn_off(self) -> None:
        """Turn projector off."""
        with self.projector() as projector:
            projector.set_power('off')

    def turn_on(self) -> None:
        """Turn projector on."""
        with self.projector() as projector:
            projector.set_power('on')

    def mute_volume(self, mute: bool) -> None:
        """Mute (true) of unmute (false) media player."""
        with self.projector() as projector:
            projector.set_mute(MUTE_AUDIO, mute)

    def select_source(self, source: str) -> None:
        """Set the input source."""
        source_tuple: Tuple[str, str] = self._source_name_mapping[source]
        with self.projector() as projector:
            projector.set_input(*source_tuple)

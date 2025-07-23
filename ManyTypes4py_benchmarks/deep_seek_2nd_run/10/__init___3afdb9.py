"""Support to control a Zehnder ComfoAir Q350/450/600 ventilation unit."""
import logging
from typing import Any, Dict, List, Optional, cast
from pycomfoconnect import Bridge, ComfoConnect
import voluptuous as vol
from homeassistant.const import (
    CONF_HOST,
    CONF_NAME,
    CONF_PIN,
    CONF_TOKEN,
    EVENT_HOMEASSISTANT_STOP,
    Platform,
)
from homeassistant.core import Event, HomeAssistant
from homeassistant.helpers import config_validation as cv, discovery
from homeassistant.helpers.dispatcher import dispatcher_send
from homeassistant.helpers.typing import ConfigType

_LOGGER = logging.getLogger(__name__)
DOMAIN = "comfoconnect"
SIGNAL_COMFOCONNECT_UPDATE_RECEIVED = "comfoconnect_update_received_{}"
CONF_USER_AGENT = "user_agent"
DEFAULT_NAME = "ComfoAirQ"
DEFAULT_PIN = 0
DEFAULT_TOKEN = "00000000000000000000000000000001"
DEFAULT_USER_AGENT = "Home Assistant"

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.Schema(
            {
                vol.Required(CONF_HOST): cv.string,
                vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
                vol.Optional(CONF_TOKEN, default=DEFAULT_TOKEN): vol.Length(
                    min=32, max=32, msg="invalid token"
                ),
                vol.Optional(CONF_USER_AGENT, default=DEFAULT_USER_AGENT): cv.string,
                vol.Optional(CONF_PIN, default=DEFAULT_PIN): cv.positive_int,
            }
        )
    },
    extra=vol.ALLOW_EXTRA,
)


def setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the ComfoConnect bridge."""
    conf: Dict[str, Any] = config[DOMAIN]
    host: str = conf[CONF_HOST]
    name: str = conf[CONF_NAME]
    token: str = conf[CONF_TOKEN]
    user_agent: str = conf[CONF_USER_AGENT]
    pin: int = conf[CONF_PIN]
    bridges: List[Bridge] = Bridge.discover(host)
    if not bridges:
        _LOGGER.error("Could not connect to ComfoConnect bridge on %s", host)
        return False
    bridge: Bridge = bridges[0]
    _LOGGER.debug("Bridge found: %s (%s)", bridge.uuid.hex(), bridge.host)
    ccb = ComfoConnectBridge(hass, bridge, name, token, user_agent, pin)
    hass.data[DOMAIN] = ccb
    ccb.connect()

    def _shutdown(_event: Event) -> None:
        ccb.disconnect()

    hass.bus.listen_once(EVENT_HOMEASSISTANT_STOP, _shutdown)
    discovery.load_platform(hass, Platform.FAN, DOMAIN, {}, config)
    return True


class ComfoConnectBridge:
    """Representation of a ComfoConnect bridge."""

    def __init__(
        self,
        hass: HomeAssistant,
        bridge: Bridge,
        name: str,
        token: str,
        friendly_name: str,
        pin: int,
    ) -> None:
        """Initialize the ComfoConnect bridge."""
        self.name: str = name
        self.hass: HomeAssistant = hass
        self.unique_id: str = bridge.uuid.hex()
        self.comfoconnect: ComfoConnect = ComfoConnect(
            bridge=bridge,
            local_uuid=bytes.fromhex(token),
            local_devicename=friendly_name,
            pin=pin,
        )
        self.comfoconnect.callback_sensor = self.sensor_callback

    def connect(self) -> None:
        """Connect with the bridge."""
        _LOGGER.debug("Connecting with bridge")
        self.comfoconnect.connect(True)

    def disconnect(self) -> None:
        """Disconnect from the bridge."""
        _LOGGER.debug("Disconnecting from bridge")
        self.comfoconnect.disconnect()

    def sensor_callback(self, var: Any, value: Any) -> None:
        """Notify listeners that we have received an update."""
        _LOGGER.debug("Received update for %s: %s", var, value)
        dispatcher_send(
            self.hass, SIGNAL_COMFOCONNECT_UPDATE_RECEIVED.format(var), value
        )

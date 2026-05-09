from homeassistant.const import Platform
from homeassistant.core import Event, HomeAssistant
from homeassistant.helpers import config_validation as cv, discovery
from homeassistant.helpers.typing import ConfigType
from pycomfoconnect import Bridge, ComfoConnect
import logging

_LOGGER = logging.getLogger(__name__)
DOMAIN: str = 'comfoconnect'
SIGNAL_COMFOCONNECT_UPDATE_RECEIVED: str
CONF_USER_AGENT: str
DEFAULT_NAME: str
DEFAULT_PIN: int
DEFAULT_TOKEN: str
DEFAULT_USER_AGENT: str
CONFIG_SCHEMA: vol.Schema

def setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the ComfoConnect bridge."""
    conf: ConfigType = config[DOMAIN]
    host: str = conf[CONF_HOST]
    name: str = conf[CONF_NAME]
    token: str = conf[CONF_TOKEN]
    user_agent: str = conf[CONF_USER_AGENT]
    pin: int = conf[CONF_PIN]
    bridges = Bridge.discover(host)
    if not bridges:
        _LOGGER.error('Could not connect to ComfoConnect bridge on %s', host)
        return False
    bridge: Bridge = bridges[0]
    _LOGGER.debug('Bridge found: %s (%s)', bridge.uuid.hex(), bridge.host)
    ccb: ComfoConnectBridge = ComfoConnectBridge(hass, bridge, name, token, user_agent, pin)
    hass.data[DOMAIN] = ccb
    ccb.connect()

    def _shutdown(_event: Event):
        ccb.disconnect()
    hass.bus.listen_once(EVENT_HOMEASSISTANT_STOP, _shutdown)
    discovery.load_platform(hass, Platform.FAN, DOMAIN, {}, config)
    return True

class ComfoConnectBridge:
    """Representation of a ComfoConnect bridge."""

    def __init__(self, hass: HomeAssistant, bridge: Bridge, name: str, token: str, friendly_name: str, pin: int):
        """Initialize the ComfoConnect bridge."""
        self.name: str = name
        self.hass: HomeAssistant = hass
        self.unique_id: str = bridge.uuid.hex()
        self.comfoconnect: ComfoConnect = ComfoConnect(bridge=bridge, local_uuid=bytes.fromhex(token), local_devicename=friendly_name, pin=pin)
        self.comfoconnect.callback_sensor = self.sensor_callback

    def connect(self):
        """Connect with the bridge."""
        _LOGGER.debug('Connecting with bridge')
        self.comfoconnect.connect(True)

    def disconnect(self):
        """Disconnect from the bridge."""
        _LOGGER.debug('Disconnecting from bridge')
        self.comfoconnect.disconnect()

    def sensor_callback(self, var: str, value: str):
        """Notify listeners that we have received an update."""
        _LOGGER.debug('Received update for %s: %s', var, value)
        dispatcher_send(self.hass, SIGNAL_COMFOCONNECT_UPDATE_RECEIVED.format(var), value)

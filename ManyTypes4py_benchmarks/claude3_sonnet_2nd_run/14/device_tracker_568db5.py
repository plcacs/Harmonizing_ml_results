"""Support for APRS device tracking."""
from __future__ import annotations
import logging
import threading
from typing import Any, List, Dict, Optional, Tuple, Union, cast
import aprslib
from aprslib import ConnectionError as AprsConnectionError, LoginError
import geopy.distance
import voluptuous as vol
from homeassistant.components.device_tracker import PLATFORM_SCHEMA as DEVICE_TRACKER_PLATFORM_SCHEMA, SeeCallback
from homeassistant.const import ATTR_GPS_ACCURACY, ATTR_LATITUDE, ATTR_LONGITUDE, CONF_HOST, CONF_PASSWORD, CONF_TIMEOUT, CONF_USERNAME, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import Event, HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import slugify

DOMAIN: str = 'aprs'
_LOGGER = logging.getLogger(__name__)
ATTR_ALTITUDE: str = 'altitude'
ATTR_COURSE: str = 'course'
ATTR_COMMENT: str = 'comment'
ATTR_FROM: str = 'from'
ATTR_FORMAT: str = 'format'
ATTR_OBJECT_NAME: str = 'object_name'
ATTR_POS_AMBIGUITY: str = 'posambiguity'
ATTR_SPEED: str = 'speed'
CONF_CALLSIGNS: str = 'callsigns'
DEFAULT_HOST: str = 'rotate.aprs2.net'
DEFAULT_PASSWORD: str = '-1'
DEFAULT_TIMEOUT: float = 30.0
FILTER_PORT: int = 14580
MSG_FORMATS: List[str] = ['compressed', 'uncompressed', 'mic-e', 'object']
PLATFORM_SCHEMA = DEVICE_TRACKER_PLATFORM_SCHEMA.extend({vol.Required(CONF_CALLSIGNS): cv.ensure_list, vol.Required(CONF_USERNAME): cv.string, vol.Optional(CONF_PASSWORD, default=DEFAULT_PASSWORD): cv.string, vol.Optional(CONF_HOST, default=DEFAULT_HOST): cv.string, vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): vol.Coerce(float)})

def make_filter(callsigns: List[str]) -> str:
    """Make a server-side filter from a list of callsigns."""
    return ' '.join((f'b/{sign.upper()}' for sign in callsigns))

def gps_accuracy(gps: Tuple[float, float], posambiguity: int) -> int:
    """Calculate the GPS accuracy based on APRS posambiguity."""
    pos_a_map: Dict[int, float] = {0: 0, 1: 1 / 600, 2: 1 / 60, 3: 1 / 6, 4: 1}
    if posambiguity in pos_a_map:
        degrees: float = pos_a_map[posambiguity]
        gps2: Tuple[float, float] = (gps[0], gps[1] + degrees)
        dist_m: float = geopy.distance.distance(gps, gps2).m
        accuracy: int = round(dist_m)
    else:
        message: str = f"APRS position ambiguity must be 0-4, not '{posambiguity}'."
        raise ValueError(message)
    return accuracy

def setup_scanner(hass: HomeAssistant, config: ConfigType, see: SeeCallback, discovery_info: Optional[DiscoveryInfoType] = None) -> bool:
    """Set up the APRS tracker."""
    callsigns: List[str] = config[CONF_CALLSIGNS]
    server_filter: str = make_filter(callsigns)
    callsign: str = config[CONF_USERNAME]
    password: str = config[CONF_PASSWORD]
    host: str = config[CONF_HOST]
    timeout: float = config[CONF_TIMEOUT]
    aprs_listener = AprsListenerThread(callsign, password, host, server_filter, see)

    def aprs_disconnect(event: Event) -> None:
        """Stop the APRS connection."""
        aprs_listener.stop()
    aprs_listener.start()
    hass.bus.listen_once(EVENT_HOMEASSISTANT_STOP, aprs_disconnect)
    if not aprs_listener.start_event.wait(timeout):
        _LOGGER.error('Timeout waiting for APRS to connect')
        return False
    if not aprs_listener.start_success:
        _LOGGER.error(aprs_listener.start_message)
        return False
    _LOGGER.debug(aprs_listener.start_message)
    return True

class AprsListenerThread(threading.Thread):
    """APRS message listener."""

    def __init__(self, callsign: str, password: str, host: str, server_filter: str, see: SeeCallback) -> None:
        """Initialize the class."""
        super().__init__()
        self.callsign: str = callsign
        self.host: str = host
        self.start_event: threading.Event = threading.Event()
        self.see: SeeCallback = see
        self.server_filter: str = server_filter
        self.start_message: str = ''
        self.start_success: bool = False
        self.ais: aprslib.IS = aprslib.IS(self.callsign, passwd=password, host=self.host, port=FILTER_PORT)

    def start_complete(self, success: bool, message: str) -> None:
        """Complete startup process."""
        self.start_message = message
        self.start_success = success
        self.start_event.set()

    def run(self) -> None:
        """Connect to APRS and listen for data."""
        self.ais.set_filter(self.server_filter)
        try:
            _LOGGER.debug('Opening connection to %s with callsign %s', self.host, self.callsign)
            self.ais.connect()
            self.start_complete(True, f'Connected to {self.host} with callsign {self.callsign}.')
            self.ais.consumer(callback=self.rx_msg, immortal=True)
        except (AprsConnectionError, LoginError) as err:
            self.start_complete(False, str(err))
        except OSError:
            _LOGGER.debug('Closing connection to %s with callsign %s', self.host, self.callsign)

    def stop(self) -> None:
        """Close the connection to the APRS network."""
        self.ais.close()

    def rx_msg(self, msg: Dict[str, Any]) -> None:
        """Receive message and process if position."""
        _LOGGER.debug('APRS message received: %s', str(msg))
        if msg[ATTR_FORMAT] in MSG_FORMATS:
            if msg[ATTR_FORMAT] == 'object':
                dev_id: str = slugify(msg[ATTR_OBJECT_NAME])
            else:
                dev_id = slugify(msg[ATTR_FROM])
            lat: float = msg[ATTR_LATITUDE]
            lon: float = msg[ATTR_LONGITUDE]
            attrs: Dict[str, Any] = {}
            if ATTR_POS_AMBIGUITY in msg:
                pos_amb: int = msg[ATTR_POS_AMBIGUITY]
                try:
                    attrs[ATTR_GPS_ACCURACY] = gps_accuracy((lat, lon), pos_amb)
                except ValueError:
                    _LOGGER.warning('APRS message contained invalid posambiguity: %s', str(pos_amb))
            for attr in (ATTR_ALTITUDE, ATTR_COMMENT, ATTR_COURSE, ATTR_SPEED):
                if attr in msg:
                    attrs[attr] = msg[attr]
            self.see(dev_id=dev_id, gps=(lat, lon), attributes=attrs)

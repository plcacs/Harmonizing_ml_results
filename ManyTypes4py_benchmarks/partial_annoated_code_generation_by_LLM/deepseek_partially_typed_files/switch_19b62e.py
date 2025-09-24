"""Use serial protocol of Acer projector to obtain state of the projector."""
from __future__ import annotations
import logging
import re
from typing import Any, cast
import serial
import voluptuous as vol
from homeassistant.components.switch import PLATFORM_SCHEMA as SWITCH_PLATFORM_SCHEMA, SwitchEntity
from homeassistant.const import CONF_FILENAME, CONF_NAME, CONF_TIMEOUT, STATE_OFF, STATE_ON, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from .const import CMD_DICT, CONF_WRITE_TIMEOUT, DEFAULT_NAME, DEFAULT_TIMEOUT, DEFAULT_WRITE_TIMEOUT, ECO_MODE, ICON, INPUT_SOURCE, LAMP, LAMP_HOURS
_LOGGER = logging.getLogger(__name__)
PLATFORM_SCHEMA = SWITCH_PLATFORM_SCHEMA.extend({vol.Required(CONF_FILENAME): cv.isdevice, vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string, vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): cv.positive_int, vol.Optional(CONF_WRITE_TIMEOUT, default=DEFAULT_WRITE_TIMEOUT): cv.positive_int})

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType | None = None) -> None:
    """Connect with serial port and return Acer Projector."""
    serial_port: str = config[CONF_FILENAME]
    name: str = config[CONF_NAME]
    timeout: int = config[CONF_TIMEOUT]
    write_timeout: int = config[CONF_WRITE_TIMEOUT]
    add_entities([AcerSwitch(serial_port, name, timeout, write_timeout)], True)

class AcerSwitch(SwitchEntity):
    """Represents an Acer Projector as a switch."""
    _attr_icon = ICON

    def __init__(self, serial_port: str, name: str, timeout: int, write_timeout: int) -> None:
        """Init of the Acer projector."""
        self.serial: serial.Serial = serial.Serial(port=serial_port, timeout=timeout, write_timeout=write_timeout)
        self._serial_port: str = serial_port
        self._attr_name: str = name
        self._attributes: dict[str, str] = {LAMP_HOURS: STATE_UNKNOWN, INPUT_SOURCE: STATE_UNKNOWN, ECO_MODE: STATE_UNKNOWN}
        self._attr_is_on: bool | None = None
        self._attr_available: bool = False
        self._attr_extra_state_attributes: dict[str, str] = {}

    def _write_read(self, msg: str) -> str:
        """Write to the projector and read the return."""
        ret: str = ''
        try:
            if not self.serial.is_open:
                self.serial.open()
            self.serial.write(msg.encode('utf-8'))
            ret_bytes: bytes = self.serial.read_until(size=20)
            ret = ret_bytes.decode('utf-8')
        except serial.SerialException:
            _LOGGER.error('Problem communicating with %s', self._serial_port)
        self.serial.close()
        return ret

    def _write_read_format(self, msg: str) -> str:
        """Write msg, obtain answer and format output."""
        awns: str = self._write_read(msg)
        if (match := re.search('\\r(.+)\\r', awns)):
            return match.group(1)
        return STATE_UNKNOWN

    def update(self) -> None:
        """Get the latest state from the projector."""
        awns: str = self._write_read_format(CMD_DICT[LAMP])
        if awns == 'Lamp 1':
            self._attr_is_on = True
            self._attr_available = True
        elif awns == 'Lamp 0':
            self._attr_is_on = False
            self._attr_available = True
        else:
            self._attr_available = False
        for key in self._attributes:
            if (msg := CMD_DICT.get(key)):
                awns = self._write_read_format(msg)
                self._attributes[key] = awns
        self._attr_extra_state_attributes = self._attributes

    def turn_on(self, **kwargs: Any) -> None:
        """Turn the projector on."""
        msg: str = CMD_DICT[STATE_ON]
        self._write_read(msg)
        self._attr_is_on = True

    def turn_off(self, **kwargs: Any) -> None:
        """Turn the projector off."""
        msg: str = CMD_DICT[STATE_OFF]
        self._write_read(msg)
        self._attr_is_on = False

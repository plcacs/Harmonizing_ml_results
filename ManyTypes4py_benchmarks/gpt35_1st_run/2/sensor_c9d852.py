from __future__ import annotations
import asyncio
import json
import logging
from serial import SerialException
import serial_asyncio_fast as serial_asyncio
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_NAME, CONF_VALUE_TEMPLATE, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
_LOGGER: logging.Logger = logging.getLogger(__name__)
CONF_SERIAL_PORT: str = 'serial_port'
CONF_BAUDRATE: str = 'baudrate'
CONF_BYTESIZE: str = 'bytesize'
CONF_PARITY: str = 'parity'
CONF_STOPBITS: str = 'stopbits'
CONF_XONXOFF: str = 'xonxoff'
CONF_RTSCTS: str = 'rtscts'
CONF_DSRDTR: str = 'dsrdtr'
DEFAULT_NAME: str = 'Serial Sensor'
DEFAULT_BAUDRATE: int = 9600
DEFAULT_BYTESIZE: int = serial_asyncio.serial.EIGHTBITS
DEFAULT_PARITY: int = serial_asyncio.serial.PARITY_NONE
DEFAULT_STOPBITS: int = serial_asyncio.serial.STOPBITS_ONE
DEFAULT_XONXOFF: bool = False
DEFAULT_RTSCTS: bool = False
DEFAULT_DSRDTR: bool = False
PLATFORM_SCHEMA: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend({vol.Required(CONF_SERIAL_PORT): cv.string, vol.Optional(CONF_BAUDRATE, default=DEFAULT_BAUDRATE): cv.positive_int, vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string, vol.Optional(CONF_VALUE_TEMPLATE): cv.template, vol.Optional(CONF_BYTESIZE, default=DEFAULT_BYTESIZE): vol.In([serial_asyncio.serial.FIVEBITS, serial_asyncio.serial.SIXBITS, serial_asyncio.serial.SEVENBITS, serial_asyncio.serial.EIGHTBITS]), vol.Optional(CONF_PARITY, default=DEFAULT_PARITY): vol.In([serial_asyncio.serial.PARITY_NONE, serial_asyncio.serial.PARITY_EVEN, serial_asyncio.serial.PARITY_ODD, serial_asyncio.serial.PARITY_MARK, serial_asyncio.serial.PARITY_SPACE]), vol.Optional(CONF_STOPBITS, default=DEFAULT_STOPBITS): vol.In([serial_asyncio.serial.STOPBITS_ONE, serial_asyncio.serial.STOPBITS_ONE_POINT_FIVE, serial_asyncio.serial.STOPBITS_TWO]), vol.Optional(CONF_XONXOFF, default=DEFAULT_XONXOFF): cv.boolean, vol.Optional(CONF_RTSCTS, default=DEFAULT_RTSCTS): cv.boolean, vol.Optional(CONF_DSRDTR, default=DEFAULT_DSRDTR): cv.boolean})

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    """Set up the Serial sensor platform."""
    name: str = config.get(CONF_NAME)
    port: str = config.get(CONF_SERIAL_PORT)
    baudrate: int = config.get(CONF_BAUDRATE)
    bytesize: int = config.get(CONF_BYTESIZE)
    parity: int = config.get(CONF_PARITY)
    stopbits: int = config.get(CONF_STOPBITS)
    xonxoff: bool = config.get(CONF_XONXOFF)
    rtscts: bool = config.get(CONF_RTSCTS)
    dsrdtr: bool = config.get(CONF_DSRDTR)
    value_template = config.get(CONF_VALUE_TEMPLATE)
    sensor: SerialSensor = SerialSensor(name, port, baudrate, bytesize, parity, stopbits, xonxoff, rtscts, dsrdtr, value_template)
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, sensor.stop_serial_read)
    async_add_entities([sensor], True)

class SerialSensor(SensorEntity):
    """Representation of a Serial sensor."""
    _attr_should_poll: bool = False

    def __init__(self, name: str, port: str, baudrate: int, bytesize: int, parity: int, stopbits: int, xonxoff: bool, rtscts: bool, dsrdtr: bool, value_template) -> None:
        """Initialize the Serial sensor."""
        self._name: str = name
        self._state: str = None
        self._port: str = port
        self._baudrate: int = baudrate
        self._bytesize: int = bytesize
        self._parity: int = parity
        self._stopbits: int = stopbits
        self._xonxoff: bool = xonxoff
        self._rtscts: bool = rtscts
        self._dsrdtr: bool = dsrdtr
        self._serial_loop_task = None
        self._template = value_template
        self._attributes = None

    async def async_added_to_hass(self) -> None:
        """Handle when an entity is about to be added to Home Assistant."""
        self._serial_loop_task = self.hass.loop.create_task(self.serial_read(self._port, self._baudrate, self._bytesize, self._parity, self._stopbits, self._xonxoff, self._rtscts, self._dsrdtr))

    async def serial_read(self, device: str, baudrate: int, bytesize: int, parity: int, stopbits: int, xonxoff: bool, rtscts: bool, dsrdtr: bool, **kwargs) -> None:
        """Read the data from the port."""
        logged_error: bool = False
        while True:
            try:
                reader, _ = await serial_asyncio.open_serial_connection(url=device, baudrate=baudrate, bytesize=bytesize, parity=parity, stopbits=stopbits, xonxoff=xonxoff, rtscts=rtscts, dsrdtr=dsrdtr, **kwargs)
            except SerialException:
                if not logged_error:
                    _LOGGER.exception('Unable to connect to the serial device %s. Will retry', device)
                    logged_error = True
                await self._handle_error()
            else:
                _LOGGER.debug('Serial device %s connected', device)
                while True:
                    try:
                        line = await reader.readline()
                    except SerialException:
                        _LOGGER.exception('Error while reading serial device %s', device)
                        await self._handle_error()
                        break
                    else:
                        line = line.decode('utf-8').strip()
                        try:
                            data = json.loads(line)
                        except ValueError:
                            pass
                        else:
                            if isinstance(data, dict):
                                self._attributes = data
                        if self._template is not None:
                            line = self._template.async_render_with_possible_json_value(line)
                        _LOGGER.debug('Received: %s', line)
                        self._state = line
                        self.async_write_ha_state()

    async def _handle_error(self) -> None:
        """Handle error for serial connection."""
        self._state = None
        self._attributes = None
        self.async_write_ha_state()
        await asyncio.sleep(5)

    @callback
    def stop_serial_read(self, event) -> None:
        """Close resources."""
        if self._serial_loop_task:
            self._serial_loop_task.cancel()

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def extra_state_attributes(self) -> dict:
        """Return the attributes of the entity (if any JSON present)."""
        return self._attributes

    @property
    def native_value(self) -> str:
        """Return the state of the sensor."""
        return self._state

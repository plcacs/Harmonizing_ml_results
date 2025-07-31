from __future__ import annotations
from datetime import timedelta
import importlib
import logging
from typing import Any, Dict, List, Optional
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_HOST, CONF_PORT, PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger = logging.getLogger(__name__)

ATTR_MARKER_TYPE: str = 'marker_type'
ATTR_MARKER_LOW_LEVEL: str = 'marker_low_level'
ATTR_MARKER_HIGH_LEVEL: str = 'marker_high_level'
ATTR_PRINTER_NAME: str = 'printer_name'
ATTR_DEVICE_URI: str = 'device_uri'
ATTR_PRINTER_INFO: str = 'printer_info'
ATTR_PRINTER_IS_SHARED: str = 'printer_is_shared'
ATTR_PRINTER_LOCATION: str = 'printer_location'
ATTR_PRINTER_MODEL: str = 'printer_model'
ATTR_PRINTER_STATE_MESSAGE: str = 'printer_state_message'
ATTR_PRINTER_STATE_REASON: str = 'printer_state_reason'
ATTR_PRINTER_TYPE: str = 'printer_type'
ATTR_PRINTER_URI_SUPPORTED: str = 'printer_uri_supported'

CONF_PRINTERS: str = 'printers'
CONF_IS_CUPS_SERVER: str = 'is_cups_server'
DEFAULT_HOST: str = '127.0.0.1'
DEFAULT_PORT: int = 631
DEFAULT_IS_CUPS_SERVER: bool = True
ICON_PRINTER: str = 'mdi:printer'
ICON_MARKER: str = 'mdi:water'
SCAN_INTERVAL: timedelta = timedelta(minutes=1)
PRINTER_STATES: Dict[int, str] = {3: 'idle', 4: 'printing', 5: 'stopped'}

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_PRINTERS): vol.All(cv.ensure_list, [cv.string]),
    vol.Optional(CONF_IS_CUPS_SERVER, default=DEFAULT_IS_CUPS_SERVER): cv.boolean,
    vol.Optional(CONF_HOST, default=DEFAULT_HOST): cv.string,
    vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port
})


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    host: str = config[CONF_HOST]
    port: int = config[CONF_PORT]
    printers: List[str] = config[CONF_PRINTERS]
    is_cups: bool = config[CONF_IS_CUPS_SERVER]
    if is_cups:
        data: CupsData = CupsData(host, port, None)
        data.update()
        if data.available is False:
            _LOGGER.error('Unable to connect to CUPS server: %s:%s', host, port)
            raise PlatformNotReady
        assert data.printers is not None
        dev: List[SensorEntity] = []
        for printer in printers:
            if printer not in data.printers:
                _LOGGER.error('Printer is not present: %s', printer)
                continue
            dev.append(CupsSensor(data, printer))
            if 'marker-names' in data.attributes[printer]:
                dev.extend(
                    (MarkerSensor(data, printer, marker, True) for marker in data.attributes[printer]['marker-names'])
                )
        add_entities(dev, True)
        return
    data = CupsData(host, port, printers)
    data.update()
    if data.available is False:
        _LOGGER.error('Unable to connect to IPP printer: %s:%s', host, port)
        raise PlatformNotReady
    dev = []  # type: List[SensorEntity]
    for printer in printers:
        dev.append(IPPSensor(data, printer))
        if 'marker-names' in data.attributes[printer]:
            for marker in data.attributes[printer]['marker-names']:
                dev.append(MarkerSensor(data, printer, marker, False))
    add_entities(dev, True)


class CupsSensor(SensorEntity):
    _attr_icon: str = ICON_PRINTER

    def __init__(self, data: CupsData, printer_name: str) -> None:
        self.data: CupsData = data
        self._name: str = printer_name
        self._printer: Optional[Dict[str, Any]] = None
        self._attr_available: bool = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def native_value(self) -> Optional[Any]:
        if self._printer is None:
            return None
        key = self._printer['printer-state']
        return PRINTER_STATES.get(key, key)

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        if self._printer is None:
            return None
        return {
            ATTR_DEVICE_URI: self._printer['device-uri'],
            ATTR_PRINTER_INFO: self._printer['printer-info'],
            ATTR_PRINTER_IS_SHARED: self._printer['printer-is-shared'],
            ATTR_PRINTER_LOCATION: self._printer['printer-location'],
            ATTR_PRINTER_MODEL: self._printer['printer-make-and-model'],
            ATTR_PRINTER_STATE_MESSAGE: self._printer['printer-state-message'],
            ATTR_PRINTER_STATE_REASON: self._printer['printer-state-reasons'],
            ATTR_PRINTER_TYPE: self._printer['printer-type'],
            ATTR_PRINTER_URI_SUPPORTED: self._printer['printer-uri-supported']
        }

    def update(self) -> None:
        self.data.update()
        assert self.data.printers is not None
        self._printer = self.data.printers.get(self.name)
        self._attr_available = self.data.available


class IPPSensor(SensorEntity):
    _attr_icon: str = ICON_PRINTER

    def __init__(self, data: CupsData, printer_name: str) -> None:
        self.data: CupsData = data
        self._printer_name: str = printer_name
        self._attributes: Optional[Dict[str, Any]] = None
        self._attr_available: bool = False

    @property
    def name(self) -> str:
        if self._attributes is None:
            return ""
        return self._attributes['printer-make-and-model']

    @property
    def native_value(self) -> Optional[Any]:
        if self._attributes is None:
            return None
        key = self._attributes['printer-state']
        return PRINTER_STATES.get(key, key)

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        if self._attributes is None:
            return None
        state_attributes: Dict[str, Any] = {}
        if 'printer-info' in self._attributes:
            state_attributes[ATTR_PRINTER_INFO] = self._attributes['printer-info']
        if 'printer-location' in self._attributes:
            state_attributes[ATTR_PRINTER_LOCATION] = self._attributes['printer-location']
        if 'printer-state-message' in self._attributes:
            state_attributes[ATTR_PRINTER_STATE_MESSAGE] = self._attributes['printer-state-message']
        if 'printer-state-reasons' in self._attributes:
            state_attributes[ATTR_PRINTER_STATE_REASON] = self._attributes['printer-state-reasons']
        if 'printer-uri-supported' in self._attributes:
            state_attributes[ATTR_PRINTER_URI_SUPPORTED] = self._attributes['printer-uri-supported']
        return state_attributes

    def update(self) -> None:
        self.data.update()
        self._attributes = self.data.attributes.get(self._printer_name)
        self._attr_available = self.data.available


class MarkerSensor(SensorEntity):
    _attr_icon: str = ICON_MARKER
    _attr_native_unit_of_measurement: str = PERCENTAGE

    def __init__(self, data: CupsData, printer: str, name: str, is_cups: bool) -> None:
        self.data: CupsData = data
        self._attr_name: str = name
        self._printer: str = printer
        self._index: int = data.attributes[printer]['marker-names'].index(name)
        self._is_cups: bool = is_cups
        self._attributes: Optional[Dict[str, Any]] = None

    @property
    def native_value(self) -> Optional[Any]:
        if self._attributes is None:
            return None
        return self._attributes[self._printer]['marker-levels'][self._index]

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        if self._attributes is None:
            return None
        high_level: Any = self._attributes[self._printer].get('marker-high-levels')
        if isinstance(high_level, list):
            high_level = high_level[self._index]
        low_level: Any = self._attributes[self._printer].get('marker-low-levels')
        if isinstance(low_level, list):
            low_level = low_level[self._index]
        marker_types: Any = self._attributes[self._printer]['marker-types']
        if isinstance(marker_types, list):
            marker_types = marker_types[self._index]
        if self._is_cups:
            printer_name: str = self._printer
        else:
            printer_name = self._attributes[self._printer]['printer-make-and-model']
        return {
            ATTR_MARKER_HIGH_LEVEL: high_level,
            ATTR_MARKER_LOW_LEVEL: low_level,
            ATTR_MARKER_TYPE: marker_types,
            ATTR_PRINTER_NAME: printer_name
        }

    def update(self) -> None:
        self._attributes = self.data.attributes


class CupsData:
    printers: Optional[Dict[str, Any]] = None
    attributes: Dict[str, Any]
    available: bool

    def __init__(self, host: str, port: int, ipp_printers: Optional[List[str]]) -> None:
        self._host: str = host
        self._port: int = port
        self._ipp_printers: Optional[List[str]] = ipp_printers
        self.is_cups: bool = ipp_printers is None
        self.printers = None
        self.attributes = {}
        self.available = False

    def update(self) -> None:
        cups = importlib.import_module('cups')
        try:
            conn = cups.Connection(host=self._host, port=self._port)
            if self.is_cups:
                self.printers = conn.getPrinters()
                assert self.printers is not None
                for printer in self.printers:
                    self.attributes[printer] = conn.getPrinterAttributes(name=printer)
            else:
                assert self._ipp_printers is not None
                for ipp_printer in self._ipp_printers:
                    self.attributes[ipp_printer] = conn.getPrinterAttributes(uri=f'ipp://{self._host}:{self._port}/{ipp_printer}')
            self.available = True
        except RuntimeError:
            self.available = False
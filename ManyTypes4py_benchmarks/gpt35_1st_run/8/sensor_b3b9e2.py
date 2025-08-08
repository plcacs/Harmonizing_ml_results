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

_LOGGER: logging.Logger
ATTR_MARKER_TYPE: str
ATTR_MARKER_LOW_LEVEL: str
ATTR_MARKER_HIGH_LEVEL: str
ATTR_PRINTER_NAME: str
ATTR_DEVICE_URI: str
ATTR_PRINTER_INFO: str
ATTR_PRINTER_IS_SHARED: str
ATTR_PRINTER_LOCATION: str
ATTR_PRINTER_MODEL: str
ATTR_PRINTER_STATE_MESSAGE: str
ATTR_PRINTER_STATE_REASON: str
ATTR_PRINTER_TYPE: str
ATTR_PRINTER_URI_SUPPORTED: str
CONF_PRINTERS: str
CONF_IS_CUPS_SERVER: str
DEFAULT_HOST: str
DEFAULT_PORT: int
DEFAULT_IS_CUPS_SERVER: bool
ICON_PRINTER: str
ICON_MARKER: str
SCAN_INTERVAL: timedelta
PRINTER_STATES: Dict[int, str]
PLATFORM_SCHEMA: vol.Schema

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: Optional[DiscoveryInfoType]) -> None:

class CupsSensor(SensorEntity):

    def __init__(self, data: CupsData, printer_name: str) -> None:

    @property
    def name(self) -> str:

    @property
    def native_value(self) -> Optional[str]:

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:

    def update(self) -> None:

class IPPSensor(SensorEntity):

    def __init__(self, data: CupsData, printer_name: str) -> None:

    @property
    def name(self) -> str:

    @property
    def native_value(self) -> Optional[str]:

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:

    def update(self) -> None:

class MarkerSensor(SensorEntity):

    def __init__(self, data: CupsData, printer: str, name: str, is_cups: bool) -> None:

    @property
    def native_value(self) -> Optional[float]:

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:

    def update(self) -> None:

class CupsData:

    def __init__(self, host: str, port: int, ipp_printers: Optional[List[str]]) -> None:

    def update(self) -> None:

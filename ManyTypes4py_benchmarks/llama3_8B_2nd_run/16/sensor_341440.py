from __future__ import annotations
import datetime
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Union

from homeassistant.components.sensor import PLATFORM_SCHEMA, SensorDeviceClass, SensorEntity
from homeassistant.const import CONF_NAME, CONF_OFFSET, CONF_TOMORROW, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER = logging.getLogger(__name__)

CONF_DATA: str
CONF_DESTINATION: str
CONF_ORIGIN: str
CONF_NAME: str
CONF_OFFSET: Union[int, float]
CONF_TOMORROW: bool
DEFAULT_NAME: str
DEFAULT_PATH: str
PLATFORM_SCHEMA: Dict[str, Any]

class GTFSDepartureSensor(SensorEntity):
    """Implementation of a GTFS departure sensor."""
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.TIMESTAMP

    def __init__(self, gtfs: Any, name: str, origin: str, destination: str, offset: int, include_tomorrow: bool) -> None:
        """Initialize the sensor."""
        self._pygtfs: Any = gtfs
        self.origin: str = origin
        self.destination: str = destination
        self._include_tomorrow: bool = include_tomorrow
        self._offset: int = offset
        self._custom_name: str = name
        self._available: bool = False
        self._icon: str = ICON
        self._name: str = ''
        self._state: Optional[datetime.datetime] = None
        self._attributes: Dict[str, Any] = {}
        self._agency: Optional[Any] = None
        self._departure: Optional[Dict[str, Any]] = None
        self._destination: Optional[Any] = None
        self._origin: Optional[Any] = None
        self._route: Optional[Any] = None
        self._trip: Optional[Any] = None
        self.lock: threading.Lock = threading.Lock()

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def native_value(self) -> Optional[datetime.datetime]:
        """Return the state of the sensor."""
        return self._state

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._available

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        return self._attributes

    @property
    def icon(self) -> str:
        """Icon to use in the frontend, if any."""
        return self._icon

    @callback
    def update(self) -> None:
        """Get the latest data from GTFS and update the states."""
        with self.lock:
            # ... (rest of the method remains the same)

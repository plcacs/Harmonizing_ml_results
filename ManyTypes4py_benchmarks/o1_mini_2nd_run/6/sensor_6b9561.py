"""Support for Open Hardware Monitor Sensor Platform."""
from __future__ import annotations
from datetime import timedelta, datetime
import logging
from typing import Any, Dict, List, Optional

import requests
import voluptuous as vol
from homeassistant.components.sensor import (
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle
from homeassistant.util.dt import utcnow

_LOGGER = logging.getLogger(__name__)

STATE_MIN_VALUE = 'minimal_value'
STATE_MAX_VALUE = 'maximum_value'
STATE_VALUE = 'value'
STATE_OBJECT = 'object'

CONF_INTERVAL = 'interval'

MIN_TIME_BETWEEN_UPDATES = timedelta(seconds=15)
SCAN_INTERVAL = timedelta(seconds=30)
RETRY_INTERVAL = timedelta(seconds=30)

OHM_VALUE = 'Value'
OHM_MIN = 'Min'
OHM_MAX = 'Max'
OHM_CHILDREN = 'Children'
OHM_NAME = 'Text'

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_HOST): cv.string,
    vol.Optional(CONF_PORT, default=8085): cv.port
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the Open Hardware Monitor platform."""
    data = OpenHardwareMonitorData(config, hass)
    if data.data is None:
        raise PlatformNotReady
    add_entities(data.devices, True)

class OpenHardwareMonitorDevice(SensorEntity):
    """Device used to display information from OpenHardwareMonitor."""
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(
        self,
        data: OpenHardwareMonitorData,
        name: str,
        path: List[int],
        unit_of_measurement: str
    ) -> None:
        """Initialize an OpenHardwareMonitor sensor."""
        self._name: str = name
        self._data: OpenHardwareMonitorData = data
        self.path: List[int] = path
        self.attributes: Dict[str, Any] = {}
        self._unit_of_measurement: str = unit_of_measurement
        self.value: Optional[str] = None

    @property
    def name(self) -> str:
        """Return the name of the device."""
        return self._name

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit of measurement."""
        return self._unit_of_measurement

    @property
    def native_value(self) -> Optional[float]:
        """Return the state of the device."""
        if self.value == '-':
            return None
        try:
            return float(self.value)
        except ValueError:
            return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes of the entity."""
        return self.attributes

    @classmethod
    def parse_number(cls, string: str) -> str:
        """In some locales a decimal numbers uses ',' instead of '.'."""
        return string.replace(',', '.')

    def update(self) -> None:
        """Update the device from a new JSON object."""
        self._data.update()
        if not self._data.data:
            return
        array: List[Dict[str, Any]] = self._data.data[OHM_CHILDREN]
        _attributes: Dict[str, Any] = {}
        for path_index, path_number in enumerate(self.path):
            values: Dict[str, Any] = array[path_number]
            if path_index == len(self.path) - 1:
                self.value = self.parse_number(values[OHM_VALUE].split(' ')[0])
                _attributes.update({
                    'name': values[OHM_NAME],
                    STATE_MIN_VALUE: self.parse_number(values[OHM_MIN].split(' ')[0]),
                    STATE_MAX_VALUE: self.parse_number(values[OHM_MAX].split(' ')[0])
                })
                self.attributes = _attributes
                return
            array = values.get(OHM_CHILDREN, [])
            _attributes.update({f'level_{path_index}': values.get(OHM_NAME, '')})

class OpenHardwareMonitorData:
    """Class used to pull data from OHM and create sensors."""

    def __init__(self, config: ConfigType, hass: HomeAssistant) -> None:
        """Initialize the Open Hardware Monitor data-handler."""
        self.data: Optional[Dict[str, Any]] = None
        self._config: ConfigType = config
        self._hass: HomeAssistant = hass
        self.devices: List[OpenHardwareMonitorDevice] = []
        self.initialize(utcnow())

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> None:
        """Hit by the timer with the configured interval."""
        if self.data is None:
            self.initialize(utcnow())
        else:
            self.refresh()

    def refresh(self) -> None:
        """Download and parse JSON from OHM."""
        data_url = f"http://{self._config.get(CONF_HOST)}:{self._config.get(CONF_PORT)}/data.json"
        try:
            response = requests.get(data_url, timeout=30)
            self.data = response.json()
        except requests.exceptions.ConnectionError:
            _LOGGER.debug('ConnectionError: Is OpenHardwareMonitor running?')

    def initialize(self, now: datetime) -> None:
        """Parse of the sensors and adding of devices."""
        self.refresh()
        if self.data is None:
            return
        self.devices = self.parse_children(self.data, [], [], [])

    def parse_children(
        self,
        json: Dict[str, Any],
        devices: List[OpenHardwareMonitorDevice],
        path: List[int],
        names: List[str]
    ) -> List[OpenHardwareMonitorDevice]:
        """Recursively loop through child objects, finding the values."""
        result = devices.copy()
        children = json.get(OHM_CHILDREN)
        if children:
            for child_index in range(len(children)):
                child_path = path.copy()
                child_path.append(child_index)
                child_names = names.copy()
                if path:
                    child_names.append(json.get(OHM_NAME, ''))
                obj = children[child_index]
                added_devices = self.parse_children(obj, devices, child_path, child_names)
                result.extend(added_devices)
            return result
        value = json.get(OHM_VALUE, '')
        if ' ' not in value:
            return result
        parts = value.split(' ')
        if len(parts) < 2:
            return result
        unit_of_measurement = parts[1]
        child_names = names.copy()
        child_names.append(json.get(OHM_NAME, ''))
        fullname = ' '.join(child_names)
        dev = OpenHardwareMonitorDevice(
            data=self,
            name=fullname,
            path=path,
            unit_of_measurement=unit_of_measurement
        )
        result.append(dev)
        return result

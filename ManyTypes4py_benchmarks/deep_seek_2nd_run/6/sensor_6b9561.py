"""Support for Open Hardware Monitor Sensor Platform."""
from __future__ import annotations
from datetime import timedelta
from typing import Any, Dict, List, Optional, cast
import logging
import requests
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity, SensorStateClass
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle
from homeassistant.util.dt import utcnow

_LOGGER: logging.Logger = logging.getLogger(__name__)
STATE_MIN_VALUE: str = 'minimal_value'
STATE_MAX_VALUE: str = 'maximum_value'
STATE_VALUE: str = 'value'
STATE_OBJECT: str = 'object'
CONF_INTERVAL: str = 'interval'
MIN_TIME_BETWEEN_UPDATES: timedelta = timedelta(seconds=15)
SCAN_INTERVAL: timedelta = timedelta(seconds=30)
RETRY_INTERVAL: timedelta = timedelta(seconds=30)
OHM_VALUE: str = 'Value'
OHM_MIN: str = 'Min'
OHM_MAX: str = 'Max'
OHM_CHILDREN: str = 'Children'
OHM_NAME: str = 'Text'
PLATFORM_SCHEMA: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend({
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
    data: OpenHardwareMonitorData = OpenHardwareMonitorData(config, hass)
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
        return float(self.value) if self.value is not None else None

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
        if self._data.data is None:
            return
        array: Dict[str, Any] = self._data.data[OHM_CHILDREN]
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
            array = array[path_number][OHM_CHILDREN]
            _attributes.update({f'level_{path_index}': values[OHM_NAME]})

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
        data_url: str = f'http://{self._config.get(CONF_HOST)}:{self._config.get(CONF_PORT)}/data.json'
        try:
            response: requests.Response = requests.get(data_url, timeout=30)
            self.data = cast(Dict[str, Any], response.json())
        except requests.exceptions.ConnectionError:
            _LOGGER.debug('ConnectionError: Is OpenHardwareMonitor running?')

    def initialize(self, now: Any) -> None:
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
        result: List[OpenHardwareMonitorDevice] = devices.copy()
        if json[OHM_CHILDREN]:
            for child_index in range(len(json[OHM_CHILDREN])):
                child_path: List[int] = path.copy()
                child_path.append(child_index)
                child_names: List[str] = names.copy()
                if path:
                    child_names.append(json[OHM_NAME])
                obj: Dict[str, Any] = json[OHM_CHILDREN][child_index]
                added_devices: List[OpenHardwareMonitorDevice] = self.parse_children(
                    obj, devices, child_path, child_names
                )
                result = result + added_devices
            return result
        if json[OHM_VALUE].find(' ') == -1:
            return result
        unit_of_measurement: str = json[OHM_VALUE].split(' ')[1]
        child_names: List[str] = names.copy()
        child_names.append(json[OHM_NAME])
        fullname: str = ' '.join(child_names)
        dev: OpenHardwareMonitorDevice = OpenHardwareMonitorDevice(
            self, fullname, path, unit_of_measurement
        )
        result.append(dev)
        return result

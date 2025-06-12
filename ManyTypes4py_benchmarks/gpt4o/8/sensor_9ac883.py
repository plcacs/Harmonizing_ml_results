"""Support for the Tank Utility propane monitor."""
from __future__ import annotations
import datetime
import logging
import requests
from tank_utility import auth, device as tank_monitor
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_DEVICES, CONF_EMAIL, CONF_PASSWORD, PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from typing import Any, Dict, List, Optional

_LOGGER: logging.Logger = logging.getLogger(__name__)
SCAN_INTERVAL: datetime.timedelta = datetime.timedelta(hours=1)
PLATFORM_SCHEMA: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_EMAIL): cv.string,
    vol.Required(CONF_PASSWORD): cv.string,
    vol.Required(CONF_DEVICES): vol.All(cv.ensure_list, vol.Length(min=1))
})
SENSOR_TYPE: str = 'tank'
SENSOR_ROUNDING_PRECISION: int = 1
SENSOR_ATTRS: List[str] = ['name', 'address', 'capacity', 'fuelType', 'orientation', 'status', 'time', 'time_iso']

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the Tank Utility sensor."""
    email: str = config[CONF_EMAIL]
    password: str = config[CONF_PASSWORD]
    devices: List[str] = config[CONF_DEVICES]
    try:
        token: str = auth.get_token(email, password)
    except requests.exceptions.HTTPError as http_error:
        if http_error.response.status_code == requests.codes.unauthorized:
            _LOGGER.error('Invalid credentials')
            return
    all_sensors: List[TankUtilitySensor] = []
    for device in devices:
        sensor = TankUtilitySensor(email, password, token, device)
        all_sensors.append(sensor)
    add_entities(all_sensors, True)

class TankUtilitySensor(SensorEntity):
    """Representation of a Tank Utility sensor."""

    def __init__(self, email: str, password: str, token: str, device: str) -> None:
        """Initialize the sensor."""
        self._email: str = email
        self._password: str = password
        self._token: str = token
        self._device: str = device
        self._state: Optional[float] = None
        self._name: str = f'Tank Utility {self.device}'
        self._unit_of_measurement: str = PERCENTAGE
        self._attributes: Dict[str, Any] = {}

    @property
    def device(self) -> str:
        """Return the device identifier."""
        return self._device

    @property
    def native_value(self) -> Optional[float]:
        """Return the state of the device."""
        return self._state

    @property
    def name(self) -> str:
        """Return the name of the device."""
        return self._name

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit of measurement of the device."""
        return self._unit_of_measurement

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the attributes of the device."""
        return self._attributes

    def get_data(self) -> Dict[str, Any]:
        """Get data from the device.

        Flatten dictionary to map device to map of device data.

        """
        data: Dict[str, Any] = {}
        try:
            data = tank_monitor.get_device_data(self._token, self.device)
        except requests.exceptions.HTTPError as http_error:
            if http_error.response.status_code in (requests.codes.unauthorized, requests.codes.bad_request):
                _LOGGER.debug('Getting new token')
                self._token = auth.get_token(self._email, self._password, force=True)
                data = tank_monitor.get_device_data(self._token, self.device)
            else:
                raise
        data.update(data.pop('device', {}))
        data.update(data.pop('lastReading', {}))
        return data

    def update(self) -> None:
        """Set the device state and attributes."""
        data: Dict[str, Any] = self.get_data()
        self._state = round(data[SENSOR_TYPE], SENSOR_ROUNDING_PRECISION)
        self._attributes = {k: v for k, v in data.items() if k in SENSOR_ATTRS}

from __future__ import annotations
import re
from typing import Any, Dict, Optional
from aiohttp import web
import voluptuous as vol
from homeassistant.components.http import HomeAssistantView
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_EMAIL, CONF_NAME, DEGREE
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

API_PATH: str = '/api/torque'
DEFAULT_NAME: str = 'vehicle'
DOMAIN: str = 'torque'
ENTITY_NAME_FORMAT: str = '{0} {1}'
SENSOR_EMAIL_FIELD: str = 'eml'
SENSOR_NAME_KEY: str = r'userFullName(\w+)'
SENSOR_UNIT_KEY: str = r'userUnit(\w+)'
SENSOR_VALUE_KEY: str = r'k(\w+)'
NAME_KEY = re.compile(SENSOR_NAME_KEY)
UNIT_KEY = re.compile(SENSOR_UNIT_KEY)
VALUE_KEY = re.compile(SENSOR_VALUE_KEY)
PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_EMAIL): cv.string,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
})

def convert_pid(value: str) -> int:
    """Convert pid from hex string to integer."""
    return int(value, 16)

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the Torque platform."""
    vehicle: str = config.get(CONF_NAME)
    email: str = config.get(CONF_EMAIL)
    sensors: Dict[int, TorqueSensor] = {}
    hass.http.register_view(TorqueReceiveDataView(email, vehicle, sensors, async_add_entities))

class TorqueReceiveDataView(HomeAssistantView):
    """Handle data from Torque requests."""
    url: str = API_PATH
    name: str = 'api:torque'

    def __init__(
        self,
        email: str,
        vehicle: str,
        sensors: Dict[int, TorqueSensor],
        async_add_entities: AddEntitiesCallback
    ) -> None:
        """Initialize a Torque view."""
        self.email: str = email
        self.vehicle: str = vehicle
        self.sensors: Dict[int, TorqueSensor] = sensors
        self.async_add_entities: AddEntitiesCallback = async_add_entities

    @callback
    def get(self, request: web.Request) -> Optional[str]:
        """Handle Torque data request."""
        data: Dict[str, Any] = request.query
        if self.email is not None and self.email != data[SENSOR_EMAIL_FIELD]:
            return None
        names: Dict[int, str] = {}
        units: Dict[int, str] = {}
        for key in data:
            is_name = NAME_KEY.match(key)
            is_unit = UNIT_KEY.match(key)
            is_value = VALUE_KEY.match(key)
            if is_name:
                pid: int = convert_pid(is_name.group(1))
                names[pid] = data[key]
            elif is_unit:
                pid = convert_pid(is_unit.group(1))
                temp_unit: str = data[key]
                if '\\xC2\\xB0' in temp_unit:
                    temp_unit = temp_unit.replace('\\xC2\\xB0', DEGREE)
                units[pid] = temp_unit
            elif is_value:
                pid = convert_pid(is_value.group(1))
                if pid in self.sensors:
                    self.sensors[pid].async_on_update(data[key])
        new_sensor_entities: list[TorqueSensor] = []
        for pid, name in names.items():
            if pid not in self.sensors:
                torque_sensor_entity: TorqueSensor = TorqueSensor(
                    ENTITY_NAME_FORMAT.format(self.vehicle, name),
                    units.get(pid)
                )
                new_sensor_entities.append(torque_sensor_entity)
                self.sensors[pid] = torque_sensor_entity
        if new_sensor_entities:
            self.async_add_entities(new_sensor_entities)
        return 'OK!'

class TorqueSensor(SensorEntity):
    """Representation of a Torque sensor."""

    def __init__(self, name: str, unit: Optional[str]) -> None:
        """Initialize the sensor."""
        self._name: str = name
        self._unit: Optional[str] = unit
        self._state: Optional[Any] = None

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def native_unit_of_measurement(self) -> Optional[str]:
        """Return the unit of measurement."""
        return self._unit

    @property
    def native_value(self) -> Optional[Any]:
        """Return the state of the sensor."""
        return self._state

    @property
    def icon(self) -> str:
        """Return the default icon of the sensor."""
        return 'mdi:car'

    @callback
    def async_on_update(self, value: Any) -> None:
        """Receive an update."""
        self._state = value
        self.async_write_ha_state()
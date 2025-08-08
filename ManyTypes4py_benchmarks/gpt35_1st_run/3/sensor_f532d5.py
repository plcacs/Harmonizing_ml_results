from __future__ import annotations
import logging
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_NAME, CONF_PAYLOAD, CONF_UNIT_OF_MEASUREMENT
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import EVENT

_LOGGER: logging.Logger = logging.getLogger(__name__)

CONF_VARIABLE: str = 'variable'
DEFAULT_NAME: str = 'Pilight Sensor'
PLATFORM_SCHEMA: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_VARIABLE): cv.string,
    vol.Required(CONF_PAYLOAD): vol.Schema(dict),
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_UNIT_OF_MEASUREMENT): cv.string
})

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    add_entities([PilightSensor(hass=hass, name=config.get(CONF_NAME), variable=config.get(CONF_VARIABLE), payload=config.get(CONF_PAYLOAD), unit_of_measurement=config.get(CONF_UNIT_OF_MEASUREMENT)])

class PilightSensor(SensorEntity):
    _attr_should_poll: bool = False

    def __init__(self, hass: HomeAssistant, name: str, variable: str, payload: dict, unit_of_measurement: str) -> None:
        self._state: any = None
        self._hass: HomeAssistant = hass
        self._name: str = name
        self._variable: str = variable
        self._payload: dict = payload
        self._unit_of_measurement: str = unit_of_measurement
        hass.bus.listen(EVENT, self._handle_code)

    @property
    def name(self) -> str:
        return self._name

    @property
    def native_unit_of_measurement(self) -> str:
        return self._unit_of_measurement

    @property
    def native_value(self) -> any:
        return self._state

    def _handle_code(self, call: dict) -> None:
        if self._payload.items() <= call.data.items():
            try:
                value: any = call.data[self._variable]
                self._state = value
                self.schedule_update_ha_state()
            except KeyError:
                _LOGGER.error('No variable %s in received code data %s', str(self._variable), str(call.data))

"""Support for Pilight sensors."""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_NAME, CONF_PAYLOAD, CONF_UNIT_OF_MEASUREMENT
from homeassistant.core import HomeAssistant, Event
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import EVENT
_LOGGER = logging.getLogger(__name__)
CONF_VARIABLE = 'variable'
DEFAULT_NAME = 'Pilight Sensor'
PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({vol.Required(CONF_VARIABLE
    ): cv.string, vol.Required(CONF_PAYLOAD): vol.Schema(dict), vol.
    Optional(CONF_NAME, default=DEFAULT_NAME): cv.string, vol.Optional(
    CONF_UNIT_OF_MEASUREMENT): cv.string})


def setup_platform(hass, config, add_entities, discovery_info=None):
    """Set up Pilight Sensor."""
    add_entities([PilightSensor(hass=hass, name=config.get(CONF_NAME),
        variable=config.get(CONF_VARIABLE), payload=config.get(CONF_PAYLOAD
        ), unit_of_measurement=config.get(CONF_UNIT_OF_MEASUREMENT))])


class PilightSensor(SensorEntity):
    """Representation of a sensor that can be updated using Pilight."""
    _attr_should_poll: bool = False

    def __init__(self, hass, name, variable, payload, unit_of_measurement):
        """Initialize the sensor."""
        self._state: Optional[Any] = None
        self._hass: HomeAssistant = hass
        self._name: str = name
        self._variable: str = variable
        self._payload: Dict[str, Any] = payload
        self._unit_of_measurement: Optional[str] = unit_of_measurement
        hass.bus.listen(EVENT, self._handle_code)

    @property
    def name(self):
        """Return the name of the sensor."""
        return self._name

    @property
    def native_unit_of_measurement(self):
        """Return the unit this state is expressed in."""
        return self._unit_of_measurement

    @property
    def native_value(self):
        """Return the state of the entity."""
        return self._state

    def _handle_code(self, call):
        """Handle received code by the pilight-daemon.

        If the code matches the defined payload
        of this sensor the sensor state is changed accordingly.
        """
        if self._payload.items() <= call.data.items():
            try:
                value: Any = call.data[self._variable]
                self._state = value
                self.schedule_update_ha_state()
            except KeyError:
                _LOGGER.error('No variable %s in received code data %s',
                    str(self._variable), str(call.data))

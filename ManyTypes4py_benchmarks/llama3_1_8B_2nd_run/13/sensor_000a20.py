"""Support for an exposed aREST RESTful API of a device."""
from __future__ import annotations
from datetime import timedelta
from http import HTTPStatus
import logging
import requests
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_MONITORED_VARIABLES, CONF_NAME, CONF_RESOURCE, CONF_UNIT_OF_MEASUREMENT, CONF_VALUE_TEMPLATE
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle

_LOGGER: logging.Logger = logging.getLogger(__name__)

MIN_TIME_BETWEEN_UPDATES: timedelta = timedelta(seconds=30)

CONF_FUNCTIONS: str = 'functions'
CONF_PINS: str = 'pins'
DEFAULT_NAME: str = 'aREST sensor'

PIN_VARIABLE_SCHEMA: vol.Schema = vol.Schema({vol.Optional(CONF_NAME): cv.string, vol.Optional(CONF_UNIT_OF_MEASUREMENT): cv.string, vol.Optional(CONF_VALUE_TEMPLATE): cv.template})
PLATFORM_SCHEMA: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend({vol.Required(CONF_RESOURCE): cv.url, vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string, vol.Optional(CONF_PINS, default={}): vol.Schema({cv.string: PIN_VARIABLE_SCHEMA}), vol.Optional(CONF_MONITORED_VARIABLES, default={}): vol.Schema({cv.string: PIN_VARIABLE_SCHEMA})})

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType | None = None) -> None:
    """Set up the aREST sensor."""
    resource: str = config[CONF_RESOURCE]
    var_conf: ConfigType | None = config.get(CONF_MONITORED_VARIABLES)
    pins: ConfigType | None = config.get(CONF_PINS)
    try:
        response: requests.Response = requests.get(resource, timeout=10)
        response.raise_for_status()
        response_json: dict = response.json()
    except requests.exceptions.MissingSchema:
        _LOGGER.error('Missing resource or schema in configuration. Add http:// to your URL')
        return
    except requests.exceptions.ConnectionError:
        _LOGGER.error('No route to device at %s', resource)
        return
    arest: ArestData = ArestData(resource)

    def make_renderer(value_template: str | None) -> callable:
        """Create a renderer based on variable_template value."""
        if value_template is None:
            return lambda value: value

        def _render(value: str) -> str:
            try:
                return value_template.async_render({'value': value}, parse_result=False)
            except TemplateError:
                _LOGGER.exception('Error parsing value')
                return value
        return _render
    devices: list[ArestSensor] = []
    if var_conf is not None:
        for variable, var_data in var_conf.items():
            if variable not in response_json['variables']:
                _LOGGER.error('Variable: %s does not exist', variable)
                continue
            renderer: callable = make_renderer(var_data.get(CONF_VALUE_TEMPLATE))
            devices.append(ArestSensor(arest, resource, config.get(CONF_NAME, response_json[CONF_NAME]), var_data.get(CONF_NAME, variable), variable=variable, unit_of_measurement=var_data.get(CONF_UNIT_OF_MEASUREMENT), renderer=renderer))
    if pins is not None:
        for pinnum, pin in pins.items():
            renderer: callable = make_renderer(pin.get(CONF_VALUE_TEMPLATE))
            devices.append(ArestSensor(ArestData(resource, pinnum), resource, config.get(CONF_NAME, response_json[CONF_NAME]), pin.get(CONF_NAME), pin=pinnum, unit_of_measurement=pin.get(CONF_UNIT_OF_MEASUREMENT), renderer=renderer))
    add_entities(devices, True)

class ArestSensor(SensorEntity):
    """Implementation of an aREST sensor for exposed variables."""

    def __init__(self, arest: ArestData, resource: str, location: str, name: str, variable: str | None = None, pin: str | None = None, unit_of_measurement: str | None = None, renderer: callable | None = None) -> None:
        """Initialize the sensor."""
        self.arest: ArestData = arest
        self._attr_name: str = f'{location.title()} {name.title()}'
        self._variable: str | None = variable
        self._attr_native_unit_of_measurement: str | None = unit_of_measurement
        self._renderer: callable | None = renderer
        if pin is not None:
            request: requests.Response = requests.get(f'{resource}/mode/{pin}/i', timeout=10)
            if request.status_code != HTTPStatus.OK:
                _LOGGER.error("Can't set mode of %s", resource)

    def update(self) -> None:
        """Get the latest data from aREST API."""
        self.arest.update()
        self._attr_available: bool = self.arest.available
        values: dict = self.arest.data
        if 'error' in values:
            self._attr_native_value: str = values['error']
        else:
            self._attr_native_value: str = self._renderer(values.get('value', values.get(self._variable, None)))

class ArestData:
    """The Class for handling the data retrieval for variables."""

    def __init__(self, resource: str, pin: str | None = None) -> None:
        """Initialize the data object."""
        self._resource: str = resource
        self._pin: str | None = pin
        self.data: dict = {}
        self.available: bool = True

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> None:
        """Get the latest data from aREST device."""
        try:
            if self._pin is None:
                response: requests.Response = requests.get(self._resource, timeout=10)
                response.raise_for_status()
                response_json: dict = response.json()
                self.data = response_json['variables']
            else:
                try:
                    if str(self._pin[0]) == 'A':
                        response: requests.Response = requests.get(f'{self._resource}/analog/{self._pin[1:]}', timeout=10)
                        response.raise_for_status()
                        response_json: dict = response.json()
                        self.data = {'value': response_json['return_value']}
                except TypeError:
                    response: requests.Response = requests.get(f'{self._resource}/digital/{self._pin}', timeout=10)
                    response.raise_for_status()
                    response_json: dict = response.json()
                    self.data = {'value': response_json['return_value']}
            self.available = True
        except requests.exceptions.ConnectionError:
            _LOGGER.error('No route to device %s', self._resource)
            self.available = False

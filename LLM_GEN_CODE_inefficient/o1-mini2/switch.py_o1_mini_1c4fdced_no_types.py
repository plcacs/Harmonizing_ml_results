"""Support for an exposed aREST RESTful API of a device."""
from __future__ import annotations
from http import HTTPStatus
import logging
from typing import Any, Dict, List, Optional, Union
import requests
import voluptuous as vol
from homeassistant.components.switch import PLATFORM_SCHEMA as SWITCH_PLATFORM_SCHEMA, SwitchEntity
from homeassistant.const import CONF_NAME, CONF_RESOURCE
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
_LOGGER = logging.getLogger(__name__)
CONF_FUNCTIONS = 'functions'
CONF_PINS = 'pins'
CONF_INVERT = 'invert'
DEFAULT_NAME = 'aREST switch'
PIN_FUNCTION_SCHEMA = vol.Schema({vol.Optional(CONF_NAME): cv.string, vol.
    Optional(CONF_INVERT, default=False): cv.boolean})
PLATFORM_SCHEMA = SWITCH_PLATFORM_SCHEMA.extend({vol.Required(CONF_RESOURCE
    ): cv.url, vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_PINS, default={}): vol.Schema({cv.string:
    PIN_FUNCTION_SCHEMA}), vol.Optional(CONF_FUNCTIONS, default={}): vol.
    Schema({cv.string: PIN_FUNCTION_SCHEMA})})


def setup_platform(hass, config, add_entities, discovery_info=None):
    """Set up the aREST switches."""
    resource: str = config[CONF_RESOURCE]
    try:
        response: requests.Response = requests.get(resource, timeout=10)
    except requests.exceptions.MissingSchema:
        _LOGGER.error(
            'Missing resource or schema in configuration. Add http:// to your URL'
            )
        return
    except requests.exceptions.ConnectionError:
        _LOGGER.error('No route to device at %s', resource)
        return
    dev: List[SwitchEntity] = []
    pins: Dict[str, Any] = config[CONF_PINS]
    for pinnum, pin in pins.items():
        dev.append(ArestSwitchPin(resource, config.get(CONF_NAME, response.
            json().get(CONF_NAME, DEFAULT_NAME)), pin.get(CONF_NAME, pinnum
            ), pinnum, pin[CONF_INVERT]))
    functions: Dict[str, Any] = config[CONF_FUNCTIONS]
    for funcname, func in functions.items():
        dev.append(ArestSwitchFunction(resource, config.get(CONF_NAME,
            response.json().get(CONF_NAME, DEFAULT_NAME)), func.get(
            CONF_NAME, funcname), funcname))
    add_entities(dev)


class ArestSwitchBase(SwitchEntity):
    """Representation of an aREST switch."""
    _resource: str
    _attr_name: str
    _attr_available: bool
    _attr_is_on: bool

    def __init__(self, resource, location, name):
        """Initialize the switch."""
        self._resource = resource
        self._attr_name = f'{location.title()} {name.title()}'
        self._attr_available = True
        self._attr_is_on = False


class ArestSwitchFunction(ArestSwitchBase):
    """Representation of an aREST switch."""
    _func: str

    def __init__(self, resource, location, name, func):
        """Initialize the switch."""
        super().__init__(resource, location, name)
        self._func = func
        try:
            request: requests.Response = requests.get(
                f'{self._resource}/{self._func}', timeout=10)
            if request.status_code != HTTPStatus.OK:
                _LOGGER.error("Can't find function")
                return
            if 'return_value' not in request.json():
                _LOGGER.error('No return_value received')
        except requests.exceptions.RequestException as e:
            _LOGGER.error('Error initializing function switch: %s', e)

    def turn_on(self, **kwargs: Any):
        """Turn the device on."""
        try:
            request: requests.Response = requests.get(
                f'{self._resource}/{self._func}', timeout=10, params={
                'params': '1'})
            if request.status_code == HTTPStatus.OK:
                self._attr_is_on = True
            else:
                _LOGGER.error("Can't turn on function %s at %s", self._func,
                    self._resource)
        except requests.exceptions.RequestException as e:
            _LOGGER.error('Request failed to turn on function %s at %s: %s',
                self._func, self._resource, e)

    def turn_off(self, **kwargs: Any):
        """Turn the device off."""
        try:
            request: requests.Response = requests.get(
                f'{self._resource}/{self._func}', timeout=10, params={
                'params': '0'})
            if request.status_code == HTTPStatus.OK:
                self._attr_is_on = False
            else:
                _LOGGER.error("Can't turn off function %s at %s", self.
                    _func, self._resource)
        except requests.exceptions.RequestException as e:
            _LOGGER.error('Request failed to turn off function %s at %s: %s',
                self._func, self._resource, e)

    def update(self):
        """Get the latest data from aREST API and update the state."""
        try:
            request: requests.Response = requests.get(
                f'{self._resource}/{self._func}', timeout=10)
            data: Dict[str, Any] = request.json()
            self._attr_is_on = data.get('return_value', 0) != 0
            self._attr_available = True
        except requests.exceptions.RequestException:
            _LOGGER.warning('No route to device %s', self._resource)
            self._attr_available = False


class ArestSwitchPin(ArestSwitchBase):
    """Representation of an aREST switch. Based on digital I/O."""
    _pin: str
    invert: bool

    def __init__(self, resource, location, name, pin, invert):
        """Initialize the switch."""
        super().__init__(resource, location, name)
        self._pin = pin
        self.invert = invert
        self.__set_pin_output()

    def turn_on(self, **kwargs: Any):
        """Turn the device on."""
        turn_on_payload: int = int(not self.invert)
        try:
            request: requests.Response = requests.get(
                f'{self._resource}/digital/{self._pin}/{turn_on_payload}',
                timeout=10)
            if request.status_code == HTTPStatus.OK:
                self._attr_is_on = True
            else:
                _LOGGER.error("Can't turn on pin %s at %s", self._pin, self
                    ._resource)
        except requests.exceptions.RequestException as e:
            _LOGGER.error('Request failed to turn on pin %s at %s: %s',
                self._pin, self._resource, e)

    def turn_off(self, **kwargs: Any):
        """Turn the device off."""
        turn_off_payload: int = int(self.invert)
        try:
            request: requests.Response = requests.get(
                f'{self._resource}/digital/{self._pin}/{turn_off_payload}',
                timeout=10)
            if request.status_code == HTTPStatus.OK:
                self._attr_is_on = False
            else:
                _LOGGER.error("Can't turn off pin %s at %s", self._pin,
                    self._resource)
        except requests.exceptions.RequestException as e:
            _LOGGER.error('Request failed to turn off pin %s at %s: %s',
                self._pin, self._resource, e)

    def update(self):
        """Get the latest data from aREST API and update the state."""
        try:
            request: requests.Response = requests.get(
                f'{self._resource}/digital/{self._pin}', timeout=10)
            data: Dict[str, Any] = request.json()
            status_value: int = int(self.invert)
            self._attr_is_on = data.get('return_value', status_value
                ) != status_value
            if not self._attr_available:
                self._attr_available = True
                self.__set_pin_output()
        except requests.exceptions.RequestException:
            _LOGGER.warning('No route to device %s', self._resource)
            self._attr_available = False

    def __set_pin_output(self):
        """Set the pin mode to output."""
        try:
            request: requests.Response = requests.get(
                f'{self._resource}/mode/{self._pin}/o', timeout=10)
            if request.status_code != HTTPStatus.OK:
                _LOGGER.error("Can't set mode")
                self._attr_available = False
        except requests.exceptions.RequestException as e:
            _LOGGER.error('Failed to set pin mode for pin %s: %s', self._pin, e
                )
            self._attr_available = False

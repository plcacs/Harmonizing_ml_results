"""Support for an exposed aREST RESTful API of a device."""
from __future__ import annotations
from datetime import timedelta
from http import HTTPStatus
import logging
from typing import Any, Callable, Dict, Optional

import requests
import voluptuous as vol
from homeassistant.components.sensor import (
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    SensorEntity,
)
from homeassistant.const import (
    CONF_MONITORED_VARIABLES,
    CONF_NAME,
    CONF_RESOURCE,
    CONF_UNIT_OF_MEASUREMENT,
    CONF_VALUE_TEMPLATE,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle

_LOGGER = logging.getLogger(__name__)

MIN_TIME_BETWEEN_UPDATES = timedelta(seconds=30)

CONF_FUNCTIONS = "functions"
CONF_PINS = "pins"

DEFAULT_NAME = "aREST sensor"

PIN_VARIABLE_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_NAME): cv.string,
        vol.Optional(CONF_UNIT_OF_MEASUREMENT): cv.string,
        vol.Optional(CONF_VALUE_TEMPLATE): cv.template,
    }
)

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_RESOURCE): cv.url,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_PINS, default={}): vol.Schema(
            {cv.string: PIN_VARIABLE_SCHEMA}
        ),
        vol.Optional(CONF_MONITORED_VARIABLES, default={}): vol.Schema(
            {cv.string: PIN_VARIABLE_SCHEMA}
        ),
    }
)


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the aREST sensor."""
    resource: str = config[CONF_RESOURCE]
    var_conf: Dict[str, Any] = config[CONF_MONITORED_VARIABLES]
    pins: Dict[str, Any] = config[CONF_PINS]
    try:
        response = requests.get(resource, timeout=10).json()
    except requests.exceptions.MissingSchema:
        _LOGGER.error("Missing resource or schema in configuration. Add http:// to your URL")
        return
    except requests.exceptions.ConnectionError:
        _LOGGER.error("No route to device at %s", resource)
        return
    arest = ArestData(resource)

    def make_renderer(
        value_template: Optional[vol.Template]
    ) -> Callable[[Any], Any]:
        """Create a renderer based on variable_template value."""
        if value_template is None:
            return lambda value: value

        def _render(value: Any) -> Any:
            try:
                return value_template.async_render({"value": value}, parse_result=False)
            except TemplateError:
                _LOGGER.exception("Error parsing value")
                return value

        return _render

    dev: list[ArestSensor] = []
    if var_conf is not None:
        for variable, var_data in var_conf.items():
            if variable not in response.get("variables", {}):
                _LOGGER.error("Variable: %s does not exist", variable)
                continue
            renderer = make_renderer(var_data.get(CONF_VALUE_TEMPLATE))
            sensor_name: str = var_data.get(CONF_NAME, variable)
            sensor_unit: Optional[str] = var_data.get(CONF_UNIT_OF_MEASUREMENT)
            dev.append(
                ArestSensor(
                    arest=arest,
                    resource=resource,
                    location=config.get(CONF_NAME, response.get(CONF_NAME, DEFAULT_NAME)),
                    name=sensor_name,
                    variable=variable,
                    unit_of_measurement=sensor_unit,
                    renderer=renderer,
                )
            )
    if pins is not None:
        for pinnum, pin in pins.items():
            renderer = make_renderer(pin.get(CONF_VALUE_TEMPLATE))
            pin_name: str = pin.get(CONF_NAME, pinnum)
            pin_unit: Optional[str] = pin.get(CONF_UNIT_OF_MEASUREMENT)
            dev.append(
                ArestSensor(
                    arest=ArestData(resource, pinnum),
                    resource=resource,
                    location=config.get(CONF_NAME, response.get(CONF_NAME, DEFAULT_NAME)),
                    name=pin_name,
                    pin=pinnum,
                    unit_of_measurement=pin_unit,
                    renderer=renderer,
                )
            )
    add_entities(dev, True)


class ArestSensor(SensorEntity):
    """Implementation of an aREST sensor for exposed variables."""

    def __init__(
        self,
        arest: ArestData,
        resource: str,
        location: str,
        name: str,
        variable: Optional[str] = None,
        pin: Optional[str] = None,
        unit_of_measurement: Optional[str] = None,
        renderer: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Initialize the sensor."""
        self.arest = arest
        self._attr_name = f"{location.title()} {name.title()}"
        self._variable = variable
        self._attr_native_unit_of_measurement = unit_of_measurement
        self._renderer = renderer
        self._attr_available: bool = False
        self._attr_native_value: Any = None
        if pin is not None:
            try:
                request = requests.get(f"{resource}/mode/{pin}/i", timeout=10)
                if request.status_code != HTTPStatus.OK:
                    _LOGGER.error("Can't set mode of %s", resource)
            except requests.RequestException:
                _LOGGER.error("Request exception when setting pin mode for %s", resource)

    def update(self) -> None:
        """Get the latest data from aREST API."""
        self.arest.update()
        self._attr_available = self.arest.available
        values: Dict[str, Any] = self.arest.data
        if "error" in values:
            self._attr_native_value = values["error"]
        else:
            if self._variable:
                raw_value = values.get(self._variable, None)
            else:
                raw_value = values.get("value", None)
            if self._renderer:
                self._attr_native_value = self._renderer(raw_value)
            else:
                self._attr_native_value = raw_value


class ArestData:
    """The Class for handling the data retrieval for variables."""

    def __init__(self, resource: str, pin: Optional[str] = None) -> None:
        """Initialize the data object."""
        self._resource: str = resource
        self._pin: Optional[str] = pin
        self.data: Dict[str, Any] = {}
        self.available: bool = True

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> None:
        """Get the latest data from aREST device."""
        try:
            if self._pin is None:
                response = requests.get(self._resource, timeout=10)
                self.data = response.json().get("variables", {})
            else:
                try:
                    if self._pin.startswith("A"):
                        analog_port = self._pin[1:]
                        response = requests.get(
                            f"{self._resource}/analog/{analog_port}", timeout=10
                        )
                        self.data = {"value": response.json().get("return_value")}
                except TypeError:
                    response = requests.get(
                        f"{self._resource}/digital/{self._pin}", timeout=10
                    )
                    self.data = {"value": response.json().get("return_value")}
            self.available = True
        except requests.exceptions.ConnectionError:
            _LOGGER.error("No route to device %s", self._resource)
            self.available = False
        except requests.RequestException as e:
            _LOGGER.error("Request exception for device %s: %s", self._resource, e)
            self.available = False

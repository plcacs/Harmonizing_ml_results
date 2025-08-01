from __future__ import annotations
from datetime import timedelta
import logging
from http import HTTPStatus
from typing import Any, Optional
import requests
import voluptuous as vol

from homeassistant.components.binary_sensor import (
    DEVICE_CLASSES_SCHEMA,
    PLATFORM_SCHEMA as BINARY_SENSOR_PLATFORM_SCHEMA,
    BinarySensorEntity,
)
from homeassistant.const import CONF_DEVICE_CLASS, CONF_NAME, CONF_PIN, CONF_RESOURCE
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle

_LOGGER = logging.getLogger(__name__)
MIN_TIME_BETWEEN_UPDATES = timedelta(seconds=30)
PLATFORM_SCHEMA = BINARY_SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_RESOURCE): cv.url,
    vol.Optional(CONF_NAME): cv.string,
    vol.Required(CONF_PIN): cv.string,
    vol.Optional(CONF_DEVICE_CLASS): DEVICE_CLASSES_SCHEMA
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    resource: str = config[CONF_RESOURCE]
    pin: str = config[CONF_PIN]
    device_class: Optional[str] = config.get(CONF_DEVICE_CLASS)
    try:
        response: dict[str, Any] = requests.get(resource, timeout=10).json()
    except requests.exceptions.MissingSchema:
        _LOGGER.error('Missing resource or schema in configuration. Add http:// to your URL')
        return
    except requests.exceptions.ConnectionError:
        _LOGGER.error('No route to device at %s', resource)
        return
    arest = ArestData(resource, pin)
    name: str = config.get(CONF_NAME, response[CONF_NAME])
    add_entities([ArestBinarySensor(arest, resource, name, device_class, pin)], True)

class ArestBinarySensor(BinarySensorEntity):
    def __init__(
        self,
        arest: ArestData,
        resource: str,
        name: str,
        device_class: Optional[str],
        pin: str
    ) -> None:
        self.arest: ArestData = arest
        self._attr_name: str = name
        self._attr_device_class: Optional[str] = device_class
        if pin is not None:
            request = requests.get(f'{resource}/mode/{pin}/i', timeout=10)
            if request.status_code != HTTPStatus.OK:
                _LOGGER.error("Can't set mode of %s", resource)

    def update(self) -> None:
        self.arest.update()
        self._attr_is_on = bool(self.arest.data.get('state'))

class ArestData:
    def __init__(self, resource: str, pin: str) -> None:
        self._resource: str = resource
        self._pin: str = pin
        self.data: dict[str, Any] = {}

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> None:
        try:
            response = requests.get(f'{self._resource}/digital/{self._pin}', timeout=10)
            self.data = {'state': response.json()['return_value']}
        except requests.exceptions.ConnectionError:
            _LOGGER.error("No route to device '%s'", self._resource)
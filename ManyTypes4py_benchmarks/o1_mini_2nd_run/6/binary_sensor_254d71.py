"""Support for an exposed aREST RESTful API of a device."""
from __future__ import annotations
from datetime import timedelta
from http import HTTPStatus
import logging
import requests
import voluptuous as vol
from typing import Any, Dict, Optional
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
PLATFORM_SCHEMA = BINARY_SENSOR_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_RESOURCE): cv.url,
        vol.Optional(CONF_NAME): cv.string,
        vol.Required(CONF_PIN): cv.string,
        vol.Optional(CONF_DEVICE_CLASS): DEVICE_CLASSES_SCHEMA,
    }
)


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the aREST binary sensor."""
    resource: str = config[CONF_RESOURCE]
    pin: str = config[CONF_PIN]
    device_class: Optional[str] = config.get(CONF_DEVICE_CLASS)
    try:
        response: Dict[str, Any] = requests.get(resource, timeout=10).json()
    except requests.exceptions.MissingSchema:
        _LOGGER.error(
            "Missing resource or schema in configuration. Add http:// to your URL"
        )
        return
    except requests.exceptions.ConnectionError:
        _LOGGER.error("No route to device at %s", resource)
        return
    arest: ArestData = ArestData(resource, pin)
    name: str = config.get(CONF_NAME, response.get(CONF_NAME, "aREST Sensor"))
    add_entities(
        [
            ArestBinarySensor(
                arest,
                resource,
                name,
                device_class,
                pin,
            )
        ],
        True,
    )


class ArestBinarySensor(BinarySensorEntity):
    """Implement an aREST binary sensor for a pin."""

    _attr_name: str
    _attr_device_class: Optional[str]
    _attr_is_on: bool

    def __init__(
        self,
        arest: ArestData,
        resource: str,
        name: str,
        device_class: Optional[str],
        pin: str,
    ) -> None:
        """Initialize the aREST device."""
        self.arest: ArestData = arest
        self._attr_name = name
        self._attr_device_class = device_class
        if pin is not None:
            try:
                request = requests.get(
                    f"{resource}/mode/{pin}/i", timeout=10
                )
                if request.status_code != HTTPStatus.OK:
                    _LOGGER.error("Can't set mode of %s", resource)
            except requests.exceptions.RequestException as e:
                _LOGGER.error("Error setting mode of %s: %s", resource, e)

    def update(self) -> None:
        """Get the latest data from aREST API."""
        self.arest.update()
        state = self.arest.data.get("state")
        self._attr_is_on = bool(state) if state is not None else False


class ArestData:
    """Class for handling the data retrieval for pins."""

    def __init__(self, resource: str, pin: str) -> None:
        """Initialize the aREST data object."""
        self._resource: str = resource
        self._pin: str = pin
        self.data: Dict[str, Any] = {}

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> None:
        """Get the latest data from aREST device."""
        try:
            response = requests.get(
                f"{self._resource}/digital/{self._pin}", timeout=10
            )
            response_json: Dict[str, Any] = response.json()
            self.data = {"state": response_json.get("return_value")}
        except requests.exceptions.ConnectionError:
            _LOGGER.error("No route to device '%s'", self._resource)
        except requests.exceptions.RequestException as e:
            _LOGGER.error("Error updating data from '%s': %s", self._resource, e)

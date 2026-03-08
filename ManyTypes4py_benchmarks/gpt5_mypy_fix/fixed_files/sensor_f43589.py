"""Support for showing values from Dweet.io."""
from __future__ import annotations

from datetime import timedelta
import json
import logging
from typing import Any, cast

import dweepy
import voluptuous as vol

from homeassistant.components.sensor import (
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    SensorEntity,
)
from homeassistant.const import (
    CONF_DEVICE,
    CONF_NAME,
    CONF_UNIT_OF_MEASUREMENT,
    CONF_VALUE_TEMPLATE,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.template import Template
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger = logging.getLogger(__name__)

DEFAULT_NAME: str = "Dweet.io Sensor"
SCAN_INTERVAL: timedelta = timedelta(minutes=1)

PLATFORM_SCHEMA: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_DEVICE): cv.string,
        vol.Required(CONF_VALUE_TEMPLATE): cv.template,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_UNIT_OF_MEASUREMENT): cv.string,
    }
)


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the Dweet sensor."""
    name: str = cast(str, config.get(CONF_NAME))
    device: str = cast(str, config.get(CONF_DEVICE))
    value_template: Template = cast(Template, config.get(CONF_VALUE_TEMPLATE))
    unit: str | None = cast(str | None, config.get(CONF_UNIT_OF_MEASUREMENT))

    try:
        content: str = json.dumps(
            dweepy.get_latest_dweet_for(device)[0]["content"]  # type: ignore[no-any-return]
        )
    except dweepy.DweepyError:
        _LOGGER.error("Device/thing %s could not be found", device)
        return

    if value_template and value_template.render_with_possible_json_value(content) == "":
        _LOGGER.error("%s was not found", value_template)
        return

    dweet: DweetData = DweetData(device)
    add_entities([DweetSensor(hass, dweet, name, value_template, unit)], True)


class DweetSensor(SensorEntity):
    """Representation of a Dweet sensor."""

    _name: str
    _value_template: Template
    _state: str | int | float | None
    _unit_of_measurement: str | None

    def __init__(
        self,
        hass: HomeAssistant,
        dweet: DweetData,
        name: str,
        value_template: Template,
        unit_of_measurement: str | None,
    ) -> None:
        """Initialize the sensor."""
        self.hass: HomeAssistant = hass
        self.dweet: DweetData = dweet
        self._name = name
        self._value_template = value_template
        self._state = None
        self._unit_of_measurement = unit_of_measurement

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def native_unit_of_measurement(self) -> str | None:
        """Return the unit the value is expressed in."""
        return self._unit_of_measurement

    @property
    def native_value(self) -> str | int | float | None:
        """Return the state."""
        return self._state

    def update(self) -> None:
        """Get the latest data from REST API."""
        self.dweet.update()
        if self.dweet.data is None:
            self._state = None
        else:
            values: str = json.dumps(self.dweet.data[0]["content"])
            self._state = self._value_template.render_with_possible_json_value(
                values, None
            )


class DweetData:
    """The class for handling the data retrieval."""

    def __init__(self, device: str) -> None:
        """Initialize the sensor."""
        self._device: str = device
        self.data: list[dict[str, Any]] | None = None

    def update(self) -> None:
        """Get the latest data from Dweet.io."""
        try:
            self.data = dweepy.get_latest_dweet_for(self._device)  # type: ignore[no-any-return]
        except dweepy.DweepyError:
            _LOGGER.warning("Device %s doesn't contain any data", self._device)
            self.data = None
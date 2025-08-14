from __future__ import annotations

from datetime import timedelta
import logging
import time
from typing import Any, Dict, Optional

import requests
import voluptuous as vol

from homeassistant.components.sensor import (
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    SensorEntity,
)
from homeassistant.const import CONF_API_KEY, CONF_LATITUDE, CONF_LONGITUDE, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER = logging.getLogger(__name__)

ATTRIBUTION = "Data provided by WorldTides"

DEFAULT_NAME = "WorldTidesInfo"

SCAN_INTERVAL = timedelta(seconds=3600)

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_API_KEY): cv.string,
        vol.Optional(CONF_LATITUDE): cv.latitude,
        vol.Optional(CONF_LONGITUDE): cv.longitude,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    }
)


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the WorldTidesInfo sensor."""
    name: str = config.get(CONF_NAME)

    lat: Optional[float] = config.get(CONF_LATITUDE, hass.config.latitude)
    lon: Optional[float] = config.get(CONF_LONGITUDE, hass.config.longitude)
    key: str = config.get(CONF_API_KEY)

    if None in (lat, lon):
        _LOGGER.error("Latitude or longitude not set in Home Assistant config")

    tides: WorldTidesInfoSensor = WorldTidesInfoSensor(name, lat, lon, key)
    tides.update()
    if tides.data is not None and tides.data.get("error") == "No location found":
        _LOGGER.error("Location not available")
        return

    add_entities([tides])


class WorldTidesInfoSensor(SensorEntity):
    """Representation of a WorldTidesInfo sensor."""

    _attr_attribution: str = ATTRIBUTION

    def __init__(self, name: str, lat: Optional[float], lon: Optional[float], key: str) -> None:
        """Initialize the sensor."""
        self._name: str = name
        self._lat: Optional[float] = lat
        self._lon: Optional[float] = lon
        self._key: str = key
        self.data: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes of this device."""
        attr: Dict[str, Any] = {}

        if self.data and "extremes" in self.data and len(self.data["extremes"]) >= 2:
            if "High" in str(self.data["extremes"][0]["type"]):
                attr["high_tide_time_utc"] = self.data["extremes"][0]["date"]
                attr["high_tide_height"] = self.data["extremes"][0]["height"]
                attr["low_tide_time_utc"] = self.data["extremes"][1]["date"]
                attr["low_tide_height"] = self.data["extremes"][1]["height"]
            elif "Low" in str(self.data["extremes"][0]["type"]):
                attr["high_tide_time_utc"] = self.data["extremes"][1]["date"]
                attr["high_tide_height"] = self.data["extremes"][1]["height"]
                attr["low_tide_time_utc"] = self.data["extremes"][0]["date"]
                attr["low_tide_height"] = self.data["extremes"][0]["height"]
        return attr

    @property
    def native_value(self) -> Optional[str]:
        """Return the state of the device."""
        if self.data and "extremes" in self.data and len(self.data["extremes"]) >= 1:
            if "High" in str(self.data["extremes"][0]["type"]):
                tidetime: str = time.strftime(
                    "%I:%M %p", time.localtime(self.data["extremes"][0]["dt"])
                )
                return f"High tide at {tidetime}"
            if "Low" in str(self.data["extremes"][0]["type"]):
                tidetime = time.strftime(
                    "%I:%M %p", time.localtime(self.data["extremes"][0]["dt"])
                )
                return f"Low tide at {tidetime}"
        return None

    def update(self) -> None:
        """Get the latest data from WorldTidesInfo API."""
        start: int = int(time.time())
        resource: str = (
            "https://www.worldtides.info/api?extremes&length=86400"
            f"&key={self._key}&lat={self._lat}&lon={self._lon}&start={start}"
        )

        try:
            self.data = requests.get(resource, timeout=10).json()
            _LOGGER.debug("Data: %s", self.data)
            _LOGGER.debug("Tide data queried with start time set to: %s", start)
        except ValueError as err:
            _LOGGER.error("Error retrieving data from WorldTidesInfo: %s", err.args)
            self.data = None
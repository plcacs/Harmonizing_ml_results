"""Support for the worldtides.info API."""
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

ATTRIBUTION: str = "Data provided by WorldTides"
DEFAULT_NAME: str = "WorldTidesInfo"
SCAN_INTERVAL: timedelta = timedelta(seconds=3600)

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
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the WorldTidesInfo sensor."""
    name: str = config.get(CONF_NAME)
    lat: Optional[float] = config.get(CONF_LATITUDE, hass.config.latitude)
    lon: Optional[float] = config.get(CONF_LONGITUDE, hass.config.longitude)
    key: str = config.get(CONF_API_KEY)

    if None in (lat, lon):
        _LOGGER.error("Latitude or longitude not set in Home Assistant config")
        return

    tides = WorldTidesInfoSensor(name, lat, lon, key)
    tides.update()
    if tides.data and tides.data.get("error") == "No location found":
        _LOGGER.error("Location not available")
        return

    add_entities([tides])


class WorldTidesInfoSensor(SensorEntity):
    """Representation of a WorldTidesInfo sensor."""

    _attr_attribution: str = ATTRIBUTION

    def __init__(self, name: str, lat: float, lon: float, key: str) -> None:
        """Initialize the sensor."""
        self._name: str = name
        self._lat: float = lat
        self._lon: float = lon
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
        if self.data and "extremes" in self.data:
            extremes = self.data["extremes"]
            if extremes:
                if "High" in str(extremes[0].get("type", "")):
                    attr["high_tide_time_utc"] = extremes[0].get("date")
                    attr["high_tide_height"] = extremes[0].get("height")
                    if len(extremes) > 1:
                        attr["low_tide_time_utc"] = extremes[1].get("date")
                        attr["low_tide_height"] = extremes[1].get("height")
                elif "Low" in str(extremes[0].get("type", "")):
                    if len(extremes) > 1:
                        attr["high_tide_time_utc"] = extremes[1].get("date")
                        attr["high_tide_height"] = extremes[1].get("height")
                    attr["low_tide_time_utc"] = extremes[0].get("date")
                    attr["low_tide_height"] = extremes[0].get("height")
        return attr

    @property
    def native_value(self) -> Optional[str]:
        """Return the state of the device."""
        if self.data and "extremes" in self.data:
            extremes = self.data["extremes"]
            if extremes:
                tide_type = extremes[0].get("type", "")
                tide_dt = extremes[0].get("dt")
                if tide_type and tide_dt:
                    tidetime = time.strftime("%I:%M %p", time.localtime(tide_dt))
                    if "High" in str(tide_type):
                        return f"High tide at {tidetime}"
                    if "Low" in str(tide_type):
                        return f"Low tide at {tidetime}"
        return None

    def update(self) -> None:
        """Get the latest data from WorldTidesInfo API."""
        start: int = int(time.time())
        resource: str = (
            f"https://www.worldtides.info/api?extremes&length=86400&key={self._key}"
            f"&lat={self._lat}&lon={self._lon}&start={start}"
        )
        try:
            response = requests.get(resource, timeout=10)
            response.raise_for_status()
            self.data = response.json()
            _LOGGER.debug("Data: %s", self.data)
            _LOGGER.debug("Tide data queried with start time set to: %s", start)
        except (ValueError, requests.RequestException) as err:
            _LOGGER.error("Error retrieving data from WorldTidesInfo: %s", err)
            self.data = None

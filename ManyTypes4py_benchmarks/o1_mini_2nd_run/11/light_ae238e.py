"""Support for Greenwave Reality (TCP Connected) lights."""
from __future__ import annotations

from datetime import timedelta
import logging
import os
from typing import Any, Dict, Optional

import greenwavereality as greenwave
import voluptuous as vol
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    PLATFORM_SCHEMA as LIGHT_PLATFORM_SCHEMA,
    ColorMode,
    LightEntity,
)
from homeassistant.const import CONF_HOST
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle

_LOGGER: logging.Logger = logging.getLogger(__name__)

CONF_VERSION: str = 'version'

PLATFORM_SCHEMA = LIGHT_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_HOST): cv.string,
        vol.Required(CONF_VERSION): cv.positive_int,
    }
)

MIN_TIME_BETWEEN_UPDATES: timedelta = timedelta(minutes=1)


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the Greenwave Reality Platform."""
    host: str = config.get(CONF_HOST)
    tokenfilename: str = hass.config.path('.greenwave')
    version: int = config.get(CONF_VERSION)

    if version == 3:
        if os.path.exists(tokenfilename):
            with open(tokenfilename, encoding='utf8') as tokenfile:
                token: str = tokenfile.read()
        else:
            try:
                token = greenwave.grab_token(host, 'hass', 'homeassistant')
            except PermissionError:
                _LOGGER.error('The Gateway Is Not In Sync Mode')
                raise
            with open(tokenfilename, 'w+', encoding='utf8') as tokenfile:
                tokenfile.write(token)
    else:
        token = None

    bulbs: Dict[str, Any] = greenwave.grab_bulbs(host, token)
    add_entities(
        (
            GreenwaveLight(device, host, token, GatewayData(host, token))
            for device in bulbs.values()
        )
    )


class GreenwaveLight(LightEntity):
    """Representation of a Greenwave Reality Light."""

    _attr_color_mode: ColorMode = ColorMode.BRIGHTNESS
    _attr_supported_color_modes: set[ColorMode] = {ColorMode.BRIGHTNESS}

    def __init__(
        self,
        light: Dict[str, Any],
        host: str,
        token: Optional[str],
        gatewaydata: GatewayData,
    ) -> None:
        """Initialize a Greenwave Reality Light."""
        self._did: int = int(light['did'])
        self._attr_name: str = light['name']
        self._state: bool = bool(int(light['state']))
        self._attr_brightness: int = greenwave.hass_brightness(light)
        self._host: str = host
        self._attr_available: bool = greenwave.check_online(light)
        self._token: Optional[str] = token
        self._gatewaydata: GatewayData = gatewaydata

    @property
    def is_on(self) -> bool:
        """Return true if light is on."""
        return self._state

    def turn_on(self, **kwargs: Any) -> None:
        """Instruct the light to turn on."""
        brightness: int = kwargs.get(ATTR_BRIGHTNESS, 255)
        temp_brightness: int = int(brightness / 255 * 100)
        greenwave.set_brightness(self._host, self._did, temp_brightness, self._token)
        greenwave.turn_on(self._host, self._did, self._token)

    def turn_off(self, **kwargs: Any) -> None:
        """Instruct the light to turn off."""
        greenwave.turn_off(self._host, self._did, self._token)

    def update(self) -> None:
        """Fetch new state data for this light."""
        self._gatewaydata.update()
        bulbs: Dict[str, Any] = self._gatewaydata.greenwave
        bulb: Dict[str, Any] = bulbs.get(str(self._did), {})
        self._state = bool(int(bulb.get('state', 0)))
        self._attr_brightness = greenwave.hass_brightness(bulb)
        self._attr_available = greenwave.check_online(bulb)
        self._attr_name = bulb.get('name', self._attr_name)


class GatewayData:
    """Handle Gateway data and limit updates."""

    def __init__(self, host: str, token: Optional[str]) -> None:
        """Initialize the data object."""
        self._host: str = host
        self._token: Optional[str] = token
        self._greenwave: Dict[str, Any] = greenwave.grab_bulbs(host, token)

    @property
    def greenwave(self) -> Dict[str, Any]:
        """Return Gateway API object."""
        return self._greenwave

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> None:
        """Get the latest data from the gateway."""
        self._greenwave = greenwave.grab_bulbs(self._host, self._token)

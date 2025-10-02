"""Support for Greenwave Reality (TCP Connected) lights."""
from __future__ import annotations
from datetime import timedelta
import logging
import os
from typing import Any
import greenwavereality as greenwave
import voluptuous as vol
from homeassistant.components.light import ATTR_BRIGHTNESS, PLATFORM_SCHEMA as LIGHT_PLATFORM_SCHEMA, ColorMode, LightEntity
from homeassistant.const import CONF_HOST
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle
_LOGGER = logging.getLogger(__name__)
CONF_VERSION = 'version'
PLATFORM_SCHEMA = LIGHT_PLATFORM_SCHEMA.extend({vol.Required(CONF_HOST): cv.string, vol.Required(CONF_VERSION): cv.positive_int})
MIN_TIME_BETWEEN_UPDATES = timedelta(minutes=1)

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    """Set up the Greenwave Reality Platform."""
    host: str = config.get(CONF_HOST)
    tokenfilename: str = hass.config.path('.greenwave')
    token: str | None = None
    
    if config.get(CONF_VERSION) == 3:
        if os.path.exists(tokenfilename):
            with open(tokenfilename, encoding='utf8') as tokenfile:
                token = tokenfile.read()
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
    bulbs: dict[int, dict[str, Any]] = greenwave.grab_bulbs(host, token)
    add_entities((GreenwaveLight(device, host, token, GatewayData(host, token)) for device in bulbs.values()))

class GreenwaveLight(LightEntity):
    """Representation of an Greenwave Reality Light."""
    _attr_color_mode = ColorMode.BRIGHTNESS
    _attr_supported_color_modes = {ColorMode.BRIGHTNESS}

    def __init__(self, light: dict[str, Any], host: str, token: str | None, gatewaydata: GatewayData) -> None:
        """Initialize a Greenwave Reality Light."""
        self._did: int = int(light['did'])
        self._attr_name: str = light['name']
        self._state: int = int(light['state'])
        self._attr_brightness: int = greenwave.hass_brightness(light)
        self._host: str = host
        self._attr_available: bool = greenwave.check_online(light)
        self._token: str | None = token
        self._gatewaydata: GatewayData = gatewaydata

    @property
    def is_on(self) -> bool:
        """Return true if light is on."""
        return bool(self._state)

    def turn_on(self, **kwargs: Any) -> None:
        """Instruct the light to turn on."""
        temp_brightness: int = int(kwargs.get(ATTR_BRIGHTNESS, 255) / 255 * 100)
        greenwave.set_brightness(self._host, self._did, temp_brightness, self._token)
        greenwave.turn_on(self._host, self._did, self._token)

    def turn_off(self, **kwargs: Any) -> None:
        """Instruct the light to turn off."""
        greenwave.turn_off(self._host, self._did, self._token)

    def update(self) -> None:
        """Fetch new state data for this light."""
        self._gatewaydata.update()
        bulbs: dict[int, dict[str, Any]] = self._gatewaydata.greenwave
        self._state = int(bulbs[self._did]['state'])
        self._attr_brightness = greenwave.hass_brightness(bulbs[self._did])
        self._attr_available = greenwave.check_online(bulbs[self._did])
        self._attr_name = bulbs[self._did]['name']

class GatewayData:
    """Handle Gateway data and limit updates."""

    def __init__(self, host: str, token: str | None) -> None:
        """Initialize the data object."""
        self._host: str = host
        self._token: str | None = token
        self._greenwave: dict[int, dict[str, Any]] = greenwave.grab_bulbs(host, token)

    @property
    def greenwave(self) -> dict[int, dict[str, Any]]:
        """Return Gateway API object."""
        return self._greenwave

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> dict[int, dict[str, Any]]:
        """Get the latest data from the gateway."""
        self._greenwave = greenwave.grab_bulbs(self._host, self._token)
        return self._greenwave

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
import tikteck
import voluptuous as vol
from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_HS_COLOR, PLATFORM_SCHEMA as LIGHT_PLATFORM_SCHEMA, ColorMode, LightEntity
from homeassistant.const import CONF_DEVICES, CONF_NAME, CONF_PASSWORD
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import color as color_util

_LOGGER: logging.Logger = logging.getLogger(__name__)
DEVICE_SCHEMA: vol.Schema = vol.Schema({vol.Optional(CONF_NAME): cv.string, vol.Required(CONF_PASSWORD): cv.string})
PLATFORM_SCHEMA: vol.Schema = LIGHT_PLATFORM_SCHEMA.extend({vol.Optional(CONF_DEVICES, default={}): {cv.string: DEVICE_SCHEMA}})

def func_49maqmic(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: Optional[DiscoveryInfoType] = None) -> None:
    """Set up the Tikteck platform."""
    lights: List[TikteckLight] = []
    for address, device_config in config[CONF_DEVICES].items():
        device: Dict[str, Any] = {}
        device['name'] = device_config[CONF_NAME]
        device['password'] = device_config[CONF_PASSWORD]
        device['address'] = address
        light = TikteckLight(device)
        if light.is_valid:
            lights.append(light)
    add_entities(lights)

class TikteckLight(LightEntity):
    """Representation of a Tikteck light."""
    _attr_assumed_state: bool = True
    _attr_color_mode: ColorMode = ColorMode.HS
    _attr_should_poll: bool = False
    _attr_supported_color_modes: Set[ColorMode] = {ColorMode.HS}

    def __init__(self, device: Dict[str, Any]) -> None:
        """Initialize the light."""
        address: str = device['address']
        self._attr_unique_id: str = address
        self._attr_name: str = device['name']
        self._attr_brightness: int = 255
        self._attr_hs_color: Tuple[int, int] = (0, 0)
        self._attr_is_on: bool = False
        self.is_valid: bool = True
        self._bulb = tikteck.tikteck(address, 'Smart Light', device['password'])
        if self._bulb.connect() is False:
            self.is_valid = False
            _LOGGER.error('Failed to connect to bulb %s, %s', address, self.name)

    def func_81lt5oor(self, red: int, green: int, blue: int, brightness: int) -> Any:
        """Set the bulb state."""
        return self._bulb.set_state(red, green, blue, brightness)

    def func_fp88lovl(self, **kwargs: Any) -> None:
        """Turn the specified light on."""
        self._attr_is_on = True
        hs_color = kwargs.get(ATTR_HS_COLOR)
        brightness = kwargs.get(ATTR_BRIGHTNESS)
        if hs_color is not None:
            self._attr_hs_color = hs_color
        if brightness is not None:
            self._attr_brightness = brightness
        rgb = color_util.color_hs_to_RGB(self.hs_color[0], self.hs_color[1])
        self.set_state(rgb[0], rgb[1], rgb[2], self.brightness)
        self.schedule_update_ha_state()

    def func_gcgp7w86(self, **kwargs: Any) -> None:
        """Turn the specified light off."""
        self._attr_is_on = False
        self.set_state(0, 0, 0, 0)
        self.schedule_update_ha_state()

"""Support for ADS light sources."""
from __future__ import annotations
from typing import Any
import pyads
import voluptuous as vol
from homeassistant.components.light import ATTR_BRIGHTNESS, PLATFORM_SCHEMA as LIGHT_PLATFORM_SCHEMA, ColorMode, LightEntity
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from .const import CONF_ADS_VAR, DATA_ADS, STATE_KEY_STATE
from .entity import AdsEntity
from .hub import AdsHub
CONF_ADS_VAR_BRIGHTNESS = 'adsvar_brightness'
STATE_KEY_BRIGHTNESS = 'brightness'
DEFAULT_NAME = 'ADS Light'
PLATFORM_SCHEMA = LIGHT_PLATFORM_SCHEMA.extend({vol.Required(CONF_ADS_VAR): cv.string, vol.Optional(CONF_ADS_VAR_BRIGHTNESS): cv.string, vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string})

def setup_platform(hass, config, add_entities, discovery_info=None):
    """Set up the light platform for ADS."""
    ads_hub = hass.data[DATA_ADS]
    ads_var_enable = config[CONF_ADS_VAR]
    ads_var_brightness = config.get(CONF_ADS_VAR_BRIGHTNESS)
    name = config[CONF_NAME]
    add_entities([AdsLight(ads_hub, ads_var_enable, ads_var_brightness, name)])

class AdsLight(AdsEntity, LightEntity):
    """Representation of ADS light."""

    def __init__(self, ads_hub, ads_var_enable, ads_var_brightness, name):
        """Initialize AdsLight entity."""
        super().__init__(ads_hub, name, ads_var_enable)
        self._state_dict[STATE_KEY_BRIGHTNESS] = None
        self._ads_var_brightness = ads_var_brightness
        if ads_var_brightness is not None:
            self._attr_color_mode = ColorMode.BRIGHTNESS
            self._attr_supported_color_modes = {ColorMode.BRIGHTNESS}
        else:
            self._attr_color_mode = ColorMode.ONOFF
            self._attr_supported_color_modes = {ColorMode.ONOFF}

    async def async_added_to_hass(self):
        """Register device notification."""
        await self.async_initialize_device(self._ads_var, pyads.PLCTYPE_BOOL)
        if self._ads_var_brightness is not None:
            await self.async_initialize_device(self._ads_var_brightness, pyads.PLCTYPE_UINT, STATE_KEY_BRIGHTNESS)

    @property
    def brightness(self):
        """Return the brightness of the light (0..255)."""
        return self._state_dict[STATE_KEY_BRIGHTNESS]

    @property
    def is_on(self):
        """Return True if the entity is on."""
        return self._state_dict[STATE_KEY_STATE]

    def turn_on(self, **kwargs):
        """Turn the light on or set a specific dimmer value."""
        brightness = kwargs.get(ATTR_BRIGHTNESS)
        self._ads_hub.write_by_name(self._ads_var, True, pyads.PLCTYPE_BOOL)
        if self._ads_var_brightness is not None and brightness is not None:
            self._ads_hub.write_by_name(self._ads_var_brightness, brightness, pyads.PLCTYPE_UINT)

    def turn_off(self, **kwargs):
        """Turn the light off."""
        self._ads_hub.write_by_name(self._ads_var, False, pyads.PLCTYPE_BOOL)
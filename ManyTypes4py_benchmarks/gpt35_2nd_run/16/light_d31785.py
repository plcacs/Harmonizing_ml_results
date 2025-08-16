from __future__ import annotations
from typing import Any, Dict, Optional
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

CONF_ADS_VAR_BRIGHTNESS: str = 'adsvar_brightness'
STATE_KEY_BRIGHTNESS: str = 'brightness'
DEFAULT_NAME: str = 'ADS Light'
PLATFORM_SCHEMA: vol.Schema = LIGHT_PLATFORM_SCHEMA.extend({vol.Required(CONF_ADS_VAR): cv.string, vol.Optional(CONF_ADS_VAR_BRIGHTNESS): cv.string, vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string})

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: Optional[DiscoveryInfoType] = None) -> None:
    ads_hub: AdsHub = hass.data[DATA_ADS]
    ads_var_enable: str = config[CONF_ADS_VAR]
    ads_var_brightness: Optional[str] = config.get(CONF_ADS_VAR_BRIGHTNESS)
    name: str = config[CONF_NAME]
    add_entities([AdsLight(ads_hub, ads_var_enable, ads_var_brightness, name)])

class AdsLight(AdsEntity, LightEntity):
    """Representation of ADS light."""

    def __init__(self, ads_hub: AdsHub, ads_var_enable: str, ads_var_brightness: Optional[str], name: str) -> None:
        super().__init__(ads_hub, name, ads_var_enable)
        self._state_dict: Dict[str, Any] = {STATE_KEY_BRIGHTNESS: None}
        self._ads_var_brightness: Optional[str] = ads_var_brightness
        if ads_var_brightness is not None:
            self._attr_color_mode: ColorMode = ColorMode.BRIGHTNESS
            self._attr_supported_color_modes: set[ColorMode] = {ColorMode.BRIGHTNESS}
        else:
            self._attr_color_mode: ColorMode = ColorMode.ONOFF
            self._attr_supported_color_modes: set[ColorMode] = {ColorMode.ONOFF}

    async def async_added_to_hass(self) -> None:
        await self.async_initialize_device(self._ads_var, pyads.PLCTYPE_BOOL)
        if self._ads_var_brightness is not None:
            await self.async_initialize_device(self._ads_var_brightness, pyads.PLCTYPE_UINT, STATE_KEY_BRIGHTNESS)

    @property
    def brightness(self) -> int:
        return self._state_dict[STATE_KEY_BRIGHTNESS]

    @property
    def is_on(self) -> bool:
        return self._state_dict[STATE_KEY_STATE]

    def turn_on(self, **kwargs: Any) -> None:
        brightness: Optional[int] = kwargs.get(ATTR_BRIGHTNESS)
        self._ads_hub.write_by_name(self._ads_var, True, pyads.PLCTYPE_BOOL)
        if self._ads_var_brightness is not None and brightness is not None:
            self._ads_hub.write_by_name(self._ads_var_brightness, brightness, pyads.PLCTYPE_UINT)

    def turn_off(self, **kwargs: Any) -> None:
        self._ads_hub.write_by_name(self._ads_var, False, pyads.PLCTYPE_BOOL)

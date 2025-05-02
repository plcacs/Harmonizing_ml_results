"""Support for ADS covers."""
from __future__ import annotations
from typing import Any, Optional
import pyads
import voluptuous as vol
from homeassistant.components.cover import ATTR_POSITION, DEVICE_CLASSES_SCHEMA, PLATFORM_SCHEMA as COVER_PLATFORM_SCHEMA, CoverDeviceClass, CoverEntity, CoverEntityFeature
from homeassistant.const import CONF_DEVICE_CLASS, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from .const import CONF_ADS_VAR, DATA_ADS, STATE_KEY_STATE
from .entity import AdsEntity
from .hub import AdsHub
DEFAULT_NAME: str = 'ADS Cover'
CONF_ADS_VAR_SET_POS: str = 'adsvar_set_position'
CONF_ADS_VAR_OPEN: str = 'adsvar_open'
CONF_ADS_VAR_CLOSE: str = 'adsvar_close'
CONF_ADS_VAR_STOP: str = 'adsvar_stop'
CONF_ADS_VAR_POSITION: str = 'adsvar_position'
STATE_KEY_POSITION: str = 'position'
PLATFORM_SCHEMA = COVER_PLATFORM_SCHEMA.extend({vol.Required(CONF_ADS_VAR):
    cv.string, vol.Optional(CONF_ADS_VAR_POSITION): cv.string, vol.Optional
    (CONF_ADS_VAR_SET_POS): cv.string, vol.Optional(CONF_ADS_VAR_CLOSE): cv
    .string, vol.Optional(CONF_ADS_VAR_OPEN): cv.string, vol.Optional(
    CONF_ADS_VAR_STOP): cv.string, vol.Optional(CONF_NAME, default=
    DEFAULT_NAME): cv.string, vol.Optional(CONF_DEVICE_CLASS):
    DEVICE_CLASSES_SCHEMA})


def setup_platform(hass, config, add_entities, discovery_info=None):
    """Set up the cover platform for ADS."""
    ads_hub: AdsHub = hass.data[DATA_ADS]
    ads_var_is_closed: str = config[CONF_ADS_VAR]
    ads_var_position: Optional[str] = config.get(CONF_ADS_VAR_POSITION)
    ads_var_pos_set: Optional[str] = config.get(CONF_ADS_VAR_SET_POS)
    ads_var_open: Optional[str] = config.get(CONF_ADS_VAR_OPEN)
    ads_var_close: Optional[str] = config.get(CONF_ADS_VAR_CLOSE)
    ads_var_stop: Optional[str] = config.get(CONF_ADS_VAR_STOP)
    name: str = config[CONF_NAME]
    device_class: Optional[CoverDeviceClass] = config.get(CONF_DEVICE_CLASS)
    add_entities([AdsCover(ads_hub, ads_var_is_closed, ads_var_position,
        ads_var_pos_set, ads_var_open, ads_var_close, ads_var_stop, name,
        device_class)])


class AdsCover(AdsEntity, CoverEntity):
    """Representation of ADS cover."""
    _ads_var_position: Optional[str]
    _ads_var_pos_set: Optional[str]
    _ads_var_open: Optional[str]
    _ads_var_close: Optional[str]
    _ads_var_stop: Optional[str]

    def __init__(self, ads_hub, ads_var_is_closed, ads_var_position,
        ads_var_pos_set, ads_var_open, ads_var_close, ads_var_stop, name,
        device_class):
        """Initialize AdsCover entity."""
        super().__init__(ads_hub, name, ads_var_is_closed)
        if self._attr_unique_id is None:
            if ads_var_position is not None:
                self._attr_unique_id = ads_var_position
            elif ads_var_pos_set is not None:
                self._attr_unique_id = ads_var_pos_set
            elif ads_var_open is not None:
                self._attr_unique_id = ads_var_open
        self._state_dict[STATE_KEY_POSITION] = None
        self._ads_var_position = ads_var_position
        self._ads_var_pos_set = ads_var_pos_set
        self._ads_var_open = ads_var_open
        self._ads_var_close = ads_var_close
        self._ads_var_stop = ads_var_stop
        self._attr_device_class = device_class
        self._attr_supported_features: CoverEntityFeature = (
            CoverEntityFeature.OPEN | CoverEntityFeature.CLOSE)
        if ads_var_stop is not None:
            self._attr_supported_features |= CoverEntityFeature.STOP
        if ads_var_pos_set is not None:
            self._attr_supported_features |= CoverEntityFeature.SET_POSITION

    async def async_added_to_hass(self) ->None:
        """Register device notification."""
        if self._ads_var is not None:
            await self.async_initialize_device(self._ads_var, pyads.
                PLCTYPE_BOOL)
        if self._ads_var_position is not None:
            await self.async_initialize_device(self._ads_var_position,
                pyads.PLCTYPE_BYTE, STATE_KEY_POSITION)

    @property
    def is_closed(self):
        """Return if the cover is closed."""
        if self._ads_var is not None:
            return self._state_dict[STATE_KEY_STATE]
        if self._ads_var_position is not None:
            return self._state_dict[STATE_KEY_POSITION] == 0
        return None

    @property
    def current_cover_position(self):
        """Return current position of cover."""
        position = self._state_dict.get(STATE_KEY_POSITION)
        return position if isinstance(position, int) else 0

    def stop_cover(self, **kwargs: Any):
        """Fire the stop action."""
        if self._ads_var_stop:
            self._ads_hub.write_by_name(self._ads_var_stop, True, pyads.
                PLCTYPE_BOOL)

    def set_cover_position(self, **kwargs: Any):
        """Set cover position."""
        position: int = kwargs[ATTR_POSITION]
        if self._ads_var_pos_set is not None:
            self._ads_hub.write_by_name(self._ads_var_pos_set, position,
                pyads.PLCTYPE_BYTE)

    def open_cover(self, **kwargs: Any):
        """Move the cover up."""
        if self._ads_var_open is not None:
            self._ads_hub.write_by_name(self._ads_var_open, True, pyads.
                PLCTYPE_BOOL)
        elif self._ads_var_pos_set is not None:
            self.set_cover_position(position=100)

    def close_cover(self, **kwargs: Any):
        """Move the cover down."""
        if self._ads_var_close is not None:
            self._ads_hub.write_by_name(self._ads_var_close, True, pyads.
                PLCTYPE_BOOL)
        elif self._ads_var_pos_set is not None:
            self.set_cover_position(position=0)

    @property
    def available(self):
        """Return False if state has not been updated yet."""
        if self._ads_var is not None or self._ads_var_position is not None:
            return self._state_dict.get(STATE_KEY_STATE
                ) is not None or self._state_dict.get(STATE_KEY_POSITION
                ) is not None
        return True

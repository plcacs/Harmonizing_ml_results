"""Support for IHC lights."""
from __future__ import annotations
from typing import Any, List, Optional
from ihcsdk.ihccontroller import IHCController
from homeassistant.components.light import ATTR_BRIGHTNESS, ColorMode, LightEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from .const import CONF_DIMMABLE, CONF_OFF_ID, CONF_ON_ID, DOMAIN, IHC_CONTROLLER
from .entity import IHCEntity
from .util import async_pulse, async_set_bool, async_set_int

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the IHC lights platform."""
    if discovery_info is None:
        return
    devices: List[IhcLight] = []
    for name, device in discovery_info.items():
        ihc_id: str = device['ihc_id']
        product_cfg: dict = device['product_cfg']
        product: Any = device['product']
        controller_id: str = device['ctrl_id']
        ihc_controller: IHCController = hass.data[DOMAIN][controller_id][IHC_CONTROLLER]
        ihc_off_id: Optional[str] = product_cfg.get(CONF_OFF_ID)
        ihc_on_id: Optional[str] = product_cfg.get(CONF_ON_ID)
        dimmable: bool = product_cfg[CONF_DIMMABLE]
        light = IhcLight(
            ihc_controller,
            controller_id,
            name,
            ihc_id,
            ihc_off_id,
            ihc_on_id,
            dimmable,
            product
        )
        devices.append(light)
    add_entities(devices)

class IhcLight(IHCEntity, LightEntity):
    """Representation of a IHC light.

    For dimmable lights, the associated IHC resource should be a light
    level (integer). For non dimmable light the IHC resource should be
    an on/off (boolean) resource
    """

    _brightness: int
    _dimmable: bool
    _state: bool
    _ihc_off_id: Optional[str]
    _ihc_on_id: Optional[str]

    def __init__(
        self,
        ihc_controller: IHCController,
        controller_id: str,
        name: str,
        ihc_id: str,
        ihc_off_id: Optional[str],
        ihc_on_id: Optional[str],
        dimmable: bool = False,
        product: Any = None
    ) -> None:
        """Initialize the light."""
        super().__init__(ihc_controller, controller_id, name, ihc_id, product)
        self._ihc_off_id: Optional[str] = ihc_off_id
        self._ihc_on_id: Optional[str] = ihc_on_id
        self._brightness: int = 0
        self._dimmable: bool = dimmable
        self._state: bool = False
        if self._dimmable:
            self._attr_color_mode: ColorMode = ColorMode.BRIGHTNESS
        else:
            self._attr_color_mode: ColorMode = ColorMode.ONOFF
        self._attr_supported_color_modes: set[ColorMode] = {self._attr_color_mode}

    @property
    def brightness(self) -> int:
        """Return the brightness of this light between 0..255."""
        return self._brightness

    @property
    def is_on(self) -> bool:
        """Return true if light is on."""
        return self._state

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the light on."""
        if ATTR_BRIGHTNESS in kwargs:
            brightness: int = kwargs[ATTR_BRIGHTNESS]
        elif (brightness := self._brightness) == 0:
            brightness = 255
        if self._dimmable:
            await async_set_int(
                self.hass,
                self.ihc_controller,
                self.ihc_id,
                int(brightness * 100 / 255)
            )
        elif self._ihc_on_id:
            await async_pulse(self.hass, self.ihc_controller, self._ihc_on_id)
        else:
            await async_set_bool(self.hass, self.ihc_controller, self.ihc_id, True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the light off."""
        if self._dimmable:
            await async_set_int(self.hass, self.ihc_controller, self.ihc_id, 0)
        elif self._ihc_off_id:
            await async_pulse(self.hass, self.ihc_controller, self._ihc_off_id)
        else:
            await async_set_bool(self.hass, self.ihc_controller, self.ihc_id, False)

    def on_ihc_change(self, ihc_id: str, value: Any) -> None:
        """Handle IHC notifications."""
        if isinstance(value, bool):
            self._dimmable = False
            self._state = value != 0
        else:
            self._dimmable = True
            self._state = value > 0
            if self._state:
                self._brightness = int(value * 255 / 100)
        self.schedule_update_ha_state()

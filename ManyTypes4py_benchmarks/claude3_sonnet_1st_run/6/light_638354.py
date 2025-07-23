"""Support for TPLink lights."""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

from kasa import Device, DeviceType, KasaException, LightState, Module
from kasa.interfaces import LightEffect
from kasa.iot import IotDevice
import voluptuous as vol

from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_COLOR_TEMP_KELVIN, ATTR_EFFECT, ATTR_HS_COLOR, ATTR_TRANSITION, DOMAIN as LIGHT_DOMAIN, EFFECT_OFF, ColorMode, LightEntity, LightEntityDescription, LightEntityFeature, filter_supported_color_modes
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback, EntityPlatform
from homeassistant.helpers.typing import VolDictType

from . import TPLinkConfigEntry, legacy_device_id
from .const import DOMAIN
from .coordinator import TPLinkDataUpdateCoordinator
from .entity import CoordinatedTPLinkModuleEntity, TPLinkModuleEntityDescription, async_refresh_after

PARALLEL_UPDATES = 0
_LOGGER = logging.getLogger(__name__)

SERVICE_RANDOM_EFFECT: str = 'random_effect'
SERVICE_SEQUENCE_EFFECT: str = 'sequence_effect'

HUE = vol.Range(min=0, max=360)
SAT = vol.Range(min=0, max=100)
VAL = vol.Range(min=0, max=100)
TRANSITION = vol.Range(min=0, max=6000)
HSV_SEQUENCE = vol.ExactSequence((HUE, SAT, VAL))

BASE_EFFECT_DICT: dict[vol.Marker, Any] = {
    vol.Optional('brightness', default=100): vol.All(vol.Coerce(int), vol.Range(min=0, max=100)),
    vol.Optional('duration', default=0): vol.All(vol.Coerce(int), vol.Range(min=0, max=5000)),
    vol.Optional('transition', default=0): vol.All(vol.Coerce(int), TRANSITION),
    vol.Optional('segments', default=[0]): vol.All(cv.ensure_list_csv, vol.Length(min=1, max=80), [vol.All(vol.Coerce(int), vol.Range(min=0, max=80))])
}

SEQUENCE_EFFECT_DICT: dict[vol.Marker, Any] = {
    **BASE_EFFECT_DICT,
    vol.Required('sequence'): vol.All(cv.ensure_list, vol.Length(min=1, max=16), [vol.All(vol.Coerce(tuple), HSV_SEQUENCE)]),
    vol.Optional('repeat_times', default=0): vol.All(vol.Coerce(int), vol.Range(min=0, max=10)),
    vol.Optional('spread', default=1): vol.All(vol.Coerce(int), vol.Range(min=1, max=16)),
    vol.Optional('direction', default=4): vol.All(vol.Coerce(int), vol.Range(min=1, max=4))
}

RANDOM_EFFECT_DICT: dict[vol.Marker, Any] = {
    **BASE_EFFECT_DICT,
    vol.Optional('fadeoff', default=0): vol.All(vol.Coerce(int), vol.Range(min=0, max=3000)),
    vol.Optional('hue_range'): vol.All(cv.ensure_list_csv, [vol.Coerce(int)], vol.ExactSequence((HUE, HUE))),
    vol.Optional('saturation_range'): vol.All(cv.ensure_list_csv, [vol.Coerce(int)], vol.ExactSequence((SAT, SAT))),
    vol.Optional('brightness_range'): vol.All(cv.ensure_list_csv, [vol.Coerce(int)], vol.ExactSequence((VAL, VAL))),
    vol.Optional('transition_range'): vol.All(cv.ensure_list_csv, [vol.Coerce(int)], vol.ExactSequence((TRANSITION, TRANSITION))),
    vol.Required('init_states'): vol.All(cv.ensure_list_csv, [vol.Coerce(int)], HSV_SEQUENCE),
    vol.Optional('random_seed', default=100): vol.All(vol.Coerce(int), vol.Range(min=1, max=600)),
    vol.Optional('backgrounds'): vol.All(cv.ensure_list, vol.Length(min=1, max=16), [vol.All(vol.Coerce(tuple), HSV_SEQUENCE)])
}

@callback
def _async_build_base_effect(brightness: int, duration: int, transition: int, segments: List[int]) -> Dict[str, Any]:
    return {
        'custom': 1,
        'id': 'yMwcNpLxijmoKamskHCvvravpbnIqAIN',
        'brightness': brightness,
        'name': 'Custom',
        'segments': segments,
        'expansion_strategy': 1,
        'enable': 1,
        'duration': duration,
        'transition': transition
    }

def _get_backwards_compatible_light_unique_id(device: Device, entity_description: TPLinkLightEntityDescription) -> str:
    """Return unique ID for the entity."""
    if device.device_type is DeviceType.Dimmer and isinstance(device, IotDevice):
        return legacy_device_id(device)
    if device.parent or device.children:
        return legacy_device_id(device)
    return device.mac.replace(':', '').upper()

@dataclass(frozen=True, kw_only=True)
class TPLinkLightEntityDescription(LightEntityDescription, TPLinkModuleEntityDescription):
    """Base class for tplink light entity description."""
    unique_id_fn: Callable[[Device, TPLinkLightEntityDescription], str] = _get_backwards_compatible_light_unique_id

LIGHT_DESCRIPTIONS: Tuple[TPLinkLightEntityDescription, ...] = (
    TPLinkLightEntityDescription(
        key='light',
        exists_fn=lambda dev, _: Module.Light in dev.modules and Module.LightEffect not in dev.modules
    ),
)

LIGHT_EFFECT_DESCRIPTIONS: Tuple[TPLinkLightEntityDescription, ...] = (
    TPLinkLightEntityDescription(
        key='light_effect',
        exists_fn=lambda dev, _: Module.Light in dev.modules and Module.LightEffect in dev.modules
    ),
)

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    """Set up lights."""
    data: TPLinkConfigEntry = config_entry.runtime_data
    parent_coordinator: TPLinkDataUpdateCoordinator = data.parent_coordinator
    device: Device = parent_coordinator.device
    known_child_device_ids_light: Set[str] = set()
    known_child_device_ids_light_effect: Set[str] = set()
    first_check: bool = True

    def _check_device() -> None:
        entities: List[Union[TPLinkLightEntity, TPLinkLightEffectEntity]] = CoordinatedTPLinkModuleEntity.entities_for_device_and_its_children(
            hass=hass,
            device=device,
            coordinator=parent_coordinator,
            entity_class=TPLinkLightEntity,
            descriptions=LIGHT_DESCRIPTIONS,
            platform_domain=LIGHT_DOMAIN,
            known_child_device_ids=known_child_device_ids_light,
            first_check=first_check
        )
        entities.extend(
            CoordinatedTPLinkModuleEntity.entities_for_device_and_its_children(
                hass=hass,
                device=device,
                coordinator=parent_coordinator,
                entity_class=TPLinkLightEffectEntity,
                descriptions=LIGHT_EFFECT_DESCRIPTIONS,
                platform_domain=LIGHT_DOMAIN,
                known_child_device_ids=known_child_device_ids_light_effect,
                first_check=first_check
            )
        )
        async_add_entities(entities)

    _check_device()
    first_check = False
    config_entry.async_on_unload(parent_coordinator.async_add_listener(_check_device))

class TPLinkLightEntity(CoordinatedTPLinkModuleEntity, LightEntity):
    """Representation of a TPLink Smart Bulb."""
    _attr_supported_features: LightEntityFeature = LightEntityFeature.TRANSITION
    _fixed_color_mode: Optional[ColorMode] = None
    _light_module: Any
    platform: EntityPlatform

    def __init__(self, device: Device, coordinator: TPLinkDataUpdateCoordinator, description: TPLinkLightEntityDescription, *, parent: Optional[Device] = None) -> None:
        """Initialize the light."""
        super().__init__(device, coordinator, description, parent=parent)
        light_module = device.modules[Module.Light]
        self._light_module = light_module
        modes: Set[ColorMode] = {ColorMode.ONOFF}
        if (color_temp_feat := light_module.get_feature('color_temp')):
            modes.add(ColorMode.COLOR_TEMP)
            self._attr_min_color_temp_kelvin = color_temp_feat.minimum_value
            self._attr_max_color_temp_kelvin = color_temp_feat.maximum_value
        if light_module.has_feature('hsv'):
            modes.add(ColorMode.HS)
        if light_module.has_feature('brightness'):
            modes.add(ColorMode.BRIGHTNESS)
        self._attr_supported_color_modes = filter_supported_color_modes(modes)
        if len(self._attr_supported_color_modes) == 1:
            self._fixed_color_mode = next(iter(self._attr_supported_color_modes))

    @callback
    def _async_extract_brightness_transition(self, **kwargs: Any) -> Tuple[Optional[int], Optional[int]]:
        transition: Optional[int] = None
        brightness: Optional[int] = None
        if (transition_val := kwargs.get(ATTR_TRANSITION)) is not None:
            transition = int(transition_val * 1000)
        if (brightness_val := kwargs.get(ATTR_BRIGHTNESS)) is not None:
            brightness = round(brightness_val * 100.0 / 255.0)
        if self._device.device_type is DeviceType.Dimmer and transition is None:
            transition = 1
        return (brightness, transition)

    async def _async_set_hsv(self, hs_color: Tuple[float, float], brightness: Optional[int], transition: Optional[int]) -> None:
        hue, sat = tuple((int(val) for val in hs_color))
        await self._light_module.set_hsv(hue, sat, brightness, transition=transition)

    async def _async_set_color_temp(self, color_temp: int, brightness: Optional[int], transition: Optional[int]) -> None:
        light_module = self._light_module
        color_temp_feat = light_module.get_feature('color_temp')
        assert color_temp_feat
        requested_color_temp = round(color_temp)
        clamped_color_temp = min(color_temp_feat.maximum_value, max(color_temp_feat.minimum_value, requested_color_temp))
        await light_module.set_color_temp(clamped_color_temp, brightness=brightness, transition=transition)

    async def _async_turn_on_with_brightness(self, brightness: Optional[int], transition: Optional[int]) -> None:
        if brightness is not None:
            await self._light_module.set_brightness(brightness, transition=transition)
            return
        await self._light_module.set_state(LightState(light_on=True, transition=transition))

    @async_refresh_after
    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the light on."""
        brightness, transition = self._async_extract_brightness_transition(**kwargs)
        if ATTR_COLOR_TEMP_KELVIN in kwargs:
            await self._async_set_color_temp(kwargs[ATTR_COLOR_TEMP_KELVIN], brightness, transition)
        if ATTR_HS_COLOR in kwargs:
            await self._async_set_hsv(kwargs[ATTR_HS_COLOR], brightness, transition)
        else:
            await self._async_turn_on_with_brightness(brightness, transition)

    @async_refresh_after
    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the light off."""
        transition: Optional[int] = None
        if (transition_val := kwargs.get(ATTR_TRANSITION)) is not None:
            transition = int(transition_val * 1000)
        await self._light_module.set_state(LightState(light_on=False, transition=transition))

    def _determine_color_mode(self) -> ColorMode:
        """Return the active color mode."""
        if self._fixed_color_mode:
            return self._fixed_color_mode
        if self._light_module.has_feature('color_temp') and self._light_module.color_temp:
            return ColorMode.COLOR_TEMP
        return ColorMode.HS

    @callback
    def _async_update_attrs(self) -> bool:
        """Update the entity's attributes."""
        light_module = self._light_module
        self._attr_is_on = light_module.state.light_on is True
        if light_module.has_feature('brightness'):
            self._attr_brightness = round(light_module.brightness * 255.0 / 100.0)
        color_mode = self._determine_color_mode()
        self._attr_color_mode = color_mode
        if color_mode is ColorMode.COLOR_TEMP:
            self._attr_color_temp_kelvin = light_module.color_temp
        elif color_mode is ColorMode.HS:
            hue, saturation, _ = light_module.hsv
            self._attr_hs_color = (hue, saturation)
        return True

class TPLinkLightEffectEntity(TPLinkLightEntity):
    """Representation of a TPLink Smart Light Strip."""
    _attr_supported_features: LightEntityFeature = LightEntityFeature.TRANSITION | LightEntityFeature.EFFECT
    _effect_module: Any

    def __init__(self, device: Device, coordinator: TPLinkDataUpdateCoordinator, description: TPLinkLightEntityDescription, *, parent: Optional[Device] = None) -> None:
        """Initialize the light strip."""
        super().__init__(device, coordinator, description, parent=parent)
        self._effect_module = device.modules[Module.LightEffect]

    async def async_added_to_hass(self) -> None:
        """Call update attributes after the device is added to the platform."""
        await super().async_added_to_hass()
        self._register_effects_services()

    def _register_effects_services(self) -> None:
        if self._effect_module.has_custom_effects:
            self.platform.async_register_entity_service(SERVICE_RANDOM_EFFECT, RANDOM_EFFECT_DICT, 'async_set_random_effect')
            self.platform.async_register_entity_service(SERVICE_SEQUENCE_EFFECT, SEQUENCE_EFFECT_DICT, 'async_set_sequence_effect')

    @callback
    def _async_update_attrs(self) -> bool:
        """Update the entity's attributes."""
        super()._async_update_attrs()
        effect_module = self._effect_module
        if effect_module.effect != LightEffect.LIGHT_EFFECTS_OFF:
            self._attr_effect = effect_module.effect
            self._attr_color_mode = ColorMode.BRIGHTNESS
        else:
            self._attr_effect = EFFECT_OFF
        if (effect_list := effect_module.effect_list):
            self._attr_effect_list = effect_list
        else:
            self._attr_effect_list = None
        return True

    @async_refresh_after
    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the light on."""
        brightness, transition = self._async_extract_brightness_transition(**kwargs)
        effect_off_called: bool = False
        if (effect := kwargs.get(ATTR_EFFECT)):
            if effect in {LightEffect.LIGHT_EFFECTS_OFF, EFFECT_OFF}:
                if self._effect_module.effect is not LightEffect.LIGHT_EFFECTS_OFF:
                    await self._effect_module.set_effect(LightEffect.LIGHT_EFFECTS_OFF)
                    effect_off_called = True
                if len(kwargs) == 1:
                    return
            elif effect in self._effect_module.effect_list:
                await self._effect_module.set_effect(kwargs[ATTR_EFFECT], brightness=brightness, transition=transition)
                return
            else:
                _LOGGER.error('Invalid effect %s for %s', effect, self._device.host)
                return
        if ATTR_COLOR_TEMP_KELVIN in kwargs:
            if self.effect and self.effect != EFFECT_OFF and (not effect_off_called):
                await self._effect_module.set_effect(LightEffect.LIGHT_EFFECTS_OFF)
            await self._async_set_color_temp(kwargs[ATTR_COLOR_TEMP_KELVIN], brightness, transition)
        elif ATTR_HS_COLOR in kwargs:
            await self._async_set_hsv(kwargs[ATTR_HS_COLOR], brightness, transition)
        else:
            await self._async_turn_on_with_brightness(brightness, transition)

    async def async_set_random_effect(
        self, 
        brightness: int, 
        duration: int, 
        transition: int, 
        segments: List[int], 
        fadeoff: int, 
        init_states: Tuple[int, int, int], 
        random_seed: int, 
        backgrounds: Optional[List[Tuple[int, int, int]]] = None, 
        hue_range: Optional[Tuple[int, int]] = None, 
        saturation_range: Optional[Tuple[int, int]] = None, 
        brightness_range: Optional[Tuple[int, int]] = None, 
        transition_range: Optional[Tuple[int, int]] = None
    ) -> None:
        """Set a random effect."""
        effect: Dict[str, Any] = {
            **_async_build_base_effect(brightness, duration, transition, segments),
            'type': 'random',
            'init_states': [init_states],
            'random_seed': random_seed
        }
        if backgrounds:
            effect['backgrounds'] = backgrounds
        if fadeoff:
            effect['fadeoff'] = fadeoff
        if hue_range:
            effect['hue_range'] = hue_range
        if saturation_range:
            effect['saturation_range'] = saturation_range
        if brightness_range:
            effect['brightness_range'] = brightness_range
            effect['brightness'] = min(brightness_range[1], max(brightness, brightness_range[0]))
        if transition_range:
            effect['transition_range'] = transition_range
            effect['transition'] = 0
        try:
            await self._effect_module.set_custom_effect(effect)
        except KasaException as ex:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key='set_custom_effect',
                translation_placeholders={'effect': str(effect), 'exc': str(ex)}
            ) from ex

    async def async_set_sequence_effect(
        self, 
        brightness: int, 
        duration: int, 
        transition: int, 
        segments: List[int], 
        sequence: List[Tuple[int, int, int]], 
        repeat_times: int, 
        spread: int, 
        direction: int
    ) -> None:
        """Set a sequence effect."""
        effect: Dict[str, Any] = {
            **_async_build_base_effect(brightness, duration, transition, segments),
            'type': 'sequence',
            'sequence': sequence,
            'repeat_times': repeat_times,
            'spread': spread,
            'direction': direction
        }
        try:
            await self._effect_module.set_custom_effect(effect)
        except KasaException as ex:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key='set_custom_effect',
                translation_placeholders={'effect': str(effect), 'exc': str(ex)}
            ) from ex

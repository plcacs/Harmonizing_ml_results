"""Support for Hyperion-NG remotes."""
from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
import functools
import logging
from types import MappingProxyType
from typing import Any, cast
from hyperion import client, const
from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_EFFECT, ATTR_HS_COLOR, ColorMode, LightEntity, LightEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import color as color_util
from . import get_hyperion_device_id, get_hyperion_unique_id, listen_for_instance_updates
from .const import CONF_EFFECT_HIDE_LIST, CONF_INSTANCE_CLIENTS, CONF_PRIORITY, DEFAULT_ORIGIN, DEFAULT_PRIORITY, DOMAIN, HYPERION_MANUFACTURER_NAME, HYPERION_MODEL_NAME, SIGNAL_ENTITY_REMOVE, TYPE_HYPERION_LIGHT
_LOGGER = logging.getLogger(__name__)
CONF_DEFAULT_COLOR: str = 'default_color'
CONF_HDMI_PRIORITY: str = 'hdmi_priority'
CONF_EFFECT_LIST: str = 'effect_list'
KEY_EFFECT_SOLID: str = 'Solid'
DEFAULT_COLOR: list[int] = [255, 255, 255]
DEFAULT_BRIGHTNESS: int = 255
DEFAULT_EFFECT: str = KEY_EFFECT_SOLID
DEFAULT_NAME: str = 'Hyperion'
DEFAULT_PORT: int = const.DEFAULT_PORT_JSON
DEFAULT_HDMI_PRIORITY: int = 880
DEFAULT_EFFECT_LIST: list[str] = []
ICON_LIGHTBULB: str = 'mdi:lightbulb'
ICON_EFFECT: str = 'mdi:lava-lamp'

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    """Set up a Hyperion platform from config entry."""
    entry_data: dict[str, Any] = hass.data[DOMAIN][config_entry.entry_id]
    server_id: str | None = config_entry.unique_id

    @callback
    def instance_add(instance_num: int, instance_name: str) -> None:
        """Add entities for a new Hyperion instance."""
        assert server_id
        args = (server_id, instance_num, instance_name, config_entry.options, entry_data[CONF_INSTANCE_CLIENTS][instance_num])
        async_add_entities([HyperionLight(*args)])

    @callback
    def instance_remove(instance_num: int) -> None:
        """Remove entities for an old Hyperion instance."""
        assert server_id
        async_dispatcher_send(hass, SIGNAL_ENTITY_REMOVE.format(get_hyperion_unique_id(server_id, instance_num, TYPE_HYPERION_LIGHT)))
    listen_for_instance_updates(hass, config_entry, instance_add, instance_remove)

class HyperionLight(LightEntity):
    """A Hyperion light that acts as a client for the configured priority."""
    _attr_has_entity_name: bool = True
    _attr_name: None = None
    _attr_color_mode: ColorMode = ColorMode.HS
    _attr_should_poll: bool = False
    _attr_supported_color_modes: set[ColorMode] = {ColorMode.HS}
    _attr_supported_features: LightEntityFeature = LightEntityFeature.EFFECT

    def __init__(self, server_id: str, instance_num: int, instance_name: str, options: dict[str, Any], hyperion_client: client.HyperionClient) -> None:
        """Initialize the light."""
        self._attr_unique_id: str = self._compute_unique_id(server_id, instance_num)
        self._device_id: str = get_hyperion_device_id(server_id, instance_num)
        self._instance_name: str = instance_name
        self._options: dict[str, Any] = options
        self._client: client.HyperionClient = hyperion_client
        self._brightness: int = 255
        self._rgb_color: list[int] = DEFAULT_COLOR
        self._effect: str = KEY_EFFECT_SOLID
        self._static_effect_list: list[str] = [KEY_EFFECT_SOLID]
        self._effect_list: list[str] = self._static_effect_list[:]
        self._client_callbacks: dict[str, Callable[[Any], None]] = {f'{const.KEY_ADJUSTMENT}-{const.KEY_UPDATE}': self._update_adjustment, f'{const.KEY_COMPONENTS}-{const.KEY_UPDATE}': self._update_components, f'{const.KEY_EFFECTS}-{const.KEY_UPDATE}': self._update_effect_list, f'{const.KEY_PRIORITIES}-{const.KEY_UPDATE}': self._update_priorities, f'{const.KEY_CLIENT}-{const.KEY_UPDATE}': self._update_client}
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, self._device_id)}, manufacturer=HYPERION_MANUFACTURER_NAME, model=HYPERION_MODEL_NAME, name=self._instance_name, configuration_url=self._client.remote_url)

    def _compute_unique_id(self, server_id: str, instance_num: int) -> str:
        """Compute a unique id for this instance."""
        return get_hyperion_unique_id(server_id, instance_num, TYPE_HYPERION_LIGHT)

    @property
    def brightness(self) -> int:
        """Return the brightness of this light between 0..255."""
        return self._brightness

    @property
    def hs_color(self) -> tuple[float, float]:
        """Return last color value set."""
        return color_util.color_RGB_to_hs(*self._rgb_color)

    @property
    def icon(self) -> str:
        """Return state specific icon."""
        if self.is_on:
            if self.effect != KEY_EFFECT_SOLID:
                return ICON_EFFECT
        return ICON_LIGHTBULB

    @property
    def effect(self) -> str:
        """Return the current effect."""
        return self._effect

    @property
    def effect_list(self) -> list[str]:
        """Return the list of supported effects."""
        return self._effect_list

    @property
    def available(self) -> bool:
        """Return server availability."""
        return bool(self._client.has_loaded_state)

    def _get_option(self, key: str) -> Any:
        """Get a value from the provided options."""
        defaults: dict[str, Any] = {CONF_PRIORITY: DEFAULT_PRIORITY, CONF_EFFECT_HIDE_LIST: []}
        return self._options.get(key, defaults[key])

    @property
    def is_on(self) -> bool:
        """Return true if light is on. Light is considered on when there is a source at the configured HA priority."""
        return self._get_priority_entry_that_dictates_state() is not None

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on the light."""
        if ATTR_EFFECT not in kwargs and ATTR_HS_COLOR in kwargs:
            effect = KEY_EFFECT_SOLID
        else:
            effect = kwargs.get(ATTR_EFFECT, self._effect)
        if ATTR_HS_COLOR in kwargs:
            rgb_color = color_util.color_hs_to_RGB(*kwargs[ATTR_HS_COLOR])
        else:
            rgb_color = self._rgb_color
        if ATTR_BRIGHTNESS in kwargs:
            brightness = kwargs[ATTR_BRIGHTNESS]
            for item in self._client.adjustment or []:
                if const.KEY_ID in item and (not await self._client.async_send_set_adjustment(**{const.KEY_ADJUSTMENT: {const.KEY_BRIGHTNESS: int(round(float(brightness) * 100 / 255)), const.KEY_ID: item[const.KEY_ID]}})):
                    return
        if effect and effect != KEY_EFFECT_SOLID:
            if not await self._client.async_send_set_effect(**{const.KEY_PRIORITY: self._get_option(CONF_PRIORITY), const.KEY_EFFECT: {const.KEY_NAME: effect}, const.KEY_ORIGIN: DEFAULT_ORIGIN}):
                return
        elif not await self._client.async_send_set_color(**{const.KEY_PRIORITY: self._get_option(CONF_PRIORITY), const.KEY_COLOR: rgb_color, const.KEY_ORIGIN: DEFAULT_ORIGIN}):
            return

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off the light i.e. clear the configured priority."""
        if not await self._client.async_send_clear(**{const.KEY_PRIORITY: self._get_option(CONF_PRIORITY)}):
            return

    def _set_internal_state(self, brightness: int | None = None, rgb_color: list[int] | None = None, effect: str | None = None) -> None:
        """Set the internal state."""
        if brightness is not None:
            self._brightness = brightness
        if rgb_color is not None:
            self._rgb_color = rgb_color
        if effect is not None:
            self._effect = effect

    @callback
    def _update_components(self, _: Any = None) -> None:
        """Update Hyperion components."""
        self.async_write_ha_state()

    @callback
    def _update_adjustment(self, _: Any = None) -> None:
        """Update Hyperion adjustments."""
        if self._client.adjustment:
            brightness_pct = self._client.adjustment[0].get(const.KEY_BRIGHTNESS, DEFAULT_BRIGHTNESS)
            if brightness_pct < 0 or brightness_pct > 100:
                return
            self._set_internal_state(brightness=int(round(brightness_pct * 255 / float(100))))
            self.async_write_ha_state()

    @callback
    def _update_priorities(self, _: Any = None) -> None:
        """Update Hyperion priorities."""
        priority = self._get_priority_entry_that_dictates_state()
        if priority:
            component_id = priority.get(const.KEY_COMPONENTID)
            if component_id == const.KEY_COMPONENTID_EFFECT:
                self._set_internal_state(rgb_color=DEFAULT_COLOR, effect=priority[const.KEY_OWNER])
            elif component_id == const.KEY_COMPONENTID_COLOR:
                self._set_internal_state(rgb_color=priority[const.KEY_VALUE][const.KEY_RGB], effect=KEY_EFFECT_SOLID)
        self.async_write_ha_state()

    @callback
    def _update_effect_list(self, _: Any = None) -> None:
        """Update Hyperion effects."""
        if not self._client.effects:
            return
        effect_list: list[str] = []
        hide_effects: list[str] = self._get_option(CONF_EFFECT_HIDE_LIST)
        for effect in self._client.effects or []:
            if const.KEY_NAME in effect:
                effect_name = effect[const.KEY_NAME]
                if effect_name not in hide_effects:
                    effect_list.append(effect_name)
        self._effect_list = [effect for effect in self._static_effect_list if effect not in hide_effects] + effect_list
        self.async_write_ha_state()

    @callback
    def _update_full_state(self) -> None:
        """Update full Hyperion state."""
        self._update_adjustment()
        self._update_priorities()
        self._update_effect_list()
        _LOGGER.debug('Hyperion full state update: On=%s,Brightness=%i,Effect=%s (%i effects total),Color=%s', self.is_on, self._brightness, self._effect, len(self._effect_list), self._rgb_color)

    @callback
    def _update_client(self, _: Any = None) -> None:
        """Update client connection state."""
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Register callbacks when entity added to hass."""
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_ENTITY_REMOVE.format(self.unique_id), functools.partial(self.async_remove, force_remove=True)))
        self._client.add_callbacks(self._client_callbacks)
        self._update_full_state()

    async def async_will_remove_from_hass(self) -> None:
        """Cleanup prior to hass removal."""
        self._client.remove_callbacks(self._client_callbacks)

    def _get_priority_entry_that_dictates_state(self) -> dict[str, Any] | None:
        """Get the relevant Hyperion priority entry to consider."""
        for priority in self._client.priorities or []:
            if priority.get(const.KEY_PRIORITY) == self._get_option(CONF_PRIORITY):
                return priority
        return None

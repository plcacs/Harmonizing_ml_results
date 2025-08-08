from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
import functools
import logging
from types import MappingProxyType
from typing import Any

from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_EFFECT, ATTR_HS_COLOR, ColorMode, LightEntity, LightEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import color as color_util

from . import get_hyperion_device_id, get_hyperion_unique_id, listen_for_instance_updates
from .const import CONF_EFFECT_HIDE_LIST, CONF_INSTANCE_CLIENTS, CONF_PRIORITY, DEFAULT_ORIGIN, DEFAULT_PRIORITY, DOMAIN, HYPERION_MANUFACTURER_NAME, HYPERION_MODEL_NAME, SIGNAL_ENTITY_REMOVE, TYPE_HYPERION_LIGHT

_LOGGER: logging.Logger

CONF_DEFAULT_COLOR: str
CONF_HDMI_PRIORITY: str
CONF_EFFECT_LIST: str

KEY_EFFECT_SOLID: str
DEFAULT_COLOR: list[int]
DEFAULT_BRIGHTNESS: int
DEFAULT_EFFECT: str
DEFAULT_NAME: str
DEFAULT_PORT: int
DEFAULT_HDMI_PRIORITY: int
DEFAULT_EFFECT_LIST: list

ICON_LIGHTBULB: str
ICON_EFFECT: str

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class HyperionLight(LightEntity):

    _attr_has_entity_name: bool
    _attr_name: str
    _attr_color_mode: ColorMode
    _attr_should_poll: bool
    _attr_supported_color_modes: set[ColorMode]
    _attr_supported_features: LightEntityFeature

    def __init__(self, server_id: str, instance_num: int, instance_name: str, options: Mapping[str, Any], hyperion_client: client.HyperionClient) -> None:

    def _compute_unique_id(self, server_id: str, instance_num: int) -> str:

    @property
    def brightness(self) -> int:

    @property
    def hs_color(self) -> tuple[float, float]:

    @property
    def icon(self) -> str:

    @property
    def effect(self) -> str:

    @property
    def effect_list(self) -> list[str]:

    @property
    def available(self) -> bool:

    def _get_option(self, key: str) -> Any:

    @property
    def is_on(self) -> bool:

    async def async_turn_on(self, **kwargs: Any) -> None:

    async def async_turn_off(self, **kwargs: Any) -> None:

    def _set_internal_state(self, brightness: int = None, rgb_color: list[int] = None, effect: str = None) -> None:

    @callback
    def _update_components(self, _=None) -> None:

    @callback
    def _update_adjustment(self, _=None) -> None:

    @callback
    def _update_priorities(self, _=None) -> None:

    @callback
    def _update_effect_list(self, _=None) -> None:

    @callback
    def _update_full_state(self) -> None:

    @callback
    def _update_client(self, _=None) -> None:

    async def async_added_to_hass(self) -> None:

    async def async_will_remove_from_hass(self) -> None:

    def _get_priority_entry_that_dictates_state(self) -> Mapping[str, Any]:

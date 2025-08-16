from __future__ import annotations
import logging
from typing import Any, List, Optional
import voluptuous as vol
from homeassistant.components.fan import ATTR_DIRECTION, ATTR_OSCILLATING, ATTR_PERCENTAGE, ATTR_PRESET_MODE, DIRECTION_FORWARD, DIRECTION_REVERSE, ENTITY_ID_FORMAT, FanEntity, FanEntityFeature
from homeassistant.const import CONF_ENTITY_ID, CONF_FRIENDLY_NAME, CONF_UNIQUE_ID, CONF_VALUE_TEMPLATE, STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity import async_generate_entity_id
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.script import Script
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from .const import DOMAIN
from .template_entity import TEMPLATE_ENTITY_AVAILABILITY_SCHEMA_LEGACY, TemplateEntity, rewrite_common_legacy_to_modern_conf

_LOGGER: logging.Logger

CONF_FANS: str
CONF_SPEED_COUNT: str
CONF_PRESET_MODES: str
CONF_PERCENTAGE_TEMPLATE: str
CONF_PRESET_MODE_TEMPLATE: str
CONF_OSCILLATING_TEMPLATE: str
CONF_DIRECTION_TEMPLATE: str
CONF_ON_ACTION: str
CONF_OFF_ACTION: str
CONF_SET_PERCENTAGE_ACTION: str
CONF_SET_OSCILLATING_ACTION: str
CONF_SET_DIRECTION_ACTION: str
CONF_SET_PRESET_MODE_ACTION: str
_VALID_DIRECTIONS: List[str]

FAN_SCHEMA: vol.Schema
PLATFORM_SCHEMA: vol.Schema

async def _async_create_entities(hass: HomeAssistant, config: ConfigType) -> List[FanEntity]:
    ...

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class TemplateFan(TemplateEntity, FanEntity):
    _attr_should_poll: bool

    def __init__(self, hass: HomeAssistant, object_id: str, config: ConfigType, unique_id: Optional[str]) -> None:
        ...

    @property
    def speed_count(self) -> int:
        ...

    @property
    def preset_modes(self) -> List[str]:
        ...

    @property
    def is_on(self) -> bool:
        ...

    @property
    def preset_mode(self) -> Optional[str]:
        ...

    @property
    def percentage(self) -> Optional[int]:
        ...

    @property
    def oscillating(self) -> Optional[bool]:
        ...

    @property
    def current_direction(self) -> Optional[str]:
        ...

    async def async_turn_on(self, percentage: Optional[int] = None, preset_mode: Optional[str] = None, **kwargs: Any) -> None:
        ...

    async def async_turn_off(self, **kwargs: Any) -> None:
        ...

    async def async_set_percentage(self, percentage: int) -> None:
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        ...

    async def async_oscillate(self, oscillating: bool) -> None:
        ...

    async def async_set_direction(self, direction: str) -> None:
        ...

    @callback
    def _update_state(self, result: Any) -> None:
        ...

    @callback
    def _async_setup_templates(self) -> None:
        ...

    @callback
    def _update_percentage(self, percentage: Any) -> None:
        ...

    @callback
    def _update_preset_mode(self, preset_mode: Any) -> None:
        ...

    @callback
    def _update_oscillating(self, oscillating: Any) -> None:
        ...

    @callback
    def _update_direction(self, direction: Any) -> None:
        ...

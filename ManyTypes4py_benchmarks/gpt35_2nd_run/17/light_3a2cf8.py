from __future__ import annotations
import logging
from typing import Any

from homeassistant.components.light import ATTR_BRIGHTNESS, ColorMode, LightEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import HomeworksData
from .const import CONF_ADDR, CONF_CONTROLLER_ID, CONF_DIMMERS, CONF_RATE, DOMAIN
from .entity import HomeworksEntity

_LOGGER: logging.Logger

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class HomeworksLight(HomeworksEntity, LightEntity):

    def __init__(self, controller: Homeworks, controller_id: str, addr: int, name: str, rate: int) -> None:

    async def async_added_to_hass(self) -> None:

    def turn_on(self, **kwargs: Any) -> None:

    def turn_off(self, **kwargs: Any) -> None:

    @property
    def brightness(self) -> int:

    def _set_brightness(self, level: int) -> None:

    @property
    def is_on(self) -> bool:

    @callback
    def _update_callback(self, msg_type: int, values: list) -> None:

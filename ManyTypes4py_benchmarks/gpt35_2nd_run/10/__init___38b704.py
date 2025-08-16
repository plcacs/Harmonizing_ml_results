from __future__ import annotations
import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Union
import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_ID, CONF_NAME, CONF_PASSWORD, CONF_PORT, CONF_USERNAME, EVENT_HOMEASSISTANT_STOP, Platform
from homeassistant.core import Event, HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import ConfigEntryNotReady, ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.dispatcher import async_dispatcher_connect, dispatcher_send
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import slugify
from .const import CONF_ADDR, CONF_CONTROLLER_ID, CONF_KEYPADS, DOMAIN

@dataclass
class HomeworksData:
    controller: Any
    controller_id: str
    keypads: Dict[str, HomeworksKeypad]

async def async_send_command(hass: HomeAssistant, data: Dict[str, Union[str, List[str]]]) -> None:
    def get_controller_ids() -> List[str]: ...
    def get_homeworks_data(controller_id: str) -> Optional[HomeworksData]: ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool: ...

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool: ...

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool: ...

async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None: ...

class HomeworksKeypad:
    def __init__(self, hass: HomeAssistant, controller: Any, controller_id: str, addr: str, name: str) -> None: ...
    def _update_callback(self, msg_type: str, values: List[Any]) -> None: ...
    def _request_keypad_led_states(self) -> None: ...
    async def request_keypad_led_states(self) -> None: ...

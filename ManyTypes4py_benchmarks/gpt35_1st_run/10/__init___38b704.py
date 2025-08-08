from __future__ import annotations
import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Union
from pyhomeworks import exceptions as hw_exceptions
from pyhomeworks.pyhomeworks import HW_BUTTON_PRESSED, HW_BUTTON_RELEASED, HW_LOGIN_INCORRECT, Homeworks
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

_LOGGER: logging.Logger
PLATFORMS: List[Platform]
CONF_COMMAND: str
EVENT_BUTTON_PRESS: str
EVENT_BUTTON_RELEASE: str
KEYPAD_LEDSTATE_POLL_COOLDOWN: float
CONFIG_SCHEMA: Dict[str, Any]
SERVICE_SEND_COMMAND_SCHEMA: vol.Schema
@dataclass
class HomeworksData:
    """Container for config entry data."""
@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Set up services for Lutron Homeworks Series 4 and 8 integration."""
async def async_send_command(hass: HomeAssistant, data: Dict[str, Any]) -> None:
    """Send command to a controller."""
async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Start Homeworks controller."""
async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Homeworks from a config entry."""
async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
class HomeworksKeypad:
    """When you want signals instead of entities."""

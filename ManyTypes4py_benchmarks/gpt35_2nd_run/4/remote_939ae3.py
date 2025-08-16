from __future__ import annotations
import asyncio
from datetime import timedelta
import logging
import time
from typing import Any, Dict, List, Optional
from miio import ChuangmiIr, DeviceException
import voluptuous as vol
from homeassistant.components import persistent_notification
from homeassistant.components.remote import ATTR_DELAY_SECS, ATTR_NUM_REPEATS, DEFAULT_DELAY_SECS, PLATFORM_SCHEMA as REMOTE_PLATFORM_SCHEMA, RemoteEntity
from homeassistant.const import CONF_COMMAND, CONF_HOST, CONF_NAME, CONF_TIMEOUT, CONF_TOKEN
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers import config_validation as cv, entity_platform
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util.dt import utcnow
from .const import SERVICE_LEARN, SERVICE_SET_REMOTE_LED_OFF, SERVICE_SET_REMOTE_LED_ON

_LOGGER: logging.Logger
DATA_KEY: str
CONF_SLOT: str
CONF_COMMANDS: str
DEFAULT_TIMEOUT: int
DEFAULT_SLOT: int
COMMAND_SCHEMA: vol.Schema
PLATFORM_SCHEMA: vol.Schema

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

async def async_service_led_off_handler(entity: XiaomiMiioRemote, service: Dict[str, Any]) -> None:

async def async_service_led_on_handler(entity: XiaomiMiioRemote, service: Dict[str, Any]) -> None:

async def async_service_learn_handler(entity: XiaomiMiioRemote, service: Dict[str, Any]) -> None:

class XiaomiMiioRemote(RemoteEntity):

    def __init__(self, friendly_name: str, device: ChuangmiIr, unique_id: str, slot: int, timeout: int, commands: Dict[str, Any]) -> None:

    @property
    def unique_id(self) -> str:

    @property
    def name(self) -> str:

    @property
    def device(self) -> ChuangmiIr:

    @property
    def slot(self) -> int:

    @property
    def timeout(self) -> int:

    @property
    def is_on(self) -> bool:

    async def async_turn_on(self, **kwargs: Any) -> None:

    async def async_turn_off(self, **kwargs: Any) -> None:

    def _send_command(self, payload: str) -> None:

    def send_command(self, command: str, **kwargs: Any) -> None:

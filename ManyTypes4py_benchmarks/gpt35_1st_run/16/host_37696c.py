from __future__ import annotations
import asyncio
from collections import defaultdict
from collections.abc import Mapping
import logging
from time import time
from typing import Any, Literal
import aiohttp
from aiohttp.web import Request
from reolink_aio.api import ALLOWED_SPECIAL_CHARS, Host
from reolink_aio.enums import SubType
from reolink_aio.exceptions import NotSupportedError, ReolinkError, SubscriptionError
from homeassistant.components import webhook
from homeassistant.const import CONF_HOST, CONF_PASSWORD, CONF_PORT, CONF_PROTOCOL, CONF_USERNAME
from homeassistant.core import CALLBACK_TYPE, HassJob, HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import format_mac
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.network import NoURLAvailableError, get_url
from homeassistant.helpers.storage import Store
from homeassistant.util.ssl import SSLCipherList
from .const import CONF_SUPPORTS_PRIVACY_MODE, CONF_USE_HTTPS, DOMAIN
from .exceptions import PasswordIncompatible, ReolinkSetupException, ReolinkWebhookException, UserNotAdmin
from .util import get_store

DEFAULT_TIMEOUT: int = 30
FIRST_TCP_PUSH_TIMEOUT: int = 10
FIRST_ONVIF_TIMEOUT: int = 10
FIRST_ONVIF_LONG_POLL_TIMEOUT: int = 90
SUBSCRIPTION_RENEW_THRESHOLD: int = 300
POLL_INTERVAL_NO_PUSH: int = 5
LONG_POLL_COOLDOWN: float = 0.75
LONG_POLL_ERROR_COOLDOWN: int = 30
BATTERY_WAKE_UPDATE_INTERVAL: int = 3600
_LOGGER: logging.Logger = logging.getLogger(__name__)

class ReolinkHost:
    def __init__(self, hass: HomeAssistant, config: Mapping[str, Any], options: Mapping[str, Any], config_entry_id: str = None) -> None:
        ...

    @callback
    def async_register_update_cmd(self, cmd: str, channel: Any = None) -> None:
        ...

    @callback
    def async_unregister_update_cmd(self, cmd: str, channel: Any = None) -> None:
        ...

    @property
    def unique_id(self) -> str:
        ...

    @property
    def api(self) -> Host:
        ...

    async def async_init(self) -> None:
        ...

    async def _async_check_tcp_push(self, *args: Any) -> None:
        ...

    async def _async_check_onvif(self, *args: Any) -> None:
        ...

    async def _async_check_onvif_long_poll(self, *args: Any) -> None:
        ...

    async def update_states(self) -> None:
        ...

    async def disconnect(self) -> None:
        ...

    async def _async_start_long_polling(self, initial: bool = False) -> None:
        ...

    async def _async_stop_long_polling(self) -> None:
        ...

    async def stop(self, *args: Any) -> None:
        ...

    async def subscribe(self) -> None:
        ...

    async def renew(self) -> None:
        ...

    async def _renew(self, sub_type: SubType) -> None:
        ...

    def register_webhook(self) -> None:
        ...

    def unregister_webhook(self) -> None:
        ...

    async def _async_long_polling(self, *args: Any) -> None:
        ...

    async def _async_poll_all_motion(self, *args: Any) -> None:
        ...

    async def handle_webhook(self, hass: HomeAssistant, webhook_id: str, request: Request) -> None:
        ...

    async def _process_webhook_data(self, hass: HomeAssistant, webhook_id: str, data: bytes) -> None:
        ...

    def _signal_write_ha_state(self, channels: Any = None) -> None:
        ...

    @property
    def event_connection(self) -> Literal['TCP push', 'ONVIF push', 'ONVIF long polling', 'Fast polling']:
        ...

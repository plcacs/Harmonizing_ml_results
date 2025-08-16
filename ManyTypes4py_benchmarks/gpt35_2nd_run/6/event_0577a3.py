from __future__ import annotations
from typing import List, Set, Optional, Callable
import asyncio
import datetime as dt
from aiohttp.web import Request
from onvif import ONVIFCamera
from onvif.client import NotificationManager, PullPointManager as ONVIFPullPointManager, retry_connection_error
from onvif.exceptions import ONVIFError
from onvif.util import stringify_onvif_error
from zeep.exceptions import Fault, ValidationError, XMLParseError
from homeassistant.components import webhook
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HassJob, HomeAssistant, callback
from homeassistant.helpers.device_registry import format_mac
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.network import NoURLAvailableError, get_url
from .const import DOMAIN, LOGGER
from .models import Event, PullPointManagerState, WebHookManagerState
from .parsers import PARSERS

UNHANDLED_TOPICS: Set[str] = {'tns1:MediaControl/VideoEncoderConfiguration'}
SUBSCRIPTION_ERRORS: Tuple[type, ...] = (Fault, TimeoutError, TransportError)
CREATE_ERRORS: Tuple[type, ...] = (ONVIFError, Fault, RequestError, XMLParseError, ValidationError)
SET_SYNCHRONIZATION_POINT_ERRORS: Tuple[type, ...] = (*SUBSCRIPTION_ERRORS, TypeError)
UNSUBSCRIBE_ERRORS: Tuple[type, ...] = (XMLParseError, *SUBSCRIPTION_ERRORS)
RENEW_ERRORS: Tuple[type, ...] = (ONVIFError, RequestError, XMLParseError, *SUBSCRIPTION_ERRORS)

SUBSCRIPTION_TIME: dt.timedelta = dt.timedelta(minutes=10)
SUBSCRIPTION_RENEW_INTERVAL: int = 8 * 60
SUBSCRIPTION_ATTEMPTS: int = 2
SUBSCRIPTION_RESTART_INTERVAL_ON_ERROR: int = 60
PULLPOINT_POLL_TIME: dt.timedelta = dt.timedelta(seconds=60)
PULLPOINT_MESSAGE_LIMIT: int = 100
PULLPOINT_COOLDOWN_TIME: float = 0.75

class EventManager:
    def __init__(self, hass: HomeAssistant, device: ONVIFCamera, config_entry: ConfigEntry, name: str) -> None:
        ...

    @property
    def started(self) -> bool:
        ...

    @callback
    def async_add_listener(self, update_callback: Callable) -> Callable:
        ...

    @callback
    def async_remove_listener(self, update_callback: Callable) -> None:
        ...

    async def async_start(self, try_pullpoint: bool, try_webhook: bool) -> bool:
        ...

    async def async_stop(self) -> None:
        ...

    @callback
    def async_callback_listeners(self) -> None:
        ...

    async def async_parse_messages(self, messages: List) -> None:
        ...

    def get_uid(self, uid: str) -> Optional[Event]:
        ...

    def get_platform(self, platform: str) -> List[Event]:
        ...

    def get_uids_by_platform(self, platform: str) -> Set[str]:
        ...

    @callback
    def async_webhook_failed(self) -> None:
        ...

    @callback
    def async_webhook_working(self) -> None:
        ...

    @callback
    def async_mark_events_stale(self) -> None:
        ...

class PullPointManager:
    def __init__(self, event_manager: EventManager) -> None:
        ...

    async def async_start(self) -> bool:
        ...

    @callback
    def async_pause(self) -> None:
        ...

    @callback
    def async_resume(self) -> None:
        ...

    async def async_stop(self) -> None:
        ...

    async def _async_start_pullpoint(self) -> bool:
        ...

    async def _async_cancel_and_unsubscribe(self) -> None:
        ...

    @retry_connection_error(SUBSCRIPTION_ATTEMPTS)
    async def _async_create_pullpoint_subscription(self) -> None:
        ...

    async def _async_unsubscribe_pullpoint(self) -> None:
        ...

    async def _async_pull_messages(self) -> None:
        ...

    @callback
    def async_cancel_pull_messages(self) -> None:
        ...

    @callback
    def async_schedule_pull_messages(self, delay: Optional[float] = None) -> None:
        ...

    @callback
    def _async_background_pull_messages_or_reschedule(self, _now=None) -> None:
        ...

class WebHookManager:
    def __init__(self, event_manager: EventManager) -> None:
        ...

    async def async_start(self) -> bool:
        ...

    async def async_stop(self) -> None:
        ...

    @retry_connection_error(SUBSCRIPTION_ATTEMPTS)
    async def _async_create_webhook_subscription(self) -> None:
        ...

    async def _async_start_webhook(self) -> bool:
        ...

    @callback
    def _async_register_webhook(self) -> None:
        ...

    @callback
    def _async_unregister_webhook(self) -> None:
        ...

    async def _async_handle_webhook(self, hass: HomeAssistant, webhook_id: str, request: Request) -> None:
        ...

    async def _async_process_webhook(self, hass: HomeAssistant, webhook_id: str, content: bytes) -> None:
        ...

    async def _async_unsubscribe_webhook(self) -> None:
        ...

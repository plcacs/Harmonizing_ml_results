"""ONVIF event abstraction."""

from __future__ import annotations
import asyncio
from collections.abc import Callable, Iterable
import datetime as dt
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)
from aiohttp.web import Request
from homeassistant.components import webhook
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HassJob, HomeAssistant, callback
from homeassistant.helpers.device_registry import format_mac
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.network import NoURLAvailableError
from onvif.client import NotificationManager, PullPointManager as ONVIFPullPointManager
from onvif.exceptions import ONVIFError
from onvif.util import stringify_onvif_error
from zeep.exceptions import Fault, XMLParseError

UNHANDLED_TOPICS: Set[str] = ...
SUBSCRIPTION_ERRORS: Tuple[Type[Exception], ...] = ...
CREATE_ERRORS: Tuple[Type[Exception], ...] = ...
SET_SYNCHRONIZATION_POINT_ERRORS: Tuple[Type[Exception], ...] = ...
UNSUBSCRIBE_ERRORS: Tuple[Type[Exception], ...] = ...
RENEW_ERRORS: Tuple[Type[Exception], ...] = ...
SUBSCRIPTION_TIME: dt.timedelta = ...
SUBSCRIPTION_RENEW_INTERVAL: int = ...
SUBSCRIPTION_ATTEMPTS: int = ...
SUBSCRIPTION_RESTART_INTERVAL_ON_ERROR: int = ...
PULLPOINT_POLL_TIME: dt.timedelta = ...
PULLPOINT_MESSAGE_LIMIT: int = ...
PULLPOINT_COOLDOWN_TIME: float = ...

class EventManager:
    """ONVIF Event Manager."""

    def __init__(self, hass: HomeAssistant, device: Any, config_entry: ConfigEntry, name: str) -> None:
        ...

    @property
    def started(self) -> bool:
        ...

    @callback
    def async_add_listener(self, update_callback: CALLBACK_TYPE) -> Callable[[], None]:
        ...

    @callback
    def async_remove_listener(self, update_callback: CALLBACK_TYPE) -> None:
        ...

    async def async_start(self, try_pullpoint: bool, try_webhook: bool) -> bool:
        ...

    async def async_stop(self) -> None:
        ...

    @callback
    def async_callback_listeners(self) -> None:
        ...

    async def async_parse_messages(self, messages: Iterable[Any]) -> None:
        ...

    def get_uid(self, uid: str) -> Optional[Any]:
        ...

    def get_platform(self, platform: str) -> List[Any]:
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
    """ONVIF PullPoint Manager."""

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
    def _async_background_pull_messages_or_reschedule(self, _now: Optional[Any] = None) -> None:
        ...

class WebHookManager:
    """Manage ONVIF webhook subscriptions."""

    def __init__(self, event_manager: EventManager) -> None:
        ...

    async def async_start(self) -> bool:
        ...

    async def async_stop(self) -> None:
        ...

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

    async def _async_process_webhook(self, hass: HomeAssistant, webhook_id: str, content: Optional[bytes]) -> None:
        ...

    async def _async_unsubscribe_webhook(self) -> None:
        ...
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from typing import Any

import aiohttp
from amcrest import ApiWrapper
import voluptuous as vol

from homeassistant.auth.models import User
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.helpers.typing import ConfigType

_LOGGER: logging.Logger
CONF_RESOLUTION: str
CONF_STREAM_SOURCE: str
CONF_FFMPEG_ARGUMENTS: str
CONF_CONTROL_LIGHT: str
DEFAULT_NAME: str
DEFAULT_PORT: int
DEFAULT_RESOLUTION: str
DEFAULT_ARGUMENTS: str
MAX_ERRORS: int
RECHECK_INTERVAL: timedelta
NOTIFICATION_ID: str
NOTIFICATION_TITLE: str
SCAN_INTERVAL: timedelta
AUTHENTICATION_LIST: dict[str, str]

import logging

def _has_unique_names(devices: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

AMCREST_SCHEMA: vol.Schema
CONFIG_SCHEMA: vol.Schema

class AmcrestChecker(ApiWrapper):
    """amcrest.ApiWrapper wrapper for catching errors."""

    _hass: HomeAssistant
    _wrap_name: str
    _wrap_errors: int
    _wrap_lock: threading.Lock
    _async_wrap_lock: asyncio.Lock
    _wrap_login_err: bool
    _wrap_event_flag: threading.Event
    _async_wrap_event_flag: asyncio.Event
    _unsub_recheck: Callable[[], None] | None

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        host: str,
        port: int,
        user: str,
        password: str,
    ) -> None: ...

    @property
    def available(self) -> bool: ...

    @property
    def available_flag(self) -> threading.Event: ...

    @property
    def async_available_flag(self) -> asyncio.Event: ...

    @callback
    def _async_start_recovery(self) -> None: ...

    def command(self, *args: Any, **kwargs: Any) -> Any: ...

    async def async_command(self, *args: Any, **kwargs: Any) -> Any: ...

    @asynccontextmanager
    async def async_stream_command(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Any]: ...

    @asynccontextmanager
    async def _async_command_wrapper(self) -> AsyncIterator[None]: ...

    def _handle_offline_thread_safe(self, ex: Exception) -> bool: ...
    def _handle_offline(self, ex: Exception) -> None: ...

    @callback
    def _async_handle_offline(self, ex: Exception) -> None: ...

    def _handle_error_thread_safe(self) -> bool: ...
    def _handle_error(self) -> None: ...

    @callback
    def _async_handle_error(self) -> None: ...

    def _set_online_thread_safe(self) -> bool: ...
    def _set_online(self) -> None: ...

    @callback
    def _async_set_online(self) -> None: ...

    @callback
    def _async_signal_online(self) -> None: ...

    async def _wrap_test_online(self, now: datetime) -> None: ...

def _monitor_events(
    hass: HomeAssistant, name: str, api: AmcrestChecker, event_codes: set[str]
) -> None: ...

def _start_event_monitor(
    hass: HomeAssistant, name: str, api: AmcrestChecker, event_codes: set[str]
) -> None: ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool: ...

@dataclass
class AmcrestDevice:
    """Representation of a base Amcrest discovery device."""

    api: AmcrestChecker
    authentication: aiohttp.BasicAuth | None
    ffmpeg_arguments: str
    stream_source: str
    resolution: str
    control_light: bool | None
    channel: int = ...
"""Support for Amcrest IP cameras."""

from __future__ import annotations
import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import threading
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Any,
    Any,
)

import aiohttp
from amcrest import AmcrestError, ApiWrapper, LoginError
import voluptuous as vol
from homeassistant.auth.models import User
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.typing import ConfigType

class AmcrestChecker(ApiWrapper):
    """amcrest.ApiWrapper wrapper for catching errors."""

    def __init__(self, hass: HomeAssistant, name: str, host: str, port: int, user: str, password: str) -> None:
        ...

    @property
    def available(self) -> bool:
        ...

    @property
    def available_flag(self) -> threading.Event:
        ...

    @property
    def async_available_flag(self) -> asyncio.Event:
        ...

    @callback
    def _async_start_recovery(self) -> None:
        ...

    def command(self, *args: Any, **kwargs: Any) -> Any:
        ...

    async def async_command(self, *args: Any, **kwargs: Any) -> Any:
        ...

    @asynccontextmanager
    async def async_stream_command(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        ...

    @asynccontextmanager
    async def _async_command_wrapper(self) -> AsyncIterator[Any]:
        ...

    def _handle_offline_thread_safe(self, ex: Exception) -> bool:
        ...

    def _handle_offline(self, ex: Exception) -> None:
        ...

    @callback
    def _async_handle_offline(self, ex: Exception) -> None:
        ...

    def _handle_error_thread_safe(self) -> bool:
        ...

    def _handle_error(self) -> None:
        ...

    @callback
    def _async_handle_error(self) -> None:
        ...

    def _set_online_thread_safe(self) -> bool:
        ...

    def _set_online(self) -> None:
        ...

    @callback
    def _async_set_online(self) -> None:
        ...

    @callback
    def _async_signal_online(self) -> None:
        ...

    async def _wrap_test_online(self, now: datetime) -> None:
        ...

def _monitor_events(hass: HomeAssistant, name: str, api: AmcrestChecker, event_codes: Set[str]) -> None:
    ...

def _start_event_monitor(hass: HomeAssistant, name: str, api: AmcrestChecker, event_codes: Set[str]) -> None:
    ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    ...

@dataclass
class AmcrestDevice:
    """Representation of a base Amcrest discovery device."""
    api: AmcrestChecker
    authentication: Optional[aiohttp.BasicAuth]
    ffmpeg_arguments: str
    stream_source: str
    resolution: str
    control_light: bool
    channel: int = 0
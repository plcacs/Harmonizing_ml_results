```python
from __future__ import annotations
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import threading
from typing import Any, ClassVar
import aiohttp
from amcrest import AmcrestError, ApiWrapper, LoginError
import httpx
import voluptuous as vol
from homeassistant.auth.models import User
from homeassistant.const import (
    ATTR_ENTITY_ID,
    CONF_AUTHENTICATION,
    CONF_BINARY_SENSORS,
    CONF_HOST,
    CONF_NAME,
    CONF_PASSWORD,
    CONF_PORT,
    CONF_SCAN_INTERVAL,
    CONF_SENSORS,
    CONF_SWITCHES,
    CONF_USERNAME,
    ENTITY_MATCH_ALL,
    ENTITY_MATCH_NONE,
    HTTP_BASIC_AUTHENTICATION,
    Platform,
)
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import Unauthorized, UnknownUser
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.discovery import async_load_platform
from homeassistant.helpers.dispatcher import async_dispatcher_send, dispatcher_send
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.service import async_extract_entity_ids
from homeassistant.helpers.typing import ConfigType

_LOGGER: logging.Logger = ...
CONF_RESOLUTION: str = ...
CONF_STREAM_SOURCE: str = ...
CONF_FFMPEG_ARGUMENTS: str = ...
CONF_CONTROL_LIGHT: str = ...
DEFAULT_NAME: str = ...
DEFAULT_PORT: int = ...
DEFAULT_RESOLUTION: str = ...
DEFAULT_ARGUMENTS: str = ...
MAX_ERRORS: int = ...
RECHECK_INTERVAL: timedelta = ...
NOTIFICATION_ID: str = ...
NOTIFICATION_TITLE: str = ...
SCAN_INTERVAL: timedelta = ...
AUTHENTICATION_LIST: dict[str, str] = ...

def _has_unique_names(devices: list[dict[str, Any]]) -> list[dict[str, Any]]: ...
AMCREST_SCHEMA: vol.Schema = ...
CONFIG_SCHEMA: vol.Schema = ...

class AmcrestChecker(ApiWrapper):
    _hass: HomeAssistant
    _wrap_name: str
    _wrap_errors: int
    _wrap_lock: threading.Lock
    _async_wrap_lock: asyncio.Lock
    _wrap_login_err: bool
    _wrap_event_flag: threading.Event
    _async_wrap_event_flag: asyncio.Event
    _unsub_recheck: Any
    
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
    async def async_stream_command(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]: ...
    
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
    hass: HomeAssistant,
    name: str,
    api: AmcrestChecker,
    event_codes: set[str],
) -> None: ...

def _start_event_monitor(
    hass: HomeAssistant,
    name: str,
    api: AmcrestChecker,
    event_codes: set[str],
) -> None: ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool: ...

@dataclass
class AmcrestDevice:
    channel: ClassVar[int] = ...
    api: AmcrestChecker
    authentication: aiohttp.BasicAuth | None
    ffmpeg_arguments: str
    stream_source: str
    resolution: tuple[int, int]
    control_light: bool
    
    def __init__(
        self,
        api: AmcrestChecker,
        authentication: aiohttp.BasicAuth | None,
        ffmpeg_arguments: str,
        stream_source: str,
        resolution: tuple[int, int],
        control_light: bool,
    ) -> None: ...
```
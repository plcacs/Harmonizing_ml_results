"""Support for Amcrest IP cameras."""

from __future__ import annotations
import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import aiohttp
from amcrest import AmcrestError, ApiWrapper, LoginError
import httpx
import voluptuous as vol
from homeassistant.auth.models import User
from homeassistant.auth.permissions.const import POLICY_CONTROL
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
from homeassistant.helpers import config_validation as cv, discovery
from homeassistant.helpers.dispatcher import async_dispatcher_send, dispatcher_send
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.service import async_extract_entity_ids
from homeassistant.helpers.typing import ConfigType

_AUTHENTICATION_LIST = dict[str, str]
_AMCREST_SCHEMA = vol.Schema[
    Dict[
        str, 
        Union[
            str, 
            int, 
            bool, 
            vol.All[vol.In[List[str]]], 
            cv.time_period
        ]
    ]
]
CONFIG_SCHEMA = vol.Schema[
    Dict[
        str, 
        List[
            Dict[
                str, 
                Union[
                    str, 
                    int, 
                    bool, 
                    vol.All[vol.In[List[str]]], 
                    cv.time_period
                ]
            ]
        ]
    ]
]

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
    _unsub_recheck: Optional[Callable[[], None]]

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
    async def _async_command_wrapper(self) -> AsyncIterator[None]:
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

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    ...
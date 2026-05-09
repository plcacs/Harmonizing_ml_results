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

_LOGGER: logging.Logger = logging.getLogger(__name__)

DEFAULT_NAME: str = "Amcrest Camera"
DEFAULT_PORT: int = 80
DEFAULT_RESOLUTION: str = "high"
DEFAULT_ARGUMENTS: str = "-pred 1"
MAX_ERRORS: int = 5
RECHECK_INTERVAL: timedelta = timedelta(minutes=1)
NOTIFICATION_ID: str = "amcrest_notification"
NOTIFICATION_TITLE: str = "Amcrest Camera Setup"
SCAN_INTERVAL: timedelta = timedelta(seconds=10)
AUTHENTICATION_LIST: Dict[str, str] = {"basic": "basic"}

def _has_unique_names(devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ...

AMCREST_SCHEMA: vol.Schema = vol.Schema(
    {
        vol.Required(CONF_HOST): cv.string,
        vol.Required(CONF_USERNAME): cv.string,
        vol.Required(CONF_PASSWORD): cv.string,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
        vol.Optional(CONF_AUTHENTICATION, default=HTTP_BASIC_AUTHENTICATION): vol.All(vol.In(AUTHENTICATION_LIST)),
        vol.Optional(CONF_RESOLUTION, default=DEFAULT_RESOLUTION): vol.All(vol.In(RESOLUTION_LIST)),
        vol.Optional(CONF_STREAM_SOURCE, default=STREAM_SOURCE_LIST[0]): vol.All(vol.In(STREAM_SOURCE_LIST)),
        vol.Optional(CONF_FFMPEG_ARGUMENTS, default=DEFAULT_ARGUMENTS): cv.string,
        vol.Optional(CONF_SCAN_INTERVAL, default=SCAN_INTERVAL): cv.time_period,
        vol.Optional(CONF_BINARY_SENSORS): vol.All(cv.ensure_list, [vol.In(BINARY_SENSOR_KEYS)], vol.Unique(), check_binary_sensors),
        vol.Optional(CONF_SWITCHES): vol.All(cv.ensure_list, [vol.In(SWITCH_KEYS)], vol.Unique()),
        vol.Optional(CONF_SENSORS): vol.All(cv.ensure_list, [vol.In(SENSOR_KEYS)], vol.Unique()),
        vol.Optional(CONF_CONTROL_LIGHT, default=True): cv.boolean,
    }
)

CONFIG_SCHEMA: vol.Schema = vol.Schema(
    {
        DOMAIN: vol.All(cv.ensure_list, [AMCREST_SCHEMA], _has_unique_names),
    },
    extra=vol.ALLOW_EXTRA,
)

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
    async def _async_command_wrapper(self) -> AsyncIterator[None]:
        ...

    def _handle_offline_thread_safe(self, ex: LoginError) -> bool:
        ...

    def _handle_offline(self, ex: LoginError) -> None:
        ...

    @callback
    def _async_handle_offline(self, ex: LoginError) -> None:
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

def _monitor_events(hass: HomeAssistant, name: str, api: AmcrestChecker, event_codes: Set[int]) -> None:
    ...

def _start_event_monitor(hass: HomeAssistant, name: str, api: AmcrestChecker, event_codes: Set[int]) -> None:
    ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    ...

@dataclass
class AmcrestDevice:
    """Representation of a base Amcrest discovery device."""
    api: AmcrestChecker
    authentication: aiohttp.BasicAuth
    ffmpeg_arguments: str
    stream_source: str
    resolution: str
    control_light: bool
    channel: int = 0
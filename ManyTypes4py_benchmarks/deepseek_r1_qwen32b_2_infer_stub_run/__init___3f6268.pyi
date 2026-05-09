"""Provide functionality to stream video source."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    cast,
    dict,
    Final,
    list,
    MappingProxyType,
    Optional,
    tuple,
    Union,
)
import asyncio
import logging
import secrets
import threading
import time
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.typing import ConfigType
from homeassistant.exceptions import HomeAssistantError
from voluptuous import vol
from yarl import URL
from .const import (
    ATTR_ENDPOINTS,
    ATTR_PREFER_TCP,
    ATTR_SETTINGS,
    ATTR_STREAMS,
    CONF_EXTRA_PART_WAIT_TIME,
    CONF_LL_HLS,
    CONF_PART_DURATION,
    CONF_RTSP_TRANSPORT,
    CONF_SEGMENT_DURATION,
    CONF_USE_WALLCLOCK_AS_TIMESTAMPS,
    DOMAIN,
    FORMAT_CONTENT_TYPE,
    HLS_PROVIDER,
    MAX_SEGMENTS,
    OUTPUT_FORMATS,
    OUTPUT_IDLE_TIMEOUT,
    RECORDER_PROVIDER,
    RTSP_TRANSPORTS,
    SEGMENT_DURATION_ADJUSTER,
    SOURCE_TIMEOUT,
    STREAM_RESTART_INCREMENT,
    STREAM_RESTART_RESET_TIME,
    StreamClientError,
)
from .core import (
    PROVIDERS,
    STREAM_SETTINGS_NON_LL_HLS,
    IdleTimer,
    KeyFrameConverter,
    Orientation,
    StreamOutput,
    StreamSettings,
)
from .exceptions import StreamOpenClientError, StreamWorkerError
from .hls import HlsStreamOutput

if TYPE_CHECKING:
    from homeassistant.components.camera import DynamicStreamSettings

__all__: list[str] = [
    "ATTR_SETTINGS",
    "CONF_EXTRA_PART_WAIT_TIME",
    "CONF_RTSP_TRANSPORT",
    "CONF_USE_WALLCLOCK_AS_TIMESTAMPS",
    "DOMAIN",
    "FORMAT_CONTENT_TYPE",
    "HLS_PROVIDER",
    "OUTPUT_FORMATS",
    "RTSP_TRANSPORTS",
    "SOURCE_TIMEOUT",
    "Orientation",
    "Stream",
    "StreamClientError",
    "StreamOpenClientError",
    "create_stream",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)

async def async_check_stream_client_error(
    hass: HomeAssistant, source: str, pyav_options: dict | None = None
) -> None:
    ...

def _check_stream_client_error(
    hass: HomeAssistant, source: str, options: dict | None = None
) -> None:
    ...

def redact_credentials(url: str) -> str:
    ...

def _convert_stream_options(
    hass: HomeAssistant, stream_source: str, stream_options: dict
) -> tuple[dict, StreamSettings]:
    ...

def create_stream(
    hass: HomeAssistant,
    stream_source: str,
    options: dict,
    dynamic_stream_settings: DynamicStreamSettings,
    stream_label: str | None = None,
) -> Stream:
    ...

class Stream:
    def __init__(
        self,
        hass: HomeAssistant,
        source: str,
        pyav_options: dict,
        stream_settings: StreamSettings,
        dynamic_stream_settings: DynamicStreamSettings,
        stream_label: str | None = None,
    ) -> None:
        ...

    def endpoint_url(self, fmt: str) -> str:
        ...

    def outputs(self) -> MappingProxyType[dict[str, StreamOutput]]:
        ...

    def add_provider(
        self, fmt: str, timeout: float = OUTPUT_IDLE_TIMEOUT
    ) -> StreamOutput:
        ...

    async def remove_provider(self, provider: StreamOutput) -> None:
        ...

    def check_idle(self) -> None:
        ...

    @property
    def available(self) -> bool:
        ...

    def set_update_callback(self, update_callback: Callable | None) -> None:
        ...

    @callback
    def _async_update_state(self, available: bool) -> None:
        ...

    async def start(self) -> None:
        ...

    def update_source(self, new_source: str) -> None:
        ...

    def _set_state(self, available: bool) -> None:
        ...

    def _run_worker(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def _stop(self) -> None:
        ...

    async def async_record(
        self,
        video_path: str,
        duration: int = 30,
        lookback: int = 5,
    ) -> None:
        ...

    async def async_get_image(
        self,
        width: int | None = None,
        height: int | None = None,
        wait_for_next_keyframe: bool = False,
    ) -> bytes:
        ...

    def get_diagnostics(self) -> dict:
        ...

STREAM_OPTIONS_SCHEMA: vol.Schema = ...
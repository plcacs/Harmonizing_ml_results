from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import voluptuous as vol

from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.typing import ConfigType

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
from .diagnostics import Diagnostics
from .exceptions import StreamOpenClientError, StreamWorkerError
from .hls import HlsStreamOutput, async_setup_hls

if TYPE_CHECKING:
    from homeassistant.components.camera import DynamicStreamSettings

__all__: list[str]

_LOGGER: logging.Logger

async def async_check_stream_client_error(
    hass: HomeAssistant,
    source: str,
    pyav_options: dict[str, str] | None = ...,
) -> None: ...

def _check_stream_client_error(
    hass: HomeAssistant,
    source: str,
    options: dict[str, str] | None = ...,
) -> None: ...

def redact_credentials(url: str) -> str: ...

def _convert_stream_options(
    hass: HomeAssistant,
    stream_source: str,
    stream_options: dict[str, Any],
) -> tuple[dict[str, str], StreamSettings]: ...

def create_stream(
    hass: HomeAssistant,
    stream_source: str,
    options: dict[str, Any],
    dynamic_stream_settings: DynamicStreamSettings,
    stream_label: str | None = ...,
) -> Stream: ...

DOMAIN_SCHEMA: vol.Schema
CONFIG_SCHEMA: vol.Schema

def set_pyav_logging(enable: bool) -> None: ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool: ...

class Stream:
    hass: HomeAssistant
    source: str
    pyav_options: dict[str, str]
    _stream_settings: StreamSettings
    _stream_label: str | None
    dynamic_stream_settings: DynamicStreamSettings
    access_token: str | None
    _start_stop_lock: asyncio.Lock
    _thread: threading.Thread | None
    _thread_quit: threading.Event
    _outputs: dict[str, StreamOutput]
    _fast_restart_once: bool
    _keyframe_converter: KeyFrameConverter
    _available: bool
    _update_callback: Callable[[], None] | None
    _logger: logging.Logger
    _diagnostics: Diagnostics

    def __init__(
        self,
        hass: HomeAssistant,
        source: str,
        pyav_options: dict[str, str],
        stream_settings: StreamSettings,
        dynamic_stream_settings: DynamicStreamSettings,
        stream_label: str | None = ...,
    ) -> None: ...
    def endpoint_url(self, fmt: str) -> str: ...
    def outputs(self) -> MappingProxyType[str, StreamOutput]: ...
    def add_provider(
        self, fmt: str, timeout: int = ...
    ) -> StreamOutput: ...
    async def remove_provider(self, provider: StreamOutput) -> None: ...
    def check_idle(self) -> None: ...
    @property
    def available(self) -> bool: ...
    def set_update_callback(self, update_callback: Callable[[], None]) -> None: ...
    @callback
    def _async_update_state(self, available: bool) -> None: ...
    async def start(self) -> None: ...
    def update_source(self, new_source: str) -> None: ...
    def _set_state(self, available: bool) -> None: ...
    def _run_worker(self) -> None: ...
    async def stop(self) -> None: ...
    async def _stop(self) -> None: ...
    async def async_record(
        self,
        video_path: str,
        duration: int = ...,
        lookback: int = ...,
    ) -> None: ...
    async def async_get_image(
        self,
        width: int | None = ...,
        height: int | None = ...,
        wait_for_next_keyframe: bool = ...,
    ) -> bytes | None: ...
    def get_diagnostics(self) -> dict[str, Any]: ...

def _should_retry() -> bool: ...

STREAM_OPTIONS_SCHEMA: vol.Schema
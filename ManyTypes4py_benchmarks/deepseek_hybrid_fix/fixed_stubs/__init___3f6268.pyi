from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Optional

import voluptuous as vol
from homeassistant.const import EVENT_HOMEASSISTANT_STOP, EVENT_LOGGING_CHANGED
from homeassistant.core import Event, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.typing import ConfigType
from homeassistant.setup import SetupPhases
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
from .diagnostics import Diagnostics
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

_LOGGER: logging.Logger = ...

async def async_check_stream_client_error(
    hass: HomeAssistant,
    source: str,
    pyav_options: Optional[dict[str, Any]] = None,
) -> None: ...

def _check_stream_client_error(
    hass: HomeAssistant,
    source: str,
    options: Optional[dict[str, Any]] = None,
) -> None: ...

def redact_credentials(url: str) -> str: ...

def _convert_stream_options(
    hass: HomeAssistant,
    stream_source: str,
    stream_options: dict[str, Any],
) -> tuple[dict[str, Any], StreamSettings]: ...

def create_stream(
    hass: HomeAssistant,
    stream_source: str,
    options: dict[str, Any],
    dynamic_stream_settings: DynamicStreamSettings,
    stream_label: Optional[str] = None,
) -> Stream: ...

DOMAIN_SCHEMA: vol.Schema = ...
CONFIG_SCHEMA: vol.Schema = ...

def set_pyav_logging(enable: bool) -> None: ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool: ...

class Stream:
    def __init__(
        self,
        hass: HomeAssistant,
        source: str,
        pyav_options: dict[str, Any],
        stream_settings: StreamSettings,
        dynamic_stream_settings: DynamicStreamSettings,
        stream_label: Optional[str] = None,
    ) -> None:
        self.hass: HomeAssistant = ...
        self.source: str = ...
        self.pyav_options: dict[str, Any] = ...
        self._stream_settings: StreamSettings = ...
        self._stream_label: Optional[str] = ...
        self.dynamic_stream_settings: DynamicStreamSettings = ...
        self.access_token: Optional[str] = ...
        self._outputs: dict[str, StreamOutput] = ...
        self._start_stop_lock: asyncio.Lock = ...
        self._thread: threading.Thread | None = ...
        self._thread_quit: threading.Event = ...
        self._fast_restart_once: bool = ...
        self._keyframe_converter: KeyFrameConverter = ...
        self._available: bool = ...
        self._update_callback: Callable[[], None] | None = ...
        self._logger: logging.Logger = ...
        self._diagnostics: Diagnostics = ...

    def endpoint_url(self, fmt: str) -> str: ...

    def outputs(self) -> MappingProxyType: ...

    def add_provider(
        self,
        fmt: str,
        timeout: float = OUTPUT_IDLE_TIMEOUT,
    ) -> StreamOutput: ...

    async def remove_provider(self, provider: StreamOutput) -> None: ...

    def check_idle(self) -> None: ...

    @property
    def available(self) -> bool: ...

    def set_update_callback(self, update_callback: Optional[Callable[[], None]]) -> None: ...

    async def start(self) -> None: ...

    def update_source(self, new_source: str) -> None: ...

    async def stop(self) -> None: ...

    async def async_record(
        self,
        video_path: str,
        duration: int = 30,
        lookback: int = 5,
    ) -> None: ...

    async def async_get_image(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        wait_for_next_keyframe: bool = False,
    ) -> bytes: ...

    def get_diagnostics(self) -> dict[str, Any]: ...

def _should_retry() -> bool: ...

STREAM_OPTIONS_SCHEMA: vol.Schema = ...
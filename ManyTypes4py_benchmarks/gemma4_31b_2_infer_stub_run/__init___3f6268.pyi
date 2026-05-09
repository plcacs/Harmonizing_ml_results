from __future__ import annotations
import asyncio
import logging
from collections.abc import Mapping
from typing import Any, Optional, Union, overload
from yarl import URL

from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType
from .const import (
    ATTR_SETTINGS,
    CONF_EXTRA_PART_WAIT_TIME,
    CONF_RTSP_TRANSPORT,
    CONF_USE_WALLCLOCK_AS_TIMESTAMPS,
    DOMAIN,
    FORMAT_CONTENT_TYPE,
    HLS_PROVIDER,
    OUTPUT_FORMATS,
    RTSP_TRANSPORTS,
    SOURCE_TIMEOUT,
    StreamClientError,
)
from .core import (
    Orientation,
    StreamOutput,
    StreamSettings,
)
from .exceptions import StreamOpenClientError
from .hls import HlsStreamOutput
from homeassistant.components.camera import DynamicStreamSettings

ATTR_SETTINGS: str = "..."
CONF_EXTRA_PART_WAIT_TIME: str = "..."
CONF_RTSP_TRANSPORT: str = "..."
CONF_USE_WALLCLOCK_AS_TIMESTAMPS: str = "..."
DOMAIN: str = "..."
FORMAT_CONTENT_TYPE: str = "..."
HLS_PROVIDER: str = "..."
OUTPUT_FORMATS: list[str] = "..."
RTSP_TRANSPORTS: list[str] = "..."
SOURCE_TIMEOUT: int = "..."
Orientation: Any = "..."
StreamClientError: Any = "..."
StreamOpenClientError: Any = "..."

async def async_check_stream_client_error(
    hass: HomeAssistant,
    source: str,
    pyav_options: Optional[Mapping[str, Any]] = None,
) -> None: ...

def redact_credentials(url: str) -> str: ...

def create_stream(
    hass: HomeAssistant,
    stream_source: str,
    options: Mapping[str, Any],
    dynamic_stream_settings: DynamicStreamSettings,
    stream_label: Optional[str] = None,
) -> Stream: ...

def set_pyav_logging(enable: bool) -> None: ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool: ...

class Stream:
    """Represents a single stream."""

    def __init__(
        self,
        hass: HomeAssistant,
        source: str,
        pyav_options: Mapping[str, Any],
        stream_settings: StreamSettings,
        dynamic_stream_settings: DynamicStreamSettings,
        stream_label: Optional[str] = None,
    ) -> None: ...

    def endpoint_url(self, fmt: str) -> str: ...

    def outputs(self) -> Mapping[str, StreamOutput]: ...

    def add_provider(self, fmt: str, timeout: float = 60.0) -> StreamOutput: ...

    async def remove_provider(self, provider: StreamOutput) -> None: ...

    def check_idle(self) -> None: ...

    @property
    def available(self) -> bool: ...

    def set_update_callback(self, update_callback: Any) -> None: ...

    async def start(self) -> None: ...

    def update_source(self, new_source: str) -> None: ...

    def _set_state(self, available: bool) -> None: ...

    def _run_worker(self) -> None: ...

    async def stop(self) -> None: ...

    async def _stop(self) -> None: ...

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
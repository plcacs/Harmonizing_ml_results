from __future__ import annotations
import asyncio
from collections.abc import Callable, Mapping
import copy
import logging
import secrets
import threading
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, cast
import voluptuous as vol
from yarl import URL
from homeassistant.const import EVENT_HOMEASSISTANT_STOP, EVENT_LOGGING_CHANGED
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType
from homeassistant.setup import SetupPhases, async_pause_setup
from homeassistant.util.async_ import create_eager_task
from .const import ATTR_ENDPOINTS, ATTR_PREFER_TCP, ATTR_SETTINGS, ATTR_STREAMS, CONF_EXTRA_PART_WAIT_TIME, CONF_LL_HLS, CONF_PART_DURATION, CONF_RTSP_TRANSPORT, CONF_SEGMENT_DURATION, CONF_USE_WALLCLOCK_AS_TIMESTAMPS, DOMAIN, FORMAT_CONTENT_TYPE, HLS_PROVIDER, MAX_SEGMENTS, OUTPUT_FORMATS, OUTPUT_IDLE_TIMEOUT, RECORDER_PROVIDER, RTSP_TRANSPORTS, SEGMENT_DURATION_ADJUSTER, SOURCE_TIMEOUT, STREAM_RESTART_INCREMENT, STREAM_RESTART_RESET_TIME, StreamClientError
from .core import PROVIDERS, STREAM_SETTINGS_NON_LL_HLS, IdleTimer, KeyFrameConverter, Orientation, StreamOutput, StreamSettings
from .diagnostics import Diagnostics
from .exceptions import StreamOpenClientError, StreamWorkerError
from .hls import HlsStreamOutput, async_setup_hls
if TYPE_CHECKING:
    from homeassistant.components.camera import DynamicStreamSettings
__all__ = ['ATTR_SETTINGS', 'CONF_EXTRA_PART_WAIT_TIME', 'CONF_RTSP_TRANSPORT', 'CONF_USE_WALLCLOCK_AS_TIMESTAMPS', 'DOMAIN', 'FORMAT_CONTENT_TYPE', 'HLS_PROVIDER', 'OUTPUT_FORMATS', 'RTSP_TRANSPORTS', 'SOURCE_TIMEOUT', 'Orientation', 'Stream', 'StreamClientError', 'StreamOpenClientError', 'create_stream']
_LOGGER: logging.Logger = logging.getLogger(__name__)

async def async_check_stream_client_error(hass: HomeAssistant, source: str, pyav_options: dict = None) -> None:
    ...

def _check_stream_client_error(hass: HomeAssistant, source: str, options: dict = None) -> None:
    ...

def redact_credentials(url: str) -> str:
    ...

def _convert_stream_options(hass: HomeAssistant, stream_source: str, stream_options: dict) -> tuple[dict, StreamSettings]:
    ...

def create_stream(hass: HomeAssistant, stream_source: str, options: dict, dynamic_stream_settings: DynamicStreamSettings, stream_label: str = None) -> Stream:
    ...

DOMAIN_SCHEMA: vol.Schema = vol.Schema({vol.Optional(CONF_LL_HLS, default=True): cv.boolean, vol.Optional(CONF_SEGMENT_DURATION, default=6): vol.All(cv.positive_float, vol.Range(min=2, max=10)), vol.Optional(CONF_PART_DURATION, default=1): vol.All(cv.positive_float, vol.Range(min=0.2, max=1.5)})
CONFIG_SCHEMA: vol.Schema = vol.Schema({DOMAIN: DOMAIN_SCHEMA}, extra=vol.ALLOW_EXTRA)

def set_pyav_logging(enable: bool) -> None:
    ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    ...

class Stream:
    def __init__(self, hass: HomeAssistant, source: str, pyav_options: dict, stream_settings: StreamSettings, dynamic_stream_settings: DynamicStreamSettings, stream_label: str = None) -> None:
        ...

    def endpoint_url(self, fmt: str) -> str:
        ...

    def outputs(self) -> MappingProxyType:
        ...

    def add_provider(self, fmt: str, timeout: int = OUTPUT_IDLE_TIMEOUT) -> StreamOutput:
        ...

    async def remove_provider(self, provider: StreamOutput) -> None:
        ...

    def check_idle(self) -> None:
        ...

    @property
    def available(self) -> bool:
        ...

    def set_update_callback(self, update_callback: Callable) -> None:
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

    async def async_record(self, video_path: str, duration: int = 30, lookback: int = 5) -> None:
        ...

    async def async_get_image(self, width: int = None, height: int = None, wait_for_next_keyframe: bool = False) -> bytes:
        ...

    def get_diagnostics(self) -> dict:
        ...

def _should_retry() -> bool:
    ...

STREAM_OPTIONS_SCHEMA: vol.Schema = vol.Schema({vol.Optional(CONF_RTSP_TRANSPORT): vol.In(RTSP_TRANSPORTS), vol.Optional(CONF_USE_WALLCLOCK_AS_TIMESTAMPS): bool, vol.Optional(CONF_EXTRA_PART_WAIT_TIME): cv.positive_float})

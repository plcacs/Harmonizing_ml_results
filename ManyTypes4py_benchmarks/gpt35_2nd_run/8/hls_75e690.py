from __future__ import annotations
from typing import TYPE_CHECKING, cast
from aiohttp import web
from homeassistant.core import HomeAssistant, callback
from .core import PROVIDERS, IdleTimer, Segment, StreamOutput, StreamSettings, StreamView
from .const import EXT_X_START_LL_HLS, EXT_X_START_NON_LL_HLS, FORMAT_CONTENT_TYPE, HLS_PROVIDER, MAX_SEGMENTS, NUM_PLAYLIST_SEGMENTS
from .fmp4utils import get_codec_string, transform_init

if TYPE_CHECKING:
    from homeassistant.components.camera import DynamicStreamSettings
    from . import Stream

@callback
def async_setup_hls(hass: HomeAssistant) -> str:
    ...

@PROVIDERS.register(HLS_PROVIDER)
class HlsStreamOutput(StreamOutput):
    def __init__(self, hass: HomeAssistant, idle_timer: IdleTimer, stream_settings: StreamSettings, dynamic_stream_settings: DynamicStreamSettings):
        ...

    @property
    def name(self) -> str:
        ...

    def cleanup(self) -> None:
        ...

    @property
    def target_duration(self) -> float:
        ...

    @callback
    def _async_put(self, segment: Segment) -> None:
        ...

    def discontinuity(self) -> None:
        ...

    @callback
    def _async_discontinuity(self) -> None:
        ...

class HlsMasterPlaylistView(StreamView):
    @staticmethod
    def render(track: Stream) -> str:
        ...

    async def handle(self, request: web.Request, stream: Stream, sequence: int, part_num: int) -> web.Response:
        ...

class HlsPlaylistView(StreamView):
    @classmethod
    def render(cls, track: Stream) -> str:
        ...

    @staticmethod
    def bad_request(blocking: bool, target_duration: float) -> web.Response:
        ...

    @staticmethod
    def not_found(blocking: bool, target_duration: float) -> web.Response:
        ...

    async def handle(self, request: web.Request, stream: Stream, sequence: int, part_num: int) -> web.Response:
        ...

class HlsInitView(StreamView):
    async def handle(self, request: web.Request, stream: Stream, sequence: int, part_num: int) -> web.Response:
        ...

class HlsPartView(StreamView):
    async def handle(self, request: web.Request, stream: Stream, sequence: int, part_num: int) -> web.Response:
        ...

class HlsSegmentView(StreamView):
    async def handle(self, request: web.Request, stream: Stream, sequence: int, part_num: int) -> web.Response:
        ...

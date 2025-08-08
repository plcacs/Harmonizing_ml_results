from __future__ import annotations
from typing import TYPE_CHECKING, cast
from aiohttp import web
from homeassistant.core import HomeAssistant, callback
from .const import EXT_X_START_LL_HLS, EXT_X_START_NON_LL_HLS, FORMAT_CONTENT_TYPE, HLS_PROVIDER, MAX_SEGMENTS, NUM_PLAYLIST_SEGMENTS
from .core import PROVIDERS, IdleTimer, Segment, StreamOutput, StreamSettings, StreamView
from .fmp4utils import get_codec_string, transform_init

if TYPE_CHECKING:
    from homeassistant.components.camera import DynamicStreamSettings
    from . import Stream

@callback
def async_setup_hls(hass: HomeAssistant) -> str:
    """Set up api endpoints."""
    hass.http.register_view(HlsPlaylistView())
    hass.http.register_view(HlsSegmentView())
    hass.http.register_view(HlsInitView())
    hass.http.register_view(HlsMasterPlaylistView())
    hass.http.register_view(HlsPartView())
    return '/api/hls/{}/master_playlist.m3u8'

@PROVIDERS.register(HLS_PROVIDER)
class HlsStreamOutput(StreamOutput):
    """Represents HLS Output formats."""

    def __init__(self, hass: HomeAssistant, idle_timer: IdleTimer, stream_settings: StreamSettings, dynamic_stream_settings: DynamicStreamSettings):
        """Initialize HLS output."""
        super().__init__(hass, idle_timer, stream_settings, dynamic_stream_settings, deque_maxlen=MAX_SEGMENTS)
        self._target_duration = stream_settings.min_segment_duration

    @property
    def name(self) -> str:
        """Return provider name."""
        return HLS_PROVIDER

    def cleanup(self) -> None:
        """Handle cleanup."""
        super().cleanup()
        self._segments.clear()

    @property
    def target_duration(self) -> float:
        """Return the target duration."""
        return self._target_duration

    @callback
    def _async_put(self, segment: Segment) -> None:
        """Async put and also update the target duration."""
        super()._async_put(segment)
        self._target_duration = max((s.duration for s in self._segments), default=segment.duration) or self.stream_settings.min_segment_duration

    def discontinuity(self) -> None:
        """Fix incomplete segment at end of deque."""
        self._hass.loop.call_soon_threadsafe(self._async_discontinuity)

    @callback
    def _async_discontinuity(self) -> None:
        """Fix incomplete segment at end of deque in event loop."""
        if self._segments:
            if (last_segment := self._segments[-1]).parts:
                last_segment.duration = sum((part.duration for part in last_segment.parts))
            else:
                self._segments.pop()

class HlsMasterPlaylistView(StreamView):
    """Stream view used only for Chromecast compatibility."""
    url: str = '/api/hls/{token:[a-f0-9]+}/master_playlist.m3u8'
    name: str = 'api:stream:hls:master_playlist'
    cors_allowed: bool = True

    @staticmethod
    def render(track: Stream) -> str:
        """Render M3U8 file."""
        if not (segment := track.get_segment(track.sequences[-2])):
            return ''
        bandwidth = round(segment.data_size_with_init * 8 / segment.duration * 1.2)
        codecs = get_codec_string(segment.init)
        lines = ['#EXTM3U', f'#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},CODECS="{codecs}"', 'playlist.m3u8']
        return '\n'.join(lines) + '\n'

    async def handle(self, request: web.Request, stream: Stream, sequence: int, part_num: int) -> web.Response:
        """Return m3u8 playlist."""
        track = stream.add_provider(HLS_PROVIDER)
        await stream.start()
        if not track.sequences and (not await track.recv()):
            return web.HTTPNotFound()
        if len(track.sequences) == 1 and (not await track.recv()):
            return web.HTTPNotFound()
        response = web.Response(body=self.render(track).encode('utf-8'), headers={'Content-Type': FORMAT_CONTENT_TYPE[HLS_PROVIDER]})
        response.enable_compression(web.ContentCoding.gzip)
        return response

# Remaining classes and methods have similar type annotations

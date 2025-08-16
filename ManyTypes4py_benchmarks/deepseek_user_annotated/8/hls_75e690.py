"""Provide functionality to stream HLS."""

from __future__ import annotations

from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, Optional, Tuple, cast, List, Dict, Any

from aiohttp import web

from homeassistant.core import HomeAssistant, callback

from .const import (
    EXT_X_START_LL_HLS,
    EXT_X_START_NON_LL_HLS,
    FORMAT_CONTENT_TYPE,
    HLS_PROVIDER,
    MAX_SEGMENTS,
    NUM_PLAYLIST_SEGMENTS,
)
from .core import (
    PROVIDERS,
    IdleTimer,
    Segment,
    StreamOutput,
    StreamSettings,
    StreamView,
    Stream,
)
from .fmp4utils import get_codec_string, transform_init

if TYPE_CHECKING:
    from homeassistant.components.camera import DynamicStreamSettings


@callback
def async_setup_hls(hass: HomeAssistant) -> str:
    """Set up api endpoints."""
    hass.http.register_view(HlsPlaylistView())
    hass.http.register_view(HlsSegmentView())
    hass.http.register_view(HlsInitView())
    hass.http.register_view(HlsMasterPlaylistView())
    hass.http.register_view(HlsPartView())
    return "/api/hls/{}/master_playlist.m3u8"


@PROVIDERS.register(HLS_PROVIDER)
class HlsStreamOutput(StreamOutput):
    """Represents HLS Output formats."""

    def __init__(
        self,
        hass: HomeAssistant,
        idle_timer: IdleTimer,
        stream_settings: StreamSettings,
        dynamic_stream_settings: DynamicStreamSettings,
    ) -> None:
        """Initialize HLS output."""
        super().__init__(
            hass,
            idle_timer,
            stream_settings,
            dynamic_stream_settings,
            deque_maxlen=MAX_SEGMENTS,
        )
        self._target_duration: float = stream_settings.min_segment_duration

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
        self._target_duration = (
            max((s.duration for s in self._segments), default=segment.duration)
            or self.stream_settings.min_segment_duration
        )

    def discontinuity(self) -> None:
        """Fix incomplete segment at end of deque."""
        self._hass.loop.call_soon_threadsafe(self._async_discontinuity)

    @callback
    def _async_discontinuity(self) -> None:
        """Fix incomplete segment at end of deque in event loop."""
        if self._segments:
            if (last_segment := self._segments[-1]).parts:
                last_segment.duration = sum(
                    part.duration for part in last_segment.parts
                )
            else:
                self._segments.pop()


class HlsMasterPlaylistView(StreamView):
    """Stream view used only for Chromecast compatibility."""

    url: str = r"/api/hls/{token:[a-f0-9]+}/master_playlist.m3u8"
    name: str = "api:stream:hls:master_playlist"
    cors_allowed: bool = True

    @staticmethod
    def render(track: StreamOutput) -> str:
        """Render M3U8 file."""
        if not (segment := track.get_segment(track.sequences[-2])):
            return ""
        bandwidth: int = round(segment.data_size_with_init * 8 / segment.duration * 1.2)
        codecs: str = get_codec_string(segment.init)
        lines: List[str] = [
            "#EXTM3U",
            f'#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},CODECS="{codecs}"',
            "playlist.m3u8",
        ]
        return "\n".join(lines) + "\n"

    async def handle(
        self, request: web.Request, stream: Stream, sequence: str, part_num: str
    ) -> web.Response:
        """Return m3u8 playlist."""
        track: StreamOutput = stream.add_provider(HLS_PROVIDER)
        await stream.start()
        if not track.sequences and not await track.recv():
            return web.HTTPNotFound()
        if len(track.sequences) == 1 and not await track.recv():
            return web.HTTPNotFound()
        response: web.Response = web.Response(
            body=self.render(track).encode("utf-8"),
            headers={
                "Content-Type": FORMAT_CONTENT_TYPE[HLS_PROVIDER],
            },
        )
        response.enable_compression(web.ContentCoding.gzip)
        return response


class HlsPlaylistView(StreamView):
    """Stream view to serve a M3U8 stream."""

    url: str = r"/api/hls/{token:[a-f0-9]+}/playlist.m3u8"
    name: str = "api:stream:hls:playlist"
    cors_allowed: bool = True

    @classmethod
    def render(cls, track: HlsStreamOutput) -> str:
        """Render HLS playlist file."""
        segments: List[Segment] = list(track.get_segments())[-(NUM_PLAYLIST_SEGMENTS + 1) :]

        if segments[-1].complete:
            segments = segments[-NUM_PLAYLIST_SEGMENTS:]

        first_segment: Segment = segments[0]
        playlist: List[str] = [
            "#EXTM3U",
            "#EXT-X-VERSION:6",
            "#EXT-X-INDEPENDENT-SEGMENTS",
            '#EXT-X-MAP:URI="init.mp4"',
            f"#EXT-X-TARGETDURATION:{track.target_duration:.0f}",
            f"#EXT-X-MEDIA-SEQUENCE:{first_segment.sequence}",
            f"#EXT-X-DISCONTINUITY-SEQUENCE:{first_segment.stream_id}",
        ]

        if track.stream_settings.ll_hls:
            playlist.extend(
                [
                    "#EXT-X-PART-INF:PART-TARGET="
                    f"{track.stream_settings.part_target_duration:.3f}",
                    "#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK="
                    f"{2 * track.stream_settings.part_target_duration:.3f}",
                    "#EXT-X-START:TIME-OFFSET=-"
                    f"{EXT_X_START_LL_HLS * track.stream_settings.part_target_duration:.3f}"
                    ",PRECISE=YES",
                ]
            )
        else:
            playlist.append(
                "#EXT-X-START:TIME-OFFSET=-"
                f"{EXT_X_START_NON_LL_HLS * track.target_duration:.3f}"
                ",PRECISE=YES"
            )

        last_stream_id: int = first_segment.stream_id

        for i, segment in enumerate(segments[:-1], 3 - len(segments)):
            playlist.append(
                segment.render_hls(
                    last_stream_id=last_stream_id,
                    render_parts=i >= 0 and track.stream_settings.ll_hls,
                    add_hint=False,
                )
            )
            last_stream_id = segment.stream_id

        playlist.append(
            segments[-1].render_hls(
                last_stream_id=last_stream_id,
                render_parts=track.stream_settings.ll_hls,
                add_hint=track.stream_settings.ll_hls,
            )
        )

        return "\n".join(playlist) + "\n"

    @staticmethod
    def bad_request(blocking: bool, target_duration: float) -> web.Response:
        """Return a HTTP Bad Request response."""
        return web.Response(
            body=None,
            status=HTTPStatus.BAD_REQUEST,
        )

    @staticmethod
    def not_found(blocking: bool, target_duration: float) -> web.Response:
        """Return a HTTP Not Found response."""
        return web.Response(
            body=None,
            status=HTTPStatus.NOT_FOUND,
        )

    async def handle(
        self, request: web.Request, stream: Stream, sequence: str, part_num: str
    ) -> web.Response:
        """Return m3u8 playlist."""
        track: HlsStreamOutput = cast(
            HlsStreamOutput, stream.add_provider(HLS_PROVIDER)
        )
        await stream.start()

        hls_msn: Optional[str] = request.query.get("_HLS_msn")
        hls_part: Optional[str] = request.query.get("_HLS_part")
        blocking_request: bool = bool(hls_msn or hls_part)

        if hls_msn is None and hls_part:
            return web.HTTPBadRequest()

        hls_msn_int: int = int(hls_msn or 0)

        if hls_msn_int > track.last_sequence + 2:
            return self.bad_request(blocking_request, track.target_duration)

        if hls_part is None:
            hls_part_int: int = -1
            hls_msn_int += 1
        else:
            hls_part_int = int(hls_part)

        while hls_msn_int > track.last_sequence:
            if not await track.recv():
                return self.not_found(blocking_request, track.target_duration)
        if track.last_segment is None:
            return self.not_found(blocking_request, 0)
        if (
            (last_segment := track.last_segment)
            and hls_msn_int == last_segment.sequence
            and hls_part_int
            >= len(last_segment.parts)
            - 1
            + track.stream_settings.hls_advance_part_limit
        ):
            return self.bad_request(blocking_request, track.target_duration)

        while (
            (last_segment := track.last_segment)
            and hls_msn_int == last_segment.sequence
            and hls_part_int >= len(last_segment.parts)
        ):
            if not await track.part_recv(
                timeout=track.stream_settings.hls_part_timeout
            ):
                return self.not_found(blocking_request, track.target_duration)
        
        if hls_msn_int + 1 == last_segment.sequence:
            if not (previous_segment := track.get_segment(hls_msn_int)) or (
                hls_part_int >= len(previous_segment.parts)
                and not last_segment.parts
                and not await track.part_recv(
                    timeout=track.stream_settings.hls_part_timeout
                )
            ):
                return self.not_found(blocking_request, track.target_duration)

        response: web.Response = web.Response(
            body=self.render(track).encode("utf-8"),
            headers={
                "Content-Type": FORMAT_CONTENT_TYPE[HLS_PROVIDER],
            },
        )
        response.enable_compression(web.ContentCoding.gzip)
        return response


class HlsInitView(StreamView):
    """Stream view to serve HLS init.mp4."""

    url: str = r"/api/hls/{token:[a-f0-9]+}/init.mp4"
    name: str = "api:stream:hls:init"
    cors_allowed: bool = True

    async def handle(
        self, request: web.Request, stream: Stream, sequence: str, part_num: str
    ) -> web.Response:
        """Return init.mp4."""
        track: StreamOutput = stream.add_provider(HLS_PROVIDER)
        if not (segments := track.get_segments()) or not (body := segments[0].init):
            return web.HTTPNotFound()
        return web.Response(
            body=transform_init(body, stream.dynamic_stream_settings.orientation),
            headers={"Content-Type": "video/mp4"},
        )


class HlsPartView(StreamView):
    """Stream view to serve a HLS fmp4 segment."""

    url: str = r"/api/hls/{token:[a-f0-9]+}/segment/{sequence:\d+}.{part_num:\d+}.m4s"
    name: str = "api:stream:hls:part"
    cors_allowed: bool = True

    async def handle(
        self, request: web.Request, stream: Stream, sequence: str, part_num: str
    ) -> web.Response:
        """Handle part."""
        track: HlsStreamOutput = cast(
            HlsStreamOutput, stream.add_provider(HLS_PROVIDER)
        )
        track.idle_timer.awake()
        if not (
            (segment := track.get_segment(int(sequence)))
            or (
                await track.part_recv(timeout=track.stream_settings.hls_part_timeout)
                and (segment := track.get_segment(int(sequence)))
            )
        ):
            return web.Response(
                body=None,
                status=HTTPStatus.NOT_FOUND,
            )
        if int(part_num) == len(segment.parts):
            await track.part_recv(timeout=track.stream_settings.hls_part_timeout)
        if int(part_num) >= len(segment.parts):
            return web.HTTPRequestRangeNotSatisfiable()
        return web.Response(
            body=segment.parts[int(part_num)].data,
            headers={
                "Content-Type": "video/iso.segment",
            },
        )


class HlsSegmentView(StreamView):
    """Stream view to serve a HLS fmp4 segment."""

    url: str = r"/api/hls/{token:[a-f0-9]+}/segment/{sequence:\d+}.m4s"
    name: str = "api:stream:hls:segment"
    cors_allowed: bool = True

    async def handle(
        self, request: web.Request, stream: Stream, sequence: str, part_num: str
    ) -> web.StreamResponse:
        """Handle segments."""
        track: HlsStreamOutput = cast(
            HlsStreamOutput, stream.add_provider(HLS_PROVIDER)
        )
        track.idle_timer.awake()
        if not (
            (segment := track.get_segment(int(sequence)))
            or (
                await track.part_recv(timeout=track.stream_settings.hls_part_timeout)
                and (segment := track.get_segment(int(sequence)))
            )
        ):
            return web.Response(
                body=None,
                status=HTTPStatus.NOT_FOUND,
            )
        return web.Response(
            body=segment.get_data(),
            headers={
                "Content-Type": "video/iso.segment",
            },
        )

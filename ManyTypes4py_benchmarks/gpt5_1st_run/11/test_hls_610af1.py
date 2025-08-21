from datetime import timedelta
from http import HTTPStatus
from typing import Any, Awaitable, Callable, List, Mapping, Optional
from unittest.mock import patch
from urllib.parse import ParseResult, urlparse

import av
import pytest
from aiohttp import ClientResponse
from aiohttp.test_utils import TestClient
from homeassistant.components.stream import Stream, create_stream
from homeassistant.components.stream.const import (
    EXT_X_START_LL_HLS,
    EXT_X_START_NON_LL_HLS,
    HLS_PROVIDER,
    MAX_SEGMENTS,
    NUM_PLAYLIST_SEGMENTS,
)
from homeassistant.components.stream.core import Orientation, Part
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util

from .common import (
    FAKE_TIME,
    DefaultSegment as Segment,
    assert_mp4_has_transform_matrix,
    dynamic_stream_settings,
)
from tests.common import async_fire_time_changed
from tests.typing import ClientSessionGenerator

STREAM_SOURCE: str = "some-stream-source"
INIT_BYTES: bytes = b"\x00\x00\x00\x08moov"
FAKE_PAYLOAD: bytes = b"fake-payload"
SEGMENT_DURATION: int = 10
TEST_TIMEOUT: float = 5.0
MAX_ABORT_SEGMENTS: int = 20
HLS_CONFIG: dict[str, dict[str, bool]] = {"stream": {"ll_hls": False}}


@pytest.fixture
async def setup_component(hass: HomeAssistant) -> None:
    """Test fixture to setup the stream component."""
    await async_setup_component(hass, "stream", HLS_CONFIG)


class HlsClient:
    """Test fixture for fetching the hls stream."""

    def __init__(self, http_client: TestClient, parsed_url: ParseResult) -> None:
        """Initialize HlsClient."""
        self.http_client: TestClient = http_client
        self.parsed_url: ParseResult = parsed_url

    async def get(
        self, path: Optional[str] = None, headers: Optional[Mapping[str, str]] = None
    ) -> ClientResponse:
        """Fetch the hls stream for the specified path."""
        url = self.parsed_url.path
        if path:
            url = "/".join(self.parsed_url.path.split("/")[:-1]) + path
        return await self.http_client.get(url, headers=headers)


@pytest.fixture
def hls_stream(
    hass: HomeAssistant, hass_client: ClientSessionGenerator
) -> Callable[[Stream], Awaitable[HlsClient]]:
    """Create test fixture for creating an HLS client for a stream."""

    async def create_client_for_stream(stream: Stream) -> HlsClient:
        http_client: TestClient = await hass_client()
        parsed_url: ParseResult = urlparse(stream.endpoint_url(HLS_PROVIDER))
        return HlsClient(http_client, parsed_url)

    return create_client_for_stream


def make_segment(segment: int, discontinuity: bool = False) -> str:
    """Create a playlist response for a segment."""
    response: List[str] = ["#EXT-X-DISCONTINUITY"] if discontinuity else []
    response.extend(
        [
            "#EXT-X-PROGRAM-DATE-TIME:"
            + FAKE_TIME.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            + "Z",
            f"#EXTINF:{SEGMENT_DURATION:.3f},",
            f"./segment/{segment}.m4s",
        ]
    )
    return "\n".join(response)


def make_playlist(
    sequence: int,
    discontinuity_sequence: int = 0,
    segments: Optional[List[str]] = None,
    hint: Optional[str] = None,
    segment_duration: Optional[float] = None,
    part_target_duration: Optional[float] = None,
) -> str:
    """Create a an hls playlist response for tests to assert on."""
    if not segment_duration:
        segment_duration = SEGMENT_DURATION
    response: List[str] = [
        "#EXTM3U",
        "#EXT-X-VERSION:6",
        "#EXT-X-INDEPENDENT-SEGMENTS",
        '#EXT-X-MAP:URI="init.mp4"',
        f"#EXT-X-TARGETDURATION:{segment_duration}",
        f"#EXT-X-MEDIA-SEQUENCE:{sequence}",
        f"#EXT-X-DISCONTINUITY-SEQUENCE:{discontinuity_sequence}",
    ]
    if hint:
        response.extend(
            [
                f"#EXT-X-PART-INF:PART-TARGET={part_target_duration:.3f}",
                f"#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK={2 * part_target_duration:.3f}",
                f"#EXT-X-START:TIME-OFFSET=-{EXT_X_START_LL_HLS * part_target_duration:.3f},PRECISE=YES",
            ]
        )
    else:
        response.append(
            f"#EXT-X-START:TIME-OFFSET=-{EXT_X_START_NON_LL_HLS * segment_duration:.3f},PRECISE=YES"
        )
    if segments:
        response.extend(segments)
    if hint:
        response.append(hint)
    response.append("")
    return "\n".join(response)


async def test_hls_stream(
    hass: HomeAssistant,
    setup_component: Any,
    hls_stream: Callable[[Stream], Awaitable[HlsClient]],
    stream_worker_sync: Any,
    h264_video: str,
) -> None:
    """Test hls stream.

    Purposefully not mocking anything here to test full
    integration with the stream component.
    """
    stream_worker_sync.pause()
    stream = create_stream(hass, h264_video, {}, dynamic_stream_settings())
    stream.add_provider(HLS_PROVIDER)
    await stream.start()
    hls_client = await hls_stream(stream)
    master_playlist_response = await hls_client.get()
    assert master_playlist_response.status == HTTPStatus.OK
    master_playlist = await master_playlist_response.text()
    init_response = await hls_client.get("/init.mp4")
    assert init_response.status == HTTPStatus.OK
    playlist_url = "/" + master_playlist.splitlines()[-1]
    playlist_response = await hls_client.get(playlist_url)
    assert playlist_response.status == HTTPStatus.OK
    playlist = await playlist_response.text()
    segment_url = "/" + [line for line in playlist.splitlines() if line][-1]
    segment_response = await hls_client.get(segment_url)
    assert segment_response.status == HTTPStatus.OK
    stream_worker_sync.resume()
    await stream.stop()
    fail_response = await hls_client.get()
    assert fail_response.status == HTTPStatus.NOT_FOUND
    assert stream.get_diagnostics() == {
        "container_format": "mov,mp4,m4a,3gp,3g2,mj2",
        "keepalive": False,
        "orientation": Orientation.NO_TRANSFORM,
        "start_worker": 1,
        "video_codec": "h264",
        "worker_error": 1,
    }


async def test_stream_timeout(
    hass: HomeAssistant,
    hass_client: ClientSessionGenerator,
    setup_component: Any,
    stream_worker_sync: Any,
    h264_video: str,
) -> None:
    """Test hls stream timeout."""
    stream_worker_sync.pause()
    stream = create_stream(hass, h264_video, {}, dynamic_stream_settings())
    available_states: List[bool] = []

    def update_callback() -> None:
        nonlocal available_states
        available_states.append(stream.available)

    stream.set_update_callback(update_callback)
    stream.add_provider(HLS_PROVIDER)
    await stream.start()
    url = stream.endpoint_url(HLS_PROVIDER)
    http_client: TestClient = await hass_client()
    parsed_url = urlparse(url)
    playlist_response = await http_client.get(parsed_url.path)
    assert playlist_response.status == HTTPStatus.OK
    future = dt_util.utcnow() + timedelta(minutes=1)
    async_fire_time_changed(hass, future)
    await hass.async_block_till_done()
    playlist_response = await http_client.get(parsed_url.path)
    assert playlist_response.status == HTTPStatus.OK
    stream_worker_sync.resume()
    future = dt_util.utcnow() + timedelta(minutes=5)
    async_fire_time_changed(hass, future)
    await hass.async_block_till_done()
    fail_response = await http_client.get(parsed_url.path)
    assert fail_response.status == HTTPStatus.NOT_FOUND
    assert available_states == [True]


async def test_stream_timeout_after_stop(
    hass: HomeAssistant,
    hass_client: ClientSessionGenerator,
    setup_component: Any,
    stream_worker_sync: Any,
    h264_video: str,
) -> None:
    """Test hls stream timeout after the stream has been stopped already."""
    stream_worker_sync.pause()
    stream = create_stream(hass, h264_video, {}, dynamic_stream_settings())
    stream.add_provider(HLS_PROVIDER)
    await stream.start()
    stream_worker_sync.resume()
    await stream.stop()
    future = dt_util.utcnow() + timedelta(minutes=5)
    async_fire_time_changed(hass, future)
    await hass.async_block_till_done()


@pytest.mark.parametrize(
    "exception",
    [av.error.InvalidDataError(-2, "error"), av.HTTPBadRequestError(500, "error")],
)
async def test_stream_retries(
    hass: HomeAssistant, setup_component: Any, should_retry: Any, exception: Exception
) -> None:
    """Test hls stream is retried on failure."""
    source = "test_stream_keepalive_source"
    stream = create_stream(hass, source, {}, dynamic_stream_settings())
    track = stream.add_provider(HLS_PROVIDER)
    track.num_segments = 2
    available_states: List[bool] = []

    def update_callback() -> None:
        nonlocal available_states
        available_states.append(stream.available)

    stream.set_update_callback(update_callback)
    open_future1 = hass.loop.create_future()
    open_future2 = hass.loop.create_future()
    futures = [open_future2, open_future1]
    original_set_state = Stream._set_state

    def set_state_wrapper(self: Stream, state: bool) -> None:
        if state is False:
            should_retry.return_value = False
        original_set_state(self, state)

    def av_open_side_effect(*args: Any, **kwargs: Any) -> None:
        hass.loop.call_soon_threadsafe(futures.pop().set_result, None)
        raise exception

    with patch("av.open") as av_open, patch(
        "homeassistant.components.stream.Stream._set_state", set_state_wrapper
    ), patch("homeassistant.components.stream.STREAM_RESTART_INCREMENT", 0):
        av_open.side_effect = av_open_side_effect
        should_retry.return_value = True
        await stream.start()
        await open_future1
        await open_future2
        await hass.async_add_executor_job(stream._thread.join)
        stream._thread = None
        assert av_open.call_count == 2
        await hass.async_block_till_done()
    await stream.stop()
    assert available_states == [True, False, True]


async def test_hls_playlist_view_no_output(
    hass: HomeAssistant,
    setup_component: Any,
    hls_stream: Callable[[Stream], Awaitable[HlsClient]],
) -> None:
    """Test rendering the hls playlist with no output segments."""
    stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream.add_provider(HLS_PROVIDER)
    hls_client = await hls_stream(stream)
    resp = await hls_client.get("/playlist.m3u8")
    assert resp.status == HTTPStatus.NOT_FOUND


async def test_hls_playlist_view(
    hass: HomeAssistant,
    setup_component: Any,
    hls_stream: Callable[[Stream], Awaitable[HlsClient]],
    stream_worker_sync: Any,
) -> None:
    """Test rendering the hls playlist with 1 and 2 output segments."""
    stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream_worker_sync.pause()
    hls = stream.add_provider(HLS_PROVIDER)
    for i in range(2):
        segment = Segment(sequence=i, duration=SEGMENT_DURATION)
        hls.put(segment)
    await hass.async_block_till_done()
    hls_client = await hls_stream(stream)
    resp = await hls_client.get("/playlist.m3u8")
    assert resp.status == HTTPStatus.OK
    assert await resp.text() == make_playlist(
        sequence=0, segments=[make_segment(0), make_segment(1)]
    )
    segment = Segment(sequence=2, duration=SEGMENT_DURATION)
    hls.put(segment)
    await hass.async_block_till_done()
    resp = await hls_client.get("/playlist.m3u8")
    assert resp.status == HTTPStatus.OK
    assert await resp.text() == make_playlist(
        sequence=0,
        segments=[make_segment(0), make_segment(1), make_segment(2)],
    )
    stream_worker_sync.resume()
    await stream.stop()


async def test_hls_max_segments(
    hass: HomeAssistant,
    setup_component: Any,
    hls_stream: Callable[[Stream], Awaitable[HlsClient]],
    stream_worker_sync: Any,
) -> None:
    """Test rendering the hls playlist with more segments than the segment deque can hold."""
    stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream_worker_sync.pause()
    hls = stream.add_provider(HLS_PROVIDER)
    hls_client = await hls_stream(stream)
    for sequence in range(MAX_SEGMENTS + 1):
        segment = Segment(sequence=sequence, duration=SEGMENT_DURATION)
        hls.put(segment)
        await hass.async_block_till_done()
    resp = await hls_client.get("/playlist.m3u8")
    assert resp.status == HTTPStatus.OK
    start = MAX_SEGMENTS + 1 - NUM_PLAYLIST_SEGMENTS
    segments = [make_segment(sequence) for sequence in range(start, MAX_SEGMENTS + 1)]
    assert await resp.text() == make_playlist(sequence=start, segments=segments)
    for segment in hls.get_segments():
        segment.init = INIT_BYTES
        segment.parts = [Part(duration=SEGMENT_DURATION, has_keyframe=True, data=FAKE_PAYLOAD)]
    with patch.object(hls.stream_settings, "hls_part_timeout", 0.1):
        segment_response = await hls_client.get("/segment/0.m4s")
    assert segment_response.status == HTTPStatus.NOT_FOUND
    for sequence in range(1, MAX_SEGMENTS + 1):
        segment_response = await hls_client.get(f"/segment/{sequence}.m4s")
        assert segment_response.status == HTTPStatus.OK
    stream_worker_sync.resume()
    await stream.stop()


async def test_hls_playlist_view_discontinuity(
    hass: HomeAssistant,
    setup_component: Any,
    hls_stream: Callable[[Stream], Awaitable[HlsClient]],
    stream_worker_sync: Any,
) -> None:
    """Test a discontinuity across segments in the stream with 3 segments."""
    stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream_worker_sync.pause()
    hls = stream.add_provider(HLS_PROVIDER)
    segment = Segment(sequence=0, stream_id=0, duration=SEGMENT_DURATION)
    hls.put(segment)
    segment = Segment(sequence=1, stream_id=0, duration=SEGMENT_DURATION)
    hls.put(segment)
    segment = Segment(sequence=2, stream_id=1, duration=SEGMENT_DURATION)
    hls.put(segment)
    await hass.async_block_till_done()
    hls_client = await hls_stream(stream)
    resp = await hls_client.get("/playlist.m3u8")
    assert resp.status == HTTPStatus.OK
    assert await resp.text() == make_playlist(
        sequence=0,
        segments=[make_segment(0), make_segment(1), make_segment(2, discontinuity=True)],
    )
    stream_worker_sync.resume()
    await stream.stop()


async def test_hls_max_segments_discontinuity(
    hass: HomeAssistant,
    setup_component: Any,
    hls_stream: Callable[[Stream], Awaitable[HlsClient]],
    stream_worker_sync: Any,
) -> None:
    """Test a discontinuity with more segments than the segment deque can hold."""
    stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream_worker_sync.pause()
    hls = stream.add_provider(HLS_PROVIDER)
    hls_client = await hls_stream(stream)
    segment = Segment(sequence=0, stream_id=0, duration=SEGMENT_DURATION)
    hls.put(segment)
    for sequence in range(MAX_SEGMENTS + 1):
        segment = Segment(sequence=sequence, stream_id=1, duration=SEGMENT_DURATION)
        hls.put(segment)
    await hass.async_block_till_done()
    resp = await hls_client.get("/playlist.m3u8")
    assert resp.status == HTTPStatus.OK
    start = MAX_SEGMENTS + 1 - NUM_PLAYLIST_SEGMENTS
    segments = [make_segment(sequence) for sequence in range(start, MAX_SEGMENTS + 1)]
    assert await resp.text() == make_playlist(
        sequence=start, discontinuity_sequence=1, segments=segments
    )
    stream_worker_sync.resume()
    await stream.stop()


async def test_remove_incomplete_segment_on_exit(
    hass: HomeAssistant, setup_component: Any, stream_worker_sync: Any
) -> None:
    """Test that the incomplete segment gets removed when the worker thread quits."""
    stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream_worker_sync.pause()
    await stream.start()
    hls = stream.add_provider(HLS_PROVIDER)
    segment = Segment(sequence=0, stream_id=0, duration=SEGMENT_DURATION)
    hls.put(segment)
    segment = Segment(sequence=1, stream_id=0, duration=SEGMENT_DURATION)
    hls.put(segment)
    segment = Segment(sequence=2, stream_id=0, duration=0)
    hls.put(segment)
    await hass.async_block_till_done()
    segments = hls._segments
    assert len(segments) == 3
    assert not segments[-1].complete
    stream_worker_sync.resume()
    with patch("homeassistant.components.stream.Stream.remove_provider"):
        stream._thread_quit.set()
        stream._thread.join()
        stream._thread = None
        await hass.async_block_till_done()
        assert segments[-1].complete
        assert len(segments) == 2
    await stream.stop()


async def test_hls_stream_rotate(
    hass: HomeAssistant,
    setup_component: Any,
    hls_stream: Callable[[Stream], Awaitable[HlsClient]],
    stream_worker_sync: Any,
    h264_video: str,
) -> None:
    """Test hls stream with rotation applied.

    Purposefully not mocking anything here to test full
    integration with the stream component.
    """
    stream_worker_sync.pause()
    stream = create_stream(hass, h264_video, {}, dynamic_stream_settings())
    stream.add_provider(HLS_PROVIDER)
    await stream.start()
    hls_client = await hls_stream(stream)
    master_playlist_response = await hls_client.get()
    assert master_playlist_response.status == HTTPStatus.OK
    stream.dynamic_stream_settings.orientation = Orientation.ROTATE_LEFT
    init_response = await hls_client.get("/init.mp4")
    assert init_response.status == HTTPStatus.OK
    init = await init_response.read()
    stream_worker_sync.resume()
    assert_mp4_has_transform_matrix(
        init, stream.dynamic_stream_settings.orientation
    )
    await stream.stop()
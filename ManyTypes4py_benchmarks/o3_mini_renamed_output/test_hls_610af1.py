from __future__ import annotations
from datetime import timedelta, datetime
from http import HTTPStatus
from typing import Any, Callable, List, Optional, Awaitable
from urllib.parse import urlparse, ParseResult

import av
import pytest
from homeassistant.components.stream import Stream, create_stream
from homeassistant.components.stream.const import EXT_X_START_LL_HLS, EXT_X_START_NON_LL_HLS, HLS_PROVIDER, MAX_SEGMENTS, NUM_PLAYLIST_SEGMENTS
from homeassistant.components.stream.core import Orientation, Part
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from .common import FAKE_TIME, DefaultSegment as Segment, assert_mp4_has_transform_matrix, dynamic_stream_settings
from tests.common import async_fire_time_changed
from tests.typing import ClientSessionGenerator

STREAM_SOURCE: str = 'some-stream-source'
INIT_BYTES: bytes = b'\x00\x00\x00\x08moov'
FAKE_PAYLOAD: bytes = b'fake-payload'
SEGMENT_DURATION: float = 10.0
TEST_TIMEOUT: float = 5.0
MAX_ABORT_SEGMENTS: int = 20
HLS_CONFIG: dict[str, Any] = {'stream': {'ll_hls': False}}


@pytest.fixture
async def func_8hos21rh(hass: HomeAssistant) -> None:
    """Test fixture to setup the stream component."""
    await async_setup_component(hass, 'stream', HLS_CONFIG)


class HlsClient:
    """Test fixture for fetching the hls stream."""

    def __init__(self, http_client: Any, parsed_url: ParseResult) -> None:
        """Initialize HlsClient."""
        self.http_client: Any = http_client
        self.parsed_url: ParseResult = parsed_url

    async def func_wogu6is2(
        self, path: Optional[str] = None, headers: Optional[dict[str, str]] = None
    ) -> Any:
        """Fetch the hls stream for the specified path."""
        url: str = self.parsed_url.path
        if path:
            url = '/'.join(self.parsed_url.path.split('/')[:-1]) + path
        return await self.http_client.get(url, headers=headers)


@pytest.fixture
def func_mz7ekale(hass: HomeAssistant, hass_client: Callable[[], Awaitable[Any]]) -> Callable[[Stream], Awaitable[HlsClient]]:
    """Create test fixture for creating an HLS client for a stream."""

    async def func_9rxbd5x4(stream: Stream) -> HlsClient:
        http_client: Any = await hass_client()
        parsed_url: ParseResult = urlparse(stream.endpoint_url(HLS_PROVIDER))
        return HlsClient(http_client, parsed_url)

    return func_9rxbd5x4


def func_sfepvqb4(segment: int, discontinuity: bool = False) -> str:
    """Create a playlist response for a segment."""
    response: List[str] = ['#EXT-X-DISCONTINUITY'] if discontinuity else []
    response.extend([
        '#EXT-X-PROGRAM-DATE-TIME:' + FAKE_TIME.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
        f'#EXTINF:{SEGMENT_DURATION:.3f},',
        f'./segment/{segment}.m4s'
    ])
    return '\n'.join(response)


def func_vaozyqx3(
    sequence: int,
    discontinuity_sequence: int = 0,
    segments: Optional[List[str]] = None,
    hint: Optional[str] = None,
    segment_duration: Optional[float] = None,
    part_target_duration: Optional[float] = None,
) -> str:
    """Create an hls playlist response for tests to assert on."""
    if segment_duration is None:
        segment_duration = SEGMENT_DURATION
    response: List[str] = [
        '#EXTM3U',
        '#EXT-X-VERSION:6',
        '#EXT-X-INDEPENDENT-SEGMENTS',
        '#EXT-X-MAP:URI="init.mp4"',
        f'#EXT-X-TARGETDURATION:{segment_duration}',
        f'#EXT-X-MEDIA-SEQUENCE:{sequence}',
        f'#EXT-X-DISCONTINUITY-SEQUENCE:{discontinuity_sequence}'
    ]
    if hint:
        # part_target_duration is assumed to be provided if hint is given.
        assert part_target_duration is not None
        response.extend([
            f'#EXT-X-PART-INF:PART-TARGET={part_target_duration:.3f}',
            f'#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK={2 * part_target_duration:.3f}',
            f'#EXT-X-START:TIME-OFFSET=-{EXT_X_START_LL_HLS * part_target_duration:.3f},PRECISE=YES'
        ])
    else:
        response.append(f'#EXT-X-START:TIME-OFFSET=-{EXT_X_START_NON_LL_HLS * segment_duration:.3f},PRECISE=YES')
    if segments:
        response.extend(segments)
    if hint:
        response.append(hint)
    response.append('')
    return '\n'.join(response)


async def func_b46zer5l(
    hass: HomeAssistant,
    setup_component: Any,
    hls_stream: Any,
    stream_worker_sync: Any,
    h264_video: Any
) -> None:
    """Test hls stream.

    Purposefully not mocking anything here to test full
    integration with the stream component.
    """
    stream_worker_sync.pause()
    stream: Stream = create_stream(hass, h264_video, {}, dynamic_stream_settings())
    stream.add_provider(HLS_PROVIDER)
    await stream.start()
    hls_client: HlsClient = await func_mz7ekale(hass, lambda: None)(stream)  # type: ignore
    master_playlist_response: Any = await hls_client.func_wogu6is2()
    assert master_playlist_response.status == HTTPStatus.OK
    master_playlist: str = await master_playlist_response.text()
    init_response: Any = await hls_client.func_wogu6is2('/init.mp4')
    assert init_response.status == HTTPStatus.OK
    playlist_url: str = '/' + master_playlist.splitlines()[-1]
    playlist_response: Any = await hls_client.func_wogu6is2(playlist_url)
    assert playlist_response.status == HTTPStatus.OK
    playlist: str = await playlist_response.text()
    segment_url: str = '/' + [line for line in playlist.splitlines() if line][-1]
    segment_response: Any = await hls_client.func_wogu6is2(segment_url)
    assert segment_response.status == HTTPStatus.OK
    stream_worker_sync.resume()
    await stream.stop()
    fail_response: Any = await hls_client.func_wogu6is2()
    assert fail_response.status == HTTPStatus.NOT_FOUND
    diagnostics: dict[str, Any] = stream.get_diagnostics()
    assert diagnostics == {
        'container_format': 'mov,mp4,m4a,3gp,3g2,mj2',
        'keepalive': False,
        'orientation': Orientation.NO_TRANSFORM,
        'start_worker': 1,
        'video_codec': 'h264',
        'worker_error': 1,
    }


async def func_yvhwp5fm(
    hass: HomeAssistant,
    hass_client: Callable[[], Awaitable[Any]],
    setup_component: Any,
    stream_worker_sync: Any,
    h264_video: Any
) -> None:
    """Test hls stream timeout."""
    stream_worker_sync.pause()
    stream: Stream = create_stream(hass, h264_video, {}, dynamic_stream_settings())
    available_states: List[bool] = []

    def update_callback() -> None:
        nonlocal available_states
        available_states.append(stream.available)

    stream.set_update_callback(update_callback)
    stream.add_provider(HLS_PROVIDER)
    await stream.start()
    url: str = stream.endpoint_url(HLS_PROVIDER)
    http_client: Any = await hass_client()
    parsed_url: ParseResult = urlparse(url)
    playlist_response: Any = await http_client.get(parsed_url.path)
    assert playlist_response.status == HTTPStatus.OK
    future: datetime = dt_util.utcnow() + timedelta(minutes=1)
    async_fire_time_changed(hass, future)
    await hass.async_block_till_done()
    playlist_response = await http_client.get(parsed_url.path)
    assert playlist_response.status == HTTPStatus.OK
    stream_worker_sync.resume()
    future = dt_util.utcnow() + timedelta(minutes=5)
    async_fire_time_changed(hass, future)
    await hass.async_block_till_done()
    fail_response: Any = await http_client.get(parsed_url.path)
    assert fail_response.status == HTTPStatus.NOT_FOUND
    assert available_states == [True]


async def func_4quur94j(
    hass: HomeAssistant,
    hass_client: Callable[[], Awaitable[Any]],
    setup_component: Any,
    stream_worker_sync: Any,
    h264_video: Any
) -> None:
    """Test hls stream timeout after the stream has been stopped already."""
    stream_worker_sync.pause()
    stream: Stream = create_stream(hass, h264_video, {}, dynamic_stream_settings())
    stream.add_provider(HLS_PROVIDER)
    await stream.start()
    stream_worker_sync.resume()
    await stream.stop()
    future: datetime = dt_util.utcnow() + timedelta(minutes=5)
    async_fire_time_changed(hass, future)
    await hass.async_block_till_done()


@pytest.mark.parametrize('exception', [av.error.InvalidDataError(-2, 'error'), av.HTTPBadRequestError(500, 'error')])
async def func_nqe3cudw(
    hass: HomeAssistant,
    setup_component: Any,
    should_retry: Any,
    exception: Exception
) -> None:
    """Test hls stream is retried on failure."""
    source: str = 'test_stream_keepalive_source'
    stream: Stream = create_stream(hass, source, {}, dynamic_stream_settings())
    track: Any = stream.add_provider(HLS_PROVIDER)
    track.num_segments = 2
    available_states: List[bool] = []

    def update_callback() -> None:
        nonlocal available_states
        available_states.append(stream.available)

    stream.set_update_callback(update_callback)
    open_future1: Any = hass.loop.create_future()
    open_future2: Any = hass.loop.create_future()
    futures: List[Any] = [open_future2, open_future1]
    original_set_state: Callable[[Stream, bool], None] = Stream._set_state  # type: ignore

    def func_whxzq999(self: Stream, state: bool) -> None:
        if state is False:
            should_retry.return_value = False
        original_set_state(self, state)

    def func_958r8yua(*args: Any, **kwargs: Any) -> Any:
        hass.loop.call_soon_threadsafe(futures.pop().set_result, None)
        raise exception

    # Note: set_state_wrapper and av_open_side_effect should be defined.
    from unittest.mock import patch
    with patch('av.open') as av_open, patch('homeassistant.components.stream.Stream._set_state', new=func_whxzq999), patch(
        'homeassistant.components.stream.STREAM_RESTART_INCREMENT', 0
    ):
        av_open.side_effect = func_958r8yua
        should_retry.return_value = True
        await stream.start()
        await open_future1
        await open_future2
        await hass.async_add_executor_job(stream._thread.join)  # type: ignore
        stream._thread = None  # type: ignore
        assert av_open.call_count == 2
        await hass.async_block_till_done()
    await stream.stop()
    assert available_states == [True, False, True]


async def func_d7f109ah(
    hass: HomeAssistant,
    setup_component: Any,
    hls_stream: Any
) -> None:
    """Test rendering the hls playlist with no output segments."""
    stream: Stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream.add_provider(HLS_PROVIDER)
    hls_client: HlsClient = await func_mz7ekale(hass, lambda: None)(stream)  # type: ignore
    resp: Any = await hls_client.func_wogu6is2('/playlist.m3u8')
    assert resp.status == HTTPStatus.NOT_FOUND


async def func_xf6fqje2(
    hass: HomeAssistant,
    setup_component: Any,
    hls_stream: Any,
    stream_worker_sync: Any
) -> None:
    """Test rendering the hls playlist with 1 and 2 output segments."""
    stream: Stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream_worker_sync.pause()
    hls: Any = stream.add_provider(HLS_PROVIDER)
    for i in range(2):
        segment: Segment = Segment(sequence=i, duration=SEGMENT_DURATION)
        hls.put(segment)
    await hass.async_block_till_done()
    hls_client: HlsClient = await func_mz7ekale(hass, lambda: None)(stream)  # type: ignore
    resp: Any = await hls_client.func_wogu6is2('/playlist.m3u8')
    assert resp.status == HTTPStatus.OK
    text1: str = await resp.text()
    assert text1 == func_vaozyqx3(sequence=0, segments=[func_sfepvqb4(0), func_sfepvqb4(1)])
    segment = Segment(sequence=2, duration=SEGMENT_DURATION)
    hls.put(segment)
    await hass.async_block_till_done()
    resp = await hls_client.func_wogu6is2('/playlist.m3u8')
    assert resp.status == HTTPStatus.OK
    text2: str = await resp.text()
    assert text2 == func_vaozyqx3(sequence=0, segments=[func_sfepvqb4(0), func_sfepvqb4(1), func_sfepvqb4(2)])
    stream_worker_sync.resume()
    await stream.stop()


async def func_7o8cm8hl(
    hass: HomeAssistant,
    setup_component: Any,
    hls_stream: Any,
    stream_worker_sync: Any
) -> None:
    """Test rendering the hls playlist with more segments than the segment deque can hold."""
    stream: Stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream_worker_sync.pause()
    hls: Any = stream.add_provider(HLS_PROVIDER)
    hls_client: HlsClient = await func_mz7ekale(hass, lambda: None)(stream)  # type: ignore
    for sequence in range(MAX_SEGMENTS + 1):
        segment: Segment = Segment(sequence=sequence, duration=SEGMENT_DURATION)
        hls.put(segment)
        await hass.async_block_till_done()
    resp: Any = await hls_client.func_wogu6is2('/playlist.m3u8')
    assert resp.status == HTTPStatus.OK
    start: int = MAX_SEGMENTS + 1 - NUM_PLAYLIST_SEGMENTS
    segments: List[str] = [func_sfepvqb4(sequence) for sequence in range(start, MAX_SEGMENTS + 1)]
    text: str = await resp.text()
    assert text == func_vaozyqx3(sequence=start, segments=segments)
    for segment in hls.get_segments():
        segment.init = INIT_BYTES
        segment.parts = [Part(duration=SEGMENT_DURATION, has_keyframe=True, data=FAKE_PAYLOAD)]
    from unittest.mock import patch
    with patch.object(hls.stream_settings, 'hls_part_timeout', 0.1):
        segment_response: Any = await hls_client.func_wogu6is2('/segment/0.m4s')
    assert segment_response.status == HTTPStatus.NOT_FOUND
    for sequence in range(1, MAX_SEGMENTS + 1):
        segment_response = await hls_client.func_wogu6is2(f'/segment/{sequence}.m4s')
        assert segment_response.status == HTTPStatus.OK
    stream_worker_sync.resume()
    await stream.stop()


async def func_j60i47t5(
    hass: HomeAssistant,
    setup_component: Any,
    hls_stream: Any,
    stream_worker_sync: Any
) -> None:
    """Test a discontinuity across segments in the stream with 3 segments."""
    stream: Stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream_worker_sync.pause()
    hls: Any = stream.add_provider(HLS_PROVIDER)
    segment: Segment = Segment(sequence=0, stream_id=0, duration=SEGMENT_DURATION)
    hls.put(segment)
    segment = Segment(sequence=1, stream_id=0, duration=SEGMENT_DURATION)
    hls.put(segment)
    segment = Segment(sequence=2, stream_id=1, duration=SEGMENT_DURATION)
    hls.put(segment)
    await hass.async_block_till_done()
    hls_client: HlsClient = await func_mz7ekale(hass, lambda: None)(stream)  # type: ignore
    resp: Any = await hls_client.func_wogu6is2('/playlist.m3u8')
    assert resp.status == HTTPStatus.OK
    playlist_text: str = await resp.text()
    assert playlist_text == func_vaozyqx3(sequence=0, segments=[
        func_sfepvqb4(0), func_sfepvqb4(1), func_sfepvqb4(2, discontinuity=True)
    ])
    stream_worker_sync.resume()
    await stream.stop()


async def func_ljh1axvq(
    hass: HomeAssistant,
    setup_component: Any,
    hls_stream: Any,
    stream_worker_sync: Any
) -> None:
    """Test a discontinuity with more segments than the segment deque can hold."""
    stream: Stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream_worker_sync.pause()
    hls: Any = stream.add_provider(HLS_PROVIDER)
    hls_client: HlsClient = await func_mz7ekale(hass, lambda: None)(stream)  # type: ignore
    segment: Segment = Segment(sequence=0, stream_id=0, duration=SEGMENT_DURATION)
    hls.put(segment)
    for sequence in range(MAX_SEGMENTS + 1):
        segment = Segment(sequence=sequence, stream_id=1, duration=SEGMENT_DURATION)
        hls.put(segment)
    await hass.async_block_till_done()
    resp: Any = await hls_client.func_wogu6is2('/playlist.m3u8')
    assert resp.status == HTTPStatus.OK
    start: int = MAX_SEGMENTS + 1 - NUM_PLAYLIST_SEGMENTS
    segments: List[str] = [func_sfepvqb4(sequence) for sequence in range(start, MAX_SEGMENTS + 1)]
    playlist: str = await resp.text()
    assert playlist == func_vaozyqx3(sequence=start, discontinuity_sequence=1, segments=segments)
    stream_worker_sync.resume()
    await stream.stop()


async def func_02m7r5i3(
    hass: HomeAssistant,
    setup_component: Any,
    stream_worker_sync: Any
) -> None:
    """Test that the incomplete segment gets removed when the worker thread quits."""
    stream: Stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream_worker_sync.pause()
    await stream.start()
    hls: Any = stream.add_provider(HLS_PROVIDER)
    segment: Segment = Segment(sequence=0, stream_id=0, duration=SEGMENT_DURATION)
    hls.put(segment)
    segment = Segment(sequence=1, stream_id=0, duration=SEGMENT_DURATION)
    hls.put(segment)
    segment = Segment(sequence=2, stream_id=0, duration=0)
    hls.put(segment)
    await hass.async_block_till_done()
    segments = hls._segments  # type: ignore
    assert len(segments) == 3
    assert not segments[-1].complete
    stream_worker_sync.resume()
    from unittest.mock import patch
    with patch('homeassistant.components.stream.Stream.remove_provider'):
        stream._thread_quit.set()  # type: ignore
        stream._thread.join()  # type: ignore
        stream._thread = None  # type: ignore
        await hass.async_block_till_done()
        assert segments[-1].complete
        assert len(segments) == 2
    await stream.stop()


async def func_iee7vp41(
    hass: HomeAssistant,
    setup_component: Any,
    hls_stream: Any,
    stream_worker_sync: Any,
    h264_video: Any
) -> None:
    """Test hls stream with rotation applied.

    Purposefully not mocking anything here to test full
    integration with the stream component.
    """
    stream_worker_sync.pause()
    stream: Stream = create_stream(hass, h264_video, {}, dynamic_stream_settings())
    stream.add_provider(HLS_PROVIDER)
    await stream.start()
    hls_client: HlsClient = await func_mz7ekale(hass, lambda: None)(stream)  # type: ignore
    master_playlist_response: Any = await hls_client.func_wogu6is2()
    assert master_playlist_response.status == HTTPStatus.OK
    stream.dynamic_stream_settings.orientation = Orientation.ROTATE_LEFT
    init_response: Any = await hls_client.func_wogu6is2('/init.mp4')
    assert init_response.status == HTTPStatus.OK
    init: bytes = await init_response.read()
    stream_worker_sync.resume()
    assert_mp4_has_transform_matrix(init, stream.dynamic_stream_settings.orientation)
    await stream.stop()
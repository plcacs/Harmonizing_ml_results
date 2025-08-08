"""The tests for hls streams."""
from datetime import timedelta
from http import HTTPStatus
from typing import Any, AsyncGenerator, Callable, List, Optional, Tuple, cast
from unittest.mock import patch
from urllib.parse import ParseResult, urlparse
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
SEGMENT_DURATION: int = 10
TEST_TIMEOUT: float = 5.0
MAX_ABORT_SEGMENTS: int = 20
HLS_CONFIG: dict[str, Any] = {'stream': {'ll_hls': False}}


@pytest.fixture
async def func_8hos21rh(hass: HomeAssistant) -> AsyncGenerator[None, None]:
    """Test fixture to setup the stream component."""
    await async_setup_component(hass, 'stream', HLS_CONFIG)
    yield


class HlsClient:
    """Test fixture for fetching the hls stream."""

    def __init__(self, http_client: ClientSessionGenerator, parsed_url: ParseResult) -> None:
        """Initialize HlsClient."""
        self.http_client = http_client
        self.parsed_url = parsed_url

    async def func_wogu6is2(self, path: Optional[str] = None, headers: Optional[dict[str, str]] = None) -> Any:
        """Fetch the hls stream for the specified path."""
        url = self.parsed_url.path
        if path:
            url = '/'.join(self.parsed_url.path.split('/')[:-1]) + path
        return await self.http_client.get(url, headers=headers)


@pytest.fixture
def func_mz7ekale(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> Callable[[Stream], Any]:
    """Create test fixture for creating an HLS client for a stream."""

    async def func_9rxbd5x4(stream: Stream) -> HlsClient:
        http_client = await hass_client()
        parsed_url = urlparse(stream.endpoint_url(HLS_PROVIDER))
        return HlsClient(http_client, parsed_url)
    return func_9rxbd5x4


def func_sfepvqb4(segment: int, discontinuity: bool = False) -> str:
    """Create a playlist response for a segment."""
    response = ['#EXT-X-DISCONTINUITY'] if discontinuity else []
    response.extend(['#EXT-X-PROGRAM-DATE-TIME:' + FAKE_TIME.strftime(
        '%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
        f'#EXTINF:{SEGMENT_DURATION:.3f},', f'./segment/{segment}.m4s'])
    return '\n'.join(response)


def func_vaozyqx3(sequence: int, discontinuity_sequence: int = 0, segments: Optional[List[str]] = None,
                 hint: Optional[str] = None, segment_duration: Optional[float] = None,
                 part_target_duration: Optional[float] = None) -> str:
    """Create a an hls playlist response for tests to assert on."""
    if not segment_duration:
        segment_duration = SEGMENT_DURATION
    response = ['#EXTM3U', '#EXT-X-VERSION:6',
        '#EXT-X-INDEPENDENT-SEGMENTS', '#EXT-X-MAP:URI="init.mp4"',
        f'#EXT-X-TARGETDURATION:{segment_duration}',
        f'#EXT-X-MEDIA-SEQUENCE:{sequence}',
        f'#EXT-X-DISCONTINUITY-SEQUENCE:{discontinuity_sequence}']
    if hint:
        response.extend([
            f'#EXT-X-PART-INF:PART-TARGET={part_target_duration:.3f}',
            f'#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK={2 * part_target_duration:.3f}'
            ,
            f'#EXT-X-START:TIME-OFFSET=-{EXT_X_START_LL_HLS * part_target_duration:.3f},PRECISE=YES'
            ])
    else:
        response.append(
            f'#EXT-X-START:TIME-OFFSET=-{EXT_X_START_NON_LL_HLS * segment_duration:.3f},PRECISE=YES'
            )
    if segments:
        response.extend(segments)
    if hint:
        response.append(hint)
    response.append('')
    return '\n'.join(response)


async def func_b46zer5l(hass: HomeAssistant, setup_component: Any, hls_stream: Any,
                       stream_worker_sync: Any, h264_video: Any) -> None:
    """Test hls stream."""
    stream_worker_sync.pause()
    stream = create_stream(hass, h264_video, {}, dynamic_stream_settings())
    stream.add_provider(HLS_PROVIDER)
    await stream.start()
    hls_client = await func_mz7ekale(stream)
    master_playlist_response = await hls_client.func_wogu6is2()
    assert master_playlist_response.status == HTTPStatus.OK
    master_playlist = await master_playlist_response.text()
    init_response = await hls_client.func_wogu6is2('/init.mp4')
    assert init_response.status == HTTPStatus.OK
    playlist_url = '/' + master_playlist.splitlines()[-1]
    playlist_response = await hls_client.func_wogu6is2(playlist_url)
    assert playlist_response.status == HTTPStatus.OK
    playlist = await playlist_response.text()
    segment_url = '/' + [line for line in playlist.splitlines() if line][-1]
    segment_response = await hls_client.func_wogu6is2(segment_url)
    assert segment_response.status == HTTPStatus.OK
    stream_worker_sync.resume()
    await stream.stop()
    fail_response = await hls_client.func_wogu6is2()
    assert fail_response.status == HTTPStatus.NOT_FOUND
    assert stream.get_diagnostics() == {'container_format':
        'mov,mp4,m4a,3gp,3g2,mj2', 'keepalive': False, 'orientation':
        Orientation.NO_TRANSFORM, 'start_worker': 1, 'video_codec': 'h264',
        'worker_error': 1}


async def func_yvhwp5fm(hass: HomeAssistant, hass_client: ClientSessionGenerator, setup_component: Any,
                       stream_worker_sync: Any, h264_video: Any) -> None:
    """Test hls stream timeout."""
    stream_worker_sync.pause()
    stream = create_stream(hass, h264_video, {}, dynamic_stream_settings())
    available_states: List[bool] = []

    def func_y9fswkxs() -> None:
        nonlocal available_states
        available_states.append(stream.available)
    stream.set_update_callback(func_y9fswkxs)
    stream.add_provider(HLS_PROVIDER)
    await stream.start()
    url = stream.endpoint_url(HLS_PROVIDER)
    http_client = await hass_client()
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


async def func_4quur94j(hass: HomeAssistant, hass_client: ClientSessionGenerator, setup_component: Any,
                        stream_worker_sync: Any, h264_video: Any) -> None:
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


@pytest.mark.parametrize('exception', [av.error.InvalidDataError(-2,
    'error'), av.HTTPBadRequestError(500, 'error')])
async def func_nqe3cudw(hass: HomeAssistant, setup_component: Any, should_retry: Any, exception: Exception) -> None:
    """Test hls stream is retried on failure."""
    source = 'test_stream_keepalive_source'
    stream = create_stream(hass, source, {}, dynamic_stream_settings())
    track = stream.add_provider(HLS_PROVIDER)
    track.num_segments = 2
    available_states: List[bool] = []

    def func_y9fswkxs() -> None:
        nonlocal available_states
        available_states.append(stream.available)
    stream.set_update_callback(func_y9fswkxs)
    open_future1 = hass.loop.create_future()
    open_future2 = hass.loop.create_future()
    futures = [open_future2, open_future1]
    original_set_state = Stream._set_state

    def func_whxzq999(self: Any, state: bool) -> None:
        if state is False:
            should_retry.return_value = False
        original_set_state(self, state)

    def func_958r8yua(*args: Any, **kwargs: Any) -> None:
        hass.loop.call_soon_threadsafe(futures.pop().set_result, None)
        raise exception
    with patch('av.open') as av_open, patch(
        'homeassistant.components.stream.Stream._set_state', func_whxzq999
        ), patch('homeassistant.components.stream.STREAM_RESTART_INCREMENT', 0
        ):
        av_open.side_effect = func_958r8yua
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


async def func_d7f109ah(hass: HomeAssistant, setup_component: Any, hls_stream: Any) -> None:
    """Test rendering the hls playlist with no output segments."""
    stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream.add_provider(HLS_PROVIDER)
    hls_client = await func_mz7ekale(stream)
    resp = await hls_client.func_wogu6is2('/playlist.m3u8')
    assert resp.status == HTTPStatus.NOT_FOUND


async def func_xf6fqje2(hass: HomeAssistant, setup_component: Any, hls_stream: Any, stream_worker_sync: Any) -> None:
    """Test rendering the hls playlist with 1 and 2 output segments."""
    stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream_worker_sync.pause()
    hls = stream.add_provider(HLS_PROVIDER)
    for i in range(2):
        segment = Segment(sequence=i, duration=SEGMENT_DURATION)
        hls.put(segment)
    await hass.async_block_till_done()
    hls_client = await func_mz7ekale(stream)
    resp = await hls_client.func_wogu6is2('/playlist.m3u8')
    assert resp.status == HTTPStatus.OK
    assert await resp.text() == func_vaozyqx3(sequence=0, segments=[
        func_sfepvqb4(0), func_sfepvqb4(1)])
    segment = Segment(sequence=2, duration=SEGMENT_DURATION)
    hls.put(segment)
    await hass.async_block_till_done()
    resp = await hls_client.func_wogu6is2('/playlist.m3u8')
    assert resp.status == HTTPStatus.OK
    assert await resp.text() == func_vaozyqx3(sequence=0, segments=[
        func_sfepvqb4(0), func_sfepvqb4(1), func_sfepvqb4(2)])
    stream_worker_sync.resume()
    await stream.stop()


async def func_7o8cm8hl(hass: HomeAssistant, setup_component: Any, hls_stream: Any, stream_worker_sync: Any) -> None:
    """Test rendering the hls playlist with more segments than the segment deque can hold."""
    stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream_worker_sync.pause()
    hls = stream.add_provider(HLS_PROVIDER)
    hls_client = await func_mz7ekale(stream)
    for sequence in range(MAX_SEGMENTS + 1):
        segment = Segment(sequence=sequence, duration=SEGMENT_DURATION)
        hls.put(segment)
        await hass.async_block_till_done()
    resp = await hls_client.func_wogu6is2('/playlist.m3u8')
    assert resp.status == HTTPStatus.OK
    start = MAX_SEGMENTS + 1 - NUM_PLAYLIST_SEGMENTS
    segments = [func_sfepvqb4(sequence) for sequence in range(start, 
        MAX_SEGMENTS + 1)]
    assert await resp.text() == func_vaozyqx3(sequence=start, segments=segments
        )
    for segment in hls.get_segments():
        segment.init = INIT_BYTES
        segment.parts = [Part(duration=SEGMENT_DURATION, has_keyframe=True,
            data=FAKE_PAYLOAD)]
    with patch.object(hls.stream_settings, 'hls_part_timeout', 0.1):
        segment_response = await hls_client.func_wogu6is2('/segment/0.m4s')
    assert segment_response.status == HTTPStatus.NOT_FOUND
    for sequence in range(1, MAX_SEGMENTS + 1):
        segment_response = await hls_client.func_wogu6is2(f'/segment/{sequence}.m4s')
        assert segment_response.status == HTTPStatus.OK
    stream_worker_sync.resume()
    await stream.stop()


async def func_j60i47t5(hass: HomeAssistant, setup_component: Any, hls_stream: Any, stream_worker_sync: Any) -> None:
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
    hls_client = await func_mz7ekale(stream)
    resp = await hls_client.func_wogu6is2('/playlist.m3u8')
    assert resp.status == HTTPStatus.OK
    assert await resp.text() == func_vaozyqx3(sequence=0, segments=[
        func_sfepvqb4(0), func_sfepvqb4(1), func_sfepvqb4(2, discontinuity=
        True)])
    stream_worker_sync.resume()
    await stream.stop()


async def func_ljh1axvq(hass: HomeAssistant, setup_component: Any, hls_stream: Any, stream_worker_sync: Any) -> None:
    """Test a discontinuity with more segments than the segment deque can hold."""
    stream = create_stream(hass, STREAM_SOURCE, {}, dynamic_stream_settings())
    stream_worker_sync.pause()
    hls = stream.add_provider(HLS_PROVIDER)
    hls_client = await func_mz7ekale(stream)
    segment = Segment(sequence=0, stream_id=0, duration=SEGMENT_DURATION)
    hls.put(segment)
    for sequence in range(MAX_SEGMENTS + 1):
        segment = Segment(sequence=sequence, stream_id=1, duration=
            SEGMENT_DURATION)
        hls.put(segment)
    await hass.async_block_till_done()
    resp = await hls_client.func_wogu6is2('/playlist.m3u8')
    assert resp.status == HTTPStatus.OK
    start = MAX_SEGMENTS + 1 - NUM_PLAYLIST_SEGMENTS

from typing import Any, Coroutine, Dict, List, Tuple
from aiohttp import ClientResponse
from homeassistant.core import HomeAssistant
from homeassistant.components.stream import Stream
from homeassistant.components.stream.const import HLS_PROVIDER
from homeassistant.components.stream.core import Orientation, Part
from homeassistant.util import dt as dt_util

async def test_hls_stream(hass: HomeAssistant, setup_component: Coroutine, hls_stream: Coroutine, stream_worker_sync: Any, h264_video: Any) -> None:
async def test_stream_timeout(hass: HomeAssistant, hass_client: Coroutine, setup_component: Coroutine, stream_worker_sync: Any, h264_video: Any) -> None:
async def test_stream_timeout_after_stop(hass: HomeAssistant, hass_client: Coroutine, setup_component: Coroutine, stream_worker_sync: Any, h264_video: Any) -> None:
async def test_stream_retries(hass: HomeAssistant, setup_component: Coroutine, should_retry: Any, exception: Any) -> None:
async def test_hls_playlist_view_no_output(hass: HomeAssistant, setup_component: Coroutine, hls_stream: Coroutine) -> None:
async def test_hls_playlist_view(hass: HomeAssistant, setup_component: Coroutine, hls_stream: Coroutine, stream_worker_sync: Any) -> None:
async def test_hls_max_segments(hass: HomeAssistant, setup_component: Coroutine, hls_stream: Coroutine, stream_worker_sync: Any) -> None:
async def test_hls_playlist_view_discontinuity(hass: HomeAssistant, setup_component: Coroutine, hls_stream: Coroutine, stream_worker_sync: Any) -> None:
async def test_hls_max_segments_discontinuity(hass: HomeAssistant, setup_component: Coroutine, hls_stream: Coroutine, stream_worker_sync: Any) -> None:
async def test_remove_incomplete_segment_on_exit(hass: HomeAssistant, setup_component: Coroutine, stream_worker_sync: Any) -> None:
async def test_hls_stream_rotate(hass: HomeAssistant, setup_component: Coroutine, hls_stream: Coroutine, stream_worker_sync: Any, h264_video: Any) -> None:

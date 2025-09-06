from datetime import timedelta
from http import HTTPStatus
from unittest.mock import patch
from urllib.parse import urlparse
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
HLS_CONFIG: dict = {'stream': {'ll_hls': False}}

@pytest.fixture
async def func_8hos21rh(hass: HomeAssistant) -> None:
    ...

class HlsClient:
    def __init__(self, http_client, parsed_url):
        ...

    async def func_wogu6is2(self, path=None, headers=None):
        ...

@pytest.fixture
def func_mz7ekale(hass: HomeAssistant, hass_client: ClientSessionGenerator) -> None:
    ...

def func_sfepvqb4(segment, discontinuity=False) -> str:
    ...

def func_vaozyqx3(sequence, discontinuity_sequence=0, segments=None, hint=None, segment_duration=None, part_target_duration=None) -> str:
    ...

async def func_b46zer5l(hass: HomeAssistant, setup_component, hls_stream, stream_worker_sync, h264_video) -> None:
    ...

async def func_yvhwp5fm(hass: HomeAssistant, hass_client: ClientSessionGenerator, setup_component, stream_worker_sync, h264_video) -> None:
    ...

async def func_4quur94j(hass: HomeAssistant, hass_client: ClientSessionGenerator, setup_component, stream_worker_sync, h264_video) -> None:
    ...

@pytest.mark.parametrize('exception', [av.error.InvalidDataError(-2, 'error'), av.HTTPBadRequestError(500, 'error')])
async def func_nqe3cudw(hass: HomeAssistant, setup_component, should_retry, exception) -> None:
    ...

async def func_d7f109ah(hass: HomeAssistant, setup_component, hls_stream) -> None:
    ...

async def func_xf6fqje2(hass: HomeAssistant, setup_component, hls_stream, stream_worker_sync) -> None:
    ...

async def func_7o8cm8hl(hass: HomeAssistant, setup_component, hls_stream, stream_worker_sync) -> None:
    ...

async def func_ljh1axvq(hass: HomeAssistant, setup_component, hls_stream, stream_worker_sync) -> None:
    ...

async def func_02m7r5i3(hass: HomeAssistant, setup_component, stream_worker_sync) -> None:
    ...

async def func_iee7vp41(hass: HomeAssistant, setup_component, hls_stream, stream_worker_sync, h264_video) -> None:
    ...

"""Test for Nest Media Source.

These tests simulate recent camera events received by the subscriber exposed
as media in the media source.
"""
from collections.abc import Generator
import datetime
from http import HTTPStatus
import io
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import patch
import aiohttp
import av
import numpy as np
import pytest
from homeassistant.components.media_player import BrowseError
from homeassistant.components.media_source import URI_SCHEME, Unresolvable, async_browse_media, async_resolve_media
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.template import DATE_STR_FORMAT
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from .common import DEVICE_ID, TEST_CLIP_URL, TEST_IMAGE_URL, CreateDevice, create_nest_event
from .conftest import FakeAuth
from tests.common import MockUser, async_capture_events
from tests.typing import ClientSessionGenerator

DOMAIN: str = 'nest'
DEVICE_NAME: str = 'Front'
PLATFORM: str = 'camera'
NEST_EVENT: str = 'nest_event'
EVENT_ID: str = '1aXEvi9ajKVTdDsXdJda8fzfCa'
EVENT_SESSION_ID: str = 'CjY5Y3VKaTZwR3o4Y19YbTVfMF'
CAMERA_DEVICE_TYPE: str = 'sdm.devices.types.CAMERA'
CAMERA_TRAITS: Dict[str, Any] = {
    'sdm.devices.traits.Info': {'customName': DEVICE_NAME},
    'sdm.devices.traits.CameraImage': {},
    'sdm.devices.traits.CameraLiveStream': {'supportedProtocols': ['RTSP']},
    'sdm.devices.traits.CameraEventImage': {},
    'sdm.devices.traits.CameraPerson': {},
    'sdm.devices.traits.CameraMotion': {}
}
BATTERY_CAMERA_TRAITS: Dict[str, Any] = {
    'sdm.devices.traits.Info': {'customName': DEVICE_NAME},
    'sdm.devices.traits.CameraClipPreview': {},
    'sdm.devices.traits.CameraLiveStream': {'supportedProtocols': ['WEB_RTC']},
    'sdm.devices.traits.CameraPerson': {},
    'sdm.devices.traits.CameraMotion': {}
}
PERSON_EVENT: str = 'sdm.devices.events.CameraPerson.Person'
MOTION_EVENT: str = 'sdm.devices.events.CameraMotion.Motion'
GENERATE_IMAGE_URL_RESPONSE: Dict[str, Any] = {'results': {'url': TEST_IMAGE_URL, 'token': 'g.0.eventToken'}}
IMAGE_BYTES_FROM_EVENT: bytes = b'test url image bytes'
IMAGE_AUTHORIZATION_HEADERS: Dict[str, str] = {'Authorization': 'Basic g.0.eventToken'}

def frame_image_data(frame_i: int, total_frames: int) -> np.ndarray:
    """Generate image content for a frame of a video."""
    img = np.empty((480, 320, 3))
    img[:, :, 0] = 0.5 + 0.5 * np.sin(2 * np.pi * (0 / 3 + frame_i / total_frames))
    img[:, :, 1] = 0.5 + 0.5 * np.sin(2 * np.pi * (1 / 3 + frame_i / total_frames))
    img[:, :, 2] = 0.5 + 0.5 * np.sin(2 * np.pi * (2 / 3 + frame_i / total_frames))
    img = np.round(255 * img).astype(np.uint8)
    return np.clip(img, 0, 255)

@pytest.fixture
def platforms() -> List[str]:
    """Fixture for platforms to setup."""
    return [PLATFORM]

@pytest.fixture(autouse=True)
async def setup_components(hass: HomeAssistant) -> None:
    """Fixture to initialize the integration."""
    await async_setup_component(hass, 'media_source', {})

@pytest.fixture
def device_type() -> str:
    """Fixture for the type of device under test."""
    return CAMERA_DEVICE_TYPE

@pytest.fixture
def device_traits() -> Dict[str, Any]:
    """Fixture for the present traits of the device under test."""
    return CAMERA_TRAITS

@pytest.fixture(autouse=True)
def device(device_type: str, device_traits: Dict[str, Any], create_device: CreateDevice) -> Dict[str, Any]:
    """Fixture to create a device under test."""
    return create_device.create(raw_data={'name': DEVICE_ID, 'type': device_type, 'traits': device_traits})

@pytest.fixture
def mp4() -> io.BytesIO:
    """Generate test mp4 clip."""
    total_frames = 10
    fps = 10
    output = io.BytesIO()
    output.name = 'test.mp4'
    container = av.open(output, mode='w', format='mp4')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = 480
    stream.height = 320
    stream.pix_fmt = 'yuv420p'
    for frame_i in range(total_frames):
        img = frame_image_data(frame_i, total_frames)
        frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    output.seek(0)
    return output

@pytest.fixture
def cache_size() -> int:
    """Fixture for overrideing cache size."""
    return 100

@pytest.fixture(autouse=True)
def apply_cache_size(cache_size: int) -> Generator[None, None, None]:
    """Fixture for patching the cache size."""
    with patch('homeassistant.components.nest.EVENT_MEDIA_CACHE_SIZE', new=cache_size):
        yield

def create_event(event_session_id: str, event_id: str, event_type: str, timestamp: Optional[datetime.datetime] = None, device_id: Optional[str] = None) -> Dict[str, Any]:
    """Create an EventMessage for a single event type."""
    if not timestamp:
        timestamp = dt_util.now()
    event_data = {event_type: {'eventSessionId': event_session_id, 'eventId': event_id}}
    return create_event_message(event_data, timestamp, device_id=device_id)

def create_event_message(event_data: Dict[str, Any], timestamp: datetime.datetime, device_id: Optional[str] = None) -> Dict[str, Any]:
    """Create an EventMessage for a single event type."""
    if device_id is None:
        device_id = DEVICE_ID
    return create_nest_event({
        'eventId': f'{EVENT_ID}-{timestamp}',
        'timestamp': timestamp.isoformat(timespec='seconds'),
        'resourceUpdate': {
            'name': device_id,
            'events': event_data
        }
    })

def create_battery_event_data(event_type: str, event_session_id: str = EVENT_SESSION_ID, event_id: str = 'n:2') -> Dict[str, Any]:
    """Return event payload data for a battery camera event."""
    return {
        event_type: {'eventSessionId': event_session_id, 'eventId': event_id},
        'sdm.devices.events.CameraClipPreview.ClipPreview': {
            'eventSessionId': event_session_id,
            'previewUrl': TEST_CLIP_URL
        }
    }

@pytest.mark.parametrize(('device_type', 'device_traits'), [('sdm.devices.types.THERMOSTAT', {'sdm.devices.traits.Temperature': {'ambientTemperatureCelsius': 22.0}})])
async def test_no_eligible_devices(hass: HomeAssistant, setup_platform: Any) -> None:
    """Test a media source with no eligible camera devices."""
    await setup_platform()
    browse = await async_browse_media(hass, f'{URI_SCHEME}{DOMAIN}')
    assert browse.domain == DOMAIN
    assert browse.identifier == ''
    assert browse.title == 'Nest'
    assert not browse.children

@pytest.mark.parametrize('device_traits', [CAMERA_TRAITS, BATTERY_CAMERA_TRAITS])
async def test_supported_device(hass: HomeAssistant, device_registry: dr.DeviceRegistry, setup_platform: Any) -> None:
    """Test a media source with a supported camera."""
    await setup_platform()
    assert len(hass.states.async_all()) == 1
    camera = hass.states.get('camera.front')
    assert camera is not None
    device = device_registry.async_get_device(identifiers={(DOMAIN, DEVICE_ID)})
    assert device
    assert device.name == DEVICE_NAME
    browse = await async_browse_media(hass, f'{URI_SCHEME}{DOMAIN}')
    assert browse.domain == DOMAIN
    assert browse.title == 'Nest'
    assert browse.identifier == ''
    assert browse.can_expand
    assert len(browse.children) == 1
    assert browse.children[0].domain == DOMAIN
    assert browse.children[0].identifier == device.id
    assert browse.children[0].title == 'Front: Recent Events'
    browse = await async_browse_media(hass, f'{URI_SCHEME}{DOMAIN}/{device.id}')
    assert browse.domain == DOMAIN
    assert browse.identifier == device.id
    assert browse.title == 'Front: Recent Events'
    assert len(browse.children) == 0

async def test_integration_unloaded(hass: HomeAssistant, auth: FakeAuth, setup_platform: Any) -> None:
    """Test the media player loads, but has no devices, when config unloaded."""
    await setup_platform()
    browse = await async_browse_media(hass, f'{URI_SCHEME}{DOMAIN}')
    assert browse.domain == DOMAIN
    assert browse.identifier == ''
    assert browse.title == 'Nest'
    assert len(browse.children) == 1
    entries = hass.config_entries.async_entries(DOMAIN)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.state is ConfigEntryState.LOADED
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert entry.state is ConfigEntryState.NOT_LOADED
    browse = await async_browse_media(hass, f'{URI_SCHEME}{DOMAIN}')
    assert browse.domain == DOMAIN
    assert browse.identifier == ''
    assert browse.title == 'Nest'
    assert len(browse.children) == 0

async def test_camera_event(hass: HomeAssistant, hass_client: ClientSessionGenerator, device_registry: dr.DeviceRegistry, subscriber: Any, auth: FakeAuth, setup_platform: Any) -> None:
    """Test a media source and image created for an event."""
    await setup_platform()
    assert len(hass.states.async_all()) == 1
    camera = hass.states.get('camera.front')
    assert camera is not None
    device = device_registry.async_get_device(identifiers={(DOMAIN, DEVICE_ID)})
    assert device
    assert device.name == DEVICE_NAME
    received_events = async_capture_events(hass, NEST_EVENT)
    auth.responses = [
        aiohttp.web.json_response(GENERATE_IMAGE_URL_RESPONSE),
        aiohttp.web.Response(body=IMAGE_BYTES_FROM_EVENT)
    ]
    event_timestamp = dt_util.now()
    await subscriber.async_receive_event(create_event(EVENT_SESSION_ID, EVENT_ID, PERSON_EVENT, timestamp=event_timestamp))
    await hass.async_block_till_done()
    assert len(received_events) == 1
    received_event = received_events[0]
    assert received_event.data['device_id'] == device.id
    assert received_event.data['type'] == 'camera_person'
    event_identifier = received_event.data['nest_event_id']
    browse = await async_browse_media(hass, f'{URI_SCHEME}{DOMAIN}')
    assert browse.title == 'Nest'
    assert browse.identifier == ''
    assert browse.can_expand
    assert len(browse.children) == 1
    assert browse.children[0].domain == DOMAIN
    assert browse.children[0].identifier == device.id
    assert browse.children[0].title == 'Front: Recent Events'
    assert browse.children[0].can_expand
    assert browse.children[0].can_play
    assert len(browse.children[0].children) == 0
    browse = await async_browse_media(hass, f'{URI_SCHEME}{DOMAIN}/{device.id}')
    assert browse.domain == DOMAIN
    assert browse.identifier == device.id
    assert browse.title == 'Front: Recent Events'
    assert browse.can_expand
    assert len(browse.children) == 1
    assert browse.children[0].domain == DOMAIN
    assert browse.children[0].identifier == f'{device.id}/{event_identifier}'
    event_timestamp_string = event_timestamp.strftime(DATE_STR_FORMAT)
    assert browse.children[0].title == f'Person @ {event_timestamp_string}'
    assert not browse.children[0].can_expand
    assert len(browse.children[0].children) == 0
    browse = await async_browse_media(hass, f'{URI_SCHEME}{DOMAIN}/{device.id}/{event_identifier}')
    assert browse.domain == DOMAIN
    assert browse.identifier == f'{device.id}/{event_identifier}'
    assert 'Person' in browse.title
    assert not browse.can_expand
    assert not browse.children
    assert not browse.can_play
    media = await async_resolve_media(hass, f'{URI_SCHEME}{DOMAIN}/{device.id}/{event_identifier}', None)
    assert media.url == f'/api/nest/event_media/{device.id}/{event_identifier}'
    assert media.mime_type == 'image/jpeg'
    client = await hass_client()
    response = await client.get(media.url)
    assert response.status == HTTPStatus.OK, f'Response not matched: {response}'
    contents = await response.read()
    assert contents == IMAGE_BYTES_FROM_EVENT
    media = await async_resolve_media(hass, f'{URI_SCHEME}{DOMAIN}/{device.id}', None)
    assert media.url == f'/api/nest/event_media/{device.id}/{event_identifier}'
    assert media.mime_type == 'image/jpeg'

async def test_event_order(hass: HomeAssistant, device_registry: dr.DeviceRegistry, auth: FakeAuth, subscriber: Any, setup_platform: Any) -> None:
    """Test multiple events are in descending timestamp order."""
    await setup_platform()
    auth.responses = [
        aiohttp.web.json_response(GENERATE_IMAGE_URL_RESPONSE),
        aiohttp.web.Response(body=IMAGE_BYTES_FROM_EVENT),
        aiohttp.web.json_response(GENERATE_IMAGE_URL_RESPONSE),
        aiohttp.web.Response(body=IMAGE_BYTES_FROM_EVENT)
    ]
    event_session_id1 = 'FWWVQVUdGNUlTU2V4MGV2aTNXV...'
    event_timestamp1 = dt_util.now()
    await subscriber.async_receive_event(create_event(event_session_id1, EVENT_ID + '1', PERSON_EVENT, timestamp=event_timestamp1))
    await hass.async_block_till_done()
    event_session_id2 = 'GXXWRWVeHNUlUU3V3MGV3bUOYW...'
    event_timestamp2 = event_timestamp1 + datetime.timedelta(seconds=5)
    await subscriber.async_receive_event(create_event(event_session_id2, EVENT_ID + '2', MOTION_EVENT, timestamp=event_timestamp2))
    await hass.async_block_till_done()
    assert len(hass.states.async_all()) == 1
    camera = hass.states.get('camera.front')
    assert camera is not None
    device = device_registry.async_get_device(identifiers={(DOMAIN, DEVICE_ID)})
    assert device
    assert device.name == DEVICE_NAME
    browse = await async_browse_media(hass, f'{URI_SCHEME}{DOMAIN}/{device.id}')
    assert browse.domain == DOMAIN
    assert browse.identifier == device.id
    assert browse.title == 'Front: Recent Events'
    assert browse.can_expand
    assert len(browse.children) == 2
    assert browse.children[0].domain == DOMAIN
    event_timestamp_string = event_timestamp2.strftime(DATE_STR_FORMAT)
    assert browse.children[0].title == f'Motion @ {event_timestamp_string}'
    assert not browse.children[0].can_expand
    assert not browse.children[0].can_play
    assert browse.children[1].domain == DOMAIN
    event_timestamp_string = event_timestamp1.strftime(DATE_STR_FORMAT)
    assert browse.children[1].title == f'Person @ {event_timestamp_string}'
    assert not browse.children[1].can_expand
    assert not browse.children[1].can_play

async def test_multiple_image_events_in_session(hass: HomeAssistant, device_registry: dr.DeviceRegistry, auth: FakeAuth, hass_client: ClientSessionGenerator, subscriber: Any, setup_platform: Any) -> None:
    """Test multiple events published within the same event session."""
    await setup_platform()
    event_session_id = 'FWWVQVUdGNUlTU2V4MGV2aTNXV...'
    event_timestamp1 = dt_util.now()
    event_timestamp2 = event_timestamp1 + datetime.timedelta(seconds=5)
    assert len(hass.states.async_all()) == 1
    camera = hass.states.get('camera.front')
    assert camera is not None
    device = device
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
def device(device_type: str, device_traits: Dict[str, Any], create_device: CreateDevice) -> dr.DeviceEntry:
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

# ... (continue with the rest of the functions, adding type annotations to each)

async def test_event_media_attachment(hass: HomeAssistant, hass_client: ClientSessionGenerator, device_registry: dr.DeviceRegistry, subscriber: Any, auth: FakeAuth, setup_platform: Any) -> None:
    """Verify that an event media attachment is successfully resolved."""
    await setup_platform()
    assert len(hass.states.async_all()) == 1
    camera = hass.states.get('camera.front')
    assert camera is not None
    device = device_registry.async_get_device(identifiers={(DOMAIN, DEVICE_ID)})
    assert device
    assert device.name == DEVICE_NAME
    received_events = async_capture_events(hass, NEST_EVENT)
    auth.responses = [aiohttp.web.json_response(GENERATE_IMAGE_URL_RESPONSE), aiohttp.web.Response(body=IMAGE_BYTES_FROM_EVENT)]
    event_timestamp = dt_util.now()
    await subscriber.async_receive_event(create_event(EVENT_SESSION_ID, EVENT_ID, PERSON_EVENT, timestamp=event_timestamp))
    await hass.async_block_till_done()
    assert len(received_events) == 1
    received_event = received_events[0]
    attachment = received_event.data.get('attachment')
    assert attachment
    assert list(attachment.keys()) == ['image']
    assert attachment['image'].startswith('/api/nest/event_media')
    assert attachment['image'].endswith('/thumbnail')
    client = await hass_client()
    response = await client.get(attachment['image'])
    assert response.status == HTTPStatus.OK, f'Response not matched: {response}'
    await response.read()

@pytest.mark.parametrize('device_traits', [BATTERY_CAMERA_TRAITS])
async def test_event_clip_media_attachment(hass: HomeAssistant, hass_client: ClientSessionGenerator, device_registry: dr.DeviceRegistry, subscriber: Any, auth: FakeAuth, setup_platform: Any, mp4: io.BytesIO) -> None:
    """Verify that an event media attachment is successfully resolved."""
    await setup_platform()
    assert len(hass.states.async_all()) == 1
    camera = hass.states.get('camera.front')
    assert camera is not None
    device = device_registry.async_get_device(identifiers={(DOMAIN, DEVICE_ID)})
    assert device
    assert device.name == DEVICE_NAME
    received_events = async_capture_events(hass, NEST_EVENT)
    auth.responses = [aiohttp.web.Response(body=mp4.getvalue())]
    event_timestamp = dt_util.now()
    await subscriber.async_receive_event(create_event_message(create_battery_event_data(MOTION_EVENT), timestamp=event_timestamp))
    await hass.async_block_till_done()
    assert len(received_events) == 1
    received_event = received_events[0]
    attachment = received_event.data.get('attachment')
    assert attachment
    assert list(attachment.keys()) == ['image', 'video']
    assert attachment['image'].startswith('/api/nest/event_media')
    assert attachment['image'].endswith('/thumbnail')
    assert attachment['video'].startswith('/api/nest/event_media')
    assert not attachment['video'].endswith('/thumbnail')
    for content_path in attachment.values():
        client = await hass_client()
        response = await client.get(content_path)
        assert response.status == HTTPStatus.OK, f'Response not matched: {response}'
        await response.read()

"""Component providing support to the Ring Door Bell camera."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
import logging
from typing import TYPE_CHECKING, Any, Generic, Optional, Tuple, Dict, Awaitable
from aiohttp import web
from haffmpeg.camera import CameraMjpeg
from ring_doorbell import RingDoorBell
from ring_doorbell.webrtcstream import RingWebRtcMessage
from homeassistant.components import ffmpeg
from homeassistant.components.camera import (
    Camera,
    CameraEntityDescription,
    CameraEntityFeature,
    RTCIceCandidateInit,
    WebRTCAnswer,
    WebRTCCandidate,
    WebRTCError,
    WebRTCSendMessage,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_aiohttp_proxy_stream
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import dt as dt_util
from . import RingConfigEntry
from .const import DOMAIN
from .coordinator import RingDataCoordinator
from .entity import RingDeviceT, RingEntity, exception_wrap

PARALLEL_UPDATES: int = 1
FORCE_REFRESH_INTERVAL: timedelta = timedelta(minutes=3)
MOTION_DETECTION_CAPABILITY: str = 'motion_detection'
_LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True, kw_only=True)
class RingCameraEntityDescription(CameraEntityDescription, Generic[RingDeviceT]):
    """Base class for event entity description."""

CAMERA_DESCRIPTIONS: Tuple[RingCameraEntityDescription, ...] = (
    RingCameraEntityDescription(
        key='live_view',
        translation_key='live_view',
        exists_fn=lambda _: True,
        live_stream=True,
        motion_detection=False
    ),
    RingCameraEntityDescription(
        key='last_recording',
        translation_key='last_recording',
        entity_registry_enabled_default=False,
        exists_fn=lambda camera: camera.has_subscription,
        live_stream=False,
        motion_detection=True
    )
)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: RingConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up a Ring Door Bell and StickUp Camera."""
    ring_data: RingDataCoordinator = entry.runtime_data
    devices_coordinator: RingDataCoordinator = ring_data.devices_coordinator
    ffmpeg_manager = ffmpeg.get_ffmpeg_manager(hass)
    cams: list[RingCam] = [
        RingCam(
            camera=camera,
            coordinator=devices_coordinator,
            description=description,
            ffmpeg_manager=ffmpeg_manager
        )
        for description in CAMERA_DESCRIPTIONS
        for camera in ring_data.devices.video_devices
        if description.exists_fn(camera)
    ]
    async_add_entities(cams)

class RingCam(RingEntity[RingDoorBell], Camera):
    """An implementation of a Ring Door Bell camera."""

    def __init__(
        self,
        device: RingDoorBell,
        coordinator: RingDataCoordinator,
        description: RingCameraEntityDescription,
        *,
        ffmpeg_manager
    ) -> None:
        """Initialize a Ring Door Bell camera."""
        super().__init__(device, coordinator)
        self.entity_description: RingCameraEntityDescription = description
        Camera.__init__(self)
        self._ffmpeg_manager = ffmpeg_manager
        self._last_event: Optional[dict[str, Any]] = None
        self._last_video_id: Optional[int] = None
        self._video_url: Optional[str] = None
        self._images: Dict[Tuple[Optional[int], Optional[int]], bytes] = {}
        self._expires_at: dt_util.dt.datetime = dt_util.utcnow() - FORCE_REFRESH_INTERVAL
        self._attr_unique_id: str = f'{device.id}-{description.key}'
        if description.motion_detection and device.has_capability(MOTION_DETECTION_CAPABILITY):
            self._attr_motion_detection_enabled: bool = device.motion_detection
        if description.live_stream:
            self._attr_supported_features |= CameraEntityFeature.STREAM

    @callback
    def _handle_coordinator_update(self) -> None:
        """Call update method."""
        self._device = self._get_coordinator_data().get_video_device(self._device.device_api_id)
        history_data: Optional[list[dict[str, Any]]] = self._device.last_history
        if history_data:
            self._last_event = history_data[0]
            self.async_schedule_update_ha_state(True)
        else:
            self._last_event = None
            self._last_video_id = None
            self._video_url = None
            self._images = {}
            self._expires_at = dt_util.utcnow()
            self.async_write_ha_state()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""
        return {'video_url': self._video_url, 'last_video_id': self._last_video_id}

    async def async_camera_image(self, width: Optional[int] = None, height: Optional[int] = None) -> Optional[bytes]:
        """Return a still image response from the camera."""
        key: Tuple[Optional[int], Optional[int]] = (width, height)
        image: Optional[bytes] = self._images.get(key)
        if not image and self._video_url is not None:
            image = await ffmpeg.async_get_image(self.hass, self._video_url, width=width, height=height)
            if image:
                self._images[key] = image
        return image

    async def handle_async_mjpeg_stream(self, request: web.Request) -> Optional[web.StreamResponse]:
        """Generate an HTTP MJPEG stream from the camera."""
        if self._video_url is None:
            return None
        stream = CameraMjpeg(self._ffmpeg_manager.binary)
        await stream.open_camera(self._video_url)
        try:
            stream_reader = await stream.get_reader()
            return await async_aiohttp_proxy_stream(
                self.hass,
                request,
                stream_reader,
                self._ffmpeg_manager.ffmpeg_stream_content_type
            )
        finally:
            await stream.close()

    async def async_handle_async_webrtc_offer(
        self,
        offer_sdp: str,
        session_id: str,
        send_message: Callable[[WebRTCSendMessage], Awaitable[None]]
    ) -> Awaitable[None]:
        """Return the source of the stream."""

        def message_wrapper(ring_message: RingWebRtcMessage) -> None:
            if ring_message.error_code:
                msg = ring_message.error_message or ''
                send_message(WebRTCError(ring_message.error_code, msg))
            elif ring_message.answer:
                send_message(WebRTCAnswer(ring_message.answer))
            elif ring_message.candidate:
                send_message(WebRTCCandidate(
                    RTCIceCandidateInit(
                        candidate=ring_message.candidate,
                        sdp_m_line_index=ring_message.sdp_m_line_index or 0
                    )
                ))

        return await self._device.generate_async_webrtc_stream(
            offer_sdp=offer_sdp,
            session_id=session_id,
            message_wrapper=message_wrapper,
            keep_alive_timeout=None
        )

    async def async_on_webrtc_candidate(
        self,
        session_id: str,
        candidate: WebRTCCandidate
    ) -> None:
        """Handle a WebRTC candidate."""
        if candidate.sdp_m_line_index is None:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key='sdp_m_line_index_required',
                translation_placeholders={'device': self._device.name}
            )
        await self._device.on_webrtc_candidate(
            session_id=session_id,
            candidate=candidate.candidate,
            sdp_m_line_index=candidate.sdp_m_line_index
        )

    @callback
    def close_webrtc_session(self, session_id: str) -> None:
        """Close a WebRTC session."""
        self._device.sync_close_webrtc_stream(session_id)

    async def async_update(self) -> None:
        """Update camera entity and refresh attributes."""
        if self._device.has_capability(MOTION_DETECTION_CAPABILITY) and \
           self._attr_motion_detection_enabled != self._device.motion_detection:
            self._attr_motion_detection_enabled = self._device.motion_detection
            self.async_write_ha_state()
        if TYPE_CHECKING:
            assert self._last_event
        if self._last_event['recording']['status'] != 'ready':
            return
        utcnow: dt_util.dt.datetime = dt_util.utcnow()
        if self._last_video_id == self._last_event['id'] and utcnow <= self._expires_at:
            return
        if self._last_video_id != self._last_event['id']:
            self._images = {}
        self._video_url = await self._async_get_video()
        self._last_video_id = self._last_event['id']
        self._expires_at = FORCE_REFRESH_INTERVAL + utcnow

    @exception_wrap
    async def _async_get_video(self) -> Optional[str]:
        if TYPE_CHECKING:
            assert self._last_event
        event_id: Optional[int] = self._last_event.get('id')
        assert event_id and isinstance(event_id, int)
        return await self._device.async_recording_url(event_id)

    @exception_wrap
    async def _async_set_motion_detection_enabled(self, new_state: bool) -> None:
        if not self._device.has_capability(MOTION_DETECTION_CAPABILITY):
            _LOGGER.error(
                'Entity %s does not have motion detection capability',
                self.entity_id
            )
            return
        await self._device.async_set_motion_detection(new_state)
        self._attr_motion_detection_enabled = new_state
        self.async_write_ha_state()

    async def async_enable_motion_detection(self) -> None:
        """Enable motion detection in the camera."""
        await self._async_set_motion_detection_enabled(True)

    async def async_disable_motion_detection(self) -> None:
        """Disable motion detection in camera."""
        await self._async_set_motion_detection_enabled(False)

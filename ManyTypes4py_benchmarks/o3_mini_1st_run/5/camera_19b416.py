from __future__ import annotations
from collections.abc import Generator
import logging
from typing import Any, List, Optional, Tuple

from uiprotect.data import (
    Camera as UFPCamera,
    CameraChannel,
    ProtectAdoptableDeviceModel,
    StateType,
)
from homeassistant.components.camera import Camera, CameraEntityFeature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.issue_registry import IssueSeverity
from .const import ATTR_BITRATE, ATTR_CHANNEL_ID, ATTR_FPS, ATTR_HEIGHT, ATTR_WIDTH, DOMAIN
from .data import ProtectData, ProtectDeviceType, UFPConfigEntry
from .entity import ProtectDeviceEntity
from .utils import get_camera_base_name

_LOGGER = logging.getLogger(__name__)

@callback
def _create_rtsp_repair(
    hass: HomeAssistant,
    entry: UFPConfigEntry,
    data: ProtectData,
    camera: UFPCamera,
) -> None:
    edit_key = "readonly"
    if camera.can_write(data.api.bootstrap.auth_user):
        edit_key = "writable"
    translation_key = f"rtsp_disabled_{edit_key}"
    issue_key = f"rtsp_disabled_{camera.id}"
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_key,
        is_fixable=True,
        is_persistent=False,
        learn_more_url="https://www.home-assistant.io/integrations/unifiprotect/#camera-streams",
        severity=IssueSeverity.WARNING,
        translation_key=translation_key,
        translation_placeholders={"camera": camera.display_name},
        data={"entry_id": entry.entry_id, "camera_id": camera.id},
    )

@callback
def _get_camera_channels(
    hass: HomeAssistant,
    entry: UFPConfigEntry,
    data: ProtectData,
    ufp_device: Optional[UFPCamera] = None,
) -> Generator[Tuple[UFPCamera, CameraChannel, bool], None, None]:
    """Get all the camera channels."""
    cameras: List[UFPCamera] = data.get_cameras() if ufp_device is None else [ufp_device]
    for camera in cameras:
        if not camera.channels:
            if ufp_device is None:
                _LOGGER.warning("Camera does not have any channels: %s (id: %s)", camera.display_name, camera.id)
            data.async_add_pending_camera_id(camera.id)
            continue
        is_default: bool = True
        for channel in camera.channels:
            if channel.is_package:
                yield (camera, channel, True)
            elif channel.is_rtsp_enabled:
                yield (camera, channel, is_default)
                is_default = False
        if is_default and (not camera.is_third_party_camera):
            _create_rtsp_repair(hass, entry, data, camera)
            yield (camera, camera.channels[0], True)
        else:
            ir.async_delete_issue(hass, DOMAIN, f"rtsp_disabled_{camera.id}")

def _async_camera_entities(
    hass: HomeAssistant,
    entry: UFPConfigEntry,
    data: ProtectData,
    ufp_device: Optional[UFPCamera] = None,
) -> List[ProtectCamera]:
    disable_stream: bool = data.disable_stream
    entities: List[ProtectCamera] = []
    for camera, channel, is_default in _get_camera_channels(hass, entry, data, ufp_device):
        entities.append(ProtectCamera(data, camera, channel, is_default, True, disable_stream or channel.is_package))
        if channel.is_rtsp_enabled and (not channel.is_package):
            entities.append(ProtectCamera(data, camera, channel, is_default, False, disable_stream))
    return entities

async def async_setup_entry(
    hass: HomeAssistant,
    entry: UFPConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Discover cameras on a UniFi Protect NVR."""
    data: ProtectData = entry.runtime_data

    @callback
    def _add_new_device(device: Any) -> None:
        if not isinstance(device, UFPCamera):
            return
        async_add_entities(_async_camera_entities(hass, entry, data, ufp_device=device))

    data.async_subscribe_adopt(_add_new_device)
    entry.async_on_unload(async_dispatcher_connect(hass, data.channels_signal, _add_new_device))
    async_add_entities(_async_camera_entities(hass, entry, data))

_DISABLE_FEATURE: CameraEntityFeature = CameraEntityFeature(0)
_ENABLE_FEATURE: CameraEntityFeature = CameraEntityFeature.STREAM

class ProtectCamera(ProtectDeviceEntity, Camera):
    """A Ubiquiti UniFi Protect Camera."""
    _state_attrs = ("_attr_available", "_attr_is_recording", "_attr_motion_detection_enabled")

    def __init__(
        self,
        data: ProtectData,
        camera: UFPCamera,
        channel: CameraChannel,
        is_default: bool,
        secure: bool,
        disable_stream: bool,
    ) -> None:
        """Initialize an UniFi camera."""
        self.channel: CameraChannel = channel
        self._secure: bool = secure
        self._disable_stream: bool = disable_stream
        self._last_image: Optional[bytes] = None
        self._stream_source: Optional[str] = None
        super().__init__(data, camera)
        device = self.device
        camera_name: str = get_camera_base_name(channel)
        if self._secure:
            self._attr_unique_id = f"{device.mac}_{channel.id}"
            self._attr_name = camera_name
        else:
            self._attr_unique_id = f"{device.mac}_{channel.id}_insecure"
            self._attr_name = f"{camera_name} (insecure)"
        self._attr_entity_registry_enabled_default = is_default and secure
        self._async_set_stream_source()

    @callback
    def _async_set_stream_source(self) -> None:
        channel: CameraChannel = self.channel
        enable_stream: bool = not self._disable_stream and channel.is_rtsp_enabled
        rtsp_url: str = channel.rtsps_no_srtp_url if self._secure else channel.rtsp_url
        source: Optional[str] = rtsp_url if enable_stream else None
        self._attr_supported_features = _ENABLE_FEATURE if source else _DISABLE_FEATURE
        self._stream_source = source

    @callback
    def _async_update_device_from_protect(self, device: UFPCamera) -> None:
        super()._async_update_device_from_protect(device)
        updated_device: UFPCamera = self.device
        channel: CameraChannel = updated_device.channels[self.channel.id]
        self.channel = channel
        motion_enabled: Optional[bool] = updated_device.recording_settings.enable_motion_detection
        self._attr_motion_detection_enabled = motion_enabled if motion_enabled is not None else True
        state_type_is_connected: bool = updated_device.state is StateType.CONNECTED
        self._attr_is_recording = state_type_is_connected and updated_device.is_recording
        is_connected: bool = self.data.last_update_success and state_type_is_connected
        self._attr_available = is_connected and updated_device.is_video_ready
        self._async_set_stream_source()
        self._attr_extra_state_attributes = {
            ATTR_WIDTH: channel.width,
            ATTR_HEIGHT: channel.height,
            ATTR_FPS: channel.fps,
            ATTR_BITRATE: channel.bitrate,
            ATTR_CHANNEL_ID: channel.id,
        }

    async def async_camera_image(self, width: Optional[int] = None, height: Optional[int] = None) -> bytes:
        """Return the Camera Image."""
        if self.channel.is_package:
            last_image: bytes = await self.device.get_package_snapshot(width, height)
        else:
            last_image = await self.device.get_snapshot(width, height)
        self._last_image = last_image
        return self._last_image

    async def stream_source(self) -> Optional[str]:
        """Return the Stream Source."""
        return self._stream_source

    async def async_enable_motion_detection(self) -> None:
        """Call the job and enable motion detection."""
        await self.device.set_motion_detection(True)

    async def async_disable_motion_detection(self) -> None:
        """Call the job and disable motion detection."""
        await self.device.set_motion_detection(False)
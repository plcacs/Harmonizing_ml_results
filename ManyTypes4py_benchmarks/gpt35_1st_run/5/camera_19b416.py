from __future__ import annotations
from collections.abc import Generator
import logging
from uiprotect.data import Camera as UFPCamera, CameraChannel, ProtectAdoptableDeviceModel, StateType
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
def _create_rtsp_repair(hass: HomeAssistant, entry: UFPConfigEntry, data: ProtectData, camera: UFPCamera) -> None:
    edit_key: str = 'readonly'
    if camera.can_write(data.api.bootstrap.auth_user):
        edit_key = 'writable'
    translation_key: str = f'rtsp_disabled_{edit_key}'
    issue_key: str = f'rtsp_disabled_{camera.id}'
    ir.async_create_issue(hass, DOMAIN, issue_key, is_fixable=True, is_persistent=False, learn_more_url='https://www.home-assistant.io/integrations/unifiprotect/#camera-streams', severity=IssueSeverity.WARNING, translation_key=translation_key, translation_placeholders={'camera': camera.display_name}, data={'entry_id': entry.entry_id, 'camera_id': camera.id})

@callback
def _get_camera_channels(hass: HomeAssistant, entry: UFPConfigEntry, data: ProtectData, ufp_device: ProtectAdoptableDeviceModel = None) -> Generator[tuple[UFPCamera, CameraChannel, bool], None, None]:
    cameras = data.get_cameras() if ufp_device is None else [ufp_device]
    for camera in cameras:
        if not camera.channels:
            if ufp_device is None:
                _LOGGER.warning('Camera does not have any channels: %s (id: %s)', camera.display_name, camera.id)
            data.async_add_pending_camera_id(camera.id)
            continue
        is_default = True
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
            ir.async_delete_issue(hass, DOMAIN, f'rtsp_disabled_{camera.id}')

def _async_camera_entities(hass: HomeAssistant, entry: UFPConfigEntry, data: ProtectData, ufp_device: ProtectAdoptableDeviceModel = None) -> list[ProtectCamera]:
    disable_stream: bool = data.disable_stream
    entities: list[ProtectCamera] = []
    for camera, channel, is_default in _get_camera_channels(hass, entry, data, ufp_device):
        entities.append(ProtectCamera(data, camera, channel, is_default, True, disable_stream or channel.is_package))
        if channel.is_rtsp_enabled and (not channel.is_package):
            entities.append(ProtectCamera(data, camera, channel, is_default, False, disable_stream))
    return entities

async def async_setup_entry(hass: HomeAssistant, entry: UFPConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    data: ProtectData = entry.runtime_data

    @callback
    def _add_new_device(device: ProtectAdoptableDeviceModel) -> None:
        if not isinstance(device, UFPCamera):
            return
        async_add_entities(_async_camera_entities(hass, entry, data, ufp_device=device))
    data.async_subscribe_adopt(_add_new_device)
    entry.async_on_unload(async_dispatcher_connect(hass, data.channels_signal, _add_new_device))
    async_add_entities(_async_camera_entities(hass, entry, data))

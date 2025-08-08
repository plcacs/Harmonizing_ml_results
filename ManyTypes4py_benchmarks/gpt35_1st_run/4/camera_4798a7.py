from __future__ import annotations
from datetime import datetime
import logging
import re
from typing import Any, List, Optional, Union, cast

from uvcclient import camera as uvc_camera, nvr
from uvcclient.camera import UVCCameraClient
from uvcclient.nvr import UVCRemote

import voluptuous as vol

from homeassistant.components.camera import PLATFORM_SCHEMA as CAMERA_PLATFORM_SCHEMA, Camera, CameraEntityFeature
from homeassistant.const import CONF_PASSWORD, CONF_PORT, CONF_SSL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util.dt import utc_from_timestamp

_LOGGER: logging.Logger = logging.getLogger(__name__)

CONF_NVR: str = 'nvr'
CONF_KEY: str = 'key'
DEFAULT_PASSWORD: str = 'ubnt'
DEFAULT_PORT: int = 7080
DEFAULT_SSL: bool = False

PLATFORM_SCHEMA: vol.Schema = CAMERA_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_NVR): cv.string,
    vol.Required(CONF_KEY): cv.string,
    vol.Optional(CONF_PASSWORD, default=DEFAULT_PASSWORD): cv.string,
    vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
    vol.Optional(CONF_SSL, default=DEFAULT_SSL): cv.boolean
})

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    addr: str = config[CONF_NVR]
    key: str = config[CONF_KEY]
    password: str = config[CONF_PASSWORD]
    port: int = config[CONF_PORT]
    ssl: bool = config[CONF_SSL]

    try:
        nvrconn: UVCRemote = nvr.UVCRemote(addr, port, key, ssl=ssl)
        cameras: List[dict[str, Any]] = nvrconn.index()
        identifier: str = nvrconn.camera_identifier
        cameras = [camera for camera in cameras if 'airCam' not in nvrconn.get_camera(camera[identifier])['model']]
    except nvr.NotAuthorized:
        _LOGGER.error('Authorization failure while connecting to NVR')
        return
    except nvr.NvrError as ex:
        _LOGGER.error('NVR refuses to talk to me: %s', str(ex))
        raise PlatformNotReady from ex

    add_entities((UnifiVideoCamera(nvrconn, camera[identifier], camera['name'], password) for camera in cameras), True)

class UnifiVideoCamera(Camera):
    _attr_should_poll: bool = True
    _attr_brand: str = 'Ubiquiti'
    _attr_is_streaming: bool = False

    def __init__(self, camera: UVCRemote, uuid: str, name: str, password: str) -> None:
        super().__init__()
        self._nvr: UVCRemote = camera
        self._uuid: str = self._attr_unique_id = uuid
        self._attr_name: str = name
        self._password: str = password
        self._connect_addr: Optional[str] = None
        self._camera: Optional[UVCCameraClient] = None

    @property
    def supported_features(self) -> Union[int, CameraEntityFeature]:
        channels: List[dict[str, Any]] = self._caminfo['channels']
        for channel in channels:
            if channel['isRtspEnabled']:
                return CameraEntityFeature.STREAM
        return CameraEntityFeature(0)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        attr: dict[str, Any] = {}
        if self.motion_detection_enabled:
            attr['last_recording_start_time'] = timestamp_ms_to_date(self._caminfo['lastRecordingStartTime'])
        return attr

    @property
    def is_recording(self) -> bool:
        recording_state: str = 'DISABLED'
        if 'recordingIndicator' in self._caminfo:
            recording_state = self._caminfo['recordingIndicator']
        return self._caminfo['recordingSettings']['fullTimeRecordEnabled'] or recording_state in ('MOTION_INPROGRESS', 'MOTION_FINISHED')

    @property
    def motion_detection_enabled(self) -> bool:
        return bool(self._caminfo['recordingSettings']['motionRecordEnabled'])

    @property
    def model(self) -> str:
        return cast(str, self._caminfo['model'])

    def _login(self) -> bool:
        caminfo: dict[str, Any] = self._caminfo
        if self._connect_addr:
            addrs: List[str] = [self._connect_addr]
        else:
            addrs = [caminfo['host'], caminfo['internalHost']
        if self._nvr.server_version >= (3, 2, 0):
            client_cls = uvc_camera.UVCCameraClientV320
        else:
            client_cls = uvc_camera.UVCCameraClient
        if caminfo['username'] is None:
            caminfo['username'] = 'ubnt'
        assert isinstance(caminfo['username'], str)
        camera: Optional[UVCCameraClient] = None
        for addr in addrs:
            try:
                camera = client_cls(addr, caminfo['username'], self._password)
                camera.login()
                _LOGGER.debug('Logged into UVC camera %s via %s', self._attr_name, addr)
                self._connect_addr = addr
                break
            except OSError:
                pass
            except uvc_camera.CameraConnectError:
                pass
            except uvc_camera.CameraAuthError:
                pass
        if not self._connect_addr:
            _LOGGER.error('Unable to login to camera')
            return False
        self._camera = camera
        self._caminfo = caminfo
        return True

    def camera_image(self, width: Optional[int] = None, height: Optional[int] = None) -> Optional[bytes]:
        if not self._camera and (not self._login()):
            return None

        def _get_image(retry: bool = True) -> Optional[bytes]:
            assert self._camera is not None
            try:
                return self._camera.get_snapshot()
            except uvc_camera.CameraConnectError:
                _LOGGER.error('Unable to contact camera')
                return None
            except uvc_camera.CameraAuthError:
                if retry:
                    self._login()
                    return _get_image(retry=False)
                _LOGGER.error('Unable to log into camera, unable to get snapshot')
                raise
        return _get_image()

    def set_motion_detection(self, mode: bool) -> None:
        set_mode: str = 'motion' if mode is True else 'none'
        try:
            self._nvr.set_recordmode(self._uuid, set_mode)
        except nvr.NvrError as err:
            _LOGGER.error('Unable to set recordmode to %s', set_mode)
            _LOGGER.debug(err)

    def enable_motion_detection(self) -> None:
        self.set_motion_detection(True)

    def disable_motion_detection(self) -> None:
        self.set_motion_detection(False)

    async def stream_source(self) -> Optional[str]:
        for channel in self._caminfo['channels']:
            if channel['isRtspEnabled']:
                return cast(str, next((uri for i, uri in enumerate(channel['rtspUris']) if re.search(self._nvr._host, uri))))
        return None

    def update(self) -> None:
        self._caminfo = self._nvr.get_camera(self._uuid)

def timestamp_ms_to_date(epoch_ms: Optional[int]) -> Optional[datetime]:
    if epoch_ms:
        return utc_from_timestamp(epoch_ms / 1000)
    return None

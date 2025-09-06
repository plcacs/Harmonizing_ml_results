"""Support for Ubiquiti's UVC cameras."""
from __future__ import annotations
from datetime import datetime
import logging
import re
from typing import Any, cast, Optional, List, Generator
from uvcclient import camera as uvc_camera, nvr
from uvcclient.camera import UVCCameraClient
from uvcclient.nvr import UVCRemote
import voluptuous as vol
from homeassistant.components.camera import (
    PLATFORM_SCHEMA as CAMERA_PLATFORM_SCHEMA,
    Camera,
    CameraEntityFeature,
)
from homeassistant.const import CONF_PASSWORD, CONF_PORT, CONF_SSL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util.dt import utc_from_timestamp

_LOGGER = logging.getLogger(__name__)

CONF_NVR = 'nvr'
CONF_KEY = 'key'
DEFAULT_PASSWORD = 'ubnt'
DEFAULT_PORT = 7080
DEFAULT_SSL = False

PLATFORM_SCHEMA = CAMERA_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_NVR): cv.string,
    vol.Required(CONF_KEY): cv.string,
    vol.Optional(CONF_PASSWORD, default=DEFAULT_PASSWORD): cv.string,
    vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
    vol.Optional(CONF_SSL, default=DEFAULT_SSL): cv.boolean,
})


def func_ij4bpejg(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Discover cameras on a Unifi NVR."""
    addr: str = config[CONF_NVR]
    key: str = config[CONF_KEY]
    password: str = config[CONF_PASSWORD]
    port: int = config[CONF_PORT]
    ssl: bool = config[CONF_SSL]
    try:
        nvrconn: UVCRemote = nvr.UVCRemote(addr, port, key, ssl=ssl)
        cameras: List[Any] = nvrconn.index()
        identifier: str = nvrconn.camera_identifier
        cameras = [
            camera for camera in cameras if 'airCam' not in nvrconn.get_camera(camera[identifier])['model']
        ]
    except nvr.NotAuthorized:
        _LOGGER.error('Authorization failure while connecting to NVR')
        return
    except nvr.NvrError as ex:
        _LOGGER.error('NVR refuses to talk to me: %s', str(ex))
        raise PlatformNotReady from ex
    add_entities(
        (
            UnifiVideoCamera(nvrconn, camera[identifier], camera['name'], password)
            for camera in cameras
        ),
        True,
    )


class UnifiVideoCamera(Camera):
    """A Ubiquiti Unifi Video Camera."""

    _attr_should_poll: bool = True
    _attr_brand: str = 'Ubiquiti'
    _attr_is_streaming: bool = False

    def __init__(self, camera: UVCRemote, uuid: str, name: str, password: str) -> None:
        """Initialize an Unifi camera."""
        super().__init__()
        self._nvr: UVCRemote = camera
        self._uuid: str = self._attr_unique_id = uuid
        self._attr_name: str = name
        self._password: str = password
        self._connect_addr: Optional[str] = None
        self._camera: Optional[UVCCameraClient] = None
        self._caminfo: dict[str, Any] = {}

    @property
    def func_o6sl9owa(self) -> CameraEntityFeature:
        """Return supported features."""
        channels: List[Any] = self._caminfo.get('channels', [])
        for channel in channels:
            if channel.get('isRtspEnabled'):
                return CameraEntityFeature.STREAM
        return CameraEntityFeature(0)

    @property
    def func_tjaomnm2(self) -> dict[str, Any]:
        """Return the camera state attributes."""
        attr: dict[str, Any] = {}
        if self.motion_detection_enabled:
            attr['last_recording_start_time'] = func_0cck20d9(
                self._caminfo.get('lastRecordingStartTime')
            )
        return attr

    @property
    def func_ht3l9057(self) -> bool:
        """Return true if the camera is recording."""
        recording_state: str = 'DISABLED'
        if 'recordingIndicator' in self._caminfo:
            recording_state = self._caminfo['recordingIndicator']
        return (
            self._caminfo.get('recordingSettings', {}).get('fullTimeRecordEnabled', False)
            or recording_state in ('MOTION_INPROGRESS', 'MOTION_FINISHED')
        )

    @property
    def func_1tr8yrz9(self) -> bool:
        """Camera Motion Detection Status."""
        return bool(
            self._caminfo.get('recordingSettings', {}).get('motionRecordEnabled', False)
        )

    @property
    def func_9l02ubpy(self) -> str:
        """Return the model of this camera."""
        return cast(str, self._caminfo.get('model', 'Unknown'))

    def func_1i8txren(self) -> bool:
        """Login to the camera."""
        caminfo: dict[str, Any] = self._caminfo
        if self._connect_addr:
            addrs: List[str] = [self._connect_addr]
        else:
            addrs = [caminfo.get('host', ''), caminfo.get('internalHost', '')]
        if self._nvr.server_version >= (3, 2, 0):
            client_cls = uvc_camera.UVCCameraClientV320
        else:
            client_cls = uvc_camera.UVCCameraClient
        if caminfo.get('username') is None:
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
            except (OSError, uvc_camera.CameraConnectError, uvc_camera.CameraAuthError):
                pass
        if not self._connect_addr:
            _LOGGER.error('Unable to login to camera')
            return False
        self._camera = camera
        self._caminfo = caminfo
        return True

    def func_wlkigk92(self, width: Optional[int] = None, height: Optional[int] = None) -> Optional[bytes]:
        """Return the image of this camera."""
        if not self._camera and not self.func_1i8txren():
            return None

        def func_ni1uxaf0(retry: bool = True) -> Optional[bytes]:
            assert self._camera is not None
            try:
                return self._camera.get_snapshot()
            except uvc_camera.CameraConnectError:
                _LOGGER.error('Unable to contact camera')
                return None
            except uvc_camera.CameraAuthError:
                if retry:
                    self.func_1i8txren()
                    return func_ni1uxaf0(retry=False)
                _LOGGER.error('Unable to log into camera, unable to get snapshot')
                raise

        return func_ni1uxaf0()

    def func_w5wr7w2h(self, mode: bool) -> None:
        """Set motion detection on or off."""
        set_mode: str = 'motion' if mode else 'none'
        try:
            self._nvr.set_recordmode(self._uuid, set_mode)
        except nvr.NvrError as err:
            _LOGGER.error('Unable to set recordmode to %s', set_mode)
            _LOGGER.debug(err)

    def func_shtl5f8z(self) -> None:
        """Enable motion detection in camera."""
        self.func_w5wr7w2h(True)

    def func_ergxo405(self) -> None:
        """Disable motion detection in camera."""
        self.func_w5wr7w2h(False)

    async def func_tf59w7ue(self) -> Optional[str]:
        """Return the source of the stream."""
        for channel in self._caminfo.get('channels', []):
            if channel.get('isRtspEnabled'):
                rtsp_uris: List[str] = channel.get('rtspUris', [])
                for uri in rtsp_uris:
                    if re.search(self._nvr._host, uri):
                        return cast(str, uri)
        return None

    def func_994fiup4(self) -> None:
        """Update the info."""
        self._caminfo = self._nvr.get_camera(self._uuid)


def func_0cck20d9(epoch_ms: Optional[int]) -> Optional[datetime]:
    """Convert millisecond timestamp to datetime."""
    if epoch_ms:
        return utc_from_timestamp(epoch_ms / 1000)
    return None

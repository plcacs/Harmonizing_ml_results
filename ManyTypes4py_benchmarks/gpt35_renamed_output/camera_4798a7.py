from __future__ import annotations
from datetime import datetime
import logging
import re
from typing import Any, List, Optional, Union, cast
from uvcclient import camera as uvc_camera, nvr
from uvcclient.camera import UVCCameraClient
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

def func_ij4bpejg(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
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
    def func_o6sl9owa(self) -> CameraEntityFeature:
        ...

    @property
    def func_tjaomnm2(self) -> dict[str, Any]:
        ...

    @property
    def func_ht3l9057(self) -> bool:
        ...

    @property
    def func_1tr8yrz9(self) -> bool:
        ...

    @property
    def func_9l02ubpy(self) -> str:
        ...

    def func_1i8txren(self) -> bool:
        ...

    def func_wlkigk92(self, width: Optional[int] = None, height: Optional[int] = None) -> Any:
        ...

    def func_w5wr7w2h(self, mode: bool) -> None:
        ...

    def func_shtl5f8z(self) -> None:
        ...

    def func_ergxo405(self) -> None:
        ...

    async def func_tf59w7ue(self) -> Optional[str]:
        ...

    def func_994fiup4(self) -> None:
        ...

def func_0cck20d9(epoch_ms: Optional[int]) -> Optional[datetime]:
    ...

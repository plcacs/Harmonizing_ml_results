from __future__ import annotations
import logging
from pyqvrpro.client import QVRResponseError
from homeassistant.components.camera import Camera
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from .const import DOMAIN, SHORT_NAME
_LOGGER: logging.Logger = logging.getLogger(__name__)

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    if discovery_info is None:
        return
    client = hass.data[DOMAIN]['client']
    entities: list[QVRProCamera] = []
    for channel in hass.data[DOMAIN]['channels']:
        stream_source = get_stream_source(channel['guid'], client)
        entities.append(QVRProCamera(**channel, stream_source=stream_source, client=client))
    add_entities(entities)

def get_stream_source(guid: str, client) -> str:
    try:
        resp = client.get_channel_live_stream(guid, protocol='rtsp')
    except QVRResponseError as ex:
        _LOGGER.error(ex)
        return None
    full_url = resp['resourceUris']
    protocol = full_url[:7]
    auth = f'{client.get_auth_string()}@'
    url = full_url[7:]
    return f'{protocol}{auth}{url}'

class QVRProCamera(Camera):
    def __init__(self, name: str, model: str, brand: str, channel_index: int, guid: str, stream_source: str, client) -> None:
        self._name: str = f'{SHORT_NAME} {name}'
        self._model: str = model
        self._brand: str = brand
        self.index: int = channel_index
        self.guid: str = guid
        self._client = client
        self._stream_source: str = stream_source
        super().__init__()

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def brand(self) -> str:
        return self._brand

    @property
    def extra_state_attributes(self) -> dict:
        return {'qvr_guid': self.guid}

    def camera_image(self, width: int = None, height: int = None) -> bytes:
        try:
            return self._client.get_snapshot(self.guid)
        except QVRResponseError as ex:
            _LOGGER.error('Error getting image: %s', ex)
            self._client.connect()
        return self._client.get_snapshot(self.guid)

    async def stream_source(self) -> str:
        return self._stream_source

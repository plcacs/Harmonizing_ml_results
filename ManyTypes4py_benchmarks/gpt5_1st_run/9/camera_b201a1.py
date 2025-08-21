"""Support for QVR Pro streams."""
from __future__ import annotations

import logging
from typing import Iterable, Protocol, TypedDict, runtime_checkable

from pyqvrpro.client import QVRResponseError

from homeassistant.components.camera import Camera
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from .const import DOMAIN, SHORT_NAME

_LOGGER: logging.Logger = logging.getLogger(__name__)


class LiveStreamResponse(TypedDict, total=False):
    resourceUris: str


class ChannelInfo(TypedDict):
    name: str
    model: str
    brand: str
    channel_index: int
    guid: str


@runtime_checkable
class QVRClientProtocol(Protocol):
    def get_channel_live_stream(self, guid: str, protocol: str = ...) -> LiveStreamResponse: ...
    def get_auth_string(self) -> str: ...
    def get_snapshot(self, guid: str) -> bytes: ...
    def connect(self) -> None: ...


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the QVR Pro camera platform."""
    if discovery_info is None:
        return
    client: QVRClientProtocol = hass.data[DOMAIN]['client']
    entities: list[QVRProCamera] = []
    channels: Iterable[ChannelInfo] = hass.data[DOMAIN]['channels']
    for channel in channels:
        stream_source: str | None = get_stream_source(channel['guid'], client)
        entities.append(QVRProCamera(**channel, stream_source=stream_source, client=client))
    add_entities(entities)


def get_stream_source(guid: str, client: QVRClientProtocol) -> str | None:
    """Get channel stream source."""
    try:
        resp: LiveStreamResponse = client.get_channel_live_stream(guid, protocol='rtsp')
    except QVRResponseError as ex:
        _LOGGER.error(ex)
        return None
    full_url: str = resp['resourceUris']
    protocol: str = full_url[:7]
    auth: str = f'{client.get_auth_string()}@'
    url: str = full_url[7:]
    return f'{protocol}{auth}{url}'


class QVRProCamera(Camera):
    """Representation of a QVR Pro camera."""

    _name: str
    _model: str
    _brand: str
    index: int
    guid: str
    _client: QVRClientProtocol
    _stream_source: str | None

    def __init__(
        self,
        name: str,
        model: str,
        brand: str,
        channel_index: int,
        guid: str,
        stream_source: str | None,
        client: QVRClientProtocol,
    ) -> None:
        """Init QVR Pro camera."""
        self._name = f'{SHORT_NAME} {name}'
        self._model = model
        self._brand = brand
        self.index = channel_index
        self.guid = guid
        self._client = client
        self._stream_source = stream_source
        super().__init__()

    @property
    def name(self) -> str:
        """Return the name of the entity."""
        return self._name

    @property
    def model(self) -> str:
        """Return the model of the entity."""
        return self._model

    @property
    def brand(self) -> str:
        """Return the brand of the entity."""
        return self._brand

    @property
    def extra_state_attributes(self) -> dict[str, str]:
        """Get the state attributes."""
        return {'qvr_guid': self.guid}

    def camera_image(self, width: int | None = None, height: int | None = None) -> bytes | None:
        """Get image bytes from camera."""
        try:
            return self._client.get_snapshot(self.guid)
        except QVRResponseError as ex:
            _LOGGER.error('Error getting image: %s', ex)
            self._client.connect()
        return self._client.get_snapshot(self.guid)

    async def stream_source(self) -> str | None:
        """Get stream source."""
        return self._stream_source
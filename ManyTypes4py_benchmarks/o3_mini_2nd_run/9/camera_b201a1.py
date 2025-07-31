"""Support for QVR Pro streams."""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from pyqvrpro.client import QVRResponseError
from homeassistant.components.camera import Camera
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from .const import DOMAIN, SHORT_NAME

_LOGGER = logging.getLogger(__name__)

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the QVR Pro camera platform."""
    if discovery_info is None:
        return
    client: Any = hass.data[DOMAIN]['client']
    entities: list[QVRProCamera] = []
    for channel in hass.data[DOMAIN]['channels']:
        stream_source: Optional[str] = get_stream_source(channel['guid'], client)
        entities.append(
            QVRProCamera(
                name=channel['name'],
                model=channel['model'],
                brand=channel['brand'],
                channel_index=channel['channel_index'],
                guid=channel['guid'],
                stream_source=stream_source,
                client=client,
            )
        )
    add_entities(entities)

def get_stream_source(guid: str, client: Any) -> Optional[str]:
    """Get channel stream source."""
    try:
        resp: Dict[str, Any] = client.get_channel_live_stream(guid, protocol='rtsp')
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

    def __init__(
        self,
        name: str,
        model: str,
        brand: str,
        channel_index: int,
        guid: str,
        stream_source: Optional[str],
        client: Any,
    ) -> None:
        """Init QVR Pro camera."""
        self._name: str = f'{SHORT_NAME} {name}'
        self._model: str = model
        self._brand: str = brand
        self.index: int = channel_index
        self.guid: str = guid
        self._client: Any = client
        self._stream_source: Optional[str] = stream_source
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
    def extra_state_attributes(self) -> Dict[str, str]:
        """Get the state attributes."""
        return {'qvr_guid': self.guid}

    def camera_image(self, width: Optional[int] = None, height: Optional[int] = None) -> Optional[bytes]:
        """Get image bytes from camera."""
        try:
            return self._client.get_snapshot(self.guid)
        except QVRResponseError as ex:
            _LOGGER.error('Error getting image: %s', ex)
            self._client.connect()
        return self._client.get_snapshot(self.guid)

    async def stream_source(self) -> Optional[str]:
        """Get stream source."""
        return self._stream_source

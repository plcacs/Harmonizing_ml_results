"""motionEye Media Source Implementation."""
from __future__ import annotations
import logging
from pathlib import PurePath
from typing import cast, Tuple, Optional, List, Dict
from motioneye_client.const import KEY_MEDIA_LIST, KEY_MIME_TYPE, KEY_PATH
from homeassistant.components.media_player import MediaClass, MediaType
from homeassistant.components.media_source import BrowseMediaSource, MediaSource, MediaSourceError, MediaSourceItem, PlayMedia, Unresolvable
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr

MIME_TYPE_MAP: Dict[str, str] = {'movies': 'video/mp4', 'images': 'image/jpeg'}
MEDIA_CLASS_MAP: Dict[str, MediaClass] = {'movies': MediaClass.VIDEO, 'images': MediaClass.IMAGE}
_LOGGER = logging.getLogger(__name__)

async def async_get_media_source(hass: HomeAssistant) -> MediaSource:
    """Set up motionEye media source."""
    return MotionEyeMediaSource(hass)

class MotionEyeMediaSource(MediaSource):
    """Provide motionEye stills and videos as media sources."""
    name: str = 'motionEye Media'

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize MotionEyeMediaSource."""
        super().__init__(DOMAIN)
        self.hass = hass

    async def async_resolve_media(self, item: MediaSourceItem) -> PlayMedia:
        """Resolve media to a url."""
        config_id, device_id, kind, path = self._parse_identifier(item.identifier)
        if not config_id or not device_id or (not kind) or (not path):
            raise Unresolvable(f'Incomplete media identifier specified: {item.identifier}')
        config = self._get_config_or_raise(config_id)
        device = self._get_device_or_raise(device_id)
        self._verify_kind_or_raise(kind)
        url = get_media_url(self.hass.data[DOMAIN][config.entry_id][CONF_CLIENT], self._get_camera_id_or_raise(config, device), self._get_path_or_raise(path), kind == 'images')
        if not url:
            raise Unresolvable(f'Could not resolve media item: {item.identifier}')
        return PlayMedia(url, MIME_TYPE_MAP[kind])

    @callback
    @classmethod
    def _parse_identifier(cls, identifier: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        base = [None] * 4
        data = identifier.split('#', 3)
        return cast(Tuple[str | None, str | None, str | None, str | None], tuple(data + base)[:4])

    async def async_browse_media(self, item: MediaSourceItem) -> BrowseMediaSource:
        """Return media."""
        if item.identifier:
            config_id, device_id, kind, path = self._parse_identifier(item.identifier)
            config = device = None
            if config_id:
                config = self._get_config_or_raise(config_id)
            if device_id:
                device = self._get_device_or_raise(device_id)
            if kind:
                self._verify_kind_or_raise(kind)
            path = self._get_path_or_raise(path)
            if config and device and kind:
                return await self._build_media_path(config, device, kind, path)
            if config and device:
                return self._build_media_kinds(config, device)
            if config:
                return self._build_media_configs()
        return self._build_media_configs()

    # ... (rest of the code remains the same)

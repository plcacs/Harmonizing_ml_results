"""motionEye Media Source Implementation."""
from __future__ import annotations
import logging
from pathlib import PurePath
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
from motioneye_client.const import KEY_MEDIA_LIST, KEY_MIME_TYPE, KEY_PATH
from homeassistant.components.media_player import MediaClass, MediaType
from homeassistant.components.media_source import (
    BrowseMedia,
    MediaClass as HMMediaClass,
    MediaContentType,
    BrowseMediaSource,
    MediaSource,
    MediaSourceError,
    MediaSourceItem,
    PlayMedia,
    Unresolvable,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from . import get_media_url, split_motioneye_device_identifier
from .const import CONF_CLIENT, DOMAIN

MIME_TYPE_MAP: Dict[str, str] = {'movies': 'video/mp4', 'images': 'image/jpeg'}
MEDIA_CLASS_MAP: Dict[str, MediaClass] = {'movies': MediaClass.VIDEO, 'images': MediaClass.IMAGE}
_LOGGER = logging.getLogger(__name__)


async def async_get_media_source(hass: HomeAssistant) -> MotionEyeMediaSource:
    """Set up motionEye media source."""
    return MotionEyeMediaSource(hass)


class MotionEyeMediaSource(MediaSource):
    """Provide motionEye stills and videos as media sources."""

    name: str = 'motionEye Media'

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize MotionEyeMediaSource."""
        super().__init__(DOMAIN)
        self.hass = hass

    async def async_resolve_media(self, item: MediaSourceItem) -> Union[PlayMedia, Unresolvable]:
        """Resolve media to a url."""
        config_id: Optional[str]
        device_id: Optional[str]
        kind: Optional[str]
        path: Optional[str]
        config_id, device_id, kind, path = self._parse_identifier(item.identifier)
        if not config_id or not device_id or not kind or not path:
            raise Unresolvable(f'Incomplete media identifier specified: {item.identifier}')
        config: ConfigEntry = self._get_config_or_raise(config_id)
        device: dr.DeviceEntry = self._get_device_or_raise(device_id)
        self._verify_kind_or_raise(kind)
        client = self.hass.data[DOMAIN][config.entry_id][CONF_CLIENT]
        camera_id: str = self._get_camera_id_or_raise(config, device)
        media_path: str = self._get_path_or_raise(path)
        is_image: bool = kind == 'images'
        url: Optional[str] = get_media_url(client, camera_id, media_path, is_image)
        if not url:
            raise Unresolvable(f'Could not resolve media item: {item.identifier}')
        return PlayMedia(url, MIME_TYPE_MAP[kind])

    @callback
    @classmethod
    def _parse_identifier(cls, identifier: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        base: List[Optional[str]] = [None] * 4
        data: List[str] = identifier.split('#', 3)
        return cast(
            Tuple[Optional[str], Optional[str], Optional[str], Optional[str]],
            tuple(data + base)[:4]
        )

    async def async_browse_media(self, item: BrowseMedia | None) -> BrowseMedia:
        """Return media."""
        if item and item.identifier:
            config_id, device_id, kind, path = self._parse_identifier(item.identifier)
            config: Optional[ConfigEntry] = None
            device: Optional[dr.DeviceEntry] = None
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
                return self._build_media_devices(config)
        return self._build_media_configs()

    def _get_config_or_raise(self, config_id: str) -> ConfigEntry:
        """Get a config entry from a URL."""
        entry: Optional[ConfigEntry] = self.hass.config_entries.async_get_entry(config_id)
        if not entry:
            raise MediaSourceError(f'Unable to find config entry with id: {config_id}')
        return entry

    def _get_device_or_raise(self, device_id: str) -> dr.DeviceEntry:
        """Get a device entry from a device ID."""
        device_registry = dr.async_get(self.hass)
        device: Optional[dr.DeviceEntry] = device_registry.async_get(device_id)
        if not device:
            raise MediaSourceError(f'Unable to find device with id: {device_id}')
        return device

    @classmethod
    def _verify_kind_or_raise(cls, kind: str) -> None:
        """Verify kind is an expected value."""
        if kind in MEDIA_CLASS_MAP:
            return
        raise MediaSourceError(f'Unknown media type: {kind}')

    @classmethod
    def _get_path_or_raise(cls, path: Optional[str]) -> str:
        """Verify path is a valid motionEye path."""
        if not path:
            return '/'
        if PurePath(path).is_absolute():
            return path
        raise MediaSourceError(f"motionEye media path must start with '/', received: {path}")

    @classmethod
    def _get_camera_id_or_raise(cls, config: ConfigEntry, device: dr.DeviceEntry) -> str:
        """Get the camera ID from a device entry."""
        for identifier in device.identifiers:
            data: Optional[Tuple[str, ...]] = split_motioneye_device_identifier(identifier)
            if data is not None and len(data) > 2:
                return data[2]
        raise MediaSourceError(f'Could not find camera id for device id: {device.id}')

    @classmethod
    def _build_media_config(cls, config: ConfigEntry) -> BrowseMediaSource:
        return BrowseMediaSource(
            domain=DOMAIN,
            identifier=config.entry_id,
            media_class=MediaClass.DIRECTORY,
            media_content_type='',
            title=config.title,
            can_play=False,
            can_expand=True,
            children_media_class=MediaClass.DIRECTORY
        )

    def _build_media_configs(self) -> BrowseMediaSource:
        """Build the media sources for config entries."""
        return BrowseMediaSource(
            domain=DOMAIN,
            identifier='',
            media_class=MediaClass.DIRECTORY,
            media_content_type='',
            title='motionEye Media',
            can_play=False,
            can_expand=True,
            children=[
                self._build_media_config(entry) for entry in self.hass.config_entries.async_entries(DOMAIN)
            ],
            children_media_class=MediaClass.DIRECTORY
        )

    @classmethod
    def _build_media_device(cls, config: ConfigEntry, device: dr.DeviceEntry, full_title: bool = True) -> BrowseMediaSource:
        title: str = f'{config.title} {device.name}' if full_title else device.name
        return BrowseMediaSource(
            domain=DOMAIN,
            identifier=f'{config.entry_id}#{device.id}',
            media_class=MediaClass.DIRECTORY,
            media_content_type='',
            title=title,
            can_play=False,
            can_expand=True,
            children_media_class=MediaClass.DIRECTORY
        )

    def _build_media_devices(self, config: ConfigEntry) -> BrowseMediaSource:
        """Build the media sources for device entries."""
        device_registry = dr.async_get(self.hass)
        devices: List[dr.DeviceEntry] = dr.async_entries_for_config_entry(device_registry, config.entry_id)
        base: BrowseMediaSource = self._build_media_config(config)
        base.children = [
            self._build_media_device(config, device, full_title=False) for device in devices
        ]
        return base

    @classmethod
    def _build_media_kind(cls, config: ConfigEntry, device: dr.DeviceEntry, kind: str, full_title: bool = True) -> BrowseMediaSource:
        media_content_type: MediaContentType = MediaType.VIDEO if kind == 'movies' else MediaType.IMAGE
        title: str
        if full_title:
            title = f'{config.title} {device.name} {kind.title()}'
        else:
            title = kind.title()
        media_class: MediaClass = MediaClass.VIDEO if kind == 'movies' else MediaClass.IMAGE
        return BrowseMediaSource(
            domain=DOMAIN,
            identifier=f'{config.entry_id}#{device.id}#{kind}',
            media_class=MediaClass.DIRECTORY,
            media_content_type=media_content_type,
            title=title,
            can_play=False,
            can_expand=True,
            children_media_class=media_class
        )

    def _build_media_kinds(self, config: ConfigEntry, device: dr.DeviceEntry) -> BrowseMediaSource:
        base: BrowseMediaSource = self._build_media_device(config, device)
        base.children = [
            self._build_media_kind(config, device, kind, full_title=False) for kind in MEDIA_CLASS_MAP
        ]
        return base

    async def _build_media_path(
        self,
        config: ConfigEntry,
        device: dr.DeviceEntry,
        kind: str,
        path: str
    ) -> BrowseMediaSource:
        """Build the media sources for media kinds."""
        base: BrowseMediaSource = self._build_media_kind(config, device, kind)
        parsed_path: PurePath = PurePath(path)
        if path != '/':
            display_path: str = str(PurePath(*parsed_path.parts[1:]))
            base.title += f' {display_path}'
        base.children = []
        client = self.hass.data[DOMAIN][config.entry_id][CONF_CLIENT]
        camera_id: str = self._get_camera_id_or_raise(config, device)
        if kind == 'movies':
            resp: Dict[str, Any] = await client.async_get_movies(camera_id)
        else:
            resp = await client.async_get_images(camera_id)
        sub_dirs: Set[str] = set()
        parts: Tuple[str, ...] = parsed_path.parts
        media_list: List[Dict[str, Any]] = resp.get(KEY_MEDIA_LIST, [])

        def get_media_sort_key(media: Dict[str, Any]) -> str:
            """Get media sort key."""
            return media.get(KEY_PATH, '')

        for media in sorted(media_list, key=get_media_sort_key):
            media_path: Optional[str] = media.get(KEY_PATH)
            media_mime_type: Optional[str] = media.get(KEY_MIME_TYPE)
            if not media_path or not media_mime_type or media_mime_type not in MIME_TYPE_MAP.values():
                continue
            parts_media: Tuple[str, ...] = PurePath(media_path).parts
            if parts_media[:len(parts)] == parts and len(parts_media) > len(parts):
                full_child_path: str = str(PurePath(*parts_media[:len(parts) + 1]))
                display_child_path: str = parts_media[len(parts)]
                if len(parts) + 1 == len(parts_media):
                    if kind == 'movies':
                        thumbnail_url: str = client.get_movie_url(camera_id, full_child_path, preview=True)
                    else:
                        thumbnail_url = client.get_image_url(camera_id, full_child_path, preview=True)
                    media_class: MediaClass = MEDIA_CLASS_MAP[kind]
                    content_type: str = media_mime_type
                    can_play: bool = kind == 'movies'
                    base.children.append(
                        BrowseMediaSource(
                            domain=DOMAIN,
                            identifier=f'{config.entry_id}#{device.id}#{kind}#{full_child_path}',
                            media_class=media_class,
                            media_content_type=content_type,
                            title=display_child_path,
                            can_play=can_play,
                            can_expand=False,
                            thumbnail=thumbnail_url
                        )
                    )
                elif len(parts) + 1 < len(parts_media):
                    if full_child_path not in sub_dirs:
                        sub_dirs.add(full_child_path)
                        media_content_type: MediaContentType = MediaType.VIDEO if kind == 'movies' else MediaType.IMAGE
                        child_media_class: MediaClass = MediaClass.DIRECTORY
                        base.children.append(
                            BrowseMediaSource(
                                domain=DOMAIN,
                                identifier=f'{config.entry_id}#{device.id}#{kind}#{full_child_path}',
                                media_class=MediaClass.DIRECTORY,
                                media_content_type=media_content_type,
                                title=display_child_path,
                                can_play=False,
                                can_expand=True,
                                children_media_class=MediaClass.DIRECTORY
                            )
                        )
        return base

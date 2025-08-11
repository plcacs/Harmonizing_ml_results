"""motionEye Media Source Implementation."""
from __future__ import annotations
import logging
from pathlib import PurePath
from typing import cast
from motioneye_client.const import KEY_MEDIA_LIST, KEY_MIME_TYPE, KEY_PATH
from homeassistant.components.media_player import MediaClass, MediaType
from homeassistant.components.media_source import BrowseMediaSource, MediaSource, MediaSourceError, MediaSourceItem, PlayMedia, Unresolvable
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from . import get_media_url, split_motioneye_device_identifier
from .const import CONF_CLIENT, DOMAIN
MIME_TYPE_MAP = {'movies': 'video/mp4', 'images': 'image/jpeg'}
MEDIA_CLASS_MAP = {'movies': MediaClass.VIDEO, 'images': MediaClass.IMAGE}
_LOGGER = logging.getLogger(__name__)

async def async_get_media_source(hass):
    """Set up motionEye media source."""
    return MotionEyeMediaSource(hass)

class MotionEyeMediaSource(MediaSource):
    """Provide motionEye stills and videos as media sources."""
    name = 'motionEye Media'

    def __init__(self, hass: Union[homeassistancore.HomeAssistant, homeassistanconfig_entries.ConfigEntry]) -> None:
        """Initialize MotionEyeMediaSource."""
        super().__init__(DOMAIN)
        self.hass = hass

    async def async_resolve_media(self, item):
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
    def _parse_identifier(cls: str, identifier: str) -> Union[str, None]:
        base = [None] * 4
        data = identifier.split('#', 3)
        return cast(tuple[str | None, str | None, str | None, str | None], tuple(data + base)[:4])

    async def async_browse_media(self, item):
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
                return self._build_media_devices(config)
        return self._build_media_configs()

    def _get_config_or_raise(self, config_id: Union[str, dict[str, str]]) -> dict[str, typing.Union[bool,str]]:
        """Get a config entry from a URL."""
        entry = self.hass.config_entries.async_get_entry(config_id)
        if not entry:
            raise MediaSourceError(f'Unable to find config entry with id: {config_id}')
        return entry

    def _get_device_or_raise(self, device_id: Union[str, int]) -> bool:
        """Get a config entry from a URL."""
        device_registry = dr.async_get(self.hass)
        if not (device := device_registry.async_get(device_id)):
            raise MediaSourceError(f'Unable to find device with id: {device_id}')
        return device

    @classmethod
    def _verify_kind_or_raise(cls: Union[str, None], kind: str) -> None:
        """Verify kind is an expected value."""
        if kind in MEDIA_CLASS_MAP:
            return
        raise MediaSourceError(f'Unknown media type: {kind}')

    @classmethod
    def _get_path_or_raise(cls: Union[pathlib.Path, str], path: Union[str, pathlib.Path]) -> Union[typing.Text, str, pathlib.Path]:
        """Verify path is a valid motionEye path."""
        if not path:
            return '/'
        if PurePath(path).root == '/':
            return path
        raise MediaSourceError(f"motionEye media path must start with '/', received: {path}")

    @classmethod
    def _get_camera_id_or_raise(cls: Union[int, dict[str, typing.Any], typing.Sequence[str]], config: Union[int, dict[str, typing.Any], typing.Sequence[str]], device: str) -> Union[bytes, typing.Any]:
        """Get a config entry from a URL."""
        for identifier in device.identifiers:
            data = split_motioneye_device_identifier(identifier)
            if data is not None:
                return data[2]
        raise MediaSourceError(f'Could not find camera id for device id: {device.id}')

    @classmethod
    def _build_media_config(cls: Union[typing.Type, static_frame.core.display_config.DisplayConfig, None, dict], config: Union[dict[str, typing.Any], dict, str]) -> BrowseMediaSource:
        return BrowseMediaSource(domain=DOMAIN, identifier=config.entry_id, media_class=MediaClass.DIRECTORY, media_content_type='', title=config.title, can_play=False, can_expand=True, children_media_class=MediaClass.DIRECTORY)

    def _build_media_configs(self) -> BrowseMediaSource:
        """Build the media sources for config entries."""
        return BrowseMediaSource(domain=DOMAIN, identifier='', media_class=MediaClass.DIRECTORY, media_content_type='', title='motionEye Media', can_play=False, can_expand=True, children=[self._build_media_config(entry) for entry in self.hass.config_entries.async_entries(DOMAIN)], children_media_class=MediaClass.DIRECTORY)

    @classmethod
    def _build_media_device(cls: str, config: Union[dict, dict[str, typing.Any], str], device: Union[dict, dict[str, typing.Any], str], full_title: bool=True) -> BrowseMediaSource:
        return BrowseMediaSource(domain=DOMAIN, identifier=f'{config.entry_id}#{device.id}', media_class=MediaClass.DIRECTORY, media_content_type='', title=f'{config.title} {device.name}' if full_title else device.name, can_play=False, can_expand=True, children_media_class=MediaClass.DIRECTORY)

    def _build_media_devices(self, config: Union[dict[str, typing.Any], typing.Mapping, dict]) -> Union[str, tuple[typing.Type], dict[str, typing.Any], None]:
        """Build the media sources for device entries."""
        device_registry = dr.async_get(self.hass)
        devices = dr.async_entries_for_config_entry(device_registry, config.entry_id)
        base = self._build_media_config(config)
        base.children = [self._build_media_device(config, device, full_title=False) for device in devices]
        return base

    @classmethod
    def _build_media_kind(cls: str, config: str, device: str, kind: str, full_title: bool=True) -> BrowseMediaSource:
        return BrowseMediaSource(domain=DOMAIN, identifier=f'{config.entry_id}#{device.id}#{kind}', media_class=MediaClass.DIRECTORY, media_content_type=MediaType.VIDEO if kind == 'movies' else MediaType.IMAGE, title=f'{config.title} {device.name} {kind.title()}' if full_title else kind.title(), can_play=False, can_expand=True, children_media_class=MediaClass.VIDEO if kind == 'movies' else MediaClass.IMAGE)

    def _build_media_kinds(self, config: Union[dict, str], device: Union[dict, str]) -> Union[str, dict[typing.Any, list], typing.Callable]:
        base = self._build_media_device(config, device)
        base.children = [self._build_media_kind(config, device, kind, full_title=False) for kind in MEDIA_CLASS_MAP]
        return base

    async def _build_media_path(self, config, device, kind, path):
        """Build the media sources for media kinds."""
        base = self._build_media_kind(config, device, kind)
        parsed_path = PurePath(path)
        if path != '/':
            base.title += f' {PurePath(*parsed_path.parts[1:])}'
        base.children = []
        client = self.hass.data[DOMAIN][config.entry_id][CONF_CLIENT]
        camera_id = self._get_camera_id_or_raise(config, device)
        if kind == 'movies':
            resp = await client.async_get_movies(camera_id)
        else:
            resp = await client.async_get_images(camera_id)
        sub_dirs = set()
        parts = parsed_path.parts
        media_list = resp.get(KEY_MEDIA_LIST, [])

        def get_media_sort_key(media: Any):
            """Get media sort key."""
            return media.get(KEY_PATH, '')
        for media in sorted(media_list, key=get_media_sort_key):
            if KEY_PATH not in media or KEY_MIME_TYPE not in media or media[KEY_MIME_TYPE] not in MIME_TYPE_MAP.values():
                continue
            parts_media = PurePath(media[KEY_PATH]).parts
            if parts_media[:len(parts)] == parts and len(parts_media) > len(parts):
                full_child_path = str(PurePath(*parts_media[:len(parts) + 1]))
                display_child_path = parts_media[len(parts)]
                if len(parts) + 1 == len(parts_media):
                    if kind == 'movies':
                        thumbnail_url = client.get_movie_url(camera_id, full_child_path, preview=True)
                    else:
                        thumbnail_url = client.get_image_url(camera_id, full_child_path, preview=True)
                    base.children.append(BrowseMediaSource(domain=DOMAIN, identifier=f'{config.entry_id}#{device.id}#{kind}#{full_child_path}', media_class=MEDIA_CLASS_MAP[kind], media_content_type=media[KEY_MIME_TYPE], title=display_child_path, can_play=kind == 'movies', can_expand=False, thumbnail=thumbnail_url))
                elif len(parts) + 1 < len(parts_media):
                    if full_child_path not in sub_dirs:
                        sub_dirs.add(full_child_path)
                        base.children.append(BrowseMediaSource(domain=DOMAIN, identifier=f'{config.entry_id}#{device.id}#{kind}#{full_child_path}', media_class=MediaClass.DIRECTORY, media_content_type=MediaType.VIDEO if kind == 'movies' else MediaType.IMAGE, title=display_child_path, can_play=False, can_expand=True, children_media_class=MediaClass.DIRECTORY))
        return base
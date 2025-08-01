"""motionEye Media Source Implementation."""
from __future__ import annotations
import logging
from pathlib import PurePath
from typing import Any, cast, Tuple, Optional, List
from motioneye_client.const import KEY_MEDIA_LIST, KEY_MIME_TYPE, KEY_PATH
from homeassistant.components.media_player import MediaClass, MediaType
from homeassistant.components.media_source import (
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

MIME_TYPE_MAP: dict[str, str] = {"movies": "video/mp4", "images": "image/jpeg"}
MEDIA_CLASS_MAP: dict[str, MediaClass] = {"movies": MediaClass.VIDEO, "images": MediaClass.IMAGE}

_LOGGER: logging.Logger = logging.getLogger(__name__)


async def async_get_media_source(hass: HomeAssistant) -> MotionEyeMediaSource:
    """Set up motionEye media source."""
    return MotionEyeMediaSource(hass)


class MotionEyeMediaSource(MediaSource):
    """Provide motionEye stills and videos as media sources."""
    name: str = "motionEye Media"

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize MotionEyeMediaSource."""
        super().__init__(DOMAIN)
        self.hass: HomeAssistant = hass

    async def async_resolve_media(self, item: MediaSourceItem) -> PlayMedia:
        """Resolve media to a url."""
        config_id, device_id, kind, path = self._parse_identifier(item.identifier)
        if not config_id or not device_id or (not kind) or (not path):
            raise Unresolvable(f"Incomplete media identifier specified: {item.identifier}")
        config: ConfigEntry = self._get_config_or_raise(config_id)
        device: dr.DeviceEntry = self._get_device_or_raise(device_id)
        self._verify_kind_or_raise(kind)
        url: Optional[str] = get_media_url(
            self.hass.data[DOMAIN][config.entry_id][CONF_CLIENT],
            self._get_camera_id_or_raise(config, device),
            self._get_path_or_raise(path),
            kind == "images",
        )
        if not url:
            raise Unresolvable(f"Could not resolve media item: {item.identifier}")
        return PlayMedia(url, MIME_TYPE_MAP[kind])

    @callback
    @classmethod
    def _parse_identifier(cls, identifier: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        base: List[Optional[str]] = [None] * 4
        data: List[str] = identifier.split("#", 3)
        return cast(Tuple[Optional[str], Optional[str], Optional[str], Optional[str]], tuple(data + base)[:4])

    async def async_browse_media(self, item: MediaSourceItem) -> BrowseMediaSource:
        """Return media."""
        if item.identifier:
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
            raise MediaSourceError(f"Unable to find config entry with id: {config_id}")
        return entry

    def _get_device_or_raise(self, device_id: str) -> dr.DeviceEntry:
        """Get a device entry from a URL."""
        device_registry = dr.async_get(self.hass)
        device: Optional[dr.DeviceEntry] = device_registry.async_get(device_id)
        if not device:
            raise MediaSourceError(f"Unable to find device with id: {device_id}")
        return device

    @classmethod
    def _verify_kind_or_raise(cls, kind: str) -> None:
        """Verify kind is an expected value."""
        if kind in MEDIA_CLASS_MAP:
            return
        raise MediaSourceError(f"Unknown media type: {kind}")

    @classmethod
    def _get_path_or_raise(cls, path: Optional[str]) -> str:
        """Verify path is a valid motionEye path."""
        if not path:
            return "/"
        if PurePath(path).root == "/":
            return path
        raise MediaSourceError(f"motionEye media path must start with '/', received: {path}")

    @classmethod
    def _get_camera_id_or_raise(cls, config: ConfigEntry, device: dr.DeviceEntry) -> str:
        """Get the camera id from a device identifier."""
        for identifier in device.identifiers:
            data = split_motioneye_device_identifier(identifier)
            if data is not None:
                return data[2]
        raise MediaSourceError(f"Could not find camera id for device id: {device.id}")

    @classmethod
    def _build_media_config(cls, config: ConfigEntry) -> BrowseMediaSource:
        return BrowseMediaSource(
            domain=DOMAIN,
            identifier=config.entry_id,
            media_class=MediaClass.DIRECTORY,
            media_content_type="",
            title=config.title,
            can_play=False,
            can_expand=True,
            children_media_class=MediaClass.DIRECTORY,
        )

    def _build_media_configs(self) -> BrowseMediaSource:
        """Build the media sources for config entries."""
        return BrowseMediaSource(
            domain=DOMAIN,
            identifier="",
            media_class=MediaClass.DIRECTORY,
            media_content_type="",
            title="motionEye Media",
            can_play=False,
            can_expand=True,
            children=[self._build_media_config(entry) for entry in self.hass.config_entries.async_entries(DOMAIN)],
            children_media_class=MediaClass.DIRECTORY,
        )

    @classmethod
    def _build_media_device(cls, config: ConfigEntry, device: dr.DeviceEntry, full_title: bool = True) -> BrowseMediaSource:
        return BrowseMediaSource(
            domain=DOMAIN,
            identifier=f"{config.entry_id}#{device.id}",
            media_class=MediaClass.DIRECTORY,
            media_content_type="",
            title=f"{config.title} {device.name}" if full_title else device.name,
            can_play=False,
            can_expand=True,
            children_media_class=MediaClass.DIRECTORY,
        )

    def _build_media_devices(self, config: ConfigEntry) -> BrowseMediaSource:
        """Build the media sources for device entries."""
        device_registry = dr.async_get(self.hass)
        devices: List[dr.DeviceEntry] = dr.async_entries_for_config_entry(device_registry, config.entry_id)
        base: BrowseMediaSource = self._build_media_config(config)
        base.children = [self._build_media_device(config, device, full_title=False) for device in devices]
        return base

    @classmethod
    def _build_media_kind(cls, config: ConfigEntry, device: dr.DeviceEntry, kind: str, full_title: bool = True) -> BrowseMediaSource:
        return BrowseMediaSource(
            domain=DOMAIN,
            identifier=f"{config.entry_id}#{device.id}#{kind}",
            media_class=MediaClass.DIRECTORY,
            media_content_type=MediaType.VIDEO if kind == "movies" else MediaType.IMAGE,
            title=f"{config.title} {device.name} {kind.title()}" if full_title else kind.title(),
            can_play=False,
            can_expand=True,
            children_media_class=MediaClass.VIDEO if kind == "movies" else MediaClass.IMAGE,
        )

    def _build_media_kinds(self, config: ConfigEntry, device: dr.DeviceEntry) -> BrowseMediaSource:
        base: BrowseMediaSource = self._build_media_device(config, device)
        base.children = [
            self._build_media_kind(config, device, kind, full_title=False) for kind in MEDIA_CLASS_MAP
        ]
        return base

    async def _build_media_path(self, config: ConfigEntry, device: dr.DeviceEntry, kind: str, path: str) -> BrowseMediaSource:
        """Build the media sources for media kinds."""
        base: BrowseMediaSource = self._build_media_kind(config, device, kind)
        parsed_path = PurePath(path)
        if path != "/":
            base.title += f" {PurePath(*parsed_path.parts[1:])}"
        base.children = []
        client: Any = self.hass.data[DOMAIN][config.entry_id][CONF_CLIENT]
        camera_id: str = self._get_camera_id_or_raise(config, device)
        if kind == "movies":
            resp: dict = await client.async_get_movies(camera_id)
        else:
            resp: dict = await client.async_get_images(camera_id)
        sub_dirs: set[str] = set()
        parts: Tuple[str, ...] = parsed_path.parts
        media_list: List[dict[str, Any]] = resp.get(KEY_MEDIA_LIST, [])

        def get_media_sort_key(media: dict[str, Any]) -> str:
            """Get media sort key."""
            return media.get(KEY_PATH, "")

        for media in sorted(media_list, key=get_media_sort_key):
            if KEY_PATH not in media or KEY_MIME_TYPE not in media or media[KEY_MIME_TYPE] not in MIME_TYPE_MAP.values():
                continue
            parts_media: Tuple[str, ...] = PurePath(media[KEY_PATH]).parts
            if parts_media[: len(parts)] == parts and len(parts_media) > len(parts):
                full_child_path: str = str(PurePath(*parts_media[: len(parts) + 1]))
                display_child_path: str = parts_media[len(parts)]
                if len(parts) + 1 == len(parts_media):
                    if kind == "movies":
                        thumbnail_url: str = client.get_movie_url(camera_id, full_child_path, preview=True)
                    else:
                        thumbnail_url: str = client.get_image_url(camera_id, full_child_path, preview=True)
                    base.children.append(
                        BrowseMediaSource(
                            domain=DOMAIN,
                            identifier=f"{config.entry_id}#{device.id}#{kind}#{full_child_path}",
                            media_class=MEDIA_CLASS_MAP[kind],
                            media_content_type=media[KEY_MIME_TYPE],
                            title=display_child_path,
                            can_play=kind == "movies",
                            can_expand=False,
                            thumbnail=thumbnail_url,
                        )
                    )
                elif len(parts) + 1 < len(parts_media):
                    if full_child_path not in sub_dirs:
                        sub_dirs.add(full_child_path)
                        base.children.append(
                            BrowseMediaSource(
                                domain=DOMAIN,
                                identifier=f"{config.entry_id}#{device.id}#{kind}#{full_child_path}",
                                media_class=MediaClass.DIRECTORY,
                                media_content_type=MediaType.VIDEO if kind == "movies" else MediaType.IMAGE,
                                title=display_child_path,
                                can_play=False,
                                can_expand=True,
                                children_media_class=MediaClass.DIRECTORY,
                            )
                        )
        return base
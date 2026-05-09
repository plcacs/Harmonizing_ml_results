"""Type stubs for media_source_5cb2e6 module."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from google_nest_sdm.camera_traits import CameraClipPreviewTrait, CameraEventImageTrait
from google_nest_sdm.device import Device
from google_nest_sdm.event import EventImageType, ImageEventBase
from google_nest_sdm.event_media import ClipPreviewSession, EventMediaStore, ImageSession
from google_nest_sdm.google_nest_subscriber import GoogleNestSubscriber
from homeassistant.components.media_player import BrowseError, MediaClass, MediaType
from homeassistant.components.media_source import (
    BrowseMediaSource,
    MediaSource,
    MediaSourceItem,
    PlayMedia,
    Unresolvable,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store
from homeassistant.helpers.template import DATE_STR_FORMAT
from homeassistant.util import dt as dt_util
from .const import DOMAIN

__all__ = [
    "NestEventMediaStore",
    "NestMediaSource",
    "MediaId",
    "async_get_media_event_store",
    "async_get_transcoder",
    "async_get_media_source",
    "async_get_media_source_devices",
]

class NestEventMediaStore(EventMediaStore):
    """Storage hook to locally persist nest media for events."""

    def __init__(
        self,
        hass: HomeAssistant,
        subscriber: GoogleNestSubscriber,
        store: Store[dict[str, Any]],
        media_path: str,
    ) -> None: ...
    
    async def async_load(self) -> dict[str, Any]: ...
    async def async_save(self, data: dict[str, Any]) -> None: ...
    def get_media_key(self, device_id: str, event: ImageEventBase) -> str: ...
    def _map_device_id(self, device_id: str) -> str: ...
    def get_image_media_key(self, device_id: str, event: ImageEventBase) -> str: ...
    def get_clip_preview_media_key(self, device_id: str, event: ImageEventBase) -> str: ...
    def get_clip_preview_thumbnail_media_key(self, device_id: str, event: ImageEventBase) -> str: ...
    def get_media_filename(self, media_key: str) -> str: ...
    async def async_load_media(self, media_key: str) -> Optional[bytes]: ...
    async def async_save_media(self, media_key: str, content: bytes) -> None: ...
    async def async_remove_media(self, media_key: str) -> None: ...
    async def _get_devices(self) -> dict[str, str]: ...

@callback
def async_get_media_source_devices(hass: HomeAssistant) -> dict[str, Device]: ...

@dataclass
class MediaId:
    """Media identifier for a node in the Media Browse tree."""
    device_id: str
    event_token: Optional[str] = None

    @property
    def identifier(self) -> str: ...

def parse_media_id(identifier: Optional[str]) -> Optional[MediaId]: ...

class NestMediaSource(MediaSource):
    """Provide Nest Media Sources for Nest Cameras."""
    name: str

    def __init__(self, hass: HomeAssistant) -> None: ...
    
    async def async_resolve_media(
        self,
        item: MediaSourceItem,
    ) -> PlayMedia: ...
    
    async def async_browse_media(
        self,
        item: MediaSourceItem,
    ) -> BrowseMediaSource: ...

async def async_get_media_event_store(
    hass: HomeAssistant,
    subscriber: GoogleNestSubscriber,
) -> NestEventMediaStore: ...

async def async_get_transcoder(hass: HomeAssistant) -> Transcoder: ...

async def async_get_media_source(hass: HomeAssistant) -> NestMediaSource: ...

async def _async_get_clip_preview_sessions(
    device: Device,
) -> dict[str, ClipPreviewSession]: ...

async def _async_get_image_sessions(
    device: Device,
) -> dict[str, ImageSession]: ...

def _browse_root() -> BrowseMediaSource: ...

async def _async_get_recent_event_id(
    device_id: MediaId,
    device: Device,
) -> Optional[MediaId]: ...

def _browse_device(
    device_id: MediaId,
    device: Device,
) -> BrowseMediaSource: ...

def _browse_clip_preview(
    event_id: MediaId,
    device: Device,
    event: ClipPreviewSession,
) -> BrowseMediaSource: ...

def _browse_image_event(
    event_id: MediaId,
    device: Device,
    event: ImageSession,
) -> BrowseMediaSource: ...
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import logging
import os
from typing import Any
from google_nest_sdm.camera_traits import CameraClipPreviewTrait, CameraEventImageTrait
from google_nest_sdm.device import Device
from google_nest_sdm.event import EventImageType, ImageEventBase
from google_nest_sdm.event_media import ClipPreviewSession, EventMediaStore, ImageSession
from google_nest_sdm.google_nest_subscriber import GoogleNestSubscriber
from google_nest_sdm.transcoder import Transcoder
from homeassistant.components.ffmpeg import get_ffmpeg_manager
from homeassistant.components.media_player import BrowseError, MediaClass, MediaType
from homeassistant.components.media_source import BrowseMediaSource, MediaSource, MediaSourceItem, PlayMedia, Unresolvable
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.storage import Store
from homeassistant.helpers.template import DATE_STR_FORMAT
from homeassistant.util import dt as dt_util
from .const import DOMAIN
from .device_info import NestDeviceInfo, async_nest_devices_by_device_id
from .events import EVENT_NAME_MAP, MEDIA_SOURCE_EVENT_TITLE_MAP

_LOGGER: logging.Logger = logging.getLogger(__name__)
MEDIA_SOURCE_TITLE: str = 'Nest'
DEVICE_TITLE_FORMAT: str = '{device_name}: Recent Events'
CLIP_TITLE_FORMAT: str = '{event_name} @ {event_time}'
EVENT_MEDIA_API_URL_FORMAT: str = '/api/nest/event_media/{device_id}/{event_token}'
EVENT_THUMBNAIL_URL_FORMAT: str = '/api/nest/event_media/{device_id}/{event_token}/thumbnail'
STORAGE_KEY: str = 'nest.event_media'
STORAGE_VERSION: int = 1
STORAGE_SAVE_DELAY_SECONDS: int = 120
MEDIA_PATH: str = f'{DOMAIN}/event_media'
DISK_READ_LRU_MAX_SIZE: int = 32

async def async_get_media_event_store(hass: HomeAssistant, subscriber: GoogleNestSubscriber) -> NestEventMediaStore:
    ...

class NestEventMediaStore(EventMediaStore):
    ...

async def async_get_transcoder(hass: HomeAssistant) -> Transcoder:
    ...

@dataclass
class MediaId:
    ...

def parse_media_id(identifier: str | None) -> MediaId | None:
    ...

class NestMediaSource(MediaSource):
    ...

async def async_resolve_media(self, item: MediaSourceItem) -> PlayMedia:
    ...

async def async_browse_media(self, item: MediaSourceItem) -> BrowseMediaSource:
    ...

async def _async_get_clip_preview_sessions(device: Device) -> dict[str, ClipPreviewSession]:
    ...

async def _async_get_image_sessions(device: Device) -> dict[str, ImageSession]:
    ...

def _browse_root() -> BrowseMediaSource:
    ...

async def _async_get_recent_event_id(device_id: str, device: Device) -> MediaId | None:
    ...

def _browse_device(device_id: str, device: Device) -> BrowseMediaSource:
    ...

def _browse_clip_preview(event_id: MediaId, device: Device, event: ClipPreviewSession) -> BrowseMediaSource:
    ...

def _browse_image_event(event_id: MediaId, device: Device, event: ImageSession) -> BrowseMediaSource:
    ...

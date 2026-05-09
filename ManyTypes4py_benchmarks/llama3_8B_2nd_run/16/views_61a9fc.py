from __future__ import annotations
from datetime import datetime
from http import HTTPStatus
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode
from aiohttp import web
from uiprotect.data import Camera, Event
from uiprotect.exceptions import ClientError
from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from .data import ProtectData, async_get_data_for_entry_id, async_get_data_for_nvr_id

@callback
def async_generate_thumbnail_url(event_id: str, nvr_id: str, width: int | None = None, height: int | None = None) -> str:
    ...

@callback
def async_generate_snapshot_url(nvr_id: str, camera_id: str, timestamp: datetime, width: int | None = None, height: int | None = None) -> str:
    ...

@callback
def async_generate_event_video_url(event: Event) -> str:
    ...

@callback
def async_generate_proxy_event_video_url(nvr_id: str, event_id: str) -> str:
    ...

@callback
def _client_error(message: str, code: HTTPStatus) -> web.Response:
    ...

@callback
def _400(message: str) -> web.Response:
    ...

@callback
def _403(message: str) -> web.Response:
    ...

@callback
def _404(message: str) -> web.Response:
    ...

class ProtectProxyView(HomeAssistantView):
    ...

class ThumbnailProxyView(ProtectProxyView):
    ...

class SnapshotProxyView(ProtectProxyView):
    ...

class VideoProxyView(ProtectProxyView):
    ...

class VideoEventProxyView(ProtectProxyView):
    ...

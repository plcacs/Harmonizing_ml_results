from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aiohttp.web import Request, Response
from motioneye_client.client import MotionEyeClient

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntry, DeviceRegistry

import logging

_LOGGER: logging.Logger
PLATFORMS: list[str]

def create_motioneye_client(*args: Any, **kwargs: Any) -> MotionEyeClient: ...

def get_motioneye_device_identifier(
    config_entry_id: str, camera_id: int
) -> tuple[str, str]: ...

def split_motioneye_device_identifier(
    identifier: tuple[str, ...],
) -> tuple[str, str, int] | None: ...

def get_camera_from_cameras(
    camera_id: int, data: dict[str, Any] | None
) -> dict[str, Any] | None: ...

def is_acceptable_camera(camera: dict[str, Any] | None) -> bool: ...

def listen_for_new_cameras(
    hass: HomeAssistant, entry: ConfigEntry, add_func: Callable[..., Any]
) -> None: ...

def async_generate_motioneye_webhook(
    hass: HomeAssistant, webhook_id: str
) -> str | None: ...

def _add_camera(
    hass: HomeAssistant,
    device_registry: DeviceRegistry,
    client: MotionEyeClient,
    entry: ConfigEntry,
    camera_id: int,
    camera: dict[str, Any],
    device_identifier: tuple[str, str],
) -> None: ...

async def _async_entry_updated(
    hass: HomeAssistant, config_entry: ConfigEntry
) -> None: ...

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool: ...

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool: ...

async def handle_webhook(
    hass: HomeAssistant, webhook_id: str, request: Request
) -> Response | None: ...

def _get_media_event_data(
    hass: HomeAssistant,
    device: DeviceEntry,
    event_file_path: str,
    event_file_type: int,
) -> dict[str, Any]: ...

def get_media_url(
    client: MotionEyeClient, camera_id: int, path: str, image: bool
) -> str | None: ...
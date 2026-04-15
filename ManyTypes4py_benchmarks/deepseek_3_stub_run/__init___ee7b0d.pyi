from __future__ import annotations

from collections.abc import Callable
from http import HTTPStatus
import logging
from typing import Any, Optional, Union

from aiohttp.web import Request, Response
from motioneye_client.client import MotionEyeClient, MotionEyeClientError, MotionEyeClientInvalidAuthError, MotionEyeClientPathError
from motioneye_client.const import KEY_CAMERAS, KEY_HTTP_METHOD_POST_JSON, KEY_ID, KEY_NAME, KEY_ROOT_DIRECTORY, KEY_WEB_HOOK_CONVERSION_SPECIFIERS, KEY_WEB_HOOK_CS_FILE_PATH, KEY_WEB_HOOK_CS_FILE_TYPE, KEY_WEB_HOOK_NOTIFICATIONS_ENABLED, KEY_WEB_HOOK_NOTIFICATIONS_HTTP_METHOD, KEY_WEB_HOOK_NOTIFICATIONS_URL, KEY_WEB_HOOK_STORAGE_ENABLED, KEY_WEB_HOOK_STORAGE_HTTP_METHOD, KEY_WEB_HOOK_STORAGE_URL
from homeassistant.components.camera import DOMAIN as CAMERA_DOMAIN
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_URL, CONF_WEBHOOK_ID
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import CONF_ADMIN_PASSWORD, CONF_ADMIN_USERNAME, CONF_CLIENT, CONF_COORDINATOR, CONF_SURVEILLANCE_PASSWORD, CONF_SURVEILLANCE_USERNAME, CONF_WEBHOOK_SET, CONF_WEBHOOK_SET_OVERWRITE, DEFAULT_SCAN_INTERVAL, DEFAULT_WEBHOOK_SET, DEFAULT_WEBHOOK_SET_OVERWRITE, DOMAIN, EVENT_FILE_STORED, EVENT_FILE_STORED_KEYS, EVENT_FILE_URL, EVENT_MEDIA_CONTENT_ID, EVENT_MOTION_DETECTED, EVENT_MOTION_DETECTED_KEYS, MOTIONEYE_MANUFACTURER, SIGNAL_CAMERA_ADD, WEB_HOOK_SENTINEL_KEY, WEB_HOOK_SENTINEL_VALUE

_LOGGER: logging.Logger = ...
PLATFORMS: list[str] = ...

def create_motioneye_client(
    url: str,
    admin_username: Optional[str] = ...,
    admin_password: Optional[str] = ...,
    surveillance_username: Optional[str] = ...,
    surveillance_password: Optional[str] = ...,
    session: Optional[Any] = ...
) -> MotionEyeClient: ...

def get_motioneye_device_identifier(config_entry_id: str, camera_id: int) -> tuple[str, str]: ...

def split_motioneye_device_identifier(identifier: tuple[str, str]) -> Optional[tuple[str, str, int]]: ...

def get_camera_from_cameras(camera_id: int, data: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]: ...

def is_acceptable_camera(camera: Optional[dict[str, Any]]) -> bool: ...

def listen_for_new_cameras(
    hass: HomeAssistant,
    entry: ConfigEntry,
    add_func: Callable[[dict[str, Any]], None]
) -> None: ...

def async_generate_motioneye_webhook(hass: HomeAssistant, webhook_id: str) -> Optional[str]: ...

def _add_camera(
    hass: HomeAssistant,
    device_registry: dr.DeviceRegistry,
    client: MotionEyeClient,
    entry: ConfigEntry,
    camera_id: int,
    camera: dict[str, Any],
    device_identifier: tuple[str, str]
) -> None: ...

async def _async_entry_updated(hass: HomeAssistant, config_entry: ConfigEntry) -> None: ...

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool: ...

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool: ...

async def handle_webhook(
    hass: HomeAssistant,
    webhook_id: str,
    request: Request
) -> Optional[Response]: ...

def _get_media_event_data(
    hass: HomeAssistant,
    device: dr.DeviceEntry,
    event_file_path: str,
    event_file_type: int
) -> dict[str, str]: ...

def get_media_url(
    client: MotionEyeClient,
    camera_id: int,
    path: str,
    image: bool
) -> Optional[str]: ...
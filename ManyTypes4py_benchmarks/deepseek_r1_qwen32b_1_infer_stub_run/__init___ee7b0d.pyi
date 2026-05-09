"""The motionEye integration."""

from __future__ import annotations
from collections.abc import Callable
from http import HTTPStatus
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from aiohttp.web import Request, Response
from motioneye_client.client import MotionEyeClient, MotionEyeClientError
from homeassistant.components.camera import DOMAIN as CAMERA_DOMAIN
from homeassistant.components.media_source import URI_SCHEME
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.components.webhook import (
    async_generate_id,
    async_generate_path,
    async_unregister as webhook_unregister,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.const import (
    ATTR_DEVICE_ID,
    ATTR_NAME,
    CONF_URL,
    CONF_WEBHOOK_ID,
)

_LOGGER: Any = ...
PLATFORMS: List[str] = ...

def create_motioneye_client(*args: Any, **kwargs: Any) -> MotionEyeClient:
    ...

def get_motioneye_device_identifier(config_entry_id: str, camera_id: int) -> Tuple[str, str]:
    ...

def split_motioneye_device_identifier(identifier: Tuple[str, str]) -> Optional[Tuple[str, str, int]]:
    ...

def get_camera_from_cameras(camera_id: int, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ...

def is_acceptable_camera(camera: Dict[str, Any]) -> bool:
    ...

@callback
def listen_for_new_cameras(hass: HomeAssistant, entry: ConfigEntry, add_func: Callable) -> None:
    ...

@callback
def async_generate_motioneye_webhook(hass: HomeAssistant, webhook_id: str) -> Optional[str]:
    ...

@callback
def _add_camera(
    hass: HomeAssistant,
    device_registry: dr.DeviceRegistry,
    client: MotionEyeClient,
    entry: ConfigEntry,
    camera_id: int,
    camera: Dict[str, Any],
    device_identifier: Tuple[str, str],
) -> None:
    ...

@callback
def _set_webhook(
    url: str,
    key_url: str,
    key_method: str,
    key_enabled: str,
    camera: Dict[str, Any],
) -> bool:
    ...

@callback
def _build_url(
    device: Any,
    base: str,
    event_type: str,
    keys: Dict[str, Any],
) -> str:
    ...

@callback
def _async_process_motioneye_cameras() -> None:
    ...

async def _async_entry_updated(hass: HomeAssistant, config_entry: ConfigEntry) -> None:
    ...

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    ...

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    ...

async def handle_webhook(hass: HomeAssistant, webhook_id: str, request: Request) -> Optional[Response]:
    ...

def _get_media_event_data(
    hass: HomeAssistant,
    device: Any,
    event_file_path: str,
    event_file_type: int,
) -> Dict[str, Any]:
    ...

def get_media_url(client: MotionEyeClient, camera_id: int, path: str, image: bool) -> Optional[str]:
    ...
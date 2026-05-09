"""The motionEye integration."""

from typing import Any, Optional, Union, overload
from aiohttp.web import Request, Response
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from motioneye_client.client import MotionEyeClient

PLATFORMS: list[str] = ...

def create_motioneye_client(*args: Any, **kwargs: Any) -> MotionEyeClient:
    """Create a MotionEyeClient."""
    ...

def get_motioneye_device_identifier(config_entry_id: str, camera_id: int) -> tuple[str, str]:
    """Get the identifiers for a motionEye device."""
    ...

def split_motioneye_device_identifier(identifier: str) -> Optional[tuple[str, str, int]]:
    """Get the identifiers for a motionEye device."""
    ...

def get_camera_from_cameras(camera_id: int, data: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Get an individual camera dict from a multiple cameras data response."""
    ...

def is_acceptable_camera(camera: Optional[dict[str, Any]]) -> bool:
    """Determine if a camera dict is acceptable."""
    ...

def listen_for_new_cameras(hass: HomeAssistant, entry: ConfigEntry, add_func: Any) -> None:
    """Listen for new cameras."""
    ...

def async_generate_motioneye_webhook(hass: HomeAssistant, webhook_id: str) -> Optional[str]:
    """Generate the full local URL for a webhook_id."""
    ...

def _add_camera(
    hass: HomeAssistant,
    device_registry: Any,
    client: MotionEyeClient,
    entry: ConfigEntry,
    camera_id: int,
    camera: dict[str, Any],
    device_identifier: tuple[str, str],
) -> None:
    """Add a motionEye camera to hass."""
    ...

async def _async_entry_updated(hass: HomeAssistant, config_entry: ConfigEntry) -> None:
    """Handle entry updates."""
    ...

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up motionEye from a config entry."""
    ...

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    ...

async def handle_webhook(hass: HomeAssistant, webhook_id: str, request: Request) -> Optional[Response]:
    """Handle webhook callback."""
    ...

def _get_media_event_data(
    hass: HomeAssistant,
    device: Any,
    event_file_path: str,
    event_file_type: int,
) -> dict[str, Any]:
    ...

def get_media_url(client: MotionEyeClient, camera_id: int, path: str, image: bool) -> Optional[str]:
    """Get the URL for a motionEye media item."""
    ...
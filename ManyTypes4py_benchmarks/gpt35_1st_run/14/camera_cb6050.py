from __future__ import annotations
import asyncio
from datetime import timedelta
import io
import logging
from PIL import Image
import voluptuous as vol
from homeassistant.components.camera import PLATFORM_SCHEMA as CAMERA_PLATFORM_SCHEMA, Camera, async_get_image, async_get_mjpeg_stream, async_get_still_stream
from homeassistant.const import CONF_ENTITY_ID, CONF_MODE, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util

_LOGGER: logging.Logger

CONF_CACHE_IMAGES: str = 'cache_images'
CONF_FORCE_RESIZE: str = 'force_resize'
CONF_IMAGE_QUALITY: str = 'image_quality'
CONF_IMAGE_REFRESH_RATE: str = 'image_refresh_rate'
CONF_MAX_IMAGE_WIDTH: str = 'max_image_width'
CONF_MAX_IMAGE_HEIGHT: str = 'max_image_height'
CONF_MAX_STREAM_WIDTH: str = 'max_stream_width'
CONF_MAX_STREAM_HEIGHT: str = 'max_stream_height'
CONF_IMAGE_TOP: str = 'image_top'
CONF_IMAGE_LEFT: str = 'image_left'
CONF_STREAM_QUALITY: str = 'stream_quality'
MODE_RESIZE: str = 'resize'
MODE_CROP: str = 'crop'
DEFAULT_BASENAME: str = 'Camera Proxy'
DEFAULT_QUALITY: int = 75
PLATFORM_SCHEMA: vol.Schema

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

def _precheck_image(image: bytes, opts: ImageOpts) -> Image:

def _resize_image(image: bytes, opts: ImageOpts) -> bytes:

def _crop_image(image: bytes, opts: ImageOpts) -> bytes:

class ImageOpts:
    max_width: int
    max_height: int
    left: int
    top: int
    quality: int
    force_resize: bool

    def __init__(self, max_width: int, max_height: int, left: int, top: int, quality: int, force_resize: bool) -> None:

    def __bool__(self) -> bool:

class ProxyCamera(Camera):
    def __init__(self, hass: HomeAssistant, config: ConfigType) -> None:

    def camera_image(self, width: int = None, height: int = None) -> bytes:

    async def async_camera_image(self, width: int = None, height: int = None) -> bytes:

    async def handle_async_mjpeg_stream(self, request: Request) -> bytes:

    @property
    def name(self) -> str:

    async def _async_stream_image(self) -> bytes:

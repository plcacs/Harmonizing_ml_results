from __future__ import annotations
import io
import logging
import os
import time
from PIL import Image, ImageDraw, UnidentifiedImageError
from pydoods import PyDOODS
import voluptuous as vol
from homeassistant.components.image_processing import CONF_CONFIDENCE, PLATFORM_SCHEMA as IMAGE_PROCESSING_PLATFORM_SCHEMA, ImageProcessingEntity
from homeassistant.const import CONF_COVERS, CONF_ENTITY_ID, CONF_NAME, CONF_SOURCE, CONF_TIMEOUT, CONF_URL
from homeassistant.core import HomeAssistant, split_entity_id
from homeassistant.helpers import config_validation as cv, template
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util.pil import draw_box

_LOGGER: logging.Logger

ATTR_MATCHES: str = 'matches'
ATTR_SUMMARY: str = 'summary'
ATTR_TOTAL_MATCHES: str = 'total_matches'
ATTR_PROCESS_TIME: str = 'process_time'
CONF_AUTH_KEY: str = 'auth_key'
CONF_DETECTOR: str = 'detector'
CONF_LABELS: str = 'labels'
CONF_AREA: str = 'area'
CONF_TOP: str = 'top'
CONF_BOTTOM: str = 'bottom'
CONF_RIGHT: str = 'right'
CONF_LEFT: str = 'left'
CONF_FILE_OUT: str = 'file_out'

AREA_SCHEMA: vol.Schema
LABEL_SCHEMA: vol.Schema
PLATFORM_SCHEMA: vol.Schema

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

class Doods(ImageProcessingEntity):

    def __init__(self, hass: HomeAssistant, camera_entity: str, name: str, doods: PyDOODS, detector: dict, config: ConfigType) -> None:

    @property
    def camera_entity(self) -> str:

    @property
    def name(self) -> str:

    @property
    def state(self) -> int:

    @property
    def extra_state_attributes(self) -> dict:

    def _save_image(self, image: bytes, matches: dict, paths: list[str]) -> None:

    def process_image(self, image: bytes) -> None:

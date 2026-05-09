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
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util.pil import draw_box
_LOGGER: logging.Logger = logging.getLogger(__name__)
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
AREA_SCHEMA: vol.Schema = vol.Schema({vol.Optional(CONF_BOTTOM, default=1): cv.small_float, vol.Optional(CONF_LEFT, default=0): cv.small_float, vol.Optional(CONF_RIGHT, default=1): cv.small_float, vol.Optional(CONF_TOP, default=0): cv.small_float, vol.Optional(CONF_COVERS, default=True): cv.boolean})
LABEL_SCHEMA: vol.Schema = vol.Schema({vol.Required(CONF_NAME): cv.string, vol.Optional(CONF_AREA): AREA_SCHEMA, vol.Optional(CONF_CONFIDENCE): vol.Range(min=0, max=100)})
PLATFORM_SCHEMA: vol.Schema = IMAGE_PROCESSING_PLATFORM_SCHEMA.extend({vol.Required(CONF_URL): cv.string, vol.Required(CONF_DETECTOR): cv.string, vol.Required(CONF_TIMEOUT, default=90): cv.positive_int, vol.Optional(CONF_AUTH_KEY, default=''): cv.string, vol.Optional(CONF_FILE_OUT, default=[]): vol.All(cv.ensure_list, [cv.template]), vol.Optional(CONF_CONFIDENCE, default=0.0): vol.Range(min=0, max=100), vol.Optional(CONF_LABELS, default=[]): vol.All(cv.ensure_list, [vol.Any(cv.string, LABEL_SCHEMA)])})

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class Doods(ImageProcessingEntity):
    """Doods image processing service client."""

    def __init__(self, hass: HomeAssistant, camera_entity: str, name: str, doods: PyDOODS, detector: dict, config: ConfigType) -> None:
        ...

    @property
    def camera_entity(self) -> str:
        """Return camera entity id from process pictures."""
        return self._camera_entity

    @property
    def name(self) -> str:
        """Return the name of the image processor."""
        return self._name

    @property
    def state(self) -> int:
        """Return the state of the entity."""
        return self._total_matches

    @property
    def extra_state_attributes(self) -> dict:
        """Return device specific state attributes."""
        return {ATTR_MATCHES: self._matches, ATTR_SUMMARY: {label: len(values) for label, values in self._matches.items()}, ATTR_TOTAL_MATCHES: self._total_matches, ATTR_PROCESS_TIME: self._process_time}

    def _save_image(self, image: bytes, matches: dict, paths: list) -> None:
        ...

    def process_image(self, image: bytes) -> None:
        ...

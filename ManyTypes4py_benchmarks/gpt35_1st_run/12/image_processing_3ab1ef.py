from __future__ import annotations
import io
import logging
import os
import subprocess
from PIL import Image
import voluptuous as vol
from homeassistant.components.image_processing import PLATFORM_SCHEMA as IMAGE_PROCESSING_PLATFORM_SCHEMA, ImageProcessingDeviceClass, ImageProcessingEntity
from homeassistant.const import CONF_ENTITY_ID, CONF_NAME, CONF_SOURCE
from homeassistant.core import HomeAssistant, split_entity_id
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
_LOGGER: logging.Logger = logging.getLogger(__name__)
CONF_DIGITS: str = 'digits'
CONF_EXTRA_ARGUMENTS: str = 'extra_arguments'
CONF_HEIGHT: str = 'height'
CONF_ROTATE: str = 'rotate'
CONF_SSOCR_BIN: str = 'ssocr_bin'
CONF_THRESHOLD: str = 'threshold'
CONF_WIDTH: str = 'width'
CONF_X_POS: str = 'x_position'
CONF_Y_POS: str = 'y_position'
DEFAULT_BINARY: str = 'ssocr'
PLATFORM_SCHEMA: vol.Schema = IMAGE_PROCESSING_PLATFORM_SCHEMA.extend({vol.Optional(CONF_EXTRA_ARGUMENTS, default=''): cv.string, vol.Optional(CONF_DIGITS): cv.positive_int, vol.Optional(CONF_HEIGHT, default=0): cv.positive_int, vol.Optional(CONF_SSOCR_BIN, default=DEFAULT_BINARY): cv.string, vol.Optional(CONF_THRESHOLD, default=0): cv.positive_int, vol.Optional(CONF_ROTATE, default=0): cv.positive_int, vol.Optional(CONF_WIDTH, default=0): cv.positive_int, vol.Optional(CONF_X_POS, default=0): cv.string, vol.Optional(CONF_Y_POS, default=0): cv.positive_int})

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    """Set up the Seven segments OCR platform."""
    async_add_entities((ImageProcessingSsocr(hass, camera[CONF_ENTITY_ID], config, camera.get(CONF_NAME)) for camera in config[CONF_SOURCE]))

class ImageProcessingSsocr(ImageProcessingEntity):
    """Representation of the seven segments OCR image processing entity."""
    _attr_device_class: ImageProcessingDeviceClass = ImageProcessingDeviceClass.OCR

    def __init__(self, hass: HomeAssistant, camera_entity: str, config: ConfigType, name: str) -> None:
        """Initialize seven segments processing."""
        self.hass: HomeAssistant = hass
        self._camera_entity: str = camera_entity
        if name:
            self._name: str = name
        else:
            self._name: str = f'SevenSegment OCR {split_entity_id(camera_entity)[1]}'
        self._state: str = None
        self.filepath: str = os.path.join(self.hass.config.config_dir, f'ssocr-{self._name.replace(' ', '_')}.png')
        crop: List[str] = ['crop', str(config[CONF_X_POS]), str(config[CONF_Y_POS]), str(config[CONF_WIDTH]), str(config[CONF_HEIGHT])]
        digits: List[str] = ['-d', str(config.get(CONF_DIGITS, -1))]
        rotate: List[str] = ['rotate', str(config[CONF_ROTATE])]
        threshold: List[str] = ['-t', str(config[CONF_THRESHOLD])]
        extra_arguments: List[str] = config[CONF_EXTRA_ARGUMENTS].split(' ')
        self._command: List[str] = [config[CONF_SSOCR_BIN], *crop, *digits, *threshold, *rotate, *extra_arguments]
        self._command.append(self.filepath)

    @property
    def camera_entity(self) -> str:
        """Return camera entity id from process pictures."""
        return self._camera_entity

    @property
    def name(self) -> str:
        """Return the name of the image processor."""
        return self._name

    @property
    def state(self) -> str:
        """Return the state of the entity."""
        return self._state

    def process_image(self, image: bytes) -> None:
        """Process the image."""
        stream: io.BytesIO = io.BytesIO(image)
        img: Image.Image = Image.open(stream)
        img.save(self.filepath, 'png')
        with subprocess.Popen(self._command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=False) as ocr:
            out: Tuple[bytes, bytes] = ocr.communicate()
            if out[0] != b'':
                self._state = out[0].strip().decode('utf-8')
            else:
                self._state = None
                _LOGGER.warning('Unable to detect value: %s', out[1].strip().decode('utf-8'))

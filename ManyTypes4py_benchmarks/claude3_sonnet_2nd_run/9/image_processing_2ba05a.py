"""Support for the DOODS service."""
from __future__ import annotations
import io
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union, cast
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

_LOGGER = logging.getLogger(__name__)

ATTR_MATCHES = 'matches'
ATTR_SUMMARY = 'summary'
ATTR_TOTAL_MATCHES = 'total_matches'
ATTR_PROCESS_TIME = 'process_time'
CONF_AUTH_KEY = 'auth_key'
CONF_DETECTOR = 'detector'
CONF_LABELS = 'labels'
CONF_AREA = 'area'
CONF_TOP = 'top'
CONF_BOTTOM = 'bottom'
CONF_RIGHT = 'right'
CONF_LEFT = 'left'
CONF_FILE_OUT = 'file_out'

AREA_SCHEMA = vol.Schema({
    vol.Optional(CONF_BOTTOM, default=1): cv.small_float,
    vol.Optional(CONF_LEFT, default=0): cv.small_float,
    vol.Optional(CONF_RIGHT, default=1): cv.small_float,
    vol.Optional(CONF_TOP, default=0): cv.small_float,
    vol.Optional(CONF_COVERS, default=True): cv.boolean
})

LABEL_SCHEMA = vol.Schema({
    vol.Required(CONF_NAME): cv.string,
    vol.Optional(CONF_AREA): AREA_SCHEMA,
    vol.Optional(CONF_CONFIDENCE): vol.Range(min=0, max=100)
})

PLATFORM_SCHEMA = IMAGE_PROCESSING_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_URL): cv.string,
    vol.Required(CONF_DETECTOR): cv.string,
    vol.Required(CONF_TIMEOUT, default=90): cv.positive_int,
    vol.Optional(CONF_AUTH_KEY, default=''): cv.string,
    vol.Optional(CONF_FILE_OUT, default=[]): vol.All(cv.ensure_list, [cv.template]),
    vol.Optional(CONF_CONFIDENCE, default=0.0): vol.Range(min=0, max=100),
    vol.Optional(CONF_LABELS, default=[]): vol.All(cv.ensure_list, [vol.Any(cv.string, LABEL_SCHEMA)]),
    vol.Optional(CONF_AREA): AREA_SCHEMA
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the Doods client."""
    url: str = config[CONF_URL]
    auth_key: str = config[CONF_AUTH_KEY]
    detector_name: str = config[CONF_DETECTOR]
    timeout: int = config[CONF_TIMEOUT]
    doods = PyDOODS(url, auth_key, timeout)
    response = doods.get_detectors()
    if not isinstance(response, dict):
        _LOGGER.warning('Could not connect to doods server: %s', url)
        return
    detector: Dict[str, Any] = {}
    for server_detector in response['detectors']:
        if server_detector['name'] == detector_name:
            detector = server_detector
            break
    if not detector:
        _LOGGER.warning('Detector %s is not supported by doods server %s', detector_name, url)
        return
    add_entities((Doods(hass, camera[CONF_ENTITY_ID], camera.get(CONF_NAME), doods, detector, config) for camera in config[CONF_SOURCE]))

class Doods(ImageProcessingEntity):
    """Doods image processing service client."""

    def __init__(
        self,
        hass: HomeAssistant,
        camera_entity: str,
        name: Optional[str],
        doods: PyDOODS,
        detector: Dict[str, Any],
        config: Dict[str, Any]
    ) -> None:
        """Initialize the DOODS entity."""
        self.hass = hass
        self._camera_entity = camera_entity
        if name:
            self._name = name
        else:
            name = split_entity_id(camera_entity)[1]
            self._name = f'Doods {name}'
        self._doods = doods
        self._file_out: List[template.Template] = config[CONF_FILE_OUT]
        self._detector_name: str = detector['name']
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._aspect: Optional[float] = None
        if detector['width'] and detector['height']:
            self._width = detector['width']
            self._height = detector['height']
            self._aspect = self._width / self._height
        dconfig: Dict[str, float] = {}
        confidence: float = config[CONF_CONFIDENCE]
        labels: List[Union[str, Dict[str, Any]]] = config[CONF_LABELS]
        self._label_areas: Dict[str, List[float]] = {}
        self._label_covers: Dict[str, bool] = {}
        for label in labels:
            if isinstance(label, dict):
                label_name: str = label[CONF_NAME]
                if label_name not in detector['labels'] and label_name != '*':
                    _LOGGER.warning('Detector does not support label %s', label_name)
                    continue
                label_confidence: float = label.get(CONF_CONFIDENCE, confidence)
                if label_name not in dconfig or dconfig[label_name] > label_confidence:
                    dconfig[label_name] = label_confidence
                label_area = label.get(CONF_AREA)
                self._label_areas[label_name] = [0, 0, 1, 1]
                self._label_covers[label_name] = True
                if label_area:
                    self._label_areas[label_name] = [label_area[CONF_TOP], label_area[CONF_LEFT], label_area[CONF_BOTTOM], label_area[CONF_RIGHT]]
                    self._label_covers[label_name] = label_area[CONF_COVERS]
            else:
                if label not in detector['labels'] and label != '*':
                    _LOGGER.warning('Detector does not support label %s', label)
                    continue
                self._label_areas[label] = [0, 0, 1, 1]
                self._label_covers[label] = True
                if label not in dconfig or dconfig[label] > confidence:
                    dconfig[label] = confidence
        if not dconfig:
            dconfig['*'] = confidence
        self._area: List[float] = [0, 0, 1, 1]
        self._covers: bool = True
        if (area_config := config.get(CONF_AREA)):
            self._area = [area_config[CONF_TOP], area_config[CONF_LEFT], area_config[CONF_BOTTOM], area_config[CONF_RIGHT]]
            self._covers = area_config[CONF_COVERS]
        self._dconfig = dconfig
        self._matches: Dict[str, List[Dict[str, Any]]] = {}
        self._total_matches: int = 0
        self._last_image: Optional[bytes] = None
        self._process_time: float = 0

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
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return device specific state attributes."""
        return {
            ATTR_MATCHES: self._matches,
            ATTR_SUMMARY: {label: len(values) for label, values in self._matches.items()},
            ATTR_TOTAL_MATCHES: self._total_matches,
            ATTR_PROCESS_TIME: self._process_time
        }

    def _save_image(self, image: bytes, matches: Dict[str, List[Dict[str, Any]]], paths: List[str]) -> None:
        img = Image.open(io.BytesIO(bytearray(image))).convert('RGB')
        img_width, img_height = img.size
        draw = ImageDraw.Draw(img)
        if self._area != [0, 0, 1, 1]:
            draw_box(draw, self._area, img_width, img_height, 'Detection Area', (0, 255, 255))
        for label, values in matches.items():
            if label in self._label_areas and self._label_areas[label] != [0, 0, 1, 1]:
                box_label = f'{label.capitalize()} Detection Area'
                draw_box(draw, self._label_areas[label], img_width, img_height, box_label, (0, 255, 0))
            for instance in values:
                box_label = f'{label} {instance["score"]:.1f}%'
                draw_box(draw, instance['box'], img_width, img_height, box_label, (255, 255, 0))
        for path in paths:
            _LOGGER.debug('Saving results image to %s', path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img.save(path)

    def process_image(self, image: bytes) -> None:
        """Process the image."""
        try:
            img = Image.open(io.BytesIO(bytearray(image))).convert('RGB')
        except UnidentifiedImageError:
            _LOGGER.warning('Unable to process image, bad data')
            return
        img_width, img_height = img.size
        if self._aspect and abs(img_width / img_height - self._aspect) > 0.1:
            _LOGGER.debug('The image aspect: %s and the detector aspect: %s differ by more than 0.1', img_width / img_height, self._aspect)
        start = time.monotonic()
        response = self._doods.detect(image, dconfig=self._dconfig, detector_name=self._detector_name)
        _LOGGER.debug('doods detect: %s response: %s duration: %s', self._dconfig, response, time.monotonic() - start)
        matches: Dict[str, List[Dict[str, Any]]] = {}
        total_matches: int = 0
        if not response or 'error' in response:
            if 'error' in response:
                _LOGGER.error(response['error'])
            self._matches = matches
            self._total_matches = total_matches
            self._process_time = time.monotonic() - start
            return
        for detection in response['detections']:
            score: float = detection['confidence']
            boxes: List[float] = [detection['top'], detection['left'], detection['bottom'], detection['right']]
            label: str = detection['label']
            if '*' not in self._dconfig and label not in self._dconfig:
                continue
            if self._covers:
                if boxes[0] < self._area[0] or boxes[1] < self._area[1] or boxes[2] > self._area[2] or (boxes[3] > self._area[3]):
                    continue
            elif boxes[0] > self._area[2] or boxes[1] > self._area[3] or boxes[2] < self._area[0] or (boxes[3] < self._area[1]):
                continue
            if self._label_areas.get(label):
                if self._label_covers[label]:
                    if boxes[0] < self._label_areas[label][0] or boxes[1] < self._label_areas[label][1] or boxes[2] > self._label_areas[label][2] or (boxes[3] > self._label_areas[label][3]):
                        continue
                elif boxes[0] > self._label_areas[label][2] or boxes[1] > self._label_areas[label][3] or boxes[2] < self._label_areas[label][0] or (boxes[3] < self._label_areas[label][1]):
                    continue
            if label not in matches:
                matches[label] = []
            matches[label].append({'score': float(score), 'box': boxes})
            total_matches += 1
        if total_matches and self._file_out:
            paths: List[str] = []
            for path_template in self._file_out:
                if isinstance(path_template, template.Template):
                    paths.append(path_template.render(camera_entity=self._camera_entity))
                else:
                    paths.append(cast(str, path_template))
            self._save_image(image, matches, paths)
        else:
            _LOGGER.debug('Not saving image(s), no detections found or no output file configured')
        self._matches = matches
        self._total_matches = total_matches
        self._process_time = time.monotonic() - start

from __future__ import annotations
import io
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
import tensorflow as tf
import voluptuous as vol

from homeassistant.components.image_processing import CONF_CONFIDENCE, PLATFORM_SCHEMA as IMAGE_PROCESSING_PLATFORM_SCHEMA, ImageProcessingEntity
from homeassistant.const import CONF_ENTITY_ID, CONF_MODEL, CONF_NAME, CONF_SOURCE, EVENT_HOMEASSISTANT_START
from homeassistant.core import HomeAssistant, split_entity_id
from homeassistant.helpers import config_validation as cv, template
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util.pil import draw_box

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DOMAIN: str = 'tensorflow'
_LOGGER: Any = logging.getLogger(__name__)
ATTR_MATCHES: str = 'matches'
ATTR_SUMMARY: str = 'summary'
ATTR_TOTAL_MATCHES: str = 'total_matches'
ATTR_PROCESS_TIME: str = 'process_time'
CONF_AREA: str = 'area'
CONF_BOTTOM: str = 'bottom'
CONF_CATEGORIES: str = 'categories'
CONF_CATEGORY: str = 'category'
CONF_FILE_OUT: str = 'file_out'
CONF_GRAPH: str = 'graph'
CONF_LABELS: str = 'labels'
CONF_LABEL_OFFSET: str = 'label_offset'
CONF_LEFT: str = 'left'
CONF_MODEL_DIR: str = 'model_dir'
CONF_RIGHT: str = 'right'
CONF_TOP: str = 'top'

AREA_SCHEMA: vol.Schema = vol.Schema({
    vol.Optional(CONF_BOTTOM, default=1): cv.small_float,
    vol.Optional(CONF_LEFT, default=0): cv.small_float,
    vol.Optional(CONF_RIGHT, default=1): cv.small_float,
    vol.Optional(CONF_TOP, default=0): cv.small_float
})
CATEGORY_SCHEMA: vol.Schema = vol.Schema({
    vol.Required(CONF_CATEGORY): cv.string,
    vol.Optional(CONF_AREA): AREA_SCHEMA
})
PLATFORM_SCHEMA: vol.Schema = IMAGE_PROCESSING_PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_FILE_OUT, default=[]): vol.All(cv.ensure_list, [cv.template]),
    vol.Required(CONF_MODEL): vol.Schema({
        vol.Required(CONF_GRAPH): cv.isdir,
        vol.Optional(CONF_AREA): AREA_SCHEMA,
        vol.Optional(CONF_CATEGORIES, default=[]): vol.All(cv.ensure_list, [vol.Any(cv.string, CATEGORY_SCHEMA)]),
        vol.Optional(CONF_LABELS): cv.isfile,
        vol.Optional(CONF_LABEL_OFFSET, default=1): int,
        vol.Optional(CONF_MODEL_DIR): cv.isdir
    })
})


def get_model_detection_function(model: Any) -> Callable[[tf.Tensor], Dict[str, Any]]:
    """Get a tf.function for detection."""
    @tf.function
    def detect_fn(image: tf.Tensor) -> Dict[str, Any]:
        """Detect objects in image."""
        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        return model.postprocess(prediction_dict, shapes)
    return detect_fn


def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: Optional[DiscoveryInfoType] = None) -> None:
    """Set up the TensorFlow image processing platform."""
    model_config: Dict[str, Any] = config[CONF_MODEL]
    model_dir: str = model_config.get(CONF_MODEL_DIR) or hass.config.path('tensorflow')
    labels: str = model_config.get(CONF_LABELS) or hass.config.path('tensorflow', 'object_detection', 'data', 'mscoco_label_map.pbtxt')
    checkpoint: str = os.path.join(model_config[CONF_GRAPH], 'checkpoint')
    pipeline_config: str = os.path.join(model_config[CONF_GRAPH], 'pipeline.config')
    if not os.path.isdir(model_dir) or not os.path.isdir(checkpoint) or (not os.path.exists(pipeline_config)) or (not os.path.exists(labels)):
        _LOGGER.error('Unable to locate tensorflow model or label map')
        return
    sys.path.append(model_dir)
    try:
        from object_detection.builders import model_builder  # type: ignore
        from object_detection.utils import config_util, label_map_util  # type: ignore
    except ImportError:
        _LOGGER.error('No TensorFlow Object Detection library found! Install or compile for your system following instructions here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md#installation')
        return
    try:
        import cv2  # type: ignore
    except ImportError:
        _LOGGER.warning('No OpenCV library found. TensorFlow will process image with PIL at reduced resolution')
    hass.data[DOMAIN] = {CONF_MODEL: None}

    def tensorflow_hass_start(_event: Any) -> None:
        """Set up TensorFlow model on hass start."""
        start: float = time.perf_counter()
        pipeline_configs: Dict[str, Any] = config_util.get_configs_from_pipeline_file(pipeline_config)
        detection_model: Any = model_builder.build(model_config=pipeline_configs['model'], is_training=False)
        ckpt: tf.train.Checkpoint = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(checkpoint, 'ckpt-0')).expect_partial()
        _LOGGER.debug('Model checkpoint restore took %d seconds', time.perf_counter() - start)
        model: Callable[[tf.Tensor], Dict[str, Any]] = get_model_detection_function(detection_model)
        inp: np.ndarray = np.zeros([2160, 3840, 3], dtype=np.uint8)
        input_tensor: tf.Tensor = tf.convert_to_tensor(inp, dtype=tf.float32)
        input_tensor = input_tensor[tf.newaxis, ...]
        model(input_tensor)
        _LOGGER.debug('Model load took %d seconds', time.perf_counter() - start)
        hass.data[DOMAIN][CONF_MODEL] = model

    hass.bus.listen_once(EVENT_HOMEASSISTANT_START, tensorflow_hass_start)
    category_index: Dict[int, Dict[str, Any]] = label_map_util.create_category_index_from_labelmap(labels, use_display_name=True)
    add_entities(
        (
            TensorFlowImageProcessor(
                hass,
                camera[CONF_ENTITY_ID],
                camera.get(CONF_NAME),
                category_index,
                config
            )
            for camera in config[CONF_SOURCE]
        )
    )


class TensorFlowImageProcessor(ImageProcessingEntity):
    """Representation of an TensorFlow image processor."""

    def __init__(self, hass: HomeAssistant, camera_entity: str, name: Optional[str], category_index: Dict[int, Dict[str, Any]], config: ConfigType) -> None:
        """Initialize the TensorFlow entity."""
        model_config: Dict[str, Any] = config.get(CONF_MODEL)
        self.hass: HomeAssistant = hass
        self._camera_entity: str = camera_entity
        if name:
            self._name: str = name
        else:
            self._name = f'TensorFlow {split_entity_id(camera_entity)[1]}'
        self._category_index: Dict[int, Dict[str, Any]] = category_index
        self._min_confidence: float = config.get(CONF_CONFIDENCE)
        self._file_out: List[Union[str, template.Template]] = config.get(CONF_FILE_OUT)
        self._label_id_offset: int = model_config.get(CONF_LABEL_OFFSET)
        categories: List[Any] = model_config.get(CONF_CATEGORIES)
        self._include_categories: List[str] = []
        self._category_areas: Dict[str, List[float]] = {}
        for category in categories:
            if isinstance(category, dict):
                category_name: str = category.get(CONF_CATEGORY)
                category_area: Optional[Dict[str, Any]] = category.get(CONF_AREA)
                self._include_categories.append(category_name)
                self._category_areas[category_name] = [0, 0, 1, 1]
                if category_area:
                    self._category_areas[category_name] = [
                        category_area.get(CONF_TOP), 
                        category_area.get(CONF_LEFT), 
                        category_area.get(CONF_BOTTOM), 
                        category_area.get(CONF_RIGHT)
                    ]
            else:
                self._include_categories.append(category)
                self._category_areas[category] = [0, 0, 1, 1]
        self._area: List[float] = [0, 0, 1, 1]
        if (area_config := model_config.get(CONF_AREA)):
            self._area = [
                area_config.get(CONF_TOP),
                area_config.get(CONF_LEFT),
                area_config.get(CONF_BOTTOM),
                area_config.get(CONF_RIGHT)
            ]
        self._matches: Dict[str, List[Dict[str, Any]]] = {}
        self._total_matches: int = 0
        self._last_image: Optional[Any] = None
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
            ATTR_SUMMARY: {category: len(values) for (category, values) in self._matches.items()},
            ATTR_TOTAL_MATCHES: self._total_matches,
            ATTR_PROCESS_TIME: self._process_time
        }

    def _save_image(self, image: bytes, matches: Dict[str, List[Dict[str, Any]]], paths: List[str]) -> None:
        img: Image.Image = Image.open(io.BytesIO(bytearray(image))).convert('RGB')
        img_width, img_height = img.size
        draw: ImageDraw.Draw = ImageDraw.Draw(img)
        if self._area != [0, 0, 1, 1]:
            draw_box(draw, self._area, img_width, img_height, 'Detection Area', (0, 255, 255))
        for category, values in matches.items():
            if category in self._category_areas and self._category_areas[category] != [0, 0, 1, 1]:
                label: str = f'{category.capitalize()} Detection Area'
                draw_box(draw, self._category_areas[category], img_width, img_height, label, (0, 255, 0))
            for instance in values:
                label = f"{category} {instance['score']:.1f}%"
                draw_box(draw, instance['box'], img_width, img_height, label, (255, 255, 0))
        for path in paths:
            _LOGGER.debug('Saving results image to %s', path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img.save(path)

    def process_image(self, image: bytes) -> None:
        """Process the image."""
        model: Optional[Callable[[tf.Tensor], Dict[str, Any]]] = self.hass.data[DOMAIN].get(CONF_MODEL)
        if not model:
            _LOGGER.debug('Model not yet ready')
            return
        start: float = time.perf_counter()
        try:
            import cv2  # type: ignore
            img_arr = np.asarray(bytearray(image))
            img_cv = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
            inp: np.ndarray = img_cv[:, :, [2, 1, 0]]
            inp_expanded: np.ndarray = inp.reshape(1, inp.shape[0], inp.shape[1], 3)
        except ImportError:
            try:
                img_pil: Image.Image = Image.open(io.BytesIO(bytearray(image))).convert('RGB')
            except UnidentifiedImageError:
                _LOGGER.warning('Unable to process image, bad data')
                return
            img_pil.thumbnail((460, 460), Image.ANTIALIAS)
            img_width, img_height = img_pil.size
            inp = np.array(list(img_pil.getdata())).reshape((img_height, img_width, 3)).astype(np.uint8)
            inp_expanded = np.expand_dims(inp, axis=0)
        input_tensor: tf.Tensor = tf.convert_to_tensor(inp_expanded, dtype=tf.float32)
        detections: Dict[str, Any] = model(input_tensor)
        boxes: np.ndarray = detections['detection_boxes'][0].numpy()
        scores: np.ndarray = detections['detection_scores'][0].numpy()
        classes: np.ndarray = (detections['detection_classes'][0].numpy() + self._label_id_offset).astype(int)
        matches: Dict[str, List[Dict[str, Any]]] = {}
        total_matches: int = 0
        for box, score, obj_class in zip(boxes, scores, classes, strict=False):
            score_percentage: float = score * 100
            box_list: List[float] = box.tolist()
            if score_percentage < self._min_confidence:
                continue
            if (box_list[0] < self._area[0] or box_list[1] < self._area[1] or
                    box_list[2] > self._area[2] or box_list[3] > self._area[3]):
                continue
            category: str = self._category_index[obj_class]['name']
            if self._include_categories and category not in self._include_categories:
                continue
            if self._category_areas and (
                box_list[0] < self._category_areas[category][0] or 
                box_list[1] < self._category_areas[category][1] or 
                box_list[2] > self._category_areas[category][2] or 
                box_list[3] > self._category_areas[category][3]
            ):
                continue
            if category not in matches:
                matches[category] = []
            matches[category].append({'score': float(score_percentage), 'box': box_list})
            total_matches += 1
        if total_matches and self._file_out:
            paths: List[str] = []
            for path_template in self._file_out:
                if isinstance(path_template, template.Template):
                    paths.append(path_template.render(camera_entity=self._camera_entity))
                else:
                    paths.append(path_template)
            self._save_image(image, matches, paths)
        self._matches = matches
        self._total_matches = total_matches
        self._process_time = time.perf_counter() - start
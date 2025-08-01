from __future__ import annotations
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image, ImageDraw, UnidentifiedImageError

import simplehound.core as hound
import voluptuous as vol
from homeassistant.components.image_processing import PLATFORM_SCHEMA as IMAGE_PROCESSING_PLATFORM_SCHEMA, ImageProcessingEntity
from homeassistant.const import ATTR_ENTITY_ID, CONF_API_KEY, CONF_ENTITY_ID, CONF_NAME, CONF_SOURCE
from homeassistant.core import HomeAssistant, split_entity_id
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util
from homeassistant.util.pil import draw_box

_LOGGER = logging.getLogger(__name__)
EVENT_PERSON_DETECTED = 'sighthound.person_detected'
ATTR_BOUNDING_BOX = 'bounding_box'
ATTR_PEOPLE = 'people'
CONF_ACCOUNT_TYPE = 'account_type'
CONF_SAVE_FILE_FOLDER = 'save_file_folder'
CONF_SAVE_TIMESTAMPTED_FILE = 'save_timestamped_file'
DATETIME_FORMAT = '%Y-%m-%d_%H:%M:%S'
DEV = 'dev'
PROD = 'prod'

PLATFORM_SCHEMA = IMAGE_PROCESSING_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_API_KEY): cv.string,
    vol.Optional(CONF_ACCOUNT_TYPE, default=DEV): vol.In([DEV, PROD]),
    vol.Optional(CONF_SAVE_FILE_FOLDER): cv.isdir,
    vol.Optional(CONF_SAVE_TIMESTAMPTED_FILE, default=False): cv.boolean
})


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    api_key: str = config[CONF_API_KEY]
    account_type: str = config[CONF_ACCOUNT_TYPE]
    api: Any = hound.cloud(api_key, account_type)
    try:
        api.detect(b'Test')
    except hound.SimplehoundException as exc:
        _LOGGER.error('Sighthound error %s setup aborted', exc)
        return
    save_file_folder_config: Optional[str] = config.get(CONF_SAVE_FILE_FOLDER)
    save_file_folder: Optional[Path] = Path(save_file_folder_config) if save_file_folder_config else None
    entities: List[SighthoundEntity] = []
    for camera in config[CONF_SOURCE]:
        sighthound: SighthoundEntity = SighthoundEntity(
            api,
            camera[CONF_ENTITY_ID],
            camera.get(CONF_NAME),
            save_file_folder,
            config[CONF_SAVE_TIMESTAMPTED_FILE]
        )
        entities.append(sighthound)
    add_entities(entities)


class SighthoundEntity(ImageProcessingEntity):
    _attr_should_poll: bool = False
    _attr_unit_of_measurement: str = ATTR_PEOPLE

    def __init__(
        self,
        api: Any,
        camera_entity: str,
        name: Optional[str],
        save_file_folder: Optional[Path],
        save_timestamped_file: bool
    ) -> None:
        self._api: Any = api
        self._camera: str = camera_entity
        if name is not None:
            self._name: str = name
        else:
            camera_name: str = split_entity_id(camera_entity)[1]
            self._name = f'sighthound_{camera_name}'
        self._state: Optional[int] = None
        self._last_detection: Optional[str] = None
        self._image_width: Optional[int] = None
        self._image_height: Optional[int] = None
        self._save_file_folder: Optional[Path] = save_file_folder
        self._save_timestamped_file: bool = save_timestamped_file

    def process_image(self, image: bytes) -> None:
        detections: Any = self._api.detect(image)
        people: List[Dict[str, Any]] = hound.get_people(detections)
        self._state = len(people)
        if self._state > 0:
            self._last_detection = dt_util.now().strftime(DATETIME_FORMAT)
        metadata: Dict[str, Any] = hound.get_metadata(detections)
        self._image_width = metadata['image_width']
        self._image_height = metadata['image_height']
        for person in people:
            self.fire_person_detected_event(person)
        if self._save_file_folder and self._state > 0:
            self.save_image(image, people, self._save_file_folder)

    def fire_person_detected_event(self, person: Dict[str, Any]) -> None:
        bounding_box: Any = hound.bbox_to_tf_style(person['boundingBox'], self._image_width, self._image_height)
        self.hass.bus.fire(
            EVENT_PERSON_DETECTED,
            {
                ATTR_ENTITY_ID: self.entity_id,
                ATTR_BOUNDING_BOX: bounding_box
            }
        )

    def save_image(self, image: bytes, people: List[Dict[str, Any]], directory: Path) -> None:
        try:
            img: Image.Image = Image.open(io.BytesIO(bytearray(image))).convert('RGB')
        except UnidentifiedImageError:
            _LOGGER.warning('Sighthound unable to process image, bad data')
            return
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(img)
        for person in people:
            box: Any = hound.bbox_to_tf_style(person['boundingBox'], self._image_width, self._image_height)
            draw_box(draw, box, self._image_width, self._image_height)
        latest_save_path: Path = directory / f'{self._name}_latest.jpg'
        img.save(latest_save_path)
        if self._save_timestamped_file:
            timestamp_save_path: Path = directory / f'{self._name}_{self._last_detection}.jpg'
            img.save(timestamp_save_path)
            _LOGGER.debug('Sighthound saved file %s', timestamp_save_path)

    @property
    def camera_entity(self) -> str:
        return self._camera

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> Optional[int]:
        return self._state

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        if not self._last_detection:
            return {}
        return {'last_person': self._last_detection}
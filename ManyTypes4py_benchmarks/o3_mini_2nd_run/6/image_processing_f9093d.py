from __future__ import annotations
import io
import logging
from typing import Any, Dict, List, Optional, Union

import face_recognition
import voluptuous as vol
from homeassistant.components.image_processing import (
    CONF_CONFIDENCE,
    PLATFORM_SCHEMA as IMAGE_PROCESSING_PLATFORM_SCHEMA,
    ImageProcessingFaceEntity,
)
from homeassistant.const import ATTR_NAME, CONF_ENTITY_ID, CONF_NAME, CONF_SOURCE
from homeassistant.core import HomeAssistant, split_entity_id
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER = logging.getLogger(__name__)

CONF_FACES = "faces"

PLATFORM_SCHEMA = IMAGE_PROCESSING_PLATFORM_SCHEMA.extend(
    {vol.Required(CONF_FACES): {cv.string: cv.isfile}, vol.Optional(CONF_CONFIDENCE, default=0.6): vol.Coerce(float)}
)


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the Dlib Face detection platform."""
    add_entities(
        (
            DlibFaceIdentifyEntity(
                camera[CONF_ENTITY_ID],
                config[CONF_FACES],
                camera.get(CONF_NAME),
                config[CONF_CONFIDENCE],
            )
            for camera in config[CONF_SOURCE]
        )
    )


class DlibFaceIdentifyEntity(ImageProcessingFaceEntity):
    """Dlib Face API entity for identify."""

    def __init__(self, camera_entity: str, faces: Dict[str, str], name: Optional[str], tolerance: float) -> None:
        """Initialize Dlib face identify entry."""
        super().__init__()
        self._camera: str = camera_entity
        if name:
            self._name: str = name
        else:
            self._name = f'Dlib Face {split_entity_id(camera_entity)[1]}'
        self._faces: Dict[str, Any] = {}
        for face_name, face_file in faces.items():
            try:
                image = face_recognition.load_image_file(face_file)
                self._faces[face_name] = face_recognition.face_encodings(image)[0]
            except IndexError as err:
                _LOGGER.error("Failed to parse %s. Error: %s", face_file, err)
        self._tolerance: float = tolerance

    @property
    def camera_entity(self) -> str:
        """Return camera entity id from process pictures."""
        return self._camera

    @property
    def name(self) -> str:
        """Return the name of the entity."""
        return self._name

    def process_image(self, image: bytes) -> None:
        """Process image."""
        fak_file: io.BytesIO = io.BytesIO(image)
        fak_file.name = "snapshot.jpg"
        fak_file.seek(0)
        image_data = face_recognition.load_image_file(fak_file)
        unknowns: List[Any] = face_recognition.face_encodings(image_data)
        found: List[Dict[str, str]] = []
        for unknown_face in unknowns:
            for name, face in self._faces.items():
                result = face_recognition.compare_faces([face], unknown_face, tolerance=self._tolerance)
                if result[0]:
                    found.append({ATTR_NAME: name})
        self.process_faces(found, len(unknowns))
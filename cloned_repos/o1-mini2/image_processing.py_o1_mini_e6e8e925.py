"""Component that will help set the OpenALPR cloud for ALPR processing."""

from __future__ import annotations

import asyncio
from base64 import b64encode
from http import HTTPStatus
import logging
from typing import Any, Dict, Optional, Set

import aiohttp
import voluptuous as vol

from homeassistant.components.image_processing import (
    ATTR_CONFIDENCE,
    CONF_CONFIDENCE,
    PLATFORM_SCHEMA as IMAGE_PROCESSING_PLATFORM_SCHEMA,
    ImageProcessingDeviceClass,
    ImageProcessingEntity,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    CONF_API_KEY,
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_REGION,
    CONF_SOURCE,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import ClientSession, async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util.async_ import run_callback_threadsafe

_LOGGER: logging.Logger = logging.getLogger(__name__)

ATTR_PLATE: str = "plate"
ATTR_PLATES: str = "plates"
ATTR_VEHICLES: str = "vehicles"

EVENT_FOUND_PLATE: str = "image_processing.found_plate"

OPENALPR_API_URL: str = "https://api.openalpr.com/v1/recognize"

OPENALPR_REGIONS: list[str] = [
    "au",
    "auwide",
    "br",
    "eu",
    "fr",
    "gb",
    "kr",
    "kr2",
    "mx",
    "sg",
    "us",
    "vn2",
]

PLATFORM_SCHEMA = IMAGE_PROCESSING_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_API_KEY): cv.string,
        vol.Required(CONF_REGION): vol.All(vol.Lower, vol.In(OPENALPR_REGIONS)),
    }
)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the OpenALPR cloud API platform."""
    confidence: Optional[float] = config.get(CONF_CONFIDENCE)
    params: Dict[str, Any] = {
        "secret_key": config[CONF_API_KEY],
        "tasks": "plate",
        "return_image": 0,
        "country": config[CONF_REGION],
    }

    sources: list[Dict[str, Any]] = config.get(CONF_SOURCE, [])

    async_add_entities(
        OpenAlprCloudEntity(
            camera.get(CONF_ENTITY_ID),
            params,
            confidence,
            camera.get(CONF_NAME),
        )
        for camera in sources
    )


class ImageProcessingAlprEntity(ImageProcessingEntity):
    """Base entity class for ALPR image processing."""

    _attr_device_class: ImageProcessingDeviceClass = ImageProcessingDeviceClass.ALPR

    plates: Dict[str, float]
    vehicles: int

    def __init__(self) -> None:
        """Initialize base ALPR entity."""
        super().__init__()
        self.plates = {}
        self.vehicles = 0

    @property
    def state(self) -> Optional[str]:
        """Return the state of the entity."""
        confidence: float = 0.0
        plate: Optional[str] = None

        # search high plate
        for i_pl, i_co in self.plates.items():
            if i_co > confidence:
                confidence = i_co
                plate = i_pl
        return plate

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return device specific state attributes."""
        return {ATTR_PLATES: self.plates, ATTR_VEHICLES: self.vehicles}

    def process_plates(self, plates: Dict[str, float], vehicles: int) -> None:
        """Send event with new plates and store data."""
        run_callback_threadsafe(
            self.hass.loop, self.async_process_plates, plates, vehicles
        ).result()

    @callback
    def async_process_plates(
        self, plates: Dict[str, float], vehicles: int
    ) -> None:
        """Send event with new plates and store data.

        Plates are a dict in follow format:
          { '<plate>': confidence }
        This method must be run in the event loop.
        """
        filtered_plates: Dict[str, float] = {
            plate: confidence
            for plate, confidence in plates.items()
            if self.confidence is None or confidence >= self.confidence
        }
        new_plates: Set[str] = set(filtered_plates) - set(self.plates)

        # Send events
        for i_plate in new_plates:
            self.hass.bus.async_fire(
                EVENT_FOUND_PLATE,
                {
                    ATTR_PLATE: i_plate,
                    ATTR_ENTITY_ID: self.entity_id,
                    ATTR_CONFIDENCE: filtered_plates.get(i_plate),
                },
            )

        # Update entity store
        self.plates = filtered_plates
        self.vehicles = vehicles


class OpenAlprCloudEntity(ImageProcessingAlprEntity):
    """Representation of an OpenALPR cloud entity."""

    _params: Dict[str, Any]
    _camera: str
    _confidence: Optional[float]
    _name: str

    def __init__(
        self,
        camera_entity: str,
        params: Dict[str, Any],
        confidence: Optional[float],
        name: Optional[str] = None,
    ) -> None:
        """Initialize OpenALPR cloud API."""
        super().__init__()

        self._params = params
        self._camera = camera_entity
        self._confidence = confidence

        if name:
            self._name = name
        else:
            _, camera_id = split_entity_id(camera_entity)
            self._name = f"OpenAlpr {camera_id}"

    @property
    def confidence(self) -> Optional[float]:
        """Return minimum confidence for send events."""
        return self._confidence

    @property
    def camera_entity(self) -> str:
        """Return camera entity id from process pictures."""
        return self._camera

    @property
    def name(self) -> str:
        """Return the name of the entity."""
        return self._name

    async def async_process_image(self, image: bytes) -> None:
        """Process image.

        This method is a coroutine.
        """
        websession: ClientSession = async_get_clientsession(self.hass)
        params: Dict[str, Any] = self._params.copy()

        body: Dict[str, str] = {"image_bytes": b64encode(image).decode("utf-8")}

        try:
            async with asyncio.timeout(self.timeout):
                request: aiohttp.ClientResponse = await websession.post(
                    OPENALPR_API_URL, params=params, data=body
                )

                data: Dict[str, Any] = await request.json()

                if request.status != HTTPStatus.OK:
                    _LOGGER.error("Error %d -> %s", request.status, data.get("error"))
                    return

        except (asyncio.TimeoutError, aiohttp.ClientError):
            _LOGGER.error("Timeout for OpenALPR API")
            return

        # Processing API data
        vehicles: int = 0
        result: Dict[str, float] = {}

        plate_results: list[Dict[str, Any]] = data.get("plate", {}).get("results", [])

        for row in plate_results:
            vehicles += 1

            candidates: list[Dict[str, Any]] = row.get("candidates", [])
            for p_data in candidates:
                plate: Optional[str] = p_data.get("plate")
                confidence_str: Optional[str] = p_data.get("confidence")
                if plate is not None and confidence_str is not None:
                    try:
                        confidence: float = float(confidence_str)
                        result[plate] = confidence
                    except ValueError:
                        continue

        self.async_process_plates(result, vehicles)

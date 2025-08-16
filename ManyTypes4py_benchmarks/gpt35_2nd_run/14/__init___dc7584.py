from __future__ import annotations
import asyncio
from collections.abc import Coroutine
import json
import logging
from typing import Any, Dict, List, Optional
import aiohttp
from aiohttp.hdrs import CONTENT_TYPE
import voluptuous as vol
from homeassistant.components import camera
from homeassistant.const import ATTR_NAME, CONF_API_KEY, CONF_TIMEOUT, CONTENT_TYPE_JSON
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import slugify

_LOGGER: logging.Logger = logging.getLogger(__name__)
ATTR_CAMERA_ENTITY: str = 'camera_entity'
ATTR_GROUP: str = 'group'
ATTR_PERSON: str = 'person'
CONF_AZURE_REGION: str = 'azure_region'
DATA_MICROSOFT_FACE: str = 'microsoft_face'
DEFAULT_TIMEOUT: int = 10
DOMAIN: str = 'microsoft_face'
FACE_API_URL: str = 'api.cognitive.microsoft.com/face/v1.0/{0}'
SERVICE_CREATE_GROUP: str = 'create_group'
SERVICE_CREATE_PERSON: str = 'create_person'
SERVICE_DELETE_GROUP: str = 'delete_group'
SERVICE_DELETE_PERSON: str = 'delete_person'
SERVICE_FACE_PERSON: str = 'face_person'
SERVICE_TRAIN_GROUP: str = 'train_group'

CONFIG_SCHEMA: vol.Schema = vol.Schema({
    DOMAIN: vol.Schema({
        vol.Required(CONF_API_KEY): cv.string,
        vol.Optional(CONF_AZURE_REGION, default='westus'): cv.string,
        vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): cv.positive_int
    })
}, extra=vol.ALLOW_EXTRA)

SCHEMA_GROUP_SERVICE: vol.Schema = vol.Schema({
    vol.Required(ATTR_NAME): cv.string
})

SCHEMA_PERSON_SERVICE: vol.Schema = SCHEMA_GROUP_SERVICE.extend({
    vol.Required(ATTR_GROUP): cv.slugify
})

SCHEMA_FACE_SERVICE: vol.Schema = vol.Schema({
    vol.Required(ATTR_PERSON): cv.string,
    vol.Required(ATTR_GROUP): cv.slugify,
    vol.Required(ATTR_CAMERA_ENTITY): cv.entity_id
})

SCHEMA_TRAIN_SERVICE: vol.Schema = vol.Schema({
    vol.Required(ATTR_GROUP): cv.slugify
})

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    component: EntityComponent[MicrosoftFaceGroupEntity] = EntityComponent[MicrosoftFaceGroupEntity](logging.getLogger(__name__), DOMAIN, hass)
    entities: Dict[str, MicrosoftFaceGroupEntity] = {}
    face: MicrosoftFace = MicrosoftFace(hass, config[DOMAIN].get(CONF_AZURE_REGION), config[DOMAIN].get(CONF_API_KEY), config[DOMAIN].get(CONF_TIMEOUT), component, entities)
    
    async def async_create_group(service: ServiceCall) -> None:
    
    async def async_delete_group(service: ServiceCall) -> None:
    
    async def async_train_group(service: ServiceCall) -> None:
    
    async def async_create_person(service: ServiceCall) -> None:
    
    async def async_delete_person(service: ServiceCall) -> None:
    
    async def async_face_person(service: ServiceCall) -> None:
    
    return True

class MicrosoftFaceGroupEntity(Entity):
    _attr_should_poll: bool = False

    def __init__(self, hass: HomeAssistant, api: MicrosoftFace, g_id: str, name: str) -> None:
        self.hass: HomeAssistant = hass
        self._api: MicrosoftFace = api
        self._id: str = g_id
        self._name: str = name

    @property
    def name(self) -> str:
    
    @property
    def entity_id(self) -> str:
    
    @property
    def state(self) -> int:
    
    @property
    def extra_state_attributes(self) -> Dict[str, Any]:

class MicrosoftFace:
    def __init__(self, hass: HomeAssistant, server_loc: str, api_key: str, timeout: int, component: EntityComponent, entities: Dict[str, MicrosoftFaceGroupEntity]) -> None:
    
    @property
    def store(self) -> Dict[str, Dict[str, str]]:
    
    async def update_store(self) -> None:
    
    async def call_api(self, method: str, function: str, data: Optional[Dict[str, Any]] = None, binary: bool = False, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

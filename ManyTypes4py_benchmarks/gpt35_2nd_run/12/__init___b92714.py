from __future__ import annotations
import asyncio
import logging
import pathlib
import secrets
import shutil
from typing import Any, Dict
from aiohttp import hdrs, web
from aiohttp.web_request import FileField
from PIL import Image, ImageOps, UnidentifiedImageError
import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.components.http import KEY_HASS, HomeAssistantView
from homeassistant.components.http.static import CACHE_HEADERS
from homeassistant.const import CONF_ID
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import collection, config_validation as cv
from homeassistant.helpers.storage import Store
from homeassistant.helpers.typing import ConfigType, VolDictType
from homeassistant.util import dt as dt_util
from .const import DOMAIN

_LOGGER: logging.Logger = logging.getLogger(__name__)
STORAGE_KEY: str = 'image'
STORAGE_VERSION: int = 1
VALID_SIZES: set[int] = {256, 512}
MAX_SIZE: int = 1024 * 1024 * 10
CREATE_FIELDS: Dict[str, Any] = {vol.Required('file'): FileField}
UPDATE_FIELDS: Dict[str, Any] = {vol.Optional('name'): vol.All(str, vol.Length(min=1))}
CONFIG_SCHEMA: vol.Schema = cv.empty_config_schema(DOMAIN)

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    image_dir: pathlib.Path = pathlib.Path(hass.config.path('image'))
    hass.data[DOMAIN]: ImageStorageCollection = storage_collection = ImageStorageCollection(hass, image_dir)
    await storage_collection.async_load()
    ImageUploadStorageCollectionWebsocket(storage_collection, 'image', 'image', CREATE_FIELDS, UPDATE_FIELDS).async_setup(hass)
    hass.http.register_view(ImageUploadView)
    hass.http.register_view(ImageServeView(image_dir, storage_collection))
    return True

class ImageStorageCollection(collection.DictStorageCollection):
    CREATE_SCHEMA: vol.Schema = vol.Schema(CREATE_FIELDS)
    UPDATE_SCHEMA: vol.Schema = vol.Schema(UPDATE_FIELDS)

    def __init__(self, hass: HomeAssistant, image_dir: pathlib.Path) -> None:
        super().__init__(Store(hass, STORAGE_VERSION, STORAGE_KEY))
        self.async_add_listener(self._change_listener)
        self.image_dir: pathlib.Path = image_dir

    async def _process_create_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = self.CREATE_SCHEMA(dict(data))
        uploaded_file: FileField = data['file']
        if uploaded_file.content_type not in ('image/gif', 'image/jpeg', 'image/png'):
            raise vol.Invalid('Only jpeg, png, and gif images are allowed')
        data[CONF_ID]: str = secrets.token_hex(16)
        data['filesize']: int = await self.hass.async_add_executor_job(self._move_data, data)
        data['content_type']: str = uploaded_file.content_type
        data['name']: str = uploaded_file.filename
        data['uploaded_at']: str = dt_util.utcnow().isoformat()
        return data

    def _move_data(self, data: Dict[str, Any]) -> int:
        uploaded_file: FileField = data.pop('file')
        try:
            image: Image.Image = Image.open(uploaded_file.file)
        except UnidentifiedImageError as err:
            raise vol.Invalid('Unable to identify image file') from err
        uploaded_file.file.seek(0)
        media_folder: pathlib.Path = self.image_dir / data[CONF_ID]
        media_folder.mkdir(parents=True)
        media_file: pathlib.Path = media_folder / 'original'
        media_file.relative_to(media_folder)
        _LOGGER.debug('Storing file %s', media_file)
        with media_file.open('wb') as target:
            shutil.copyfileobj(uploaded_file.file, target)
        image.close()
        return media_file.stat().st_size

    @callback
    def _get_suggested_id(self, info: Dict[str, Any]) -> str:
        return str(info[CONF_ID])

    async def _update_data(self, item: Dict[str, Any], update_data: Dict[str, Any]) -> Dict[str, Any]:
        return {**item, **self.UPDATE_SCHEMA(update_data)}

    async def _change_listener(self, change_type: str, item_id: str, data: Dict[str, Any]) -> None:
        if change_type != collection.CHANGE_REMOVED:
            return
        await self.hass.async_add_executor_job(shutil.rmtree, self.image_dir / item_id)

class ImageUploadStorageCollectionWebsocket(collection.DictStorageCollectionWebsocket):
    async def ws_create_item(self, hass: HomeAssistant, connection: Any, msg: Any) -> None:
        raise NotImplementedError

class ImageUploadView(HomeAssistantView):
    async def post(self, request: web.Request) -> web.Response:
        request._client_max_size = MAX_SIZE
        data: Dict[str, Any] = await request.post()
        item: Dict[str, Any] = await request.app[KEY_HASS].data[DOMAIN].async_create_item(data)
        return self.json(item)

class ImageServeView(HomeAssistantView):
    def __init__(self, image_folder: pathlib.Path, image_collection: ImageStorageCollection) -> None:
        self.transform_lock: asyncio.Lock = asyncio.Lock()
        self.image_folder: pathlib.Path = image_folder
        self.image_collection: ImageStorageCollection = image_collection

    async def get(self, request: web.Request, image_id: str, filename: str) -> web.FileResponse:
        image_info: Dict[str, Any] = self.image_collection.data.get(image_id)
        if image_info is None:
            raise web.HTTPNotFound
        if filename == 'original':
            target_file: pathlib.Path = self.image_folder / image_id / filename
        else:
            try:
                width, height = _validate_size_from_filename(filename)
            except (ValueError, IndexError) as err:
                raise web.HTTPBadRequest from err
            hass: HomeAssistant = request.app[KEY_HASS]
            target_file: pathlib.Path = self.image_folder / image_id / f'{width}x{height}'
            if not await hass.async_add_executor_job(target_file.is_file):
                async with self.transform_lock:
                    await hass.async_add_executor_job(_generate_thumbnail_if_file_does_not_exist, target_file, self.image_folder / image_id / 'original', image_info['content_type'], target_file, (width, height))
        return web.FileResponse(target_file, headers={**CACHE_HEADERS, hdrs.CONTENT_TYPE: image_info['content_type']})

def _generate_thumbnail_if_file_does_not_exist(target_file: pathlib.Path, original_path: pathlib.Path, content_type: str, target_path: pathlib.Path, target_size: tuple[int, int]) -> None:
    if not target_file.is_file():
        image: Image.Image = ImageOps.exif_transpose(Image.open(original_path))
        image.thumbnail(target_size)
        image.save(target_path, format=content_type.partition('/')[-1])

def _validate_size_from_filename(filename: str) -> tuple[int, int]:
    image_size: str = filename.partition('-')[0]
    if not image_size:
        raise ValueError('Invalid filename')
    width_s, _, height_s = image_size.partition('x')
    width: int = int(width_s)
    height: int = int(height_s)
    if not width or width != height or width not in VALID_SIZES:
        raise ValueError(f'Invalid size {image_size}')
    return (width, height)

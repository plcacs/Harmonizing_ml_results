from __future__ import annotations
import asyncio
import base64
import binascii
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import functools
from typing import Any
from aiohttp import web
from hyperion import client
from hyperion.const import KEY_IMAGE, KEY_IMAGE_STREAM, KEY_LEDCOLORS, KEY_RESULT, KEY_UPDATE
from homeassistant.components.camera import DEFAULT_CONTENT_TYPE, Camera, async_get_still_stream
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import get_hyperion_device_id, get_hyperion_unique_id, listen_for_instance_updates
from .const import CONF_INSTANCE_CLIENTS, DOMAIN, HYPERION_MANUFACTURER_NAME, HYPERION_MODEL_NAME, SIGNAL_ENTITY_REMOVE, TYPE_HYPERION_CAMERA
IMAGE_STREAM_JPG_SENTINEL = 'data:image/jpg;base64,'

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    entry_data = hass.data[DOMAIN][config_entry.entry_id]
    server_id = config_entry.unique_id

    def camera_unique_id(instance_num: int) -> str:
        assert server_id
        return get_hyperion_unique_id(server_id, instance_num, TYPE_HYPERION_CAMERA)

    @callback
    def instance_add(instance_num: int, instance_name: str) -> None:
        assert server_id
        async_add_entities([HyperionCamera(server_id, instance_num, instance_name, entry_data[CONF_INSTANCE_CLIENTS][instance_num])])

    @callback
    def instance_remove(instance_num: int) -> None:
        assert server_id
        async_dispatcher_send(hass, SIGNAL_ENTITY_REMOVE.format(camera_unique_id(instance_num)))
    listen_for_instance_updates(hass, config_entry, instance_add, instance_remove)

class HyperionCamera(Camera):
    _attr_has_entity_name: bool = True
    _attr_name: str = None

    def __init__(self, server_id: str, instance_num: int, instance_name: str, hyperion_client: Any) -> None:
        super().__init__()
        self._attr_unique_id = get_hyperion_unique_id(server_id, instance_num, TYPE_HYPERION_CAMERA)
        self._device_id = get_hyperion_device_id(server_id, instance_num)
        self._instance_name = instance_name
        self._client = hyperion_client
        self._image_cond = asyncio.Condition()
        self._image = None
        self._image_stream_clients = 0
        self._client_callbacks = {f'{KEY_LEDCOLORS}-{KEY_IMAGE_STREAM}-{KEY_UPDATE}': self._update_imagestream}
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, self._device_id)}, manufacturer=HYPERION_MANUFACTURER_NAME, model=HYPERION_MODEL_NAME, name=instance_name, configuration_url=hyperion_client.remote_url)

    @property
    def is_on(self) -> bool:
        return self.available

    @property
    def available(self) -> bool:
        return bool(self._client.has_loaded_state)

    async def _update_imagestream(self, img: Any = None) -> None:
        if not img:
            return
        img_data = img.get(KEY_RESULT, {}).get(KEY_IMAGE)
        if not img_data or not img_data.startswith(IMAGE_STREAM_JPG_SENTINEL):
            return
        async with self._image_cond:
            try:
                self._image = base64.b64decode(img_data.removeprefix(IMAGE_STREAM_JPG_SENTINEL))
            except binascii.Error:
                return
            self._image_cond.notify_all()

    async def _async_wait_for_camera_image(self) -> Any:
        async with self._image_cond:
            await self._image_cond.wait()
            return self._image if self.available else None

    async def _start_image_streaming_for_client(self) -> bool:
        if not self._image_stream_clients and (not await self._client.async_send_image_stream_start()):
            return False
        self._image_stream_clients += 1
        self._attr_is_streaming = True
        self.async_write_ha_state()
        return True

    async def _stop_image_streaming_for_client(self) -> None:
        self._image_stream_clients -= 1
        if not self._image_stream_clients:
            await self._client.async_send_image_stream_stop()
            self._attr_is_streaming = False
            self.async_write_ha_state()

    @asynccontextmanager
    async def _image_streaming(self) -> AsyncGenerator[bool, None]:
        try:
            yield (await self._start_image_streaming_for_client())
        finally:
            await self._stop_image_streaming_for_client()

    async def async_camera_image(self, width: int = None, height: int = None) -> Any:
        async with self._image_streaming() as is_streaming:
            if is_streaming:
                return await self._async_wait_for_camera_image()
        return None

    async def handle_async_mjpeg_stream(self, request: web.Request) -> Any:
        async with self._image_streaming() as is_streaming:
            if is_streaming:
                return await async_get_still_stream(request, self._async_wait_for_camera_image, DEFAULT_CONTENT_TYPE, 0.0)
        return None

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_ENTITY_REMOVE.format(self._attr_unique_id), functools.partial(self.async_remove, force_remove=True)))
        self._client.add_callbacks(self._client_callbacks)

    async def async_will_remove_from_hass(self) -> None:
        self._client.remove_callbacks(self._client_callbacks)
CAMERA_TYPES: dict[str, Any] = {TYPE_HYPERION_CAMERA: HyperionCamera}

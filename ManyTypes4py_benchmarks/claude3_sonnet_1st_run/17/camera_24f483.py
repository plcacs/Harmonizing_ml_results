"""Switch platform for Hyperion."""
from __future__ import annotations
import asyncio
import base64
import binascii
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import functools
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast
from aiohttp import web
from hyperion import client
from hyperion.const import KEY_IMAGE, KEY_IMAGE_STREAM, KEY_LEDCOLORS, KEY_RESULT, KEY_UPDATE
from homeassistant.components.camera import DEFAULT_CONTENT_TYPE, Camera, async_get_still_stream
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from . import get_hyperion_device_id, get_hyperion_unique_id, listen_for_instance_updates
from .const import CONF_INSTANCE_CLIENTS, DOMAIN, HYPERION_MANUFACTURER_NAME, HYPERION_MODEL_NAME, SIGNAL_ENTITY_REMOVE, TYPE_HYPERION_CAMERA
IMAGE_STREAM_JPG_SENTINEL: str = 'data:image/jpg;base64,'

async def async_setup_entry(
    hass: HomeAssistant, 
    config_entry: ConfigEntry, 
    async_add_entities: AddEntitiesCallback
) -> None:
    """Set up a Hyperion platform from config entry."""
    entry_data: Dict[str, Any] = hass.data[DOMAIN][config_entry.entry_id]
    server_id: Optional[str] = config_entry.unique_id

    def camera_unique_id(instance_num: int) -> str:
        """Return the camera unique_id."""
        assert server_id
        return get_hyperion_unique_id(server_id, instance_num, TYPE_HYPERION_CAMERA)

    @callback
    def instance_add(instance_num: int, instance_name: str) -> None:
        """Add entities for a new Hyperion instance."""
        assert server_id
        async_add_entities([HyperionCamera(server_id, instance_num, instance_name, entry_data[CONF_INSTANCE_CLIENTS][instance_num])])

    @callback
    def instance_remove(instance_num: int) -> None:
        """Remove entities for an old Hyperion instance."""
        assert server_id
        async_dispatcher_send(hass, SIGNAL_ENTITY_REMOVE.format(camera_unique_id(instance_num)))
    listen_for_instance_updates(hass, config_entry, instance_add, instance_remove)

class HyperionCamera(Camera):
    """ComponentBinarySwitch switch class."""
    _attr_has_entity_name: bool = True
    _attr_name: Optional[str] = None

    def __init__(
        self, 
        server_id: str, 
        instance_num: int, 
        instance_name: str, 
        hyperion_client: client.HyperionClient
    ) -> None:
        """Initialize the switch."""
        super().__init__()
        self._attr_unique_id: str = get_hyperion_unique_id(server_id, instance_num, TYPE_HYPERION_CAMERA)
        self._device_id: str = get_hyperion_device_id(server_id, instance_num)
        self._instance_name: str = instance_name
        self._client: client.HyperionClient = hyperion_client
        self._image_cond: asyncio.Condition = asyncio.Condition()
        self._image: Optional[bytes] = None
        self._image_stream_clients: int = 0
        self._client_callbacks: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            f'{KEY_LEDCOLORS}-{KEY_IMAGE_STREAM}-{KEY_UPDATE}': self._update_imagestream
        }
        self._attr_device_info: DeviceInfo = DeviceInfo(
            identifiers={(DOMAIN, self._device_id)}, 
            manufacturer=HYPERION_MANUFACTURER_NAME, 
            model=HYPERION_MODEL_NAME, 
            name=instance_name, 
            configuration_url=hyperion_client.remote_url
        )

    @property
    def is_on(self) -> bool:
        """Return true if the camera is on."""
        return self.available

    @property
    def available(self) -> bool:
        """Return server availability."""
        return bool(self._client.has_loaded_state)

    async def _update_imagestream(self, img: Optional[Dict[str, Any]] = None) -> None:
        """Update Hyperion components."""
        if not img:
            return
        img_data: Optional[str] = img.get(KEY_RESULT, {}).get(KEY_IMAGE)
        if not img_data or not img_data.startswith(IMAGE_STREAM_JPG_SENTINEL):
            return
        async with self._image_cond:
            try:
                self._image = base64.b64decode(img_data.removeprefix(IMAGE_STREAM_JPG_SENTINEL))
            except binascii.Error:
                return
            self._image_cond.notify_all()

    async def _async_wait_for_camera_image(self) -> Optional[bytes]:
        """Return a single camera image in a stream."""
        async with self._image_cond:
            await self._image_cond.wait()
            return self._image if self.available else None

    async def _start_image_streaming_for_client(self) -> bool:
        """Start streaming for a client."""
        if not self._image_stream_clients and (not await self._client.async_send_image_stream_start()):
            return False
        self._image_stream_clients += 1
        self._attr_is_streaming = True
        self.async_write_ha_state()
        return True

    async def _stop_image_streaming_for_client(self) -> None:
        """Stop streaming for a client."""
        self._image_stream_clients -= 1
        if not self._image_stream_clients:
            await self._client.async_send_image_stream_stop()
            self._attr_is_streaming = False
            self.async_write_ha_state()

    @asynccontextmanager
    async def _image_streaming(self) -> AsyncGenerator[bool, None]:
        """Async context manager to start/stop image streaming."""
        try:
            yield (await self._start_image_streaming_for_client())
        finally:
            await self._stop_image_streaming_for_client()

    async def async_camera_image(
        self, 
        width: Optional[int] = None, 
        height: Optional[int] = None
    ) -> Optional[bytes]:
        """Return single camera image bytes."""
        async with self._image_streaming() as is_streaming:
            if is_streaming:
                return await self._async_wait_for_camera_image()
        return None

    async def handle_async_mjpeg_stream(self, request: web.Request) -> Optional[web.StreamResponse]:
        """Serve an HTTP MJPEG stream from the camera."""
        async with self._image_streaming() as is_streaming:
            if is_streaming:
                return await async_get_still_stream(request, self._async_wait_for_camera_image, DEFAULT_CONTENT_TYPE, 0.0)
        return None

    async def async_added_to_hass(self) -> None:
        """Register callbacks when entity added to hass."""
        self.async_on_remove(async_dispatcher_connect(
            self.hass, 
            SIGNAL_ENTITY_REMOVE.format(self._attr_unique_id), 
            functools.partial(self.async_remove, force_remove=True)
        ))
        self._client.add_callbacks(self._client_callbacks)

    async def async_will_remove_from_hass(self) -> None:
        """Cleanup prior to hass removal."""
        self._client.remove_callbacks(self._client_callbacks)

CAMERA_TYPES: Dict[str, type[HyperionCamera]] = {TYPE_HYPERION_CAMERA: HyperionCamera}

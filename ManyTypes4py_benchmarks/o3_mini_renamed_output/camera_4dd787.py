from __future__ import annotations
from datetime import timedelta
import logging
from typing import Final, Any, Optional, Dict, Generator
from aiohttp.web import Request, StreamResponse
from canary.live_stream_api import LiveStreamSession
from canary.model import Device, Location
from haffmpeg.camera import CameraMjpeg
import voluptuous as vol
from homeassistant.components import ffmpeg
from homeassistant.components.camera import PLATFORM_SCHEMA as CAMERA_PLATFORM_SCHEMA, Camera
from homeassistant.components.ffmpeg import FFmpegManager, get_ffmpeg_manager
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_aiohttp_proxy_stream
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util
from .const import CONF_FFMPEG_ARGUMENTS, DEFAULT_FFMPEG_ARGUMENTS, DOMAIN, MANUFACTURER
from .coordinator import CanaryConfigEntry, CanaryDataUpdateCoordinator

FORCE_CAMERA_REFRESH_INTERVAL: Final = timedelta(minutes=15)

PLATFORM_SCHEMA = vol.All(
    cv.deprecated(CONF_FFMPEG_ARGUMENTS),
    CAMERA_PLATFORM_SCHEMA.extend(
        {
            vol.Optional(CONF_FFMPEG_ARGUMENTS, default=DEFAULT_FFMPEG_ARGUMENTS): cv.string
        }
    ),
)

_LOGGER = logging.getLogger(__name__)


async def func_ehghn4iz(
    hass: HomeAssistant,
    entry: CanaryConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Canary sensors based on a config entry."""
    coordinator: CanaryDataUpdateCoordinator = entry.runtime_data
    ffmpeg_arguments: str = entry.options.get(CONF_FFMPEG_ARGUMENTS, DEFAULT_FFMPEG_ARGUMENTS)
    async_add_entities(
        (
            CanaryCamera(hass, coordinator, location_id, device, ffmpeg_arguments)
            for location_id, location in coordinator.data["locations"].items()
            for device in location.devices
            if device.is_online
        ),
        True,
    )


class CanaryCamera(CoordinatorEntity[CanaryDataUpdateCoordinator], Camera):
    """An implementation of a Canary security camera."""

    def __init__(
        self,
        hass: HomeAssistant,
        coordinator: CanaryDataUpdateCoordinator,
        location_id: str,
        device: Device,
        ffmpeg_args: str,
    ) -> None:
        """Initialize a Canary security camera."""
        super().__init__(coordinator)
        Camera.__init__(self)
        self._ffmpeg: FFmpegManager = get_ffmpeg_manager(hass)
        self._ffmpeg_arguments: str = ffmpeg_args
        self._location_id: str = location_id
        self._device: Device = device
        self._live_stream_session: Optional[LiveStreamSession] = None
        self._attr_name: str = device.name
        self._attr_unique_id: str = str(device.device_id)
        self._attr_device_info: DeviceInfo = DeviceInfo(
            identifiers={(DOMAIN, str(device.device_id))},
            manufacturer=MANUFACTURER,
            model=device.device_type["name"],
            name=device.name,
        )
        self._image: Optional[bytes] = None
        self._expires_at = dt_util.utcnow()
        _LOGGER.debug("%s %s has been initialized", self.name, device.device_type["name"])

    @property
    def func_4lm9v1gj(self) -> Location:
        """Return information about the location."""
        return self.coordinator.data["locations"][self._location_id]

    @property
    def func_809xdcgj(self) -> bool:
        """Return true if the device is recording."""
        # Assuming self.func_4lm9v1gj is the location with an attribute is_recording.
        return self.func_4lm9v1gj.is_recording

    @property
    def func_m135g15n(self) -> bool:
        """Return the camera motion detection status."""
        return not self.func_4lm9v1gj.is_recording

    async def func_wl0eosgi(self, width: Optional[int] = None, height: Optional[int] = None) -> Optional[bytes]:
        """Return a still image response from the camera."""
        utcnow = dt_util.utcnow()
        if self._expires_at <= utcnow:
            _LOGGER.debug("Grabbing a live view image from %s", self.name)
            await self.hass.async_add_executor_job(self.func_vroiiggs)
            live_stream_session: Optional[LiveStreamSession] = self._live_stream_session
            if live_stream_session is None:
                return None
            if not (live_stream_url := live_stream_session.live_stream_url):
                return None
            image: Optional[bytes] = await ffmpeg.async_get_image(
                self.hass, live_stream_url, extra_cmd=self._ffmpeg_arguments, width=width, height=height
            )
            if image:
                self._image = image
                self._expires_at = FORCE_CAMERA_REFRESH_INTERVAL + utcnow
                _LOGGER.debug("Grabbed a live view image from %s", self.name)
            await self.hass.async_add_executor_job(live_stream_session.stop_session)
            _LOGGER.debug("Stopped live session from %s", self.name)
        return self._image

    async def func_1flwg7sn(self, request: Request) -> Optional[StreamResponse]:
        """Generate an HTTP MJPEG stream from the camera."""
        if self._live_stream_session is None:
            return None
        live_stream_url: str = await self.hass.async_add_executor_job(getattr, self._live_stream_session, "live_stream_url")
        stream: CameraMjpeg = CameraMjpeg(self._ffmpeg.binary)
        await stream.open_camera(live_stream_url, extra_cmd=self._ffmpeg_arguments)
        try:
            stream_reader: Any = await stream.get_reader()
            return await async_aiohttp_proxy_stream(self.hass, request, stream_reader, self._ffmpeg.ffmpeg_stream_content_type)
        finally:
            await stream.close()

    def func_vroiiggs(self) -> None:
        """Renew live stream session."""
        self._live_stream_session = self.coordinator.canary.get_live_stream_session(self._device)
        _LOGGER.debug("Live Stream URL for %s is %s", self.name, self._live_stream_session.live_stream_url)
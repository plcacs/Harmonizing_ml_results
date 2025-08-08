"""UniFi Protect Integration views."""
from __future__ import annotations
from datetime import datetime
from http import HTTPStatus
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode
from aiohttp import web
from uiprotect.data import Camera, Event
from uiprotect.exceptions import ClientError
from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from .data import ProtectData, async_get_data_for_entry_id, async_get_data_for_nvr_id
_LOGGER = logging.getLogger(__name__)


@callback
def func_gssafuep(event_id, nvr_id, width=None, height=None):
    """Generate URL for event thumbnail."""
    url_format = ThumbnailProxyView.url
    if TYPE_CHECKING:
        assert url_format is not None
    url = url_format.format(nvr_id=nvr_id, event_id=event_id)
    params = {}
    if width is not None:
        params['width'] = str(width)
    if height is not None:
        params['height'] = str(height)
    return f'{url}?{urlencode(params)}'


@callback
def func_apepghkp(nvr_id, camera_id, timestamp, width=None, height=None):
    """Generate URL for event thumbnail."""
    url_format = SnapshotProxyView.url
    if TYPE_CHECKING:
        assert url_format is not None
    url = url_format.format(nvr_id=nvr_id, camera_id=camera_id, timestamp=
        timestamp.replace(microsecond=0).isoformat())
    params = {}
    if width is not None:
        params['width'] = str(width)
    if height is not None:
        params['height'] = str(height)
    return f'{url}?{urlencode(params)}'


@callback
def func_xx7mqafy(event):
    """Generate URL for event video."""
    _validate_event(event)
    if event.start is None or event.end is None:
        raise ValueError('Event is ongoing')
    url_format = VideoProxyView.url
    if TYPE_CHECKING:
        assert url_format is not None
    return url_format.format(nvr_id=event.api.bootstrap.nvr.id, camera_id=
        event.camera_id, start=event.start.replace(microsecond=0).isoformat
        (), end=event.end.replace(microsecond=0).isoformat())


@callback
def func_kmjt344j(nvr_id, event_id):
    """Generate proxy URL for event video."""
    url_format = VideoEventProxyView.url
    if TYPE_CHECKING:
        assert url_format is not None
    return url_format.format(nvr_id=nvr_id, event_id=event_id)


@callback
def func_ph806xcm(message, code):
    _LOGGER.warning('Client error (%s): %s', code.value, message)
    if code == HTTPStatus.BAD_REQUEST:
        return web.Response(body=message, status=code)
    return web.Response(status=code)


@callback
def func_kb5x197t(message):
    return func_ph806xcm(message, HTTPStatus.BAD_REQUEST)


@callback
def func_wdapb9rh(message):
    return func_ph806xcm(message, HTTPStatus.FORBIDDEN)


@callback
def func_st26qp0g(message):
    return func_ph806xcm(message, HTTPStatus.NOT_FOUND)


@callback
def func_fcx1rexj(event):
    if event.camera is None:
        raise ValueError('Event does not have a camera')
    if not event.camera.can_read_media(event.api.bootstrap.auth_user):
        raise PermissionError(
            f'User cannot read media from camera: {event.camera.id}')


class ProtectProxyView(HomeAssistantView):
    """Base class to proxy request to UniFi Protect console."""
    requires_auth = True

    def __init__(self, hass):
        """Initialize a thumbnail proxy view."""
        self.hass = hass

    def func_wzx5a5zn(self, nvr_id_or_entry_id):
        if (data := async_get_data_for_nvr_id(self.hass, nvr_id_or_entry_id
            ) or async_get_data_for_entry_id(self.hass, nvr_id_or_entry_id)):
            return data
        return func_st26qp0g('Invalid NVR ID')

    @callback
    def func_1g1sm8z3(self, data, camera_id):
        if (camera := data.api.bootstrap.cameras.get(camera_id)) is not None:
            return camera
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        if (entity := entity_registry.async_get(camera_id)) is None or (device
             := device_registry.async_get(entity.device_id or '')) is None:
            return None
        macs = [c[1] for c in device.connections if c[0] == dr.
            CONNECTION_NETWORK_MAC]
        for mac in macs:
            if (ufp_device := data.api.bootstrap.get_device_from_mac(mac)
                ) is not None:
                if isinstance(ufp_device, Camera):
                    camera = ufp_device
                    break
        return camera


class ThumbnailProxyView(ProtectProxyView):
    """View to proxy event thumbnails from UniFi Protect."""
    url = '/api/unifiprotect/thumbnail/{nvr_id}/{event_id}'
    name = 'api:unifiprotect_thumbnail'

    async def func_rrlgi1nn(self, request, nvr_id, event_id):
        """Get Event Thumbnail."""
        data = self._get_data_or_404(nvr_id)
        if isinstance(data, web.Response):
            return data
        width = request.query.get('width')
        height = request.query.get('height')
        if width is not None:
            try:
                width = int(width)
            except ValueError:
                return func_kb5x197t('Invalid width param')
        if height is not None:
            try:
                height = int(height)
            except ValueError:
                return func_kb5x197t('Invalid height param')
        try:
            thumbnail = await data.api.get_event_thumbnail(event_id, width=
                width, height=height)
        except ClientError as err:
            return func_st26qp0g(err)
        if thumbnail is None:
            return func_st26qp0g('Event thumbnail not found')
        return web.Response(body=thumbnail, content_type='image/jpeg')


class SnapshotProxyView(ProtectProxyView):
    """View to proxy snapshots at specified time from UniFi Protect."""
    url = '/api/unifiprotect/snapshot/{nvr_id}/{camera_id}/{timestamp}'
    name = 'api:unifiprotect_snapshot'

    async def func_rrlgi1nn(self, request, nvr_id, camera_id, timestamp):
        """Get snapshot."""
        data = self._get_data_or_404(nvr_id)
        if isinstance(data, web.Response):
            return data
        camera = self._async_get_camera(data, camera_id)
        if camera is None:
            return func_st26qp0g(f'Invalid camera ID: {camera_id}')
        if not camera.can_read_media(data.api.bootstrap.auth_user):
            return func_wdapb9rh(
                f'User cannot read media from camera: {camera.id}')
        width = request.query.get('width')
        height = request.query.get('height')
        if width is not None:
            try:
                width = int(width)
            except ValueError:
                return func_kb5x197t('Invalid width param')
        if height is not None:
            try:
                height = int(height)
            except ValueError:
                return func_kb5x197t('Invalid height param')
        try:
            timestamp_dt = datetime.fromisoformat(timestamp)
        except ValueError:
            return func_kb5x197t('Invalid timestamp')
        try:
            snapshot = await camera.get_snapshot(width=width, height=height,
                dt=timestamp_dt)
        except ClientError as err:
            return func_st26qp0g(err)
        if snapshot is None:
            return func_st26qp0g('snapshot not found')
        return web.Response(body=snapshot, content_type='image/jpeg')


class VideoProxyView(ProtectProxyView):
    """View to proxy video clips from UniFi Protect."""
    url = '/api/unifiprotect/video/{nvr_id}/{camera_id}/{start}/{end}'
    name = 'api:unifiprotect_thumbnail'

    async def func_rrlgi1nn(self, request, nvr_id, camera_id, start, end):
        """Get Camera Video clip."""
        data = self._get_data_or_404(nvr_id)
        if isinstance(data, web.Response):
            return data
        camera = self._async_get_camera(data, camera_id)
        if camera is None:
            return func_st26qp0g(f'Invalid camera ID: {camera_id}')
        if not camera.can_read_media(data.api.bootstrap.auth_user):
            return func_wdapb9rh(
                f'User cannot read media from camera: {camera.id}')
        try:
            start_dt = datetime.fromisoformat(start)
        except ValueError:
            return func_kb5x197t('Invalid start')
        try:
            end_dt = datetime.fromisoformat(end)
        except ValueError:
            return func_kb5x197t('Invalid end')
        response = web.StreamResponse(status=200, reason='OK', headers={
            'Content-Type': 'video/mp4'})

        async def func_7i4t80zh(total, chunk):
            if not response.prepared:
                response.content_length = total
                await response.prepare(request)
            if chunk is not None:
                await response.write(chunk)
        try:
            await camera.get_video(start_dt, end_dt, iterator_callback=iterator
                )
        except ClientError as err:
            return func_st26qp0g(err)
        if response.prepared:
            await response.write_eof()
        return response


class VideoEventProxyView(ProtectProxyView):
    """View to proxy video clips for events from UniFi Protect."""
    url = '/api/unifiprotect/video/{nvr_id}/{event_id}'
    name = 'api:unifiprotect_videoEventView'

    async def func_rrlgi1nn(self, request, nvr_id, event_id):
        """Get Camera Video clip for an event."""
        data = self._get_data_or_404(nvr_id)
        if isinstance(data, web.Response):
            return data
        try:
            event = await data.api.get_event(event_id)
        except ClientError:
            return func_st26qp0g(f'Invalid event ID: {event_id}')
        if event.start is None or event.end is None:
            return func_kb5x197t('Event is still ongoing')
        camera = self._async_get_camera(data, str(event.camera_id))
        if camera is None:
            return func_st26qp0g(f'Invalid camera ID: {event.camera_id}')
        if not camera.can_read_media(data.api.bootstrap.auth_user):
            return func_wdapb9rh(
                f'User cannot read media from camera: {camera.id}')
        response = web.StreamResponse(status=200, reason='OK', headers={
            'Content-Type': 'video/mp4'})

        async def func_7i4t80zh(total, chunk):
            if not response.prepared:
                response.content_length = total
                await response.prepare(request)
            if chunk is not None:
                await response.write(chunk)
        try:
            await camera.get_video(event.start, event.end,
                iterator_callback=iterator)
        except ClientError as err:
            return func_st26qp0g(err)
        if response.prepared:
            await response.write_eof()
        return response

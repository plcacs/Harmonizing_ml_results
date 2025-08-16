from __future__ import annotations
import asyncio
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, NoReturn, cast
from yarl import URL
from homeassistant.components.camera import CameraImageView
from homeassistant.components.media_player import BrowseError, MediaClass
from homeassistant.components.media_source import BrowseMediaSource, MediaSource, MediaSourceItem, PlayMedia
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .data import ProtectData, async_get_ufp_entries
from .views import async_generate_event_video_url, async_generate_thumbnail_url
from .data import Camera, Event, EventType, SmartDetectObjectType
from .exceptions import NvrError

VIDEO_FORMAT: str = 'video/mp4'
THUMBNAIL_WIDTH: int = 185
THUMBNAIL_HEIGHT: int = 185

class SimpleEventType(str, Enum):
    """Enum to Camera Video events."""
    ALL: str = 'all'
    RING: str = 'ring'
    MOTION: str = 'motion'
    SMART: str = 'smart'
    AUDIO: str = 'audio'

class IdentifierType(str, Enum):
    """UniFi Protect identifier type."""
    EVENT: str = 'event'
    EVENT_THUMB: str = 'eventthumb'
    BROWSE: str = 'browse'

class IdentifierTimeType(str, Enum):
    """UniFi Protect identifier subtype."""
    RECENT: str = 'recent'
    RANGE: str = 'range'

EVENT_MAP: dict[SimpleEventType, set[EventType]] = {SimpleEventType.ALL: {EventType.RING, EventType.MOTION, EventType.SMART_DETECT, EventType.SMART_DETECT_LINE, EventType.SMART_AUDIO_DETECT}, SimpleEventType.RING: {EventType.RING}, SimpleEventType.MOTION: {EventType.MOTION}, SimpleEventType.SMART: {EventType.SMART_DETECT, EventType.SMART_DETECT_LINE}, SimpleEventType.AUDIO: {EventType.SMART_AUDIO_DETECT}}
EVENT_NAME_MAP: dict[SimpleEventType, str] = {SimpleEventType.ALL: 'All Events', SimpleEventType.RING: 'Ring Events', SimpleEventType.MOTION: 'Motion Events', SimpleEventType.SMART: 'Object Detections', SimpleEventType.AUDIO: 'Audio Detections'}

async def async_get_media_source(hass: HomeAssistant) -> ProtectMediaSource:
    """Set up UniFi Protect media source."""
    return ProtectMediaSource(hass, {entry.runtime_data.api.bootstrap.nvr.id: entry.runtime_data for entry in async_get_ufp_entries(hass)})

@callback
def _get_month_start_end(start: datetime) -> tuple[datetime, datetime]:
    start = dt_util.as_local(start)
    end = dt_util.now()
    start = start.replace(day=1, hour=0, minute=0, second=1, microsecond=0)
    end = end.replace(day=1, hour=0, minute=0, second=2, microsecond=0)
    return (start, end)

@callback
def _bad_identifier(identifier: str, err: Exception = None) -> NoReturn:
    msg = f'Unexpected identifier: {identifier}'
    if err is None:
        raise BrowseError(msg)
    raise BrowseError(msg) from err

@callback
def _format_duration(duration: timedelta) -> str:
    formatted: str = ''
    seconds: int = int(duration.total_seconds())
    if seconds > 3600:
        hours: int = seconds // 3600
        formatted += f'{hours}h '
        seconds -= hours * 3600
    if seconds > 60:
        minutes: int = seconds // 60
        formatted += f'{minutes}m '
        seconds -= minutes * 60
    if seconds > 0:
        formatted += f'{seconds}s '
    return formatted.strip()

@callback
def _get_object_name(event: Any) -> str:
    if isinstance(event, Event):
        event = event.unifi_dict()
    names: list[str] = []
    types: set[str] = set(event['smartDetectTypes'])
    metadata: dict = event.get('metadata') or {}
    for thumb in metadata.get('detectedThumbnails', []):
        thumb_type: str = thumb.get('type')
        if thumb_type not in types:
            continue
        types.remove(thumb_type)
        if thumb_type == SmartDetectObjectType.VEHICLE.value:
            attributes: dict = thumb.get('attributes') or {}
            color: str = attributes.get('color', {}).get('val', '')
            vehicle_type: str = attributes.get('vehicleType', {}).get('val', 'vehicle')
            license_plate: str = metadata.get('licensePlate', {}).get('name')
            name: str = f'{color} {vehicle_type}'.strip().title()
            if license_plate:
                types.remove(SmartDetectObjectType.LICENSE_PLATE.value)
                name = f'{name}: {license_plate}'
            names.append(name)
        else:
            smart_type: SmartDetectObjectType = SmartDetectObjectType(thumb_type)
            names.append(smart_type.name.title().replace('_', ' '))
    for raw in types:
        smart_type: SmartDetectObjectType = SmartDetectObjectType(raw)
        names.append(smart_type.name.title().replace('_', ' '))
    return ', '.join(sorted(names))

@callback
def _get_audio_name(event: Any) -> str:
    if isinstance(event, Event):
        event = event.unifi_dict()
    smart_types: list[SmartDetectObjectType] = [SmartDetectObjectType(e) for e in event['smartDetectTypes']]
    return ', '.join([s.name.title().replace('_', ' ') for s in smart_types])

class ProtectMediaSource(MediaSource):
    """Represents all UniFi Protect NVRs."""
    name: str = 'UniFi Protect'

    def __init__(self, hass: HomeAssistant, data_sources: dict[str, ProtectData]):
        """Initialize the UniFi Protect media source."""
        super().__init__(DOMAIN)
        self.hass: HomeAssistant = hass
        self.data_sources: dict[str, ProtectData] = data_sources
        self._registry: er.EntityRegistry | None = None

    async def async_resolve_media(self, item: MediaSourceItem) -> PlayMedia:
        """Return a streamable URL and associated mime type for a UniFi Protect event."""
        parts: list[str] = item.identifier.split(':')
        if len(parts) != 3 or parts[1] not in ('event', 'eventthumb'):
            _bad_identifier(item.identifier)
        thumbnail_only: bool = parts[1] == 'eventthumb'
        try:
            data: ProtectData = self.data_sources[parts[0]]
        except (KeyError, IndexError) as err:
            _bad_identifier(item.identifier, err)
        event: Event | None = data.api.bootstrap.events.get(parts[2])
        if event is None:
            try:
                event = await data.api.get_event(parts[2])
            except NvrError as err:
                _bad_identifier(item.identifier, err)
            else:
                data.api.bootstrap.events[event.id] = event
        nvr: Any = data.api.bootstrap.nvr
        if thumbnail_only:
            return PlayMedia(async_generate_thumbnail_url(event.id, nvr.id), 'image/jpeg')
        return PlayMedia(async_generate_event_video_url(event), 'video/mp4')

    async def async_browse_media(self, item: MediaSourceItem) -> BrowseMediaSource:
        """Return a browsable UniFi Protect media source."""
        if not item.identifier:
            return await self._build_sources()
        parts: list[str] = item.identifier.split(':')
        try:
            data: ProtectData = self.data_sources[parts[0]]
        except (KeyError, IndexError) as err:
            _bad_identifier(item.identifier, err)
        if len(parts) < 2:
            _bad_identifier(item.identifier)
        try:
            identifier_type: IdentifierType = IdentifierType(parts[1])
        except ValueError as err:
            _bad_identifier(item.identifier, err)
        if identifier_type in (IdentifierType.EVENT, IdentifierType.EVENT_THUMB):
            thumbnail_only: bool = identifier_type == IdentifierType.EVENT_THUMB
            return await self._resolve_event(data, parts[2], thumbnail_only)
        parts = parts[2:]
        if len(parts) == 0:
            return await self._build_console(data)
        camera_id: str = parts.pop(0)
        if len(parts) == 0:
            return await self._build_camera(data, camera_id, build_children=True)
        try:
            event_type: SimpleEventType = SimpleEventType(parts.pop(0).lower())
        except (IndexError, ValueError) as err:
            _bad_identifier(item.identifier, err)
        if len(parts) == 0:
            return await self._build_events_type(data, camera_id, event_type, build_children=True)
        try:
            time_type: IdentifierTimeType = IdentifierTimeType(parts.pop(0))
        except ValueError as err:
            _bad_identifier(item.identifier, err)
        if len(parts) == 0:
            _bad_identifier(item.identifier)
        if time_type == IdentifierTimeType.RECENT:
            try:
                days: int = int(parts.pop(0))
            except (IndexError, ValueError) as err:
                _bad_identifier(item.identifier, err)
            return await self._build_recent(data, camera_id, event_type, days, build_children=True)
        try:
            start, is_month, is_all = self._parse_range(parts)
        except (IndexError, ValueError) as err:
            _bad_identifier(item.identifier, err)
        if is_month:
            return await self._build_month(data, camera_id, event_type, start, build_children=True)
        return await self._build_days(data, camera_id, event_type, start, build_children=True, is_all=is_all)

    def _parse_range(self, parts: list[str]) -> tuple[date, bool, bool]:
        day: int = 1
        is_month: bool = True
        is_all: bool = True
        year: int = int(parts[0])
        month: int = int(parts[1])
        if len(parts) == 3:
            is_month = False
            if parts[2] != 'all':
                is_all = False
                day = int(parts[2])
        start: date = date(year=year, month=month, day=day)
        return (start, is_month, is_all)

    async def _resolve_event(self, data: ProtectData, event_id: str, thumbnail_only: bool = False) -> BrowseMediaSource:
        """Resolve a specific event."""
        subtype: str = 'eventthumb' if thumbnail_only else 'event'
        try:
            event: Event = await data.api.get_event(event_id)
        except NvrError as err:
            _bad_identifier(f'{data.api.bootstrap.nvr.id}:{subtype}:{event_id}', err)
        if event.start is None or event.end is None:
            raise BrowseError('Event is still ongoing')
        return await self._build_event(data, event, thumbnail_only)

    @callback
    def async_get_registry(self) -> er.EntityRegistry:
        """Get or return Entity Registry."""
        if self._registry is None:
            self._registry = er.async_get(self.hass)
        return self._registry

    def _breadcrumb(self, data: ProtectData, base_title: str, camera: Any = None, event_type: SimpleEventType = None, count: int = None) -> str:
        title: str = base_title
        if count is not None:
            if count == data.max_events:
                title = f'{title} ({count} TRUNCATED)'
            else:
                title = f'{title} ({count})'
        if event_type is not None:
            title = f'{EVENT_NAME_MAP[event_type].title()} > {title}'
        if camera is not None:
            title = f'{camera.display_name} > {title}'
        return f'{data.api.bootstrap.nvr.display_name} > {title}'

    async def _build_event(self, data: ProtectData, event: Any, thumbnail_only: bool = False) -> BrowseMediaSource:
        """Build media source for an individual event."""
        if isinstance(event, Event):
            event_id: str = event.id
            event_type: EventType = event.type
            start: datetime = event.start
            end: datetime = event.end
        else:
            event_id: str = event['id']
            event_type: EventType = EventType(event['type'])
            start: datetime = from_js_time(event['start'])
            end: datetime = from_js_time(event['end'])
        assert end is not None
        title: str = dt_util.as_local(start).strftime('%x %X')
        duration: timedelta = end - start
        title += f' {_format_duration(duration)}'
        if event_type in EVENT_MAP[SimpleEventType.RING]:
            event_text: str = 'Ring Event'
        elif event_type in EVENT_MAP[SimpleEventType.MOTION]:
            event_text: str = 'Motion Event'
        elif event_type in EVENT_MAP[SimpleEventType.SMART]:
            event_text: str = f'Object Detection - {_get_object_name(event)}'
        elif event_type in EVENT_MAP[SimpleEventType.AUDIO]:
            event_text: str = f'Audio Detection - {_get_audio_name(event)}'
        title += f' {event_text}'
        nvr: Any = data.api.bootstrap.nvr
        if thumbnail_only:
            return BrowseMediaSource(domain=DOMAIN, identifier=f'{nvr.id}:eventthumb:{event_id}', media_class=MediaClass.IMAGE, media_content_type='image/jpeg', title=title, can_play=True, can_expand=False, thumbnail=async_generate_thumbnail_url(event_id, nvr.id, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))
        return BrowseMediaSource(domain=DOMAIN, identifier=f'{nvr.id}:event:{event_id}', media_class=MediaClass.VIDEO, media_content_type='video/mp4', title=title, can_play=True, can_expand=False, thumbnail=async_generate_thumbnail_url(event_id, nvr.id, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))

    async def _build_events(self, data: ProtectData, start: datetime, end: datetime, camera_id: str = None, event_types: set[EventType] = None, reserve: bool = False) -> list[BrowseMediaSource]:
        """Build media source for a given range of time and event type."""
        event_types = event_types or EVENT_MAP[SimpleEventType.ALL]
        types: list[EventType] = list(event_types)
        sources: list[BrowseMediaSource] = []
        events: list[dict] = await data.api.get_events_raw(start=start, end=end, types=types, limit=data.max_events)
        events = sorted(events, key=lambda e: cast(int, e['start']), reverse=reserve)
        for event in events:
            if event.get('start') is None or event.get('end') is None:
                continue
            if camera_id is not None and event.get('camera') != camera_id:
                continue
            if event.get('type') == EventType.MOTION.value and event.get('smartDetectEvents'):
                continue
            sources.append(await self._build_event(data, event))
        return sources

    async def _build_recent(self, data: ProtectData, camera_id: str, event_type: SimpleEventType, days: int, build_children: bool = False) -> BrowseMediaSource:
        """Build media source for events in relative days."""
        base_id: str = f'{data.api.bootstrap.nvr.id}:browse:{camera_id}:{event_type.value}'
        title: str = f'Last {days} Days'
        if days == 1:
            title = 'Last 24 Hours'
        source: BrowseMediaSource = BrowseMediaSource(domain=DOMAIN, identifier=f'{base_id}:recent:{days}', media_class=MediaClass.DIRECTORY, media_content_type='video/mp4', title=title, can_play=False, can_expand=True, children_media_class=MediaClass.VIDEO)
        if not build_children:
            return source
        now: datetime = dt_util.now()
        camera: Any = None
        event_camera_id: str = None
        if camera_id != 'all':
            camera = data.api.bootstrap.cameras.get(camera_id)
            event_camera_id = camera_id
        events: list[BrowseMediaSource] = await self._build_events(data=data, start=now - timedelta(days=days), end=now, camera_id=event_camera_id, event_types=EVENT_MAP[event_type], reserve=True)
        source.children = events
        source.title = self._breadcrumb(data, title, camera=camera, event_type=event_type, count=len(events))
        return source

    async def _build_month(self, data: ProtectData, camera_id: str, event_type: SimpleEventType, start: date, build_children: bool = False) -> BrowseMediaSource:
        """Build media source for selectors for a given month."""
        base_id: str = f'{data.api.bootstrap.nvr.id}:browse:{camera_id}:{event_type.value}'
        title: str = f'{start.strftime('%B %Y')}'
        source: BrowseMediaSource = BrowseMediaSource(domain=DOMAIN, identifier=f'{base_id}:range:{start.year}:{start.month}', media_class=MediaClass.DIRECTORY, media_content_type=MediaClass.VIDEO, title=title, can_play=False, can_expand=True, children_media_class=MediaClass.VIDEO)
        if not build_children:
            return source
        if data.api.bootstrap.recording_start is not None:
            recording_start: date = data.api.bootstrap.recording_start.date()
        start: date = max(recording_start, start)
        recording_end: date = dt_util.now().date()
        end: date = start.replace(month=start.month + 1) - timedelta(days=1)
        end: date = min(recording_end, end)
        children: list[BrowseMediaSource] = [self._build_days(data, camera_id, event_type, start, is_all=True)]
        while start <= end:
            children.append(self._build_days(data, camera_id, event_type, start, is_all=False))
            start = start + timedelta(hours=24)
        camera: Any = None
        if camera_id != 'all':
            camera = data.api.bootstrap.cameras.get(camera_id)
        source.children = await asyncio.gather(*children)
        source.title = self._breadcrumb(data, title, camera=camera, event_type=event_type)
        return source

    async def _build_days(self, data: ProtectData, camera_id: str, event_type: SimpleEventType, start: date, is_all: bool = True, build_children: bool = False) -> BrowseMediaSource:
        """Build media source for events for a given day or whole month."""
        base_id: str = f'{data.api.bootstrap.nvr.id}:browse:{camera_id}:{event_type.value}'
        if is_all:
            title: str = 'Whole Month'
            identifier: str = f'{base_id}:range:{start.year}:{start.month}:all'
        else:
            title: str = f'{start.strftime('%x')}'
            identifier: str = f'{base_id}:range:{start.year}:{start.month}:{start.day}'
        source: BrowseMedia
"""UniFi Protect media sources."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, NoReturn, cast, Optional, Union, Dict, List, Set, Tuple

from uiprotect.data import Camera, Event, EventType, SmartDetectObjectType
from uiprotect.exceptions import NvrError
from uiprotect.utils import from_js_time
from yarl import URL

from homeassistant.components.camera import CameraImageView
from homeassistant.components.media_player import BrowseError, MediaClass
from homeassistant.components.media_source import (
    BrowseMediaSource,
    MediaSource,
    MediaSourceItem,
    PlayMedia,
)
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .data import ProtectData, async_get_ufp_entries
from .views import async_generate_event_video_url, async_generate_thumbnail_url

VIDEO_FORMAT: str = "video/mp4"
THUMBNAIL_WIDTH: int = 185
THUMBNAIL_HEIGHT: int = 185


class SimpleEventType(str, Enum):
    """Enum to Camera Video events."""

    ALL = "all"
    RING = "ring"
    MOTION = "motion"
    SMART = "smart"
    AUDIO = "audio"


class IdentifierType(str, Enum):
    """UniFi Protect identifier type."""

    EVENT = "event"
    EVENT_THUMB = "eventthumb"
    BROWSE = "browse"


class IdentifierTimeType(str, Enum):
    """UniFi Protect identifier subtype."""

    RECENT = "recent"
    RANGE = "range"


EVENT_MAP: Dict[SimpleEventType, Set[EventType]] = {
    SimpleEventType.ALL: {
        EventType.RING,
        EventType.MOTION,
        EventType.SMART_DETECT,
        EventType.SMART_DETECT_LINE,
        EventType.SMART_AUDIO_DETECT,
    },
    SimpleEventType.RING: {EventType.RING},
    SimpleEventType.MOTION: {EventType.MOTION},
    SimpleEventType.SMART: {EventType.SMART_DETECT, EventType.SMART_DETECT_LINE},
    SimpleEventType.AUDIO: {EventType.SMART_AUDIO_DETECT},
}
EVENT_NAME_MAP: Dict[SimpleEventType, str] = {
    SimpleEventType.ALL: "All Events",
    SimpleEventType.RING: "Ring Events",
    SimpleEventType.MOTION: "Motion Events",
    SimpleEventType.SMART: "Object Detections",
    SimpleEventType.AUDIO: "Audio Detections",
}


async def async_get_media_source(hass: HomeAssistant) -> MediaSource:
    """Set up UniFi Protect media source."""
    return ProtectMediaSource(
        hass,
        {
            entry.runtime_data.api.bootstrap.nvr.id: entry.runtime_data
            for entry in async_get_ufp_entries(hass)
        },
    )


@callback
def _get_month_start_end(start: datetime) -> Tuple[datetime, datetime]:
    start = dt_util.as_local(start)
    end = dt_util.now()

    start = start.replace(day=1, hour=0, minute=0, second=1, microsecond=0)
    end = end.replace(day=1, hour=0, minute=0, second=2, microsecond=0)

    return start, end


@callback
def _bad_identifier(identifier: str, err: Exception | None = None) -> NoReturn:
    msg = f"Unexpected identifier: {identifier}"
    if err is None:
        raise BrowseError(msg)
    raise BrowseError(msg) from err


@callback
def _format_duration(duration: timedelta) -> str:
    formatted = ""
    seconds = int(duration.total_seconds())
    if seconds > 3600:
        hours = seconds // 3600
        formatted += f"{hours}h "
        seconds -= hours * 3600
    if seconds > 60:
        minutes = seconds // 60
        formatted += f"{minutes}m "
        seconds -= minutes * 60
    if seconds > 0:
        formatted += f"{seconds}s "

    return formatted.strip()


@callback
def _get_object_name(event: Union[Event, Dict[str, Any]]) -> str:
    if isinstance(event, Event):
        event = event.unifi_dict()

    names: List[str] = []
    types = set(event["smartDetectTypes"])
    metadata = event.get("metadata") or {}
    for thumb in metadata.get("detectedThumbnails", []):
        thumb_type = thumb.get("type")
        if thumb_type not in types:
            continue

        types.remove(thumb_type)
        if thumb_type == SmartDetectObjectType.VEHICLE.value:
            attributes = thumb.get("attributes") or {}
            color = attributes.get("color", {}).get("val", "")
            vehicle_type = attributes.get("vehicleType", {}).get("val", "vehicle")
            license_plate = metadata.get("licensePlate", {}).get("name")

            name = f"{color} {vehicle_type}".strip().title()
            if license_plate:
                types.remove(SmartDetectObjectType.LICENSE_PLATE.value)
                name = f"{name}: {license_plate}"
            names.append(name)
        else:
            smart_type = SmartDetectObjectType(thumb_type)
            names.append(smart_type.name.title().replace("_", " "))

    for raw in types:
        smart_type = SmartDetectObjectType(raw)
        names.append(smart_type.name.title().replace("_", " "))

    return ", ".join(sorted(names))


@callback
def _get_audio_name(event: Union[Event, Dict[str, Any]]) -> str:
    if isinstance(event, Event):
        event = event.unifi_dict()

    smart_types = [SmartDetectObjectType(e) for e in event["smartDetectTypes"]]
    return ", ".join([s.name.title().replace("_", " ") for s in smart_types])


class ProtectMediaSource(MediaSource):
    """Represents all UniFi Protect NVRs."""

    name: str = "UniFi Protect"
    _registry: Optional[er.EntityRegistry] = None

    def __init__(
        self, hass: HomeAssistant, data_sources: Dict[str, ProtectData]
    ) -> None:
        """Initialize the UniFi Protect media source."""

        super().__init__(DOMAIN)
        self.hass = hass
        self.data_sources = data_sources
        self._registry = None

    async def async_resolve_media(self, item: MediaSourceItem) -> PlayMedia:
        """Return a streamable URL and associated mime type for a UniFi Protect event."""

        parts = item.identifier.split(":")
        if len(parts) != 3 or parts[1] not in ("event", "eventthumb"):
            _bad_identifier(item.identifier)

        thumbnail_only = parts[1] == "eventthumb"
        try:
            data = self.data_sources[parts[0]]
        except (KeyError, IndexError) as err:
            _bad_identifier(item.identifier, err)

        event = data.api.bootstrap.events.get(parts[2])
        if event is None:
            try:
                event = await data.api.get_event(parts[2])
            except NvrError as err:
                _bad_identifier(item.identifier, err)
            else:
                # cache the event for later
                data.api.bootstrap.events[event.id] = event

        nvr = data.api.bootstrap.nvr
        if thumbnail_only:
            return PlayMedia(
                async_generate_thumbnail_url(event.id, nvr.id), "image/jpeg"
            )
        return PlayMedia(async_generate_event_video_url(event), "video/mp4")

    async def async_browse_media(self, item: MediaSourceItem) -> BrowseMediaSource:
        """Return a browsable UniFi Protect media source."""

        if not item.identifier:
            return await self._build_sources()

        parts = item.identifier.split(":")

        try:
            data = self.data_sources[parts[0]]
        except (KeyError, IndexError) as err:
            _bad_identifier(item.identifier, err)

        if len(parts) < 2:
            _bad_identifier(item.identifier)

        try:
            identifier_type = IdentifierType(parts[1])
        except ValueError as err:
            _bad_identifier(item.identifier, err)

        if identifier_type in (IdentifierType.EVENT, IdentifierType.EVENT_THUMB):
            thumbnail_only = identifier_type == IdentifierType.EVENT_THUMB
            return await self._resolve_event(data, parts[2], thumbnail_only)

        # rest are params for browse
        parts = parts[2:]

        # {nvr_id}:browse
        if len(parts) == 0:
            return await self._build_console(data)

        # {nvr_id}:browse:all|{camera_id}
        camera_id = parts.pop(0)
        if len(parts) == 0:
            return await self._build_camera(data, camera_id, build_children=True)

        # {nvr_id}:browse:all|{camera_id}:all|{event_type}
        try:
            event_type = SimpleEventType(parts.pop(0).lower())
        except (IndexError, ValueError) as err:
            _bad_identifier(item.identifier, err)

        if len(parts) == 0:
            return await self._build_events_type(
                data, camera_id, event_type, build_children=True
            )

        try:
            time_type = IdentifierTimeType(parts.pop(0))
        except ValueError as err:
            _bad_identifier(item.identifier, err)

        if len(parts) == 0:
            _bad_identifier(item.identifier)

        # {nvr_id}:browse:all|{camera_id}:all|{event_type}:recent:{day_count}
        if time_type == IdentifierTimeType.RECENT:
            try:
                days = int(parts.pop(0))
            except (IndexError, ValueError) as err:
                _bad_identifier(item.identifier, err)

            return await self._build_recent(
                data, camera_id, event_type, days, build_children=True
            )

        # {nvr_id}:all|{camera_id}:all|{event_type}:range:{year}:{month}
        # {nvr_id}:all|{camera_id}:all|{event_type}:range:{year}:{month}:all|{day}
        try:
            start, is_month, is_all = self._parse_range(parts)
        except (IndexError, ValueError) as err:
            _bad_identifier(item.identifier, err)

        if is_month:
            return await self._build_month(
                data, camera_id, event_type, start, build_children=True
            )
        return await self._build_days(
            data, camera_id, event_type, start, build_children=True, is_all=is_all
        )

    def _parse_range(self, parts: List[str]) -> Tuple[date, bool, bool]:
        day = 1
        is_month = True
        is_all = True
        year = int(parts[0])
        month = int(parts[1])
        if len(parts) == 3:
            is_month = False
            if parts[2] != "all":
                is_all = False
                day = int(parts[2])

        start = date(year=year, month=month, day=day)
        return start, is_month, is_all

    async def _resolve_event(
        self, data: ProtectData, event_id: str, thumbnail_only: bool = False
    ) -> BrowseMediaSource:
        """Resolve a specific event."""

        subtype = "eventthumb" if thumbnail_only else "event"
        try:
            event = await data.api.get_event(event_id)
        except NvrError as err:
            _bad_identifier(f"{data.api.bootstrap.nvr.id}:{subtype}:{event_id}", err)

        if event.start is None or event.end is None:
            raise BrowseError("Event is still ongoing")

        return await self._build_event(data, event, thumbnail_only)

    @callback
    def async_get_registry(self) -> er.EntityRegistry:
        """Get or return Entity Registry."""
        if self._registry is None:
            self._registry = er.async_get(self.hass)
        return self._registry

    def _breadcrumb(
        self,
        data: ProtectData,
        base_title: str,
        camera: Optional[Camera] = None,
        event_type: Optional[SimpleEventType] = None,
        count: Optional[int] = None,
    ) -> str:
        title = base_title
        if count is not None:
            if count == data.max_events:
                title = f"{title} ({count} TRUNCATED)"
            else:
                title = f"{title} ({count})"

        if event_type is not None:
            title = f"{EVENT_NAME_MAP[event_type].title()} > {title}"

        if camera is not None:
            title = f"{camera.display_name} > {title}"
        return f"{data.api.bootstrap.nvr.display_name} > {title}"

    async def _build_event(
        self,
        data: ProtectData,
        event: Union[Dict[str, Any], Event],
        thumbnail_only: bool = False,
    ) -> BrowseMediaSource:
        """Build media source for an individual event."""

        if isinstance(event, Event):
            event_id = event.id
            event_type = event.type
            start = event.start
            end = event.end
        else:
            event_id = event["id"]
            event_type = EventType(event["type"])
            start = from_js_time(event["start"])
            end = from_js_time(event["end"])

        assert end is not None

        title = dt_util.as_local(start).strftime("%x %X")
        duration = end - start
        title += f" {_format_duration(duration)}"
        if event_type in EVENT_MAP[SimpleEventType.RING]:
            event_text = "Ring Event"
        elif event_type in EVENT_MAP[SimpleEventType.MOTION]:
            event_text = "Motion Event"
        elif event_type in EVENT_MAP[SimpleEventType.SMART]:
            event_text = f"Object Detection - {_get_object_name(event)}"
        elif event_type in EVENT_MAP[SimpleEventType.AUDIO]:
            event_text = f"Audio Detection - {_get_audio_name(event)}"
        title += f" {event_text}"

        nvr = data.api.bootstrap.nvr
        if thumbnail_only:
            return BrowseMediaSource(
                domain=DOMAIN,
                identifier=f"{nvr.id}:eventthumb:{event_id}",
                media_class=MediaClass.IMAGE,
                media_content_type="image/jpeg",
                title=title,
                can_play=True,
                can_expand=False,
                thumbnail=async_generate_thumbnail_url(
                    event_id, nvr.id, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT
                ),
            )

        return BrowseMediaSource(
            domain=DOMAIN,
            identifier=f"{nvr.id}:event:{event_id}",
            media_class=MediaClass.VIDEO,
            media_content_type="video/mp4",
            title=title,
            can_play=True,
            can_expand=False,
            thumbnail=async_generate_thumbnail_url(
                event_id, nvr.id, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT
            ),
        )

    async def _build_events(
        self,
        data: ProtectData,
        start: datetime,
        end: datetime,
        camera_id: Optional[str] = None,
        event_types: Optional[Set[EventType]] = None,
        reserve: bool = False,
    ) -> List[BrowseMediaSource]:
        """Build media source for a given range of time and event type."""

        event_types = event_types or EVENT_MAP[SimpleEventType.ALL]
        types = list(event_types)
        sources: List[BrowseMediaSource] = []
        events = await data.api.get_events_raw(
            start=start, end=end, types=types, limit=data.max_events
        )
        events = sorted(events, key=lambda e: cast(int, e["start"]), reverse=reserve)
        for event in events:
            # do not process ongoing events
            if event.get("start") is None or event.get("end") is None:
                continue

            if camera_id is not None and event.get("camera") != camera_id:
                continue

            # smart detect events have a paired motion event
            if event.get("type") == EventType.MOTION.value and event.get(
                "smartDetectEvents"
            ):
                continue

            sources.append(await self._build_event(data, event))

        return sources

    async def _build_recent(
        self,
        data: ProtectData,
        camera_id: str,
        event_type: SimpleEventType,
        days: int,
        build_children: bool = False,
    ) -> BrowseMediaSource:
        """Build media source for events in relative days."""

        base_id = f"{data.api.bootstrap.nvr.id}:browse:{camera_id}:{event_type.value}"
        title = f"Last {days} Days"
        if days == 1:
            title = "Last 24 Hours"

        source = BrowseMediaSource(
            domain=DOMAIN,
            identifier=f"{base_id}:recent:{days}",
            media_class=MediaClass.DIRECTORY,
            media_content_type="video/mp4",
            title=title,
            can_play=False,
            can_expand=True,
            children_media_class=MediaClass.VIDEO,
        )

        if not build_children:
            return source

        now = dt_util.now()
        camera: Optional[Camera] = None
        event_camera_id: Optional[str] = None
        if camera_id != "all":
            camera = data.api.bootstrap.cameras.get(camera_id)
            event_camera_id = camera_id

        events = await self._build_events(
            data=data,
            start=now - timedelta(days=days),
            end=now,
            camera_id=event
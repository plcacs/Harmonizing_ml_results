"""Support for DoorBird devices."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus
import logging
from typing import Any, DefaultDict, Dict, List, Optional
from aiohttp import ClientResponseError
from doorbirdpy import (
    DoorBird,
    DoorBirdScheduleEntry,
    DoorBirdScheduleEntryOutput,
    DoorBirdScheduleEntrySchedule,
)
from propcache.api import cached_property
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant
from homeassistant.helpers.network import get_url
from homeassistant.util import dt as dt_util, slugify
from .const import API_URL, DEFAULT_EVENT_TYPES, HTTP_EVENT_TYPE, MAX_WEEKDAY, MIN_WEEKDAY

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DoorbirdEvent:
    """Describes a doorbird event."""
    event: str
    input: int


@dataclass(slots=True)
class DoorbirdEventConfig:
    """Describes the configuration of doorbird events."""
    events: List[DoorbirdEvent]
    schedule: List[DoorBirdScheduleEntry]
    unconfigured_favorites: DefaultDict[int, List[str]]


class ConfiguredDoorBird:
    """Attach additional information to pass along with configured device."""

    def __init__(
        self,
        hass: HomeAssistant,
        device: DoorBird,
        name: str,
        custom_url: Optional[str],
        token: str,
        event_entity_ids: Dict[str, str],
    ) -> None:
        """Initialize configured device."""
        self._hass: HomeAssistant = hass
        self._name: str = name
        self._device: DoorBird = device
        self._custom_url: Optional[str] = custom_url
        self._token: str = token
        self._event_entity_ids: Dict[str, str] = event_entity_ids
        self.events: List[DoorbirdEvent] = []
        self.door_station_events: List[str] = []
        self.event_descriptions: List[DoorbirdEvent] = []

    def update_events(self, events: List[DoorbirdEvent]) -> None:
        """Update the doorbird events."""
        self.events = events
        self.door_station_events = [self._get_event_name(event) for event in self.events]

    @cached_property
    def name(self) -> str:
        """Get custom device name."""
        return self._name

    @cached_property
    def device(self) -> DoorBird:
        """Get the configured device."""
        return self._device

    @cached_property
    def custom_url(self) -> Optional[str]:
        """Get custom url for device."""
        return self._custom_url

    @cached_property
    def token(self) -> str:
        """Get token for device."""
        return self._token

    async def async_register_events(self) -> None:
        """Register events on device."""
        if not self.door_station_events:
            return
        http_fav: Dict[str, Any] = await self._async_register_events()
        event_config: DoorbirdEventConfig = await self._async_get_event_config(http_fav)
        _LOGGER.debug('%s: Event config: %s', self.name, event_config)
        if event_config.unconfigured_favorites:
            await self._configure_unconfigured_favorites(event_config)
            event_config = await self._async_get_event_config(http_fav)
        self.event_descriptions = event_config.events

    async def _configure_unconfigured_favorites(
        self, event_config: DoorbirdEventConfig
    ) -> None:
        """Configure unconfigured favorites."""
        for entry in event_config.schedule:
            modified_schedule: bool = False
            for identifier in event_config.unconfigured_favorites.get(entry.input, []):
                schedule: DoorBirdScheduleEntrySchedule = DoorBirdScheduleEntrySchedule()
                schedule.add_weekday(MIN_WEEKDAY, MAX_WEEKDAY)
                entry.output.append(
                    DoorBirdScheduleEntryOutput(
                        enabled=True, event=HTTP_EVENT_TYPE, param=identifier, schedule=schedule
                    )
                )
                modified_schedule = True
            if modified_schedule:
                update_ok, code = await self.device.change_schedule(entry)
                if not update_ok:
                    _LOGGER.error(
                        'Unable to update schedule entry %s to %s. Error code: %s',
                        self.name,
                        entry.export,
                        code,
                    )

    async def _async_register_events(self) -> Dict[str, Any]:
        """Register events on device."""
        if (custom_url := self.custom_url):
            hass_url: str = custom_url
        else:
            hass_url = get_url(self._hass, prefer_external=False)
        http_fav: Dict[str, Any] = await self._async_get_http_favorites()
        if any(
            [
                await self._async_register_event(hass_url, event, http_fav)
                for event in self.door_station_events
            ]
        ):
            http_fav = await self._async_get_http_favorites()
        return http_fav

    async def _async_get_event_config(
        self, http_fav: Dict[str, Any]
    ) -> DoorbirdEventConfig:
        """Get events and unconfigured favorites from http favorites."""
        device: DoorBird = self.device
        events: List[DoorbirdEvent] = []
        unconfigured_favorites: DefaultDict[int, List[str]] = defaultdict(list)
        try:
            schedule: List[DoorBirdScheduleEntry] = await device.schedule()
        except ClientResponseError as ex:
            if ex.status == HTTPStatus.NOT_FOUND:
                return DoorbirdEventConfig(events, [], unconfigured_favorites)
            raise
        favorite_input_type: Dict[str, int] = {
            output.param: entry.input
            for entry in schedule
            for output in entry.output
            if output.event == HTTP_EVENT_TYPE
        }
        default_event_types: Dict[str, int] = {
            self._get_event_name(event): event_type for event, event_type in DEFAULT_EVENT_TYPES
        }
        for identifier, data in http_fav.items():
            title: Optional[str] = data.get('title')
            if not title or not title.startswith('Home Assistant'):
                continue
            event: str = title.partition('(')[2].strip(')')
            if (input_type := favorite_input_type.get(identifier)) is not None:
                events.append(DoorbirdEvent(event, input_type))
            elif (input_type := default_event_types.get(event)) is not None:
                unconfigured_favorites[input_type].append(identifier)
        return DoorbirdEventConfig(events, schedule, unconfigured_favorites)

    @cached_property
    def slug(self) -> str:
        """Get device slug."""
        return slugify(self._name)

    def _get_event_name(self, event: DoorbirdEvent) -> str:
        return f'{self.slug}_{event.event}'

    async def _async_get_http_favorites(self) -> Dict[str, Any]:
        """Get the HTTP favorites from the device."""
        favorites: Dict[str, Any] = (await self.device.favorites()).get(HTTP_EVENT_TYPE) or {}
        return favorites

    async def _async_register_event(
        self, hass_url: str, event: str, http_fav: Dict[str, Any]
    ) -> bool:
        """Register an event.

        Returns True if the event was registered, False if
        the event was already registered or registration failed.
        """
        url: str = f'{hass_url}{API_URL}/{event}?token={self._token}'
        _LOGGER.debug('Registering URL %s for event %s', url, event)
        if any((fav['value'] == url for fav in http_fav.values())):
            _LOGGER.debug('URL already registered for %s', event)
            return False
        if not await self.device.change_favorite(
            HTTP_EVENT_TYPE, f'Home Assistant ({event})', url
        ):
            _LOGGER.warning(
                'Unable to set favorite URL "%s". Event "%s" will not fire', url, event
            )
            return False
        _LOGGER.debug('Successfully registered URL for %s on %s', event, self.name)
        return True

    def get_event_data(self, event: str) -> Dict[str, Any]:
        """Get data to pass along with HA event."""
        return {
            'timestamp': dt_util.utcnow().isoformat(),
            'live_video_url': self._device.live_video_url,
            'live_image_url': self._device.live_image_url,
            'rtsp_live_video_url': self._device.rtsp_live_video_url,
            'html5_viewer_url': self._device.html5_viewer_url,
            ATTR_ENTITY_ID: self._event_entity_ids.get(event),
        }


async def async_reset_device_favorites(door_station: ConfiguredDoorBird) -> None:
    """Handle clearing favorites on device."""
    door_bird: DoorBird = door_station.device
    favorites: Dict[str, Any] = await door_bird.favorites()
    for favorite_type, favorite_ids in favorites.items():
        for favorite_id in favorite_ids:
            await door_bird.delete_favorite(favorite_type, favorite_id)
    await door_station.async_register_events()
from __future__ import annotations
from collections.abc import Iterable
from datetime import datetime, timedelta
import itertools
import logging
from gcal_sync.api import GoogleCalendarService, ListEventsRequest
from gcal_sync.exceptions import ApiException
from gcal_sync.model import Event
from gcal_sync.sync import CalendarEventSyncManager
from gcal_sync.timeline import Timeline
from ical.iter import SortableItemValue
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

_LOGGER: logging.Logger = logging.getLogger(__name__)
MIN_TIME_BETWEEN_UPDATES: timedelta = timedelta(minutes=15)
MAX_UPCOMING_EVENTS: int = 20

def _truncate_timeline(timeline: Timeline, max_events: int) -> Timeline:
    ...

class CalendarSyncUpdateCoordinator(DataUpdateCoordinator[Timeline]):
    ...

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry, sync: CalendarEventSyncManager, name: str):
        ...

    async def _async_update_data(self) -> Timeline:
        ...

    async def async_get_events(self, start_date: datetime, end_date: datetime) -> Iterable[Event]:
        ...

    @property
    def upcoming(self) -> Iterable[Event]:
        ...

class CalendarQueryUpdateCoordinator(DataUpdateCoordinator[list[Event]]):
    ...

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry, calendar_service: GoogleCalendarService, name: str, calendar_id: str, search: str):
        ...

    async def async_get_events(self, start_date: datetime, end_date: datetime) -> list[Event]:
        ...

    async def _async_update_data(self) -> list[Event]:
        ...

    @property
    def upcoming(self) -> list[Event]:
        ...

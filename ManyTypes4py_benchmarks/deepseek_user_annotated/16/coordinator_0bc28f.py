"""Support for Google Calendar Search binary sensors."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
import itertools
import logging
from typing import Optional, List, cast

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
    """Truncate the timeline to a maximum number of events."""
    upcoming: Iterable[Event] = timeline.active_after(dt_util.now())
    truncated: List[Event] = list(itertools.islice(upcoming, max_events))
    return Timeline(
        [
            SortableItemValue(event.timespan_of(dt_util.get_default_time_zone()), event)
            for event in truncated
        ]
    )


class CalendarSyncUpdateCoordinator(DataUpdateCoordinator[Timeline]):
    """Coordinator for calendar RPC calls that use an efficient sync."""

    config_entry: ConfigEntry

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        sync: CalendarEventSyncManager,
        name: str,
    ) -> None:
        """Create the CalendarSyncUpdateCoordinator."""
        super().__init__(
            hass,
            _LOGGER,
            config_entry=config_entry,
            name=name,
            update_interval=MIN_TIME_BETWEEN_UPDATES,
        )
        self.sync: CalendarEventSyncManager = sync
        self._upcoming_timeline: Optional[Timeline] = None

    async def _async_update_data(self) -> Timeline:
        """Fetch data from API endpoint."""
        try:
            await self.sync.run()
        except ApiException as err:
            raise UpdateFailed(f"Error communicating with API: {err}") from err

        timeline: Timeline = await self.sync.store_service.async_get_timeline(
            dt_util.get_default_time_zone()
        )
        self._upcoming_timeline = _truncate_timeline(timeline, MAX_UPCOMING_EVENTS)
        return timeline

    async def async_get_events(
        self, start_date: datetime, end_date: datetime
    ) -> Iterable[Event]:
        """Get all events in a specific time frame."""
        if not self.data:
            raise HomeAssistantError(
                "Unable to get events: Sync from server has not completed"
            )
        return self.data.overlapping(
            start_date,
            end_date,
        )

    @property
    def upcoming(self) -> Optional[Iterable[Event]]:
        """Return upcoming events if any."""
        if self._upcoming_timeline:
            return self._upcoming_timeline.active_after(dt_util.now())
        return None


class CalendarQueryUpdateCoordinator(DataUpdateCoordinator[List[Event]]):
    """Coordinator for calendar RPC calls."""

    config_entry: ConfigEntry

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        calendar_service: GoogleCalendarService,
        name: str,
        calendar_id: str,
        search: Optional[str],
    ) -> None:
        """Create the CalendarQueryUpdateCoordinator."""
        super().__init__(
            hass,
            _LOGGER,
            config_entry=config_entry,
            name=name,
            update_interval=MIN_TIME_BETWEEN_UPDATES,
        )
        self.calendar_service: GoogleCalendarService = calendar_service
        self.calendar_id: str = calendar_id
        self._search: Optional[str] = search

    async def async_get_events(
        self, start_date: datetime, end_date: datetime
    ) -> Iterable[Event]:
        """Get all events in a specific time frame."""
        request: ListEventsRequest = ListEventsRequest(
            calendar_id=self.calendar_id,
            start_time=start_date,
            end_time=end_date,
            search=self._search,
        )
        result_items: List[Event] = []
        try:
            result = await self.calendar_service.async_list_events(request)
            async for result_page in result:
                result_items.extend(result_page.items)
        except ApiException as err:
            self.async_set_update_error(err)
            raise HomeAssistantError(str(err)) from err
        return result_items

    async def _async_update_data(self) -> List[Event]:
        """Fetch data from API endpoint."""
        request: ListEventsRequest = ListEventsRequest(calendar_id=self.calendar_id, search=self._search)
        try:
            result = await self.calendar_service.async_list_events(request)
        except ApiException as err:
            raise UpdateFailed(f"Error communicating with API: {err}") from err
        return result.items

    @property
    def upcoming(self) -> Optional[Iterable[Event]]:
        """Return the next upcoming event if any."""
        return self.data

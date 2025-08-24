"""Calendar platform for a Local Calendar."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
import logging
from typing import Any

from ical.calendar import Calendar
from ical.calendar_stream import IcsCalendarStream
from ical.event import Event
from ical.exceptions import CalendarParseError
from ical.store import EventStore, EventStoreError
from ical.types import Range, Recur
import voluptuous as vol

from homeassistant.components.calendar import (
    EVENT_END,
    EVENT_RRULE,
    EVENT_START,
    CalendarEntity,
    CalendarEntityFeature,
    CalendarEvent,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import dt as dt_util

from .const import CONF_CALENDAR_NAME, DOMAIN
from .store import LocalCalendarStore

_LOGGER: logging.Logger = logging.getLogger(__name__)
PRODID: str = "-//homeassistant.io//local_calendar 1.0//EN"


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the local calendar platform."""
    store: LocalCalendarStore = hass.data[DOMAIN][config_entry.entry_id]
    ics: str = await store.async_load()
    calendar: Calendar = await hass.async_add_executor_job(
        IcsCalendarStream.calendar_from_ics, ics
    )
    calendar.prodid = PRODID
    name: str = config_entry.data[CONF_CALENDAR_NAME]
    entity = LocalCalendarEntity(
        store, calendar, name, unique_id=config_entry.entry_id
    )
    async_add_entities([entity], True)


class LocalCalendarEntity(CalendarEntity):
    """A calendar entity backed by a local iCalendar file."""

    _attr_has_entity_name: bool = True
    _attr_supported_features: int = (
        CalendarEntityFeature.CREATE_EVENT
        | CalendarEntityFeature.DELETE_EVENT
        | CalendarEntityFeature.UPDATE_EVENT
    )

    def __init__(
        self,
        store: LocalCalendarStore,
        calendar: Calendar,
        name: str,
        unique_id: str,
    ) -> None:
        """Initialize LocalCalendarEntity."""
        self._store: LocalCalendarStore = store
        self._calendar: Calendar = calendar
        self._calendar_lock: asyncio.Lock = asyncio.Lock()
        self._event: CalendarEvent | None = None
        self._attr_name = name
        self._attr_unique_id = unique_id

    @property
    def event(self) -> CalendarEvent | None:
        """Return the next upcoming event."""
        return self._event

    async def async_get_events(
        self, hass: HomeAssistant, start_date: datetime, end_date: datetime
    ) -> list[CalendarEvent]:
        """Get all events in a specific time frame."""
        events = self._calendar.timeline_tz(start_date.tzinfo).overlapping(
            start_date, end_date
        )
        return [_get_calendar_event(event) for event in events]

    async def async_update(self) -> None:
        """Update entity state with the next upcoming event."""
        now: datetime = dt_util.now()
        events = self._calendar.timeline_tz(now.tzinfo).active_after(now)
        if (event := next(events, None)) is not None:
            self._event = _get_calendar_event(event)
        else:
            self._event = None

    async def _async_store(self) -> None:
        """Persist the calendar to disk."""
        content: str = IcsCalendarStream.calendar_to_ics(self._calendar)
        await self._store.async_store(content)

    async def async_create_event(self, **kwargs: Any) -> None:
        """Add a new event to calendar."""
        event: Event = _parse_event(kwargs)
        async with self._calendar_lock:
            event_store = EventStore(self._calendar)
            await self.hass.async_add_executor_job(event_store.add, event)
            await self._async_store()
        await self.async_update_ha_state(force_refresh=True)

    async def async_delete_event(
        self,
        uid: str,
        recurrence_id: date | datetime | None = None,
        recurrence_range: Range | None = None,
    ) -> None:
        """Delete an event on the calendar."""
        range_value: Range = Range.NONE
        if recurrence_range == Range.THIS_AND_FUTURE:
            range_value = Range.THIS_AND_FUTURE
        async with self._calendar_lock:
            try:
                EventStore(self._calendar).delete(
                    uid, recurrence_id=recurrence_id, recurrence_range=range_value
                )
            except EventStoreError as err:
                raise HomeAssistantError(f"Error while deleting event: {err}") from err
            await self._async_store()
        await self.async_update_ha_state(force_refresh=True)

    async def async_update_event(
        self,
        uid: str,
        event: dict[str, Any],
        recurrence_id: date | datetime | None = None,
        recurrence_range: Range | None = None,
    ) -> None:
        """Update an existing event on the calendar."""
        new_event: Event = _parse_event(event)
        range_value: Range = Range.NONE
        if recurrence_range == Range.THIS_AND_FUTURE:
            range_value = Range.THIS_AND_FUTURE
        async with self._calendar_lock:
            event_store = EventStore(self._calendar)

            def apply_edit() -> None:
                event_store.edit(
                    uid,
                    new_event,
                    recurrence_id=recurrence_id,
                    recurrence_range=range_value,
                )

            try:
                await self.hass.async_add_executor_job(apply_edit)
            except EventStoreError as err:
                raise HomeAssistantError(f"Error while updating event: {err}") from err
            await self._async_store()
        await self.async_update_ha_state(force_refresh=True)


def _parse_event(event: dict[str, Any]) -> Event:
    """Parse an ical event from a home assistant event dictionary."""
    if (rrule := event.get(EVENT_RRULE)):
        event[EVENT_RRULE] = Recur.from_rrule(rrule)
    for key in (EVENT_START, EVENT_END):
        value = event[key]
        if isinstance(value, datetime) and (value.tzinfo is not None):
            event[key] = dt_util.as_local(value).replace(tzinfo=None)
    try:
        return Event(**event)
    except CalendarParseError as err:
        _LOGGER.debug("Error parsing event input fields: %s (%s)", event, str(err))
        raise vol.Invalid("Error parsing event input fields") from err


def _get_calendar_event(event: Event) -> CalendarEvent:
    """Return a CalendarEvent from an API event."""
    if isinstance(event.start, datetime) and isinstance(event.end, datetime):
        start: datetime = dt_util.as_local(event.start)
        end: datetime = dt_util.as_local(event.end)
        if end - start <= timedelta(seconds=0):
            end = start + timedelta(minutes=30)
    else:
        start = event.start  # type: ignore[assignment]
        end = event.end  # type: ignore[assignment]
        if end - start < timedelta(days=0):  # type: ignore[operator]
            end = start + timedelta(days=1)  # type: ignore[operator]
    return CalendarEvent(
        summary=event.summary,
        start=start,  # type: ignore[arg-type]
        end=end,  # type: ignore[arg-type]
        description=event.description,
        uid=event.uid,
        rrule=event.rrule.as_rrule_str() if event.rrule else None,
        recurrence_id=event.recurrence_id,
        location=event.location,
    )
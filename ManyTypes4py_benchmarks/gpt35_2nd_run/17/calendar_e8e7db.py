from __future__ import annotations
import asyncio
from datetime import date, datetime, timedelta
import logging
from typing import Any, List
from ical.calendar import Calendar
from ical.calendar_stream import IcsCalendarStream
from ical.event import Event
from ical.exceptions import CalendarParseError
from ical.store import EventStore, EventStoreError
from ical.types import Range, Recur
import voluptuous as vol
from homeassistant.components.calendar import EVENT_END, EVENT_RRULE, EVENT_START, CalendarEntity, CalendarEntityFeature, CalendarEvent
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import dt as dt_util
from .const import CONF_CALENDAR_NAME, DOMAIN
from .store import LocalCalendarStore
_LOGGER: logging.Logger = logging.getLogger(__name__)
PRODID: str = '-//homeassistant.io//local_calendar 1.0//EN'

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    store: LocalCalendarStore = hass.data[DOMAIN][config_entry.entry_id]
    ics: str = await store.async_load()
    calendar: Calendar = await hass.async_add_executor_job(IcsCalendarStream.calendar_from_ics, ics)
    calendar.prodid = PRODID
    name: str = config_entry.data[CONF_CALENDAR_NAME]
    entity: LocalCalendarEntity = LocalCalendarEntity(store, calendar, name, unique_id=config_entry.entry_id)
    async_add_entities([entity], True)

class LocalCalendarEntity(CalendarEntity):
    _attr_has_entity_name: bool = True
    _attr_supported_features: int = CalendarEntityFeature.CREATE_EVENT | CalendarEntityFeature.DELETE_EVENT | CalendarEntityFeature.UPDATE_EVENT

    def __init__(self, store: LocalCalendarStore, calendar: Calendar, name: str, unique_id: str) -> None:
        self._store: LocalCalendarStore = store
        self._calendar: Calendar = calendar
        self._calendar_lock: asyncio.Lock = asyncio.Lock()
        self._event: CalendarEvent = None
        self._attr_name: str = name
        self._attr_unique_id: str = unique_id

    @property
    def event(self) -> CalendarEvent:
        return self._event

    async def async_get_events(self, hass: HomeAssistant, start_date: datetime, end_date: datetime) -> List[CalendarEvent]:
        events = self._calendar.timeline_tz(start_date.tzinfo).overlapping(start_date, end_date)
        return [_get_calendar_event(event) for event in events]

    async def async_update(self) -> None:
        now: datetime = dt_util.now()
        events = self._calendar.timeline_tz(now.tzinfo).active_after(now)
        if (event := next(events, None)):
            self._event = _get_calendar_event(event)
        else:
            self._event = None

    async def _async_store(self) -> None:
        content: str = IcsCalendarStream.calendar_to_ics(self._calendar)
        await self._store.async_store(content)

    async def async_create_event(self, **kwargs: Any) -> None:
        event: Event = _parse_event(kwargs)
        async with self._calendar_lock:
            event_store: EventStore = EventStore(self._calendar)
            await self.hass.async_add_executor_job(event_store.add, event)
            await self._async_store()
        await self.async_update_ha_state(force_refresh=True)

    async def async_delete_event(self, uid: str, recurrence_id: str = None, recurrence_range: Range = None) -> None:
        range_value: Range = Range.NONE
        if recurrence_range == Range.THIS_AND_FUTURE:
            range_value = Range.THIS_AND_FUTURE
        async with self._calendar_lock:
            try:
                EventStore(self._calendar).delete(uid, recurrence_id=recurrence_id, recurrence_range=range_value)
            except EventStoreError as err:
                raise HomeAssistantError(f'Error while deleting event: {err}') from err
            await self._async_store()
        await self.async_update_ha_state(force_refresh=True)

    async def async_update_event(self, uid: str, event: Any, recurrence_id: str = None, recurrence_range: Range = None) -> None:
        new_event: Event = _parse_event(event)
        range_value: Range = Range.NONE
        if recurrence_range == Range.THIS_AND_FUTURE:
            range_value = Range.THIS_AND_FUTURE
        async with self._calendar_lock:
            event_store: EventStore = EventStore(self._calendar)

            def apply_edit() -> None:
                event_store.edit(uid, new_event, recurrence_id=recurrence_id, recurrence_range=range_value)
            try:
                await self.hass.async_add_executor_job(apply_edit)
            except EventStoreError as err:
                raise HomeAssistantError(f'Error while updating event: {err}') from err
            await self._async_store()
        await self.async_update_ha_state(force_refresh=True)

def _parse_event(event: Any) -> Event:
    if (rrule := event.get(EVENT_RRULE)):
        event[EVENT_RRULE] = Recur.from_rrule(rrule)
    for key in (EVENT_START, EVENT_END):
        if (value := event[key]) and isinstance(value, datetime) and (value.tzinfo is not None):
            event[key] = dt_util.as_local(value).replace(tzinfo=None)
    try:
        return Event(**event)
    except CalendarParseError as err:
        _LOGGER.debug('Error parsing event input fields: %s (%s)', event, str(err))
        raise vol.Invalid('Error parsing event input fields') from err

def _get_calendar_event(event: Event) -> CalendarEvent:
    if isinstance(event.start, datetime) and isinstance(event.end, datetime):
        start: datetime = dt_util.as_local(event.start)
        end: datetime = dt_util.as_local(event.end)
        if end - start <= timedelta(seconds=0):
            end = start + timedelta(minutes=30)
    else:
        start = event.start
        end = event.end
        if end - start < timedelta(days=0):
            end = start + timedelta(days=1)
    return CalendarEvent(summary=event.summary, start=start, end=end, description=event.description, uid=event.uid, rrule=event.rrule.as_rrule_str() if event.rrule else None, recurrence_id=event.recurrence_id, location=event.location)

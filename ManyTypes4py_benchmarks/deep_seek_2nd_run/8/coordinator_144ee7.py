"""Data update coordinator for caldav."""
from __future__ import annotations
from datetime import date, datetime, time, timedelta
from functools import partial
import logging
import re
from typing import TYPE_CHECKING, Optional, List, Pattern, Union, cast
import caldav
from homeassistant.components.calendar import CalendarEvent, extract_offset
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from .api import get_attr_value
if TYPE_CHECKING:
    from . import CalDavConfigEntry
    from caldav.objects import Event, Calendar
    from icalendar.prop import vDDDTypes

_LOGGER = logging.getLogger(__name__)
MIN_TIME_BETWEEN_UPDATES = timedelta(minutes=15)
OFFSET = '!!'

class CalDavUpdateCoordinator(DataUpdateCoordinator[Optional[CalendarEvent]]):
    """Class to utilize the calendar dav client object to get next event."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: CalDavConfigEntry,
        calendar: Calendar,
        days: int,
        include_all_day: bool,
        search: Optional[str]
    ) -> None:
        """Set up how we are going to search the WebDav calendar."""
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=f'CalDAV {calendar.name}',
            update_interval=MIN_TIME_BETWEEN_UPDATES
        )
        self.calendar: Calendar = calendar
        self.days: int = days
        self.include_all_day: bool = include_all_day
        self.search: Optional[str] = search
        self.offset: Optional[str] = None

    async def async_get_events(
        self,
        hass: HomeAssistant,
        start_date: datetime,
        end_date: datetime
    ) -> List[CalendarEvent]:
        """Get all events in a specific time frame."""
        vevent_list: List[Event] = await hass.async_add_executor_job(
            partial(
                self.calendar.search,
                start=start_date,
                end=end_date,
                event=True,
                expand=True
            )
        )
        event_list: List[CalendarEvent] = []
        for event in vevent_list:
            if not hasattr(event.instance, 'vevent'):
                _LOGGER.warning("Skipped event with missing 'vevent' property")
                continue
            vevent = event.instance.vevent
            if not self.is_matching(vevent, self.search):
                continue
            event_list.append(
                CalendarEvent(
                    summary=get_attr_value(vevent, 'summary') or '',
                    start=self.to_local(vevent.dtstart.value),
                    end=self.to_local(self.get_end_date(vevent)),
                    location=get_attr_value(vevent, 'location'),
                    description=get_attr_value(vevent, 'description')
                )
            )
        return event_list

    async def _async_update_data(self) -> Optional[CalendarEvent]:
        """Get the latest data."""
        start_of_today: datetime = dt_util.start_of_local_day()
        start_of_tomorrow: datetime = dt_util.start_of_local_day() + timedelta(days=self.days)
        results: List[Event] = await self.hass.async_add_executor_job(
            partial(
                self.calendar.search,
                start=start_of_today,
                end=start_of_tomorrow,
                event=True,
                expand=True
            )
        )
        new_events: List[Event] = []
        for event in results:
            if not hasattr(event.instance, 'vevent'):
                _LOGGER.warning("Skipped event with missing 'vevent' property")
                continue
            vevent = event.instance.vevent
            for start_dt in vevent.getrruleset() or []:
                if self.is_all_day(vevent):
                    start_dt = start_dt.date()
                    _start_of_today = start_of_today.date()
                    _start_of_tomorrow = start_of_tomorrow.date()
                else:
                    _start_of_today = start_of_today
                    _start_of_tomorrow = start_of_tomorrow
                if _start_of_today <= start_dt < _start_of_tomorrow:
                    new_event = event.copy()
                    new_vevent = new_event.instance.vevent
                    if hasattr(new_vevent, 'dtend'):
                        dur = new_vevent.dtend.value - new_vevent.dtstart.value
                        new_vevent.dtend.value = start_dt + dur
                    new_vevent.dtstart.value = start_dt
                    new_events.append(new_event)
                elif _start_of_tomorrow <= start_dt:
                    break
        vevents: List[vDDDTypes] = [
            event.instance.vevent for event in results + new_events
            if hasattr(event.instance, 'vevent')
        ]
        vevents.sort(key=lambda x: self.to_datetime(x.dtstart.value))
        vevent: Optional[vDDDTypes] = next(
            (
                vevent for vevent in vevents
                if self.is_matching(vevent, self.search)
                and (not self.is_all_day(vevent) or self.include_all_day)
                and (not self.is_over(vevent))
            ),
            None
        )
        if vevent is None:
            _LOGGER.debug(
                'No matching event found in the %d results for %s',
                len(vevents),
                self.calendar.name
            )
            self.offset = None
            return None
        summary, offset = extract_offset(get_attr_value(vevent, 'summary') or '', OFFSET)
        self.offset = offset
        return CalendarEvent(
            summary=summary,
            start=self.to_local(vevent.dtstart.value),
            end=self.to_local(self.get_end_date(vevent)),
            location=get_attr_value(vevent, 'location'),
            description=get_attr_value(vevent, 'description')
        )

    @staticmethod
    def is_matching(vevent: vDDDTypes, search: Optional[str]) -> bool:
        """Return if the event matches the filter criteria."""
        if search is None:
            return True
        pattern: Pattern[str] = re.compile(search)
        return (
            hasattr(vevent, 'summary') and pattern.match(vevent.summary.value)
            or (hasattr(vevent, 'location') and pattern.match(vevent.location.value))
            or (hasattr(vevent, 'description') and pattern.match(vevent.description.value))
        )

    @staticmethod
    def is_all_day(vevent: vDDDTypes) -> bool:
        """Return if the event last the whole day."""
        return not isinstance(vevent.dtstart.value, datetime)

    @staticmethod
    def is_over(vevent: vDDDTypes) -> bool:
        """Return if the event is over."""
        return dt_util.now() >= CalDavUpdateCoordinator.to_datetime(
            CalDavUpdateCoordinator.get_end_date(vevent)
        )

    @staticmethod
    def to_datetime(obj: Union[datetime, date]) -> datetime:
        """Return a datetime."""
        if isinstance(obj, datetime):
            return CalDavUpdateCoordinator.to_local(obj)
        return datetime.combine(obj, time.min).replace(tzinfo=dt_util.get_default_time_zone())

    @staticmethod
    def to_local(obj: Union[datetime, date]) -> Union[datetime, date]:
        """Return a datetime as a local datetime, leaving dates unchanged."""
        if isinstance(obj, datetime):
            return dt_util.as_local(obj)
        return obj

    @staticmethod
    def get_end_date(obj: vDDDTypes) -> Union[datetime, date]:
        """Return the end datetime as determined by dtend or duration."""
        if hasattr(obj, 'dtend'):
            enddate = obj.dtend.value
        elif hasattr(obj, 'duration'):
            enddate = obj.dtstart.value + obj.duration.value
        else:
            enddate = obj.dtstart.value + timedelta(days=1)
        if not isinstance(enddate, datetime) and obj.dtstart.value == enddate:
            enddate += timedelta(days=1)
        return enddate

"""Holiday Calendar."""
from __future__ import annotations
from datetime import datetime, timedelta, date
from typing import Optional, Tuple, List, Any, Callable, Dict
from holidays import PUBLIC, HolidayBase, country_holidays
from homeassistant.components.calendar import CalendarEntity, CalendarEvent
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_COUNTRY
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.util import dt as dt_util
from .const import CONF_CATEGORIES, CONF_PROVINCE, DOMAIN

def _get_obj_holidays_and_language(
    country: str,
    province: Optional[str],
    language: str,
    selected_categories: Optional[List[Any]]
) -> Tuple[HolidayBase, str]:
    """Get the object for the requested country and year."""
    if selected_categories is None:
        categories: List[Any] = [PUBLIC]
    else:
        categories = [PUBLIC, *selected_categories]
    current_year = dt_util.now().year
    obj_holidays: HolidayBase = country_holidays(
        country,
        subdiv=province,
        years={current_year, current_year + 1},
        language=language,
        categories=categories
    )
    if language == 'en':
        for lang in obj_holidays.supported_languages:
            if lang.startswith('en'):
                obj_holidays = country_holidays(
                    country,
                    subdiv=province,
                    years={current_year, current_year + 1},
                    language=lang,
                    categories=categories
                )
                language = lang
                break
    if obj_holidays.supported_languages and language not in obj_holidays.supported_languages and (default_language := obj_holidays.default_language):
        obj_holidays = country_holidays(
            country,
            subdiv=province,
            years={current_year, current_year + 1},
            language=default_language,
            categories=categories
        )
        language = default_language
    return obj_holidays, language

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up the Holiday Calendar config entry."""
    country: str = config_entry.data[CONF_COUNTRY]
    province: Optional[str] = config_entry.data.get(CONF_PROVINCE)
    categories: Optional[List[Any]] = config_entry.options.get(CONF_CATEGORIES)
    language: str = hass.config.language
    obj_holidays, language = await hass.async_add_executor_job(
        _get_obj_holidays_and_language, country, province, language, categories
    )
    async_add_entities([
        HolidayCalendarEntity(
            config_entry.title,
            country,
            province,
            language,
            categories,
            obj_holidays,
            config_entry.entry_id
        )
    ], True)

class HolidayCalendarEntity(CalendarEntity):
    """Representation of a Holiday Calendar element."""
    _attr_has_entity_name: bool = True
    _attr_name: Optional[str] = None
    _attr_event: Optional[CalendarEvent] = None
    _attr_should_poll: bool = False
    unsub: Optional[CALLBACK_TYPE] = None

    def __init__(
        self,
        name: str,
        country: str,
        province: Optional[str],
        language: str,
        categories: Optional[List[Any]],
        obj_holidays: HolidayBase,
        unique_id: str
    ) -> None:
        """Initialize HolidayCalendarEntity."""
        self._country: str = country
        self._province: Optional[str] = province
        self._location: str = name
        self._language: str = language
        self._categories: Optional[List[Any]] = categories
        self._attr_unique_id: str = unique_id
        self._attr_device_info: DeviceInfo = DeviceInfo(
            identifiers={(DOMAIN, unique_id)},
            entry_type=DeviceEntryType.SERVICE,
            name=name
        )
        self._obj_holidays: HolidayBase = obj_holidays

    def get_next_interval(self, now: datetime) -> datetime:
        """Compute next time an update should occur."""
        tomorrow: datetime = dt_util.as_local(now) + timedelta(days=1)
        return dt_util.start_of_local_day(tomorrow)

    def _update_state_and_setup_listener(self) -> None:
        """Update state and setup listener for next interval."""
        now: datetime = dt_util.now()
        self._attr_event = self.update_event(now)
        self.unsub = async_track_point_in_utc_time(
            self.hass,
            self.point_in_time_listener,
            self.get_next_interval(now)
        )

    @callback
    def point_in_time_listener(self, time_date: datetime) -> None:
        """Get the latest data and update state."""
        self._update_state_and_setup_listener()
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Set up first update."""
        self._update_state_and_setup_listener()

    def update_event(self, now: datetime) -> Optional[CalendarEvent]:
        """Return the next upcoming event."""
        next_holiday: Optional[Tuple[date, Any]] = None
        for holiday_date, holiday_name in sorted(self._obj_holidays.items(), key=lambda x: x[0]):
            if holiday_date >= now.date():
                next_holiday = (holiday_date, holiday_name)
                break
        if next_holiday is None:
            return None
        return CalendarEvent(
            summary=next_holiday[1],
            start=next_holiday[0],
            end=next_holiday[0],
            location=self._location
        )

    @property
    def event(self) -> Optional[CalendarEvent]:
        """Return the next upcoming event."""
        return self._attr_event

    async def async_get_events(
        self,
        hass: HomeAssistant,
        start_date: datetime,
        end_date: datetime
    ) -> List[CalendarEvent]:
        """Get all events in a specific time frame."""
        years = list({start_date.year, end_date.year})
        obj_holidays: HolidayBase = country_holidays(
            self._country,
            subdiv=self._province,
            years=years,
            language=self._language,
            categories=self._categories
        )
        event_list: List[CalendarEvent] = []
        for holiday_date, holiday_name in obj_holidays.items():
            if start_date.date() <= holiday_date <= end_date.date():
                event = CalendarEvent(
                    summary=holiday_name,
                    start=holiday_date,
                    end=holiday_date,
                    location=self._location
                )
                event_list.append(event)
        return event_list

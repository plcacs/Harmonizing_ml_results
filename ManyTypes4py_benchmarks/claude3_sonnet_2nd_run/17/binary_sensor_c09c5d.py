"""Sensor to indicate whether the current day is a workday."""
from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Final, List, Optional, Set, Tuple, Union, Dict, Any, cast
from holidays import PUBLIC, HolidayBase, __version__ as python_holidays_version, country_holidays
import voluptuous as vol
from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_COUNTRY, CONF_LANGUAGE, CONF_NAME
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, ServiceResponse, SupportsResponse, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, async_get_current_platform
from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue
from homeassistant.util import dt as dt_util, slugify
from .const import ALLOWED_DAYS, CONF_ADD_HOLIDAYS, CONF_CATEGORY, CONF_EXCLUDES, CONF_OFFSET, CONF_PROVINCE, CONF_REMOVE_HOLIDAYS, CONF_WORKDAYS, DOMAIN, LOGGER
SERVICE_CHECK_DATE: str = 'check_date'
CHECK_DATE: str = 'check_date'

def validate_dates(holiday_list: List[str]) -> List[str]:
    """Validate and adds to list of dates to add or remove."""
    calc_holidays: List[str] = []
    for add_date in holiday_list:
        if add_date.find(',') > 0:
            dates = add_date.split(',', maxsplit=1)
            d1 = dt_util.parse_date(dates[0])
            d2 = dt_util.parse_date(dates[1])
            if d1 is None or d2 is None:
                LOGGER.error('Incorrect dates in date range: %s', add_date)
                continue
            _range = d2 - d1
            for i in range(_range.days + 1):
                day = d1 + timedelta(days=i)
                calc_holidays.append(day.strftime('%Y-%m-%d'))
            continue
        calc_holidays.append(add_date)
    return calc_holidays

def _get_obj_holidays(country: Optional[str], province: Optional[str], year: int, language: Optional[str], categories: Optional[List[str]]) -> HolidayBase:
    """Get the object for the requested country and year."""
    if not country:
        return HolidayBase()
    set_categories: Optional[Tuple[str, ...]] = None
    if categories:
        category_list: List[str] = [PUBLIC]
        category_list.extend(categories)
        set_categories = tuple(category_list)
    obj_holidays = country_holidays(country, subdiv=province, years=[year, year + 1], language=language, categories=set_categories)
    if (supported_languages := obj_holidays.supported_languages) and language and language.startswith('en'):
        for lang in supported_languages:
            if lang.startswith('en'):
                obj_holidays = country_holidays(country, subdiv=province, years=year, language=lang, categories=set_categories)
            LOGGER.debug('Changing language from %s to %s', language, lang)
    return obj_holidays

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the Workday sensor."""
    add_holidays: List[str] = entry.options[CONF_ADD_HOLIDAYS]
    remove_holidays: List[str] = entry.options[CONF_REMOVE_HOLIDAYS]
    country: Optional[str] = entry.options.get(CONF_COUNTRY)
    days_offset: int = int(entry.options[CONF_OFFSET])
    excludes: List[str] = entry.options[CONF_EXCLUDES]
    province: Optional[str] = entry.options.get(CONF_PROVINCE)
    sensor_name: str = entry.options[CONF_NAME]
    workdays: List[str] = entry.options[CONF_WORKDAYS]
    language: Optional[str] = entry.options.get(CONF_LANGUAGE)
    categories: Optional[List[str]] = entry.options.get(CONF_CATEGORY)
    year: int = (dt_util.now() + timedelta(days=days_offset)).year
    obj_holidays: HolidayBase = await hass.async_add_executor_job(_get_obj_holidays, country, province, year, language, categories)
    calc_add_holidays: List[str] = validate_dates(add_holidays)
    calc_remove_holidays: List[str] = validate_dates(remove_holidays)
    next_year: int = dt_util.now().year + 1
    try:
        obj_holidays.append(calc_add_holidays)
    except ValueError as error:
        LOGGER.error('Could not add custom holidays: %s', error)
    for remove_holiday in calc_remove_holidays:
        try:
            if dt_util.parse_date(remove_holiday):
                removed = obj_holidays.pop(remove_holiday)
                LOGGER.debug('Removed %s', remove_holiday)
            else:
                LOGGER.debug("Treating '%s' as named holiday", remove_holiday)
                removed = obj_holidays.pop_named(remove_holiday)
                for holiday in removed:
                    LOGGER.debug("Removed %s by name '%s'", holiday, remove_holiday)
        except KeyError as unmatched:
            LOGGER.warning('No holiday found matching %s', unmatched)
            if (_date := dt_util.parse_date(remove_holiday)):
                if _date.year <= next_year:
                    async_create_issue(hass, DOMAIN, f'bad_date_holiday-{entry.entry_id}-{slugify(remove_holiday)}', is_fixable=True, is_persistent=False, severity=IssueSeverity.WARNING, translation_key='bad_date_holiday', translation_placeholders={CONF_COUNTRY: country if country else '-', 'title': entry.title, CONF_REMOVE_HOLIDAYS: remove_holiday}, data={'entry_id': entry.entry_id, 'country': country, 'named_holiday': remove_holiday})
            else:
                async_create_issue(hass, DOMAIN, f'bad_named_holiday-{entry.entry_id}-{slugify(remove_holiday)}', is_fixable=True, is_persistent=False, severity=IssueSeverity.WARNING, translation_key='bad_named_holiday', translation_placeholders={CONF_COUNTRY: country if country else '-', 'title': entry.title, CONF_REMOVE_HOLIDAYS: remove_holiday}, data={'entry_id': entry.entry_id, 'country': country, 'named_holiday': remove_holiday})
    LOGGER.debug('Found the following holidays for your configuration:')
    for holiday_date, name in sorted(obj_holidays.items()):
        _holiday_string = holiday_date.strftime('%Y-%m-%d')
        LOGGER.debug('%s %s', _holiday_string, name)
    platform = async_get_current_platform()
    platform.async_register_entity_service(SERVICE_CHECK_DATE, {vol.Required(CHECK_DATE): cv.date}, 'check_date', None, SupportsResponse.ONLY)
    async_add_entities([IsWorkdaySensor(obj_holidays, workdays, excludes, days_offset, sensor_name, entry.entry_id)])

class IsWorkdaySensor(BinarySensorEntity):
    """Implementation of a Workday sensor."""
    _attr_has_entity_name: bool = True
    _attr_name: Optional[str] = None
    _attr_translation_key: str = DOMAIN
    _attr_should_poll: bool = False
    unsub: Optional[CALLBACK_TYPE] = None

    def __init__(self, obj_holidays: HolidayBase, workdays: List[str], excludes: List[str], days_offset: int, name: str, entry_id: str) -> None:
        """Initialize the Workday sensor."""
        self._obj_holidays: HolidayBase = obj_holidays
        self._workdays: List[str] = workdays
        self._excludes: List[str] = excludes
        self._days_offset: int = days_offset
        self._attr_extra_state_attributes: Dict[str, Any] = {CONF_WORKDAYS: workdays, CONF_EXCLUDES: excludes, CONF_OFFSET: days_offset}
        self._attr_unique_id: str = entry_id
        self._attr_device_info: DeviceInfo = DeviceInfo(entry_type=DeviceEntryType.SERVICE, identifiers={(DOMAIN, entry_id)}, manufacturer='python-holidays', model=python_holidays_version, name=name)

    def is_include(self, day: str, now: datetime) -> bool:
        """Check if given day is in the includes list."""
        if day in self._workdays:
            return True
        if 'holiday' in self._workdays and now in self._obj_holidays:
            return True
        return False

    def is_exclude(self, day: str, now: datetime) -> bool:
        """Check if given day is in the excludes list."""
        if day in self._excludes:
            return True
        if 'holiday' in self._excludes and now in self._obj_holidays:
            return True
        return False

    def get_next_interval(self, now: datetime) -> datetime:
        """Compute next time an update should occur."""
        tomorrow = dt_util.as_local(now) + timedelta(days=1)
        return dt_util.start_of_local_day(tomorrow)

    def _update_state_and_setup_listener(self) -> None:
        """Update state and setup listener for next interval."""
        now = dt_util.now()
        self.update_data(now)
        self.unsub = async_track_point_in_utc_time(self.hass, self.point_in_time_listener, self.get_next_interval(now))

    @callback
    def point_in_time_listener(self, time_date: datetime) -> None:
        """Get the latest data and update state."""
        self._update_state_and_setup_listener()
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Set up first update."""
        self._update_state_and_setup_listener()

    def update_data(self, now: datetime) -> None:
        """Get date and look whether it is a holiday."""
        self._attr_is_on = self.date_is_workday(now)

    def check_date(self, check_date: date) -> ServiceResponse:
        """Service to check if date is workday or not."""
        return {'workday': self.date_is_workday(check_date)}

    def date_is_workday(self, check_date: Union[datetime, date]) -> bool:
        """Check if date is workday."""
        is_workday: bool = False
        adjusted_date = check_date + timedelta(days=self._days_offset)
        day = adjusted_date.isoweekday() - 1
        day_of_week = ALLOWED_DAYS[day]
        if self.is_include(day_of_week, adjusted_date):
            is_workday = True
        if self.is_exclude(day_of_week, adjusted_date):
            is_workday = False
        return is_workday

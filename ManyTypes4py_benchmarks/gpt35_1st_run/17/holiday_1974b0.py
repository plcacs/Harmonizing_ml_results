from __future__ import annotations
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, List, Tuple, Union, Callable
import warnings
from dateutil.relativedelta import FR, MO, SA, SU, TH, TU, WE
import numpy as np
from pandas._libs.tslibs.offsets import BaseOffset
from pandas.errors import PerformanceWarning
from pandas import DateOffset, DatetimeIndex, Series, Timestamp, concat, date_range
from pandas.tseries.offsets import Day, Easter

if TYPE_CHECKING:
    from collections.abc import Callable

def next_monday(dt: datetime) -> datetime:
    ...

def next_monday_or_tuesday(dt: datetime) -> datetime:
    ...

def previous_friday(dt: datetime) -> datetime:
    ...

def sunday_to_monday(dt: datetime) -> datetime:
    ...

def weekend_to_monday(dt: datetime) -> datetime:
    ...

def nearest_workday(dt: datetime) -> datetime:
    ...

def next_workday(dt: datetime) -> datetime:
    ...

def previous_workday(dt: datetime) -> datetime:
    ...

def before_nearest_workday(dt: datetime) -> datetime:
    ...

def after_nearest_workday(dt: datetime) -> datetime:
    ...

class Holiday:
    def __init__(self, name: str, year: Union[int, None] = None, month: Union[int, None] = None, day: Union[int, None] = None, offset: Union[List[BaseOffset], BaseOffset, None] = None, observance: Union[Callable, None] = None, start_date: Union[datetime, None] = None, end_date: Union[datetime, None] = None, days_of_week: Union[Tuple[int, ...], None] = None):
        ...

    def __repr__(self) -> str:
        ...

    def dates(self, start_date: datetime, end_date: datetime, return_name: bool = False) -> Union[Series, DatetimeIndex]:
        ...

    def _reference_dates(self, start_date: datetime, end_date: datetime) -> DatetimeIndex:
        ...

    def _apply_rule(self, dates: DatetimeIndex) -> DatetimeIndex:
        ...

def register(cls: type) -> None:
    ...

def get_calendar(name: str) -> AbstractHolidayCalendar:
    ...

class HolidayCalendarMetaClass(type):
    def __new__(cls, clsname: str, bases: Tuple[type, ...], attrs: dict) -> type:
        ...

class AbstractHolidayCalendar(metaclass=HolidayCalendarMetaClass):
    def __init__(self, name: str = '', rules: Union[List[Holiday], None] = None) -> None:
        ...

    def rule_from_name(self, name: str) -> Union[Holiday, None]:
        ...

    def holidays(self, start: Union[datetime, None] = None, end: Union[datetime, None] = None, return_name: bool = False) -> Union[Series, DatetimeIndex]:
        ...

    @staticmethod
    def merge_class(base: Union[AbstractHolidayCalendar, List[Holiday]], other: Union[AbstractHolidayCalendar, List[Holiday]]) -> List[Holiday]:
        ...

    def merge(self, other: Union[AbstractHolidayCalendar, List[Holiday]], inplace: bool = False) -> Union[List[Holiday], None]:
        ...

USMemorialDay = Holiday('Memorial Day', month=5, day=31, offset=DateOffset(weekday=MO(-1)))
USLaborDay = Holiday('Labor Day', month=9, day=1, offset=DateOffset(weekday=MO(1)))
USColumbusDay = Holiday('Columbus Day', month=10, day=1, offset=DateOffset(weekday=MO(2)))
USThanksgivingDay = Holiday('Thanksgiving Day', month=11, day=1, offset=DateOffset(weekday=TH(4)))
USMartinLutherKingJr = Holiday('Birthday of Martin Luther King, Jr.', start_date=datetime(1986, 1, 1), month=1, day=1, offset=DateOffset(weekday=MO(3)))
USPresidentsDay = Holiday("Washington's Birthday", month=2, day=1, offset=DateOffset(weekday=MO(3)))
GoodFriday = Holiday('Good Friday', month=1, day=1, offset=[Easter(), Day(-2)])
EasterMonday = Holiday('Easter Monday', month=1, day=1, offset=[Easter(), Day(1)])

class USFederalHolidayCalendar(AbstractHolidayCalendar):
    rules = [Holiday("New Year's Day", month=1, day=1, observance=nearest_workday), USMartinLutherKingJr, USPresidentsDay, USMemorialDay, Holiday('Juneteenth National Independence Day', month=6, day=19, start_date='2021-06-18', observance=nearest_workday), Holiday('Independence Day', month=7, day=4, observance=nearest_workday), USLaborDay, USColumbusDay, Holiday('Veterans Day', month=11, day=11, observance=nearest_workday), USThanksgivingDay, Holiday('Christmas Day', month=12, day=25, observance=nearest_workday)]

def HolidayCalendarFactory(name: str, base: Union[AbstractHolidayCalendar, List[Holiday]], other: Union[AbstractHolidayCalendar, List[Holiday]], base_class: type = AbstractHolidayCalendar) -> type:
    ...

__all__ = ['FR', 'MO', 'SA', 'SU', 'TH', 'TU', 'WE', 'HolidayCalendarFactory', 'after_nearest_workday', 'before_nearest_workday', 'get_calendar', 'nearest_workday', 'next_monday', 'next_monday_or_tuesday', 'next_workday', 'previous_friday', 'previous_workday', 'register', 'sunday_to_monday', 'weekend_to_monday']

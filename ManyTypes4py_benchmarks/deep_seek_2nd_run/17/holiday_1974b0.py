from __future__ import annotations
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional, Union, List, Tuple, Dict, cast
import warnings
from dateutil.relativedelta import FR, MO, SA, SU, TH, TU, WE
import numpy as np
from pandas._libs.tslibs.offsets import BaseOffset
from pandas.errors import PerformanceWarning
from pandas import DateOffset, DatetimeIndex, Series, Timestamp, concat, date_range
from pandas.tseries.offsets import Day, Easter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas import DataFrame

DateLike = Union[str, datetime, Timestamp]
Weekday = Union[int, str]
OffsetLike = Union[BaseOffset, List[BaseOffset]]
ObservanceFunc = Callable[[Timestamp], Timestamp]

def next_monday(dt: Timestamp) -> Timestamp:
    """
    If holiday falls on Saturday, use following Monday instead;
    if holiday falls on Sunday, use Monday instead
    """
    if dt.weekday() == 5:
        return dt + timedelta(2)
    elif dt.weekday() == 6:
        return dt + timedelta(1)
    return dt

def next_monday_or_tuesday(dt: Timestamp) -> Timestamp:
    """
    For second holiday of two adjacent ones!
    If holiday falls on Saturday, use following Monday instead;
    if holiday falls on Sunday or Monday, use following Tuesday instead
    (because Monday is already taken by adjacent holiday on the day before)
    """
    dow = dt.weekday()
    if dow in (5, 6):
        return dt + timedelta(2)
    if dow == 0:
        return dt + timedelta(1)
    return dt

def previous_friday(dt: Timestamp) -> Timestamp:
    """
    If holiday falls on Saturday or Sunday, use previous Friday instead.
    """
    if dt.weekday() == 5:
        return dt - timedelta(1)
    elif dt.weekday() == 6:
        return dt - timedelta(2)
    return dt

def sunday_to_monday(dt: Timestamp) -> Timestamp:
    """
    If holiday falls on Sunday, use day thereafter (Monday) instead.
    """
    if dt.weekday() == 6:
        return dt + timedelta(1)
    return dt

def weekend_to_monday(dt: Timestamp) -> Timestamp:
    """
    If holiday falls on Sunday or Saturday,
    use day thereafter (Monday) instead.
    Needed for holidays such as Christmas observation in Europe
    """
    if dt.weekday() == 6:
        return dt + timedelta(1)
    elif dt.weekday() == 5:
        return dt + timedelta(2)
    return dt

def nearest_workday(dt: Timestamp) -> Timestamp:
    """
    If holiday falls on Saturday, use day before (Friday) instead;
    if holiday falls on Sunday, use day thereafter (Monday) instead.
    """
    if dt.weekday() == 5:
        return dt - timedelta(1)
    elif dt.weekday() == 6:
        return dt + timedelta(1)
    return dt

def next_workday(dt: Timestamp) -> Timestamp:
    """
    returns next workday used for observances
    """
    dt += timedelta(days=1)
    while dt.weekday() > 4:
        dt += timedelta(days=1)
    return dt

def previous_workday(dt: Timestamp) -> Timestamp:
    """
    returns previous workday used for observances
    """
    dt -= timedelta(days=1)
    while dt.weekday() > 4:
        dt -= timedelta(days=1)
    return dt

def before_nearest_workday(dt: Timestamp) -> Timestamp:
    """
    returns previous workday before nearest workday
    """
    return previous_workday(nearest_workday(dt))

def after_nearest_workday(dt: Timestamp) -> Timestamp:
    """
    returns next workday after nearest workday
    needed for Boxing day or multiple holidays in a series
    """
    return next_workday(nearest_workday(dt))

class Holiday:
    """
    Class that defines a holiday with start/end dates and rules
    for observance.
    """

    def __init__(
        self,
        name: str,
        year: Optional[int] = None,
        month: Optional[int] = None,
        day: Optional[int] = None,
        offset: Optional[OffsetLike] = None,
        observance: Optional[ObservanceFunc] = None,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        days_of_week: Optional[Tuple[Weekday, ...]] = None
    ) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the holiday , defaults to class name
        year : int, default None
            Year of the holiday
        month : int, default None
            Month of the holiday
        day : int, default None
            Day of the holiday
        offset : list of pandas.tseries.offsets or
                class from pandas.tseries.offsets, default None
            Computes offset from date
        observance : function, default None
            Computes when holiday is given a pandas Timestamp
        start_date : datetime-like, default None
            First date the holiday is observed
        end_date : datetime-like, default None
            Last date the holiday is observed
        days_of_week : tuple of int or dateutil.relativedelta weekday strs, default None
            Provide a tuple of days e.g  (0,1,2,3,) for Monday Through Thursday
            Monday=0,..,Sunday=6
        """
        if offset is not None:
            if observance is not None:
                raise NotImplementedError('Cannot use both offset and observance.')
            if not (isinstance(offset, BaseOffset) or (isinstance(offset, list) and all((isinstance(off, BaseOffset) for off in offset))):
                raise ValueError('Only BaseOffsets and flat lists of them are supported for offset.')
        self.name = name
        self.year = year
        self.month = month
        self.day = day
        self.offset = offset
        self.start_date = Timestamp(start_date) if start_date is not None else start_date
        self.end_date = Timestamp(end_date) if end_date is not None else end_date
        self.observance = observance
        assert days_of_week is None or isinstance(days_of_week, tuple)
        self.days_of_week = days_of_week

    def __repr__(self) -> str:
        info = ''
        if self.year is not None:
            info += f'year={self.year}, '
        info += f'month={self.month}, day={self.day}, '
        if self.offset is not None:
            info += f'offset={self.offset}'
        if self.observance is not None:
            info += f'observance={self.observance}'
        repr = f'Holiday: {self.name} ({info})'
        return repr

    def dates(
        self,
        start_date: DateLike,
        end_date: DateLike,
        return_name: bool = False
    ) -> Union[DatetimeIndex, Series]:
        """
        Calculate holidays observed between start date and end date

        Parameters
        ----------
        start_date : starting date, datetime-like, optional
        end_date : ending date, datetime-like, optional
        return_name : bool, optional, default=False
            If True, return a series that has dates and holiday names.
            False will only return dates.

        Returns
        -------
        Series or DatetimeIndex
            Series if return_name is True
        """
        start_date = Timestamp(start_date)
        end_date = Timestamp(end_date)
        filter_start_date = start_date
        filter_end_date = end_date
        if self.year is not None:
            dt = Timestamp(datetime(self.year, self.month, self.day))
            dti = DatetimeIndex([dt])
            if return_name:
                return Series(self.name, index=dti)
            else:
                return dti
        dates = self._reference_dates(start_date, end_date)
        holiday_dates = self._apply_rule(dates)
        if self.days_of_week is not None:
            holiday_dates = holiday_dates[np.isin(holiday_dates.dayofweek, self.days_of_week).ravel()]
        if self.start_date is not None:
            filter_start_date = max(self.start_date.tz_localize(filter_start_date.tz), filter_start_date)
        if self.end_date is not None:
            filter_end_date = min(self.end_date.tz_localize(filter_end_date.tz), filter_end_date)
        holiday_dates = holiday_dates[(holiday_dates >= filter_start_date) & (holiday_dates <= filter_end_date)]
        if return_name:
            return Series(self.name, index=holiday_dates)
        return holiday_dates

    def _reference_dates(self, start_date: Timestamp, end_date: Timestamp) -> DatetimeIndex:
        """
        Get reference dates for the holiday.

        Return reference dates for the holiday also returning the year
        prior to the start_date and year following the end_date.  This ensures
        that any offsets to be applied will yield the holidays within
        the passed in dates.
        """
        if self.start_date is not None:
            start_date = self.start_date.tz_localize(start_date.tz)
        if self.end_date is not None:
            end_date = self.end_date.tz_localize(start_date.tz)
        year_offset = DateOffset(years=1)
        reference_start_date = Timestamp(datetime(start_date.year - 1, self.month, self.day))
        reference_end_date = Timestamp(datetime(end_date.year + 1, self.month, self.day))
        dates = date_range(start=reference_start_date, end=reference_end_date, freq=year_offset, tz=start_date.tz)
        return dates

    def _apply_rule(self, dates: DatetimeIndex) -> DatetimeIndex:
        """
        Apply the given offset/observance to a DatetimeIndex of dates.

        Parameters
        ----------
        dates : DatetimeIndex
            Dates to apply the given offset/observance rule

        Returns
        -------
        Dates with rules applied
        """
        if dates.empty:
            return dates.copy()
        if self.observance is not None:
            return dates.map(lambda d: self.observance(d))
        if self.offset is not None:
            if not isinstance(self.offset, list):
                offsets = [self.offset]
            else:
                offsets = self.offset
            for offset in offsets:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', PerformanceWarning)
                    dates += offset
        return dates

holiday_calendars: Dict[str, Any] = {}

def register(cls: type) -> None:
    try:
        name = cls.name
    except AttributeError:
        name = cls.__name__
    holiday_calendars[name] = cls

def get_calendar(name: str) -> Any:
    """
    Return an instance of a calendar based on its name.

    Parameters
    ----------
    name : str
        Calendar name to return an instance of
    """
    return holiday_calendars[name]()

class HolidayCalendarMetaClass(type):
    def __new__(
        cls,
        clsname: str,
        bases: Tuple[type, ...],
        attrs: Dict[str, Any]
    ) -> HolidayCalendarMetaClass:
        calendar_class = super().__new__(cls, clsname, bases, attrs)
        register(calendar_class)
        return calendar_class

class AbstractHolidayCalendar(metaclass=HolidayCalendarMetaClass):
    """
    Abstract interface to create holidays following certain rules.
    """
    rules: List[Holiday] = []
    start_date: Timestamp = Timestamp(datetime(1970, 1, 1))
    end_date: Timestamp = Timestamp(datetime(2200, 12, 31))
    _cache: Optional[Tuple[Timestamp, Timestamp, Series]] = None

    def __init__(self, name: str = '', rules: Optional[List[Holiday]] = None) -> None:
        """
        Initializes holiday object with a given set a rules.  Normally
        classes just have the rules defined within them.

        Parameters
        ----------
        name : str
            Name of the holiday calendar, defaults to class name
        rules : array of Holiday objects
            A set of rules used to create the holidays.
        """
        super().__init__()
        if not name:
            name = type(self).__name__
        self.name = name
        if rules is not None:
            self.rules = rules

    def rule_from_name(self, name: str) -> Optional[Holiday]:
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None

    def holidays(
        self,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        return_name: bool = False
    ) -> Union[DatetimeIndex, Series]:
        """
        Returns a curve with holidays between start_date and end_date

        Parameters
        ----------
        start : starting date, datetime-like, optional
        end : ending date, datetime-like, optional
        return_name : bool, optional
            If True, return a series that has dates and holiday names.
            False will only return a DatetimeIndex of dates.

        Returns
        -------
            DatetimeIndex of holidays
        """
        if self.rules is None:
            raise Exception(f'Holiday Calendar {self.name} does not have any rules specified')
        if start is None:
            start = AbstractHolidayCalendar.start_date
        if end is None:
            end = AbstractHolidayCalendar.end_date
        start = Timestamp(start)
        end = Timestamp(end)
        if self._cache is None or start < self._cache[0] or end > self._cache[1]:
            pre_holidays = [rule.dates(start, end, return_name=True) for rule in self.rules]
            if pre_holidays:
                holidays = concat(pre_holidays)
            else:
                holidays = Series(index=DatetimeIndex([]), dtype=object)
            self._cache = (start, end, holidays.sort_index())
        holidays = self._cache[2]
        holidays = holidays[start:end]
        if return_name:
            return holidays
        else:
            return holidays.index

    @staticmethod
    def merge_class(
        base: Union[AbstractHolidayCalendar, List[Holiday]],
        other: Union[AbstractHolidayCalendar, List[Holiday]]
    ) -> List[Holiday]:
        """
        Merge holiday calendars together. The base calendar
        will take precedence to other. The merge will be done
        based on each holiday's name.

        Parameters
        ----------
        base : AbstractHolidayCalendar
          instance/subclass or array of Holiday objects
        other : AbstractHolidayCalendar
          instance/subclass or array of Holiday objects
        """
        try:
            other = other.rules
        except AttributeError:
            pass
        if not isinstance(other, list):
            other = [other]
        other_holidays = {holiday.name: holiday for holiday in other}
        try:
            base = base.rules
        except AttributeError:
            pass
        if not isinstance(base, list):
            base = [base]
        base_holidays = {holiday.name: holiday for holiday in base}
        other_holidays.update(base_holidays)
        return list(other_holidays.values())

    def merge(
        self,
        other: Union[AbstractHolidayCalendar, List[Holiday]],
        inplace: bool = False
    ) -> Optional[List[Holiday]]:
        """
        Merge holiday calendars together.  The caller's class
        rules take precedence.  The merge will be done
        based on each holiday's name.

        Parameters
        ----------
        other : holiday calendar
        inplace : bool (default=False)
            If True set rule_table to holidays, else return array of Holidays
        """
        holidays = self.merge_class(self, other)
        if inplace:
            self.rules = holidays
        else:
            return holidays

USMemorialDay = Holiday('Memorial Day', month=5, day=31, offset=DateOffset(weekday=MO(-1)))
USLaborDay = Holiday('Labor Day', month=9, day=1, offset=DateOffset(weekday=MO(1)))
USColumbusDay = Holiday('Columbus Day', month=10, day=1, offset=DateOffset(weekday=MO(2)))
USThanksgivingDay = Holiday('Thanksgiving Day', month=11, day=1, offset=DateOffset(weekday=TH(4)))
USMartinLutherKingJr = Holiday('Birthday of Martin Luther King, Jr.', start_date=datetime(1986, 1, 1), month=1, day=1, offset=DateOffset(weekday=MO(3)))
USPresidentsDay = Holiday("Washington's Birthday", month=2, day=1, offset=DateOffset(weekday=MO(3)))
GoodFriday = Holiday('Good Friday', month=1, day=1, offset=[Easter(), Day(-2)])
EasterMonday = Holiday('Easter Monday', month=1, day=1, offset=[Easter(), Day(1)])

class USFederalHolidayCalendar(AbstractHolidayCalendar):
    """
    US Federal Government Holiday Calendar based on rules specified by:
    https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/
    """
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        Holiday('Juneteenth National Independence Day', month=6, day=19, start_date='2021-06-18', observance=nearest_workday),
        Holiday('Independence Day', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USColumbusDay,
        Holiday('Veterans Day', month=11, day=11
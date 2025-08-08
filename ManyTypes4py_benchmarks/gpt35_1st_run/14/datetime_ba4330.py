import datetime as dt
import operator as op
import zoneinfo
from calendar import monthrange
from functools import lru_cache, partial
from importlib import resources
from pathlib import Path
from typing import Optional
from hypothesis.errors import InvalidArgument
from hypothesis.internal.validation import check_type, check_valid_interval
from hypothesis.strategies._internal.core import sampled_from
from hypothesis.strategies._internal.misc import just, none, nothing
from hypothesis.strategies._internal.strategies import SearchStrategy
from hypothesis.strategies._internal.utils import defines_strategy

DATENAMES: tuple[str] = ('year', 'month', 'day')
TIMENAMES: tuple[str] = ('hour', 'minute', 'second', 'microsecond')

def is_pytz_timezone(tz: dt.tzinfo) -> bool:
    if not isinstance(tz, dt.tzinfo):
        return False
    module = type(tz).__module__
    return module == 'pytz' or module.startswith('pytz.')

def replace_tzinfo(value: dt.datetime, timezone: dt.tzinfo) -> dt.datetime:
    if is_pytz_timezone(timezone):
        return timezone.localize(value, is_dst=not value.fold)
    return value.replace(tzinfo=timezone)

def datetime_does_not_exist(value: dt.datetime) -> bool:
    if value.tzinfo is None:
        return False
    try:
        roundtrip = value.astimezone(dt.timezone.utc).astimezone(value.tzinfo)
    except OverflowError:
        return True
    if value.tzinfo is not roundtrip.tzinfo and value.utcoffset() != roundtrip.utcoffset():
        return True
    assert value.tzinfo is roundtrip.tzinfo, 'so only the naive portions are compared'
    return value != roundtrip

def draw_capped_multipart(data, min_value: dt.datetime, max_value: dt.datetime, duration_names: tuple[str] = DATENAMES + TIMENAMES) -> dict:
    assert isinstance(min_value, (dt.date, dt.time, dt.datetime))
    assert type(min_value) == type(max_value)
    assert min_value <= max_value
    result = {}
    cap_low, cap_high = (True, True)
    for name in duration_names:
        low = getattr(min_value if cap_low else dt.datetime.min, name)
        high = getattr(max_value if cap_high else dt.datetime.max, name)
        if name == 'day' and (not cap_high):
            _, high = monthrange(**result)
        if name == 'year':
            val = data.draw_integer(low, high, shrink_towards=2000)
        else:
            val = data.draw_integer(low, high)
        result[name] = val
        cap_low = cap_low and val == low
        cap_high = cap_high and val == high
    if hasattr(min_value, 'fold'):
        result['fold'] = data.draw_integer(0, 1)
    return result

class DatetimeStrategy(SearchStrategy):

    def __init__(self, min_value: dt.datetime, max_value: dt.datetime, timezones_strat: SearchStrategy, allow_imaginary: bool):
        assert isinstance(min_value, dt.datetime)
        assert isinstance(max_value, dt.datetime)
        assert min_value.tzinfo is None
        assert max_value.tzinfo is None
        assert min_value <= max_value
        assert isinstance(timezones_strat, SearchStrategy)
        assert isinstance(allow_imaginary, bool)
        self.min_value = min_value
        self.max_value = max_value
        self.tz_strat = timezones_strat
        self.allow_imaginary = allow_imaginary

    def do_draw(self, data):
        tz = data.draw(self.tz_strat)
        result = self.draw_naive_datetime_and_combine(data, tz)
        if not self.allow_imaginary and datetime_does_not_exist(result):
            data.mark_invalid(f'{result} does not exist (usually a DST transition)')
        return result

    def draw_naive_datetime_and_combine(self, data, tz):
        result = draw_capped_multipart(data, self.min_value, self.max_value)
        try:
            return replace_tzinfo(dt.datetime(**result), timezone=tz)
        except (ValueError, OverflowError):
            data.mark_invalid(f'Failed to draw a datetime between {self.min_value!r} and {self.max_value!r} with timezone from {self.tz_strat!r}.')

@defines_strategy(force_reusable_values=True)
def datetimes(min_value: dt.datetime = dt.datetime.min, max_value: dt.datetime = dt.datetime.max, *, timezones: Optional[SearchStrategy] = none(), allow_imaginary: bool = True) -> DatetimeStrategy:
    ...

class TimeStrategy(SearchStrategy):

    def __init__(self, min_value: dt.time, max_value: dt.time, timezones_strat: SearchStrategy):
        ...

    def do_draw(self, data):
        ...

@defines_strategy(force_reusable_values=True)
def times(min_value: dt.time = dt.time.min, max_value: dt.time = dt.time.max, *, timezones: Optional[SearchStrategy] = none()) -> TimeStrategy:
    ...

class DateStrategy(SearchStrategy):

    def __init__(self, min_value: dt.date, max_value: dt.date):
        ...

    def do_draw(self, data):
        ...

    def filter(self, condition):
        ...

@defines_strategy(force_reusable_values=True)
def dates(min_value: dt.date = dt.date.min, max_value: dt.date = dt.date.max) -> DateStrategy:
    ...

class TimedeltaStrategy(SearchStrategy):

    def __init__(self, min_value: dt.timedelta, max_value: dt.timedelta):
        ...

    def do_draw(self, data):
        ...

@defines_strategy(force_reusable_values=True)
def timedeltas(min_value: dt.timedelta = dt.timedelta.min, max_value: dt.timedelta = dt.timedelta.max) -> TimedeltaStrategy:
    ...

@lru_cache(maxsize=None)
def _valid_key_cacheable(tzpath: tuple, key: str) -> bool:
    ...

@defines_strategy(force_reusable_values=True)
def timezone_keys(*, allow_prefix: bool = True) -> SearchStrategy:
    ...

@defines_strategy(force_reusable_values=True)
def timezones(*, no_cache: bool = False) -> SearchStrategy:
    ...

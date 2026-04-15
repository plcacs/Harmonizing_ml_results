import datetime as dt
import operator as op
from calendar import monthrange
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

from hypothesis.errors import InvalidArgument
from hypothesis.internal.validation import check_type, check_valid_interval
from hypothesis.strategies._internal.core import sampled_from
from hypothesis.strategies._internal.misc import just, none, nothing
from hypothesis.strategies._internal.strategies import SearchStrategy
from hypothesis.strategies._internal.utils import defines_strategy

DATENAMES: Tuple[str, str, str] = ...
TIMENAMES: Tuple[str, str, str, str] = ...

def is_pytz_timezone(tz: Any) -> bool: ...

def replace_tzinfo(value: dt.datetime, timezone: Optional[dt.tzinfo]) -> dt.datetime: ...

def datetime_does_not_exist(value: dt.datetime) -> bool: ...

def draw_capped_multipart(
    data: Any,
    min_value: Union[dt.date, dt.time, dt.datetime],
    max_value: Union[dt.date, dt.time, dt.datetime],
    duration_names: Tuple[str, ...] = ...,
) -> dict[str, int]: ...

class DatetimeStrategy(SearchStrategy[dt.datetime]):
    min_value: dt.datetime
    max_value: dt.datetime
    tz_strat: SearchStrategy[Optional[dt.tzinfo]]
    allow_imaginary: bool
    
    def __init__(
        self,
        min_value: dt.datetime,
        max_value: dt.datetime,
        timezones_strat: SearchStrategy[Optional[dt.tzinfo]],
        allow_imaginary: bool,
    ) -> None: ...
    
    def do_draw(self, data: Any) -> dt.datetime: ...
    
    def draw_naive_datetime_and_combine(
        self,
        data: Any,
        tz: Optional[dt.tzinfo],
    ) -> dt.datetime: ...

@defines_strategy(force_reusable_values=True)
def datetimes(
    min_value: dt.datetime = ...,
    max_value: dt.datetime = ...,
    *,
    timezones: SearchStrategy[Optional[dt.tzinfo]] = ...,
    allow_imaginary: bool = True,
) -> SearchStrategy[dt.datetime]: ...

class TimeStrategy(SearchStrategy[dt.time]):
    min_value: dt.time
    max_value: dt.time
    tz_strat: SearchStrategy[Optional[dt.tzinfo]]
    
    def __init__(
        self,
        min_value: dt.time,
        max_value: dt.time,
        timezones_strat: SearchStrategy[Optional[dt.tzinfo]],
    ) -> None: ...
    
    def do_draw(self, data: Any) -> dt.time: ...

@defines_strategy(force_reusable_values=True)
def times(
    min_value: dt.time = ...,
    max_value: dt.time = ...,
    *,
    timezones: SearchStrategy[Optional[dt.tzinfo]] = ...,
) -> SearchStrategy[dt.time]: ...

class DateStrategy(SearchStrategy[dt.date]):
    min_value: dt.date
    max_value: dt.date
    
    def __init__(self, min_value: dt.date, max_value: dt.date) -> None: ...
    
    def do_draw(self, data: Any) -> dt.date: ...
    
    def filter(self, condition: Callable[[dt.date], bool]) -> SearchStrategy[dt.date]: ...

@defines_strategy(force_reusable_values=True)
def dates(
    min_value: dt.date = ...,
    max_value: dt.date = ...,
) -> SearchStrategy[dt.date]: ...

class TimedeltaStrategy(SearchStrategy[dt.timedelta]):
    min_value: dt.timedelta
    max_value: dt.timedelta
    
    def __init__(self, min_value: dt.timedelta, max_value: dt.timedelta) -> None: ...
    
    def do_draw(self, data: Any) -> dt.timedelta: ...

@defines_strategy(force_reusable_values=True)
def timedeltas(
    min_value: dt.timedelta = ...,
    max_value: dt.timedelta = ...,
) -> SearchStrategy[dt.timedelta]: ...

@defines_strategy(force_reusable_values=True)
def timezone_keys(*, allow_prefix: bool = True) -> SearchStrategy[str]: ...

@defines_strategy(force_reusable_values=True)
def timezones(*, no_cache: bool = False) -> SearchStrategy[dt.tzinfo]: ...
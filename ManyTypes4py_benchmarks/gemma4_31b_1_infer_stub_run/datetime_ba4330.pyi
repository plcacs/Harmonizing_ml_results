import datetime as dt
import zoneinfo
from typing import Optional, Union, Any, Sequence, overload
from hypothesis.strategies._internal.strategies import SearchStrategy

DATENAMES: tuple[str, ...] = ...
TIMENAMES: tuple[str, ...] = ...

def is_pytz_timezone(tz: Any) -> bool: ...

def replace_tzinfo(value: dt.datetime, timezone: Optional[dt.tzinfo]) -> dt.datetime: ...

def datetime_does_not_exist(value: dt.datetime) -> bool: ...

def draw_capped_multipart(
    data: Any,
    min_value: Union[dt.date, dt.time, dt.datetime],
    max_value: Union[dt.date, dt.time, dt.datetime],
    duration_names: Sequence[str] = DATENAMES + TIMENAMES,
) -> dict[str, int]: ...

class DatetimeStrategy(SearchStrategy):
    min_value: dt.datetime
    max_value: dt.datetime
    tz_strat: SearchStrategy
    allow_imaginary: bool

    def __init__(
        self,
        min_value: dt.datetime,
        max_value: dt.datetime,
        timezones_strat: SearchStrategy,
        allow_imaginary: bool,
    ) -> None: ...

    def do_draw(self, data: Any) -> dt.datetime: ...

    def draw_naive_datetime_and_combine(self, data: Any, tz: Optional[dt.tzinfo]) -> dt.datetime: ...

def datetimes(
    min_value: dt.datetime = ...,
    max_value: dt.datetime = ...,
    *,
    timezones: SearchStrategy = ...,
    allow_imaginary: bool = True,
) -> DatetimeStrategy: ...

class TimeStrategy(SearchStrategy):
    min_value: dt.time
    max_value: dt.time
    tz_strat: SearchStrategy

    def __init__(self, min_value: dt.time, max_value: dt.time, timezones_strat: SearchStrategy) -> None: ...

    def do_draw(self, data: Any) -> dt.time: ...

def times(
    min_value: dt.time = ...,
    max_value: dt.time = ...,
    *,
    timezones: SearchStrategy = ...,
) -> TimeStrategy: ...

class DateStrategy(SearchStrategy):
    min_value: dt.date
    max_value: dt.date

    def __init__(self, min_value: dt.date, max_value: dt.date) -> None: ...

    def do_draw(self, data: Any) -> dt.date: ...

    def filter(self, condition: Any) -> SearchStrategy: ...

def dates(min_value: dt.date = ..., max_value: dt.date = ...) -> Union[DateStrategy, SearchStrategy]: ...

class TimedeltaStrategy(SearchStrategy):
    min_value: dt.timedelta
    max_value: dt.timedelta

    def __init__(self, min_value: dt.timedelta, max_value: dt.timedelta) -> None: ...

    def do_draw(self, data: Any) -> dt.timedelta: ...

def timedeltas(min_value: dt.timedelta = ..., max_value: dt.timedelta = ...) -> Union[TimedeltaStrategy, SearchStrategy]: ...

def _valid_key_cacheable(tzpath: tuple[str, ...], key: str) -> bool: ...

def timezone_keys(*, allow_prefix: bool = True) -> SearchStrategy: ...

def timezones(*, no_cache: bool = False) -> SearchStrategy: ...
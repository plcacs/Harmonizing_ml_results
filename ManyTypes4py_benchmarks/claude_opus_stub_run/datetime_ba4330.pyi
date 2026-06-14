import datetime as dt
from functools import partial
from typing import Optional, Tuple

from hypothesis.strategies._internal.strategies import SearchStrategy

DATENAMES: Tuple[str, ...]
TIMENAMES: Tuple[str, ...]

def is_pytz_timezone(tz: object) -> bool: ...
def replace_tzinfo(value: dt.datetime, timezone: Optional[dt.tzinfo]) -> dt.datetime: ...
def datetime_does_not_exist(value: dt.datetime) -> bool: ...
def draw_capped_multipart(
    data: object,
    min_value: dt.date | dt.time | dt.datetime,
    max_value: dt.date | dt.time | dt.datetime,
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
    def do_draw(self, data: object) -> dt.datetime: ...
    def draw_naive_datetime_and_combine(
        self, data: object, tz: Optional[dt.tzinfo]
    ) -> dt.datetime: ...

def datetimes(
    min_value: dt.datetime = ...,
    max_value: dt.datetime = ...,
    *,
    timezones: SearchStrategy[Optional[dt.tzinfo]] = ...,
    allow_imaginary: bool = ...,
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
    def do_draw(self, data: object) -> dt.time: ...

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
    def do_draw(self, data: object) -> dt.date: ...
    def filter(self, condition: object) -> SearchStrategy[dt.date]: ...

def dates(
    min_value: dt.date = ...,
    max_value: dt.date = ...,
) -> SearchStrategy[dt.date]: ...

class TimedeltaStrategy(SearchStrategy[dt.timedelta]):
    min_value: dt.timedelta
    max_value: dt.timedelta
    def __init__(self, min_value: dt.timedelta, max_value: dt.timedelta) -> None: ...
    def do_draw(self, data: object) -> dt.timedelta: ...

def timedeltas(
    min_value: dt.timedelta = ...,
    max_value: dt.timedelta = ...,
) -> SearchStrategy[dt.timedelta]: ...

def _valid_key_cacheable(tzpath: tuple[str, ...], key: str) -> bool: ...

def timezone_keys(
    *,
    allow_prefix: bool = ...,
) -> SearchStrategy[str]: ...

def timezones(
    *,
    no_cache: bool = ...,
) -> SearchStrategy[dt.tzinfo]: ...
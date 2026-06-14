import datetime as dt
import zoneinfo
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from hypothesis.strategies._internal.misc import none
from hypothesis.strategies._internal.strategies import SearchStrategy

DATENAMES: Tuple[str, ...]
TIMENAMES: Tuple[str, ...]


def is_pytz_timezone(tz: object) -> bool: ...
def replace_tzinfo(value: dt.datetime, timezone: Optional[dt.tzinfo]) -> dt.datetime: ...
def datetime_does_not_exist(value: dt.datetime) -> bool: ...
def draw_capped_multipart(
    data: Any,
    min_value: Union[dt.date, dt.time, dt.datetime],
    max_value: Union[dt.date, dt.time, dt.datetime],
    duration_names: Iterable[str] = DATENAMES + TIMENAMES,
) -> Dict[str, int]: ...


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
    def draw_naive_datetime_and_combine(self, data: Any, tz: Optional[dt.tzinfo]) -> dt.datetime: ...


def datetimes(
    min_value: dt.datetime = dt.datetime.min,
    max_value: dt.datetime = dt.datetime.max,
    *,
    timezones: SearchStrategy[Optional[dt.tzinfo]] = none(),
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


def times(
    min_value: dt.time = dt.time.min,
    max_value: dt.time = dt.time.max,
    *,
    timezones: SearchStrategy[Optional[dt.tzinfo]] = none(),
) -> SearchStrategy[dt.time]: ...


class DateStrategy(SearchStrategy[dt.date]):
    min_value: dt.date
    max_value: dt.date

    def __init__(self, min_value: dt.date, max_value: dt.date) -> None: ...
    def do_draw(self, data: Any) -> dt.date: ...
    def filter(self, condition: Callable[[dt.date], object]) -> SearchStrategy[dt.date]: ...


def dates(
    min_value: dt.date = dt.date.min,
    max_value: dt.date = dt.date.max,
) -> SearchStrategy[dt.date]: ...


class TimedeltaStrategy(SearchStrategy[dt.timedelta]):
    min_value: dt.timedelta
    max_value: dt.timedelta

    def __init__(self, min_value: dt.timedelta, max_value: dt.timedelta) -> None: ...
    def do_draw(self, data: Any) -> dt.timedelta: ...


def timedeltas(
    min_value: dt.timedelta = dt.timedelta.min,
    max_value: dt.timedelta = dt.timedelta.max,
) -> SearchStrategy[dt.timedelta]: ...


def _valid_key_cacheable(tzpath: Tuple[str, ...], key: str) -> bool: ...
def timezone_keys(*, allow_prefix: bool = True) -> SearchStrategy[str]: ...
def timezones(*, no_cache: bool = False) -> SearchStrategy[zoneinfo.ZoneInfo]: ...
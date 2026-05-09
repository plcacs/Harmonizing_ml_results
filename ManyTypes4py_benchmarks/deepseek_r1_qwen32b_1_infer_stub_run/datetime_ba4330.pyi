import datetime as dt
import operator as op
import zoneinfo
from calendar import monthrange
from functools import lru_cache, partial
from importlib import resources
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from hypothesis.errors import InvalidArgument
from hypothesis.internal.validation import check_type, check_valid_interval
from hypothesis.strategies._internal.core import sampled_from
from hypothesis.strategies._internal.misc import just, none, nothing
from hypothesis.strategies._internal.strategies import SearchStrategy
from hypothesis.strategies._internal.utils import defines_strategy

DATENAMES: Tuple[str, ...] = ('year', 'month', 'day')
TIMENAMES: Tuple[str, ...] = ('hour', 'minute', 'second', 'microsecond')

def is_pytz_timezone(tz: Optional[dt.tzinfo]) -> bool: ...

def replace_tzinfo(value: dt.datetime, timezone: Optional[dt.tzinfo]) -> dt.datetime: ...

def datetime_does_not_exist(value: dt.datetime) -> bool: ...

def draw_capped_multipart(
    data: Any,
    min_value: Union[dt.date, dt.time, dt.datetime],
    max_value: Union[dt.date, dt.time, dt.datetime],
    duration_names: Tuple[str, ...] = DATENAMES + TIMENAMES,
) -> Dict[str, int]: ...

class DatetimeStrategy(SearchStrategy):
    def __init__(
        self,
        min_value: dt.datetime,
        max_value: dt.datetime,
        timezones_strat: SearchStrategy,
        allow_imaginary: bool,
    ) -> None: ...

    def do_draw(self, data: Any) -> dt.datetime: ...

    def draw_naive_datetime_and_combine(self, data: Any, tz: dt.tzinfo) -> dt.datetime: ...

@defines_strategy(force_reusable_values=True)
def datetimes(
    min_value: dt.datetime = dt.datetime.min,
    max_value: dt.datetime = dt.datetime.max,
    *,
    timezones: SearchStrategy = none(),
    allow_imaginary: bool = True,
) -> DatetimeStrategy: ...

class TimeStrategy(SearchStrategy):
    def __init__(self, min_value: dt.time, max_value: dt.time, timezones_strat: SearchStrategy) -> None: ...

    def do_draw(self, data: Any) -> dt.time: ...

@defines_strategy(force_reusable_values=True)
def times(
    min_value: dt.time = dt.time.min,
    max_value: dt.time = dt.time.max,
    *,
    timezones: SearchStrategy = none(),
) -> TimeStrategy: ...

class DateStrategy(SearchStrategy):
    def __init__(self, min_value: dt.date, max_value: dt.date) -> None: ...

    def do_draw(self, data: Any) -> dt.date: ...

    def filter(self, condition: Any) -> SearchStrategy: ...

@defines_strategy(force_reusable_values=True)
def dates(
    min_value: dt.date = dt.date.min,
    max_value: dt.date = dt.date.max,
) -> Union[DateStrategy, SearchStrategy]: ...

class TimedeltaStrategy(SearchStrategy):
    def __init__(self, min_value: dt.timedelta, max_value: dt.timedelta) -> None: ...

    def do_draw(self, data: Any) -> dt.timedelta: ...

@defines_strategy(force_reusable_values=True)
def timedeltas(
    min_value: dt.timedelta = dt.timedelta.min,
    max_value: dt.timedelta = dt.timedelta.max,
) -> TimedeltaStrategy: ...

@lru_cache(maxsize=None)
def _valid_key_cacheable(tzpath: Tuple[str, ...], key: str) -> bool: ...

@defines_strategy(force_reusable_values=True)
def timezone_keys(*, allow_prefix: bool = True) -> SearchStrategy: ...

@defines_strategy(force_reusable_values=True)
def timezones(*, no_cache: bool = False) -> SearchStrategy: ...
```python
import datetime as dt
from typing import Any, Optional, Union
from hypothesis.strategies._internal.strategies import SearchStrategy

DATENAMES: tuple[str, str, str] = ...
TIMENAMES: tuple[str, str, str, str] = ...

def is_pytz_timezone(tz: Any) -> bool: ...

def replace_tzinfo(value: Any, timezone: Any) -> Any: ...

def datetime_does_not_exist(value: dt.datetime) -> bool: ...

def draw_capped_multipart(
    data: Any,
    min_value: Union[dt.date, dt.time, dt.datetime],
    max_value: Union[dt.date, dt.time, dt.datetime],
    duration_names: tuple[str, ...] = ...
) -> dict[str, Any]: ...

class DatetimeStrategy(SearchStrategy[dt.datetime]):
    min_value: dt.datetime
    max_value: dt.datetime
    tz_strat: SearchStrategy
    allow_imaginary: bool
    
    def __init__(
        self,
        min_value: dt.datetime,
        max_value: dt.datetime,
        timezones_strat: SearchStrategy,
        allow_imaginary: bool
    ) -> None: ...
    
    def do_draw(self, data: Any) -> dt.datetime: ...
    
    def draw_naive_datetime_and_combine(self, data: Any, tz: Any) -> dt.datetime: ...

def datetimes(
    min_value: dt.datetime = ...,
    max_value: dt.datetime = ...,
    *,
    timezones: SearchStrategy = ...,
    allow_imaginary: bool = True
) -> SearchStrategy[dt.datetime]: ...

class TimeStrategy(SearchStrategy[dt.time]):
    min_value: dt.time
    max_value: dt.time
    tz_strat: SearchStrategy
    
    def __init__(
        self,
        min_value: dt.time,
        max_value: dt.time,
        timezones_strat: SearchStrategy
    ) -> None: ...
    
    def do_draw(self, data: Any) -> dt.time: ...

def times(
    min_value: dt.time = ...,
    max_value: dt.time = ...,
    *,
    timezones: SearchStrategy = ...
) -> SearchStrategy[dt.time]: ...

class DateStrategy(SearchStrategy[dt.date]):
    min_value: dt.date
    max_value: dt.date
    
    def __init__(self, min_value: dt.date, max_value: dt.date) -> None: ...
    
    def do_draw(self, data: Any) -> dt.date: ...
    
    def filter(self, condition: Any) -> SearchStrategy[dt.date]: ...

def dates(
    min_value: dt.date = ...,
    max_value: dt.date = ...
) -> SearchStrategy[dt.date]: ...

class TimedeltaStrategy(SearchStrategy[dt.timedelta]):
    min_value: dt.timedelta
    max_value: dt.timedelta
    
    def __init__(self, min_value: dt.timedelta, max_value: dt.timedelta) -> None: ...
    
    def do_draw(self, data: Any) -> dt.timedelta: ...

def timedeltas(
    min_value: dt.timedelta = ...,
    max_value: dt.timedelta = ...
) -> SearchStrategy[dt.timedelta]: ...

def timezone_keys(*, allow_prefix: bool = True) -> SearchStrategy[str]: ...

def timezones(*, no_cache: bool = False) -> SearchStrategy[Any]: ...
```
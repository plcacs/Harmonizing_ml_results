import datetime as dt
import operator as op
import zoneinfo
from calendar import monthrange
from functools import lru_cache, partial
from importlib import resources
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, cast
from hypothesis.errors import InvalidArgument
from hypothesis.internal.validation import check_type, check_valid_interval
from hypothesis.strategies._internal.core import sampled_from
from hypothesis.strategies._internal.misc import just, none, nothing
from hypothesis.strategies._internal.strategies import SearchStrategy, DataObject
from hypothesis.strategies._internal.utils import defines_strategy

DATENAMES: Tuple[str, str, str] = ('year', 'month', 'day')
TIMENAMES: Tuple[str, str, str, str] = ('hour', 'minute', 'second', 'microsecond')

def is_pytz_timezone(tz: Optional[dt.tzinfo]) -> bool:
    if not isinstance(tz, dt.tzinfo):
        return False
    module = type(tz).__module__
    return module == 'pytz' or module.startswith('pytz.')

def replace_tzinfo(value: dt.datetime, timezone: Optional[dt.tzinfo]) -> dt.datetime:
    if is_pytz_timezone(timezone):
        return timezone.localize(value, is_dst=not value.fold)  # type: ignore
    return value.replace(tzinfo=timezone)

def datetime_does_not_exist(value: dt.datetime) -> bool:
    """This function tests whether the given datetime can be round-tripped to and
    from UTC.  It is an exact inverse of (and very similar to) the dateutil method
    https://dateutil.readthedocs.io/en/stable/tz.html#dateutil.tz.datetime_exists
    """
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

def draw_capped_multipart(data: DataObject, min_value: Union[dt.date, dt.time, dt.datetime], max_value: Union[dt.date, dt.time, dt.datetime], duration_names: Tuple[str, ...] = DATENAMES + TIMENAMES) -> Dict[str, Any]:
    assert isinstance(min_value, (dt.date, dt.time, dt.datetime))
    assert type(min_value) == type(max_value)
    assert min_value <= max_value
    result: Dict[str, Any] = {}
    cap_low, cap_high = True, True
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

class DatetimeStrategy(SearchStrategy[dt.datetime]):

    def __init__(self, min_value: dt.datetime, max_value: dt.datetime, timezones_strat: SearchStrategy[Optional[dt.tzinfo]], allow_imaginary: bool):
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

    def do_draw(self, data: DataObject) -> dt.datetime:
        tz = data.draw(self.tz_strat)
        result = self.draw_naive_datetime_and_combine(data, tz)
        if not self.allow_imaginary and datetime_does_not_exist(result):
            data.mark_invalid(f'{result} does not exist (usually a DST transition)')
        return result

    def draw_naive_datetime_and_combine(self, data: DataObject, tz: Optional[dt.tzinfo]) -> dt.datetime:
        result = draw_capped_multipart(data, self.min_value, self.max_value)
        try:
            return replace_tzinfo(dt.datetime(**result), timezone=tz)
        except (ValueError, OverflowError):
            data.mark_invalid(f'Failed to draw a datetime between {self.min_value!r} and {self.max_value!r} with timezone from {self.tz_strat!r}.')

@defines_strategy(force_reusable_values=True)
def datetimes(min_value: dt.datetime = dt.datetime.min, max_value: dt.datetime = dt.datetime.max, *, timezones: SearchStrategy[Optional[dt.tzinfo]] = none(), allow_imaginary: bool = True) -> SearchStrategy[dt.datetime]:
    """datetimes(min_value=datetime.datetime.min, max_value=datetime.datetime.max, *, timezones=none(), allow_imaginary=True)

    A strategy for generating datetimes, which may be timezone-aware.

    This strategy works by drawing a naive datetime between ``min_value``
    and ``max_value``, which must both be naive (have no timezone).

    ``timezones`` must be a strategy that generates either ``None``, for naive
    datetimes, or :class:`~python:datetime.tzinfo` objects for 'aware' datetimes.
    You can construct your own, though we recommend using one of these built-in
    strategies:

    * with the standard library: :func:`hypothesis.strategies.timezones`;
    * with :pypi:`dateutil <python-dateutil>`:
      :func:`hypothesis.extra.dateutil.timezones`; or
    * with :pypi:`pytz`: :func:`hypothesis.extra.pytz.timezones`.

    You may pass ``allow_imaginary=False`` to filter out "imaginary" datetimes
    which did not (or will not) occur due to daylight savings, leap seconds,
    timezone and calendar adjustments, etc.  Imaginary datetimes are allowed
    by default, because malformed timestamps are a common source of bugs.

    Examples from this strategy shrink towards midnight on January 1st 2000,
    local time.
    """
    check_type(bool, allow_imaginary, 'allow_imaginary')
    check_type(dt.datetime, min_value, 'min_value')
    check_type(dt.datetime, max_value, 'max_value')
    if min_value.tzinfo is not None:
        raise InvalidArgument(f'min_value={min_value!r} must not have tzinfo')
    if max_value.tzinfo is not None:
        raise InvalidArgument(f'max_value={max_value!r} must not have tzinfo')
    check_valid_interval(min_value, max_value, 'min_value', 'max_value')
    if not isinstance(timezones, SearchStrategy):
        raise InvalidArgument(f'timezones={timezones!r} must be a SearchStrategy that can provide tzinfo for datetimes (either None or dt.tzinfo objects)')
    return DatetimeStrategy(min_value, max_value, timezones, allow_imaginary)

class TimeStrategy(SearchStrategy[dt.time]):

    def __init__(self, min_value: dt.time, max_value: dt.time, timezones_strat: SearchStrategy[Optional[dt.tzinfo]]):
        self.min_value = min_value
        self.max_value = max_value
        self.tz_strat = timezones_strat

    def do_draw(self, data: DataObject) -> dt.time:
        result = draw_capped_multipart(data, self.min_value, self.max_value, TIMENAMES)
        tz = data.draw(self.tz_strat)
        return dt.time(**result, tzinfo=tz)

@defines_strategy(force_reusable_values=True)
def times(min_value: dt.time = dt.time.min, max_value: dt.time = dt.time.max, *, timezones: SearchStrategy[Optional[dt.tzinfo]] = none()) -> SearchStrategy[dt.time]:
    """times(min_value=datetime.time.min, max_value=datetime.time.max, *, timezones=none())

    A strategy for times between ``min_value`` and ``max_value``.

    The ``timezones`` argument is handled as for :py:func:`datetimes`.

    Examples from this strategy shrink towards midnight, with the timezone
    component shrinking as for the strategy that provided it.
    """
    check_type(dt.time, min_value, 'min_value')
    check_type(dt.time, max_value, 'max_value')
    if min_value.tzinfo is not None:
        raise InvalidArgument(f'min_value={min_value!r} must not have tzinfo')
    if max_value.tzinfo is not None:
        raise InvalidArgument(f'max_value={max_value!r} must not have tzinfo')
    check_valid_interval(min_value, max_value, 'min_value', 'max_value')
    return TimeStrategy(min_value, max_value, timezones)

class DateStrategy(SearchStrategy[dt.date]):

    def __init__(self, min_value: dt.date, max_value: dt.date):
        assert isinstance(min_value, dt.date)
        assert isinstance(max_value, dt.date)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def do_draw(self, data: DataObject) -> dt.date:
        return dt.date(**draw_capped_multipart(data, self.min_value, self.max_value, DATENAMES))

    def filter(self, condition: Any) -> SearchStrategy[dt.date]:
        if isinstance(condition, partial) and len((args := condition.args)) == 1 and (not condition.keywords) and isinstance((arg := args[0]), dt.date) and (condition.func in (op.lt, op.le, op.eq, op.ge, op.gt)):
            try:
                arg += dt.timedelta(days={op.lt: 1, op.gt: -1}.get(condition.func, 0))
            except OverflowError:
                return nothing()
            lo, hi = {op.lt: (arg, self.max_value), op.le: (arg, self.max_value), op.eq: (arg, arg), op.ge: (self.min_value, arg), op.gt: (self.min_value, arg)}[condition.func]
            lo = max(lo, self.min_value)
            hi = min(hi, self.max_value)
            print(lo, hi)
            if hi < lo:
                return nothing()
            if lo <= self.min_value and self.max_value <= hi:
                return self
            return dates(lo, hi)
        return super().filter(condition)

@defines_strategy(force_reusable_values=True)
def dates(min_value: dt.date = dt.date.min, max_value: dt.date = dt.date.max) -> SearchStrategy[dt.date]:
    """dates(min_value=datetime.date.min, max_value=datetime.date.max)

    A strategy for dates between ``min_value`` and ``max_value``.

    Examples from this strategy shrink towards January 1st 2000.
    """
    check_type(dt.date, min_value, 'min_value')
    check_type(dt.date, max_value, 'max_value')
    check_valid_interval(min_value, max_value, 'min_value', 'max_value')
    if min_value == max_value:
        return just(min_value)
    return DateStrategy(min_value, max_value)

class TimedeltaStrategy(SearchStrategy[dt.timedelta]):

    def __init__(self, min_value: dt.timedelta, max_value: dt.timedelta):
        assert isinstance(min_value, dt.timedelta)
        assert isinstance(max_value, dt.timedelta)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def do_draw(self, data: DataObject) -> dt.timedelta:
        result: Dict[str, int] = {}
        low_bound = True
        high_bound = True
        for name in ('days', 'seconds', 'microseconds'):
            low = getattr(self.min_value if low_bound else dt.timedelta.min, name)
            high = getattr(self.max_value if high_bound else dt.timedelta.max, name)
            val = data.draw_integer(low, high)
            result[name] = val
            low_bound = low_bound and val == low
            high_bound = high_bound and val == high
        return dt.timedelta(**result)

@defines_strategy(force_reusable_values=True)
def timedeltas(min_value: dt.timedelta = dt.timedelta.min, max_value: dt.timedelta = dt.timedelta.max) -> SearchStrategy[dt.timedelta]:
    """timedeltas(min_value=datetime.timedelta.min, max_value=datetime.timedelta.max)

    A strategy for timedeltas between ``min_value`` and ``max_value``.

    Examples from this strategy shrink towards zero.
    """
    check_type(dt.timedelta, min_value, 'min_value')
    check_type(dt.timedelta, max_value, 'max_value')
    check_valid_interval(min_value, max_value, 'min_value', 'max_value')
    if min_value == max_value:
        return just(min_value)
    return TimedeltaStrategy(min_value=min_value, max_value=max_value)

@lru_cache(maxsize=None)
def _valid_key_cacheable(tzpath: Tuple[str, ...], key: str) -> bool:
    assert isinstance(tzpath, tuple)
    for root in tzpath:
        if Path(root).joinpath(key).exists():
            return True
    else:
        package_loc, resource_name = key.split('/')[:-1], key.split('/')[-1]
        package = 'tzdata.zoneinfo.' + '.'.join(package_loc)
        try:
            return (resources.files(package) / resource_name).exists()
        except ModuleNotFoundError:
            return False

@defines_strategy(force_reusable_values=True)
def timezone_keys(*, allow_prefix: bool = True) -> SearchStrategy[str]:
    """A strategy for :wikipedia:`IANA timezone names <List_of_tz_database_time_zones>`.

    As well as timezone names like ``"UTC"``, ``"Australia/Sydney"``, or
    ``"America/New_York"``, this strategy can generate:

    - Aliases such as ``"Antarctica/McMurdo"``, which links to ``"Pacific/Auckland"``.
    - Deprecated names such as ``"Antarctica/South_Pole"``, which *also* links to
      ``"Pacific/Auckland"``.  Note that most but
      not all deprecated timezone names are also aliases.
    - Timezone names with the ``"posix/"`` or ``"right/"`` prefixes, unless
      ``allow_prefix=False``.

    These strings are provided separately from Tzinfo objects - such as ZoneInfo
    instances from the timezones() strategy - to facilitate testing of timezone
    logic without needing workarounds to access non-canonical names.

    .. note::

        `The tzdata package is required on Windows
        <https://docs.python.org/3/library/zoneinfo.html#data-sources>`__.
        ``pip install hypothesis[zoneinfo]`` installs it, if and only if needed.

    On Windows, you may need to access IANA timezone data via the :pypi:`tzdata`
    package.  For non-IANA timezones, such as Windows-native names or GNU TZ
    strings, we recommend using :func:`~hypothesis.strategies.sampled_from` with
    the :pypi:`dateutil <python-dateutil>` package, e.g.
    :meth:`dateutil:dateutil.tz.tzwin.list`.
    """
    check_type(bool, allow_prefix, 'allow_prefix')
    available_timezones = ('UTC', *sorted(zoneinfo.available_timezones()))

    def valid_key(key: str) -> bool:
        return key == 'UTC' or _valid_key_cacheable(zoneinfo.TZPATH, key)
    strategy = sampled_from([key for key in available_timezones if valid_key(key)])
    if not allow_prefix:
        return strategy

    def sample_with_prefixes(zone: str) -> SearchStrategy[str]:
        keys_with_prefixes = (zone, f'posix/{zone}', f'right/{zone}')
        return sampled_from([key for key in keys_with_prefixes if valid_key(key)])
    return strategy.flatmap(sample_with_prefixes)

@defines_strategy(force_reusable_values=True)
def timezones(*, no_cache: bool = False) -> SearchStrategy[zoneinfo.ZoneInfo]:
    """A strategy for :class:`python:zoneinfo.ZoneInfo` objects.

    If ``no_cache=True``, the generated instances are constructed using
    :meth:`ZoneInfo.no_cache <python:zoneinfo.ZoneInfo.no_cache>` instead
    of the usual constructor.  This may change the semantics of your datetimes
    in surprising ways, so only use it if you know that you need to!

    .. note::

        `The tzdata package is required on Windows
        <https://docs.python.org/3/library/zoneinfo.html#data-sources>`__.
        ``pip install hypothesis[zoneinfo]`` installs it, if and only if needed.
    """
    check_type(bool, no_cache, 'no_cache')
    return timezone_keys().map(zoneinfo.ZoneInfo.no_cache if no_cache else zoneinfo.ZoneInfo)

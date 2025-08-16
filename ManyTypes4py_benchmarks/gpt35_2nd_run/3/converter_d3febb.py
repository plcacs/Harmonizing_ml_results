from __future__ import annotations
import contextlib
import datetime as pydt
from datetime import datetime, tzinfo
import functools
from typing import TYPE_CHECKING, Any, cast
import warnings
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.units as munits
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import Timestamp, to_offset
from pandas._libs.tslibs.dtypes import FreqGroup, periods_per_day
from pandas._typing import F, npt
from pandas.core.dtypes.common import is_float, is_float_dtype, is_integer, is_integer_dtype, is_nested_list_like
from pandas import Index, Series, get_option
import pandas.core.common as com
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import Period, PeriodIndex, period_range
import pandas.core.tools.datetimes as tools
if TYPE_CHECKING:
    from collections.abc import Generator
    from matplotlib.axis import Axis
    from pandas._libs.tslibs.offsets import BaseOffset
_mpl_units: dict = {}

def get_pairs() -> list[tuple[type, type]]:
    pairs = [(Timestamp, DatetimeConverter), (Period, PeriodConverter), (pydt.datetime, DatetimeConverter), (pydt.date, DatetimeConverter), (pydt.time, TimeConverter), (np.datetime64, DatetimeConverter)]
    return pairs

def register_pandas_matplotlib_converters(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with pandas_converters():
            return func(*args, **kwargs)
    return cast(F, wrapper)

@contextlib.contextmanager
def pandas_converters():
    value = get_option('plotting.matplotlib.register_converters')
    if value:
        register()
    try:
        yield
    finally:
        if value == 'auto':
            deregister()

def register():
    pairs = get_pairs()
    for type_, cls in pairs:
        if type_ in munits.registry and (not isinstance(munits.registry[type_], cls)):
            previous = munits.registry[type_]
            _mpl_units[type_] = previous
        munits.registry[type_] = cls()

def deregister():
    for type_, cls in get_pairs():
        if type(munits.registry.get(type_)) is cls:
            munits.registry.pop(type_)
    for unit, formatter in _mpl_units.items():
        if type(formatter) not in {DatetimeConverter, PeriodConverter, TimeConverter}:
            munits.registry[unit] = formatter

def _to_ordinalf(tm: datetime) -> float:
    tot_sec = tm.hour * 3600 + tm.minute * 60 + tm.second + tm.microsecond / 10 ** 6
    return tot_sec

def time2num(d: Any) -> Any:
    if isinstance(d, str):
        parsed = Timestamp(d)
        return _to_ordinalf(parsed.time())
    if isinstance(d, pydt.time):
        return _to_ordinalf(d)
    return d

class TimeConverter(munits.ConversionInterface):

    @staticmethod
    def convert(value: Any, unit: str, axis: mpl.axis.Axis) -> Any:
        valid_types = (str, pydt.time)
        if isinstance(value, valid_types) or is_integer(value) or is_float(value):
            return time2num(value)
        if isinstance(value, Index):
            return value.map(time2num)
        if isinstance(value, (list, tuple, np.ndarray, Index)):
            return [time2num(x) for x in value]
        return value

    @staticmethod
    def axisinfo(unit: str, axis: mpl.axis.Axis) -> munits.AxisInfo:
        if unit != 'time':
            return None
        majloc = mpl.ticker.AutoLocator()
        majfmt = TimeFormatter(majloc)
        return munits.AxisInfo(majloc=majloc, majfmt=majfmt, label='time')

    @staticmethod
    def default_units(x: Any, axis: mpl.axis.Axis) -> str:
        return 'time'

class TimeFormatter(mpl.ticker.Formatter):

    def __init__(self, locs: mpl.ticker.Locator):
        self.locs = locs

    def __call__(self, x: float, pos: int = 0) -> str:
        fmt = '%H:%M:%S.%f'
        s = int(x)
        msus = round((x - s) * 10 ** 6)
        ms = msus // 1000
        us = msus % 1000
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        _, h = divmod(h, 24)
        if us != 0:
            return pydt.time(h, m, s, msus).strftime(fmt)
        elif ms != 0:
            return pydt.time(h, m, s, msus).strftime(fmt)[:-3]
        elif s != 0:
            return pydt.time(h, m, s).strftime('%H:%M:%S')
        return pydt.time(h, m).strftime('%H:%M')

class PeriodConverter(mdates.DateConverter):

    @staticmethod
    def convert(values: Any, units: str, axis: mpl.axis.Axis) -> Any:
        if is_nested_list_like(values):
            values = [PeriodConverter._convert_1d(v, units, axis) for v in values]
        else:
            values = PeriodConverter._convert_1d(values, units, axis)
        return values

    @staticmethod
    def _convert_1d(values: Any, units: str, axis: mpl.axis.Axis) -> Any:
        if not hasattr(axis, 'freq'):
            raise TypeError('Axis must have `freq` set to convert to Periods')
        valid_types = (str, datetime, Period, pydt.date, pydt.time, np.datetime64)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Period with BDay freq is deprecated', category=FutureWarning)
            warnings.filterwarnings('ignore', 'PeriodDtype\\[B\\] is deprecated', category=FutureWarning)
            if isinstance(values, valid_types) or is_integer(values) or is_float(values):
                return get_datevalue(values, axis.freq)
            elif isinstance(values, PeriodIndex):
                return values.asfreq(axis.freq).asi8
            elif isinstance(values, Index):
                return values.map(lambda x: get_datevalue(x, axis.freq))
            elif lib.infer_dtype(values, skipna=False) == 'period':
                return PeriodIndex(values, freq=axis.freq).asi8
            elif isinstance(values, (list, tuple, np.ndarray, Index)):
                return [get_datevalue(x, axis.freq) for x in values]
        return values

def get_datevalue(date: Any, freq: Any) -> Any:
    if isinstance(date, Period):
        return date.asfreq(freq).ordinal
    elif isinstance(date, (str, datetime, pydt.date, pydt.time, np.datetime64)):
        return Period(date, freq).ordinal
    elif is_integer(date) or is_float(date) or (isinstance(date, (np.ndarray, Index)) and date.size == 1):
        return date
    elif date is None:
        return None
    raise ValueError(f"Unrecognizable date '{date}'")

class DatetimeConverter(mdates.DateConverter):

    @staticmethod
    def convert(values: Any, unit: str, axis: mpl.axis.Axis) -> Any:
        if is_nested_list_like(values):
            values = [DatetimeConverter._convert_1d(v, unit, axis) for v in values]
        else:
            values = DatetimeConverter._convert_1d(values, unit, axis)
        return values

    @staticmethod
    def _convert_1d(values: Any, unit: str, axis: mpl.axis.Axis) -> Any:

        def try_parse(values: Any) -> Any:
            try:
                return mdates.date2num(tools.to_datetime(values))
            except Exception:
                return values
        if isinstance(values, (datetime, pydt.date, np.datetime64, pydt.time)):
            return mdates.date2num(values)
        elif is_integer(values) or is_float(values):
            return values
        elif isinstance(values, str):
            return try_parse(values)
        elif isinstance(values, (list, tuple, np.ndarray, Index, Series)):
            if isinstance(values, Series):
                values = Index(values)
            if isinstance(values, Index):
                values = values.values
            if not isinstance(values, np.ndarray):
                values = com.asarray_tuplesafe(values)
            if is_integer_dtype(values) or is_float_dtype(values):
                return values
            try:
                values = tools.to_datetime(values)
            except Exception:
                pass
            values = mdates.date2num(values)
        return values

    @staticmethod
    def axisinfo(unit: str, axis: mpl.axis.Axis) -> munits.AxisInfo:
        tz = unit
        majloc = PandasAutoDateLocator(tz=tz)
        majfmt = PandasAutoDateFormatter(majloc, tz=tz)
        datemin = pydt.date(2000, 1, 1)
        datemax = pydt.date(2010, 1, 1)
        return munits.AxisInfo(majloc=majloc, majfmt=majfmt, label='', default_limits=(datemin, datemax))

class PandasAutoDateFormatter(mdates.AutoDateFormatter):

    def __init__(self, locator: mpl.ticker.Locator, tz: Any = None, defaultfmt: str = '%Y-%m-%d'):
        mdates.AutoDateFormatter.__init__(self, locator, tz, defaultfmt)

class PandasAutoDateLocator(mdates.AutoDateLocator):

    def get_locator(self, dmin: datetime, dmax: datetime) -> mpl.ticker.Locator:
        tot_sec = (dmax - dmin).total_seconds()
        if abs(tot_sec) < self.minticks:
            self._freq = -1
            locator = MilliSecondLocator(self.tz)
            locator.set_axis(self.axis)
            locator.axis.set_view_interval(*self.axis.get_view_interval())
            locator.axis.set_data_interval(*self.axis.get_data_interval())
            return locator
        return mdates.AutoDateLocator.get_locator(self, dmin, dmax)

    def _get_unit(self) -> float:
        return MilliSecondLocator.get_unit_generic(self._freq)

class MilliSecondLocator(mdates.DateLocator):
    UNIT: float = 1.0 / (24 * 3600 * 1000)

    def __init__(self, tz: Any):
        mdates.DateLocator.__init__(self, tz)
        self._interval = 1.0

    def _get_unit(self) -> float:
        return self.get_unit_generic(-1)

    @staticmethod
    def get_unit_generic(freq: int) -> float:
        unit = mdates.RRuleLocator.get_unit_generic(freq)
        if unit < 0:
            return MilliSecondLocator.UNIT
        return unit

    def __call__(self) -> list[float]:
        try:
            dmin, dmax = self.viewlim_to_dt()
        except ValueError:
            return []
        nmax, nmin = mdates.date2num((dmax, dmin))
        num = (nmax - nmin) * 86400 * 1000
        max_millis_ticks = 6
        for interval in [1, 10, 50, 100, 200, 500]:
            if num <= interval * (max_millis_ticks - 1):
                self._interval = interval
                break
            self._interval = 1000.0
        estimate = (nmax - nmin) / (self._get_unit() * self._get_interval())
        if estimate > self.MAXTICKS * 2:
            raise RuntimeError(f'MillisecondLocator estimated to generate {estimate:d} ticks from {dmin} to {dmax}: exceeds Locator.MAXTICKS* 2 ({self.MAXTICKS * 2:d}) ')
        interval = self._get_interval()
        freq = f'{interval}ms'
        tz = self.tz.tzname(None)
        st = dmin.replace(tzinfo=None)
        ed = dmax.replace(tzinfo=None)
        all_dates = date_range(start=st, end=ed, freq=freq, tz=tz).astype(object)
        try:
            if len(all_dates) > 0:
                locs = self.raise_if_exceeds(mdates.date2num(all_dates))
                return locs
        except Exception:
            pass
        lims = mdates.date2num([dmin, dmax])
        return lims

    def _get_interval(self) -> float:
        return self._interval

    def autoscale(self) -> tuple[float, float]:
        dmin, dmax = self.datalim_to_dt()
        vmin = mdates.date2num(dmin)
        vmax = mdates.date2num(dmax)
        return self.nonsingular(vmin, vmax)

def _get_default_annual_spacing(nyears: int) -> tuple[int, int]:
    if nyears < 11:
        min_spacing, maj_spacing = (1, 1)
    elif nyears < 20:
        min_spacing, maj_spacing = (1, 2)
    elif nyears < 50:
        min_spacing, maj_spacing = (1, 5)
    elif nyears < 100:
        min_spacing, maj_spacing = (5, 10)
    elif nyears < 200:
        min_spacing, maj_spacing = (5, 25)
    elif nyears < 600:
        min_spacing, maj_spacing = (10, 50)
    else:
        factor = nyears // 1000 + 1
        min_spacing, maj_spacing = (factor * 20, factor * 100)
    return (min_spacing, maj_spacing)

def _period_break(dates: PeriodIndex, period: str) -> np.ndarray:
    mask = _period_break_mask(dates, period)
    return np.nonzero(mask)[0]

def _period_break_mask(dates: PeriodIndex, period: str) -> np.ndarray:
    current = getattr(dates, period)
    previous = getattr(dates - 1 * dates.freq, period)
    return current != previous

def has_level_label(label_flags: np.ndarray, vmin: float) -> bool:
    if label_flags.size == 0 or (label_flags.size == 1 and label_flags[0] == 0 and (vmin % 1 > 0.0)):
        return False
    else:
        return True

def _get_periods_per_ymd(freq: Any) -> tuple[int, int, int]:
    dtype_code = freq._period_dtype_code
    freq_group = FreqGroup.from_period_dtype_code(dtype_code)
    ppd = -1
    if dtype_code >= FreqGroup.FR_HR.value:
        ppd = periods_per_day(freq._creso)
        ppm = 28 * ppd
        ppy = 365 * ppd
    elif freq_group == FreqGroup.FR_BUS:
        ppm = 19
        ppy = 261
    elif freq_group == FreqGroup.FR_DAY:
        ppm = 28
        ppy = 365
    elif freq_group == FreqGroup.FR_WK:
        ppm = 3
        ppy = 52
    elif freq_group == FreqGroup.FR_MTH:
        ppm = 1
        ppy = 12
    elif freq_group == FreqGroup.FR_QTR:
        ppm = -1
        ppy = 4
    elif freq_group == FreqGroup.FR_ANN:
        ppm = -1
        ppy = 1
    else:
        raise NotImplementedError(f'Unsupported frequency: {dtype_code}')
    return (ppd, ppm, ppy)

@functools.cache
def _daily_finder(vmin: float, vmax: float, freq: Any) -> np.ndarray:
    dtype_code = freq._period_dtype_code
    periodsperday, periodspermonth, periodsperyear = _get_periods_per_ymd(freq)
    vmin_orig = vmin
    vmin, vmax = (int(vmin), int(vmax))
    span = vmax - vmin + 1
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Period with BDay freq is deprecated', category=FutureWarning)
        warnings.filterwarnings('ignore', 'PeriodDtype\\[B\\] is deprecated', category=FutureWarning)
        dates_ = period_range(start=Period(ordinal=vmin, freq=freq), end=Period(ordinal=vmax, freq=freq), freq=freq)
    info = np.zeros(span, dtype=[('val', np.int64), ('maj', bool), ('min', bool), ('fmt', '|S20')])
    info['val'][:] = dates_.asi8
    info['fmt'][:] = ''
    info['maj'][[0, -1]] = True
    info_maj = info['maj']
    info_min = info['min']
    info_fmt = info['fmt']

    def first_label(label_flags: np.ndarray) -> bool:
        if label_flags[0] == 0 and label_flags.size > 1 and (vmin_orig % 1 > 0.0):
            return label_flags[1]
        else:
            return label_flags[0]
    if span <= periodspermonth:
        day_start = _period_break(dates_, 'day')
        month_start = _period_break(dates_, 'month')
        year_start = _period_break(dates_, 'year')

        def _hour_finder(label_interval: int, force_year_start: bool) -> None:
            target = dates_.hour
            mask = _period_break_mask(dates_, 'hour')
            info_maj[day_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = '%H:%M'
            info_fmt[day_start] = '%H:%M\n%d-%b'
            info_fmt[year_start] = '%H:%M\n%d-%b\n%Y'
            if force_year_start and (not has_level_label(year_start, vmin_orig)):
                info_fmt[first_label(day_start)] = '%H:%M\n%d-%b\n%Y'

        def _minute_finder(label_interval: int) -> None:
            target = dates_.minute
            hour_start = _period_break(dates_, 'hour')
            mask = _period_break_mask(dates_, 'minute')
            info_maj[hour_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = '%H:%M'
            info_fmt[day_start] = '%H:%M\n%d-%b'
            info_fmt[year_start] = '%H:%M\n%d-%b\n%Y'

        def _second_finder(label_interval: int) -> None:
            target = dates_.second
           
from __future__ import annotations
import contextlib
import datetime as pydt
from datetime import datetime, tzinfo
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    Callable,
    ContextManager,
    Generator,
    List,
    Tuple,
    Type,
)
import warnings
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.units as munits
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import Timestamp, to_offset
from pandas._libs.tslibs.dtypes import FreqGroup, periods_per_day
from pandas._typing import F, npt
from pandas.core.dtypes.common import (
    is_float,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_nested_list_like,
)
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


def func_nsnpmosf() -> List[Tuple[Type[Any], Type[munits.ConversionInterface]]]:
    pairs: List[Tuple[Type[Any], Type[munits.ConversionInterface]]] = [
        (Timestamp, DatetimeConverter),
        (Period, PeriodConverter),
        (pydt.datetime, DatetimeConverter),
        (pydt.date, DatetimeConverter),
        (pydt.time, TimeConverter),
        (np.datetime64, DatetimeConverter),
    ]
    return pairs


def func_e8bwjllm(func: F) -> F:
    """
    Decorator applying pandas_converters.
    """

    @functools.wraps(func)
    def func_2932xs5c(*args: Any, **kwargs: Any) -> Any:
        with pandas_converters():
            return func(*args, **kwargs)

    return cast(F, func_2932xs5c)


@contextlib.contextmanager
def func_qvapo2k0() -> Generator[None, None, None]:
    """
    Context manager registering pandas' converters for a plot.

    See Also
    --------
    register_pandas_matplotlib_converters : Decorator that applies this.
    """
    value: Any = get_option("plotting.matplotlib.register_converters")
    if value:
        register()
    try:
        yield
    finally:
        if value == "auto":
            deregister()


def func_swriha8l() -> None:
    pairs: List[Tuple[Type[Any], Type[munits.ConversionInterface]]] = func_nsnpmosf()
    for type_, cls in pairs:
        if type_ in munits.registry and not isinstance(munits.registry[type_], cls):
            previous: munits.ConversionInterface = munits.registry[type_]
            _mpl_units[type_] = previous
        munits.registry[type_] = cls()


def func_m7j2fvrr() -> None:
    for type_, cls in func_nsnpmosf():
        if type(munits.registry.get(type_)) is cls:
            munits.registry.pop(type_)
    for unit, formatter in _mpl_units.items():
        if type(formatter) not in {DatetimeConverter, PeriodConverter, TimeConverter}:
            munits.registry[unit] = formatter


def func_w4sj649z(tm: pydt.time) -> float:
    tot_sec: float = (
        tm.hour * 3600
        + tm.minute * 60
        + tm.second
        + tm.microsecond / 10 ** 6
    )
    return tot_sec


def func_psz74bbu(d: Any) -> float | Any:
    if isinstance(d, str):
        parsed: Timestamp = Timestamp(d)
        return func_w4sj649z(parsed.time())
    if isinstance(d, pydt.time):
        return func_w4sj649z(d)
    return d


class TimeConverter(munits.ConversionInterface):
    @staticmethod
    def func_e053ttco(value: Any, unit: Any, axis: Any) -> Any:
        valid_types: Tuple[type, ...] = (str, pydt.time)
        if isinstance(value, valid_types) or is_integer(value) or is_float(value):
            return func_psz74bbu(value)
        if isinstance(value, Index):
            return value.map(time2num)
        if isinstance(value, (list, tuple, np.ndarray, Index)):
            return [func_psz74bbu(x) for x in value]
        return value

    @staticmethod
    def func_v5jogtm1(unit: Any, axis: Any) -> munits.AxisInfo | None:
        if unit != "time":
            return None
        majloc: mpl.ticker.AutoLocator = mpl.ticker.AutoLocator()
        majfmt: TimeFormatter = TimeFormatter(majloc)
        return munits.AxisInfo(
            majloc=majloc, majfmt=majfmt, label="time"
        )

    @staticmethod
    def func_707exffp(x: Any, axis: Any) -> str:
        return "time"


class TimeFormatter(mpl.ticker.Formatter):
    def __init__(self, locs: Any) -> None:
        self.locs = locs

    def __call__(self, x: float, pos: int = 0) -> str:
        """
        Return the time of day as a formatted string.

        Parameters
        ----------
        x : float
            The time of day specified as seconds since 00:00 (midnight),
            with up to microsecond precision.
        pos
            Unused

        Returns
        -------
        str
            A string in HH:MM:SS.mmmuuu format. Microseconds,
            milliseconds and seconds are only displayed if non-zero.
        """
        fmt: str = "%H:%M:%S.%f"
        s: int = int(x)
        msus: int = round((x - s) * 10**6)
        ms: int = msus // 1000
        us: int = msus % 1000
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        _, h = divmod(h, 24)
        if us != 0:
            return pydt.time(h, m, s, msus).strftime(fmt)
        elif ms != 0:
            return pydt.time(h, m, s, msus).strftime(fmt)[:-3]
        elif s != 0:
            return pydt.time(h, m, s).strftime("%H:%M:%S")
        return pydt.time(h, m).strftime("%H:%M")


class PeriodConverter(mdates.DateConverter):
    @staticmethod
    def func_e053ttco(
        values: Any, units: Any, axis: Any
    ) -> Any:
        if is_nested_list_like(values):
            values = [
                PeriodConverter._convert_1d(v, units, axis) for v in values
            ]
        else:
            values = PeriodConverter._convert_1d(values, units, axis)
        return values

    @staticmethod
    def func_wysz6rgv(values: Any, units: Any, axis: Any) -> Any:
        if not hasattr(axis, "freq"):
            raise TypeError(
                "Axis must have `freq` set to convert to Periods"
            )
        valid_types: Tuple[type, ...] = (
            str,
            datetime,
            Period,
            pydt.date,
            pydt.time,
            np.datetime64,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Period with BDay freq is deprecated",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                "PeriodDtype\\[B\\] is deprecated",
                category=FutureWarning,
            )
            if isinstance(values, valid_types) or is_integer(values) or is_float(
                values
            ):
                return get_datevalue(values, axis.freq)
            elif isinstance(values, PeriodIndex):
                return values.asfreq(axis.freq).asi8
            elif isinstance(values, Index):
                return values.map(lambda x: get_datevalue(x, axis.freq))
            elif lib.infer_dtype(values, skipna=False) == "period":
                return PeriodIndex(values, freq=axis.freq).asi8
            elif isinstance(values, (list, tuple, np.ndarray, Index)):
                return [get_datevalue(x, axis.freq) for x in values]
        return values


def func_l347n8ga(date: Any, freq: Any) -> Any:
    if isinstance(date, Period):
        return date.asfreq(freq).ordinal
    elif isinstance(date, (str, datetime, pydt.date, pydt.time, np.datetime64)):
        return Period(date, freq).ordinal
    elif is_integer(date) or is_float(date) or (
        isinstance(date, (np.ndarray, Index)) and date.size == 1
    ):
        return date
    elif date is None:
        return None
    raise ValueError(f"Unrecognizable date '{date}'")


class DatetimeConverter(mdates.DateConverter):
    @staticmethod
    def func_e053ttco(
        values: Any, unit: Any, axis: Any
    ) -> Any:
        if is_nested_list_like(values):
            values = [
                DatetimeConverter._convert_1d(v, unit, axis) for v in values
            ]
        else:
            values = DatetimeConverter._convert_1d(values, unit, axis)
        return values

    @staticmethod
    def func_wysz6rgv(values: Any, unit: Any, axis: Any) -> Any:

        def func_r6rnzgqg(values_inner: Any) -> Any:
            try:
                return mdates.date2num(tools.to_datetime(values_inner))
            except Exception:
                return values_inner

        if isinstance(
            values, (datetime, pydt.date, np.datetime64, pydt.time)
        ):
            return mdates.date2num(values)
        elif is_integer(values) or is_float(values):
            return values
        elif isinstance(values, str):
            return func_r6rnzgqg(values)
        elif isinstance(
            values, (list, tuple, np.ndarray, Index, Series)
        ):
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
    def func_v5jogtm1(unit: Any, axis: Any) -> munits.AxisInfo:
        """
        Return the :class:`~matplotlib.units.AxisInfo` for *unit*.

        *unit* is a tzinfo instance or None.
        The *axis* argument is required but not used.
        """
        tz: tzinfo | None = unit
        majloc: PandasAutoDateLocator = PandasAutoDateLocator(tz=tz)
        majfmt: PandasAutoDateFormatter = PandasAutoDateFormatter(
            majloc, tz=tz
        )
        datemin: pydt.date = pydt.date(2000, 1, 1)
        datemax: pydt.date = pydt.date(2010, 1, 1)
        return munits.AxisInfo(
            majloc=majloc,
            majfmt=majfmt,
            label="",
            default_limits=(datemin, datemax),
        )


class PandasAutoDateFormatter(mdates.AutoDateFormatter):
    def __init__(
        self,
        locator: mdates.AutoDateLocator,
        tz: tzinfo | None = None,
        defaultfmt: str = "%Y-%m-%d",
    ) -> None:
        super().__init__(locator, tz, defaultfmt)


class PandasAutoDateLocator(mdates.AutoDateLocator):
    def func_i2v28etc(self, dmin: datetime, dmax: datetime) -> mdates.DateLocator:
        """Pick the best locator based on a distance."""
        tot_sec: float = (dmax - dmin).total_seconds()
        if abs(tot_sec) < self.minticks:
            self._freq = -1
            locator: MilliSecondLocator = MilliSecondLocator(self.tz)
            locator.set_axis(self.axis)
            locator.axis.set_view_interval(*self.axis.get_view_interval())
            locator.axis.set_data_interval(*self.axis.get_data_interval())
            return locator
        return mdates.AutoDateLocator.get_locator(self, dmin, dmax)

    def func_6gpp0pjy(self) -> float:
        return MilliSecondLocator.get_unit_generic(self._freq)


class MilliSecondLocator(mdates.DateLocator):
    UNIT: float = 1.0 / (24 * 3600 * 1000)

    def __init__(self, tz: tzinfo | None) -> None:
        super().__init__(tz)
        self._interval: float = 1.0

    def func_6gpp0pjy(self) -> float:
        return self.get_unit_generic(-1)

    @staticmethod
    def func_or2w6pro(freq: Any) -> float:
        unit: float = mdates.RRuleLocator.get_unit_generic(freq)
        if unit < 0:
            return MilliSecondLocator.UNIT
        return unit

    def __call__(self) -> np.ndarray:
        try:
            dmin, dmax = self.viewlim_to_dt()
        except ValueError:
            return np.array([])
        nmax, nmin = mdates.date2num((dmax, dmin))
        num: float = (nmax - nmin) * 86400 * 1000
        max_millis_ticks: int = 6
        for interval in [1, 10, 50, 100, 200, 500]:
            if num <= interval * (max_millis_ticks - 1):
                self._interval = interval
                break
            self._interval = 1000.0
        estimate: float = (nmax - nmin) / (self._get_unit() * self._get_interval())
        if estimate > self.MAXTICKS * 2:
            raise RuntimeError(
                f"MillisecondLocator estimated to generate {int(estimate)} ticks from {dmin} to {dmax}: exceeds Locator.MAXTICKS* 2 ({self.MAXTICKS * 2}) "
            )
        interval: float = self._get_interval()
        freq: str = f"{int(interval)}ms"
        tz: str | None = self.tz.tzname(None) if self.tz else None
        st: datetime = dmin.replace(tzinfo=None)
        ed: datetime = dmax.replace(tzinfo=None)
        all_dates: np.ndarray = date_range(
            start=st, end=ed, freq=freq, tz=tz
        ).astype(object)
        try:
            if len(all_dates) > 0:
                locs: np.ndarray = self.raise_if_exceeds(
                    mdates.date2num(all_dates)
                )
                return locs
        except Exception:
            pass
        lims: np.ndarray = mdates.date2num([dmin, dmax])
        return lims

    def func_bqor3ugn(self) -> float:
        return self._interval

    def func_oywae6cj(self) -> tuple[float, float]:
        """
        Set the view limits to include the data range.
        """
        dmin, dmax = self.datalim_to_dt()
        vmin: float = mdates.date2num(dmin)
        vmax: float = mdates.date2num(dmax)
        return self.nonsingular(vmin, vmax)


def func_m35cyxc1(nyears: float) -> Tuple[int, int]:
    """
    Returns a default spacing between consecutive ticks for annual data.
    """
    if nyears < 11:
        min_spacing, maj_spacing = 1, 1
    elif nyears < 20:
        min_spacing, maj_spacing = 1, 2
    elif nyears < 50:
        min_spacing, maj_spacing = 1, 5
    elif nyears < 100:
        min_spacing, maj_spacing = 5, 10
    elif nyears < 200:
        min_spacing, maj_spacing = 5, 25
    elif nyears < 600:
        min_spacing, maj_spacing = 10, 50
    else:
        factor: int = nyears // 1000 + 1
        min_spacing, maj_spacing = factor * 20, factor * 100
    return min_spacing, maj_spacing


def func_fwjk7zy0(dates: PeriodIndex, period: str) -> np.ndarray:
    """
    Returns the indices where the given period changes.

    Parameters
    ----------
    dates : PeriodIndex
        Array of intervals to monitor.
    period : str
        Name of the period to monitor.
    """
    mask: np.ndarray = _period_break_mask(dates, period)
    return np.nonzero(mask)[0]


def func_yuzk120x(dates: PeriodIndex, period: str) -> np.ndarray:
    current = getattr(dates, period)
    previous = getattr(dates - 1 * dates.freq, period)
    return current != previous


def func_pjswpgcx(label_flags: np.ndarray, vmin: float) -> bool:
    """
    Returns true if the ``label_flags`` indicate there is at least one label
    for this level.

    if the minimum view limit is not an exact integer, then the first tick
    label won't be shown, so we must adjust for that.
    """
    if (
        label_flags.size == 0
        or (label_flags.size == 1 and label_flags[0] == 0 and vmin % 1 > 0.0)
    ):
        return False
    else:
        return True


def func_p3bs53wa(freq: BaseOffset) -> Tuple[int, int, int]:
    dtype_code: int = freq._period_dtype_code
    freq_group: FreqGroup = FreqGroup.from_period_dtype_code(dtype_code)
    ppd: int = -1
    if dtype_code >= FreqGroup.FR_HR.value:
        ppd = periods_per_day(freq._creso)
        ppm: int = 28 * ppd
        ppy: int = 365 * ppd
    elif freq_group == FreqGroup.FR_BUS:
        ppm, ppy = 19, 261
    elif freq_group == FreqGroup.FR_DAY:
        ppm, ppy = 28, 365
    elif freq_group == FreqGroup.FR_WK:
        ppm, ppy = 3, 52
    elif freq_group == FreqGroup.FR_MTH:
        ppm, ppy = 1, 12
    elif freq_group == FreqGroup.FR_QTR:
        ppm, ppy = -1, 4
    elif freq_group == FreqGroup.FR_ANN:
        ppm, ppy = -1, 1
    else:
        raise NotImplementedError(f"Unsupported frequency: {dtype_code}")
    return ppd, ppm, ppy


@functools.cache
def func_1dr5ri72(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    dtype_code: int = freq._period_dtype_code
    periodsperday, periodspermonth, periodsperyear = func_p3bs53wa(freq)
    vmin_orig: float = vmin
    vmin, vmax = int(vmin), int(vmax)
    span: int = vmax - vmin + 1
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "Period with BDay freq is deprecated",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            "PeriodDtype\\[B\\] is deprecated",
            category=FutureWarning,
        )
        dates_: PeriodIndex = period_range(
            start=Period(ordinal=vmin, freq=freq),
            end=Period(ordinal=vmax, freq=freq),
            freq=freq,
        )
    info: np.ndarray = np.zeros(
        span,
        dtype=[
            ("val", np.int64),
            ("maj", bool),
            ("min", bool),
            ("fmt", "|S20"),
        ],
    )
    info["val"][:] = dates_.asi8
    info["fmt"][:] = ""
    info["maj"][[0, -1]] = True
    info_maj: np.ndarray = info["maj"]
    info_min: np.ndarray = info["min"]
    info_fmt: np.ndarray = info["fmt"]

    def func_6wadagu1(label_flags_inner: np.ndarray) -> bool:
        if (
            label_flags_inner[0] == 0
            and label_flags_inner.size > 1
            and vmin_orig % 1 > 0.0
        ):
            return label_flags_inner[1]
        else:
            return label_flags_inner[0]

    if span <= periodspermonth:
        day_start: np.ndarray = func_fwjk7zy0(dates_, "day")
        month_start: np.ndarray = func_fwjk7zy0(dates_, "month")
        year_start: np.ndarray = func_fwjk7zy0(dates_, "year")

        def func_5y0turm6(label_interval: int, force_year_start: bool) -> None:
            target: Any = dates_.hour
            mask: np.ndarray = func_yuzk120x(dates_, "hour")
            info_maj[day_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = "%H:%M"
            info_fmt[day_start] = "%H:%M\n%d-%b"
            info_fmt[year_start] = "%H:%M\n%d-%b\n%Y"
            if force_year_start and not func_pjswpgcx(year_start, vmin_orig):
                info_fmt[func_6wadagu1(day_start)] = "%H:%M\n%d-%b\n%Y"

        def func_nh1sjaj8(label_interval: int) -> None:
            target: Any = dates_.minute
            hour_start: np.ndarray = func_fwjk7zy0(dates_, "hour")
            mask: np.ndarray = func_yuzk120x(dates_, "minute")
            info_maj[hour_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = "%H:%M"
            info_fmt[day_start] = "%H:%M\n%d-%b"
            info_fmt[year_start] = "%H:%M\n%d-%b\n%Y"

        def func_kjh8occr(label_interval: int) -> None:
            target: Any = dates_.second
            minute_start: np.ndarray = func_fwjk7zy0(dates_, "minute")
            mask: np.ndarray = func_yuzk120x(dates_, "second")
            info_maj[minute_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = "%H:%M:%S"
            info_fmt[day_start] = "%H:%M:%S\n%d-%b"
            info_fmt[year_start] = "%H:%M:%S\n%d-%b\n%Y"

        if span < periodsperday / 12000:
            func_kjh8occr(1)
        elif span < periodsperday / 6000:
            func_kjh8occr(2)
        elif span < periodsperday / 2400:
            func_kjh8occr(5)
        elif span < periodsperday / 1200:
            func_kjh8occr(10)
        elif span < periodsperday / 800:
            func_kjh8occr(15)
        elif span < periodsperday / 400:
            func_kjh8occr(30)
        elif span < periodsperday / 150:
            func_nh1sjaj8(1)
        elif span < periodsperday / 70:
            func_nh1sjaj8(2)
        elif span < periodsperday / 24:
            func_nh1sjaj8(5)
        elif span < periodsperday / 12:
            func_nh1sjaj8(15)
        elif span < periodsperday / 6:
            func_nh1sjaj8(30)
        elif span < periodsperday / 2.5:
            func_5y0turm6(1, False)
        elif span < periodsperday / 1.5:
            func_5y0turm6(2, False)
        elif span < periodsperday * 1.25:
            func_5y0turm6(3, False)
        elif span < periodsperday * 2.5:
            func_5y0turm6(6, True)
        elif span < periodsperday * 4:
            func_5y0turm6(12, True)
        else:
            info_maj[month_start] = True
            info_min[day_start] = True
            info_fmt[day_start] = "%d"
            info_fmt[month_start] = "%d\n%b"
            info_fmt[year_start] = "%d\n%b\n%Y"
            if not func_pjswpgcx(year_start, vmin_orig):
                if not func_pjswpgcx(month_start, vmin_orig):
                    info_fmt[func_6wadagu1(day_start)] = "%d\n%b\n%Y"
                else:
                    info_fmt[func_6wadagu1(month_start)] = "%d\n%b\n%Y"
    elif span <= periodsperyear // 4:
        month_start: np.ndarray = func_fwjk7zy0(dates_, "month")
        info_maj[month_start] = True
        if dtype_code < FreqGroup.FR_HR.value:
            info["min"] = True
        else:
            day_start: np.ndarray = func_fwjk7zy0(dates_, "day")
            info["min"][day_start] = True
        week_start: np.ndarray = func_fwjk7zy0(dates_, "week")
        year_start: np.ndarray = func_fwjk7zy0(dates_, "year")
        info_fmt[week_start] = "%d"
        info_fmt[month_start] = "\n\n%b"
        info_fmt[year_start] = "\n\n%b\n%Y"
        if not func_pjswpgcx(year_start, vmin_orig):
            if not func_pjswpgcx(month_start, vmin_orig):
                info_fmt[func_6wadagu1(week_start)] = "\n\n%b\n%Y"
            else:
                info_fmt[func_6wadagu1(month_start)] = "\n\n%b\n%Y"
    elif span <= 1.15 * periodsperyear:
        year_start: np.ndarray = func_fwjk7zy0(dates_, "year")
        month_start: np.ndarray = func_fwjk7zy0(dates_, "month")
        week_start: np.ndarray = func_fwjk7zy0(dates_, "week")
        info_maj[month_start] = True
        info_min[week_start] = True
        info_min[year_start] = False
        info_min[month_start] = False
        info_fmt[month_start] = "%b"
        info_fmt[year_start] = "%b\n%Y"
        if not func_pjswpgcx(year_start, vmin_orig):
            if not func_pjswpgcx(month_start, vmin_orig):
                info_fmt[func_6wadagu1(month_start)] = "%b\n%Y"
    elif span <= 2.5 * periodsperyear:
        year_start: np.ndarray = func_fwjk7zy0(dates_, "year")
        quarter_start: np.ndarray = func_fwjk7zy0(dates_, "quarter")
        month_start: np.ndarray = func_fwjk7zy0(dates_, "month")
        info_maj[quarter_start] = True
        info_min[month_start] = True
        info_fmt[quarter_start] = "%b"
        info_fmt[year_start] = "%b\n%Y"
    elif span <= 4 * periodsperyear:
        year_start: np.ndarray = func_fwjk7zy0(dates_, "year")
        month_start: np.ndarray = func_fwjk7zy0(dates_, "month")
        info_maj[year_start] = True
        info_min[month_start] = True
        info_min[year_start] = False
        jan_or_jul: np.ndarray = month_start[
            (dates_[month_start].month == 1) | (dates_[month_start].month == 7)
        ]
        info_fmt[jan_or_jul] = "%b"
        info_fmt[year_start] = "%b\n%Y"
    elif span <= 11 * periodsperyear:
        year_start: np.ndarray = func_fwjk7zy0(dates_, "year")
        quarter_start: np.ndarray = func_fwjk7zy0(dates_, "quarter")
        info_maj[year_start] = True
        info_min[quarter_start] = True
        info_min[year_start] = False
        info_fmt[year_start] = "%Y"
    else:
        year_start: np.ndarray = func_fwjk7zy0(dates_, "year")
        year_break: np.ndarray = dates_[year_start].year
        nyears: float = span / periodsperyear
        min_anndef, maj_anndef = func_m35cyxc1(nyears)
        major_idx: np.ndarray = year_start[year_break % maj_anndef == 0]
        info_maj[major_idx] = True
        minor_idx: np.ndarray = year_start[year_break % min_anndef == 0]
        info_min[minor_idx] = True
        info_fmt[major_idx] = "%Y"
    return info


@functools.cache
def func_3ogxgdss(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    _, _, periodsperyear = func_p3bs53wa(freq)
    vmin_orig: float = vmin
    vmin, vmax = int(vmin), int(vmax)
    span: int = vmax - vmin + 1
    info: np.ndarray = np.zeros(
        span,
        dtype=[
            ("val", int),
            ("maj", bool),
            ("min", bool),
            ("fmt", "|S8"),
        ],
    )
    info["val"] = np.arange(vmin, vmax + 1)
    dates_: np.ndarray = info["val"]
    info["fmt"] = ""
    year_start: np.ndarray = (dates_ % 12 == 0).nonzero()[0]
    info_maj: np.ndarray = info["maj"]
    info_fmt: np.ndarray = info["fmt"]
    if span <= 1.15 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True
        info_fmt[:] = "%b"
        info_fmt[year_start] = "%b\n%Y"
        if not func_pjswpgcx(year_start, vmin_orig):
            if dates_.size > 1:
                idx: int = 1
            else:
                idx = 0
            info_fmt[idx] = "%b\n%Y"
    elif span <= 2.5 * periodsperyear:
        quarter_start: np.ndarray = (dates_ % 3 == 0).nonzero()[0]
        info_maj[year_start] = True
        info["min"][quarter_start] = True
        info_fmt[quarter_start] = "%b"
        info_fmt[year_start] = "%b\n%Y"
    elif span <= 4 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True
        jan_or_jul: np.ndarray = (dates_ % 12 == 0) | (dates_ % 12 == 6)
        info_fmt[jan_or_jul] = "%b"
        info_fmt[year_start] = "%b\n%Y"
    elif span <= 11 * periodsperyear:
        quarter_start: np.ndarray = (dates_ % 3 == 0).nonzero()[0]
        info_maj[year_start] = True
        info["min"][quarter_start] = True
        info_fmt[year_start] = "%Y"
    else:
        nyears: float = span / periodsperyear
        min_anndef, maj_anndef = func_m35cyxc1(nyears)
        years: np.ndarray = dates_[year_start] // 12 + 1
        major_idx: np.ndarray = year_start[years % maj_anndef == 0]
        info_maj[major_idx] = True
        info["min"][year_start[years % min_anndef == 0]] = True
        info_fmt[major_idx] = "%Y"
    return info


@functools.cache
def func_4gjzoai8(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    _, _, periodsperyear = func_p3bs53wa(freq)
    vmin_orig: float = vmin
    vmin, vmax = int(vmin), int(vmax)
    span: int = vmax - vmin + 1
    info: np.ndarray = np.zeros(
        span,
        dtype=[
            ("val", int),
            ("maj", bool),
            ("min", bool),
            ("fmt", "|S8"),
        ],
    )
    info["val"] = np.arange(vmin, vmax + 1)
    info["fmt"] = ""
    dates_: np.ndarray = info["val"]
    info_maj: np.ndarray = info["maj"]
    info_fmt: np.ndarray = info["fmt"]
    year_start: np.ndarray = (dates_ % 4 == 0).nonzero()[0]
    if span <= 3.5 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True
        info_fmt[:] = "Q%q"
        info_fmt[year_start] = "Q%q\n%F"
        if not func_pjswpgcx(year_start, vmin_orig):
            if dates_.size > 1:
                idx: int = 1
            else:
                idx = 0
            info_fmt[idx] = "Q%q\n%F"
    elif span <= 11 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True
        info_fmt[year_start] = "%F"
    else:
        years: np.ndarray = dates_[year_start] // 4 + 1970
        nyears: float = span / periodsperyear
        min_anndef, maj_anndef = func_m35cyxc1(nyears)
        major_idx: np.ndarray = year_start[years % maj_anndef == 0]
        info_maj[major_idx] = True
        info["min"][year_start[years % min_anndef == 0]] = True
        info_fmt[major_idx] = "%F"
    return info


@functools.cache
def func_tapbvj7h(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    vmin, vmax = int(vmin), int(vmax + 1)
    span: int = vmax - vmin + 1
    info: np.ndarray = np.zeros(
        span,
        dtype=[
            ("val", int),
            ("maj", bool),
            ("min", bool),
            ("fmt", "|S8"),
        ],
    )
    info["val"] = np.arange(vmin, vmax + 1)
    info["fmt"] = ""
    dates_: np.ndarray = info["val"]
    min_anndef, maj_anndef = func_m35cyxc1(span)
    major_idx: np.ndarray = dates_ % maj_anndef == 0
    minor_idx: np.ndarray = dates_ % min_anndef == 0
    info["maj"][major_idx] = True
    info["min"][minor_idx] = True
    info["fmt"][major_idx] = "%Y"
    return info


def func_nwhnz3x7(freq: BaseOffset) -> Callable[[float, float, BaseOffset], np.ndarray]:
    dtype_code: int = freq._period_dtype_code
    fgroup: FreqGroup = FreqGroup.from_period_dtype_code(dtype_code)
    if fgroup == FreqGroup.FR_ANN:
        return _annual_finder
    elif fgroup == FreqGroup.FR_QTR:
        return _quarterly_finder
    elif fgroup == FreqGroup.FR_MTH:
        return _monthly_finder
    elif dtype_code >= FreqGroup.FR_BUS.value or fgroup == FreqGroup.FR_WK:
        return _daily_finder
    else:
        raise NotImplementedError(f"Unsupported frequency: {dtype_code}")


class TimeSeries_DateLocator(mpl.ticker.Locator):
    """
    Locates the ticks along an axis controlled by a :class:`Series`.

    Parameters
    ----------
    freq : BaseOffset
        Valid frequency specifier.
    minor_locator : {False, True}, optional
        Whether the locator is for minor ticks (True) or not.
    dynamic_mode : {True, False}, optional
        Whether the locator should work in dynamic mode.
    base : {int}, optional
    quarter : {int}, optional
    month : {int}, optional
    day : {int}, optional
    plot_obj : Any, optional
    """

    def __init__(
        self,
        freq: BaseOffset,
        minor_locator: bool = False,
        dynamic_mode: bool = True,
        base: int = 1,
        quarter: int = 1,
        month: int = 1,
        day: int = 1,
        plot_obj: Any = None,
    ) -> None:
        freq = to_offset(freq, is_period=True)
        self.freq: BaseOffset = freq
        self.base: int = base
        self.quarter: int = quarter
        self.month: int = month
        self.day: int = day
        self.isminor: bool = minor_locator
        self.isdynamic: bool = dynamic_mode
        self.offset: int = 0
        self.plot_obj: Any = plot_obj
        self.finder: Callable[[float, float, BaseOffset], np.ndarray] = func_nwhnz3x7(freq)

    def func_f8llenfn(self, vmin: float, vmax: float) -> np.ndarray:
        """Returns the default locations of ticks."""
        locator: np.ndarray = self.finder(vmin, vmax, self.freq)
        if self.isminor:
            return np.compress(locator["min"], locator["val"])
        return np.compress(locator["maj"], locator["val"])

    def __call__(self) -> List[float]:
        """Return the locations of the ticks."""
        vi: Tuple[float, float] = tuple(self.axis.get_view_interval())
        vmin, vmax = vi
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        if self.isdynamic:
            locs: np.ndarray = self.func_f8llenfn(vmin, vmax)
            return locs.tolist()
        else:
            base = self.base
            d, m = divmod(vmin, base)
            vmin = (d + 1) * base
            locs: List[int] = list(range(vmin, vmax + 1, base))
            return locs

    def func_oywae6cj(self) -> tuple[float, float]:
        """
        Sets the view limits to the nearest multiples of base that contain the
        data.
        """
        vmin, vmax = self.axis.get_data_interval()
        locs: List[float] = self.func_f8llenfn(vmin, vmax).tolist()
        vmin, vmax = locs[0], locs[-1]
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        return mpl.transforms.nonsingular(vmin, vmax)


class TimeSeries_DateFormatter(mpl.ticker.Formatter):
    """
    Formats the ticks along an axis controlled by a :class:`PeriodIndex`.

    Parameters
    ----------
    freq : BaseOffset
        Valid frequency specifier.
    minor_locator : bool, default False
        Whether the current formatter should apply to minor ticks (True) or
        major ticks (False).
    dynamic_mode : bool, default True
        Whether the formatter works in dynamic mode or not.
    plot_obj : Any, optional
    """

    def __init__(
        self,
        freq: BaseOffset,
        minor_locator: bool = False,
        dynamic_mode: bool = True,
        plot_obj: Any = None,
    ) -> None:
        freq = to_offset(freq, is_period=True)
        self.format: Any = None
        self.freq: BaseOffset = freq
        self.locs: List[float] = []
        self.formatdict: dict[float, str] | None = None
        self.isminor: bool = minor_locator
        self.isdynamic: bool = dynamic_mode
        self.offset: int = 0
        self.plot_obj: Any = plot_obj
        self.finder: Callable[[float, float, BaseOffset], np.ndarray] = func_nwhnz3x7(freq)

    def func_4qbsq9c5(self, vmin: float, vmax: float) -> dict[float, str]:
        """Returns the default ticks spacing."""
        info: np.ndarray = self.finder(vmin, vmax, self.freq)
        if self.isminor:
            format_array: np.ndarray = np.compress(
                info["min"] & ~info["maj"], info
            )
        else:
            format_array = np.compress(info["maj"], info)
        self.formatdict = {x: f.decode("utf-8") for x, _, _, f in format_array}
        return self.formatdict

    def func_gd98p30z(self, locs: List[float]) -> None:
        """Sets the locations of the ticks"""
        self.locs = locs
        vmin, vmax = tuple(self.axis.get_view_interval())
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        self.func_4qbsq9c5(vmin, vmax)

    def __call__(self, x: float, pos: int = 0) -> str:
        if self.formatdict is None:
            return ""
        else:
            fmt: str = self.formatdict.pop(x, "")
            if isinstance(fmt, np.bytes_):
                fmt = fmt.decode("utf-8")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Period with BDay freq is deprecated",
                    category=FutureWarning,
                )
                period: Period = Period(int(x), freq=self.freq)
            assert isinstance(period, Period)
            return period.strftime(fmt)


class TimeSeries_TimedeltaFormatter(mpl.ticker.Formatter):
    """
    Formats the ticks along an axis controlled by a :class:`TimedeltaIndex`.
    """

    @staticmethod
    def func_qu47f13f(x: int, pos: int, n_decimals: int) -> str:
        """
        Convert seconds to 'D days HH:MM:SS.F'
        """
        s, ns = divmod(x, 10**9)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        decimals: int = int(ns * 10**(n_decimals - 9))
        s_str: str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        if n_decimals > 0:
            s_str += f".{decimals:0{n_decimals}d}"
        if d != 0:
            s_str = f"{int(d)} days {s_str}"
        return s_str

    def __call__(self, x: float, pos: int = 0) -> str:
        vmin, vmax = tuple(self.axis.get_view_interval())
        if vmax - vmin == 0:
            n_decimals = 0
        else:
            n_decimals = min(
                int(np.ceil(np.log10(100 * 10**9 / abs(vmax - vmin)))),
                9,
            )
        return self.func_qu47f13f(x, pos, n_decimals)


def time2num(t: pydt.time) -> float:
    return func_w4sj649z(t)


@contextlib.contextmanager
def pandas_converters() -> Generator[None, None, None]:
    func_swriha8l()
    try:
        yield
    finally:
        func_m7j2fvrr()


def register() -> None:
    mpl.rcParams["date.converter"] = register_pandas_matplotlib_converters


def deregister() -> None:
    "Unregister pandas converters"
    munits.registry.update(_mpl_units)
    _mpl_units.clear()


def register_pandas_matplotlib_converters() -> Callable[[F], F]:
    return func_e8bwjllm


# Placeholder functions for _annual_finder, _quarterly_finder, _monthly_finder, _daily_finder
def _annual_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    return np.array([])


def _quarterly_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    return np.array([])


def _monthly_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    return np.array([])


def _daily_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    return np.array([])


def _period_break_mask(dates: PeriodIndex, period: str) -> np.ndarray:
    return np.array([])


def get_datevalue(value: Any, freq: BaseOffset) -> float:
    return 0.0

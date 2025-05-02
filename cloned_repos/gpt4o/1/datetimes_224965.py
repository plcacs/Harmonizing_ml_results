from __future__ import annotations
import datetime as dt
import operator
from typing import TYPE_CHECKING, Optional, Union
import warnings
import numpy as np
from pandas._libs import NaT, Period, Timestamp, index as libindex, lib
from pandas._libs.tslibs import Resolution, Tick, Timedelta, periods_per_day, timezones, to_offset
from pandas._libs.tslibs.offsets import prefix_mapping
from pandas.util._decorators import cache_readonly, doc, set_module
from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.arrays.datetimes import DatetimeArray, tz_to_dtype
import pandas.core.common as com
from pandas.core.indexes.base import Index, maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names
from pandas.core.tools.times import to_time
if TYPE_CHECKING:
    from collections.abc import Hashable
    from pandas._typing import Dtype, DtypeObj, Frequency, IntervalClosedType, Self, TimeAmbiguous, TimeNonexistent, npt
    from pandas.core.api import DataFrame, PeriodIndex
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR

def _new_DatetimeIndex(cls: type[DatetimeIndex], d: dict) -> DatetimeIndex:
    if 'data' in d and (not isinstance(d['data'], DatetimeIndex)):
        data = d.pop('data')
        if not isinstance(data, DatetimeArray):
            tz = d.pop('tz')
            freq = d.pop('freq')
            dta = DatetimeArray._simple_new(data, dtype=tz_to_dtype(tz), freq=freq)
        else:
            dta = data
            for key in ['tz', 'freq']:
                if key in d:
                    assert d[key] == getattr(dta, key)
                    d.pop(key)
        result = cls._simple_new(dta, **d)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = cls.__new__(cls, **d)
    return result

@inherit_names(DatetimeArray._field_ops + [method for method in DatetimeArray._datetimelike_methods if method not in ('tz_localize', 'tz_convert', 'strftime')], DatetimeArray, wrap=True)
@inherit_names(['is_normalized'], DatetimeArray, cache=True)
@inherit_names(['tz', 'tzinfo', 'dtype', 'to_pydatetime', 'date', 'time', 'timetz', 'std'] + DatetimeArray._bool_ops, DatetimeArray)
@set_module('pandas')
class DatetimeIndex(DatetimeTimedeltaMixin):
    _typ: str = 'datetimeindex'
    _data_cls: type = DatetimeArray
    _supports_partial_string_indexing: bool = True

    @property
    def _engine_type(self) -> type:
        return libindex.DatetimeEngine

    @doc(DatetimeArray.strftime)
    def strftime(self, date_format: str) -> Index:
        arr = self._data.strftime(date_format)
        return Index(arr, name=self.name, dtype=arr.dtype)

    @doc(DatetimeArray.tz_convert)
    def tz_convert(self, tz: Union[str, dt.tzinfo]) -> DatetimeIndex:
        arr = self._data.tz_convert(tz)
        return type(self)._simple_new(arr, name=self.name, refs=self._references)

    @doc(DatetimeArray.tz_localize)
    def tz_localize(self, tz: Union[str, dt.tzinfo], ambiguous: str = 'raise', nonexistent: str = 'raise') -> DatetimeIndex:
        arr = self._data.tz_localize(tz, ambiguous, nonexistent)
        return type(self)._simple_new(arr, name=self.name)

    @doc(DatetimeArray.to_period)
    def to_period(self, freq: Optional[str] = None) -> PeriodIndex:
        from pandas.core.indexes.api import PeriodIndex
        arr = self._data.to_period(freq)
        return PeriodIndex._simple_new(arr, name=self.name)

    @doc(DatetimeArray.to_julian_date)
    def to_julian_date(self) -> Index:
        arr = self._data.to_julian_date()
        return Index._simple_new(arr, name=self.name)

    @doc(DatetimeArray.isocalendar)
    def isocalendar(self) -> DataFrame:
        df = self._data.isocalendar()
        return df.set_index(self)

    @cache_readonly
    def _resolution_obj(self) -> Resolution:
        return self._data._resolution_obj

    def __new__(cls: type[DatetimeIndex], data: Optional[Union[np.ndarray, DatetimeArray, Index, ABCSeries]] = None, freq: Optional[Union[str, Tick]] = lib.no_default, tz: Optional[Union[str, dt.tzinfo]] = lib.no_default, ambiguous: str = 'raise', dayfirst: bool = False, yearfirst: bool = False, dtype: Optional[Union[np.dtype, DatetimeTZDtype, str]] = None, copy: bool = False, name: Optional[Hashable] = None) -> DatetimeIndex:
        if is_scalar(data):
            cls._raise_scalar_data_error(data)
        name = maybe_extract_name(name, data, cls)
        if isinstance(data, DatetimeArray) and freq is lib.no_default and (tz is lib.no_default) and (dtype is None):
            if copy:
                data = data.copy()
            return cls._simple_new(data, name=name)
        dtarr = DatetimeArray._from_sequence_not_strict(data, dtype=dtype, copy=copy, tz=tz, freq=freq, dayfirst=dayfirst, yearfirst=yearfirst, ambiguous=ambiguous)
        refs = None
        if not copy and isinstance(data, (Index, ABCSeries)):
            refs = data._references
        subarr = cls._simple_new(dtarr, name=name, refs=refs)
        return subarr

    @cache_readonly
    def _is_dates_only(self) -> bool:
        if isinstance(self.freq, Tick):
            delta = Timedelta(self.freq)
            if delta % dt.timedelta(days=1) != dt.timedelta(days=0):
                return False
        return self._values._is_dates_only

    def __reduce__(self) -> tuple:
        d = {'data': self._data, 'name': self.name}
        return (_new_DatetimeIndex, (type(self), d), None)

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        if self.tz is not None:
            return isinstance(dtype, DatetimeTZDtype)
        return lib.is_np_dtype(dtype, 'M')

    @cache_readonly
    def _formatter_func(self) -> callable:
        from pandas.io.formats.format import get_format_datetime64
        formatter = get_format_datetime64(is_dates_only=self._is_dates_only)
        return lambda x: f"'{formatter(x)}'"

    def _can_range_setop(self, other: DatetimeIndex) -> bool:
        if self.tz is not None and (not timezones.is_utc(self.tz)) and (not timezones.is_fixed_offset(self.tz)):
            return False
        if other.tz is not None and (not timezones.is_utc(other.tz)) and (not timezones.is_fixed_offset(other.tz)):
            return False
        return super()._can_range_setop(other)

    def _get_time_micros(self) -> np.ndarray:
        values = self._data._local_timestamps()
        ppd = periods_per_day(self._data._creso)
        frac = values % ppd
        if self.unit == 'ns':
            micros = frac // 1000
        elif self.unit == 'us':
            micros = frac
        elif self.unit == 'ms':
            micros = frac * 1000
        elif self.unit == 's':
            micros = frac * 1000000
        else:
            raise NotImplementedError(self.unit)
        micros[self._isnan] = -1
        return micros

    def snap(self, freq: Union[str, Tick, dt.timedelta, Timedelta] = 'S') -> DatetimeIndex:
        freq = to_offset(freq)
        dta = self._data.copy()
        for i, v in enumerate(self):
            s = v
            if not freq.is_on_offset(s):
                t0 = freq.rollback(s)
                t1 = freq.rollforward(s)
                if abs(s - t0) < abs(t1 - s):
                    s = t0
                else:
                    s = t1
            dta[i] = s
        return DatetimeIndex._simple_new(dta, name=self.name)

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: dt.datetime) -> tuple[Timestamp, Timestamp]:
        freq = OFFSET_TO_PERIOD_FREQSTR.get(reso.attr_abbrev, reso.attr_abbrev)
        per = Period(parsed, freq=freq)
        start, end = (per.start_time, per.end_time)
        start = start.as_unit(self.unit)
        end = end.as_unit(self.unit)
        start = start.tz_localize(parsed.tzinfo)
        end = end.tz_localize(parsed.tzinfo)
        if parsed.tzinfo is not None:
            if self.tz is None:
                raise ValueError('The index must be timezone aware when indexing with a date string with a UTC offset')
        return (start, end)

    def _parse_with_reso(self, label: str) -> tuple[Timestamp, Resolution]:
        parsed, reso = super()._parse_with_reso(label)
        parsed = Timestamp(parsed)
        if self.tz is not None and parsed.tzinfo is None:
            parsed = parsed.tz_localize(self.tz)
        return (parsed, reso)

    def _disallow_mismatched_indexing(self, key: Union[dt.datetime, Timestamp]) -> None:
        try:
            self._data._assert_tzawareness_compat(key)
        except TypeError as err:
            raise KeyError(key) from err

    def get_loc(self, key: Union[str, dt.datetime, dt.time, dt.timedelta, Timestamp]) -> int:
        self._check_indexing_error(key)
        orig_key = key
        if is_valid_na_for_dtype(key, self.dtype):
            key = NaT
        if isinstance(key, self._data._recognized_scalars):
            self._disallow_mismatched_indexing(key)
            key = Timestamp(key)
        elif isinstance(key, str):
            try:
                parsed, reso = self._parse_with_reso(key)
            except ValueError as err:
                raise KeyError(key) from err
            self._disallow_mismatched_indexing(parsed)
            if self._can_partial_date_slice(reso):
                try:
                    return self._partial_date_slice(reso, parsed)
                except KeyError as err:
                    raise KeyError(key) from err
            key = parsed
        elif isinstance(key, dt.timedelta):
            raise TypeError(f'Cannot index {type(self).__name__} with {type(key).__name__}')
        elif isinstance(key, dt.time):
            return self.indexer_at_time(key)
        else:
            raise KeyError(key)
        try:
            return Index.get_loc(self, key)
        except KeyError as err:
            raise KeyError(orig_key) from err

    @doc(DatetimeTimedeltaMixin._maybe_cast_slice_bound)
    def _maybe_cast_slice_bound(self, label: Union[str, dt.date, dt.datetime], side: str) -> Timestamp:
        if isinstance(label, dt.date) and (not isinstance(label, dt.datetime)):
            label = Timestamp(label).to_pydatetime()
        label = super()._maybe_cast_slice_bound(label, side)
        self._data._assert_tzawareness_compat(label)
        return Timestamp(label)

    def slice_indexer(self, start: Optional[Union[str, dt.datetime, dt.time]] = None, end: Optional[Union[str, dt.datetime, dt.time]] = None, step: Optional[int] = None) -> Union[slice, np.ndarray]:
        if isinstance(start, dt.time) and isinstance(end, dt.time):
            if step is not None and step != 1:
                raise ValueError('Must have step size of 1 with time slices')
            return self.indexer_between_time(start, end)
        if isinstance(start, dt.time) or isinstance(end, dt.time):
            raise KeyError('Cannot mix time and non-time slice keys')

        def check_str_or_none(point: Optional[Union[str, dt.datetime]]) -> bool:
            return point is not None and (not isinstance(point, str))
        if check_str_or_none(start) or check_str_or_none(end) or self.is_monotonic_increasing:
            return Index.slice_indexer(self, start, end, step)
        mask = np.array(True)
        in_index = True
        if start is not None:
            start_casted = self._maybe_cast_slice_bound(start, 'left')
            mask = start_casted <= self
            in_index &= (start_casted == self).any()
        if end is not None:
            end_casted = self._maybe_cast_slice_bound(end, 'right')
            mask = (self <= end_casted) & mask
            in_index &= (end_casted == self).any()
        if not in_index:
            raise KeyError('Value based partial slicing on non-monotonic DatetimeIndexes with non-existing keys is not allowed.')
        indexer = mask.nonzero()[0][::step]
        if len(indexer) == len(self):
            return slice(None)
        else:
            return indexer

    @property
    def inferred_type(self) -> str:
        return 'datetime64'

    def indexer_at_time(self, time: Union[dt.time, str], asof: bool = False) -> np.ndarray:
        if asof:
            raise NotImplementedError("'asof' argument is not supported")
        if isinstance(time, str):
            from dateutil.parser import parse
            time = parse(time).time()
        if time.tzinfo:
            if self.tz is None:
                raise ValueError('Index must be timezone aware.')
            time_micros = self.tz_convert(time.tzinfo)._get_time_micros()
        else:
            time_micros = self._get_time_micros()
        micros = _time_to_micros(time)
        return (time_micros == micros).nonzero()[0]

    def indexer_between_time(self, start_time: Union[dt.time, str], end_time: Union[dt.time, str], include_start: bool = True, include_end: bool = True) -> np.ndarray:
        start_time = to_time(start_time)
        end_time = to_time(end_time)
        time_micros = self._get_time_micros()
        start_micros = _time_to_micros(start_time)
        end_micros = _time_to_micros(end_time)
        if include_start and include_end:
            lop = rop = operator.le
        elif include_start:
            lop = operator.le
            rop = operator.lt
        elif include_end:
            lop = operator.lt
            rop = operator.le
        else:
            lop = rop = operator.lt
        if start_time <= end_time:
            join_op = operator.and_
        else:
            join_op = operator.or_
        mask = join_op(lop(start_micros, time_micros), rop(time_micros, end_micros))
        return mask.nonzero()[0]

@set_module('pandas')
def date_range(start: Optional[Union[str, dt.datetime]] = None, end: Optional[Union[str, dt.datetime]] = None, periods: Optional[int] = None, freq: Optional[Union[str, Tick, dt.timedelta, Timedelta]] = None, tz: Optional[Union[str, dt.tzinfo]] = None, normalize: bool = False, name: Optional[str] = None, inclusive: str = 'both', *, unit: Optional[str] = None, **kwargs) -> DatetimeIndex:
    if freq is None and com.any_none(periods, start, end):
        freq = 'D'
    dtarr = DatetimeArray._generate_range(start=start, end=end, periods=periods, freq=freq, tz=tz, normalize=normalize, inclusive=inclusive, unit=unit, **kwargs)
    return DatetimeIndex._simple_new(dtarr, name=name)

@set_module('pandas')
def bdate_range(start: Optional[Union[str, dt.datetime]] = None, end: Optional[Union[str, dt.datetime]] = None, periods: Optional[int] = None, freq: Union[str, Tick, dt.timedelta, Timedelta] = 'B', tz: Optional[Union[str, dt.tzinfo]] = None, normalize: bool = True, name: Optional[str] = None, weekmask: Optional[str] = None, holidays: Optional[list] = None, inclusive: str = 'both', **kwargs) -> DatetimeIndex:
    if freq is None:
        msg = 'freq must be specified for bdate_range; use date_range instead'
        raise TypeError(msg)
    if isinstance(freq, str) and freq.startswith('C'):
        try:
            weekmask = weekmask or 'Mon Tue Wed Thu Fri'
            freq = prefix_mapping[freq](holidays=holidays, weekmask=weekmask)
        except (KeyError, TypeError) as err:
            msg = f'invalid custom frequency string: {freq}'
            raise ValueError(msg) from err
    elif holidays or weekmask:
        msg = f'a custom frequency string is required when holidays or weekmask are passed, got frequency {freq}'
        raise ValueError(msg)
    return date_range(start=start, end=end, periods=periods, freq=freq, tz=tz, normalize=normalize, name=name, inclusive=inclusive, **kwargs)

def _time_to_micros(time_obj: dt.time) -> int:
    seconds = time_obj.hour * 60 * 60 + 60 * time_obj.minute + time_obj.second
    return 1000000 * seconds + time_obj.microsecond

from __future__ import annotations
from datetime import datetime, timedelta, tzinfo
from typing import TYPE_CHECKING, TypeVar, cast, overload, Any, Optional, Union, List, Tuple, Generator, Iterator, Dict, Sequence
import warnings
import numpy as np
from pandas._config import using_string_dtype
from pandas._config.config import get_option
from pandas._libs import lib, tslib
from pandas._libs.tslibs import BaseOffset, NaT, NaTType, Resolution, Timestamp, astype_overflowsafe, fields, get_resolution, get_supported_dtype, get_unit_from_dtype, ints_to_pydatetime, is_date_array_normalized, is_supported_dtype, is_unitless, normalize_i8_timestamps, timezones, to_offset, tz_convert_from_utc, tzconversion
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_inclusive
from pandas.core.dtypes.common import DT64NS_DTYPE, INT64_DTYPE, is_bool_dtype, is_float_dtype, is_string_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype, ExtensionDtype, PeriodDtype
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.tseries.frequencies import get_period_alias
from pandas.tseries.offsets import Day, Tick

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from pandas._typing import ArrayLike, DateTimeErrorChoices, DtypeObj, IntervalClosedType, Self, TimeAmbiguous, TimeNonexistent, npt
    from pandas import DataFrame, Timedelta
    from pandas.core.arrays import PeriodArray
    _TimestampNoneT1 = TypeVar('_TimestampNoneT1', Timestamp, None)
    _TimestampNoneT2 = TypeVar('_TimestampNoneT2', Timestamp, None)

_ITER_CHUNKSIZE = 10000

@overload
def func_s4181h5i(tz: Optional[tzinfo], unit: str = ...) -> np.dtype: ...

@overload
def func_s4181h5i(tz: Optional[tzinfo], unit: str = ...) -> DatetimeTZDtype: ...

def func_s4181h5i(tz: Optional[tzinfo], unit: str = 'ns') -> Union[np.dtype, DatetimeTZDtype]:
    if tz is None:
        return np.dtype(f'M8[{unit}]')
    else:
        return DatetimeTZDtype(tz=tz, unit=unit)

def func_wk1mu9r9(name: str, field: str, docstring: Optional[str] = None) -> property:
    def func_ry3g5jtp(self: Any) -> Any:
        values = self._local_timestamps()
        if field in self._bool_ops:
            if field.endswith(('start', 'end')):
                freq = self.freq
                month_kw = 12
                if freq:
                    kwds = freq.kwds
                    month_kw = kwds.get('startingMonth', kwds.get('month', month_kw))
                if freq is not None:
                    freq_name = freq.name
                else:
                    freq_name = None
                result = fields.get_start_end_field(values, field, freq_name, month_kw, reso=self._creso)
            else:
                result = fields.get_date_field(values, field, reso=self._creso)
            return result
        result = fields.get_date_field(values, field, reso=self._creso)
        result = self._maybe_mask_results(result, fill_value=None, convert='float64')
        return result
    f = func_ry3g5jtp
    f.__name__ = name
    f.__doc__ = docstring
    return property(f)

class DatetimeArray(dtl.TimelikeOps, dtl.DatelikeOps):
    _typ: str = 'datetimearray'
    _internal_fill_value: np.datetime64 = np.datetime64('NaT', 'ns')
    _recognized_scalars: Tuple[type, type] = (datetime, np.datetime64)
    _is_recognized_dtype: Any = lambda x: lib.is_np_dtype(x, 'M') or isinstance(x, DatetimeTZDtype)
    _infer_matches: Tuple[str, str, str] = ('datetime', 'datetime64', 'date')

    @property
    def func_quui8io2(self) -> type:
        return Timestamp

    _bool_ops: List[str] = ['is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end', 'is_leap_year']
    _field_ops: List[str] = ['year', 'month', 'day', 'hour', 'minute', 'second', 'weekday', 'dayofweek', 'day_of_week', 'dayofyear', 'day_of_year', 'quarter', 'days_in_month', 'daysinmonth', 'microsecond', 'nanosecond']
    _other_ops: List[str] = ['date', 'time', 'timetz']
    _datetimelike_ops: List[str] = _field_ops + _bool_ops + _other_ops + ['unit', 'freq', 'tz']
    _datetimelike_methods: List[str] = ['to_period', 'tz_localize', 'tz_convert', 'normalize', 'strftime', 'round', 'floor', 'ceil', 'month_name', 'day_name', 'as_unit']
    __array_priority__: int = 1000
    _freq: Optional[BaseOffset] = None

    @classmethod
    def func_xksorsp6(cls, scalars: Sequence[Any], *, dtype: DtypeObj) -> DatetimeArray:
        if lib.infer_dtype(scalars, skipna=True) not in ['datetime', 'datetime64']:
            raise ValueError
        return cls._from_sequence(scalars, dtype=dtype)

    @classmethod
    def func_9ila1bj4(cls, values: np.ndarray, dtype: DtypeObj) -> DtypeObj:
        dtype = _validate_dt64_dtype(dtype)
        _validate_dt64_dtype(values.dtype)
        if isinstance(dtype, np.dtype):
            if values.dtype != dtype:
                raise ValueError('Values resolution does not match dtype.')
        else:
            vunit = np.datetime_data(values.dtype)[0]
            if vunit != dtype.unit:
                raise ValueError('Values resolution does not match dtype.')
        return dtype

    @classmethod
    def func_370zkfv7(cls, values: np.ndarray, freq: Optional[BaseOffset] = None, dtype: DtypeObj = DT64NS_DTYPE) -> DatetimeArray:
        assert isinstance(values, np.ndarray)
        assert dtype.kind == 'M'
        if isinstance(dtype, np.dtype):
            assert dtype == values.dtype
            assert not is_unitless(dtype)
        else:
            assert dtype._creso == get_unit_from_dtype(values.dtype)
        result = super()._simple_new(values, dtype)
        result._freq = freq
        return result

    @classmethod
    def func_st3qxwkz(cls, scalars: Sequence[Any], *, dtype: Optional[DtypeObj] = None, copy: bool = False) -> DatetimeArray:
        return cls._from_sequence_not_strict(scalars, dtype=dtype, copy=copy)

    @classmethod
    def func_w233dtr0(cls, data: Any, *, dtype: Optional[DtypeObj] = None, copy: bool = False, tz: Any = lib.no_default, freq: Any = lib.no_default, dayfirst: bool = False, yearfirst: bool = False, ambiguous: str = 'raise') -> DatetimeArray:
        explicit_tz_none = tz is None
        if tz is lib.no_default:
            tz = None
        else:
            tz = timezones.maybe_get_tz(tz)
        dtype = _validate_dt64_dtype(dtype)
        tz = _validate_tz_from_dtype(dtype, tz, explicit_tz_none)
        unit = None
        if dtype is not None:
            unit = dtl.dtype_to_unit(dtype)
        data, copy = dtl.ensure_arraylike_for_datetimelike(data, copy, cls_name='DatetimeArray')
        inferred_freq = None
        if isinstance(data, DatetimeArray):
            inferred_freq = data.freq
        subarr, tz = _sequence_to_dt64(data, copy=copy, tz=tz, dayfirst=dayfirst, yearfirst=yearfirst, ambiguous=ambiguous, out_unit=unit)
        _validate_tz_from_dtype(dtype, tz, explicit_tz_none)
        if tz is not None and explicit_tz_none:
            raise ValueError("Passed data is timezone-aware, incompatible with 'tz=None'. Use obj.tz_localize(None) instead.")
        data_unit = np.datetime_data(subarr.dtype)[0]
        data_dtype = func_s4181h5i(tz, data_unit)
        result = cls._simple_new(subarr, freq=inferred_freq, dtype=data_dtype)
        if unit is not None and unit != result.unit:
            result = result.as_unit(unit)
        validate_kwds = {'ambiguous': ambiguous}
        result._maybe_pin_freq(freq, validate_kwds)
        return result

    @classmethod
    def func_qszdfm6v(cls, start: Optional[Union[Timestamp, str]], end: Optional[Union[Timestamp, str]], periods: Optional[int], freq: Any, tz: Optional[tzinfo] = None, normalize: bool = False, ambiguous: str = 'raise', nonexistent: str = 'raise', inclusive: str = 'both', *, unit: Optional[str] = None) -> DatetimeArray:
        periods = dtl.validate_periods(periods)
        if freq is None and any(x is None for x in [periods, start, end]):
            raise ValueError('Must provide freq argument if no data is supplied')
        if com.count_not_none(start, end, periods, freq) != 3:
            raise ValueError('Of the four parameters: start, end, periods, and freq, exactly three must be specified')
        freq = to_offset(freq)
        if start is not None:
            start = Timestamp(start)
        if end is not None:
            end = Timestamp(end)
        if start is NaT or end is NaT:
            raise ValueError('Neither `start` nor `end` can be NaT')
        if unit is not None:
            if unit not in ['s', 'ms', 'us', 'ns']:
                raise ValueError("'unit' must be one of 's', 'ms', 'us', 'ns'")
        else:
            unit = 'ns'
        if start is not None:
            start = start.as_unit(unit, round_ok=False)
        if end is not None:
            end = end.as_unit(unit, round_ok=False)
        left_inclusive, right_inclusive = validate_inclusive(inclusive)
        start, end = _maybe_normalize_endpoints(start, end, normalize)
        tz = _infer_tz_from_endpoints(start, end, tz)
        if tz is not None:
            start = _maybe_localize_point(start, freq, tz, ambiguous, nonexistent)
            end = _maybe_localize_point(end, freq, tz, ambiguous, nonexistent)
        if freq is not None:
            if isinstance(freq, Day):
                if start is not None:
                    start = start.tz_localize(None)
                if end is not None:
                    end = end.tz_localize(None)
            if isinstance(freq, Tick):
                i8values = generate_regular_range(start, end, periods, freq, unit=unit)
            else:
                xdr = func_qszdfm6v(start=start, end=end, periods=periods, offset=freq, unit=unit)
                i8values = np.array([x._value for x in xdr], dtype=np.int64)
            endpoint_tz = start.tz if start is not None else end.tz
            if tz is not None and endpoint_tz is None:
                if not timezones.is_utc(tz):
                    creso = abbrev_to_npy_unit(unit)
                    i8values = tzconversion.tz_localize_to_utc(i8values, tz, ambiguous=ambiguous, nonexistent=nonexistent, creso=creso)
                if start is not None:
                    start = start.tz_localize(tz, ambiguous, nonexistent)
                if end is not None:
                    end = end.tz_localize(tz, ambiguous, nonexistent)
        else:
            periods = cast(int, periods)
            i8values = np.linspace(0, end._value - start._value, periods, dtype='int64') + start._value
            if i8values.dtype != 'i8':
                i8values = i8values.astype('i8')
        if start == end:
            if not left_inclusive and not right_inclusive:
                i8values = i8values[1:-1]
        else:
            start_i8 = Timestamp(start)._value
            end_i8 = Timestamp(end)._value
            if not left_inclusive or not right_inclusive:
                if not left_inclusive and len(i8values) and i8values[0] == start_i8:
                    i8values = i8values[1:]
                if not right_inclusive and len(i8values) and i8values[-1] == end_i8:
                    i8values = i8values[:-1]
        dt64_values = i8values.view(f'datetime64[{unit}]')
        dtype = func_s4181h5i(tz, unit=unit)
        return cls._simple_new(dt64_values, freq=freq, dtype=dtype)

    def func_o1qkqv2i(self, value: Any) -> np.datetime64:
        if not isinstance(value, self._scalar_type) and value is not NaT:
            raise ValueError("'value' should be a Timestamp.")
        self._check_compatible_with(value)
        if value is NaT:
            return np.datetime64(value._value, self.unit)
        else:
            return value.as_unit(self.unit, round_ok=False).asm8

    def func_zisz75xe(self, value: Any) -> Timestamp:
        return Timestamp(value, tz=self.tz)

    def func_tdmk59k7(self, other: Any) -> None:
        if other is NaT:
            return
        self._assert_tzawareness_compat(other)

    def func_g1sc1n2n(self, x: Any) -> Timestamp:
        value = x.view('i8')
        ts = Timestamp._from_value_and_reso(value, reso=self._creso, tz=self.tz)
        return ts

    @property
    def func_in7hk2ew(self) -> Union[np.dtype, DatetimeTZDtype]:
        return self._dtype

    @property
    def func_aizu4xmw(self) -> Optional[tzinfo]:
        return getattr(self.dtype, 'tz', None)

    @func_aizu4xmw.setter
    def func_aizu4xmw(self, value: Any) -> None:
        raise AttributeError('Cannot directly set timezone. Use tz_localize() or tz_convert() as appropriate')

    @property
    def func_4736otgv(self) -> Optional[tzinfo]:
        return self.tz

    @property
    def func_ia8hy46t(self) -> bool:
        return is_date_array_normalized(self.asi8, self.tz, reso=self._creso)

    @property
    def func_t1nqcx4s(self) -> Resolution:
        return get_resolution(self.asi8, self.tz, reso=self._creso)

    def __array__(self, dtype: Optional[DtypeObj] = None, copy: Optional[bool] = None) -> np.ndarray:
        if dtype is None and self.tz:
            dtype = object
        return super().__array__(dtype=dtype, copy=copy)

    def __iter__(self) -> Iterator[Timestamp]:
        if self.ndim > 1:
            for i in range(len(self)):
                yield self[i]
        else:
            data = self.asi8
            length = len(self)
            chunksize = _ITER_CHUNKSIZE
            chunks = length // chunksize + 1
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, length)
                converted = ints_to_pydatetime(data[start_i:end_i], tz=self.tz, box='timestamp', reso=self._creso)
                yield from converted

    def func_n81j45de(self, dtype: DtypeObj, copy: bool = True) -> Any:
        dtype = pandas_dtype(dtype)
        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self
        elif isinstance(dtype, ExtensionDtype):
            if not isinstance(dtype, DatetimeTZDtype):
                return super().astype(dtype, copy=copy)
            elif self.tz is None:
                raise TypeError('Cannot use .astype to convert from timezone-naive dtype to timezone-aware dtype. Use obj.tz_localize instead or series.dt.tz_localize instead')
            else:
                np_dtype = np.dtype(dtype.str)
                res_values = astype_overflowsafe(self._ndarray, np_dtype, copy=copy)
                return type(self)._simple_new(res_values, dtype=dtype, freq=self.freq)
        elif self.tz is None and lib.is_np_dtype(dtype, 'M') and not is_unitless(dtype) and is_supported_dtype(dtype):
            res_values = ast
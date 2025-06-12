from __future__ import annotations
from datetime import timedelta
import operator
from typing import TYPE_CHECKING, cast, Union, Optional, Tuple
import numpy as np
from pandas._libs import lib, tslibs
from pandas._libs.tslibs import NaT, NaTType, Tick, Timedelta, astype_overflowsafe, get_supported_dtype, iNaT, is_supported_dtype, periods_per_second
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.fields import get_timedelta_days, get_timedelta_field
from pandas._libs.tslibs.timedeltas import array_to_timedelta64, floordiv_object_array, ints_to_pytimedelta, parse_timedelta_unit, truediv_object_array
from pandas.compat.numpy import function as nv
from pandas.util._validators import validate_endpoints
from pandas.core.dtypes.common import TD64NS_DTYPE, is_float_dtype, is_integer_dtype, is_object_dtype, is_scalar, is_string_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import isna
from pandas.core import nanops, roperator
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.core.ops.common import unpack_zerodim_and_defer
if TYPE_CHECKING:
    from collections.abc import Iterator
    from pandas._typing import AxisInt, DateTimeErrorChoices, DtypeObj, NpDtype, Self, npt
    from pandas import DataFrame
import textwrap

def _field_accessor(name: str, alias: str, docstring: str) -> property:
    def f(self) -> np.ndarray:
        values = self.asi8
        if alias == 'days':
            result = get_timedelta_days(values, reso=self._creso)
        else:
            result = get_timedelta_field(values, alias, reso=self._creso)
        if self._hasna:
            result = self._maybe_mask_results(result, fill_value=None, convert='float64')
        return result
    f.__name__ = name
    f.__doc__ = f'\n{docstring}\n'
    return property(f)

class TimedeltaArray(dtl.TimelikeOps):
    _typ: str = 'timedeltaarray'
    _internal_fill_value: np.timedelta64 = np.timedelta64('NaT', 'ns')
    _recognized_scalars: Tuple[type, ...] = (timedelta, np.timedelta64, Tick)
    _is_recognized_dtype: callable = lambda x: lib.is_np_dtype(x, 'm')
    _infer_matches: Tuple[str, ...] = ('timedelta', 'timedelta64')

    @property
    def _scalar_type(self) -> type:
        return Timedelta
    __array_priority__: int = 1000
    _other_ops: list = []
    _bool_ops: list = []
    _field_ops: list = ['days', 'seconds', 'microseconds', 'nanoseconds']
    _datetimelike_ops: list = _field_ops + _bool_ops + ['unit', 'freq']
    _datetimelike_methods: list = ['to_pytimedelta', 'total_seconds', 'round', 'floor', 'ceil', 'as_unit']

    def _box_func(self, x: np.timedelta64) -> Union[Timedelta, NaTType]:
        y = x.view('i8')
        if y == NaT._value:
            return NaT
        return Timedelta._from_value_and_reso(y, reso=self._creso)

    @property
    def dtype(self) -> np.dtype:
        return self._ndarray.dtype
    _freq: Optional[Tick] = None

    @classmethod
    def _validate_dtype(cls, values: np.ndarray, dtype: np.dtype) -> np.dtype:
        dtype = _validate_td64_dtype(dtype)
        _validate_td64_dtype(values.dtype)
        if dtype != values.dtype:
            raise ValueError('Values resolution does not match dtype.')
        return dtype

    @classmethod
    def _simple_new(cls, values: np.ndarray, freq: Optional[Tick] = None, dtype: np.dtype = TD64NS_DTYPE) -> TimedeltaArray:
        assert lib.is_np_dtype(dtype, 'm')
        assert not tslibs.is_unitless(dtype)
        assert isinstance(values, np.ndarray), type(values)
        assert dtype == values.dtype
        assert freq is None or isinstance(freq, Tick)
        result = super()._simple_new(values=values, dtype=dtype)
        result._freq = freq
        return result

    @classmethod
    def _from_sequence(cls, data: Union[list, np.ndarray], *, dtype: Optional[np.dtype] = None, copy: bool = False) -> TimedeltaArray:
        if dtype:
            dtype = _validate_td64_dtype(dtype)
        data, freq = sequence_to_td64ns(data, copy=copy, unit=None)
        if dtype is not None:
            data = astype_overflowsafe(data, dtype=dtype, copy=False)
        return cls._simple_new(data, dtype=data.dtype, freq=freq)

    @classmethod
    def _from_sequence_not_strict(cls, data: Union[list, np.ndarray], *, dtype: Optional[np.dtype] = None, copy: bool = False, freq: Optional[Tick] = lib.no_default, unit: Optional[str] = None) -> TimedeltaArray:
        if dtype:
            dtype = _validate_td64_dtype(dtype)
        assert unit not in ['Y', 'y', 'M']
        data, inferred_freq = sequence_to_td64ns(data, copy=copy, unit=unit)
        if dtype is not None:
            data = astype_overflowsafe(data, dtype=dtype, copy=False)
        result = cls._simple_new(data, dtype=data.dtype, freq=inferred_freq)
        result._maybe_pin_freq(freq, {})
        return result

    @classmethod
    def _generate_range(cls, start: Optional[Union[str, Timedelta]], end: Optional[Union[str, Timedelta]], periods: Optional[int], freq: Optional[Tick], closed: Optional[str] = None, *, unit: Optional[str] = None) -> TimedeltaArray:
        periods = dtl.validate_periods(periods)
        if freq is None and any((x is None for x in [periods, start, end])):
            raise ValueError('Must provide freq argument if no data is supplied')
        if com.count_not_none(start, end, periods, freq) != 3:
            raise ValueError('Of the four parameters: start, end, periods, and freq, exactly three must be specified')
        if start is not None:
            start = Timedelta(start).as_unit('ns')
        if end is not None:
            end = Timedelta(end).as_unit('ns')
        if unit is not None:
            if unit not in ['s', 'ms', 'us', 'ns']:
                raise ValueError("'unit' must be one of 's', 'ms', 'us', 'ns'")
        else:
            unit = 'ns'
        if start is not None and unit is not None:
            start = start.as_unit(unit, round_ok=False)
        if end is not None and unit is not None:
            end = end.as_unit(unit, round_ok=False)
        left_closed, right_closed = validate_endpoints(closed)
        if freq is not None:
            index = generate_regular_range(start, end, periods, freq, unit=unit)
        else:
            index = np.linspace(start._value, end._value, periods).astype('i8')
        if not left_closed:
            index = index[1:]
        if not right_closed:
            index = index[:-1]
        td64values = index.view(f'm8[{unit}]')
        return cls._simple_new(td64values, dtype=td64values.dtype, freq=freq)

    def _unbox_scalar(self, value: Union[Timedelta, NaTType]) -> np.timedelta64:
        if not isinstance(value, self._scalar_type) and value is not NaT:
            raise ValueError("'value' should be a Timedelta.")
        self._check_compatible_with(value)
        if value is NaT:
            return np.timedelta64(value._value, self.unit)
        else:
            return value.as_unit(self.unit, round_ok=False).asm8

    def _scalar_from_string(self, value: str) -> Timedelta:
        return Timedelta(value)

    def _check_compatible_with(self, other: Union[Timedelta, NaTType]) -> None:
        pass

    def astype(self, dtype: Union[str, np.dtype], copy: bool = True) -> Union[TimedeltaArray, np.ndarray]:
        dtype = pandas_dtype(dtype)
        if lib.is_np_dtype(dtype, 'm'):
            if dtype == self.dtype:
                if copy:
                    return self.copy()
                return self
            if is_supported_dtype(dtype):
                res_values = astype_overflowsafe(self._ndarray, dtype, copy=False)
                return type(self)._simple_new(res_values, dtype=res_values.dtype, freq=self.freq)
            else:
                raise ValueError(f"Cannot convert from {self.dtype} to {dtype}. Supported resolutions are 's', 'ms', 'us', 'ns'")
        return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy=copy)

    def __iter__(self) -> Iterator:
        if self.ndim > 1:
            for i in range(len(self)):
                yield self[i]
        else:
            data = self._ndarray
            length = len(self)
            chunksize = 10000
            chunks = length // chunksize + 1
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, length)
                converted = ints_to_pytimedelta(data[start_i:end_i], box=True)
                yield from converted

    def sum(self, *, axis: Optional[AxisInt] = None, dtype: Optional[np.dtype] = None, out: Optional[np.ndarray] = None, keepdims: bool = False, initial: Optional[Union[int, float]] = None, skipna: bool = True, min_count: int = 0) -> Union[Timedelta, np.ndarray]:
        nv.validate_sum((), {'dtype': dtype, 'out': out, 'keepdims': keepdims, 'initial': initial})
        result = nanops.nansum(self._ndarray, axis=axis, skipna=skipna, min_count=min_count)
        return self._wrap_reduction_result(axis, result)

    def std(self, *, axis: Optional[AxisInt] = None, dtype: Optional[np.dtype] = None, out: Optional[np.ndarray] = None, ddof: int = 1, keepdims: bool = False, skipna: bool = True) -> Union[Timedelta, np.ndarray]:
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='std')
        result = nanops.nanstd(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        if axis is None or self.ndim == 1:
            return self._box_func(result)
        return self._from_backing_data(result)

    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs) -> TimedeltaArray:
        if name == 'cumsum':
            op = getattr(datetimelike_accumulations, name)
            result = op(self._ndarray.copy(), skipna=skipna, **kwargs)
            return type(self)._simple_new(result, freq=None, dtype=self.dtype)
        elif name == 'cumprod':
            raise TypeError('cumprod not supported for Timedelta.')
        else:
            return super()._accumulate(name, skipna=skipna, **kwargs)

    def _formatter(self, boxed: bool = False) -> callable:
        from pandas.io.formats.format import get_format_timedelta64
        return get_format_timedelta64(self, box=True)

    def _format_native_types(self, *, na_rep: str = 'NaT', date_format: Optional[str] = None, **kwargs) -> np.ndarray:
        from pandas.io.formats.format import get_format_timedelta64
        formatter = get_format_timedelta64(self, na_rep)
        return np.frompyfunc(formatter, 1, 1)(self._ndarray)

    def _add_offset(self, other: Union[Timedelta, NaTType]) -> None:
        assert not isinstance(other, Tick)
        raise TypeError(f'cannot add the type {type(other).__name__} to a {type(self).__name__}')

    @unpack_zerodim_and_defer('__mul__')
    def __mul__(self, other: Union[int, float, np.ndarray]) -> TimedeltaArray:
        if is_scalar(other):
            result = self._ndarray * other
            if result.dtype.kind != 'm':
                raise TypeError(f'Cannot multiply with {type(other).__name__}')
            freq = None
            if self.freq is not None and (not isna(other)):
                freq = self.freq * other
                if freq.n == 0:
                    freq = None
            return type(self)._simple_new(result, dtype=result.dtype, freq=freq)
        if not hasattr(other, 'dtype'):
            other = np.array(other)
        if len(other) != len(self) and (not lib.is_np_dtype(other.dtype, 'm')):
            raise ValueError('Cannot multiply with unequal lengths')
        if is_object_dtype(other.dtype):
            arr = self._ndarray
            result = [arr[n] * other[n] for n in range(len(self))]
            result = np.array(result)
            return type(self)._simple_new(result, dtype=result.dtype)
        result = self._ndarray * other
        if result.dtype.kind != 'm':
            raise TypeError(f'Cannot multiply with {type(other).__name__}')
        return type(self)._simple_new(result, dtype=result.dtype)
    __rmul__ = __mul__

    def _scalar_divlike_op(self, other: Union[Timedelta, NaTType, int, float], op: callable) -> Union[np.ndarray, TimedeltaArray]:
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
            if cast('Timedelta | NaTType', other) is NaT:
                res = np.empty(self.shape, dtype=np.float64)
                res.fill(np.nan)
                return res
            return op(self._ndarray, other)
        else:
            if op in [roperator.rtruediv, roperator.rfloordiv]:
                raise TypeError(f'Cannot divide {type(other).__name__} by {type(self).__name__}')
            result = op(self._ndarray, other)
            freq = None
            if self.freq is not None:
                freq = self.freq / other
                if freq.nanos == 0 and self.freq.nanos != 0:
                    freq = None
            return type(self)._simple_new(result, dtype=result.dtype, freq=freq)

    def _cast_divlike_op(self, other: Union[np.ndarray, list]) -> np.ndarray:
        if not hasattr(other, 'dtype'):
            other = np.array(other)
        if len(other) != len(self):
            raise ValueError('Cannot divide vectors with unequal lengths')
        return other

    def _vector_divlike_op(self, other: np.ndarray, op: callable) -> Union[np.ndarray, TimedeltaArray]:
        result = op(self._ndarray, np.asarray(other))
        if (is_integer_dtype(other.dtype) or is_float_dtype(other.dtype)) and op in [operator.truediv, operator.floordiv]:
            return type(self)._simple_new(result, dtype=result.dtype)
        if op in [operator.floordiv, roperator.rfloordiv]:
            mask = self.isna() | isna(other)
            if mask.any():
                result = result.astype(np.float64)
                np.putmask(result, mask, np.nan)
        return result

    @unpack_zerodim_and_defer('__truediv__')
    def __truediv__(self, other: Union[int, float, np.ndarray]) -> Union[np.ndarray, TimedeltaArray]:
        op = operator.truediv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)
        other = self._cast_divlike_op(other)
        if lib.is_np_dtype(other.dtype, 'm') or is_integer_dtype(other.dtype) or is_float_dtype(other.dtype):
            return self._vector_divlike_op(other, op)
        if is_object_dtype(other.dtype):
            other = np.asarray(other)
            if self.ndim > 1:
                res_cols = [left / right for left, right in zip(self, other)]
                res_cols2 = [x.reshape(1, -1) for x in res_cols]
                result = np.concatenate(res_cols2, axis=0)
            else:
                result = truediv_object_array(self._ndarray, other)
            return result
        else:
            return NotImplemented

    @unpack_zerodim_and_defer('__rtruediv__')
    def __rtruediv__(self, other: Union[int, float, np.ndarray]) -> Union[np.ndarray, TimedeltaArray]:
        op = roperator.rtruediv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)
        other = self._cast_divlike_op(other)
        if lib.is_np_dtype(other.dtype, 'm'):
            return self._vector_divlike_op(other, op)
        elif is_object_dtype(other.dtype):
            result_list = [other[n] / self[n] for n in range(len(self))]
            return np.array(result_list)
        else:
            return NotImplemented

    @unpack_zerodim_and_defer('__floordiv__')
    def __floordiv__(self, other: Union[int, float, np.ndarray]) -> Union[np.ndarray, TimedeltaArray]:
        op = operator.floordiv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)
        other = self._cast_divlike_op(other)
        if lib.is_np_dtype(other.dtype, 'm') or is_integer_dtype(other.dtype) or is_float_dtype(other.dtype):
            return self._vector_divlike_op(other, op)
        elif is_object_dtype(other.dtype):
            other = np.asarray(other)
            if self.ndim > 1:
                res_cols = [left // right for left, right in zip(self, other)]
                res_cols2 = [x.reshape(1, -1) for x in res_cols]
                result = np.concatenate(res_cols2, axis=0)
            else:
                result = floordiv_object_array(self._ndarray, other)
            assert result.dtype == object
            return result
        else:
            return NotImplemented

    @unpack_zerodim_and_defer('__rfloordiv__')
    def __rfloordiv__(self, other: Union[int, float, np.ndarray]) -> Union[np.ndarray, TimedeltaArray]:
        op = roperator.rfloordiv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)
        other = self._cast_divlike_op(other)
        if lib.is_np_dtype(other.dtype, 'm'):
            return self._vector_divlike_op(other, op)
        elif is_object_dtype(other.dtype):
            result_list = [other[n] // self[n] for n in range(len(self))]
            result = np.array(result_list)
            return result
        else:
            return NotImplemented

    @unpack_zerodim_and_defer('__mod__')
    def __mod__(self, other: Union[Timedelta, NaTType, int, float]) -> TimedeltaArray:
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        return self - self // other * other

    @unpack_zerodim_and_defer('__rmod__')
    def __rmod__(self, other: Union[Timedelta, NaTType, int, float]) -> TimedeltaArray:
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        return other - other // self * self

    @unpack_zerodim_and_defer('__divmod__')
    def __divmod__(self, other: Union[Timedelta, NaTType, int, float]) -> Tuple[TimedeltaArray, TimedeltaArray]:
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        res1 = self // other
        res2 = self - res1 * other
        return (res1, res2)

    @unpack_zerodim_and_defer('__rdivmod__')
    def __rdivmod__(self, other: Union[Timedelta, NaTType, int, float]) -> Tuple[TimedeltaArray, TimedeltaArray]:
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        res1 = other // self
        res2 = other - res1 * self
        return (res1, res2)

    def __neg__(self) -> TimedeltaArray:
        freq = None
        if self.freq is not None:
            freq = -self.freq
        return type(self)._simple_new(-self._ndarray, dtype=self.dtype, freq=freq)

    def __pos__(self) -> TimedeltaArray:
        return type(self)._simple_new(self._ndarray.copy(), dtype=self.dtype, freq=self.freq)

    def __abs__(self) -> TimedeltaArray:
        return type(self)._simple_new(np.abs(self._ndarray), dtype=self.dtype)

    def total_seconds(self) -> np.ndarray:
        pps = periods_per_second(self._creso)
        return self._maybe_mask_results(self.asi8 / pps, fill_value=None)

    def to_pytimedelta(self) -> np.ndarray:
        return ints_to_pytimedelta(self._ndarray)
    days_docstring: str = textwrap.dedent('Number of days for each element.\n\n    See Also\n    --------\n    Series.dt.seconds : Return number of seconds for each element.\n    Series.dt.microseconds : Return number of microseconds for each element.\n    Series.dt.nanoseconds : Return number of nanoseconds for each element.\n\n    Examples\n    --------\n    For Series:\n\n    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit=\'D\'))\n    >>> ser\n    0   1 days\n    1   2 days\n    2   3 days\n    dtype: timedelta64[ns]\n    >>> ser.dt.days\n    0    1\n    1    2\n    2    3\n    dtype: int64\n\n    For TimedeltaIndex:\n\n    >>> tdelta_idx = pd.to_timedelta(["0 days", "10 days", "20 days"])\n    >>> tdelta_idx\n    TimedeltaIndex([\'0 days\', \'10 days\', \'20 days\'],\n                    dtype=\'timedelta64[ns]\', freq=None)\n    >>> tdelta_idx.days\n    Index([0, 10, 20], dtype=\'int64\')')
    days: property = _field_accessor('days', 'days', days_docstring)
    seconds_docstring: str = textwrap.dedent("Number of seconds (>= 0 and less than 1 day) for each element.\n\n    See Also\n    --------\n    Series.dt.seconds : Return number of seconds for each element.\n    Series.dt.nanoseconds : Return number of nanoseconds for each element.\n\n    Examples\n    --------\n    For Series:\n\n    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='s'))\n    >>> ser\n    0   0 days 00:00:01\n    1   0 days 00:00:02\n    2   0 days 00:00:03\n    dtype: timedelta64[ns]\n    >>> ser.dt.seconds\n    0    1\n    1    2\n    2    3\n    dtype: int32\n\n    For TimedeltaIndex:\n\n    >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='s')\n    >>> tdelta_idx\n    TimedeltaIndex(['0 days 00:00:01', '0 days 00:00:02', '0 days 00:00:03'],\n                   dtype='timedelta64[ns]', freq=None)\n    >>> tdelta_idx.seconds\n    Index([1, 2, 3], dtype='int32')")
    seconds: property = _field_accessor('seconds', 'seconds', seconds_docstring)
    microseconds_docstring: str = textwrap.dedent("Number of microseconds (>= 0 and less than 1 second) for each element.\n\n    See Also\n    --------\n    pd.Timedelta.microseconds : Number of microseconds (>= 0 and less than 1 second).\n    pd.Timedelta.to_pytimedelta.microseconds : Number of microseconds (>= 0 and less\n        than 1 second) of a datetime.timedelta.\n\n    Examples\n    --------\n    For Series:\n\n    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='us'))\n    >>> ser\n    0   0 days 00:00:00.000001\n    1   0 days 00:00:00.000002\n    2   0 days 00:00:00.000003\n    dtype: timedelta64[ns]\n    >>> ser.dt.microseconds\n    0    1\n    1    2\n    2    3\n    dtype: int32\n\n    For TimedeltaIndex:\n\n    >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='us')\n    >>> tdelta_idx\n    TimedeltaIndex(['0 days 00:00:00.000001', '0 days 00:00:00.000002',\n                    '0 days 00:00:00.000003'],\n                   dtype='timedelta64[ns]', freq=None)\n    >>> tdelta_idx.microseconds\n    Index([1, 2, 3], dtype='int32')")
    microseconds: property = _field_accessor('microseconds', 'microseconds', microseconds_docstring)
    nanoseconds_docstring: str = textwrap.dedent("Number of nanoseconds (>= 0 and less than 1 microsecond) for each element.\n\n    See Also\n    --------\n    Series.dt.seconds : Return number of seconds for each element.\n    Series.dt.microseconds : Return number of nanoseconds for each element.\n\n    Examples\n    --------\n    For Series:\n\n    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='ns'))\n    >>> ser\n    0   0 days 00:00:00.000000001\n    1   0 days 00:00:00.000000002\n    2   0 days 00:00:00.000000003\n    dtype: timedelta64[ns]\n    >>> ser.dt.nanoseconds\n    0    1\n    1    2\n    2    3\n    dtype: int32\n\n    For TimedeltaIndex:\n\n    >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='ns')\n    >>> tdelta_idx\n    TimedeltaIndex(['0 days 00:00:00.000000001', '0 days 00:00:00.000000002',\n                    '0 days 00:00:00.000000003'],\n                   dtype='timedelta64[ns]', freq=None)\n    >>> tdelta_idx.nanoseconds\n    Index([1, 2, 3], dtype='int32')")
    nanoseconds: property = _field_accessor('nanoseconds', 'nanoseconds', nanoseconds_docstring)

    @property
    def components(self) -> DataFrame:
        from pandas import DataFrame
        columns = ['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds', 'nanoseconds']
        hasnans = self._hasna
        if hasnans:
            def f(x: Timedelta) -> list:
                if isna(x):
                    return [np.nan] * len(columns)
                return x.components
        else:
            def f(x: Timedelta) -> list:
                return x.components
        result = DataFrame([f(x) for x in self], columns=columns)
        if not hasnans:
            result = result.astype('int64')
        return result

def sequence_to_td64ns(data: Union[list, np.ndarray], copy: bool = False, unit: Optional[str] = None, errors: str = 'raise') -> Tuple[np.ndarray, Optional[Tick]]:
    assert unit not in ['Y', 'y', 'M']
    inferred_freq = None
    if unit is not None:
        unit = parse_timedelta_unit(unit)
    data, copy = dtl.ensure_arraylike_for_datetimelike(data, copy, cls_name='TimedeltaArray')
    if isinstance(data, TimedeltaArray):
        inferred_freq = data.freq
    if data.dtype == object or is_string_dtype(data.dtype):
        data = _objects_to_td64ns(data, unit=unit, errors=errors)
        copy = False
    elif is_integer_dtype(data.dtype):
        data, copy_made = _ints_to_td64ns(data, unit=unit)
        copy = copy and (not copy_made)
    elif is_float_dtype(data.dtype):
        if isinstance(data.dtype, ExtensionDtype):
            mask = data._mask
            data = data._data
        else:
            mask = np.isnan(data)
        data = cast_from_unit_vectorized(data, unit or 'ns')
        data[mask] = iNaT
        data = data.view('m8[ns]')
        copy = False
    elif lib.is_np_dtype(data.dtype, 'm'):
        if not is_supported_dtype(data.dtype):
            new_dtype = get_supported_dtype(data.dtype)
            data = astype_overflowsafe(data, dtype=new_dtype, copy=False)
            copy = False
    else:
        raise TypeError(f'dtype {data.dtype} cannot be converted to timedelta64[ns]')
    if not copy:
        data = np.asarray(data)
    else:
        data = np.array(data, copy=copy)
    assert data.dtype.kind == 'm'
    assert data.dtype != 'm8'
    return (data, inferred_freq)

def _ints_to_td64ns(data: np.ndarray, unit: str = 'ns') -> Tuple[np.ndarray, bool]:
    copy_made = False
    unit = unit if unit is not None else 'ns'
    if data.dtype != np.int64:
        data = data.astype(np.int64)
        copy_made = True
    if unit != 'ns':
        dtype_str = f'timedelta64[{unit}]'
        data = data.view(dtype_str)
        data = astype_overflowsafe(data, dtype=TD64NS_DTYPE)
        copy_made = True
    else:
        data = data.view('timedelta64[ns]')
    return (data, copy_made)

def _objects_to_td64ns(data: Union[np.ndarray, list], unit: Optional[str] = None, errors: str = 'raise') -> np.ndarray:
    values = np.asarray(data, dtype=np.object_)
    result = array_to_timedelta64(values, unit=unit, errors=errors)
    return result.view('timedelta64[ns]')

def _validate_td64_dtype(dtype: Union[str, np.dtype]) -> np.dtype:
    dtype = pandas_dtype(dtype)
    if dtype == np.dtype('m8'):
        msg = "Passing in 'timedelta' dtype with no precision is not allowed. Please pass in 'timedelta64[ns]' instead."
        raise ValueError(msg)
    if not lib.is_np_dtype(dtype, 'm'):
        raise ValueError(f"dtype '{dtype}' is invalid, should be np.timedelta64 dtype")
    elif not is_supported_dtype(dtype):
        raise ValueError("Supported timedelta64 resolutions are 's', 'ms', 'us', 'ns'")
    return dtype

from __future__ import annotations
from datetime import timedelta
import operator
from typing import TYPE_CHECKING, cast, Any, overload
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
    """
    Pandas ExtensionArray for timedelta data.

    .. warning::

       TimedeltaArray is currently experimental, and its API may change
       without warning. In particular, :attr:`TimedeltaArray.dtype` is
       expected to change to be an instance of an ``ExtensionDtype``
       subclass.

    Parameters
    ----------
    data : array-like
        The timedelta data.
    dtype : numpy.dtype
        Currently, only ``numpy.dtype("timedelta64[ns]")`` is accepted.
    freq : Offset, optional
        Frequency of the data.
    copy : bool, default False
        Whether to copy the underlying array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    Timedelta : Represents a duration, the difference between two dates or times.
    TimedeltaIndex : Immutable Index of timedelta64 data.
    to_timedelta : Convert argument to timedelta.

    Examples
    --------
    >>> pd.arrays.TimedeltaArray._from_sequence(pd.TimedeltaIndex(["1h", "2h"]))
    <TimedeltaArray>
    ['0 days 01:00:00', '0 days 02:00:00']
    Length: 2, dtype: timedelta64[ns]
    """
    _typ: str = 'timedeltaarray'
    _internal_fill_value: np.timedelta64 = np.timedelta64('NaT', 'ns')
    _recognized_scalars: tuple[type, ...] = (timedelta, np.timedelta64, Tick)
    _is_recognized_dtype = lambda x: lib.is_np_dtype(x, 'm')
    _infer_matches: tuple[str, ...] = ('timedelta', 'timedelta64')

    @property
    def _scalar_type(self) -> type[Timedelta]:
        return Timedelta
    __array_priority__: int = 1000
    _other_ops: list[str] = []
    _bool_ops: list[str] = []
    _field_ops: list[str] = ['days', 'seconds', 'microseconds', 'nanoseconds']
    _datetimelike_ops: list[str] = _field_ops + _bool_ops + ['unit', 'freq']
    _datetimelike_methods: list[str] = ['to_pytimedelta', 'total_seconds', 'round', 'floor', 'ceil', 'as_unit']

    def _box_func(self, x: np.timedelta64) -> Timedelta | NaTType:
        y = x.view('i8')
        if y == NaT._value:
            return NaT
        return Timedelta._from_value_and_reso(y, reso=self._creso)

    @property
    def dtype(self) -> np.dtype[np.timedelta64]:
        """
        The dtype for the TimedeltaArray.

        .. warning::

           A future version of pandas will change dtype to be an instance
           of a :class:`pandas.api.extensions.ExtensionDtype` subclass,
           not a ``numpy.dtype``.

        Returns
        -------
        numpy.dtype
        """
        return self._ndarray.dtype
    _freq: Tick | None = None

    @classmethod
    def _validate_dtype(cls, values: np.ndarray, dtype: DtypeObj | None) -> DtypeObj:
        dtype = _validate_td64_dtype(dtype)
        _validate_td64_dtype(values.dtype)
        if dtype != values.dtype:
            raise ValueError('Values resolution does not match dtype.')
        return dtype

    @classmethod
    def _simple_new(cls, values: npt.NDArray[np.timedelta64], freq: Tick | None = None, dtype: np.dtype[np.timedelta64] = TD64NS_DTYPE) -> Self:
        assert lib.is_np_dtype(dtype, 'm')
        assert not tslibs.is_unitless(dtype)
        assert isinstance(values, np.ndarray), type(values)
        assert dtype == values.dtype
        assert freq is None or isinstance(freq, Tick)
        result = super()._simple_new(values=values, dtype=dtype)
        result._freq = freq
        return result

    @classmethod
    def _from_sequence(cls, data: Any, *, dtype: DtypeObj | None = None, copy: bool = False) -> Self:
        if dtype:
            dtype = _validate_td64_dtype(dtype)
        (data, freq) = sequence_to_td64ns(data, copy=copy, unit=None)
        if dtype is not None:
            data = astype_overflowsafe(data, dtype=dtype, copy=False)
        return cls._simple_new(data, dtype=data.dtype, freq=freq)

    @classmethod
    def _from_sequence_not_strict(cls, data: Any, *, dtype: DtypeObj | None = None, copy: bool = False, freq: Any = lib.no_default, unit: str | None = None) -> Self:
        """
        _from_sequence_not_strict but without responsibility for finding the
        result's `freq`.
        """
        if dtype:
            dtype = _validate_td64_dtype(dtype)
        assert unit not in ['Y', 'y', 'M']
        (data, inferred_freq) = sequence_to_td64ns(data, copy=copy, unit=unit)
        if dtype is not None:
            data = astype_overflowsafe(data, dtype=dtype, copy=False)
        result = cls._simple_new(data, dtype=data.dtype, freq=inferred_freq)
        result._maybe_pin_freq(freq, {})
        return result

    @classmethod
    def _generate_range(cls, start: Timedelta | None, end: Timedelta | None, periods: int | None, freq: Tick | None, closed: str | None = None, *, unit: str | None = None) -> Self:
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
        (left_closed, right_closed) = validate_endpoints(closed)
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

    def _unbox_scalar(self, value: Any) -> np.timedelta64:
        if not isinstance(value, self._scalar_type) and value is not NaT:
            raise ValueError("'value' should be a Timedelta.")
        self._check_compatible_with(value)
        if value is NaT:
            return np.timedelta64(value._value, self.unit)
        else:
            return value.as_unit(self.unit, round_ok=False).asm8

    def _scalar_from_string(self, value: str) -> Timedelta | NaTType:
        return Timedelta(value)

    def _check_compatible_with(self, other: Any) -> None:
        pass

    def astype(self, dtype: DtypeObj, copy: bool = True) -> Any:
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

    def __iter__(self) -> Iterator[Timedelta | NaTType]:
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

    def sum(self, *, axis: AxisInt | None = None, dtype: NpDtype | None = None, out: Any = None, keepdims: bool = False, initial: Any = None, skipna: bool = True, min_count: int = 0) -> Timedelta | NaTType | Self:
        nv.validate_sum((), {'dtype': dtype, 'out': out, 'keepdims': keepdims, 'initial': initial})
        result = nanops.nansum(self._ndarray, axis=axis, skipna=skipna, min_count=min_count)
        return self._wrap_reduction_result(axis, result)

    def std(self, *, axis: AxisInt | None = None, dtype: NpDtype | None = None, out: Any = None, ddof: int = 1, keepdims: bool = False, skipna: bool = True) -> Timedelta | NaTType | Self:
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='std')
        result = nanops.nanstd(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        if axis is None or self.ndim == 1:
            return self._box_func(result)
        return self._from_backing_data(result)

    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs: Any) -> Self:
        if name == 'cumsum':
            op = getattr(datetimelike_accumulations, name)
            result = op(self._ndarray.copy(), skipna=skipna, **kwargs)
            return type(self)._simple_new(result, freq=None, dtype=self.dtype)
        elif name == 'cumprod':
            raise TypeError('cumprod not supported for Timedelta.')
        else:
            return super()._accumulate(name, skipna=skipna, **kwargs)

    def _formatter(self, boxed: bool = False) -> Any:
        from pandas.io.formats.format import get_format_timedelta64
        return get_format_timedelta64(self, box=True)

    def _format_native_types(self, *, na_rep: str | float = 'NaT', date_format: str | None = None, **kwargs: Any) -> npt.NDArray[np.object_]:
        from pandas.io.formats.format import get_format_timedelta64
        formatter = get_format_timedelta64(self, na_rep)
        return np.frompyfunc(formatter, 1, 1)(self._ndarray)

    def _add_offset(self, other: Any) -> None:
        assert not isinstance(other, Tick)
        raise TypeError(f'cannot add the type {type(other).__name__} to a {type(self).__name__}')

    @unpack_zerodim_and_defer('__mul__')
    def __mul__(self, other: Any) -> Self:
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

    def _scalar_divlike_op(self, other: Any, op: Any) -> Any:
        """
        Shared logic for __truediv__, __rtruediv__, __floordiv__, __rfloordiv__
        with scalar 'other'.
        """
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

    def _cast_divlike_op(self, other: Any) -> np.ndarray:
        if not hasattr(other, 'dtype'):
            other = np.array(other)
        if len(other) != len(self):
            raise ValueError('Cannot divide vectors with unequal lengths')
        return other

    def _vector_divlike_op(self, other: Any, op: Any) -> np.ndarray | Self:
        """
        Shared logic for __truediv__, __floordiv__, and their reversed versions
        with timedelta64-dtype ndarray other.
        """
        result = op(self._ndarray, np.asarray(other))
        if (is_integer_dtype(other.dtype) or is_float_dtype(other.dtype)) and op in [operator
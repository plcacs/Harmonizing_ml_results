from __future__ import annotations
from datetime import timedelta
import operator
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload, Union, Optional, List, Tuple, Dict, Sequence
import warnings
import numpy as np
from pandas._libs import algos as libalgos, lib
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import BaseOffset, NaT, NaTType, Timedelta, add_overflowsafe, astype_overflowsafe, dt64arr_to_periodarr as c_dt64arr_to_periodarr, get_unit_from_dtype, iNaT, parsing, period as libperiod, to_offset
from pandas._libs.tslibs.dtypes import FreqGroup, PeriodDtypeBase
from pandas._libs.tslibs.fields import isleapyear_arr
from pandas._libs.tslibs.offsets import Tick, delta_to_tick
from pandas._libs.tslibs.period import DIFFERENT_FREQ, IncompatibleFrequency, Period, get_period_field_arr, period_asfreq_arr
from pandas.util._decorators import cache_readonly, doc
from pandas.core.dtypes.common import ensure_object, pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype, PeriodDtype
from pandas.core.dtypes.generic import ABCIndex, ABCPeriodIndex, ABCSeries, ABCTimedeltaArray
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import datetimelike as dtl
import pandas.core.common as com
if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pandas._typing import AnyArrayLike, Dtype, FillnaOptions, NpDtype, NumpySorter, NumpyValueArrayLike, Self, npt
    from pandas.core.dtypes.dtypes import ExtensionDtype
    from pandas.core.arrays import DatetimeArray, TimedeltaArray
    from pandas.core.arrays.base import ExtensionArray
    from pyarrow import Array as ArrowArray

BaseOffsetT = TypeVar('BaseOffsetT', bound=BaseOffset)
_shared_doc_kwargs: Dict[str, str] = {'klass': 'PeriodArray'}

def _field_accessor(name: str, docstring: Optional[str] = None) -> property:

    def f(self: PeriodArray) -> np.ndarray:
        base = self.dtype._dtype_code
        result = get_period_field_arr(name, self.asi8, base)
        return result
    f.__name__ = name
    f.__doc__ = docstring
    return property(f)

class PeriodArray(dtl.DatelikeOps, libperiod.PeriodMixin):
    __array_priority__: int = 1000
    _typ: str = 'periodarray'
    _internal_fill_value: np.int64 = np.int64(iNaT)
    _recognized_scalars: Tuple[type] = (Period,)
    _is_recognized_dtype: Callable[[Any], bool] = lambda x: isinstance(x, PeriodDtype)
    _infer_matches: Tuple[str, ...] = ('period',)

    @property
    def _scalar_type(self) -> type:
        return Period
    _other_ops: List[str] = []
    _bool_ops: List[str] = ['is_leap_year']
    _object_ops: List[str] = ['start_time', 'end_time', 'freq']
    _field_ops: List[str] = ['year', 'month', 'day', 'hour', 'minute', 'second', 'weekofyear', 'weekday', 'week', 'dayofweek', 'day_of_week', 'dayofyear', 'day_of_year', 'quarter', 'qyear', 'days_in_month', 'daysinmonth']
    _datetimelike_ops: List[str] = _field_ops + _object_ops + _bool_ops
    _datetimelike_methods: List[str] = ['strftime', 'to_timestamp', 'asfreq']

    def __init__(self, values: Union[PeriodArray, ABCSeries, np.ndarray, ABCPeriodIndex], dtype: Optional[PeriodDtype] = None, copy: bool = False) -> None:
        if dtype is not None:
            dtype = pandas_dtype(dtype)
            if not isinstance(dtype, PeriodDtype):
                raise ValueError(f'Invalid dtype {dtype} for PeriodArray')
        if isinstance(values, ABCSeries):
            values = values._values
            if not isinstance(values, type(self)):
                raise TypeError('Incorrect dtype')
        elif isinstance(values, ABCPeriodIndex):
            values = values._values
        if isinstance(values, type(self)):
            if dtype is not None and dtype != values.dtype:
                raise raise_on_incompatible(values, dtype.freq)
            values, dtype = (values._ndarray, values.dtype)
        if not copy:
            values = np.asarray(values, dtype='int64')
        else:
            values = np.array(values, dtype='int64', copy=copy)
        if dtype is None:
            raise ValueError('dtype is not specified and cannot be inferred')
        dtype = cast(PeriodDtype, dtype)
        NDArrayBacked.__init__(self, values, dtype)

    @classmethod
    def _simple_new(cls, values: np.ndarray, dtype: PeriodDtype) -> PeriodArray:
        assertion_msg = 'Should be numpy array of type i8'
        assert isinstance(values, np.ndarray) and values.dtype == 'i8', assertion_msg
        return cls(values, dtype=dtype)

    @classmethod
    def _from_sequence(cls, scalars: Sequence[Any], *, dtype: Optional[PeriodDtype] = None, copy: bool = False) -> PeriodArray:
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        if dtype and isinstance(dtype, PeriodDtype):
            freq = dtype.freq
        else:
            freq = None
        if isinstance(scalars, cls):
            validate_dtype_freq(scalars.dtype, freq)
            if copy:
                scalars = scalars.copy()
            return scalars
        periods = np.asarray(scalars, dtype=object)
        freq = freq or libperiod.extract_freq(periods)
        ordinals = libperiod.extract_ordinals(periods, freq)
        dtype = PeriodDtype(freq)
        return cls(ordinals, dtype=dtype)

    @classmethod
    def _from_sequence_of_strings(cls, strings: Sequence[str], *, dtype: PeriodDtype, copy: bool = False) -> PeriodArray:
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _from_datetime64(cls, data: np.ndarray, freq: Union[str, Tick], tz: Optional[Any] = None) -> PeriodArray:
        if isinstance(freq, BaseOffset):
            freq = PeriodDtype(freq)._freqstr
        data, freq = dt64arr_to_periodarr(data, freq, tz)
        dtype = PeriodDtype(freq)
        return cls(data, dtype=dtype)

    @classmethod
    def _generate_range(cls, start: Optional[Period], end: Optional[Period], periods: Optional[int], freq: Optional[Union[str, Tick]]) -> Tuple[np.ndarray, Any]:
        periods = dtl.validate_periods(periods)
        if freq is not None:
            freq = Period._maybe_convert_freq(freq)
        if start is not None or end is not None:
            subarr, freq = _get_ordinal_range(start, end, periods, freq)
        else:
            raise ValueError('Not enough parameters to construct Period range')
        return (subarr, freq)

    @classmethod
    def _from_fields(cls, *, fields: Dict[str, Any], freq: Any) -> PeriodArray:
        subarr, freq = _range_from_fields(freq=freq, **fields)
        dtype = PeriodDtype(freq)
        return cls._simple_new(subarr, dtype=dtype)

    def _unbox_scalar(self, value: Union[NaTType, Period, Any]) -> np.int64:
        if value is NaT:
            return np.int64(value._value)
        elif isinstance(value, self._scalar_type):
            self._check_compatible_with(value)
            return np.int64(value.ordinal)
        else:
            raise ValueError(f"'value' should be a Period. Got '{value}' instead.")

    def _scalar_from_string(self, value: str) -> Period:
        return Period(value, freq=self.freq)

    def _check_compatible_with(self, other: Period) -> None:
        if other is NaT:
            return
        self._require_matching_freq(other.freq)

    @cache_readonly
    def dtype(self) -> PeriodDtype:
        return self._dtype

    @property
    def freq(self) -> BaseOffset:
        return self.dtype.freq

    @property
    def freqstr(self) -> str:
        return PeriodDtype(self.freq)._freqstr

    def __array__(self, dtype: Optional[Union[np.dtype, str]] = None, copy: Optional[bool] = None) -> np.ndarray:
        if dtype == 'i8':
            if not copy:
                return np.asarray(self.asi8, dtype=dtype)
            else:
                return np.array(self.asi8, dtype=dtype)
        if copy is False:
            raise ValueError('Unable to avoid copy while creating an array as requested.')
        if dtype == bool:
            return ~self._isnan
        return np.array(list(self), dtype=object)

    def __arrow_array__(self, type: Optional[Any] = None) -> ArrowArray:
        import pyarrow
        from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
        if type is not None:
            if pyarrow.types.is_integer(type):
                return pyarrow.array(self._ndarray, mask=self.isna(), type=type)
            elif isinstance(type, ArrowPeriodType):
                if self.freqstr != type.freq:
                    raise TypeError(f"Not supported to convert PeriodArray to array with different 'freq' ({self.freqstr} vs {type.freq})")
            else:
                raise TypeError(f"Not supported to convert PeriodArray to '{type}' type")
        period_type = ArrowPeriodType(self.freqstr)
        storage_array = pyarrow.array(self._ndarray, mask=self.isna(), type='int64')
        return pyarrow.ExtensionArray.from_storage(period_type, storage_array)

    year = _field_accessor('year', '\n        The year of the period.\n\n        See Also\n        --------\n        PeriodIndex.day_of_year : The ordinal day of the year.\n        PeriodIndex.dayofyear : The ordinal day of the year.\n        PeriodIndex.is_leap_year : Logical indicating if the date belongs to a\n            leap year.\n        PeriodIndex.weekofyear : The week ordinal of the year.\n        PeriodIndex.year : The year of the period.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")\n        >>> idx.year\n        Index([2023, 2024, 2025], dtype=\'int64\')\n        ')
    month = _field_accessor('month', '\n        The month as January=1, December=12.\n\n        See Also\n        --------\n        PeriodIndex.days_in_month : The number of days in the month.\n        PeriodIndex.daysinmonth : The number of days in the month.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")\n        >>> idx.month\n        Index([1, 2, 3], dtype=\'int64\')\n        ')
    day = _field_accessor('day', "\n        The days of the period.\n\n        See Also\n        --------\n        PeriodIndex.day_of_week : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.day_of_year : The ordinal day of the year.\n        PeriodIndex.dayofweek : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.dayofyear : The ordinal day of the year.\n        PeriodIndex.days_in_month : The number of days in the month.\n        PeriodIndex.daysinmonth : The number of days in the month.\n        PeriodIndex.weekday : The day of the week with Monday=0, Sunday=6.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(['2020-01-31', '2020-02-28'], freq='D')\n        >>> idx.day\n        Index([31, 28], dtype='int64')\n        ")
    hour = _field_accessor('hour', '\n        The hour of the period.\n\n        See Also\n        --------\n        PeriodIndex.minute : The minute of the period.\n        PeriodIndex.second : The second of the period.\n        PeriodIndex.to_timestamp : Cast to DatetimeArray/Index.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01-01 10:00", "2023-01-01 11:00"], freq=\'h\')\n        >>> idx.hour\n        Index([10, 11], dtype=\'int64\')\n        ')
    minute = _field_accessor('minute', '\n        The minute of the period.\n\n        See Also\n        --------\n        PeriodIndex.hour : The hour of the period.\n        PeriodIndex.second : The second of the period.\n        PeriodIndex.to_timestamp : Cast to DatetimeArray/Index.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01-01 10:30:00",\n        ...                       "2023-01-01 11:50:00"], freq=\'min\')\n        >>> idx.minute\n        Index([30, 50], dtype=\'int64\')\n        ')
    second = _field_accessor('second', '\n        The second of the period.\n\n        See Also\n        --------\n        PeriodIndex.hour : The hour of the period.\n        PeriodIndex.minute : The minute of the period.\n        PeriodIndex.to_timestamp : Cast to DatetimeArray/Index.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01-01 10:00:30",\n        ...                       "2023-01-01 10:00:31"], freq=\'s\')\n        >>> idx.second\n        Index([30, 31], dtype=\'int64\')\n        ')
    weekofyear = _field_accessor('week', '\n        The week ordinal of the year.\n\n        See Also\n        --------\n        PeriodIndex.day_of_week : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.dayofweek : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.week : The week ordinal of the year.\n        PeriodIndex.weekday : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.year : The year of the period.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")\n        >>> idx.week  # It can be written `weekofyear`\n        Index([5, 9, 13], dtype=\'int64\')\n        ')
    week = weekofyear
    day_of_week = _field_accessor('day_of_week', '\n        The day of the week with Monday=0, Sunday=6.\n\n        See Also\n        --------\n        PeriodIndex.day : The days of the period.\n        PeriodIndex.day_of_week : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.day_of_year : The ordinal day of the year.\n        PeriodIndex.dayofweek : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.dayofyear : The ordinal day of the year.\n        PeriodIndex.week : The week ordinal of the year.\n        PeriodIndex.weekday : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.weekofyear : The week ordinal of the year.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01-01", "2023-01-02", "2023-01-03"], freq="D")\n        >>> idx.weekday\n        Index([6, 0, 1], dtype=\'int64\')\n        ')
    dayofweek = day_of_week
    weekday = dayofweek
    dayofyear = day_of_year = _field_accessor('day_of_year', '\n        The ordinal day of the year.\n\n        See Also\n        --------\n        PeriodIndex.day : The days of the period.\n        PeriodIndex.day_of_week : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.day_of_year : The ordinal day of the year.\n        PeriodIndex.dayofweek : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.dayofyear : The ordinal day of the year.\n        PeriodIndex.weekday : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.weekofyear : The week ordinal of the year.\n        PeriodIndex.year : The year of the period.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(["2023-01-10", "2023-02-01", "2023-03-01"], freq="D")\n        >>> idx.dayofyear\n        Index([10, 32, 60], dtype=\'int64\')\n\n        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")\n        >>>
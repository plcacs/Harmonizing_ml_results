from __future__ import annotations
from datetime import datetime, timedelta, tzinfo
from typing import TYPE_CHECKING, TypeVar, cast, overload, Any, Optional, Union, Tuple, List
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

_ITER_CHUNKSIZE: int = 10000

@overload
def tz_to_dtype(tz: Optional[tzinfo], unit: str = ...) -> np.dtype: ...

@overload
def tz_to_dtype(tz: Optional[tzinfo], unit: str = ...) -> DatetimeTZDtype: ...

def tz_to_dtype(tz: Optional[tzinfo], unit: str = 'ns') -> Union[np.dtype, DatetimeTZDtype]:
    """
    Return a datetime64[ns] dtype appropriate for the given timezone.

    Parameters
    ----------
    tz : tzinfo or None
    unit : str, default "ns"

    Returns
    -------
    np.dtype or Datetime64TZDtype
    """
    if tz is None:
        return np.dtype(f'M8[{unit}]')
    else:
        return DatetimeTZDtype(tz=tz, unit=unit)

def _field_accessor(name: str, field: str, docstring: Optional[str] = None) -> property:

    def f(self: DatetimeArray) -> np.ndarray:
        values: np.ndarray = self._local_timestamps()
        if field in self._bool_ops:
            result: np.ndarray
            if field.endswith(('start', 'end')):
                freq: Optional[BaseOffset] = self.freq
                month_kw: int = 12
                if freq:
                    kwds: dict = freq.kwds
                    month_kw = kwds.get('startingMonth', kwds.get('month', month_kw))
                if freq is not None:
                    freq_name: Optional[str] = freq.name
                else:
                    freq_name = None
                result = fields.get_start_end_field(values, field, freq_name, month_kw, reso=self._creso)
            else:
                result = fields.get_date_field(values, field, reso=self._creso)
            return result
        result = fields.get_date_field(values, field, reso=self._creso)
        result = self._maybe_mask_results(result, fill_value=None, convert='float64')
        return result
    f.__name__ = name
    f.__doc__ = docstring
    return property(f)

class DatetimeArray(dtl.TimelikeOps, dtl.DatelikeOps):
    """
    Pandas ExtensionArray for tz-naive or tz-aware datetime data.

    .. warning::

       DatetimeArray is currently experimental, and its API may change
       without warning. In particular, :attr:`DatetimeArray.dtype` is
       expected to change to always be an instance of an ``ExtensionDtype``
       subclass.

    Parameters
    ----------
    data : Series, Index, DatetimeArray, ndarray
        The datetime data.

        For DatetimeArray `values` (or a Series or Index boxing one),
        `dtype` and `freq` will be extracted from `values`.

    dtype : numpy.dtype or DatetimeTZDtype
        Note that the only NumPy dtype allowed is 'datetime64[ns]'.
    freq : str or Offset, optional
        The frequency.
    copy : bool, default False
        Whether to copy the underlying array of values.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    DatetimeIndex : Immutable Index for datetime-like data.
    Series : One-dimensional labeled array capable of holding datetime-like data.
    Timestamp : Pandas replacement for python datetime.datetime object.
    to_datetime : Convert argument to datetime.
    period_range : Return a fixed frequency PeriodIndex.

    Examples
    --------
    >>> pd.arrays.DatetimeArray._from_sequence(
    ...     pd.DatetimeIndex(["2023-01-01", "2023-01-02"], freq="D")
    ... )
    <DatetimeArray>
    ['2023-01-01 00:00:00', '2023-01-02 00:00:00']
    Length: 2, dtype: datetime64[s]
    """
    _typ: str = 'datetimearray'
    _internal_fill_value: np.datetime64 = np.datetime64('NaT', 'ns')
    _recognized_scalars: Tuple[type, type] = (datetime, np.datetime64)
    _is_recognized_dtype = lambda self, x: lib.is_np_dtype(x, 'M') or isinstance(x, DatetimeTZDtype)
    _infer_matches: Tuple[str, ...] = ('datetime', 'datetime64', 'date')

    @property
    def _scalar_type(self) -> type[Timestamp]:
        return Timestamp
    _bool_ops: List[str] = ['is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end', 'is_leap_year']
    _field_ops: List[str] = ['year', 'month', 'day', 'hour', 'minute', 'second', 'weekday', 'dayofweek', 'day_of_week', 'dayofyear', 'day_of_year', 'quarter', 'days_in_month', 'daysinmonth', 'microsecond', 'nanosecond']
    _other_ops: List[str] = ['date', 'time', 'timetz']
    _datetimelike_ops: List[str] = _field_ops + _bool_ops + _other_ops + ['unit', 'freq', 'tz']
    _datetimelike_methods: List[str] = ['to_period', 'tz_localize', 'tz_convert', 'normalize', 'strftime', 'round', 'floor', 'ceil', 'month_name', 'day_name', 'as_unit']
    __array_priority__: int = 1000
    _dtype: Union[np.dtype, DatetimeTZDtype]
    _freq: Optional[BaseOffset] = None

    @classmethod
    def _from_scalars(cls, scalars: Any, *, dtype: DtypeObj) -> Self:
        if lib.infer_dtype(scalars, skipna=True) not in ['datetime', 'datetime64']:
            raise ValueError
        return cls._from_sequence(scalars, dtype=dtype)

    @classmethod
    def _validate_dtype(cls, values: Any, dtype: DtypeObj) -> DtypeObj:
        dtype = _validate_dt64_dtype(dtype)
        _validate_dt64_dtype(values.dtype)
        if isinstance(dtype, np.dtype):
            if values.dtype != dtype:
                raise ValueError('Values resolution does not match dtype.')
        else:
            vunit: str = np.datetime_data(values.dtype)[0]
            if vunit != dtype.unit:
                raise ValueError('Values resolution does not match dtype.')
        return dtype

    @classmethod
    def _simple_new(cls, values: np.ndarray, freq: Optional[BaseOffset] = None, dtype: Union[np.dtype, DatetimeTZDtype] = DT64NS_DTYPE) -> Self:
        assert isinstance(values, np.ndarray)
        assert dtype.kind == 'M'
        if isinstance(dtype, np.dtype):
            assert dtype == values.dtype
            assert not is_unitless(dtype)
        else:
            assert dtype._creso == get_unit_from_dtype(values.dtype)
        result: Self = super()._simple_new(values, dtype)
        result._freq = freq
        return result

    @classmethod
    def _from_sequence(cls, scalars: Any, *, dtype: Optional[DtypeObj] = None, copy: bool = False) -> Self:
        return cls._from_sequence_not_strict(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_sequence_not_strict(cls, data: Any, *, dtype: Optional[DtypeObj] = None, copy: bool = False, tz: Any = lib.no_default, freq: Union[str, BaseOffset, lib.NoDefault, None] = lib.no_default, dayfirst: bool = False, yearfirst: bool = False, ambiguous: TimeAmbiguous = 'raise') -> Self:
        """
        A non-strict version of _from_sequence, called from DatetimeIndex.__new__.
        """
        explicit_tz_none: bool = tz is None
        if tz is lib.no_default:
            tz = None
        else:
            tz = timezones.maybe_get_tz(tz)
        dtype = _validate_dt64_dtype(dtype)
        tz = _validate_tz_from_dtype(dtype, tz, explicit_tz_none)
        unit: Optional[str] = None
        if dtype is not None:
            unit = dtl.dtype_to_unit(dtype)
        (data, copy) = dtl.ensure_arraylike_for_datetimelike(data, copy, cls_name='DatetimeArray')
        inferred_freq: Optional[BaseOffset] = None
        if isinstance(data, DatetimeArray):
            inferred_freq = data.freq
        (subarr, tz) = _sequence_to_dt64(data, copy=copy, tz=tz, dayfirst=dayfirst, yearfirst=yearfirst, ambiguous=ambiguous, out_unit=unit)
        _validate_tz_from_dtype(dtype, tz, explicit_tz_none)
        if tz is not None and explicit_tz_none:
            raise ValueError("Passed data is timezone-aware, incompatible with 'tz=None'. Use obj.tz_localize(None) instead.")
        data_unit: str = np.datetime_data(subarr.dtype)[0]
        data_dtype: Union[np.dtype, DatetimeTZDtype] = tz_to_dtype(tz, data_unit)
        result: Self = cls._simple_new(subarr, freq=inferred_freq, dtype=data_dtype)
        if unit is not None and unit != result.unit:
            result = result.as_unit(unit)
        validate_kwds: dict = {'ambiguous': ambiguous}
        result._maybe_pin_freq(freq, validate_kwds)
        return result

    @classmethod
    def _generate_range(cls, start: Optional[Union[Timestamp, str]], end: Optional[Union[Timestamp, str]], periods: Optional[int], freq: Any, tz: Optional[tzinfo] = None, normalize: bool = False, ambiguous: TimeAmbiguous = 'raise', nonexistent: TimeNonexistent = 'raise', inclusive: IntervalClosedType = 'both', *, unit: Optional[str] = None) -> Self:
        periods = dtl.validate_periods(periods)
        if freq is None and any((x is None for x in [periods, start, end])):
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
        (left_inclusive, right_inclusive) = validate_inclusive(inclusive)
        (start, end) = _maybe_normalize_endpoints(start, end, normalize)
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
                i8values: np.ndarray = generate_regular_range(start, end, periods, freq, unit=unit)
            else:
                xdr: Generator[Timestamp] = _generate_range(start=start, end=end, periods=periods, offset=freq, unit=unit)
                i8values = np.array([x._value for x in xdr], dtype=np.int64)
            endpoint_tz: Optional[tzinfo] = start.tz if start is not None else end.tz
            if tz is not None and endpoint_tz is None:
                if not timezones.is_utc(tz):
                    creso: int = abbrev_to_npy_unit(unit)
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
            if not left_inclusive and (not right_inclusive):
                i8values = i8values[1:-1]
        else:
            start_i8: int = Timestamp(start)._value
            end_i8: int = Timestamp(end)._value
            if not left_inclusive or not right_inclusive:
                if not left_inclusive and len(i8values) and (i8values[0] == start_i8):
                    i8values = i8values[1:]
                if not right_inclusive and len(i8values) and (i8values[-1] == end_i8):
                    i8values = i8values[:-1]
        dt64_values: np.ndarray = i8values.view(f'datetime64[{unit}]')
        dtype: Union[np.dtype, DatetimeTZDtype] = tz_to_dtype(tz, unit=unit)
        return cls._simple_new(dt64_values, freq=freq, dtype=dtype)

    def _unbox_scalar(self, value: Any) -> np.datetime64:
        if not isinstance(value, self._scalar_type) and value is not NaT:
            raise ValueError("'value' should be a Timestamp.")
        self._check_compatible_with(value)
        if value is NaT:
            return np.datetime64(value._value, self.unit)
        else:
            return value.as_unit(self.unit, round_ok=False).asm8

    def _scalar_from_string(self, value: str) -> Union[Timestamp, NaTType]:
        return Timestamp(value, tz=self.tz)

    def _check_compatible_with(self, other: Any) -> None:
        if other is NaT:
            return
        self._assert_tzawareness_compat(other)

    def _box_func(self, x: np.datetime64) -> Union[Timestamp, NaTType]:
        value: int = x.view('i8')
        ts: Union[Timestamp, NaTType] = Timestamp._from_value_and_reso(value, reso=self._creso, tz=self.tz)
        return ts

    @property
    def dtype(self) -> Union[np.dtype, DatetimeTZDtype]:
        """
        The dtype for the DatetimeArray.

        .. warning::

           A future version of pandas will change dtype to never be a
           ``numpy.dtype``. Instead, :attr:`DatetimeArray.dtype` will
           always be an instance of an ``ExtensionDtype`` subclass.

        Returns
        -------
        numpy.dtype or DatetimeTZDtype
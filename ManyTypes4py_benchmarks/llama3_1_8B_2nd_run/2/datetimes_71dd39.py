from __future__ import annotations
from datetime import datetime, timedelta, tzinfo
from typing import TYPE_CHECKING, TypeVar, cast, overload
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
def tz_to_dtype(tz: tzinfo | None, unit: str | None = ...) -> np.dtype | DatetimeTZDtype:
    ...

@overload
def tz_to_dtype(tz: tzinfo | None, unit: str | None = ...) -> np.dtype | DatetimeTZDtype:
    ...

def tz_to_dtype(tz: tzinfo | None, unit: str | None = 'ns') -> np.dtype | DatetimeTZDtype:
    """
    Return a datetime64[ns] dtype appropriate for the given timezone.

    Parameters
    ----------
    tz : tzinfo or None
    unit : str, default "ns"

    Returns
    -------
    np.dtype or DatetimeTZDtype
    """
    if tz is None:
        return np.dtype(f'M8[{unit}]')
    else:
        return DatetimeTZDtype(tz=tz, unit=unit)

def _field_accessor(name: str, field: str, docstring: str | None = None) -> property:
    """
    Create a property accessor for a field.

    Parameters
    ----------
    name : str
    field : str
    docstring : str | None, default None

    Returns
    -------
    property
    """
    def f(self) -> np.ndarray:
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
    _typ = 'datetimearray'
    _internal_fill_value = np.datetime64('NaT', 'ns')
    _recognized_scalars = (datetime, np.datetime64)
    _is_recognized_dtype = lambda x: lib.is_np_dtype(x, 'M') or isinstance(x, DatetimeTZDtype)
    _infer_matches = ('datetime', 'datetime64', 'date')

    @property
    def _scalar_type(self) -> type[Timestamp]:
        return Timestamp

    _bool_ops: list[str] = ['is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end', 'is_leap_year']
    _field_ops: list[str] = ['year', 'month', 'day', 'hour', 'minute', 'second', 'weekday', 'dayofweek', 'day_of_week', 'dayofyear', 'day_of_year', 'quarter', 'days_in_month', 'daysinmonth', 'microsecond', 'nanosecond']
    _other_ops: list[str] = ['date', 'time', 'timetz']
    _datetimelike_ops: list[str] = _field_ops + _bool_ops + _other_ops + ['unit', 'freq', 'tz']
    _datetimelike_methods: list[str] = ['to_period', 'tz_localize', 'tz_convert', 'normalize', 'strftime', 'round', 'floor', 'ceil', 'month_name', 'day_name', 'as_unit']
    __array_priority__ = 1000
    _freq: BaseOffset | None = None

    @classmethod
    def _from_scalars(cls: type['DatetimeArray'], scalars: np.ndarray, *, dtype: np.dtype | DatetimeTZDtype) -> 'DatetimeArray':
        """
        Create a DatetimeArray from scalars.

        Parameters
        ----------
        scalars : np.ndarray
        dtype : np.dtype | DatetimeTZDtype

        Returns
        -------
        DatetimeArray
        """
        if lib.infer_dtype(scalars, skipna=True) not in ['datetime', 'datetime64']:
            raise ValueError
        return cls._from_sequence(scalars, dtype=dtype)

    @classmethod
    def _validate_dtype(cls: type['DatetimeArray'], values: np.ndarray, dtype: np.dtype | DatetimeTZDtype) -> np.dtype | DatetimeTZDtype:
        """
        Validate the dtype of a DatetimeArray.

        Parameters
        ----------
        values : np.ndarray
        dtype : np.dtype | DatetimeTZDtype

        Returns
        -------
        np.dtype | DatetimeTZDtype

        Raises
        ------
        ValueError
        """
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
    def _simple_new(cls: type['DatetimeArray'], values: np.ndarray, freq: BaseOffset | None, dtype: np.dtype | DatetimeTZDtype) -> 'DatetimeArray':
        """
        Create a new DatetimeArray.

        Parameters
        ----------
        values : np.ndarray
        freq : BaseOffset | None
        dtype : np.dtype | DatetimeTZDtype

        Returns
        -------
        DatetimeArray
        """
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
    def _from_sequence(cls: type['DatetimeArray'], scalars: np.ndarray | ArrayLike, *, dtype: np.dtype | DatetimeTZDtype | None = None, copy: bool = False) -> 'DatetimeArray':
        """
        Create a DatetimeArray from a sequence.

        Parameters
        ----------
        scalars : np.ndarray | ArrayLike
        dtype : np.dtype | DatetimeTZDtype | None, default None
        copy : bool, default False

        Returns
        -------
        DatetimeArray
        """
        return cls._from_sequence_not_strict(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_sequence_not_strict(cls: type['DatetimeArray'], data: np.ndarray | ArrayLike, *, dtype: np.dtype | DatetimeTZDtype | None = None, copy: bool = False, tz: tzinfo | None = lib.no_default, freq: BaseOffset | None = lib.no_default, dayfirst: bool = False, yearfirst: bool = False, ambiguous: DateTimeErrorChoices = 'raise') -> 'DatetimeArray':
        """
        Create a DatetimeArray from a sequence.

        Parameters
        ----------
        data : np.ndarray | ArrayLike
        dtype : np.dtype | DatetimeTZDtype | None, default None
        copy : bool, default False
        tz : tzinfo | None, default lib.no_default
        freq : BaseOffset | None, default lib.no_default
        dayfirst : bool, default False
        yearfirst : bool, default False
        ambiguous : DateTimeErrorChoices, default 'raise'

        Returns
        -------
        DatetimeArray
        """
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
        data_dtype = tz_to_dtype(tz, data_unit)
        result = cls._simple_new(subarr, freq=inferred_freq, dtype=data_dtype)
        if unit is not None and unit != result.unit:
            result = result.as_unit(unit)
        validate_kwds = {'ambiguous': ambiguous}
        result._maybe_pin_freq(freq, validate_kwds)
        return result

    @classmethod
    def _generate_range(cls: type['DatetimeArray'], start: Timestamp | None, end: Timestamp | None, periods: int | None, freq: BaseOffset | None, tz: tzinfo | None, normalize: bool, ambiguous: DateTimeErrorChoices, nonexistent: DateTimeErrorChoices, inclusive: IntervalClosedType, *, unit: str | None) -> 'DatetimeArray':
        """
        Generate a DatetimeArray from a start and end point.

        Parameters
        ----------
        start : Timestamp | None
        end : Timestamp | None
        periods : int | None
        freq : BaseOffset | None
        tz : tzinfo | None
        normalize : bool
        ambiguous : DateTimeErrorChoices
        nonexistent : DateTimeErrorChoices
        inclusive : IntervalClosedType
        unit : str | None

        Returns
        -------
        DatetimeArray
        """
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
                xdr = _generate_range(start=start, end=end, periods=periods, offset=freq, unit=unit)
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
            if not left_inclusive and (not right_inclusive):
                i8values = i8values[1:-1]
        else:
            start_i8 = Timestamp(start)._value
            end_i8 = Timestamp(end)._value
            if not left_inclusive or not right_inclusive:
                if not left_inclusive and len(i8values) and (i8values[0] == start_i8):
                    i8values = i8values[1:]
                if not right_inclusive and len(i8values) and (i8values[-1] == end_i8):
                    i8values = i8values[:-1]
        dt64_values = i8values.view(f'datetime64[{unit}]')
        dtype = tz_to_dtype(tz, unit=unit)
        return cls._simple_new(dt64_values, freq=freq, dtype=dtype)

    def _unbox_scalar(self, value: Timestamp | NaT) -> np.ndarray:
        """
        Unbox a scalar.

        Parameters
        ----------
        value : Timestamp | NaT

        Returns
        -------
        np.ndarray
        """
        if not isinstance(value, self._scalar_type) and value is not NaT:
            raise ValueError("'value' should be a Timestamp.")
        self._check_compatible_with(value)
        if value is NaT:
            return np.datetime64(value._value, self.unit)
        else:
            return value.as_unit(self.unit, round_ok=False).asm8

    def _scalar_from_string(self, value: str) -> Timestamp:
        """
        Create a scalar from a string.

        Parameters
        ----------
        value : str

        Returns
        -------
        Timestamp
        """
        return Timestamp(value, tz=self.tz)

    def _check_compatible_with(self, other: Timestamp | NaT) -> None:
        """
        Check that a value is compatible with this DatetimeArray.

        Parameters
        ----------
        other : Timestamp | NaT

        Returns
        -------
        None
        """
        if other is NaT:
            return
        self._assert_tzawareness_compat(other)

    def _box_func(self, x: np.ndarray) -> Timestamp:
        """
        Box an array.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        Timestamp
        """
        value = x.view('i8')
        ts = Timestamp._from_value_and_reso(value, reso=self._creso, tz=self.tz)
        return ts

    @property
    def dtype(self) -> np.dtype | DatetimeTZDtype:
        """
        Get the dtype of the DatetimeArray.

        Returns
        -------
        np.dtype | DatetimeTZDtype
        """
        return self._dtype

    @property
    def tz(self) -> tzinfo | None:
        """
        Get the timezone of the DatetimeArray.

        Returns
        -------
        tzinfo | None
        """
        return getattr(self.dtype, 'tz', None)

    @tz.setter
    def tz(self, value: tzinfo | None) -> None:
        """
        Set the timezone of the DatetimeArray.

        Parameters
        ----------
        value : tzinfo | None

        Returns
        -------
        None
        """
        raise AttributeError('Cannot directly set timezone. Use tz_localize() or tz_convert() as appropriate')

    @property
    def tzinfo(self) -> tzinfo | None:
        """
        Get the timezone of the DatetimeArray.

        Returns
        -------
        tzinfo | None
        """
        return self.tz

    @property
    def is_normalized(self) -> bool:
        """
        Check if the DatetimeArray is normalized.

        Returns
        -------
        bool
        """
        return is_date_array_normalized(self.asi8, self.tz, reso=self._creso)

    @property
    def _resolution_obj(self) -> Resolution:
        """
        Get the resolution of the DatetimeArray.

        Returns
        -------
        Resolution
        """
        return get_resolution(self.asi8, self.tz, reso=self._creso)

    def __array__(self, dtype: type[np.ndarray] | None = None, copy: bool | None = None) -> np.ndarray:
        """
        Get the array representation of the DatetimeArray.

        Parameters
        ----------
        dtype : type[np.ndarray] | None, default None
        copy : bool | None, default None

        Returns
        -------
        np.ndarray
        """
        if dtype is None and self.tz:
            dtype = object
        return super().__array__(dtype=dtype, copy=copy)

    def __iter__(self) -> Iterator[Timestamp]:
        """
        Get an iterator over the boxed values.

        Yields
        ------
        Timestamp
        """
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

    def astype(self, dtype: type[np.ndarray] | np.dtype | ExtensionDtype, copy: bool = True) -> 'DatetimeArray':
        """
        Cast the DatetimeArray to a different dtype.

        Parameters
        ----------
        dtype : type[np.ndarray] | np.dtype | ExtensionDtype
        copy : bool, default True

        Returns
        -------
        DatetimeArray
        """
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
        elif self.tz is None and lib.is_np_dtype(dtype, 'M') and (not is_unitless(dtype)) and is_supported_dtype(dtype):
            res_values = astype_overflowsafe(self._ndarray, dtype, copy=True)
            return type(self)._simple_new(res_values, dtype=res_values.dtype)
        elif self.tz is not None and lib.is_np_dtype(dtype, 'M'):
            raise TypeError("Cannot use .astype to convert from timezone-aware dtype to timezone-naive dtype. Use obj.tz_localize(None) or obj.tz_convert('UTC').tz_localize(None) instead.")
        elif self.tz is None and lib.is_np_dtype(dtype, 'M') and (dtype != self.dtype) and is_unitless(dtype):
            raise TypeError("Casting to unit-less dtype 'datetime64' is not supported. Pass e.g. 'datetime64[ns]' instead.")
        elif isinstance(dtype, PeriodDtype):
            return self.to_period(freq=dtype.freq)
        return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy)

    def _format_native_types(self, *, na_rep: str = 'NaT', date_format: str | None = None, **kwargs: Any) -> np.ndarray:
        """
        Format the DatetimeArray as native types.

        Parameters
        ----------
        na_rep : str, default 'NaT'
        date_format : str | None, default None
        **kwargs : Any

        Returns
        -------
        np.ndarray
        """
        if date_format is None and self._is_dates_only:
            date_format = '%Y-%m-%d'
        return tslib.format_array_from_datetime(self.asi8, tz=self.tz, format=date_format, na_rep=na_rep, reso=self._creso)

    def _assert_tzawareness_compat(self, other: Timestamp | NaT) -> None:
        """
        Check that two values are compatible in terms of timezone awareness.

        Parameters
        ----------
        other : Timestamp | NaT

        Returns
        -------
        None
        """
        other_tz = getattr(other, 'tzinfo', None)
        other_dtype = getattr(other, 'dtype', None)
        if isinstance(other_dtype, DatetimeTZDtype):
            other_tz = other.dtype.tz
        if other is NaT:
            pass
        elif self.tz is None:
            if other_tz is not None:
                raise TypeError('Cannot compare tz-naive and tz-aware datetime-like objects.')
        elif other_tz is None:
            raise TypeError('Cannot compare tz-naive and tz-aware datetime-like objects')

    def _add_offset(self, offset: BaseOffset) -> 'DatetimeArray':
        """
        Add an offset to the DatetimeArray.

        Parameters
        ----------
        offset : BaseOffset

        Returns
        -------
        DatetimeArray
        """
        assert not isinstance(offset, Tick)
        if self.tz is not None:
            values = self.tz_localize(None)
        else:
            values = self
        try:
            res_values = offset._apply_array(values._ndarray)
            if res_values.dtype.kind == 'i':
                res_values = res_values.view(values.dtype)
        except NotImplementedError:
            if get_option('performance_warnings'):
                warnings.warn('Non-vectorized DateOffset being applied to Series or DatetimeIndex.', PerformanceWarning, stacklevel=find_stack_level())
            res_values = self.astype('O') + offset
            result = type(self)._from_sequence(res_values, dtype=self.dtype)
        else:
            result = type(self)._simple_new(res_values, dtype=res_values.dtype)
            if offset.normalize:
                result = result.normalize()
                result._freq = None
            if self.tz is not None:
                result = result.tz_localize(self.tz)
        return result

    def _local_timestamps(self) -> np.ndarray:
        """
        Convert to an i8 (unix-like nanosecond timestamp) representation
        while keeping the local timezone and not using UTC.

        Returns
        -------
        np.ndarray
        """
        if self.tz is None or timezones.is_utc(self.tz):
            return self.asi8
        return tz_convert_from_utc(self.asi8, self.tz, reso=self._creso)

    def tz_convert(self, tz: tzinfo | None) -> 'DatetimeArray':
        """
        Convert the timezone of the DatetimeArray.

        Parameters
        ----------
        tz : tzinfo | None

        Returns
        -------
        DatetimeArray
        """
        tz = timezones.maybe_get_tz(tz)
        if self.tz is None:
            raise TypeError('Cannot convert tz-naive timestamps, use tz_localize to localize')
        dtype = tz_to_dtype(tz, unit=self.unit)
        return self._simple_new(self._ndarray, dtype=dtype, freq=self.freq)

    @dtl.ravel_compat
    def tz_localize(self, tz: tzinfo | None, ambiguous: DateTimeErrorChoices = 'raise', nonexistent: DateTimeErrorChoices = 'raise') -> 'DatetimeArray':
        """
        Localize the DatetimeArray.

        Parameters
        ----------
        tz : tzinfo | None
        ambiguous : DateTimeErrorChoices, default 'raise'
        nonexistent : DateTimeErrorChoices, default 'raise'

        Returns
        -------
        DatetimeArray
        """
        nonexistent_options = ('raise', 'NaT', 'shift_forward', 'shift_backward')
        if nonexistent not in nonexistent_options and (not isinstance(nonexistent, timedelta)):
            raise ValueError("The nonexistent argument must be one of 'raise', 'NaT', 'shift_forward', 'shift_backward' or a timedelta object")
        if self.tz is not None:
            if tz is None:
                new_dates = tz_convert_from_utc(self.asi8, self.tz, reso=self._creso)
            else:
                raise TypeError('Already tz-aware, use tz_convert to convert.')
        else:
            tz = timezones.maybe_get_tz(tz)
            new_dates = tzconversion.tz_localize_to_utc(self.asi8, tz, ambiguous=ambiguous, nonexistent=nonexistent, creso=self._creso)
        new_dates_dt64 = new_dates.view(f'M8[{self.unit}]')
        dtype = tz_to_dtype(tz, unit=self.unit)
        freq = None
        if timezones.is_utc(tz) or (len(self) == 1 and (not isna(new_dates_dt64[0]))):
            freq = self.freq
        elif tz is None and self.tz is None:
            freq = self.freq
        return self._simple_new(new_dates_dt64, dtype=dtype, freq=freq)

    def to_pydatetime(self) -> np.ndarray:
        """
        Convert the DatetimeArray to a numpy array of datetime.datetime objects.

        Returns
        -------
        np.ndarray
        """
        return ints_to_pydatetime(self.asi8, tz=self.tz, reso=self._creso)

    def normalize(self) -> 'DatetimeArray':
        """
        Normalize the DatetimeArray.

        Returns
        -------
        DatetimeArray
        """
        new_values = normalize_i8_timestamps(self.asi8, self.tz, reso=self._creso)
        dt64_values = new_values.view(self._ndarray.dtype)
        dta = type(self)._simple_new(dt64_values, dtype=dt64_values.dtype)
        dta = dta._with_freq('infer')
        if self.tz is not None:
            dta = dta.tz_localize(self.tz)
        return dta

    def to_period(self, freq: str | PeriodDtype | None) -> PeriodArray:
        """
        Convert the DatetimeArray to a PeriodArray.

        Parameters
        ----------
        freq : str | PeriodDtype | None

        Returns
        -------
        PeriodArray
        """
        from pandas.core.arrays import PeriodArray
        if self.tz is not None:
            warnings.warn('Converting to PeriodArray/Index representation will drop timezone information.', UserWarning, stacklevel=find_stack_level())
        if freq is None:
            freq = self.freqstr or self.inferred_freq
            if isinstance(self.freq, BaseOffset) and hasattr(self.freq, '_period_dtype_code'):
                freq = PeriodDtype(self.freq)._freqstr
            if freq is None:
                raise ValueError('You must pass a freq argument as current index has none.')
            res = get_period_alias(freq)
            if res is None:
                res = freq
            freq = res
        return PeriodArray._from_datetime64(self._ndarray, freq, tz=self.tz)

    def month_name(self, locale: str | None = None) -> np.ndarray:
        """
        Get the month names.

        Parameters
        ----------
        locale : str | None, default None

        Returns
        -------
        np.ndarray
        """
        values = self._local_timestamps()
        result = fields.get_date_name_field(values, 'month_name', locale=locale, reso=self._creso)
        result = self._maybe_mask_results(result, fill_value=None)
        if using_string_dtype():
            from pandas import StringDtype, array as pd_array
            return pd_array(result, dtype=StringDtype(na_value=np.nan))
        return result

    def day_name(self, locale: str | None = None) -> np.ndarray:
        """
        Get the day names.

        Parameters
        ----------
        locale : str | None, default None

        Returns
        -------
        np.ndarray
        """
        values = self._local_timestamps()
        result = fields.get_date_name_field(values, 'day_name', locale=locale, reso=self._creso)
        result = self._maybe_mask_results(result, fill_value=None)
        if using_string_dtype():
            from pandas import StringDtype, array as pd_array
            return pd_array(result, dtype=StringDtype(na_value=np.nan))
        return result

    @property
    def time(self) -> np.ndarray:
        """
        Get the time part of the DatetimeArray.

        Returns
        -------
        np.ndarray
        """
        timestamps = self._local_timestamps()
        return ints_to_pydatetime(timestamps, box='time', reso=self._creso)

    @property
    def timetz(self) -> np.ndarray:
        """
        Get the time part of the DatetimeArray with timezone.

        Returns
        -------
        np.ndarray
        """
        return ints_to_pydatetime(self.asi8, self.tz, box='time', reso=self._creso)

    @property
    def date(self) -> np.ndarray:
        """
        Get the date part of the DatetimeArray.

        Returns
        -------
        np.ndarray
        """
        timestamps = self._local_timestamps()
        return ints_to_pydatetime(timestamps, box='date', reso=self._creso)

    def isocalendar(self) -> DataFrame:
        """
        Get the ISO calendar information.

        Returns
        -------
        DataFrame
        """
        from pandas import DataFrame
        values = self._local_timestamps()
        sarray = fields.build_isocalendar_sarray(values, reso=self._creso)
        iso_calendar_df = DataFrame(sarray, columns=['year', 'week', 'day'], dtype='UInt32')
        if self._hasna:
            iso_calendar_df.iloc[self._isnan] = None
        return iso_calendar_df

    year = _field_accessor('year', 'Y', '\n        The year of the datetime.\n\n        See Also\n        --------\n        DatetimeIndex.month: The month as January=1, December=12.\n        DatetimeIndex.day: The day of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="YE")\n        ... )\n        >>> datetime_series\n        0   2000-12-31\n        1   2001-12-31\n        2   2002-12-31\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.year\n        0    2000\n        1    2001\n        2    2002\n        dtype: int32\n        ')
    month = _field_accessor('month', 'M', '\n        The month as January=1, December=12.\n\n        See Also\n        --------\n        DatetimeIndex.year: The year of the datetime.\n        DatetimeIndex.day: The day of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="ME")\n        ... )\n        >>> datetime_series\n        0   2000-01-31\n        1   2000-02-29\n        2   2000-03-31\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.month\n        0    1\n        1    2\n        2    3\n        dtype: int32\n        ')
    day = _field_accessor('day', 'D', '\n        The day of the datetime.\n\n        See Also\n        --------\n        DatetimeIndex.year: The year of the datetime.\n        DatetimeIndex.month: The month as January=1, December=12.\n        DatetimeIndex.hour: The hours of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="D")\n        ... )\n        >>> datetime_series\n        0   2000-01-01\n        1   2000-01-02\n        2   2000-01-03\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.day\n        0    1\n        1    2\n        2    3\n        dtype: int32\n        ')
    hour = _field_accessor('hour', 'h', '\n        The hours of the datetime.\n\n        See Also\n        --------\n        DatetimeIndex.day: The day of the datetime.\n        DatetimeIndex.minute: The minutes of the datetime.\n        DatetimeIndex.second: The seconds of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="h")\n        ... )\n        >>> datetime_series\n        0   2000-01-01 00:00:00\n        1   2000-01-01 01:00:00\n        2   2000-01-01 02:00:00\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.hour\n        0    0\n        1    1\n        2    2\n        dtype: int32\n        ')
    minute = _field_accessor('minute', 'm', '\n        The minutes of the datetime.\n\n        See Also\n        --------\n        DatetimeIndex.hour: The hours of the datetime.\n        DatetimeIndex.second: The seconds of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="min")\n        ... )\n        >>> datetime_series\n        0   2000-01-01 00:00:00\n        1   2000-01-01 00:01:00\n        2   2000-01-01 00:02:00\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.minute\n        0    0\n        1    1\n        2    2\n        dtype: int32\n        ')
    second = _field_accessor('second', 's', '\n        The seconds of the datetime.\n\n        See Also\n        --------\n        DatetimeIndex.minute: The minutes of the datetime.\n        DatetimeIndex.microsecond: The microseconds of the datetime.\n        DatetimeIndex.nanosecond: The nanoseconds of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="s")\n        ... )\n        >>> datetime_series\n        0   2000-01-01 00:00:00\n        1   2000-01-01 00:00:01\n        2   2000-01-01 00:00:02\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.second\n        0    0\n        1    1\n        2    2\n        dtype: int32\n        ')
    microsecond = _field_accessor('microsecond', 'us', '\n        The microseconds of the datetime.\n\n        See Also\n        --------\n        DatetimeIndex.second: The seconds of the datetime.\n        DatetimeIndex.nanosecond: The nanoseconds of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="us")\n        ... )\n        >>> datetime_series\n        0   2000-01-01 00:00:00.000000\n        1   2000-01-01 00:00:00.000001\n        2   2000-01-01 00:00:00.000002\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.microsecond\n        0       0\n        1       1\n        2       2\n        dtype: int32\n        ')
    nanosecond = _field_accessor('nanosecond', 'ns', '\n        The nanoseconds of the datetime.\n\n        See Also\n        --------\n        DatetimeIndex.second: The seconds of the datetime.\n        DatetimeIndex.microsecond: The microseconds of the datetime.\n\n        Examples\n        --------\n        >>> datetime_series = pd.Series(\n        ...     pd.date_range("2000-01-01", periods=3, freq="ns")\n        ... )\n        >>> datetime_series\n        0   2000-01-01 00:00:00.000000000\n        1   2000-01-01 00:00:00.000000001\n        2   2000-01-01 00:00:00.000000002\n        dtype: datetime64[ns]\n        >>> datetime_series.dt.nanosecond\n        0       0\n        1       1\n        2       2\n        dtype: int32\n        ')
    _dayofweek_doc = "\n    The day of the week with Monday=0, Sunday=6.\n\n    Return the day of the week. It is assumed the week starts on\n    Monday, which is denoted by 0 and ends on Sunday which is denoted\n    by 6. This method is available on both Series with datetime\n    values (using the `dt` accessor) or DatetimeIndex.\n\n    Returns\n    -------\n    Series or Index\n        Containing integers indicating the day number.\n\n    See Also\n    --------\n    Series.dt.dayofweek : Alias.\n    Series.dt.weekday : Alias.\n    Series.dt.day_name : Returns the name of the day of the week.\n\n    Examples\n    --------\n    >>> s = pd.date_range('2016-12-31', '2017-01-08', freq='D').to_series()\n    >>> s.dt.dayofweek\n    2016-12-31    5\n    2017-01-01    6\n    2017-01-02    0\n    2017-01-03    1\n    2017-01-04    2\n    2017-01-05    3\n    2017-01-06    4\n    2017-01-07    5\n    2017-01-08    6\n    Freq: D, dtype: int32\n    "
    day_of_week = _field_accessor('day_of_week', 'dow', _dayofweek_doc)
    dayofweek = day_of_week
    weekday = day_of_week
    day_of_year = _field_accessor('dayofyear', 'doy', '\n        The ordinal day of the year.\n\n        See Also\n        --------\n        DatetimeIndex.dayofweek : The day of the week with Monday=0, Sunday=6.\n        DatetimeIndex.day : The day of the datetime.\n\n        Examples\n        --------\n        For Series:\n\n        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])\n        >>> s = pd.to_datetime(s)\n        >>> s\n        0   2020-01-01 10:00:00+00:00\n        1   2020-02-01 11:00:00+00:00\n        dtype: datetime64[s, UTC]\n        >>> s.dt.dayofyear\n        0    1\n        1   32\n        dtype: int32\n\n        For DatetimeIndex:\n\n        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00",\n        ...                         "2/1/2020 11:00:00+00:00"])\n        >>> idx.dayofyear\n        Index([1, 32], dtype=\'int32\')\n        ')
    dayofyear = day_of_year
    quarter = _field_accessor('quarter', 'q', '\n        The quarter of the date.\n\n        See Also\n        --------\n        DatetimeIndex.snap : Snap time stamps to nearest occurring frequency.\n        DatetimeIndex.time : Returns numpy array of datetime.time objects.\n            The time part of the Timestamps.\n\n        Examples\n        --------\n        For Series:\n\n        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "4/1/2020 11:00:00+00:00"])\n        >>> s = pd.to_datetime(s)\n        >>> s\n        0   2020-01-01 10:00:00+00:00\n        1   2020-04-01 11:00:00+00:00\n        dtype: datetime64[s, UTC]\n        >>> s.dt.quarter\n        0    1\n        1    2\n        dtype: int32\n\n        For DatetimeIndex:\n\n        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00",\n        ...                         "2/1/2020 11:00:00+00:00"])\n        >>> idx.quarter\n        Index([1, 1], dtype=\'int32\')\n        ')
    days_in_month = _field_accessor('days_in_month', 'dim', '\n        The number of days in the month.\n\n        See Also\n        --------\n        Series.dt.day : Return the day of the month.\n        Series.dt.is_month_end : Return a boolean indicating if the\n            date is the last day of the month.\n        Series.dt.is_month_start : Return a boolean indicating if the\n            date is the first day of the month.\n        Series.dt.month : Return the month as January=1 through December=12.\n\n        Examples\n        --------\n        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])\n        >>> s = pd.to_datetime(s)\n        >>> s\n        0   2020-01-01 10:00:00+00:00\n        1   2020-02-01 11:00:00+00:00\n        dtype: datetime64[s, UTC]\n        >>> s.dt.daysinmonth\n        0    31\n        1    29\n        dtype: int32\n        ')
    daysinmonth = days_in_month
    _is_month_doc = '\n        Indicates whether the date is the {first_or_last} day of the month.\n\n        Returns\n        -------\n        Series or array\n            For Series, returns a Series with boolean values.\n            For DatetimeIndex, returns a boolean array.\n\n        See Also\n        --------\n        is_month_start : Return a boolean indicating whether the date\n            is the first day of the month.\n        is_month_end : Return a boolean indicating whether the date\n            is the last day of the month.\n\n        Examples\n        --------\n        This method is available on Series with datetime values under\n        the ``.dt`` accessor, and directly on DatetimeIndex.\n\n        >>> s = pd.Series(pd.date_range("2018-02-27", periods=3))\n        >>> s\n        0   2018-02-27\n        1   2018-02-28\n        2   2018-03-01\n        dtype: datetime64[ns]\n        >>> s.dt.is_month_start\n        0    False\n        1    False\n        2    True\n        dtype: bool\n        >>> s.dt.is_month_end\n        0    False\n        1    True\n        2    False\n        dtype: bool\n\n        >>> idx = pd.date_range("2018-02-27", periods=3)\n        >>> idx.is_month_start\n        array([False, False, True])\n        >>> idx.is_month_end\n        array([False, True, False])\n    '
    is_month_start = _field_accessor('is_month_start', 'is_month_start', _is_month_doc.format(first_or_last='first'))
    is_month_end = _field_accessor('is_month_end', 'is_month_end', _is_month_doc.format(first_or_last='last'))
    is_quarter_start = _field_accessor('is_quarter_start', 'is_quarter_start', '\n        Indicator for whether the date is the first day of a quarter.\n\n        Returns\n        -------\n        is_quarter_start : Series or DatetimeIndex\n            The same type as the original data with boolean values. Series will\n            have the same name and index. DatetimeIndex will have the same\n            name.\n\n        See Also\n        --------\n        quarter : Return the quarter of the date.\n        is_quarter_end : Similar property for indicating the quarter end.\n\n        Examples\n        --------\n        This method is available on Series with datetime values under\n        the ``.dt`` accessor, and directly on DatetimeIndex.\n\n        >>> df = pd.DataFrame({\'dates\': pd.date_range("2017-03-30",\n        ...                   periods=4)})\n        >>> df.assign(quarter=df.dates.dt.quarter,\n        ...           is_quarter_start=df.dates.dt.is_quarter_start)\n               dates  quarter  is_quarter_start\n        0 2017-03-30        1             False\n        1 2017-03-31        1             False\n        2 2017-04-01        2              True\n        3 2017-04-02        2             False\n\n        >>> idx = pd.date_range(\'2017-03-30\', periods=4)\n        >>> idx\n        DatetimeIndex([\'2017-03-30\', \'2017-03-31\', \'2017-04-01\', \'2017-04-02\'],\n                      dtype=\'datetime64[ns]\', freq=\'D\')\n\n        >>> idx.is_quarter_start\n        array([False, False,  True, False])\n        ')
    is_quarter_end = _field_accessor('is_quarter_end', 'is_quarter_end', '\n        Indicator for whether the date is the last day of a quarter.\n\n        Returns\n        -------\n        is_quarter_end : Series or DatetimeIndex\n            The same type as the original data with boolean values. Series will\n            have the same name and index. DatetimeIndex will have the same\n            name.\n\n        See Also\n        --------\n        quarter : Return the quarter of the date.\n        is_quarter_start : Similar property indicating the quarter start.\n\n        Examples\n        --------\n        This method is available on Series with datetime values under\n        the ``.dt`` accessor, and directly on DatetimeIndex.\n\n        >>> df = pd.DataFrame({\'dates\': pd.date_range("2017-03-30",\n        ...                    periods=4)})\n        >>> df.assign(quarter=df.dates.dt.quarter,\n        ...           is_quarter_end=df.dates.dt.is_quarter_end)\n               dates  quarter    is_quarter_end\n        0 2017-03-30        1             False\n        1 2017-03-31        1              True\n        2 2017-04-01        2             False\n        3 2017-04-02        2             False\n\n        >>> idx = pd.date_range(\'2017-03-30\', periods=4)\n        >>> idx\n        DatetimeIndex([\'2017-03-30\', \'2017-03-31\', \'2017-04-01\', \'2017-04-02\'],\n                      dtype=\'datetime64[ns]\', freq=\'D\')\n\n        >>> idx.is_quarter_end\n        array([False,  True, False, False])\n        ')
    is_year_start = _field_accessor('is_year_start', 'is_year_start', '\n        Indicate whether the date is the first day of a year.\n\n        Returns\n        -------\n        Series or DatetimeIndex\n            The same type as the original data with boolean values. Series will\n            have the same name and index. DatetimeIndex will have the same\n            name.\n\n        See Also\n        --------\n        is_year_end : Similar property indicating the last day of the year.\n\n        Examples\n        --------\n        This method is available on Series with datetime values under\n        the ``.dt`` accessor, and directly on DatetimeIndex.\n\n        >>> dates = pd.Series(pd.date_range("2017-12-30", periods=3))\n        >>> dates\n        0   2017-12-30\n        1   2017-12-31\n        2   2018-01-01\n        dtype: datetime64[ns]\n\n        >>> dates.dt.is_year_start\n        0    False\n        1    False\n        2    True\n        dtype: bool\n\n        >>> idx = pd.date_range("2017-12-30", periods=3)\n        >>> idx\n        DatetimeIndex([\'2017-12-30\', \'2017-12-31\', \'2018-01-01\'],\n                      dtype=\'datetime64[ns]\', freq=\'D\')\n\n        >>> idx.is_year_start\n        array([False, False,  True])\n\n        This method, when applied to Series with datetime values under\n        the ``.dt`` accessor, will lose information about Business offsets.\n\n        >>> dates = pd.Series(pd.date_range("2020-10-30", periods=4, freq="BYS"))\n        >>> dates\n        0   2021-01-01\n        1   2022-01-03\n        2   2023-01-02\n        3   2024-01-01\n        dtype: datetime64[ns]\n\n        >>> dates.dt.is_year_start\n        0    True\n        1    False\n        2    False\n        3    True\n        dtype: bool\n\n        >>> idx = pd.date_range("2020-10-30", periods=4, freq="BYS")\n        >>> idx\n        DatetimeIndex([\'2021-01-01\', \'2022-01-03\', \'2023-01-02\', \'2024-01-01\'],\n                      dtype=\'datetime64[ns]\', freq=\'BYS-JAN\')\n\n        >>> idx.is_year_start\n        array([ True,  True,  True,  True])\n        ')
    is_year_end = _field_accessor('is_year_end', 'is_year_end', '\n        Indicate whether the date is the last day of the year.\n\n        Returns\n        -------\n        Series or DatetimeIndex\n            The same type as the original data with boolean values. Series will\n            have the same name and index. DatetimeIndex will have the same\n            name.\n\n        See Also\n        --------\n        is_year_start : Similar property indicating the start of the year.\n\n        Examples\n        --------\n        This method is available on Series with datetime values under\n        the ``.dt`` accessor, and directly on DatetimeIndex.\n\n        >>> dates = pd.Series(pd.date_range("2017-12-30", periods=3))\n        >>> dates\n        0   2017-12-30\n        1   2017-12-31\n        2   2018-01-01\n        dtype: datetime64[ns]\n\n        >>> dates.dt.is_year_end\n        0    False\n        1     True\n        2    False\n        dtype: bool\n\n        >>> idx = pd.date_range("2017-12-30", periods=3)\n        >>> idx\n        DatetimeIndex([\'2017-12-30\', \'2017-12-31\', \'2018-01-01\'],\n                      dtype=\'datetime64[ns]\', freq=\'D\')\n\n        >>> idx.is_year_end\n        array([False,  True, False])\n        ')
    is_leap_year = _field_accessor('is_leap_year', 'is_leap_year', '\n        Boolean indicator if the date belongs to a leap year.\n\n        A leap year is a year, which has 366 days (instead of 365) including\n        29th of February as an intercalary day.\n        Leap years are years which are multiples of four with the exception\n        of years divisible by 100 but not by 400.\n\n        Returns\n        -------\n        Series or ndarray\n             Booleans indicating if dates belong to a leap year.\n\n        See Also\n        --------\n        DatetimeIndex.is_year_end : Indicate whether the date is the\n            last day of the year.\n        DatetimeIndex.is_year_start : Indicate whether the date is the first\n            day of a year.\n\n        Examples\n        --------\n        This method is available on Series with datetime values under\n        the ``.dt`` accessor, and directly on DatetimeIndex.\n\n        >>> idx = pd.date_range("2012-01-01", "2015-01-01", freq="YE")\n        >>> idx\n        DatetimeIndex([\'2012-12-31\', \'2013-12-31\', \'2014-12-31\'],\n                      dtype=\'datetime64[ns]\', freq=\'YE-DEC\')\n        >>> idx.is_leap_year\n        array([ True, False, False])\n\n        >>> dates_series = pd.Series(idx)\n        >>> dates_series\n        0   2012-12-31\n        1   2013-12-31\n        2   2014-12-31\n        dtype: datetime64[ns]\n        >>> dates_series.dt.is_leap_year\n        0     True\n        1    False\n        2    False\n        dtype: bool\n        ')

    def to_julian_date(self) -> np.ndarray:
        """
        Convert the DatetimeArray to a float64 ndarray of Julian Dates.

        Returns
        -------
        np.ndarray
        """
        year = np.asarray(self.year)
        month = np.asarray(self.month)
        day = np.asarray(self.day)
        testarr = month < 3
        year[testarr] -= 1
        month[testarr] += 12
        return day + np.fix((153 * month - 457) / 5) + 365 * year + np.floor(year / 4) - np.floor(year / 100) + np.floor(year / 400) + 1721118.5 + (self.hour + self.minute / 60 + self.second / 3600 + self.microsecond / 3600 / 10 ** 6 + self.nanosecond / 3600 / 10 ** 9) / 24

    def _reduce(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs: Any) -> Any:
        """
        Reduce the DatetimeArray.

        Parameters
        ----------
        name : str
        skipna : bool, default True
        keepdims : bool, default False
        **kwargs : Any

        Returns
        -------
        Any
        """
        result = super()._reduce(name, skipna=skipna, keepdims=keepdims, **kwargs)
        if keepdims and isinstance(result, np.ndarray):
            if name == 'std':
                from pandas.core.arrays import TimedeltaArray
                return TimedeltaArray._from_sequence(result)
            else:
                return self._from_sequence(result, dtype=self.dtype)
        return result

    def std(self, axis: int | None = None, dtype: type[np.ndarray] | None = None, out: np.ndarray | None = None, ddof: int = 1, keepdims: bool = False, skipna: bool = True) -> Timedelta:
        """
        Calculate the standard deviation.

        Parameters
        ----------
        axis : int | None, default None
        dtype : type[np.ndarray] | None, default None
        out : np.ndarray | None, default None
        ddof : int, default 1
        keepdims : bool, default False
        skipna : bool, default True

        Returns
        -------
        Timedelta
        """
        from pandas.core.arrays import TimedeltaArray
        dtype_str = self._ndarray.dtype.name.replace('datetime64', 'timedelta64')
        dtype = np.dtype(dtype_str)
        tda = TimedeltaArray._simple_new(self._ndarray.view(dtype), dtype=dtype)
        return tda.std(axis=axis, out=out, ddof=ddof, keepdims=keepdims, skipna=skipna)

def _sequence_to_dt64(data: np.ndarray | ExtensionArray, *, copy: bool = False, tz: tzinfo | None = None, dayfirst: bool = False, yearfirst: bool = False, ambiguous: DateTimeErrorChoices = 'raise', out_unit: str | None) -> tuple[np.ndarray, tzinfo | None]:
    """
    Parameters
    ----------
    data : np.ndarray | ExtensionArray
    copy : bool, default False
    tz : tzinfo | None, default None
    dayfirst : bool, default False
    yearfirst : bool, default False
    ambiguous : str, bool, or arraylike, default 'raise'
        See pandas._libs.tslibs.tzconversion.tz_localize_to_utc.
    out_unit : str or None, default None
        Desired output resolution.

    Returns
    -------
    result : numpy.ndarray
        The sequence converted to a numpy array with dtype ``datetime64[unit]``.
        Where `unit` is "ns" unless specified otherwise by `out_unit`.
    tz : tzinfo or None
        Either the user-provided tzinfo or one inferred from the data.

    Raises
    ------
    TypeError : PeriodDType data is passed
    """
    data, copy = maybe_convert_dtype(data, copy, tz=tz)
    data_dtype = getattr(data, 'dtype', None)
    out_dtype = DT64NS_DTYPE
    if out_unit is not None:
        out_dtype = np.dtype(f'M8[{out_unit}]')
    if data_dtype == object or is_string_dtype(data_dtype):
        data = cast(np.ndarray, data)
        copy = False
        if lib.infer_dtype(data, skipna=False) == 'integer':
            data = data.astype(np.int64)
        elif tz is not None and ambiguous == 'raise':
            obj_data = np.asarray(data, dtype=object)
            result = tslib.array_to_datetime_with_tz(obj_data, tz=tz, dayfirst=dayfirst, yearfirst=yearfirst, creso=abbrev_to_npy_unit(out_unit))
            return (result, tz)
        else:
            converted, inferred_tz = objects_to_datetime64(data, dayfirst=dayfirst, yearfirst=yearfirst, allow_object=False, out_unit=out_unit)
            copy = False
            if tz and inferred_tz:
                result = converted
            elif inferred_tz:
                tz = inferred_tz
                result = converted
            else:
                result, _ = _construct_from_dt64_naive(converted, tz=tz, copy=copy, ambiguous=ambiguous)
            return (result, tz)
        data_dtype = data.dtype
    if isinstance(data_dtype, DatetimeTZDtype):
        data = cast(DatetimeArray, data)
        tz = _maybe_infer_tz(tz, data.tz)
        result = data._ndarray
    elif lib.is_np_dtype(data_dtype, 'M'):
        if isinstance(data, DatetimeArray):
            data = data._ndarray
        data = cast(np.ndarray, data)
        result, copy = _construct_from_dt64_naive(data, tz=tz, copy=copy, ambiguous=ambiguous)
    else:
        if data.dtype != INT64_DTYPE:
            data = data.astype(np.int64, copy=False)
            copy = False
        data = cast(np.ndarray, data)
        result = data.view(out_dtype)
    if copy:
        result = result.copy()
    assert isinstance(result, np.ndarray), type(result)
    assert result.dtype.kind == 'M'
    assert result.dtype != 'M8'
    assert is_supported_dtype(result.dtype)
    return (result, tz)

def _construct_from_dt64_naive(data: np.ndarray, *, tz: tzinfo | None, copy: bool, ambiguous: DateTimeErrorChoices) -> tuple[np.ndarray, bool]:
    """
    Convert datetime64 data to a supported dtype, localizing if necessary.

    Parameters
    ----------
    data : np.ndarray
    tz : tzinfo | None
    copy : bool
    ambiguous : DateTimeErrorChoices

    Returns
    -------
    result : np.ndarray
        The converted data.
    copy : bool
        Whether the data was copied.
    """
    new_dtype = data.dtype
    if not is_supported_dtype(new_dtype):
        new_dtype = get_supported_dtype(new_dtype)
        data = astype_overflowsafe(data, dtype=new_dtype, copy=False)
        copy = False
    if data.dtype.byteorder == '>':
        data = data.astype(data.dtype.newbyteorder('<'))
        new_dtype = data.dtype
        copy = False
    if tz is not None:
        shape = data.shape
        if data.ndim > 1:
            data = data.ravel()
        data_unit = get_unit_from_dtype(new_dtype)
        data = tzconversion.tz_localize_to_utc(data.view('i8'), tz, ambiguous=ambiguous, creso=data_unit)
        data = data.view(new_dtype)
        data = data.reshape(shape)
    assert data.dtype == new_dtype, data.dtype
    result = data
    return (result, copy)

def objects_to_datetime64(data: np.ndarray[object], dayfirst: bool, yearfirst: bool, utc: bool, errors: str, allow_object: bool, out_unit: str | None) -> tuple[np.ndarray, tzinfo | None]:
    """
    Convert data to array of timestamps.

    Parameters
    ----------
    data : np.ndarray[object]
    dayfirst : bool
    yearfirst : bool
    utc : bool, default False
        Whether to convert/localize timestamps to UTC.
    errors : {'raise', 'coerce'}
    allow_object : bool
        Whether to return an object-dtype ndarray instead of raising if the
        data contains more than one timezone.
    out_unit : str or None, default None
        None indicates we should do resolution inference.

    Returns
    -------
    result : ndarray
        np.datetime64[out_unit] if returned values represent wall times or UTC
        timestamps.
        object if mixed timezones
    inferred_tz : tzinfo or None
        If not None, then the datetime64 values in `result` denote UTC timestamps.

    Raises
    ------
    ValueError : if data cannot be converted to datetimes
    TypeError  : When a type cannot be converted to datetime
    """
    assert errors in ['raise', 'coerce']
    data = np.asarray(data, dtype=np.object_)
    result, tz_parsed = tslib.array_to_datetime(data, errors=errors, utc=utc, dayfirst=dayfirst, yearfirst=yearfirst, creso=abbrev_to_npy_unit(out_unit))
    if tz_parsed is not None:
        return (result, tz_parsed)
    elif result.dtype.kind == 'M':
        return (result, tz_parsed)
    elif result.dtype == object:
        if allow_object:
            return (result, tz_parsed)
        raise TypeError('DatetimeIndex has mixed timezones')
    else:
        raise TypeError(result)

def maybe_convert_dtype(data: np.ndarray | pd.Index, copy: bool, tz: tzinfo | None) -> tuple[np.ndarray, bool]:
    """
    Convert data based on dtype conventions, issuing
    errors where appropriate.

    Parameters
    ----------
    data : np.ndarray | pd.Index
    copy : bool
    tz : tzinfo | None

    Returns
    -------
    data : np.ndarray | pd.Index
    copy : bool

    Raises
    ------
    TypeError : PeriodDType data is passed
    """
    if not hasattr(data, 'dtype'):
        return (data, copy)
    if is_float_dtype(data.dtype):
        data = data.astype(DT64NS_DTYPE).view('i8')
        copy = False
    elif lib.is_np_dtype(data.dtype, 'm') or is_bool_dtype(data.dtype):
        raise TypeError(f'dtype {data.dtype} cannot be converted to datetime64[ns]')
    elif isinstance(data.dtype, PeriodDtype):
        raise TypeError('Passing PeriodDtype data is invalid. Use `data.to_timestamp()` instead')
    elif isinstance(data.dtype, ExtensionDtype) and (not isinstance(data.dtype, DatetimeTZDtype)):
        data = np.array(data, dtype=np.object_)
        copy = False
    return (data, copy)

def _maybe_infer_tz(tz: tzinfo | None, inferred_tz: tzinfo | None) -> tzinfo | None:
    """
    If a timezone is inferred from data, check that it is compatible with
    the user-provided timezone, if any.

    Parameters
    ----------
    tz : tzinfo | None
    inferred_tz : tzinfo | None

    Returns
    -------
    tz : tzinfo | None

    Raises
    ------
    TypeError : if both timezones are present but do not match
    """
    if tz is None:
        tz = inferred_tz
    elif inferred_tz is None:
        pass
    elif not timezones.tz_compare(tz, inferred_tz):
        raise TypeError(f'data is already tz-aware {inferred_tz}, unable to set specified tz: {tz}')
    return tz

def _validate_dt64_dtype(dtype: np.dtype | DatetimeTZDtype) -> np.dtype | DatetimeTZDtype:
    """
    Check that a dtype, if passed, represents either a numpy datetime64[ns]
    dtype or a pandas DatetimeTZDtype.

    Parameters
    ----------
    dtype : object

    Returns
    -------
    dtype : None, numpy.dtype, or DatetimeTZDtype

    Raises
    ------
    ValueError : invalid dtype

    Notes
    -----
    Unlike _validate_tz_from_dtype, this does _not_ allow non-existent
    tz errors to go through
    """
    if dtype is not None:
        dtype = pandas_dtype(dtype)
        if dtype == np.dtype('M8'):
            msg = "Passing in 'datetime64' dtype with no precision is not allowed. Please pass in 'datetime64[ns]' instead."
            raise ValueError(msg)
        if isinstance(dtype, np.dtype) and (dtype.kind != 'M' or not is_supported_dtype(dtype)) or not isinstance(dtype, (np.dtype, DatetimeTZDtype)):
            raise ValueError(f"Unexpected value for 'dtype': '{dtype}'. Must be 'datetime64[s]', 'datetime64[ms]', 'datetime64[us]', 'datetime64[ns]' or DatetimeTZDtype'.")
        if getattr(dtype, 'tz', None):
            dtype = cast(DatetimeTZDtype, dtype)
            dtype = DatetimeTZDtype(unit=dtype.unit, tz=timezones.tz_standardize(dtype.tz))
    return dtype

def _validate_tz_from_dtype(dtype: np.dtype | DatetimeTZDtype, tz: tzinfo | None, explicit_tz_none: bool) -> tzinfo | None:
    """
    If the given dtype is a DatetimeTZDtype, extract the implied
    tzinfo object from it and check that it does not conflict with the given
    tz.

    Parameters
    ----------
    dtype : dtype, str
    tz : None, tzinfo
    explicit_tz_none : bool, default False
        Whether tz=None was passed explicitly, as opposed to lib.no_default.

    Returns
    -------
    tz : consensus tzinfo

    Raises
    ------
    ValueError : on tzinfo mismatch
    """
    if dtype is not None:
        if isinstance(dtype, str):
            try:
                dtype = DatetimeTZDtype.construct_from_string(dtype)
            except TypeError:
                pass
        dtz = getattr(dtype, 'tz', None)
        if dtz is not None:
            if tz is not None and (not timezones.tz_compare(tz, dtz)):
                raise ValueError('cannot supply both a tz and a dtype with a tz')
            if explicit_tz_none:
                raise ValueError('Cannot pass both a timezone-aware dtype and tz=None')
            tz = dtz
        if tz is not None and lib.is_np_dtype(dtype, 'M'):
            if tz is not None and (not timezones.tz_compare(tz, dtz)):
                raise ValueError('cannot supply both a tz and a timezone-naive dtype (i.e. datetime64[ns])')
    return tz

def _infer_tz_from_endpoints(start: Timestamp | None, end: Timestamp | None, tz: tzinfo | None) -> tzinfo | None:
    """
    If a timezone is not explicitly given via `tz`, see if one can
    be inferred from the `start` and `end` endpoints.  If more than one
    of these inputs provides a timezone, require that they all agree.

    Parameters
    ----------
    start : Timestamp | None
    end : Timestamp | None
    tz : tzinfo | None

    Returns
    -------
    tz : tzinfo | None

    Raises
    ------
    TypeError : if start and end timezones do not agree
    """
    try:
        inferred_tz = timezones.infer_tzinfo(start, end)
    except AssertionError as err:
        raise TypeError('Start and end cannot both be tz-aware with different timezones') from err
    inferred_tz = timezones.maybe_get_tz(inferred_tz)
    tz = timezones.maybe_get_tz(tz)
    if tz is not None and inferred_tz is not None:
        if not timezones.tz_compare(inferred_tz, tz):
            raise AssertionError('Inferred time zone not equal to passed time zone')
    elif inferred_tz is not None:
        tz = inferred_tz
    return tz

def _maybe_normalize_endpoints(start: Timestamp | None, end: Timestamp | None, normalize: bool) -> tuple[Timestamp | None, Timestamp | None]:
    """
    Normalize the start and end points.

    Parameters
    ----------
    start : Timestamp | None
    end : Timestamp | None
    normalize : bool

    Returns
    -------
    start : Timestamp | None
    end : Timestamp | None
    """
    if normalize:
        if start is not None:
            start = start.normalize()
        if end is not None:
            end = end.normalize()
    return (start, end)

def _maybe_localize_point(ts: Timestamp | None, freq: BaseOffset | None, tz: tzinfo | None, ambiguous: DateTimeErrorChoices, nonexistent: DateTimeErrorChoices) -> Timestamp:
    """
    Localize a start or end Timestamp to the timezone of the corresponding
    start or end Timestamp

    Parameters
    ----------
    ts : start or end Timestamp to potentially localize
    freq : Tick, DateOffset, or None
    tz : str, timezone object or None
    ambiguous: str, localization behavior for ambiguous times
    nonexistent: str, localization behavior for nonexistent times

    Returns
    -------
    ts : Timestamp
    """
    if ts is not None and ts.tzinfo is None:
        ambiguous = ambiguous if ambiguous != 'infer' else False
        localize_args = {'ambiguous': ambiguous, 'nonexistent': nonexistent, 'tz': None}
        if isinstance(freq, Tick) or freq is None:
            localize_args['tz'] = tz
        ts = ts.tz_localize(**localize_args)
    return ts

def _generate_range(start: Timestamp | None, end: Timestamp | None, periods: int | None, offset: BaseOffset, *, unit: str) -> Generator[Timestamp, None, None]:
    """
    Generates a sequence of dates corresponding to the specified time
    offset. Similar to dateutil.rrule except uses pandas DateOffset
    objects to represent time increments.

    Parameters
    ----------
    start : Timestamp or None
    end : Timestamp or None
    periods : int or None
    offset : DateOffset
    unit : str

    Notes
    -----
    * This method is faster for generating weekdays than dateutil.rrule
    * At least two of (start, end, periods) must be specified.
    * If both start and end are specified, the returned dates will
    satisfy start <= date <= end.

    Yields
    ------
    dates : generator object
    """
    offset = to_offset(offset)
    start = Timestamp(start)
    if start is not NaT:
        start = start.as_unit(unit)
    else:
        start = None
    end = Timestamp(end)
    if end is not NaT:
        end = end.as_unit(unit)
    else:
        end = None
    if start and (not offset.is_on_offset(start)):
        if offset.n >= 0:
            start = offset.rollforward(start)
        else:
            start = offset.rollback(start)
    if periods is None and end < start and (offset.n >= 0):
        end = None
        periods = 0
    if end is None:
        end = start + (periods - 1) * offset
    if start is None:
        start = end - (periods - 1) * offset
    start = cast(Timestamp, start)
    end = cast(Timestamp, end)
    cur = start
    if offset.n >= 0:
        while cur <= end:
            yield cur
            if cur == end:
                break
            next_date = offset._apply(cur)
            next_date = next_date.as_unit(unit)
            if next_date <= cur:
                raise ValueError(f'Offset {offset} did not increment date')
            cur = next_date
    else:
        while cur >= end:
            yield cur
            if cur == end:
                break
            next_date = offset._apply(cur)
            next_date = next_date.as_unit(unit)
            if next_date >= cur:
                raise ValueError(f'Offset {offset} did not decrement date')
            cur = next_date

from __future__ import annotations
from datetime import timedelta
import operator
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union, overload
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
    from collections.abc import Callable as CallableType
    from pandas._typing import AnyArrayLike, Dtype, FillnaOptions, NpDtype, NumpySorter, NumpyValueArrayLike, Self, npt
    from pandas.core.dtypes.dtypes import ExtensionDtype
    from pandas.core.arrays import DatetimeArray, TimedeltaArray
    from pandas.core.arrays.base import ExtensionArray

BaseOffsetT = TypeVar("BaseOffsetT", bound=BaseOffset)
_PeriodArrayT = TypeVar("_PeriodArrayT", bound="PeriodArray")
_shared_doc_kwargs = {"klass": "PeriodArray"}

def _field_accessor(name: str, docstring: Optional[str] = None) -> property:
    def f(self: Any) -> Any:
        base = self.dtype._dtype_code
        result = get_period_field_arr(name, self.asi8, base)
        return result
    f.__name__ = name
    f.__doc__ = docstring
    return property(f)

class PeriodArray(dtl.DatelikeOps, libperiod.PeriodMixin, NDArrayBacked):
    """
    Pandas ExtensionArray for storing Period data.

    Users should use :func:`~pandas.array` to create new instances.

    Parameters
    ----------
    values : Union[PeriodArray, Series[period], ndarray[int], PeriodIndex]
        The data to store. These should be arrays that can be directly
        converted to ordinals without inference or copy (PeriodArray,
        ndarray[int64]), or a box around such an array (Series[period],
        PeriodIndex).
    dtype : PeriodDtype, optional
        A PeriodDtype instance from which to extract a `freq`. If both
        `freq` and `dtype` are specified, then the frequencies must match.
    copy : bool, default False
        Whether to copy the ordinals before storing.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    Period: Represents a period of time.
    PeriodIndex : Immutable Index for period data.
    period_range: Create a fixed-frequency PeriodArray.
    array: Construct a pandas array.

    Notes
    -----
    There are two components to a PeriodArray

    - ordinals : integer ndarray
    - freq : pd.tseries.offsets.Offset

    The values are physically stored as a 1-D ndarray of integers. These are
    called "ordinals" and represent some kind of offset from a base.

    The `freq` indicates the span covered by each element of the array.
    All elements in the PeriodArray have the same `freq`.

    Examples
    --------
    >>> pd.arrays.PeriodArray(pd.PeriodIndex(["2023-01-01", "2023-01-02"], freq="D"))
    <PeriodArray>
    ['2023-01-01', '2023-01-02']
    Length: 2, dtype: period[D]
    """
    __array_priority__ = 1000
    _typ: str = "periodarray"
    _internal_fill_value: np.int64 = np.int64(iNaT)
    _recognized_scalars: Tuple[type, ...] = (Period,)
    _is_recognized_dtype = lambda x: isinstance(x, PeriodDtype)
    _infer_matches: Tuple[str, ...] = ("period",)

    @property
    def _scalar_type(self) -> type[Period]:
        return Period

    _other_ops: List[Any] = []
    _bool_ops: List[str] = ["is_leap_year"]
    _object_ops: List[str] = ["start_time", "end_time", "freq"]
    _field_ops: List[str] = ["year", "month", "day", "hour", "minute", "second", "weekofyear", "weekday", "week", "dayofweek", "day_of_week", "dayofyear", "day_of_year", "quarter", "qyear", "days_in_month", "daysinmonth"]
    _datetimelike_ops: List[str] = _field_ops + _object_ops + _bool_ops
    _datetimelike_methods: List[str] = ["strftime", "to_timestamp", "asfreq"]

    def __init__(
        self,
        values: Union[PeriodArray, ABCSeries, ABCPeriodIndex, np.ndarray, list],
        dtype: Optional[PeriodDtype] = None,
        copy: bool = False,
    ) -> None:
        if dtype is not None:
            dtype = pandas_dtype(dtype)
            if not isinstance(dtype, PeriodDtype):
                raise ValueError(f"Invalid dtype {dtype} for PeriodArray")
        if isinstance(values, ABCSeries):
            values = values._values
            if not isinstance(values, type(self)):
                raise TypeError("Incorrect dtype")
        elif isinstance(values, ABCPeriodIndex):
            values = values._values
        if isinstance(values, type(self)):
            if dtype is not None and dtype != values.dtype:
                raise raise_on_incompatible(values, dtype.freq)
            values, dtype = (values._ndarray, values.dtype)
        if not copy:
            values = np.asarray(values, dtype="int64")
        else:
            values = np.array(values, dtype="int64", copy=copy)
        if dtype is None:
            raise ValueError("dtype is not specified and cannot be inferred")
        dtype = cast(PeriodDtype, dtype)
        NDArrayBacked.__init__(self, values, dtype)

    @classmethod
    def _simple_new(cls: Type[_PeriodArrayT], values: np.ndarray, dtype: PeriodDtype) -> _PeriodArrayT:
        assertion_msg = "Should be numpy array of type i8"
        assert isinstance(values, np.ndarray) and values.dtype == "i8", assertion_msg
        return cls(values, dtype=dtype)

    @classmethod
    def _from_sequence(
        cls: Type[PeriodArray],
        scalars: Sequence[Any],
        *,
        dtype: Optional[Any] = None,
        copy: bool = False,
    ) -> PeriodArray:
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
    def _from_sequence_of_strings(
        cls: Type[PeriodArray],
        strings: Sequence[str],
        *,
        dtype: PeriodDtype,
        copy: bool = False,
    ) -> PeriodArray:
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _from_datetime64(
        cls: Type[PeriodArray],
        data: np.ndarray,
        freq: Union[str, Tick],
        tz: Optional[Any] = None,
    ) -> PeriodArray:
        """
        Construct a PeriodArray from a datetime64 array

        Parameters
        ----------
        data : ndarray[datetime64[ns], datetime64[ns, tz]]
        freq : str or Tick
        tz : tzinfo, optional

        Returns
        -------
        PeriodArray[freq]
        """
        if isinstance(freq, BaseOffset):
            freq = PeriodDtype(freq)._freqstr
        data, freq = dt64arr_to_periodarr(data, freq, tz)
        dtype = PeriodDtype(freq)
        return cls(data, dtype=dtype)

    @classmethod
    def _generate_range(
        cls: Type[PeriodArray],
        start: Optional[Any],
        end: Optional[Any],
        periods: Optional[int],
        freq: Union[str, BaseOffset],
    ) -> Tuple[np.ndarray, BaseOffset]:
        periods = dtl.validate_periods(periods)
        if freq is not None:
            freq = Period._maybe_convert_freq(freq)
        if start is not None or end is not None:
            subarr, freq = _get_ordinal_range(start, end, periods, freq)
        else:
            raise ValueError("Not enough parameters to construct Period range")
        return (subarr, freq)

    @classmethod
    def _from_fields(cls: Type[PeriodArray], *, fields: Mapping[str, Any], freq: Union[str, BaseOffset]) -> PeriodArray:
        subarr, freq = _range_from_fields(freq=freq, **fields)
        dtype = PeriodDtype(freq)
        return cls._simple_new(subarr, dtype=dtype)

    def _unbox_scalar(self, value: Any) -> np.int64:
        if value is NaT:
            return np.int64(value._value)
        elif isinstance(value, self._scalar_type):
            self._check_compatible_with(value)
            return np.int64(value.ordinal)
        else:
            raise ValueError(f"'value' should be a Period. Got '{value}' instead.")

    def _scalar_from_string(self, value: str) -> Period:
        return Period(value, freq=self.freq)

    def _check_compatible_with(self, other: Any) -> None:
        if other is NaT:
            return
        self._require_matching_freq(other.freq)

    @cache_readonly
    def dtype(self) -> PeriodDtype:
        return self._dtype

    @property
    def freq(self) -> BaseOffset:
        """
        Return the frequency object for this PeriodArray.
        """
        return self.dtype.freq

    @property
    def freqstr(self) -> str:
        return PeriodDtype(self.freq)._freqstr

    def __array__(self, dtype: Optional[Any] = None, copy: Optional[bool] = None) -> np.ndarray:
        if dtype == "i8":
            if not copy:
                return np.asarray(self.asi8, dtype=dtype)
            else:
                return np.array(self.asi8, dtype=dtype)
        if copy is False:
            raise ValueError("Unable to avoid copy while creating an array as requested.")
        if dtype == bool:
            return ~self._isnan
        return np.array(list(self), dtype=object)

    def __arrow_array__(self, type: Optional[Any] = None) -> Any:
        """
        Convert myself into a pyarrow Array.
        """
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
        storage_array = pyarrow.array(self._ndarray, mask=self.isna(), type="int64")
        return pyarrow.ExtensionArray.from_storage(period_type, storage_array)

    year = _field_accessor(
        "year",
        "\n        The year of the period.\n\n        See Also\n        --------\n        PeriodIndex.day_of_year : The ordinal day of the year.\n        PeriodIndex.dayofyear : The ordinal day of the year.\n        PeriodIndex.is_leap_year : Logical indicating if the date belongs to a\n            leap year.\n        PeriodIndex.weekofyear : The week ordinal of the year.\n        PeriodIndex.year : The year of the period.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex([\"2023\", \"2024\", \"2025\"], freq=\"Y\")\n        >>> idx.year\n        Index([2023, 2024, 2025], dtype:'int64')\n        "
    )
    month = _field_accessor(
        "month",
        "\n        The month as January=1, December=12.\n\n        See Also\n        --------\n        PeriodIndex.days_in_month : The number of days in the month.\n        PeriodIndex.daysinmonth : The number of days in the month.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex([\"2023-01\", \"2023-02\", \"2023-03\"], freq=\"M\")\n        >>> idx.month\n        Index([1, 2, 3], dtype:'int64')\n        "
    )
    day = _field_accessor(
        "day",
        "\n        The days of the period.\n\n        See Also\n        --------\n        PeriodIndex.day_of_week : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.day_of_year : The ordinal day of the year.\n        PeriodIndex.dayofweek : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.dayofyear : The ordinal day of the year.\n        PeriodIndex.days_in_month : The number of days in the month.\n        PeriodIndex.daysinmonth : The number of days in the month.\n        PeriodIndex.weekday : The day of the week with Monday=0, Sunday=6.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex(['2020-01-31', '2020-02-28'], freq='D')\n        >>> idx.day\n        Index([31, 28], dtype='int64')\n        "
    )
    hour = _field_accessor(
        "hour",
        "\n        The hour of the period.\n\n        See Also\n        --------\n        PeriodIndex.minute : The minute of the period.\n        PeriodIndex.second : The second of the period.\n        PeriodIndex.to_timestamp : Cast to DatetimeArray/Index.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex([\"2023-01-01 10:00\", \"2023-01-01 11:00\"], freq='h')\n        >>> idx.hour\n        Index([10, 11], dtype='int64')\n        "
    )
    minute = _field_accessor(
        "minute",
        "\n        The minute of the period.\n\n        See Also\n        --------\n        PeriodIndex.hour : The hour of the period.\n        PeriodIndex.second : The second of the period.\n        PeriodIndex.to_timestamp : Cast to DatetimeArray/Index.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex([\"2023-01-01 10:30:00\",\n        ...                       \"2023-01-01 11:50:00\"], freq='min')\n        >>> idx.minute\n        Index([30, 50], dtype='int64')\n        "
    )
    second = _field_accessor(
        "second",
        "\n        The second of the period.\n\n        See Also\n        --------\n        PeriodIndex.hour : The hour of the period.\n        PeriodIndex.minute : The minute of the period.\n        PeriodIndex.to_timestamp : Cast to DatetimeArray/Index.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex([\"2023-01-01 10:00:30\",\n        ...                       \"2023-01-01 10:00:31\"], freq='s')\n        >>> idx.second\n        Index([30, 31], dtype='int64')\n        "
    )
    weekofyear = _field_accessor(
        "week",
        "\n        The week ordinal of the year.\n\n        See Also\n        --------\n        PeriodIndex.day_of_week : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.dayofweek : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.week : The week ordinal of the year.\n        PeriodIndex.weekday : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.year : The year of the period.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex([\"2023-01\", \"2023-02\", \"2023-03\"], freq=\"M\")\n        >>> idx.week  # It can be written `weekofyear`\n        Index([5, 9, 13], dtype:'int64')\n        "
    )
    week = weekofyear
    day_of_week = _field_accessor(
        "day_of_week",
        "\n        The day of the week with Monday=0, Sunday=6.\n\n        See Also\n        --------\n        PeriodIndex.day : The days of the period.\n        PeriodIndex.day_of_week : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.day_of_year : The ordinal day of the year.\n        PeriodIndex.dayofweek : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.dayofyear : The ordinal day of the year.\n        PeriodIndex.week : The week ordinal of the year.\n        PeriodIndex.weekday : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.weekofyear : The week ordinal of the year.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex([\"2023-01-01\", \"2023-01-02\", \"2023-01-03\"], freq=\"D\")\n        >>> idx.weekday\n        Index([6, 0, 1], dtype:'int64')\n        "
    )
    dayofweek = day_of_week
    weekday = dayofweek
    dayofyear = _field_accessor(
        "day_of_year",
        "\n        The ordinal day of the year.\n\n        See Also\n        --------\n        PeriodIndex.day : The days of the period.\n        PeriodIndex.day_of_week : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.day_of_year : The ordinal day of the year.\n        PeriodIndex.dayofweek : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.dayofyear : The ordinal day of the year.\n        PeriodIndex.weekday : The day of the week with Monday=0, Sunday=6.\n        PeriodIndex.weekofyear : The week ordinal of the year.\n        PeriodIndex.year : The year of the period.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex([\"2023-01-10\", \"2023-02-01\", \"2023-03-01\"], freq=\"D\")\n        >>> idx.dayofyear\n        Index([10, 32, 60], dtype:'int64')\n\n        >>> idx = pd.PeriodIndex([\"2023\", \"2024\", \"2025\"], freq=\"Y\")\n        >>> idx\n        PeriodIndex(['2023', '2024', '2025'], dtype:'period[Y-DEC]')\n        >>> idx.dayofyear\n        Index([365, 366, 365], dtype:'int64')\n        "
    )
    quarter = _field_accessor(
        "quarter",
        "\n        The quarter of the date.\n\n        See Also\n        --------\n        PeriodIndex.qyear : Fiscal year the Period lies in according to its\n            starting-quarter.\n\n        Examples\n        --------\n        >>> idx = pd.PeriodIndex([\"2023-01\", \"2023-02\", \"2023-03\"], freq=\"M\")\n        >>> idx.quarter\n        Index([1, 1, 1], dtype:'int64')\n        "
    )
    qyear = _field_accessor(
        "qyear",
        "\n        Fiscal year the Period lies in according to its starting-quarter.\n\n        The `year` and the `qyear` of the period will be the same if the fiscal\n        and calendar years are the same. When they are not, the fiscal year\n        can be different from the calendar year of the period.\n\n        Returns\n        -------\n        int\n            The fiscal year of the period.\n\n        See Also\n        --------\n        PeriodIndex.quarter : The quarter of the date.\n        PeriodIndex.year : The year of the period.\n\n        Examples\n        --------\n        If the natural and fiscal year are the same, `qyear` and `year` will\n        be the same.\n\n        >>> per = pd.Period('2018Q1', freq='Q')\n        >>> per.qyear\n        2018\n        >>> per.year\n        2018\n\n        If the fiscal year starts in April (`Q-MAR`), the first quarter of\n        2018 will start in April 2017. `year` will then be 2017, but `qyear`\n        will be the fiscal year, 2018.\n\n        >>> per = pd.Period('2018Q1', freq='Q-MAR')\n        >>> per.start_time\n        Timestamp('2017-04-01 00:00:00')\n        >>> per.qyear\n        2018\n        >>> per.year\n        2017\n        "
    )
    days_in_month = _field_accessor(
        "days_in_month",
        "\n        The number of days in the month.\n\n        See Also\n        --------\n        PeriodIndex.day : The days of the period.\n        PeriodIndex.days_in_month : The number of days in the month.\n        PeriodIndex.daysinmonth : The number of days in the month.\n        PeriodIndex.month : The month as January=1, December=12.\n\n        Examples\n        --------\n        For Series:\n\n        >>> period = pd.period_range('2020-1-1 00:00', '2020-3-1 00:00', freq='M')\n        >>> s = pd.Series(period)\n        >>> s\n        0   2020-01\n        1   2020-02\n        2   2020-03\n        dtype: period[M]\n        >>> s.dt.days_in_month\n        0    31\n        1    29\n        2    31\n        dtype: int64\n\n        For PeriodIndex:\n\n        >>> idx = pd.PeriodIndex([\"2023-01\", \"2023-02\", \"2023-03\"], freq=\"M\")\n        >>> idx.days_in_month   # It can be also entered as `daysinmonth`\n        Index([31, 28, 31], dtype:'int64')\n        "
    )
    daysinmonth = days_in_month

    @property
    def is_leap_year(self) -> np.ndarray:
        """
        Logical indicating if the date belongs to a leap year.

        See Also
        --------
        PeriodIndex.qyear : Fiscal year the Period lies in according to its
            starting-quarter.
        PeriodIndex.year : The year of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> idx.is_leap_year
        array([False,  True, False])
        """
        return isleapyear_arr(np.asarray(self.year))

    def to_timestamp(
        self, freq: Optional[Union[str, BaseOffset]] = None, how: str = "start"
    ) -> Any:
        """
        Cast to DatetimeArray/Index.

        Parameters
        ----------
        freq : str or DateOffset, optional
            Target frequency. The default is 'D' for week or longer,
            's' otherwise.
        how : {'s', 'e', 'start', 'end'}
            Whether to use the start or end of the time period being converted.

        Returns
        -------
        DatetimeArray/Index
            Timestamp representation of given Period-like object.

        See Also
        --------
        PeriodIndex.day : The days of the period.
        PeriodIndex.from_fields : Construct a PeriodIndex from fields
            (year, month, day, etc.).
        PeriodIndex.from_ordinals : Construct a PeriodIndex from ordinals.
        PeriodIndex.hour : The hour of the period.
        PeriodIndex.minute : The minute of the period.
        PeriodIndex.month : The month as January=1, December=12.
        PeriodIndex.second : The second of the period.
        PeriodIndex.year : The year of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.to_timestamp()
        DatetimeIndex(['2023-01-01', '2023-02-01', '2023-03-01'],
        dtype='datetime64[ns]', freq='MS')

        The frequency will not be inferred if the index contains less than
        three elements, or if the values of index are not strictly monotonic:

        >>> idx = pd.PeriodIndex(["2023-01", "2023-02"], freq="M")
        >>> idx.to_timestamp()
        DatetimeIndex(['2023-01-01', '2023-02-01'], dtype='datetime64[ns]', freq=None)

        >>> idx = pd.PeriodIndex(
        ...     ["2023-01", "2023-02", "2023-02", "2023-03"], freq="2M"
        ... )
        >>> idx.to_timestamp()
        DatetimeIndex(['2023-01-01', '2023-02-01', '2023-02-01', '2023-03-01'],
        dtype='datetime64[ns]', freq=None)
        """
        from pandas.core.arrays import DatetimeArray

        how = libperiod.validate_end_alias(how)
        end = how == "E"
        if end:
            if freq == "B" or self.freq == "B":
                adjust = Timedelta(1, "D") - Timedelta(1, "ns")
                return self.to_timestamp(how="start") + adjust
            else:
                adjust = Timedelta(1, "ns")
                return (self + self.freq).to_timestamp(how="start") - adjust
        if freq is None:
            freq_code = self._dtype._get_to_timestamp_base()
            dtype = PeriodDtypeBase(freq_code, 1)
            freq = dtype._freqstr
            base = freq_code
        else:
            freq = Period._maybe_convert_freq(freq)
            base = freq._period_dtype_code
        new_parr = self.asfreq(freq, how=how)
        new_data = libperiod.periodarr_to_dt64arr(new_parr.asi8, base)
        dta = DatetimeArray._from_sequence(new_data, dtype=np.dtype("M8[ns]"))
        if self.freq.name == "B":
            diffs = libalgos.unique_deltas(self.asi8)
            if len(diffs) == 1:
                diff = diffs[0]
                if diff == self.dtype._n:
                    dta._freq = self.freq
                elif diff == 1:
                    dta._freq = self.freq.base
            return dta
        else:
            return dta._with_freq("infer")

    def _box_func(self, x: Any) -> Period:
        return Period._from_ordinal(ordinal=x, freq=self.freq)

    @doc(**_shared_doc_kwargs, other="PeriodIndex", other_name="PeriodIndex")
    def asfreq(self, freq: Optional[Union[str, BaseOffset]] = None, how: str = "E") -> PeriodArray:
        """
        Convert the {klass} to the specified frequency `freq`.

        Equivalent to applying :meth:`pandas.Period.asfreq` with the given arguments
        to each :class:`~pandas.Period` in this {klass}.

        Parameters
        ----------
        freq : str
            A frequency.
        how : str {'E', 'S'}, default 'E'
            Whether the elements should be aligned to the end
            or start within pa period.

            * 'E', 'END', or 'FINISH' for end,
            * 'S', 'START', or 'BEGIN' for start.

            January 31st ('END') vs. January 1st ('START') for example.

        Returns
        -------
        {klass}
            The transformed {klass} with the new frequency.

        See Also
        --------
        {other}.asfreq: Convert each Period in a {other_name} to the given frequency.
        Period.asfreq : Convert a :class:`~pandas.Period` object to the given frequency.

        Examples
        --------
        >>> pidx = pd.period_range("2010-01-01", "2015-01-01", freq="Y")
        >>> pidx
        PeriodIndex(['2010', '2011', '2012', '2013', '2014', '2015'],
        dtype='period[Y-DEC]')

        >>> pidx.asfreq("M")
        PeriodIndex(['2010-12', '2011-12', '2012-12', '2013-12', '2014-12',
        '2015-12'], dtype='period[M]')

        >>> pidx.asfreq("M", how="S")
        PeriodIndex(['2010-01', '2011-01', '2012-01', '2013-01', '2014-01',
        '2015-01'], dtype='period[M]')
        """
        how = libperiod.validate_end_alias(how)
        if isinstance(freq, BaseOffset) and hasattr(freq, "_period_dtype_code"):
            freq = PeriodDtype(freq)._freqstr
        freq = Period._maybe_convert_freq(freq)
        base1 = self._dtype._dtype_code
        base2 = freq._period_dtype_code
        asi8 = self.asi8
        end = how == "E"
        if end:
            ordinal = asi8 + self.dtype._n - 1
        else:
            ordinal = asi8
        new_data = period_asfreq_arr(ordinal, base1, base2, end)
        if self._hasna:
            new_data[self._isnan] = iNaT
        dtype = PeriodDtype(freq)
        return type(self)(new_data, dtype=dtype)

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str]:
        if boxed:
            return str
        return lambda x: "'{}'".format(x)

    def _format_native_types(self, *, na_rep: str = "NaT", date_format: Optional[str] = None, **kwargs: Any) -> Any:
        """
        actually format my specific types
        """
        return libperiod.period_array_strftime(self.asi8, self.dtype._dtype_code, na_rep, date_format)

    def astype(self, dtype: Any, copy: bool = True) -> Any:
        dtype = pandas_dtype(dtype)
        if dtype == self._dtype:
            if not copy:
                return self
            else:
                return self.copy()
        if isinstance(dtype, PeriodDtype):
            return self.asfreq(dtype.freq)
        if lib.is_np_dtype(dtype, "M") or isinstance(dtype, DatetimeTZDtype):
            tz = getattr(dtype, "tz", None)
            unit = dtl.dtype_to_unit(dtype)
            return self.to_timestamp().tz_localize(tz).as_unit(unit)
        return super().astype(dtype, copy=copy)

    def searchsorted(self, value: Any, side: str = "left", sorter: Optional[Any] = None) -> np.ndarray:
        npvalue = self._validate_setitem_value(value).view("M8[ns]")
        m8arr = self._ndarray.view("M8[ns]")
        return m8arr.searchsorted(npvalue, side=side, sorter=sorter)

    def _pad_or_backfill(
        self, *, method: str, limit: Optional[int] = None, limit_area: Optional[str] = None, copy: bool = True
    ) -> PeriodArray:
        dta = self.view("M8[ns]")
        result = dta._pad_or_backfill(method=method, limit=limit, limit_area=limit_area, copy=copy)
        if copy:
            return cast("PeriodArray", result.view(self.dtype))
        else:
            return self

    def _addsub_int_array_or_scalar(self, other: Union[np.ndarray, int], op: Callable[[Any, Any], Any]) -> PeriodArray:
        """
        Add or subtract array of integers.

        Parameters
        ----------
        other : np.ndarray[int64] or int
        op : {operator.add, operator.sub}

        Returns
        -------
        result : PeriodArray
        """
        assert op in [operator.add, operator.sub]
        if op is operator.sub:
            other = -other
        res_values = add_overflowsafe(self.asi8, np.asarray(other, dtype="i8"))
        return type(self)(res_values, dtype=self.dtype)

    def _add_offset(self, other: Any) -> PeriodArray:
        assert not isinstance(other, Tick)
        self._require_matching_freq(other, base=True)
        return self._addsub_int_array_or_scalar(other.n, operator.add)

    def _add_timedeltalike_scalar(self, other: Any) -> PeriodArray:
        """
        Parameters
        ----------
        other : timedelta, Tick, np.timedelta64

        Returns
        -------
        PeriodArray
        """
        if not isinstance(self.freq, Tick):
            raise raise_on_incompatible(self, other)
        if isna(other):
            return super()._add_timedeltalike_scalar(other)
        td = np.asarray(Timedelta(other).asm8)
        return self._add_timedelta_arraylike(td)

    def _add_timedelta_arraylike(self, other: Union[np.ndarray, Any]) -> PeriodArray:
        """
        Parameters
        ----------
        other : TimedeltaArray or ndarray[timedelta64]

        Returns
        -------
        PeriodArray
        """
        if not self.dtype._is_tick_like():
            raise TypeError(f"Cannot add or subtract timedelta64[ns] dtype from {self.dtype}")
        dtype = np.dtype(f"m8[{self.dtype._td64_unit}]")
        try:
            delta = astype_overflowsafe(np.asarray(other), dtype=dtype, copy=False, round_ok=False)
        except ValueError as err:
            raise IncompatibleFrequency("Cannot add/subtract timedelta-like from PeriodArray that is not an integer multiple of the PeriodArray's freq.") from err
        res_values = add_overflowsafe(self.asi8, np.asarray(delta.view("i8")))
        return type(self)(res_values, dtype=self.dtype)

    def _check_timedeltalike_freq_compat(self, other: Any) -> Union[int, np.ndarray]:
        """
        Arithmetic operations with timedelta-like scalars or array `other`
        are only valid if `other` is an integer multiple of `self.freq`.
        If the operation is valid, find that integer multiple.  Otherwise,
        raise because the operation is invalid.

        Parameters
        ----------
        other : timedelta, np.timedelta64, Tick,
                ndarray[timedelta64], TimedeltaArray, TimedeltaIndex

        Returns
        -------
        multiple : int or ndarray[int64]

        Raises
        ------
        IncompatibleFrequency
        """
        assert self.dtype._is_tick_like()
        dtype = np.dtype(f"m8[{self.dtype._td64_unit}]")
        if isinstance(other, (timedelta, np.timedelta64, Tick)):
            td = np.asarray(Timedelta(other).asm8)
        else:
            td = np.asarray(other)
        try:
            delta = astype_overflowsafe(td, dtype=dtype, copy=False, round_ok=False)
        except ValueError as err:
            raise raise_on_incompatible(self, other) from err
        delta = delta.view("i8")
        return lib.item_from_zerodim(delta)

    def _reduce(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs: Any) -> Any:
        result = super()._reduce(name, skipna=skipna, keepdims=keepdims, **kwargs)
        if keepdims and isinstance(result, np.ndarray):
            return self._from_sequence(result, dtype=self.dtype)
        return result

def raise_on_incompatible(left: PeriodArray, right: Any) -> IncompatibleFrequency:
    """
    Helper function to render a consistent error message when raising
    IncompatibleFrequency.

    Parameters
    ----------
    left : PeriodArray
    right : None, DateOffset, Period, ndarray, or timedelta-like

    Returns
    -------
    IncompatibleFrequency
        Exception to be raised by the caller.
    """
    if isinstance(right, (np.ndarray, ABCTimedeltaArray)) or right is None:
        other_freq = None
    elif isinstance(right, BaseOffset):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "PeriodDtype\\[B\\] is deprecated", category=FutureWarning)
            other_freq = PeriodDtype(right)._freqstr
    elif isinstance(right, (ABCPeriodIndex, PeriodArray, Period)):
        other_freq = right.freqstr  # type: ignore[attr-defined]
    else:
        other_freq = delta_to_tick(Timedelta(right)).freqstr
    own_freq = PeriodDtype(left.freq)._freqstr
    msg = DIFFERENT_FREQ.format(cls=type(left).__name__, own_freq=own_freq, other_freq=other_freq)
    return IncompatibleFrequency(msg)

def period_array(
    data: Sequence[Any],
    freq: Optional[Union[str, Tick, BaseOffset]] = None,
    copy: bool = False,
) -> PeriodArray:
    """
    Construct a new PeriodArray from a sequence of Period scalars.

    Parameters
    ----------
    data : Sequence of Period objects
        A sequence of Period objects. These are required to all have
        the same ``freq.`` Missing values can be indicated by ``None``
        or ``pandas.NaT``.
    freq : str, Tick, or Offset
        The frequency of every element of the array. This can be specified
        to avoid inferring the `freq` from `data`.
    copy : bool, default False
        Whether to ensure a copy of the data is made.

    Returns
    -------
    PeriodArray

    See Also
    --------
    PeriodArray
    pandas.PeriodIndex

    Examples
    --------
    >>> period_array([pd.Period("2017", freq="Y"), pd.Period("2018", freq="Y")])
    <PeriodArray>
    ['2017', '2018']
    Length: 2, dtype: period[Y-DEC]

    >>> period_array([pd.Period("2017", freq="Y"), pd.Period("2018", freq="Y"), pd.NaT])
    <PeriodArray>
    ['2017', '2018', 'NaT']
    Length: 3, dtype: period[Y-DEC]

    Integers that look like years are handled

    >>> period_array([2000, 2001, 2002], freq="D")
    <PeriodArray>
    ['2000-01-01', '2001-01-01', '2002-01-01']
    Length: 3, dtype: period[D]

    Datetime-like strings may also be passed

    >>> period_array(["2000-Q1", "2000-Q2", "2000-Q3", "2000-Q4"], freq="Q")
    <PeriodArray>
    ['2000Q1', '2000Q2', '2000Q3', '2000Q4']
    Length: 4, dtype: period[Q-DEC]
    """
    data_dtype = getattr(data, "dtype", None)
    if lib.is_np_dtype(data_dtype, "M"):
        return PeriodArray._from_datetime64(data, freq)  # type: ignore[arg-type]
    if isinstance(data_dtype, PeriodDtype):
        out = PeriodArray(data)
        if freq is not None:
            if freq == data_dtype.freq:
                return out
            return out.asfreq(freq)
        return out
    if not isinstance(data, (np.ndarray, list, tuple, ABCSeries)):
        data = list(data)
    arrdata = np.asarray(data)
    if freq:
        dtype: Optional[PeriodDtype] = PeriodDtype(freq)  # type: ignore[arg-type]
    else:
        dtype = None
    if arrdata.dtype.kind == "f" and len(arrdata) > 0:
        raise TypeError("PeriodIndex does not allow floating point in construction")
    if arrdata.dtype.kind in "iu":
        arr = arrdata.astype(np.int64, copy=False)
        ordinals = libperiod.from_ordinals(arr, freq)
        return PeriodArray(ordinals, dtype=dtype)  # type: ignore[arg-type]
    data = ensure_object(arrdata)
    if freq is None:
        freq = libperiod.extract_freq(data)
    dtype = PeriodDtype(freq)
    return PeriodArray._from_sequence(data, dtype=dtype)

@overload
def validate_dtype_freq(dtype: Any, freq: Any) -> Any:
    ...

@overload
def validate_dtype_freq(dtype: Any, freq: Any) -> Any:
    ...

def validate_dtype_freq(dtype: Any, freq: Optional[Any]) -> Any:
    """
    If both a dtype and a freq are available, ensure they match.  If only
    dtype is available, extract the implied freq.

    Parameters
    ----------
    dtype : dtype
    freq : DateOffset or None

    Returns
    -------
    freq : DateOffset

    Raises
    ------
    ValueError : non-period dtype
    IncompatibleFrequency : mismatch between dtype and freq
    """
    if freq is not None:
        freq = to_offset(freq, is_period=True)
    if dtype is not None:
        dtype = pandas_dtype(dtype)
        if not isinstance(dtype, PeriodDtype):
            raise ValueError("dtype must be PeriodDtype")
        if freq is None:
            freq = dtype.freq
        elif freq != dtype.freq:
            raise IncompatibleFrequency("specified freq and dtype are different")
    return freq

def dt64arr_to_periodarr(
    data: np.ndarray, freq: Union[str, Tick, BaseOffset], tz: Optional[Any] = None
) -> Tuple[np.ndarray, BaseOffset]:
    """
    Convert an datetime-like array to values Period ordinals.

    Parameters
    ----------
    data : Union[Series[datetime64[ns]], DatetimeIndex, ndarray[datetime64ns]]
    freq : Optional[Union[str, Tick]]
        Must match the `freq` on the `data` if `data` is a DatetimeIndex
        or Series.
    tz : Optional[tzinfo]

    Returns
    -------
    ordinals : ndarray[int64]
    freq : Tick
        The frequency extracted from the Series or DatetimeIndex if that's
        used.

    """
    if not isinstance(data.dtype, np.dtype) or data.dtype.kind != "M":
        raise ValueError(f"Wrong dtype: {data.dtype}")
    if freq is None:
        if isinstance(data, ABCIndex):
            data, freq = (data._values, data.freq)
        elif isinstance(data, ABCSeries):
            data, freq = (data._values, data.dt.freq)
    elif isinstance(data, (ABCIndex, ABCSeries)):
        data = data._values
    reso = get_unit_from_dtype(data.dtype)
    freq = Period._maybe_convert_freq(freq)
    base = freq._period_dtype_code
    return (c_dt64arr_to_periodarr(data.view("i8"), base, tz, reso=reso), freq)

def _get_ordinal_range(
    start: Optional[Any],
    end: Optional[Any],
    periods: Optional[int],
    freq: Optional[Union[str, BaseOffset]],
    mult: int = 1,
) -> Tuple[np.ndarray, BaseOffset]:
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError("Of the three parameters: start, end, and periods, exactly two must be specified")
    if freq is not None:
        freq = to_offset(freq, is_period=True)
        mult = freq.n
    if start is not None:
        start = Period(start, freq)
    if end is not None:
        end = Period(end, freq)
    is_start_per = isinstance(start, Period)
    is_end_per = isinstance(end, Period)
    if is_start_per and is_end_per and (start.freq != end.freq):
        raise ValueError("start and end must have same freq")
    if start is NaT or end is NaT:
        raise ValueError("start and end must not be NaT")
    if freq is None:
        if is_start_per:
            freq = start.freq
        elif is_end_per:
            freq = end.freq
        else:
            raise ValueError("Could not infer freq from start/end")
        mult = freq.n
    if periods is not None:
        periods = periods * mult
        if start is None:
            data = np.arange(end.ordinal - periods + mult, end.ordinal + 1, mult, dtype=np.int64)
        else:
            data = np.arange(start.ordinal, start.ordinal + periods, mult, dtype=np.int64)
    else:
        data = np.arange(start.ordinal, end.ordinal + 1, mult, dtype=np.int64)
    return (data, freq)

def _range_from_fields(
    year: Optional[Any] = None,
    month: Optional[Any] = None,
    quarter: Optional[Any] = None,
    day: Optional[Any] = None,
    hour: Optional[Any] = None,
    minute: Optional[Any] = None,
    second: Optional[Any] = None,
    freq: Union[str, BaseOffset] = None,
) -> Tuple[np.ndarray, BaseOffset]:
    if hour is None:
        hour = 0
    if minute is None:
        minute = 0
    if second is None:
        second = 0
    if day is None:
        day = 1
    ordinals: List[int] = []
    if quarter is not None:
        if freq is None:
            freq = to_offset("Q", is_period=True)
            base = FreqGroup.FR_QTR.value
        else:
            freq = to_offset(freq, is_period=True)
            base = libperiod.freq_to_dtype_code(freq)
            if base != FreqGroup.FR_QTR.value:
                raise AssertionError("base must equal FR_QTR")
        freqstr = freq.freqstr
        year, quarter = _make_field_arrays(year, quarter)
        for y, q in zip(year, quarter):
            calendar_year, calendar_month = parsing.quarter_to_myear(y, q, freqstr)
            val = libperiod.period_ordinal(calendar_year, calendar_month, 1, 1, 1, 1, 0, 0, base)
            ordinals.append(val)
    else:
        freq = to_offset(freq, is_period=True)
        base = libperiod.freq_to_dtype_code(freq)
        arrays = _make_field_arrays(year, month, day, hour, minute, second)
        for y, mth, d, h, mn, s in zip(*arrays):
            ordinals.append(libperiod.period_ordinal(y, mth, d, h, mn, s, 0, 0, base))
    return (np.array(ordinals, dtype=np.int64), freq)

def _make_field_arrays(*fields: Any) -> List[np.ndarray]:
    length: Optional[int] = None
    for x in fields:
        if isinstance(x, (list, np.ndarray, ABCSeries)):
            if length is not None and len(x) != length:
                raise ValueError("Mismatched Period array lengths")
            if length is None:
                length = len(x)
    return [np.asarray(x) if isinstance(x, (np.ndarray, list, ABCSeries)) else np.repeat(x, length) for x in fields]
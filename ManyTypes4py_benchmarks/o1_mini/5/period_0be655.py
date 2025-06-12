from __future__ import annotations
from datetime import timedelta
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Literal,
    TypeVar,
    cast,
    overload,
    Sequence,
)
import warnings
import numpy as np
from pandas._libs import algos as libalgos, lib
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    NaTType,
    Timedelta,
    add_overflowsafe,
    astype_overflowsafe,
    dt64arr_to_periodarr as c_dt64arr_to_periodarr,
    get_unit_from_dtype,
    iNaT,
    parsing,
    period as libperiod,
    to_offset,
)
from pandas._libs.tslibs.dtypes import FreqGroup, PeriodDtypeBase
from pandas._libs.tslibs.fields import isleapyear_arr
from pandas._libs.tslibs.offsets import Tick, delta_to_tick
from pandas._libs.tslibs.period import (
    DIFFERENT_FREQ,
    IncompatibleFrequency,
    Period,
    get_period_field_arr,
    period_asfreq_arr,
)
from pandas.util._decorators import cache_readonly, doc
from pandas.core.dtypes.common import ensure_object, pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype, PeriodDtype
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCPeriodIndex,
    ABCSeries,
    ABCTimedeltaArray,
)
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import datetimelike as dtl
import pandas.core.common as com

if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas._typing import (
        AnyArrayLike,
        Dtype,
        FillnaOptions,
        NpDtype,
        NumpySorter,
        NumpyValueArrayLike,
        Self,
        npt,
    )
    from pandas.core.dtypes.dtypes import ExtensionDtype
    from pandas.core.arrays import DatetimeArray, TimedeltaArray
    from pandas.core.arrays.base import ExtensionArray

BaseOffsetT = TypeVar("BaseOffsetT", bound=BaseOffset)
_shared_doc_kwargs = {"klass": "PeriodArray"}


def _field_accessor(
    name: str, docstring: Optional[str] = None
) -> property:
    def f(self: PeriodArray) -> Any:
        base = self.dtype._dtype_code
        result = get_period_field_arr(name, self.asi8, base)
        return result

    f.__name__ = name
    f.__doc__ = docstring
    return property(f)


class PeriodArray(dtl.DatelikeOps, libperiod.PeriodMixin):
    """
    Pandas ExtensionArray for storing Period data.

    Users should use :func:`~pandas.array` to create new instances.

    Parameters
    ----------
    values : Union[PeriodArray, ABCSeries[period], np.ndarray[int], ABCPeriodIndex]
        The data to store. These should be arrays that can be directly
        converted to ordinals without inference or copy (PeriodArray,
        ndarray[int64]), or a box around such an array (Series[period],
        PeriodIndex).
    dtype : Optional[PeriodDtype] = None
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
    _recognized_scalars: tuple[type, ...] = (Period,)
    _is_recognized_dtype: Callable[[Any], bool] = lambda x: isinstance(x, PeriodDtype)
    _infer_matches: tuple[str, ...] = ("period",)

    @property
    def _scalar_type(self) -> type:
        return Period

    _other_ops: list[str] = []
    _bool_ops: list[str] = ["is_leap_year"]
    _object_ops: list[str] = ["start_time", "end_time", "freq"]
    _field_ops: list[str] = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "weekofyear",
        "weekday",
        "week",
        "dayofweek",
        "day_of_week",
        "dayofyear",
        "day_of_year",
        "quarter",
        "qyear",
        "days_in_month",
        "daysinmonth",
    ]
    _datetimelike_ops: list[str] = _field_ops + _object_ops + _bool_ops
    _datetimelike_methods: list[str] = ["strftime", "to_timestamp", "asfreq"]

    def __init__(
        self, values: Any, dtype: Optional[Dtype] = None, copy: bool = False
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
    def _simple_new(
        cls, values: np.ndarray[np.int64], dtype: PeriodDtype
    ) -> PeriodArray:
        assertion_msg = "Should be numpy array of type i8"
        assert isinstance(values, np.ndarray) and values.dtype == "i8", assertion_msg
        return cls(values, dtype=dtype)

    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Any],
        *,
        dtype: Optional[Dtype] = None,
        copy: bool = False,
    ) -> PeriodArray:
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        if isinstance(dtype, PeriodDtype):
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
        cls, strings: Sequence[str], *, dtype: Dtype, copy: bool = False
    ) -> PeriodArray:
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _from_datetime64(
        cls,
        data: np.ndarray[np.datetime64],
        freq: Optional[Union[str, Tick, BaseOffset]] = None,
        tz: Optional[Any] = None,
    ) -> PeriodArray:
        """
        Construct a PeriodArray from a datetime64 array

        Parameters
        ----------
        data : np.ndarray[np.datetime64]
        freq : Optional[Union[str, Tick, BaseOffset]], optional
            Must match the `freq` on the `data` if `data` is a DatetimeIndex
            or Series.
        tz : Optional[Any], optional

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
        cls,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        periods: Optional[int] = None,
        freq: Optional[Union[str, Tick, BaseOffset]] = None,
    ) -> tuple[np.ndarray[np.int64], BaseOffset]:
        periods = dtl.validate_periods(periods)
        if freq is not None:
            freq = Period._maybe_convert_freq(freq)
        if start is not None or end is not None:
            subarr, freq = _get_ordinal_range(start, end, periods, freq)
        else:
            raise ValueError("Not enough parameters to construct Period range")
        return subarr, freq

    @classmethod
    def _from_fields(
        cls,
        *,
        fields: dict[str, Any],
        freq: Union[str, Tick, BaseOffset],
    ) -> PeriodArray:
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

    def __array__(
        self, dtype: Optional[NpDtype] = None, copy: Optional[bool] = None
    ) -> np.ndarray[Any, Any]:
        if dtype == "i8":
            if not copy:
                return np.asarray(self.asi8, dtype=dtype)
            else:
                return np.array(self.asi8, dtype=dtype)
        if copy is False:
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )
        if dtype == bool:
            return ~self._isnan
        return np.array(list(self), dtype=object)

    def __arrow_array__(
        self, type: Optional[Any] = None
    ) -> Any:
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow
        from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

        if type is not None:
            if pyarrow.types.is_integer(type):
                return pyarrow.array(
                    self._ndarray, mask=self.isna(), type=type
                )
            elif isinstance(type, ArrowPeriodType):
                if self.freqstr != type.freq:
                    raise TypeError(
                        f"Not supported to convert PeriodArray to array with different 'freq' ({self.freqstr} vs {type.freq})"
                    )
            else:
                raise TypeError(
                    f"Not supported to convert PeriodArray to '{type}' type"
                )
        period_type = ArrowPeriodType(self.freqstr)
        storage_array = pyarrow.array(
            self._ndarray, mask=self.isna(), type="int64"
        )
        return pyarrow.ExtensionArray.from_storage(period_type, storage_array)

    year = _field_accessor(
        "year",
        """
        The year of the period.

        See Also
        --------
        PeriodIndex.day_of_year : The ordinal day of the year.
        PeriodIndex.dayofyear : The ordinal day of the year.
        PeriodIndex.is_leap_year : Logical indicating if the date belongs to a
            leap year.
        PeriodIndex.weekofyear : The week ordinal of the year.
        PeriodIndex.year : The year of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> idx.year
        Index([2023, 2024, 2025], dtype='int64')
        """
    )
    month = _field_accessor(
        "month",
        """
        The month as January=1, December=12.

        See Also
        --------
        PeriodIndex.days_in_month : The number of days in the month.
        PeriodIndex.daysinmonth : The number of days in the month.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.month
        Index([1, 2, 3], dtype='int64')
        """,
    )
    day = _field_accessor(
        "day",
        """
        The days of the period.

        See Also
        --------
        PeriodIndex.day_of_week : The day of the week with Monday=0, Sunday=6.
        PeriodIndex.day_of_year : The ordinal day of the year.
        PeriodIndex.dayofweek : The day of the week with Monday=0, Sunday=6.
        PeriodIndex.dayofyear : The ordinal day of the year.
        PeriodIndex.days_in_month : The number of days in the month.
        PeriodIndex.daysinmonth : The number of days in the month.
        PeriodIndex.weekday : The day of the week with Monday=0, Sunday=6.

        Examples
        --------
        >>> idx = pd.PeriodIndex(['2020-01-31', '2020-02-28'], freq='D')
        >>> idx.day
        Index([31, 28], dtype='int64')
        """,
    )
    hour = _field_accessor(
        "hour",
        """
        The hour of the period.

        See Also
        --------
        PeriodIndex.minute : The minute of the period.
        PeriodIndex.second : The second of the period.
        PeriodIndex.to_timestamp : Cast to DatetimeArray/Index.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01 10:00", "2023-01-01 11:00"], freq='h')
        >>> idx.hour
        Index([10, 11], dtype='int64')
        """,
    )
    minute = _field_accessor(
        "minute",
        """
        The minute of the period.

        See Also
        --------
        PeriodIndex.hour : The hour of the period.
        PeriodIndex.second : The second of the period.
        PeriodIndex.to_timestamp : Cast to DatetimeArray/Index.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01 10:30:00",
        ...                       "2023-01-01 11:50:00"], freq='min')
        >>> idx.minute
        Index([30, 50], dtype='int64')
        """,
    )
    second = _field_accessor(
        "second",
        """
        The second of the period.

        See Also
        --------
        PeriodIndex.hour : The hour of the period.
        PeriodIndex.minute : The minute of the period.
        PeriodIndex.to_timestamp : Cast to DatetimeArray/Index.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01 10:00:30",
        ...                       "2023-01-01 10:00:31"], freq='s')
        >>> idx.second
        Index([30, 31], dtype='int64')
        """,
    )
    weekofyear = _field_accessor(
        "week",
        """
        The week ordinal of the year.

        See Also
        --------
        PeriodIndex.day_of_week : The day of the week with Monday=0, Sunday=6.
        PeriodIndex.dayofweek : The day of the week with Monday=0, Sunday=6.
        PeriodIndex.week : The week ordinal of the year.
        PeriodIndex.weekday : The day of the week with Monday=0, Sunday=6.
        PeriodIndex.year : The year of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.week  # It can be written `weekofyear`
        Index([5, 9, 13], dtype='int64')
        """,
    )
    week = weekofyear
    day_of_week = _field_accessor(
        "day_of_week",
        """
        The day of the week with Monday=0, Sunday=6.

        See Also
        --------
        PeriodIndex.day : The days of the period.
        PeriodIndex.day_of_week : The day of the week with Monday=0, Sunday=6.
        PeriodIndex.day_of_year : The ordinal day of the year.
        PeriodIndex.dayofweek : The day of the week with Monday=0, Sunday=6.
        PeriodIndex.dayofyear : The ordinal day of the year.
        PeriodIndex.week : The week ordinal of the year.
        PeriodIndex.weekday : The day of the week with Monday=0, Sunday=6.
        PeriodIndex.weekofyear : The week ordinal of the year.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01", "2023-01-02", "2023-01-03"], freq="D")
        >>> idx.weekday
        Index([6, 0, 1], dtype='int64')
        """,
    )
    dayofweek = day_of_week
    weekday = dayofweek
    dayofyear = day_of_year = _field_accessor(
        "day_of_year",
        """
        The ordinal day of the year.

        See Also
        --------
        PeriodIndex.day : The days of the period.
        PeriodIndex.day_of_week : The day of the week with Monday=0, Sunday=6.
        PeriodIndex.dayofweek : The day of the week with Monday=0, Sunday=6.
        PeriodIndex.dayofyear : The ordinal day of the year.
        PeriodIndex.weekday : The day of the week with Monday=0, Sunday=6.
        PeriodIndex.weekofyear : The week ordinal of the year.
        PeriodIndex.year : The year of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-10", "2023-02-01", "2023-03-01"], freq="D")
        >>> idx.dayofyear
        Index([10, 32, 60], dtype='int64')

        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> idx
        PeriodIndex(['2023', '2024', '2025'], dtype='period[Y-DEC]')
        >>> idx.dayofyear
        Index([365, 366, 365], dtype='int64')
        """,
    )
    quarter = _field_accessor(
        "quarter",
        """
        The quarter of the date.

        See Also
        --------
        PeriodIndex.qyear : Fiscal year the Period lies in according to its
            starting-quarter.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.quarter
        Index([1, 1, 1], dtype='int64')
        """,
    )
    qyear = _field_accessor(
        "qyear",
        """
        Fiscal year the Period lies in according to its starting-quarter.

        The `year` and the `qyear` of the period will be the same if the fiscal
        and calendar years are the same. When they are not, the fiscal year
        can be different from the calendar year of the period.

        Returns
        -------
        int
            The fiscal year of the period.

        See Also
        --------
        PeriodIndex.quarter : The quarter of the date.
        PeriodIndex.year : The year of the period.

        Examples
        --------
        If the natural and fiscal year are the same, `qyear` and `year` will
        be the same.

        >>> per = pd.Period('2018Q1', freq='Q')
        >>> per.qyear
        2018
        >>> per.year
        2018

        If the fiscal year starts in April (`Q-MAR`), the first quarter of
        2018 will start in April 2017. `year` will then be 2017, but `qyear`
        will be the fiscal year, 2018.

        >>> per = pd.Period('2018Q1', freq='Q-MAR')
        >>> per.start_time
        Timestamp('2017-04-01 00:00:00')
        >>> per.qyear
        2018
        >>> per.year
        2017
        """,
    )
    days_in_month = _field_accessor(
        "days_in_month",
        """
        The number of days in the month.

        See Also
        --------
        PeriodIndex.day : The days of the period.
        PeriodIndex.days_in_month : The number of days in the month.
        PeriodIndex.daysinmonth : The number of days in the month.
        PeriodIndex.month : The month as January=1, December=12.

        Examples
        --------
        For Series:

        >>> period = pd.period_range('2020-1-1 00:00', '2020-3-1 00:00', freq='M')
        >>> s = pd.Series(period)
        >>> s
        0   2020-01
        1   2020-02
        2   2020-03
        dtype: period[M]
        >>> s.dt.days_in_month
        0    31
        1    29
        2    31
        dtype: int64

        For PeriodIndex:

        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.days_in_month   # It can be also entered as `daysinmonth`
        Index([31, 28, 31], dtype='int64')
        """,
    )
    daysinmonth = days_in_month

    @property
    def is_leap_year(self) -> np.ndarray[np.bool_]:
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
        self,
        freq: Optional[Union[str, Tick, BaseOffset]] = None,
        how: Literal["start", "end", "E", "S"] = "start",
    ) -> DatetimeArray:
        """
        Cast to DatetimeArray/Index.

        Parameters
        ----------
        freq : Union[str, Tick, BaseOffset], optional
            Target frequency. The default is 'D' for week or longer,
            's' otherwise.
        how : {'s', 'e', 'start', 'end'}, default 'start'
            Whether to use the start or end of the time period being converted.

            * 'E', 'END', or 'FINISH' for end,
            * 'S', 'START', or 'BEGIN' for start.

            January 31st ('END') vs. January 1st ('START') for example.

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

    def _box_func(self, x: np.int64) -> Period:
        return Period._from_ordinal(ordinal=x, freq=self.freq)

    @doc(**_shared_doc_kwargs, other="PeriodIndex", other_name="PeriodIndex")
    def asfreq(
        self,
        freq: Optional[Union[str, Tick, BaseOffset]] = None,
        how: Literal["E", "S", "end", "start", "finisht", "begin"] = "E",
    ) -> PeriodArray:
        """
        Convert the {klass} to the specified frequency `freq`.

        Equivalent to applying :meth:`pandas.Period.asfreq` with the given arguments
        to each :class:`~pandas.Period` in this {klass}.

        Parameters
        ----------
        freq : str
            A frequency.
        how : str {{'E', 'S'}}, default 'E'
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
        {other}.asfreq : Convert each Period in a {other_name} to the given frequency.
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
        base2 = freq._period_dtype_code if freq else 0
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

    def _formatter(
        self, boxed: bool = False
    ) -> Callable[[Any], str]:
        if boxed:
            return str
        return "'{}'".format

    def _format_native_types(
        self,
        *,
        na_rep: str = "NaT",
        date_format: Optional[str] = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        actually format my specific types
        """
        return libperiod.period_array_strftime(
            self.asi8, self.dtype._dtype_code, na_rep, date_format
        )

    def astype(
        self,
        dtype: Dtype,
        copy: bool = True,
    ) -> Any:
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

    def searchsorted(
        self,
        value: Any,
        side: Literal["left", "right"] = "left",
        sorter: Optional[np.ndarray[np.intp]] = None,
    ) -> int:
        npvalue = self._validate_setitem_value(value).view("M8[ns]")
        m8arr = self._ndarray.view("M8[ns]")
        return m8arr.searchsorted(npvalue, side=side, sorter=sorter)

    def _pad_or_backfill(
        self,
        *,
        method: Literal["pad", "backfill"],
        limit: Optional[int] = None,
        limit_area: Optional[str] = None,
        copy: bool = True,
    ) -> PeriodArray:
        dta = self.view("M8[ns]")
        result = dta._pad_or_backfill(
            method=method, limit=limit, limit_area=limit_area, copy=copy
        )
        if copy:
            return cast("Self", result.view(self.dtype))
        else:
            return self

    def _addsub_int_array_or_scalar(
        self,
        other: Union[np.ndarray[np.int64], int],
        op: Callable[[Any, Any], Any],
    ) -> PeriodArray:
        """
        Add or subtract array of integers.

        Parameters
        ----------
        other : np.ndarray[np.int64] or int
        op : {operator.add, operator.sub}
            The operation to perform.

        Returns
        -------
        PeriodArray
        """
        assert op in [operator.add, operator.sub]
        if op is operator.sub:
            other = -cast(int, other)
        else:
            other = cast(int, other)
        res_values = add_overflowsafe(self.asi8, np.asarray(other, dtype="i8"))
        return type(self)(res_values, dtype=self.dtype)

    def _add_offset(self, other: BaseOffset) -> PeriodArray:
        assert not isinstance(other, Tick)
        self._require_matching_freq(other, base=True)
        return self._addsub_int_array_or_scalar(other.n, operator.add)

    def _add_timedeltalike_scalar(
        self, other: Union[timedelta, Tick, np.timedelta64]
    ) -> PeriodArray:
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

    def _add_timedelta_arraylike(
        self, other: np.ndarray[np.timedelta64]
    ) -> PeriodArray:
        """
        Parameters
        ----------
        other : TimedeltaArray or np.ndarray[np.timedelta64]

        Returns
        -------
        PeriodArray
        """
        if not self.dtype._is_tick_like():
            raise TypeError(
                f"Cannot add or subtract timedelta64[ns] dtype from {self.dtype}"
            )
        dtype = np.dtype(f"m8[{self.dtype._td64_unit}]")
        try:
            delta = astype_overflowsafe(
                np.asarray(other), dtype=dtype, copy=False, round_ok=False
            )
        except ValueError as err:
            raise IncompatibleFrequency(
                "Cannot add/subtract timedelta-like from PeriodArray that is not an integer multiple of the PeriodArray's freq."
            ) from err
        res_values = add_overflowsafe(self.asi8, np.asarray(delta.view("i8")))
        return type(self)(res_values, dtype=self.dtype)

    def _check_timedeltalike_freq_compat(
        self, other: Union[timedelta, np.timedelta64, Tick, np.ndarray[np.timedelta64], ABCTimedeltaArray]
    ) -> Union[int, np.ndarray[np.int64]]:
        """
        Arithmetic operations with timedelta-like scalars or array `other`
        are only valid if `other` is an integer multiple of `self.freq`.
        If the operation is valid, find that integer multiple.  Otherwise,
        raise because the operation is invalid.

        Parameters
        ----------
        other : timedelta, np.timedelta64, Tick,
                ndarray[np.timedelta64], TimedeltaArray, TimedeltaIndex

        Returns
        -------
        multiple : int or np.ndarray[np.int64]

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

    def _reduce(
        self,
        name: str,
        *,
        skipna: bool = True,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> Any:
        result = super()._reduce(
            name, skipna=skipna, keepdims=keepdims, **kwargs
        )
        if keepdims and isinstance(result, np.ndarray):
            return self._from_sequence(result, dtype=self.dtype)
        return result


def raise_on_incompatible(
    left: PeriodArray, right: Optional[Union[BaseOffset, Period, np.ndarray[Any, Any], ABCTimedeltaArray]]
) -> IncompatibleFrequency:
    """
    Helper function to render a consistent error message when raising
    IncompatibleFrequency.

    Parameters
    ----------
    left : PeriodArray
    right : Optional[Union[BaseOffset, Period, np.ndarray[Any, Any], ABCTimedeltaArray]]
        The other object to compare frequencies with.

    Returns
    -------
    IncompatibleFrequency
        Exception to be raised by the caller.
    """
    if isinstance(right, (np.ndarray, ABCTimedeltaArray)) or right is None:
        other_freq: Optional[str] = None
    elif isinstance(right, BaseOffset):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "PeriodDtype\\[B\\] is deprecated", category=FutureWarning
            )
            other_freq = PeriodDtype(right)._freqstr
    elif isinstance(right, (ABCPeriodIndex, PeriodArray, Period)):
        other_freq = right.freqstr
    else:
        other_freq = delta_to_tick(Timedelta(right)).freqstr
    own_freq = PeriodDtype(left.freq)._freqstr
    msg = DIFFERENT_FREQ.format(
        cls=type(left).__name__, own_freq=own_freq, other_freq=other_freq
    )
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
    data : Sequence[Any]
        A sequence of Period objects. These are required to all have
        the same ``freq.`` Missing values can be indicated by ``None``
        or ``pandas.NaT``.
    freq : Optional[Union[str, Tick, BaseOffset]], default None
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
        return PeriodArray._from_datetime64(data, freq)
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
        dtype = PeriodDtype(freq)
    else:
        dtype = None
    if arrdata.dtype.kind == "f" and len(arrdata) > 0:
        raise TypeError(
            "PeriodIndex does not allow floating point in construction"
        )
    if arrdata.dtype.kind in ("i", "u"):
        arr = arrdata.astype(np.int64, copy=False)
        ordinals = libperiod.from_ordinals(arr, freq)
        return PeriodArray(ordinals, dtype=dtype)
    data = ensure_object(arrdata)
    if freq is None:
        freq = libperiod.extract_freq(data)
    dtype = PeriodDtype(freq)
    return PeriodArray._from_sequence(data, dtype=dtype)


@overload
def validate_dtype_freq(
    dtype: Optional[Dtype], freq: Optional[Union[str, Tick, BaseOffset]]
) -> BaseOffset:
    ...


@overload
def validate_dtype_freq(
    dtype: Dtype, freq: Optional[Union[str, Tick, BaseOffset]]
) -> BaseOffset:
    ...


def validate_dtype_freq(
    dtype: Optional[Dtype],
    freq: Optional[Union[str, Tick, BaseOffset]],
) -> BaseOffset:
    """
    If both a dtype and a freq are available, ensure they match.  If only
    dtype is available, extract the implied freq.

    Parameters
    ----------
    dtype : Optional[Dtype]
    freq : Optional[Union[str, Tick, BaseOffset]]

    Returns
    -------
    BaseOffset
        The validated frequency.

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
    return cast(BaseOffset, freq)


def dt64arr_to_periodarr(
    data: np.ndarray[np.datetime64],
    freq: Optional[Union[str, Tick, BaseOffset]],
    tz: Optional[Any] = None,
) -> tuple[np.ndarray[np.int64], BaseOffset]:
    """
    Convert an datetime-like array to values Period ordinals.

    Parameters
    ----------
    data : np.ndarray[np.datetime64]
    freq : Optional[Union[str, Tick, BaseOffset]]
        Must match the `freq` on the `data` if `data` is a DatetimeIndex
        or Series.
    tz : Optional[Any]

    Returns
    -------
    tuple[np.ndarray[np.int64], BaseOffset]
        (ordinals, freq)
    """
    if not isinstance(data.dtype, np.dtype) or data.dtype.kind != "M":
        raise ValueError(f"Wrong dtype: {data.dtype}")
    if freq is None:
        if isinstance(data, ABCIndex):
            data, freq = data._values, data.freq
        elif isinstance(data, ABCSeries):
            data, freq = data._values, data.dt.freq
    elif isinstance(data, (ABCIndex, ABCSeries)):
        data = data._values
    reso = get_unit_from_dtype(data.dtype)
    freq = Period._maybe_convert_freq(freq)
    base = freq._period_dtype_code
    return (
        c_dt64arr_to_periodarr(data.view("i8"), base, tz, reso=reso),
        freq,
    )


def _get_ordinal_range(
    start: Optional[Any],
    end: Optional[Any],
    periods: Optional[int],
    freq: Optional[Union[str, Tick, BaseOffset]],
    mult: int = 1,
) -> tuple[np.ndarray[np.int64], BaseOffset]:
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError(
            "Of the three parameters: start, end, and periods, exactly two must be specified"
        )
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
            data = np.arange(
                end.ordinal - periods + mult,
                end.ordinal + 1,
                mult,
                dtype=np.int64,
            )
        else:
            data = np.arange(
                start.ordinal, start.ordinal + periods, mult, dtype=np.int64
            )
    else:
        data = np.arange(
            start.ordinal, end.ordinal + 1, mult, dtype=np.int64
        )
    return data, freq


def _range_from_fields(
    year: Optional[Any] = None,
    month: Optional[Any] = None,
    quarter: Optional[Any] = None,
    day: Optional[Any] = None,
    hour: Optional[Any] = None,
    minute: Optional[Any] = None,
    second: Optional[Any] = None,
    freq: Optional[Union[str, Tick, BaseOffset]] = None,
) -> tuple[np.ndarray[np.int64], BaseOffset]:
    if hour is None:
        hour = 0
    if minute is None:
        minute = 0
    if second is None:
        second = 0
    if day is None:
        day = 1
    ordinals: list[np.int64] = []
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
        year_arr, quarter_arr = _make_field_arrays(year, quarter)
        for y, q in zip(year_arr, quarter_arr):
            calendar_year, calendar_month = parsing.quarter_to_myear(
                y, q, freqstr
            )
            val = libperiod.period_ordinal(
                calendar_year,
                calendar_month,
                1,
                1,
                1,
                1,
                0,
                0,
                base,
            )
            ordinals.append(val)
    else:
        freq = to_offset(freq, is_period=True)
        base = libperiod.freq_to_dtype_code(freq)
        arrays = _make_field_arrays(year, month, day, hour, minute, second)
        for y, mth, d, h, mn, s in zip(*arrays):
            ordinals.append(
                libperiod.period_ordinal(y, mth, d, h, mn, s, 0, 0, base)
            )
    return np.array(ordinals, dtype=np.int64), freq


def _make_field_arrays(*fields: Any) -> list[np.ndarray[Any, Any]]:
    length: Optional[int] = None
    for x in fields:
        if isinstance(x, (list, np.ndarray, ABCSeries)):
            if length is not None and len(x) != length:
                raise ValueError("Mismatched Period array lengths")
            if length is None:
                length = len(x)
    return [
        np.asarray(x) if isinstance(x, (np.ndarray, list, ABCSeries)) else np.repeat(x, length)
        for x in fields
    ]

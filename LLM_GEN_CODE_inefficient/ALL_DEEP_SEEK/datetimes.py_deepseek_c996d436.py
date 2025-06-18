from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    tzinfo,
)
from typing import (
    TYPE_CHECKING,
    TypeVar,
    cast,
    overload,
    Optional,
    Union,
    List,
    Tuple,
    Generator,
    Iterator,
    Any,
    Dict,
    Sequence,
    Callable,
    Type,
    Literal,
)
import warnings

import numpy as np
from numpy.typing import NDArray

from pandas._config import using_string_dtype
from pandas._config.config import get_option

from pandas._libs import (
    lib,
    tslib,
)
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    NaTType,
    Resolution,
    Timestamp,
    astype_overflowsafe,
    fields,
    get_resolution,
    get_supported_dtype,
    get_unit_from_dtype,
    ints_to_pydatetime,
    is_date_array_normalized,
    is_supported_dtype,
    is_unitless,
    normalize_i8_timestamps,
    timezones,
    to_offset,
    tz_convert_from_utc,
    tzconversion,
)
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_inclusive

from pandas.core.dtypes.common import (
    DT64NS_DTYPE,
    INT64_DTYPE,
    is_bool_dtype,
    is_float_dtype,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)
from pandas.core.dtypes.missing import isna

from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com

from pandas.tseries.frequencies import get_period_alias
from pandas.tseries.offsets import (
    Day,
    Tick,
)

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Iterator,
    )

    from pandas._typing import (
        ArrayLike,
        DateTimeErrorChoices,
        DtypeObj,
        IntervalClosedType,
        Self,
        TimeAmbiguous,
        TimeNonexistent,
        npt,
    )

    from pandas import (
        DataFrame,
        Timedelta,
    )
    from pandas.core.arrays import PeriodArray

    _TimestampNoneT1 = TypeVar("_TimestampNoneT1", Timestamp, None)
    _TimestampNoneT2 = TypeVar("_TimestampNoneT2", Timestamp, None)


_ITER_CHUNKSIZE = 10_000


@overload
def tz_to_dtype(tz: tzinfo, unit: str = ...) -> DatetimeTZDtype: ...


@overload
def tz_to_dtype(tz: None, unit: str = ...) -> np.dtype[np.datetime64]: ...


def tz_to_dtype(
    tz: Optional[tzinfo], unit: str = "ns"
) -> Union[np.dtype[np.datetime64], DatetimeTZDtype]:
    """
    Return a datetime64[ns] dtype appropriate for the given timezone.

    Parameters
    ----------
    tz : tzinfo or None
    unit : str, default "ns"

    Returns
    -------
    np.dtype or Datetime64TZDType
    """
    if tz is None:
        return np.dtype(f"M8[{unit}]")
    else:
        return DatetimeTZDtype(tz=tz, unit=unit)


def _field_accessor(name: str, field: str, docstring: Optional[str] = None) -> property:
    def f(self) -> NDArray[Any]:
        values = self._local_timestamps()

        if field in self._bool_ops:
            result: NDArray[Any]

            if field.endswith(("start", "end")):
                freq = self.freq
                month_kw = 12
                if freq:
                    kwds = freq.kwds
                    month_kw = kwds.get("startingMonth", kwds.get("month", month_kw))

                if freq is not None:
                    freq_name = freq.name
                else:
                    freq_name = None
                result = fields.get_start_end_field(
                    values, field, freq_name, month_kw, reso=self._creso
                )
            else:
                result = fields.get_date_field(values, field, reso=self._creso)

            # these return a boolean by-definition
            return result

        result = fields.get_date_field(values, field, reso=self._creso)
        result = self._maybe_mask_results(result, fill_value=None, convert="float64")

        return result

    f.__name__ = name
    f.__doc__ = docstring
    return property(f)


# error: Definition of "_concat_same_type" in base class "NDArrayBacked" is
# incompatible with definition in base class "ExtensionArray"
class DatetimeArray(dtl.TimelikeOps, dtl.DatelikeOps):  # type: ignore[misc]
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

    _typ = "datetimearray"
    _internal_fill_value = np.datetime64("NaT", "ns")
    _recognized_scalars = (datetime, np.datetime64)
    _is_recognized_dtype = lambda x: lib.is_np_dtype(x, "M") or isinstance(
        x, DatetimeTZDtype
    )
    _infer_matches = ("datetime", "datetime64", "date")

    @property
    def _scalar_type(self) -> Type[Timestamp]:
        return Timestamp

    # define my properties & methods for delegation
    _bool_ops: List[str] = [
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
        "is_leap_year",
    ]
    _field_ops: List[str] = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "weekday",
        "dayofweek",
        "day_of_week",
        "dayofyear",
        "day_of_year",
        "quarter",
        "days_in_month",
        "daysinmonth",
        "microsecond",
        "nanosecond",
    ]
    _other_ops: List[str] = ["date", "time", "timetz"]
    _datetimelike_ops: List[str] = (
        _field_ops + _bool_ops + _other_ops + ["unit", "freq", "tz"]
    )
    _datetimelike_methods: List[str] = [
        "to_period",
        "tz_localize",
        "tz_convert",
        "normalize",
        "strftime",
        "round",
        "floor",
        "ceil",
        "month_name",
        "day_name",
        "as_unit",
    ]

    # ndim is inherited from ExtensionArray, must exist to ensure
    #  Timestamp.__richcmp__(DateTimeArray) operates pointwise

    # ensure that operations with numpy arrays defer to our implementation
    __array_priority__ = 1000

    # -----------------------------------------------------------------
    # Constructors

    _dtype: Union[np.dtype[np.datetime64], DatetimeTZDtype]
    _freq: Optional[BaseOffset] = None

    @classmethod
    def _from_scalars(cls, scalars: Any, *, dtype: DtypeObj) -> Self:
        if lib.infer_dtype(scalars, skipna=True) not in ["datetime", "datetime64"]:
            # TODO: require any NAs be valid-for-DTA
            # TODO: if dtype is passed, check for tzawareness compat?
            raise ValueError
        return cls._from_sequence(scalars, dtype=dtype)

    @classmethod
    def _validate_dtype(cls, values: Any, dtype: DtypeObj) -> DtypeObj:
        # used in TimeLikeOps.__init__
        dtype = _validate_dt64_dtype(dtype)
        _validate_dt64_dtype(values.dtype)
        if isinstance(dtype, np.dtype):
            if values.dtype != dtype:
                raise ValueError("Values resolution does not match dtype.")
        else:
            vunit = np.datetime_data(values.dtype)[0]
            if vunit != dtype.unit:
                raise ValueError("Values resolution does not match dtype.")
        return dtype

    # error: Signature of "_simple_new" incompatible with supertype "NDArrayBacked"
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls,
        values: npt.NDArray[np.datetime64],
        freq: Optional[BaseOffset] = None,
        dtype: Union[np.dtype[np.datetime64], DatetimeTZDtype] = DT64NS_DTYPE,
    ) -> Self:
        assert isinstance(values, np.ndarray)
        assert dtype.kind == "M"
        if isinstance(dtype, np.dtype):
            assert dtype == values.dtype
            assert not is_unitless(dtype)
        else:
            # DatetimeTZDtype. If we have e.g. DatetimeTZDtype[us, UTC],
            #  then values.dtype should be M8[us].
            assert dtype._creso == get_unit_from_dtype(values.dtype)

        result = super()._simple_new(values, dtype)
        result._freq = freq
        return result

    @classmethod
    def _from_sequence(cls, scalars: Any, *, dtype: Optional[DtypeObj] = None, copy: bool = False) -> Self:
        return cls._from_sequence_not_strict(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_sequence_not_strict(
        cls,
        data: Any,
        *,
        dtype: Optional[DtypeObj] = None,
        copy: bool = False,
        tz: Union[tzinfo, Literal[lib.no_default]] = lib.no_default,
        freq: Union[str, BaseOffset, Literal[lib.no_default], None] = lib.no_default,
        dayfirst: bool = False,
        yearfirst: bool = False,
        ambiguous: TimeAmbiguous = "raise",
    ) -> Self:
        """
        A non-strict version of _from_sequence, called from DatetimeIndex.__new__.
        """

        # if the user either explicitly passes tz=None or a tz-naive dtype, we
        #  disallows inferring a tz.
        explicit_tz_none = tz is None
        if tz is lib.no_default:
            tz = None
        else:
            tz = timezones.maybe_get_tz(tz)

        dtype = _validate_dt64_dtype(dtype)
        # if dtype has an embedded tz, capture it
        tz = _validate_tz_from_dtype(dtype, tz, explicit_tz_none)

        unit = None
        if dtype is not None:
            unit = dtl.dtype_to_unit(dtype)

        data, copy = dtl.ensure_arraylike_for_datetimelike(
            data, copy, cls_name="DatetimeArray"
        )
        inferred_freq = None
        if isinstance(data, DatetimeArray):
            inferred_freq = data.freq

        subarr, tz = _sequence_to_dt64(
            data,
            copy=copy,
            tz=tz,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            ambiguous=ambiguous,
            out_unit=unit,
        )
        # We have to call this again after possibly inferring a tz above
        _validate_tz_from_dtype(dtype, tz, explicit_tz_none)
        if tz is not None and explicit_tz_none:
            raise ValueError(
                "Passed data is timezone-aware, incompatible with 'tz=None'. "
                "Use obj.tz_localize(None) instead."
            )

        data_unit = np.datetime_data(subarr.dtype)[0]
        data_dtype = tz_to_dtype(tz, data_unit)
        result = cls._simple_new(subarr, freq=inferred_freq, dtype=data_dtype)
        if unit is not None and unit != result.unit:
            # If unit was specified in user-passed dtype, cast to it here
            result = result.as_unit(unit)

        validate_kwds = {"ambiguous": ambiguous}
        result._maybe_pin_freq(freq, validate_kwds)
        return result

    @classmethod
    def _generate_range(
        cls,
        start: Optional[Timestamp],
        end: Optional[Timestamp],
        periods: Optional[int],
        freq: Any,
        tz: Optional[tzinfo] = None,
        normalize: bool = False,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
        inclusive: IntervalClosedType = "both",
        *,
        unit: Optional[str] = None,
    ) -> Self:
        periods = dtl.validate_periods(periods)
        if freq is None and any(x is None for x in [periods, start, end]):
            raise ValueError("Must provide freq argument if no data is supplied")

        if com.count_not_none(start, end, periods, freq) != 3:
            raise ValueError(
                "Of the four parameters: start, end, periods, "
                "and freq, exactly three must be specified"
            )
        freq = to_offset(freq)

        if start is not None:
            start = Timestamp(start)

        if end is not None:
            end = Timestamp(end)

        if start is NaT or end is NaT:
            raise ValueError("Neither `start` nor `end` can be NaT")

        if unit is not None:
            if unit not in ["s", "ms", "us", "ns"]:
                raise ValueError("'unit' must be one of 's', 'ms', 'us', 'ns'")
        else:
            unit = "ns"

        if start is not None:
            start = start.as_unit(unit, round_ok=False)
        if end is not None:
            end = end.as_unit(unit, round_ok=False)

        left_inclusive, right_inclusive = validate_inclusive(inclusive)
        start, end = _maybe_normalize_endpoints(start, end, normalize)
        tz = _infer_tz_from_endpoints(start, end, tz)

        if tz is not None:
            # Localize the start and end arguments
            start = _maybe_localize_point(start, freq, tz, ambiguous, nonexistent)
            end = _maybe_localize_point(end, freq, tz, ambiguous, nonexistent)

        if freq is not None:
            # We break Day arithmetic (fixed 24 hour) here and opt for
            # Day to mean calendar day (23/24/25 hour). Therefore, strip
            # tz info from start and day to avoid DST arithmetic
            if isinstance(freq, Day):
                if start is not None:
                    start = start.tz_localize(None)
                if end is not None:
                    end = end.tz_localize(None)

            if isinstance(freq, Tick):
                i8values = generate_regular_range(start, end, periods, freq, unit=unit)
            else:
                xdr = _generate_range(
                    start=start, end=end, periods=periods, offset=freq, unit=unit
                )
                i8values = np.array([x._value for x in xdr], dtype=np.int64)

            endpoint_tz = start.tz if start is not None else end.tz

            if tz is not None and endpoint_tz is None:
                if not timezones.is_utc(tz):
                    # short-circuit tz_localize_to_utc which would make
                    #  an unnecessary copy with UTC but be a no-op.
                    creso = abbrev_to_npy_unit(unit)
                    i8values = tzconversion.tz_localize_to_utc(
                        i8values,
                        tz,
                        ambiguous=ambiguous,
                        nonexistent=nonexistent,
                        creso=creso,
                    )

                # i8values is localized datetime64 array -> have to convert
                # start/end as well to compare
                if start is not None:
                    start = start.tz_localize(tz, ambiguous, nonexistent)
                if end is not None:
                    end = end.tz_localize(tz, ambiguous, nonexistent)
        else:
            # Create a linearly spaced date_range in local time
            # Nanosecond-granularity timestamps aren't always correctly
            # representable with doubles, so we limit the range that we
            # pass to np.linspace as much as possible
            periods = cast(int, periods)
            i8values = (
                np.linspace(0, end._value - start._value, periods, dtype="int64")
                + start._value
            )
            if i8values.dtype != "i8":
                # 2022-01-09 I (brock) am not sure if it is possible for this
                #  to overflow and cast to e.g. f8, but if it does we need to cast
                i8values = i8values.astype("i8")

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

        dt64_values = i8values.view(f"datetime64[{unit}]")
        dtype = tz_to_dtype(tz, unit=unit)
        return cls._simple_new(dt64_values, freq=freq, dtype=dtype)

    # -----------------------------------------------------------------
    # DatetimeLike Interface

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

    # -----------------------------------------------------------------
    # Descriptive Properties

    def _box_func(self, x: np.datetime64) -> Union[Timestamp, NaTType]:
        # GH#42228
        value = x.view("i8")
        ts = Timestamp._from_value_and_reso(value, reso=self._creso, tz=self.tz)
        return ts

    @property
    # error: Return type "Union[dtype, DatetimeTZDtype]" of "dtype"
    # incompatible with return type "ExtensionDtype" in supertype
    # "ExtensionArray"
    def dtype(self) -> Union[np.dtype[np.datetime64], DatetimeTZDtype]:  # type: ignore[override]
        """
        The dtype for the DatetimeArray.

        .. warning::

           DatetimeArray is currently experimental, and its API may change
           without warning. In particular, :attr:`DatetimeArray.dtype` is
           expected to change to always be an instance of an ``ExtensionDtype``
           subclass.

        Returns
        -------
        numpy.dtype or DatetimeTZDtype
            If the values are tz-naive, then ``np.dtype('datetime64[ns]')``
            is returned.

            If the values are tz-aware, then the ``DatetimeTZDtype``
            is returned.
        """
        return self._dtype

    @property
    def tz(self) -> Optional[tzinfo]:
        """
        Return the timezone.

        Returns
        -------
        zoneinfo.ZoneInfo,, datetime.tzinfo, pytz.tzinfo.BaseTZInfo, dateutil.tz.tz.tzfile, or None
            Returns None when the array is tz-naive.

        See Also
        --------
        DatetimeIndex.tz_localize : Localize tz-naive DatetimeIndex to a
            given time zone, or remove timezone from a tz-aware DatetimeIndex.
        DatetimeIndex.tz_convert : Convert tz-aware DatetimeIndex from
            one time zone to another.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[s, UTC]
        >>> s.dt.tz
        datetime.timezone.utc

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(
        ...     ["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"]
        ... )
        >>> idx.tz
        datetime.timezone.utc
        """  # noqa: E501
        # GH 18595
        return getattr(self.dtype, "tz", None)

    @tz.setter
    def tz(self, value: Any) -> None:
        # GH 3746: Prevent localizing or converting the index by setting tz
        raise AttributeError(
            "Cannot directly set timezone. Use tz_localize() "
            "or tz_convert() as appropriate"
        )

    @property
    def tzinfo(self) -> Optional[tzinfo]:
        """
        Alias for tz attribute
        """
        return self.tz

    @property  # NB: override with cache_readonly in immutable subclasses
    def is_normalized(self) -> bool:
        """
        Returns True if all of the dates are at midnight ("no time")
        """
        return is_date_array_normalized(self.asi8, self.tz, reso=self._creso)

    @property  # NB: override with cache_readonly in immutable subclasses
    def _resolution_obj(self) -> Resolution:
        return get_resolution(self.asi8, self.tz, reso=self._creso)

    # ----------------------------------------------------------------
    # Array-Like / EA-Interface Methods

    def __array__(self, dtype: Optional[DtypeObj] = None, copy: Optional[bool] = None) -> NDArray[Any]:
        if dtype is None and self.tz:
            # The default for tz-aware is object, to preserve tz info
            dtype = object

        return super().__array__(dtype=dtype, copy=copy)

    def __iter__(self) -> Iterator[Union[Timestamp, NaTType]]:
        """
        Return an iterator over the boxed values

        Yields
        ------
        tstamp : Timestamp
        """
        if self.ndim > 1:
            for i in range(len(self)):
                yield self[i]
        else:
            # convert in chunks of 10k for efficiency
            data = self.asi8
            length = len(self)
            chunksize = _ITER_CHUNKSIZE
            chunks = (length // chunksize) + 1

            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, length)
                converted = ints_to_pydatetime(
                    data[start_i:end_i],
                    tz=self.tz,
                    box="timestamp",
                    reso=self._creso,
                )
                yield from converted

    def astype(self, dtype: DtypeObj, copy: bool = True) -> Any:
        # We handle
        #   --> datetime
        #   --> period
        # DatetimeLikeArrayMixin Super handles the rest.
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self

        elif isinstance(dtype, ExtensionDtype):
            if not isinstance(dtype, DatetimeTZDtype):
                # e.g. Sparse[datetime64[ns]]
                return super().astype(dtype, copy=copy)
            elif self.tz is None:
                # pre-2.0 this did self.tz_localize(dtype.tz), which did not match
                #  the Series behavior which did
                #  values.tz_localize("UTC").tz_convert(dtype.tz)
                raise TypeError(
                    "Cannot use .astype to convert from timezone-naive dtype to "
                    "timezone-aware dtype. Use obj.tz_localize instead or "
                    "series.dt.tz_localize instead"
                )
            else:
                # tzaware unit conversion e.g. datetime64[s, UTC]
                np_dtype = np.dtype(dtype.str)
                res_values = astype_overflowsafe(self._ndarray, np_dtype, copy=copy)
                return type(self)._simple_new(res_values, dtype=dtype, freq=self.freq)

        elif (
            self.tz is None
            and lib.is_np_dtype(dtype, "M")
            and not is_unitless(dtype)
            and is_supported_dtype(dtype)
        ):
            # unit conversion e.g. datetime64[s]
            res_values = astype_overflowsafe(self._ndarray, dtype, copy=True)
            return type(self)._simple_new(res_values, dtype=res_values.dtype)
            # TODO: preserve freq?

        elif self.tz is not None and lib.is_np_dtype(dtype, "M"):
            # pre-2.0 behavior for DTA/DTI was
            #  values.tz_convert("UTC").tz_localize(None), which did not match
            #  the Series behavior
            raise TypeError(
                "Cannot use .astype to convert from timezone-aware dtype to "
                "timezone-naive dtype. Use obj.tz_localize(None) or "
                "obj.tz_convert('UTC').tz_localize(None) instead."
            )

        elif (
            self.tz is None
            and lib.is_np_dtype(dtype, "M")
            and dtype != self.dtype
            and is_unitless(dtype)
        ):
            raise TypeError(
                "Casting to unit-less dtype 'datetime64' is not supported. "
                "Pass e.g. 'datetime64[ns]' instead."
            )

        elif isinstance(dtype, PeriodDtype):
            return self.to_period(freq=dtype.freq)
        return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy)

    # -----------------------------------------------------------------
    # Rendering Methods

    def _format_native_types(
        self, *, na_rep: Union[str, float] = "NaT", date_format: Optional[str] = None, **kwargs
    ) -> NDArray[np.object_]:
        if date_format is None and self._is_dates_only:
            # Only dates and no timezone: provide a default format
            date_format = "%Y-%m-%d"

        return tslib.format_array_from_datetime(
            self.asi8, tz=self.tz, format=date_format, na_rep=na_rep, reso=self._creso
        )

    # -----------------------------------------------------------------
    # Comparison Methods

    def _assert_tzawareness_compat(self, other: Any) -> None:
        # adapted from _Timestamp._assert_tzawareness_compat
        other_tz = getattr(other, "tzinfo", None)
        other_dtype = getattr(other, "dtype", None)

        if isinstance(other_dtype, DatetimeTZDtype):
            # Get tzinfo from Series dtype
            other_tz = other.dtype.tz
        if other is NaT:
            # pd.NaT quacks both aware and naive
            pass
        elif self.tz is None:
            if other_tz is not None:
                raise TypeError(
                    "Cannot compare tz-naive and tz-aware datetime-like objects."
                )
        elif other_tz is None:
            raise TypeError(
                "Cannot compare tz-naive and tz-aware datetime-like objects"
            )

    # -----------------------------------------------------------------
    # Arithmetic Methods

    def _add_offset(self, offset: BaseOffset) -> Self:
        assert not isinstance(offset, Tick)

        if self.tz is not None:
            values = self.tz_localize(None)
        else:
            values = self

        try:
            res_values = offset._apply_array(values._ndarray)
            if res_values.dtype.kind == "i":
                # error: Argument 1 to "view" of "ndarray" has incompatible type
                # "dtype[datetime64] | DatetimeTZDtype"; expected
                # "dtype[Any] | type[Any] | _SupportsDType[dtype[Any]]"
                res_values = res_values.view(values.dtype)  # type: ignore[arg-type]
        except NotImplementedError:
            if get_option("performance_warnings"):
                warnings.warn(
                    "Non-vectorized DateOffset being applied to Series or "
                    "DatetimeIndex.",
                    PerformanceWarning,
                    stacklevel=find_stack_level(),
                )
            res_values = self.astype("O") + offset
            result = type(self)._from_sequence(res_values, dtype=self.dtype)

        else:
            result = type(self)._simple_new(res_values, dtype=res_values.dtype)
            if offset.normalize:
                result = result.normalize()
                result._freq = None

            if self.tz is not None:
                result = result.tz_localize(self.tz)

        return result

    # -----------------------------------------------------------------
    # Timezone Conversion and Localization Methods

    def _local_timestamps(self) -> NDArray[np.int64]:
        """
        Convert to an i8 (unix-like nanosecond timestamp) representation
        while keeping the local timezone and not using UTC.
        This is used to calculate time-of-day information as if the timestamps
        were timezone-naive.
        """
        if self.tz is None or timezones.is_utc(self.tz):
            # Avoid the copy that would be made in tzconversion
            return self.asi8
        return tz_convert_from_utc(self.asi8, self.tz, reso=self._creso)

    def tz_convert(self, tz: Union[str, tzinfo]) -> Self:
        """
        Convert tz-aware Datetime Array/Index from one time zone to another.

        Parameters
        ----------
        tz : str, zoneinfo.ZoneInfo, pytz.timezone, dateutil.tz.tzfile, datetime.tzinfo or None
            Time zone for time. Corresponding timestamps would be converted
            to this time zone of the Datetime Array/Index. A `tz` of None will
            convert to UTC and remove the timezone information.

        Returns
        -------
        Array or Index
            Datetme Array/Index with target `tz`.

        Raises
        ------
        TypeError
            If Datetime Array/Index is tz-naive.

        See Also
        --------
        DatetimeIndex.tz : A timezone that has a variable offset from UTC.
        DatetimeIndex.tz_localize : Localize tz-naive DatetimeIndex to a
            given time zone, or remove timezone from a tz-aware DatetimeIndex.

        Examples
        --------
        With the `tz` parameter, we can change the DatetimeIndex
        to other time zones:

        >>> dti = pd.date_range(
        ...     start="2014-08-01 09:00", freq="h", periods=3, tz="Europe/Berlin"
        ... )

        >>> dti
        DatetimeIndex(['2014-08-01 09:00:00+02:00',
                       '2014-08-01 10:00:00+02:00',
                       '2014-08-01 11:00:00+02:00'],
                      dtype='datetime64[ns, Europe/Berlin]', freq='h')

        >>> dti.tz_convert("US/Central")
        DatetimeIndex(['2014-08-01 02:00:00-05:00',
                       '2014-08-01 03:00:00-05:00',
                       '2014-08-01 04:00:00-05:00'],
                      dtype='datetime64[ns, US/Central]', freq='h')

        With the ``tz=None``, we can remove the timezone (after converting
        to UTC if necessary):

        >>> dti = pd.date_range(
        ...     start="2014-08-01 09:00", freq="h", periods=3, tz="Europe/Berlin"
        ... )

        >>> dti
        DatetimeIndex(['2014-08-01 09:00:00+02:00',
                       '2014-08-01 10:00:00+02:00',
                       '2014-08-01 11:00:00+02:00'],
                        dtype='datetime64[ns, Europe/Berlin]', freq='h')

        >>> dti.tz_convert(None)
        DatetimeIndex(['2014-08-01 07:00:00',
                       '2014-08-01 08:00:00',
                       '2014-08-01 09:00:00'],
                        dtype='datetime64[ns]', freq='h')
        """  # noqa: E501
        tz = timezones.maybe_get
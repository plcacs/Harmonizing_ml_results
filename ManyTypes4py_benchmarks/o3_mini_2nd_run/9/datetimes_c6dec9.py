from __future__ import annotations
from collections import abc
from datetime import date
from functools import partial
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, overload
import warnings
import numpy as np
from pandas._libs import lib, tslib
from pandas._libs.tslibs import OutOfBoundsDatetime, Timedelta, Timestamp, astype_overflowsafe, is_supported_dtype, timezones as libtimezones
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.parsing import DateParseError, guess_datetime_format
from pandas._libs.tslibs.strptime import array_strptime
from pandas._typing import AnyArrayLike, ArrayLike, DateTimeErrorChoices
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import ensure_object, is_float, is_float_dtype, is_integer, is_integer_dtype, is_list_like, is_numeric_dtype
from pandas.core.dtypes.dtypes import ArrowDtype, DatetimeTZDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.arrays import DatetimeArray, IntegerArray, NumpyExtensionArray
from pandas.core.algorithms import unique
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.datetimes import maybe_convert_dtype, objects_to_datetime64, tz_to_dtype
from pandas.core.construction import extract_array
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex

if TYPE_CHECKING:
    from collections.abc import Hashable
    from pandas._libs.tslibs.nattype import NaTType
    from pandas._libs.tslibs.timedeltas import UnitChoices
    from pandas import DataFrame, Series

ArrayConvertible = Union[list, tuple, AnyArrayLike]
Scalar = Union[float, str]
DatetimeScalar = Union[Scalar, date, np.datetime64]
DatetimeScalarOrArrayConvertible = Union[DatetimeScalar, ArrayConvertible]
DatetimeDictArg = Union[list[Scalar], tuple[Scalar, ...], AnyArrayLike]

class YearMonthDayDict(dict, total=True):  # type: ignore
    pass

class FulldatetimeDict(YearMonthDayDict, total=False):  # type: ignore
    pass

DictConvertible = Union[FulldatetimeDict, "DataFrame"]

start_caching_at: int = 50


def _guess_datetime_format_for_array(arr: Any, dayfirst: bool = False) -> Optional[str]:
    if (first_non_null := tslib.first_non_null(arr)) != -1:
        first_non_nan_element = arr[first_non_null]
        if isinstance(first_non_nan_element, str):
            guessed_format: Optional[str] = guess_datetime_format(first_non_nan_element, dayfirst=dayfirst)
            if guessed_format is not None:
                return guessed_format
            if tslib.first_non_null(arr[first_non_null + 1:]) != -1:
                warnings.warn(
                    "Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
    return None


def should_cache(arg: Any, unique_share: float = 0.7, check_count: Optional[int] = None) -> bool:
    """
    Decides whether to do caching.
    """
    do_caching: bool = True
    if check_count is None:
        if len(arg) <= start_caching_at:
            return False
        if len(arg) <= 5000:
            check_count = len(arg) // 10
        else:
            check_count = 500
    else:
        assert 0 <= check_count <= len(arg), "check_count must be in next bounds: [0; len(arg)]"
        if check_count == 0:
            return False
    assert 0 < unique_share < 1, "unique_share must be in next bounds: (0; 1)"
    try:
        unique_elements = set(islice(arg, check_count))
    except TypeError:
        return False
    if len(unique_elements) > check_count * unique_share:
        do_caching = False
    return do_caching


def _maybe_cache(
    arg: Any,
    format: Optional[str],
    cache: bool,
    convert_listlike: Callable[[Any, Optional[str]], Any],
) -> "Series":
    """
    Create a cache of unique dates from an array of dates
    """
    from pandas import Series

    cache_array = Series(dtype=object)
    if cache:
        if not should_cache(arg):
            return cache_array
        if not isinstance(arg, (np.ndarray, ExtensionArray, Index, ABCSeries)):
            arg = np.array(arg)
        unique_dates = unique(arg)
        if len(unique_dates) < len(arg):
            cache_dates = convert_listlike(unique_dates, format)
            try:
                cache_array = Series(cache_dates, index=unique_dates, copy=False)
            except OutOfBoundsDatetime:
                return cache_array
            if not cache_array.index.is_unique:
                cache_array = cache_array[~cache_array.index.duplicated()]
    return cache_array


def _box_as_indexlike(dt_array: Any, utc: bool = False, name: Optional[str] = None) -> Index:
    """
    Properly boxes the ndarray of datetimes to DatetimeIndex
    if it is possible or to generic Index instead
    """
    if lib.is_np_dtype(dt_array.dtype, "M"):
        tz = "utc" if utc else None
        return DatetimeIndex(dt_array, tz=tz, name=name)
    return Index(dt_array, name=name, dtype=dt_array.dtype)


def _convert_and_box_cache(arg: Any, cache_array: "Series", name: Optional[str] = None) -> Index:
    """
    Convert array of dates with a cache and wrap the result in an Index.
    """
    from pandas import Series

    result = Series(arg, dtype=cache_array.index.dtype).map(cache_array)
    return _box_as_indexlike(result._values, utc=False, name=name)


def _convert_listlike_datetimes(
    arg: Any,
    format: Optional[str],
    name: Optional[str] = None,
    utc: bool = False,
    unit: Optional[str] = None,
    errors: str = "raise",
    dayfirst: Optional[bool] = None,
    yearfirst: Optional[bool] = None,
    exact: Any = lib.no_default,
) -> Union[DatetimeIndex, Index]:
    """
    Helper function for to_datetime. Performs the conversions of 1D listlike of dates
    """
    if isinstance(arg, (list, tuple)):
        arg = np.array(arg, dtype="O")
    elif isinstance(arg, NumpyExtensionArray):
        arg = np.array(arg)
    arg_dtype = getattr(arg, "dtype", None)
    tz = "utc" if utc else None
    if isinstance(arg_dtype, DatetimeTZDtype):
        if not isinstance(arg, (DatetimeArray, DatetimeIndex)):
            return DatetimeIndex(arg, tz=tz, name=name)
        if utc:
            arg = arg.tz_convert(None).tz_localize("utc")
        return arg
    elif isinstance(arg_dtype, ArrowDtype) and arg_dtype.type is Timestamp:
        if utc:
            if isinstance(arg, Index):
                arg_array = cast(ArrowExtensionArray, arg.array)
                if arg_dtype.pyarrow_dtype.tz is not None:
                    arg_array = arg_array._dt_tz_convert("UTC")
                else:
                    arg_array = arg_array._dt_tz_localize("UTC")
                arg = Index(arg_array)
            elif arg_dtype.pyarrow_dtype.tz is not None:
                arg = arg._dt_tz_convert("UTC")
            else:
                arg = arg._dt_tz_localize("UTC")
        return arg
    elif lib.is_np_dtype(arg_dtype, "M"):
        if not is_supported_dtype(arg_dtype):
            arg = astype_overflowsafe(np.asarray(arg), np.dtype("M8[s]"), is_coerce=errors == "coerce")
        if not isinstance(arg, (DatetimeArray, DatetimeIndex)):
            return DatetimeIndex(arg, tz=tz, name=name)
        elif utc:
            return arg.tz_localize("utc")
        return arg
    elif unit is not None:
        if format is not None:
            raise ValueError("cannot specify both format and unit")
        return _to_datetime_with_unit(arg, unit, name, utc, errors)
    elif getattr(arg, "ndim", 1) > 1:
        raise TypeError("arg must be a string, datetime, list, tuple, 1-d array, or Series")
    try:
        arg, _ = maybe_convert_dtype(arg, copy=False, tz=libtimezones.maybe_get_tz(tz))
    except TypeError:
        if errors == "coerce":
            npvalues = np.full(len(arg), np.datetime64("NaT", "ns"))
            return DatetimeIndex(npvalues, name=name)
        raise
    arg = ensure_object(arg)
    if format is None:
        format = _guess_datetime_format_for_array(arg, dayfirst=dayfirst)  # type: ignore
    if format is not None and format != "mixed":
        return _array_strptime_with_fallback(arg, name, utc, format, exact, errors)
    result, tz_parsed = objects_to_datetime64(arg, dayfirst=dayfirst, yearfirst=yearfirst, utc=utc, errors=errors, allow_object=True)
    if tz_parsed is not None:
        out_unit = np.datetime_data(result.dtype)[0]
        dtype = tz_to_dtype(tz_parsed, out_unit)
        dt64_values = result.view(f"M8[{dtype.unit}]")
        dta = DatetimeArray._simple_new(dt64_values, dtype=dtype)
        return DatetimeIndex._simple_new(dta, name=name)
    return _box_as_indexlike(result, utc=utc, name=name)


def _array_strptime_with_fallback(
    arg: Any, name: Optional[str], utc: bool, fmt: str, exact: Any, errors: str
) -> Index:
    """
    Call array_strptime, with fallback behavior depending on 'errors'.
    """
    result, tz_out = array_strptime(arg, fmt, exact=exact, errors=errors, utc=utc)
    if tz_out is not None:
        unit = np.datetime_data(result.dtype)[0]
        dtype = DatetimeTZDtype(tz=tz_out, unit=unit)
        dta = DatetimeArray._simple_new(result, dtype=dtype)
        if utc:
            dta = dta.tz_convert("UTC")
        return Index(dta, name=name)
    elif result.dtype != object and utc:
        unit = np.datetime_data(result.dtype)[0]
        res = Index(result, dtype=f"M8[{unit}, UTC]", name=name)
        return res
    return Index(result, dtype=result.dtype, name=name)


def _to_datetime_with_unit(
    arg: Any, unit: str, name: Optional[str], utc: bool, errors: str
) -> DatetimeIndex:
    """
    to_datetime specialized to the case where a 'unit' is passed.
    """
    arg = extract_array(arg, extract_numpy=True)
    if isinstance(arg, IntegerArray):
        arr = arg.astype(f"datetime64[{unit}]")
        tz_parsed = None
    else:
        arg = np.asarray(arg)
        if arg.dtype.kind in "iu":
            arr = arg.astype(f"datetime64[{unit}]", copy=False)
            try:
                arr = astype_overflowsafe(arr, np.dtype("M8[ns]"), copy=False)
            except OutOfBoundsDatetime:
                if errors == "raise":
                    raise
                arg = arg.astype(object)
                return _to_datetime_with_unit(arg, unit, name, utc, errors)
            tz_parsed = None
        elif arg.dtype.kind == "f":
            with np.errstate(over="raise"):
                try:
                    arr = cast_from_unit_vectorized(arg, unit=unit)
                except OutOfBoundsDatetime as err:
                    if errors != "raise":
                        return _to_datetime_with_unit(arg.astype(object), unit, name, utc, errors)
                    raise OutOfBoundsDatetime(f"cannot convert input with unit '{unit}'") from err
            arr = arr.view("M8[ns]")
            tz_parsed = None
        else:
            arg = arg.astype(object, copy=False)
            arr, tz_parsed = tslib.array_to_datetime(
                arg, utc=utc, errors=errors, unit_for_numerics=unit, creso=NpyDatetimeUnit.NPY_FR_ns.value
            )
    result = DatetimeIndex(arr, name=name)
    if not isinstance(result, DatetimeIndex):
        return result
    result = result.tz_localize("UTC").tz_convert(tz_parsed)
    if utc:
        if result.tz is None:
            result = result.tz_localize("utc")
        else:
            result = result.tz_convert("utc")
    return result


def _adjust_to_origin(arg: Any, origin: Union[str, Timestamp], unit: str) -> Any:
    """
    Helper function for to_datetime.
    Adjust input argument to the specified origin
    """
    if origin == "julian":
        original = arg
        j0 = Timestamp(0).to_julian_date()
        if unit != "D":
            raise ValueError("unit must be 'D' for origin='julian'")
        try:
            arg = arg - j0
        except TypeError as err:
            raise ValueError("incompatible 'arg' type for given 'origin'='julian'") from err
        j_max = Timestamp.max.to_julian_date() - j0
        j_min = Timestamp.min.to_julian_date() - j0
        if np.any(arg > j_max) or np.any(arg < j_min):
            raise OutOfBoundsDatetime(f"{original} is Out of Bounds for origin='julian'")
    else:
        if not ((is_integer(arg) or is_float(arg)) or is_numeric_dtype(np.asarray(arg))):
            raise ValueError(f"'{arg}' is not compatible with origin='{origin}'; it must be numeric with a unit specified")
        try:
            offset = Timestamp(origin, unit=unit)
        except OutOfBoundsDatetime as err:
            raise OutOfBoundsDatetime(f"origin {origin} is Out of Bounds") from err
        except ValueError as err:
            raise ValueError(f"origin {origin} cannot be converted to a Timestamp") from err
        if offset.tz is not None:
            raise ValueError(f"origin offset {offset} must be tz-naive")
        td_offset = offset - Timestamp(0)
        ioffset = td_offset // Timedelta(1, unit=unit)
        if is_list_like(arg) and (not isinstance(arg, (ABCSeries, Index, np.ndarray))):
            arg = np.asarray(arg)
        arg = arg + ioffset
    return arg


@overload
def to_datetime(
    arg: Any,
    errors: str = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool = ...,
    format: Optional[str] = ...,
    exact: Any = ...,
    unit: Optional[str] = ...,
    origin: Union[str, Timestamp] = ...,
    cache: bool = ...,
) -> Any:
    ...


@overload
def to_datetime(
    arg: Any,
    errors: str = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool = ...,
    format: Optional[str] = ...,
    exact: Any = ...,
    unit: Optional[str] = ...,
    origin: Union[str, Timestamp] = ...,
    cache: bool = ...,
) -> Any:
    ...


@overload
def to_datetime(
    arg: Any,
    errors: str = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool = ...,
    format: Optional[str] = ...,
    exact: Any = ...,
    unit: Optional[str] = ...,
    origin: Union[str, Timestamp] = ...,
    cache: bool = ...,
) -> Any:
    ...


def to_datetime(
    arg: Any,
    errors: str = "raise",
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: Optional[str] = None,
    exact: Any = lib.no_default,
    unit: Optional[str] = None,
    origin: Union[str, Timestamp] = "unix",
    cache: bool = True,
) -> Any:
    """
    Convert argument to datetime.
    """
    if exact is not lib.no_default and format in {"mixed", "ISO8601"}:
        raise ValueError("Cannot use 'exact' when 'format' is 'mixed' or 'ISO8601'")
    if arg is None:
        return None
    if origin != "unix":
        arg = _adjust_to_origin(arg, origin, unit)  # type: ignore
    convert_listlike = partial(
        _convert_listlike_datetimes,
        utc=utc,
        unit=unit,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
        errors=errors,
        exact=exact,
    )
    if isinstance(arg, Timestamp):
        result = arg
        if utc:
            if arg.tz is not None:
                result = arg.tz_convert("utc")
            else:
                result = arg.tz_localize("utc")
    elif isinstance(arg, ABCSeries):
        cache_array = _maybe_cache(arg, format, cache, convert_listlike)
        if not cache_array.empty:
            result = arg.map(cache_array)
        else:
            values = convert_listlike(arg._values, format)
            from pandas import Series

            result = arg._constructor(values, index=arg.index, name=arg.name)
    elif isinstance(arg, (ABCDataFrame, abc.MutableMapping)):
        result = _assemble_from_unit_mappings(arg, errors, utc)
    elif isinstance(arg, Index):
        cache_array = _maybe_cache(arg, format, cache, convert_listlike)
        if not cache_array.empty:
            result = _convert_and_box_cache(arg, cache_array, name=arg.name)
        else:
            result = convert_listlike(arg, format, name=arg.name)
    elif is_list_like(arg):
        try:
            argc = cast(Union[list, tuple, ExtensionArray, np.ndarray, "Series", Index], arg)
            cache_array = _maybe_cache(argc, format, cache, convert_listlike)
        except OutOfBoundsDatetime:
            if errors == "raise":
                raise
            from pandas import Series

            cache_array = Series([], dtype=object)
        if not cache_array.empty:
            result = _convert_and_box_cache(argc, cache_array)
        else:
            result = convert_listlike(argc, format)
    else:
        result = convert_listlike(np.array([arg]), format)[0]
        if isinstance(arg, bool) and isinstance(result, np.bool_):
            result = bool(result)
    return result


_unit_map: dict[str, str] = {
    "year": "year",
    "years": "year",
    "month": "month",
    "months": "month",
    "day": "day",
    "days": "day",
    "hour": "h",
    "hours": "h",
    "minute": "m",
    "minutes": "m",
    "second": "s",
    "seconds": "s",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "us": "us",
    "microsecond": "us",
    "microseconds": "us",
    "ns": "ns",
    "nanosecond": "ns",
    "nanoseconds": "ns",
}


def _assemble_from_unit_mappings(arg: Any, errors: str, utc: bool) -> "Series":
    """
    Assemble the unit specified fields from the arg (DataFrame)
    Return a Series for actual parsing
    """
    from pandas import DataFrame, to_numeric, to_timedelta, Series

    arg = DataFrame(arg)
    if not arg.columns.is_unique:
        raise ValueError("cannot assemble with duplicate keys")

    def f(value: Any) -> Any:
        if value in _unit_map:
            return _unit_map[value]
        if isinstance(value, str) and value.lower() in _unit_map:
            return _unit_map[value.lower()]
        return value

    unit = {k: f(k) for k in arg.keys()}
    unit_rev = {v: k for k, v in unit.items()}
    required = ["year", "month", "day"]
    req = set(required) - set(unit_rev.keys())
    if len(req):
        _required = ",".join(sorted(req))
        raise ValueError(
            f"to assemble mappings requires at least that [year, month, day] be specified: [{_required}] is missing"
        )
    excess = set(unit_rev.keys()) - set(_unit_map.values())
    if len(excess):
        _excess = ",".join(sorted(excess))
        raise ValueError(f"extra keys have been passed to the datetime assemblage: [{_excess}]")

    def coerce(values: Any) -> Any:
        values = to_numeric(values, errors=errors)
        if is_float_dtype(values.dtype):
            values = values.astype("float64")
        if is_integer_dtype(values.dtype):
            values = values.astype("int64")
        return values

    values = coerce(arg[unit_rev["year"]]) * 10000 + coerce(arg[unit_rev["month"]]) * 100 + coerce(arg[unit_rev["day"]])
    try:
        values = to_datetime(values, format="%Y%m%d", errors=errors, utc=utc)
    except (TypeError, ValueError) as err:
        raise ValueError(f"cannot assemble the datetimes: {err}") from err
    units = ["h", "m", "s", "ms", "us", "ns"]
    for u in units:
        value = unit_rev.get(u)
        if value is not None and value in arg:
            try:
                values += to_timedelta(coerce(arg[value]), unit=u, errors=errors)
            except (TypeError, ValueError) as err:
                raise ValueError(f"cannot assemble the datetimes [{value}]: {err}") from err
    return values


__all__ = ["DateParseError", "should_cache", "to_datetime"]
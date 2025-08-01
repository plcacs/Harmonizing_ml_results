#!/usr/bin/env python3
"""
Routines for casting.
"""
from __future__ import annotations
import datetime as dt
import functools
import warnings
from typing import Any, Optional, Union, List, Tuple, overload
import numpy as np

from pandas._config import using_string_dtype
from pandas._libs import Interval, Period, lib
from pandas._libs.missing import NA, NAType, checknull
from pandas._libs.tslibs import NaT, OutOfBoundsDatetime, OutOfBoundsTimedelta, Timedelta, Timestamp, is_supported_dtype
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.errors import IntCastingNaNError, LossySetitemError
from pandas.core.dtypes.common import (
    ensure_int8,
    ensure_int16,
    ensure_int32,
    ensure_int64,
    ensure_object,
    ensure_str,
    is_bool,
    is_complex,
    is_float,
    is_integer,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype as pandas_dtype_func,
)
from pandas.core.dtypes.dtypes import ArrowDtype, BaseMaskedDtype, CategoricalDtype, DatetimeTZDtype, ExtensionDtype, IntervalDtype, PandasExtensionDtype, PeriodDtype
from pandas.core.dtypes.generic import ABCExtensionArray, ABCIndex, ABCSeries
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import is_valid_na_for_dtype, isna, na_value_for_dtype, notna
from pandas.io._util import _arrow_dtype_mapping

_int8_max = np.iinfo(np.int8).max
_int16_max = np.iinfo(np.int16).max
_int32_max = np.iinfo(np.int32).max
_dtype_obj = np.dtype(object)

def maybe_convert_platform(values: Union[list[Any], tuple[Any, ...], range, np.ndarray]) -> np.ndarray:
    """try to do platform conversion, allow ndarray or list here"""
    if isinstance(values, (list, tuple, range)):
        arr = construct_1d_object_array_from_listlike(values)
    else:
        arr = values
    if arr.dtype == _dtype_obj:
        arr = cast(np.ndarray, arr)
        arr = lib.maybe_convert_objects(arr)
    return arr

def is_nested_object(obj: Any) -> bool:
    """
    return a boolean if we have a nested object, e.g. a Series with 1 or
    more Series elements
    """
    return bool(
        isinstance(obj, ABCSeries)
        and is_object_dtype(obj.dtype)
        and any((isinstance(v, ABCSeries) for v in obj._values))
    )

def maybe_box_datetimelike(value: Any, dtype: Optional[Any] = None) -> Any:
    """
    Cast scalar to Timestamp or Timedelta if scalar is datetime-like
    and dtype is not object.
    """
    if dtype == _dtype_obj:
        pass
    elif isinstance(value, (np.datetime64, dt.datetime)):
        value = Timestamp(value)
    elif isinstance(value, (np.timedelta64, dt.timedelta)):
        value = Timedelta(value)
    return value

def maybe_box_native(value: Any) -> Any:
    """
    If passed a scalar cast the scalar to a python native type.
    """
    if is_float(value):
        value = float(value)
    elif is_integer(value):
        value = int(value)
    elif is_bool(value):
        value = bool(value)
    elif isinstance(value, (np.datetime64, np.timedelta64)):
        value = maybe_box_datetimelike(value)
    elif value is NA:
        value = None
    return value

def _maybe_unbox_datetimelike(value: Any, dtype: Any) -> Any:
    """
    Convert a Timedelta or Timestamp to timedelta64 or datetime64 for setting
    into a numpy array.
    """
    if is_valid_na_for_dtype(value, dtype):
        value = dtype.type('NaT', 'ns')
    elif isinstance(value, Timestamp):
        if value.tz is None:
            value = value.to_datetime64()
        elif not isinstance(dtype, DatetimeTZDtype):
            raise TypeError('Cannot unbox tzaware Timestamp to tznaive dtype')
    elif isinstance(value, Timedelta):
        value = value.to_timedelta64()
    _disallow_mismatched_datetimelike(value, dtype)
    return value

def _disallow_mismatched_datetimelike(value: Any, dtype: Any) -> None:
    """
    numpy allows np.array(dt64values, dtype="timedelta64[ns]") and
    vice-versa, but we do not want to allow this, so we need to
    check explicitly
    """
    vdtype = getattr(value, 'dtype', None)
    if vdtype is None:
        return
    elif vdtype.kind == 'm' and dtype.kind == 'M' or (vdtype.kind == 'M' and dtype.kind == 'm'):
        raise TypeError(f'Cannot cast {value!r} to {dtype}')

@overload
def maybe_downcast_to_dtype(result: Any, dtype: Any) -> Any:
    ...

@overload
def maybe_downcast_to_dtype(result: Any, dtype: Any) -> Any:
    ...

def maybe_downcast_to_dtype(result: Any, dtype: Any) -> Any:
    """
    try to cast to the specified dtype
    """
    if isinstance(result, ABCSeries):
        result = result._values
    do_round = False
    if isinstance(dtype, str):
        if dtype == 'infer':
            inferred_type = lib.infer_dtype(result, skipna=False)
            if inferred_type == 'boolean':
                dtype = 'bool'
            elif inferred_type == 'integer':
                dtype = 'int64'
            elif inferred_type == 'datetime64':
                dtype = 'datetime64[ns]'
            elif inferred_type in ['timedelta', 'timedelta64']:
                dtype = 'timedelta64[ns]'
            elif inferred_type == 'floating':
                dtype = 'int64'
                if issubclass(result.dtype.type, np.number):
                    do_round = True
            else:
                dtype = 'object'
        dtype = np.dtype(dtype)
    if not isinstance(dtype, np.dtype):
        raise TypeError(dtype)
    converted = maybe_downcast_numeric(result, dtype, do_round)
    if converted is not result:
        return converted
    if dtype.kind in 'mM' and result.dtype.kind in 'if':
        result = result.astype(dtype)
    elif dtype.kind == 'm' and result.dtype == _dtype_obj:
        result = cast(np.ndarray, result)
        result = array_to_timedelta64(result)
    elif dtype == np.dtype('M8[ns]') and result.dtype == _dtype_obj:
        result = cast(np.ndarray, result)
        return np.asarray(maybe_cast_to_datetime(result, dtype=dtype))
    return result

@overload
def maybe_downcast_numeric(result: Any, dtype: Any, do_round: bool = False) -> Any:
    ...

@overload
def maybe_downcast_numeric(result: Any, dtype: Any, do_round: bool = False) -> Any:
    ...

def maybe_downcast_numeric(result: Any, dtype: Any, do_round: bool = False) -> Any:
    """
    Subset of maybe_downcast_to_dtype restricted to numeric dtypes.
    """
    if not isinstance(dtype, np.dtype) or not isinstance(result.dtype, np.dtype):
        return result

    def trans(x: Any) -> Any:
        if do_round:
            return x.round()
        return x

    if dtype.kind == result.dtype.kind:
        if result.dtype.itemsize <= dtype.itemsize and result.size:
            return result
    if dtype.kind in 'biu':
        if not result.size:
            return trans(result).astype(dtype)
        if isinstance(result, np.ndarray):
            element = result.item(0)
        else:
            element = result.iloc[0]
        if not isinstance(element, (np.integer, np.floating, int, float, bool)):
            return result
        if issubclass(result.dtype.type, (np.object_, np.number)) and notna(result).all():
            new_result = trans(result).astype(dtype)
            if new_result.dtype.kind == 'O' or result.dtype.kind == 'O':
                if (new_result == result).all():
                    return new_result
            elif np.allclose(new_result, result, rtol=0):
                return new_result
    elif issubclass(dtype.type, np.floating) and result.dtype.kind != 'b' and (not is_string_dtype(result.dtype)):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'overflow encountered in cast', RuntimeWarning)
            new_result = result.astype(dtype)
        size_tols = {4: 0.0005, 8: 5e-08, 16: 5e-16}
        atol = size_tols.get(new_result.dtype.itemsize, 0.0)
        if np.allclose(new_result, result, equal_nan=True, rtol=0.0, atol=atol):
            return new_result
    elif dtype.kind == result.dtype.kind == 'c':
        new_result = result.astype(dtype)
        if np.array_equal(new_result, result, equal_nan=True):
            return new_result
    return result

def maybe_upcast_numeric_to_64bit(arr: Any) -> Any:
    """
    If array is a int/uint/float bit size lower than 64 bit, upcast it to 64 bit.
    """
    dtype = arr.dtype
    if dtype.kind == 'i' and dtype != np.int64:
        return arr.astype(np.int64)
    elif dtype.kind == 'u' and dtype != np.uint64:
        return arr.astype(np.uint64)
    elif dtype.kind == 'f' and dtype != np.float64:
        return arr.astype(np.float64)
    else:
        return arr

def maybe_cast_pointwise_result(result: Any, dtype: Any, numeric_only: bool = False, same_dtype: bool = True) -> Any:
    """
    Try casting result of a pointwise operation back to the original dtype if
    appropriate.
    """
    if isinstance(dtype, ExtensionDtype):
        cls = dtype.construct_array_type()
        if same_dtype:
            result = _maybe_cast_to_extension_array(cls, result, dtype=dtype)
        else:
            result = _maybe_cast_to_extension_array(cls, result)
    elif (numeric_only and dtype.kind in 'iufcb') or (not numeric_only):
        result = maybe_downcast_to_dtype(result, dtype)
    return result

def _maybe_cast_to_extension_array(cls: Any, obj: Any, dtype: Optional[ExtensionDtype] = None) -> Any:
    """
    Call to `_from_sequence` that returns the object unchanged on Exception.
    """
    if dtype is not None:
        try:
            result = cls._from_scalars(obj, dtype=dtype)
        except (TypeError, ValueError):
            return obj
        return result
    try:
        result = cls._from_sequence(obj, dtype=dtype)
    except Exception:
        result = obj
    return result

@overload
def ensure_dtype_can_hold_na(dtype: Any) -> Any:
    ...

@overload
def ensure_dtype_can_hold_na(dtype: Any) -> Any:
    ...

def ensure_dtype_can_hold_na(dtype: Any) -> Any:
    """
    If we have a dtype that cannot hold NA values, find the best match that can.
    """
    if isinstance(dtype, ExtensionDtype):
        if dtype._can_hold_na:
            return dtype
        elif isinstance(dtype, IntervalDtype):
            return IntervalDtype(np.float64, closed=dtype.closed)
        return _dtype_obj
    elif dtype.kind == 'b':
        return _dtype_obj
    elif dtype.kind in 'iu':
        return np.dtype(np.float64)
    return dtype

_canonical_nans = {
    np.datetime64: np.datetime64('NaT', 'ns'),
    np.timedelta64: np.timedelta64('NaT', 'ns'),
    type(np.nan): np.nan,
}

def maybe_promote(dtype: Any, fill_value: Any = np.nan) -> Tuple[Any, Any]:
    """
    Find the minimal dtype that can hold both the given dtype and fill_value.
    """
    orig = fill_value
    orig_is_nat = False
    if checknull(fill_value):
        if fill_value is not NA:
            try:
                orig_is_nat = np.isnat(fill_value)
            except TypeError:
                pass
        fill_value = _canonical_nans.get(type(fill_value), fill_value)
    try:
        dtype, fill_value = _maybe_promote_cached(dtype, fill_value, type(fill_value))
    except TypeError:
        dtype, fill_value = _maybe_promote(dtype, fill_value)
    if dtype == _dtype_obj and orig is not None or (orig_is_nat and np.datetime_data(orig)[0] != 'ns'):
        fill_value = orig
    return (dtype, fill_value)

@functools.lru_cache
def _maybe_promote_cached(dtype: Any, fill_value: Any, fill_value_type: Any) -> Tuple[Any, Any]:
    return _maybe_promote(dtype, fill_value)

def _maybe_promote(dtype: Any, fill_value: Any = np.nan) -> Tuple[Any, Any]:
    if not is_scalar(fill_value):
        if dtype != object:
            raise ValueError('fill_value must be a scalar')
        dtype = _dtype_obj
        return (dtype, fill_value)
    if is_valid_na_for_dtype(fill_value, dtype) and dtype.kind in 'iufcmM':
        dtype = ensure_dtype_can_hold_na(dtype)
        fv = na_value_for_dtype(dtype)
        return (dtype, fv)
    elif isinstance(dtype, CategoricalDtype):
        if fill_value in dtype.categories or isna(fill_value):
            return (dtype, fill_value)
        else:
            return (object, ensure_object(fill_value))
    elif isna(fill_value):
        dtype = _dtype_obj
        if fill_value is None:
            fill_value = np.nan
        return (dtype, fill_value)
    if issubclass(dtype.type, np.datetime64):
        inferred, fv = infer_dtype_from_scalar(fill_value)
        if inferred == dtype:
            return (dtype, fv)
        from pandas.core.arrays import DatetimeArray
        dta = DatetimeArray._from_sequence([], dtype='M8[ns]')
        try:
            fv = dta._validate_setitem_value(fill_value)
            return (dta.dtype, fv)
        except (ValueError, TypeError):
            return (_dtype_obj, fill_value)
    elif issubclass(dtype.type, np.timedelta64):
        inferred, fv = infer_dtype_from_scalar(fill_value)
        if inferred == dtype:
            return (dtype, fv)
        elif inferred.kind == 'm':
            unit = np.datetime_data(dtype)[0]
            try:
                td = Timedelta(fill_value).as_unit(unit, round_ok=False)
            except OutOfBoundsTimedelta:
                return (_dtype_obj, fill_value)
            else:
                return (dtype, td.asm8)
        return (_dtype_obj, fill_value)
    elif is_float(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)
        elif issubclass(dtype.type, np.integer):
            dtype = np.dtype(np.float64)
        elif dtype.kind == 'f':
            mst = np.min_scalar_type(fill_value)
            if mst > dtype:
                dtype = mst
        elif dtype.kind == 'c':
            mst = np.min_scalar_type(fill_value)
            dtype = np.promote_types(dtype, mst)
    elif is_bool(fill_value):
        if not issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)
    elif is_integer(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)
        elif issubclass(dtype.type, np.integer):
            if not np_can_cast_scalar(fill_value, dtype):
                mst = np.min_scalar_type(fill_value)
                dtype = np.promote_types(dtype, mst)
                if dtype.kind == 'f':
                    dtype = np.dtype(np.object_)
    elif is_complex(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)
        elif issubclass(dtype.type, (np.integer, np.floating)):
            mst = np.min_scalar_type(fill_value)
            dtype = np.promote_types(dtype, mst)
        elif dtype.kind == 'c':
            mst = np.min_scalar_type(fill_value)
            if mst > dtype:
                dtype = mst
    else:
        dtype = np.dtype(np.object_)
    if issubclass(dtype.type, (bytes, str)):
        dtype = np.dtype(np.object_)
    fill_value = _ensure_dtype_type(fill_value, dtype)
    return (dtype, fill_value)

def _ensure_dtype_type(value: Any, dtype: Any) -> Any:
    """
    Ensure that the given value is an instance of the given dtype.
    """
    if dtype == _dtype_obj:
        return value
    return dtype.type(value)

def infer_dtype_from(val: Any) -> Tuple[Any, Any]:
    """
    Interpret the dtype from a scalar or array.
    """
    if not is_list_like(val):
        return infer_dtype_from_scalar(val)
    return infer_dtype_from_array(val)

def infer_dtype_from_scalar(val: Any) -> Tuple[Any, Any]:
    """
    Interpret the dtype from a scalar.
    """
    dtype = _dtype_obj
    if isinstance(val, np.ndarray):
        if val.ndim != 0:
            msg = 'invalid ndarray passed to infer_dtype_from_scalar'
            raise ValueError(msg)
        dtype = val.dtype
        val = lib.item_from_zerodim(val)
    elif isinstance(val, str):
        dtype = _dtype_obj
        if using_string_dtype():
            from pandas.core.arrays.string_ import StringDtype
            dtype = StringDtype(na_value=np.nan)
    elif isinstance(val, (np.datetime64, dt.datetime)):
        try:
            val = Timestamp(val)
        except OutOfBoundsDatetime:
            return (_dtype_obj, val)
        if val is NaT or val.tz is None:
            val = val.to_datetime64()
            dtype = val.dtype
        else:
            dtype = DatetimeTZDtype(unit=val.unit, tz=val.tz)
    elif isinstance(val, (np.timedelta64, dt.timedelta)):
        try:
            val = Timedelta(val)
        except (OutOfBoundsTimedelta, OverflowError):
            dtype = _dtype_obj
        else:
            if val is NaT:
                val = np.timedelta64('NaT', 'ns')
            else:
                val = val.asm8
            dtype = val.dtype
    elif is_bool(val):
        dtype = np.dtype(np.bool_)
    elif is_integer(val):
        if isinstance(val, np.integer):
            dtype = np.dtype(type(val))
        else:
            dtype = np.dtype(np.int64)
        try:
            np.array(val, dtype=dtype)
        except OverflowError:
            dtype = np.array(val).dtype
    elif is_float(val):
        if isinstance(val, np.floating):
            dtype = np.dtype(type(val))
        else:
            dtype = np.dtype(np.float64)
    elif is_complex(val):
        dtype = np.dtype(np.complex128)
    if isinstance(val, Period):
        dtype = PeriodDtype(freq=val.freq)
    elif isinstance(val, Interval):
        subtype, _ = infer_dtype_from_scalar(val.left)
        dtype = IntervalDtype(subtype=subtype, closed=val.closed)
    return (dtype, val)

def dict_compat(d: dict[Any, Any]) -> dict[Any, Any]:
    """
    Convert datetimelike-keyed dicts to a Timestamp-keyed dict.
    """
    return {maybe_box_datetimelike(key): value for key, value in d.items()}

def infer_dtype_from_array(arr: Any) -> Tuple[Any, Any]:
    """
    Infer the dtype from an array.
    """
    if isinstance(arr, np.ndarray):
        return (arr.dtype, arr)
    if not is_list_like(arr):
        raise TypeError("'arr' must be list-like")
    arr_dtype = getattr(arr, 'dtype', None)
    if isinstance(arr_dtype, ExtensionDtype):
        return (arr.dtype, arr)
    elif isinstance(arr, ABCSeries):
        return (arr.dtype, np.asarray(arr))
    inferred = lib.infer_dtype(arr, skipna=False)
    if inferred in ['string', 'bytes', 'mixed', 'mixed-integer']:
        return (np.dtype(np.object_), arr)
    arr = np.asarray(arr)
    return (arr.dtype, arr)

def _maybe_infer_dtype_type(element: Any) -> Optional[Any]:
    """
    Try to infer an object's dtype, for use in arithmetic ops.
    """
    tipo = None
    if hasattr(element, 'dtype'):
        tipo = element.dtype
    elif is_list_like(element):
        element = np.asarray(element)
        tipo = element.dtype
    return tipo

def invalidate_string_dtypes(dtype_set: set[Any]) -> None:
    """
    Change string like dtypes to object for
    DataFrame.select_dtypes().
    """
    non_string_dtypes = dtype_set - {np.dtype('S').type, np.dtype('<U').type}
    if non_string_dtypes != dtype_set:
        raise TypeError("string dtypes are not allowed, use 'object' instead")

def coerce_indexer_dtype(indexer: np.ndarray, categories: Any) -> np.ndarray:
    """coerce the indexer input array to the smallest dtype possible"""
    length = len(categories)
    if length < _int8_max:
        return ensure_int8(indexer)
    elif length < _int16_max:
        return ensure_int16(indexer)
    elif length < _int32_max:
        return ensure_int32(indexer)
    return ensure_int64(indexer)

def convert_dtypes(
    input_array: Any,
    convert_string: bool = True,
    convert_integer: bool = True,
    convert_boolean: bool = True,
    convert_floating: bool = True,
    infer_objects: bool = False,
    dtype_backend: str = 'numpy_nullable'
) -> Any:
    """
    Convert objects to best possible type, and optionally,
    to types supporting pd.NA.
    """
    from pandas.core.arrays.string_ import StringDtype
    if (convert_string or convert_integer or convert_boolean or convert_floating) and isinstance(input_array, np.ndarray):
        if input_array.dtype == object:
            inferred_dtype = lib.infer_dtype(input_array)
        else:
            inferred_dtype = input_array.dtype
        if is_string_dtype(inferred_dtype):
            if not convert_string or inferred_dtype == 'bytes':
                inferred_dtype = input_array.dtype
            else:
                inferred_dtype = pandas_dtype_func('string')
        if convert_integer:
            target_int_dtype = pandas_dtype_func('Int64')
            if input_array.dtype.kind in 'iu':
                from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE
                inferred_dtype = NUMPY_INT_TO_DTYPE.get(input_array.dtype, target_int_dtype)
            elif input_array.dtype.kind in 'fcb':
                arr = input_array[notna(input_array)]
                if (arr.astype(int) == arr).all():
                    inferred_dtype = target_int_dtype
                else:
                    inferred_dtype = input_array.dtype
            elif infer_objects and input_array.dtype == object and (isinstance(inferred_dtype, str) and inferred_dtype == 'integer'):
                inferred_dtype = target_int_dtype
        if convert_floating:
            if input_array.dtype.kind in 'fcb':
                from pandas.core.arrays.floating import NUMPY_FLOAT_TO_DTYPE
                inferred_float_dtype = NUMPY_FLOAT_TO_DTYPE.get(input_array.dtype, pandas_dtype_func('Float64'))
                if convert_integer:
                    arr = input_array[notna(input_array)]
                    if (arr.astype(int) == arr).all():
                        inferred_dtype = pandas_dtype_func('Int64')
                    else:
                        inferred_dtype = inferred_float_dtype
                else:
                    inferred_dtype = inferred_float_dtype
            elif infer_objects and input_array.dtype == object and (isinstance(inferred_dtype, str) and inferred_dtype == 'mixed-integer-float'):
                inferred_dtype = pandas_dtype_func('Float64')
        if convert_boolean:
            if input_array.dtype.kind == 'b':
                inferred_dtype = pandas_dtype_func('boolean')
            elif isinstance(inferred_dtype, str) and inferred_dtype == 'boolean':
                inferred_dtype = pandas_dtype_func('boolean')
        if isinstance(inferred_dtype, str):
            inferred_dtype = input_array.dtype
    elif convert_string and isinstance(input_array.dtype, StringDtype) and (input_array.dtype.na_value is np.nan):
        inferred_dtype = pandas_dtype_func('string')
    else:
        inferred_dtype = input_array.dtype
    if dtype_backend == 'pyarrow':
        from pandas.core.arrays.arrow.array import to_pyarrow_type
        from pandas.core.arrays.string_ import StringDtype
        assert not isinstance(inferred_dtype, str)
        if (convert_integer and inferred_dtype.kind in 'iu') or (convert_floating and inferred_dtype.kind in 'fc') or (convert_boolean and inferred_dtype.kind == 'b') or (convert_string and isinstance(inferred_dtype, StringDtype)) or (inferred_dtype.kind not in 'iufcb' and (not isinstance(inferred_dtype, StringDtype))):
            if isinstance(inferred_dtype, PandasExtensionDtype) and (not isinstance(inferred_dtype, DatetimeTZDtype)):
                base_dtype = inferred_dtype.base
            elif isinstance(inferred_dtype, (BaseMaskedDtype, ArrowDtype)):
                base_dtype = inferred_dtype.numpy_dtype
            elif isinstance(inferred_dtype, StringDtype):
                base_dtype = np.dtype(str)
            else:
                base_dtype = inferred_dtype
            if base_dtype.kind == 'O' and input_array.size > 0 and isna(input_array).all():
                import pyarrow as pa
                pa_type = pa.null()
            else:
                pa_type = to_pyarrow_type(base_dtype)
            if pa_type is not None:
                inferred_dtype = ArrowDtype(pa_type)
    elif dtype_backend == 'numpy_nullable' and isinstance(inferred_dtype, ArrowDtype):
        inferred_dtype = _arrow_dtype_mapping()[inferred_dtype.pyarrow_dtype]
    return inferred_dtype

def maybe_infer_to_datetimelike(value: np.ndarray, convert_to_nullable_dtype: bool = False) -> Any:
    """
    we might have a array (or single object) that is datetime like,
    and no dtype is passed don't change the value unless we find a
    datetime/timedelta set
    """
    if not isinstance(value, np.ndarray) or value.dtype != object:
        raise TypeError(type(value))
    if value.ndim != 1:
        raise ValueError(value.ndim)
    if not len(value):
        return value
    return lib.maybe_convert_objects(
        value,
        convert_numeric=False,
        convert_non_numeric=True,
        convert_to_nullable_dtype=convert_to_nullable_dtype,
        dtype_if_all_nat=np.dtype('M8[s]')
    )

def maybe_cast_to_datetime(value: Any, dtype: Any) -> Any:
    """
    try to cast the array/value to a datetimelike dtype, converting float
    nan to iNaT
    """
    from pandas.core.arrays.datetimes import DatetimeArray
    from pandas.core.arrays.timedeltas import TimedeltaArray
    assert dtype.kind in 'mM'
    if not is_list_like(value):
        raise TypeError('value must be listlike')
    _ensure_nanosecond_dtype(dtype)
    if lib.is_np_dtype(dtype, 'm'):
        res = TimedeltaArray._from_sequence(value, dtype=dtype)
        return res
    else:
        try:
            dta = DatetimeArray._from_sequence(value, dtype=dtype)
        except ValueError as err:
            if 'cannot supply both a tz and a timezone-naive dtype' in str(err):
                raise ValueError('Cannot convert timezone-aware data to timezone-naive dtype. Use pd.Series(values).dt.tz_localize(None) instead.') from err
            raise
        return dta

def _ensure_nanosecond_dtype(dtype: Any) -> None:
    """
    Convert dtypes with granularity less than nanosecond to nanosecond.
    """
    msg = f"The '{dtype.name}' dtype has no unit. Please pass in '{dtype.name}[ns]' instead."
    dtype = getattr(dtype, 'subtype', dtype)
    if not isinstance(dtype, np.dtype):
        pass
    elif dtype.kind in 'mM':
        if not is_supported_dtype(dtype):
            if dtype.name in ['datetime64', 'timedelta64']:
                raise ValueError(msg)
            raise TypeError(f"dtype={dtype} is not supported. Supported resolutions are 's', 'ms', 'us', and 'ns'")

def find_result_type(left_dtype: Any, right: Any) -> Any:
    """
    Find the type/dtype for the result of an operation between objects.
    """
    if isinstance(left_dtype, np.dtype) and left_dtype.kind in 'iuc' and (lib.is_integer(right) or lib.is_float(right)):
        if lib.is_float(right) and right.is_integer() and (left_dtype.kind != 'f'):
            right = int(right)
        if isinstance(right, int) and (not isinstance(right, np.integer)):
            right_dtype = np.min_scalar_type(right)
            if right == 0:
                right = left_dtype
            elif not np.issubdtype(left_dtype, np.unsignedinteger) and 0 < right <= np.iinfo(right_dtype).max:
                right = np.dtype(f'i{right_dtype.itemsize}')
            else:
                right = right_dtype
        new_dtype = np.result_type(left_dtype, right)
    elif is_valid_na_for_dtype(right, left_dtype):
        new_dtype = ensure_dtype_can_hold_na(left_dtype)
    else:
        dtype, _ = infer_dtype_from(right)
        new_dtype = find_common_type([left_dtype, dtype])
    return new_dtype

def common_dtype_categorical_compat(objs: List[Any], dtype: Any) -> Any:
    """
    Update the result of find_common_type to account for NAs in a Categorical.
    """
    if lib.is_np_dtype(dtype, 'iu'):
        for obj in objs:
            obj_dtype = getattr(obj, 'dtype', None)
            if isinstance(obj_dtype, CategoricalDtype):
                if isinstance(obj, ABCIndex):
                    hasnas = obj.hasnans
                else:
                    hasnas = cast('Categorical', obj)._hasna
                if hasnas:
                    dtype = np.dtype(np.float64)
                    break
    return dtype

def np_find_common_type(*dtypes: Any) -> Any:
    """
    np.find_common_type implementation pre-1.25 deprecation using np.result_type.
    """
    try:
        common_dtype = np.result_type(*dtypes)
        if common_dtype.kind in 'mMSU':
            common_dtype = np.dtype('O')
    except TypeError:
        common_dtype = np.dtype('O')
    return common_dtype

@overload
def find_common_type(types: List[Any]) -> Any:
    ...

@overload
def find_common_type(types: List[Any]) -> Any:
    ...

@overload
def find_common_type(types: List[Any]) -> Any:
    ...

def find_common_type(types: List[Any]) -> Any:
    """
    Find a common data type among the given dtypes.
    """
    if not types:
        raise ValueError('no types given')
    first = types[0]
    if lib.dtypes_all_equal(list(types)):
        return first
    types = list(dict.fromkeys(types).keys())
    if any((isinstance(t, ExtensionDtype) for t in types)):
        for t in types:
            if isinstance(t, ExtensionDtype):
                res = t._get_common_dtype(types)
                if res is not None:
                    return res
        return np.dtype('object')
    if all((lib.is_np_dtype(t, 'M') for t in types)):
        return np.dtype(max(types))
    if all((lib.is_np_dtype(t, 'm') for t in types)):
        return np.dtype(max(types))
    has_bools = any((t.kind == 'b' for t in types))
    if has_bools:
        for t in types:
            if t.kind in 'iufc':
                return np.dtype('object')
    return np_find_common_type(*types)

def construct_2d_arraylike_from_scalar(value: Any, length: int, width: int, dtype: Any, copy: bool) -> np.ndarray:
    shape = (length, width)
    if dtype.kind in 'mM':
        value = _maybe_box_and_unbox_datetimelike(value, dtype)
    elif dtype == _dtype_obj:
        if isinstance(value, (np.timedelta64, np.datetime64)):
            out = np.empty(shape, dtype=object)
            out.fill(value)
            return out
    try:
        if not copy:
            arr = np.asarray(value, dtype=dtype)
        else:
            arr = np.array(value, dtype=dtype, copy=copy)
    except (ValueError, TypeError) as err:
        raise TypeError(f'DataFrame constructor called with incompatible data and dtype: {err}') from err
    if arr.ndim != 0:
        raise ValueError('DataFrame constructor not properly called!')
    return np.full(shape, arr)

def construct_1d_arraylike_from_scalar(value: Any, length: int, dtype: Any) -> Any:
    """
    create a np.ndarray / pandas type of specified shape and dtype
    filled with value
    """
    if dtype is None:
        try:
            dtype, value = infer_dtype_from_scalar(value)
        except OutOfBoundsDatetime:
            dtype = _dtype_obj
    if isinstance(dtype, ExtensionDtype):
        cls = dtype.construct_array_type()
        seq = [] if length == 0 else [value]
        return cls._from_sequence(seq, dtype=dtype).repeat(length)
    if length and dtype.kind in 'iu' and isna(value):
        dtype = np.dtype('float64')
    elif lib.is_np_dtype(dtype, 'US'):
        dtype = np.dtype('object')
        if not isna(value):
            value = ensure_str(value)
    elif dtype.kind in 'mM':
        value = _maybe_box_and_unbox_datetimelike(value, dtype)
    subarr = np.empty(length, dtype=dtype)
    if length:
        subarr.fill(value)
    return subarr

def _maybe_box_and_unbox_datetimelike(value: Any, dtype: Any) -> Any:
    if isinstance(value, dt.datetime):
        value = maybe_box_datetimelike(value, dtype)
    return _maybe_unbox_datetimelike(value, dtype)

def construct_1d_object_array_from_listlike(values: Any) -> np.ndarray:
    """
    Transform any list-like object in a 1-dimensional numpy array of object dtype.
    """
    return np.fromiter(values, dtype='object', count=len(values))

def maybe_cast_to_integer_array(arr: Any, dtype: Any) -> np.ndarray:
    """
    Takes any dtype and returns the casted version, raising for when data is
    incompatible with integer/unsigned integer dtypes.
    """
    assert dtype.kind in 'iu'
    try:
        if not isinstance(arr, np.ndarray):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    'NumPy will stop allowing conversion of out-of-bound Python int',
                    DeprecationWarning
                )
                casted = np.asarray(arr, dtype=dtype)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                casted = arr.astype(dtype, copy=False)
    except OverflowError as err:
        raise OverflowError(f'The elements provided in the data cannot all be casted to the dtype {dtype}') from err
    if isinstance(arr, np.ndarray) and arr.dtype == dtype:
        return casted
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', 'elementwise comparison failed', FutureWarning)
        if np.array_equal(arr, casted):
            return casted
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, str):
        if (casted.astype(str) == arr).all():
            return casted
        raise ValueError(f'string values cannot be losslessly cast to {dtype}')
    if dtype.kind == 'u' and (arr < 0).any():
        raise OverflowError('Trying to coerce negative values to unsigned integers')
    if arr.dtype.kind == 'f':
        if not np.isfinite(arr).all():
            raise IntCastingNaNError('Cannot convert non-finite values (NA or inf) to integer')
        raise ValueError('Trying to coerce float values to integers')
    if arr.dtype == object:
        raise ValueError('Trying to coerce object values to integers')
    if casted.dtype < arr.dtype:
        raise ValueError(f'Values are too large to be losslessly converted to {dtype}. To cast anyway, use pd.Series(values).astype({dtype})')
    if arr.dtype.kind in 'mM':
        raise TypeError(f'Constructing a Series or DataFrame from {arr.dtype} values and dtype={dtype} is not supported. Use values.view({dtype}) instead.')
    raise ValueError(f'values cannot be losslessly cast to {dtype}')

def can_hold_element(arr: Any, element: Any) -> bool:
    """
    Can we do an inplace setitem with this element in an array with this dtype?
    """
    dtype = arr.dtype
    if not isinstance(dtype, np.dtype) or dtype.kind in 'mM':
        if isinstance(dtype, (PeriodDtype, IntervalDtype, DatetimeTZDtype, np.dtype)):
            arr = cast('PeriodArray | DatetimeArray | TimedeltaArray | IntervalArray', arr)
            try:
                arr._validate_setitem_value(element)
                return True
            except (ValueError, TypeError):
                return False
        if dtype == 'string':
            try:
                arr._maybe_convert_setitem_value(element)
                return True
            except (ValueError, TypeError):
                return False
        return True
    try:
        np_can_hold_element(dtype, element)
        return True
    except (TypeError, LossySetitemError):
        return False

def np_can_hold_element(dtype: Any, element: Any) -> Any:
    """
    Raise if we cannot losslessly set this element into an ndarray with this dtype.
    """
    if dtype == _dtype_obj:
        return element
    tipo = _maybe_infer_dtype_type(element)
    if dtype.kind in 'iu':
        if isinstance(element, range):
            if _dtype_can_hold_range(element, dtype):
                return element
            raise LossySetitemError
        if is_integer(element) or (is_float(element) and element.is_integer()):
            info = np.iinfo(dtype)
            if info.min <= element <= info.max:
                return dtype.type(element)
            raise LossySetitemError
        if tipo is not None:
            if tipo.kind not in 'iu':
                if isinstance(element, np.ndarray) and element.dtype.kind == 'f':
                    with np.errstate(invalid='ignore'):
                        casted = element.astype(dtype)
                    comp = casted == element
                    if comp.all():
                        return casted
                    raise LossySetitemError
                elif isinstance(element, ABCExtensionArray) and isinstance(element.dtype, CategoricalDtype):
                    try:
                        casted = element.astype(dtype)
                    except (ValueError, TypeError) as err:
                        raise LossySetitemError from err
                    comp = casted == element
                    if not comp.all():
                        raise LossySetitemError
                    return casted
                raise LossySetitemError
            if dtype.kind == 'u' and isinstance(element, np.ndarray) and (element.dtype.kind == 'i'):
                casted = element.astype(dtype)
                if (casted == element).all():
                    return casted
                raise LossySetitemError
            if dtype.itemsize < tipo.itemsize:
                raise LossySetitemError
            if not isinstance(tipo, np.dtype):
                arr = element._values if isinstance(element, ABCSeries) else element
                if arr._hasna:
                    raise LossySetitemError
                return element
            return element
        raise LossySetitemError
    if dtype.kind == 'f':
        if lib.is_integer(element) or lib.is_float(element):
            casted = dtype.type(element)
            if np.isnan(casted) or casted == element:
                return casted
            raise LossySetitemError
        if tipo is not None:
            if tipo.kind not in 'iuf':
                raise LossySetitemError
            if not isinstance(tipo, np.dtype):
                if element._hasna:
                    raise LossySetitemError
                return element
            elif tipo.itemsize > dtype.itemsize or tipo.kind != dtype.kind:
                if isinstance(element, np.ndarray):
                    casted = element.astype(dtype)
                    if np.array_equal(casted, element, equal_nan=True):
                        return casted
                    raise LossySetitemError
            return element
        raise LossySetitemError
    if dtype.kind == 'c':
        if lib.is_integer(element) or lib.is_complex(element) or lib.is_float(element):
            if np.isnan(element):
                return dtype.type(element)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                casted = dtype.type(element)
            if casted == element:
                return casted
            raise LossySetitemError
        if tipo is not None:
            if tipo.kind in 'iufc':
                return element
            raise LossySetitemError
        raise LossySetitemError
    if dtype.kind == 'b':
        if tipo is not None:
            if tipo.kind == 'b':
                if not isinstance(tipo, np.dtype):
                    if element._hasna:
                        raise LossySetitemError
                return element
            raise LossySetitemError
        if lib.is_bool(element):
            return element
        raise LossySetitemError
    if dtype.kind == 'S':
        if tipo is not None:
            if tipo.kind == 'S' and tipo.itemsize <= dtype.itemsize:
                return element
            raise LossySetitemError
        if isinstance(element, bytes) and len(element) <= dtype.itemsize:
            return element
        raise LossySetitemError
    if dtype.kind == 'V':
        raise LossySetitemError
    raise NotImplementedError(dtype)

def _dtype_can_hold_range(rng: range, dtype: Any) -> bool:
    """
    Check if a range can be held by the given dtype.
    """
    if not len(rng):
        return True
    return np_can_cast_scalar(rng.start, dtype) and np_can_cast_scalar(rng.stop, dtype)

def np_can_cast_scalar(element: Any, dtype: Any) -> bool:
    """
    np.can_cast pandas-equivalent for pre 2-0 behavior that allowed scalar
    inference.
    """
    try:
        np_can_hold_element(dtype, element)
        return True
    except (LossySetitemError, NotImplementedError):
        return False

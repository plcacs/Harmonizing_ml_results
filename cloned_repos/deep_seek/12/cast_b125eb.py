"""
Routines for casting.
"""
from __future__ import annotations
import datetime as dt
import functools
from typing import (
    TYPE_CHECKING, Any, Literal, TypeVar, cast, overload, Union, Optional, 
    List, Tuple, Dict, Set, Callable, Type, Sequence, Collection, Iterable
)
import warnings
import numpy as np
from pandas._config import using_string_dtype
from pandas._libs import Interval, Period, lib
from pandas._libs.missing import NA, NAType, checknull
from pandas._libs.tslibs import (
    NaT, OutOfBoundsDatetime, OutOfBoundsTimedelta, Timedelta, Timestamp, 
    is_supported_dtype
)
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.errors import IntCastingNaNError, LossySetitemError
from pandas.core.dtypes.common import (
    ensure_int8, ensure_int16, ensure_int32, ensure_int64, ensure_object, 
    ensure_str, is_bool, is_complex, is_float, is_integer, is_object_dtype, 
    is_scalar, is_string_dtype, pandas_dtype as pandas_dtype_func
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype, BaseMaskedDtype, CategoricalDtype, DatetimeTZDtype, 
    ExtensionDtype, IntervalDtype, PandasExtensionDtype, PeriodDtype
)
from pandas.core.dtypes.generic import ABCExtensionArray, ABCIndex, ABCSeries
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype, isna, na_value_for_dtype, notna
)
from pandas.io._util import _arrow_dtype_mapping

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from pandas._typing import (
        ArrayLike, Dtype, DtypeObj, NumpyIndexT, Scalar, npt, 
        DtypeBackend, Self
    )
    from pandas import Index, Series
    from pandas.core.arrays import (
        Categorical, DatetimeArray, ExtensionArray, IntervalArray, 
        PeriodArray, TimedeltaArray
    )

_int8_max: int = np.iinfo(np.int8).max
_int16_max: int = np.iinfo(np.int16).max
_int32_max: int = np.iinfo(np.int32).max
_dtype_obj: np.dtype = np.dtype(object)
NumpyArrayT = TypeVar('NumpyArrayT', bound=np.ndarray)

def maybe_convert_platform(values: Union[list, tuple, range, np.ndarray]) -> np.ndarray:
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

    This may not be necessarily be performant.
    """
    return bool(isinstance(obj, ABCSeries) and is_object_dtype(obj.dtype) and 
               any((isinstance(v, ABCSeries) for v in obj._values))

def maybe_box_datetimelike(value: Any, dtype: Optional[np.dtype] = None) -> Any:
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

def _maybe_unbox_datetimelike(value: Any, dtype: np.dtype) -> Any:
    """
    Convert a Timedelta or Timestamp to timedelta64 or datetime64 for setting
    into a numpy array. Failing to unbox would risk dropping nanoseconds.
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

def _disallow_mismatched_datetimelike(value: Any, dtype: np.dtype) -> None:
    """
    numpy allows np.array(dt64values, dtype="timedelta64[ns]") and
    vice-versa, but we do not want to allow this, so we need to
    check explicitly
    """
    vdtype = getattr(value, 'dtype', None)
    if vdtype is None:
        return
    elif (vdtype.kind == 'm' and dtype.kind == 'M') or (vdtype.kind == 'M' and dtype.kind == 'm'):
        raise TypeError(f'Cannot cast {value!r} to {dtype}')

@overload
def maybe_downcast_to_dtype(result: np.ndarray, dtype: np.dtype) -> np.ndarray: ...

@overload
def maybe_downcast_to_dtype(result: ABCExtensionArray, dtype: ExtensionDtype) -> ABCExtensionArray: ...

def maybe_downcast_to_dtype(result: Union[np.ndarray, ABCExtensionArray], 
                          dtype: Union[np.dtype, ExtensionDtype]) -> Union[np.ndarray, ABCExtensionArray]:
    """
    try to cast to the specified dtype (e.g. convert back to bool/int
    or could be an astype of float64->float32
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
def maybe_downcast_numeric(result: np.ndarray, dtype: np.dtype, do_round: bool = False) -> np.ndarray: ...

@overload
def maybe_downcast_numeric(result: ABCExtensionArray, dtype: ExtensionDtype, do_round: bool = False) -> ABCExtensionArray: ...

def maybe_downcast_numeric(result: Union[np.ndarray, ABCExtensionArray], 
                         dtype: Union[np.dtype, ExtensionDtype], 
                         do_round: bool = False) -> Union[np.ndarray, ABCExtensionArray]:
    """
    Subset of maybe_downcast_to_dtype restricted to numeric dtypes.
    """
    if not isinstance(dtype, np.dtype) or not isinstance(result.dtype, np.dtype):
        return result

    def trans(x: np.ndarray) -> np.ndarray:
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
    elif (issubclass(dtype.type, np.floating) and result.dtype.kind != 'b' and 
          not is_string_dtype(result.dtype)):
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

def maybe_upcast_numeric_to_64bit(arr: Union[np.ndarray, ABCExtensionArray]) -> Union[np.ndarray, ABCExtensionArray]:
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

def maybe_cast_pointwise_result(
    result: ArrayLike, 
    dtype: Union[np.dtype, ExtensionDtype], 
    numeric_only: bool = False, 
    same_dtype: bool = True
) -> ArrayLike:
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
    elif numeric_only and dtype.kind in 'iufcb' or not numeric_only:
        result = maybe_downcast_to_dtype(result, dtype)
    return result

def _maybe_cast_to_extension_array(
    cls: Type[ExtensionArray], 
    obj: Any, 
    dtype: Optional[ExtensionDtype] = None
) -> Union[ExtensionArray, Any]:
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
def ensure_dtype_can_hold_na(dtype: np.dtype) -> np.dtype: ...

@overload
def ensure_dtype_can_hold_na(dtype: ExtensionDtype) -> Union[ExtensionDtype, np.dtype]: ...

def ensure_dtype_can_hold_na(dtype: Union[np.dtype, ExtensionDtype]) -> Union[np.dtype, ExtensionDtype]:
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

_canonical_nans: Dict[Type, Any] = {
    np.datetime64: np.datetime64('NaT', 'ns'), 
    np.timedelta64: np.timedelta64('NaT', 'ns'), 
    type(np.nan): np.nan
}

def maybe_promote(
    dtype: np.dtype, 
    fill_value: Any = np.nan
) -> Tuple[np.dtype, Any]:
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
def _maybe_promote_cached(
    dtype: np.dtype, 
    fill_value: Any, 
    fill_value_type: Type
) -> Tuple[np.dtype, Any]:
    return _maybe_promote(dtype, fill_value)

def _maybe_promote(
    dtype: np.dtype, 
    fill_value: Any = np.nan
) -> Tuple[np.dtype, Any]:
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
        if not issub
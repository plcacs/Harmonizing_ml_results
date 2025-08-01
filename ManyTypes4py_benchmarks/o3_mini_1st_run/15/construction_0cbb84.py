#!/usr/bin/env python
"""
Constructor functions intended to be shared by pd.array, Series.__init__,
and Index.__new__.

These should not depend on core.internals.
"""

from __future__ import annotations
from typing import Any, Optional, Union, overload
import numpy as np
from numpy import ma
from pandas._config import using_string_dtype
from pandas._libs import lib
from pandas._libs.tslibs import get_supported_dtype, is_supported_dtype
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import (
    construct_1d_arraylike_from_scalar,
    construct_1d_object_array_from_listlike,
    maybe_cast_to_datetime,
    maybe_cast_to_integer_array,
    maybe_convert_platform,
    maybe_infer_to_datetimelike,
    maybe_promote,
)
from pandas.core.dtypes.common import ensure_object, is_list_like, is_object_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCExtensionArray, ABCIndex, ABCSeries
from pandas.core.dtypes.missing import isna
import pandas.core.common as com

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pandas._typing import AnyArrayLike, ArrayLike, Dtype, DtypeObj, T
    from pandas import Index, Series
    from pandas.core.arrays import DatetimeArray, ExtensionArray, TimedeltaArray

# Import ExtensionArray types for type annotations below.
from pandas.core.arrays import (
    BooleanArray,
    DatetimeArray,
    ExtensionArray,
    FloatingArray,
    IntegerArray,
    NumpyExtensionArray,
    TimedeltaArray,
)
from pandas.core.arrays.string_ import StringDtype

def array(
    data: Any,
    dtype: Optional[Union[str, np.dtype, ExtensionDtype]] = None,
    copy: bool = True,
) -> ExtensionArray:
    """
    Create an array.

    This method constructs an array using pandas extension types when possible.
    If `dtype` is specified, it determines the type of array returned. Otherwise,
    pandas attempts to infer the appropriate dtype based on `data`.

    Parameters
    ----------
    data : Sequence of objects
        The scalars inside `data` should be instances of the
        scalar type for `dtype`. It's expected that `data`
        represents a 1-dimensional array of data.

        When `data` is a Index or Series, the underlying array
        will be extracted from `data`.

    dtype : str, np.dtype, or ExtensionDtype, optional
        The dtype to use for the array.
    copy : bool, default True
        Whether to copy the data, even if not necessary.

    Returns
    -------
    ExtensionArray
        The newly created array.

    Raises
    ------
    ValueError
        When `data` is not 1-dimensional.
    """
    if lib.is_scalar(data):
        msg = f"Cannot pass scalar '{data}' to 'pandas.array'."
        raise ValueError(msg)
    elif isinstance(data, ABCDataFrame):
        raise TypeError("Cannot pass DataFrame to 'pandas.array'")
    if dtype is None and isinstance(data, (ABCSeries, ABCIndex, ABCExtensionArray)):
        dtype = data.dtype  # type: ignore[union-attr]
    data = extract_array(data, extract_numpy=True)
    if dtype is not None:
        dtype = pandas_dtype(dtype)
    if isinstance(data, ABCExtensionArray) and (dtype is None or data.dtype == dtype):
        if copy:
            return data.copy()  # type: ignore[union-attr]
        return data
    if isinstance(dtype, ExtensionDtype):
        cls = dtype.construct_array_type()
        return cls._from_sequence(data, dtype=dtype, copy=copy)  # type: ignore
    if dtype is None:
        was_ndarray: bool = isinstance(data, np.ndarray)
        if not was_ndarray or data.dtype == object:
            result = lib.maybe_convert_objects(
                ensure_object(data),
                convert_non_numeric=True,
                convert_to_nullable_dtype=True,
                dtype_if_all_nat=None,
            )
            result = ensure_wrapped_if_datetimelike(result)
            if isinstance(result, np.ndarray):
                if len(result) == 0 and (not was_ndarray):
                    return FloatingArray._from_sequence(data, dtype='Float64')
                return NumpyExtensionArray._from_sequence(data, dtype=result.dtype, copy=copy)
            if result is data and copy:
                return result.copy()
            return result
        data = data  # type: np.ndarray
        result = ensure_wrapped_if_datetimelike(data)
        if result is not data:
            # result is a DatetimeArray or TimedeltaArray
            if copy and result.dtype == data.dtype:
                return result.copy()
            return result
        if data.dtype.kind in 'SU':
            dtype = StringDtype()
            cls = dtype.construct_array_type()
            return cls._from_sequence(data, dtype=dtype, copy=copy)
        elif data.dtype.kind in 'iu':
            dtype = IntegerArray._dtype_cls._get_dtype_mapping()[data.dtype]
            return IntegerArray._from_sequence(data, dtype=dtype, copy=copy)
        elif data.dtype.kind == 'f':
            if data.dtype == np.float16:
                return NumpyExtensionArray._from_sequence(data, dtype=data.dtype, copy=copy)
            dtype = FloatingArray._dtype_cls._get_dtype_mapping()[data.dtype]
            return FloatingArray._from_sequence(data, dtype=dtype, copy=copy)
        elif data.dtype.kind == 'b':
            return BooleanArray._from_sequence(data, dtype='boolean', copy=copy)
        else:
            return NumpyExtensionArray._from_sequence(data, dtype=data.dtype, copy=copy)
    if lib.is_np_dtype(dtype, 'M') and is_supported_dtype(dtype):
        return DatetimeArray._from_sequence(data, dtype=dtype, copy=copy)
    if lib.is_np_dtype(dtype, 'm') and is_supported_dtype(dtype):
        return TimedeltaArray._from_sequence(data, dtype=dtype, copy=copy)
    elif lib.is_np_dtype(dtype, 'mM'):
        raise ValueError("datetime64 and timedelta64 dtype resolutions other than 's', 'ms', 'us', and 'ns' are no longer supported.")
    return NumpyExtensionArray._from_sequence(data, dtype=dtype, copy=copy)

_typs: frozenset[str] = frozenset({
    'index', 'rangeindex', 'multiindex', 'datetimeindex', 'timedeltaindex',
    'periodindex', 'categoricalindex', 'intervalindex', 'series'
})

@overload
def extract_array(obj: Any, extract_numpy: bool = ..., extract_range: bool = ...) -> Any:
    ...

@overload
def extract_array(obj: Any, extract_numpy: bool = ..., extract_range: bool = ...) -> Any:
    ...

def extract_array(obj: Any, extract_numpy: bool = False, extract_range: bool = False) -> Any:
    """
    Extract the ndarray or ExtensionArray from a Series or Index.

    For all other types, `obj` is just returned as is.

    Parameters
    ----------
    obj : object
        For Series / Index, the underlying ExtensionArray is unboxed.

    extract_numpy : bool, default False
        Whether to extract the ndarray from a NumpyExtensionArray.

    extract_range : bool, default False
        If we have a RangeIndex, return range._values if True
        (which is a materialized integer ndarray), otherwise return unchanged.

    Returns
    -------
    arr : object
    """
    typ: Optional[str] = getattr(obj, '_typ', None)
    if typ in _typs:
        if typ == 'rangeindex':
            if extract_range:
                return obj._values
            return obj
        return obj._values
    elif extract_numpy and typ == 'npy_extension':
        return obj.to_numpy()
    return obj

def ensure_wrapped_if_datetimelike(arr: Any) -> Any:
    """
    Wrap datetime64 and timedelta64 ndarrays in DatetimeArray/TimedeltaArray.
    """
    if isinstance(arr, np.ndarray):
        if arr.dtype.kind == 'M':
            dtype = get_supported_dtype(arr.dtype)
            return DatetimeArray._from_sequence(arr, dtype=dtype)
        elif arr.dtype.kind == 'm':
            dtype = get_supported_dtype(arr.dtype)
            return TimedeltaArray._from_sequence(arr, dtype=dtype)
    return arr

def sanitize_masked_array(data: ma.MaskedArray) -> ma.MaskedArray:
    """
    Convert numpy MaskedArray to ensure mask is softened.
    """
    mask: np.ndarray = ma.getmaskarray(data)
    if mask.any():
        dtype, fill_value = maybe_promote(data.dtype, np.nan)
        dtype = cast(np.dtype, dtype)
        data = ma.asarray(data.astype(dtype, copy=True))
        data.soften_mask()
        data[mask] = fill_value
    else:
        data = data.copy()
    return data

def sanitize_array(
    data: Any,
    index: Optional[Any],
    dtype: Optional[Union[np.dtype, ExtensionDtype]] = None,
    copy: bool = False,
    *,
    allow_2d: bool = False,
) -> Union[np.ndarray, ExtensionArray]:
    """
    Sanitize input data to an ndarray or ExtensionArray, copy if specified,
    coerce to the dtype if specified.

    Parameters
    ----------
    data : Any
    index : Index or None
    dtype : np.dtype, ExtensionDtype, or None, default None
    copy : bool, default False
    allow_2d : bool, default False
        If False, raise if we have a 2D Arraylike.

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    original_dtype: Optional[Union[np.dtype, ExtensionDtype]] = dtype
    if isinstance(data, ma.MaskedArray):
        data = sanitize_masked_array(data)
    if isinstance(dtype, NumpyEADtype):
        dtype = dtype.numpy_dtype
    infer_object: bool = not isinstance(data, (ABCIndex, ABCSeries))
    data = extract_array(data, extract_numpy=True, extract_range=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        if dtype is None:
            dtype = data.dtype
        data = lib.item_from_zerodim(data)
    elif isinstance(data, range):
        data = range_to_ndarray(data)
        copy = False
    if not is_list_like(data):
        if index is None:
            raise ValueError('index must be specified when data is not list-like')
        if isinstance(data, str) and using_string_dtype() and (original_dtype is None):
            from pandas.core.arrays.string_ import StringDtype
            dtype = StringDtype(na_value=np.nan)
        data = construct_1d_arraylike_from_scalar(data, len(index), dtype)
        return data
    elif isinstance(data, ABCExtensionArray):
        if dtype is not None:
            subarr = data.astype(dtype, copy=copy)  # type: ignore
        elif copy:
            subarr = data.copy()  # type: ignore
        else:
            subarr = data
    elif isinstance(dtype, ExtensionDtype):
        _sanitize_non_ordered(data)
        cls = dtype.construct_array_type()
        if not hasattr(data, '__array__'):
            data = list(data)
        subarr = cls._from_sequence(data, dtype=dtype, copy=copy)  # type: ignore
    elif isinstance(data, np.ndarray):
        if isinstance(data, np.matrix):
            data = data.A
        if dtype is None:
            subarr: Union[np.ndarray, ExtensionArray] = data
            if data.dtype == object and infer_object:
                subarr = maybe_infer_to_datetimelike(data)
            elif data.dtype.kind == 'U' and using_string_dtype():
                from pandas.core.arrays.string_ import StringDtype
                dtype = StringDtype(na_value=np.nan)
                subarr = dtype.construct_array_type()._from_sequence(data, dtype=dtype)
            if (subarr is data or (subarr.dtype == 'str' and subarr.dtype.storage == 'python')) and copy:
                subarr = subarr.copy()
        else:
            subarr = _try_cast(data, dtype, copy)
    elif hasattr(data, '__array__'):
        if not copy:
            data = np.asarray(data)
        else:
            data = np.array(data, copy=copy)
        return sanitize_array(data, index=index, dtype=dtype, copy=False, allow_2d=allow_2d)
    else:
        _sanitize_non_ordered(data)
        data = list(data)
        if len(data) == 0 and dtype is None:
            subarr = np.array([], dtype=np.float64)
        elif dtype is not None:
            subarr = _try_cast(data, dtype, copy)
        else:
            subarr = maybe_convert_platform(data)
            if subarr.dtype == object:
                subarr = cast(np.ndarray, subarr)
                subarr = maybe_infer_to_datetimelike(subarr)
    subarr = _sanitize_ndim(subarr, data, dtype, index, allow_2d=allow_2d)
    if isinstance(subarr, np.ndarray):
        dtype = cast(np.dtype, dtype)
        subarr = _sanitize_str_dtypes(subarr, data, dtype, copy)
    return subarr

def range_to_ndarray(rng: range) -> np.ndarray:
    """
    Cast a range object to ndarray.
    """
    try:
        arr: np.ndarray = np.arange(rng.start, rng.stop, rng.step, dtype='int64')
    except OverflowError:
        if rng.start >= 0 and rng.step > 0 or rng.step < 0 <= rng.stop:
            try:
                arr = np.arange(rng.start, rng.stop, rng.step, dtype='uint64')
            except OverflowError:
                arr = construct_1d_object_array_from_listlike(list(rng))
        else:
            arr = construct_1d_object_array_from_listlike(list(rng))
    return arr

def _sanitize_non_ordered(data: Any) -> None:
    """
    Raise only for unordered sets, e.g., not for dict_keys
    """
    if isinstance(data, (set, frozenset)):
        raise TypeError(f"'{type(data).__name__}' type is unordered")

def _sanitize_ndim(
    result: Any,
    data: Any,
    dtype: Optional[Union[np.dtype, ExtensionDtype]],
    index: Optional[Any],
    *,
    allow_2d: bool = False,
) -> Any:
    """
    Ensure we have a 1-dimensional result array.
    """
    if getattr(result, 'ndim', 0) == 0:
        raise ValueError('result should be arraylike with ndim > 0')
    if result.ndim == 1:
        result = _maybe_repeat(result, index)
    elif result.ndim > 1:
        if isinstance(data, np.ndarray):
            if allow_2d:
                return result
            raise ValueError(f'Data must be 1-dimensional, got ndarray of shape {data.shape} instead')
        if is_object_dtype(dtype) and isinstance(dtype, ExtensionDtype):
            result = com.asarray_tuplesafe(data, dtype=np.dtype('object'))
            cls = dtype.construct_array_type()
            result = cls._from_sequence(result, dtype=dtype)
        else:
            result = com.asarray_tuplesafe(data, dtype=dtype)  # type: ignore
    return result

def _sanitize_str_dtypes(
    result: np.ndarray, data: Any, dtype: Union[np.dtype, ExtensionDtype], copy: bool
) -> np.ndarray:
    """
    Ensure we have a dtype that is supported by pandas.
    """
    if issubclass(result.dtype.type, str):
        if not lib.is_scalar(data):
            if not np.all(isna(data)):
                data = np.asarray(data, dtype=dtype)
            if not copy:
                result = np.asarray(data, dtype=object)
            else:
                result = np.array(data, dtype=object, copy=copy)
    return result

def _maybe_repeat(arr: np.ndarray, index: Optional[Any]) -> np.ndarray:
    """
    If we have a length-1 array and an index describing how long we expect
    the result to be, repeat the array.
    """
    if index is not None:
        if 1 == len(arr) != len(index):
            arr = arr.repeat(len(index))
    return arr

def _try_cast(
    arr: Any, dtype: Union[np.dtype, ExtensionDtype], copy: bool
) -> Union[np.ndarray, ExtensionArray]:
    """
    Convert input to numpy ndarray and optionally cast to a given dtype.

    Parameters
    ----------
    arr : ndarray or list
        Excludes: ExtensionArray, Series, Index.
    dtype : np.dtype or ExtensionDtype
    copy : bool
        If False, don't copy the data if not needed.

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    is_ndarray: bool = isinstance(arr, np.ndarray)
    if dtype == object:
        if not is_ndarray:
            subarr = construct_1d_object_array_from_listlike(arr)
            return subarr
        return ensure_wrapped_if_datetimelike(arr).astype(dtype, copy=copy)
    elif dtype.kind == 'U':
        if is_ndarray:
            arr = cast(np.ndarray, arr)
            shape = arr.shape
            if arr.ndim > 1:
                arr = arr.ravel()
        else:
            shape = (len(arr),)
        return lib.ensure_string_array(arr, convert_na_value=False, copy=copy).reshape(shape)
    elif dtype.kind in 'mM':
        if is_ndarray:
            arr = cast(np.ndarray, arr)
            if arr.ndim == 2 and arr.shape[1] == 1:
                return maybe_cast_to_datetime(arr[:, 0], dtype).reshape(arr.shape)
        return maybe_cast_to_datetime(arr, dtype)
    elif dtype.kind in 'iu':
        subarr = maybe_cast_to_integer_array(arr, dtype)
    elif not copy:
        subarr = np.asarray(arr, dtype=dtype)
    else:
        subarr = np.array(arr, dtype=dtype, copy=copy)
    return subarr

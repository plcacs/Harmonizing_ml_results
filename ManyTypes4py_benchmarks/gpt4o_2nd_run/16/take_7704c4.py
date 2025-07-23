from __future__ import annotations
import functools
from typing import TYPE_CHECKING, cast, overload, Any, Tuple, Optional, Callable
import numpy as np
from pandas._libs import algos as libalgos, lib
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import ensure_platform_int, is_1d_only_ea_dtype
from pandas.core.dtypes.missing import na_value_for_dtype
from pandas.core.construction import ensure_wrapped_if_datetimelike

if TYPE_CHECKING:
    from pandas._typing import ArrayLike, AxisInt, npt
    from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
    from pandas.core.arrays.base import ExtensionArray

@overload
def take_nd(arr: np.ndarray, indexer: np.ndarray, axis: int = ..., fill_value: Any = ..., allow_fill: bool = ...) -> np.ndarray:
    ...

@overload
def take_nd(arr: ExtensionArray, indexer: np.ndarray, axis: int = ..., fill_value: Any = ..., allow_fill: bool = ...) -> ExtensionArray:
    ...

def take_nd(arr: ArrayLike, indexer: np.ndarray, axis: int = 0, fill_value: Any = lib.no_default, allow_fill: bool = True) -> ArrayLike:
    if fill_value is lib.no_default:
        fill_value = na_value_for_dtype(arr.dtype, compat=False)
    elif lib.is_np_dtype(arr.dtype, 'mM'):
        dtype, fill_value = maybe_promote(arr.dtype, fill_value)
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
    if not isinstance(arr, np.ndarray):
        if not is_1d_only_ea_dtype(arr.dtype):
            arr = cast('NDArrayBackedExtensionArray', arr)
            return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill, axis=axis)
        return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill)
    arr = np.asarray(arr)
    return _take_nd_ndarray(arr, indexer, axis, fill_value, allow_fill)

def _take_nd_ndarray(arr: np.ndarray, indexer: np.ndarray, axis: int, fill_value: Any, allow_fill: bool) -> np.ndarray:
    if indexer is None:
        indexer = np.arange(arr.shape[axis], dtype=np.intp)
        dtype, fill_value = (arr.dtype, arr.dtype.type())
    else:
        indexer = ensure_platform_int(indexer)
    dtype, fill_value, mask_info = _take_preprocess_indexer_and_fill_value(arr, indexer, fill_value, allow_fill)
    flip_order = False
    if arr.ndim == 2 and arr.flags.f_contiguous:
        flip_order = True
    if flip_order:
        arr = arr.T
        axis = arr.ndim - axis - 1
    out_shape_ = list(arr.shape)
    out_shape_[axis] = len(indexer)
    out_shape = tuple(out_shape_)
    if arr.flags.f_contiguous and axis == arr.ndim - 1:
        out = np.empty(out_shape, dtype=dtype, order='F')
    else:
        out = np.empty(out_shape, dtype=dtype)
    func = _get_take_nd_function(arr.ndim, arr.dtype, out.dtype, axis=axis, mask_info=mask_info)
    func(arr, indexer, out, fill_value)
    if flip_order:
        out = out.T
    return out

def take_2d_multi(arr: np.ndarray, indexer: Tuple[np.ndarray, np.ndarray], fill_value: Any = np.nan) -> np.ndarray:
    assert indexer is not None
    assert indexer[0] is not None
    assert indexer[1] is not None
    row_idx, col_idx = indexer
    row_idx = ensure_platform_int(row_idx)
    col_idx = ensure_platform_int(col_idx)
    indexer = (row_idx, col_idx)
    mask_info = None
    dtype, fill_value = maybe_promote(arr.dtype, fill_value)
    if dtype != arr.dtype:
        row_mask = row_idx == -1
        col_mask = col_idx == -1
        row_needs = row_mask.any()
        col_needs = col_mask.any()
        mask_info = ((row_mask, col_mask), (row_needs, col_needs))
        if not (row_needs or col_needs):
            dtype, fill_value = (arr.dtype, arr.dtype.type())
    out_shape = (len(row_idx), len(col_idx))
    out = np.empty(out_shape, dtype=dtype)
    func = _take_2d_multi_dict.get((arr.dtype.name, out.dtype.name), None)
    if func is None and arr.dtype != out.dtype:
        func = _take_2d_multi_dict.get((out.dtype.name, out.dtype.name), None)
        if func is not None:
            func = _convert_wrapper(func, out.dtype)
    if func is not None:
        func(arr, indexer, out=out, fill_value=fill_value)
    else:
        _take_2d_multi_object(arr, indexer, out, fill_value=fill_value, mask_info=mask_info)
    return out

@functools.lru_cache
def _get_take_nd_function_cached(ndim: int, arr_dtype: np.dtype, out_dtype: np.dtype, axis: int) -> Optional[Callable]:
    tup = (arr_dtype.name, out_dtype.name)
    if ndim == 1:
        func = _take_1d_dict.get(tup, None)
    elif ndim == 2:
        if axis == 0:
            func = _take_2d_axis0_dict.get(tup, None)
        else:
            func = _take_2d_axis1_dict.get(tup, None)
    if func is not None:
        return func
    tup = (out_dtype.name, out_dtype.name)
    if ndim == 1:
        func = _take_1d_dict.get(tup, None)
    elif ndim == 2:
        if axis == 0:
            func = _take_2d_axis0_dict.get(tup, None)
        else:
            func = _take_2d_axis1_dict.get(tup, None)
    if func is not None:
        func = _convert_wrapper(func, out_dtype)
        return func
    return None

def _get_take_nd_function(ndim: int, arr_dtype: np.dtype, out_dtype: np.dtype, axis: int = 0, mask_info: Optional[Tuple[np.ndarray, bool]] = None) -> Callable:
    func = None
    if ndim <= 2:
        func = _get_take_nd_function_cached(ndim, arr_dtype, out_dtype, axis)
    if func is None:
        def func(arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value: Any = np.nan) -> None:
            indexer = ensure_platform_int(indexer)
            _take_nd_object(arr, indexer, out, axis=axis, fill_value=fill_value, mask_info=mask_info)
    return func

def _view_wrapper(f: Callable, arr_dtype: Optional[np.dtype] = None, out_dtype: Optional[np.dtype] = None, fill_wrap: Optional[Callable] = None) -> Callable:
    def wrapper(arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value: Any = np.nan) -> None:
        if arr_dtype is not None:
            arr = arr.view(arr_dtype)
        if out_dtype is not None:
            out = out.view(out_dtype)
        if fill_wrap is not None:
            if fill_value.dtype.kind == 'm':
                fill_value = fill_value.astype('m8[ns]')
            else:
                fill_value = fill_value.astype('M8[ns]')
            fill_value = fill_wrap(fill_value)
        f(arr, indexer, out, fill_value=fill_value)
    return wrapper

def _convert_wrapper(f: Callable, conv_dtype: np.dtype) -> Callable:
    def wrapper(arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value: Any = np.nan) -> None:
        if conv_dtype == object:
            arr = ensure_wrapped_if_datetimelike(arr)
        arr = arr.astype(conv_dtype)
        f(arr, indexer, out, fill_value=fill_value)
    return wrapper

_take_1d_dict: dict[Tuple[str, str], Callable] = {
    ('int8', 'int8'): libalgos.take_1d_int8_int8,
    ('int8', 'int32'): libalgos.take_1d_int8_int32,
    ('int8', 'int64'): libalgos.take_1d_int8_int64,
    ('int8', 'float64'): libalgos.take_1d_int8_float64,
    ('int16', 'int16'): libalgos.take_1d_int16_int16,
    ('int16', 'int32'): libalgos.take_1d_int16_int32,
    ('int16', 'int64'): libalgos.take_1d_int16_int64,
    ('int16', 'float64'): libalgos.take_1d_int16_float64,
    ('int32', 'int32'): libalgos.take_1d_int32_int32,
    ('int32', 'int64'): libalgos.take_1d_int32_int64,
    ('int32', 'float64'): libalgos.take_1d_int32_float64,
    ('int64', 'int64'): libalgos.take_1d_int64_int64,
    ('uint8', 'uint8'): libalgos.take_1d_bool_bool,
    ('uint16', 'int64'): libalgos.take_1d_uint16_uint16,
    ('uint32', 'int64'): libalgos.take_1d_uint32_uint32,
    ('uint64', 'int64'): libalgos.take_1d_uint64_uint64,
    ('int64', 'float64'): libalgos.take_1d_int64_float64,
    ('float32', 'float32'): libalgos.take_1d_float32_float32,
    ('float32', 'float64'): libalgos.take_1d_float32_float64,
    ('float64', 'float64'): libalgos.take_1d_float64_float64,
    ('object', 'object'): libalgos.take_1d_object_object,
    ('bool', 'bool'): _view_wrapper(libalgos.take_1d_bool_bool, np.uint8, np.uint8),
    ('bool', 'object'): _view_wrapper(libalgos.take_1d_bool_object, np.uint8, None),
    ('datetime64[ns]', 'datetime64[ns]'): _view_wrapper(libalgos.take_1d_int64_int64, np.int64, np.int64, np.int64),
    ('timedelta64[ns]', 'timedelta64[ns]'): _view_wrapper(libalgos.take_1d_int64_int64, np.int64, np.int64, np.int64)
}

_take_2d_axis0_dict: dict[Tuple[str, str], Callable] = {
    ('int8', 'int8'): libalgos.take_2d_axis0_int8_int8,
    ('int8', 'int32'): libalgos.take_2d_axis0_int8_int32,
    ('int8', 'int64'): libalgos.take_2d_axis0_int8_int64,
    ('int8', 'float64'): libalgos.take_2d_axis0_int8_float64,
    ('int16', 'int16'): libalgos.take_2d_axis0_int16_int16,
    ('int16', 'int32'): libalgos.take_2d_axis0_int16_int32,
    ('int16', 'int64'): libalgos.take_2d_axis0_int16_int64,
    ('int16', 'float64'): libalgos.take_2d_axis0_int16_float64,
    ('int32', 'int32'): libalgos.take_2d_axis0_int32_int32,
    ('int32', 'int64'): libalgos.take_2d_axis0_int32_int64,
    ('int32', 'float64'): libalgos.take_2d_axis0_int32_float64,
    ('int64', 'int64'): libalgos.take_2d_axis0_int64_int64,
    ('int64', 'float64'): libalgos.take_2d_axis0_int64_float64,
    ('uint8', 'uint8'): libalgos.take_2d_axis0_bool_bool,
    ('uint16', 'uint16'): libalgos.take_2d_axis0_uint16_uint16,
    ('uint32', 'uint32'): libalgos.take_2d_axis0_uint32_uint32,
    ('uint64', 'uint64'): libalgos.take_2d_axis0_uint64_uint64,
    ('float32', 'float32'): libalgos.take_2d_axis0_float32_float32,
    ('float32', 'float64'): libalgos.take_2d_axis0_float32_float64,
    ('float64', 'float64'): libalgos.take_2d_axis0_float64_float64,
    ('object', 'object'): libalgos.take_2d_axis0_object_object,
    ('bool', 'bool'): _view_wrapper(libalgos.take_2d_axis0_bool_bool, np.uint8, np.uint8),
    ('bool', 'object'): _view_wrapper(libalgos.take_2d_axis0_bool_object, np.uint8, None),
    ('datetime64[ns]', 'datetime64[ns]'): _view_wrapper(libalgos.take_2d_axis0_int64_int64, np.int64, np.int64, fill_wrap=np.int64),
    ('timedelta64[ns]', 'timedelta64[ns]'): _view_wrapper(libalgos.take_2d_axis0_int64_int64, np.int64, np.int64, fill_wrap=np.int64)
}

_take_2d_axis1_dict: dict[Tuple[str, str], Callable] = {
    ('int8', 'int8'): libalgos.take_2d_axis1_int8_int8,
    ('int8', 'int32'): libalgos.take_2d_axis1_int8_int32,
    ('int8', 'int64'): libalgos.take_2d_axis1_int8_int64,
    ('int8', 'float64'): libalgos.take_2d_axis1_int8_float64,
    ('int16', 'int16'): libalgos.take_2d_axis1_int16_int16,
    ('int16', 'int32'): libalgos.take_2d_axis1_int16_int32,
    ('int16', 'int64'): libalgos.take_2d_axis1_int16_int64,
    ('int16', 'float64'): libalgos.take_2d_axis1_int16_float64,
    ('int32', 'int32'): libalgos.take_2d_axis1_int32_int32,
    ('int32', 'int64'): libalgos.take_2d_axis1_int32_int64,
    ('int32', 'float64'): libalgos.take_2d_axis1_int32_float64,
    ('int64', 'int64'): libalgos.take_2d_axis1_int64_int64,
    ('int64', 'float64'): libalgos.take_2d_axis1_int64_float64,
    ('uint8', 'uint8'): libalgos.take_2d_axis1_bool_bool,
    ('uint16', 'uint16'): libalgos.take_2d_axis1_uint16_uint16,
    ('uint32', 'uint32'): libalgos.take_2d_axis1_uint32_uint32,
    ('uint64', 'uint64'): libalgos.take_2d_axis1_uint64_uint64,
    ('float32', 'float32'): libalgos.take_2d_axis1_float32_float32,
    ('float32', 'float64'): libalgos.take_2d_axis1_float32_float64,
    ('float64', 'float64'): libalgos.take_2d_axis1_float64_float64,
    ('object', 'object'): libalgos.take_2d_axis1_object_object,
    ('bool', 'bool'): _view_wrapper(libalgos.take_2d_axis1_bool_bool, np.uint8, np.uint8),
    ('bool', 'object'): _view_wrapper(libalgos.take_2d_axis1_bool_object, np.uint8, None),
    ('datetime64[ns]', 'datetime64[ns]'): _view_wrapper(libalgos.take_2d_axis1_int64_int64, np.int64, np.int64, fill_wrap=np.int64),
    ('timedelta64[ns]', 'timedelta64[ns]'): _view_wrapper(libalgos.take_2d_axis1_int64_int64, np.int64, np.int64, fill_wrap=np.int64)
}

_take_2d_multi_dict: dict[Tuple[str, str], Callable] = {
    ('int8', 'int8'): libalgos.take_2d_multi_int8_int8,
    ('int8', 'int32'): libalgos.take_2d_multi_int8_int32,
    ('int8', 'int64'): libalgos.take_2d_multi_int8_int64,
    ('int8', 'float64'): libalgos.take_2d_multi_int8_float64,
    ('int16', 'int16'): libalgos.take_2d_multi_int16_int16,
    ('int16', 'int32'): libalgos.take_2d_multi_int16_int32,
    ('int16', 'int64'): libalgos.take_2d_multi_int16_int64,
    ('int16', 'float64'): libalgos.take_2d_multi_int16_float64,
    ('int32', 'int32'): libalgos.take_2d_multi_int32_int32,
    ('int32', 'int64'): libalgos.take_2d_multi_int32_int64,
    ('int32', 'float64'): libalgos.take_2d_multi_int32_float64,
    ('int64', 'int64'): libalgos.take_2d_multi_int64_int64,
    ('int64', 'float64'): libalgos.take_2d_multi_int64_float64,
    ('float32', 'float32'): libalgos.take_2d_multi_float32_float32,
    ('float32', 'float64'): libalgos.take_2d_multi_float32_float64,
    ('float64', 'float64'): libalgos.take_2d_multi_float64_float64,
    ('object', 'object'): libalgos.take_2d_multi_object_object,
    ('bool', 'bool'): _view_wrapper(libalgos.take_2d_multi_bool_bool, np.uint8, np.uint8),
    ('bool', 'object'): _view_wrapper(libalgos.take_2d_multi_bool_object, np.uint8, None),
    ('datetime64[ns]', 'datetime64[ns]'): _view_wrapper(libalgos.take_2d_multi_int64_int64, np.int64, np.int64, fill_wrap=np.int64),
    ('timedelta64[ns]', 'timedelta64[ns]'): _view_wrapper(libalgos.take_2d_multi_int64_int64, np.int64, np.int64, fill_wrap=np.int64)
}

def _take_nd_object(arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, axis: int, fill_value: Any, mask_info: Optional[Tuple[np.ndarray, bool]]) -> None:
    if mask_info is not None:
        mask, needs_masking = mask_info
    else:
        mask = indexer == -1
        needs_masking = mask.any()
    if arr.dtype != out.dtype:
        arr = arr.astype(out.dtype)
    if arr.shape[axis] > 0:
        arr.take(indexer, axis=axis, out=out)
    if needs_masking:
        outindexer = [slice(None)] * arr.ndim
        outindexer[axis] = mask
        out[tuple(outindexer)] = fill_value

def _take_2d_multi_object(arr: np.ndarray, indexer: Tuple[np.ndarray, np.ndarray], out: np.ndarray, fill_value: Any, mask_info: Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[bool, bool]]]) -> None:
    row_idx, col_idx = indexer
    if mask_info is not None:
        (row_mask, col_mask), (row_needs, col_needs) = mask_info
    else:
        row_mask = row_idx == -1
        col_mask = col_idx == -1
        row_needs = row_mask.any()
        col_needs = col_mask.any()
    if fill_value is not None:
        if row_needs:
            out[row_mask, :] = fill_value
        if col_needs:
            out[:, col_mask] = fill_value
    for i, u_ in enumerate(row_idx):
        if u_ != -1:
            for j, v in enumerate(col_idx):
                if v != -1:
                    out[i, j] = arr[u_, v]

def _take_preprocess_indexer_and_fill_value(arr: np.ndarray, indexer: np.ndarray, fill_value: Any, allow_fill: bool, mask: Optional[np.ndarray] = None) -> Tuple[np.dtype, Any, Optional[Tuple[np.ndarray, bool]]]:
    mask_info = None
    if not allow_fill:
        dtype, fill_value = (arr.dtype, arr.dtype.type())
        mask_info = (None, False)
    else:
        dtype, fill_value = maybe_promote(arr.dtype, fill_value)
        if dtype != arr.dtype:
            if mask is not None:
                needs_masking = True
            else:
                mask = indexer == -1
                needs_masking = bool(mask.any())
            mask_info = (mask, needs_masking)
            if not needs_masking:
                dtype, fill_value = (arr.dtype, arr.dtype.type())
    return (dtype, fill_value, mask_info)

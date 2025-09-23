"""
Functions for preparing various inputs passed to the DataFrame or Series
constructors before passing them to a BlockManager.
"""
from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, cast
import numpy as np
from numpy import ma
from pandas._config import using_string_dtype
from pandas._libs import lib
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar, dict_compat, maybe_cast_to_datetime, maybe_convert_platform, maybe_infer_to_datetimelike
from pandas.core.dtypes.common import is_1d_only_ea_dtype, is_integer_dtype, is_list_like, is_named_tuple, is_object_dtype, is_scalar
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core import algorithms, common as com
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import array as pd_array, extract_array, range_to_ndarray, sanitize_array
from pandas.core.indexes.api import DatetimeIndex, Index, TimedeltaIndex, default_index, ensure_index, get_objs_combined_axis, maybe_sequence_to_range, union_indexes
from pandas.core.internals.blocks import BlockPlacement, ensure_block_shape, new_block, new_block_2d
from pandas.core.internals.managers import create_block_manager_from_blocks, create_block_manager_from_column_arrays
if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from pandas._typing import ArrayLike, DtypeObj, Manager, npt
    from pandas import DataFrame

def arrays_to_mgr(arrays: Sequence[Any], columns: Any, index: Any, *, dtype: DtypeObj | None = None, verify_integrity: bool = True, consolidate: bool = True) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.

    Needs to handle a lot of exceptional cases.
    """
    if verify_integrity:
        if index is None:
            index = _extract_index(arrays)
        else:
            index = ensure_index(index)
        (arrays, refs) = _homogenize(arrays, index, dtype)
    else:
        index = ensure_index(index)
        arrays = [extract_array(x, extract_numpy=True) for x in arrays]
        refs = [None] * len(arrays)
        for arr in arrays:
            if not isinstance(arr, (np.ndarray, ExtensionArray)) or arr.ndim != 1 or len(arr) != len(index):
                raise ValueError('Arrays must be 1-dimensional np.ndarray or ExtensionArray with length matching len(index)')
    columns = ensure_index(columns)
    if len(columns) != len(arrays):
        raise ValueError('len(arrays) must match len(columns)')
    axes = [columns, index]
    return create_block_manager_from_column_arrays(arrays, axes, consolidate=consolidate, refs=refs)

def rec_array_to_mgr(data: np.recarray | np.ndarray, index: Any, columns: Any, dtype: DtypeObj | None, copy: bool) -> Manager:
    """
    Extract from a masked rec array and create the manager.
    """
    fdata = ma.getdata(data)
    if index is None:
        index = default_index(len(fdata))
    else:
        index = ensure_index(index)
    if columns is not None:
        columns = ensure_index(columns)
    (arrays, arr_columns) = to_arrays(fdata, columns)
    (arrays, arr_columns) = reorder_arrays(arrays, arr_columns, columns, len(index))
    if columns is None:
        columns = arr_columns
    mgr = arrays_to_mgr(arrays, columns, index, dtype=dtype)
    if copy:
        mgr = mgr.copy()
    return mgr

def ndarray_to_mgr(values: ArrayLike, index: Any, columns: Any, dtype: DtypeObj | None, copy: bool) -> Manager:
    infer_object = not isinstance(values, (ABCSeries, Index, ExtensionArray))
    if isinstance(values, ABCSeries):
        if columns is None:
            if values.name is not None:
                columns = Index([values.name])
        if index is None:
            index = values.index
        else:
            values = values.reindex(index)
        if not len(values) and columns is not None and len(columns):
            values = np.empty((0, 1), dtype=object)
    vdtype = getattr(values, 'dtype', None)
    refs = None
    if is_1d_only_ea_dtype(vdtype) or is_1d_only_ea_dtype(dtype):
        if isinstance(values, (np.ndarray, ExtensionArray)) and values.ndim > 1:
            values = [values[:, n] for n in range(values.shape[1])]
        else:
            values = [values]
        if columns is None:
            columns = Index(range(len(values)))
        else:
            columns = ensure_index(columns)
        return arrays_to_mgr(values, columns, index, dtype=dtype)
    elif isinstance(vdtype, ExtensionDtype):
        values = extract_array(values, extract_numpy=True)
        if copy:
            values = values.copy()
        if values.ndim == 1:
            values = values.reshape(-1, 1)
    elif isinstance(values, (ABCSeries, Index)):
        if not copy and (dtype is None or astype_is_view(values.dtype, dtype)):
            refs = values._references
        if copy:
            values = values._values.copy()
        else:
            values = values._values
        values = _ensure_2d(values)
    elif isinstance(values, (np.ndarray, ExtensionArray)):
        if copy and (dtype is None or astype_is_view(values.dtype, dtype)):
            values = np.array(values, copy=True, order='F')
        else:
            values = np.asarray(values)
        values = _ensure_2d(values)
    else:
        values = _prep_ndarraylike(values, copy=copy)
    if dtype is not None and values.dtype != dtype:
        values = sanitize_array(values, None, dtype=dtype, copy=copy, allow_2d=True)
    (index, columns) = _get_axes(values.shape[0], values.shape[1], index=index, columns=columns)
    _check_values_indices_shape_match(values, index, columns)
    values = values.T
    if dtype is None and infer_object and is_object_dtype(values.dtype):
        obj_columns = list(values)
        maybe_datetime = [maybe_infer_to_datetimelike(x) for x in obj_columns]
        if any((x is not y for (x, y) in zip(obj_columns, maybe_datetime))):
            block_values = [new_block_2d(ensure_block_shape(dval, 2), placement=BlockPlacement(n)) for (n, dval) in enumerate(maybe_datetime)]
        else:
            bp = BlockPlacement(slice(len(columns)))
            nb = new_block_2d(values, placement=bp, refs=refs)
            block_values = [nb]
    elif dtype is None and values.dtype.kind == 'U' and using_string_dtype():
        dtype = StringDtype(na_value=np.nan)
        obj_columns = list(values)
        block_values = [new_block(dtype.construct_array_type()._from_sequence(data, dtype=dtype), BlockPlacement(slice(i, i + 1)), ndim=2) for (i, data) in enumerate(obj_columns)]
    else:
        bp = BlockPlacement(slice(len(columns)))
        nb = new_block_2d(values, placement=bp, refs=refs)
        block_values = [nb]
    if len(columns) == 0:
        block_values = []
    return create_block_manager_from_blocks(block_values, [columns, index], verify_integrity=False)

def _check_values_indices_shape_match(values: np.ndarray, index: Index, columns: Index) -> None:
    """
    Check that the shape implied by our axes matches the actual shape of the
    data.
    """
    if values.shape[1] != len(columns) or values.shape[0] != len(index):
        if values.shape[0] == 0 < len(index):
            raise ValueError('Empty data passed with indices specified.')
        passed = values.shape
        implied = (len(index), len(columns))
        raise ValueError(f'Shape of passed values is {passed}, indices imply {implied}')

def dict_to_mgr(data: dict, index: Any, columns: Any, *, dtype: DtypeObj | None = None, copy: bool = True) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.
    Needs to handle a lot of exceptional cases.

    Used in DataFrame.__init__
    """
    arrays: Sequence[Any]
    if columns is not None:
        columns = ensure_index(columns)
        arrays = [np.nan] * len(columns)
        midxs = set()
        data_keys = ensure_index(data.keys())
        data_values = list(data.values())
        for (i, column) in enumerate(columns):
            try:
                idx = data_keys.get_loc(column)
            except KeyError:
                midxs.add(i)
                continue
            array = data_values[idx]
            arrays[i] = array
            if is_scalar(array) and isna(array):
                midxs.add(i)
        if index is None:
            if midxs:
                index = _extract_index([array for (i, array) in enumerate(arrays) if i not in midxs])
            else:
                index = _extract_index(arrays)
        else:
            index = ensure_index(index)
        if midxs and (not is_integer_dtype(dtype)):
            for i in midxs:
                arr = construct_1d_arraylike_from_scalar(arrays[i], len(index), dtype if dtype is not None else np.dtype('object'))
                arrays[i] = arr
    else:
        keys = maybe_sequence_to_range(list(data.keys()))
        columns = Index(keys) if keys else default_index(0)
        arrays = [com.maybe_iterable_to_list(data[k]) for k in keys]
    if copy:
        arrays = [x.copy() if isinstance(x, ExtensionArray) else x.copy(deep=True) if isinstance(x, Index) or (isinstance(x, ABCSeries) and is_1d_only_ea_dtype(x.dtype)) else x for x in arrays]
    return arrays_to_mgr(arrays, columns, index, dtype=dtype, consolidate=copy)

def nested_data_to_arrays(data: Sequence[Any], columns: Any, index: Any, dtype: DtypeObj | None) -> tuple[list[ArrayLike], Index, Index]:
    """
    Convert a single sequence of arrays to multiple arrays.
    """
    if is_named_tuple(data[0]) and columns is None:
        columns = ensure_index(data[0]._fields)
    (arrays, columns) = to_arrays(data, columns, dtype=dtype)
    columns = ensure_index(columns)
    if index is None:
        if isinstance(data[0], ABCSeries):
            index = _get_names_from_index(data)
        else:
            index = default_index(len(data))
    return (arrays, columns, index)

def treat_as_nested(data: Sequence[Any]) -> bool:
    """
    Check if we should use nested_data_to_arrays.
    """
    return len(data) > 0 and is_list_like(data[0]) and (getattr(data[0], 'ndim', 1) == 1) and (not (isinstance(data, ExtensionArray) and data.ndim == 2))

def _prep_ndarraylike(values: Any, copy: bool = True) -> np.ndarray:
    if len(values) == 0:
        return np.empty((0, 0), dtype=object)
    elif isinstance(values, range):
        arr = range_to_ndarray(values)
        return arr[..., np.newaxis]

    def convert(v: Any) -> Any:
        if not is_list_like(v) or isinstance(v, ABCDataFrame):
            return v
        v = extract_array(v, extract_numpy=True)
        res = maybe_convert_platform(v)
        return res
    if is_list_like(values[0]):
        values = np.array([convert(v) for v in values])
    elif isinstance(values[0], np.ndarray) and values[0].ndim == 0:
        values = np.array([convert(v) for v in values])
    else:
        values = convert(values)
    return _ensure_2d(values)

def _ensure_2d(values: np.ndarray) -> np.ndarray:
    """
    Reshape 1D values, raise on anything else other than 2D.
    """
    if values.ndim == 1:
        values = values.reshape((values.shape[0], 1))
    elif values.ndim != 2:
        raise ValueError(f'Must pass 2-d input. shape={values.shape}')
    return values

def _homogenize(data: Sequence[Any], index: Index, dtype: DtypeObj | None) -> tuple[list[ArrayLike], list[Any]]:
    oindex = None
    homogenized = []
    refs: list[Any] = []
    for val in data:
        if isinstance(val, (ABCSeries, Index)):
            if dtype is not None:
                val = val.astype(dtype)
            if isinstance(val, ABCSeries) and val.index is not index:
                val = val.reindex(index)
            refs.append(val._references)
            val = val._values
        else:
            if isinstance(val, dict):
                if oindex is None:
                    oindex = index.astype('O')
                if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
                    val = dict_compat(val)
                else:
                    val = dict(val)
                val = lib.fast_multiget(val, oindex._values, default=np.nan)
            val = sanitize_array(val, index, dtype=dtype, copy=False)
            com.require_length_match(val, index)
            refs.append(None)
        homogenized.append(val)
    return (homogenized, refs)

def _extract_index(data: Sequence[Any]) -> Index:
    """
    Try to infer an Index from the passed data, raise ValueError on failure.
    """
    index: Index
    if len(data) == 0:
        return default_index(0)
    raw_lengths = set()
    indexes: list[list[Hashable] | Index] = []
    have_raw_arrays = False
    have_series = False
    have_dicts = False
    for val in data:
        if isinstance(val, ABCSeries):
            have_series = True
            indexes.append(val.index)
        elif isinstance(val, dict):
            have_dicts = True
            indexes.append(list(val.keys()))
        elif is_list_like(val) and getattr(val, 'ndim', 1) == 1:
            have_raw_arrays = True
            raw_lengths.add(len(val))
        elif isinstance(val, np.ndarray) and val.ndim > 1:
            raise ValueError('Per-column arrays must each be 1-dimensional')
    if not indexes and (not raw_lengths):
        raise ValueError('If using all scalar values, you must pass an index')
    if have_series:
        index = union_indexes(indexes)
    elif have_dicts:
        index = union_indexes(indexes, sort=False)
    if have_raw_arrays:
        if len(raw_lengths) > 1:
            raise ValueError('All arrays must be of the same length')
        if have_dicts:
            raise ValueError('Mixing dicts with non-Series may lead to ambiguous ordering.')
        raw_length = raw_lengths.pop()
        if have_series:
            if raw_length != len(index):
                msg = f'array length {raw_length} does not match index length {len(index)}'
                raise ValueError(msg)
        else:
            index = default_index(raw_length)
    return ensure_index(index)

def reorder_arrays(arrays: list[ArrayLike], arr_columns: Index, columns: Index | None, length: int) -> tuple[list[ArrayLike], Index]:
    """
    Preemptively (cheaply) reindex arrays with new columns.
    """
    if columns is not None:
        if not columns.equals(arr_columns):
            new_arrays: list[ArrayLike] = []
            indexer = arr_columns.get_indexer(columns)
            for (i, k) in enumerate(indexer):
                if k == -1:
                    arr = np.empty(length, dtype=object)
                    arr.fill(np.nan)
                else:
                    arr = arrays[k]
                new_arrays.append(arr)
            arrays = new_arrays
            arr_columns = columns
    return (arrays, arr_columns)

def _get_names_from_index(data: Sequence[Any]) -> Index:
    has_some_name = any((getattr(s, 'name', None) is not None for s in data))
    if not has_some_name:
        return default_index(len(data))
    index: list[Hashable] = list(range(len(data)))
    count = 0
    for (i, s) in enumerate(data):
        n = getattr(s, 'name', None)
        if n is not None:
            index[i] = n
        else:
            index[i] = f'Unnamed {count}'
            count += 1
    return Index(index)

def _get_axes(N: int, K: int, index: Index | None, columns: Index | None) -> tuple[Index, Index]:
    if index is None:
        index = default_index(N)
    else:
        index = ensure_index(index)
    if columns is None:
        columns = default_index(K)
    else:
        columns = ensure_index(columns)
    return (index, columns)

def dataclasses_to_dicts(data: list[Any]) -> list[dict]:
    """
    Converts a list of dataclass instances to a list of dictionaries.

    Parameters
    ----------
    data : List[Type[dataclass]]

    Returns
    --------
    list_dict : List[dict]

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Point:
    ...     x: int

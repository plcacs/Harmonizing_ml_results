from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, List, Tuple, Union
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

def arrays_to_mgr(
    arrays: List[ArrayLike],
    columns: Index,
    index: Index,
    *,
    dtype: DtypeObj | None = None,
    verify_integrity: bool = True,
    consolidate: bool = True
) -> Manager:
    if verify_integrity:
        if index is None:
            index = _extract_index(arrays)
        else:
            index = ensure_index(index)
        arrays, refs = _homogenize(arrays, index, dtype)
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

def rec_array_to_mgr(
    data: np.recarray,
    index: Index | None,
    columns: Index | None,
    dtype: DtypeObj,
    copy: bool
) -> Manager:
    fdata = ma.getdata(data)
    if index is None:
        index = default_index(len(fdata))
    else:
        index = ensure_index(index)
    if columns is not None:
        columns = ensure_index(columns)
    arrays, arr_columns = to_arrays(fdata, columns)
    arrays, arr_columns = reorder_arrays(arrays, arr_columns, columns, len(index))
    if columns is None:
        columns = arr_columns
    mgr = arrays_to_mgr(arrays, columns, index, dtype=dtype)
    if copy:
        mgr = mgr.copy()
    return mgr

def ndarray_to_mgr(
    values: ArrayLike,
    index: Index | None,
    columns: Index | None,
    dtype: DtypeObj,
    copy: bool
) -> Manager:
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
    index, columns = _get_axes(values.shape[0], values.shape[1], index=index, columns=columns)
    _check_values_indices_shape_match(values, index, columns)
    values = values.T
    if dtype is None and infer_object and is_object_dtype(values.dtype):
        obj_columns = list(values)
        maybe_datetime = [maybe_infer_to_datetimelike(x) for x in obj_columns]
        if any((x is not y for x, y in zip(obj_columns, maybe_datetime))):
            block_values = [new_block_2d(ensure_block_shape(dval, 2), placement=BlockPlacement(n)) for n, dval in enumerate(maybe_datetime)]
        else:
            bp = BlockPlacement(slice(len(columns)))
            nb = new_block_2d(values, placement=bp, refs=refs)
            block_values = [nb]
    elif dtype is None and values.dtype.kind == 'U' and using_string_dtype():
        dtype = StringDtype(na_value=np.nan)
        obj_columns = list(values)
        block_values = [new_block(dtype.construct_array_type()._from_sequence(data, dtype=dtype), BlockPlacement(slice(i, i + 1)), ndim=2) for i, data in enumerate(obj_columns)]
    else:
        bp = BlockPlacement(slice(len(columns)))
        nb = new_block_2d(values, placement=bp, refs=refs)
        block_values = [nb]
    if len(columns) == 0:
        block_values = []
    return create_block_manager_from_blocks(block_values, [columns, index], verify_integrity=False)

def _check_values_indices_shape_match(values: np.ndarray, index: Index, columns: Index) -> None:
    if values.shape[1] != len(columns) or values.shape[0] != len(index):
        if values.shape[0] == 0 < len(index):
            raise ValueError('Empty data passed with indices specified.')
        passed = values.shape
        implied = (len(index), len(columns))
        raise ValueError(f'Shape of passed values is {passed}, indices imply {implied}')

def dict_to_mgr(
    data: dict[Hashable, Any],
    index: Index | None,
    columns: Index | None,
    *,
    dtype: DtypeObj | None = None,
    copy: bool = True
) -> Manager:
    if columns is not None:
        columns = ensure_index(columns)
        arrays = [np.nan] * len(columns)
        midxs = set()
        data_keys = ensure_index(data.keys())
        data_values = list(data.values())
        for i, column in enumerate(columns):
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
                index = _extract_index([array for i, array in enumerate(arrays) if i not in midxs])
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

def nested_data_to_arrays(
    data: Sequence,
    columns: Index | None,
    index: Index | None,
    dtype: DtypeObj
) -> Tuple[List[ArrayLike], Index, Index]:
    if is_named_tuple(data[0]) and columns is None:
        columns = ensure_index(data[0]._fields)
    arrays, columns = to_arrays(data, columns, dtype=dtype)
    columns = ensure_index(columns)
    if index is None:
        if isinstance(data[0], ABCSeries):
            index = _get_names_from_index(data)
        else:
            index = default_index(len(data))
    return (arrays, columns, index)

def treat_as_nested(data: Sequence) -> bool:
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
    if values.ndim == 1:
        values = values.reshape((values.shape[0], 1))
    elif values.ndim != 2:
        raise ValueError(f'Must pass 2-d input. shape={values.shape}')
    return values

def _homogenize(
    data: Sequence,
    index: Index,
    dtype: DtypeObj | None
) -> Tuple[List[np.ndarray], List[Any]]:
    oindex = None
    homogenized = []
    refs = []
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

def _extract_index(data: Sequence) -> Index:
    if len(data) == 0:
        return default_index(0)
    raw_lengths = set()
    indexes = []
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

def reorder_arrays(
    arrays: List[ArrayLike],
    arr_columns: Index,
    columns: Index,
    length: int
) -> Tuple[List[ArrayLike], Index]:
    if columns is not None:
        if not columns.equals(arr_columns):
            new_arrays = []
            indexer = arr_columns.get_indexer(columns)
            for i, k in enumerate(indexer):
                if k == -1:
                    arr = np.empty(length, dtype=object)
                    arr.fill(np.nan)
                else:
                    arr = arrays[k]
                new_arrays.append(arr)
            arrays = new_arrays
            arr_columns = columns
    return (arrays, arr_columns)

def _get_names_from_index(data: Sequence) -> Index:
    has_some_name = any((getattr(s, 'name', None) is not None for s in data))
    if not has_some_name:
        return default_index(len(data))
    index = list(range(len(data)))
    count = 0
    for i, s in enumerate(data):
        n = getattr(s, 'name', None)
        if n is not None:
            index[i] = n
        else:
            index[i] = f'Unnamed {count}'
            count += 1
    return Index(index)

def _get_axes(N: int, K: int, index: Index | None, columns: Index | None) -> Tuple[Index, Index]:
    if index is None:
        index = default_index(N)
    else:
        index = ensure_index(index)
    if columns is None:
        columns = default_index(K)
    else:
        columns = ensure_index(columns)
    return (index, columns)

def dataclasses_to_dicts(data: List[Any]) -> List[dict]:
    from dataclasses import asdict
    return list(map(asdict, data))

def to_arrays(
    data: Sequence,
    columns: Index | None,
    dtype: DtypeObj | None = None
) -> Tuple[List[ArrayLike], Index]:
    if not len(data):
        if isinstance(data, np.ndarray):
            if data.dtype.names is not None:
                columns = ensure_index(data.dtype.names)
                arrays = [data[name] for name in columns]
                if len(data) == 0:
                    for i, arr in enumerate(arrays):
                        if arr.ndim == 2:
                            arrays[i] = arr[:, 0]
                return (arrays, columns)
        return ([], ensure_index([]))
    elif isinstance(data, np.ndarray) and data.dtype.names is not None:
        if columns is None:
            columns = Index(data.dtype.names)
        arrays = [data[k] for k in columns]
        return (arrays, columns)
    if isinstance(data[0], (list, tuple)):
        arr = _list_to_arrays(data)
    elif isinstance(data[0], abc.Mapping):
        arr, columns = _list_of_dict_to_arrays(data, columns)
    elif isinstance(data[0], ABCSeries):
        arr, columns = _list_of_series_to_arrays(data, columns)
    else:
        data = [tuple(x) for x in data]
        arr = _list_to_arrays(data)
    content, columns = _finalize_columns_and_data(arr, columns, dtype)
    return (content, columns)

def _list_to_arrays(data: Sequence) -> np.ndarray:
    if isinstance(data[0], tuple):
        content = lib.to_object_array_tuples(data)
    else:
        content = lib.to_object_array(data)
    return content

def _list_of_series_to_arrays(data: Sequence, columns: Index | None) -> Tuple[np.ndarray, Index]:
    if columns is None:
        pass_data = [x for x in data if isinstance(x, (ABCSeries, ABCDataFrame))]
        columns = get_objs_combined_axis(pass_data, sort=False)
    indexer_cache = {}
    aligned_values = []
    for s in data:
        index = getattr(s, 'index', None)
        if index is None:
            index = default_index(len(s))
        if id(index) in indexer_cache:
            indexer = indexer_cache[id(index)]
        else:
            indexer = indexer_cache[id(index)] = index.get_indexer(columns)
        values = extract_array(s, extract_numpy=True)
        aligned_values.append(algorithms.take_nd(values, indexer))
    content = np.vstack(aligned_values)
    return (content, columns)

def _list_of_dict_to_arrays(data: Sequence, columns: Index | None) -> Tuple[np.ndarray, Index]:
    if columns is None:
        gen = (list(x.keys()) for x in data)
        sort = not any((isinstance(d, dict) for d in data))
        pre_cols = lib.fast_unique_multiple_list_gen(gen, sort=sort)
        columns = ensure_index(pre_cols)
    data = [d if type(d) is dict else dict(d) for d in data]
    content = lib.dicts_to_array(data, list(columns))
    return (content, columns)

def _finalize_columns_and_data(
    content: np.ndarray,
    columns: Index | None,
    dtype: DtypeObj | None
) -> Tuple[List[ArrayLike], Index]:
    contents = list(content.T)
    try:
        columns = _validate_or_indexify_columns(contents, columns)
    except AssertionError as err:
        raise ValueError(err) from err
    if len(contents) and contents[0].dtype == np.object_:
        contents = convert_object_array(contents, dtype=dtype)
    return (contents, columns)

def _validate_or_indexify_columns(content: List[np.ndarray], columns: Index | None) -> Index:
    if columns is None:
        columns = default_index(len(content))
    else:
        is_mi_list = isinstance(columns, list) and all((isinstance(col, list) for col in columns))
        if not is_mi_list and len(columns) != len(content):
            raise AssertionError(f'{len(columns)} columns passed, passed data had {len(content)} columns')
        if is_mi_list:
            if len({len(col) for col in columns}) > 1:
                raise ValueError('Length of columns passed for MultiIndex columns is different')
            if columns and len(columns[0]) != len(content):
                raise ValueError(f'{len(columns[0])} columns passed, passed data had {len(content)} columns')
    return columns

def convert_object_array(
    content: List[np.ndarray],
    dtype: DtypeObj | None,
    dtype_backend: str = 'numpy',
    coerce_float: bool = False
) -> List[ArrayLike]:
    def convert(arr: np.ndarray) -> ArrayLike:
        if dtype != np.dtype('O'):
            arr = lib.maybe_convert_objects(arr, try_float=coerce_float, convert_to_nullable_dtype=dtype_backend != 'numpy')
            if dtype is None:
                if arr.dtype == np.dtype('O'):
                    convert_to_nullable_dtype = dtype_backend != 'numpy'
                    arr = maybe_infer_to_datetimelike(arr, convert_to_nullable_dtype)
                    if convert_to_nullable_dtype and arr.dtype == np.dtype('O'):
                        new_dtype = StringDtype()
                        arr_cls = new_dtype.construct_array_type()
                        arr = arr_cls._from_sequence(arr, dtype=new_dtype)
                elif dtype_backend != 'numpy' and isinstance(arr, np.ndarray):
                    if arr.dtype.kind in 'iufb':
                        arr = pd_array(arr, copy=False)
            elif isinstance(dtype, ExtensionDtype):
                cls = dtype.construct_array_type()
                arr = cls._from_sequence(arr, dtype=dtype, copy=False)
            elif dtype.kind in 'mM':
                arr = maybe_cast_to_datetime(arr, dtype)
        return arr
    arrays = [convert(arr) for arr in content]
    return arrays

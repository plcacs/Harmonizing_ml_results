from __future__ import annotations
from collections import abc
import types
from typing import TYPE_CHECKING, Literal, cast, overload, Union, Sequence, Optional, List, Tuple
import warnings
import numpy as np
from pandas._libs import lib
from pandas.util._decorators import set_module
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_bool, is_scalar
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core.arrays.categorical import factorize_from_iterable, factorize_from_iterables
import pandas.core.common as com
from pandas.core.indexes.api import Index, MultiIndex, all_indexes_same, default_index, ensure_index, get_objs_combined_axis, get_unanimous_names
from pandas.core.internals import concatenate_managers

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping
    from pandas._typing import Axis, AxisInt, HashableT
    from pandas import DataFrame, Series

@overload
def concat(
    objs: Union[Iterable[Union[Series, DataFrame]], Mapping[Hashable, Union[Series, DataFrame]]],
    *,
    axis: Axis = ...,
    join: Literal['inner', 'outer'] = ...,
    ignore_index: bool = ...,
    keys: Optional[Sequence[Hashable]] = ...,
    levels: Optional[Sequence[Sequence[Hashable]]] = ...,
    names: Optional[Sequence[Hashable]] = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool = ...
) -> Union[Series, DataFrame]:
    ...

@set_module('pandas')
def concat(
    objs: Union[Iterable[Union[Series, DataFrame]], Mapping[Hashable, Union[Series, DataFrame]]],
    *,
    axis: Axis = 0,
    join: Literal['inner', 'outer'] = 'outer',
    ignore_index: bool = False,
    keys: Optional[Sequence[Hashable]] = None,
    levels: Optional[Sequence[Sequence[Hashable]]] = None,
    names: Optional[Sequence[Hashable]] = None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: bool = lib.no_default
) -> Union[Series, DataFrame]:
    if ignore_index and keys is not None:
        raise ValueError(f'Cannot set ignore_index={ignore_index!r} and specify keys. Either should be used.')
    if copy is not lib.no_default:
        warnings.warn('The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 which utilizes a lazy copy mechanism that defers copies until necessary. Use .copy() to make an eager copy if necessary.', DeprecationWarning, stacklevel=find_stack_level())
    if join == 'outer':
        intersect = False
    elif join == 'inner':
        intersect = True
    else:
        raise ValueError('Only can inner (intersect) or outer (union) join the other axis')
    if not is_bool(sort):
        raise ValueError(f"The 'sort' keyword only accepts boolean values; {sort} was passed.")
    sort = bool(sort)
    objs, keys, ndims = _clean_keys_and_objs(objs, keys)
    sample, objs = _get_sample_object(objs, ndims, keys, names, levels, intersect)
    if sample.ndim == 1:
        from pandas import DataFrame
        bm_axis = DataFrame._get_axis_number(axis)
        is_frame = False
        is_series = True
    else:
        bm_axis = sample._get_axis_number(axis)
        is_frame = True
        is_series = False
        bm_axis = sample._get_block_manager_axis(bm_axis)
    if len(ndims) > 1:
        objs = _sanitize_mixed_ndim(objs, sample, ignore_index, bm_axis)
    axis = 1 - bm_axis if is_frame else 0
    names = names or getattr(keys, 'names', None)
    return _get_result(objs, is_series, bm_axis, ignore_index, intersect, sort, keys, levels, verify_integrity, names, axis)

def _sanitize_mixed_ndim(
    objs: List[Union[Series, DataFrame]],
    sample: Union[Series, DataFrame],
    ignore_index: bool,
    axis: int
) -> List[Union[Series, DataFrame]]:
    new_objs = []
    current_column = 0
    max_ndim = sample.ndim
    for obj in objs:
        ndim = obj.ndim
        if ndim == max_ndim:
            pass
        elif ndim != max_ndim - 1:
            raise ValueError('cannot concatenate unaligned mixed dimensional NDFrame objects')
        else:
            name = getattr(obj, 'name', None)
            if ignore_index or name is None:
                if axis == 1:
                    name = 0
                else:
                    name = current_column
                    current_column += 1
                obj = sample._constructor(obj, copy=False)
                if isinstance(obj, ABCDataFrame):
                    obj.columns = range(name, name + 1, 1)
            else:
                obj = sample._constructor({name: obj}, copy=False)
        new_objs.append(obj)
    return new_objs

def _get_result(
    objs: List[Union[Series, DataFrame]],
    is_series: bool,
    bm_axis: int,
    ignore_index: bool,
    intersect: bool,
    sort: bool,
    keys: Optional[Sequence[Hashable]],
    levels: Optional[Sequence[Sequence[Hashable]]],
    verify_integrity: bool,
    names: Optional[Sequence[Hashable]],
    axis: int
) -> Union[Series, DataFrame]:
    if is_series:
        sample = cast('Series', objs[0])
        if bm_axis == 0:
            name = com.consensus_name_attr(objs)
            cons = sample._constructor
            arrs = [ser._values for ser in objs]
            res = concat_compat(arrs, axis=0)
            if ignore_index:
                new_index = default_index(len(res))
            else:
                new_index = _get_concat_axis_series(objs, ignore_index, bm_axis, keys, levels, verify_integrity, names)
            mgr = type(sample._mgr).from_array(res, index=new_index)
            result = sample._constructor_from_mgr(mgr, axes=mgr.axes)
            result._name = name
            return result.__finalize__(types.SimpleNamespace(objs=objs), method='concat')
        else:
            data = dict(enumerate(objs))
            cons = sample._constructor_expanddim
            index = get_objs_combined_axis(objs, axis=objs[0]._get_block_manager_axis(0), intersect=intersect, sort=sort)
            columns = _get_concat_axis_series(objs, ignore_index, bm_axis, keys, levels, verify_integrity, names)
            df = cons(data, index=index, copy=False)
            df.columns = columns
            return df.__finalize__(types.SimpleNamespace(objs=objs), method='concat')
    else:
        sample = cast('DataFrame', objs[0])
        mgrs_indexers = []
        result_axes = new_axes(objs, bm_axis, intersect, sort, keys, names, axis, levels, verify_integrity, ignore_index)
        for obj in objs:
            indexers = {}
            for ax, new_labels in enumerate(result_axes):
                if ax == bm_axis:
                    continue
                obj_labels = obj.axes[1 - ax]
                if not new_labels.equals(obj_labels):
                    indexers[ax] = obj_labels.get_indexer(new_labels)
            mgrs_indexers.append((obj._mgr, indexers))
        new_data = concatenate_managers(mgrs_indexers, result_axes, concat_axis=bm_axis, copy=False)
        out = sample._constructor_from_mgr(new_data, axes=new_data.axes)
        return out.__finalize__(types.SimpleNamespace(objs=objs), method='concat')

def new_axes(
    objs: List[Union[Series, DataFrame]],
    bm_axis: int,
    intersect: bool,
    sort: bool,
    keys: Optional[Sequence[Hashable]],
    names: Optional[Sequence[Hashable]],
    axis: int,
    levels: Optional[Sequence[Sequence[Hashable]]],
    verify_integrity: bool,
    ignore_index: bool
) -> List[Index]:
    return [
        _get_concat_axis_dataframe(objs, axis, ignore_index, keys, names, levels, verify_integrity) if i == bm_axis
        else get_objs_combined_axis(objs, axis=objs[0]._get_block_manager_axis(i), intersect=intersect, sort=sort)
        for i in range(2)
    ]

def _get_concat_axis_series(
    objs: List[Series],
    ignore_index: bool,
    bm_axis: int,
    keys: Optional[Sequence[Hashable]],
    levels: Optional[Sequence[Sequence[Hashable]]],
    verify_integrity: bool,
    names: Optional[Sequence[Hashable]]
) -> Index:
    if ignore_index:
        return default_index(len(objs))
    elif bm_axis == 0:
        indexes = [x.index for x in objs]
        if keys is None:
            if levels is not None:
                raise ValueError('levels supported only when keys is not None')
            concat_axis = _concat_indexes(indexes)
        else:
            concat_axis = _make_concat_multiindex(indexes, keys, levels, names)
        if verify_integrity and (not concat_axis.is_unique):
            overlap = concat_axis[concat_axis.duplicated()].unique()
            raise ValueError(f'Indexes have overlapping values: {overlap}')
        return concat_axis
    elif keys is None:
        result_names = [None] * len(objs)
        num = 0
        has_names = False
        for i, x in enumerate(objs):
            if x.ndim != 1:
                raise TypeError(f"Cannot concatenate type 'Series' with object of type '{type(x).__name__}'")
            if x.name is not None:
                result_names[i] = x.name
                has_names = True
            else:
                result_names[i] = num
                num += 1
        if has_names:
            return Index(result_names)
        else:
            return default_index(len(objs))
    else:
        return ensure_index(keys).set_names(names)

def _get_concat_axis_dataframe(
    objs: List[DataFrame],
    axis: int,
    ignore_index: bool,
    keys: Optional[Sequence[Hashable]],
    names: Optional[Sequence[Hashable]],
    levels: Optional[Sequence[Sequence[Hashable]]],
    verify_integrity: bool
) -> Index:
    indexes_gen = (x.axes[axis] for x in objs)
    if ignore_index:
        return default_index(sum((len(i) for i in indexes_gen)))
    else:
        indexes = list(indexes_gen)
    if keys is None:
        if levels is not None:
            raise ValueError('levels supported only when keys is not None')
        concat_axis = _concat_indexes(indexes)
    else:
        concat_axis = _make_concat_multiindex(indexes, keys, levels, names)
    if verify_integrity and (not concat_axis.is_unique):
        overlap = concat_axis[concat_axis.duplicated()].unique()
        raise ValueError(f'Indexes have overlapping values: {overlap}')
    return concat_axis

def _clean_keys_and_objs(
    objs: Union[Iterable[Union[Series, DataFrame]], Mapping[Hashable, Union[Series, DataFrame]]],
    keys: Optional[Sequence[Hashable]]
) -> Tuple[List[Union[Series, DataFrame]], Optional[Index], set[int]]:
    if isinstance(objs, abc.Mapping):
        if keys is None:
            keys = objs.keys()
        objs = [objs[k] for k in keys]
    elif isinstance(objs, (ABCSeries, ABCDataFrame)) or is_scalar(objs):
        raise TypeError(f'first argument must be an iterable of pandas objects, you passed an object of type "{type(objs).__name__}"')
    elif not isinstance(objs, abc.Sized):
        objs = list(objs)
    if len(objs) == 0:
        raise ValueError('No objects to concatenate')
    if keys is not None:
        if not isinstance(keys, Index):
            keys = Index(keys)
        if len(keys) != len(objs):
            raise ValueError(f'The length of the keys ({len(keys)}) must match the length of the objects to concatenate ({len(objs)})')
    key_indices = []
    clean_objs = []
    ndims = set()
    for i, obj in enumerate(objs):
        if obj is None:
            continue
        elif isinstance(obj, (ABCSeries, ABCDataFrame)):
            key_indices.append(i)
            clean_objs.append(obj)
            ndims.add(obj.ndim)
        else:
            msg = f"cannot concatenate object of type '{type(obj)}'; only Series and DataFrame objs are valid"
            raise TypeError(msg)
    if keys is not None and len(key_indices) < len(keys):
        keys = keys.take(key_indices)
    if len(clean_objs) == 0:
        raise ValueError('All objects passed were None')
    return (clean_objs, keys, ndims)

def _get_sample_object(
    objs: List[Union[Series, DataFrame]],
    ndims: set[int],
    keys: Optional[Sequence[Hashable]],
    names: Optional[Sequence[Hashable]],
    levels: Optional[Sequence[Sequence[Hashable]]],
    intersect: bool
) -> Tuple[Union[Series, DataFrame], List[Union[Series, DataFrame]]]:
    if len(ndims) > 1:
        max_ndim = max(ndims)
        for obj in objs:
            if obj.ndim == max_ndim and sum(obj.shape):
                return (obj, objs)
    elif keys is None and names is None and (levels is None) and (not intersect):
        if ndims.pop() == 2:
            non_empties = [obj for obj in objs if sum(obj.shape)]
        else:
            non_empties = objs
        if len(non_empties):
            return (non_empties[0], non_empties)
    return (objs[0], objs)

def _concat_indexes(indexes: List[Index]) -> Index:
    return indexes[0].append(indexes[1:])

def validate_unique_levels(levels: Sequence[Index]) -> None:
    for level in levels:
        if not level.is_unique:
            raise ValueError(f'Level values not unique: {level.tolist()}')

def _make_concat_multiindex(
    indexes: List[Index],
    keys: Sequence[Hashable],
    levels: Optional[Sequence[Sequence[Hashable]]] = None,
    names: Optional[Sequence[Hashable]] = None
) -> MultiIndex:
    if levels is None and isinstance(keys[0], tuple) or (levels is not None and len(levels) > 1):
        zipped = list(zip(*keys))
        if names is None:
            names = [None] * len(zipped)
        if levels is None:
            _, levels = factorize_from_iterables(zipped)
        else:
            levels = [ensure_index(x) for x in levels]
            validate_unique_levels(levels)
    else:
        zipped = [keys]
        if names is None:
            names = [None]
        if levels is None:
            levels = [ensure_index(keys).unique()]
        else:
            levels = [ensure_index(x) for x in levels]
            validate_unique_levels(levels)
    if not all_indexes_same(indexes):
        codes_list = []
        for hlevel, level in zip(zipped, levels):
            to_concat = []
            if isinstance(hlevel, Index) and hlevel.equals(level):
                lens = [len(idx) for idx in indexes]
                codes_list.append(np.repeat(np.arange(len(hlevel)), lens))
            else:
                for key, index in zip(hlevel, indexes):
                    mask = isna(level) & isna(key) | (level == key)
                    if not mask.any():
                        raise ValueError(f'Key {key} not in level {level}')
                    i = np.nonzero(mask)[0][0]
                    to_concat.append(np.repeat(i, len(index)))
                codes_list.append(np.concatenate(to_concat))
        concat_index = _concat_indexes(indexes)
        if isinstance(concat_index, MultiIndex):
            levels.extend(concat_index.levels)
            codes_list.extend(concat_index.codes)
        else:
            codes, categories = factorize_from_iterable(concat_index)
            levels.append(categories)
            codes_list.append(codes)
        if len(names) == len(levels):
            names = list(names)
        else:
            if not len({idx.nlevels for idx in indexes}) == 1:
                raise AssertionError('Cannot concat indices that do not have the same number of levels')
            names = list(names) + list(get_unanimous_names(*indexes))
        return MultiIndex(levels=levels, codes=codes_list, names=names, verify_integrity=False)
    new_index = indexes[0]
    n = len(new_index)
    kpieces = len(indexes)
    new_names = list(names)
    new_levels = list(levels)
    new_codes = []
    for hlevel, level in zip(zipped, levels):
        hlevel_index = ensure_index(hlevel)
        mapped = level.get_indexer(hlevel_index)
        mask = mapped == -1
        if mask.any():
            raise ValueError(f'Values not found in passed level: {hlevel_index[mask]!s}')
        new_codes.append(np.repeat(mapped, n))
    if isinstance(new_index, MultiIndex):
        new_levels.extend(new_index.levels)
        new_codes.extend((np.tile(lab, kpieces) for lab in new_index.codes))
    else:
        new_levels.append(new_index.unique())
        single_codes = new_index.unique().get_indexer(new_index)
        new_codes.append(np.tile(single_codes, kpieces))
    if len(new_names) < len(new_levels):
        new_names.extend(new_index.names)
    return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)

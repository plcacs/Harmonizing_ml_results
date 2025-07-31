#!/usr/bin/env python3
"""miscellaneous sorting / groupby utilities"""
from __future__ import annotations
import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from typing import cast
import numpy as np
from numpy.typing import NDArray
from pandas._libs import algos, hashtable, lib
from pandas._libs.hashtable import unique_label_indices
from pandas.core.dtypes.common import ensure_int64, ensure_platform_int
from pandas.core.dtypes.generic import ABCMultiIndex, ABCRangeIndex
from pandas.core.dtypes.missing import isna
from pandas.core.construction import extract_array
if False:  # TYPE_CHECKING
    from collections.abc import Callable as Cb
    from pandas._typing import ArrayLike, AxisInt, IndexKeyFunc, Level, NaPosition, Shape, SortKind, npt
    from pandas import MultiIndex, Series
    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.base import Index

IndexType = Any  # placeholder for pandas.core.indexes.base.Index
MultiIndexType = Any  # placeholder for pandas.core.indexes.multi.MultiIndex

def get_indexer_indexer(
    target: IndexType,
    level: Optional[Union[int, str, List[int], List[str]]] = None,
    ascending: Union[bool, List[bool]] = True,
    kind: str = "quicksort",
    na_position: Literal["first", "last"] = "last",
    sort_remaining: bool = True,
    key: Optional[Callable[[Any], Any]] = None
) -> Optional[NDArray[np.intp]]:
    """
    Helper method that return the indexer according to input parameters for
    the sort_index method of DataFrame and Series.
    """
    target = ensure_key_mapped(target, key, levels=level)
    target = target._sort_levels_monotonic()
    if level is not None:
        _, indexer = target.sortlevel(
            level, ascending=ascending, sort_remaining=sort_remaining, na_position=na_position
        )
    elif np.all(ascending) and target.is_monotonic_increasing or (not np.any(ascending) and target.is_monotonic_decreasing):
        return None
    elif isinstance(target, ABCMultiIndex):
        codes = [lev.codes for lev in target._get_codes_for_sorting()]
        indexer = lexsort_indexer(codes, orders=ascending, na_position=na_position, codes_given=True)
    else:
        indexer = nargsort(target, kind=kind, ascending=cast(bool, ascending), na_position=na_position)
    return indexer

def get_group_index(
    labels: Sequence[NDArray[Any]],
    shape: Tuple[int, ...],
    sort: bool,
    xnull: bool
) -> NDArray[np.int64]:
    """
    For the particular label_list, gets the offsets into the hypothetical list
    representing the totally ordered cartesian product of all possible label
    combinations, *as long as* this space fits within int64 bounds;
    otherwise, though group indices identify unique combinations of
    labels, they cannot be deconstructed.
    """
    def _int64_cut_off(shape: Tuple[int, ...]) -> int:
        acc = 1
        for i, mul in enumerate(shape):
            acc *= int(mul)
            if not acc < lib.i8max:
                return i
        return len(shape)

    def maybe_lift(lab: NDArray[Any], size: int) -> Tuple[NDArray[Any], int]:
        return (lab + 1, size + 1) if (lab == -1).any() else (lab, size)
    labels = [ensure_int64(x) for x in labels]
    lshape = list(shape)
    if not xnull:
        for i, (lab, size) in enumerate(zip(labels, shape)):
            labels[i], lshape[i] = maybe_lift(lab, size)
    while True:
        nlev = _int64_cut_off(tuple(lshape))
        stride = np.prod(lshape[1:nlev], dtype='i8')
        out = stride * labels[0].astype('i8', subok=False, copy=False)
        for i in range(1, nlev):
            if lshape[i] == 0:
                stride = np.int64(0)
            else:
                stride //= lshape[i]
            out += labels[i] * stride
        if xnull:
            mask = labels[0] == -1
            for lab in labels[1:nlev]:
                mask |= lab == -1
            out[mask] = -1
        if nlev == len(lshape):
            break
        comp_ids, obs_ids = compress_group_index(out, sort=sort)
        labels = [comp_ids] + list(labels[nlev:])
        lshape = [len(obs_ids)] + lshape[nlev:]
    return out

def get_compressed_ids(
    labels: List[NDArray[Any]],
    sizes: Tuple[int, ...]
) -> Tuple[NDArray[np.intp], NDArray[np.int64]]:
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).
    """
    ids = get_group_index(labels, sizes, sort=True, xnull=False)
    return compress_group_index(ids, sort=True)

def is_int64_overflow_possible(shape: Sequence[int]) -> bool:
    the_prod = 1
    for x in shape:
        the_prod *= int(x)
    return the_prod >= lib.i8max

def _decons_group_index(
    comp_labels: NDArray[np.intp],
    shape: Tuple[int, ...]
) -> List[NDArray[np.int64]]:
    if is_int64_overflow_possible(shape):
        raise ValueError('cannot deconstruct factorized group indices!')
    label_list: List[NDArray[np.int64]] = []
    factor = 1
    y = np.array(0)
    x = comp_labels
    for i in reversed(range(len(shape))):
        labels = (x - y) % (factor * shape[i]) // factor
        np.putmask(labels, comp_labels < 0, -1)
        label_list.append(labels)
        y = labels * factor
        factor *= shape[i]
    return label_list[::-1]

def decons_obs_group_ids(
    comp_ids: NDArray[np.intp],
    obs_ids: NDArray[np.intp],
    shape: Tuple[int, ...],
    labels: Sequence[NDArray[Any]],
    xnull: bool
) -> List[NDArray[np.intp]]:
    """
    Reconstruct labels from observed group ids.
    """
    if not xnull:
        lift = np.fromiter(((a == -1).any() for a in labels), dtype=np.intp)
        arr_shape = np.asarray(shape, dtype=np.intp) + lift
        shape = tuple(arr_shape)
    if not is_int64_overflow_possible(shape):
        out = _decons_group_index(obs_ids, shape)
        return out if xnull or not lift.any() else [x - y for x, y in zip(out, lift)]
    indexer = unique_label_indices(comp_ids)
    return [lab[indexer].astype(np.intp, subok=False, copy=True) for lab in labels]

def lexsort_indexer(
    keys: Sequence[Any],
    orders: Optional[Union[bool, Sequence[bool]]] = None,
    na_position: Literal["first", "last"] = "last",
    key: Optional[Callable[[Any], Any]] = None,
    codes_given: bool = False
) -> NDArray[np.intp]:
    """
    Performs lexical sorting on a set of keys
    """
    from pandas.core.arrays import Categorical
    if na_position not in ['last', 'first']:
        raise ValueError(f'invalid na_position: {na_position}')
    if isinstance(orders, bool):
        orders = itertools.repeat(orders, len(keys))
    elif orders is None:
        orders = itertools.repeat(True, len(keys))
    else:
        orders = reversed(orders)
    labels: List[NDArray[Any]] = []
    for k, order in zip(reversed(keys), orders):
        k = ensure_key_mapped(k, key)
        if codes_given:
            codes = cast(NDArray[Any], k)
            n = codes.max() + 1 if len(codes) else 0
        else:
            cat = Categorical(k, ordered=True)
            codes = cat.codes
            n = len(cat.categories)
        mask = codes == -1
        if na_position == 'last' and mask.any():
            codes = np.where(mask, n, codes)
        if not order:
            codes = np.where(mask, codes, n - codes - 1)
        labels.append(codes)
    return np.lexsort(labels)

def nargsort(
    items: Any,
    kind: str = "quicksort",
    ascending: bool = True,
    na_position: Literal["first", "last"] = "last",
    key: Optional[Callable[[Any], Any]] = None,
    mask: Optional[NDArray[bool]] = None
) -> NDArray[np.intp]:
    """
    Intended to be a drop-in replacement for np.argsort which handles NaNs.
    """
    if key is not None:
        items = ensure_key_mapped(items, key)
        return nargsort(items, kind=kind, ascending=ascending, na_position=na_position, key=None, mask=mask)
    if isinstance(items, ABCRangeIndex):
        return items.argsort(ascending=ascending)
    elif not isinstance(items, ABCMultiIndex):
        items = extract_array(items)
    else:
        raise TypeError('nargsort does not support MultiIndex. Use index.sort_values instead.')
    if mask is None:
        mask = np.asarray(isna(items))
    if not isinstance(items, np.ndarray):
        return items.argsort(ascending=ascending, kind=kind, na_position=na_position)
    idx = np.arange(len(items))
    non_nans = items[~mask]
    non_nan_idx = idx[~mask]
    nan_idx = np.nonzero(mask)[0]
    if not ascending:
        non_nans = non_nans[::-1]
        non_nan_idx = non_nan_idx[::-1]
    indexer = non_nan_idx[non_nans.argsort(kind=kind)]
    if not ascending:
        indexer = indexer[::-1]
    if na_position == 'last':
        indexer = np.concatenate([indexer, nan_idx])
    elif na_position == 'first':
        indexer = np.concatenate([nan_idx, indexer])
    else:
        raise ValueError(f'invalid na_position: {na_position}')
    return ensure_platform_int(indexer)

def nargminmax(
    values: Any,
    method: Literal["argmax", "argmin"],
    axis: int = 0
) -> Union[int, NDArray[np.intp]]:
    """
    Implementation of np.argmin/argmax but for ExtensionArray and which
    handles missing values.
    """
    assert method in {'argmax', 'argmin'}
    func = np.argmax if method == 'argmax' else np.argmin
    mask = np.asarray(isna(values))
    arr_values = values._values_for_argsort()
    if arr_values.ndim > 1:
        if mask.any():
            if axis == 1:
                zipped = zip(arr_values, mask)
            else:
                zipped = zip(arr_values.T, mask.T)
            return np.array([_nanargminmax(v, m, func) for v, m in zipped])
        return func(arr_values, axis=axis)
    return _nanargminmax(arr_values, mask, func)

def _nanargminmax(
    values: NDArray[Any],
    mask: NDArray[bool],
    func: Callable[[NDArray[Any]], int]
) -> int:
    """
    See nanargminmax.__doc__.
    """
    idx = np.arange(values.shape[0])
    non_nans = values[~mask]
    non_nan_idx = idx[~mask]
    return non_nan_idx[func(non_nans)]

def _ensure_key_mapped_multiindex(
    index: MultiIndexType,
    key: Callable[[Any], Any],
    level: Optional[Union[int, str, Sequence[Union[int, str]]]] = None
) -> MultiIndexType:
    """
    Returns a new MultiIndex in which key has been applied
    to all levels specified in level (or all levels if level
    is None).
    """
    if level is not None:
        if isinstance(level, (str, int)):
            level_iter = [level]
        else:
            level_iter = level
        sort_levels = {index._get_level_number(lev) for lev in level_iter}
    else:
        sort_levels = range(index.nlevels)
    mapped = [
        ensure_key_mapped(index._get_level_values(lvl), key)
        if lvl in sort_levels else index._get_level_values(lvl)
        for lvl in range(index.nlevels)
    ]
    return type(index).from_arrays(mapped)

T = Any
def ensure_key_mapped(
    values: T,
    key: Optional[Callable[[Any], Any]],
    levels: Optional[Union[int, str, Sequence[Union[int, str]]]] = None
) -> T:
    """
    Applies a callable key function to the values and checks
    that the resulting value has the same shape.
    """
    from pandas.core.indexes.api import Index
    if not key:
        return values
    if isinstance(values, ABCMultiIndex):
        return _ensure_key_mapped_multiindex(values, key, level=levels)
    result = key(values.copy())
    if len(result) != len(values):
        raise ValueError('User-provided `key` function must not change the shape of the array.')
    try:
        if isinstance(values, Index):
            result = Index(result, tupleize_cols=False)
        else:
            type_of_values = type(values)
            result = type_of_values(result)
    except TypeError as err:
        raise TypeError(f'User-provided `key` function returned an invalid type {type(result)}             which could not be converted to {type(values)}.') from err
    return result

def get_indexer_dict(
    label_list: List[NDArray[Any]],
    keys: Sequence[Any]
) -> Dict[Any, Any]:
    """
    Returns
    -------
    dict:
        Labels mapped to indexers.
    """
    shape = tuple((len(x) for x in keys))
    group_index = get_group_index(label_list, shape, sort=True, xnull=True)
    if np.all(group_index == -1):
        return {}
    ngroups = (group_index.size and group_index.max()) + 1 if is_int64_overflow_possible(shape) else np.prod(shape, dtype='i8')
    sorter = get_group_index_sorter(group_index, ngroups)
    sorted_labels = [lab.take(sorter) for lab in label_list]
    group_index = group_index.take(sorter)
    return lib.indices_fast(sorter, group_index, keys, sorted_labels)

def get_group_index_sorter(
    group_index: NDArray[np.intp],
    ngroups: Optional[int] = None
) -> NDArray[np.intp]:
    """
    algos.groupsort_indexer implements `counting sort`...
    """
    if ngroups is None:
        ngroups = 1 + group_index.max()
    count = len(group_index)
    alpha = 0.0
    beta = 1.0
    do_groupsort = count > 0 and alpha + beta * ngroups < count * np.log(count)
    if do_groupsort:
        sorter, _ = algos.groupsort_indexer(ensure_platform_int(group_index), ngroups)
    else:
        sorter = group_index.argsort(kind='mergesort')
    return ensure_platform_int(sorter)

def compress_group_index(
    group_index: NDArray[Any],
    sort: bool = True
) -> Tuple[NDArray[np.intp], NDArray[np.int64]]:
    """
    Group_index is offsets into cartesian product of all possible labels.
    """
    if len(group_index) and np.all(group_index[1:] >= group_index[:-1]):
        unique_mask = np.concatenate([group_index[:1] > -1, group_index[1:] != group_index[:-1]])
        comp_ids = unique_mask.cumsum()
        comp_ids -= 1
        obs_group_ids = group_index[unique_mask]
    else:
        size_hint = len(group_index)
        table = hashtable.Int64HashTable(size_hint)
        group_index = ensure_int64(group_index)
        comp_ids, obs_group_ids = table.get_labels_groupby(group_index)
        if sort and len(obs_group_ids) > 0:
            obs_group_ids, comp_ids = _reorder_by_uniques(obs_group_ids, comp_ids)
    return (ensure_int64(comp_ids), ensure_int64(obs_group_ids))

def _reorder_by_uniques(
    uniques: NDArray[np.int64],
    labels: NDArray[np.intp]
) -> Tuple[NDArray[np.int64], NDArray[np.intp]]:
    """
    Parameters
    ----------
    uniques : np.ndarray[np.int64]
    labels : np.ndarray[np.intp]
    """
    sorter = uniques.argsort()
    reverse_indexer = np.empty(len(sorter), dtype=np.intp)
    reverse_indexer.put(sorter, np.arange(len(sorter)))
    mask = labels < 0
    labels = reverse_indexer.take(labels)
    np.putmask(labels, mask, -1)
    uniques = uniques.take(sorter)
    return (uniques, labels)
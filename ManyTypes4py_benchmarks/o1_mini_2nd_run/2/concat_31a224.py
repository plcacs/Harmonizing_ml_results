"""
Concat routines.
"""
from __future__ import annotations
from collections import abc
import types
from typing import TYPE_CHECKING, Literal, cast, overload, Any, Union, Iterable, Mapping, Optional, Sequence, Tuple, List, Set
import warnings
import numpy as np
import pandas as pd
from pandas._libs import lib
from pandas.util._decorators import set_module
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_bool, is_scalar
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core.arrays.categorical import factorize_from_iterable, factorize_from_iterables
import pandas.core.common as com
from pandas.core.indexes.api import (
    Index, MultiIndex, all_indexes_same, default_index, ensure_index, get_objs_combined_axis, get_unanimous_names
)
from pandas.core.internals import concatenate_managers

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable
    from pandas._typing import Axis, AxisInt, HashableT
    from pandas import DataFrame, Series


@overload
def concat(
    objs: Iterable[Union[Series, DataFrame]],
    *,
    axis: Axis = ...,
    join: Literal["inner", "outer"] = ...,
    ignore_index: bool = ...,
    keys: Optional[Sequence[Hashable]] = ...,
    levels: Optional[Sequence[Sequence[Hashable]]] = ...,
    names: Optional[Sequence[Hashable]] = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool = ...,
) -> Union[Series, DataFrame]:
    ...


@overload
def concat(
    objs: Mapping[Hashable, Union[Series, DataFrame]],
    *,
    axis: Axis = ...,
    join: Literal["inner", "outer"] = ...,
    ignore_index: bool = ...,
    keys: Optional[Sequence[Hashable]] = ...,
    levels: Optional[Sequence[Sequence[Hashable]]] = ...,
    names: Optional[Sequence[Hashable]] = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool = ...,
) -> Union[Series, DataFrame]:
    ...


@overload
def concat(
    objs: Union[Series, DataFrame, Any],
    *,
    axis: Axis = ...,
    join: Literal["inner", "outer"] = ...,
    ignore_index: bool = ...,
    keys: Optional[Sequence[Hashable]] = ...,
    levels: Optional[Sequence[Sequence[Hashable]]] = ...,
    names: Optional[Sequence[Hashable]] = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool = ...,
) -> Union[Series, DataFrame]:
    ...


@set_module('pandas')
def concat(
    objs: Union[Iterable[Union[Series, DataFrame]], Mapping[Hashable, Union[Series, DataFrame]]],
    *,
    axis: Axis = 0,
    join: Literal["outer", "inner"] = 'outer',
    ignore_index: bool = False,
    keys: Optional[Sequence[Hashable]] = None,
    levels: Optional[Sequence[Sequence[Hashable]]] = None,
    names: Optional[Sequence[Hashable]] = None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: Any = lib.no_default,
) -> Union[Series, DataFrame]:
    """
    Concatenate pandas objects along a particular axis.

    Allows optional set logic along the other axes.

    Can also add a layer of hierarchical indexing on the concatenation axis,
    which may be useful if the labels are the same (or overlapping) on
    the passed axis number.

    Parameters
    ----------
    objs : an iterable or mapping of Series or DataFrame objects
        If a mapping is passed, the keys will be used as the `keys`
        argument, unless it is passed, in which case the values will be
        selected (see below). Any None objects will be dropped silently unless
        they are all None in which case a ValueError will be raised.
    axis : {0/'index', 1/'columns'}, default 0
        The axis to concatenate along.
    join : {'inner', 'outer'}, default 'outer'
        How to handle indexes on other axis (or axes).
    ignore_index : bool, default False
        If True, do not use the index values along the concatenation axis. The
        resulting axis will be labeled 0, ..., n - 1. This is useful if you are
        concatenating objects where the concatenation axis does not have
        meaningful indexing information. Note the index values on the other
        axes are still respected in the join.
    keys : sequence, default None
        If multiple levels passed, should contain tuples. Construct
        hierarchical index using the passed keys as the outermost level.
    levels : list of sequences, default None
        Specific levels (unique values) to use for constructing a
        MultiIndex. Otherwise they will be inferred from the keys.
    names : list, default None
        Names for the levels in the resulting hierarchical index.
    verify_integrity : bool, default False
        Check whether the new concatenated axis contains duplicates. This can
        be very expensive relative to the actual data concatenation.
    sort : bool, default False
        Sort non-concatenation axis. One exception to this is when the
        non-concatenation axis is a DatetimeIndex and join='outer' and the axis is
        not already aligned. In that case, the non-concatenation axis is always
        sorted lexicographically.
    copy : bool, default False
        If False, do not copy data unnecessarily.

        .. note::
            The `copy` keyword will change behavior in pandas 3.0.
            `Copy-on-Write
            <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
            will be enabled by default, which means that all methods with a
            `copy` keyword will use a lazy copy mechanism to defer the copy and
            ignore the `copy` keyword. The `copy` keyword will be removed in a
            future version of pandas.

            You can already get the future behavior and improvements through
            enabling copy on write ``pd.options.mode.copy_on_write = True``

        .. deprecated:: 3.0.0

    Returns
    -------
    object, type of objs
        When concatenating all ``Series`` along the index (axis=0), a
        ``Series`` is returned. When ``objs`` contains at least one
        ``DataFrame``, a ``DataFrame`` is returned. When concatenating along
        the columns (axis=1), a ``DataFrame`` is returned.

    See Also
    --------
    DataFrame.join : Join DataFrames using indexes.
    DataFrame.merge : Merge DataFrames by indexes or columns.

    Notes
    -----
    The keys, levels, and names arguments are all optional.

    A walkthrough of how this method fits in with other tools for combining
    pandas objects can be found `here
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html>`__.

    It is not recommended to build DataFrames by adding single rows in a
    for loop. Build a list of rows and make a DataFrame in a single concat.

    Examples
    --------
    Combine two ``Series``.

    >>> s1 = pd.Series(["a", "b"])
    >>> s2 = pd.Series(["c", "d"])
    >>> pd.concat([s1, s2])
    0    a
    1    b
    0    c
    1    d
    dtype: object

    Clear the existing index and reset it in the result
    by setting the ``ignore_index`` option to ``True``.

    >>> pd.concat([s1, s2], ignore_index=True)
    0    a
    1    b
    2    c
    3    d
    dtype: object

    Add a hierarchical index at the outermost level of
    the data with the ``keys`` option.

    >>> pd.concat([s1, s2], keys=["s1", "s2"])
    s1  0    a
        1    b
    s2  0    c
        1    d
    dtype: object

    Label the index keys you create with the ``names`` option.

    >>> pd.concat([s1, s2], keys=["s1", "s2"], names=["Series name", "Row ID"])
    Series name  Row ID
    s1           0         a
                 1         b
    s2           0         c
                 1         d
    dtype: object

    Combine two ``DataFrame`` objects with identical columns.

    >>> df1 = pd.DataFrame([["a", 1], ["b", 2]], columns=["letter", "number"])
    >>> df1
      letter  number
    0      a       1
    1      b       2
    >>> df2 = pd.DataFrame([["c", 3], ["d", 4]], columns=["letter", "number"])
    >>> df2
      letter  number
    0      c       3
    1      d       4
    >>> pd.concat([df1, df2])
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    Combine ``DataFrame`` objects with overlapping columns
    and return everything. Columns outside the intersection will
    be filled with ``NaN`` values.

    >>> df3 = pd.DataFrame(
    ...     [["c", 3, "cat"], ["d", 4, "dog"]], columns=["letter", "number", "animal"]
    ... )
    >>> df3
      letter  number animal
    0      c       3    cat
    1      d       4    dog
    >>> pd.concat([df1, df3], sort=False)
      letter  number animal
    0      a       1    NaN
    1      b       2    NaN
    0      c       3    cat
    1      d       4    dog

    Combine ``DataFrame`` objects with overlapping columns
    and return only those that are shared by passing ``inner`` to
    the ``join`` keyword argument.

    >>> pd.concat([df1, df3], join="inner")
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    Combine ``DataFrame`` objects horizontally along the x axis by
    passing in ``axis=1``.

    >>> df4 = pd.DataFrame(
    ...     [["bird", "polly"], ["monkey", "george"]], columns=["animal", "name"]
    ... )
    >>> pd.concat([df1, df4], axis=1)
      letter  number  animal    name
    0      a       1    bird   polly
    1      b       2  monkey  george

    Prevent the result from including duplicate index values with the
    ``verify_integrity`` option.

    >>> df5 = pd.DataFrame([1], index=["a"])
    >>> df5
       0
    a  1
    >>> df6 = pd.DataFrame([2], index=["a"])
    >>> df6
       0
    a  2
    >>> pd.concat([df5, df6], verify_integrity=True)
    Traceback (most recent call last):
        ...
    ValueError: Indexes have overlapping values: ['a']

    Append a single row to the end of a ``DataFrame`` object.

    >>> df7 = pd.DataFrame({"a": 1, "b": 2}, index=[0])
    >>> df7
        a   b
    0   1   2
    >>> new_row = pd.Series({"a": 3, "b": 4})
    >>> new_row
    a    3
    b    4
    dtype: int64
    >>> pd.concat([df7, new_row.to_frame().T], ignore_index=True)
        a   b
    0   1   2
    1   3   4
    """
    if ignore_index and keys is not None:
        raise ValueError(f'Cannot set ignore_index={ignore_index!r} and specify keys. Either should be used.')
    if copy is not lib.no_default:
        warnings.warn(
            'The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 which utilizes a lazy copy mechanism that defers copies until necessary. Use .copy() to make an eager copy if necessary.',
            DeprecationWarning,
            stacklevel=find_stack_level(),
        )
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
        sample_df: Optional[DataFrame] = None
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
    axis_final: int
    axis_final = 1 - bm_axis if is_frame else 0
    names = names or getattr(keys, 'names', None)
    return _get_result(
        objs, is_series, bm_axis, ignore_index, intersect, sort, keys, levels, verify_integrity, names, axis_final
    )


def _sanitize_mixed_ndim(
    objs: List[Union[Series, DataFrame]],
    sample: Union[Series, DataFrame],
    ignore_index: bool,
    axis: int,
) -> List[Union[Series, DataFrame]]:
    new_objs: List[Union[Series, DataFrame]] = []
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
    axis: int,
) -> Union[Series, DataFrame]:
    if is_series:
        sample = cast(Series, objs[0])
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
        sample = cast(DataFrame, objs[0])
        mgrs_indexers: List[Tuple[Any, Mapping[int, Any]]] = []
        result_axes = new_axes(objs, bm_axis, intersect, sort, keys, names, axis, levels, verify_integrity, ignore_index)
        for obj in objs:
            indexers: Mapping[int, Any] = {}
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
    ignore_index: bool,
) -> List[Index]:
    """Return the new [index, column] result for concat."""
    return [
        _get_concat_axis_dataframe(objs, axis, ignore_index, keys, names, levels, verify_integrity)
        if i == bm_axis
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
    names: Optional[Sequence[Hashable]],
) -> Index:
    """Return result concat axis when concatenating Series objects."""
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
        result_names: List[Optional[Hashable]] = [None] * len(objs)
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
    axis: Axis,
    ignore_index: bool,
    keys: Optional[Sequence[Hashable]],
    names: Optional[Sequence[Hashable]],
    levels: Optional[Sequence[Sequence[Hashable]]],
    verify_integrity: bool,
) -> Index:
    """Return result concat axis when concatenating DataFrame objects."""
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
    objs: Union[Iterable[Union[Series, DataFrame]], Mapping[Hashable, Union[Series, DataFrame]], Any],
    keys: Optional[Sequence[Hashable]],
) -> Tuple[List[Union[Series, DataFrame]], Optional[Index], Set[int]]:
    """
    Returns
    -------
    clean_objs : list[Series | DataFrame]
        List of DataFrame and Series with Nones removed.
    keys : Index | None
        None if keys was None
        Index if objs was a Mapping or keys was not None. Filtered where objs was None.
    ndim : set[int]
        Unique .ndim attribute of obj encountered.
    """
    if isinstance(objs, abc.Mapping):
        if keys is None:
            keys = list(objs.keys())
        objs = [objs[k] for k in keys]
    elif isinstance(objs, (ABCSeries, ABCDataFrame)) or is_scalar(objs):
        raise TypeError(
            f'first argument must be an iterable of pandas objects, you passed an object of type "{type(objs).__name__}"'
        )
    elif not isinstance(objs, abc.Sized):
        objs = list(objs)
    if len(objs) == 0:
        raise ValueError('No objects to concatenate')
    if keys is not None:
        if not isinstance(keys, Index):
            keys = Index(keys)
        if len(keys) != len(objs):
            raise ValueError(
                f'The length of the keys ({len(keys)}) must match the length of the objects to concatenate ({len(objs)})'
            )
    key_indices: List[int] = []
    clean_objs: List[Union[Series, DataFrame]] = []
    ndims: Set[int] = set()
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
    ndims: Set[int],
    keys: Optional[Sequence[Hashable]],
    names: Optional[Sequence[Hashable]],
    levels: Optional[Sequence[Sequence[Hashable]]],
    intersect: bool,
) -> Tuple[Union[Series, DataFrame], List[Union[Series, DataFrame]]]:
    if len(ndims) > 1:
        max_ndim = max(ndims)
        for obj in objs:
            if obj.ndim == max_ndim and sum(obj.shape):
                return (obj, objs)
    elif keys is None and names is None and (levels is None) and (not intersect):
        if len(ndims) == 0:
            pass
        elif list(ndims)[0] == 2:
            non_empties = [obj for obj in objs if sum(obj.shape)]
        else:
            non_empties = objs
        if len(non_empties):
            return (non_empties[0], non_empties)
    return (objs[0], objs)


def _concat_indexes(indexes: List[Index]) -> Index:
    return indexes[0].append(indexes[1:])


def validate_unique_levels(levels: List[Index]) -> None:
    for level in levels:
        if not level.is_unique:
            raise ValueError(f'Level values not unique: {level.tolist()}')


def _make_concat_multiindex(
    indexes: List[Index],
    keys: Sequence[Hashable],
    levels: Optional[Sequence[Sequence[Hashable]]] = None,
    names: Optional[Sequence[Hashable]] = None,
) -> MultiIndex:
    if levels is None and (isinstance(keys[0], tuple) or (levels is not None and len(levels) > 1)):
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
        codes_list: List[np.ndarray] = []
        for hlevel, level in zip(zipped, levels):
            to_concat: List[np.ndarray] = []
            if isinstance(hlevel, Index) and hlevel.equals(level):
                lens = [len(idx) for idx in indexes]
                codes_list.append(np.repeat(np.arange(len(hlevel)), lens))
            else:
                for key, index in zip(hlevel, indexes):
                    mask = (isna(level) & isna(key)) | (level == key)
                    if not mask.any():
                        raise ValueError(f'Key {key} not in level {level}')
                    i = np.nonzero(mask)[0][0]
                    to_concat.append(np.repeat(i, len(index)))
                codes_list.append(np.concatenate(to_concat))
        concat_index = _concat_indexes(indexes)
        if isinstance(concat_index, MultiIndex):
            levels_extended = list(levels) + list(concat_index.levels)
            codes_list_extended = list(codes_list) + list(concat_index.codes)
        else:
            codes, categories = factorize_from_iterable(concat_index)
            levels_extended = list(levels) + [categories]
            codes_list_extended = list(codes_list) + [codes]
        if len(names) == len(levels_extended):
            names_final = list(names)
        else:
            if not len({idx.nlevels for idx in indexes}) == 1:
                raise AssertionError('Cannot concat indices that do not have the same number of levels')
            names_final = list(names) + list(get_unanimous_names(*indexes))
        return MultiIndex(levels=levels_extended, codes=codes_list_extended, names=names_final, verify_integrity=False)
    new_index = indexes[0]
    n = len(new_index)
    kpieces = len(indexes)
    new_names = list(names) if names is not None else []
    new_levels = list(levels) if levels is not None else []
    new_codes: List[np.ndarray] = []
    for hlevel, level in zip(zipped, levels):
        hlevel_index = ensure_index(hlevel)
        mapped = level.get_indexer(hlevel_index)
        mask = mapped == -1
        if mask.any():
            raise ValueError(f'Values not found in passed level: {hlevel_index[mask]!s}')
        new_codes.append(np.repeat(mapped, n))
    if isinstance(new_index, MultiIndex):
        new_levels.extend(new_index.levels)
        for lab in new_index.codes:
            new_codes.append(np.tile(lab, kpieces))
    else:
        new_levels.append(new_index.unique())
        single_codes = new_index.unique().get_indexer(new_index)
        new_codes.append(np.tile(single_codes, kpieces))
    if new_names and len(new_names) < len(new_levels):
        if new_index.names:
            new_names.extend(new_index.names)
    return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)

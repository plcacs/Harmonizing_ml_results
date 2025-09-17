from __future__ import annotations
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import datetime
from functools import partial, wraps
from textwrap import dedent
from typing import (Any, Concatenate, Dict, Generator, List, Literal, Mapping as TypingMapping,
                    Optional, Sequence, Tuple, TypeVar, Union)
import warnings
import numpy as np
from pandas._libs import Timestamp, lib
from pandas._libs.algos import rank_1d
import pandas._libs.groupby as libgroupby
from pandas._libs.missing import NA
from pandas._typing import AnyArrayLike, ArrayLike, DtypeObj, IndexLabel, IntervalClosedType, NDFrameT, PositionalIndexer, RandomState, npt
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError, DataError
from pandas.util._decorators import Appender, Substitution, cache_readonly, doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import coerce_indexer_dtype, ensure_dtype_can_hold_na
from pandas.core.dtypes.common import is_bool_dtype, is_float_dtype, is_hashable, is_integer, is_integer_dtype, is_list_like, is_numeric_dtype, is_object_dtype, is_scalar, needs_i8_conversion, pandas_dtype
from pandas.core.dtypes.missing import isna, na_value_for_dtype, notna
from pandas.core import algorithms, sample
from pandas.core._numba import executor
from pandas.core.arrays import ArrowExtensionArray, BaseMaskedArray, ExtensionArray, FloatingArray, IntegerArray, SparseArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.arrays.string_arrow import ArrowStringArray, ArrowStringArrayNumpySemantics
from pandas.core.base import PandasObject, SelectionMixin
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import base, numba_, ops
from pandas.core.groupby.grouper import get_grouper
from pandas.core.groupby.indexing import GroupByIndexingMixin, GroupByNthSelector
from pandas.core.indexes.api import Index, MultiIndex, default_index
from pandas.core.internals.blocks import ensure_block_shape
from pandas.core.series import Series
from pandas.core.sorting import get_group_index_sorter
from pandas.core.util.numba_ import get_jit_arguments, maybe_use_numba, prepare_function_arguments
if False:
    from pandas._libs.tslibs.offsets import BaseOffset

_common_see_also = dedent(
    """
    See also
    --------
    Series.agg : Aggregate using function(s)
    DataFrame.agg : Aggregate using function(s)
    """
)
_groupby_agg_method_engine_template = dedent(
    """
    Parameters
    ----------
    numeric_only : bool, default {no}
        For the series, specify whether to only consider numeric data.
    min_count : int, default {mc}
        The required number of non-NA observations in the group to perform the operation.
    skipna : bool, default {s}
        Exclude NA/null values. If an entire group is NA, the result will be NA.
    engine : str, optional
        Experimental. Numba/Cython engines.
    engine_kwargs : dict, optional
        Additional engine-specific keyword arguments.

    Returns
    -------
    Series or DataFrame
    """
)
_groupby_agg_method_skipna_engine_template = dedent(
    """
    Parameters
    ----------
    numeric_only : bool, default {no}
        For the series, specify whether to only consider numeric data.
    min_count : int, default {mc}
        The required number of non-NA values within each group.
    skipna : bool, default {s}
        Exclude NA/null values. If an entire group is NA, the result will be NA.
    engine : str, optional
        Experimental.
    engine_kwargs : dict, optional
        Additional engine-specific keyword arguments.

    Returns
    -------
    Series or DataFrame

    Examples
    --------
    {example}
    """
)
_pipe_template = dedent(
    """
    Parameters
    ----------
    func : callable
        The function to apply to this object.
    *args : tuple
        Positional arguments to pass into the function.
    **kwargs : dict
        Keyword arguments to pass into the function.

    Returns
    -------
    Same type as caller
    """
)
_transform_template = dedent(
    """
    Parameters
    ----------
    func : callable or str
        Function to apply.

    Returns
    -------
    Series or DataFrame
    """
)


@final
class GroupByPlot(PandasObject):
    """
    Class implementing the .plot attribute for groupby objects.
    """

    def __init__(self, groupby: Any) -> None:
        self._groupby = groupby

    def __call__(self, *args: Any, **kwargs: Any) -> Any:

        def f(self: Any) -> Any:
            return self.plot(*args, **kwargs)
        f.__name__ = 'plot'
        return self._groupby._python_apply_general(f, self._groupby._selected_obj)

    def __getattr__(self, name: str) -> Any:

        def attr(*args: Any, **kwargs: Any) -> Any:

            def f(self: Any) -> Any:
                return getattr(self.plot, name)(*args, **kwargs)
            return self._groupby._python_apply_general(f, self._groupby._selected_obj)
        return attr


_KeysArgType = Union[Hashable, List[Hashable], Callable[[Hashable], Hashable],
                     List[Callable[[Hashable], Hashable]], Mapping[Hashable, Hashable]]


class BaseGroupBy(PandasObject, SelectionMixin[NDFrameT], GroupByIndexingMixin):
    _hidden_attrs = PandasObject._hidden_attrs | {'as_index', 'dropna', 'exclusions', 'grouper', 'group_keys', 'keys', 'level', 'obj', 'observed', 'sort'}
    keys = None
    level = None

    @final
    def __len__(self) -> int:
        return self._grouper.ngroups

    @final
    def __repr__(self) -> str:
        return object.__repr__(self)

    @final
    @property
    def groups(self) -> Any:
        if isinstance(self.keys, list) and len(self.keys) == 1:
            warnings.warn("`groups` by one element list returns scalar is deprecated and will be removed. In a future version `groups` by one element list will return tuple. Use ``df.groupby(by='a').groups`` instead of ``df.groupby(by=['a']).groups`` to avoid this warning", FutureWarning, stacklevel=find_stack_level())
        return self._grouper.groups

    @final
    @property
    def ngroups(self) -> int:
        return self._grouper.ngroups

    @final
    @property
    def indices(self) -> Any:
        return self._grouper.indices

    @final
    def _get_indices(self, names: Sequence[Any]) -> List[List[int]]:
        def get_converter(s: Any) -> Callable[[Any], Any]:
            if isinstance(s, datetime.datetime):
                return lambda key: Timestamp(key)
            elif isinstance(s, np.datetime64):
                return lambda key: Timestamp(key).asm8
            else:
                return lambda key: key
        if len(names) == 0:
            return []
        if len(self.indices) > 0:
            index_sample = next(iter(self.indices))
        else:
            index_sample = None
        name_sample = names[0]
        if isinstance(index_sample, tuple):
            if not isinstance(name_sample, tuple):
                msg = 'must supply a tuple to get_group with multiple grouping keys'
                raise ValueError(msg)
            if not len(name_sample) == len(index_sample):
                try:
                    return [self.indices[name] for name in names]
                except KeyError as err:
                    msg = 'must supply a same-length tuple to get_group with multiple grouping keys'
                    raise ValueError(msg) from err
            converters = (get_converter(s) for s in index_sample)
            names = (tuple((f(n) for f, n in zip(converters, name))) for name in names)
        else:
            converter = get_converter(index_sample)
            names = (converter(name) for name in names)
        return [self.indices.get(name, []) for name in names]

    @final
    def _get_index(self, name: Any) -> List[int]:
        return self._get_indices([name])[0]

    @final
    @cache_readonly
    def _selected_obj(self) -> NDFrameT:
        if isinstance(self.obj, Series):
            return self.obj
        if self._selection is not None:
            if is_hashable(self._selection):
                return self.obj[self._selection]
            return self._obj_with_exclusions
        return self.obj

    @final
    def _dir_additions(self) -> Any:
        return self.obj._dir_additions()

    @overload
    def pipe(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        ...

    @overload
    def pipe(self, func: Union[Callable[..., Any], Tuple[Callable[..., Any], str]], *args: Any, **kwargs: Any) -> Any:
        ...

    @Substitution(klass='GroupBy', examples=dedent("        >>> df = pd.DataFrame({'A': 'a b a b'.split(), 'B': [1, 2, 3, 4]})\n        >>> df\n           A  B\n        0  a  1\n        1  b  2\n        2  a  3\n        3  b  4\n\n        To get the difference between each groups maximum and minimum value in one\n        pass, you can do\n\n        >>> df.groupby('A').pipe(lambda x: x.max() - x.min())\n           B\n        A\n        a  2\n        b  2"))
    @Appender(_pipe_template)
    def pipe(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return com.pipe(self, func, *args, **kwargs)

    @final
    def get_group(self, name: Any) -> NDFrameT:
        keys = self.keys
        level = self.level
        if is_list_like(level) and len(level) == 1 or (is_list_like(keys) and len(keys) == 1):
            if isinstance(name, tuple) and len(name) == 1:
                name = name[0]
            else:
                raise KeyError(name)
        inds = self._get_index(name)
        if not len(inds):
            raise KeyError(name)
        return self._selected_obj.iloc[inds]

    @final
    def __iter__(self) -> Iterator[Tuple[Any, NDFrameT]]:
        keys = self.keys
        level = self.level
        result = self._grouper.get_iterator(self._selected_obj)
        if is_list_like(level) and len(level) == 1 or (isinstance(keys, list) and len(keys) == 1):
            result = (((key,), group) for key, group in result)
        return result


OutputFrameOrSeries = TypeVar('OutputFrameOrSeries', bound=NDFrame)


class GroupBy(BaseGroupBy[NDFrameT]):
    """
    Class for grouping and aggregating relational data.
    """

    @final
    def __init__(self, obj: NDFrame, keys: Optional[Any]=None, level: Optional[Any]=None, grouper: Optional[Any]=None, exclusions: Optional[Any]=None, selection: Optional[Any]=None, as_index: bool=True, sort: bool=True, group_keys: bool=True, observed: bool=False, dropna: bool=True) -> None:
        self._selection = selection
        assert isinstance(obj, NDFrame), type(obj)
        self.level = level
        self.as_index = as_index
        self.keys = keys
        self.sort = sort
        self.group_keys = group_keys
        self.dropna = dropna
        if grouper is None:
            grouper, exclusions, obj = get_grouper(obj, keys, level=level, sort=sort, observed=observed, dropna=self.dropna)
        self.observed = observed
        self.obj = obj
        self._grouper = grouper
        self.exclusions = frozenset(exclusions) if exclusions else frozenset()

    def __getattr__(self, attr: str) -> Any:
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        if attr in self.obj:
            return self[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    @final
    def _op_via_apply(self, name: str, *args: Any, **kwargs: Any) -> Any:
        f = getattr(type(self._obj_with_exclusions), name)

        def curried(x: NDFrameT) -> Any:
            return f(x, *args, **kwargs)
        curried.__name__ = name
        if name in base.plotting_methods:
            return self._python_apply_general(curried, self._selected_obj)
        is_transform = name in base.transformation_kernels
        result = self._python_apply_general(curried, self._obj_with_exclusions, is_transform=is_transform, not_indexed_same=not is_transform)
        if self._grouper.has_dropped_na and is_transform:
            result = self._set_result_index_ordered(result)
        return result

    @final
    def _concat_objects(self, values: Sequence[Any], not_indexed_same: bool=False, is_transform: bool=False) -> Any:
        from pandas.core.reshape.concat import concat
        if self.group_keys and (not is_transform):
            if self.as_index:
                group_keys = self._grouper.result_index
                group_levels = self._grouper.levels
                group_names = self._grouper.names
                result = concat(values, axis=0, keys=group_keys, levels=group_levels, names=group_names, sort=False)
            else:
                result = concat(values, axis=0)
        elif not not_indexed_same:
            result = concat(values, axis=0)
            ax = self._selected_obj.index
            if self.dropna:
                labels = self._grouper.ids
                mask = labels != -1
                ax = ax[mask]
            if ax.has_duplicates and (not result.axes[0].equals(ax)):
                target = algorithms.unique1d(ax._values)
                indexer, _ = result.index.get_indexer_non_unique(target)
                result = result.take(indexer, axis=0)
            else:
                result = result.reindex(ax, axis=0)
        else:
            result = concat(values, axis=0)
        if self.obj.ndim == 1:
            name = self.obj.name
        elif is_hashable(self._selection):
            name = self._selection
        else:
            name = None
        if isinstance(result, Series) and name is not None:
            result.name = name
        return result

    @final
    def _set_result_index_ordered(self, result: Any) -> Any:
        index = self.obj.index
        if self._grouper.is_monotonic and (not self._grouper.has_dropped_na):
            result = result.set_axis(index, axis=0)
            return result
        original_positions = Index(self._grouper.result_ilocs)
        result = result.set_axis(original_positions, axis=0)
        result = result.sort_index(axis=0)
        if self._grouper.has_dropped_na:
            result = result.reindex(default_index(len(index)), axis=0)
        result = result.set_axis(index, axis=0)
        return result

    @final
    def _insert_inaxis_grouper(self, result: DataFrame, qs: Optional[Any]=None) -> DataFrame:
        if isinstance(result, Series):
            result = result.to_frame()
        n_groupings = len(self._grouper.groupings)
        if qs is not None:
            result.insert(0, f'level_{n_groupings}', np.tile(qs, len(result) // len(qs)))
        for level, (name, lev) in enumerate(zip(reversed(self._grouper.names), self._grouper.get_group_levels())):
            if name is None:
                name = 'index' if n_groupings == 1 and qs is None else f'level_{n_groupings - level - 1}'
            if name not in result.columns:
                if qs is None:
                    result.insert(0, name, lev)
                else:
                    result.insert(0, name, Index(np.repeat(lev, len(qs))))
        return result

    @final
    def _wrap_aggregated_output(self, result: Any, qs: Optional[Any]=None) -> Any:
        if not self.as_index:
            result = self._insert_inaxis_grouper(result, qs=qs)
            result = result._consolidate()
            result.index = default_index(len(result))
        else:
            index = self._grouper.result_index
            if qs is not None:
                index = _insert_quantile_level(index, qs)
            result.index = index
        return result

    def _wrap_applied_output(self, data: NDFrameT, values: Any, not_indexed_same: bool=False, is_transform: bool=False) -> Any:
        raise AbstractMethodError(self)

    @final
    def _numba_prep(self, data: NDFrameT) -> Tuple[Any, Any, Any, Any]:
        ngroups = self._grouper.ngroups
        sorted_index = self._grouper.result_ilocs
        sorted_ids = self._grouper._sorted_ids
        sorted_data = data.take(sorted_index, axis=0).to_numpy()
        index_data = data.index
        if isinstance(index_data, MultiIndex):
            if len(self._grouper.groupings) > 1:
                raise NotImplementedError("Grouping with more than 1 grouping labels and a MultiIndex is not supported with engine='numba'")
            group_key = self._grouper.groupings[0].name
            index_data = index_data.get_level_values(group_key)
        sorted_index_data = index_data.take(sorted_index).to_numpy()
        starts, ends = lib.generate_slices(sorted_ids, ngroups)
        return (starts, ends, sorted_index_data, sorted_data)

    def _numba_agg_general(self, func: Callable[..., Any], dtype_mapping: Any, engine_kwargs: Optional[dict], **aggregator_kwargs: Any) -> Any:
        if not self.as_index:
            raise NotImplementedError('as_index=False is not supported. Use .reset_index() instead.')
        data = self._obj_with_exclusions
        df = data if data.ndim == 2 else data.to_frame()
        aggregator = executor.generate_shared_aggregator(func, dtype_mapping, True, **get_jit_arguments(engine_kwargs))
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups
        res_mgr = df._mgr.apply(aggregator, labels=ids, ngroups=ngroups, **aggregator_kwargs)
        res_mgr.axes[1] = self._grouper.result_index
        result = df._constructor_from_mgr(res_mgr, axes=res_mgr.axes)
        if data.ndim == 1:
            result = result.squeeze('columns')
            result.name = data.name
        else:
            result.columns = data.columns
        return result

    @final
    def _transform_with_numba(self, func: Callable[..., Any], *args: Any, engine_kwargs: Optional[dict]=None, **kwargs: Any) -> NDFrameT:
        data = self._obj_with_exclusions
        index_sorting = self._grouper.result_ilocs
        df = data if data.ndim == 2 else data.to_frame()
        starts, ends, sorted_index, sorted_data = self._numba_prep(df)
        numba_.validate_udf(func)
        args, kwargs = prepare_function_arguments(func, args, kwargs, num_required_args=2)
        numba_transform_func = numba_.generate_numba_transform_func(func, **get_jit_arguments(engine_kwargs))
        result = numba_transform_func(sorted_data, sorted_index, starts, ends, len(df.columns), *args)
        result = result.take(np.argsort(index_sorting), axis=0)
        index = data.index
        if data.ndim == 1:
            result_kwargs = {'name': data.name}
            result = result.ravel()
        else:
            result_kwargs = {'columns': data.columns}
        return data._constructor(result, index=index, **result_kwargs)

    @final
    def _aggregate_with_numba(self, func: Callable[..., Any], *args: Any, engine_kwargs: Optional[dict]=None, **kwargs: Any) -> NDFrameT:
        data = self._obj_with_exclusions
        df = data if data.ndim == 2 else data.to_frame()
        starts, ends, sorted_index, sorted_data = self._numba_prep(df)
        numba_.validate_udf(func)
        args, kwargs = prepare_function_arguments(func, args, kwargs, num_required_args=2)
        numba_agg_func = numba_.generate_numba_agg_func(func, **get_jit_arguments(engine_kwargs))
        result = numba_agg_func(sorted_data, sorted_index, starts, ends, len(df.columns), *args)
        index = self._grouper.result_index
        if data.ndim == 1:
            result_kwargs = {'name': data.name}
            result = result.ravel()
        else:
            result_kwargs = {'columns': data.columns}
        res = data._constructor(result, index=index, **result_kwargs)
        if not self.as_index:
            res = self._insert_inaxis_grouper(res)
            res.index = default_index(len(res))
        return res

    def apply(self, func: Callable[..., Any], *args: Any, include_groups: bool=False, **kwargs: Any) -> Any:
        if include_groups:
            raise ValueError('include_groups=True is no longer allowed.')
        if isinstance(func, str):
            if hasattr(self, func):
                res = getattr(self, func)
                if callable(res):
                    return res(*args, **kwargs)
                elif args or kwargs:
                    raise ValueError(f'Cannot pass arguments to property {func}')
                return res
            else:
                raise TypeError(f"apply func should be callable, not '{func}'")
        elif args or kwargs:
            if callable(func):

                @wraps(func)
                def f(g: NDFrameT) -> Any:
                    return func(g, *args, **kwargs)
            else:
                raise ValueError('func must be a callable if args or kwargs are supplied')
        else:
            f = func
        return self._python_apply_general(f, self._obj_with_exclusions)

    @final
    def _python_apply_general(self, f: Callable[..., Any], data: NDFrameT, not_indexed_same: Optional[bool]=None, is_transform: bool=False, is_agg: bool=False) -> Any:
        values, mutated = self._grouper.apply_groupwise(f, data)
        if not_indexed_same is None:
            not_indexed_same = mutated
        return self._wrap_applied_output(data, values, not_indexed_same, is_transform)

    @final
    def _agg_general(self, numeric_only: bool=False, min_count: int=-1, *, alias: str, npfunc: Optional[Callable[..., Any]]=None, **kwargs: Any) -> Any:
        result = self._cython_agg_general(how=alias, alt=npfunc, numeric_only=numeric_only, min_count=min_count, **kwargs)
        return result.__finalize__(self.obj, method='groupby')

    def _agg_py_fallback(self, how: str, values: Any, ndim: int, alt: Callable[..., Any]) -> Any:
        assert alt is not None
        if values.ndim == 1:
            ser = Series(values, copy=False)
        else:
            df = DataFrame(values.T, dtype=values.dtype)
            assert df.shape[1] == 1
            ser = df.iloc[:, 0]
        try:
            res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
        except Exception as err:
            msg = f'agg function failed [how->{how},dtype->{ser.dtype}]'
            raise type(err)(msg) from err
        if ser.dtype == object:
            res_values = res_values.astype(object, copy=False)
        return ensure_block_shape(res_values, ndim=ndim)

    @final
    def _cython_agg_general(self, how: str, alt: Optional[Callable[..., Any]]=None, numeric_only: bool=False, min_count: int=-1, **kwargs: Any) -> Any:
        data = self._get_data_to_aggregate(numeric_only=numeric_only, name=how)

        def array_func(values: Any) -> Any:
            try:
                result = self._grouper._cython_operation('aggregate', values, how, axis=data.ndim - 1, min_count=min_count, **kwargs)
            except NotImplementedError:
                if how in ['any', 'all'] and isinstance(values, SparseArray):
                    pass
                elif alt is None or how in ['any', 'all', 'std', 'sem']:
                    raise
            else:
                return result
            assert alt is not None
            result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
            return result
        new_mgr = data.grouped_reduce(array_func)
        res = self._wrap_agged_manager(new_mgr)
        if how in ['idxmin', 'idxmax']:
            res = self._wrap_idxmax_idxmin(res)
        out = self._wrap_aggregated_output(res)
        return out

    def _cython_transform(self, how: str, numeric_only: bool=False, **kwargs: Any) -> Any:
        raise AbstractMethodError(self)

    @final
    def _transform(self, func: Any, *args: Any, engine: Optional[Any]=None, engine_kwargs: Optional[dict]=None, **kwargs: Any) -> Any:
        if not isinstance(func, str):
            return self._transform_general(func, engine, engine_kwargs, *args, **kwargs)
        elif func not in base.transform_kernel_allowlist:
            msg = f"'{func}' is not a valid function name for transform(name)"
            raise ValueError(msg)
        elif func in base.cythonized_kernels or func in base.transformation_kernels:
            if engine is not None:
                kwargs['engine'] = engine
                kwargs['engine_kwargs'] = engine_kwargs
            return getattr(self, func)(*args, **kwargs)
        else:
            if self.observed:
                return self._reduction_kernel_transform(func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs)
            with com.temp_setattr(self, 'observed', True), com.temp_setattr(self, '_grouper', self._grouper.observed_grouper):
                return self._reduction_kernel_transform(func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs)

    @final
    def _reduction_kernel_transform(self, func: Any, *args: Any, engine: Optional[Any]=None, engine_kwargs: Optional[dict]=None, **kwargs: Any) -> Any:
        with com.temp_setattr(self, 'as_index', True):
            if func in ['idxmin', 'idxmax']:
                func = cast(Literal['idxmin', 'idxmax'], func)
                result = self._idxmax_idxmin(func, True, *args, **kwargs)
            else:
                if engine is not None:
                    kwargs['engine'] = engine
                    kwargs['engine_kwargs'] = engine_kwargs
                result = getattr(self, func)(*args, **kwargs)
        return self._wrap_transform_fast_result(result)

    @final
    def _wrap_transform_fast_result(self, result: Any) -> Any:
        obj = self._obj_with_exclusions
        ids = self._grouper.ids
        result = result.reindex(self._grouper.result_index, axis=0)
        if self.obj.ndim == 1:
            out = algorithms.take_nd(result._values, ids)
            output = obj._constructor(out, index=obj.index, name=obj.name)
        else:
            new_ax = result.index.take(ids)
            output = result._reindex_with_indexers({0: (new_ax, ids)}, allow_dups=True)
            output = output.set_axis(obj.index, axis=0)
        return output

    @final
    def _apply_filter(self, indices: Sequence[np.ndarray], dropna: bool) -> Any:
        if len(indices) == 0:
            indices = np.array([], dtype='int64')
        else:
            indices = np.sort(np.concatenate(indices))
        if dropna:
            filtered = self._selected_obj.take(indices, axis=0)
        else:
            mask = np.empty(len(self._selected_obj.index), dtype=bool)
            mask.fill(False)
            mask[indices.astype(int)] = True
            mask = np.tile(mask, list(self._selected_obj.shape[1:]) + [1]).T
            filtered = self._selected_obj.where(mask)
        return filtered

    @final
    def _cumcount_array(self, ascending: bool=True) -> np.ndarray:
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups
        sorter = get_group_index_sorter(ids, ngroups)
        ids, count = (ids[sorter], len(ids))
        if count == 0:
            return np.empty(0, dtype=np.int64)
        run = np.r_[True, ids[:-1] != ids[1:]]
        rep = np.diff(np.r_[np.nonzero(run)[0], count])
        out = (~run).cumsum()
        if ascending:
            out -= np.repeat(out[run], rep)
        else:
            out = np.repeat(out[np.r_[run[1:], True]], rep) - out
        if self._grouper.has_dropped_na:
            out = np.where(ids == -1, np.nan, out.astype(np.float64, copy=False))
        else:
            out = out.astype(np.int64, copy=False)
        rev = np.empty(count, dtype=np.intp)
        rev[sorter] = np.arange(count, dtype=np.intp)
        return out[rev]

    @final
    @property
    def _obj_1d_constructor(self) -> Any:
        if isinstance(self.obj, DataFrame):
            return self.obj._constructor_sliced
        assert isinstance(self.obj, Series)
        return self.obj._constructor

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def any(self, skipna: bool=True) -> Any:
        return self._cython_agg_general('any', alt=lambda x: Series(x, copy=False).any(skipna=skipna), skipna=skipna)

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def all(self, skipna: bool=True) -> Any:
        return self._cython_agg_general('all', alt=lambda x: Series(x, copy=False).all(skipna=skipna), skipna=skipna)

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def count(self) -> Any:
        data = self._get_data_to_aggregate()
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups
        mask = ids != -1
        is_series = data.ndim == 1

        def hfunc(bvalues: Any) -> Any:
            if bvalues.ndim == 1:
                masked = mask & ~isna(bvalues).reshape(1, -1)
            else:
                masked = mask & ~isna(bvalues)
            counted = lib.count_level_2d(masked, labels=ids, max_bin=ngroups)
            if isinstance(bvalues, BaseMaskedArray):
                return IntegerArray(counted[0], mask=np.zeros(counted.shape[1], dtype=np.bool_))
            elif isinstance(bvalues, ArrowExtensionArray) and (not isinstance(bvalues.dtype, StringDtype)):
                dtype = pandas_dtype('int64[pyarrow]')
                return type(bvalues)._from_sequence(counted[0], dtype=dtype)
            if is_series:
                assert counted.ndim == 2
                assert counted.shape[0] == 1
                return counted[0]
            return counted
        new_mgr = data.grouped_reduce(hfunc)
        new_obj = self._wrap_agged_manager(new_mgr)
        result = self._wrap_aggregated_output(new_obj)
        return result

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def mean(self, numeric_only: bool=False, skipna: bool=True, engine: Optional[Any]=None, engine_kwargs: Optional[dict]=None) -> Any:
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_mean
            return self._numba_agg_general(grouped_mean, executor.float_dtype_mapping, engine_kwargs, min_periods=0, skipna=skipna)
        else:
            result = self._cython_agg_general('mean', alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only, skipna=skipna), numeric_only=numeric_only, skipna=skipna)
            return result.__finalize__(self.obj, method='groupby')

    @final
    def median(self, numeric_only: bool=False, skipna: bool=True) -> Any:
        result = self._cython_agg_general('median', alt=lambda x: Series(x, copy=False).median(numeric_only=numeric_only, skipna=skipna), numeric_only=numeric_only, skipna=skipna)
        return result.__finalize__(self.obj, method='groupby')

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def std(self, ddof: int=1, engine: Optional[Any]=None, engine_kwargs: Optional[dict]=None, numeric_only: bool=False, skipna: bool=True) -> Any:
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_var
            return np.sqrt(self._numba_agg_general(grouped_var, executor.float_dtype_mapping, engine_kwargs, min_periods=0, ddof=ddof, skipna=skipna))
        else:
            return self._cython_agg_general('std', alt=lambda x: Series(x, copy=False).std(ddof=ddof, skipna=skipna), numeric_only=numeric_only, ddof=ddof, skipna=skipna)

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def var(self, ddof: int=1, engine: Optional[Any]=None, engine_kwargs: Optional[dict]=None, numeric_only: bool=False, skipna: bool=True) -> Any:
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_var
            return self._numba_agg_general(grouped_var, executor.float_dtype_mapping, engine_kwargs, min_periods=0, ddof=ddof, skipna=skipna)
        else:
            return self._cython_agg_general('var', alt=lambda x: Series(x, copy=False).var(ddof=ddof, skipna=skipna), numeric_only=numeric_only, ddof=ddof, skipna=skipna)

    @final
    def _value_counts(self, subset: Optional[Any]=None, normalize: bool=False, sort: bool=True, ascending: bool=False, dropna: bool=True) -> Any:
        name = 'proportion' if normalize else 'count'
        df = self.obj
        obj = self._obj_with_exclusions
        in_axis_names = {grouping.name for grouping in self._grouper.groupings if grouping.in_axis}
        if isinstance(obj, Series):
            _name = obj.name
            keys = [] if _name in in_axis_names else [obj]
        else:
            unique_cols = set(obj.columns)
            if subset is not None:
                subsetted = set(subset)
                clashing = subsetted & set(in_axis_names)
                if clashing:
                    raise ValueError(f'Keys {clashing} in subset cannot be in the groupby column keys.')
                doesnt_exist = subsetted - unique_cols
                if doesnt_exist:
                    raise ValueError(f'Keys {doesnt_exist} in subset do not exist in the DataFrame.')
            else:
                subsetted = unique_cols
            keys = (obj.iloc[:, idx] for idx, _name in enumerate(obj.columns) if _name not in in_axis_names and _name in subsetted)
        groupings = list(self._grouper.groupings)
        for key in keys:
            grouper, _, _ = get_grouper(df, key=key, sort=False, observed=False, dropna=dropna)
            groupings += list(grouper.groupings)
        gb = df.groupby(groupings, sort=False, observed=self.observed, dropna=self.dropna)
        result_series = cast(Series, gb.size())
        result_series.name = name
        if sort:
            result_series = result_series.sort_values(ascending=ascending, kind='stable')
        if self.sort:
            names = result_series.index.names
            result_series.index.names = list(range(len(names)))
            index_level = list(range(len(self._grouper.groupings)))
            result_series = result_series.sort_index(level=index_level, sort_remaining=False)
            result_series.index.names = names
        if normalize:
            levels = list(range(len(self._grouper.groupings), result_series.index.nlevels))
            indexed_group_size = result_series.groupby(result_series.index.droplevel(levels), sort=self.sort, dropna=self.dropna, observed=False).transform('sum')
            result_series /= indexed_group_size
            result_series = result_series.fillna(0.0)
        if self.as_index:
            result = result_series
        else:
            index = result_series.index
            columns = com.fill_missing_names(index.names)
            if name in columns:
                raise ValueError(f"Column label '{name}' is duplicate of result column")
            result_series.name = name
            result_series.index = index.set_names(list(range(len(columns))))
            result_frame = result_series.reset_index()
            orig_dtype = self._grouper.groupings[0].obj.columns.dtype
            cols = Index(columns, dtype=orig_dtype).insert(len(columns), name)
            result_frame.columns = cols
            result = result_frame
        return result.__finalize__(self.obj, method='value_counts')

    @final
    def sem(self, ddof: int=1, numeric_only: bool=False, skipna: bool=True) -> Any:
        if numeric_only and self.obj.ndim == 1 and (not is_numeric_dtype(self.obj.dtype)):
            raise TypeError(f'{type(self).__name__}.sem called with numeric_only={numeric_only} and dtype {self.obj.dtype}')
        return self._cython_agg_general('sem', alt=lambda x: Series(x, copy=False).sem(ddof=ddof, skipna=skipna), numeric_only=numeric_only, ddof=ddof, skipna=skipna)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def size(self) -> Any:
        result = self._grouper.size()
        dtype_backend = None
        if isinstance(self.obj, Series):
            if isinstance(self.obj.array, ArrowExtensionArray):
                if isinstance(self.obj.array, ArrowStringArrayNumpySemantics):
                    dtype_backend = None
                elif isinstance(self.obj.array, ArrowStringArray):
                    dtype_backend = 'numpy_nullable'
                else:
                    dtype_backend = 'pyarrow'
            elif isinstance(self.obj.array, BaseMaskedArray):
                dtype_backend = 'numpy_nullable'
        if isinstance(self.obj, Series):
            result = self._obj_1d_constructor(result, name=self.obj.name)
        else:
            result = self._obj_1d_constructor(result)
        if dtype_backend is not None:
            result = result.convert_dtypes(infer_objects=False, convert_string=False, convert_boolean=False, convert_floating=False, dtype_backend=dtype_backend)
        if not self.as_index:
            result = result.rename('size').reset_index()
        return result

    @final
    @doc(_groupby_agg_method_skipna_engine_template, fname='sum', no=False, mc=0, s=True, e=None, ek=None, example=dedent('        For SeriesGroupBy:\n\n        >>> lst = [\'a\', \'a\', \'b\', \'b\']\n        >>> ser = pd.Series([1, 2, 3, 4], index=lst)\n        >>> ser\n        a    1\n        a    2\n        b    3\n        b    4\n        dtype: int64\n        >>> ser.groupby(level=0).sum()\n        a    3\n        b    7\n        dtype: int64\n\n        For DataFrameGroupBy:\n\n        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]\n        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],\n        ...                   index=["tiger", "leopard", "cheetah", "lion"])\n        >>> df\n                  a  b  c\n          tiger   1  8  2\n        leopard   1  2  5\n        cheetah   2  5  8\n           lion   2  6  9\n        >>> df.groupby("a").sum()\n             b   c\n        a\n        1   10   7\n        2   11  17'))
    def sum(self, numeric_only: bool=False, min_count: int=0, skipna: bool=True, engine: Optional[Any]=None, engine_kwargs: Optional[dict]=None) -> Any:
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_sum
            return self._numba_agg_general(grouped_sum, executor.default_dtype_mapping, engine_kwargs, min_periods=min_count, skipna=skipna)
        else:
            with com.temp_setattr(self, 'observed', True):
                result = self._agg_general(numeric_only=numeric_only, min_count=min_count, alias='sum', npfunc=np.sum, skipna=skipna)
            return result

    @final
    def prod(self, numeric_only: bool=False, min_count: int=0, skipna: bool=True) -> Any:
        return self._agg_general(numeric_only=numeric_only, min_count=min_count, skipna=skipna, alias='prod', npfunc=np.prod)

    @final
    @doc(_groupby_agg_method_skipna_engine_template, fname='min', no=False, mc=-1, e=None, ek=None, s=True, example=dedent('        For SeriesGroupBy:\n\n        >>> lst = [\'a\', \'a\', \'b\', \'b\']\n        >>> ser = pd.Series([1, 2, 3, 4], index=lst)\n        >>> ser\n        a    1\n        a    2\n        b    3\n        b    4\n        dtype: int64\n        >>> ser.groupby(level=0).min()\n        a    1\n        b    3\n        dtype: int64\n\n        For DataFrameGroupBy:\n\n        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]\n        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],\n        ...                   index=["tiger", "leopard", "cheetah", "lion"])\n        >>> df\n                  a  b  c\n          tiger   1  8  2\n        leopard   1  2  5\n        cheetah   2  5  8\n           lion   2  6  9\n        >>> df.groupby("a").min()\n            b  c\n        a\n        1   2  2\n        2   5  8'))
    def min(self, numeric_only: bool=False, min_count: int=-1, skipna: bool=True, engine: Optional[Any]=None, engine_kwargs: Optional[dict]=None) -> Any:
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_min_max
            return self._numba_agg_general(grouped_min_max, executor.identity_dtype_mapping, engine_kwargs, min_periods=min_count, is_max=False, skipna=skipna)
        else:
            return self._agg_general(numeric_only=numeric_only, min_count=min_count, skipna=skipna, alias='min', npfunc=np.min)

    @final
    @doc(_groupby_agg_method_skipna_engine_template, fname='max', no=False, mc=-1, e=None, ek=None, s=True, example=dedent('        For SeriesGroupBy:\n\n        >>> lst = [\'a\', \'a\', \'b\', \'b\']\n        >>> ser = pd.Series([1, 2, 3, 4], index=lst)\n        >>> ser\n        a    1\n        a    2\n        b    3\n        b    4\n        dtype: int64\n        >>> ser.groupby(level=0).max()\n        a    2\n        b    4\n        dtype: int64\n\n        For DataFrameGroupBy:\n\n        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]\n        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],\n        ...                   index=["tiger", "leopard", "cheetah", "lion"])\n        >>> df\n                  a  b  c\n          tiger   1  8  2\n        leopard   1  2  5\n        cheetah   2  5  8\n           lion   2  6  9\n        >>> df.groupby("a").max()\n            b  c\n        a\n        1   8  5\n        2   6  9'))
    def max(self, numeric_only: bool=False, min_count: int=-1, skipna: bool=True, engine: Optional[Any]=None, engine_kwargs: Optional[dict]=None) -> Any:
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_min_max
            return self._numba_agg_general(grouped_min_max, executor.identity_dtype_mapping, engine_kwargs, min_periods=min_count, is_max=True, skipna=skipna)
        else:
            return self._agg_general(numeric_only=numeric_only, min_count=min_count, skipna=skipna, alias='max', npfunc=np.max)

    @final
    def first(self, numeric_only: bool=False, min_count: int=-1, skipna: bool=True) -> Any:
        def first_compat(obj: Any) -> Any:

            def first(x: Any) -> Any:
                arr = x.array[notna(x.array)]
                if not len(arr):
                    return x.array.dtype.na_value
                return arr[0]
            if isinstance(obj, DataFrame):
                return obj.apply(first)
            elif isinstance(obj, Series):
                return first(obj)
            else:
                raise TypeError(type(obj))
        return self._agg_general(numeric_only=numeric_only, min_count=min_count, alias='first', npfunc=first_compat, skipna=skipna)

    @final
    def last(self, numeric_only: bool=False, min_count: int=-1, skipna: bool=True) -> Any:
        def last_compat(obj: Any) -> Any:

            def last(x: Any) -> Any:
                arr = x.array[notna(x.array)]
                if not len(arr):
                    return x.array.dtype.na_value
                return arr[-1]
            if isinstance(obj, DataFrame):
                return obj.apply(last)
            elif isinstance(obj, Series):
                return last(obj)
            else:
                raise TypeError(type(obj))
        return self._agg_general(numeric_only=numeric_only, min_count=min_count, alias='last', npfunc=last_compat, skipna=skipna)

    @final
    def ohlc(self) -> Any:
        if self.obj.ndim == 1:
            obj = self._selected_obj
            is_numeric = is_numeric_dtype(obj.dtype)
            if not is_numeric:
                raise DataError('No numeric types to aggregate')
            res_values = self._grouper._cython_operation('aggregate', obj._values, 'ohlc', axis=0, min_count=-1)
            agg_names = ['open', 'high', 'low', 'close']
            result = self.obj._constructor_expanddim(res_values, index=self._grouper.result_index, columns=agg_names)
            return result
        result = self._apply_to_column_groupbys(lambda sgb: sgb.ohlc())
        return result

    @doc(DataFrame.describe)
    def describe(self, percentiles: Optional[Sequence[float]]=None, include: Optional[Any]=None, exclude: Optional[Any]=None) -> Any:
        obj = self._obj_with_exclusions
        if len(obj) == 0:
            described = obj.describe(percentiles=percentiles, include=include, exclude=exclude)
            if obj.ndim == 1:
                result = described
            else:
                result = described.unstack()
            return result.to_frame().T.iloc[:0]
        with com.temp_setattr(self, 'as_index', True):
            result = self._python_apply_general(lambda x: x.describe(percentiles=percentiles, include=include, exclude=exclude), obj, not_indexed_same=True)
        result = result.unstack()
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return result

    @final
    def resample(self, rule: Union[str, Any], *args: Any, include_groups: bool=False, **kwargs: Any) -> Any:
        from pandas.core.resample import get_resampler_for_grouping
        if include_groups:
            raise ValueError('include_groups=True is no longer allowed.')
        return get_resampler_for_grouping(self, rule, *args, **kwargs)

    @final
    def rolling(self, window: Union[int, str, datetime.timedelta], min_periods: Optional[int]=None, center: bool=False, win_type: Optional[str]=None, on: Optional[str]=None, closed: Optional[str]=None, method: str='single') -> Any:
        from pandas.core.window import RollingGroupby
        return RollingGroupby(self._selected_obj, window=window, min_periods=min_periods, center=center, win_type=win_type, on=on, closed=closed, method=method, _grouper=self._grouper, _as_index=self.as_index)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def expanding(self, *args: Any, **kwargs: Any) -> Any:
        from pandas.core.window import ExpandingGroupby
        return ExpandingGroupby(self._selected_obj, *args, _grouper=self._grouper, **kwargs)

    @final
    @Substitution(name='groupby')
    @Appender(_common_see_also)
    def ewm(self, *args: Any, **kwargs: Any) -> Any:
        from pandas.core.window import ExponentialMovingWindowGroupby
        return ExponentialMovingWindowGroupby(self._selected_obj, *args, _grouper=self._grouper, **kwargs)

    @final
    def _fill(self, direction: str, limit: Optional[int]=None) -> Any:
        if limit is None:
            limit = -1
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups
        col_func = partial(libgroupby.group_fillna_indexer, labels=ids, limit=limit, compute_ffill=direction == 'ffill', ngroups=ngroups)

        def blk_func(values: Any) -> Any:
            mask = isna(values)
            if values.ndim == 1:
                indexer = np.empty(values.shape, dtype=np.intp)
                col_func(out=indexer, mask=mask)
                return algorithms.take_nd(values, indexer)
            else:
                if isinstance(values, np.ndarray):
                    dtype = values.dtype
                    if self._grouper.has_dropped_na:
                        dtype = ensure_dtype_can_hold_na(values.dtype)
                    out = np.empty(values.shape, dtype=dtype)
                else:
                    out = type(values)._empty(values.shape, dtype=values.dtype)
                for i, value_element in enumerate(values):
                    indexer = np.empty(values.shape[1], dtype=np.intp)
                    col_func(out=indexer, mask=mask[i])
                    out[i, :] = algorithms.take_nd(value_element, indexer)
                return out
        mgr = self._get_data_to_aggregate()
        res_mgr = mgr.apply(blk_func)
        new_obj = self._wrap_agged_manager(res_mgr)
        new_obj.index = self.obj.index
        return new_obj

    @final
    @Substitution(name='groupby')
    def ffill(self, limit: Optional[int]=None) -> Any:
        return self._fill('ffill', limit=limit)

    @final
    @Substitution(name='groupby')
    def bfill(self, limit: Optional[int]=None) -> Any:
        return self._fill('bfill', limit=limit)

    @final
    @property
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def nth(self) -> GroupByNthSelector:
        return GroupByNthSelector(self)

    def _nth(self, n: Any, dropna: Optional[Union[Literal[None], str]]=None) -> Any:
        if not dropna:
            mask = self._make_mask_from_positional_indexer(n)
            ids = self._grouper.ids
            mask = mask & (ids != -1)
            out = self._mask_selected_obj(mask)
            return out
        if not is_integer(n):
            raise ValueError('dropna option only supported for an integer argument')
        if dropna not in ['any', 'all']:
            raise ValueError(f"For a DataFrame or Series groupby.nth, dropna must be either None, 'any' or 'all', (was passed {dropna}).")
        n = cast(int, n)
        dropped = self._selected_obj.dropna(how=dropna, axis=0)
        if len(dropped) == len(self._selected_obj):
            grouper = self._grouper
        else:
            axis = self._grouper.axis
            grouper = self._grouper.codes_info[axis.isin(dropped.index)]
            if self._grouper.has_dropped_na:
                nulls = grouper == -1
                values = np.where(nulls, NA, grouper)
                grouper = Index(values, dtype='Int64')
        grb = dropped.groupby(grouper, as_index=self.as_index, sort=self.sort)
        return grb.nth(n)

    @final
    def quantile(self, q: Union[float, Sequence[float]]=0.5, interpolation: str='linear', numeric_only: bool=False) -> Any:
        mgr = self._get_data_to_aggregate(numeric_only=numeric_only, name='quantile')
        obj = self._wrap_agged_manager(mgr)
        splitter = self._grouper._get_splitter(obj)
        sdata = splitter._sorted_data
        starts, ends = lib.generate_slices(splitter._slabels, splitter.ngroups)

        def pre_processor(vals: Any) -> Tuple[Any, Optional[Any]]:
            if isinstance(vals.dtype, StringDtype) or is_object_dtype(vals.dtype):
                raise TypeError(f"dtype '{vals.dtype}' does not support operation 'quantile'")
            inference = None
            if isinstance(vals, BaseMaskedArray) and is_numeric_dtype(vals.dtype):
                out = vals.to_numpy(dtype=float, na_value=np.nan)
                inference = vals.dtype
            elif is_integer_dtype(vals.dtype):
                if isinstance(vals, ExtensionArray):
                    out = vals.to_numpy(dtype=float, na_value=np.nan)
                else:
                    out = vals
                inference = np.dtype(np.int64)
            elif is_bool_dtype(vals.dtype) and isinstance(vals, ExtensionArray):
                out = vals.to_numpy(dtype=float, na_value=np.nan)
            elif is_bool_dtype(vals.dtype):
                raise TypeError('Cannot use quantile with bool dtype')
            elif needs_i8_conversion(vals.dtype):
                inference = vals.dtype
                return (vals, inference)
            elif isinstance(vals, ExtensionArray) and is_float_dtype(vals.dtype):
                inference = np.dtype(np.float64)
                out = vals.to_numpy(dtype=float, na_value=np.nan)
            else:
                out = np.asarray(vals)
            return (out, inference)

        def post_processor(vals: Any, inference: Optional[Any], result_mask: Optional[Any], orig_vals: Any) -> Any:
            if inference:
                if isinstance(orig_vals, BaseMaskedArray):
                    assert result_mask is not None
                    if interpolation in {'linear', 'midpoint'} and (not is_float_dtype(orig_vals)):
                        return FloatingArray(vals, result_mask)
                    else:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=RuntimeWarning)
                            return type(orig_vals)(vals.astype(inference.numpy_dtype), result_mask)
                elif not (is_integer_dtype(inference) and interpolation in {'linear', 'midpoint'}):
                    if needs_i8_conversion(inference):
                        vals = vals.astype('i8').view(orig_vals._ndarray.dtype)
                        return orig_vals._from_backing_data(vals)
                    assert isinstance(inference, np.dtype)
                    return vals.astype(inference)
            return vals
        if is_scalar(q):
            qs = np.array([q], dtype=np.float64)
            pass_qs = None
        else:
            qs = np.asarray(q, dtype=np.float64)
            pass_qs = qs
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups
        if self.dropna:
            ids = ids[ids >= 0]
        nqs = len(qs)
        func = partial(libgroupby.group_quantile, labels=ids, qs=qs, interpolation=interpolation, starts=starts, ends=ends)

        def blk_func(values: Any) -> Any:
            orig_vals = values
            if isinstance(values, BaseMaskedArray):
                mask = values._mask
                result_mask = np.zeros((ngroups, nqs), dtype=np.bool_)
            else:
                mask = isna(values)
                result_mask = None
            is_datetimelike = needs_i8_conversion(values.dtype)
            vals, inference = pre_processor(values)
            ncols = 1
            if vals.ndim == 2:
                ncols = vals.shape[0]
            out = np.empty((ncols, ngroups, nqs), dtype=np.float64)
            if is_datetimelike:
                vals = vals.view('i8')
            if vals.ndim == 1:
                func(out[0], values=vals, mask=mask, result_mask=result_mask, is_datetimelike=is_datetimelike)
            else:
                for i in range(ncols):
                    func(out[i], values=vals[i], mask=mask[i], result_mask=None, is_datetimelike=is_datetimelike)
            if vals.ndim == 1:
                out = out.ravel('K')
                if result_mask is not None:
                    result_mask = result_mask.ravel('K')
            else:
                out = out.reshape(ncols, ngroups * nqs)
            return post_processor(out, inference, result_mask, orig_vals)
        res_mgr = sdata._mgr.grouped_reduce(blk_func)
        res = self._wrap_agged_manager(res_mgr)
        return self._wrap_aggregated_output(res, qs=pass_qs)

    @final
    @Substitution(name='groupby')
    def ngroup(self, ascending: bool=True) -> Series:
        obj = self._obj_with_exclusions
        index = obj.index
        comp_ids = self._grouper.ids
        if self._grouper.has_dropped_na:
            comp_ids = np.where(comp_ids == -1, np.nan, comp_ids)
            dtype = np.float64
        else:
            dtype = np.int64
        if any((ping._passed_categorical for ping in self._grouper.groupings)):
            comp_ids = rank_1d(comp_ids, ties_method='dense') - 1
        result = self._obj_1d_constructor(comp_ids, index, dtype=dtype)
        if not ascending:
            result = self.ngroups - 1 - result
        return result

    @final
    @Substitution(name='groupby')
    def cumcount(self, ascending: bool=True) -> Series:
        index = self._obj_with_exclusions.index
        cumcounts = self._cumcount_array(ascending=ascending)
        return self._obj_1d_constructor(cumcounts, index)

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def rank(self, method: str='average', ascending: bool=True, na_option: str='keep', pct: bool=False) -> Any:
        if na_option not in {'keep', 'top', 'bottom'}:
            msg = "na_option must be one of 'keep', 'top', or 'bottom'"
            raise ValueError(msg)
        kwargs = {'ties_method': method, 'ascending': ascending, 'na_option': na_option, 'pct': pct}
        return self._cython_transform('rank', numeric_only=False, **kwargs)

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def cumprod(self, numeric_only: bool=False, *args: Any, **kwargs: Any) -> Any:
        nv.validate_groupby_func('cumprod', args, kwargs, ['skipna'])
        return self._cython_transform('cumprod', numeric_only, **kwargs)

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def cumsum(self, numeric_only: bool=False, *args: Any, **kwargs: Any) -> Any:
        nv.validate_groupby_func('cumsum', args, kwargs, ['skipna'])
        return self._cython_transform('cumsum', numeric_only, **kwargs)

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def cummin(self, numeric_only: bool=False, **kwargs: Any) -> Any:
        skipna = kwargs.get('skipna', True)
        return self._cython_transform('cummin', numeric_only=numeric_only, skipna=skipna)

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def cummax(self, numeric_only: bool=False, **kwargs: Any) -> Any:
        skipna = kwargs.get('skipna', True)
        return self._cython_transform('cummax', numeric_only=numeric_only, skipna=skipna)

    @final
    @Substitution(name='groupby')
    def shift(self, periods: Union[int, Sequence[int]]=1, freq: Optional[Any]=None, fill_value: Any=lib.no_default, suffix: Optional[str]=None) -> Any:
        if is_list_like(periods):
            periods = cast(Sequence[int], periods)
            if len(periods) == 0:
                raise ValueError('If `periods` is an iterable, it cannot be empty.')
            from pandas.core.reshape.concat import concat
            add_suffix = True
        else:
            if not is_integer(periods):
                raise TypeError(f'Periods must be integer, but {periods} is {type(periods)}.')
            if suffix:
                raise ValueError('Cannot specify `suffix` if `periods` is an int.')
            periods = [cast(int, periods)]
            add_suffix = False
        shifted_dataframes: List[Any] = []
        for period in periods:
            if not is_integer(period):
                raise TypeError(f'Periods must be integer, but {period} is {type(period)}.')
            period = cast(int, period)
            if freq is not None:
                f = lambda x: x.shift(period, freq, 0, fill_value)
                shifted = self._python_apply_general(f, self._selected_obj, is_transform=True)
            else:
                if fill_value is lib.no_default:
                    fill_value = None
                ids = self._grouper.ids
                ngroups = self._grouper.ngroups
                res_indexer = np.zeros(len(ids), dtype=np.int64)
                libgroupby.group_shift_indexer(res_indexer, ids, ngroups, period)
                obj = self._obj_with_exclusions
                shifted = obj._reindex_with_indexers({0: (obj.index, res_indexer)}, fill_value=fill_value, allow_dups=True)
            if add_suffix:
                if isinstance(shifted, Series):
                    shifted = cast(NDFrameT, shifted.to_frame())
                shifted = shifted.add_suffix(f'{suffix}_{period}' if suffix else f'_{period}')
            shifted_dataframes.append(cast(Union[Series, DataFrame], shifted))
        return shifted_dataframes[0] if len(shifted_dataframes) == 1 else concat(shifted_dataframes, axis=1)

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def diff(self, periods: int=1) -> Any:
        obj = self._obj_with_exclusions
        shifted = self.shift(periods=periods)
        dtypes_to_f32 = ['int8', 'int16']
        if obj.ndim == 1:
            if obj.dtype in dtypes_to_f32:
                shifted = shifted.astype('float32')
        else:
            to_coerce = [c for c, dtype in obj.dtypes.items() if dtype in dtypes_to_f32]
            if len(to_coerce):
                shifted = shifted.astype({c: 'float32' for c in to_coerce})
        return obj - shifted

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def pct_change(self, periods: int=1, fill_method: Optional[Any]=None, freq: Optional[Any]=None) -> Any:
        if fill_method is not None:
            raise ValueError(f'fill_method must be None; got fill_method={fill_method!r}.')
        if freq is not None:
            f = lambda x: x.pct_change(periods=periods, freq=freq, axis=0)
            return self._python_apply_general(f, self._selected_obj, is_transform=True)
        if fill_method is None:
            op = 'ffill'
        else:
            op = fill_method
        filled = getattr(self, op)(limit=0)
        fill_grp = filled.groupby(self._grouper.codes, group_keys=self.group_keys)
        shifted = fill_grp.shift(periods=periods, freq=freq)
        return filled / shifted - 1

    @final
    @Substitution(name='groupby')
    def head(self, n: int=5) -> Any:
        mask = self._make_mask_from_positional_indexer(slice(None, n))
        return self._mask_selected_obj(mask)

    @final
    @Substitution(name='groupby')
    def tail(self, n: int=5) -> Any:
        if n:
            mask = self._make_mask_from_positional_indexer(slice(-n, None))
        else:
            mask = self._make_mask_from_positional_indexer([])
        return self._mask_selected_obj(mask)

    @final
    def _mask_selected_obj(self, mask: np.ndarray) -> Any:
        ids = self._grouper.ids
        mask = mask & (ids != -1)
        return self._selected_obj[mask]

    @final
    def sample(self, n: Optional[int]=None, frac: Optional[float]=None, replace: bool=False, weights: Optional[Any]=None, random_state: Optional[Union[int, Any]]=None) -> Any:
        if self._selected_obj.empty:
            return self._selected_obj
        size = sample.process_sampling_size(n, frac, replace)
        if weights is not None:
            weights_arr = sample.preprocess_weights(self._selected_obj, weights, axis=0)
        random_state = com.random_state(random_state)
        group_iterator = self._grouper.get_iterator(self._selected_obj)
        sampled_indices: List[np.ndarray] = []
        for labels, obj in group_iterator:
            grp_indices = self.indices[labels]
            group_size = len(grp_indices)
            sample_size = size if size is not None else round(frac * group_size)
            grp_sample = sample.sample(group_size, size=sample_size, replace=replace, weights=None if weights is None else weights_arr[grp_indices], random_state=random_state)
            sampled_indices.append(grp_indices[grp_sample])
        sampled_indices = np.concatenate(sampled_indices)
        return self._selected_obj.take(sampled_indices, axis=0)

    def _idxmax_idxmin(self, how: Literal['idxmin', 'idxmax'], ignore_unobserved: bool=False, skipna: bool=True, numeric_only: bool=False) -> Any:
        if not self.observed and any((ping._passed_categorical for ping in self._grouper.groupings)):
            expected_len = len(self._grouper.result_index)
            group_sizes = self._grouper.size()
            result_len = group_sizes[group_sizes > 0].shape[0]
            has_unobserved = result_len < expected_len
            raise_err = not ignore_unobserved and has_unobserved
            data = self._obj_with_exclusions
            if raise_err and isinstance(data, DataFrame):
                if numeric_only:
                    data = data._get_numeric_data()
                raise_err = len(data.columns) > 0
            if raise_err:
                raise ValueError(f"Can't get {how} of an empty group due to unobserved categories. Specify observed=True in groupby instead.")
        elif not skipna and self._obj_with_exclusions.isna().any(axis=None):
            raise ValueError(f'{type(self).__name__}.{how} with skipna=False encountered an NA value.')
        result = self._agg_general(numeric_only=numeric_only, min_count=1, alias=how, skipna=skipna)
        return result

    def _wrap_idxmax_idxmin(self, res: Any) -> Any:
        index = self.obj.index
        if res.size == 0:
            result = res.astype(index.dtype)
        else:
            if isinstance(index, MultiIndex):
                index = index.to_flat_index()
            values = res._values
            assert isinstance(values, np.ndarray)
            na_value = na_value_for_dtype(index.dtype, compat=False)
            if isinstance(res, Series):
                result = res._constructor(index.array.take(values, allow_fill=True, fill_value=na_value), index=res.index, name=res.name)
            else:
                data: Dict[Any, Any] = {}
                for k, column_values in enumerate(values.T):
                    data[k] = index.array.take(column_values, allow_fill=True, fill_value=na_value)
                result = self.obj._constructor(data, index=res.index)
                result.columns = res.columns
        return result


@doc(GroupBy)
def get_groupby(obj: Union[Series, DataFrame], by: Optional[Any]=None, grouper: Optional[Any]=None, group_keys: bool=True) -> Union["SeriesGroupBy", "DataFrameGroupBy"]:
    if isinstance(obj, Series):
        from pandas.core.groupby.generic import SeriesGroupBy
        klass = SeriesGroupBy
    elif isinstance(obj, DataFrame):
        from pandas.core.groupby.generic import DataFrameGroupBy
        klass = DataFrameGroupBy
    else:
        raise TypeError(f'invalid type: {obj}')
    return klass(obj=obj, keys=by, grouper=grouper, group_keys=group_keys)


def _insert_quantile_level(idx: Index, qs: np.ndarray) -> MultiIndex:
    nqs = len(qs)
    lev_codes, lev = Index(qs).factorize()
    lev_codes = coerce_indexer_dtype(lev_codes, lev)
    if idx._is_multi:
        idx = cast(MultiIndex, idx)
        levels = list(idx.levels) + [lev]
        codes = [np.repeat(x, nqs) for x in idx.codes] + [np.tile(lev_codes, len(idx))]
        mi = MultiIndex(levels=levels, codes=codes, names=idx.names + [None])
    else:
        nidx = len(idx)
        idx_codes = coerce_indexer_dtype(np.arange(nidx), idx)
        levels = [idx, lev]
        codes = [np.repeat(idx_codes, nqs), np.tile(lev_codes, nidx)]
        mi = MultiIndex(levels=levels, codes=codes, names=[idx.name, None])
    return mi
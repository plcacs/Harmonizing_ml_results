from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
import datetime
from functools import partial, wraps
from textwrap import dedent
from typing import TYPE_CHECKING, Literal, TypeVar, Union, cast, final, overload, Optional, Any, Sequence as TypingSequence, Tuple
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
if TYPE_CHECKING:
    from pandas._libs.tslibs import BaseOffset
    from pandas._typing import Concatenate, P, Self, T
    from pandas.core.indexers.objects import BaseIndexer
    from pandas.core.resample import Resampler
    from pandas.core.window import ExpandingGroupby, ExponentialMovingWindowGroupby, RollingGroupby

_common_see_also: str = '\n        See Also\n        --------\n        Series.%(name)s : Apply a function %(name)s to a Series.\n        DataFrame.%(name)s : Apply a function %(name)s\n            to each row or column of a DataFrame.\n'

_groupby_agg_method_engine_template: str = "\nCompute {fname} of group values.\n\nParameters\n----------\nnumeric_only : bool, default {no}\n    Include only float, int, boolean columns.\n\n    .. versionchanged:: 2.0.0\n\n        numeric_only no longer accepts ``None``.\n\nmin_count : int, default {mc}\n    The required number of valid values to perform the operation. If fewer\n    than ``min_count`` non-NA values are present the result will be NA.\n\nengine : str, default None {e}\n    * ``'cython'`` : Runs rolling apply through C-extensions from cython.\n    * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.\n        Only available when ``raw`` is set to ``True``.\n    * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``\n\nengine_kwargs : dict, default None {ek}\n    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``\n    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``\n        and ``parallel`` dictionary keys. The values must either be ``True`` or\n        ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is\n        ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be\n        applied to both the ``func`` and the ``apply`` groupby aggregation.\n\nReturns\n-------\nSeries or DataFrame\n    Computed {fname} of values within each group.\n\nSee Also\n--------\nSeriesGroupBy.min : Return the min of the group values.\nDataFrameGroupBy.min : Return the min of the group values.\nSeriesGroupBy.max : Return the max of the group values.\nDataFrameGroupBy.max : Return the max of the group values.\nSeriesGroupBy.sum : Return the sum of the group values.\nDataFrameGroupBy.sum : Return the sum of the group values.\n\nExamples\n--------\n{example}\n"

_groupby_agg_method_skipna_engine_template: str = "\nCompute {fname} of group values.\n\nParameters\n----------\nnumeric_only : bool, default {no}\n    Include only float, int, boolean columns.\n\n    .. versionchanged:: 2.0.0\n\n        numeric_only no longer accepts ``None``.\n\nmin_count : int, default {mc}\n    The required number of valid values to perform the operation. If fewer\n    than ``min_count`` non-NA values are present the result will be NA.\n\nskipna : bool, default {s}\n    Exclude NA/null values. If the entire group is NA and ``skipna`` is\n    ``True``, the result will be NA.\n\n    .. versionchanged:: 3.0.0\n\nengine : str, default None {e}\n    * ``'cython'`` : Runs rolling apply through C-extensions from cython.\n    * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.\n        Only available when ``raw`` is set to ``True``.\n    * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``\n\nengine_kwargs : dict, default None {ek}\n    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``\n    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``\n        and ``parallel`` dictionary keys. The values must either be ``True`` or\n        ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is\n        ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be\n        applied to both the ``func`` and the ``apply`` groupby aggregation.\n\nReturns\n-------\nSeries or DataFrame\n    Computed {fname} of values within each group.\n\nSee Also\n--------\nSeriesGroupBy.min : Return the min of the group values.\nDataFrameGroupBy.min : Return the min of the group values.\nSeriesGroupBy.max : Return the max of the group values.\nDataFrameGroupBy.max : Return the max of the group values.\nSeriesGroupBy.sum : Return the sum of the group values.\nDataFrameGroupBy.sum : Return the sum of the group values.\n\nExamples\n--------\n{example}\n"

_pipe_template: str = '\nApply a ``func`` with arguments to this %(klass)s object and return its result.\n\nUse `.pipe` when you want to improve readability by chaining together\nfunctions that expect Series, DataFrames, GroupBy or Resampler objects.\nInstead of writing\n\n>>> h = lambda x, arg2, arg3: x + 1 - arg2 * arg3\n>>> g = lambda x, arg1: x * 5 / arg1\n>>> f = lambda x: x ** 4\n>>> df = pd.DataFrame([["a", 4], ["b", 5]], columns=["group", "value"])\n>>> h(g(f(df.groupby(\'group\')), arg1=1), arg2=2, arg3=3)  # doctest: +SKIP\n\nYou can write\n\n>>> (df.groupby(\'group\')\n...    .pipe(f)\n...    .pipe(g, arg1=1)\n...    .pipe(h, arg2=2, arg3=3))  # doctest: +SKIP\n\nwhich is much more readable.\n\nParameters\n----------\nfunc : callable or tuple of (callable, str)\n    Function to apply to this %(klass)s object or, alternatively,\n    a `(callable, data_keyword)` tuple where `data_keyword` is a\n    string indicating the keyword of `callable` that expects the\n    %(klass)s object.\n*args : iterable, optional\n       Positional arguments passed into `func`.\n**kwargs : dict, optional\n         A dictionary of keyword arguments passed into `func`.\n\nReturns\n-------\n%(klass)s\n    The original object with the function `func` applied.\n\nSee Also\n--------\nSeries.pipe : Apply a function with arguments to a series.\nDataFrame.pipe: Apply a function with arguments to a dataframe.\napply : Apply function to each group instead of to the\n    full %(klass)s object.\n\nNotes\n-----\nSee more `here\n<https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls>`_\n\nExamples\n--------\n%(examples)s\n'

_transform_template: str = '\nCall function producing a same-indexed %(klass)s on each group.\n\nReturns a %(klass)s having the same indexes as the original object\nfilled with the transformed values.\n\nParameters\n----------\nfunc : function, str\n    Function to apply to each group. See the Notes section below for requirements.\n\n    Accepted inputs are:\n\n    - String\n    - Python function\n    - Numba JIT function with ``engine=\'numba\'`` specified.\n\n    Only passing a single function is supported with this engine.\n    If the ``\'numba\'`` engine is chosen, the function must be\n    a user defined function with ``values`` and ``index`` as the\n    first and second arguments respectively in the function signature.\n    Each group\'s index will be passed to the user defined function\n    and optionally available for use.\n\n    If a string is chosen, then it needs to be the name\n    of the groupby method you want to use.\n*args\n    Positional arguments to pass to func.\nengine : str, default None\n    * ``\'cython\'`` : Runs the function through C-extensions from cython.\n    * ``\'numba\'`` : Runs the function through JIT compiled code from numba.\n    * ``None`` : Defaults to ``\'cython\'`` or the global setting ``compute.use_numba``\n\nengine_kwargs : dict, default None\n    * For ``\'cython\'`` engine, there are no accepted ``engine_kwargs``\n    * For ``\'numba\'`` engine, the engine can accept ``nopython``, ``nogil``\n      and ``parallel`` dictionary keys. The values must either be ``True`` or\n      ``False``. The default ``engine_kwargs`` for the ``\'numba\'`` engine is\n      ``{\'nopython\': True, \'nogil\': False, \'parallel\': False}`` and will be\n      applied to the function\n\n**kwargs\n    Keyword arguments to be passed into func.\n\nReturns\n-------\n%(klass)s\n    %(klass)s with the same indexes as the original object filled\n    with transformed values.\n\nSee Also\n--------\n%(klass)s.groupby.apply : Apply function ``func`` group-wise and combine\n    the results together.\n%(klass)s.groupby.aggregate : Aggregate using one or more operations.\n%(klass)s.transform : Call ``func`` on self producing a %(klass)s with the\n    same axis shape as self.\n\nNotes\n-----\nEach group is endowed the attribute \'name\' in case you need to know\nwhich group you are working on.\n\nThe current implementation imposes three requirements on f:\n\n* f must return a value that either has the same shape as the input\n  subframe or can be broadcast to the shape of the input subframe.\n  For example, if `f` returns a scalar it will be broadcast to have the\n  same shape as the input subframe.\n* if this is a DataFrame, f must support application column-by-column\n  in the subframe. If f also supports application to the entire subframe,\n  then a fast path is used starting from the second chunk.\n* f must not mutate groups. Mutation is not supported and may\n  produce unexpected results. See :ref:`gotchas.udf-mutation` for more details.\n\nWhen using ``engine=\'numba\'``, there will be no "fall back" behavior internally.\nThe group data and group index will be passed as numpy arrays to the JITed\nuser defined function, and no alternative execution attempts will be tried.\n\n.. versionchanged:: 1.3.0\n\n    The resulting dtype will reflect the return value of the passed ``func``,\n    see the examples below.\n\n.. versionchanged:: 2.0.0\n\n    When using ``.transform`` on a grouped DataFrame and the transformation function\n    returns a DataFrame, pandas now aligns the result\'s index\n    with the input\'s index. You can call ``.to_numpy()`` on the\n    result of the transformation function to avoid alignment.\n\nExamples\n--------\n%(example)s'

@final
class GroupByPlot(PandasObject):
    """
    Class implementing the .plot attribute for groupby objects.
    """

    def __init__(self, groupby: GroupBy) -> None:
        self._groupby = groupby

    def __call__(self, *args: Any, **kwargs: Any) -> NDFrameT:
        def f(self: GroupByPlot) -> Any:
            return self.plot(*args, **kwargs)
        f.__name__ = 'plot'
        return self._groupby._python_apply_general(f, self._groupby._selected_obj)

    def __getattr__(self, name: str) -> Callable[..., NDFrameT]:
        def attr(*args: Any, **kwargs: Any) -> NDFrameT:
            def f(self: GroupByPlot) -> Any:
                return getattr(self.plot, name)(*args, **kwargs)
            return self._groupby._python_apply_general(f, self._groupby._selected_obj)
        return attr

_KeysArgType = Union[Hashable, list[Hashable], Callable[[Hashable], Hashable], list[Callable[[Hashable], Hashable]], Mapping[Hashable, Hashable]]

if TYPE_CHECKING:
    LevelArgType = Union[int, list[int], Hashable, list[Hashable]]

class BaseGroupBy(PandasObject, SelectionMixin[NDFrameT], GroupByIndexingMixin):
    _hidden_attrs: frozenset[str] = PandasObject._hidden_attrs | {'as_index', 'dropna', 'exclusions', 'grouper', 'group_keys', 'keys', 'level', 'obj', 'observed', 'sort'}
    keys: Optional[_KeysArgType] = None
    level: Optional[LevelArgType] = None

    @final
    def __len__(self) -> int:
        return self._grouper.ngroups

    @final
    def __repr__(self) -> str:
        return object.__repr__(self)

    @final
    @property
    def groups(self) -> Mapping[Hashable, Sequence[int]]:
        """
        Dict {group name -> group labels}.
        """
        if isinstance(self.keys, list) and len(self.keys) == 1:
            warnings.warn("`groups` by one element list returns scalar is deprecated and will be removed. In a future version `groups` by one element list will return tuple. Use ``df.groupby(by='a').groups`` instead of ``df.groupby(by=['a']).groups`` to avoid this warning", FutureWarning, stacklevel=find_stack_level())
        return self._grouper.groups

    @final
    @property
    def ngroups(self) -> int:
        return self._grouper.ngroups

    @final
    @property
    def indices(self) -> Mapping[Hashable, list[int]]:
        """
        Dict {group name -> group indices}.
        """
        return self._grouper.indices

    @final
    def _get_indices(self, names: TypingSequence[Hashable]) -> list[list[int]]:
        """
        Safe get multiple indices, translate keys for
        datelike to underlying repr.
        """
        def get_converter(s: Any) -> Callable[[Hashable], Hashable]:
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
            converters = tuple(get_converter(s) for s in index_sample)
            names = tuple(tuple(f(n) for f, n in zip(converters, name)) for name in names)
        else:
            converter = get_converter(index_sample)
            names = tuple(converter(name) for name in names)
        return [self.indices.get(name, []) for name in names]

    @final
    def _get_index(self, name: Hashable) -> list[int]:
        """
        Safe get index, translate keys for datelike to underlying repr.
        """
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
    def _dir_additions(self) -> list[str]:
        return self.obj._dir_additions()

    @overload
    def pipe(self, func: Callable[[GroupBy], NDFrameT], *args: Any, **kwargs: Any) -> NDFrameT:
        ...

    @overload
    def pipe(self, func: Callable[..., NDFrameT], *args: Any, **kwargs: Any) -> NDFrameT:
        ...

    @Substitution(klass='GroupBy', examples=dedent("        >>> df = pd.DataFrame({'A': 'a b a b'.split(), 'B': [1, 2, 3, 4]})\n        >>> df\n           A  B\n        0  a  1\n        1  b  2\n        2  a  3\n        3  b  4\n\n        To get the difference between each groups maximum and minimum value in one\n        pass, you can do\n\n        >>> df.groupby('A').pipe(lambda x: x.max() - x.min())\n           B\n        A\n        a  2\n        b  2"))
    @Appender(_pipe_template)
    def pipe(self, func: Callable[..., NDFrameT], *args: Any, **kwargs: Any) -> NDFrameT:
        return com.pipe(self, func, *args, **kwargs)

    @final
    def get_group(self, name: Hashable) -> Union[Series, DataFrame]:
        """
        Construct DataFrame from group with provided name.
        """
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
    def __iter__(self) -> Iterator[Tuple[Union[Hashable, Tuple[Hashable, ...]], Union[Series, DataFrame]]]:
        """
        Groupby iterator.
        """
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
    def __init__(
        self,
        obj: NDFrame,
        keys: Optional[_KeysArgType] = None,
        level: Optional[LevelArgType] = None,
        grouper: Optional[Any] = None,
        exclusions: Optional[Iterable[Hashable]] = None,
        selection: Optional[Hashable] = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True
    ) -> None:
        self._selection: Optional[Hashable] = selection
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
    def _op_via_apply(
        self,
        name: str,
        *args: Any,
        **kwargs: Any
    ) -> Union[Series, DataFrame]:
        """Compute the result of an operation by using GroupBy's apply."""
        f = getattr(type(self._obj_with_exclusions), name)

        def curried(x: NDFrame) -> Any:
            return f(x, *args, **kwargs)
        curried.__name__ = name
        if name in base.plotting_methods:
            return self._python_apply_general(curried, self._selected_obj)
        is_transform = name in base.transformation_kernels
        result = self._python_apply_general(
            curried, 
            self._obj_with_exclusions, 
            is_transform=is_transform, 
            not_indexed_same=not is_transform
        )
        if self._grouper.has_dropped_na and is_transform:
            result = self._set_result_index_ordered(result)
        return result

    @final
    def _concat_objects(
        self,
        values: list[Union[Series, DataFrame]],
        not_indexed_same: bool = False,
        is_transform: bool = False
    ) -> Union[Series, DataFrame]:
        from pandas.core.reshape.concat import concat
        if self.group_keys and (not is_transform):
            if self.as_index:
                group_keys = self._grouper.result_index
                group_levels = self._grouper.levels
                group_names = self._grouper.names
                result = concat(
                    values, 
                    axis=0, 
                    keys=group_keys, 
                    levels=group_levels, 
                    names=group_names, 
                    sort=False
                )
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
    def _set_result_index_ordered(self, result: Union[Series, DataFrame]) -> Union[Series, DataFrame]:
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
    def _insert_inaxis_grouper(
        self,
        result: Union[Series, DataFrame],
        qs: Optional[np.ndarray] = None
    ) -> Union[Series, DataFrame]:
        if isinstance(result, Series):
            result = cast(NDFrameT, result.to_frame())
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
    def _wrap_aggregated_output(
        self,
        result: Union[Series, DataFrame],
        qs: Optional[np.ndarray] = None
    ) -> Union[Series, DataFrame]:
        """
        Wraps the output of GroupBy aggregations into the expected result.
        """
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

    def _wrap_applied_output(
        self,
        data: NDFrame,
        values: Any,
        not_indexed_same: Optional[bool] = None,
        is_transform: bool = False,
        is_agg: bool = False
    ) -> Union[Series, DataFrame]:
        raise AbstractMethodError(self)

    @final
    def _numba_prep(self, data: NDFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    def _numba_agg_general(
        self,
        func: Callable[..., Any],
        dtype_mapping: dict[str, DtypeObj],
        engine_kwargs: Optional[dict[str, bool]],
        **aggregator_kwargs: Any
    ) -> NDFrameT:
        """
        Perform groupby with a standard numerical aggregation function (e.g. mean)
        with Numba.
        """
        if not self.as_index:
            raise NotImplementedError('as_index=False is not supported. Use .reset_index() instead.')
        data = self._obj_with_exclusions
        df = data if data.ndim == 2 else data.to_frame()
        aggregator = executor.generate_shared_aggregator(func, dtype_mapping, True, **get_jit_arguments(engine_kwargs))
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups
        res_mgr = df._mgr.apply(
            aggregator, 
            labels=ids, 
            ngroups=ngroups, 
            **aggregator_kwargs
        )
        res_mgr.axes[1] = self._grouper.result_index
        result = df._constructor_from_mgr(res_mgr, axes=res_mgr.axes)
        if data.ndim == 1:
            result = result.squeeze('columns')
            result.name = data.name
        else:
            result.columns = data.columns
        return result

    @final
    def _transform_with_numba(
        self,
        func: Callable[..., Any],
        *args: Any,
        engine_kwargs: Optional[dict[str, bool]] = None,
        **kwargs: Any
    ) -> NDFrameT:
        """
        Perform groupby transform routine with the numba engine.
        """
        data = self._obj_with_exclusions
        index_sorting = self._grouper.result_ilocs
        df = data if data.ndim == 2 else data.to_frame()
        starts, ends, sorted_index, sorted_data = self._numba_prep(df)
        numba_.validate_udf(func)
        args, kwargs = prepare_function_arguments(func, args, kwargs, num_required_args=2)
        numba_transform_func = numba_.generate_numba_transform_func(func, **get_jit_arguments(engine_kwargs))
        result = numba_transform_func(
            sorted_data,
            sorted_index,
            starts,
            ends,
            len(df.columns),
            *args
        )
        result = result.take(np.argsort(index_sorting), axis=0)
        index = data.index
        if data.ndim == 1:
            result_kwargs: dict[str, Any] = {'name': data.name}
            result = result.ravel()
        else:
            result_kwargs = {'columns': data.columns}
        return data._constructor(result, index=index, **result_kwargs)

    @final
    def _aggregate_with_numba(
        self,
        func: Callable[..., Any],
        *args: Any,
        engine_kwargs: Optional[dict[str, bool]] = None,
        **kwargs: Any
    ) -> NDFrameT:
        """
        Perform groupby aggregation routine with the numba engine.
        """
        data = self._obj_with_exclusions
        df = data if data.ndim == 2 else data.to_frame()
        starts, ends, sorted_index, sorted_data = self._numba_prep(df)
        numba_.validate_udf(func)
        args, kwargs = prepare_function_arguments(func, args, kwargs, num_required_args=2)
        numba_agg_func = numba_.generate_numba_agg_func(func, **get_jit_arguments(engine_kwargs))
        result = numba_agg_func(
            sorted_data,
            sorted_index,
            starts,
            ends,
            len(df.columns),
            *args
        )
        index = self._grouper.result_index
        if data.ndim == 1:
            result_kwargs: dict[str, Any] = {'name': data.name}
            result = result.ravel()
        else:
            result_kwargs = {'columns': data.columns}
        res = data._constructor(result, index=index, **result_kwargs)
        if not self.as_index:
            res = self._insert_inaxis_grouper(res)
            res.index = default_index(len(res))
        return res

    def apply(
        self,
        func: Callable[..., Any],
        *args: Any,
        include_groups: bool = False,
        **kwargs: Any
    ) -> Union[Series, DataFrame]:
        """
        Apply function ``func`` group-wise and combine the results together.
        """
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
                def f(g: NDFrame) -> Any:
                    return func(g, *args, **kwargs)
            else:
                raise ValueError('func must be a callable if args or kwargs are supplied')
        else:
            f = func
        return self._python_apply_general(f, self._obj_with_exclusions)

    @final
    def _python_apply_general(
        self,
        f: Callable[[NDFrame], Any],
        data: NDFrame,
        not_indexed_same: Optional[bool] = None,
        is_transform: bool = False,
        is_agg: bool = False
    ) -> Union[Series, DataFrame]:
        """
        Apply function f in python space
        """
        values, mutated = self._grouper.apply_groupwise(f, data)
        if not_indexed_same is None:
            not_indexed_same = mutated
        return self._wrap_applied_output(data, values, not_indexed_same, is_transform)

    @final
    def _agg_general(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        *,
        alias: str,
        npfunc: Optional[Callable[..., Any]] = None,
        **kwargs: Any
    ) -> Union[Series, DataFrame]:
        result = self._cython_agg_general(
            how=alias, 
            alt=npfunc, 
            numeric_only=numeric_only, 
            min_count=min_count, 
            **kwargs
        )
        return result.__finalize__(self.obj, method='groupby')

    def _agg_py_fallback(
        self,
        how: str,
        values: Any,
        ndim: int,
        alt: Callable[[Any], Any]
    ) -> Any:
        """
        Fallback to pure-python aggregation if _cython_operation raises
        NotImplementedError.
        """
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
    def _cython_agg_general(
        self,
        how: str,
        alt: Optional[Callable[[Any], Any]] = None,
        numeric_only: bool = False,
        min_count: int = -1,
        **kwargs: Any
    ) -> Union[Series, DataFrame]:
        data = self._get_data_to_aggregate(numeric_only=numeric_only, name=how)

        def array_func(values: Any) -> Any:
            try:
                result = self._grouper._cython_operation(
                    'aggregate', 
                    values, 
                    how, 
                    axis=data.ndim - 1, 
                    min_count=min_count, 
                    **kwargs
                )
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

    def _cython_transform(
        self,
        how: str,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Union[Series, DataFrame]:
        raise AbstractMethodError(self)

    @final
    def _transform(
        self,
        func: Union[str, Callable[..., Any]],
        *args: Any,
        engine: Optional[str] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
        **kwargs: Any
    ) -> Union[Series, DataFrame]:
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
            with com.temp_setattr(self, 'observed', True), \
                 com.temp_setattr(self, '_grouper', self._grouper.observed_grouper):
                return self._reduction_kernel_transform(func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs)

    @final
    def _reduction_kernel_transform(
        self,
        func: str,
        *args: Any,
        engine: Optional[str] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
        **kwargs: Any
    ) -> Union[Series, DataFrame]:
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
    def _wrap_transform_fast_result(
        self,
        result: Union[Series, DataFrame]
    ) -> Union[Series, DataFrame]:
        """
        Fast transform path for aggregations.
        """
        obj = self._obj_with_exclusions
        ids = self._grouper.ids
        result = result.reindex(self._grouper.result_index, axis=0)
        if self.obj.ndim == 1:
            out = algorithms.take_nd(result._values, ids)
            output = obj._constructor(out, index=obj.index, name=obj.name)
        else:
            new_ax = result.index.take(ids)
            output = result._reindex_with_indexers(
                {0: (new_ax, ids)}, 
                allow_dups=True
            )
            output = output.set_axis(obj.index, axis=0)
        return output

    @final
    def _apply_filter(
        self,
        indices: list[int],
        dropna: bool
    ) -> Union[Series, DataFrame]:
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
    def _cumcount_array(self, ascending: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.

        Notes
        -----
        this is currently implementing sort=False
        (though the default is sort=True) for groupby in general
        """
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups
        sorter = get_group_index_sorter(ids, ngroups)
        ids, count = cast(Tuple[np.ndarray, int], (ids[sorter], len(ids)))
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
    def _obj_1d_constructor(self) -> Callable[..., Series]:
        if isinstance(self.obj, DataFrame):
            return self.obj._constructor_sliced
        assert isinstance(self.obj, Series)
        return self.obj._constructor

    @final
    @Substitution(name='groupby')
    @Appender(_groupby_agg_method_skipna_engine_template.format(
        fname='any', no='False', mc=0, s='True', e=None, ek=None, example=dedent('        For SeriesGroupBy:\n\n        >>> lst = ["a", "a", "b"]\n        >>> ser = pd.Series([1, 2, 0], index=lst)\n        >>> ser\n        a    1\n        a    2\n        b    0\n        dtype: int64\n        >>> ser.groupby(level=0).any()\n        a     True\n        b    False\n        dtype: bool\n\n        For DataFrameGroupBy:\n\n        >>> data = [[1, np.nan, 3], [1, np.nan, 6], [7, 8, 9]]\n        >>> df = pd.DataFrame(\n        ...     data, columns=["a", "b", "c"], index=["cow", "horse", "bull"]\n        ... )\n        >>> df\n                        a	  b	c\n            cow     1	NaN	3\n            horse	1	NaN	6\n            bull    7	8.0	9\n        >>> df.groupby(by=["a"]).any()\n               b      c\n            a\n            1   False   True\n            7   True   True\n\n        For Resampler:\n\n        >>> ser = pd.Series(\n        ...     [1, 2, 3, 4],\n        ...     index=pd.DatetimeIndex(\n        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]\n        ...     ),\n        ... )\n        >>> ser\n        2023-01-01    1\n        2023-01-15    2\n        2023-02-01    3\n        2023-02-15    4\n        dtype: int64\n        >>> ser.resample("MS").any()\n        2023-01-01    2\n        2023-02-01    4\n        Freq: MS, dtype: int64\n'))
    def any(
        self,
        skipna: bool = True
    ) -> Union[Series, DataFrame]:
        return self._cython_agg_general(
            'any',
            alt=lambda x: Series(x, copy=False).any(skipna=skipna),
            skipna=skipna
        )

    @final
    @Substitution(name='groupby')
    @Appender(_groupby_agg_method_skipna_engine_template.format(
        fname='all', no='False', mc=-1, s='True', e=None, ek=None, example=dedent('        For SeriesGroupBy:\n\n        >>> lst = ["a", "a", "b"]\n        >>> ser = pd.Series([1, 2, 0], index=lst)\n        >>> ser\n        a    1\n        a    2\n        b    0\n        dtype: int64\n        >>> ser.groupby(level=0).all()\n        a     True\n        b    False\n        dtype: bool\n\n        For DataFrameGroupBy:\n\n        >>> data = [[1, 0, 3], [1, 5, 6], [7, 8, 9]]\n        >>> df = pd.DataFrame(\n        ...     data, columns=["a", "b", "c"], index=["dog", "dog", "eagle"]\n        ... )\n        >>> df\n                        a  b  c\n            dog    1  0  3\n            dog    1  5  6\n            eagle   7  8  9\n        >>> df.groupby(by=["a"]).all()\n               b      c\n            a\n            1  False   True\n            7   True   True\n'))
    def all(
        self,
        skipna: bool = True
    ) -> Union[Series, DataFrame]:
        return self._cython_agg_general(
            'all',
            alt=lambda x: Series(x, copy=False).all(skipna=skipna),
            skipna=skipna
        )

    @final
    @Substitution(name='groupby')
    @Appender(_groupby_agg_method_skipna_engine_template.format(
        fname='count', no='False', mc=0, s='True', e=None, ek=None, example=dedent('        For SeriesGroupBy:\n\n        >>> lst = ["a", "a", "b"]\n        >>> ser = pd.Series([1, 2, np.nan], index=lst)\n        >>> ser\n        a    1.0\n        a    2.0\n        b    NaN\n        dtype: float64\n        >>> ser.groupby(level=0).count()\n        a    2\n        b    0\n        dtype: int64\n\n        For DataFrameGroupBy:\n\n        >>> data = [[1, np.nan, 3], [1, np.nan, 6], [7, 8, 9]]\n        >>> df = pd.DataFrame(\n        ...     data, columns=["a", "b", "c"], index=["cow", "horse", "bull"]\n        ... )\n        >>> df\n                        a	  b	c\n            cow     1	NaN	3\n            horse	1	NaN	6\n            bull    7	8.0	9\n        >>> df.groupby("a").count()\n                b   c\n            a\n            1   0   2\n            7   1   1\n'))
    def count(self) -> Union[Series, DataFrame]:
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
    @Appender(_groupby_agg_method_skipna_engine_template.format(
        fname='mean', no='False', mc=0, s='True', e=None, ek=None, example=dedent('        >>> df = pd.DataFrame(\n        ...     {"A": [1, 1, 2, 1, 2], "B": [np.nan, 2, 3, 4, 5], "C": [1, 2, 1, 1, 2]},\n        ...     columns=["A", "B", "C"],\n        ... )\n\n        Groupby one column and return the mean of the remaining columns in\n        each group.\n\n        >>> df.groupby("A").mean()\n             B         C\n        A\n        1  3.0  1.333333\n        2  4.0  1.500000\n\n        Groupby two columns and return the mean of the remaining column.\n\n        >>> df.groupby(["A", "B"]).mean()\n                 C\n        A B\n        1 2.0  2.0\n          4.0  1.0\n        2 3.0  1.0\n          5.0  2.0\n\n        Groupby one column and return the mean of only particular column in\n        the group.\n\n        >>> df.groupby("A")["B"].mean()\n        A\n        1    3.0\n        2    4.0\n        Name: B, dtype: float64\n'))
    def mean(
        self,
        numeric_only: bool = False,
        skipna: bool = True,
        engine: Optional[str] = None,
        engine_kwargs: Optional[dict[str, bool]] = None
    ) -> Union[Series, DataFrame]:
        """
        Compute mean of groups, excluding missing values.
        """
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_mean
            return self._numba_agg_general(
                grouped_mean, 
                executor.float_dtype_mapping, 
                engine_kwargs, 
                min_periods=0, 
                skipna=skipna
            )
        else:
            result = self._cython_agg_general(
                'mean',
                alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only, skipna=skipna),
                numeric_only=numeric_only,
                skipna=skipna
            )
            return result.__finalize__(self.obj, method='groupby')

    @final
    def median(
        self,
        numeric_only: bool = False,
        skipna: bool = True
    ) -> Union[Series, DataFrame]:
        """
        Compute median of groups, excluding missing values.
        """
        result = self._cython_agg_general(
            'median',
            alt=lambda x: Series(x, copy=False).median(numeric_only=numeric_only, skipna=skipna),
            numeric_only=numeric_only,
            skipna=skipna
        )
        return result.__finalize__(self.obj, method='groupby')

    @final
    @Substitution(name='groupby')
    @Appender(_groupby_agg_method_skipna_engine_template.format(
        fname='std', no='False', mc=-1, s='True', e=None, ek=None, example=dedent('        For SeriesGroupBy:\n\n        >>> lst = ["a", "a", "a", "b", "b", "b"]\n        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)\n        >>> ser\n        a     7\n        a     2\n        a     8\n        b     4\n        b     3\n        b     3\n        dtype: int64\n        >>> ser.groupby(level=0).std()\n        a    3.21455\n        b    0.57735\n        dtype: float64\n\n        For DataFrameGroupBy:\n\n        >>> data = {"a": [1, 3, 5, 7, 7, 8, 3], "b": [1, 4, 8, 4, 4, 2, 1]}\n        >>> df = pd.DataFrame(\n        ...     data, index=["dog", "dog", "dog", "mouse", "mouse", "mouse", "mouse"]\n        ... )\n        >>> df\n                     a  b\n            dog    1  1\n            dog    3  4\n            dog    5  8\n            mouse    7  4\n            mouse    7  4\n            mouse    8  2\n            mouse    3  1\n        >>> df.groupby(level=0).std()\n                          a         b\n            dog    2.000000  3.511885\n            mouse  2.217356  1.500000\n'))
    def std(
        self,
        ddof: int = 1,
        skipna: bool = True,
        engine: Optional[str] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
        numeric_only: bool = False
    ) -> Union[Series, DataFrame]:
        """
        Compute standard deviation of groups, excluding missing values.
        """
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_var
            return np.sqrt(
                self._numba_agg_general(
                    grouped_var, 
                    executor.float_dtype_mapping, 
                    engine_kwargs, 
                    min_periods=0, 
                    ddof=ddof, 
                    skipna=skipna
                )
            )
        else:
            return self._cython_agg_general(
                'std',
                alt=lambda x: Series(x, copy=False).std(ddof=ddof, skipna=skipna),
                numeric_only=numeric_only,
                ddof=ddof,
                skipna=skipna
            )

    @final
    @Substitution(name='groupby')
    @Appender(_groupby_agg_method_skipna_engine_template.format(
        fname='var', no='False', mc=-1, s='True', e=None, ek=None, example=dedent('        For SeriesGroupBy:\n\n        >>> lst = ["a", "a", "a", "b", "b", "b"]\n        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)\n        >>> ser\n        a     7\n        a     2\n        a     8\n        b     4\n        b     3\n        b     3\n        dtype: int64\n        >>> ser.groupby(level=0).var()\n        a    10.333333\n        b     0.333333\n        dtype: float64\n\n        For DataFrameGroupBy:\n\n        >>> data = {"a": [1, 3, 5, 7, 7, 8, 3], "b": [1, 4, 8, 4, 4, 2, 1]}\n        >>> df = pd.DataFrame(\n        ...     data, index=["dog", "dog", "dog", "mouse", "mouse", "mouse", "mouse"]\n        ... )\n        >>> df\n                     a  b\n            dog    1  1\n            dog    3  4\n            dog    5  8\n            mouse    7  4\n            mouse    7  4\n            mouse    8  2\n            mouse    3  1\n        >>> df.groupby(level=0).var()\n                          a          b\n            dog    4.000000  12.333333\n            mouse  4.916667   2.250000\n'))
    def var(
        self,
        ddof: int = 1,
        skipna: bool = True,
        engine: Optional[str] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
        numeric_only: bool = False
    ) -> Union[Series, DataFrame]:
        """
        Compute variance of groups, excluding missing values.
        """
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_var
            return self._numba_agg_general(
                grouped_var, 
                executor.float_dtype_mapping, 
                engine_kwargs, 
                min_periods=0, 
                ddof=ddof, 
                skipna=skipna
            )
        else:
            return self._cython_agg_general(
                'var',
                alt=lambda x: Series(x, copy=False).var(ddof=ddof, skipna=skipna),
                numeric_only=numeric_only,
                ddof=ddof,
                skipna=skipna
            )

    @final
    def _value_counts(
        self,
        subset: Optional[TypingSequence[Hashable]] = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True
    ) -> Union[Series, DataFrame]:
        """
        Shared implementation of value_counts for SeriesGroupBy and DataFrameGroupBy.
        """
        name = 'proportion' if normalize else 'count'
        df = self.obj
        obj = self._obj_with_exclusions
        in_axis_names = {grouping.name for grouping in self._grouper.groupings if grouping.in_axis}
        if isinstance(obj, Series):
            _name = obj.name
            keys: list[Optional[ArrayLike]] = [] if _name in in_axis_names else [obj]
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
            keys = [obj.iloc[:, idx] for idx, _name in enumerate(obj.columns) if _name not in in_axis_names and _name in subsetted]
        groupings = list(self._grouper.groupings)
        for key in keys:
            grouper, _, _ = get_grouper(df, key=key, sort=False, observed=False, dropna=dropna)
            groupings += list(grouper.groupings)
        gb = df.groupby(
            groupings, 
            sort=False, 
            observed=self.observed, 
            dropna=self.dropna
        )
        result_series = cast(Series, gb.size())
        result_series.name = name
        if sort:
            result_series = result_series.sort_values(ascending=ascending, kind='stable')
        if self.sort:
            names = result_series.index.names
            result_series.index.names = range(len(names))
            index_level = range(len(self._grouper.groupings))
            result_series = result_series.sort_index(
                level=index_level, 
                sort_remaining=False
            )
            result_series.index.names = names
        if normalize:
            levels = list(range(len(self._grouper.groupings), result_series.index.nlevels))
            indexed_group_size = result_series.groupby(
                result_series.index.droplevel(levels), 
                sort=self.sort, 
                dropna=self.dropna, 
                observed=False
            ).transform('sum')
            result_series /= indexed_group_size
            result_series = result_series.fillna(0.0)
        if self.as_index:
            result: Union[Series, DataFrame] = result_series
        else:
            index = result_series.index
            columns = com.fill_missing_names(index.names)
            if name in columns:
                raise ValueError(f"Column label '{name}' is duplicate of result column")
            result_series.name = name
            result_series.index = index.set_names(range(len(columns)))
            result_frame = result_series.reset_index()
            orig_dtype = self._grouper.groupings[0].obj.columns.dtype
            cols = Index(columns, dtype=orig_dtype).insert(len(columns), name)
            result_frame.columns = cols
            result = result_frame
        return result.__finalize__(self.obj, method='value_counts')

    @final
    def sem(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        skipna: bool = True
    ) -> Union[Series, DataFrame]:
        """
        Compute standard error of the mean of groups, excluding missing values.
        """
        if numeric_only and self.obj.ndim == 1 and (not is_numeric_dtype(self.obj.dtype)):
            raise TypeError(f'{type(self).__name__}.sem called with numeric_only={numeric_only} and dtype {self.obj.dtype}')
        return self._cython_agg_general(
            'sem',
            alt=lambda x: Series(x, copy=False).sem(ddof=ddof, skipna=skipna),
            numeric_only=numeric_only,
            ddof=ddof,
            skipna=skipna
        )

    @final
    def size(self) -> Union[Series, DataFrame]:
        """
        Compute group sizes.
        """
        result = self._grouper.size()
        dtype_backend: Optional[str] = None
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
            result = cast(NDFrameT, result.convert_dtypes(
                infer_objects=False, 
                convert_string=False, 
                convert_boolean=False, 
                convert_floating=False, 
                dtype_backend=dtype_backend
            ))
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return cast(Union[Series, DataFrame], result)

    @final
    @Substitution(name='groupby')
    @Appender(_groupby_agg_method_skipna_engine_template.format(
        fname='sum', no='False', mc=0, s='True', e=None, ek=None, example=dedent('        For SeriesGroupBy:\n\n        >>> lst = [\'a\', \'a\', \'b\', \'b\']\n        >>> ser = pd.Series([1, 2, 3, 4], index=lst)\n        >>> ser\n        a    1\n        a    2\n        b    3\n        b    4\n        dtype: int64\n        >>> ser.groupby(level=0).sum()\n        a    3\n        b    7\n        dtype: int64\n\n        For DataFrameGroupBy:\n\n        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]\n        >>> df = pd.DataFrame(\n        ...     data, columns=["a", "b", "c"],\n        ...     index=["tuna", "salmon", "catfish", "goldfish"]\n        ... )\n        >>> df\n                        a  b  c\n            tuna     1  8  2\n            salmon   1  2  5\n            catfish   2  5  8\n            goldfish   2  6  9\n        >>> df.groupby("a").sum()\n                b   c\n            a\n            1   10   7\n            2   11  17'))
    def sum(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        skipna: bool = True,
        engine: Optional[str] = None,
        engine_kwargs: Optional[dict[str, bool]] = None
    ) -> Union[Series, DataFrame]:
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_sum
            return self._numba_agg_general(
                grouped_sum, 
                executor.default_dtype_mapping, 
                engine_kwargs, 
                min_periods=min_count,
                skipna=skipna
            )
        else:
            with com.temp_setattr(self, 'observed', True):
                result = self._agg_general(
                    numeric_only=numeric_only, 
                    min_count=min_count, 
                    alias='sum', 
                    npfunc=np.sum, 
                    skipna=skipna
                )
            return result

    @final
    def prod(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        skipna: bool = True
    ) -> Union[Series, DataFrame]:
        """
        Compute prod of group values.
        """
        return self._agg_general(
            numeric_only=numeric_only, 
            min_count=min_count, 
            skipna=skipna, 
            alias='prod', 
            npfunc=np.prod
        )

    @final
    @Substitution(name='groupby')
    @Appender(_groupby_agg_method_skipna_engine_template.format(
        fname='min', no='False', mc=-1, s='True', e=None, ek=None, example=dedent('        For SeriesGroupBy:\n\n        >>> lst = [\'a\', \'a\', \'b\', \'b\']\n        >>> ser = pd.Series([1, 2, 3, 4], index=lst)\n        >>> ser\n        a    1\n        a    2\n        b    3\n        b    4\n        dtype: int64\n        >>> ser.groupby(level=0).min()\n        a    1\n        b    3\n        dtype: int64\n\n        For DataFrameGroupBy:\n\n        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]\n        >>> df = pd.DataFrame(\n        ...     data, columns=["a", "b", "c"],\n        ...     index=["dog", "dog", "catfish", "lion"]\n        ... )\n        >>> df\n                        a  b  c\n            dog    1  8  2\n            dog    1  2  5\n            catfish   2  5  8\n            lion    2  6  9\n        >>> df.groupby("a").min()\n            b  c\n        a\n        1   2  2\n        2   5  8\n'))
    def min(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        skipna: bool = True,
        engine: Optional[str] = None,
        engine_kwargs: Optional[dict[str, bool]] = None
    ) -> Union[Series, DataFrame]:
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_min_max
            return self._numba_agg_general(
                grouped_min_max, 
                executor.identity_dtype_mapping, 
                engine_kwargs, 
                min_periods=min_count, 
                is_max=False, 
                skipna=skipna
            )
        else:
            return self._agg_general(
                numeric_only=numeric_only, 
                min_count=min_count, 
                skipna=skipna, 
                alias='min', 
                npfunc=np.min
            )

    @final
    @Substitution(name='groupby')
    @Appender(_groupby_agg_method_skipna_engine_template.format(
        fname='max', no='False', mc=-1, s='True', e=None, ek=None, example=dedent('        For SeriesGroupBy:\n\n        >>> lst = [\'a\', \'a\', \'b\', \'b\']\n        >>> ser = pd.Series([1, 2, 3, 4], index=lst)\n        >>> ser\n        a    1\n        a    2\n        b    3\n        b    4\n        dtype: int64\n        >>> ser.groupby(level=0).max()\n        a    2\n        b    4\n        dtype: int64\n\n        For DataFrameGroupBy:\n\n        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]\n        >>> df = pd.DataFrame(\n        ...     data, columns=["a", "b", "c"],\n        ...     index=["dog", "dog", "catfish", "lion"]\n        ... )\n        >>> df\n                        a  b  c\n            dog    1  8  2\n            dog    1  2  5\n            catfish   2  5  8\n            lion    2  6  9\n        >>> df.groupby("a").max()\n            b  c\n        a\n        1   8  5\n        2   6  9\n'))
    def max(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        skipna: bool = True,
        engine: Optional[str] = None,
        engine_kwargs: Optional[dict[str, bool]] = None
    ) -> Union[Series, DataFrame]:
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_min_max
            return self._numba_agg_general(
                grouped_min_max, 
                executor.identity_dtype_mapping, 
                engine_kwargs, 
                min_periods=min_count, 
                is_max=True, 
                skipna=skipna
            )
        else:
            return self._agg_general(
                numeric_only=numeric_only, 
                min_count=min_count, 
                skipna=skipna, 
                alias='max', 
                npfunc=np.max
            )

    @final
    def first(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        skipna: bool = True
    ) -> Union[Series, DataFrame]:
        """
        Compute the first entry of each column within each group.
        """
        def first_compat(obj: NDFrame) -> Union[Series, DataFrame]:
            def first(x: Series) -> Any:
                """Helper function for first item that isn't NA."""
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
        return self._agg_general(
            numeric_only=numeric_only, 
            min_count=min_count, 
            alias='first', 
            npfunc=first_compat, 
            skipna=skipna
        )

    @final
    def last(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        skipna: bool = True
    ) -> Union[Series, DataFrame]:
        """
        Compute the last entry of each column within each group.
        """
        def last_compat(obj: NDFrame) -> Union[Series, DataFrame]:
            def last(x: Series) -> Any:
                """Helper function for last item that isn't NA."""
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
        return self._agg_general(
            numeric_only=numeric_only, 
            min_count=min_count, 
            alias='last', 
            npfunc=last_compat, 
            skipna=skipna
        )

    @final
    def ohlc(self) -> DataFrame:
        """
        Compute open, high, low and close values of a group, excluding missing values.
        """
        if self.obj.ndim == 1:
            obj = self._selected_obj
            is_numeric = is_numeric_dtype(obj.dtype)
            if not is_numeric:
                raise DataError('No numeric types to aggregate')
            res_values = self._grouper._cython_operation(
                'aggregate', 
                obj._values, 
                'ohlc', 
                axis=0, 
                min_count=-1
            )
            agg_names = ['open', 'high', 'low', 'close']
            result = self.obj._constructor_expanddim(
                res_values, 
                index=self._grouper.result_index, 
                columns=agg_names
            )
            return result
        result = self._apply_to_column_groupbys(lambda sgb: sgb.ohlc())
        return result

    @doc(DataFrame.describe)
    def describe(
        self,
        percentiles: Optional[TypingSequence[float]] = None,
        include: Optional[Union[str, list[str], dict[str, Any], Callable[[Any], Any]]] = None,
        exclude: Optional[Union[str, list[str], dict[str, Any], Callable[[Any], Any]]] = None
    ) -> Union[Series, DataFrame]:
        obj = self._obj_with_exclusions
        if len(obj) == 0:
            described = obj.describe(percentiles=percentiles, include=include, exclude=exclude)
            if obj.ndim == 1:
                result = described
            else:
                result = described.unstack()
            return result.to_frame().T.iloc[:0]
        with com.temp_setattr(self, 'as_index', True):
            result = self._python_apply_general(
                lambda x: x.describe(percentiles=percentiles, include=include, exclude=exclude), 
                obj, 
                not_indexed_same=True
            )
        result = result.unstack()
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return result

    @final
    def resample(
        self,
        rule: Union[str, BaseOffset],
        *args: Any,
        include_groups: bool = False,
        **kwargs: Any
    ) -> Resampler:
        """
        Provide resampling when using a TimeGrouper.
        """
        from pandas.core.resample import get_resampler_for_grouping
        if include_groups:
            raise ValueError('include_groups=True is no longer allowed.')
        return get_resampler_for_grouping(self, rule, *args, **kwargs)

    @final
    def _fill(
        self,
        direction: Literal['ffill', 'bfill'],
        limit: Optional[int] = None
    ) -> Union[Series, DataFrame]:
        """
        Shared function for `pad` and `backfill` to call Cython method.
        """
        if limit is None:
            limit = -1
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups
        col_func = partial(
            libgroupby.group_fillna_indexer, 
            labels=ids, 
            limit=limit, 
            compute_ffill=(direction == 'ffill'), 
            ngroups=ngroups
        )

        def blk_func(values: Any) -> Any:
            mask: np.ndarray
            result_mask: Optional[np.ndarray] = None
            if isinstance(values, BaseMaskedArray):
                mask = values._mask
                result_mask = np.zeros((ngroups, values.shape[1]), dtype=np.bool_)
            else:
                mask = isna(values)
                result_mask = None
            is_datetimelike = needs_i8_conversion(values.dtype)
            vals, inference = self._pre_processor(values)
            ncols = 1
            if vals.ndim == 2:
                ncols = vals.shape[0]
            out = np.empty((ncols, ngroups, vals.shape[1] if vals.ndim == 2 else 1), dtype=np.float64)
            if is_datetimelike:
                vals = vals.view('i8')
            if vals.ndim == 1:
                col_func(out[0], values=vals, mask=mask, result_mask=result_mask, is_datetimelike=is_datetimelike)
            else:
                for i in range(ncols):
                    col_func(
                        out[i], 
                        values=vals[i], 
                        mask=mask[i], 
                        result_mask=None, 
                        is_datetimelike=is_datetimelike
                    )
            if vals.ndim == 1:
                out = out.ravel('K')
                if result_mask is not None:
                    result_mask = result_mask.ravel('K')
            else:
                out = out.reshape(ncols, ngroups * vals.shape[1])
            return self._post_processor(out, inference, result_mask, original_values=values)

        mgr = self._get_data_to_aggregate()
        res_mgr = mgr.apply(blk_func)
        new_obj = self._wrap_agged_manager(res_mgr)
        new_obj.index = self.obj.index
        return new_obj

    def _pre_processor(self, values: Any) -> Tuple[np.ndarray, Optional[DtypeObj]]:
        if isinstance(values, BaseMaskedArray):
            inference = None
            if is_numeric_dtype(values.dtype):
                out = values.to_numpy(dtype=float, na_value=np.nan)
                inference = values.dtype
            elif is_integer_dtype(values.dtype):
                if isinstance(values, ExtensionArray):
                    out = values.to_numpy(dtype=float, na_value=np.nan)
                else:
                    out = values
                inference = np.dtype(np.int64)
            elif is_bool_dtype(values.dtype) and isinstance(values, ExtensionArray):
                out = values.to_numpy(dtype=float, na_value=np.nan)
            elif is_bool_dtype(values.dtype):
                raise TypeError('Cannot use quantile with bool dtype')
            elif needs_i8_conversion(values.dtype):
                inference = values.dtype
                return (values.to_numpy(), inference)
            elif isinstance(values, ExtensionArray) and is_float_dtype(values.dtype):
                inference = np.dtype(np.float64)
                out = values.to_numpy(dtype=float, na_value=np.nan)
            else:
                out = np.asarray(values)
            return (out, inference)
        else:
            mask = isna(values)
            return (np.asarray(values), None)

    def _post_processor(
        self,
        vals: np.ndarray,
        inference: Optional[DtypeObj],
        result_mask: Optional[np.ndarray],
        original_values: Any
    ) -> Any:
        if inference:
            if isinstance(original_values, BaseMaskedArray):
                assert result_mask is not None
                if inference in ['float32', 'float64'] and inference != 'object':
                    return FloatingArray(vals, result_mask)
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        return type(original_values)(vals.astype(inference.numpy_dtype), result_mask)
            elif not (is_integer_dtype(inference) and 'linear' in inference.name):
                if needs_i8_conversion(inference):
                    vals = vals.astype('i8').view(original_values._ndarray.dtype)
                    return original_values._from_backing_data(vals)
                assert isinstance(inference, np.dtype)
                return vals.astype(inference)
        return vals

    @final
    def _wrap_idxmax_idxmin(
        self,
        res: Union[Series, DataFrame]
    ) -> Union[Series, DataFrame]:
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
                result = res._constructor(
                    index.array.take(values, allow_fill=True, fill_value=na_value), 
                    index=res.index, 
                    name=res.name
                )
            else:
                data: dict[int, ArrayLike] = {}
                for k, column_values in enumerate(values.T):
                    data[k] = index.array.take(column_values, allow_fill=True, fill_value=na_value)
                result = self.obj._constructor(data, index=res.index)
                result.columns = res.columns
        return result

OutputFrameOrSeries = TypeVar('OutputFrameOrSeries', bound=NDFrame)

@doc(GroupBy)
def get_groupby(
    obj: NDFrame,
    by: Optional[_KeysArgType] = None,
    grouper: Optional[Any] = None,
    group_keys: bool = True
) -> GroupBy:
    if isinstance(obj, Series):
        from pandas.core.groupby.generic import SeriesGroupBy
        klass = SeriesGroupBy
    elif isinstance(obj, DataFrame):
        from pandas.core.groupby.generic import DataFrameGroupBy
        klass = DataFrameGroupBy
    else:
        raise TypeError(f'invalid type: {obj}')
    return klass(obj=obj, keys=by, grouper=grouper, group_keys=group_keys)

def _insert_quantile_level(
    idx: Index,
    qs: np.ndarray
) -> MultiIndex:
    """
    Insert the sequence 'qs' of quantiles as the inner-most level of a MultiIndex.
    """
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

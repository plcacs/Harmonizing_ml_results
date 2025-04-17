"""
Define the SeriesGroupBy and DataFrameGroupBy
classes that hold the groupby interfaces (and some implementations).

These are user facing as the result of the ``df.groupby(...)`` operations,
which here returns a DataFrameGroupBy object.
"""
from __future__ import annotations
from collections import abc
from collections.abc import Callable, Iterable
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable as TypingCallable, Dict, Iterator, List, Literal, NamedTuple, Optional, TypeVar, Union, cast
import warnings
import numpy as np
from pandas._libs import Interval
from pandas._libs.hashtable import duplicated
from pandas.errors import SpecificationError
from pandas.util._decorators import Appender, Substitution, doc, set_module
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import ensure_int64, is_bool, is_dict_like, is_integer_dtype, is_list_like, is_numeric_dtype, is_scalar
from pandas.core.dtypes.dtypes import CategoricalDtype, IntervalDtype
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import isna, notna
from pandas.core import algorithms
from pandas.core.apply import GroupByApply, maybe_mangle_lambdas, reconstruct_func, validate_func_kwargs
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import base
from pandas.core.groupby.groupby import GroupBy, GroupByPlot, _transform_template
from pandas.core.indexes.api import Index, MultiIndex, all_indexes_same, default_index
from pandas.core.series import Series
from pandas.core.sorting import get_group_index
from pandas.core.util.numba_ import maybe_use_numba
from pandas.plotting import boxplot_frame_groupby
if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from pandas._typing import ArrayLike, BlockManager, CorrelationMethod, IndexLabel, Manager, SingleBlockManager, TakeIndexer
    from pandas import Categorical
    from pandas.core.generic import NDFrame
AggScalar = Union[str, Callable[..., Any]]
ScalarResult = TypeVar('ScalarResult')


@set_module('pandas')
class NamedAgg(NamedTuple):
    """
    Helper for column specific aggregation with control over output column names.

    Subclass of typing.NamedTuple.

    Parameters
    ----------
    column : Hashable
        Column label in the DataFrame to apply aggfunc.
    aggfunc : function or str
        Function to apply to the provided column. If string, the name of a built-in
        pandas function.

    See Also
    --------
    DataFrame.groupby : Group DataFrame using a mapper or by a Series of columns.

    Examples
    --------
    >>> df = pd.DataFrame({"key": [1, 1, 2], "a": [-1, 0, 1], 1: [10, 11, 12]})
    >>> agg_a = pd.NamedAgg(column="a", aggfunc="min")
    >>> agg_1 = pd.NamedAgg(column=1, aggfunc=lambda x: np.mean(x))
    >>> df.groupby("key").agg(result_a=agg_a, result_1=agg_1)
         result_a  result_1
    key
    1          -1      10.5
    2           1      12.0
    """
    column: Hashable
    aggfunc: AggScalar


@set_module('pandas.api.typing')
class SeriesGroupBy(GroupBy[Series]):

    def _wrap_agged_manager(self, mgr):
        out = self.obj._constructor_from_mgr(mgr, axes=mgr.axes)
        out._name = self.obj.name
        return out

    def _get_data_to_aggregate(self, *, numeric_only: bool=False, name:
        Optional[str]=None):
        ser = self._obj_with_exclusions
        single = ser._mgr
        if numeric_only and not is_numeric_dtype(ser.dtype):
            kwd_name = 'numeric_only'
            raise TypeError(
                f'Cannot use {kwd_name}=True with {type(self).__name__}.{name} and non-numeric dtypes.'
                )
        return single
    _agg_examples_doc = dedent(
        """
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4])

    >>> s
    0    1
    1    2
    2    3
    3    4
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).min()
    1    1
    2    3
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).agg('min')
    1    1
    2    3
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).agg(['min', 'max'])
       min  max
    1    1    2
    2    3    4

    The output column names can be controlled by passing
    the desired column names and aggregations as keyword arguments.

    >>> s.groupby([1, 1, 2, 2]).agg(
    ...     minimum='min',
    ...     maximum='max',
    ... )
       minimum  maximum
    1        1        2
    2        3        4

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the aggregating function.

    >>> s.groupby([1, 1, 2, 2]).agg(lambda x: x.astype(float).min())
    1    1.0
    2    3.0
    dtype: float64
    """
        )

    def apply(self, func, *args: Any, **kwargs: Any):
        """
        Apply function ``func`` group-wise and combine the results together.

        The function passed to ``apply`` must take a series as its first
        argument and return a DataFrame, Series or scalar. ``apply`` will
        then take care of combining the results back together into a single
        dataframe or series. ``apply`` is therefore a highly flexible
        grouping method.

        While ``apply`` is a very flexible method, its downside is that
        using it can be quite a bit slower than using more specific methods
        like ``agg`` or ``transform``. Pandas offers a wide range of method that will
        be much faster than using ``apply`` for their specific purposes, so try to
        use them before reaching for ``apply``.
        """
        return super().apply(func, *args, **kwargs)

    def aggregate(self, func=None, *args: Any, engine: Optional[str]=None,
        engine_kwargs: Optional[Dict[str, bool]]=None, **kwargs: Any):
        """
        Aggregate using one or more operations.

        The ``aggregate`` method enables flexible and efficient aggregation of grouped
        data using a variety of functions, including built-in, user-defined, and
        optimized JIT-compiled functions.
        """
        relabeling = func is None
        columns: Optional[List[str]] = None
        if relabeling:
            columns, func = validate_func_kwargs(kwargs)
            kwargs = {}
        if isinstance(func, str):
            if maybe_use_numba(engine) and engine is not None:
                kwargs['engine'] = engine
            if engine_kwargs is not None:
                kwargs['engine_kwargs'] = engine_kwargs
            return getattr(self, func)(*args, **kwargs)
        elif isinstance(func, Iterable):
            func = maybe_mangle_lambdas(func)
            kwargs['engine'] = engine
            kwargs['engine_kwargs'] = engine_kwargs
            ret = self._aggregate_multiple_funcs(func, *args, **kwargs)
            if relabeling:
                assert columns is not None
                ret.columns = columns
            if not self.as_index:
                ret = ret.reset_index()
            return ret
        else:
            if maybe_use_numba(engine):
                return self._aggregate_with_numba(func, *args,
                    engine_kwargs=engine_kwargs, **kwargs)
            if self.ngroups == 0:
                obj = self._obj_with_exclusions
                return self.obj._constructor([], name=self.obj.name, index=
                    self._grouper.result_index, dtype=obj.dtype)
            return self._python_agg_general(func, *args, **kwargs)
    agg = aggregate

    def _python_agg_general(self, func, *args: Any, **kwargs: Any):
        f = lambda x: func(x, *args, **kwargs)
        obj = self._obj_with_exclusions
        result = self._grouper.agg_series(obj, f)
        res = obj._constructor(result, name=obj.name)
        return self._wrap_aggregated_output(res)

    def _aggregate_multiple_funcs(self, arg, *args: Any, **kwargs: Any):
        if isinstance(arg, dict):
            raise SpecificationError('nested renamer is not supported')
        if any(isinstance(x, (tuple, list)) for x in arg):
            arg = ((x, x) if not isinstance(x, (tuple, list)) else x for x in
                arg)
        else:
            columns = [(com.get_callable_name(f) or f) for f in arg]
            arg = zip(columns, arg)
        results: Dict[base.OutputKey, Union[DataFrame, Series]] = {}
        with com.temp_setattr(self, 'as_index', True):
            for idx, (name, func) in enumerate(arg):
                key = base.OutputKey(label=name, position=idx)
                results[key] = self.aggregate(func, *args, **kwargs)
        if any(isinstance(x, DataFrame) for x in results.values()):
            from pandas import concat
            res_df = concat(results.values(), axis=1, keys=[key.label for
                key in results])
            return res_df
        indexed_output: Dict[int, ArrayLike] = {key.position: val for key,
            val in results.items()}
        output = self.obj._constructor_expanddim(indexed_output, index=None)
        output.columns = Index([key.label for key in results])
        return output

    def _wrap_applied_output(self, data, values, not_indexed_same=False,
        is_transform=False):
        """
        Wrap the output of SeriesGroupBy.apply into the expected result.

        Parameters
        ----------
        data : Series
            Input data for groupby operation.
        values : List[Any]
            Applied output for each group.
        not_indexed_same : bool, default False
            Whether the applied outputs are not indexed the same as the group axes.

        Returns
        -------
        DataFrame or Series
        """
        if len(values) == 0:
            if is_transform:
                res_index = data.index
            elif not self.group_keys:
                res_index = None
            else:
                res_index = self._grouper.result_index
            return self.obj._constructor([], name=self.obj.name, index=
                res_index, dtype=data.dtype)
        assert values is not None
        if isinstance(values[0], dict):
            index = self._grouper.result_index
            res_df = self.obj._constructor_expanddim(values, index=index)
            res_ser = res_df.stack()
            res_ser.name = self.obj.name
            return res_ser
        elif isinstance(values[0], (Series, DataFrame)):
            result = self._concat_objects(values, not_indexed_same=
                not_indexed_same, is_transform=is_transform)
            if isinstance(result, Series):
                result.name = self.obj.name
            if not self.as_index and not_indexed_same:
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return result
        else:
            result = self.obj._constructor(data=values, index=self._grouper
                .result_index, name=self.obj.name)
            if not self.as_index:
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return result
    __examples_series_doc = dedent(
        """
    >>> ser = pd.Series([390.0, 350.0, 30.0, 20.0],
                    index=["Falcon", "Falcon", "Parrot", "Parrot"],
                    name="Max Speed")
    >>> grouped = ser.groupby([1, 1, 2, 2])
    >>> grouped.transform(lambda x: (x - x.mean()) / x.std())
        Falcon    0.707107
        Falcon   -0.707107
        Parrot    0.707107
        Parrot   -0.707107
        Name: Max Speed, dtype: float64

    Broadcast result of the transformation

    >>> grouped.transform(lambda x: x.max() - x.min())
    Falcon    40.0
    Falcon    40.0
    Parrot    10.0
    Parrot    10.0
    Name: Max Speed, dtype: float64

    >>> grouped.transform("mean")
    Falcon    370.0
    Falcon    370.0
    Parrot     25.0
    Parrot     25.0
    Name: Max Speed, dtype: float64

    .. versionchanged:: 1.3.0

    The resulting dtype will reflect the return value of the passed ``func``,
    for example:

    >>> grouped.transform(lambda x: x.astype(int).max())
    Falcon    390
    Falcon    390
    Parrot     30
    Parrot     30
    Name: Max Speed, dtype: int64
    """
        )

    @Substitution(klass='Series', example=__examples_series_doc)
    @Appender(_transform_template)
    def transform(self, func, *args: Any, engine: Optional[str]=None,
        engine_kwargs: Optional[Dict[str, bool]]=None, **kwargs: Any):
        return self._transform(func, *args, engine=engine, engine_kwargs=
            engine_kwargs, **kwargs)

    def _cython_transform(self, how, numeric_only=False, **kwargs: Any):
        obj = self._obj_with_exclusions
        try:
            result = self._grouper._cython_operation('transform', obj.
                _values, how, 0, **kwargs)
        except NotImplementedError as err:
            raise TypeError(f'{how} is not supported for {obj.dtype} dtype'
                ) from err
        return obj._constructor(result, index=self.obj.index, name=obj.name)

    def _transform_general(self, func, engine, engine_kwargs, *args: Any,
        **kwargs: Any):
        """
        Transform with a callable `func`.
        """
        if maybe_use_numba(engine):
            return self._transform_with_numba(func, *args, engine_kwargs=
                engine_kwargs, **kwargs)
        assert callable(func)
        klass = type(self.obj)
        results: List[Series] = []
        for name, group in self._grouper.get_iterator(self._obj_with_exclusions
            ):
            object.__setattr__(group, 'name', name)
            res = func(group, *args, **kwargs)
            results.append(klass(res, index=group.index))
        if results:
            from pandas.core.reshape.concat import concat
            concatenated = concat(results, ignore_index=True)
            result = self._set_result_index_ordered(concatenated)
        else:
            result = self.obj._constructor(dtype=np.float64)
        result.name = self.obj.name
        return result

    def filter(self, func, dropna=True, *args: Any, **kwargs: Any):
        """
        Filter elements from groups that don't satisfy a criterion.
        """
        if isinstance(func, str):
            wrapper: Callable[[Series], bool] = lambda x: getattr(x, func)(*
                args, **kwargs)
        else:
            wrapper = lambda x: func(x, *args, **kwargs)

        def true_and_notna(x):
            b = wrapper(x)
            return notna(b) and b
        try:
            indices: List[np.ndarray] = [self._get_index(name) for name,
                group in self._grouper.get_iterator(self.
                _obj_with_exclusions) if true_and_notna(group)]
        except (ValueError, TypeError) as err:
            raise TypeError('the filter must return a boolean result') from err
        filtered = self._apply_filter(indices, dropna)
        return filtered

    def nunique(self, dropna=True):
        """
        Return number of unique elements in the group.
        """
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups
        val = self.obj._values
        codes, uniques = algorithms.factorize(val, use_na_sentinel=dropna,
            sort=False)
        if self._grouper.has_dropped_na:
            mask = ids >= 0
            ids = ids[mask]
            codes = codes[mask]
        group_index = get_group_index(labels=[ids, codes], shape=(ngroups,
            len(uniques)), sort=False, xnull=dropna)
        if dropna:
            mask = group_index >= 0
            if (~mask).any():
                ids = ids[mask]
                group_index = group_index[mask]
        mask = duplicated(group_index, 'first')
        res = np.bincount(ids[~mask], minlength=ngroups)
        res = ensure_int64(res)
        ri = self._grouper.result_index
        result: Union[Series, DataFrame] = self.obj._constructor(res, index
            =ri, name=self.obj.name)
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return result

    @doc(Series.describe)
    def describe(self, percentiles=None, include=None, exclude=None):
        return super().describe(percentiles=percentiles, include=include,
            exclude=exclude)

    def value_counts(self, normalize=False, sort=True, ascending=False,
        bins=None, dropna=True):
        """
        Return a Series or DataFrame containing counts of unique rows.
        """
        name = 'proportion' if normalize else 'count'
        if bins is None:
            result = self._value_counts(normalize=normalize, sort=sort,
                ascending=ascending, dropna=dropna)
            result.name = name
            return result
        from pandas.core.reshape.merge import get_join_indexers
        from pandas.core.reshape.tile import cut
        ids = self._grouper.ids
        val = self.obj._values
        index_names = self._grouper.names + [self.obj.name]
        if isinstance(val.dtype, CategoricalDtype
            ) or bins is not None and not isinstance(bins, Iterable):
            ser = self.apply(Series.value_counts, normalize=normalize, sort
                =sort, ascending=ascending, bins=bins)
            ser.name = name
            ser.index.names = index_names
            return ser
        mask = ids != -1
        ids, val = ids[mask], val[mask]
        lab: Union[Index, np.ndarray]
        if bins is None:
            lab, lev = algorithms.factorize(val, sort=True)
            llab = lambda lab, inc: lab[inc]
        else:
            cat_ser = cut(Series(val, copy=False), bins, include_lowest=True)
            cat_obj = cast('Categorical', cat_ser._values)
            lev = cat_obj.categories
            lab = lev.take(cat_obj.codes, allow_fill=True, fill_value=lev.
                _na_value)
            llab = lambda lab, inc: lab[inc]._multiindex.codes[-1]
        if isinstance(lab.dtype, IntervalDtype):
            lab_interval = cast(Interval, lab)
            sorter = np.lexsort((lab_interval.left, lab_interval.right, ids))
        else:
            sorter = np.lexsort((lab, ids))
        ids, lab = ids[sorter], lab[sorter]
        idchanges = 1 + np.nonzero(ids[1:] != ids[:-1])[0]
        idx = np.r_[0, idchanges]
        if not len(ids):
            idx = idchanges
        lchanges = llab(lab, slice(1, None)) != llab(lab, slice(None, -1))
        inc = np.r_[True, lchanges]
        if not len(val):
            inc = lchanges
        inc[idx] = True
        out = np.diff(np.nonzero(np.r_[inc, True])[0])
        rep = partial(np.repeat, repeats=np.add.reduceat(inc, idx))
        if isinstance(self._grouper.result_index, MultiIndex):
            codes_list = list(self._grouper.result_index.codes)
        else:
            codes_list = [algorithms.factorize(self._grouper.result_index,
                sort=self._grouper._sort, use_na_sentinel=self._grouper.
                dropna)[0]]
        codes_list = [rep(level_codes) for level_codes in codes_list] + [llab
            (lab, inc)]
        levels = self._grouper.levels + [lev]
        if dropna:
            mask = codes_list[-1] != -1
            if mask.any():
                out, codes_list = out[mask], [level_codes[mask] for
                    level_codes in codes_list]
        mask_final = duplicated(np.stack(codes_list, axis=1), 'first')
        res = np.bincount(ids[~mask_final], minlength=self._grouper.ngroups)
        res = ensure_int64(res)
        if is_integer_dtype(res.dtype):
            res = ensure_int64(res)
        result = self.obj._constructor(res, index=self._grouper.
            result_index, name=self.obj.name)
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return result

    @doc(Series.describe)
    def describe(self, percentiles=None, include=None, exclude=None):
        return super().describe(percentiles=percentiles, include=include,
            exclude=exclude)

    def transform(self, func, *args: Any, engine: Optional[str]=None,
        engine_kwargs: Optional[Dict[str, bool]]=None, **kwargs: Any):
        return self._transform(func, *args, engine=engine, engine_kwargs=
            engine_kwargs, **kwargs)

    def _define_paths(self, func, *args: Any, **kwargs: Any):
        if isinstance(func, str):
            fast_path = lambda group: getattr(group, func)(*args, **kwargs)
            slow_path = lambda group: group.apply(lambda x: getattr(x, func
                )(*args, **kwargs), axis=0)
        else:
            fast_path = lambda group: func(group, *args, **kwargs)
            slow_path = lambda group: group.apply(lambda x: func(x, *args,
                **kwargs), axis=0)
        return fast_path, slow_path

    def _choose_path(self, fast_path, slow_path, group):
        path = slow_path
        res = slow_path(group)
        if self.ngroups == 1:
            return path, res
        try:
            res_fast = fast_path(group)
        except AssertionError:
            raise
        except Exception:
            return path, res
        if isinstance(res_fast, DataFrame):
            if not res_fast.columns.equals(group.columns):
                return path, res
        elif isinstance(res_fast, Series):
            if not res_fast.index.equals(group.columns):
                return path, res
        else:
            return path, res
        if res_fast.equals(res):
            path = fast_path
        return path, res

    def _compare_results(self, fast_res, slow_res):
        return fast_res == slow_res

    def _wrap_applied_output_series(self, values, not_indexed_same,
        first_not_none, key_index, is_transform):
        from pandas import concat
        kwargs = first_not_none._construct_axes_dict()
        backup = Series(**kwargs)
        values = [(x if x is not None else backup) for x in values]
        all_indexed_same = all_indexes_same(x.index for x in values)
        if not all_indexed_same:
            return self._concat_objects(values, not_indexed_same=True,
                is_transform=is_transform)
        stacked_values = np.vstack([np.asarray(v) for v in values])
        index = key_index
        columns = first_not_none.index.copy()
        if columns.name is None:
            names = {v.name for v in values}
            if len(names) == 1:
                columns.name = next(iter(names))
        if isinstance(stacked_values.dtype, object):
            stacked_values = stacked_values.tolist()
        result = self.obj._constructor(stacked_values, index=index, columns
            =columns)
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
        return result
    __examples_series_doc = dedent(
        """
    >>> ser = pd.Series([0, 1, 2], index="a a b".split())
    >>> g1 = ser.groupby(s.index, group_keys=False)
    >>> g2 = ser.groupby(s.index, group_keys=True)

    From ``s`` above we can see that ``g`` has two groups, ``a`` and ``b``.
    Notice that ``g1`` have ``g2`` have two groups, ``a`` and ``b``, and only
    differ in their ``group_keys`` argument. Calling `apply` in various ways,
    we can get different grouping results:

    Example 1: The function passed to `apply` takes a Series as
    its argument and returns a Series.  `apply` combines the result for
    each group together into a new Series.

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the passed ``func``.

    >>> g1.apply(lambda x: x * 2 if x.name == "a" else x / 2)
    a    0.0
    a    2.0
    b    1.0
    dtype: float64

    In the above, the groups are not part of the index. We can have them included
    by using ``g2`` where ``group_keys=True``:

    >>> g2.apply(lambda x: x * 2 if x.name == "a" else x / 2)
    a  a    0.0
       a    2.0
    b  b    1.0
    dtype: float64

    Example 2: The function passed to `apply` takes a Series as
    its argument and returns a scalar. `apply` combines the result for
    each group together into a Series, including setting the index as
    appropriate:

    >>> g1.apply(lambda x: x.max() - x.min())
    a    1
    b    0
    dtype: int64

    The ``group_keys`` argument has no effect here because the result is not
    like-indexed (i.e. :ref:`a transform <groupby.transform>`) when compared
    to the input.

    >>> g2.apply(lambda x: x.max() - x.min())
    a    1
    b    0
    dtype: int64
    """
        )

    @Substitution(klass='Series', example=__examples_series_doc)
    @Appender(_transform_template)
    def transform(self, func, *args: Any, engine: Optional[str]=None,
        engine_kwargs: Optional[Dict[str, bool]]=None, **kwargs: Any):
        return self._transform(func, *args, engine=engine, engine_kwargs=
            engine_kwargs, **kwargs)

    def filter(self, func, dropna=True, *args: Any, **kwargs: Any):
        """
        Filter elements from groups that don't satisfy a criterion.
        """
        if isinstance(func, str):
            wrapper = lambda x: getattr(x, func)(*args, **kwargs)
        else:
            wrapper = lambda x: func(x, *args, **kwargs)

        def true_and_notna(x):
            b = wrapper(x)
            return notna(b) and b
        try:
            indices: List[np.ndarray] = [self._get_index(name) for name,
                group in self._grouper.get_iterator(self.
                _obj_with_exclusions) if true_and_notna(group)]
        except (ValueError, TypeError) as err:
            raise TypeError('the filter must return a boolean result') from err
        filtered = self._apply_filter(indices, dropna)
        return filtered

    def nunique(self, dropna=True):
        """
        Return number of unique elements in the group.
        """
        return super().nunique(dropna=dropna)

    def skew(self, skipna=True, numeric_only=False, **kwargs: Any):
        """
        Return unbiased skew within groups.

        Normalized by N-1.
        """
        return self._cython_agg_general('skew', alt=None, skipna=skipna,
            numeric_only=numeric_only, **kwargs)

    def kurt(self, skipna=True, numeric_only=False, **kwargs: Any):
        """
        Return unbiased kurtosis within groups.
        """

        def alt(obj):
            raise TypeError(f"'kurt' is not supported for dtype={obj.dtype}")
        return self._cython_agg_general('kurt', alt=alt, skipna=skipna,
            numeric_only=numeric_only, **kwargs)

    @property
    @doc(Series.plot.__doc__)
    def plot(self):
        result = GroupByPlot(self)
        return result

    @doc(Series.nlargest.__doc__)
    def nlargest(self, n=5, keep='first'):
        f = partial(Series.nlargest, n=n, keep=keep)
        data = self._obj_with_exclusions
        result = self._python_apply_general(f, data, not_indexed_same=True)
        return result

    @doc(Series.nsmallest.__doc__)
    def nsmallest(self, n=5, keep='first'):
        f = partial(Series.nsmallest, n=n, keep=keep)
        data = self._obj_with_exclusions
        result = self._python_apply_general(f, data, not_indexed_same=True)
        return result

    def idxmin(self, skipna=True):
        """
        Return the row label of the minimum value.
        """
        return self._idxmax_idxmin('idxmin', skipna=skipna)

    def idxmax(self, skipna=True):
        """
        Return the row label of the maximum value.
        """
        return self._idxmax_idxmin('idxmax', skipna=skipna)

    @doc(Series.corr.__doc__)
    def corr(self, other, method='pearson', min_periods=None):
        result = self._op_via_apply('corr', other=other, method=method,
            min_periods=min_periods)
        return result

    @doc(Series.cov.__doc__)
    def cov(self, other, min_periods=None, ddof=1):
        result = self._op_via_apply('cov', other=other, min_periods=
            min_periods, ddof=ddof)
        return result

    @property
    def is_monotonic_increasing(self):
        """
        Return whether each group's values are monotonically increasing.
        """
        return self.apply(lambda ser: ser.is_monotonic_increasing)

    @property
    def is_monotonic_decreasing(self):
        """
        Return whether each group's values are monotonically decreasing.
        """
        return self.apply(lambda ser: ser.is_monotonic_decreasing)

    @doc(Series.hist.__doc__)
    def hist(self, by=None, ax=None, grid=True, xlabelsize=None, xrot=None,
        ylabelsize=None, yrot=None, figsize=None, bins=10, backend=None,
        legend=False, **kwargs: Any):
        result = self._op_via_apply('hist', by=by, ax=ax, grid=grid,
            xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=
            yrot, figsize=figsize, bins=bins, backend=backend, legend=
            legend, **kwargs)
        return result

    @property
    @doc(Series.dtype.__doc__)
    def dtype(self):
        return self.apply(lambda ser: ser.dtype)

    def unique(self):
        """
        Return unique values for each group.
        """
        result = self._op_via_apply('unique')
        return result


@set_module('pandas.api.typing')
class DataFrameGroupBy(GroupBy[DataFrame]):
    _agg_examples_doc = dedent(
        """
    Examples
    --------
    >>> data = {"A": [1, 1, 2, 2],
    ...         "B": [1, 2, 3, 4],
    ...         "C": [0.362838, 0.227877, 1.267767, -0.562860]}
    >>> df = pd.DataFrame(data)
    >>> df
       A  B         C
    0  1  1  0.362838
    1  1  2  0.227877
    2  2  3  1.267767
    3  2  4 -0.562860

    The aggregation is for each column.

    >>> df.groupby('A').agg('min')
       B         C
    A
    1  1  0.227877
    2  3 -0.562860

    Multiple aggregations

    >>> df.groupby('A').agg(['min', 'max'])
        B             C
      min max       min       max
    A
    1   1   2  0.227877  0.362838
    2   3   4 -0.562860  1.267767

    Select a column for aggregation

    >>> df.groupby('A').B.agg(['min', 'max'])
       min  max
    A
    1    1    2
    2    3    4

    User-defined function for aggregation

    >>> df.groupby('A').agg(lambda x: sum(x) + 2)
        B	       C
    A
    1	5	2.590715
    2	9	2.704907

    Different aggregations per column

    >>> df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'})
        B             C
      min max       sum
    A
    1   1   2  0.590715
    2   3   4  0.704907

    To control the output names with different aggregations per column,
    pandas supports "named aggregation"

    >>> df.groupby("A").agg(
    ...     b_min=pd.NamedAgg(column="B", aggfunc="min"),
    ...     c_sum=pd.NamedAgg(column="C", aggfunc="sum")
    ... )
       b_min     c_sum
    A
    1      1  0.590715
    2      3  0.704907

    - The keywords are the *output* column names
    - The values are tuples whose first element is the column to select
      and the second element is the aggregation to apply to that column.
      Pandas provides the ``pandas.NamedAgg`` namedtuple with the fields
      ``['column', 'aggfunc']`` to make it clearer what the arguments are.
      As usual, the aggregation can be a callable or a string alias.

    See :ref:`groupby.aggregate.named` for more.

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the aggregating function.

    >>> df.groupby("A")[["B"]].agg(lambda x: x.astype(float).min())
          B
    A
    1   1.0
    2   3.0
    """
        )

    def aggregate(self, func=None, *args: Any, engine: Optional[str]=None,
        engine_kwargs: Optional[Dict[str, bool]]=None, **kwargs: Any):
        """
        Aggregate using one or more operations.

        The ``aggregate`` function allows the application of one or more aggregation
        operations on groups of data within a DataFrameGroupBy object. It supports
        various aggregation methods, including user-defined functions and predefined
        functions such as 'sum', 'mean', etc.
        """
        relabeling, func, columns, order = reconstruct_func(func, **kwargs)
        func = maybe_mangle_lambdas(func)
        if maybe_use_numba(engine):
            kwargs['engine'] = engine
            kwargs['engine_kwargs'] = engine_kwargs
        op = GroupByApply(self, func, args=args, kwargs=kwargs)
        result = op.agg()
        if not is_dict_like(func) and result is not None:
            if not self.as_index and isinstance(func, Iterable):
                return result.reset_index()
            else:
                return result
        elif relabeling:
            result = cast(DataFrame, result)
            result = result.iloc[:, order]
            result = cast(DataFrame, result)
            result.columns = cast(List[str], columns)
        if result is None:
            if 'engine' in kwargs:
                del kwargs['engine']
                del kwargs['engine_kwargs']
            if maybe_use_numba(engine):
                return self._aggregate_with_numba(func, *args,
                    engine_kwargs=engine_kwargs, **kwargs)
            if self._grouper.nkeys > 1:
                return self._python_agg_general(func, *args, **kwargs)
            elif args or kwargs:
                result = self._aggregate_frame(func, *args, **kwargs)
            else:
                gba = GroupByApply(self, [func], args=(), kwargs={})
                try:
                    result = gba.agg()
                except ValueError as err:
                    if 'No objects to concatenate' not in str(err):
                        raise
                    result = self._aggregate_frame(func)
                else:
                    result = cast(DataFrame, result)
                    result.columns = self._obj_with_exclusions.columns.copy()
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return result
    agg = aggregate

    def _python_agg_general(self, func, *args: Any, **kwargs: Any):
        f = lambda x: func(x, *args, **kwargs)
        if self.ngroups == 0:
            return self._python_apply_general(f, self._selected_obj, is_agg
                =True)
        obj = self._obj_with_exclusions
        if not len(obj.columns):
            return self._python_apply_general(f, self._selected_obj)
        output: Dict[int, ArrayLike] = {}
        for idx, (name, ser) in enumerate(obj.items()):
            result = self._grouper.agg_series(ser, f)
            output[idx] = result
        res = self.obj._constructor(output)
        res.columns = obj.columns.copy(deep=False)
        return self._wrap_aggregated_output(res)

    def _aggregate_frame(self, func, *args: Any, **kwargs: Any):
        if self._grouper.nkeys != 1:
            raise AssertionError('Number of keys must be 1')
        obj = self._obj_with_exclusions
        result: Dict[Hashable, Union[NDFrame, np.ndarray]] = {}
        for name, grp_df in self._grouper.get_iterator(obj):
            fres = func(grp_df, *args, **kwargs)
            result[name] = fres
        result_index = self._grouper.result_index
        out = self.obj._constructor(result, index=obj.columns, columns=
            result_index)
        out = out.T
        return out

    def _wrap_applied_output(self, data, values, not_indexed_same=False,
        is_transform=False):
        if len(values) == 0:
            if is_transform:
                res_index = data.index
            elif not self.group_keys:
                res_index = None
            else:
                res_index = self._grouper.result_index
            result = self.obj._constructor(index=res_index, columns=data.
                columns)
            result = result.astype(data.dtypes)
            return result
        first_not_none = next(com.not_none(*values), None)
        if first_not_none is None:
            result = self.obj._constructor(columns=data.columns)
            result = result.astype(data.dtypes)
            return result
        elif isinstance(first_not_none, DataFrame):
            return self._concat_objects(values, not_indexed_same=
                not_indexed_same, is_transform=is_transform)
        key_index = self._grouper.result_index if self.as_index else None
        if isinstance(first_not_none, (np.ndarray, Index)):
            if not is_hashable(self._selection):
                name: Hashable = tuple(self._selection)
            else:
                name = self._selection
            return self.obj._constructor_sliced(values, index=key_index,
                name=name)
        elif not isinstance(first_not_none, Series):
            if self.as_index:
                return self.obj._constructor_sliced(values, index=key_index)
            else:
                result = self.obj._constructor(values, columns=[self.
                    _selection])
                result = self._insert_inaxis_grouper(result)
                return result
        else:
            return self._wrap_applied_output_series(values,
                not_indexed_same, first_not_none, key_index, is_transform)

    def _wrap_applied_output_series(self, values, not_indexed_same,
        first_not_none, key_index, is_transform):
        from pandas import concat
        kwargs = first_not_none._construct_axes_dict()
        backup = Series(**kwargs)
        values = [(x if x is not None else backup) for x in values]
        all_indexed_same = all_indexes_same(x.index for x in values)
        if not all_indexed_same:
            return self._concat_objects(values, not_indexed_same=True,
                is_transform=is_transform)
        stacked_values = np.vstack([np.asarray(v) for v in values])
        index = key_index
        columns = first_not_none.index.copy()
        if columns.name is None:
            names = {v.name for v in values}
            if len(names) == 1:
                columns.name = next(iter(names))
        if isinstance(stacked_values.dtype, object):
            stacked_values = stacked_values.tolist()
        result = self.obj._constructor(stacked_values, index=index, columns
            =columns)
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
        return result

    def hist(self, column=None, by=None, grid=True, xlabelsize=None, xrot=
        None, ylabelsize=None, yrot=None, ax=None, sharex=False, sharey=
        False, figsize=None, layout=None, bins=10, backend=None, legend=
        False, **kwargs: Any):
        """
        Make a histogram of the DataFrame's columns.

        A `histogram`_ is a representation of the distribution of data.
        This function calls :meth:`matplotlib.pyplot.hist`, on each series in
        the DataFrame, resulting in one histogram per column.

        .. _histogram: https://en.wikipedia.org/wiki/Histogram

        Parameters
        ----------
        column : str or sequence, optional
            If passed, will be used to limit data to a subset of columns.
        by : object, optional
            If passed, then used to form histograms for separate groups.
        grid : bool, default True
            Whether to show axis grid lines.
        xlabelsize : int, default None
            If specified changes the x-axis label size.
        xrot : float, default None
            Rotation of x axis labels. For example, a value of 90 displays the
            x labels rotated 90 degrees clockwise.
        ylabelsize : int, default None
            If specified changes the y-axis label size.
        yrot : float, default None
            Rotation of y axis labels. For example, a value of 90 displays the
            y labels rotated 90 degrees clockwise.
        ax : Matplotlib axes object, default None
            The axes to plot the histogram on.
        sharex : bool, default True if ax is None else False
            In case subplots=True, share x axis and set some x axis labels to
            invisible; defaults to True if ax is None otherwise False if an ax
            is passed in.
            Note that passing in both an ax and sharex=True will alter all x axis
            labels for all subplots in a figure.
        sharey : bool, default False
            In case subplots=True, share y axis and set some y axis labels to
            invisible.
        figsize : tuple, optional
            The size in inches of the figure to create. Uses the value in
            `matplotlib.rcParams` by default.
        layout : tuple, optional
            Tuple of (rows, columns) for the layout of the histograms.
        bins : int or sequence, default 10
            Number of histogram bins to be used. If an integer is given, bins + 1
            bin edges are calculated and returned. If bins is a sequence, gives
            bin edges, including left edge of first bin and right edge of last
            bin. In this case, bins is returned unmodified.

        backend : str, default None
            Backend to use instead of the backend specified in the option
            ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
            specify the ``plotting.backend`` for the whole session, set
            ``pd.options.plotting.backend``.
        
        legend : bool, default False
            Whether to show the legend.

        **kwargs : Any
            All other plotting keyword arguments to be passed to
            :meth:`matplotlib.pyplot.hist`.

        Returns
        -------
        matplotlib.Axes or numpy.ndarray
            A ``matplotlib.Axes`` object or an array of ``Axes`` objects, depending on
            the layout and grouping.

        See Also
        --------
        matplotlib.pyplot.hist : Plot a histogram using matplotlib.

        Examples
        --------
        This example draws a histogram based on the length and width of
        some animals, displayed in three bins

        .. plot::
            :context: close-figs

            >>> data = {
            ...     "length": [1.5, 0.5, 1.2, 0.9, 3],
            ...     "width": [0.7, 0.2, 0.15, 0.2, 1.1],
            ... }
            >>> index = ["pig", "rabbit", "duck", "chicken", "horse"]
            >>> df = pd.DataFrame(data, index=index)
            >>> hist = df.groupby("length").hist(bins=3)
        """
        result = self._op_via_apply('hist', column=column, by=by, grid=grid,
            xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=
            yrot, figsize=figsize, bins=bins, backend=backend, legend=
            legend, **kwargs)
        return result

    def take(self, indices, **kwargs: Any):
        """
        Return the elements in the given *positional* indices in each group.
        """
        result = self._op_via_apply('take', indices=indices, **kwargs)
        return result

    def skew(self, skipna=True, numeric_only=False, **kwargs: Any):
        """
        Return unbiased skew within groups.

        Normalized by N-1.
        """
        return self._cython_agg_general('skew', alt=None, skipna=skipna,
            numeric_only=numeric_only, **kwargs)

    def kurt(self, skipna=True, numeric_only=False, **kwargs: Any):
        """
        Return unbiased kurtosis within groups.
        """
        return self._cython_agg_general('kurt', alt=None, skipna=skipna,
            numeric_only=numeric_only, **kwargs)

    @property
    @doc(DataFrame.plot.__doc__)
    def plot(self):
        result = GroupByPlot(self)
        return result

    @doc(DataFrame.corr.__doc__)
    def corr(self, method='pearson', min_periods=1, numeric_only=False):
        result = self._op_via_apply('corr', method=method, min_periods=
            min_periods, numeric_only=numeric_only)
        return result

    @doc(DataFrame.cov.__doc__)
    def cov(self, min_periods=None, ddof=1, numeric_only=False):
        result = self._op_via_apply('cov', min_periods=min_periods, ddof=
            ddof, numeric_only=numeric_only)
        return result

    def value_counts(self, subset=None, normalize=False, sort=True,
        ascending=False, dropna=True):
        """
        Return a Series or DataFrame containing counts of unique rows.
        """
        return self._value_counts(subset, normalize, sort, ascending, dropna)

    def take(self, indices, **kwargs: Any):
        """
        Return the elements in the given *positional* indices in each group.
        """
        result = self._op_via_apply('take', indices=indices, **kwargs)
        return result

    def skew(self, skipna=True, numeric_only=False, **kwargs: Any):
        """
        Return unbiased skew within groups.

        Normalized by N-1.
        """
        return self._cython_agg_general('skew', alt=None, skipna=skipna,
            numeric_only=numeric_only, **kwargs)

    def kurt(self, skipna=True, numeric_only=False, **kwargs: Any):
        """
        Return unbiased kurtosis within groups.
        """
        return self._cython_agg_general('kurt', alt=None, skipna=skipna,
            numeric_only=numeric_only, **kwargs)

    @property
    @doc(DataFrame.plot.__doc__)
    def plot(self):
        result = GroupByPlot(self)
        return result

    @doc(DataFrame.corr.__doc__)
    def corr(self, method='pearson', min_periods=1, numeric_only=False):
        result = self._op_via_apply('corr', method=method, min_periods=
            min_periods, numeric_only=numeric_only)
        return result

    @doc(DataFrame.cov.__doc__)
    def cov(self, min_periods=None, ddof=1, numeric_only=False):
        result = self._op_via_apply('cov', min_periods=min_periods, ddof=
            ddof, numeric_only=numeric_only)
        return result

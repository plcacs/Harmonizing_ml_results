"""
Define the SeriesGroupBy and DataFrameGroupBy
classes that hold the groupby interfaces (and some implementations).

These are user facing as the result of the ``df.groupby(...)`` operations,
which here returns a DataFrameGroupBy object.
"""
from __future__ import annotations
from collections import abc
from collections.abc import Callable
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, NamedTuple, Optional, Tuple, TypeVar, Union, cast
import warnings
import numpy as np
from pandas._libs import Interval
from pandas._libs.hashtable import duplicated
from pandas.errors import SpecificationError
from pandas.util._decorators import Appender, Substitution, doc, set_module
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
    ensure_int64,
    is_bool,
    is_dict_like,
    is_integer_dtype,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
)
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


@set_module('pandas.api.typing')
class SeriesGroupBy(GroupBy[Series]):

    def _wrap_agged_manager(self, mgr: BlockManager) -> Series:
        out = self.obj._constructor_from_mgr(mgr, axes=mgr.axes)
        out._name = self.obj.name
        return out

    def _get_data_to_aggregate(
        self, *, numeric_only: bool = False, name: Optional[str] = None
    ) -> BlockManager:
        ser = self._obj_with_exclusions
        single = ser._mgr
        if numeric_only and (not is_numeric_dtype(ser.dtype)):
            kwd_name = 'numeric_only'
            raise TypeError(
                f'Cannot use {kwd_name}=True with {type(self).__name__}.{name} and non-numeric dtypes.'
            )
        return single

    _agg_examples_doc = dedent(
        "\n    Examples\n    --------\n    >>> s = pd.Series([1, 2, 3, 4])\n\n    >>> s\n    0    1\n    1    2\n    2    3\n    3    4\n    dtype: int64\n\n    >>> s.groupby([1, 1, 2, 2]).min()\n    1    1\n    2    3\n    dtype: int64\n\n    >>> s.groupby([1, 1, 2, 2]).agg('min')\n    1    1\n    2    3\n    dtype: int64\n\n    >>> s.groupby([1, 1, 2, 2]).agg(['min', 'max'])\n       min  max\n    1    1    2\n    2    3    4\n\n    The output column names can be controlled by passing\n    the desired column names and aggregations as keyword arguments.\n\n    >>> s.groupby([1, 1, 2, 2]).agg(\n    ...     minimum='min',\n    ...     maximum='max',\n    ... )\n       minimum  maximum\n    1        1        2\n    2        3        4\n\n    .. versionchanged:: 1.3.0\n\n        The resulting dtype will reflect the return value of the aggregating function.\n\n    >>> s.groupby([1, 1, 2, 2]).agg(lambda x: x.astype(float).min())\n    1    1.0\n    2    3.0\n    dtype: float64\n    "
    )

    def apply(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Union[Series, DataFrame]:
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
        ...
        """
        return super().apply(func, *args, **kwargs)

    def aggregate(
        self,
        func: Optional[AggScalar] = None,
        *args: Any,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, bool]] = None,
        **kwargs: Any,
    ) -> Union[Series, DataFrame]:
        """
        Aggregate using one or more operations.
        ...
        """
        relabeling = func is None
        columns: Optional[Sequence[Any]] = None
        if relabeling:
            columns, func = validate_func_kwargs(kwargs)
            kwargs = {}
        if isinstance(func, str):
            if maybe_use_numba(engine) and engine is not None:
                kwargs['engine'] = engine
            if engine_kwargs is not None:
                kwargs['engine_kwargs'] = engine_kwargs
            return getattr(self, func)(*args, **kwargs)
        elif isinstance(func, abc.Iterable):
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
                return self._aggregate_with_numba(
                    func, *args, engine_kwargs=engine_kwargs, **kwargs
                )
            if self.ngroups == 0:
                obj = self._obj_with_exclusions
                return self.obj._constructor(
                    [],
                    name=self.obj.name,
                    index=self._grouper.result_index,
                    dtype=obj.dtype,
                )
            return self._python_agg_general(func, *args, **kwargs)

    agg = aggregate

    def _python_agg_general(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Series:
        f = lambda x: func(x, *args, **kwargs)
        obj = self._obj_with_exclusions
        result = self._grouper.agg_series(obj, f)
        res: Series = obj._constructor(result, name=obj.name)
        return self._wrap_aggregated_output(res)

    def _aggregate_multiple_funcs(
        self, arg: Iterable[AggScalar], *args: Any, **kwargs: Any
    ) -> Union[DataFrame, Series]:
        if isinstance(arg, dict):
            raise SpecificationError('nested renamer is not supported')
        if any(isinstance(x, (tuple, list)) for x in arg):
            arg = ((x, x) if not isinstance(x, (tuple, list)) else x for x in arg)
        else:
            columns = (com.get_callable_name(f) or f for f in arg)
            arg = zip(columns, arg)
        results: Dict[base.OutputKey, Union[Series, DataFrame]] = {}
        with com.temp_setattr(self, 'as_index', True):
            for idx, (name, func) in enumerate(arg):
                key = base.OutputKey(label=name, position=idx)
                results[key] = self.aggregate(func, *args, **kwargs)
        if any(isinstance(x, DataFrame) for x in results.values()):
            from pandas import concat

            res_df = concat(
                results.values(), axis=1, keys=[key.label for key in results]
            )
            return res_df
        indexed_output: Dict[int, Series] = {key.position: val for key, val in results.items()}
        output: Dict[int, Any] = indexed_output
        out = self.obj._constructor_expanddim(indexed_output, index=None)
        out.columns = Index((key.label for key in results))
        return out

    def _wrap_applied_output(
        self,
        data: Series,
        values: List[Any],
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ) -> Union[Series, DataFrame]:
        if len(values) == 0:
            if is_transform:
                res_index = data.index
            elif not self.group_keys:
                res_index = None
            else:
                res_index = self._grouper.result_index
            return self.obj._constructor(
                [], name=self.obj.name, index=res_index, dtype=data.dtype
            )
        assert values is not None
        if isinstance(values[0], dict):
            index = self._grouper.result_index
            res_df = self.obj._constructor_expanddim(values, index=index)
            res_ser: Series = res_df.stack()
            res_ser.name = self.obj.name
            return res_ser
        elif isinstance(values[0], (Series, DataFrame)):
            result = self._concat_objects(
                values, not_indexed_same=not_indexed_same, is_transform=is_transform
            )
            if isinstance(result, Series):
                result.name = self.obj.name
            if not self.as_index and not_indexed_same:
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return result
        else:
            result = self.obj._constructor(
                data=values, index=self._grouper.result_index, name=self.obj.name
            )
            if not self.as_index:
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return result

    __examples_series_doc = dedent(
        '\n    >>> ser = pd.Series([390.0, 350.0, 30.0, 20.0],\n    ...                 index=["Falcon", "Falcon", "Parrot", "Parrot"],\n    ...                 name="Max Speed")\n    >>> grouped = ser.groupby([1, 1, 2, 2])\n    >>> grouped.transform(lambda x: (x - x.mean()) / x.std())\n        Falcon    0.707107\n        Falcon   -0.707107\n        Parrot    0.707107\n        Parrot   -0.707107\n        Name: Max Speed, dtype: float64\n\n    Broadcast result of the transformation\n\n    >>> grouped.transform(lambda x: x.max() - x.min())\n    Falcon    40.0\n    Falcon    40.0\n    Parrot    10.0\n    Parrot    10.0\n    Name: Max Speed, dtype: float64\n\n    >>> grouped.transform("mean")\n    Falcon    370.0\n    Falcon    370.0\n    Parrot     25.0\n    Parrot     25.0\n    Name: Max Speed, dtype: float64\n\n    .. versionchanged:: 1.3.0\n\n    The resulting dtype will reflect the return value of the passed ``func``,\n    for example:\n\n    >>> grouped.transform(lambda x: x.astype(int).max())\n    Falcon    390\n    Falcon    390\n    Parrot     30\n    Parrot     30\n    Name: Max Speed, dtype: int64\n    '
    )

    @Substitution(klass='Series', example=__examples_series_doc)
    @Appender(_transform_template)
    def transform(
        self,
        func: Callable[..., Any],
        *args: Any,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, bool]] = None,
        **kwargs: Any,
    ) -> Union[Series, DataFrame]:
        return self._transform(func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs)

    def _cython_transform(
        self, how: str, numeric_only: bool = False, **kwargs: Any
    ) -> Series:
        obj = self._obj_with_exclusions
        try:
            result = self._grouper._cython_operation('transform', obj._values, how, 0, **kwargs)
        except NotImplementedError as err:
            raise TypeError(f'{how} is not supported for {obj.dtype} dtype') from err
        return obj._constructor(result, index=self.obj.index, name=obj.name)

    def _transform_general(
        self,
        func: Callable[..., Any],
        engine: Optional[str],
        engine_kwargs: Optional[Dict[str, bool]],
        *args: Any,
        **kwargs: Any,
    ) -> Series:
        if maybe_use_numba(engine):
            return self._transform_with_numba(
                func, *args, engine_kwargs=engine_kwargs, **kwargs
            )
        assert callable(func)
        klass = type(self.obj)
        results: List[Any] = []
        for name, group in self._grouper.get_iterator(self._obj_with_exclusions):
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

    def filter(
        self,
        func: Callable[..., bool],
        dropna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series:
        """
        Filter elements from groups that don't satisfy a criterion.
        ...
        """
        if isinstance(func, str):
            wrapper = lambda x: getattr(x, func)(*args, **kwargs)
        else:
            wrapper = lambda x: func(x, *args, **kwargs)

        def true_and_notna(x: Series) -> bool:
            b = wrapper(x)
            return notna(b) and b

        try:
            indices = [
                self._get_index(name)
                for name, group in self._grouper.get_iterator(self._obj_with_exclusions)
                if true_and_notna(group)
            ]
        except (ValueError, TypeError) as err:
            raise TypeError('the filter must return a boolean result') from err
        filtered = self._apply_filter(indices, dropna)
        return filtered

    def nunique(self, dropna: bool = True) -> Series:
        """
        Return number of unique elements in the group.
        ...
        """
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups
        val = self.obj._values
        codes, uniques = algorithms.factorize(val, use_na_sentinel=dropna, sort=False)
        if self._grouper.has_dropped_na:
            mask = ids >= 0
            ids = ids[mask]
            codes = codes[mask]
        group_index = get_group_index(
            labels=[ids, codes],
            shape=(ngroups, len(uniques)),
            sort=False,
            xnull=dropna,
        )
        if dropna:
            mask = group_index >= 0
            if (~mask).any():
                ids = ids[mask]
                group_index = group_index[mask]
        mask = duplicated(group_index, 'first')
        res = np.bincount(ids[~mask], minlength=ngroups)
        res = ensure_int64(res)
        ri = self._grouper.result_index
        result: Series = self.obj._constructor(res, index=ri, name=self.obj.name)
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return result

    @doc(Series.describe)
    def describe(
        self,
        percentiles: Optional[List[float]] = None,
        include: Optional[Union[str, List[str]]] = None,
        exclude: Optional[Union[str, List[str]]] = None,
    ) -> Union[Series, DataFrame]:
        return super().describe(percentiles=percentiles, include=include, exclude=exclude)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins: Optional[Union[int, List[int]]] = None,
        dropna: bool = True,
    ) -> Union[Series, DataFrame]:
        """
        Return a Series or DataFrame containing counts of unique rows.
        ...
        """
        name = 'proportion' if normalize else 'count'
        if bins is None:
            result = self._value_counts(
                normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
            )
            result.name = name
            return result
        from pandas.core.reshape.merge import get_join_indexers
        from pandas.core.reshape.tile import cut

        ids = self._grouper.ids
        val = self.obj._values
        index_names = self._grouper.names + [self.obj.name]
        if (
            isinstance(val.dtype, CategoricalDtype)
            or (bins is not None and (not np.iterable(bins)))
        ):
            ser = self.apply(
                Series.value_counts,
                normalize=normalize,
                sort=sort,
                ascending=ascending,
                bins=bins,
            )
            ser.name = name
            ser.index.names = index_names
            return ser
        mask = ids != -1
        ids, val = cast(Tuple[np.ndarray, np.ndarray], (ids[mask], val[mask]))
        if bins is None:
            lab, lev = algorithms.factorize(val, sort=True)
            llab = lambda lab, inc: lab[inc]
        else:
            cat_ser = cut(Series(val, copy=False), bins, include_lowest=True)
            cat_obj = cast('Categorical', cat_ser._values)
            lev = cat_obj.categories
            lab = lev.take(cat_obj.codes, allow_fill=True, fill_value=lev._na_value)
            llab = lambda lab, inc: lab[inc]._multiindex.codes[-1]
        if isinstance(lab.dtype, IntervalDtype):
            lab_interval = cast(Interval, lab)
            sorter = np.lexsort((lab_interval.left, lab_interval.right, ids))
        else:
            sorter = np.lexsort((lab, ids))
        ids, lab = cast(Tuple[np.ndarray, np.ndarray], (ids[sorter], lab[sorter]))
        idchanges = 1 + np.nonzero(ids[1:] != ids[:-1])[0]
        idx = np.r_[0, idchanges]
        if not len(ids):
            idx = cast(np.ndarray, idchanges)
        lchanges = llab(lab, slice(1, None)) != llab(lab, slice(None, -1))
        inc = np.r_[True, lchanges]
        if not len(val):
            inc = cast(np.ndarray, lchanges)
        inc[idx] = True
        out = np.diff(np.nonzero(np.r_[inc, True])[0])
        rep = partial(np.repeat, repeats=np.add.reduceat(inc, idx))
        if isinstance(self._grouper.result_index, MultiIndex):
            codes = list(self._grouper.result_index.codes)
        else:
            codes = [
                algorithms.factorize(
                    self._grouper.result_index,
                    sort=self._grouper._sort,
                    use_na_sentinel=self._grouper.dropna,
                )[0]
            ]
        codes = [rep(level_codes) for level_codes in codes] + [
            llab(lab, inc)
        ]
        levels = self._grouper.levels + [lev]
        if dropna:
            mask = codes[-1] != -1
            if mask.all():
                dropna = False
            else:
                out, codes = cast(Tuple[np.ndarray, List[np.ndarray]], (
                    out[mask],
                    [level_codes[mask] for level_codes in codes],
                ))
        mask = duplicated(group_index := group_index := np.array(codes[-1]), 'first')
        res = np.bincount(ids[~mask], minlength=ngroups := self._grouper.ngroups)
        res = ensure_int64(res)
        ri = self._grouper.result_index
        result: Series = self.obj._constructor(res, index=ri, name=self.obj.name)
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return result

    @doc(Series.transform.__doc__)
    def transform(
        self,
        func: Callable[..., Any],
        *args: Any,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, bool]] = None,
        **kwargs: Any,
    ) -> Union[Series, DataFrame]:
        return self._transform(func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs)

    def _define_paths(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Tuple[Callable[[Series], Any], Callable[[Series], Any]]:
        if isinstance(func, str):
            fast_path = lambda group: getattr(group, func)(*args, **kwargs)
            slow_path = lambda group: group.apply(lambda x: getattr(x, func)(*args, **kwargs), axis=0)
        else:
            fast_path = lambda group: func(group, *args, **kwargs)
            slow_path = lambda group: group.apply(lambda x: func(x, *args, **kwargs), axis=0)
        return fast_path, slow_path

    def _choose_path(
        self,
        fast_path: Callable[[Series], Any],
        slow_path: Callable[[Series], Any],
        group: Series,
    ) -> Tuple[Callable[[Series], Any], Any]:
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

    def skew(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Series:
        """
        Return unbiased skew within groups.
        ...
        """
        return self._cython_agg_general(
            'skew', alt=None, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def kurt(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Series:
        """
        Return unbiased kurtosis within groups.
        ...
        """

        def alt(obj: Series) -> None:
            raise TypeError(f"'kurt' is not supported for dtype={obj.dtype}")

        return self._cython_agg_general(
            'kurt', alt=alt, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    @property
    @doc(Series.plot.__doc__)
    def plot(self) -> GroupByPlot:
        result = GroupByPlot(self)
        return result

    @doc(Series.nlargest.__doc__)
    def nlargest(
        self,
        n: int = 5,
        keep: Literal['first', 'last', 'all'] = 'first',
    ) -> Series:
        f = partial(Series.nlargest, n=n, keep=keep)
        data = self._obj_with_exclusions
        result = self._python_apply_general(f, data, not_indexed_same=True)
        return result

    @doc(Series.nsmallest.__doc__)
    def nsmallest(
        self,
        n: int = 5,
        keep: Literal['first', 'last', 'all'] = 'first',
    ) -> Series:
        f = partial(Series.nsmallest, n=n, keep=keep)
        data = self._obj_with_exclusions
        result = self._python_apply_general(f, data, not_indexed_same=True)
        return result

    def idxmin(
        self,
        skipna: bool = True,
    ) -> Series:
        """
        Return the row label of the minimum value.
        ...
        """
        return self._idxmax_idxmin('idxmin', skipna=skipna)

    def idxmax(
        self,
        skipna: bool = True,
    ) -> Series:
        """
        Return the row label of the maximum value.
        ...
        """
        return self._idxmax_idxmin('idxmax', skipna=skipna)

    @doc(Series.corr.__doc__)
    def corr(
        self,
        other: Series,
        method: CorrelationMethod = 'pearson',
        min_periods: Optional[int] = None,
    ) -> Series:
        result = self._op_via_apply(
            'corr', other=other, method=method, min_periods=min_periods
        )
        return result

    @doc(Series.cov.__doc__)
    def cov(
        self,
        other: Series,
        min_periods: Optional[int] = None,
        ddof: int = 1,
    ) -> Series:
        result = self._op_via_apply(
            'cov', other=other, min_periods=min_periods, ddof=ddof
        )
        return result

    @property
    def is_monotonic_increasing(self) -> Series:
        """
        Return whether each group's values are monotonically increasing.
        ...
        """
        return self.apply(lambda ser: ser.is_monotonic_increasing)

    @property
    def is_monotonic_decreasing(self) -> Series:
        """
        Return whether each group's values are monotonically decreasing.
        ...
        """
        return self.apply(lambda ser: ser.is_monotonic_decreasing)

    @doc(Series.hist.__doc__)
    def hist(
        self,
        by: Optional[Any] = None,
        ax: Optional[Any] = None,
        grid: bool = True,
        xlabelsize: Optional[int] = None,
        xrot: Optional[float] = None,
        ylabelsize: Optional[int] = None,
        yrot: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        bins: Union[int, Sequence[int]] = 10,
        backend: Optional[str] = None,
        legend: bool = False,
        **kwargs: Any,
    ) -> Any:
        result = self._op_via_apply(
            'hist',
            by=by,
            ax=ax,
            grid=grid,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            figsize=figsize,
            bins=bins,
            backend=backend,
            legend=legend,
            **kwargs,
        )
        return result

    @property
    @doc(Series.dtype.__doc__)
    def dtype(self) -> Series:
        return self.apply(lambda ser: ser.dtype)

    def unique(self) -> Series:
        """
        Return unique values for each group.
        ...
        """
        result = self._op_via_apply('unique')
        return result


@set_module('pandas.api.typing')
class DataFrameGroupBy(GroupBy[DataFrame]):
    _agg_examples_doc = dedent(
        '\n    Examples\n    --------\n    >>> data = {"A": [1, 1, 2, 2],\n    ...         "B": [1, 2, 3, 4],\n    ...         "C": [0.362838, 0.227877, 1.267767, -0.562860]}\n    >>> df = pd.DataFrame(data)\n    >>> df\n       A  B         C\n    0  1  1  0.362838\n    1  1  2  0.227877\n    2  2  3  1.267767\n    3  2  4 -0.562860\n\n    The aggregation is for each column.\n\n    >>> df.groupby(\'A\').agg(\'min\')\n       B         C\n    A\n    1  1  0.227877\n    2  3 -0.562860\n\n    Multiple aggregations\n\n    >>> df.groupby(\'A\').agg([\'min\', \'max\'])\n        B             C\n      min max       min       max\n    A\n    1   1   2  0.227877  0.362838\n    2   3   4 -0.562860  1.267767\n\n    Select a column for aggregation\n\n    >>> df.groupby(\'A\').B.agg([\'min\', \'max\'])\n       min  max\n    A\n    1    1    2\n    2    3    4\n\n    User-defined function for aggregation\n\n    >>> df.groupby(\'A\').agg(lambda x: sum(x) + 2)\n        B\t       C\n    A\n    1\t5\t2.590715\n    2\t9\t2.704907\n\n    Different aggregations per column\n\n    >>> df.groupby(\'A\').agg({\'B\': [\'min\', \'max\'], \'C\': \'sum\'})\n        B             C\n      min max       sum\n    A\n    1   1   2  0.590715\n    2   3   4  0.704907\n\n    To control the output names with different aggregations per column,\n    pandas supports "named aggregation"\n\n    >>> df.groupby("A").agg(\n    ...     b_min=pd.NamedAgg(column="B", aggfunc="min"),\n    ...     c_sum=pd.NamedAgg(column="C", aggfunc="sum")\n    ... )\n       b_min     c_sum\n    A\n    1      1  0.590715\n    2      3  0.704907\n\n    - The keywords are the *output* column names\n    - The values are tuples whose first element is the column to select\n      and the second element is the aggregation to apply to that column.\n      Pandas provides the ``pandas.NamedAgg`` namedtuple with the fields\n      ``[\'column\', \'aggfunc\']`` to make it clearer what the arguments are.\n      As usual, the aggregation can be a callable or a string alias.\n\n    See :ref:`groupby.aggregate.named` for more.\n\n    .. versionchanged:: 1.3.0\n\n        The resulting dtype will reflect the return value of the aggregating function.\n\n    >>> df.groupby("A")[["B"]].agg(lambda x: x.astype(float).min())\n          B\n    A\n    1   1.0\n    2   3.0\n    '
    )

    def aggregate(
        self,
        func: Optional[Union[AggScalar, Iterable[AggScalar], Dict[Any, Any]]] = None,
        *args: Any,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, bool]] = None,
        **kwargs: Any,
    ) -> DataFrame:
        """
        Aggregate using one or more operations.
        ...
        """
        relabeling, func, columns, order = reconstruct_func(func, **kwargs)
        func = maybe_mangle_lambdas(func)
        if maybe_use_numba(engine):
            kwargs['engine'] = engine
            kwargs['engine_kwargs'] = engine_kwargs
        op = GroupByApply(self, func, args=args, kwargs=kwargs)
        result = op.agg()
        if not is_dict_like(func) and result is not None:
            if not self.as_index and is_list_like(func):
                return result.reset_index()
            else:
                return result
        elif relabeling:
            result = cast(DataFrame, result)
            result = result.iloc[:, order]
            result = cast(DataFrame, result)
            result.columns = columns
        if result is None:
            if 'engine' in kwargs:
                del kwargs['engine']
                del kwargs['engine_kwargs']
            if maybe_use_numba(engine):
                return self._aggregate_with_numba(
                    func, *args, engine_kwargs=engine_kwargs, **kwargs
                )
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
        return cast(DataFrame, result)

    agg = aggregate

    def _python_agg_general(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> DataFrame:
        f = lambda x: func(x, *args, **kwargs)
        if self.ngroups == 0:
            return self._python_apply_general(f, self._selected_obj, is_agg=True)
        obj = self._obj_with_exclusions
        if not len(obj.columns):
            return self._python_apply_general(f, self._selected_obj)
        output: Dict[int, Series] = {}
        for idx, (name, ser) in enumerate(obj.items()):
            result = self._grouper.agg_series(ser, f)
            output[idx] = result
        res = self.obj._constructor(output)
        res.columns = obj.columns.copy(deep=False)
        return self._wrap_aggregated_output(res)

    def _aggregate_frame(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> DataFrame:
        if self._grouper.nkeys != 1:
            raise AssertionError('Number of keys must be 1')
        obj = self._obj_with_exclusions
        result: Dict[Any, Any] = {}
        for name, grp_df in self._grouper.get_iterator(obj):
            fres = func(grp_df, *args, **kwargs)
            result[name] = fres
        result_index = self._grouper.result_index
        out = self.obj._constructor(result, index=obj.columns, columns=result_index)
        out = out.T
        return out

    def _wrap_applied_output(
        self,
        data: DataFrame,
        values: List[Any],
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ) -> Union[DataFrame, Series]:
        if len(values) == 0:
            if is_transform:
                res_index = data.index
            elif not self.group_keys:
                res_index = None
            else:
                res_index = self._grouper.result_index
            result = self.obj._constructor(index=res_index, columns=data.columns)
            result = result.astype(data.dtypes)
            return result
        first_not_none = next(com.not_none(*values), None)
        if first_not_none is None:
            result = self.obj._constructor(columns=data.columns)
            result = result.astype(data.dtypes)
            return result
        elif isinstance(first_not_none, DataFrame):
            return self._concat_objects(
                values, not_indexed_same=not_indexed_same, is_transform=is_transform
            )
        key_index: Optional[Index] = self._grouper.result_index if self.as_index else None
        if isinstance(first_not_none, (np.ndarray, Index)):
            if not is_hashable(self._selection):
                name = tuple(self._selection)
            else:
                name = self._selection
            return self.obj._constructor_sliced(
                values, index=key_index, name=name
            )
        elif not isinstance(first_not_none, Series):
            if self.as_index:
                return self.obj._constructor_sliced(values, index=key_index)
            else:
                result = self.obj._constructor(values, columns=[self._selection])
                result = self._insert_inaxis_grouper(result)
                return result
        else:
            return self._wrap_applied_output_series(
                values, not_indexed_same, first_not_none, key_index, is_transform
            )

    def _wrap_applied_output_series(
        self,
        values: List[Series],
        not_indexed_same: bool,
        first_not_none: Series,
        key_index: Optional[Index],
        is_transform: bool,
    ) -> DataFrame:
        kwargs = first_not_none._construct_axes_dict()
        backup = Series(**kwargs)
        values = [x if x is not None else backup for x in values]
        all_indexed_same = all_indexes_same((x.index for x in values))
        if not all_indexed_same:
            return self._concat_objects(
                values, not_indexed_same=True, is_transform=is_transform
            )
        stacked_values = np.vstack([np.asarray(v) for v in values])
        index = key_index
        columns = first_not_none.index.copy()
        if columns.name is None:
            names = {v.name for v in values}
            if len(names) == 1:
                columns.name = next(iter(names))
        if stacked_values.dtype == object:
            stacked_values = stacked_values.tolist()
        result = self.obj._constructor(
            stacked_values, index=index, columns=columns
        )
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
        return result

    def boxplot(
        self,
        column: Optional[Union[str, List[str]]] = None,
        by: Optional[Any] = None,
        ax: Optional[Any] = None,
        fontsize: Optional[int] = None,
        rot: Optional[int] = None,
        grid: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        layout: Optional[Tuple[int, int]] = None,
        sharex: bool = False,
        sharey: bool = False,
        backend: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        return boxplot_frame_groupby(
            self,
            column=column,
            by=by,
            ax=ax,
            fontsize=fontsize,
            rot=rot,
            grid=grid,
            figsize=figsize,
            layout=layout,
            sharex=sharex,
            sharey=sharey,
            backend=backend,
            **kwargs,
        )

    def _apply_to_column_groupbys(
        self, func: Callable[[SeriesGroupBy], Any]
    ) -> DataFrame:
        from pandas.core.reshape.concat import concat
        obj = self._obj_with_exclusions
        columns = obj.columns
        sgbs = (
            SeriesGroupBy(
                obj.iloc[:, i],
                selection=colname,
                grouper=self._grouper,
                exclusions=self.exclusions,
                observed=self.observed,
            )
            for i, colname in enumerate(obj.columns)
        )
        results = [func(sgb) for sgb in sgbs]
        if not len(results):
            res_df = DataFrame([], columns=columns, index=self._grouper.result_index)
        else:
            res_df = concat(
                results, keys=columns, axis=1
            )
        if not self.as_index:
            res_df.index = default_index(len(res_df))
            res_df = self._insert_inaxis_grouper(res_df)
        return res_df

    def nunique(self, dropna: bool = True) -> DataFrame:
        """
        Return DataFrame with counts of unique elements in each position.
        ...
        """
        return self._apply_to_column_groupbys(lambda sgb: sgb.nunique(dropna))

    def idxmax(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame:
        """
        Return index of first occurrence of maximum in each group.
        ...
        """
        return self._idxmax_idxmin('idxmax', numeric_only=numeric_only, skipna=skipna)

    def idxmin(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame:
        """
        Return index of first occurrence of minimum in each group.
        ...
        """
        return self._idxmax_idxmin('idxmin', numeric_only=numeric_only, skipna=skipna)

    boxplot = boxplot_frame_groupby

    def value_counts(
        self,
        subset: Optional[List[str]] = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> Union[Series, DataFrame]:
        """
        Return a Series or DataFrame containing counts of unique rows.
        ...
        """
        return self._value_counts(subset, normalize, sort, ascending, dropna)

    def take(
        self,
        indices: Union[List[int], np.ndarray],
        **kwargs: Any,
    ) -> DataFrame:
        """
        Return the elements in the given *positional* indices in each group.
        ...
        """
        result = self._op_via_apply('take', indices=indices, **kwargs)
        return result

    def skew(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> DataFrame:
        """
        Return unbiased skew within groups.
        ...
        """
        return self._cython_agg_general(
            'skew', alt=None, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def kurt(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> DataFrame:
        """
        Return unbiased kurtosis within groups.
        ...
        """
        return self._cython_agg_general(
            'kurt', alt=None, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    @property
    @doc(DataFrame.plot.__doc__)
    def plot(self) -> GroupByPlot:
        result = GroupByPlot(self)
        return result

    @doc(DataFrame.corr.__doc__)
    def corr(
        self,
        method: CorrelationMethod = 'pearson',
        min_periods: int = 1,
        numeric_only: bool = False,
    ) -> DataFrame:
        result = self._op_via_apply(
            'corr', method=method, min_periods=min_periods, numeric_only=numeric_only
        )
        return result

    @doc(DataFrame.cov.__doc__)
    def cov(
        self,
        min_periods: Optional[int] = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> DataFrame:
        result = self._op_via_apply(
            'cov', min_periods=min_periods, ddof=ddof, numeric_only=numeric_only
        )
        return result

    def hist(
        self,
        column: Optional[Union[str, List[str]]] = None,
        by: Optional[Any] = None,
        grid: bool = True,
        xlabelsize: Optional[int] = None,
        xrot: Optional[float] = None,
        ylabelsize: Optional[int] = None,
        yrot: Optional[float] = None,
        ax: Optional[Any] = None,
        sharex: bool = False,
        sharey: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        layout: Optional[Tuple[int, int]] = None,
        bins: Union[int, Sequence[int]] = 10,
        backend: Optional[str] = None,
        legend: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Make a histogram of the DataFrame's columns.
        ...
        """
        result = self._op_via_apply(
            'hist',
            column=column,
            by=by,
            grid=grid,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            ax=ax,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            layout=layout,
            bins=bins,
            backend=backend,
            legend=legend,
            **kwargs,
        )
        return result

    def corrwith(
        self,
        other: Union[Series, DataFrame],
        drop: bool = False,
        method: CorrelationMethod = 'pearson',
        numeric_only: bool = False,
    ) -> DataFrame:
        """
        Compute pairwise correlation.
        ...
        """
        warnings.warn(
            'DataFrameGroupBy.corrwith is deprecated', FutureWarning, stacklevel=find_stack_level()
        )
        result = self._op_via_apply(
            'corrwith',
            other=other,
            drop=drop,
            method=method,
            numeric_only=numeric_only,
        )
        return result


def _wrap_transform_general_frame(
    obj: NDFrame, group: NDFrame, res: Union[Series, DataFrame]
) -> Union[DataFrame, Series]:
    from pandas import concat
    if isinstance(res, Series):
        if res.index.is_(obj.index):
            res_frame = concat([res] * len(group.columns), axis=1, ignore_index=True)
            res_frame.columns = group.columns
            res_frame.index = group.index
        else:
            res_frame = obj._constructor(
                np.tile(res.values, (len(group.index), 1)),
                columns=group.columns,
                index=group.index,
            )
            res_frame = cast(DataFrame, res_frame)
        assert isinstance(res_frame, DataFrame)
        return res_frame
    elif isinstance(res, DataFrame) and (not res.index.is_(group.index)):
        return res._align_frame(group)[0]
    else:
        return res

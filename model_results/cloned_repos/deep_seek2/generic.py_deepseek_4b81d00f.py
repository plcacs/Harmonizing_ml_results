from __future__ import annotations

from collections import abc
from collections.abc import Callable, Hashable, Sequence
from functools import partial
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    TypeVar,
    Union,
    cast,
    Optional,
    List,
    Dict,
    Tuple,
)
import warnings

import numpy as np

from pandas._libs import Interval
from pandas._libs.hashtable import duplicated
from pandas.errors import SpecificationError
from pandas.util._decorators import (
    Appender,
    Substitution,
    doc,
    set_module,
)
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
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    IntervalDtype,
)
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

from pandas.core import algorithms
from pandas.core.apply import (
    GroupByApply,
    maybe_mangle_lambdas,
    reconstruct_func,
    validate_func_kwargs,
)
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import base
from pandas.core.groupby.groupby import (
    GroupBy,
    GroupByPlot,
    _transform_template,
)
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    all_indexes_same,
    default_index,
)
from pandas.core.series import Series
from pandas.core.sorting import get_group_index
from pandas.core.util.numba_ import maybe_use_numba

from pandas.plotting import boxplot_frame_groupby

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        BlockManager,
        CorrelationMethod,
        IndexLabel,
        Manager,
        SingleBlockManager,
        TakeIndexer,
    )

    from pandas import Categorical
    from pandas.core.generic import NDFrame

# TODO(typing) the return value on this callable should be any *scalar*.
AggScalar = Union[str, Callable[..., Any]]
# TODO: validate types on ScalarResult and move to _typing
# Blocked from using by https://github.com/python/mypy/issues/1484
# See note at _mangle_lambda_list
ScalarResult = TypeVar("ScalarResult")


@set_module("pandas")
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


@set_module("pandas.api.typing")
class SeriesGroupBy(GroupBy[Series]):
    def _wrap_agged_manager(self, mgr: Manager) -> Series:
        out = self.obj._constructor_from_mgr(mgr, axes=mgr.axes)
        out._name = self.obj.name
        return out

    def _get_data_to_aggregate(
        self, *, numeric_only: bool = False, name: str | None = None
    ) -> SingleBlockManager:
        ser = self._obj_with_exclusions
        single = ser._mgr
        if numeric_only and not is_numeric_dtype(ser.dtype):
            # GH#41291 match Series behavior
            kwd_name = "numeric_only"
            raise TypeError(
                f"Cannot use {kwd_name}=True with "
                f"{type(self).__name__}.{name} and non-numeric dtypes."
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

    def apply(self, func: Callable, *args: Any, **kwargs: Any) -> Series:
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

        Parameters
        ----------
        func : callable
            A callable that takes a series as its first argument, and
            returns a dataframe, a series or a scalar. In addition the
            callable may take positional and keyword arguments.

        *args : tuple
            Optional positional arguments to pass to ``func``.

        **kwargs : dict
            Optional keyword arguments to pass to ``func``.

        Returns
        -------
        Series or DataFrame
            A pandas object with the result of applying ``func`` to each group.

        See Also
        --------
        pipe : Apply function to the full GroupBy object instead of to each
            group.
        aggregate : Apply aggregate function to the GroupBy object.
        transform : Apply function column-by-column to the GroupBy object.
        Series.apply : Apply a function to a Series.
        DataFrame.apply : Apply a function to each row or column of a DataFrame.

        Notes
        -----

        .. versionchanged:: 1.3.0

            The resulting dtype will reflect the return value of the passed ``func``,
            see the examples below.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> s = pd.Series([0, 1, 2], index="a a b".split())
        >>> g1 = s.groupby(s.index, group_keys=False)
        >>> g2 = s.groupby(s.index, group_keys=True)

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
        return super().apply(func, *args, **kwargs)

    def aggregate(
        self,
        func: Optional[Union[str, Callable, List[Union[str, Callable]], Dict[Hashable, Union[str, Callable]]]] = None,
        *args: Any,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Series:
        """
        Aggregate using one or more operations.

        The ``aggregate`` method enables flexible and efficient aggregation of grouped
        data using a variety of functions, including built-in, user-defined, and
        optimized JIT-compiled functions.

        Parameters
        ----------
        func : function, str, list, dict or None
            Function to use for aggregating the data. If a function, must either
            work when passed a Series or when passed to Series.apply.

            Accepted combinations are:

            - function
            - string function name
            - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
            - None, in which case ``**kwargs`` are used with Named Aggregation. Here
              the output has one column for each element in ``**kwargs``. The name of
              the column is keyword, whereas the value determines the aggregation
              used to compute the values in the column.

              Can also accept a Numba JIT function with
              ``engine='numba'`` specified. Only passing a single function is supported
              with this engine.

              If the ``'numba'`` engine is chosen, the function must be
              a user defined function with ``values`` and ``index`` as the
              first and second arguments respectively in the function signature.
              Each group's index will be passed to the user defined function
              and optionally available for use.

            .. deprecated:: 2.1.0

                Passing a dictionary is deprecated and will raise in a future version
                of pandas. Pass a list of aggregations instead.
        *args
            Positional arguments to pass to func.
        engine : str, default None
            * ``'cython'`` : Runs the function through C-extensions from cython.
            * ``'numba'`` : Runs the function through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
                ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
              applied to the function

        **kwargs
            * If ``func`` is None, ``**kwargs`` are used to define the output names and
              aggregations via Named Aggregation. See ``func`` entry.
            * Otherwise, keyword arguments to be passed into func.

        Returns
        -------
        Series
            Aggregated Series based on the grouping and the applied aggregation
            functions.

        See Also
        --------
        SeriesGroupBy.apply : Apply function func group-wise
            and combine the results together.
        SeriesGroupBy.transform : Transforms the Series on each group
            based on the given function.
        Series.aggregate : Aggregate using one or more operations.

        Notes
        -----
        When using ``engine='numba'``, there will be no "fall back" behavior internally.
        The group data and group index will be passed as numpy arrays to the JITed
        user defined function, and no alternative execution attempts will be tried.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        .. versionchanged:: 1.3.0

            The resulting dtype will reflect the return value of the passed ``func``,
            see the examples below.

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

        >>> s.groupby([1, 1, 2, 2]).agg("min")
        1    1
        2    3
        dtype: int64

        >>> s.groupby([1, 1, 2, 2]).agg(["min", "max"])
           min  max
        1    1    2
        2    3    4

        The output column names can be controlled by passing
        the desired column names and aggregations as keyword arguments.

        >>> s.groupby([1, 1, 2, 2]).agg(
        ...     minimum="min",
        ...     maximum="max",
        ... )
           minimum  maximum
        1        1        2
        2        3        4

        .. versionchanged:: 1.3.0

            The resulting dtype will reflect the return value of the aggregating
            function.

        >>> s.groupby([1, 1, 2, 2]).agg(lambda x: x.astype(float).min())
        1    1.0
        2    3.0
        dtype: float64
        """
        relabeling = func is None
        columns = None
        if relabeling:
            columns, func = validate_func_kwargs(kwargs)
            kwargs = {}

        if isinstance(func, str):
            if maybe_use_numba(engine) and engine is not None:
                # Not all agg functions support numba, only propagate numba kwargs
                # if user asks for numba, and engine is not None
                # (if engine is None, the called function will handle the case where
                # numba is requested via the global option)
                kwargs["engine"] = engine
            if engine_kwargs is not None:
                kwargs["engine_kwargs"] = engine_kwargs
            return getattr(self, func)(*args, **kwargs)

        elif isinstance(func, abc.Iterable):
            # Catch instances of lists / tuples
            # but not the class list / tuple itself.
            func = maybe_mangle_lambdas(func)
            kwargs["engine"] = engine
            kwargs["engine_kwargs"] = engine_kwargs
            ret = self._aggregate_multiple_funcs(func, *args, **kwargs)
            if relabeling:
                # columns is not narrowed by mypy from relabeling flag
                assert columns is not None  # for mypy
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
                # e.g. test_evaluate_with_empty_groups without any groups to
                #  iterate over, we have no output on which to do dtype
                #  inference. We default to using the existing dtype.
                #  xref GH#51445
                obj = self._obj_with_exclusions
                return self.obj._constructor(
                    [],
                    name=self.obj.name,
                    index=self._grouper.result_index,
                    dtype=obj.dtype,
                )
            return self._python_agg_general(func, *args, **kwargs)

    agg = aggregate

    def _python_agg_general(self, func: Callable, *args: Any, **kwargs: Any) -> Series:
        f = lambda x: func(x, *args, **kwargs)

        obj = self._obj_with_exclusions
        result = self._grouper.agg_series(obj, f)
        res = obj._constructor(result, name=obj.name)
        return self._wrap_aggregated_output(res)

    def _aggregate_multiple_funcs(self, arg: Any, *args: Any, **kwargs: Any) -> DataFrame:
        if isinstance(arg, dict):
            raise SpecificationError("nested renamer is not supported")

        if any(isinstance(x, (tuple, list)) for x in arg):
            arg = ((x, x) if not isinstance(x, (tuple, list)) else x for x in arg)
        else:
            # list of functions / function names
            columns = (com.get_callable_name(f) or f for f in arg)
            arg = zip(columns, arg)

        results: dict[base.OutputKey, DataFrame | Series] = {}
        with com.temp_setattr(self, "as_index", True):
            # Combine results using the index, need to adjust index after
            # if as_index=False (GH#50724)
            for idx, (name, func) in enumerate(arg):
                key = base.OutputKey(label=name, position=idx)
                results[key] = self.aggregate(func, *args, **kwargs)

        if any(isinstance(x, DataFrame) for x in results.values()):
            from pandas import concat

            res_df = concat(
                results.values(), axis=1, keys=[key.label for key in results]
            )
            return res_df

        indexed_output = {key.position: val for key, val in results.items()}
        output = self.obj._constructor_expanddim(indexed_output, index=None)
        output.columns = Index(key.label for key in results)

        return output

    def _wrap_applied_output(
        self,
        data: Series,
        values: list[Any],
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ) -> DataFrame | Series:
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
            # GH #6265
            if is_transform:
                # GH#47787 see test_group_on_empty_multiindex
                res_index = data.index
            elif not self.group_keys:
                res_index = None
            else:
                res_index = self._grouper.result_index

            return self.obj._constructor(
                [],
                name=self.obj.name,
                index=res_index,
                dtype=data.dtype,
            )
        assert values is not None

        if isinstance(values[0], dict):
            # GH #823 #24880
            index = self._grouper.result_index
            res_df = self.obj._constructor_expanddim(values, index=index)
            # if self.observed is False,
            # keep all-NaN rows created while re-indexing
            res_ser = res_df.stack()
            res_ser.name = self.obj.name
            return res_ser
        elif isinstance(values[0], (Series, DataFrame)):
            result = self._concat_objects(
                values,
                not_indexed_same=not_indexed_same,
                is_transform=is_transform,
            )
            if isinstance(result, Series):
                result.name = self.obj.name
            if not self.as_index and not_indexed_same:
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return result
        else:
            # GH #6265 #24880
            result = self.obj._constructor(
                data=values, index=self._grouper.result_index, name=self.obj.name
            )
            if not self.as_index:
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return result

    __examples_series_doc = dedent(
        """
    >>> ser = pd.Series([390.0, 350.0, 30.0, 20.0],
    ...                 index=["Falcon", "Falcon", "Parrot", "Parrot"],
    ...                 name="Max Speed")
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

    @Substitution(klass="Series", example=__examples_series_doc)
    @Appender(_transform_template)
    def transform(
        self,
        func: Callable,
        *args: Any,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Series:
        return self._transform(
            func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
        )

    def _cython_transform(self, how: str, numeric_only: bool = False, **kwargs: Any) -> Series:
        obj = self._obj_with_exclusions

        try:
            result = self._grouper._cython_operation(
                "transform", obj._values, how, 0, **kwargs
            )
        except NotImplementedError as err:
            # e.g. test_groupby_raises_string
            raise TypeError(f"{how} is not supported for {obj.dtype} dtype") from err

        return obj._constructor(result, index=self.obj.index, name=obj.name)

    def _transform_general(
        self,
        func: Callable,
        engine: Optional[str],
        engine_kwargs: Optional[Dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Series:
        """
        Transform with a callable `func`.
        """
        if maybe_use_numba(engine):
            return self._transform_with_numba(
                func, *args, engine_kwargs=engine_kwargs, **kwargs
            )
        assert callable(func)
        klass = type(self.obj)

        results = []
        for name, group in self._grouper.get_iterator(
            self._obj_with_exclusions,
        ):
            # this setattr is needed for test_transform_lambda_with_datetimetz
            object.__setattr__(group, "name", name)
            res = func(group, *args, **kwargs)

            results.append(klass(res, index=group.index))

        # check for empty "results" to avoid concat ValueError
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
        func: Callable,
        dropna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series:
        """
        Filter elements from groups that don't satisfy a criterion.

        Elements from groups are filtered if they do not satisfy the
        boolean criterion specified by func.

        Parameters
        ----------
        func : function
            Criterion to apply to each group. Should return True or False.
        dropna : bool, optional
            Drop groups that do not pass the filter. True by default; if False,
            groups that evaluate False are filled with NaNs.
        *args : tuple
            Optional positional arguments to pass to `func`.
        **kwargs : dict
            Optional keyword arguments to pass to `func`.

        Returns
        -------
        Series
            The filtered subset of the original Series.

        See Also
        --------
        Series.filter: Filter elements of ungrouped Series.
        DataFrameGroupBy.filter : Filter elements from groups base on criterion.

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "A": ["foo", "bar", "foo", "bar", "foo", "bar"],
        ...         "B": [1, 2, 3, 4, 5, 6],
        ...         "C": [2.0, 5.0, 8.0, 1.0, 2.0, 9.0],
        ...     }
        ... )
        >>> grouped = df.groupby("A")
        >>> df.groupby("A").B.filter(lambda x: x.mean() > 3.0)
        1    2
        3    4
        5    6
        Name: B, dtype: int64
        """
        if isinstance(func, str):
            wrapper = lambda x: getattr(x, func)(*args, **kwargs)
        else:
            wrapper = lambda x: func(x, *args, **kwargs)

        # Interpret np.nan as False.
        def true_and_notna(x) -> bool:
            b = wrapper(x)
            return notna(b) and b

        try:
            indices = [
                self._get_index(name)
                for name, group in self._grouper.get_iterator(self._obj_with_exclusions)
                if true_and_notna(group)
            ]
        except (ValueError, TypeError) as err:
            raise TypeError("the filter must return a boolean result") from err

        filtered = self._apply_filter(indices, dropna)
        return filtered

    def nunique(self, dropna: bool = True) -> Series | DataFrame:
        """
        Return number of unique elements in the group.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the counts.

        Returns
        -------
        Series
            Number of unique values within each group.

        See Also
        --------
        core.resample.Resampler.nunique : Method nunique for Resampler.

        Examples
        --------
        >>> lst = ["a", "a", "b", "b"]
        >>> ser = pd.Series([1, 2, 3, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    3
        dtype: int64
        >>> ser.groupby(level=0).nunique()
        a    2
        b    1
        dtype: int64
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

        mask = duplicated(group_index, "first")
        res = np.bincount(ids[~mask], minlength=ngroups)
        res = ensure_int64(res)

        ri = self._grouper.result_index
        result: Series | DataFrame = self.obj._constructor(
            res, index=ri, name=self.obj.name
        )
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return result

    @doc(Series.describe)
    def describe(
        self,
        percentiles: Optional[List[float]] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> Series:
        return super().describe(
            percentiles=percentiles, include=include, exclude=exclude
        )

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins: Optional[Union[int, List[int]]] = None,
        dropna: bool = True,
    ) -> Series | DataFrame:
        """
        Return a Series or DataFrame containing counts of unique rows.

        .. versionadded:: 1.4.0

        Parameters
        ----------
        normalize : bool, default False
            Return proportions rather than frequencies.
        sort : bool, default True
            Sort by frequencies.
        ascending : bool, default False
            Sort in ascending order.
        bins : int or list of ints, optional
            Rather than count values, group them into half-open bins,
            a convenience for pd.cut, only works with numeric data.
        dropna : bool, default True
            Don't include counts of rows that contain NA values.

        Returns
        -------
        Series or DataFrame
            Series if the groupby ``as_index`` is True, otherwise DataFrame.

        See Also
        --------
        Series.value_counts: Equivalent method on Series.
        DataFrame.value_counts: Equivalent method on DataFrame.
        DataFrameGroupBy.value_counts: Equivalent method on DataFrameGroupBy.

        Notes
        -----
        - If the groupby ``as_index`` is True then the returned Series will have a
          MultiIndex with one level per input column.
        - If the groupby ``as_index`` is False then the returned DataFrame will have an
          additional column with the value_counts. The column is labelled 'count' or
          'proportion', depending on the ``normalize`` parameter.

        By default, rows that contain any NA values are omitted from
        the result.

        By default, the result will be in descending order so that the
        first element of each group is the most frequently-occurring row.

        Examples
        --------
        >>> s = pd.Series(
        ...     [1, 1, 2, 3, 3, 3, 1, 1, 1, 3, 3, 3],
        ...     index=["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
        ... )
        >>> s
        A    1
        A    1
        A    2
        A    3
        A    2
        A    3
        B    3
        B    1
        B    1
        B    3
        B    3
        B    3
        dtype: int64
        >>> g1 = s.groupby(s.index)
        >>> g1.value_counts(bins=2)
        A  (0.997, 2.0]    4
           (2.0, 3.0]      2
        B  (2.0, 3.0]      4
           (0.997, 2.0]    2
        Name: count, dtype: int64
        >>> g1.value_counts(normalize=True)
        A  1    0.333333
           2    0.333333
           3    0.333333
        B  3    0.666667
           1    0.333333
        Name: proportion, dtype: float64
        """
        name = "proportion" if normalize else "count"

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

        if isinstance(val.dtype, CategoricalDtype) or (
            bins is not None and not np.iterable(bins)
        ):
            # scalar bins cannot be done at top level
            # in a backward compatible way
            # GH38672 relates to categorical dtype
            ser = self.apply(
                Series.value_counts,
                normalize=normalize,
                sort=sort
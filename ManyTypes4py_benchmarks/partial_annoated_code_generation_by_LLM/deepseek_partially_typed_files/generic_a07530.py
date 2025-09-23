"""
Define the SeriesGroupBy and DataFrameGroupBy
classes that hold the groupby interfaces (and some implementations).

These are user facing as the result of the ``df.groupby(...)`` operations,
which here returns a DataFrameGroupBy object.
"""
from __future__ import annotations
from collections import abc
from collections.abc import Callable, Hashable, Sequence
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeVar, Union, cast, overload
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

    def _wrap_agged_manager(self, mgr: SingleBlockManager) -> Series:
        out = self.obj._constructor_from_mgr(mgr, axes=mgr.axes)
        out._name = self.obj.name
        return out

    def _get_data_to_aggregate(self, *, numeric_only: bool = False, name: str | None = None) -> SingleBlockManager:
        ser = self._obj_with_exclusions
        single = ser._mgr
        if numeric_only and (not is_numeric_dtype(ser.dtype)):
            kwd_name = 'numeric_only'
            raise TypeError(f'Cannot use {kwd_name}=True with {type(self).__name__}.{name} and non-numeric dtypes.')
        return single

    _agg_examples_doc = dedent("\n    Examples\n    --------\n    >>> s = pd.Series([1, 2, 3, 4])\n\n    >>> s\n    0    1\n    1    2\n    2    3\n    3    4\n    dtype: int64\n\n    >>> s.groupby([1, 1, 2, 2]).min()\n    1    1\n    2    3\n    dtype: int64\n\n    >>> s.groupby([1, 1, 2, 2]).agg('min')\n    1    1\n    2    3\n    dtype: int64\n\n    >>> s.groupby([1, 1, 2, 2]).agg(['min', 'max'])\n       min  max\n    1    1    2\n    2    3    4\n\n    The output column names can be controlled by passing\n    the desired column names and aggregations as keyword arguments.\n\n    >>> s.groupby([1, 1, 2, 2]).agg(\n    ...     minimum='min',\n    ...     maximum='max',\n    ... )\n       minimum  maximum\n    1        1        2\n    2        3        4\n\n    .. versionchanged:: 1.3.0\n\n        The resulting dtype will reflect the return value of the aggregating function.\n\n    >>> s.groupby([1, 1, 2, 2]).agg(lambda x: x.astype(float).min())\n    1    1.0\n    2    3.0\n    dtype: float64\n    ")

    def apply(self, func: Callable, *args: Any, **kwargs: Any) -> Series | DataFrame:
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

    def aggregate(self, func: AggScalar | None = None, *args: Any, engine: str | None = None, engine_kwargs: dict[str, bool] | None = None, **kwargs: Any) -> Series | DataFrame:
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
            (columns, func) = validate_func_kwargs(kwargs)
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
                return self._aggregate_with_numba(func, *args, engine_kwargs=engine_kwargs, **kwargs)
            if self.ngroups == 0:
                obj = self._obj_with_exclusions
                return self.obj._constructor([], name=self.obj.name, index=self._grouper.result_index, dtype=obj.dtype)
            return self._python_agg_general(func, *args, **kwargs)

    agg = aggregate

    def _python_agg_general(self, func: Callable, *args: Any, **kwargs: Any) -> Series:
        f = lambda x: func(x, *args, **kwargs)
        obj = self._obj_with_exclusions
        result = self._grouper.agg_series(obj, f)
        res = obj._constructor(result, name=obj.name)
        return self._wrap_aggregated_output(res)

    def _aggregate_multiple_funcs(self, arg: abc.Iterable, *args: Any, **kwargs: Any) -> DataFrame:
        if isinstance(arg, dict):
            raise SpecificationError('nested renamer is not supported')
        if any((isinstance(x, (tuple, list)) for x in arg)):
            arg = ((x, x) if not isinstance(x, (tuple, list)) else x for x in arg)
        else:
            columns = (com.get_callable_name(f) or f for f in arg)
            arg = zip(columns, arg)
        results: dict[base.OutputKey, DataFrame | Series] = {}
        with com.temp_setattr(self, 'as_index', True):
            for (idx, (name, func)) in enumerate(arg):
                key = base.OutputKey(label=name, position=idx)
                results[key
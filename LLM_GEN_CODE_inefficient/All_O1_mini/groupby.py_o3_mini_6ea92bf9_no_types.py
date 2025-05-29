"""
Provide the groupby split-apply-combine paradigm. Define the GroupBy
class providing the base-class of operations.

The SeriesGroupBy and DataFrameGroupBy sub-class
(defined in pandas.core.groupby.generic)
expose these user-facing objects to provide specific functionality.
"""
from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
import datetime
from functools import partial, wraps
from textwrap import dedent
from typing import TYPE_CHECKING, Literal, TypeVar, Union, cast, final, overload
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
    from pandas._typing import Any, Concatenate, P, Self, T
    from pandas.core.indexers.objects import BaseIndexer
    from pandas.core.resample import Resampler
    from pandas.core.window import ExpandingGroupby, ExponentialMovingWindowGroupby, RollingGroupby
_common_see_also = """
        See Also
        --------
        Series.%(name)s : Apply a function %(name)s to a Series.
        DataFrame.%(name)s : Apply a function %(name)s
            to each row or column of a DataFrame.
"""
_groupby_agg_method_engine_template = """
Compute {fname} of group values.

Parameters
----------
numeric_only : bool, default {no}
    Include only float, int, boolean columns.

    .. versionchanged:: 2.0.0

        numeric_only no longer accepts ``None``.

min_count : int, default {mc}
    The required number of valid values to perform the operation. If fewer
    than ``min_count`` non-NA values are present the result will be NA.

engine : str, default None {e}
    * ``'cython'`` : Runs rolling apply through C-extensions from cython.
    * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.
        Only available when ``raw`` is set to ``True``.
    * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

engine_kwargs : dict, default None {ek}
    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
        and ``parallel`` dictionary keys. The values must either be ``True`` or
        ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
        ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be
        applied to both the ``func`` and the ``apply`` groupby aggregation.

Returns
-------
Series or DataFrame
    Computed {fname} of values within each group.

See Also
--------
SeriesGroupBy.min : Return the min of the group values.
DataFrameGroupBy.min : Return the min of the group values.
SeriesGroupBy.max : Return the max of the group values.
DataFrameGroupBy.max : Return the max of the group values.
SeriesGroupBy.sum : Return the sum of the group values.
DataFrameGroupBy.sum : Return the sum of the group values.

Examples
--------
{example}
"""
_groupby_agg_method_skipna_engine_template = """
Compute {fname} of group values.

Parameters
----------
numeric_only : bool, default {no}
    Include only float, int, boolean columns.

    .. versionchanged:: 2.0.0

        numeric_only no longer accepts ``None``.

min_count : int, default {mc}
    The required number of valid values to perform the operation. If fewer
    than ``min_count`` non-NA values are present the result will be NA.

skipna : bool, default {s}
    Exclude NA/null values. If the entire group is NA and ``skipna`` is
    ``True``, the result will be NA.

    .. versionchanged:: 3.0.0

engine : str, default None {e}
    * ``'cython'`` : Runs rolling apply through C-extensions from cython.
    * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.
        Only available when ``raw`` is set to ``True``.
    * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

engine_kwargs : dict, default None {ek}
    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
        and ``parallel`` dictionary keys. The values must either be ``True`` or
        ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
        ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be
        applied to both the ``func`` and the ``apply`` groupby aggregation.

Returns
-------
Series or DataFrame
    Computed {fname} of values within each group.

See Also
--------
SeriesGroupBy.min : Return the min of the group values.
DataFrameGroupBy.min : Return the min of the group values.
SeriesGroupBy.max : Return the max of the group values.
DataFrameGroupBy.max : Return the max of the group values.
SeriesGroupBy.sum : Return the sum of the group values.
DataFrameGroupBy.sum : Return the sum of the group values.

Examples
--------
{example}
"""
_pipe_template = """
Apply a ``func`` with arguments to this %(klass)s object and return its result.

Use `.pipe` when you want to improve readability by chaining together
functions that expect Series, DataFrames, GroupBy or Resampler objects.
Instead of writing

>>> h = lambda x, arg2, arg3: x + 1 - arg2 * arg3
>>> g = lambda x, arg1: x * 5 / arg1
>>> f = lambda x: x ** 4
>>> df = pd.DataFrame([["a", 4], ["b", 5]], columns=["group", "value"])
>>> h(g(f(df.groupby('group')), arg1=1), arg2=2, arg3=3)  # doctest: +SKIP

You can write

>>> (df.groupby('group')
...    .pipe(f)
...    .pipe(g, arg1=1)
...    .pipe(h, arg2=2, arg3=3))  # doctest: +SKIP

which is much more readable.

Parameters
----------
func : Callable[[Self, P], T] | tuple[Callable[..., T], str]
    Function to apply to this %(klass)s object or, alternatively,
    a `(callable, data_keyword)` tuple where `data_keyword` is a
    string indicating the keyword of `callable` that expects the
    %(klass)s object.
*args : P.args, optional
       Positional arguments passed into `func`.
**kwargs : P.kwargs, optional
         A dictionary of keyword arguments passed into `func`.

Returns
-------
T
    The original object with the function `func` applied.

See Also
--------
Series.pipe : Apply a function with arguments to a series.
DataFrame.pipe: Apply a function with arguments to a dataframe.
apply : Apply function to each group instead of to the
    full %(klass)s object.

Notes
-----
See more `here
<https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls>`_

Examples
--------
%(examples)s
"""
_transform_template = """
Call function producing a same-indexed %(klass)s on each group.

Returns a %(klass)s having the same indexes as the original object
filled with the transformed values.

Parameters
----------
func : Callable[[Series | DataFrame], Series | DataFrame | ArrayLike] | str
    Function to apply to each group. See the Notes section below for requirements.

    Accepted inputs are:

    - String
    - Python function
    - Numba JIT function with ``engine='numba'`` specified.

    Only passing a single function is supported with this engine.
    If the ``'numba'`` engine is chosen, the function must be
    a user defined function with ``values`` and ``index`` as the
    first and second arguments respectively in the function signature.
    Each group's index will be passed to the user defined function
    and optionally available for use.

    If a string is chosen, then it needs to be the name
    of the groupby method you want to use.
*args : Iterable[Any], optional
    Positional arguments to pass to func.
engine : Literal["cython", "numba"] | None, default None
    * ``'cython'`` : Runs the function through C-extensions from cython.
    * ``'numba'`` : Runs the function through JIT compiled code from numba.
    * ``None`` : Defaults to ``'cython'`` or the global setting ``compute.use_numba``

engine_kwargs : dict[str, bool] | None, default None
    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
      and ``parallel`` dictionary keys. The values must either be ``True`` or
      ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
      ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
      applied to the function

**kwargs : dict[str, Any], optional
    Keyword arguments to be passed into func.

Returns
-------
%(klass)s
    %(klass)s with the same indexes as the original object filled
    with transformed values.

See Also
--------
%(klass)s.groupby.apply : Apply function ``func`` group-wise and combine
    the results together.
%(klass)s.groupby.aggregate : Aggregate using one or more operations.
%(klass)s.transform : Call ``func`` on self producing a %(klass)s with the
    same axis shape as self.

Notes
-----
Each group is endowed the attribute 'name' in case you need to know
which group you are working on.

The current implementation imposes three requirements on f:

* f must return a value that either has the same shape as the input
  subframe or can be broadcast to the shape of the input subframe.
  For example, if `f` returns a scalar it will be broadcast to have the
  same shape as the input subframe.
* if this is a DataFrame, f must support application column-by-column
  in the subframe. If f also supports application to the entire subframe,
  then a fast path is used starting from the second chunk.
* f must not mutate groups. Mutation is not supported and may
  produce unexpected results. See :ref:`gotchas.udf-mutation` for more details.

When using ``engine='numba'``, there will be no "fall back" behavior internally.
The group data and group index will be passed as numpy arrays to the JITed
user defined function, and no alternative execution attempts will be tried.

.. versionchanged:: 1.3.0

    The resulting dtype will reflect the return value of the passed ``func``,
    see the examples below.

.. versionchanged:: 2.0.0

    When using ``.transform`` on a grouped DataFrame and the transformation function
    returns a DataFrame, pandas now aligns the result's index
    with the input's index. You can call ``.to_numpy()`` on the
    result of the transformation function to avoid alignment.

Examples
--------
%(example)s"""


@final
class GroupByPlot(PandasObject):
    """
    Class implementing the .plot attribute for groupby objects.
    """

    def __init__(self, groupby):
        self._groupby = groupby

    def __call__(self, *args: Any, **kwargs: Any):

        def f(self_arg):
            return self_arg.plot(*args, **kwargs)
        f.__name__ = 'plot'
        return self._groupby._python_apply_general(f, self._groupby.
            _selected_obj)

    def __getattr__(self, name):

        def attr(*args: Any, **kwargs: Any):

            def f(self_arg):
                return getattr(self_arg.plot, name)(*args, **kwargs)
            return self._groupby._python_apply_general(f, self._groupby.
                _selected_obj)
        return attr


_KeysArgType = Union[Hashable, list[Hashable], Callable[[Hashable],
    Hashable], list[Callable[[Hashable], Hashable]], Mapping[Hashable,
    Hashable]]


class BaseGroupBy(PandasObject, SelectionMixin[NDFrameT], GroupByIndexingMixin
    ):
    _hidden_attrs: frozenset[str] = PandasObject._hidden_attrs | {'as_index',
        'dropna', 'exclusions', 'grouper', 'group_keys', 'keys', 'level',
        'obj', 'observed', 'sort'}
    _grouper: ops.BaseGrouper
    keys: _KeysArgType | None = None
    level: IndexLabel | None = None
    group_keys: bool

    @final
    def __len__(self):
        return self._grouper.ngroups

    @final
    def __repr__(self):
        return object.__repr__(self)

    @final
    @property
    def groups(self):
        """
        Dict {group name -> group labels}.

        This property provides a dictionary representation of the groupings formed
        during a groupby operation, where each key represents a unique group value from
        the specified column(s), and each value is a list of index labels
        that belong to that group.

        See Also
        --------
        core.groupby.DataFrameGroupBy.get_group : Retrieve group from a
            ``DataFrameGroupBy`` object with provided name.
        core.groupby.SeriesGroupBy.get_group : Retrieve group from a
            ``SeriesGroupBy`` object with provided name.
        core.resample.Resampler.get_group : Retrieve group from a
            ``Resampler`` object with provided name.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).groups
        {'a': ['a', 'a'], 'b': ['b']}

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"])
        >>> df
           a  b  c
        0  1  2  3
        1  1  5  6
        2  7  8  9
        >>> df.groupby(by="a").groups
        {1: [0, 1], 7: [2]}

        For Resampler:

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").groups
        {Timestamp('2023-01-01 00:00:00'): 2, Timestamp('2023-02-01 00:00:00'): 4}
        """
        if isinstance(self.keys, list) and len(self.keys) == 1:
            warnings.warn(
                "`groups` by one element list returns scalar is deprecated and will be removed. In a future version `groups` by one element list will return tuple. Use ``df.groupby(by='a').groups`` instead of ``df.groupby(by=['a']).groups`` to avoid this warning"
                , FutureWarning, stacklevel=find_stack_level())
        return self._grouper.groups

    @final
    @property
    def ngroups(self):
        return self._grouper.ngroups

    @final
    @property
    def indices(self):
        """
        Dict {group name -> group indices}.

        The dictionary keys represent the group labels (e.g., timestamps for a
        time-based resampling operation), and the values are arrays of integer
        positions indicating where the elements of each group are located in the
        original data. This property is particularly useful when working with
        resampled data, as it provides insight into how the original time-series data
        has been grouped.

        See Also
        --------
        core.groupby.DataFrameGroupBy.indices : Provides a mapping of group rows to
            positions of the elements.
        core.groupby.SeriesGroupBy.indices : Provides a mapping of group rows to
            positions of the elements.
        core.resample.Resampler.indices : Provides a mapping of group rows to
            positions of the elements.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).indices
        {'a': array([0, 1]), 'b': array([2])}

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["owl", "toucan", "eagle"]
        ... )
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby(by=["a"]).indices
        {1: array([0, 1]), 7: array([2])}

        For Resampler:

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").indices
        defaultdict(<class 'list'>, {Timestamp('2023-01-01 00:00:00'): [0, 1],
        Timestamp('2023-02-01 00:00:00'): [2, 3]})
        """
        return self._grouper.indices

    @final
    def _get_indices(self, names):
        """
        Safe get multiple indices, translate keys for
        datelike to underlying repr.
        """

        def get_converter(s):
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
        name_sample = next(iter(names), None)
        if isinstance(index_sample, tuple):
            if not isinstance(name_sample, tuple):
                msg = (
                    'must supply a tuple to get_group with multiple grouping keys'
                    )
                raise ValueError(msg)
            if not len(name_sample) == len(index_sample):
                try:
                    return [self.indices[name] for name in names]
                except KeyError as err:
                    msg = (
                        'must supply a same-length tuple to get_group with multiple grouping keys'
                        )
                    raise ValueError(msg) from err
            converters = tuple(get_converter(s) for s in index_sample)
            names = [tuple(f(n) for f, n in zip(converters, name)) for name in
                names]
        else:
            converter = get_converter(index_sample)
            names = [converter(name) for name in names]
        return [self.indices.get(name, np.array([], dtype=np.intp)) for
            name in names]

    @final
    def _get_index(self, name):
        """
        Safe get index, translate keys for datelike to underlying repr.
        """
        return self._get_indices([name])[0]

    @final
    @cache_readonly
    def _selected_obj(self):
        if isinstance(self.obj, Series):
            return self.obj
        if self._selection is not None:
            if is_hashable(self._selection):
                return self.obj[self._selection]
            return self._obj_with_exclusions
        return self.obj

    @final
    def _dir_additions(self):
        return self.obj._dir_additions()

    @overload
    def pipe(self, func, *args: P.args, **kwargs: P.kwargs):
        ...

    @overload
    def pipe(self, func, *args: Any, **kwargs: Any):
        ...

    @Substitution(klass='GroupBy', examples=dedent(
        """        >>> df = pd.DataFrame({'A': 'a b a b'.split(), 'B': [1, 2, 3, 4]})
        >>> df
           A  B
        0  a  1
        1  b  2
        2  a  3
        3  b  4

        To get the difference between each groups maximum and minimum value in one
        pass, you can do

        >>> df.groupby('A').pipe(lambda x: x.max() - x.min())
           B
        A
        a  2
        b  2"""
        ))
    @Appender(_pipe_template)
    def pipe(self, func, *args: Any, **kwargs: Any):
        return com.pipe(self, func, *args, **kwargs)

    @final
    def get_group(self, name):
        """
        Construct DataFrame from group with provided name.

        Parameters
        ----------
        name : object
            The name of the group to get as a DataFrame.

        Returns
        -------
        Series or DataFrame
            Get the respective Series or DataFrame corresponding to the group provided.

        See Also
        --------
        DataFrameGroupBy.groups: Dictionary representation of the groupings formed
            during a groupby operation.
        DataFrameGroupBy.indices: Provides a mapping of group rows to positions
            of the elements.
        SeriesGroupBy.groups: Dictionary representation of the groupings formed
            during a groupby operation.
        SeriesGroupBy.indices: Provides a mapping of group rows to positions
            of the elements.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).get_group("a")
        a    1
        a    2
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["owl", "toucan", "eagle"]
        ... )
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby(by=["a"]).get_group((1,))
                a  b  c
        owl     1  2  3
        toucan  1  5  6

        For Resampler:

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").get_group("2023-01-01")
        2023-01-01    1
        2023-01-15    2
        dtype: int64
        """
        keys = self.keys
        level = self.level
        if is_list_like(level) and len(level) == 1 or is_list_like(keys
            ) and len(keys) == 1:
            if isinstance(name, tuple) and len(name) == 1:
                name = name[0]
            else:
                raise KeyError(name)
        inds = self._get_index(name)
        if not len(inds):
            raise KeyError(name)
        return self._selected_obj.iloc[inds]

    @final
    def __iter__(self):
        """
        Groupby iterator.

        This method provides an iterator over the groups created by the ``resample``
        or ``groupby`` operation on the object. The method yields tuples where
        the first element is the label (group key) corresponding to each group or
        resampled bin, and the second element is the subset of the data that falls
        within that group or bin.

        Returns
        -------
        Iterator[tuple[Hashable, NDFrameT]]
            Generator yielding a sequence of (name, subsetted object)
            for each group.

        See Also
        --------
        Series.groupby : Group data by a specific key or column.
        DataFrame.groupby : Group DataFrame using mapper or by columns.
        DataFrame.resample : Resample a DataFrame.
        Series.resample : Resample a Series.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> for x, y in ser.groupby(level=0):
        ...     print(f"{x}\\n{y}\\n")
        a
        a    1
        a    2
        dtype: int64
        b
        b    3
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"])
        >>> df
           a  b  c
        0  1  2  3
        1  1  5  6
        2  7  8  9
        >>> for x, y in df.groupby(by=["a"]):
        ...     print(f"{x}\\n{y}\\n")
        (1,)
           a  b  c
        0  1  2  3
        1  1  5  6
        (7,)
           a  b  c
        2  7  8  9

        For Resampler:

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> for x, y in ser.resample("MS"):
        ...     print(f"{x}\\n{y}\\n")
        2023-01-01 00:00:00
        2023-01-01    1
        2023-01-15    2
        dtype: int64
        2023-02-01 00:00:00
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        """
        keys = self.keys
        level = self.level
        result = self._grouper.get_iterator(self._selected_obj)
        if is_list_like(level) and len(level) == 1 or isinstance(keys, list
            ) and len(keys) == 1:
            result = (((key,), group) for key, group in result)
        return result
    OutputFrameOrSeries = TypeVar('OutputFrameOrSeries', bound=NDFrame)


    class GroupBy(BaseGroupBy[NDFrameT]):
        """
        Class for grouping and aggregating relational data.

        See aggregate, transform, and apply functions on this object.

        It's easiest to use obj.groupby(...) to use GroupBy, but you can also do:

        ::

            grouped = groupby(obj, ...)

        Parameters
        ----------
        obj : pandas object
        level : int, default None
            Level of MultiIndex
        groupings : list of Grouping objects
            Most users should ignore this
        exclusions : array-like, optional
            List of columns to exclude
        name : str
            Most users should ignore this

        Returns
        -------
        **Attributes**
        groups : dict
            {group name -> group labels}
        len(grouped) : int
            Number of groups

        Notes
        -----
        After grouping, see aggregate, apply, and transform functions. Here are
        some other brief notes about usage. When grouping by multiple groups, the
        result index will be a MultiIndex (hierarchical) by default.

        Iteration produces (key, group) tuples, i.e. chunking the data by group. So
        you can write code like:

        ::

            grouped = obj.groupby(keys)
            for key, group in grouped:
                # do something with the data

        Function calls on GroupBy, if not specially implemented, "dispatch" to the
        grouped data. So if you group a DataFrame and wish to invoke the std()
        method on each group, you can simply do:

        ::

            df.groupby(mapper).std()

        rather than

        ::

            df.groupby(mapper).aggregate(np.std)

        You can pass arguments to these "wrapped" functions, too.

        See the online documentation for full exposition on these topics and much
        more
        """
        _grouper: ops.BaseGrouper
        as_index: bool

        @final
        def __init__(self, obj, keys=None, level=None, grouper=None,
            exclusions=None, selection=None, as_index=True, sort=True,
            group_keys=True, observed=False, dropna=True):
            self._selection: IndexLabel | None = selection
            assert isinstance(obj, NDFrame), type(obj)
            self.level = level
            self.as_index = as_index
            self.keys = keys
            self.sort = sort
            self.group_keys = group_keys
            self.dropna = dropna
            if grouper is None:
                grouper, exclusions, obj = get_grouper(obj, keys, level=
                    level, sort=sort, observed=observed, dropna=self.dropna)
            self.observed = observed
            self.obj: NDFrameT = obj
            self._grouper = grouper
            self.exclusions = frozenset(exclusions
                ) if exclusions else frozenset()

        def __getattr__(self, attr):
            if attr in self._internal_names_set:
                return object.__getattribute__(self, attr)
            if attr in self.obj:
                return self[attr]
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'")

        @final
        def _op_via_apply(self, name, *args: Any, **kwargs: Any):
            """Compute the result of an operation by using GroupBy's apply."""
            f = getattr(type(self._obj_with_exclusions), name)

            def curried(x):
                return f(x, *args, **kwargs)
            curried.__name__ = name
            if name in base.plotting_methods:
                return self._python_apply_general(curried, self._selected_obj)
            is_transform = name in base.transformation_kernels
            result = self._python_apply_general(curried, self.
                _obj_with_exclusions, is_transform=is_transform,
                not_indexed_same=not is_transform)
            if self._grouper.has_dropped_na and is_transform:
                result = self._set_result_index_ordered(result)
            return result

        @final
        def _concat_objects(self, values, not_indexed_same=False,
            is_transform=False):
            from pandas.core.reshape.concat import concat
            if self.group_keys and not is_transform:
                if self.as_index:
                    group_keys = self._grouper.result_index
                    group_levels = self._grouper.levels
                    group_names = self._grouper.names
                    result = concat(values, axis=0, keys=group_keys, levels
                        =group_levels, names=group_names, sort=False)
                else:
                    result = concat(values, axis=0)
            elif not not_indexed_same:
                result = concat(values, axis=0)
                ax = self._selected_obj.index
                if self.dropna:
                    labels = self._grouper.ids
                    mask = labels != -1
                    ax = ax[mask]
                if ax.has_duplicates and not result.axes[0].equals(ax):
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
        def _set_result_index_ordered(self, result):
            index: Index = self.obj.index
            if self._grouper.is_monotonic and not self._grouper.has_dropped_na:
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
        def _insert_inaxis_grouper(self, result, qs=None):
            if isinstance(result, Series):
                result = cast(DataFrame, result.to_frame())
            n_groupings = len(self._grouper.groupings)
            if qs is not None:
                result.insert(0, f'level_{n_groupings}', np.tile(qs, len(
                    result) // len(qs)))
            for level, (name, lev) in enumerate(zip(reversed(self._grouper.
                names), self._grouper.get_group_levels())):
                if name is None:
                    name = ('index' if n_groupings == 1 and qs is None else
                        f'level_{n_groupings - level - 1}')
                if name not in result.columns:
                    if qs is None:
                        result.insert(0, name, lev)
                    else:
                        result.insert(0, name, Index(np.repeat(lev, len(qs))))
            return result

        @final
        def _wrap_aggregated_output(self, result, qs=None):
            """
            Wraps the output of GroupBy aggregations into the expected result.

            Parameters
            ----------
            result : Series, DataFrame

            Returns
            -------
            Series or DataFrame
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

        def _wrap_applied_output(self, data, values, not_indexed_same=False,
            is_transform=False):
            raise AbstractMethodError(self)

        @final
        def _numba_prep(self, data):
            ngroups = self._grouper.ngroups
            sorted_index = self._grouper.result_ilocs
            sorted_ids = self._grouper._sorted_ids
            sorted_data = data.take(sorted_index, axis=0).to_numpy()
            index_data = data.index
            if isinstance(index_data, MultiIndex):
                if len(self._grouper.groupings) > 1:
                    raise NotImplementedError(
                        "Grouping with more than 1 grouping labels and a MultiIndex is not supported with engine='numba'"
                        )
                group_key = self._grouper.groupings[0].name
                index_data = index_data.get_level_values(group_key)
            sorted_index_data = index_data.take(sorted_index).to_numpy()
            starts, ends = lib.generate_slices(sorted_ids, ngroups)
            return starts, ends, sorted_index_data, sorted_data

        def _numba_agg_general(self, func, dtype_mapping, engine_kwargs, **
            aggregator_kwargs: Any):
            """
            Perform groupby with a standard numerical aggregation function (e.g. mean)
            with Numba.
            """
            if not self.as_index:
                raise NotImplementedError(
                    'as_index=False is not supported. Use .reset_index() instead.'
                    )
            data = self._obj_with_exclusions
            df = data if data.ndim == 2 else data.to_frame()
            aggregator = executor.generate_shared_aggregator(func,
                dtype_mapping, True, **get_jit_arguments(engine_kwargs))
            ids = self._grouper.ids
            ngroups = self._grouper.ngroups
            res_mgr = df._mgr.apply(aggregator, labels=ids, ngroups=ngroups,
                **aggregator_kwargs)
            res_mgr.axes[1] = self._grouper.result_index
            result = df._constructor_from_mgr(res_mgr, axes=res_mgr.axes)
            if data.ndim == 1:
                result = cast(NDFrameT, result.squeeze('columns'))
                result.name = data.name
            else:
                result.columns = data.columns
            return result

        @final
        def _transform_with_numba(self, func, *args: Any, engine_kwargs: (
            dict[str, bool] | None)=None, **kwargs: Any):
            """
            Perform groupby transform routine with the numba engine.

            This routine mimics the data splitting routine of the DataSplitter class
            to generate the indices of each group in the sorted data and then passes the
            data and indices into a Numba jitted function.
            """
            data = self._obj_with_exclusions
            index_sorting = self._grouper.result_ilocs
            df = data if data.ndim == 2 else data.to_frame()
            starts, ends, sorted_index, sorted_data = self._numba_prep(df)
            numba_.validate_udf(func)
            args, kwargs = prepare_function_arguments(func, args, kwargs,
                num_required_args=2)
            numba_transform_func = numba_.generate_numba_transform_func(func,
                **get_jit_arguments(engine_kwargs))
            result = numba_transform_func(sorted_data, sorted_index, starts,
                ends, len(df.columns), *args)
            result = result.take(np.argsort(index_sorting), axis=0)
            index = data.index
            if data.ndim == 1:
                result_kwargs: dict[str, Any] = {'name': data.name}
                result = result.ravel()
            else:
                result_kwargs = {'columns': data.columns}
            return data._constructor(result, index=index, **result_kwargs)

        @final
        def _aggregate_with_numba(self, func, *args: Any, engine_kwargs: (
            dict[str, bool] | None)=None, **kwargs: Any):
            """
            Perform groupby aggregation routine with the numba engine.

            This routine mimics the data splitting routine of the DataSplitter class
            to generate the indices of each group in the sorted data and then passes the
            data and indices into a Numba jitted function.
            """
            data = self._obj_with_exclusions
            df = data if data.ndim == 2 else data.to_frame()
            starts, ends, sorted_index, sorted_data = self._numba_prep(df)
            numba_.validate_udf(func)
            args, kwargs = prepare_function_arguments(func, args, kwargs,
                num_required_args=2)
            numba_agg_func = numba_.generate_numba_agg_func(func, **
                get_jit_arguments(engine_kwargs))
            result = numba_agg_func(sorted_data, sorted_index, starts, ends,
                len(df.columns), *args)
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

        def apply(self, func, *args: Any, include_groups: bool=False, **
            kwargs: Any):
            """
            Apply function ``func`` group-wise and combine the results together.

            The function passed to ``apply`` must take a dataframe as its first
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
            func : Callable[..., Any]
                A callable that takes a dataframe as its first argument, and
                returns a dataframe, a series or a scalar. In addition the
                callable may take positional and keyword arguments.

            *args : Tuple[Any, ...]
                Optional positional arguments to pass to ``func``.

            include_groups : bool, default False
                When True, will attempt to apply ``func`` to the groupings in
                the case that they are columns of the DataFrame. If this raises a
                TypeError, the result will be computed with the groupings excluded.
                When False, the groupings will be excluded when applying ``func``.

                .. versionadded:: 2.2.0

                .. versionchanged:: 3.0.0

                The default changed from True to False, and True is no longer allowed.

            **kwargs : dict[str, Any]
                Optional keyword arguments to pass to ``func``.

            Returns
            -------
            NDFrameT
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
            >>> df = pd.DataFrame({"A": "a a b".split(), "B": [1, 2, 3], "C": [4, 6, 5]})
            >>> g1 = df.groupby("A", group_keys=False)
            >>> g2 = df.groupby("A", group_keys=True)

            Notice that ``g1`` and ``g2`` have two groups, ``a`` and ``b``, and only
            differ in their ``group_keys`` argument. Calling `apply` in various ways,
            we can get different grouping results:

            Example 1: below the function passed to `apply` takes a DataFrame as
            its argument and returns a DataFrame. `apply` combines the result for
            each group together into a new DataFrame:

            >>> g1[["B", "C"]].apply(lambda x: x / x.sum())
                      B    C
            0  0.333333  0.4
            1  0.666667  0.6
            2  1.000000  1.0

            In the above, the groups are not part of the index. We can have them included
            by using ``g2`` where ``group_keys=True``:

            >>> g2[["B", "C"]].apply(lambda x: x / x.sum())
                        B    C
            A
            a 0  0.333333  0.4
              1  0.666667  0.6
            b 2  1.000000  1.0

            Example 2: The function passed to `apply` takes a DataFrame as
            its argument and returns a Series.  `apply` combines the result for
            each group together into a new DataFrame.

            .. versionchanged:: 1.3.0

                The resulting dtype will reflect the return value of the passed ``func``.

            >>> g1[["B", "C"]].apply(lambda x: x.astype(float).max() - x.min())
                 B    C
            A
            a  1.0  2.0
            b  0.0  0.0

            >>> g2[["B", "C"]].apply(lambda x: x.astype(float).max() - x.min())
                 B    C
            A
            a  1.0  2.0
            b  0.0  0.0

            The ``group_keys`` argument has no effect here because the result is not
            like-indexed (i.e. :ref:`a transform <groupby.transform>`) when compared
            to the input.

            Example 3: The function passed to `apply` takes a DataFrame as
            its argument and returns a scalar. `apply` combines the result for
            each group together into a Series, including setting the index as
            appropriate:

            >>> g1.apply(lambda x: x.C.max() - x.B.min())
            A
            a    5
            b    2
            dtype: int64

            Example 4: The function passed to ``apply`` returns ``None`` for one of the
            group. This group is filtered from the result:

            >>> g1.apply(lambda x: None if x.iloc[0, 0] == 3 else x)
               B  C
            0  1  4
            1  2  6
            """
            if include_groups:
                raise ValueError('include_groups=True is no longer allowed.')
            if isinstance(func, str):
                if hasattr(self, func):
                    res = getattr(self, func)
                    if callable(res):
                        return res(*args, **kwargs)
                    elif args or kwargs:
                        raise ValueError(
                            f'Cannot pass arguments to property {func}')
                    return res
                else:
                    raise TypeError(
                        f"apply func should be callable, not '{func}'")
            elif args or kwargs:
                if callable(func):

                    @wraps(func)
                    def f(g):
                        return func(g, *args, **kwargs)
                else:
                    raise ValueError(
                        'func must be a callable if args or kwargs are supplied'
                        )
            else:
                f: Callable[[NDFrameT], Any] = func
            return self._python_apply_general(f, self._obj_with_exclusions)

        @final
        def _python_apply_general(self, f, data, not_indexed_same=None,
            is_transform=False, is_agg=False):
            """
            Apply function f in python space

            Parameters
            ----------
            f : Callable[[NDFrameT], Any]
                Function to apply
            data : NDFrameT
                Data to apply f to
            not_indexed_same: bool, optional
                When specified, overrides the value of not_indexed_same. Apply behaves
                differently when the result index is equal to the input index, but
                this can be coincidental leading to value-dependent behavior.
            is_transform : bool, default False
                Indicator for whether the function is actually a transform
                and should not have group keys prepended.
            is_agg : bool, default False
                Indicator for whether the function is an aggregation. When the
                result is empty, we don't want to warn for this case.
                See _GroupBy._python_agg_general.

            Returns
            -------
            NDFrameT
                data after applying f
            """
            values, mutated = self._grouper.apply_groupwise(f, data)
            if not_indexed_same is None:
                not_indexed_same = mutated
            return self._wrap_applied_output(data, values, not_indexed_same,
                is_transform)

        @final
        def _agg_general(self, numeric_only=False, min_count=-1, *, alias:
            str, npfunc: (Callable[..., Any] | None)=None, **kwargs: Any):
            result = self._cython_agg_general(how=alias, alt=npfunc,
                numeric_only=numeric_only, min_count=min_count, **kwargs)
            return cast(NDFrameT, result.__finalize__(self.obj, method=
                'groupby'))

        def _agg_py_fallback(self, how, values, ndim, alt):
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
                res_values = self._grouper.agg_series(ser, alt,
                    preserve_dtype=True)
            except Exception as err:
                msg = f'agg function failed [how->{how},dtype->{ser.dtype}]'
                raise type(err)(msg) from err
            if ser.dtype == object:
                res_values = res_values.astype(object, copy=False)
            return ensure_block_shape(res_values, ndim=ndim)

        @final
        def _cython_agg_general(self, how, alt=None, numeric_only=False,
            min_count=-1, **kwargs: Any):
            data = self._get_data_to_aggregate(numeric_only=numeric_only,
                name=how)

            def array_func(values):
                try:
                    result = self._grouper._cython_operation('aggregate',
                        values, how, axis=data.ndim - 1, min_count=
                        min_count, **kwargs)
                except NotImplementedError:
                    if how in ['any', 'all'] and isinstance(values, SparseArray
                        ):
                        pass
                    elif alt is None or how in ['any', 'all', 'std', 'sem']:
                        raise
                else:
                    return result
                assert alt is not None
                result = self._agg_py_fallback(how, values, ndim=data.ndim,
                    alt=alt)
                return result
            new_mgr = data.grouped_reduce(array_func)
            res = self._wrap_agged_manager(new_mgr)
            if how in ['idxmin', 'idxmax']:
                res = self._wrap_idxmax_idxmin(res)
            out = self._wrap_aggregated_output(res)
            return out

        def _cython_transform(self, how, numeric_only=False, skipna=True,
            **kwargs: Any):
            raise AbstractMethodError(self)

        @final
        def _transform(self, func, *args: Any, engine: (Literal['cython',
            'numba'] | None)=None, engine_kwargs: (dict[str, bool] | None)=
            None, **kwargs: Any):
            if not isinstance(func, str):
                return self._transform_general(func, engine, engine_kwargs,
                    *args, **kwargs)
            elif func not in base.transform_kernel_allowlist:
                msg = (
                    f"'{func}' is not a valid function name for transform(name)"
                    )
                raise ValueError(msg)
            elif func in base.cythonized_kernels or func in base.transformation_kernels:
                if engine is not None:
                    kwargs['engine'] = engine
                    kwargs['engine_kwargs'] = engine_kwargs
                return getattr(self, func)(*args, **kwargs)
            else:
                if self.observed:
                    return self._reduction_kernel_transform(func, *args,
                        engine=engine, engine_kwargs=engine_kwargs, **kwargs)
                with com.temp_setattr(self, 'observed', True
                    ), com.temp_setattr(self, '_grouper', self._grouper.
                    observed_grouper):
                    return self._reduction_kernel_transform(func, *args,
                        engine=engine, engine_kwargs=engine_kwargs, **kwargs)

        @final
        def _reduction_kernel_transform(self, func, *args: Any, engine: (
            Literal['cython', 'numba'] | None)=None, engine_kwargs: (dict[
            str, bool] | None)=None, **kwargs: Any):
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
        def _wrap_transform_fast_result(self, result):
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
                output = result._reindex_with_indexers({(0): (new_ax, ids)},
                    allow_dups=True)
                output = output.set_axis(obj.index, axis=0)
            return output

        @final
        def _apply_filter(self, indices, dropna):
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
                mask = np.tile(mask, list(self._selected_obj.shape[1:]) + [1]
                    ).T
                filtered = self._selected_obj.where(mask)
            return filtered

        @final
        def _cumcount_array(self, ascending=True):
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
            ids, count = ids[sorter], len(ids)
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
                out = np.where(ids == -1, np.nan, out.astype(np.float64,
                    copy=False))
            else:
                out = out.astype(np.int64, copy=False)
            rev = np.empty(count, dtype=np.intp)
            rev[sorter] = np.arange(count, dtype=np.intp)
            return out[rev]

        @final
        @property
        def _obj_1d_constructor(self):
            if isinstance(self.obj, DataFrame):
                return self.obj._constructor_sliced
            assert isinstance(self.obj, Series)
            return self.obj._constructor

        @final
        @Substitution(name='groupby')
        @Substitution(see_also=_common_see_also)
        def any(self, skipna=True):
            """
            Return True if any value in the group is truthful, else False.

            Parameters
            ----------
            skipna : bool, default True
                Flag to ignore nan values during truth testing.

            Returns
            -------
            Series or DataFrame
                DataFrame or Series of boolean values, where a value is True if any element
                is True within its respective group, False otherwise.
            %(see_also)s
            Examples
            --------

            For SeriesGroupBy:

            >>> lst = ["a", "a", "b"]
            >>> ser = pd.Series([1, 2, 0], index=lst)
            >>> ser
            a    1
            a    2
            b    0
            dtype: int64
            >>> ser.groupby(level=0).any()
            a     True
            b    False
            dtype: bool

            For DataFrameGroupBy:

            >>> data = [[1, 0, 3], [1, 0, 6], [2, 5, 8], [2, 6, 9]]
            >>> df = pd.DataFrame(
            ...     data, columns=["a", "b", "c"], index=["cow", "horse", "catfish", "lion"]
            ... )
            >>> df
                      a  b  c
              cow     1  0  3
              horse   1  0  6
              catfish   2  5  8
               lion    2  6  9
            >>> df.groupby(by=["a"]).any()
                   b      c
            a
            1  False   True
            2   True   True

            For Resampler:

            >>> ser = pd.Series(
            ...     [1, 2, 3, 4],
            ...     index=pd.DatetimeIndex(
            ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
            ...     ),
            ... )
            >>> ser
            2023-01-01    1
            2023-01-15    2
            2023-02-01    3
            2023-02-15    4
            dtype: int64
            >>> ser.resample("MS").any()
            2023-01-01    1
            2023-02-01    1
            Freq: MS, dtype: int64
            """
            return self._cython_agg_general('any', alt=lambda x: Series(x,
                copy=False).any(skipna=skipna), skipna=skipna)

        @final
        @Substitution(name='groupby')
        @Substitution(see_also=_common_see_also)
        def all(self, skipna=True):
            """
            Return True if all values in the group are truthful, else False.

            Parameters
            ----------
            skipna : bool, default True
                Flag to ignore nan values during truth testing.

            Returns
            -------
            Series or DataFrame
                DataFrame or Series of boolean values, where a value is True if all elements
                are True within its respective group, False otherwise.
            %(see_also)s
            Examples
            --------

            For SeriesGroupBy:

            >>> lst = ["a", "a", "b"]
            >>> ser = pd.Series([1, 2, 0], index=lst)
            >>> ser
            a    1
            a    2
            b    0
            dtype: int64
            >>> ser.groupby(level=0).all()
            a     True
            b    False
            dtype: bool

            For DataFrameGroupBy:

            >>> data = [[1, 0, 3], [1, 5, 6], [2, 5, 8], [2, 6, 9]]
            >>> df = pd.DataFrame(
            ...     data, columns=["a", "b", "c"], index=["cow", "horse", "catfish", "lion"]
            ... )
            >>> df
                      a  b  c
              cow     1  0  3
              horse   1  5  6
              catfish   2  5  8
               lion    2  6  9
            >>> df.groupby(by=["a"]).all()
                   b      c
            a
            1  False   True
            2   True   True
            """
            return self._cython_agg_general('all', alt=lambda x: Series(x,
                copy=False).all(skipna=skipna), skipna=skipna)

        @final
        @Substitution(name='groupby')
        @Substitution(see_also=_common_see_also)
        def count(self):
            """
            Compute count of group, excluding missing values.

            Returns
            -------
            Series or DataFrame
                Count of values within each group.
            %(see_also)s
            Examples
            --------

            For SeriesGroupBy:

            >>> lst = ["a", "a", "b"]
            >>> ser = pd.Series([1, 2, np.nan], index=lst)
            >>> ser
            a    1.0
            a    2.0
            b    NaN
            dtype: float64
            >>> ser.groupby(level=0).count()
            a    2
            b    0
            dtype: int64

            For DataFrameGroupBy:

            >>> data = [[1, np.nan, 3], [1, np.nan, 6], [2, 8, 9]]
            >>> df = pd.DataFrame(
            ...     data, columns=["a", "b", "c"], index=["cow", "horse", "bull"]
            ... )
            >>> df
                    a	  b	c
            cow     1	NaN	3
            horse   1	NaN	6
            bull    2	8.0	9
            >>> df.groupby("a").count()
                b   c
            a
            1   0   2
            2   1   1

            For Resampler:

            >>> ser = pd.Series(
            ...     [1, 2, 3, 4],
            ...     index=pd.DatetimeIndex(
            ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
            ...     ),
            ... )
            >>> ser
            2023-01-01    1
            2023-01-15    2
            2023-02-01    3
            2023-02-15    4
            dtype: int64
            >>> ser.resample("MS").count()
            2023-01-01    2
            2023-02-01    2
            Freq: MS, dtype: int64
            """
            data = self._get_data_to_aggregate()
            ids = self._grouper.ids
            ngroups = self._grouper.ngroups
            mask = ids != -1
            is_series = data.ndim == 1

            def hfunc(bvalues):
                if bvalues.ndim == 1:
                    masked = mask & ~isna(bvalues).reshape(1, -1)
                else:
                    masked = mask & ~isna(bvalues)
                counted = lib.count_level_2d(masked, labels=ids, max_bin=
                    ngroups)
                if isinstance(bvalues, BaseMaskedArray):
                    return IntegerArray(counted[0], mask=np.zeros(counted.
                        shape[1], dtype=np.bool_))
                elif isinstance(bvalues, ArrowExtensionArray
                    ) and not isinstance(bvalues.dtype, StringDtype):
                    dtype = pandas_dtype('int64[pyarrow]')
                    return type(bvalues)._from_sequence(counted[0], dtype=dtype
                        )
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
        def mean(self, numeric_only=False, skipna=True, engine=None,
            engine_kwargs=None):
            """
            Compute mean of groups, excluding missing values.

            Parameters
            ----------
            numeric_only : bool, default False
                Include only float, int, boolean columns.

                .. versionchanged:: 2.0.0

                    numeric_only no longer accepts ``None``.

            skipna : bool, default True
                Exclude NA/null values. If an entire group is NA, the result will be NA.

                .. versionadded:: 3.0.0

            engine : str, default None
                * ``'cython'`` : Runs the operation through C-extensions from cython.
                * ``'numba'`` : Runs the operation through JIT compiled code from numba.
                * ``None`` : Defaults to ``'cython'`` or globally setting
                  ``compute.use_numba``

                .. versionadded:: 1.4.0

            engine_kwargs : dict[str, bool] | None, default None
                * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
                * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
                  and ``parallel`` dictionary keys. The values must either be ``True`` or
                  ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
                  ``{{'nopython': True, 'nogil': False, 'parallel': False}}``
    
                .. versionadded:: 1.4.0

            Returns
            -------
            pandas.Series or pandas.DataFrame
                Mean of values within each group. Same object type as the caller.
            %(see_also)s
            Examples
            --------
            >>> df = pd.DataFrame(
            ...     {"A": "a a b".split(), "B": [np.nan, 2, 3], "C": [1, 2, 1]},
            ...     columns=["A", "B", "C"],
            ... )
            >>> g1 = df.groupby("A", group_keys=False)
            >>> g2 = df.groupby("A", group_keys=True)

            Groupby one column and return the mean of the remaining columns in
            each group.

            >>> g1["B", "C"].mean()
                 B         C
            a  2.0  1.333333
            b  3.0  1.500000

            Groupby two columns and return the mean of the remaining column.

            >>> g1.groupby(["A", "B"]).mean()
                     C
            A B
            a NaN  1.0
            a  2.0  2.0
            b  3.0  1.0

            Groupby one column and return the mean of only particular column in
            the group.

            >>> g1["B"].mean()
            A
            a    2.0
            b    3.0
            Name: B, dtype: float64
            """
            if maybe_use_numba(engine):
                from pandas.core._numba.kernels import grouped_mean
                return self._numba_agg_general(grouped_mean, executor.
                    float_dtype_mapping, engine_kwargs, min_periods=0,
                    skipna=skipna)
            else:
                result = self._cython_agg_general('mean', alt=lambda x:
                    Series(x, copy=False).mean(numeric_only=numeric_only,
                    skipna=skipna), numeric_only=numeric_only, skipna=skipna)
                return cast(NDFrameT, result.__finalize__(self.obj, method=
                    'groupby'))

        @final
        def median(self, numeric_only=False, skipna=True):
            """
            Compute median of groups, excluding missing values.

            For multiple groupings, the result index will be a MultiIndex

            Parameters
            ----------
            numeric_only : bool, default False
                Include only float, int, boolean columns.
            skipna : bool, default True
                Exclude NA/null values. If an entire group is NA, the result will be NA.

                .. versionadded:: 3.0.0

            Returns
            -------
            Series or DataFrame
                Median of values within each group.

            See Also
            --------
            Series.groupby : Apply a function groupby to a Series.
            DataFrame.groupby : Apply a function groupby to each row or column of a
                DataFrame.

            Examples
            --------
            For SeriesGroupBy:

            >>> lst = ["a", "a", "a", "b", "b", "b"]
            >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
            >>> ser
            a     7
            a     2
            a     8
            b     4
            b     3
            b     3
            dtype: int64
            >>> ser.groupby(level=0).median()
            a    7.0
            b    3.0
            dtype: float64

            For DataFrameGroupBy:

            >>> data = {"a": [1, 3, 5, 7, 7, 8, 3], "b": [1, 4, 8, 4, 4, 2, 1]}
            >>> df = pd.DataFrame(
            ...     data, columns=["a", "b", "c"], index=["dog", "dog", "dog", "mouse", "mouse", "mouse", "mouse"]
            ... )
            >>> df
                       a  b  c
                dog    1  1  3
                dog    3  4  6
                dog    5  8  9
              mouse    7  4  9
              mouse    7  4  8
              mouse    8  2  1
              mouse    3  1  1
            >>> df.groupby("a").median()
                 b    c
            a
            1   1.0  3.0
            3   2.5  3.0
            5   8.0  9.0
            7   4.0  8.5
            8   2.0  1.0

            For Resampler:

            >>> ser = pd.Series(
            ...     [1, 2, 3, 3, 4, 5],
            ...     index=pd.DatetimeIndex(
            ...         [
            ...             "2023-01-01",
            ...             "2023-01-10",
            ...             "2023-01-15",
            ...             "2023-02-01",
            ...             "2023-02-10",
            ...             "2023-02-15",
            ...         ]
            ...     ),
            ... )
            >>> ser.resample("MS").median()
            2023-01-01    2.0
            2023-02-01    4.0
            Freq: MS, dtype: float64
            """
            result = self._cython_agg_general('median', alt=lambda x:
                Series(x, copy=False).median(numeric_only=numeric_only,
                skipna=skipna), numeric_only=numeric_only, skipna=skipna)
            return cast(NDFrameT, result.__finalize__(self.obj, method=
                'groupby'))

        @final
        @Substitution(name='groupby')
        @Substitution(see_also=_common_see_also)
        def std(self, ddof=1, numeric_only=False, skipna=True, engine=None,
            engine_kwargs=None):
            """
            Compute standard deviation of groups, excluding missing values.

            For multiple groupings, the result index will be a MultiIndex.

            Parameters
            ----------
            ddof : int, default 1
                Delta Degrees of Freedom. The divisor used in calculations is ``N - ddof``,
                where ``N`` represents the number of elements.

            numeric_only : bool, default False
                Include only `float`, `int` or `boolean` data.

                .. versionadded:: 1.5.0

                .. versionchanged:: 2.0.0

                    numeric_only now defaults to ``False``.

            skipna : bool, default True
                Exclude NA/null values. If an entire group is NA, the result will be NA.

                .. versionadded:: 3.0.0

            engine : str, default None
                * ``'cython'`` : Runs the operation through C-extensions from cython.
                * ``'numba'`` : Runs the operation through JIT compiled code from numba.
                * ``None`` : Defaults to ``'cython'`` or globally setting
                  ``compute.use_numba``

                .. versionadded:: 1.4.0

            engine_kwargs : dict[str, bool] | None, default None
                * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
                * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
                  and ``parallel`` dictionary keys. The values must either be ``True`` or
                  ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
                  ``{{'nopython': True, 'nogil': False, 'parallel': False}}``
    
                .. versionadded:: 1.4.0

            Returns
            -------
            pandas.Series or pandas.DataFrame
                Standard deviation of values within each group.
            %(see_also)s
            Examples
            --------
            For SeriesGroupBy:

            >>> lst = ["a", "a", "a", "b", "b", "b"]
            >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
            >>> ser
            a     7
            a     2
            a     8
            b     4
            b     3
            b     3
            dtype: int64
            >>> ser.groupby(level=0).std()
            a    3.21455
            b    0.57735
            dtype: float64

            For DataFrameGroupBy:

            >>> data = {"a": [1, 3, 5, 7, 7, 8, 3], "b": [1, 4, 8, 4, 4, 2, 1]}
            >>> df = pd.DataFrame(
            ...     data, columns=["a", "b", "c"], index=["dog", "dog", "dog", "mouse", "mouse", "mouse", "mouse"]
            ... )
            >>> df
                       a  b  c
                dog    1  1  3
                dog    3  4  6
                dog    5  8  9
              mouse    7  4  9
              mouse    7  4  8
              mouse    8  2  1
              mouse    3  1  1
            >>> df.groupby("a").std()
                          a         b
            a    2.000000  3.511885
            3.666667  2.217356  1.500000
            7.0       0.000000  0.000000
            8.0       0.000000  0.000000
            """
            if maybe_use_numba(engine):
                from pandas.core._numba.kernels import grouped_var
                return np.sqrt(self._numba_agg_general(grouped_var,
                    executor.float_dtype_mapping, engine_kwargs,
                    min_periods=0, ddof=ddof, skipna=skipna))
            else:
                return self._cython_agg_general('std', alt=lambda x: Series
                    (x, copy=False).std(ddof=ddof, skipna=skipna),
                    numeric_only=numeric_only, ddof=ddof, skipna=skipna)

        @final
        def var(self, ddof=1, numeric_only=False, skipna=True):
            """
            Compute variance of groups, excluding missing values.

            For multiple groupings, the result index will be a MultiIndex.

            Parameters
            ----------
            ddof : int, default 1
                Degrees of freedom.

            numeric_only : bool, default False
                Include only float, int, boolean columns.

                .. versionadded:: 1.5.0

                .. versionchanged:: 2.0.0

                    numeric_only now defaults to ``False``.

            skipna : bool, default True
                Exclude NA/null values. If an entire group is NA, the result will be NA.

                .. versionadded:: 3.0.0

            Returns
            -------
            pandas.Series or pandas.DataFrame
                Variance of values within each group.
            %(see_also)s
            Examples
            --------

            For SeriesGroupBy:

            >>> lst = ["a", "a", "a", "b", "b", "b"]
            >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
            >>> ser
            a     7
            a     2
            a     8
            b     4
            b     3
            b     3
            dtype: int64
            >>> ser.groupby(level=0).var()
            a    10.333333
            b     0.333333
            dtype: float64

            For DataFrameGroupBy:

            >>> data = {"a": [1, 3, 5, 7, 7, 8, 3], "b": [1, 4, 8, 4, 4, 2, 1]}
            >>> df = pd.DataFrame(
            ...     data, columns=["a", "b", "c"], index=["dog", "dog", "dog", "mouse", "mouse", "mouse", "mouse"]
            ... )
            >>> df
                       a  b  c
                dog    1  1  3
                dog    3  4  6
                dog    5  8  9
              mouse    7  4  9
              mouse    7  4  8
              mouse    8  2  1
              mouse    3  1  1
            >>> df.groupby("a").var()
                          a          b
            a    4.000000  12.333333
            3.666667  2.217356   1.500000
            7.0       0.000000   0.000000
            8.0       0.000000   0.000000
            """
            return self._cython_agg_general('var', alt=lambda x: Series(x,
                copy=False).var(ddof=ddof, skipna=skipna), numeric_only=
                numeric_only, ddof=ddof, skipna=skipna)

        @final
        def _value_counts(self, subset=None, normalize=False, sort=True,
            ascending=False, dropna=True):
            """
            Shared implementation of value_counts for SeriesGroupBy and DataFrameGroupBy.

            SeriesGroupBy additionally supports a bins argument. See the docstring of
            DataFrameGroupBy.value_counts for a description of arguments.
            """
            name = 'proportion' if normalize else 'count'
            df = self.obj
            obj = self._obj_with_exclusions
            in_axis_names: set[Hashable] = {grouping.name for grouping in
                self._grouper.groupings if grouping.in_axis}
            if isinstance(obj, Series):
                _name = obj.name
                keys: Iterable[Series] = [] if _name in in_axis_names else [obj
                    ]
            else:
                unique_cols = set(obj.columns)
                if subset is not None:
                    subsetted = set(subset)
                    clashing = subsetted & set(in_axis_names)
                    if clashing:
                        raise ValueError(
                            f'Keys {clashing} in subset cannot be in the groupby column keys.'
                            )
                    doesnt_exist = subsetted - unique_cols
                    if doesnt_exist:
                        raise ValueError(
                            f'Keys {doesnt_exist} in subset do not exist in the DataFrame.'
                            )
                else:
                    subsetted = unique_cols
                keys = (obj.iloc[:, idx] for idx, _name in enumerate(obj.
                    columns) if _name not in in_axis_names and _name in
                    subsetted)
            groupings = list(self._grouper.groupings)
            for key in keys:
                grouper, _, _ = get_grouper(df, key=key, sort=False,
                    observed=False, dropna=dropna)
                groupings += list(grouper.groupings)
            gb = df.groupby(groupings, sort=False, observed=self.observed,
                dropna=self.dropna)
            result_series = cast(Series, gb.size())
            result_series.name = name
            if sort:
                result_series = result_series.sort_values(ascending=
                    ascending, kind='stable')
            if self.sort:
                names = result_series.index.names
                result_series.index.names = range(len(names))
                index_level = range(len(self._grouper.groupings))
                result_series = result_series.sort_index(level=index_level,
                    sort_remaining=False)
                result_series.index.names = names
            if normalize:
                levels = list(range(len(self._grouper.groupings),
                    result_series.index.nlevels))
                indexed_group_size = result_series.groupby(result_series.
                    index.droplevel(levels), sort=self.sort, dropna=self.
                    dropna, observed=False).transform('sum')
                result_series /= indexed_group_size
                result_series = result_series.fillna(0.0)
            result: Union[Series, DataFrame]
            if self.as_index:
                result = result_series
            else:
                index = result_series.index
                columns = com.fill_missing_names(index.names)
                if name in columns:
                    raise ValueError(
                        f"Column label '{name}' is duplicate of result column")
                result_series.name = name
                result_series.index = index.set_names(range(len(columns)))
                result_frame = result_series.reset_index()
                orig_dtype = self._grouper.groupings[0].obj.columns.dtype
                cols = Index(columns, dtype=orig_dtype).insert(len(columns),
                    name)
                result_frame.columns = cols
                result = result_frame
            return result.__finalize__(self.obj, method='value_counts')

        @final
        def _insert_quantile_level(idx, qs):
            """
            Insert the sequence 'qs' of quantiles as the inner-most level of a MultiIndex.

            The quantile level in the MultiIndex is a repeated copy of 'qs'.

            Parameters
            ----------
            idx : Index
            qs : np.ndarray[float64]

            Returns
            -------
            MultiIndex
            """
            nqs = len(qs)
            lev_codes, lev = Index(qs).factorize()
            lev_codes = coerce_indexer_dtype(lev_codes, lev)
            if idx._is_multi:
                idx = cast(MultiIndex, idx)
                levels = list(idx.levels) + [lev]
                codes = [np.repeat(x, nqs) for x in idx.codes] + [np.tile(
                    lev_codes, len(idx))]
                mi = MultiIndex(levels=levels, codes=codes, names=idx.names +
                    [None])
            else:
                nidx = len(idx)
                idx_codes = coerce_indexer_dtype(np.arange(nidx), idx)
                levels = [idx, lev]
                codes = [np.repeat(idx_codes, nqs), np.tile(lev_codes, nidx)]
                mi = MultiIndex(levels=levels, codes=codes, names=[idx.name,
                    None])
            return mi

        @final
        @Substitution(name='groupby')
        @Substitution(see_also=_common_see_also)
        def sum(self, numeric_only=False, min_count=0, skipna=True, engine=
            None, engine_kwargs=None):
            if maybe_use_numba(engine):
                from pandas.core._numba.kernels import grouped_sum
                return self._numba_agg_general(grouped_sum, executor.
                    default_dtype_mapping, engine_kwargs, min_periods=0,
                    skipna=skipna)
            else:
                with com.temp_setattr(self, 'observed', True):
                    result = self._agg_general(numeric_only=numeric_only,
                        min_count=min_count, skipna=skipna, alias='sum',
                        npfunc=np.sum)
                return cast(NDFrameT, result)

        @final
        def prod(self, numeric_only=False, min_count=0, skipna=True):
            """
            Compute prod of group values.

            Parameters
            ----------
            numeric_only : bool, default False
                Include only float, int, boolean columns.
            min_count : int, default 0
                The required number of valid values to perform the operation. If fewer
                than ``min_count`` non-NA values are present the result will be NA.
            skipna : bool, default True
                Exclude NA/null values. If an entire group is NA, the result will be NA.

                .. versionadded:: 3.0.0

            Returns
            -------
            Series or DataFrame
                Computed prod of values within each group.
            %(see_also)s
            Examples
            --------
            For SeriesGroupBy:

            >>> lst = ["a", "a", "b"]
            >>> ser = pd.Series([1, 2, 3], index=lst)
            >>> ser
            a    1
            a    2
            b    3
            dtype: int64
            >>> ser.groupby(level=0).prod()
            a    2
            b    3
            dtype: int64

            For DataFrameGroupBy:

            >>> data = [[1, 2, 3], [1, 5, 6], [2, 5, 8], [2, 6, 9]]
            >>> df = pd.DataFrame(
            ...     data, columns=["a", "b", "c"], index=["dog", "dog", "catfish", "lion"]
            ... )
            >>> df
                       a  b  c
                dog    1  2  3
                dog    1  5  6
              catfish   2  5  8
               lion    2  6  9
            >>> df.groupby("a").prod()
                b    c
            a
            1   10   18
            2   30   72
            """
            return self._agg_general(numeric_only=numeric_only, min_count=
                min_count, skipna=skipna, alias='prod', npfunc=np.prod)

        @final
        def _idxmax_idxmin(self, how, ignore_unobserved=False, skipna=True,
            numeric_only=False):
            """Compute idxmax/idxmin.

            Parameters
            ----------
            how : {'idxmin', 'idxmax'}
                Whether to compute idxmin or idxmax.
            numeric_only : bool, default False
                Include only float, int, boolean columns.
            skipna : bool, default True
                Exclude NA/null values. If an entire group is NA, the result will be NA.
            ignore_unobserved : bool, default False
                When True and an unobserved group is encountered, do not raise. This used
                for transform where unobserved groups do not play an impact on the result.

            Returns
            -------
            Series or DataFrame
                idxmax or idxmin for the groupby operation.
            """
            if not self.observed and any(ping._passed_categorical for ping in
                self._grouper.groupings):
                expected_len = len(self._grouper.result_index)
                group_sizes = self._grouper.size()
                result_len = group_sizes[group_sizes > 0].shape[0]
                assert result_len <= expected_len
                has_unobserved = result_len < expected_len
                raise_err: bool = not ignore_unobserved and has_unobserved
                data = self._obj_with_exclusions
                if raise_err and isinstance(data, DataFrame):
                    if numeric_only:
                        data = data._get_numeric_data()
                    raise_err = len(data.columns) > 0
                if raise_err:
                    raise ValueError(
                        f"Can't get {how} of an empty group due to unobserved categories. Specify observed=True in groupby instead."
                        )
            elif not skipna and self._obj_with_exclusions.isna().any(axis=None
                ):
                raise ValueError(
                    f'{type(self).__name__}.{how} with skipna=False encountered an NA value.'
                    )
            result = self._agg_general(numeric_only=numeric_only, min_count
                =1, alias=how, skipna=skipna)
            return result

        def _wrap_idxmax_idxmin(self, res):
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
                    result = res._constructor(index.array.take(values,
                        allow_fill=True, fill_value=na_value), index=res.
                        index, name=res.name)
                else:
                    data = {}
                    for k, column_values in enumerate(values.T):
                        data[k] = index.array.take(column_values,
                            allow_fill=True, fill_value=na_value)
                    result = self.obj._constructor(data, index=res.index)
                    result.columns = res.columns
            return result

    @doc(GroupBy)
    def get_groupby(obj, by=None, grouper=None, group_keys=True):
        klass: type[GroupBy]
        if isinstance(obj, Series):
            from pandas.core.groupby.generic import SeriesGroupBy
            klass = SeriesGroupBy
        elif isinstance(obj, DataFrame):
            from pandas.core.groupby.generic import DataFrameGroupBy
            klass = DataFrameGroupBy
        else:
            raise TypeError(f'invalid type: {obj}')
        return klass(obj=obj, keys=by, grouper=grouper, group_keys=group_keys)

    def _insert_quantile_level(idx, qs):
        """
        Insert the sequence 'qs' of quantiles as the inner-most level of a MultiIndex.

        The quantile level in the MultiIndex is a repeated copy of 'qs'.

        Parameters
        ----------
        idx : Index
        qs : np.ndarray[float64]

        Returns
        -------
        MultiIndex
        """
        nqs = len(qs)
        lev_codes, lev = Index(qs).factorize()
        lev_codes = coerce_indexer_dtype(lev_codes, lev)
        if idx._is_multi:
            idx = cast(MultiIndex, idx)
            levels = list(idx.levels) + [lev]
            codes = [np.repeat(x, nqs) for x in idx.codes] + [np.tile(
                lev_codes, len(idx))]
            mi = MultiIndex(levels=levels, codes=codes, names=idx.names + [
                None])
        else:
            nidx = len(idx)
            idx_codes = coerce_indexer_dtype(np.arange(nidx), idx)
            levels = [idx, lev]
            codes = [np.repeat(idx_codes, nqs), np.tile(lev_codes, nidx)]
            mi = MultiIndex(levels=levels, codes=codes, names=[idx.name, None])
        return mi

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def sem(self, ddof=1, numeric_only=False, skipna=True):
        """
        Compute standard error of the mean of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        skipna : bool, default True
            Exclude NA/null values. If an entire group is NA, the result will be NA.

            .. versionadded:: 3.0.0

        Returns
        -------
        Series or DataFrame
            Standard error of the mean of values within each group.

        See Also
        --------
        DataFrame.sem : Return unbiased standard error of the mean over requested axis.
        Series.sem : Return unbiased standard error of the mean over requested axis.

        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ["a", "a", "b", "b"]
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).sem()
        A
        a    0.707107
        b    0.707107
        dtype: float64

        For DataFrameGroupBy:

        >>> data = {"a": [1, 3, 5, 7, 7, 8, 3], "b": [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["dog", "dog", "dog", "mouse", "mouse", "mouse", "mouse"]
        ... )
        >>> df
                       a  b  c
                dog    1  1  3
                dog    3  4  6
                dog    5  8  9
              mouse    7  4  9
              mouse    7  4  8
              mouse    8  2  1
              mouse    3  1  1
        >>> df.groupby("a").sem()
                          a         b
            a    2.000000  3.511885
            3.666667  2.217356  1.500000
            7.0       0.000000  0.000000
            8.0       0.000000  0.000000
        """
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_var
            return np.sqrt(self._numba_agg_general(grouped_var, executor.
                float_dtype_mapping, engine_kwargs, min_periods=0, ddof=
                ddof, skipna=skipna))
        else:
            return self._cython_agg_general('sem', alt=lambda x: Series(x,
                copy=False).sem(ddof=ddof, skipna=skipna), numeric_only=
                numeric_only, ddof=ddof, skipna=skipna)

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def size(self):
        """
        Compute group sizes.

        Returns
        -------
        DataFrame or Series
            Number of rows in each group as a Series if as_index is True
            or a DataFrame if as_index is False.
        %(see_also)s
        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ["a", "a", "b"]
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).size()
        a    2
        b    1
        dtype: int64

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(
        ...     data, columns=["a", "b", "c"], index=["owl", "toucan", "eagle"]
        ... )
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby("a").size()
            b   c
        a
        1    2   2
        7    1   1

        For Resampler:

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").size()
        2023-01-01    2
        2023-02-01    2
        Freq: MS, dtype: int64
        """
        result = self._grouper.size()
        dtype_backend: Literal['pyarrow', 'numpy_nullable'] | None = None
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
            result = result.convert_dtypes(infer_objects=False,
                convert_string=False, convert_boolean=False,
                convert_floating=False, dtype_backend=dtype_backend)
        if not self.as_index:
            result = result.rename('size').reset_index()
        return result

    @final
    @Substitution(name='groupby')
    @Substitution(see_also=_common_see_also)
    def mean(self, numeric_only=False, skipna=True, engine=None,
        engine_kwargs=None):
        """
        Compute mean of groups, excluding missing values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        skipna : bool, default True
            Exclude NA/null values. If an entire group is NA, the result will be NA.

            .. versionadded:: 3.0.0

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

            .. versionadded:: 1.4.0

        engine_kwargs : dict[str, bool] | None, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}``

            .. versionadded:: 1.4.0

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Mean of values within each group. Same object type as the caller.
        %(see_also)s
        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"A": "a a b".split(), "B": [np.nan, 2, 3], "C": [1, 2, 1]},
        ...     columns=["A", "B", "C"],
        ... )
        >>> g1 = df.groupby("A", group_keys=False)
        >>> g2 = df.groupby("A", group_keys=True)

        Groupby one column and return the mean of the remaining columns in
        each group.

        >>> g1["B", "C"].mean()
             B         C
        a  2.0  1.333333
        b  3.0  1.500000

        Groupby two columns and return the mean of the remaining column.

        >>> g1.groupby(["A", "B"]).mean()
              C
        A B
        a NaN  1.0
        a  2.0  2.0
        b  3.0  1.0

        Groupby one column and return the mean of only particular column in
        the group.

        >>> g1["B"].mean()
        A
        a    2.0
        b    3.0
        Name: B, dtype: float64
        """
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_mean
            return self._numba_agg_general(grouped_mean, executor.
                float_dtype_mapping, engine_kwargs, min_periods=0, skipna=
                skipna)
        else:
            result = self._cython_agg_general('mean', alt=lambda x: Series(
                x, copy=False).mean(numeric_only=numeric_only, skipna=
                skipna), numeric_only=numeric_only, skipna=skipna)
            return cast(NDFrameT, result.__finalize__(self.obj, method=
                'groupby'))

    def _wrap_agged_manager(self, mgr):
        """
        Wrap the agged manager into a Series or DataFrame.

        Parameters
        ----------
        mgr : Manager
            The manager containing the aggregated data.

        Returns
        -------
        NDFrameT
            The wrapped Series or DataFrame.
        """
        if isinstance(mgr, pd.core.internals.managers.BlockManager):
            if mgr.ndim == 1:
                return self._obj_1d_constructor(mgr)
            else:
                return self.obj._constructor(mgr, index=mgr.axes[0],
                    columns=mgr.axes[1])
        else:
            raise TypeError('Unrecognized manager type')


OutputFrameOrSeries = TypeVar('OutputFrameOrSeries', bound=NDFrame)


class GroupBy(BaseGroupBy[NDFrameT]):
    """
    Class for grouping and aggregating relational data.

    See aggregate, transform, and apply functions on this object.

    It's easiest to use obj.groupby(...) to use GroupBy, but you can also do:

    ::

        grouped = groupby(obj, ...)

    Parameters
    ----------
    obj : pandas object
    level : int, default None
        Level of MultiIndex
    groupings : list of Grouping objects
        Most users should ignore this
    exclusions : array-like, optional
        List of columns to exclude
    name : str
        Most users should ignore this

    Returns
    -------
    **Attributes**
    groups : dict
        {group name -> group labels}
    len(grouped) : int
        Number of groups

    Notes
    -----
    After grouping, see aggregate, apply, and transform functions. Here are
    some other brief notes about usage. When grouping by multiple groups, the
    result index will be a MultiIndex (hierarchical) by default.

    Iteration produces (key, group) tuples, i.e. chunking the data by group. So
    you can write code like:

    ::

        grouped = obj.groupby(keys)
        for key, group in grouped:
            # do something with the data

    Function calls on GroupBy, if not specially implemented, "dispatch" to the
    grouped data. So if you group a DataFrame and wish to invoke the std()
    method on each group, you can simply do:

    ::

        df.groupby(mapper).std()

    rather than

    ::

        df.groupby(mapper).aggregate(np.std)

    You can pass arguments to these "wrapped" functions, too.

    See the online documentation for full exposition on these topics and much
    more
    """
    _grouper: ops.BaseGrouper
    as_index: bool

    @final
    def __init__(self, obj, keys=None, level=None, grouper=None, exclusions
        =None, selection=None, as_index=True, sort=True, group_keys=True,
        observed=False, dropna=True):
        self._selection = selection
        assert isinstance(obj, NDFrame), type(obj)
        self.level = level
        self.as_index = as_index
        self.keys = keys
        self.sort = sort
        self.group_keys = group_keys
        self.dropna = dropna
        if grouper is None:
            grouper, exclusions, obj = get_grouper(obj, keys, level=level,
                sort=sort, observed=observed, dropna=self.dropna)
        self.observed = observed
        self.obj = obj
        self._grouper = grouper
        self.exclusions = frozenset(exclusions) if exclusions else frozenset()

    def __getattr__(self, attr):
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        if attr in self.obj:
            return self[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'")

    @final
    def _op_via_apply(self, name, *args: Any, **kwargs: Any):
        """Compute the result of an operation by using GroupBy's apply."""
        f = getattr(type(self._obj_with_exclusions), name)

        def curried(x):
            return f(x, *args, **kwargs)
        curried.__name__ = name
        if name in base.plotting_methods:
            return self._python_apply_general(curried, self._selected_obj)
        is_transform = name in base.transformation_kernels
        result = self._python_apply_general(curried, self.
            _obj_with_exclusions, is_transform=is_transform,
            not_indexed_same=not is_transform)
        if self._grouper.has_dropped_na and is_transform:
            result = self._set_result_index_ordered(result)
        return result

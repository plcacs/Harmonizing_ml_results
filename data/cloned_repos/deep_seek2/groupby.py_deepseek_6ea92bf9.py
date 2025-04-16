from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
import datetime
from functools import (
    partial,
    wraps,
)
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Literal,
    TypeVar,
    Union,
    cast,
    final,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    Timestamp,
    lib,
)
from pandas._libs.algos import rank_1d
import pandas._libs.groupby as libgroupby
from pandas._libs.missing import NA
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    DtypeObj,
    IndexLabel,
    IntervalClosedType,
    NDFrameT,
    PositionalIndexer,
    RandomState,
    npt,
)
from pandas.compat.numpy import function as nv
from pandas.errors import (
    AbstractMethodError,
    DataError,
)
from pandas.util._decorators import (
    Appender,
    Substitution,
    cache_readonly,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    coerce_indexer_dtype,
    ensure_dtype_can_hold_na,
)
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_float_dtype,
    is_hashable,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
)

from pandas.core import (
    algorithms,
    sample,
)
from pandas.core._numba import executor
from pandas.core.arrays import (
    ArrowExtensionArray,
    BaseMaskedArray,
    ExtensionArray,
    FloatingArray,
    IntegerArray,
    SparseArray,
)
from pandas.core.arrays.string_ import StringDtype
from pandas.core.arrays.string_arrow import (
    ArrowStringArray,
    ArrowStringArrayNumpySemantics,
)
from pandas.core.base import (
    PandasObject,
    SelectionMixin,
)
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import (
    base,
    numba_,
    ops,
)
from pandas.core.groupby.grouper import get_grouper
from pandas.core.groupby.indexing import (
    GroupByIndexingMixin,
    GroupByNthSelector,
)
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    default_index,
)
from pandas.core.internals.blocks import ensure_block_shape
from pandas.core.series import Series
from pandas.core.sorting import get_group_index_sorter
from pandas.core.util.numba_ import (
    get_jit_arguments,
    maybe_use_numba,
    prepare_function_arguments,
)

if TYPE_CHECKING:
    from pandas._libs.tslibs import BaseOffset
    from pandas._typing import (
        Any,
        Concatenate,
        P,
        Self,
        T,
    )

    from pandas.core.indexers.objects import BaseIndexer
    from pandas.core.resample import Resampler
    from pandas.core.window import (
        ExpandingGroupby,
        ExponentialMovingWindowGroupby,
        RollingGroupby,
    )

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
func : callable or tuple of (callable, str)
    Function to apply to this %(klass)s object or, alternatively,
    a `(callable, data_keyword)` tuple where `data_keyword` is a
    string indicating the keyword of `callable` that expects the
    %(klass)s object.
*args : iterable, optional
       Positional arguments passed into `func`.
**kwargs : dict, optional
         A dictionary of keyword arguments passed into `func`.

Returns
-------
%(klass)s
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
func : function, str
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
*args
    Positional arguments to pass to func.
engine : str, default None
    * ``'cython'`` : Runs the function through C-extensions from cython.
    * ``'numba'`` : Runs the function through JIT compiled code from numba.
    * ``None`` : Defaults to ``'cython'`` or the global setting ``compute.use_numba``

engine_kwargs : dict, default None
    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
      and ``parallel`` dictionary keys. The values must either be ``True`` or
      ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
      ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
      applied to the function

**kwargs
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

    def __init__(self, groupby: GroupBy) -> None:
        self._groupby = groupby

    def __call__(self, *args, **kwargs):
        def f(self):
            return self.plot(*args, **kwargs)

        f.__name__ = "plot"
        return self._groupby._python_apply_general(f, self._groupby._selected_obj)

    def __getattr__(self, name: str):
        def attr(*args, **kwargs):
            def f(self):
                return getattr(self.plot, name)(*args, **kwargs)

            return self._groupby._python_apply_general(f, self._groupby._selected_obj)

        return attr


_KeysArgType = Union[
    Hashable,
    list[Hashable],
    Callable[[Hashable], Hashable],
    list[Callable[[Hashable], Hashable]],
    Mapping[Hashable, Hashable],
]


class BaseGroupBy(PandasObject, SelectionMixin[NDFrameT], GroupByIndexingMixin):
    _hidden_attrs = PandasObject._hidden_attrs | {
        "as_index",
        "dropna",
        "exclusions",
        "grouper",
        "group_keys",
        "keys",
        "level",
        "obj",
        "observed",
        "sort",
    }

    _grouper: ops.BaseGrouper
    keys: _KeysArgType | None = None
    level: IndexLabel | None = None
    group_keys: bool

    @final
    def __len__(self) -> int:
        return self._grouper.ngroups

    @final
    def __repr__(self) -> str:
        # TODO: Better repr for GroupBy object
        return object.__repr__(self)

    @final
    @property
    def groups(self) -> dict[Hashable, Index]:
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
                "`groups` by one element list returns scalar is deprecated "
                "and will be removed. In a future version `groups` by one element "
                "list will return tuple. Use ``df.groupby(by='a').groups`` "
                "instead of ``df.groupby(by=['a']).groups`` to avoid this warning",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return self._grouper.groups

    @final
    @property
    def ngroups(self) -> int:
        return self._grouper.ngroups

    @final
    @property
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
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
            # possibly convert to the actual key types
            # in the indices, could be a Timestamp or a np.datetime64
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
            index_sample = None  # Dummy sample

        name_sample = names[0]
        if isinstance(index_sample, tuple):
            if not isinstance(name_sample, tuple):
                msg = "must supply a tuple to get_group with multiple grouping keys"
                raise ValueError(msg)
            if not len(name_sample) == len(index_sample):
                try:
                    # If the original grouper was a tuple
                    return [self.indices[name] for name in names]
                except KeyError as err:
                    # turns out it wasn't a tuple
                    msg = (
                        "must supply a same-length tuple to get_group "
                        "with multiple grouping keys"
                    )
                    raise ValueError(msg) from err

            converters = (get_converter(s) for s in index_sample)
            names = (tuple(f(n) for f, n in zip(converters, name)) for name in names)

        else:
            converter = get_converter(index_sample)
            names = (converter(name) for name in names)

        return [self.indices.get(name, []) for name in names]

    @final
    def _get_index(self, name):
        """
        Safe get index, translate keys for datelike to underlying repr.
        """
        return self._get_indices([name])[0]

    @final
    @cache_readonly
    def _selected_obj(self):
        # Note: _selected_obj is always just `self.obj` for SeriesGroupBy
        if isinstance(self.obj, Series):
            return self.obj

        if self._selection is not None:
            if is_hashable(self._selection):
                # i.e. a single key, so selecting it will return a Series.
                #  In this case, _obj_with_exclusions would wrap the key
                #  in a list and return a single-column DataFrame.
                return self.obj[self._selection]

            # Otherwise _selection is equivalent to _selection_list, so
            #  _selected_obj matches _obj_with_exclusions, so we can reuse
            #  that and avoid making a copy.
            return self._obj_with_exclusions

        return self.obj

    @final
    def _dir_additions(self) -> set[str]:
        return self.obj._dir_additions()

    @overload
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...

    @overload
    def pipe(
        self,
        func: tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...

    @Substitution(
        klass="GroupBy",
        examples=dedent(
            """\
        >>> df = pd.DataFrame({'A': 'a b a b'.split(), 'B': [1, 2, 3, 4]})
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
        ),
    )
    @Appender(_pipe_template)
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        return com.pipe(self, func, *args, **kwargs)

    @final
    def get_group(self, name) -> DataFrame | Series:
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
        # mypy doesn't recognize level/keys as being sized when passed to len
        if (is_list_like(level) and len(level) == 1) or (  # type: ignore[arg-type]
            is_list_like(keys) and len(keys) == 1  # type: ignore[arg-type]
        ):
            # GH#25971
            if isinstance(name, tuple) and len(name) == 1:
                name = name[0]
            else:
                raise KeyError(name)

        inds = self._get_index(name)
        if not len(inds):
            raise KeyError(name)
        return self._selected_obj.iloc[inds]

    @final
    def __iter__(self) -> Iterator[tuple[Hashable, NDFrameT]]:
        """
        Groupby iterator.

        This method provides an iterator over the groups created by the ``resample``
        or ``groupby`` operation on the object. The method yields tuples where
        the first element is the label (group key) corresponding to each group or
        resampled bin, and the second element is the subset of the data that falls
        within that group or bin.

        Returns
        -------
        Iterator
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
        # mypy: Argument 1 to "len" has incompatible type "Hashable"; expected "Sized"
        if (
            (is_list_like(level) and len(level) == 1)  # type: ignore[arg-type]
            or (isinstance(keys, list) and len(keys) == 1)
        ):
            # GH#42795 - when keys is a list, return tuples even when length is 1
            result = (((key,), group) for key, group in result)
        return result


# To track operations that expand dimensions, like ohlc
OutputFrameOrSeries = TypeVar("OutputFrameOrSeries", bound=NDFrame)


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
    def __init__(
        self,
        obj: NDFrameT,
        keys: _KeysArgType | None =
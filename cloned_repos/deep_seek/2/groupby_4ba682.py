from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
import datetime
from functools import partial, wraps
from textwrap import dedent
from typing import (
    TYPE_CHECKING, 
    Literal, 
    TypeVar, 
    Union, 
    cast, 
    final, 
    overload,
    Any,
    Optional,
    Dict,
    List,
    Tuple,
    Set,
    FrozenSet,
    Type,
    Generic,
    Sequence as Seq,
    Mapping as Map,
    Callable as Func,
    Iterator as Iter,
    Hashable as Hash,
    cast,
    final,
    overload
)
import warnings
import numpy as np
from pandas._libs import Timestamp, lib
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
    npt
)
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError, DataError
from pandas.util._decorators import Appender, Substitution, cache_readonly, doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import coerce_indexer_dtype, ensure_dtype_can_hold_na
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
    pandas_dtype
)
from pandas.core.dtypes.missing import isna, na_value_for_dtype, notna
from pandas.core import algorithms, sample
from pandas.core._numba import executor
from pandas.core.arrays import (
    ArrowExtensionArray, 
    BaseMaskedArray, 
    ExtensionArray, 
    FloatingArray, 
    IntegerArray, 
    SparseArray
)
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

if TYPE_CHECKING:
    from pandas._libs.tslibs import BaseOffset
    from pandas._typing import Any, Concatenate, P, Self, T
    from pandas.core.indexers.objects import BaseIndexer
    from pandas.core.resample import Resampler
    from pandas.core.window import (
        ExpandingGroupby, 
        ExponentialMovingWindowGroupby, 
        RollingGroupby
    )

_common_see_also = '\n        See Also\n        --------\n        Series.%(name)s : Apply a function %(name)s to a Series.\n        DataFrame.%(name)s : Apply a function %(name)s\n            to each row or column of a DataFrame.\n'
_groupby_agg_method_engine_template = "\nCompute {fname} of group values.\n\nParameters\n----------\nnumeric_only : bool, default {no}\n    Include only float, int, boolean columns.\n\n    .. versionchanged:: 2.0.0\n\n        numeric_only no longer accepts ``None``.\n\nmin_count : int, default {mc}\n    The required number of valid values to perform the operation. If fewer\n    than ``min_count`` non-NA values are present the result will be NA.\n\nengine : str, default None {e}\n    * ``'cython'`` : Runs rolling apply through C-extensions from cython.\n    * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.\n        Only available when ``raw`` is set to ``True``.\n    * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``\n\nengine_kwargs : dict, default None {ek}\n    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``\n    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``\n        and ``parallel`` dictionary keys. The values must either be ``True`` or\n        ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is\n        ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be\n        applied to both the ``func`` and the ``apply`` groupby aggregation.\n\nReturns\n-------\nSeries or DataFrame\n    Computed {fname} of values within each group.\n\nSee Also\n--------\nSeriesGroupBy.min : Return the min of the group values.\nDataFrameGroupBy.min : Return the min of the group values.\nSeriesGroupBy.max : Return the max of the group values.\nDataFrameGroupBy.max : Return the max of the group values.\nSeriesGroupBy.sum : Return the sum of the group values.\nDataFrameGroupBy.sum : Return the sum of the group values.\n\nExamples\n--------\n{example}\n"
_groupby_agg_method_skipna_engine_template = "\nCompute {fname} of group values.\n\nParameters\n----------\nnumeric_only : bool, default {no}\n    Include only float, int, boolean columns.\n\n    .. versionchanged:: 2.0.0\n\n        numeric_only no longer accepts ``None``.\n\nmin_count : int, default {mc}\n    The required number of valid values to perform the operation. If fewer\n    than ``min_count`` non-NA values are present the result will be NA.\n\nskipna : bool, default {s}\n    Exclude NA/null values. If the entire group is NA and ``skipna`` is\n    ``True``, the result will be NA.\n\n    .. versionchanged:: 3.0.0\n\nengine : str, default None {e}\n    * ``'cython'`` : Runs rolling apply through C-extensions from cython.\n    * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.\n        Only available when ``raw`` is set to ``True``.\n    * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``\n\nengine_kwargs : dict, default None {ek}\n    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``\n    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``\n        and ``parallel`` dictionary keys. The values must either be ``True`` or\n        ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is\n        ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be\n        applied to both the ``func`` and the ``apply`` groupby aggregation.\n\nReturns\n-------\nSeries or DataFrame\n    Computed {fname} of values within each group.\n\nSee Also\n--------\nSeriesGroupBy.min : Return the min of the group values.\nDataFrameGroupBy.min : Return the min of the group values.\nSeriesGroupBy.max : Return the max of the group values.\nDataFrameGroupBy.max : Return the max of the group values.\nSeriesGroupBy.sum : Return the sum of the group values.\nDataFrameGroupBy.sum : Return the sum of the group values.\n\nExamples\n--------\n{example}\n"
_pipe_template = '\nApply a ``func`` with arguments to this %(klass)s object and return its result.\n\nUse `.pipe` when you want to improve readability by chaining together\nfunctions that expect Series, DataFrames, GroupBy or Resampler objects.\nInstead of writing\n\n>>> h = lambda x, arg2, arg3: x + 1 - arg2 * arg3\n>>> g = lambda x, arg1: x * 5 / arg1\n>>> f = lambda x: x ** 4\n>>> df = pd.DataFrame([["a", 4], ["b", 5]], columns=["group", "value"])\n>>> h(g(f(df.groupby(\'group\')), arg1=1), arg2=2, arg3=3)  # doctest: +SKIP\n\nYou can write\n\n>>> (df.groupby(\'group\')\n...    .pipe(f)\n...    .pipe(g, arg1=1)\n...    .pipe(h, arg2=2, arg3=3))  # doctest: +SKIP\n\nwhich is much more readable.\n\nParameters\n----------\nfunc : callable or tuple of (callable, str)\n    Function to apply to this %(klass)s object or, alternatively,\n    a `(callable, data_keyword)` tuple where `data_keyword` is a\n    string indicating the keyword of `callable` that expects the\n    %(klass)s object.\n*args : iterable, optional\n       Positional arguments passed into `func`.\n**kwargs : dict, optional\n         A dictionary of keyword arguments passed into `func`.\n\nReturns\n-------\n%(klass)s\n    The original object with the function `func` applied.\n\nSee Also\n--------\nSeries.pipe : Apply a function with arguments to a series.\nDataFrame.pipe: Apply a function with arguments to a dataframe.\napply : Apply function to each group instead of to the\n    full %(klass)s object.\n\nNotes\n-----\nSee more `here\n<https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls>`_\n\nExamples\n--------\n%(examples)s\n'
_transform_template = '\nCall function producing a same-indexed %(klass)s on each group.\n\nReturns a %(klass)s having the same indexes as the original object\nfilled with the transformed values.\n\nParameters\n----------\nfunc : function, str\n    Function to apply to each group. See the Notes section below for requirements.\n\n    Accepted inputs are:\n\n    - String\n    - Python function\n    - Numba JIT function with ``engine=\'numba\'`` specified.\n\n    Only passing a single function is supported with this engine.\n    If the ``\'numba\'`` engine is chosen, the function must be\n    a user defined function with ``values`` and ``index`` as the\n    first and second arguments respectively in the function signature.\n    Each group\'s index will be passed to the user defined function\n    and optionally available for use.\n\n    If a string is chosen, then it needs to be the name\n    of the groupby method you want to use.\n*args\n    Positional arguments to pass to func.\nengine : str, default None\n    * ``\'cython\'`` : Runs the function through C-extensions from cython.\n    * ``\'numba\'`` : Runs the function through JIT compiled code from numba.\n    * ``None`` : Defaults to ``\'cython\'`` or the global setting ``compute.use_numba``\n\nengine_kwargs : dict, default None\n    * For ``\'cython\'`` engine, there are no accepted ``engine_kwargs``\n    * For ``\'numba\'`` engine, the engine can accept ``nopython``, ``nogil``\n      and ``parallel`` dictionary keys. The values must either be ``True`` or\n      ``False``. The default ``engine_kwargs`` for the ``\'numba\'`` engine is\n      ``{\'nopython\': True, \'nogil\': False, \'parallel\': False}`` and will be\n      applied to the function\n\n**kwargs\n    Keyword arguments to be passed into func.\n\nReturns\n-------\n%(klass)s\n    %(klass)s with the same indexes as the original object filled\n    with transformed values.\n\nSee Also\n--------\n%(klass)s.groupby.apply : Apply function ``func`` group-wise and combine\n    the results together.\n%(klass)s.groupby.aggregate : Aggregate using one or more operations.\n%(klass)s.transform : Call ``func`` on self producing a %(klass)s with the\n    same axis shape as self.\n\nNotes\n-----\nEach group is endowed the attribute \'name\' in case you need to know\nwhich group you are working on.\n\nThe current implementation imposes three requirements on f:\n\n* f must return a value that either has the same shape as the input\n  subframe or can be broadcast to the shape of the input subframe.\n  For example, if `f` returns a scalar it will be broadcast to have the\n  same shape as the input subframe.\n* if this is a DataFrame, f must support application column-by-column\n  in the subframe. If f also supports application to the entire subframe,\n  then a fast path is used starting from the second chunk.\n* f must not mutate groups. Mutation is not supported and may\n  produce unexpected results. See :ref:`gotchas.udf-mutation` for more details.\n\nWhen using ``engine=\'numba\'``, there will be no "fall back" behavior internally.\nThe group data and group index will be passed as numpy arrays to the JITed\nuser defined function, and no alternative execution attempts will be tried.\n\n.. versionchanged:: 1.3.0\n\n    The resulting dtype will reflect the return value of the passed ``func``,\n    see the examples below.\n\n.. versionchanged:: 2.0.0\n\n    When using ``.transform`` on a grouped DataFrame and the transformation function\n    returns a DataFrame, pandas now aligns the result\'s index\n    with the input\'s index. You can call ``.to_numpy()`` on the\n    result of the transformation function to avoid alignment.\n\nExamples\n--------\n%(example)s'

@final
class GroupByPlot(PandasObject):
    """
    Class implementing the .plot attribute for groupby objects.
    """

    def __init__(self, groupby: GroupBy) -> None:
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

_KeysArgType = Union[Hashable, List[Hashable], Callable[[Hashable], Hashable], List[Callable[[Hashable], Hashable]], Mapping[Hashable, Hashable]]

class BaseGroupBy(PandasObject, SelectionMixin[NDFrameT], GroupByIndexingMixin):
    _hidden_attrs: Set[str] = PandasObject._hidden_attrs | {'as_index', 'dropna', 'exclusions', 'grouper', 'group_keys', 'keys', 'level', 'obj', 'observed', 'sort'}
    keys: Optional[Union[Hashable, List[Hashable]]] = None
    level: Optional[Union[int, str, List[Union[int, str]]]] = None

    @final
    def __len__(self) -> int:
        return self._grouper.ngroups

    @final
    def __repr__(self) -> str:
        return object.__repr__(self)

    @final
    @property
    def groups(self) -> Dict[Hashable, Any]:
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
    def indices(self) -> Dict[Hashable, Any]:
        """
        Dict {group name -> group indices}.
        """
        return self._grouper.indices

    @final
    def _get_indices(self, names: Any) -> List[Any]:
        """
        Safe get multiple indices, translate keys for datelike to underlying repr.
        """
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
        if
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
    Any,
    Concatenate,
    Literal,
    TypeVar,
    Union,
    cast,
    final,
    overload,
)
import warnings

import numpy as np
import numpy.typing as npt

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

if TYPE_CHECKING:
    from pandas._libs.tslibs import BaseOffset
    from pandas._typing import (
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

_KeysArgType = Union[
    Hashable,
    list[Hashable],
    Callable[[Hashable], Hashable],
    list[Callable[[Hashable], Hashable]],
    Mapping[Hashable, Hashable],
]

OutputFrameOrSeries = TypeVar("OutputFrameOrSeries", bound=NDFrame)

@final
class GroupByPlot(PandasObject):
    def __init__(self, groupby: GroupBy) -> None:
        self._groupby = groupby

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        def f(self: Any) -> Any:
            return self.plot(*args, **kwargs)
        f.__name__ = "plot"
        return self._groupby._python_apply_general(f, self._groupby._selected_obj)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        def attr(*args: Any, **kwargs: Any) -> Any:
            def f(self: Any) -> Any:
                return getattr(self.plot, name)(*args, **kwargs)
            return self._groupby._python_apply_general(f, self._groupby._selected_obj)
        return attr

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
        return object.__repr__(self)

    @final
    @property
    def groups(self) -> dict[Hashable, Index]:
        return self._grouper.groups

    @final
    @property
    def ngroups(self) -> int:
        return self._grouper.ngroups

    @final
    @property
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        return self._grouper.indices

    @final
    def _get_indices(self, names: list[Hashable]) -> list[npt.NDArray[np.intp]]:
        pass

    @final
    def _get_index(self, name: Hashable) -> npt.NDArray[np.intp]:
        pass

    @final
    @cache_readonly
    def _selected_obj(self) -> NDFrameT:
        pass

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
    def get_group(self, name: Hashable) -> DataFrame | Series:
        pass

    @final
    def __iter__(self) -> Iterator[tuple[Hashable, NDFrameT]]:
        pass

class GroupBy(BaseGroupBy[NDFrameT]):
    _grouper: ops.BaseGrouper
    as_index: bool

    @final
    def __init__(
        self,
        obj: NDFrameT,
        keys: _KeysArgType | None = None,
        level: IndexLabel | None = None,
        grouper: ops.BaseGrouper | None = None,
        exclusions: frozenset[Hashable] | None = None,
        selection: IndexLabel | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True,
    ) -> None:
        self._selection = selection
        self.level = level
        self.as_index = as_index
        self.keys = keys
        self.sort = sort
        self.group_keys = group_keys
        self.dropna = dropna
        self.observed = observed
        self.obj = obj
        self._grouper = grouper or get_grouper(
            obj,
            keys,
            level=level,
            sort=sort,
            observed=observed,
            dropna=self.dropna,
        )[0]
        self.exclusions = frozenset(exclusions) if exclusions else frozenset()

    def __getattr__(self, attr: str) -> Any:
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        if attr in self.obj:
            return self[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    @final
    def _op_via_apply(self, name: str, *args: Any, **kwargs: Any) -> NDFrameT:
        pass

    @final
    def _concat_objects(
        self,
        values: list[NDFrameT],
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ) -> NDFrameT:
        pass

    @final
    def _set_result_index_ordered(
        self, result: OutputFrameOrSeries
    ) -> OutputFrameOrSeries:
        pass

    @final
    def _insert_inaxis_grouper(
        self, result: Series | DataFrame, qs: npt.NDArray[np.float64] | None = None
    ) -> DataFrame:
        pass

    @final
    def _wrap_aggregated_output(
        self,
        result: Series | DataFrame,
        qs: npt.NDArray[np.float64] | None = None,
    ) -> NDFrameT:
        pass

    @final
    def _wrap_applied_output(
        self,
        data: NDFrameT,
        values: list[NDFrameT],
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ) -> NDFrameT:
        raise AbstractMethodError(self)

    @final
    def _numba_prep(self, data: DataFrame) -> tuple[
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
        npt.NDArray[Any],
        npt.NDArray[Any],
    ]:
        pass

    def _numba_agg_general(
        self,
        func: Callable,
        dtype_mapping: dict[np.dtype, Any],
        engine_kwargs: dict[str, bool] | None,
        **aggregator_kwargs: Any,
    ) -> NDFrameT:
        pass

    @final
    def _transform_with_numba(
        self,
        func: Callable,
        *args: Any,
        engine_kwargs: dict[str, bool] | None = None,
        **kwargs: Any,
    ) -> NDFrameT:
        pass

    @final
    def _aggregate_with_numba(
        self,
        func: Callable,
        *args: Any,
        engine_kwargs: dict[str, bool] | None = None,
        **kwargs: Any,
    ) -> NDFrameT:
        pass

    def apply(self, func: Callable, *args: Any, include_groups: bool = False, **kwargs: Any) -> NDFrameT:
        pass

    @final
    def _python_apply_general(
        self,
        f: Callable,
        data: DataFrame | Series,
        not_indexed_same: bool | None = None,
        is_transform: bool = False,
        is_agg: bool = False,
    ) -> NDFrameT:
        pass

    @final
    def _agg_general(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        *,
        alias: str,
        npfunc: Callable | None = None,
        **kwargs: Any,
    ) -> NDFrameT:
        pass

    def _agg_py_fallback(
        self, how: str, values: ArrayLike, ndim: int, alt: Callable
    ) -> ArrayLike:
        pass

    @final
    def _cython_agg_general(
        self,
        how: str,
        alt: Callable | None = None,
        numeric_only: bool = False,
        min_count: int = -1,
        **kwargs: Any,
    ) -> NDFrameT:
        pass

    def _cython_transform(self, how: str, numeric_only: bool = False, **kwargs: Any) -> NDFrameT:
        raise AbstractMethodError(self)

    @final
    def _transform(self, func: Callable, *args: Any, engine: str | None = None, engine_kwargs: dict[str, bool] | None = None, **kwargs: Any) -> NDFrameT:
        pass

    @final
    def _reduction_kernel_transform(
        self, func: Callable, *args: Any, engine: str | None = None, engine_kwargs: dict[str, bool] | None = None, **kwargs: Any
    ) -> NDFrameT:
        pass

    @final
    def _wrap_transform_fast_result(self, result: NDFrameT) -> NDFrameT:
        pass

    @final
    def _apply_filter(self, indices: list[npt.NDArray[np.intp]], dropna: bool) -> NDFrameT:
        pass

    @final
    def _cumcount_array(self, ascending: bool = True) -> npt.NDArray[np.intp]:
        pass

    @final
    @property
    def _obj_1d_constructor(self) -> Callable[..., Series]:
        pass

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def any(self, skipna: bool = True) -> NDFrameT:
        pass

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def all(self, skipna: bool = True) -> NDFrameT:
        pass

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def count(self) -> NDFrameT:
        pass

    @final
    def mean(
        self,
        numeric_only: bool = False,
        skipna: bool = True,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ) -> NDFrameT:
        pass

    @final
    def median(self, numeric_only: bool = False, skipna: bool = True) -> NDFrameT:
        pass

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def std(
        self,
        ddof: int = 1,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        numeric_only: bool = False,
        skipna: bool = True,
    ) -> NDFrameT:
        pass

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def var(
        self,
        ddof: int = 1,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        numeric_only: bool = False,
        skipna: bool = True,
    ) -> NDFrameT:
        pass

    @final
    def _value_counts(
        self,
        subset: Sequence[Hashable] | None = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> DataFrame | Series:
        pass

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def size(self) -> DataFrame | Series:
        pass

    @final
    @doc(
        _groupby_agg_method_skipna_engine_template,
        fname="sum",
        no=False,
        mc=0,
        s=True,
        e=None,
        ek=None,
        example=dedent(
            """\
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).sum()
        a    3
        b    7
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger
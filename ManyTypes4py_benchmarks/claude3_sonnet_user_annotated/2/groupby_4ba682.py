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
    P,
    Self,
    T,
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

_common_see_also: str
_groupby_agg_method_engine_template: str
_groupby_agg_method_skipna_engine_template: str
_pipe_template: str
_transform_template: str

OutputFrameOrSeries = TypeVar("OutputFrameOrSeries", bound=NDFrame)
_KeysArgType = Union[
    Hashable,
    list[Hashable],
    Callable[[Hashable], Hashable],
    list[Callable[[Hashable], Hashable]],
    Mapping[Hashable, Hashable],
]

@final
class GroupByPlot(PandasObject):
    def __init__(self, groupby: GroupBy) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...

class BaseGroupBy(PandasObject, SelectionMixin[NDFrameT], GroupByIndexingMixin):
    _hidden_attrs: set[str]
    _grouper: ops.BaseGrouper
    keys: _KeysArgType | None = None
    level: IndexLabel | None = None
    group_keys: bool

    @final
    def __len__(self) -> int: ...

    @final
    def __repr__(self) -> str: ...

    @final
    @property
    def groups(self) -> dict[Hashable, Index]: ...

    @final
    @property
    def ngroups(self) -> int: ...

    @final
    @property
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]: ...

    @final
    def _get_indices(self, names: list[Any]) -> list[Any]: ...

    @final
    def _get_index(self, name: Any) -> Any: ...

    @final
    @cache_readonly
    def _selected_obj(self) -> NDFrameT: ...

    @final
    def _dir_additions(self) -> set[str]: ...

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

    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...

    @final
    def get_group(self, name: Any) -> DataFrame | Series: ...

    @final
    def __iter__(self) -> Iterator[tuple[Hashable, NDFrameT]]: ...

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
    ) -> None: ...

    def __getattr__(self, attr: str) -> Any: ...

    @final
    def _op_via_apply(self, name: str, *args: Any, **kwargs: Any) -> Any: ...

    @final
    def _concat_objects(
        self,
        values: list[Any],
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ) -> NDFrameT: ...

    @final
    def _set_result_index_ordered(
        self, result: OutputFrameOrSeries
    ) -> OutputFrameOrSeries: ...

    @final
    def _insert_inaxis_grouper(
        self, result: Series | DataFrame, qs: npt.NDArray[np.float64] | None = None
    ) -> DataFrame: ...

    @final
    def _wrap_aggregated_output(
        self,
        result: Series | DataFrame,
        qs: npt.NDArray[np.float64] | None = None,
    ) -> Series | DataFrame: ...

    def _wrap_applied_output(
        self,
        data: Any,
        values: list[Any],
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ) -> Any: ...

    @final
    def _numba_prep(self, data: DataFrame) -> tuple[Any, Any, Any, Any]: ...

    def _numba_agg_general(
        self,
        func: Callable[..., Any],
        dtype_mapping: dict[np.dtype, Any],
        engine_kwargs: dict[str, bool] | None,
        **aggregator_kwargs: Any,
    ) -> Any: ...

    @final
    def _transform_with_numba(
        self, func: Callable[..., Any], *args: Any, engine_kwargs: Any = None, **kwargs: Any
    ) -> Any: ...

    @final
    def _aggregate_with_numba(
        self, func: Callable[..., Any], *args: Any, engine_kwargs: Any = None, **kwargs: Any
    ) -> Any: ...

    def apply(
        self, func: Callable[..., Any] | str, *args: Any, include_groups: bool = False, **kwargs: Any
    ) -> NDFrameT: ...

    @final
    def _python_apply_general(
        self,
        f: Callable[..., Any],
        data: DataFrame | Series,
        not_indexed_same: bool | None = None,
        is_transform: bool = False,
        is_agg: bool = False,
    ) -> NDFrameT: ...

    @final
    def _agg_general(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        *,
        alias: str,
        npfunc: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> Any: ...

    def _agg_py_fallback(
        self, how: str, values: ArrayLike, ndim: int, alt: Callable[..., Any]
    ) -> ArrayLike: ...

    @final
    def _cython_agg_general(
        self,
        how: str,
        alt: Callable[..., Any] | None = None,
        numeric_only: bool = False,
        min_count: int = -1,
        **kwargs: Any,
    ) -> Any: ...

    def _cython_transform(self, how: str, numeric_only: bool = False, **kwargs: Any) -> Any: ...

    @final
    def _transform(
        self, func: Callable[..., Any] | str, *args: Any, engine: Any = None, engine_kwargs: Any = None, **kwargs: Any
    ) -> Any: ...

    @final
    def _reduction_kernel_transform(
        self, func: str, *args: Any, engine: Any = None, engine_kwargs: Any = None, **kwargs: Any
    ) -> Any: ...

    @final
    def _wrap_transform_fast_result(self, result: NDFrameT) -> NDFrameT: ...

    @final
    def _apply_filter(self, indices: list[Any], dropna: bool) -> Any: ...

    @final
    def _cumcount_array(self, ascending: bool = True) -> np.ndarray: ...

    @final
    @property
    def _obj_1d_constructor(self) -> Callable[..., Any]: ...

    @final
    def any(self, skipna: bool = True) -> NDFrameT: ...

    @final
    def all(self, skipna: bool = True) -> NDFrameT: ...

    @final
    def count(self) -> NDFrameT: ...

    @final
    def mean(
        self,
        numeric_only: bool = False,
        skipna: bool = True,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ) -> Any: ...

    @final
    def median(self, numeric_only: bool = False, skipna: bool = True) -> NDFrameT: ...

    @final
    def std(
        self,
        ddof: int = 1,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        numeric_only: bool = False,
        skipna: bool = True,
    ) -> Any: ...

    @final
    def var(
        self,
        ddof: int = 1,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        numeric_only: bool = False,
        skipna: bool = True,
    ) -> Any: ...

    @final
    def _value_counts(
        self,
        subset: Sequence[Hashable] | None = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> DataFrame | Series: ...

    @final
    def sem(
        self, ddof: int = 1, numeric_only: bool = False, skipna: bool = True
    ) -> NDFrameT: ...

    @final
    def size(self) -> DataFrame | Series: ...

    @final
    def sum(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        skipna: bool = True,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ) -> Any: ...

    @final
    def prod(
        self, numeric_only: bool = False, min_count: int = 0, skipna: bool = True
    ) -> NDFrameT: ...

    @final
    def min(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        skipna: bool = True,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ) -> Any: ...

    @final
    def max(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        skipna: bool = True,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ) -> Any: ...

    @final
    def first(
        self, numeric_only: bool = False, min_count: int = -1, skipna: bool = True
    ) -> NDFrameT: ...

    @final
    def last(
        self, numeric_only: bool = False, min_count: int = -1, skipna: bool = True
    ) -> NDFrameT: ...

    @final
    def ohlc(self) -> DataFrame: ...

    def describe(
        self,
        percentiles: list[float] | None = None,
        include: list[Any] | None = None,
        exclude: list[Any] | None = None,
    ) -> NDFrameT: ...

    @final
    def resample(
        self, rule: str | BaseOffset, *args: Any, include_groups: bool = False, **kwargs: Any
    ) -> Resampler: ...

    @final
    def rolling(
        self,
        window: int | datetime.timedelta | str | BaseOffset | BaseIndexer,
        min_periods: int | None = None,
        center: bool = False,
        win_type: str | None = None,
        on: str | None = None,
        closed: IntervalClosedType | None = None,
        method: str = "single",
    ) -> RollingGroupby: ...

    @final
    def expanding(self, *args: Any, **kwargs: Any) -> ExpandingGroupby: ...

    @final
    def ewm(self, *args: Any, **kwargs: Any) -> ExponentialMovingWindowGroupby: ...

    @final
    def _fill(self, direction: Literal["ffill", "bfill"], limit: int | None = None) -> Any: ...

    @final
    def ffill(self, limit: int | None = None) -> Any: ...

    @final
    def bfill(self, limit: int | None = None) -> Any: ...

    @final
    @property
    def nth(self) -> GroupByNthSelector: ...

    def _nth(
        self,
        n: PositionalIndexer | tuple[Any, ...],
        dropna: Literal["any", "all", None] = None,
    ) -> NDFrameT: ...

    @final
    def quantile(
        self,
        q: float | AnyArrayLike = 0.5,
        interpolation: Literal[
            "linear", "lower", "higher", "nearest", "midpoint"
        ] = "linear",
        numeric_only: bool = False,
    ) -> Any: ...

    @final
    def ngroup(self, ascending: bool = True) -> Series: ...

    @final
    def cumcount(self, ascending: bool = True) -> Series: ...

    @final
    def rank(
        self,
        method: str = "average",
        ascending: bool = True,
        na_option: str = "keep",
        pct: bool = False,
    ) -> NDFrameT: ...

    @final
    def cumprod(self, numeric_only: bool = False, *args: Any, **kwargs: Any) -> NDFrameT: ...

    @final
    def cumsum(self, numeric_only: bool = False, *args: Any, **kwargs: Any) -> NDFrameT: ...

    @final
    def cummin(
        self,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> NDFrameT: ...

    @final
    def cummax(
        self,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> NDFrameT: ...

    @final
    def shift(
        self,
        periods: int | Sequence[int] = 1,
        freq: Any = None,
        fill_value: Any = lib.no_default,
        suffix: str | None = None,
    ) -> Any: ...

    @final
    def diff(
        self,
        periods: int = 1,
    ) -> NDFrameT: ...

    @final
    def pct_change(
        self,
        periods: int = 1,
        fill_method: None = None,
        freq: Any = None,
    ) -> Any: ...

    @final
    def head(self, n: int = 5) -> NDFrameT: ...

    @final
    def tail(self, n: int = 5) -> NDFrameT: ...

    @final
    def _mask_selected_obj(self, mask: npt.NDArray[np.bool_]) -> NDFrameT: ...

    @final
    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: bool = False,
        weights: Sequence[Any] | Series | None = None,
        random_state: RandomState | None = None,
    ) -> Any: ...

    def _idxmax_idxmin(
        self,
        how: Literal["idxmax", "idxmin"],
        ignore_unobserved: bool = False,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> NDFrameT: ...

    def _wrap_idxmax_idxmin(self, res: NDFrameT) -> NDFrameT: ...

def get_groupby(
    obj: NDFrame,
    by: _KeysArgType | None = None,
    grouper: ops.BaseGrouper | None = None,
    group_keys: bool = True,
) -> GroupBy: ...

def _insert_quantile_level(idx: Index, qs: npt.NDArray[np.float64]) -> MultiIndex: ...

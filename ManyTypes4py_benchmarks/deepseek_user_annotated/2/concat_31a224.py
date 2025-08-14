"""
Concat routines.
"""

from __future__ import annotations

from collections import abc
import types
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas.util._decorators import set_module
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_bool,
    is_scalar,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

from pandas.core.arrays.categorical import (
    factorize_from_iterable,
    factorize_from_iterables,
)
import pandas.core.common as com
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    all_indexes_same,
    default_index,
    ensure_index,
    get_objs_combined_axis,
    get_unanimous_names,
)
from pandas.core.internals import concatenate_managers

if TYPE_CHECKING:
    from collections.abc import (
        Callable as CallableT,
        Hashable as HashableT,
        Iterable as IterableT,
        Mapping as MappingT,
    )

    from pandas._typing import (
        Axis,
        AxisInt,
        Hashable as HashableT,
    )

    from pandas import (
        DataFrame,
        Series,
    )

T = TypeVar('T')
HashableT = TypeVar('HashableT', bound=Hashable)

# ---------------------------------------------------------------------
# Concatenate DataFrame objects


@overload
def concat(
    objs: Iterable[DataFrame] | Mapping[HashableT, DataFrame],
    *,
    axis: Literal[0, "index"] = ...,
    join: str = ...,
    ignore_index: bool = ...,
    keys: Iterable[Hashable] | None = ...,
    levels: Optional[List[Sequence[Hashable]]] = ...,
    names: List[HashableT] | None = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool | lib.NoDefault = ...,
) -> DataFrame: ...


@overload
def concat(
    objs: Iterable[Series] | Mapping[HashableT, Series],
    *,
    axis: Literal[0, "index"] = ...,
    join: str = ...,
    ignore_index: bool = ...,
    keys: Iterable[Hashable] | None = ...,
    levels: Optional[List[Sequence[Hashable]]] = ...,
    names: List[HashableT] | None = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool | lib.NoDefault = ...,
) -> Series: ...


@overload
def concat(
    objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
    *,
    axis: Literal[0, "index"] = ...,
    join: str = ...,
    ignore_index: bool = ...,
    keys: Iterable[Hashable] | None = ...,
    levels: Optional[List[Sequence[Hashable]]] = ...,
    names: List[HashableT] | None = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool | lib.NoDefault = ...,
) -> DataFrame | Series: ...


@overload
def concat(
    objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
    *,
    axis: Literal[1, "columns"],
    join: str = ...,
    ignore_index: bool = ...,
    keys: Iterable[Hashable] | None = ...,
    levels: Optional[List[Sequence[Hashable]]] = ...,
    names: List[HashableT] | None = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool | lib.NoDefault = ...,
) -> DataFrame: ...


@overload
def concat(
    objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
    *,
    axis: Axis = ...,
    join: str = ...,
    ignore_index: bool = ...,
    keys: Iterable[Hashable] | None = ...,
    levels: Optional[List[Sequence[Hashable]]] = ...,
    names: List[HashableT] | None = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool | lib.NoDefault = ...,
) -> DataFrame | Series: ...


@set_module("pandas")
def concat(
    objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
    *,
    axis: Axis = 0,
    join: str = "outer",
    ignore_index: bool = False,
    keys: Iterable[Hashable] | None = None,
    levels: Optional[List[Sequence[Hashable]]] = None,
    names: List[HashableT] | None = None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: bool | lib.NoDefault = lib.no_default,
) -> DataFrame | Series:
    # Implementation remains the same
    pass


def _sanitize_mixed_ndim(
    objs: List[Series | DataFrame],
    sample: Series | DataFrame,
    ignore_index: bool,
    axis: AxisInt,
) -> List[Series | DataFrame]:
    # Implementation remains the same
    pass


def _get_result(
    objs: List[Series | DataFrame],
    is_series: bool,
    bm_axis: AxisInt,
    ignore_index: bool,
    intersect: bool,
    sort: bool,
    keys: Iterable[Hashable] | None,
    levels: Optional[List[Sequence[Hashable]]],
    verify_integrity: bool,
    names: List[HashableT] | None,
    axis: AxisInt,
) -> DataFrame | Series:
    # Implementation remains the same
    pass


def new_axes(
    objs: List[Series | DataFrame],
    bm_axis: AxisInt,
    intersect: bool,
    sort: bool,
    keys: Iterable[Hashable] | None,
    names: List[HashableT] | None,
    axis: AxisInt,
    levels: Optional[List[Sequence[Hashable]]],
    verify_integrity: bool,
    ignore_index: bool,
) -> List[Index]:
    # Implementation remains the same
    pass


def _get_concat_axis_series(
    objs: List[Series | DataFrame],
    ignore_index: bool,
    bm_axis: AxisInt,
    keys: Iterable[Hashable] | None,
    levels: Optional[List[Sequence[Hashable]]],
    verify_integrity: bool,
    names: List[HashableT] | None,
) -> Index:
    # Implementation remains the same
    pass


def _get_concat_axis_dataframe(
    objs: List[Series | DataFrame],
    axis: AxisInt,
    ignore_index: bool,
    keys: Iterable[Hashable] | None,
    names: List[HashableT] | None,
    levels: Optional[List[Sequence[Hashable]]],
    verify_integrity: bool,
) -> Index:
    # Implementation remains the same
    pass


def _clean_keys_and_objs(
    objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
    keys: Optional[Index],
) -> Tuple[List[Series | DataFrame], Optional[Index], Set[int]]:
    # Implementation remains the same
    pass


def _get_sample_object(
    objs: List[Series | DataFrame],
    ndims: Set[int],
    keys: Optional[Index],
    names: List[HashableT] | None,
    levels: Optional[List[Sequence[Hashable]]],
    intersect: bool,
) -> Tuple[Series | DataFrame, List[Series | DataFrame]]:
    # Implementation remains the same
    pass


def _concat_indexes(indexes: List[Index]) -> Index:
    # Implementation remains the same
    pass


def validate_unique_levels(levels: List[Index]) -> None:
    # Implementation remains the same
    pass


def _make_concat_multiindex(
    indexes: List[Index],
    keys: Iterable[Hashable],
    levels: Optional[List[Sequence[Hashable]]] = None,
    names: List[HashableT] | None = None,
) -> MultiIndex:
    # Implementation remains the same
    pass

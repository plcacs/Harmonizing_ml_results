from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Hashable, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pandas import MultiIndex, Series
from pandas._typing import IndexKeyFunc, Level, NaPosition, SortKind
from pandas.core.arrays import ExtensionArray
from pandas.core.indexes.base import Index


def get_indexer_indexer(
    target: Index,
    level: Level | Sequence[Level] | None,
    ascending: bool | Sequence[bool],
    kind: SortKind,
    na_position: NaPosition,
    sort_remaining: bool,
    key: Optional[IndexKeyFunc],
) -> Optional[NDArray[np.intp]]: ...
def get_group_index(
    labels: Sequence[NDArray[np.intp]],
    shape: tuple[int, ...],
    sort: bool,
    xnull: bool,
) -> NDArray[np.int64]: ...
def get_compressed_ids(
    labels: Sequence[NDArray[np.intp]],
    sizes: tuple[int, ...],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...
def is_int64_overflow_possible(shape: tuple[int, ...]) -> bool: ...
def _decons_group_index(
    comp_labels: NDArray[np.int64],
    shape: tuple[int, ...],
) -> list[NDArray[np.int64]]: ...
def decons_obs_group_ids(
    comp_ids: NDArray[np.intp],
    obs_ids: NDArray[np.int64],
    shape: tuple[int, ...],
    labels: Sequence[NDArray[np.intp]],
    xnull: bool,
) -> list[NDArray[np.intp]]: ...
def lexsort_indexer(
    keys: Sequence[Series | Index | NDArray[Any]],
    orders: bool | Sequence[bool] | None = ...,
    na_position: NaPosition = ...,
    key: Optional[IndexKeyFunc] = ...,
    codes_given: bool = ...,
) -> NDArray[np.intp]: ...
def nargsort(
    items: NDArray[Any] | ExtensionArray | Index | Series,
    kind: SortKind = ...,
    ascending: bool = ...,
    na_position: NaPosition = ...,
    key: Optional[IndexKeyFunc] = ...,
    mask: Optional[NDArray[np.bool_]] = ...,
) -> NDArray[np.intp]: ...
def nargminmax(
    values: ExtensionArray,
    method: Literal["argmax", "argmin"],
    axis: int = ...,
) -> int | NDArray[np.intp]: ...
def _nanargminmax(
    values: NDArray[Any],
    mask: NDArray[np.bool_],
    func: Callable[[NDArray[Any]], int],
) -> int: ...
def _ensure_key_mapped_multiindex(
    index: MultiIndex,
    key: IndexKeyFunc,
    level: Sequence[int | str] | int | str | None = ...,
) -> MultiIndex: ...
def ensure_key_mapped(
    values: object,
    key: Optional[IndexKeyFunc],
    levels: Optional[Sequence[Level]] = ...,
) -> object: ...
def get_indexer_dict(
    label_list: Sequence[NDArray[np.intp]],
    keys: Sequence[Sequence[Hashable]],
) -> dict[Hashable, NDArray[np.intp]]: ...
def get_group_index_sorter(
    group_index: NDArray[np.intp],
    ngroups: Optional[int] = ...,
) -> NDArray[np.intp]: ...
def compress_group_index(
    group_index: NDArray[np.intp],
    sort: bool = ...,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...
def _reorder_by_uniques(
    uniques: NDArray[np.int64],
    labels: NDArray[np.intp],
) -> tuple[NDArray[np.int64], NDArray[np.intp]]: ...
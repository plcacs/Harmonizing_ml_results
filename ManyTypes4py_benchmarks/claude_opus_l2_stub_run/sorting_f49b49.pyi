from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pandas._libs.hashtable import unique_label_indices as unique_label_indices

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        IndexKeyFunc,
        Level,
        NaPosition,
        Shape,
        SortKind,
    )
    from pandas import MultiIndex, Series
    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.base import Index


def get_indexer_indexer(
    target: Index,
    level: Level | list[Level] | None,
    ascending: bool | list[bool],
    kind: SortKind,
    na_position: NaPosition,
    sort_remaining: bool,
    key: IndexKeyFunc,
) -> npt.NDArray[np.intp] | None: ...


def get_group_index(
    labels: list[np.ndarray],
    shape: Shape,
    sort: bool,
    xnull: bool,
) -> np.ndarray: ...


def get_compressed_ids(
    labels: list[np.ndarray],
    sizes: tuple[int, ...],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...


def is_int64_overflow_possible(shape: Shape) -> bool: ...


def _decons_group_index(
    comp_labels: np.ndarray,
    shape: Shape,
) -> list[np.ndarray]: ...


def decons_obs_group_ids(
    comp_ids: npt.NDArray[np.intp],
    obs_ids: npt.NDArray[np.intp],
    shape: tuple[int, ...],
    labels: Sequence[np.ndarray],
    xnull: bool,
) -> list[np.ndarray]: ...


def lexsort_indexer(
    keys: Sequence[ArrayLike | Index | Series],
    orders: bool | list[bool] | None = ...,
    na_position: NaPosition = ...,
    key: Callable | None = ...,
    codes_given: bool = ...,
) -> npt.NDArray[np.intp]: ...


def nargsort(
    items: np.ndarray | ExtensionArray | Index | Series,
    kind: SortKind = ...,
    ascending: bool = ...,
    na_position: NaPosition = ...,
    key: Callable | None = ...,
    mask: np.ndarray | None = ...,
) -> npt.NDArray[np.intp]: ...


def nargminmax(
    values: ExtensionArray,
    method: str,
    axis: AxisInt = ...,
) -> int | np.ndarray: ...


def _nanargminmax(
    values: np.ndarray,
    mask: np.ndarray,
    func: Callable,
) -> np.intp: ...


def _ensure_key_mapped_multiindex(
    index: MultiIndex,
    key: Callable,
    level: int | str | list[int] | list[str] | None = ...,
) -> MultiIndex: ...


def ensure_key_mapped(
    values: Series | Index | np.ndarray,
    key: Callable | None,
    levels: Level | list[Level] | None = ...,
) -> Series | Index | np.ndarray: ...


def get_indexer_dict(
    label_list: list[np.ndarray],
    keys: list[Index],
) -> dict[Hashable, npt.NDArray[np.intp]]: ...


def get_group_index_sorter(
    group_index: npt.NDArray[np.intp],
    ngroups: int | None = ...,
) -> npt.NDArray[np.intp]: ...


def compress_group_index(
    group_index: np.ndarray,
    sort: bool = ...,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...


def _reorder_by_uniques(
    uniques: npt.NDArray[np.int64],
    labels: npt.NDArray[np.intp],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.intp]]: ...
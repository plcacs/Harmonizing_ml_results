```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence
    from pandas._typing import ArrayLike, AxisInt, IndexKeyFunc, Level, NaPosition, Shape, SortKind, npt
    from pandas import MultiIndex, Series
    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.base import Index

def get_indexer_indexer(
    target: Any,
    level: Any,
    ascending: Any,
    kind: Any,
    na_position: Any,
    sort_remaining: Any,
    key: Any,
) -> Any: ...
def get_group_index(
    labels: Sequence[Any],
    shape: tuple[int, ...],
    sort: bool,
    xnull: bool,
) -> np.ndarray[Any, np.dtype[np.int64]]: ...
def get_compressed_ids(
    labels: list[Any],
    sizes: tuple[int, ...],
) -> tuple[np.ndarray[Any, np.dtype[np.intp]], np.ndarray[Any, np.dtype[np.int64]]]: ...
def is_int64_overflow_possible(shape: tuple[int, ...]) -> bool: ...
def _decons_group_index(
    comp_labels: np.ndarray[Any, np.dtype[np.intp]],
    shape: tuple[int, ...],
) -> list[np.ndarray[Any, np.dtype[np.intp]]]: ...
def decons_obs_group_ids(
    comp_ids: np.ndarray[Any, np.dtype[np.intp]],
    obs_ids: np.ndarray[Any, np.dtype[np.intp]],
    shape: tuple[int, ...],
    labels: Sequence[np.ndarray[Any, Any]],
    xnull: bool,
) -> list[np.ndarray[Any, np.dtype[np.intp]]]: ...
def lexsort_indexer(
    keys: Sequence[Any],
    orders: bool | list[bool] | None = ...,
    na_position: NaPosition = ...,
    key: Callable[[Any], Any] | None = ...,
    codes_given: bool = ...,
) -> np.ndarray[Any, np.dtype[np.intp]]: ...
def nargsort(
    items: Any,
    kind: SortKind = ...,
    ascending: bool = ...,
    na_position: NaPosition = ...,
    key: Callable[[Any], Any] | None = ...,
    mask: np.ndarray[Any, np.dtype[np.bool_]] | None = ...,
) -> np.ndarray[Any, np.dtype[np.intp]]: ...
def nargminmax(
    values: ExtensionArray,
    method: str,
    axis: int = ...,
) -> Any: ...
def _nanargminmax(
    values: np.ndarray[Any, Any],
    mask: np.ndarray[Any, np.dtype[np.bool_]],
    func: Callable[[Any], Any],
) -> Any: ...
def _ensure_key_mapped_multiindex(
    index: MultiIndex,
    key: Callable[[Any], Any],
    level: Any = ...,
) -> MultiIndex: ...
def ensure_key_mapped(
    values: Any,
    key: Callable[[Any], Any] | None,
    levels: list[Any] | None = ...,
) -> Any: ...
def get_indexer_dict(
    label_list: list[np.ndarray[Any, Any]],
    keys: Sequence[Any],
) -> dict[Any, Any]: ...
def get_group_index_sorter(
    group_index: np.ndarray[Any, np.dtype[np.intp]],
    ngroups: int | None = ...,
) -> np.ndarray[Any, np.dtype[np.intp]]: ...
def compress_group_index(
    group_index: np.ndarray[Any, np.dtype[np.intp]],
    sort: bool = ...,
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.int64]]]: ...
def _reorder_by_uniques(
    uniques: np.ndarray[Any, np.dtype[np.int64]],
    labels: np.ndarray[Any, np.dtype[np.intp]],
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.intp]]]: ...
```
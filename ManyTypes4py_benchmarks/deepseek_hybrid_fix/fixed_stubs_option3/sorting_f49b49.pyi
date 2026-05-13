from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterator, Literal, Sequence
import numpy as np
from pandas._typing import ArrayLike, AxisInt, IndexKeyFunc, Level, NaPosition, Shape, SortKind, npt

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence
    from pandas._typing import ArrayLike, AxisInt, IndexKeyFunc, Level, NaPosition, Shape, SortKind, npt
    from pandas import MultiIndex, Series
    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.base import Index

def get_indexer_indexer(
    target: Index,
    level: Level | None,
    ascending: bool | Sequence[bool],
    kind: SortKind,
    na_position: NaPosition,
    sort_remaining: bool,
    key: IndexKeyFunc | None,
) -> np.ndarray[np.intp] | None: ...

def get_group_index(
    labels: Sequence[ArrayLike],
    shape: Shape,
    sort: bool,
    xnull: bool,
) -> np.ndarray[np.int64]: ...

def get_compressed_ids(
    labels: Sequence[ArrayLike],
    sizes: Sequence[int],
) -> tuple[np.ndarray[np.intp], np.ndarray[np.int64]]: ...

def is_int64_overflow_possible(shape: Sequence[int]) -> bool: ...

def _decons_group_index(
    comp_labels: np.ndarray[np.signedinteger[Any]],
    shape: Sequence[int],
) -> list[np.ndarray[np.signedinteger[Any]]]: ...

def decons_obs_group_ids(
    comp_ids: np.ndarray[np.intp],
    obs_ids: np.ndarray[np.intp],
    shape: Sequence[int],
    labels: Sequence[np.ndarray[np.signedinteger[Any]]],
    xnull: bool,
) -> list[np.ndarray[np.intp]]: ...

def lexsort_indexer(
    keys: Sequence[ArrayLike | Series],
    orders: bool | Sequence[bool] | None = None,
    na_position: Literal['first', 'last'] = 'last',
    key: Callable[[Any], Any] | None = None,
    codes_given: bool = False,
) -> np.ndarray[np.intp]: ...

def nargsort(
    items: ArrayLike,
    kind: SortKind = 'quicksort',
    ascending: bool = True,
    na_position: Literal['first', 'last'] = 'last',
    key: Callable[[Any], Any] | None = None,
    mask: np.ndarray[np.bool_] | None = None,
) -> np.ndarray[np.intp]: ...

def nargminmax(
    values: ExtensionArray,
    method: Literal['argmax', 'argmin'],
    axis: AxisInt = 0,
) -> int | np.ndarray[np.intp]: ...

def _nanargminmax(
    values: np.ndarray[Any, Any],
    mask: np.ndarray[np.bool_],
    func: Callable[[np.ndarray[Any, Any]], int],
) -> int: ...

def _ensure_key_mapped_multiindex(
    index: MultiIndex,
    key: Callable[[Any], Any],
    level: Level | Sequence[Level] | None = None,
) -> MultiIndex: ...

def ensure_key_mapped(
    values: ArrayLike | Index | Series,
    key: Callable[[Any], Any] | None,
    levels: Sequence[Level] | None = None,
) -> ArrayLike | Index | Series: ...

def get_indexer_dict(
    label_list: Sequence[np.ndarray[np.signedinteger[Any]]],
    keys: Sequence[Sequence[Hashable]],
) -> dict[Hashable, np.ndarray[np.intp]]: ...

def get_group_index_sorter(
    group_index: np.ndarray[np.intp],
    ngroups: int | None = None,
) -> np.ndarray[np.intp]: ...

def compress_group_index(
    group_index: np.ndarray[np.signedinteger[Any]],
    sort: bool = True,
) -> tuple[np.ndarray[np.int64], np.ndarray[np.int64]]: ...

def _reorder_by_uniques(
    uniques: np.ndarray[np.int64],
    labels: np.ndarray[np.intp],
) -> tuple[np.ndarray[np.int64], np.ndarray[np.intp]]: ...
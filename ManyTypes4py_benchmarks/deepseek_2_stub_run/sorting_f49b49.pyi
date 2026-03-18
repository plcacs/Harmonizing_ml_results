```python
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Sequence,
    overload,
    Optional,
    Union,
)
import numpy as np

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        AxisInt,
        IndexKeyFunc,
        Level,
        NaPosition,
        Shape,
        SortKind,
        npt,
    )
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
) -> Optional[np.ndarray]: ...

def get_group_index(
    labels: Sequence[ArrayLike],
    shape: Shape,
    sort: bool,
    xnull: bool,
) -> np.ndarray: ...

def get_compressed_ids(
    labels: list[ArrayLike],
    sizes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]: ...

def is_int64_overflow_possible(shape: Shape) -> bool: ...

def _decons_group_index(
    comp_labels: np.ndarray,
    shape: Shape,
) -> list[np.ndarray]: ...

def decons_obs_group_ids(
    comp_ids: np.ndarray,
    obs_ids: np.ndarray,
    shape: tuple[int, ...],
    labels: Sequence[np.ndarray],
    xnull: bool,
) -> list[np.ndarray]: ...

def lexsort_indexer(
    keys: Sequence[ArrayLike | Index | Series],
    orders: bool | Sequence[bool] | None = None,
    na_position: NaPosition = "last",
    key: Callable | None = None,
    codes_given: bool = False,
) -> np.ndarray: ...

def nargsort(
    items: ArrayLike | Index | Series,
    kind: SortKind = "quicksort",
    ascending: bool = True,
    na_position: NaPosition = "last",
    key: Callable | None = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray: ...

def nargminmax(
    values: ExtensionArray,
    method: str,
    axis: AxisInt = 0,
) -> int | np.ndarray: ...

def _nanargminmax(
    values: np.ndarray,
    mask: np.ndarray,
    func: Callable,
) -> int: ...

def _ensure_key_mapped_multiindex(
    index: MultiIndex,
    key: IndexKeyFunc,
    level: Level | Sequence[Level] | None = None,
) -> MultiIndex: ...

def ensure_key_mapped(
    values: Any,
    key: IndexKeyFunc | None,
    levels: Sequence[Level] | None = None,
) -> Any: ...

def get_indexer_dict(
    label_list: Sequence[ArrayLike],
    keys: Sequence[ArrayLike],
) -> dict[Any, Any]: ...

def get_group_index_sorter(
    group_index: np.ndarray,
    ngroups: int | None = None,
) -> np.ndarray: ...

def compress_group_index(
    group_index: np.ndarray,
    sort: bool = True,
) -> tuple[np.ndarray, np.ndarray]: ...

def _reorder_by_uniques(
    uniques: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]: ...
```
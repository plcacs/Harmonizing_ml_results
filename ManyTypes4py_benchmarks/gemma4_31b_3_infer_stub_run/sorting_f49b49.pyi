from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Sequence, Callable, overload
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence as ABCSequence
    from pandas._typing import ArrayLike, AxisInt, IndexKeyFunc, Level, NaPosition, Shape, SortKind, npt
    from pandas import MultiIndex, Series
    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.base import Index

def get_indexer_indexer(
    target: Index,
    level: Optional[Union[int, str, Sequence[Union[int, str]]]],
    ascending: Union[bool, Sequence[bool]],
    kind: SortKind,
    na_position: NaPosition,
    sort_remaining: bool,
    key: Optional[IndexKeyFunc],
) -> Optional[np.ndarray[np.intp]]: ...

def get_group_index(
    labels: Sequence[np.ndarray],
    shape: Sequence[int],
    sort: bool,
    xnull: bool,
) -> np.ndarray[np.int64]: ...

def get_compressed_ids(
    labels: Sequence[np.ndarray],
    sizes: Sequence[int],
) -> tuple[np.ndarray[np.intp], np.ndarray[np.int64]]: ...

def is_int64_overflow_possible(shape: Sequence[int]) -> bool: ...

def _decons_group_index(comp_labels: np.ndarray, shape: Sequence[int]) -> list[np.ndarray]: ...

def decons_obs_group_ids(
    comp_ids: np.ndarray[np.intp],
    obs_ids: np.ndarray[np.intp],
    shape: Sequence[int],
    labels: Sequence[np.ndarray],
    xnull: bool,
) -> list[np.ndarray]: ...

def lexsort_indexer(
    keys: Sequence[Union[ArrayLike, Index, Series]],
    orders: Optional[Union[bool, Sequence[bool]]] = None,
    na_position: NaPosition = 'last',
    key: Optional[Callable] = None,
    codes_given: bool = False,
) -> np.ndarray[np.intp]: ...

def nargsort(
    items: Union[np.ndarray, ExtensionArray, Index, Series],
    kind: SortKind = 'quicksort',
    ascending: bool = True,
    na_position: NaPosition = 'last',
    key: Optional[Callable] = None,
    mask: Optional[np.ndarray[bool]] = None,
) -> np.ndarray[np.intp]: ...

def nargminmax(
    values: ExtensionArray,
    method: Union[Literal["argmax", "argmin"], str],
    axis: int = 0,
) -> Union[int, np.ndarray]: ...

def _nanargminmax(values: np.ndarray, mask: np.ndarray[bool], func: Callable) -> np.intp: ...

def _ensure_key_mapped_multiindex(
    index: MultiIndex,
    key: Callable,
    level: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
) -> MultiIndex: ...

def ensure_key_mapped(
    values: Union[Series, Index, np.ndarray],
    key: Optional[Callable],
    levels: Optional[Sequence] = None,
) -> Union[Series, Index, np.ndarray]: ...

def get_indexer_dict(label_list: Sequence[np.ndarray], keys: Sequence[Sequence]) -> dict: ...

def get_group_index_sorter(
    group_index: np.ndarray[np.intp],
    ngroups: Optional[int] = None,
) -> np.ndarray[np.intp]: ...

def compress_group_index(
    group_index: np.ndarray,
    sort: bool = True,
) -> tuple[np.ndarray[np.int64], np.ndarray[np.int64]]: ...

def _reorder_by_uniques(
    uniques: np.ndarray[np.int64],
    labels: np.ndarray[np.intp],
) -> tuple[np.ndarray[np.int64], np.ndarray[np.intp]]: ...

from typing import Literal
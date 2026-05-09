"""miscellaneous sorting / groupby utilities"""

from __future__ import annotations
import itertools
import numpy as np
from collections.abc import Callable, Hashable, Sequence
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)
from pandas._typing import (
    ArrayLike,
    AxisInt,
    IndexKeyFunc,
    Level,
    NaPosition,
    npt,
)

def get_indexer_indexer(
    target: Any,
    level: Union[Level, Sequence[Level]],
    ascending: Union[bool, Sequence[bool]],
    kind: str,
    na_position: NaPosition,
    sort_remaining: bool,
    key: Optional[IndexKeyFunc],
) -> Optional[npt.NDArray[np.intp]]:
    ...

def get_group_index(
    labels: List[np.ndarray[np.signedinteger]],
    shape: Tuple[int, ...],
    sort: bool,
    xnull: bool,
) -> npt.NDArray[np.int64]:
    ...

def get_compressed_ids(
    labels: List[np.ndarray[np.signedinteger]],
    sizes: Tuple[int, ...],
) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    ...

def is_int64_overflow_possible(shape: Tuple[int, ...]) -> bool:
    ...

def _decons_group_index(
    comp_labels: np.ndarray[np.signedinteger],
    shape: Tuple[int, ...],
) -> List[np.ndarray[np.signedinteger]]:
    ...

def decons_obs_group_ids(
    comp_ids: npt.NDArray[np.intp],
    obs_ids: npt.NDArray[np.intp],
    shape: Tuple[int, ...],
    labels: Sequence[np.ndarray[np.signedinteger]],
    xnull: bool,
) -> List[np.ndarray[np.intp]]:
    ...

def lexsort_indexer(
    keys: Sequence[Union[ArrayLike, Any, Any]],
    orders: Optional[Union[bool, Sequence[bool]]] = ...,
    na_position: NaPosition = ...,
    key: Optional[Callable] = ...,
    codes_given: bool = ...,
) -> npt.NDArray[np.intp]:
    ...

def nargsort(
    items: Union[np.ndarray, Any, Any],
    kind: str = ...,
    ascending: bool = ...,
    na_position: NaPosition = ...,
    key: Optional[Callable] = ...,
    mask: Optional[np.ndarray[bool]] = ...,
) -> npt.NDArray[np.intp]:
    ...

def nargminmax(
    values: Any,
    method: Literal["argmax", "argmin"],
    axis: int = ...,
) -> int:
    ...

def _nanargminmax(
    values: np.ndarray[np.signedinteger],
    mask: np.ndarray[bool],
    func: Callable,
) -> int:
    ...

def _ensure_key_mapped_multiindex(
    index: Any,
    key: Callable,
    level: Optional[Union[Level, Sequence[Level]]] = ...,
) -> Any:
    ...

def ensure_key_mapped(
    values: Union[Any, Any, Any],
    key: Optional[Callable],
    levels: Optional[List] = ...,
) -> Any:
    ...

def get_indexer_dict(
    label_list: List[np.ndarray[np.signedinteger]],
    keys: List[np.ndarray[np.signedinteger]],
) -> Dict[Any, npt.NDArray[np.intp]]:
    ...

def get_group_index_sorter(
    group_index: npt.NDArray[np.intp],
    ngroups: Optional[int] = ...,
) -> npt.NDArray[np.intp]:
    ...

def compress_group_index(
    group_index: npt.NDArray[np.intp],
    sort: bool = ...,
) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    ...

def _reorder_by_uniques(
    uniques: npt.NDArray[np.int64],
    labels: npt.NDArray[np.intp],
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.intp]]:
    ...
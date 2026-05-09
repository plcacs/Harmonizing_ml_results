"""miscellaneous sorting / groupby utilities"""

import numpy as np
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from pandas import Index, MultiIndex, Series
from pandas.core.arrays import ExtensionArray
from pandas.core.indexes.base import Index
from pandas._typing import ArrayLike, AxisInt, SortKind, npt

def get_indexer_indexer(
    target: Index,
    level: Union[int, str, List[Union[int, str]]],
    ascending: Union[bool, List[bool]],
    kind: SortKind,
    na_position: str,
    sort_remaining: bool,
    key: Optional[Callable],
) -> Optional[np.ndarray[npt.intp]]:
    ...

def get_group_index(
    labels: List[np.ndarray[np.signedinteger]],
    shape: Tuple[int, ...],
    sort: bool,
    xnull: bool,
) -> np.ndarray[np.int64]:
    ...

def get_compressed_ids(
    labels: List[np.ndarray[np.signedinteger]],
    sizes: Tuple[int, ...],
) -> Tuple[np.ndarray[np.intp], np.ndarray[np.int64]]:
    ...

def is_int64_overflow_possible(shape: Tuple[int, ...]) -> bool:
    ...

def _decons_group_index(
    comp_labels: np.ndarray[np.signedinteger],
    shape: Tuple[int, ...],
) -> List[np.ndarray[np.signedinteger]]:
    ...

def decons_obs_group_ids(
    comp_ids: np.ndarray[np.intp],
    obs_ids: np.ndarray[np.intp],
    shape: Tuple[int, ...],
    labels: Sequence[np.ndarray[np.signedinteger]],
    xnull: bool,
) -> List[np.ndarray[np.intp]]:
    ...

def lexsort_indexer(
    keys: Sequence[ArrayLike],
    orders: Optional[Union[bool, List[bool]]] = ...,
    na_position: str = ...,
    key: Optional[Callable] = ...,
    codes_given: bool = ...,
) -> np.ndarray[np.intp]:
    ...

def nargsort(
    items: Union[np.ndarray, ExtensionArray, Index, Series],
    kind: SortKind = ...,
    ascending: bool = ...,
    na_position: str = ...,
    key: Optional[Callable] = ...,
    mask: Optional[np.ndarray[np.bool_]] = ...,
) -> np.ndarray[np.intp]:
    ...

def nargminmax(
    values: ExtensionArray,
    method: str,
    axis: int = ...,
) -> int:
    ...

def _nanargminmax(
    values: np.ndarray,
    mask: np.ndarray[np.bool_],
    func: Callable,
) -> int:
    ...

def _ensure_key_mapped_multiindex(
    index: MultiIndex,
    key: Callable,
    level: Optional[Union[List[Union[int, str]], int, str]] = ...,
) -> MultiIndex:
    ...

def ensure_key_mapped(
    values: Union[Series, "DataFrame", Index, np.ndarray],
    key: Optional[Callable],
    levels: Optional[List[int]] = ...,
) -> Union[Series, "DataFrame", Index, np.ndarray]:
    ...

def get_indexer_dict(
    label_list: List[np.ndarray],
    keys: List[np.ndarray],
) -> Dict[Any, Any]:
    ...

def get_group_index_sorter(
    group_index: np.ndarray[np.intp],
    ngroups: Optional[int] = ...,
) -> np.ndarray[np.intp]:
    ...

def compress_group_index(
    group_index: np.ndarray[np.intp],
    sort: bool = ...,
) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]:
    ...

def _reorder_by_uniques(
    uniques: np.ndarray[np.int64],
    labels: np.ndarray[np.intp],
) -> Tuple[np.ndarray[np.int64], np.ndarray[np.intp]]:
    ...
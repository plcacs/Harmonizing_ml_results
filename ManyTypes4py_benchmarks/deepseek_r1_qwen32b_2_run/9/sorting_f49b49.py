"""miscellaneous sorting / groupby utilities"""
from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Any, Callable, Hashable, List, Optional, Sequence, Tuple, Union
import numpy as np
from pandas._libs import algos, hashtable, lib
from pandas._libs.hashtable import unique_label_indices
from pandas.core.dtypes.common import ensure_int64, ensure_platform_int
from pandas.core.dtypes.generic import ABCMultiIndex, ABCRangeIndex
from pandas.core.dtypes.missing import isna
from pandas.core.construction import extract_array
from pandas._typing import ArrayLike, Index, Level, NaPosition, SortKind
from numpy import ndarray

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence
    from pandas import MultiIndex, Series
    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.base import Index

def get_indexer_indexer(target: Index, level: Level, ascending: Union[bool, List[bool]], kind: SortKind, na_position: NaPosition, sort_remaining: bool, key: Optional[Callable]) -> Optional[ndarray]:
    ...

def get_group_index(labels: Sequence[ndarray], shape: Tuple[int, ...], sort: bool, xnull: bool) -> ndarray:
    ...

def get_compressed_ids(labels: Sequence[ndarray], sizes: Tuple[int, ...]) -> Tuple[ndarray, ndarray]:
    ...

def is_int64_overflow_possible(shape: Tuple[int, ...]) -> bool:
    ...

def _decons_group_index(comp_labels: ndarray, shape: Tuple[int, ...]) -> List[ndarray]:
    ...

def decons_obs_group_ids(comp_ids: ndarray, obs_ids: ndarray, shape: Tuple[int, ...], labels: Sequence[ndarray], xnull: bool) -> List[ndarray]:
    ...

def lexsort_indexer(keys: Sequence[ArrayLike], orders: Optional[Union[bool, List[bool]]] = None, na_position: NaPosition = 'last', key: Optional[Callable] = None, codes_given: bool = False) -> ndarray:
    ...

def nargsort(items: Union[ExtensionArray, Index, Series, ndarray], kind: SortKind = 'quicksort', ascending: bool = True, na_position: NaPosition = 'last', key: Optional[Callable] = None, mask: Optional[ndarray] = None) -> ndarray:
    ...

def nargminmax(values: ExtensionArray, method: Literal['argmax', 'argmin'], axis: int = 0) -> int:
    ...

def _nanargminmax(values: ndarray, mask: ndarray, func: Callable) -> int:
    ...

def _ensure_key_mapped_multiindex(index: MultiIndex, key: Callable, level: Optional[Union[str, int, List]] = None) -> MultiIndex:
    ...

def ensure_key_mapped(values: Union[ExtensionArray, Index, Series, ndarray], key: Optional[Callable], levels: Optional[List] = None) -> Union[ExtensionArray, Index, Series, ndarray]:
    ...

def get_indexer_dict(label_list: List[ndarray], keys: Sequence[Hashable]) -> dict:
    ...

def get_group_index_sorter(group_index: ndarray, ngroups: Optional[int] = None) -> ndarray:
    ...

def compress_group_index(group_index: ndarray, sort: bool = True) -> Tuple[ndarray, ndarray]:
    ...

def _reorder_by_uniques(uniques: ndarray, labels: ndarray) -> Tuple[ndarray, ndarray]:
    ...
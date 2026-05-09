from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Callable, Hashable, Sequence, ArrayLike, AxisInt, IndexKeyFunc, Level, NaPosition, Shape, SortKind, npt
from pandas._typing import ExtensionArray
from pandas import MultiIndex, Series
from pandas.core.dtypes.common import ensure_int64, ensure_platform_int
from pandas.core.dtypes.generic import ABCMultiIndex, ABCRangeIndex
from pandas.core.dtypes.missing import isna
from pandas.core.construction import extract_array
from pandas._libs import algos, hashtable, lib

def get_indexer_indexer(target: Index, level: AxisInt, ascending: bool, kind: SortKind, na_position: NaPosition, sort_remaining: bool, key: IndexKeyFunc) -> Optional[npt]:
    # ... (rest of the function remains the same)

def get_group_index(labels: Sequence[ArrayLike], shape: Shape, sort: bool, xnull: bool) -> ArrayLike:
    # ... (rest of the function remains the same)

def get_compressed_ids(labels: Sequence[ArrayLike], sizes: Shape) -> Tuple[npt, ArrayLike]:
    # ... (rest of the function remains the same)

def decons_obs_group_ids(comp_ids: npt, obs_ids: npt, shape: Shape, labels: Sequence[ArrayLike], xnull: bool) -> Sequence[ArrayLike]:
    # ... (rest of the function remains the same)

def lexsort_indexer(keys: Sequence[ArrayLike], orders: Optional[Sequence[bool]], na_position: NaPosition, key: Callable, codes_given: bool) -> npt:
    # ... (rest of the function remains the same)

def nargsort(items: ArrayLike, kind: SortKind, ascending: bool, na_position: NaPosition, key: Callable, mask: Optional[npt]) -> npt:
    # ... (rest of the function remains the same)

def _nanargminmax(values: ArrayLike, mask: npt, func: Callable) -> npt:
    # ... (rest of the function remains the same)

def _ensure_key_mapped_multiindex(index: MultiIndex, key: Callable, level: Optional[Level]) -> MultiIndex:
    # ... (rest of the function remains the same)

def ensure_key_mapped(values: ArrayLike, key: Callable, levels: Optional[Sequence[Level]]) -> ArrayLike:
    # ... (rest of the function remains the same)

def get_indexer_dict(label_list: Sequence[ArrayLike], keys: Sequence[ArrayLike]) -> dict:
    # ... (rest of the function remains the same)

def get_group_index_sorter(group_index: npt, ngroups: Optional[int]) -> npt:
    # ... (rest of the function remains the same)

def compress_group_index(group_index: npt, sort: bool) -> Tuple[npt, npt]:
    # ... (rest of the function remains the same)

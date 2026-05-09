"""miscellaneous sorting / groupby utilities"""

from __future__ import annotations
import itertools
from typing import (
    Any,
    Callable,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import numpy as np
from numpy import ndarray
from pandas._libs import hashtable
from pandas._typing import (
    ArrayLike,
    AxisInt,
    IndexKeyFunc,
    Level,
    NaPosition,
    SortKind,
)
from pandas.core.dtypes.missing import isna
from pandas.core.indexes.base import Index
from pandas.core.arrays import ExtensionArray
from pandas import MultiIndex, Series


def get_indexer_indexer(
    target: Index,
    level: Union[int, str, List[Union[int, str]]],
    ascending: Union[bool, List[bool]],
    kind: SortKind,
    na_position: NaPosition,
    sort_remaining: bool,
    key: Optional[IndexKeyFunc],
) -> Optional[ndarray[int]]:
    ...


def get_group_index(
    labels: List[ArrayLike],
    shape: Tuple[int, ...],
    sort: bool,
    xnull: bool,
) -> ndarray[int]:
    ...


def get_compressed_ids(
    labels: List[ArrayLike],
    sizes: Tuple[int, ...],
) -> Tuple[ndarray[int], ndarray[int]]:
    ...


def is_int64_overflow_possible(shape: Tuple[int, ...]) -> bool:
    ...


def _decons_group_index(
    comp_labels: ndarray[int],
    shape: Tuple[int, ...],
) -> List[ndarray[int]]:
    ...


def decons_obs_group_ids(
    comp_ids: ndarray[int],
    obs_ids: ndarray[int],
    shape: Tuple[int, ...],
    labels: Sequence[ndarray[int]],
    xnull: bool,
) -> List[ndarray[int]]:
    ...


def lexsort_indexer(
    keys: Sequence[Union[ArrayLike, Index, Series]],
    orders: Optional[Union[bool, List[bool]]] = None,
    na_position: NaPosition = 'last',
    key: Optional[Callable] = None,
    codes_given: bool = False,
) -> ndarray[int]:
    ...


def nargsort(
    items: Union[ndarray[Any], ExtensionArray, Index, Series],
    kind: SortKind = 'quicksort',
    ascending: bool = True,
    na_position: NaPosition = 'last',
    key: Optional[Callable] = None,
    mask: Optional[ndarray[bool]] = None,
) -> ndarray[int]:
    ...


def nargminmax(
    values: ExtensionArray,
    method: Literal['argmax', 'argmin'],
    axis: int = 0,
) -> int:
    ...


def _nanargminmax(
    values: ndarray[Any],
    mask: ndarray[bool],
    func: Callable,
) -> int:
    ...


def _ensure_key_mapped_multiindex(
    index: MultiIndex,
    key: Callable,
    level: Optional[Union[int, str, List[Union[int, str]]]] = None,
) -> MultiIndex:
    ...


def ensure_key_mapped(
    values: Union[Index, Series, ndarray[Any]],
    key: Optional[Callable],
    levels: Optional[List[int]] = None,
) -> Union[Index, Series, ndarray[Any]]:
    ...


def get_indexer_dict(
    label_list: List[ndarray[Any]],
    keys: List[ndarray[Any]],
) -> dict:
    ...


def get_group_index_sorter(
    group_index: ndarray[int],
    ngroups: Optional[int] = None,
) -> ndarray[int]:
    ...


def compress_group_index(
    group_index: ndarray[int],
    sort: bool = True,
) -> Tuple[ndarray[int], ndarray[int]]:
    ...


def _reorder_by_uniques(
    uniques: ndarray[int],
    labels: ndarray[int],
) -> Tuple[ndarray[int], ndarray[int]]:
    ...
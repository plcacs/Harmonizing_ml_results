import string
import numpy as np
from pandas import (
    NA,
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
    array,
    date_range,
)
from typing import Any, Dict, List, Optional, Tuple, Union

class GetLoc:
    mi_large: MultiIndex
    mi_med: MultiIndex
    mi_small: MultiIndex

    def setup(self) -> None:
        ...

    def time_large_get_loc(self) -> int:
        ...

    def time_large_get_loc_warm(self) -> None:
        ...

    def time_med_get_loc(self) -> int:
        ...

    def time_med_get_loc_warm(self) -> None:
        ...

    def time_string_get_loc(self) -> int:
        ...

    def time_small_get_loc_warm(self) -> None:
        ...

class GetLocs:
    mi_large: MultiIndex
    mi_med: MultiIndex
    mi_small: MultiIndex

    def setup(self) -> None:
        ...

    def time_large_get_locs(self) -> np.ndarray[int]:
        ...

    def time_med_get_locs(self) -> np.ndarray[int]:
        ...

    def time_small_get_locs(self) -> np.ndarray[int]:
        ...

class Duplicates:
    mi_unused_levels: MultiIndex

    def setup(self) -> None:
        ...

    def time_remove_unused_levels(self) -> None:
        ...

class Integer:
    mi_int: MultiIndex
    obj_index: np.ndarray[object]
    other_mi_many_mismatches: MultiIndex

    def setup(self) -> None:
        ...

    def time_get_indexer(self) -> np.ndarray[int]:
        ...

    def time_get_indexer_and_backfill(self) -> np.ndarray[int]:
        ...

    def time_get_indexer_and_pad(self) -> np.ndarray[int]:
        ...

    def time_is_monotonic(self) -> bool:
        ...

class Duplicated:
    mi: MultiIndex

    def setup(self) -> None:
        ...

    def time_duplicated(self) -> Series[bool]:
        ...

class Sortlevel:
    mi_int: MultiIndex
    mi: MultiIndex

    def setup(self) -> None:
        ...

    def time_sortlevel_int64(self) -> None:
        ...

    def time_sortlevel_zero(self) -> None:
        ...

    def time_sortlevel_one(self) -> None:
        ...

class SortValues:
    mi: MultiIndex

    def setup(self, dtype: str) -> None:
        ...

    def time_sort_values(self, dtype: str) -> None:
        ...

class Values:
    mi: MultiIndex

    def setup_cache(self) -> MultiIndex:
        ...

    def time_datetime_level_values_copy(self, mi: MultiIndex) -> np.ndarray:
        ...

    def time_datetime_level_values_sliced(self, mi: MultiIndex) -> np.ndarray:
        ...

class CategoricalLevel:
    df: DataFrame

    def setup(self) -> None:
        ...

    def time_categorical_level(self) -> None:
        ...

class Equals:
    mi: MultiIndex
    mi_deepcopy: MultiIndex
    idx_non_object: RangeIndex

    def setup(self) -> None:
        ...

    def time_equals_deepcopy(self) -> bool:
        ...

    def time_equals_non_object_index(self) -> bool:
        ...

class SetOperations:
    left: MultiIndex
    right: MultiIndex

    def setup(
        self,
        index_structure: str,
        dtype: str,
        method: str,
        sort: Optional[bool],
    ) -> None:
        ...

    def time_operation(
        self,
        index_structure: str,
        dtype: str,
        method: str,
        sort: Optional[bool],
    ) -> MultiIndex:
        ...

class Difference:
    left: MultiIndex
    right: MultiIndex

    def setup(self, dtype: str) -> None:
        ...

    def time_difference(self, dtype: str) -> MultiIndex:
        ...

class Unique:
    midx: MultiIndex
    midx_dups: MultiIndex

    def setup(self, dtype_val: Tuple[str, int]) -> None:
        ...

    def time_unique(self, dtype_val: Tuple[str, int]) -> MultiIndex:
        ...

    def time_unique_dups(self, dtype_val: Tuple[str, int]) -> MultiIndex:
        ...

class Isin:
    midx: MultiIndex
    values_small: MultiIndex
    values_large: MultiIndex

    def setup(self, dtype: str) -> None:
        ...

    def time_isin_small(self, dtype: str) -> Series[bool]:
        ...

    def time_isin_large(self, dtype: str) -> Series[bool]:
        ...

class Putmask:
    midx: MultiIndex
    midx_values: MultiIndex
    midx_values_different: MultiIndex
    mask: np.ndarray[bool]

    def setup(self) -> None:
        ...

    def time_putmask(self) -> None:
        ...

    def time_putmask_all_different(self) -> None:
        ...

class Append:
    left: MultiIndex
    right: MultiIndex

    def setup(self, dtype: str) -> None:
        ...

    def time_append(self, dtype: str) -> MultiIndex:
        ...
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

class GetLoc:
    mi_large: MultiIndex
    mi_med: MultiIndex
    mi_small: MultiIndex

    def setup(self) -> None:
        ...

    def time_large_get_loc(self) -> None:
        ...

    def time_large_get_loc_warm(self) -> None:
        ...

    def time_med_get_loc(self) -> None:
        ...

    def time_med_get_loc_warm(self) -> None:
        ...

    def time_string_get_loc(self) -> None:
        ...

    def time_small_get_loc_warm(self) -> None:
        ...

class GetLocs:
    mi_large: MultiIndex
    mi_med: MultiIndex
    mi_small: MultiIndex

    def setup(self) -> None:
        ...

    def time_large_get_locs(self) -> None:
        ...

    def time_med_get_locs(self) -> None:
        ...

    def time_small_get_locs(self) -> None:
        ...

class Duplicates:
    mi_unused_levels: MultiIndex

    def setup(self) -> None:
        ...

    def time_remove_unused_levels(self) -> None:
        ...

class Integer:
    mi_int: MultiIndex
    obj_index: np.ndarray
    other_mi_many_mismatches: MultiIndex

    def setup(self) -> None:
        ...

    def time_get_indexer(self) -> None:
        ...

    def time_get_indexer_and_backfill(self) -> None:
        ...

    def time_get_indexer_and_pad(self) -> None:
        ...

    def time_is_monotonic(self) -> None:
        ...

class Duplicated:
    mi: MultiIndex

    def setup(self) -> None:
        ...

    def time_duplicated(self) -> None:
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
    def setup_cache(self) -> MultiIndex:
        ...

    def time_datetime_level_values_copy(self, mi: MultiIndex) -> None:
        ...

    def time_datetime_level_values_sliced(self, mi: MultiIndex) -> None:
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

    def time_equals_deepcopy(self) -> None:
        ...

    def time_equals_non_object_index(self) -> None:
        ...

class SetOperations:
    left: MultiIndex
    right: MultiIndex

    def setup(
        self,
        index_structure: str,
        dtype: str,
        method: str,
        sort: bool | None,
    ) -> None:
        ...

    def time_operation(
        self,
        index_structure: str,
        dtype: str,
        method: str,
        sort: bool | None,
    ) -> None:
        ...

class Difference:
    left: MultiIndex
    right: MultiIndex

    def setup(self, dtype: str) -> None:
        ...

    def time_difference(self, dtype: str) -> None:
        ...

class Unique:
    midx: MultiIndex
    midx_dups: MultiIndex

    def setup(self, dtype_val: tuple[str, int | float]) -> None:
        ...

    def time_unique(self, dtype_val: tuple[str, int | float]) -> None:
        ...

    def time_unique_dups(self, dtype_val: tuple[str, int | float]) -> None:
        ...

class Isin:
    midx: MultiIndex
    values_small: MultiIndex
    values_large: MultiIndex

    def setup(self, dtype: str) -> None:
        ...

    def time_isin_small(self, dtype: str) -> None:
        ...

    def time_isin_large(self, dtype: str) -> None:
        ...

class Putmask:
    midx: MultiIndex
    midx_values: MultiIndex
    midx_values_different: MultiIndex
    mask: np.ndarray

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

    def time_append(self, dtype: str) -> None:
        ...
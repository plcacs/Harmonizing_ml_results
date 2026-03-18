```python
import datetime
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, RangeIndex, Series

class GetLoc:
    mi_large: MultiIndex
    mi_med: MultiIndex
    mi_small: MultiIndex
    def setup(self) -> None: ...
    def time_large_get_loc(self) -> None: ...
    def time_large_get_loc_warm(self) -> None: ...
    def time_med_get_loc(self) -> None: ...
    def time_med_get_loc_warm(self) -> None: ...
    def time_string_get_loc(self) -> None: ...
    def time_small_get_loc_warm(self) -> None: ...

class GetLocs:
    mi_large: MultiIndex
    mi_med: MultiIndex
    mi_small: MultiIndex
    def setup(self) -> None: ...
    def time_large_get_locs(self) -> None: ...
    def time_med_get_locs(self) -> None: ...
    def time_small_get_locs(self) -> None: ...

class Duplicates:
    mi_unused_levels: MultiIndex
    def setup(self) -> None: ...
    def time_remove_unused_levels(self) -> None: ...

class Integer:
    mi_int: MultiIndex
    obj_index: Any
    other_mi_many_mismatches: MultiIndex
    def setup(self) -> None: ...
    def time_get_indexer(self) -> None: ...
    def time_get_indexer_and_backfill(self) -> None: ...
    def time_get_indexer_and_pad(self) -> None: ...
    def time_is_monotonic(self) -> None: ...

class Duplicated:
    mi: MultiIndex
    def setup(self) -> None: ...
    def time_duplicated(self) -> None: ...

class Sortlevel:
    mi_int: MultiIndex
    mi: MultiIndex
    def setup(self) -> None: ...
    def time_sortlevel_int64(self) -> None: ...
    def time_sortlevel_zero(self) -> None: ...
    def time_sortlevel_one(self) -> None: ...

class SortValues:
    params: Any = ...
    param_names: Any = ...
    mi: MultiIndex
    @overload
    def setup(self, dtype: Literal["int64"]) -> None: ...
    @overload
    def setup(self, dtype: Literal["Int64"]) -> None: ...
    def setup(self, dtype: Any) -> None: ...
    @overload
    def time_sort_values(self, dtype: Literal["int64"]) -> None: ...
    @overload
    def time_sort_values(self, dtype: Literal["Int64"]) -> None: ...
    def time_sort_values(self, dtype: Any) -> None: ...

class Values:
    @staticmethod
    def setup_cache() -> MultiIndex: ...
    def time_datetime_level_values_copy(self, mi: MultiIndex) -> None: ...
    def time_datetime_level_values_sliced(self, mi: MultiIndex) -> None: ...

class CategoricalLevel:
    df: DataFrame
    def setup(self) -> None: ...
    def time_categorical_level(self) -> None: ...

class Equals:
    mi: MultiIndex
    mi_deepcopy: MultiIndex
    idx_non_object: RangeIndex
    def setup(self) -> None: ...
    def time_equals_deepcopy(self) -> None: ...
    def time_equals_non_object_index(self) -> None: ...

class SetOperations:
    params: Any = ...
    param_names: Any = ...
    left: MultiIndex
    right: MultiIndex
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["datetime"],
        method: Literal["intersection"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["datetime"],
        method: Literal["intersection"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["datetime"],
        method: Literal["union"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["datetime"],
        method: Literal["union"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["datetime"],
        method: Literal["symmetric_difference"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["datetime"],
        method: Literal["symmetric_difference"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["int"],
        method: Literal["intersection"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["int"],
        method: Literal["intersection"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["int"],
        method: Literal["union"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["int"],
        method: Literal["union"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["int"],
        method: Literal["symmetric_difference"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["int"],
        method: Literal["symmetric_difference"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["string"],
        method: Literal["intersection"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["string"],
        method: Literal["intersection"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["string"],
        method: Literal["union"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["string"],
        method: Literal["union"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["string"],
        method: Literal["symmetric_difference"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["string"],
        method: Literal["symmetric_difference"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["ea_int"],
        method: Literal["intersection"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["ea_int"],
        method: Literal["intersection"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["ea_int"],
        method: Literal["union"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["ea_int"],
        method: Literal["union"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["ea_int"],
        method: Literal["symmetric_difference"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["monotonic"],
        dtype: Literal["ea_int"],
        method: Literal["symmetric_difference"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["datetime"],
        method: Literal["intersection"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["datetime"],
        method: Literal["intersection"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["datetime"],
        method: Literal["union"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["datetime"],
        method: Literal["union"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["datetime"],
        method: Literal["symmetric_difference"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["datetime"],
        method: Literal["symmetric_difference"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["int"],
        method: Literal["intersection"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["int"],
        method: Literal["intersection"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["int"],
        method: Literal["union"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["int"],
        method: Literal["union"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["int"],
        method: Literal["symmetric_difference"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["int"],
        method: Literal["symmetric_difference"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["string"],
        method: Literal["intersection"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["string"],
        method: Literal["intersection"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["string"],
        method: Literal["union"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["string"],
        method: Literal["union"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["string"],
        method: Literal["symmetric_difference"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["string"],
        method: Literal["symmetric_difference"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["ea_int"],
        method: Literal["intersection"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["ea_int"],
        method: Literal["intersection"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["ea_int"],
        method: Literal["union"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["ea_int"],
        method: Literal["union"],
        sort: None,
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["ea_int"],
        method: Literal["symmetric_difference"],
        sort: Literal[False],
    ) -> None: ...
    @overload
    def setup(
        self,
        index_structure: Literal["non_monotonic"],
        dtype: Literal["ea_int"],
        method: Literal["symmetric_difference"],
        sort: None,
    ) -> None: ...
    def setup(self, index_structure: Any, dtype: Any, method: Any, sort: Any) -> None: ...
    def time_operation(self, index_structure: Any, dtype: Any, method: Any, sort: Any) -> None: ...

class Difference:
    params: Any = ...
    param_names: Any = ...
    left: MultiIndex
    right: MultiIndex
    @overload
    def setup(self, dtype: Literal["datetime"]) -> None: ...
    @overload
    def setup(self, dtype: Literal["int"]) -> None: ...
    @overload
    def setup(self, dtype: Literal["string"]) -> None: ...
    @overload
    def setup(self, dtype: Literal["ea_int"]) -> None: ...
    def setup(self, dtype: Any) -> None: ...
    @overload
    def time_difference(self, dtype: Literal["datetime"]) -> None: ...
    @overload
    def time_difference(self, dtype: Literal["int"]) -> None: ...
    @overload
    def time_difference(self, dtype: Literal["string"]) -> None: ...
    @overload
    def time_difference(self, dtype: Literal["ea_int"]) -> None: ...
    def time_difference(self, dtype: Any) -> None: ...

class Unique:
    params: Any = ...
    param_names: Any = ...
    midx: MultiIndex
    midx_dups: MultiIndex
    @overload
    def setup(self, dtype_val: tuple[Literal["Int64"], Any]) -> None: ...
    @overload
    def setup(self, dtype_val: tuple[Literal["int64"], Literal[0]]) -> None: ...
    def setup(self, dtype_val: Any) -> None: ...
    @overload
    def time_unique(self, dtype_val: tuple[Literal["Int64"], Any]) -> None: ...
    @overload
    def time_unique(self, dtype_val: tuple[Literal["int64"], Literal[0]]) -> None: ...
    def time_unique(self, dtype_val: Any) -> None: ...
    @overload
    def time_unique_dups(self, dtype_val: tuple[Literal["Int64"], Any]) -> None: ...
    @overload
    def time_unique_dups(self, dtype_val: tuple[Literal["int64"], Literal[0]]) -> None: ...
    def time_unique_dups(self, dtype_val: Any) -> None: ...

class Isin:
    params: Any = ...
    param_names: Any = ...
    midx: MultiIndex
    values_small: Any
    values_large: Any
    @overload
    def setup(self, dtype: Literal["string"]) -> None: ...
    @overload
    def setup(self, dtype: Literal["int"]) -> None: ...
    @overload
    def setup(self, dtype: Literal["datetime"]) -> None: ...
    def setup(self, dtype: Any) -> None: ...
    @overload
    def time_isin_small(self, dtype: Literal["string"]) -> None: ...
    @
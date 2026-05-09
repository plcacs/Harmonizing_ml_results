import string
import numpy as np
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    array,
    concat,
    date_range,
    merge,
    merge_asof,
)
try:
    from pandas import merge_ordered
except ImportError:
    from pandas import ordered_merge as merge_ordered

class Concat:
    params: list[int]
    param_names: list[str]
    
    def setup(self, axis: int) -> None:
        ...
    
    def time_concat_series(self, axis: int) -> None:
        ...
    
    def time_concat_small_frames(self, axis: int) -> None:
        ...
    
    def time_concat_empty_right(self, axis: int) -> None:
        ...
    
    def time_concat_empty_left(self, axis: int) -> None:
        ...
    
    def time_concat_mixed_ndims(self, axis: int) -> None:
        ...

class ConcatDataFrames:
    params: list[tuple[int, bool]]
    param_names: list[str]
    
    def setup(self, axis: int, ignore_index: bool) -> None:
        ...
    
    def time_c_ordered(self, axis: int, ignore_index: bool) -> None:
        ...
    
    def time_f_ordered(self, axis: int, ignore_index: bool) -> None:
        ...

class ConcatIndexDtype:
    params: list[tuple[str, str, int, bool]]
    param_names: list[str]
    
    def setup(self, dtype: str, structure: str, axis: int, sort: bool) -> None:
        ...
    
    def time_concat_series(self, dtype: str, structure: str, axis: int, sort: bool) -> None:
        ...

class Join:
    params: list[bool]
    param_names: list[str]
    
    def setup(self, sort: bool) -> None:
        ...
    
    def time_join_dataframe_index_multi(self, sort: bool) -> None:
        ...
    
    def time_join_dataframe_index_single_key_bigger(self, sort: bool) -> None:
        ...
    
    def time_join_dataframe_index_single_key_small(self, sort: bool) -> None:
        ...
    
    def time_join_dataframe_index_shuffle_key_bigger_sort(self, sort: bool) -> None:
        ...
    
    def time_join_dataframes_cross(self, sort: bool) -> None:
        ...

class JoinIndex:
    def setup(self) -> None:
        ...
    
    def time_left_outer_join_index(self) -> None:
        ...

class JoinMultiindexSubset:
    def setup(self) -> None:
        ...
    
    def time_join_multiindex_subset(self) -> None:
        ...

class JoinEmpty:
    def setup(self) -> None:
        ...
    
    def time_inner_join_left_empty(self) -> None:
        ...
    
    def time_inner_join_right_empty(self) -> None:
        ...

class JoinNonUnique:
    def setup(self) -> None:
        ...
    
    def time_join_non_unique_equal(self) -> None:
        ...

class Merge:
    params: list[bool]
    param_names: list[str]
    
    def setup(self, sort: bool) -> None:
        ...
    
    def time_merge_2intkey(self, sort: bool) -> None:
        ...
    
    def time_merge_dataframe_integer_2key(self, sort: bool) -> None:
        ...
    
    def time_merge_dataframe_integer_key(self, sort: bool) -> None:
        ...
    
    def time_merge_dataframe_empty_right(self, sort: bool) -> None:
        ...
    
    def time_merge_dataframe_empty_left(self, sort: bool) -> None:
        ...
    
    def time_merge_dataframes_cross(self, sort: bool) -> None:
        ...

class MergeEA:
    params: list[tuple[str, bool]]
    param_names: list[str]
    
    def setup(self, dtype: str, monotonic: bool) -> None:
        ...
    
    def time_merge(self, dtype: str, monotonic: bool) -> None:
        ...

class I8Merge:
    params: list[str]
    param_names: list[str]
    
    def setup(self, how: str) -> None:
        ...
    
    def time_i8merge(self, how: str) -> None:
        ...

class UniqueMerge:
    params: list[int]
    param_names: list[str]
    
    def setup(self, unique_elements: int) -> None:
        ...
    
    def time_unique_merge(self, unique_elements: int) -> None:
        ...

class MergeDatetime:
    params: list[tuple[tuple[str, str], str, bool]]
    param_names: list[str]
    
    def setup(self, units: tuple[str, str], tz: str, monotonic: bool) -> None:
        ...
    
    def time_merge(self, units: tuple[str, str], tz: str, monotonic: bool) -> None:
        ...

class MergeCategoricals:
    def setup(self) -> None:
        ...
    
    def time_merge_object(self) -> None:
        ...
    
    def time_merge_cat(self) -> None:
        ...
    
    def time_merge_on_cat_col(self) -> None:
        ...
    
    def time_merge_on_cat_idx(self) -> None:
        ...

class MergeOrdered:
    def setup(self) -> None:
        ...
    
    def time_merge_ordered(self) -> None:
        ...

class MergeAsof:
    params: list[tuple[str, int]]
    param_names: list[str]
    
    def setup(self, direction: str, tolerance: int) -> None:
        ...
    
    def time_on_int(self, direction: str, tolerance: int) -> None:
        ...
    
    def time_on_int32(self, direction: str, tolerance: int) -> None:
        ...
    
    def time_on_uint64(self, direction: str, tolerance: int) -> None:
        ...
    
    def time_by_object(self, direction: str, tolerance: int) -> None:
        ...
    
    def time_by_int(self, direction: str, tolerance: int) -> None:
        ...
    
    def time_multiby(self, direction: str, tolerance: int) -> None:
        ...

class MergeMultiIndex:
    params: list[tuple[tuple[str, str], str]]
    param_names: list[str]
    
    def setup(self, dtypes: tuple[str, str], how: str) -> None:
        ...
    
    def time_merge_sorted_multiindex(self, dtypes: tuple[str, str], how: str) -> None:
        ...

class Align:
    def setup(self) -> None:
        ...
    
    def time_series_align_int64_index(self) -> None:
        ...
    
    def time_series_align_left_monotonic(self) -> None:
        ...
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
    series: list[Series]
    small_frames: list[DataFrame]
    empty_left: list[DataFrame]
    empty_right: list[DataFrame]
    mixed_ndims: list[DataFrame]

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
    frame_c: list[DataFrame]
    frame_f: list[DataFrame]

    def setup(self, axis: int, ignore_index: bool) -> None:
        ...

    def time_c_ordered(self, axis: int, ignore_index: bool) -> None:
        ...

    def time_f_ordered(self, axis: int, ignore_index: bool) -> None:
        ...

class ConcatIndexDtype:
    params: list[tuple[str, str, int, bool]]
    param_names: list[str]
    series: list[Series]

    def setup(self, dtype: str, structure: str, axis: int, sort: bool) -> None:
        ...

    def time_concat_series(self, dtype: str, structure: str, axis: int, sort: bool) -> None:
        ...

class Join:
    params: list[bool]
    param_names: list[str]
    df_multi: DataFrame
    key1: np.ndarray
    key2: np.ndarray
    df: DataFrame
    df_key1: DataFrame
    df_key2: DataFrame
    df_shuf: DataFrame

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
    left: DataFrame
    right: DataFrame

    def setup(self) -> None:
        ...

    def time_left_outer_join_index(self) -> None:
        ...

class JoinMultiindexSubset:
    left: DataFrame
    right: DataFrame

    def setup(self) -> None:
        ...

    def time_join_multiindex_subset(self) -> None:
        ...

class JoinEmpty:
    df: DataFrame
    df_empty: DataFrame

    def setup(self) -> None:
        ...

    def time_inner_join_left_empty(self) -> None:
        ...

    def time_inner_join_right_empty(self) -> None:
        ...

class JoinNonUnique:
    fracofday: Series
    temp: Series

    def setup(self) -> None:
        ...

    def time_join_non_unique_equal(self) -> None:
        ...

class Merge:
    params: list[bool]
    param_names: list[str]
    left: DataFrame
    right: DataFrame
    df: DataFrame
    df2: DataFrame
    df3: DataFrame

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
    left: DataFrame
    right: DataFrame

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
    left: DataFrame
    right: DataFrame

    def setup(self, unique_elements: int) -> None:
        ...

    def time_unique_merge(self, unique_elements: int) -> None:
        ...

class MergeDatetime:
    params: list[tuple[tuple[str, str], str, bool]]
    param_names: list[str]
    left: DataFrame
    right: DataFrame

    def setup(self, units: tuple[str, str], tz: str, monotonic: bool) -> None:
        ...

    def time_merge(self, units: tuple[str, str], tz: str, monotonic: bool) -> None:
        ...

class MergeCategoricals:
    left_object: DataFrame
    right_object: DataFrame
    left_cat: DataFrame
    right_cat: DataFrame
    left_cat_col: DataFrame
    right_cat_col: DataFrame
    left_cat_idx: DataFrame
    right_cat_idx: DataFrame

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
    left: DataFrame
    right: DataFrame

    def setup(self) -> None:
        ...

    def time_merge_ordered(self) -> None:
        ...

class MergeAsof:
    params: list[tuple[str, int]]
    param_names: list[str]
    df1a: DataFrame
    df2a: DataFrame
    df1b: DataFrame
    df2b: DataFrame
    df1c: DataFrame
    df2c: DataFrame
    df1d: DataFrame
    df2d: DataFrame
    df1e: DataFrame
    df2e: DataFrame
    df1f: DataFrame
    df2f: DataFrame

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
    df1: DataFrame
    df2: DataFrame

    def setup(self, dtypes: tuple[str, str], how: str) -> None:
        ...

    def time_merge_sorted_multiindex(self, dtypes: tuple[str, str], how: str) -> None:
        ...

class Align:
    ts1: Series
    ts2: Series

    def setup(self) -> None:
        ...

    def time_series_align_int64_index(self) -> None:
        ...

    def time_series_align_left_monotonic(self) -> None:
        ...
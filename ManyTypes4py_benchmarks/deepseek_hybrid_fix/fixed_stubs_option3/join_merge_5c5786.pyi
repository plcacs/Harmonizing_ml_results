from typing import Any, ClassVar, List, Optional, Tuple, Union
import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series, array, concat, date_range, merge, merge_asof, merge_ordered
import string
import datetime

class Concat:
    params: ClassVar[List[int]]
    param_names: ClassVar[List[str]]
    series: List[Series]
    small_frames: List[DataFrame]
    empty_left: List[DataFrame]
    empty_right: List[DataFrame]
    mixed_ndims: List[DataFrame]

    def setup(self, axis: int) -> None: ...
    def time_concat_series(self, axis: int) -> None: ...
    def time_concat_small_frames(self, axis: int) -> None: ...
    def time_concat_empty_right(self, axis: int) -> None: ...
    def time_concat_empty_left(self, axis: int) -> None: ...
    def time_concat_mixed_ndims(self, axis: int) -> None: ...

class ConcatDataFrames:
    params: tuple[list[int], list[bool]]
    param_names: ClassVar[List[str]]
    frame_c: List[DataFrame]
    frame_f: List[DataFrame]

    def setup(self, axis: int, ignore_index: bool) -> None: ...
    def time_c_ordered(self, axis: int, ignore_index: bool) -> None: ...
    def time_f_ordered(self, axis: int, ignore_index: bool) -> None: ...

class ConcatIndexDtype:
    params: tuple[list[str], list[str], list[int], list[bool]]
    param_names: ClassVar[List[str]]
    series: List[Series]

    def setup(self, dtype: str, structure: str, axis: int, sort: bool) -> None: ...
    def time_concat_series(self, dtype: str, structure: str, axis: int, sort: bool) -> None: ...

class Join:
    params: ClassVar[List[bool]]
    param_names: ClassVar[List[str]]
    df_multi: DataFrame
    key1: Any
    key2: Any
    df: DataFrame
    df_key1: DataFrame
    df_key2: DataFrame
    df_shuf: DataFrame

    def setup(self, sort: bool) -> None: ...
    def time_join_dataframe_index_multi(self, sort: bool) -> None: ...
    def time_join_dataframe_index_single_key_bigger(self, sort: bool) -> None: ...
    def time_join_dataframe_index_single_key_small(self, sort: bool) -> None: ...
    def time_join_dataframe_index_shuffle_key_bigger_sort(self, sort: bool) -> None: ...
    def time_join_dataframes_cross(self, sort: bool) -> None: ...

class JoinIndex:
    left: DataFrame
    right: DataFrame

    def setup(self) -> None: ...
    def time_left_outer_join_index(self) -> None: ...

class JoinMultiindexSubset:
    left: DataFrame
    right: DataFrame

    def setup(self) -> None: ...
    def time_join_multiindex_subset(self) -> None: ...

class JoinEmpty:
    df: DataFrame
    df_empty: DataFrame

    def setup(self) -> None: ...
    def time_inner_join_left_empty(self) -> None: ...
    def time_inner_join_right_empty(self) -> None: ...

class JoinNonUnique:
    fracofday: Series
    temp: Series

    def setup(self) -> None: ...
    def time_join_non_unique_equal(self) -> None: ...

class Merge:
    params: ClassVar[List[bool]]
    param_names: ClassVar[List[str]]
    left: DataFrame
    right: DataFrame
    df: DataFrame
    df2: DataFrame
    df3: DataFrame

    def setup(self, sort: bool) -> None: ...
    def time_merge_2intkey(self, sort: bool) -> None: ...
    def time_merge_dataframe_integer_2key(self, sort: bool) -> None: ...
    def time_merge_dataframe_integer_key(self, sort: bool) -> None: ...
    def time_merge_dataframe_empty_right(self, sort: bool) -> None: ...
    def time_merge_dataframe_empty_left(self, sort: bool) -> None: ...
    def time_merge_dataframes_cross(self, sort: bool) -> None: ...

class MergeEA:
    params: list[list[str] | list[bool]]
    param_names: ClassVar[List[str]]
    left: DataFrame
    right: DataFrame

    def setup(self, dtype: str, monotonic: bool) -> None: ...
    def time_merge(self, dtype: str, monotonic: bool) -> None: ...

class I8Merge:
    params: ClassVar[List[str]]
    param_names: ClassVar[List[str]]
    left: DataFrame
    right: DataFrame

    def setup(self, how: str) -> None: ...
    def time_i8merge(self, how: str) -> None: ...

class UniqueMerge:
    params: ClassVar[List[int]]
    param_names: ClassVar[List[str]]
    left: DataFrame
    right: DataFrame

    def setup(self, unique_elements: int) -> None: ...
    def time_unique_merge(self, unique_elements: int) -> None: ...

class MergeDatetime:
    params: list[list[tuple[str, str]] | list[None | str] | list[bool]]
    param_names: ClassVar[List[str]]
    left: DataFrame
    right: DataFrame

    def setup(self, units: Tuple[str, str], tz: Optional[str], monotonic: bool) -> None: ...
    def time_merge(self, units: Tuple[str, str], tz: Optional[str], monotonic: bool) -> None: ...

class MergeCategoricals:
    left_object: DataFrame
    right_object: DataFrame
    left_cat: DataFrame
    right_cat: DataFrame
    left_cat_col: DataFrame
    right_cat_col: DataFrame
    left_cat_idx: DataFrame
    right_cat_idx: DataFrame

    def setup(self) -> None: ...
    def time_merge_object(self) -> None: ...
    def time_merge_cat(self) -> None: ...
    def time_merge_on_cat_col(self) -> None: ...
    def time_merge_on_cat_idx(self) -> None: ...

class MergeOrdered:
    left: DataFrame
    right: DataFrame

    def setup(self) -> None: ...
    def time_merge_ordered(self) -> None: ...

class MergeAsof:
    params: list[list[str] | list[int | None]]
    param_names: ClassVar[List[str]]
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

    def setup(self, direction: str, tolerance: Optional[int]) -> None: ...
    def time_on_int(self, direction: str, tolerance: Optional[int]) -> None: ...
    def time_on_int32(self, direction: str, tolerance: Optional[int]) -> None: ...
    def time_on_uint64(self, direction: str, tolerance: Optional[int]) -> None: ...
    def time_by_object(self, direction: str, tolerance: Optional[int]) -> None: ...
    def time_by_int(self, direction: str, tolerance: Optional[int]) -> None: ...
    def time_multiby(self, direction: str, tolerance: Optional[int]) -> None: ...

class MergeMultiIndex:
    params: list[list[tuple[str, str]] | list[str]]
    param_names: ClassVar[List[str]]
    df1: DataFrame
    df2: DataFrame

    def setup(self, dtypes: Tuple[str, str], how: str) -> None: ...
    def time_merge_sorted_multiindex(self, dtypes: Tuple[str, str], how: str) -> None: ...

class Align:
    ts1: Series
    ts2: Series

    def setup(self) -> None: ...
    def time_series_align_int64_index(self) -> None: ...
    def time_series_align_left_monotonic(self) -> None: ...
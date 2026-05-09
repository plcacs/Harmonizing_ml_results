import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series
from typing import Any, Union, Optional, Sequence, List, Tuple

class Concat:
    params: List[int]
    param_names: List[str]
    series: List[Series]
    small_frames: List[DataFrame]
    empty_left: List[DataFrame]
    empty_right: List[DataFrame]
    mixed_ndims: List[DataFrame]

    def setup(self, axis: int) -> None: ...
    def time_concat_series(self, axis: int) -> Any: ...
    def time_concat_small_frames(self, axis: int) -> Any: ...
    def time_concat_empty_right(self, axis: int) -> Any: ...
    def time_concat_empty_left(self, axis: int) -> Any: ...
    def time_concat_mixed_ndims(self, axis: int) -> Any: ...

class ConcatDataFrames:
    params: Tuple[List[int], List[bool]]
    param_names: List[str]
    frame_c: List[DataFrame]
    frame_f: List[DataFrame]

    def setup(self, axis: int, ignore_index: bool) -> None: ...
    def time_c_ordered(self, axis: int, ignore_index: bool) -> Any: ...
    def time_f_ordered(self, axis: int, ignore_index: bool) -> Any: ...

class ConcatIndexDtype:
    params: Tuple[List[str], List[str], List[int], List[bool]]
    param_names: List[str]
    series: List[Series]

    def setup(self, dtype: str, structure: str, axis: int, sort: bool) -> None: ...
    def time_concat_series(self, dtype: str, structure: str, axis: int, sort: bool) -> Any: ...

class Join:
    params: List[bool]
    param_names: List[str]
    df_multi: DataFrame
    key1: np.ndarray
    key2: np.ndarray
    df: DataFrame
    df_key1: DataFrame
    df_key2: DataFrame
    df_shuf: DataFrame

    def setup(self, sort: bool) -> None: ...
    def time_join_dataframe_index_multi(self, sort: bool) -> DataFrame: ...
    def time_join_dataframe_index_single_key_bigger(self, sort: bool) -> DataFrame: ...
    def time_join_dataframe_index_single_key_small(self, sort: bool) -> DataFrame: ...
    def time_join_dataframe_index_shuffle_key_bigger_sort(self, sort: bool) -> DataFrame: ...
    def time_join_dataframes_cross(self, sort: bool) -> DataFrame: ...

class JoinIndex:
    left: DataFrame
    right: DataFrame

    def setup(self) -> None: ...
    def time_left_outer_join_index(self) -> DataFrame: ...

class JoinMultiindexSubset:
    left: DataFrame
    right: DataFrame

    def setup(self) -> None: ...
    def time_join_multiindex_subset(self) -> DataFrame: ...

class JoinEmpty:
    df: DataFrame
    df_empty: DataFrame

    def setup(self) -> None: ...
    def time_inner_join_left_empty(self) -> DataFrame: ...
    def time_inner_join_right_empty(self) -> DataFrame: ...

class JoinNonUnique:
    fracofday: Series
    temp: Series

    def setup(self) -> None: ...
    def time_join_non_unique_equal(self) -> Series: ...

class Merge:
    params: List[bool]
    param_names: List[str]
    left: DataFrame
    right: DataFrame
    df: DataFrame
    df2: DataFrame
    df3: DataFrame

    def setup(self, sort: bool) -> None: ...
    def time_merge_2intkey(self, sort: bool) -> DataFrame: ...
    def time_merge_dataframe_integer_2key(self, sort: bool) -> DataFrame: ...
    def time_merge_dataframe_integer_key(self, sort: bool) -> DataFrame: ...
    def time_merge_dataframe_empty_right(self, sort: bool) -> DataFrame: ...
    def time_merge_dataframe_empty_left(self, sort: bool) -> DataFrame: ...
    def time_merge_dataframes_cross(self, sort: bool) -> DataFrame: ...

class MergeEA:
    params: List[Union[List[str], List[bool]]]
    param_names: List[str]
    left: DataFrame
    right: DataFrame

    def setup(self, dtype: str, monotonic: bool) -> None: ...
    def time_merge(self, dtype: str, monotonic: bool) -> DataFrame: ...

class I8Merge:
    params: List[str]
    param_names: List[str]
    left: DataFrame
    right: DataFrame

    def setup(self, how: str) -> None: ...
    def time_i8merge(self, how: str) -> DataFrame: ...

class UniqueMerge:
    params: List[int]
    param_names: List[str]
    left: DataFrame
    right: DataFrame

    def setup(self, unique_elements: int) -> None: ...
    def time_unique_merge(self, unique_elements: int) -> DataFrame: ...

class MergeDatetime:
    params: List[Union[List[Tuple[str, str]], List[Optional[str]], List[bool]]]
    param_names: List[str]
    left: DataFrame
    right: DataFrame

    def setup(self, units: Tuple[str, str], tz: Optional[str], monotonic: bool) -> None: ...
    def time_merge(self, units: Tuple[str, str], tz: Optional[str], monotonic: bool) -> DataFrame: ...

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
    def time_merge_object(self) -> DataFrame: ...
    def time_merge_cat(self) -> DataFrame: ...
    def time_merge_on_cat_col(self) -> DataFrame: ...
    def time_merge_on_cat_idx(self) -> DataFrame: ...

class MergeOrdered:
    left: DataFrame
    right: DataFrame

    def setup(self) -> None: ...
    def time_merge_ordered(self) -> DataFrame: ...

class MergeAsof:
    params: List[Union[List[str], List[Optional[int]]]]
    param_names: List[str]
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
    def time_on_int(self, direction: str, tolerance: Optional[int]) -> DataFrame: ...
    def time_on_int32(self, direction: str, tolerance: Optional[int]) -> DataFrame: ...
    def time_on_uint64(self, direction: str, tolerance: Optional[int]) -> DataFrame: ...
    def time_by_object(self, direction: str, tolerance: Optional[int]) -> DataFrame: ...
    def time_by_int(self, direction: str, tolerance: Optional[int]) -> DataFrame: ...
    def time_multiby(self, direction: str, tolerance: Optional[int]) -> DataFrame: ...

class MergeMultiIndex:
    params: List[Union[List[Tuple[str, str]], List[str]]]
    param_names: List[str]
    df1: DataFrame
    df2: DataFrame

    def setup(self, dtypes: Tuple[str, str], how: str) -> None: ...
    def time_merge_sorted_multiindex(self, dtypes: Tuple[str, str], how: str) -> DataFrame: ...

class Align:
    ts1: Series
    ts2: Series

    def setup(self) -> None: ...
    def time_series_align_int64_index(self) -> Series: ...
    def time_series_align_left_monotonic(self) -> Tuple[Series, Series]: ...
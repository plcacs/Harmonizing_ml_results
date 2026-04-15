import datetime
from typing import Any, ClassVar, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series, concat, date_range, merge, merge_asof

try:
    from pandas import merge_ordered
except ImportError:
    from pandas import ordered_merge as merge_ordered

class Concat:
    params: ClassVar[List[int]] = [0, 1]
    param_names: ClassVar[List[str]] = ['axis']
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
    params: ClassVar[List[Tuple[List[int], List[bool]]]] = ([0, 1], [True, False])
    param_names: ClassVar[List[str]] = ['axis', 'ignore_index']
    frame_c: List[DataFrame]
    frame_f: List[DataFrame]

    def setup(self, axis: int, ignore_index: bool) -> None: ...
    def time_c_ordered(self, axis: int, ignore_index: bool) -> None: ...
    def time_f_ordered(self, axis: int, ignore_index: bool) -> None: ...

class ConcatIndexDtype:
    params: ClassVar[List[Tuple[List[str], List[str], List[int], List[bool]]]] = (['datetime64[ns]', 'int64', 'Int64', 'int64[pyarrow]', 'string[python]', 'string[pyarrow]'], ['monotonic', 'non_monotonic', 'has_na'], [0, 1], [True, False])
    param_names: ClassVar[List[str]] = ['dtype', 'structure', 'axis', 'sort']
    series: List[Series]

    def setup(self, dtype: str, structure: str, axis: int, sort: bool) -> None: ...
    def time_concat_series(self, dtype: str, structure: str, axis: int, sort: bool) -> None: ...

class Join:
    params: ClassVar[List[bool]] = [True, False]
    param_names: ClassVar[List[str]] = ['sort']
    df_multi: DataFrame
    key1: np.ndarray
    key2: np.ndarray
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
    params: ClassVar[List[bool]] = [True, False]
    param_names: ClassVar[List[str]] = ['sort']
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
    params: ClassVar[List[Tuple[List[str], List[bool]]]] = [['Int64', 'Int32', 'Int16', 'UInt64', 'UInt32', 'UInt16', 'Float64', 'Float32'], [True, False]]
    param_names: ClassVar[List[str]] = ['dtype', 'monotonic']
    left: DataFrame
    right: DataFrame

    def setup(self, dtype: str, monotonic: bool) -> None: ...
    def time_merge(self, dtype: str, monotonic: bool) -> None: ...

class I8Merge:
    params: ClassVar[List[str]] = ['inner', 'outer', 'left', 'right']
    param_names: ClassVar[List[str]] = ['how']
    left: DataFrame
    right: DataFrame

    def setup(self, how: str) -> None: ...
    def time_i8merge(self, how: str) -> None: ...

class UniqueMerge:
    params: ClassVar[List[int]] = [4000000, 1000000]
    param_names: ClassVar[List[str]] = ['unique_elements']
    left: DataFrame
    right: DataFrame

    def setup(self, unique_elements: int) -> None: ...
    def time_unique_merge(self, unique_elements: int) -> None: ...

class MergeDatetime:
    params: ClassVar[List[Tuple[List[Tuple[str, str]], List[Optional[str]], List[bool]]]] = [[('ns', 'ns'), ('ms', 'ms'), ('ns', 'ms')], [None, 'Europe/Brussels'], [True, False]]
    param_names: ClassVar[List[str]] = ['units', 'tz', 'monotonic']
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
    params: ClassVar[List[Tuple[List[str], List[Optional[int]]]]] = [['backward', 'forward', 'nearest'], [None, 5]]
    param_names: ClassVar[List[str]] = ['direction', 'tolerance']
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
    params: ClassVar[List[Tuple[List[Tuple[str, str]], List[str]]]] = [[('int64', 'int64'), ('datetime64[ns]', 'int64'), ('Int64', 'Int64')], ['left', 'right', 'inner', 'outer']]
    param_names: ClassVar[List[str]] = ['dtypes', 'how']
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

def setup(*args: Any, **kwargs: Any) -> Any: ...
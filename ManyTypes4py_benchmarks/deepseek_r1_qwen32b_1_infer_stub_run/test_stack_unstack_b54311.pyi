from __future__ import annotations
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Callable,
    Iterable,
    Iterator,
    overload,
)

import numpy as np
import pandas as pd
from pandas import (
    DataFrame,
    Series,
    MultiIndex,
    Index,
    Timedelta,
    Period,
    date_range,
)
from pandas._testing import tm

@pytest.fixture(params=[True, False])
def future_stack(request: Any) -> bool:
    ...

class TestDataFrameReshape:
    def test_stack_unstack(
        self,
        float_frame: DataFrame,
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_mixed_level(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_unstack_not_consolidated(
        self,
    ) -> None:
        ...

    def test_unstack_fill(
        self,
        future_stack: bool,
        fill_value: Union[int, float, None] = ...,
    ) -> None:
        ...

    def test_unstack_fill_frame(
        self,
    ) -> None:
        ...

    def test_unstack_fill_frame_datetime(
        self,
    ) -> None:
        ...

    def test_unstack_fill_frame_timedelta(
        self,
    ) -> None:
        ...

    def test_unstack_fill_frame_period(
        self,
    ) -> None:
        ...

    def test_unstack_fill_frame_categorical(
        self,
    ) -> None:
        ...

    def test_unstack_tuplename_in_multiindex(
        self,
    ) -> None:
        ...

    def test_unstack_mixed_type_name_in_multiindex(
        self,
        unstack_idx: Union[Tuple[str, str], Tuple[Tuple[str, str], str]],
        expected_values: List[List[Any]],
        expected_index: MultiIndex,
        expected_columns: MultiIndex,
    ) -> None:
        ...

    def test_unstack_preserve_dtypes(
        self,
        using_infer_string: bool,
    ) -> None:
        ...

    def test_unstack_bool(
        self,
    ) -> None:
        ...

    def test_unstack_level_binding(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_unstack_to_series(
        self,
        float_frame: DataFrame,
    ) -> None:
        ...

    def test_unstack_dtypes(
        self,
        using_infer_string: bool,
    ) -> None:
        ...

    def test_unstack_dtypes_mixed_date(
        self,
        c: np.ndarray,
        d: np.ndarray,
    ) -> None:
        ...

    def test_unstack_non_unique_index_names(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_unstack_unused_levels(
        self,
    ) -> None:
        ...

    def test_unstack_unused_levels_mixed_with_nan(
        self,
        level: int,
        idces: List[int],
        col_level: List[Any],
        idx_level: List[Any],
    ) -> None:
        ...

    def test_unstack_unused_level(
        self,
        cols: Union[List[str], slice],
    ) -> None:
        ...

    def test_unstack_long_index(
        self,
    ) -> None:
        ...

    def test_unstack_multi_level_cols(
        self,
    ) -> None:
        ...

    def test_unstack_multi_level_rows_and_cols(
        self,
    ) -> None:
        ...

    def test_unstack_nan_index1(
        self,
        idx: Tuple[str, str],
        lev: int,
    ) -> None:
        ...

    def test_unstack_nan_index_repeats(
        self,
        idx: Tuple[str, str, str],
        lev: int,
        col: str,
    ) -> None:
        ...

    def test_unstack_nan_index2(
        self,
    ) -> None:
        ...

    def test_unstack_nan_index3(
        self,
    ) -> None:
        ...

    def test_unstack_nan_index4(
        self,
    ) -> None:
        ...

    def test_unstack_nan_index5(
        self,
    ) -> None:
        ...

    def test_stack_datetime_column_multiIndex(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_partial_multiIndex(
        self,
        multiindex_columns: List[int],
        level: Union[int, List[int]],
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_full_multiIndex(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_preserve_categorical_dtype(
        self,
        ordered: bool,
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_multi_preserve_categorical_dtype(
        self,
        ordered: bool,
        labels: List[str],
        data: List[int],
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_preserve_categorical_dtype_values(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_multi_columns_non_unique_index(
        self,
        index: List[int],
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_multi_columns_mixed_extension_types(
        self,
        vals1: List[int],
        vals2: List[Union[int, str]],
        dtype1: str,
        dtype2: str,
        expected_dtype: str,
        future_stack: bool,
    ) -> None:
        ...

    def test_unstack_mixed_extension_types(
        self,
        level: int,
    ) -> None:
        ...

    def test_unstack_swaplevel_sortlevel(
        self,
        level: int,
    ) -> None:
        ...

    def test_unstack_sort_false(
        self,
        frame_or_series: Union[DataFrame, Series],
        dtype: str,
    ) -> None:
        ...

    def test_unstack_fill_frame_object(
        self,
    ) -> None:
        ...

    def test_unstack_timezone_aware_values(
        self,
    ) -> None:
        ...

    def test_stack_timezone_aware_values(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_empty_frame(
        self,
        dropna: Union[bool, Any],
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_empty_level(
        self,
        dropna: Union[bool, Any],
        future_stack: bool,
        int_frame: DataFrame,
    ) -> None:
        ...

    def test_stack_unstack_empty_frame(
        self,
        dropna: Union[bool, Any],
        fill_value: Any,
        future_stack: bool,
    ) -> None:
        ...

    def test_unstack_single_index_series(
        self,
    ) -> None:
        ...

    def test_unstacking_multi_index_df(
        self,
    ) -> None:
        ...

    def test_stack_positional_level_duplicate_column_names(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_unstack_non_slice_like_blocks(
        self,
    ) -> None:
        ...

    def test_stack_sort_false(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_sort_false_multi_level(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_unstack(self, multiindex_year_month_day_dataframe_random_data: DataFrame) -> None:
        ...

    def test_stack(self, multiindex_year_month_day_dataframe_random_data: DataFrame, future_stack: bool) -> None:
        ...

    def test_stack_names_and_numbers(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_multiple_out_of_bounds(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_period_series(
        self,
    ) -> None:
        ...

    def test_unstack_period_frame(
        self,
    ) -> None:
        ...

    def test_stack_multiple_bug(
        self,
        future_stack: bool,
        using_infer_string: bool,
    ) -> None:
        ...

    def test_stack_dropna(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_unstack_sparse_keyspace(
        self,
    ) -> None:
        ...

    def test_unstack_unobserved_keys(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_unstack_number_of_levels_larger_than_int32(
        self,
        performance_warning: Any,
        monkeypatch: Any,
    ) -> None:
        ...

    def test_stack_order_with_unsorted_levels(
        self,
        levels: Tuple[Tuple[Any, ...], Tuple[Any, ...]],
        stack_lev: int,
        sort: bool,
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_order_with_unsorted_levels_multi_row(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_order_with_unsorted_levels_multi_row_2(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_unstack_unordered_multiindex(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_unstack_preserve_types(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        using_infer_string: bool,
    ) -> None:
        ...

    def test_unstack_group_index_overflow(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_unstack_with_missing_int_cast_to_float(
        self,
    ) -> None:
        ...

    def test_unstack_with_level_has_nan(
        self,
    ) -> None:
        ...

    def test_stack_nan_in_multiindex_columns(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_multi_level_stack_categorical(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_nan_level(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_unstack_categorical_columns(
        self,
    ) -> None:
        ...

    def test_stack_unsorted(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_stack_nullable_dtype(
        self,
        future_stack: bool,
    ) -> None:
        ...

    def test_unstack_mixed_level_names(
        self,
    ) -> None:
        ...

def test_stack_tuple_columns(
    future_stack: bool,
) -> None:
    ...

def test_stack_preserves_na(
    dtype: str,
    na_value: Any,
    test_multiindex: bool,
) -> None:
    ...
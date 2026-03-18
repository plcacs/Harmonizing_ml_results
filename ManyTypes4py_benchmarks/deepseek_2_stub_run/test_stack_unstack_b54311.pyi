```python
from typing import Any, Optional, Union, Sequence, Literal
from pandas import DataFrame, Series, Index, MultiIndex, Timedelta
from pandas._libs import lib
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def future_stack(request: Any) -> bool: ...

@pytest.fixture
def float_frame() -> DataFrame: ...

@pytest.fixture
def multiindex_year_month_day_dataframe_random_data() -> DataFrame: ...

@pytest.fixture
def multiindex_dataframe_random_data() -> DataFrame: ...

def test_unstack_sort_false(
    frame_or_series: Any,
    dtype: str
) -> None: ...

def test_unstack_fill_frame_object() -> None: ...

def test_unstack_timezone_aware_values() -> None: ...

def test_stack_timezone_aware_values(
    future_stack: bool
) -> None: ...

def test_stack_empty_frame(
    dropna: Union[bool, Any],
    future_stack: bool
) -> None: ...

def test_stack_empty_level(
    dropna: Union[bool, Any],
    future_stack: bool,
    int_frame: DataFrame
) -> None: ...

def test_stack_unstack_empty_frame(
    dropna: Union[bool, Any],
    fill_value: Optional[Any],
    future_stack: bool
) -> None: ...

def test_unstack_single_index_series() -> None: ...

def test_unstacking_multi_index_df() -> None: ...

def test_unstack_non_slice_like_blocks() -> None: ...

def test_stack_sort_false(
    future_stack: bool
) -> None: ...

def test_stack_sort_false_multi_level(
    future_stack: bool
) -> None: ...

def test_stack_tuple_columns(
    future_stack: bool
) -> None: ...

def test_stack_preserves_na(
    dtype: str,
    na_value: Any,
    test_multiindex: bool
) -> None: ...

class TestDataFrameReshape:
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unstack(
        self,
        float_frame: DataFrame,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_mixed_level(
        self,
        future_stack: bool
    ) -> None: ...
    
    def test_unstack_not_consolidated(self) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_fill(
        self,
        future_stack: bool
    ) -> None: ...
    
    def test_unstack_fill_frame(self) -> None: ...
    
    def test_unstack_fill_frame_datetime(self) -> None: ...
    
    def test_unstack_fill_frame_timedelta(self) -> None: ...
    
    def test_unstack_fill_frame_period(self) -> None: ...
    
    def test_unstack_fill_frame_categorical(self) -> None: ...
    
    def test_unstack_tuplename_in_multiindex(self) -> None: ...
    
    @pytest.mark.parametrize('unstack_idx, expected_values, expected_index, expected_columns', [])
    def test_unstack_mixed_type_name_in_multiindex(
        self,
        unstack_idx: Any,
        expected_values: Any,
        expected_index: Any,
        expected_columns: Any
    ) -> None: ...
    
    def test_unstack_preserve_dtypes(self) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_ints(
        self,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_mixed_levels(
        self,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_int_level_names(
        self,
        future_stack: bool
    ) -> None: ...
    
    def test_unstack_bool(self) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_level_binding(
        self,
        future_stack: bool
    ) -> None: ...
    
    def test_unstack_to_series(
        self,
        float_frame: DataFrame
    ) -> None: ...
    
    def test_unstack_dtypes(
        self,
        using_infer_string: bool
    ) -> None: ...
    
    @pytest.mark.parametrize('c, d', [])
    def test_unstack_dtypes_mixed_date(
        self,
        c: Any,
        d: Any
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_non_unique_index_names(
        self,
        future_stack: bool
    ) -> None: ...
    
    def test_unstack_unused_levels(self) -> None: ...
    
    @pytest.mark.parametrize('level, idces, col_level, idx_level', [])
    def test_unstack_unused_levels_mixed_with_nan(
        self,
        level: Any,
        idces: Any,
        col_level: Any,
        idx_level: Any
    ) -> None: ...
    
    @pytest.mark.parametrize('cols', [])
    def test_unstack_unused_level(
        self,
        cols: Any
    ) -> None: ...
    
    def test_unstack_long_index(self) -> None: ...
    
    def test_unstack_multi_level_cols(self) -> None: ...
    
    def test_unstack_multi_level_rows_and_cols(self) -> None: ...
    
    @pytest.mark.parametrize('idx', [])
    @pytest.mark.parametrize('lev', [])
    def test_unstack_nan_index1(
        self,
        idx: Any,
        lev: Any
    ) -> None: ...
    
    @pytest.mark.parametrize('idx', [])
    @pytest.mark.parametrize('lev', [])
    @pytest.mark.parametrize('col', [])
    def test_unstack_nan_index_repeats(
        self,
        idx: Any,
        lev: Any,
        col: Any
    ) -> None: ...
    
    def test_unstack_nan_index2(self) -> None: ...
    
    def test_unstack_nan_index3(self) -> None: ...
    
    def test_unstack_nan_index4(self) -> None: ...
    
    def test_unstack_nan_index5(self) -> None: ...
    
    def test_stack_datetime_column_multiIndex(
        self,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('multiindex_columns', [])
    @pytest.mark.parametrize('level', [])
    def test_stack_partial_multiIndex(
        self,
        multiindex_columns: Any,
        level: Any,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_full_multiIndex(
        self,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('ordered', [])
    def test_stack_preserve_categorical_dtype(
        self,
        ordered: bool,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('ordered', [])
    @pytest.mark.parametrize('labels,data', [])
    def test_stack_multi_preserve_categorical_dtype(
        self,
        ordered: bool,
        labels: Any,
        data: Any,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_preserve_categorical_dtype_values(
        self,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('index', [])
    def test_stack_multi_columns_non_unique_index(
        self,
        index: Any,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('vals1, vals2, dtype1, dtype2, expected_dtype', [])
    def test_stack_multi_columns_mixed_extension_types(
        self,
        vals1: Any,
        vals2: Any,
        dtype1: str,
        dtype2: str,
        expected_dtype: str,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.parametrize('level', [])
    def test_unstack_mixed_extension_types(
        self,
        level: Any
    ) -> None: ...
    
    @pytest.mark.parametrize('level', [])
    def test_unstack_swaplevel_sortlevel(
        self,
        level: Any
    ) -> None: ...

class TestStackUnstackMultiLevel:
    def test_unstack(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame
    ) -> None: ...
    
    @pytest.mark.parametrize('result_rows,result_columns,index_product,expected_row', [])
    def test_unstack_partial(
        self,
        result_rows: Any,
        result_columns: Any,
        index_product: Any,
        expected_row: Any
    ) -> None: ...
    
    def test_unstack_multiple_no_empty_columns(self) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('idx, exp_idx', [])
    def test_stack_duplicate_index(
        self,
        idx: Any,
        exp_idx: Any,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_odd_failure(
        self,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_mixed_dtype(
        self,
        multiindex_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_bug(
        self,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unstack_preserve_names(
        self,
        multiindex_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('method', [])
    def test_stack_unstack_wrong_level_name(
        self,
        method: str,
        multiindex_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None: ...
    
    def test_unstack_level_name(
        self,
        multiindex_dataframe_random_data: DataFrame
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_level_name(
        self,
        multiindex_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unstack_multiple(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_names_and_numbers(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_multiple_out_of_bounds(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None: ...
    
    def test_unstack_period_series(self) -> None: ...
    
    def test_unstack_period_frame(self) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_multiple_bug(
        self,
        future_stack: bool,
        using_infer_string: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_dropna(
        self,
        future_stack: bool
    ) -> None: ...
    
    def test_unstack_multiple_hierarchical(
        self,
        future_stack: bool
    ) -> None: ...
    
    def test_unstack_sparse_keyspace(self) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_unobserved_keys(
        self,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.slow
    def test_unstack_number_of_levels_larger_than_int32(
        self,
        performance_warning: Any,
        monkeypatch: Any
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('levels', [])
    @pytest.mark.parametrize('stack_lev', [])
    def test_stack_order_with_unsorted_levels(
        self,
        levels: Any,
        stack_lev: Any,
        sort: Any,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_order_with_unsorted_levels_multi_row(
        self,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_order_with_unsorted_levels_multi_row_2(
        self,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unstack_unordered_multiindex(
        self,
        future_stack: bool
    ) -> None: ...
    
    def test_unstack_preserve_types(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        using_infer_string: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_group_index_overflow(
        self,
        future_stack: bool
    ) -> None: ...
    
    def test_unstack_with_missing_int_cast_to_float(self) -> None: ...
    
    def test_unstack_with_level_has_nan(self) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_nan_in_multiindex_columns(
        self,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_multi_level_stack_categorical(
        self,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_nan_level(
        self,
        future_stack: bool
    ) -> None: ...
    
    def test_unstack_categorical_columns(self) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unsorted(
        self,
        future_stack: bool
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_nullable_dtype(
        self,
        future_stack: bool
    ) -> None: ...
    
    def test_unstack_mixed_level_names(self) -> None: ...
```
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import pytest
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, MultiIndex, Index, Timedelta, Period
from pandas._libs import lib

@pytest.fixture(params=[True, False])
def future_stack(request: pytest.FixtureRequest) -> bool:
    ...

class TestDataFrameReshape:
    def test_stack_unstack(self, float_frame: DataFrame, future_stack: bool) -> None:
        ...

    def test_stack_mixed_level(self, future_stack: bool) -> None:
        ...

    def test_unstack_not_consolidated(self) -> None:
        ...

    def test_unstack_fill(self, future_stack: bool) -> None:
        ...

    def test_unstack_fill_frame(self) -> None:
        ...

    def test_unstack_fill_frame_datetime(self) -> None:
        ...

    def test_unstack_fill_frame_timedelta(self) -> None:
        ...

    def test_unstack_fill_frame_period(self) -> None:
        ...

    def test_unstack_fill_frame_categorical(self) -> None:
        ...

    def test_unstack_tuplename_in_multiindex(self) -> None:
        ...

    @pytest.mark.parametrize('unstack_idx, expected_values, expected_index, expected_columns', [
        (('A', 'a'), [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]], 
         MultiIndex.from_tuples([(1, 3), (1, 4), (2, 3), (2, 4)], names=['B', 'C']), 
         MultiIndex.from_tuples([('d', 'a'), ('d', 'b'), ('e', 'a'), ('e', 'b')], names=[None, ('A', 'a')])),
        ((('A', 'a'), 'B'), [[1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 2, 2, 2, 2]], 
         Index([3, 4], name='C'), 
         MultiIndex.from_tuples([('d', 'a', 1), ('d', 'a', 2), ('d', 'b', 1), ('d', 'b', 2), ('e', 'a', 1), ('e', 'a', 2), ('e', 'b', 1), ('e', 'b', 2)], names=[None, ('A', 'a'), 'B']))
    ])
    def test_unstack_mixed_type_name_in_multiindex(self, unstack_idx: Union[Tuple[str, str], Tuple[Tuple[str, str], str]], 
                                                  expected_values: List[List[int]], 
                                                  expected_index: Union[MultiIndex, Index], 
                                                  expected_columns: MultiIndex) -> None:
        ...

    def test_unstack_preserve_dtypes(self) -> None:
        ...

    def test_stack_ints(self, future_stack: bool) -> None:
        ...

    def test_stack_mixed_levels(self, future_stack: bool) -> None:
        ...

    def test_stack_int_level_names(self, future_stack: bool) -> None:
        ...

    def test_unstack_bool(self) -> None:
        ...

    def test_unstack_level_binding(self, future_stack: bool) -> None:
        ...

    def test_unstack_to_series(self, float_frame: DataFrame) -> None:
        ...

    def test_unstack_dtypes(self, using_infer_string: bool) -> None:
        ...

    def test_unstack_dtypes_mixed_date(self, c: np.ndarray, d: np.ndarray) -> None:
        ...

    def test_unstack_non_unique_index_names(self, future_stack: bool) -> None:
        ...

    def test_unstack_unused_levels(self) -> None:
        ...

    @pytest.mark.parametrize('level, idces, col_level, idx_level', [
        (0, [13, 16, 6, 9, 2, 5, 8, 11], [np.nan, 'a', 2], [np.nan, 5, 1]),
        (1, [8, 11, 1, 4, 12, 15, 13, 16], [np.nan, 5, 1], [np.nan, 'a', 2])
    ])
    def test_unstack_unused_levels_mixed_with_nan(self, level: int, idces: List[int], col_level: List[Any], idx_level: List[Any]) -> None:
        ...

    @pytest.mark.parametrize('cols', [['A', 'C'], slice(None)])
    def test_unstack_unused_level(self, cols: Union[List[str], slice]) -> None:
        ...

    def test_unstack_long_index(self) -> None:
        ...

    def test_unstack_multi_level_cols(self) -> None:
        ...

    def test_unstack_multi_level_rows_and_cols(self) -> None:
        ...

    @pytest.mark.parametrize('idx', [('jim', 'joe'), ('joe', 'jim')])
    @pytest.mark.parametrize('lev', list(range(2)))
    def test_unstack_nan_index1(self, idx: Tuple[str, str], lev: int) -> None:
        ...

    @pytest.mark.parametrize('idx', itertools.permutations(['1st', '2nd', '3rd']))
    @pytest.mark.parametrize('lev', list(range(3)))
    @pytest.mark.parametrize('col', ['4th', '5th'])
    def test_unstack_nan_index_repeats(self, idx: Tuple[str, str, str], lev: int, col: str) -> None:
        ...

    def test_unstack_nan_index2(self) -> None:
        ...

    def test_unstack_nan_index3(self) -> None:
        ...

    def test_unstack_nan_index4(self) -> None:
        ...

    def test_unstack_nan_index5(self) -> None:
        ...

    def test_stack_datetime_column_multiIndex(self, future_stack: bool) -> None:
        ...

    @pytest.mark.parametrize('multiindex_columns', [
        [0, 1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 1], [0, 2], [0, 3], [0], [2], [4], [4, 3, 2, 1, 0], [3, 2, 1, 0], [4, 2, 1, 0], [2, 1, 0], [3, 2, 1], [4, 3, 2], [1, 0], [2, 0], [3, 0]
    ])
    @pytest.mark.parametrize('level', (-1, 0, 1, [0, 1], [1, 0]))
    def test_stack_partial_multiIndex(self, multiindex_columns: List[int], level: Union[int, List[int]], future_stack: bool) -> None:
        ...

    def test_stack_full_multiIndex(self, future_stack: bool) -> None:
        ...

    @pytest.mark.parametrize('ordered', [False, True])
    def test_stack_preserve_categorical_dtype(self, ordered: bool, future_stack: bool) -> None:
        ...

    @pytest.mark.parametrize('ordered', [False, True])
    @pytest.mark.parametrize('labels,data', [
        (list('xyz'), [10, 11, 12, 13, 14, 15]),
        (list('zyx'), [14, 15, 12, 13, 10, 11])
    ])
    def test_stack_multi_preserve_categorical_dtype(self, ordered: bool, labels: List[str], data: List[int], future_stack: bool) -> None:
        ...

    def test_stack_preserve_categorical_dtype_values(self, future_stack: bool) -> None:
        ...

    @pytest.mark.parametrize('index', [
        [0, 0, 1, 1], [0, 0, 2, 3], [0, 1, 2, 3]
    ])
    def test_stack_multi_columns_non_unique_index(self, index: List[int], future_stack: bool) -> None:
        ...

    @pytest.mark.parametrize('vals1, vals2, dtype1, dtype2, expected_dtype', [
        ([1, 2], [3.0, 4.0], 'Int64', 'Float64', 'Float64'),
        ([1, 2], ['foo', 'bar'], 'Int64', 'string', 'object')
    ])
    def test_stack_multi_columns_mixed_extension_types(self, vals1: List[int], vals2: List[Any], dtype1: str, dtype2: str, expected_dtype: str, future_stack: bool) -> None:
        ...

    @pytest.mark.parametrize('level', [0, 1])
    def test_unstack_mixed_extension_types(self, level: int) -> None:
        ...

    @pytest.mark.parametrize('level', [0, 'baz'])
    def test_unstack_swaplevel_sortlevel(self, level: Union[int, str]) -> None:
        ...

@pytest.mark.parametrize('dtype', ['float64', 'Float64'])
def test_unstack_sort_false(frame_or_series: Union[DataFrame, Series], dtype: str) -> None:
    ...

def test_unstack_fill_frame_object(self) -> None:
    ...

def test_unstack_timezone_aware_values(self) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_timezone_aware_values(self, future_stack: bool) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
@pytest.mark.parametrize('dropna', [True, False, lib.no_default])
def test_stack_empty_frame(self, dropna: Union[bool, Any], future_stack: bool) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
@pytest.mark.parametrize('dropna', [True, False, lib.no_default])
def test_stack_empty_level(self, dropna: Union[bool, Any], future_stack: bool, int_frame: DataFrame) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
@pytest.mark.parametrize('dropna', [True, False, lib.no_default])
@pytest.mark.parametrize('fill_value', [None, 0])
def test_stack_unstack_empty_frame(self, dropna: Union[bool, Any], fill_value: Optional[int], future_stack: bool) -> None:
    ...

def test_unstack_single_index_series(self) -> None:
    ...

def test_unstacking_multi_index_df(self) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_positional_level_duplicate_column_names(self, future_stack: bool) -> None:
    ...

def test_unstack_non_slice_like_blocks(self) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_sort_false(self, future_stack: bool) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_sort_false_multi_level(self, future_stack: bool) -> None:
    ...

def test_unstack_period_series(self) -> None:
    ...

def test_unstack_period_frame(self) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_dropna(self, future_stack: bool) -> None:
    ...

def test_unstack_sparse_keyspace(self) -> None:
    ...

@pytest.mark.slow
def test_unstack_number_of_levels_larger_than_int32(self, performance_warning: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
@pytest.mark.parametrize('levels', itertools.chain.from_iterable((itertools.product(itertools.permutations([0, 1, 2], width), repeat=2) for width in [2, 3])))
@pytest.mark.parametrize('stack_lev', range(2))
def test_stack_order_with_unsorted_levels(self, levels: Tuple[Tuple[int, ...], Tuple[int, ...]], stack_lev: int, sort: bool, future_stack: bool) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_order_with_unsorted_levels_multi_row(self, future_stack: bool) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_order_with_unsorted_levels_multi_row_2(self, future_stack: bool) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_unstack_unordered_multiindex(self, future_stack: bool) -> None:
    ...

def test_unstack_preserve_types(self, multiindex_year_month_day_dataframe_random_data: DataFrame, using_infer_string: bool) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_unstack_group_index_overflow(self, future_stack: bool) -> None:
    ...

def test_unstack_with_missing_int_cast_to_float(self) -> None:
    ...

def test_unstack_with_level_has_nan(self) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_nan_in_multiindex_columns(self, future_stack: bool) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_multi_level_stack_categorical(self, future_stack: bool) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_nan_level(self, future_stack: bool) -> None:
    ...

def test_unstack_categorical_columns(self) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_unsorted(self, future_stack: bool) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_nullable_dtype(self, future_stack: bool) -> None:
    ...

def test_unstack_mixed_level_names(self) -> None:
    ...

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_tuple_columns(self, future_stack: bool) -> None:
    ...

@pytest.mark.parametrize('dtype, na_value', [
    ('float64', np.nan),
    ('Float64', np.nan),
    ('Float64', pd.NA),
    ('Int64', pd.NA)
])
@pytest.mark.parametrize('test_multiindex', [True, False])
def test_stack_preserves_na(self, dtype: str, na_value: Any, test_multiindex: bool) -> None:
    ...
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    Index,
    array,
    concat,
    merge,
    NA,
    Int64Dtype,
)
from pandas._testing import tm

@pytest.fixture
def left_right() -> Tuple[DataFrame, DataFrame]:
    ...

class TestSorting:
    @pytest.mark.slow
    def test_int64_overflow(self) -> None:
        ...

    def test_int64_overflow_groupby_large_range(self) -> None:
        ...

    @pytest.mark.parametrize('agg', ['mean', 'median'])
    def test_int64_overflow_groupby_large_df_shuffled(self, agg: str) -> None:
        ...

    @pytest.mark.parametrize('order, na_position, exp', [
        [True, 'last', List[int]],
        [True, 'first', List[int]],
        [False, 'last', List[int]],
        [False, 'first', List[int]],
    ])
    def test_lexsort_indexer(self, order: bool, na_position: str, exp: List[int]) -> None:
        ...

    @pytest.mark.parametrize('ascending, na_position, exp', [
        [True, 'last', List[int]],
        [True, 'first', List[int]],
        [False, 'last', List[int]],
        [False, 'first', List[int]],
    ])
    def test_nargsort(self, ascending: bool, na_position: str, exp: List[int]) -> None:
        ...

class TestMerge:
    def test_int64_overflow_outer_merge(self) -> None:
        ...

    @pytest.mark.slow
    def test_int64_overflow_check_sum_col(self, left_right: Tuple[DataFrame, DataFrame]) -> None:
        ...

    @pytest.mark.slow
    def test_int64_overflow_how_merge(self, left_right: Tuple[DataFrame, DataFrame], join_type: str) -> None:
        ...

    @pytest.mark.slow
    def test_int64_overflow_sort_false_order(self, left_right: Tuple[DataFrame, DataFrame]) -> None:
        ...

    @pytest.mark.slow
    def test_int64_overflow_one_to_many_none_match(self, join_type: str, sort: bool) -> None:
        ...

@pytest.mark.parametrize('codes_list, shape', [
    [[np.ndarray, np.ndarray, np.ndarray], Tuple[int, int, int]],
    [[np.ndarray, np.ndarray], Tuple[int, int]],
])
def test_decons(codes_list: List[np.ndarray], shape: Tuple[int, ...]) -> None:
    ...

class TestSafeSort:
    @pytest.mark.parametrize('arg, exp', [
        [[List[int]], [List[int]]],
        [[np.ndarray], [np.ndarray]],
        [[], []],
    ])
    def test_basic_sort(self, arg: Union[List[int], np.ndarray], exp: Union[List[int], np.ndarray]) -> None:
        ...

    @pytest.mark.parametrize('verify', [True, False])
    @pytest.mark.parametrize('codes, exp_codes', [
        [[List[int]], [List[int]]],
        [[], []],
    ])
    def test_codes(self, verify: bool, codes: List[int], exp_codes: List[int]) -> None:
        ...

    def test_codes_out_of_bound(self) -> None:
        ...

    @pytest.mark.parametrize('codes', [[List[int]]])
    def test_codes_empty_array_out_of_bound(self, codes: List[int]) -> None:
        ...

    def test_mixed_integer(self) -> None:
        ...

    def test_mixed_integer_with_codes(self) -> None:
        ...

    def test_unsortable(self) -> None:
        ...

    @pytest.mark.parametrize('arg, codes, err, msg', [
        [[Any, None, Type, Pattern], [Any, Any, Type, Pattern]],
    ])
    def test_exceptions(self, arg: Any, codes: Any, err: Type[Exception], msg: str) -> None:
        ...

    @pytest.mark.parametrize('arg, exp', [
        [[List[int]], [List[int]]],
        [[List[Union[int, float]]], [List[Union[int, float]]]],
    ])
    def test_extension_array(self, arg: List[Union[int, float]], exp: List[Union[int, float]]) -> None:
        ...

    @pytest.mark.parametrize('verify', [True, False])
    def test_extension_array_codes(self, verify: bool) -> None:
        ...

def test_mixed_str_null(nulls_fixture: Any) -> None:
    ...

def test_safe_sort_multiindex() -> None:
    ...
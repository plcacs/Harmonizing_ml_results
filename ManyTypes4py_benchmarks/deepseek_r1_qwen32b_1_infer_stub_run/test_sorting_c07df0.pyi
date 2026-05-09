from collections import defaultdict
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas import NA, DataFrame, MultiIndex, Series, array, concat, merge
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import _decons_group_index, get_group_index, is_int64_overflow_possible, lexsort_indexer, nargsort

@pytest.fixture
def left_right() -> tuple[DataFrame, DataFrame]:
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
        [True, 'last', list[int]],
        [True, 'first', list[int]],
        [False, 'last', list[int]],
        [False, 'first', list[int]]
    ])
    def test_lexsort_indexer(self, order: bool, na_position: str, exp: list[int]) -> None:
        ...

    @pytest.mark.parametrize('ascending, na_position, exp', [
        [True, 'last', list[int]],
        [True, 'first', list[int]],
        [False, 'last', list[int]],
        [False, 'first', list[int]]
    ])
    def test_nargsort(self, ascending: bool, na_position: str, exp: list[int]) -> None:
        ...

class TestMerge:
    def test_int64_overflow_outer_merge(self) -> None:
        ...

    @pytest.mark.slow
    def test_int64_overflow_check_sum_col(self, left_right: tuple[DataFrame, DataFrame]) -> None:
        ...

    @pytest.mark.slow
    def test_int64_overflow_how_merge(self, left_right: tuple[DataFrame, DataFrame], join_type: str) -> None:
        ...

    @pytest.mark.slow
    def test_int64_overflow_sort_false_order(self, left_right: tuple[DataFrame, DataFrame]) -> None:
        ...

    @pytest.mark.slow
    def test_int64_overflow_one_to_many_none_match(self, join_type: str, sort: bool) -> None:
        ...

@pytest.mark.parametrize('codes_list, shape', [
    [[np.ndarray, np.ndarray, np.ndarray], tuple[int, int, int]],
    [[np.ndarray, np.ndarray], tuple[int, int]]
])
def test_decons(codes_list: list[np.ndarray], shape: tuple[int, ...]) -> None:
    ...

class TestSafeSort:
    @pytest.mark.parametrize('arg, exp', [
        [list[int], list[int]],
        [np.ndarray, np.ndarray],
        [[], []]
    ])
    def test_basic_sort(self, arg: Union[list[int], np.ndarray], exp: Union[list[int], np.ndarray]) -> None:
        ...

    @pytest.mark.parametrize('verify', [True, False])
    @pytest.mark.parametrize('codes, exp_codes', [
        [list[int], list[int]],
        [[], []]
    ])
    def test_codes(self, verify: bool, codes: list[int], exp_codes: list[int]) -> None:
        ...

    def test_codes_out_of_bound(self) -> None:
        ...

    @pytest.mark.parametrize('codes', [[-1, -1], [2, -1], [2, 2]])
    def test_codes_empty_array_out_of_bound(self, codes: list[int]) -> None:
        ...

    def test_mixed_integer(self) -> None:
        ...

    def test_mixed_integer_with_codes(self) -> None:
        ...

    def test_unsortable(self) -> None:
        ...

    @pytest.mark.parametrize('arg, codes, err, msg', [
        [int, None, TypeError, str],
        [np.ndarray, int, TypeError, str],
        [np.ndarray, list[int], ValueError, str]
    ])
    def test_exceptions(self, arg: Union[int, np.ndarray], codes: Union[None, int, list[int]], err: type[Exception], msg: str) -> None:
        ...

    @pytest.mark.parametrize('arg, exp', [
        [list[int], list[int]],
        [list[Union[int, float]], list[Union[int, float]]]
    ])
    def test_extension_array(self, arg: list[Union[int, float, np.nan]], exp: list[Union[int, float, np.nan]]) -> None:
        ...

    @pytest.mark.parametrize('verify', [True, False])
    def test_extension_array_codes(self, verify: bool) -> None:
        ...

def test_mixed_str_null(nulls_fixture: object) -> None:
    ...

def test_safe_sort_multiindex() -> None:
    ...
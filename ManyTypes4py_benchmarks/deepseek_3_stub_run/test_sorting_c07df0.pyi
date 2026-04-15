from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, Literal, Optional, Union, overload

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, MultiIndex, Series, NA
from pandas.core.arrays import ExtensionArray
from pandas.core.sorting import _decons_group_index, get_group_index, is_int64_overflow_possible, lexsort_indexer, nargsort

@pytest.fixture
def left_right() -> tuple[DataFrame, DataFrame]: ...

class TestSorting:
    @pytest.mark.slow
    def test_int64_overflow(self) -> None: ...
    
    def test_int64_overflow_groupby_large_range(self) -> None: ...
    
    @pytest.mark.parametrize('agg', ['mean', 'median'])
    def test_int64_overflow_groupby_large_df_shuffled(self, agg: Literal['mean', 'median']) -> None: ...
    
    @pytest.mark.parametrize('order, na_position, exp', [[True, 'last', list(range(5, 105)) + list(range(5)) + list(range(105, 110))], [True, 'first', list(range(5)) + list(range(105, 110)) + list(range(5, 105))], [False, 'last', list(range(104, 4, -1)) + list(range(5)) + list(range(105, 110))], [False, 'first', list(range(5)) + list(range(105, 110)) + list(range(104, 4, -1))]])
    def test_lexsort_indexer(self, order: bool, na_position: Literal['first', 'last'], exp: list[int]) -> None: ...
    
    @pytest.mark.parametrize('ascending, na_position, exp', [[True, 'last', list(range(5, 105)) + list(range(5)) + list(range(105, 110))], [True, 'first', list(range(5)) + list(range(105, 110)) + list(range(5, 105))], [False, 'last', list(range(104, 4, -1)) + list(range(5)) + list(range(105, 110))], [False, 'first', list(range(5)) + list(range(105, 110)) + list(range(104, 4, -1))]])
    def test_nargsort(self, ascending: bool, na_position: Literal['first', 'last'], exp: list[int]) -> None: ...

class TestMerge:
    def test_int64_overflow_outer_merge(self) -> None: ...
    
    @pytest.mark.slow
    def test_int64_overflow_check_sum_col(self, left_right: tuple[DataFrame, DataFrame]) -> None: ...
    
    @pytest.mark.slow
    def test_int64_overflow_how_merge(self, left_right: tuple[DataFrame, DataFrame], join_type: str) -> None: ...
    
    @pytest.mark.slow
    def test_int64_overflow_sort_false_order(self, left_right: tuple[DataFrame, DataFrame]) -> None: ...
    
    @pytest.mark.slow
    def test_int64_overflow_one_to_many_none_match(self, join_type: str, sort: bool) -> None: ...

@pytest.mark.parametrize('codes_list, shape', [[[np.tile([0, 1, 2, 3, 0, 1, 2, 3], 100).astype(np.int64), np.tile([0, 2, 4, 3, 0, 1, 2, 3], 100).astype(np.int64), np.tile([5, 1, 0, 2, 3, 0, 5, 4], 100).astype(np.int64)], (4, 5, 6)], [[np.tile(np.arange(10000, dtype=np.int64), 5), np.tile(np.arange(10000, dtype=np.int64), 5)], (10000, 10000)]])
def test_decons(codes_list: list[np.ndarray], shape: tuple[int, ...]) -> None: ...

class TestSafeSort:
    @pytest.mark.parametrize('arg, exp', [[[3, 1, 2, 0, 4], [0, 1, 2, 3, 4]], [np.array(list('baaacb'), dtype=object), np.array(list('aaabbc'), dtype=object)], [[], []]])
    def test_basic_sort(self, arg: Union[list[int], np.ndarray], exp: Union[list[int], np.ndarray]) -> None: ...
    
    @pytest.mark.parametrize('verify', [True, False])
    @pytest.mark.parametrize('codes, exp_codes', [[[0, 1, 1, 2, 3, 0, -1, 4], [3, 1, 1, 2, 0, 3, -1, 4]], [[], []]])
    def test_codes(self, verify: bool, codes: list[int], exp_codes: list[int]) -> None: ...
    
    def test_codes_out_of_bound(self) -> None: ...
    
    @pytest.mark.parametrize('codes', [[-1, -1], [2, -1], [2, 2]])
    def test_codes_empty_array_out_of_bound(self, codes: list[int]) -> None: ...
    
    def test_mixed_integer(self) -> None: ...
    
    def test_mixed_integer_with_codes(self) -> None: ...
    
    def test_unsortable(self) -> None: ...
    
    @pytest.mark.parametrize('arg, codes, err, msg', [[1, None, TypeError, 'Only np.ndarray, ExtensionArray, and Index'], [np.array([0, 1, 2]), 1, TypeError, 'Only list-like objects or None'], [np.array([0, 1, 2, 1]), [0, 1], ValueError, 'values should be unique']])
    def test_exceptions(self, arg: Any, codes: Any, err: type[Exception], msg: str) -> None: ...
    
    @pytest.mark.parametrize('arg, exp', [[[1, 3, 2], [1, 2, 3]], [[1, 3, np.nan, 2], [1, 2, 3, np.nan]]])
    def test_extension_array(self, arg: list[Union[int, float]], exp: list[Union[int, float]]) -> None: ...
    
    @pytest.mark.parametrize('verify', [True, False])
    def test_extension_array_codes(self, verify: bool) -> None: ...

def test_mixed_str_null(nulls_fixture: Any) -> None: ...

def test_safe_sort_multiindex() -> None: ...

@overload
def safe_sort(
    values: np.ndarray,
    codes: None = ...,
    use_na_sentinel: bool = ...,
    verify: bool = ...,
) -> np.ndarray: ...
@overload
def safe_sort(
    values: np.ndarray,
    codes: Sequence[int],
    use_na_sentinel: bool = ...,
    verify: bool = ...,
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def safe_sort(
    values: ExtensionArray,
    codes: None = ...,
    use_na_sentinel: bool = ...,
    verify: bool = ...,
) -> ExtensionArray: ...
@overload
def safe_sort(
    values: ExtensionArray,
    codes: Sequence[int],
    use_na_sentinel: bool = ...,
    verify: bool = ...,
) -> tuple[ExtensionArray, np.ndarray]: ...
@overload
def safe_sort(
    values: pd.Index,
    codes: None = ...,
    use_na_sentinel: bool = ...,
    verify: bool = ...,
) -> pd.Index: ...
@overload
def safe_sort(
    values: pd.Index,
    codes: Sequence[int],
    use_na_sentinel: bool = ...,
    verify: bool = ...,
) -> tuple[pd.Index, np.ndarray]: ...
def safe_sort(
    values: Union[np.ndarray, ExtensionArray, pd.Index],
    codes: Optional[Sequence[int]] = ...,
    use_na_sentinel: bool = ...,
    verify: bool = ...,
) -> Union[
    np.ndarray,
    ExtensionArray,
    pd.Index,
    tuple[np.ndarray, np.ndarray],
    tuple[ExtensionArray, np.ndarray],
    tuple[pd.Index, np.ndarray],
]: ...
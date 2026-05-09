import numpy as np
import pytest
import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    NaT,
    Timestamp,
    date_range,
    Index,
)
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

class TestDataFrameSortValues:
    @pytest.mark.parametrize('dtype', [np.uint8, bool])
    def test_sort_values_sparse_no_warning(self, dtype: Union[np.uint8, bool]) -> None:
        ...

    def test_sort_values(self) -> None:
        ...

    def test_sort_values_by_empty_list(self) -> None:
        ...

    def test_sort_values_inplace(self) -> None:
        ...

    def test_sort_values_multicolumn(self) -> None:
        ...

    def test_sort_values_multicolumn_uint64(self) -> None:
        ...

    def test_sort_values_nan(self) -> None:
        ...

    def test_sort_values_stable_descending_sort(self) -> None:
        ...

    @pytest.mark.parametrize('expected_idx_non_na, ascending', [[[List[int]], [List[bool]]], [[List[int]], [List[bool]]]])
    @pytest.mark.parametrize('na_position', ['first', 'last'])
    def test_sort_values_stable_multicolumn_sort(self, expected_idx_non_na: List[int], ascending: List[bool], na_position: str) -> None:
        ...

    def test_sort_values_stable_categorial(self) -> None:
        ...

    def test_sort_values_datetimes(self) -> None:
        ...

    def test_sort_values_frame_column_inplace_sort_exception(self, float_frame: DataFrame) -> None:
        ...

    def test_sort_values_nat_values_in_int_column(self) -> None:
        ...

    def test_sort_nat(self) -> None:
        ...

    def test_sort_values_na_position_with_categories(self) -> None:
        ...

    def test_sort_values_nat(self) -> None:
        ...

    def test_sort_values_na_position_with_categories_raises(self) -> None:
        ...

    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('original_dict, sorted_dict, ignore_index, output_index', [([Dict[str, List[int]]], [Dict[str, List[int]]], bool, [List[int]]), ([Dict[str, List[int]]], [Dict[str, List[int]]], bool, [List[int]])])
    def test_sort_values_ignore_index(self, inplace: bool, original_dict: Dict[str, List[int]], sorted_dict: Dict[str, List[int]], ignore_index: bool, output_index: List[int]) -> None:
        ...

    def test_sort_values_nat_na_position_default(self) -> None:
        ...

    def test_sort_values_item_cache(self) -> None:
        ...

    def test_sort_values_reshaping(self) -> None:
        ...

    def test_sort_values_no_by_inplace(self) -> None:
        ...

    def test_sort_values_no_op_reset_index(self) -> None:
        ...

class TestDataFrameSortKey:
    def test_sort_values_inplace_key(self, sort_by_key: Callable) -> None:
        ...

    def test_sort_values_key(self) -> None:
        ...

    def test_sort_values_by_key(self) -> None:
        ...

    def test_sort_values_by_key_by_name(self) -> None:
        ...

    def test_sort_values_key_string(self) -> None:
        ...

    def test_sort_values_key_empty(self, sort_by_key: Callable) -> None:
        ...

    def test_changes_length_raises(self) -> None:
        ...

    def test_sort_values_key_axes(self) -> None:
        ...

    def test_sort_values_key_dict_axis(self) -> None:
        ...

    @pytest.mark.parametrize('ordered', [True, False])
    def test_sort_values_key_casts_to_categorical(self, ordered: bool) -> None:
        ...

@pytest.fixture
def df_none() -> DataFrame:
    ...

@pytest.fixture
def df_idx(request: Any, df_none: DataFrame) -> DataFrame:
    ...

@pytest.fixture
def sort_names(request: Any) -> Union[str, List[Union[str, Tuple[str, int]]]]:
    ...
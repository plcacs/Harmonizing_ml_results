import inspect
import operator
import numpy as np
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort

class BaseMethodsTests:
    def test_hash_pandas_object(self, data: pd.Series) -> None:
    
    def test_value_counts_default_dropna(self, data: pd.Series) -> None:
    
    def test_value_counts(self, all_data: pd.Series, dropna: bool) -> None:
    
    def test_value_counts_with_normalize(self, data: pd.Series) -> None:
    
    def test_count(self, data_missing: pd.Series) -> None:
    
    def test_series_count(self, data_missing: pd.Series) -> None:
    
    def test_apply_simple_series(self, data: pd.Series) -> None:
    
    def test_map(self, data_missing: pd.Series, na_action: str) -> None:
    
    def test_argsort(self, data_for_sorting: pd.Series) -> None:
    
    def test_argsort_missing_array(self, data_missing_for_sorting: pd.Series) -> None:
    
    def test_argsort_missing(self, data_missing_for_sorting: pd.Series) -> None:
    
    def test_argmin_argmax(self, data_for_sorting: pd.Series, data_missing_for_sorting: pd.Series, na_value) -> None:
    
    def test_argmin_argmax_empty_array(self, method: str, data: pd.Series) -> None:
    
    def test_argmin_argmax_all_na(self, method: str, data: pd.Series, na_value) -> None:
    
    def test_argreduce_series(self, data_missing_for_sorting: pd.Series, op_name: str, skipna: bool, expected: int) -> None:
    
    def test_argmax_argmin_no_skipna_notimplemented(self, data_missing_for_sorting: pd.Series) -> None:
    
    def test_nargsort(self, data_missing_for_sorting: pd.Series, na_position: str, expected: np.ndarray) -> None:
    
    def test_sort_values(self, data_for_sorting: pd.Series, ascending: bool, sort_by_key) -> None:
    
    def test_sort_values_missing(self, data_missing_for_sorting: pd.Series, ascending: bool, sort_by_key) -> None:
    
    def test_sort_values_frame(self, data_for_sorting: pd.Series, ascending: bool) -> None:
    
    def test_duplicated(self, data: pd.Series, keep: str) -> None:
    
    def test_unique(self, data: pd.Series, box, method) -> None:
    
    def test_factorize(self, data_for_grouping: pd.Series) -> None:
    
    def test_factorize_equivalence(self, data_for_grouping: pd.Series) -> None:
    
    def test_factorize_empty(self, data: pd.Series) -> None:
    
    def test_fillna_limit_frame(self, data_missing: pd.Series) -> None:
    
    def test_fillna_limit_series(self, data_missing: pd.Series) -> None:
    
    def test_fillna_copy_frame(self, data_missing: pd.Series) -> None:
    
    def test_fillna_copy_series(self, data_missing: pd.Series) -> None:
    
    def test_fillna_length_mismatch(self, data_missing: pd.Series) -> None:
    
    _combine_le_expected_dtype = NumpyEADtype('bool')
    
    def test_combine_le(self, data_repeated) -> None:
    
    def test_combine_add(self, data_repeated) -> None:
    
    def test_combine_first(self, data: pd.Series) -> None:
    
    def test_container_shift(self, data: pd.Series, frame: bool, periods: int, indices: List[int]) -> None:
    
    def test_shift_0_periods(self, data: pd.Series) -> None:
    
    def test_diff(self, data: pd.Series, periods: int) -> None:
    
    def test_shift_non_empty_array(self, data: pd.Series, periods: int, indices: List[int]) -> None:
    
    def test_shift_empty_array(self, data: pd.Series, periods: int) -> None:
    
    def test_shift_zero_copies(self, data: pd.Series) -> None:
    
    def test_shift_fill_value(self, data: pd.Series) -> None:
    
    def test_not_hashable(self, data: pd.Series) -> None:
    
    def test_hash_pandas_object_works(self, data: pd.Series, as_frame: bool) -> None:
    
    def test_searchsorted(self, data_for_sorting: pd.Series, as_series: bool) -> None:
    
    def _test_searchsorted_bool_dtypes(self, data_for_sorting: pd.Series, as_series: bool) -> None:
    
    def test_where_series(self, data: pd.Series, na_value, as_frame: bool) -> None:
    
    def test_repeat(self, data: pd.Series, repeats, as_series: bool, use_numpy: bool) -> None:
    
    def test_repeat_raises(self, data: pd.Series, repeats, kwargs: Dict, error, msg, use_numpy: bool) -> None:
    
    def test_delete(self, data: pd.Series) -> None:
    
    def test_insert(self, data: pd.Series) -> None:
    
    def test_insert_invalid(self, data: pd.Series, invalid_scalar) -> None:
    
    def test_insert_invalid_loc(self, data: pd.Series) -> None:
    
    def test_equals(self, data: pd.Series, na_value, as_series: bool, box) -> None:
    
    def test_equals_same_data_different_object(self, data: pd.Series) -> None:

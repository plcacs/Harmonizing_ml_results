from datetime import date, timedelta, timezone
from decimal import Decimal
from operator import add, and_, eq, floordiv, ge, gt, le, lt, mod, mul, ne, or_, pow, rshift, sub, truediv, xor
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pytest
from pandas import (
    Categorical,
    DatetimeTZDtype,
    Index,
    Series,
    Timedelta,
    bdate_range,
    date_range,
    isna,
)
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency

class TestSeriesFlexArithmetic:
    def test_flex_method_equivalence(
        self, opname: str, ts: Tuple[Callable, Callable, bool]
    ) -> None:
        ...
    
    def test_flex_method_subclass_metadata_preservation(
        self, all_arithmetic_operators: str
    ) -> None:
        ...
    
    def test_flex_add_scalar_fill_value(self) -> None:
        ...

class TestSeriesArithmetic:
    def test_add_series_with_period_index(self) -> None:
        ...
    
    def test_string_addition(
        self, target_add: str, input_value: List[str], expected_value: List[str]
    ) -> None:
        ...
    
    def test_divmod(self) -> None:
        ...
    
    def test_series_integer_mod(self, index: Optional[List[int]]) -> None:
        ...
    
    def test_add_with_duplicate_index(self) -> None:
        ...
    
    def test_add_na_handling(self) -> None:
        ...
    
    def test_add_corner_cases(self, datetime_series: Series) -> None:
        ...
    
    def test_add_float_plus_int(self, datetime_series: Series) -> None:
        ...
    
    def test_mul_empty_int_corner_case(self) -> None:
        ...
    
    def test_sub_datetimelike_align(self) -> None:
        ...
    
    def test_alignment_doesnt_change_tz(self) -> None:
        ...
    
    def test_alignment_categorical(self) -> None:
        ...
    
    def test_arithmetic_with_duplicate_index(self) -> None:
        ...
    
    def test_masked_and_non_masked_propagate_na(self) -> None:
        ...
    
    def test_mask_div_propagate_na_for_non_na_dtype(self) -> None:
        ...
    
    def test_add_list_to_masked_array(
        self, val: Union[int, float], dtype: str
    ) -> None:
        ...
    
    def test_add_list_to_masked_array_boolean(
        self, request: pytest.fixture
    ) -> None:
        ...

class TestSeriesFlexComparison:
    def test_comparison_flex_basic(
        self, axis: Union[int, str, None], comparison_op: Callable
    ) -> None:
        ...
    
    def test_comparison_bad_axis(self, comparison_op: Callable) -> None:
        ...
    
    def test_comparison_flex_alignment(
        self, values: List[bool], op: str
    ) -> None:
        ...
    
    def test_comparison_flex_alignment_fill(
        self, values: List[bool], op: str, fill_value: int
    ) -> None:
        ...

class TestSeriesComparison:
    def test_comparison_different_length(self) -> None:
        ...
    
    def test_ser_flex_cmp_return_dtypes(self, opname: str) -> None:
        ...
    
    def test_ser_flex_cmp_return_dtypes_empty(self, opname: str) -> None:
        ...
    
    def test_ser_cmp_result_names(
        self, names: Tuple[Optional[str], Optional[str], Optional[str]], comparison_op: Callable
    ) -> None:
        ...
    
    def test_comparisons(self) -> None:
        ...
    
    def test_categorical_comparisons(self) -> None:
        ...
    
    def test_unequal_categorical_comparison_raises_type_error(self) -> None:
        ...
    
    def test_comparison_tuples(self) -> None:
        ...
    
    def test_comparison_frozenset(self) -> None:
        ...
    
    def test_comparison_operators_with_nas(self, comparison_op: Callable) -> None:
        ...
    
    def test_ne(self) -> None:
        ...
    
    def test_comp_ops_df_compat(
        self, right_data: List[int], frame_or_series: Union[type, Callable]
    ) -> None:
        ...
    
    def test_compare_series_interval_keyword(self) -> None:
        ...

class TestTimeSeriesArithmetic:
    def test_series_add_tz_mismatch_converts_to_utc(self) -> None:
        ...
    
    def test_series_add_aware_naive_raises(self) -> None:
        ...
    
    def test_datetime_understood(self, unit: str) -> None:
        ...
    
    def test_align_date_objects_with_datetimeindex(self) -> None:
        ...

class TestNamePreservation:
    def test_series_ops_name_retention(
        self, flex: bool, box: Callable, names: Tuple[str, str, str], all_binary_operators: Callable
    ) -> None:
        ...
    
    def test_binop_maybe_preserve_name(self, datetime_series: Series) -> None:
        ...
    
    def test_scalarop_preserve_name(self, datetime_series: Series) -> None:
        ...

class TestInplaceOperations:
    def test_series_inplace_ops(
        self, dtype1: str, dtype2: str, dtype_expected: str, dtype_mul: str
    ) -> None:
        ...

def test_none_comparison(
    request: pytest.fixture, series_with_simple_index: Series
) -> None:
    ...

def test_series_varied_multiindex_alignment(self) -> None:
    ...

def test_rmod_consistent_large_series(self) -> None:
    ...
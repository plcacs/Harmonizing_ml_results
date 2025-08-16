from datetime import date, timedelta, timezone
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import Series, Timedelta, Index, Categorical, DatetimeTZDtype, bdate_range, date_range, isna
from pandas._libs.tslibs import IncompatibleFrequency

def _permute(obj: Series) -> Series:
    return obj.take(np.random.default_rng(2).permutation(len(obj)))

class TestSeriesFlexArithmetic:

    def test_flex_method_equivalence(self, opname: str, ts: tuple) -> None:
        ...

    def test_flex_method_subclass_metadata_preservation(self, all_arithmetic_operators: str) -> None:
        ...

    def test_flex_add_scalar_fill_value(self) -> None:
        ...

    def test_operators_combine(self, op, equiv_op, fv) -> None:
        ...

class TestSeriesArithmetic:

    def test_add_series_with_period_index(self) -> None:
        ...

    def test_string_addition(self, target_add: str, input_value: list, expected_value: list) -> None:
        ...

    def test_divmod(self) -> None:
        ...

    def test_series_integer_mod(self, index) -> None:
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

    def test_add_list_to_masked_array(self, val: int, dtype: str) -> None:
        ...

    def test_add_list_to_masked_array_boolean(self, request) -> None:
        ...

class TestSeriesFlexComparison:

    def test_comparison_flex_basic(self, axis, comparison_op) -> None:
        ...

    def test_comparison_bad_axis(self, comparison_op) -> None:
        ...

    def test_comparison_flex_alignment(self, values: list, op: str) -> None:
        ...

    def test_comparison_flex_alignment_fill(self, values: list, op: str, fill_value: int) -> None:
        ...

class TestSeriesComparison:

    def test_comparison_different_length(self) -> None:
        ...

    def test_categorical_comparisons(self) -> None:
        ...

    def test_unequal_categorical_comparison_raises_type_error(self) -> None:
        ...

    def test_comparison_tuples(self) -> None:
        ...

    def test_comparison_frozenset(self) -> None:
        ...

    def test_comparison_operators_with_nas(self, comparison_op) -> None:
        ...

    def test_ne(self) -> None:
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

    def test_none_comparison(request, series_with_simple_index) -> None:
        ...

    def test_series_varied_multiindex_alignment() -> None:
        ...

    def test_rmod_consistent_large_series() -> None:
        ...

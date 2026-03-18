```python
import builtins
import datetime as dt
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex, Series, Timestamp
from pandas._testing import tm

def test_basic_aggregations(dtype: str) -> None: ...

@overload
def test_groupby_bool_aggs(
    skipna: bool,
    all_boolean_reductions: str,
    vals: list[str]
) -> None: ...
@overload
def test_groupby_bool_aggs(
    skipna: bool,
    all_boolean_reductions: str,
    vals: list[int]
) -> None: ...
@overload
def test_groupby_bool_aggs(
    skipna: bool,
    all_boolean_reductions: str,
    vals: list[float]
) -> None: ...
@overload
def test_groupby_bool_aggs(
    skipna: bool,
    all_boolean_reductions: str,
    vals: list[bool]
) -> None: ...
@overload
def test_groupby_bool_aggs(
    skipna: bool,
    all_boolean_reductions: str,
    vals: list[Any]
) -> None: ...
def test_groupby_bool_aggs(
    skipna: bool,
    all_boolean_reductions: str,
    vals: Any
) -> None: ...

def test_any() -> None: ...

def test_bool_aggs_dup_column_labels(all_boolean_reductions: str) -> None: ...

def test_masked_kleene_logic(
    all_boolean_reductions: str,
    skipna: bool,
    data: list[Any]
) -> None: ...

def test_masked_mixed_types(
    dtype1: str,
    dtype2: str,
    exp_col1: Any,
    exp_col2: Any
) -> None: ...

def test_masked_bool_aggs_skipna(
    all_boolean_reductions: str,
    dtype: str,
    skipna: bool,
    frame_or_series: Any
) -> None: ...

def test_object_type_missing_vals(
    bool_agg_func: str,
    data: list[Any],
    expected_res: bool,
    frame_or_series: Any
) -> None: ...

def test_object_NA_raises_with_skipna_false(all_boolean_reductions: str) -> None: ...

def test_empty(frame_or_series: Any, all_boolean_reductions: str) -> None: ...

def test_idxmin_idxmax_extremes(how: str, any_real_numpy_dtype: Any) -> None: ...

def test_idxmin_idxmax_extremes_skipna(
    skipna: bool,
    how: str,
    float_numpy_dtype: Any
) -> None: ...

def test_idxmin_idxmax_returns_int_types(
    func: str,
    values: dict[str, list[int]],
    numeric_only: bool
) -> None: ...

def test_groupby_non_arithmetic_agg_int_like_precision(
    method: str,
    data: tuple[Any, Any]
) -> None: ...

def test_first_last_skipna(
    any_real_nullable_dtype: Any,
    sort: bool,
    skipna: bool,
    how: str
) -> None: ...

def test_groupby_mean_no_overflow() -> None: ...

def test_mean_on_timedelta() -> None: ...

def test_mean_skipna(
    values: Any,
    dtype: str,
    result_dtype: str,
    skipna: bool
) -> None: ...

def test_sum_skipna(values: Any, dtype: str, skipna: bool) -> None: ...

def test_sum_skipna_object(skipna: bool) -> None: ...

def test_multifunc_skipna(
    func: str,
    values: Any,
    dtype: str,
    result_dtype: str,
    skipna: bool
) -> None: ...

def test_cython_median() -> None: ...

def test_median_empty_bins(observed: bool) -> None: ...

def test_max_min_non_numeric() -> None: ...

def test_max_min_object_multiple_columns(using_infer_string: bool) -> None: ...

def test_min_date_with_nans() -> None: ...

def test_max_inat() -> None: ...

def test_max_inat_not_all_na() -> None: ...

def test_groupby_aggregate_period_column(func: str) -> None: ...

def test_groupby_aggregate_period_frame(func: str) -> None: ...

def test_aggregate_numeric_object_dtype() -> None: ...

def test_aggregate_categorical_lost_index(func: str) -> None: ...

def test_groupby_min_max_nullable(dtype: str) -> None: ...

def test_min_max_nullable_uint64_empty_group() -> None: ...

def test_groupby_min_max_categorical(func: str) -> None: ...

def test_min_empty_string_dtype(func: str, string_dtype_no_object: Any) -> None: ...

def test_max_nan_bug() -> None: ...

def test_series_groupby_nunique(
    sort: bool,
    dropna: bool,
    as_index: bool,
    with_nan: bool,
    keys: list[str]
) -> None: ...

def test_nunique() -> None: ...

def test_nunique_with_object() -> None: ...

def test_nunique_with_empty_series() -> None: ...

def test_nunique_with_timegrouper() -> None: ...

def test_nunique_with_NaT(
    key: list[str],
    data: list[Any],
    dropna: bool,
    expected: Series
) -> None: ...

def test_nunique_preserves_column_level_names() -> None: ...

def test_nunique_transform_with_datetime() -> None: ...

def test_empty_categorical(observed: bool) -> None: ...

def test_intercept_builtin_sum() -> None: ...

def test_groupby_sum_mincount_boolean(min_count: int) -> None: ...

def test_groupby_sum_below_mincount_nullable_integer() -> None: ...

def test_groupby_sum_timedelta_with_nat() -> None: ...

def test_groupby_non_arithmetic_agg_types(
    dtype: str,
    method: str,
    data: dict[str, Any]
) -> None: ...

def test_ops_general(op: str, targop: Any) -> None: ...

def test_apply_to_nullable_integer_returns_float(
    values: dict[str, list[Any]],
    function: str
) -> None: ...

def test_regression_allowlist_methods(
    op: str,
    skipna: bool,
    sort: bool
) -> None: ...

def test_groupby_prod_with_int64_dtype() -> None: ...

def test_groupby_std_datetimelike() -> None: ...
```
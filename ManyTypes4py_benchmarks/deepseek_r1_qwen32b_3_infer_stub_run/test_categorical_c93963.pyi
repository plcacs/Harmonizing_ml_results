from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    qcut,
)
from pandas.api.typing import SeriesGroupBy

_results_for_groupbys_with_missing_categories: dict[str, Any]

def cartesian_product_for_groupers(
    result: Union[DataFrame, Series],
    args: List[Categorical],
    names: List[str],
    fill_value: Any = np.nan,
) -> Union[DataFrame, Series]:
    ...

def test_apply_use_categorical_name(df: DataFrame) -> None:
    ...

def test_basic(df: DataFrame) -> None:
    ...

def test_basic_single_grouper() -> None:
    ...

def test_basic_string(using_infer_string: bool) -> None:
    ...

def test_basic_monotonic() -> None:
    ...

def test_basic_non_monotonic() -> None:
    ...

def test_basic_cut_grouping() -> None:
    ...

def test_more_basic() -> None:
    ...

def test_level_get_group(observed: bool) -> None:
    ...

def test_sorting_with_different_categoricals() -> None:
    ...

def test_apply(ordered: bool) -> None:
    ...

def test_observed(request: Any, using_infer_string: bool, observed: bool) -> None:
    ...

def test_observed_single_column(observed: bool) -> None:
    ...

def test_observed_two_columns(observed: bool) -> None:
    ...

def test_observed_with_as_index(observed: bool) -> None:
    ...

def test_observed_codes_remap(observed: bool) -> None:
    ...

def test_observed_perf() -> None:
    ...

def test_observed_groups(observed: bool) -> None:
    ...

def test_unobserved_in_index(
    keys: Union[str, List[str]],
    expected_values: List[int],
    expected_index_levels: Union[CategoricalIndex, List[CategoricalIndex]],
    test_series: bool,
) -> None:
    ...

def test_observed_groups_with_nan(observed: bool) -> None:
    ...

def test_observed_nth() -> None:
    ...

def test_dataframe_categorical_with_nan(observed: bool) -> None:
    ...

def test_dataframe_categorical_ordered_observed_sort(
    ordered: bool, observed: bool, sort: bool
) -> None:
    ...

def test_datetime() -> None:
    ...

def test_categorical_index() -> None:
    ...

def test_describe_categorical_columns() -> None:
    ...

def test_unstack_categorical() -> None:
    ...

def test_bins_unequal_len() -> None:
    ...

def test_categorical_series(
    series: Series, data: Dict[str, List[int]]
) -> None:
    ...

def test_as_index() -> None:
    ...

def test_preserve_categories() -> None:
    ...

def test_preserve_categories_ordered_false() -> None:
    ...

def test_preserve_categorical_dtype(col: str) -> None:
    ...

def test_preserve_on_ordered_ops(func: str, values: List[str]) -> None:
    ...

def test_categorical_no_compress() -> None:
    ...

def test_categorical_no_compress_string() -> None:
    ...

def test_groupby_empty_with_category() -> None:
    ...

def test_sort(sort: bool) -> None:
    ...

def test_sort2(sort: bool, ordered: bool) -> None:
    ...

def test_sort_datetimelike(sort: bool, ordered: bool) -> None:
    ...

def test_empty_sum() -> None:
    ...

def test_empty_prod() -> None:
    ...

def test_groupby_multiindex_categorical_datetime() -> None:
    ...

def test_shift(fill_value: Any) -> None:
    ...

def test_series_groupby_on_2_categoricals_unobserved(
    reduction_func: str, observed: bool
) -> None:
    ...

def test_series_groupby_on_2_categoricals_unobserved_zeroes_or_nans(
    reduction_func: str, request: Any
) -> None:
    ...

def test_dataframe_groupby_on_2_categoricals_when_observed_is_true(
    reduction_func: str
) -> None:
    ...

def test_dataframe_groupby_on_2_categoricals_when_observed_is_false(
    reduction_func: str, observed: bool
) -> None:
    ...

def test_series_groupby_categorical_aggregation_getitem() -> None:
    ...

def test_groupby_agg_categorical_columns(func: str, expected_values: List[int]) -> None:
    ...

def test_groupby_agg_non_numeric() -> None:
    ...

def test_groupby_first_returned_categorical_instead_of_dataframe(func: str) -> None:
    ...

def test_read_only_category_no_sort() -> None:
    ...

def test_sorted_missing_category_values() -> None:
    ...

def test_agg_cython_category_not_implemented_fallback() -> None:
    ...

def test_aggregate_categorical_with_isnan() -> None:
    ...

def test_categorical_transform() -> None:
    ...

def test_series_groupby_first_on_categorical_col_grouped_on_2_categoricals(
    func: str, observed: bool
) -> None:
    ...

def test_df_groupby_first_on_categorical_col_grouped_on_2_categoricals(
    func: str, observed: bool
) -> None:
    ...

def test_groupby_categorical_indices_unused_categories() -> None:
    ...

def test_groupby_categorical_observed_nunique() -> None:
    ...

def test_groupby_categorical_aggregate_functions() -> None:
    ...

def test_groupby_categorical_dropna(observed: bool, dropna: bool) -> None:
    ...

def test_category_order_reducer(
    request: Any,
    as_index: bool,
    sort: bool,
    observed: bool,
    reduction_func: str,
    index_kind: str,
    ordered: bool,
) -> None:
    ...

def test_category_order_transformer(
    as_index: bool,
    sort: bool,
    observed: bool,
    transformation_func: str,
    index_kind: str,
    ordered: bool,
) -> None:
    ...

def test_category_order_head_tail(
    as_index: bool,
    sort: bool,
    observed: bool,
    method: str,
    index_kind: str,
    ordered: bool,
) -> None:
    ...

def test_category_order_apply(
    as_index: bool,
    sort: bool,
    observed: bool,
    method: str,
    index_kind: str,
    ordered: bool,
) -> None:
    ...

def test_many_categories(
    as_index: bool,
    sort: bool,
    index_kind: str,
    ordered: bool,
) -> None:
    ...

def test_agg_list(
    request: Any,
    as_index: bool,
    observed: bool,
    reduction_func: str,
    test_series: bool,
    keys: List[str],
) -> None:
    ...
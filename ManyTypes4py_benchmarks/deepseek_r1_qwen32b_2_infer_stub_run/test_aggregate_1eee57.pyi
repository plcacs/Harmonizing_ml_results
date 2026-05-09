"""
Stub file for test_aggregate_1eee57 module
"""

from datetime import datetime
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import numpy as np
import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pytest

def test_groupby_agg_no_extra_calls() -> None:
    ...

def test_agg_regression1(tsframe: DataFrame) -> None:
    ...

def test_agg_must_agg(df: DataFrame, msg: str) -> None:
    ...

def test_agg_ser_multi_key(df: DataFrame) -> None:
    ...

def test_agg_with_missing_values() -> None:
    ...

def test_groupby_aggregation_mixed_dtype() -> None:
    ...

def test_agg_apply_corner(ts: Series, tsframe: DataFrame) -> None:
    ...

def test_with_na_groups(any_real_numpy_dtype: str) -> None:
    ...

def test_agg_grouping_is_list_tuple(ts: Series) -> None:
    ...

def test_agg_python_multiindex(multiindex_dataframe_random_data: DataFrame) -> None:
    ...

def test_aggregate_str_func(tsframe: DataFrame, groupbyfunc: Union[Callable[[Any], Any], List[Callable[[Any], Any]]]) -> None:
    ...

def test_std_masked_dtype(any_numeric_ea_dtype: str) -> None:
    ...

def test_agg_str_with_kwarg_axis_1_raises(df: DataFrame, reduction_func: str) -> None:
    ...

def test_aggregate_item_by_item(df: DataFrame) -> None:
    ...

def test_wrap_agg_out(three_group: DataFrame) -> None:
    ...

def test_agg_multiple_functions_maintain_order(df: DataFrame) -> None:
    ...

def test_series_index_name(df: DataFrame) -> None:
    ...

def test_agg_multiple_functions_same_name() -> None:
    ...

def test_agg_multiple_functions_same_name_with_ohlc_present() -> None:
    ...

def test_multiple_functions_tuples_and_non_tuples(df: DataFrame) -> None:
    ...

def test_more_flexible_frame_multi_function(df: DataFrame) -> None:
    ...

def test_multi_function_flexible_mix(df: DataFrame) -> None:
    ...

def test_groupby_agg_coercing_bools() -> None:
    ...

def test_groupby_agg_dict_with_getitem() -> None:
    ...

def test_groupby_agg_dict_dup_columns() -> None:
    ...

def test_bool_agg_dtype(op: Callable[[Any], Any]) -> None:
    ...

def test_callable_result_dtype_frame(
    keys: Union[List[str], Tuple[str, str]],
    agg_index: Union[Index, MultiIndex],
    input_dtype: str,
    result_dtype: str,
    method: str,
) -> None:
    ...

def test_callable_result_dtype_series(
    keys: Union[List[str], Tuple[str, str]],
    agg_index: Union[Index, MultiIndex],
    input: Union[bool, int, float],
    dtype: Union[bool, int, float],
    method: str,
) -> None:
    ...

def test_order_aggregate_multiple_funcs() -> None:
    ...

def test_ohlc_ea_dtypes(any_numeric_ea_dtype: str) -> None:
    ...

def test_uint64_type_handling(dtype: Union[np.int64, np.uint64], how: str) -> None:
    ...

def test_func_duplicates_raises() -> None:
    ...

def test_agg_index_has_complex_internals(index: Union[pd.CategoricalIndex, pd.IntervalIndex, pd.PeriodIndex, MultiIndex]) -> None:
    ...

def test_agg_split_block() -> None:
    ...

def test_agg_split_object_part_datetime() -> None:
    ...

def test_series_named_agg() -> None:
    ...

def test_no_args_raises() -> None:
    ...

def test_series_named_agg_duplicates_no_raises() -> None:
    ...

def test_mangled() -> None:
    ...

def test_named_agg_nametuple(inp: Union[pd.NamedAgg, Tuple[str, str], List[str]]) -> None:
    ...

def test_agg_relabel() -> None:
    ...

def test_agg_relabel_non_identifier() -> None:
    ...

def test_duplicate_no_raises() -> None:
    ...

def test_agg_relabel_with_level() -> None:
    ...

def test_agg_relabel_other_raises() -> None:
    ...

def test_missing_raises() -> None:
    ...

def test_agg_namedtuple() -> None:
    ...

def test_mangled() -> None:
    ...

def test_agg_relabel_multiindex_column(
    agg_col1: Union[Tuple[Tuple[str, str], str], pd.NamedAgg],
    agg_col2: Union[Tuple[Tuple[str, str], str], pd.NamedAgg],
    agg_col3: Union[Tuple[Tuple[str, str], str], pd.NamedAgg],
    agg_result1: List[int],
    agg_result2: List[Union[int, float]],
    agg_result3: List[Union[int, float]],
) -> None:
    ...

def test_agg_relabel_multiindex_raises_not_exist() -> None:
    ...

def test_agg_relabel_multiindex_duplicates() -> None:
    ...

def test_groupby_aggregate_empty_key(kwargs: Dict[str, List[str]]) -> None:
    ...

def test_groupby_aggregate_empty_key_empty_return() -> None:
    ...

def test_groupby_aggregate_empty_with_multiindex_frame() -> None:
    ...

def test_grouby_agg_loses_results_with_as_index_false_relabel() -> None:
    ...

def test_grouby_agg_loses_results_with_as_index_false_relabel_multiindex() -> None:
    ...

def test_groupby_as_index_agg(df: DataFrame) -> None:
    ...

def test_multiindex_custom_func(func: Callable[[Series], Any]) -> None:
    ...

def test_lambda_named_agg(func: Union[Callable[[Series], Any], partial]) -> None:
    ...

def test_aggregate_mixed_types() -> None:
    ...

def test_aggregate_udf_na_extension_type() -> None:
    ...

def test_lambda_mangling() -> None:
    ...

def test_pass_args_kwargs_duplicate_columns(tsframe: DataFrame, as_index: bool) -> None:
    ...

def test_groupby_get_by_index() -> None:
    ...

def test_groupby_single_agg_cat_cols(
    grp_col_dict: Dict[str, Union[str, List[str]]],
    exp_data: Dict[str, List[Union[int, str]]],
) -> None:
    ...

def test_groupby_combined_aggs_cat_cols(
    grp_col_dict: Dict[str, Union[str, List[str]]],
    exp_data: List[Tuple[Union[int, str], ...]],
) -> None:
    ...

def test_nonagg_agg() -> None:
    ...

def test_aggregate_datetime_objects() -> None:
    ...

def test_groupby_index_object_dtype() -> None:
    ...

def test_timeseries_groupby_agg() -> None:
    ...

def test_groupby_agg_precision(any_real_numeric_dtype: str) -> None:
    ...

def test_groupby_aggregate_directory(reduction_func: str) -> None:
    ...

def test_group_mean_timedelta_nat() -> None:
    ...

def test_group_mean_datetime64_nat(input_data: List[str], expected_output: List[str]) -> None:
    ...

def test_groupby_complex(func: str, output: List[complex]) -> None:
    ...

def test_groupby_complex_raises(func: str) -> None:
    ...

def test_agg_of_mode_list(test: List[List[Union[int, str]]], constant: Dict[int, List[Union[int, str]]]) -> None:
    ...

def test_dataframe_groupy_agg_list_like_func_with_args() -> None:
    ...

def test_series_groupy_agg_list_like_func_with_args() -> None:
    ...

def test_agg_groupings_selection() -> None:
    ...

def test_agg_multiple_with_as_index_false_subset_to_a_single_column() -> None:
    ...

def test_agg_with_as_index_false_with_list() -> None:
    ...

def test_groupby_agg_extension_timedelta_cumsum_with_named_aggregation() -> None:
    ...

def test_groupby_aggregation_empty_group() -> None:
    ...

def test_groupby_aggregation_duplicate_columns_single_dict_value() -> None:
    ...

def test_groupby_aggregation_duplicate_columns_multiple_dict_values() -> None:
    ...

def test_groupby_aggregation_duplicate_columns_some_empty_result() -> None:
    ...

def test_groupby_aggregation_multi_index_duplicate_columns() -> None:
    ...

def test_groupby_aggregation_func_list_multi_index_duplicate_columns() -> None:
    ...
from datetime import datetime
from functools import partial
import numpy as np
import pytest
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
    to_datetime,
)
from pandas._testing import tm
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.groupby.grouper import Grouping
from pandas.errors import SpecificationError

def test_groupby_agg_no_extra_calls(df: DataFrame) -> None:
    ...

def test_agg_regression1(tsframe: DataFrame) -> None:
    ...

def test_agg_must_agg(df: DataFrame) -> None:
    ...

def test_agg_ser_multi_key(df: DataFrame) -> None:
    ...

def test_agg_with_missing_values() -> None:
    ...

def test_groupby_aggregation_mixed_dtype() -> None:
    ...

def test_agg_apply_corner(ts: Series, tsframe: DataFrame) -> None:
    ...

def test_with_na_groups(any_real_numpy_dtype: np.dtype) -> None:
    ...

def test_agg_grouping_is_list_tuple(ts: Series) -> None:
    ...

def test_agg_python_multiindex(multiindex_dataframe_random_data: DataFrame) -> None:
    ...

def test_aggregate_str_func(tsframe: DataFrame, groupbyfunc: list[lambda x: int | str]) -> None:
    ...

def test_std_masked_dtype(any_numeric_ea_dtype: np.dtype) -> None:
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

def test_bool_agg_dtype(op: lambda x: DataFrame) -> None:
    ...

def test_callable_result_dtype_frame(
    keys: list[str],
    agg_index: Index,
    input_dtype: str,
    result_dtype: str,
    method: str,
) -> None:
    ...

def test_callable_result_dtype_series(
    keys: list[str],
    agg_index: Index,
    input: bool | int | float,
    dtype: type,
    method: str,
) -> None:
    ...

def test_order_aggregate_multiple_funcs() -> None:
    ...

def test_ohlc_ea_dtypes(any_numeric_ea_dtype: np.dtype) -> None:
    ...

def test_uint64_type_handling(dtype: np.dtype, how: str) -> None:
    ...

def test_func_duplicates_raises() -> None:
    ...

def test_agg_index_has_complex_internals(index: pd.CategoricalIndex | pd.IntervalIndex | pd.PeriodIndex | MultiIndex) -> None:
    ...

def test_agg_split_block() -> None:
    ...

def test_agg_split_object_part_datetime() -> None:
    ...

class TestNamedAggregationSeries:
    def test_series_named_agg(self) -> None:
        ...
    
    def test_no_args_raises(self) -> None:
        ...
    
    def test_series_named_agg_duplicates_no_raises(self) -> None:
        ...
    
    def test_mangled(self) -> None:
        ...
    
    def test_mangled(self) -> None:
        ...

class TestNamedAggregationDataFrame:
    def test_agg_relabel(self) -> None:
        ...
    
    def test_agg_relabel_non_identifier(self) -> None:
        ...
    
    def test_duplicate_no_raises(self) -> None:
        ...
    
    def test_agg_relabel_with_level(self) -> None:
        ...
    
    def test_agg_relabel_other_raises(self) -> None:
        ...
    
    def test_missing_raises(self) -> None:
        ...
    
    def test_agg_namedtuple(self) -> None:
        ...
    
    def test_mangled(self) -> None:
        ...

def test_agg_relabel_multiindex_column(
    agg_col1: tuple[tuple[str, str], str],
    agg_col2: tuple[tuple[str, str], str],
    agg_col3: tuple[tuple[str, str], str],
    agg_result1: list[int],
    agg_result2: list[int],
    agg_result3: list[int],
) -> None:
    ...

def test_agg_relabel_multiindex_raises_not_exist() -> None:
    ...

def test_agg_relabel_multiindex_duplicates() -> None:
    ...

def test_groupby_aggregate_empty_key(kwargs: dict[str, list[str]]) -> None:
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

def test_multiindex_custom_func(func: lambda x: float) -> None:
    ...

def test_lambda_named_agg(func: lambda x: float) -> None:
    ...

def test_aggregate_mixed_types() -> None:
    ...

def test_aggregate_udf_na_extension_type() -> None:
    ...

class TestLambdaMangling:
    def test_basic(self) -> None:
        ...
    
    def test_mangle_series_groupby(self) -> None:
        ...
    
    def test_with_kwargs(self) -> None:
        ...
    
    def test_agg_with_one_lambda(self) -> None:
        ...
    
    def test_agg_multiple_lambda(self) -> None:
        ...

def test_pass_args_kwargs_duplicate_columns(tsframe: DataFrame, as_index: bool) -> None:
    ...

def test_groupby_get_by_index() -> None:
    ...

def test_groupby_single_agg_cat_cols(
    grp_col_dict: dict[str, str],
    exp_data: dict[str, list[int | str]]
) -> None:
    ...

def test_groupby_combined_aggs_cat_cols(
    grp_col_dict: dict[str, list[str]],
    exp_data: list[tuple[int, str]]
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

def test_groupby_agg_precision(any_real_numeric_dtype: np.dtype) -> None:
    ...

def test_groupby_aggregate_directory(reduction_func: str) -> None:
    ...

def test_group_mean_timedelta_nat() -> None:
    ...

def test_group_mean_datetime64_nat(
    input_data: list[str],
    expected_output: list[str]
) -> None:
    ...

def test_groupby_complex(func: str, output: list[complex]) -> None:
    ...

def test_groupby_complex_raises(func: str) -> None:
    ...

def test_agg_of_mode_list(
    test: list[list[str | int]],
    constant: dict[int, list[str | int]]
) -> None:
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
import datetime
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Index, MultiIndex, Series
from pandas._testing import TM
from pandas.core.groupby.grouper import Grouping
from pandas.errors import SpecificationError

def test_groupby_agg_no_extra_calls() -> None: ...

def test_agg_regression1(tsframe: DataFrame) -> None: ...

def test_agg_must_agg(df: DataFrame) -> None: ...

def test_agg_ser_multi_key(df: DataFrame) -> None: ...

def test_agg_with_missing_values() -> None: ...

def test_groupby_aggregation_mixed_dtype() -> None: ...

def test_agg_apply_corner(ts: Series, tsframe: DataFrame) -> None: ...

def test_with_na_groups(any_real_numpy_dtype: np.dtype) -> None: ...

def test_agg_grouping_is_list_tuple(ts: Series) -> None: ...

def test_agg_python_multiindex(multiindex_dataframe_random_data: DataFrame) -> None: ...

@pytest.mark.parametrize("groupbyfunc", [lambda x: x.weekday(), [lambda x: x.month, lambda x: x.weekday()]])
def test_aggregate_str_func(tsframe: DataFrame, groupbyfunc: Union[Callable, List[Callable]]) -> None: ...

def test_std_masked_dtype(any_numeric_ea_dtype: str) -> None: ...

def test_agg_str_with_kwarg_axis_1_raises(df: DataFrame, reduction_func: str) -> None: ...

def test_aggregate_item_by_item(df: DataFrame) -> None: ...

def test_wrap_agg_out(three_group: DataFrame) -> None: ...

def test_agg_multiple_functions_maintain_order(df: DataFrame) -> None: ...

def test_series_index_name(df: DataFrame) -> None: ...

def test_agg_multiple_functions_same_name() -> None: ...

def test_agg_multiple_functions_same_name_with_ohlc_present() -> None: ...

def test_multiple_functions_tuples_and_non_tuples(df: DataFrame) -> None: ...

def test_more_flexible_frame_multi_function(df: DataFrame) -> None: ...

def test_multi_function_flexible_mix(df: DataFrame) -> None: ...

def test_groupby_agg_coercing_bools() -> None: ...

def test_groupby_agg_dict_with_getitem() -> None: ...

def test_groupby_agg_dict_dup_columns() -> None: ...

@pytest.mark.parametrize(
    "op",
    [
        lambda x: x.sum(),
        lambda x: x.cumsum(),
        lambda x: x.transform("sum"),
        lambda x: x.transform("cumsum"),
        lambda x: x.agg("sum"),
        lambda x: x.agg("cumsum"),
    ],
)
def test_bool_agg_dtype(op: Callable) -> None: ...

@pytest.mark.parametrize(
    "keys, agg_index",
    [
        (["a"], Index([1], name="a")),
        (["a", "b"], MultiIndex([[1], [2]], [[0], [0]], names=["a", "b"])),
    ],
)
@pytest.mark.parametrize(
    "input_dtype", ["bool", "int32", "int64", "float32", "float64"]
)
@pytest.mark.parametrize(
    "result_dtype", ["bool", "int32", "int64", "float32", "float64"]
)
@pytest.mark.parametrize("method", ["apply", "aggregate", "transform"])
def test_callable_result_dtype_frame(
    keys: List[str],
    agg_index: Union[Index, MultiIndex],
    input_dtype: str,
    result_dtype: str,
    method: str,
) -> None: ...

@pytest.mark.parametrize(
    "keys, agg_index",
    [
        (["a"], Index([1], name="a")),
        (["a", "b"], MultiIndex([[1], [2]], [[0], [0]], names=["a", "b"])),
    ],
)
@pytest.mark.parametrize("input", [True, 1, 1.0])
@pytest.mark.parametrize("dtype", [bool, int, float])
@pytest.mark.parametrize("method", ["apply", "aggregate", "transform"])
def test_callable_result_dtype_series(
    keys: List[str],
    agg_index: Union[Index, MultiIndex],
    input: Union[bool, int, float],
    dtype: type,
    method: str,
) -> None: ...

def test_order_aggregate_multiple_funcs() -> None: ...

def test_ohlc_ea_dtypes(any_numeric_ea_dtype: str) -> None: ...

@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("how", ["first", "last", "min", "max", "mean", "median"])
def test_uint64_type_handling(dtype: np.dtype, how: str) -> None: ...

def test_func_duplicates_raises() -> None: ...

@pytest.mark.parametrize(
    "index",
    [
        pd.CategoricalIndex(list("abc")),
        pd.interval_range(0, 3),
        pd.period_range("2020", periods=3, freq="D"),
        MultiIndex.from_tuples([("a", 0), ("a", 1), ("b", 0)]),
    ],
)
def test_agg_index_has_complex_internals(index: Union[pd.CategoricalIndex, pd.IntervalIndex, pd.PeriodIndex, MultiIndex]) -> None: ...

def test_agg_split_block() -> None: ...

def test_agg_split_object_part_datetime() -> None: ...

class TestNamedAggregationSeries:
    def test_series_named_agg(self) -> None: ...
    def test_no_args_raises(self) -> None: ...
    def test_series_named_agg_duplicates_no_raises(self) -> None: ...
    def test_mangled(self) -> None: ...
    @pytest.mark.parametrize(
        "inp",
        [
            pd.NamedAgg(column="anything", aggfunc="min"),
            ("anything", "min"),
            ["anything", "min"],
        ],
    )
    def test_named_agg_nametuple(self, inp: Union[pd.NamedAgg, Tuple[str, str], List[str]]) -> None: ...

class TestNamedAggregationDataFrame:
    def test_agg_relabel(self) -> None: ...
    def test_agg_relabel_non_identifier(self) -> None: ...
    def test_duplicate_no_raises(self) -> None: ...
    def test_agg_relabel_with_level(self) -> None: ...
    def test_agg_relabel_other_raises(self) -> None: ...
    def test_missing_raises(self) -> None: ...
    def test_agg_namedtuple(self) -> None: ...
    def test_mangled(self) -> None: ...

@pytest.mark.parametrize(
    "agg_col1, agg_col2, agg_col3, agg_result1, agg_result2, agg_result3",
    [
        (
            (("y", "A"), "max"),
            (("y", "A"), np.mean),
            (("y", "B"), "mean"),
            [1, 3],
            [0.5, 2.5],
            [5.5, 7.5],
        ),
        (
            (("y", "A"), lambda x: max(x)),
            (("y", "A"), lambda x: 1),
            (("y", "B"), np.mean),
            [1, 3],
            [1, 1],
            [5.5, 7.5],
        ),
        (
            pd.NamedAgg(("y", "A"), "max"),
            pd.NamedAgg(("y", "B"), np.mean),
            pd.NamedAgg(("y", "A"), lambda x: 1),
            [1, 3],
            [5.5, 7.5],
            [1, 1],
        ),
    ],
)
def test_agg_relabel_multiindex_column(
    agg_col1: Union[Tuple[Tuple[str, str], str], pd.NamedAgg],
    agg_col2: Union[Tuple[Tuple[str, str], Union[str, Callable]], pd.NamedAgg],
    agg_col3: Union[Tuple[Tuple[str, str], Union[str, Callable]], pd.NamedAgg],
    agg_result1: List[int],
    agg_result2: List[float],
    agg_result3: List[float],
) -> None: ...

def test_agg_relabel_multiindex_raises_not_exist() -> None: ...

def test_agg_relabel_multiindex_duplicates() -> None: ...

@pytest.mark.parametrize("kwargs", [{"c": ["min"]}, {"b": [], "c": ["min"]}])
def test_groupby_aggregate_empty_key(kwargs: Dict[str, Union[List[str], List]]) -> None: ...

def test_groupby_aggregate_empty_key_empty_return() -> None: ...

def test_groupby_aggregate_empty_with_multiindex_frame() -> None: ...

def test_grouby_agg_loses_results_with_as_index_false_relabel() -> None: ...

def test_grouby_agg_loses_results_with_as_index_false_relabel_multiindex() -> None: ...

def test_groupby_as_index_agg(df: DataFrame) -> None: ...

@pytest.mark.parametrize(
    "func",
    [lambda s: s.mean(), lambda s: np.mean(s), lambda s: np.nanmean(s)],
)
def test_multiindex_custom_func(func: Callable) -> None: ...

def myfunc(s: Series) -> float: ...

@pytest.mark.parametrize("func", [lambda s: np.percentile(s, q=0.9), myfunc])
def test_lambda_named_agg(func: Callable) -> None: ...

def test_aggregate_mixed_types() -> None: ...

@pytest.mark.xfail(reason="Not implemented;see GH 31256")
def test_aggregate_udf_na_extension_type() -> None: ...

class TestLambdaMangling:
    def test_basic(self) -> None: ...
    def test_mangle_series_groupby(self) -> None: ...
    @pytest.mark.xfail(reason="GH-26611. kwargs for multi-agg.")
    def test_with_kwargs(self) -> None: ...
    def test_agg_with_one_lambda(self) -> None: ...
    def test_agg_multiple_lambda(self) -> None: ...

def test_pass_args_kwargs_duplicate_columns(tsframe: DataFrame, as_index: bool) -> None: ...

def test_groupby_get_by_index() -> None: ...

@pytest.mark.parametrize(
    "grp_col_dict, exp_data",
    [
        ({"nr": "min", "cat_ord": "min"}, {"nr": [1, 5], "cat_ord": ["a", "c"]}),
        ({"cat_ord": "min"}, {"cat_ord": ["a", "c"]}),
        ({"nr": "min"}, {"nr": [1, 5]}),
    ],
)
def test_groupby_single_agg_cat_cols(
    grp_col_dict: Dict[str, str], exp_data: Dict[str, List[Union[int, str]]]
) -> None: ...

@pytest.mark.parametrize(
    "grp_col_dict, exp_data",
    [
        (
            {"nr": ["min", "max"], "cat_ord": "min"},
            [(1, 4, "a"), (5, 8, "c")],
        ),
        (
            {"nr": "min", "cat_ord": ["min", "max"]},
            [(1, "a", "b"), (5, "c", "d")],
        ),
        (
            {"cat_ord": ["min", "max"]},
            [("a", "b"), ("c", "d")],
        ),
    ],
)
def test_groupby_combined_aggs_cat_cols(
    grp_col_dict: Dict[str, Union[str, List[str]]],
    exp_data: List[Tuple],
) -> None: ...

def test_nonagg_agg() -> None: ...

def test_aggregate_datetime_objects() -> None: ...

def test_groupby_index_object_dtype() -> None: ...

def test_timeseries_groupby_agg() -> None: ...

def test_groupby_agg_precision(any_real_numeric_dtype: np.dtype) -> None: ...

def test_groupby_aggregate_directory(reduction_func: str) -> None: ...

def test_group_mean_timedelta_nat() -> None: ...

@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        (
            ["2021-01-01T00:00", "NaT", "2021-01-01T02:00"],
            ["2021-01-01T01:00"],
        ),
        (
            ["2021-01-01T00:00-0100", "NaT", "2021-01-01T02:00-0100"],
            ["2021-01-01T01:00-0100"],
        ),
    ],
)
def test_group_mean_datetime64_nat(
    input_data: List[str], expected_output: List[str]
) -> None: ...

@pytest.mark.parametrize(
    "func, output",
    [
        ("mean", [8 + 18j, 10 + 22j]),
        ("sum", [40 + 90j, 50 + 110j]),
    ],
)
def test_groupby_complex(func: str, output: List[complex]) -> None: ...

@pytest.mark.parametrize("func", ["min", "max", "var"])
def test_groupby_complex_raises(func: str) -> None: ...

@pytest.mark.parametrize(
    "test, constant",
    [
        ([[20, "A"], [20, "B"], [10, "C"]], {0: [10, 20], 1: ["C", ["A", "B"]]}),
        ([[20, "A"], [20, "B"], [30, "C"]], {0: [20, 30], 1: [["A", "B"], "C"]}),
        ([["a", 1], ["a", 1], ["b", 2], ["b", 3]], {0: ["a", "b"], 1: [1, [2, 3]]}),
        pytest.param(
            [["a", 1], ["a", 2], ["b", 3], ["b", 3]],
            {0: ["a", "b"], 1: [[1, 2], 3]},
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_agg_of_mode_list(
    test: List[List[Union[int, str]]], constant: Dict[int, List]
) -> None: ...

def test_dataframe_groupy_agg_list_like_func_with_args() -> None: ...

def test_series_groupy_agg_list_like_func_with_args() -> None: ...

def test_agg_groupings_selection() -> None: ...

def test_agg_multiple_with_as_index_false_subset_to_a_single_column() -> None: ...

def test_agg_with_as_index_false_with_list() -> None: ...

def test_groupby_agg_extension_timedelta_cumsum_with_named_aggregation() -> None: ...

def test_groupby_aggregation_empty_group() -> None: ...

def test_groupby_aggregation_duplicate_columns_single_dict_value() -> None: ...

def test_groupby_aggregation_duplicate_columns_multiple_dict_values() -> None: ...

def test_groupby_aggregation_duplicate_columns_some_empty_result() -> None: ...

def test_groupby_aggregation_multi_index_duplicate_columns() -> None: ...

def test_groupby_aggregation_func_list_multi_index_duplicate_columns() -> None: ...
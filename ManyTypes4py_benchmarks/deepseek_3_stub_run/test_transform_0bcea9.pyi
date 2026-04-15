import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
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
from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
)
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args

def assert_fp_equal(a: np.ndarray, b: np.ndarray) -> None: ...

def test_transform() -> None: ...

def test_transform_fast() -> None: ...

def test_transform_fast2() -> None: ...

def test_transform_fast3() -> None: ...

def test_transform_broadcast(
    tsframe: DataFrame,
    ts: Series,
) -> None: ...

def test_transform_axis_ts(tsframe: DataFrame) -> None: ...

def test_transform_dtype() -> None: ...

def test_transform_bug() -> None: ...

def test_transform_numeric_to_boolean() -> None: ...

def test_transform_datetime_to_timedelta() -> None: ...

def test_transform_datetime_to_numeric() -> None: ...

def test_transform_casting() -> None: ...

def test_transform_multiple(ts: Series) -> None: ...

def test_dispatch_transform(tsframe: DataFrame) -> None: ...

def test_transform_transformation_func(
    transformation_func: str,
) -> None: ...

def test_transform_select_columns(df: DataFrame) -> None: ...

def test_transform_nuisance_raises(
    df: DataFrame,
    using_infer_string: bool,
) -> None: ...

def test_transform_function_aliases(df: DataFrame) -> None: ...

def test_series_fast_transform_date() -> None: ...

@pytest.mark.parametrize("func", ...)
def test_transform_length(
    func: Callable[[Any], Any],
) -> None: ...

def test_transform_coercion() -> None: ...

def test_groupby_transform_with_int(
    using_infer_string: bool,
) -> None: ...

def test_groupby_transform_with_nan_group() -> None: ...

def test_transform_mixed_type() -> None: ...

@pytest.mark.parametrize("op, args, targop", ...)
def test_cython_transform_series(
    op: str,
    args: Tuple[Any, ...],
    targop: Callable[[Series], Series],
) -> None: ...

@pytest.mark.parametrize("op", ...)
@pytest.mark.parametrize("input, exp", ...)
def test_groupby_cum_skipna(
    op: str,
    skipna: bool,
    input: Dict[str, List[Union[float, str]]],
    exp: Union[List[float], Dict[Tuple[str, bool], List[float]]],
) -> None: ...

@pytest.fixture
def frame() -> DataFrame: ...

@pytest.fixture
def frame_mi(frame: DataFrame) -> DataFrame: ...

@pytest.mark.slow
@pytest.mark.parametrize("op, args, targop", ...)
@pytest.mark.parametrize("df_fix", ...)
@pytest.mark.parametrize("gb_target", ...)
def test_cython_transform_frame(
    request: Any,
    op: str,
    args: Tuple[Any, ...],
    targop: Callable[[DataFrame], DataFrame],
    df_fix: str,
    gb_target: Dict[str, Any],
) -> None: ...

@pytest.mark.slow
@pytest.mark.parametrize("op, args, targop", ...)
@pytest.mark.parametrize("df_fix", ...)
@pytest.mark.parametrize("gb_target", ...)
@pytest.mark.parametrize("column", ...)
def test_cython_transform_frame_column(
    request: Any,
    op: str,
    args: Tuple[Any, ...],
    targop: Callable[[Series], Series],
    df_fix: str,
    gb_target: Dict[str, Any],
    column: str,
) -> None: ...

@pytest.mark.parametrize("cols,expected", ...)
@pytest.mark.parametrize("agg_func", ...)
def test_transform_numeric_ret(
    cols: Union[str, List[str]],
    expected: Union[Series, DataFrame],
    agg_func: str,
) -> None: ...

def test_transform_ffill() -> None: ...

@pytest.mark.parametrize("mix_groupings", ...)
@pytest.mark.parametrize("as_series", ...)
@pytest.mark.parametrize("val1,val2", ...)
@pytest.mark.parametrize("fill_method,limit,exp_vals", ...)
def test_group_fill_methods(
    mix_groupings: bool,
    as_series: bool,
    val1: Any,
    val2: Any,
    fill_method: str,
    limit: Optional[int],
    exp_vals: List[str],
) -> None: ...

@pytest.mark.parametrize("fill_method", ...)
def test_pad_stable_sorting(fill_method: str) -> None: ...

@pytest.mark.parametrize("freq", ...)
@pytest.mark.parametrize("periods", ...)
def test_pct_change(
    frame_or_series: Any,
    freq: Optional[str],
    periods: int,
) -> None: ...

@pytest.mark.parametrize("func, expected_status", ...)
def test_ffill_bfill_non_unique_multilevel(
    func: str,
    expected_status: List[Optional[str]],
) -> None: ...

@pytest.mark.parametrize("func", ...)
def test_any_all_np_func(func: Callable[[Any], Any]) -> None: ...

def test_groupby_transform_rename() -> None: ...

@pytest.mark.parametrize("func", ...)
def test_groupby_transform_timezone_column(
    func: Union[str, Callable[[Any], Any]],
) -> None: ...

@pytest.mark.parametrize("func, values", ...)
def test_groupby_transform_with_datetimes(
    func: str,
    values: List[str],
) -> None: ...

def test_groupby_transform_dtype() -> None: ...

def test_transform_absent_categories(
    all_numeric_accumulations: str,
) -> None: ...

@pytest.mark.parametrize("func", ...)
@pytest.mark.parametrize("key, val", ...)
def test_ffill_not_in_axis(
    func: str,
    key: str,
    val: Union[int, Series],
) -> None: ...

def test_transform_invalid_name_raises() -> None: ...

def test_transform_agg_by_name(
    request: Any,
    reduction_func: str,
    frame_or_series: Any,
) -> None: ...

def test_transform_lambda_with_datetimetz() -> None: ...

def test_transform_fastpath_raises() -> None: ...

def test_transform_lambda_indexing() -> None: ...

def test_categorical_and_not_categorical_key(observed: bool) -> None: ...

def test_string_rank_grouping() -> None: ...

def test_transform_cumcount() -> None: ...

@pytest.mark.parametrize("keys", ...)
def test_null_group_lambda_self(
    sort: bool,
    dropna: bool,
    keys: List[str],
) -> None: ...

def test_null_group_str_reducer(
    request: Any,
    dropna: bool,
    reduction_func: str,
) -> None: ...

def test_null_group_str_transformer(
    dropna: bool,
    transformation_func: str,
) -> None: ...

def test_null_group_str_reducer_series(
    request: Any,
    dropna: bool,
    reduction_func: str,
) -> None: ...

def test_null_group_str_transformer_series(
    dropna: bool,
    transformation_func: str,
) -> None: ...

@pytest.mark.parametrize("func, expected_values", ...)
@pytest.mark.parametrize("keys", ...)
@pytest.mark.parametrize("keys_in_index", ...)
def test_transform_aligns(
    func: Callable[[Any], Any],
    frame_or_series: Any,
    expected_values: List[Union[int, float, None]],
    keys: List[str],
    keys_in_index: bool,
) -> None: ...

@pytest.mark.parametrize("keys", ...)
def test_as_index_no_change(
    keys: Union[str, List[str]],
    df: DataFrame,
    groupby_func: str,
) -> None: ...

@pytest.mark.parametrize("how", ...)
@pytest.mark.parametrize("numeric_only", ...)
def test_idxmin_idxmax_transform_args(
    how: str,
    skipna: bool,
    numeric_only: bool,
) -> None: ...

def test_transform_sum_one_column_no_matching_labels() -> None: ...

def test_transform_sum_no_matching_labels() -> None: ...

def test_transform_sum_one_column_with_matching_labels() -> None: ...

def test_transform_sum_one_column_with_missing_labels() -> None: ...

def test_transform_sum_one_column_with_matching_labels_and_missing_labels() -> None: ...

@pytest.mark.parametrize("dtype", ...)
def test_min_one_unobserved_category_no_type_coercion(dtype: str) -> None: ...

def test_min_all_empty_data_no_type_coercion() -> None: ...

def test_min_one_dim_no_type_coercion() -> None: ...

def test_nan_in_cumsum_group_label() -> None: ...
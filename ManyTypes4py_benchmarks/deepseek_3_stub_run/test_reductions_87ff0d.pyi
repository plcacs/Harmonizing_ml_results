import builtins
import datetime as dt
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
from pandas import DataFrame, MultiIndex, Series, Timestamp
from pandas._libs.tslibs import iNaT
import pandas._testing as tm
from pandas.core.dtypes.missing import na_value_for_dtype
from pandas.core.dtypes.common import pandas_dtype
from pandas.util import _test_decorators as td
import pytest

def test_basic_aggregations(
    dtype: str,
) -> None: ...

@pytest.mark.parametrize('vals', [['foo', 'bar', 'baz'], ['foo', '', ''], ['', '', ''], [1, 2, 3], [1, 0, 0], [0, 0, 0], [1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [True, True, True], [True, False, False], [False, False, False], [np.nan, np.nan, np.nan]])
def test_groupby_bool_aggs(
    skipna: bool,
    all_boolean_reductions: str,
    vals: List[Union[str, int, float, bool, float]],
) -> None: ...

def test_any() -> None: ...

def test_bool_aggs_dup_column_labels(
    all_boolean_reductions: str,
) -> None: ...

@pytest.mark.parametrize('data', [[False, False, False], [True, True, True], [pd.NA, pd.NA, pd.NA], [False, pd.NA, False], [True, pd.NA, True], [True, pd.NA, False]])
def test_masked_kleene_logic(
    all_boolean_reductions: str,
    skipna: bool,
    data: List[Union[bool, Any]],
) -> None: ...

@pytest.mark.parametrize('dtype1,dtype2,exp_col1,exp_col2', [('float', 'Float64', np.array([True], dtype=bool), pd.array([pd.NA], dtype='boolean')), ('Int64', 'float', pd.array([pd.NA], dtype='boolean'), np.array([True], dtype=bool)), ('Int64', 'Int64', pd.array([pd.NA], dtype='boolean'), pd.array([pd.NA], dtype='boolean')), ('Float64', 'boolean', pd.array([pd.NA], dtype='boolean'), pd.array([pd.NA], dtype='boolean'))])
def test_masked_mixed_types(
    dtype1: str,
    dtype2: str,
    exp_col1: np.ndarray,
    exp_col2: pd.array,
) -> None: ...

@pytest.mark.parametrize('dtype', ['Int64', 'Float64', 'boolean'])
def test_masked_bool_aggs_skipna(
    all_boolean_reductions: str,
    dtype: str,
    skipna: bool,
    frame_or_series: Callable,
) -> None: ...

@pytest.mark.parametrize('bool_agg_func,data,expected_res', [('any', [pd.NA, np.nan], False), ('any', [pd.NA, 1, np.nan], True), ('all', [pd.NA, pd.NaT], True), ('all', [pd.NA, False, pd.NaT], False)])
def test_object_type_missing_vals(
    bool_agg_func: str,
    data: List[Any],
    expected_res: bool,
    frame_or_series: Callable,
) -> None: ...

def test_object_NA_raises_with_skipna_false(
    all_boolean_reductions: str,
) -> None: ...

def test_empty(
    frame_or_series: Callable,
    all_boolean_reductions: str,
) -> None: ...

@pytest.mark.parametrize('how', ['idxmin', 'idxmax'])
def test_idxmin_idxmax_extremes(
    how: str,
    any_real_numpy_dtype: type,
) -> None: ...

@pytest.mark.parametrize('how', ['idxmin', 'idxmax'])
def test_idxmin_idxmax_extremes_skipna(
    skipna: bool,
    how: str,
    float_numpy_dtype: type,
) -> None: ...

@pytest.mark.parametrize('func, values', [('idxmin', {'c_int': [0, 2], 'c_float': [1, 3], 'c_date': [1, 2]}), ('idxmax', {'c_int': [1, 3], 'c_float': [0, 2], 'c_date': [0, 3]})])
@pytest.mark.parametrize('numeric_only', [True, False])
def test_idxmin_idxmax_returns_int_types(
    func: str,
    values: Dict[str, List[int]],
    numeric_only: bool,
) -> None: ...

@pytest.mark.parametrize('data', [(Timestamp('2011-01-15 12:50:28.502376'), Timestamp('2011-01-20 12:50:28.593448')), (24650000000000001, 24650000000000002)])
@pytest.mark.parametrize('method', ['count', 'min', 'max', 'first', 'last'])
def test_groupby_non_arithmetic_agg_int_like_precision(
    method: str,
    data: Tuple[Union[Timestamp, int], Union[Timestamp, int]],
) -> None: ...

@pytest.mark.parametrize('how', ['first', 'last'])
def test_first_last_skipna(
    any_real_nullable_dtype: type,
    sort: bool,
    skipna: bool,
    how: str,
) -> None: ...

def test_groupby_mean_no_overflow() -> None: ...

def test_mean_on_timedelta() -> None: ...

@pytest.mark.parametrize('values, dtype, result_dtype', [([0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9], 'float64', 'float64'), ([0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9], 'Float64', 'Float64'), ([0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9], 'Int64', 'Float64'), ([0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9], 'timedelta64[ns]', 'timedelta64[ns]'), (pd.to_datetime(['2019-05-09', pd.NaT, '2019-05-11', '2019-05-12', '2019-05-13', '2019-05-14', '2019-05-15', '2019-05-16', '2019-05-17', '2019-05-18']), 'datetime64[ns]', 'datetime64[ns]')])
def test_mean_skipna(
    values: List[Union[int, float, pd.Timestamp, pd.NaTType]],
    dtype: str,
    result_dtype: str,
    skipna: bool,
) -> None: ...

@pytest.mark.parametrize('values, dtype', [([0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9], 'float64'), ([0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9], 'Float64'), ([0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9], 'Int64'), ([0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9], 'timedelta64[ns]')])
def test_sum_skipna(
    values: List[Union[int, float]],
    dtype: str,
    skipna: bool,
) -> None: ...

def test_sum_skipna_object(
    skipna: bool,
) -> None: ...

@pytest.mark.parametrize('func, values, dtype, result_dtype', [('prod', [0, 1, 3, np.nan, 4, 5, 6, 7, -8, 9], 'float64', 'float64'), ('prod', [0, -1, 3, 4, 5, np.nan, 6, 7, 8, 9], 'Float64', 'Float64'), ('prod', [0, 1, 3, -4, 5, 6, 7, -8, np.nan, 9], 'Int64', 'Int64'), ('prod', [np.nan] * 10, 'float64', 'float64'), ('prod', [np.nan] * 10, 'Float64', 'Float64'), ('prod', [np.nan] * 10, 'Int64', 'Int64'), ('var', [0, -1, 3, 4, np.nan, 5, 6, 7, 8, 9], 'float64', 'float64'), ('var', [0, 1, 3, -4, 5, 6, 7, -8, 9, np.nan], 'Float64', 'Float64'), ('var', [0, -1, 3, 4, 5, -6, 7, np.nan, 8, 9], 'Int64', 'Float64'), ('var', [np.nan] * 10, 'float64', 'float64'), ('var', [np.nan] * 10, 'Float64', 'Float64'), ('var', [np.nan] * 10, 'Int64', 'Float64'), ('std', [0, 1, 3, -4, 5, 6, 7, -8, np.nan, 9], 'float64', 'float64'), ('std', [0, -1, 3, 4, 5, -6, 7, np.nan, 8, 9], 'Float64', 'Float64'), ('std', [0, 1, 3, -4, 5, 6, 7, -8, 9, np.nan], 'Int64', 'Float64'), ('std', [np.nan] * 10, 'float64', 'float64'), ('std', [np.nan] * 10, 'Float64', 'Float64'), ('std', [np.nan] * 10, 'Int64', 'Float64'), ('sem', [0, -1, 3, 4, 5, -6, 7, np.nan, 8, 9], 'float64', 'float64'), ('sem', [0, 1, 3, -4, 5, 6, 7, -8, np.nan, 9], 'Float64', 'Float64'), ('sem', [0, -1, 3, 4, 5, -6, 7, 8, 9, np.nan], 'Int64', 'Float64'), ('sem', [np.nan] * 10, 'float64', 'float64'), ('sem', [np.nan] * 10, 'Float64', 'Float64'), ('sem', [np.nan] * 10, 'Int64', 'Float64'), ('min', [0, -1, 3, 4, 5, -6, 7, np.nan, 8, 9], 'float64', 'float64'), ('min', [0, 1, 3, -4, 5, 6, 7, -8, np.nan, 9], 'Float64', 'Float64'), ('min', [0, -1, 3, 4, 5, -6, 7, 8, 9, np.nan], 'Int64', 'Int64'), ('min', [0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9], 'timedelta64[ns]', 'timedelta64[ns]'), ('min', pd.to_datetime(['2019-05-09', pd.NaT, '2019-05-11', '2019-05-12', '2019-05-13', '2019-05-14', '2019-05-15', '2019-05-16', '2019-05-17', '2019-05-18']), 'datetime64[ns]', 'datetime64[ns]'), ('min', [np.nan] * 10, 'float64', 'float64'), ('min', [np.nan] * 10, 'Float64', 'Float64'), ('min', [np.nan] * 10, 'Int64', 'Int64'), ('max', [0, -1, 3, 4, 5, -6, 7, np.nan, 8, 9], 'float64', 'float64'), ('max', [0, 1, 3, -4, 5, 6, 7, -8, np.nan, 9], 'Float64', 'Float64'), ('max', [0, -1, 3, 4, 5, -6, 7, 8, 9, np.nan], 'Int64', 'Int64'), ('max', [0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9], 'timedelta64[ns]', 'timedelta64[ns]'), ('max', pd.to_datetime(['2019-05-09', pd.NaT, '2019-05-11', '2019-05-12', '2019-05-13', '2019-05-14', '2019-05-15', '2019-05-16', '2019-05-17', '2019-05-18']), 'datetime64[ns]', 'datetime64[ns]'), ('max', [np.nan] * 10, 'float64', 'float64'), ('max', [np.nan] * 10, 'Float64', 'Float64'), ('max', [np.nan] * 10, 'Int64', 'Int64'), ('median', [0, -1, 3, 4, 5, -6, 7, np.nan, 8, 9], 'float64', 'float64'), ('median', [0, 1, 3, -4, 5, 6, 7, -8, np.nan, 9], 'Float64', 'Float64'), ('median', [0, -1, 3, 4, 5, -6, 7, 8, 9, np.nan], 'Int64', 'Float64'), ('median', [0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9], 'timedelta64[ns]', 'timedelta64[ns]'), ('median', pd.to_datetime(['2019-05-09', pd.NaT, '2019-05-11', '2019-05-12', '2019-05-13', '2019-05-14', '2019-05-15', '2019-05-16', '2019-05-17', '2019-05-18']), 'datetime64[ns]', 'datetime64[ns]'), ('median', [np.nan] * 10, 'float64', 'float64'), ('median', [np.nan] * 10, 'Float64', 'Float64'), ('median', [np.nan] * 10, 'Int64', 'Float64')])
def test_multifunc_skipna(
    func: str,
    values: List[Union[int, float, pd.Timestamp, pd.NaTType]],
    dtype: str,
    result_dtype: str,
    skipna: bool,
) -> None: ...

def test_cython_median() -> None: ...

def test_median_empty_bins(
    observed: bool,
) -> None: ...

def test_max_min_non_numeric() -> None: ...

def test_max_min_object_multiple_columns(
    using_infer_string: bool,
) -> None: ...

def test_min_date_with_nans() -> None: ...

def test_max_inat() -> None: ...

def test_max_inat_not_all_na() -> None: ...

@pytest.mark.parametrize('func', ['min', 'max'])
def test_groupby_aggregate_period_column(
    func: str,
) -> None: ...

@pytest.mark.parametrize('func', ['min', 'max'])
def test_groupby_aggregate_period_frame(
    func: str,
) -> None: ...

def test_aggregate_numeric_object_dtype() -> None: ...

@pytest.mark.parametrize('func', ['min', 'max'])
def test_aggregate_categorical_lost_index(
    func: str,
) -> None: ...

@pytest.mark.parametrize('dtype', ['Int64', 'Int32', 'Float64', 'Float32', 'boolean'])
def test_groupby_min_max_nullable(
    dtype: str,
) -> None: ...

def test_min_max_nullable_uint64_empty_group() -> None: ...

@pytest.mark.parametrize('func', ['first', 'last', 'min', 'max'])
def test_groupby_min_max_categorical(
    func: str,
) -> None: ...

@pytest.mark.parametrize('func', ['min', '
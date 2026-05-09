from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    Period,
    PeriodIndex,
    RangeIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
)
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics

def get_objs() -> list[Index | Series]:
    ...

class TestReductions:
    @pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
    @pytest.mark.parametrize('opname', ['max', 'min'])
    @pytest.mark.parametrize('obj', get_objs())
    def test_ops(self, opname: str, obj: Index | Series) -> None:
        ...

    @pytest.mark.parametrize('opname', ['max', 'min'])
    @pytest.mark.parametrize('dtype, val', [('object', float), ('float64', float), ('datetime64[ns]', datetime), ('Int64', int), ('boolean', bool)])
    def test_nanminmax(self, opname: str, dtype: str, val: float | int | datetime | bool, index_or_series: Index | Series) -> None:
        ...

    @pytest.mark.parametrize('opname', ['max', 'min'])
    def test_nanargminmax(self, opname: str, index_or_series: Index | Series) -> None:
        ...

    @pytest.mark.parametrize('opname', ['max', 'min'])
    @pytest.mark.parametrize('dtype', ['M8[ns]', 'datetime64[ns, UTC]'])
    def test_nanops_empty_object(self, opname: str, index_or_series: Index | Series, dtype: str) -> None:
        ...

    def test_argminmax(self) -> None:
        ...

    @pytest.mark.parametrize('op, expected_col', [['max', 'a'], ['min', 'b']])
    def test_same_tz_min_max_axis_1(self, op: str, expected_col: str) -> None:
        ...

    @pytest.mark.parametrize('func', ['maximum', 'minimum'])
    def test_numpy_reduction_with_tz_aware_dtype(self, tz_aware_fixture: Any, func: str) -> None:
        ...

    def test_nan_int_timedelta_sum(self) -> None:
        ...

class TestIndexReductions:
    @pytest.mark.parametrize('start,stop,step', [(int, int, int), (int, int, int), (int, int, int), (int, int, int), (int, int, int)])
    def test_max_min_range(self, start: int, stop: int, step: int) -> None:
        ...

    def test_minmax_timedelta64(self) -> None:
        ...

    @pytest.mark.parametrize('op', ['min', 'max'])
    def test_minmax_timedelta_empty_or_na(self, op: str) -> None:
        ...

    def test_numpy_minmax_timedelta64(self) -> None:
        ...

    def test_timedelta_ops(self) -> None:
        ...

    @pytest.mark.parametrize('opname', ['skew', 'kurt', 'sem', 'prod', 'var'])
    def test_invalid_td64_reductions(self, opname: str) -> None:
        ...

    @pytest.mark.parametrize('op', ['min', 'max'])
    def test_minmax_tz(self, op: str, tz_naive_fixture: Any) -> None:
        ...

    @pytest.mark.parametrize('op', ['min', 'max'])
    def test_minmax_nat_datetime64(self, op: str) -> None:
        ...

    def test_numpy_minmax_integer(self) -> None:
        ...

    def test_numpy_minmax_range(self) -> None:
        ...

    def test_numpy_minmax_datetime64(self) -> None:
        ...

    def test_minmax_period(self) -> None:
        ...

    @pytest.mark.parametrize('op', ['min', 'max'])
    @pytest.mark.parametrize('data', [[], [NaT], [NaT, NaT, NaT]])
    def test_minmax_period_empty_nat(self, op: str, data: list) -> None:
        ...

    def test_numpy_minmax_period(self) -> None:
        ...

    def test_min_max_categorical(self) -> None:
        ...

class TestSeriesReductions:
    def test_sum_inf(self) -> None:
        ...

    @pytest.mark.parametrize('dtype', ['float64', 'Float32', 'Int64', 'boolean', 'object'])
    @pytest.mark.parametrize('use_bottleneck', [True, False])
    @pytest.mark.parametrize('method, unit', [('sum', float), ('prod', float)])
    def test_empty(self, method: str, unit: float, use_bottleneck: bool, dtype: str) -> None:
        ...

    @pytest.mark.parametrize('method', ['mean', 'var'])
    @pytest.mark.parametrize('dtype', ['Float64', 'Int64', 'boolean'])
    def test_ops_consistency_on_empty_nullable(self, method: str, dtype: str) -> None:
        ...

    @pytest.mark.parametrize('method', ['mean', 'median', 'std', 'var'])
    def test_ops_consistency_on_empty(self, method: str) -> None:
        ...

    def test_nansum_buglet(self) -> None:
        ...

    @pytest.mark.parametrize('use_bottleneck', [True, False])
    @pytest.mark.parametrize('dtype', ['int32', 'int64'])
    def test_sum_overflow_int(self, use_bottleneck: bool, dtype: str) -> None:
        ...

    @pytest.mark.parametrize('use_bottleneck', [True, False])
    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    def test_sum_overflow_float(self, use_bottleneck: bool, dtype: str) -> None:
        ...

    def test_mean_masked_overflow(self) -> None:
        ...

    @pytest.mark.parametrize('ddof, exp', [(int, float), (int, float)])
    def test_var_masked_array(self, ddof: int, exp: float) -> None:
        ...

    @pytest.mark.parametrize('dtype', ('m8[ns]', 'M8[ns]', 'M8[ns, UTC]'))
    def test_empty_timeseries_reductions_return_nat(self, dtype: str, skipna: bool) -> None:
        ...

    def test_numpy_argmin(self) -> None:
        ...

    def test_numpy_argmax(self) -> None:
        ...

    def test_idxmin_dt64index(self, unit: str) -> None:
        ...

    def test_idxmin(self) -> None:
        ...

    def test_idxmax(self) -> None:
        ...

    def test_all_any(self) -> None:
        ...

    def test_numpy_all_any(self, index_or_series: Index | Series) -> None:
        ...

    def test_all_any_skipna(self) -> None:
        ...

    def test_all_any_bool_only(self) -> None:
        ...

    @pytest.mark.parametrize('data', [[False, None], [None, False], [False, np.nan], [np.nan, False]])
    def test_any_all_object_dtype_missing(self, data: list, all_boolean_reductions: str) -> None:
        ...

    @pytest.mark.parametrize('dtype', ['boolean', 'Int64', 'UInt64', 'Float64'])
    @pytest.mark.parametrize('data,expected_data', [([0, 0, 0], [[bool, bool], [bool, bool]]), ([1, 1, 1], [[bool, bool], [bool, bool]]), ([pd.NA, pd.NA, pd.NA], [[pd.NA, pd.NA], [bool, bool]]), ([0, pd.NA, 0], [[pd.NA, bool], [bool, bool]]), ([1, pd.NA, 1], [[bool, pd.NA], [bool, bool]]), ([1, pd.NA, 0], [[bool, bool], [bool, bool]])])
    def test_any_all_nullable_kleene_logic(self, all_boolean_reductions: str, skipna: bool, data: list, dtype: str, expected_data: list) -> None:
        ...

    def test_any_axis1_bool_only(self) -> None:
        ...

    def test_any_all_datetimelike(self) -> None:
        ...

    def test_any_all_string_dtype(self, any_string_dtype: str) -> None:
        ...

    def test_timedelta64_analytics(self) -> None:
        ...

    def test_assert_idxminmax_empty_raises(self) -> None:
        ...

    def test_idxminmax_object_dtype(self, using_infer_string: bool) -> None:
        ...

    def test_idxminmax_object_frame(self) -> None:
        ...

    def test_idxminmax_object_tuples(self) -> None:
        ...

    def test_idxminmax_object_decimals(self) -> None:
        ...

    def test_argminmax_object_ints(self) -> None:
        ...

    def test_idxminmax_with_inf(self) -> None:
        ...

    def test_sum_uint64(self) -> None:
        ...

    def test_signedness_preserved_after_sum(self) -> None:
        ...

class TestDatetime64SeriesReductions:
    @pytest.mark.parametrize('nat_ser', [Series([NaT, NaT]), Series([NaT, Timedelta('nat')]), Series([Timedelta('nat'), Timedelta('nat')])])
    def test_minmax_nat_series(self, nat_ser: Series) -> None:
        ...

    @pytest.mark.parametrize('nat_df', [[NaT, NaT], [NaT, Timedelta('nat')], [Timedelta('nat'), Timedelta('nat')]])
    def test_minmax_nat_dataframe(self, nat_df: list) -> None:
        ...

    def test_min_max(self) -> None:
        ...

    def test_min_max_series(self) -> None:
        ...

class TestCategoricalSeriesReductions:
    @pytest.mark.parametrize('function', ['min', 'max'])
    def test_min_max_unordered_raises(self, function: str) -> None:
        ...

    @pytest.mark.parametrize('values, categories', [(list[str], list[str]), (list[str], list[str]), (list[str] + [float], list[str]), (list[int], list[int]), (list[int] + [float], list[int])])
    @pytest.mark.parametrize('function', ['min', 'max'])
    def test_min_max_ordered(self, values: list, categories: list, function: str) -> None:
        ...

    @pytest.mark.parametrize('function', ['min', 'max'])
    def test_min_max_ordered_with_nan_only(self, function: str, skipna: bool) -> None:
        ...

    @pytest.mark.parametrize('function', ['min', 'max'])
    def test_min_max_skipna(self, function: str, skipna: bool) -> None:
        ...

class TestSeriesMode:
    def test_mode_empty(self, dropna: bool) -> None:
        ...

    @pytest.mark.parametrize('dropna, data, expected', [(bool, list[int], list[int]), (bool, list[int], list[int])])
    def test_mode_numerical(self, dropna: bool, data: list[int], expected: list[int], any_real_numpy_dtype: np.dtype) -> None:
        ...

    @pytest.mark.parametrize('dropna, expected', [(bool, list[float]), (bool, list[float])])
    def test_mode_numerical_nan(self, dropna: bool, expected: list[float]) -> None:
        ...

    @pytest.mark.parametrize('dropna, expected1, expected2', [(bool, list[str], list[str]), (bool, list[str], list[str])])
    def test_mode_object(self, dropna: bool, expected1: list[str], expected2: list[str]) -> None:
        ...

    @pytest.mark.parametrize('dropna, expected1, expected2', [(bool, list[str], list[str]), (bool, list[str], list[str])])
    def test_mode_string(self, dropna: bool, expected1: list[str], expected2: list[str], any_string_dtype: str) -> None:
        ...

    @pytest.mark.parametrize('dropna, expected1, expected2', [(bool, list[str], list[str]), (bool, list[str], list[str])])
    def test_mode_mixeddtype(self, dropna: bool, expected1: list[str], expected2: list[str]) -> None:
        ...

    @pytest.mark.parametrize('dropna, expected1, expected2', [(bool, list[str], list[str]), (bool, list[str], list[str])])
    def test_mode_datetime(self, dropna: bool, expected1: list[str], expected2: list[str]) -> None:
        ...

    @pytest.mark.parametrize('dropna, expected1, expected2', [(bool, list[str], list[str]), (bool, list[str], list[str])])
    def test_mode_timedelta(self, dropna: bool, expected1: list[str], expected2: list[str]) -> None:
        ...

    @pytest.mark.parametrize('dropna, expected1, expected2, expected3', [(bool, list[Categorical], list[Categorical], list[Categorical]), (bool, list[Categorical], list[Categorical], list[Categorical])])
    def test_mode_category(self, dropna: bool, expected1: list[Categorical], expected2: list[Categorical], expected3: list[Categorical]) -> None:
        ...

    @pytest.mark.parametrize('dropna, expected1, expected2', [(bool, list[int], list[int]), (bool, list[int], list[int])])
    def test_mode_intoverflow(self, dropna: bool, expected1: list[int], expected2: list[int]) -> None:
        ...

    def test_mode_sort_with_na(self) -> None:
        ...

    def test_mode_boolean_with_na(self) -> None:
        ...

    @pytest.mark.parametrize('array,expected,dtype', [(list[complex], list[complex], np.complex128), (list[complex], list[complex], np.complex64), (list[complex], list[complex], np.complex128)])
    def test_single_mode_value_complex(self, array: list[complex], expected: list[complex], dtype: np.dtype) -> None:
        ...

    @pytest.mark.parametrize('array,expected,dtype', [(list[complex], list[complex], np.complex128), (list[complex], list[complex], np.complex64)])
    def test_multimode_complex(self, array: list[complex], expected: list[complex], dtype: np.dtype) -> None:
        ...
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pytest
from pandas import (
    DataFrame,
    DatetimeIndex,
    MultiIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
)
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay

def test_doc_string() -> None:
    ...

def test_constructor(frame_or_series: Union[DataFrame, Series]) -> None:
    ...

@pytest.mark.parametrize('w', [2.0, 'foo', np.array([2])])
def test_invalid_constructor(frame_or_series: Union[DataFrame, Series], w: Union[float, str, np.ndarray]) -> None:
    ...

@pytest.mark.parametrize('window', [timedelta(days=3), Timedelta(days=3), '3D', VariableOffsetWindowIndexer(index=date_range('2015-12-25', periods=5), offset=BusinessDay(1))])
def test_freq_window_not_implemented(window: Union[timedelta, Timedelta, str, VariableOffsetWindowIndexer]) -> None:
    ...

@pytest.mark.parametrize('agg', ['cov', 'corr'])
def test_step_not_implemented_for_cov_corr(agg: str) -> None:
    ...

@pytest.mark.parametrize('window', [timedelta(days=3), Timedelta(days=3)])
def test_constructor_with_timedelta_window(window: Union[timedelta, Timedelta]) -> None:
    ...

@pytest.mark.parametrize('window', [timedelta(days=3), Timedelta(days=3), '3D'])
def test_constructor_timedelta_window_and_minperiods(window: Union[timedelta, Timedelta, str], raw: bool) -> None:
    ...

def test_closed_fixed(closed: str, arithmetic_win_operators: str) -> None:
    ...

@pytest.mark.parametrize('closed, window_selections', [('both', [[True, True, False, False, False], [True, True, True, False, False], [False, True, True, True, False], [False, False, True, True, True], [False, False, False, True, True]]), ('left', [[True, False, False, False, False], [True, True, False, False, False], [False, True, True, False, False], [False, False, True, True, False], [False, False, False, True, True]]), ('right', [[True, True, False, False, False], [False, True, True, False, False], [False, False, True, True, False], [False, False, False, True, True], [False, False, False, False, True]]), ('neither', [[True, False, False, False, False], [False, True, False, False, False], [False, False, True, False, False], [False, False, False, True, False], [False, False, False, False, True]])])
def test_datetimelike_centered_selections(closed: str, window_selections: List[List[List[bool]]], arithmetic_win_operators: str) -> None:
    ...

@pytest.mark.parametrize('window,closed,expected', [('3s', 'right', [3.0, 3.0, 3.0]), ('3s', 'both', [3.0, 3.0, 3.0]), ('3s', 'left', [3.0, 3.0, 3.0]), ('3s', 'neither', [3.0, 3.0, 3.0]), ('2s', 'right', [3.0, 2.0, 2.0]), ('2s', 'both', [3.0, 3.0, 3.0]), ('2s', 'left', [1.0, 3.0, 3.0]), ('2s', 'neither', [1.0, 2.0, 2.0])])
def test_datetimelike_centered_offset_covers_all(window: str, closed: str, expected: List[float], frame_or_series: Union[DataFrame, Series]) -> None:
    ...

@pytest.mark.parametrize('window,closed,expected', [('2D', 'right', [4, 4, 4, 4, 4, 4, 2, 2]), ('2D', 'left', [2, 2, 4, 4, 4, 4, 4, 4]), ('2D', 'both', [4, 4, 6, 6, 6, 6, 4, 4]), ('2D', 'neither', [2, 2, 2, 2, 2, 2, 2, 2])])
def test_datetimelike_nonunique_index_centering(window: str, closed: str, expected: List[float], frame_or_series: Union[DataFrame, Series]) -> None:
    ...

@pytest.mark.parametrize('closed,expected', [('left', [np.nan, np.nan, 1, 1, 1, 10, 14, 14, 18, 21]), ('neither', [np.nan, np.nan, 1, 1, 1, 9, 5, 5, 13, 8]), ('right', [0, 1, 3, 6, 10, 14, 11, 18, 21, 17]), ('both', [0, 1, 3, 6, 10, 15, 20, 27, 26, 30])])
def test_variable_window_nonunique(closed: str, expected: List[float], frame_or_series: Union[DataFrame, Series]) -> None:
    ...

@pytest.mark.parametrize('closed,expected', [('left', [np.nan, np.nan, 1, 1, 1, 10, 15, 15, 18, 21]), ('neither', [np.nan, np.nan, 1, 1, 1, 10, 15, 15, 13, 8]), ('right', [0, 1, 3, 6, 10, 15, 21, 28, 21, 17]), ('both', [0, 1, 3, 6, 10, 15, 21, 28, 26, 30])])
def test_variable_offset_window_nonunique(closed: str, expected: List[float], frame_or_series: Union[DataFrame, Series]) -> None:
    ...

def test_even_number_window_alignment() -> None:
    ...

def test_closed_fixed_binary_col(center: bool, step: int) -> None:
    ...

@pytest.mark.parametrize('closed', ['neither', 'left'])
def test_closed_empty(closed: str, arithmetic_win_operators: str) -> None:
    ...

@pytest.mark.parametrize('func', ['min', 'max'])
def test_closed_one_entry(func: str) -> None:
    ...

@pytest.mark.parametrize('func', ['min', 'max'])
def test_closed_one_entry_groupby(func: str) -> None:
    ...

@pytest.mark.parametrize('input_dtype', ['int', 'float'])
@pytest.mark.parametrize('func,closed,expected', [('min', 'right', [0.0, 0, 0, 1, 2, 3, 4, 5, 6, 7]), ('min', 'both', [0.0, 0, 0, 0, 1, 2, 3, 4, 5, 6]), ('min', 'neither', [np.nan, 0, 0, 1, 2, 3, 4, 5, 6, 7]), ('min', 'left', [np.nan, 0, 0, 0, 1, 2, 3, 4, 5, 6]), ('max', 'right', [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), ('max', 'both', [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), ('max', 'neither', [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8]), ('max', 'left', [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8])])
def test_closed_min_max_datetime(input_dtype: str, func: str, closed: str, expected: List[float]) -> None:
    ...

def test_closed_uneven() -> None:
    ...

@pytest.mark.parametrize('func,closed,expected', [('min', 'right', [np.nan, 0, 0, 1, 2, 3, 4, 5, np.nan, np.nan]), ('min', 'both', [np.nan, 0, 0, 0, 1, 2, 3, 4, 5, np.nan]), ('min', 'neither', [np.nan, np.nan, 0, 1, 2, 3, 4, 5, np.nan, np.nan]), ('min', 'left', [np.nan, np.nan, 0, 0, 1, 2, 3, 4, 5, np.nan]), ('max', 'right', [np.nan, 1, 2, 3, 4, 5, 6, 6, np.nan, np.nan]), ('max', 'both', [np.nan, 1, 2, 3, 4, 5, 6, 6, 6, np.nan]), ('max', 'neither', [np.nan, np.nan, 1, 2, 3, 4, 5, 6, np.nan, np.nan]), ('max', 'left', [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 6, np.nan])])
def test_closed_min_max_minp(func: str, closed: str, expected: List[float]) -> None:
    ...

@pytest.mark.parametrize('closed,expected', [('right', [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]), ('both', [0, 0.5, 1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]), ('neither', [np.nan, 0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]), ('left', [np.nan, 0, 0.5, 1, 2, 3, 4, 5, 6, 7])])
def test_closed_median_quantile(closed: str, expected: List[float]) -> None:
    ...

@pytest.mark.parametrize('roller', ['1s', 1])
def tests_empty_df_rolling(roller: Union[str, int]) -> None:
    ...

def test_empty_window_median_quantile() -> None:
    ...

def test_missing_minp_zero() -> None:
    ...

def test_missing_minp_zero_variable() -> None:
    ...

def test_multi_index_names() -> None:
    ...

def test_rolling_axis_sum() -> None:
    ...

def test_rolling_axis_count() -> None:
    ...

def test_readonly_array() -> None:
    ...

def test_rolling_datetime(tz_naive_fixture: Any) -> None:
    ...

def test_rolling_window_as_string(center: bool) -> None:
    ...

def test_min_periods1() -> None:
    ...

def test_rolling_count_with_min_periods(frame_or_series: Union[DataFrame, Series]) -> None:
    ...

def test_rolling_count_default_min_periods_with_null_values(frame_or_series: Union[DataFrame, Series]) -> None:
    ...

@pytest.mark.parametrize('df,expected,window,min_periods', [({'A': [1, 2, 3], 'B': [4, 5, 6]}, [({'A': [1], 'B': [4]}, [0]), ({'A': [1, 2], 'B': [4, 5]}, [0, 1]), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [0, 1, 2])], 3, None), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [({'A': [1], 'B': [4]}, [0]), ({'A': [1, 2], 'B': [4, 5]}, [0, 1]), ({'A': [2, 3], 'B': [5, 6]}, [1, 2])], 2, 1), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [({'A': [1], 'B': [4]}, [0]), ({'A': [1, 2], 'B': [4, 5]}, [0, 1]), ({'A': [2, 3], 'B': [5, 6]}, [1, 2])], 2, 2), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [({'A': [1], 'B': [4]}, [0]), ({'A': [2], 'B': [5]}, [1]), ({'A': [3], 'B': [6]}, [2])], 1, 1), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [({'A': [1], 'B': [4]}, [0]), ({'A': [2], 'B': [5]}, [1]), ({'A': [3], 'B': [6]}, [2])], 1, 0), ({'A': [1], 'B': [4]}, [], 2, None), ({'A': [1], 'B': [4]}, [], 2, 1), (None, [({}, [])], 2, None), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [({'A': [1.0], 'B': [np.nan]}, [0]), ({'A': [1, np.nan], 'B': [np.nan, 5]}, [0, 1]), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [0, 1, 2])], 3, 2)])
def test_iter_rolling_dataframe(df: Union[Dict[str, List[float]], None], expected: List[List[Union[Dict[str, List[float]], List[int]]]], window: int, min_periods: Optional[int]) -> None:
    ...

@pytest.mark.parametrize('expected,window', [([({'A': [1], 'B': [4]}, [0]), ({'A': [1, 2], 'B': [4, 5]}, [0, 1]), ({'A': [2, 3], 'B': [5, 6]}, [1, 2])], '2D'), ([({'A': [1], 'B': [4]}, [0]), ({'A': [1, 2], 'B': [4, 5]}, [0, 1]), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [0, 1, 2])], '3D'), ([({'A': [1], 'B': [4]}, [0]), ({'A': [2], 'B': [5]}, [1]), ({'A': [3], 'B': [6]}, [2])], '1D')])
def test_iter_rolling_on_dataframe(expected: List[List[Union[Dict[str, List[float]], List[int]]]], window: str) -> None:
    ...

def test_iter_rolling_on_dataframe_unordered() -> None:
    ...

@pytest.mark.parametrize('ser,expected,window, min_periods', [(Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 3, None), (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 3, 1), (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([2, 3], [1, 2])], 2, 1), (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([2, 3], [1, 2])], 2, 2), (Series([1, 2, 3]), [([1], [0]), ([2], [1]), ([3], [2])], 1, 0), (Series([1, 2, 3]), [([1], [0]), ([2], [1]), ([3], [2])], 1, 1), (Series([1, 2]), [([1], [0]), ([1, 2], [0, 1])], 2, 0), (Series([], dtype='int64'), [], 2, 1)])
def test_iter_rolling_series(ser: Series, expected: List[List[Union[List[int], List[int]]]], window: int, min_periods: Optional[int]) -> None:
    ...

@pytest.mark.parametrize('expected,expected_index,window', [([[0], [1], [2], [3], [4]], [date_range('2020-01-01', periods=1, freq='D'), date_range('2020-01-02', periods=1, freq='D'), date_range('2020-01-03', periods=1, freq='D'), date_range('2020-01-04', periods=1, freq='D'), date_range('2020-01-05', periods=1, freq='D')], '1D'), ([[0], [0, 1], [1, 2], [2, 3], [3, 4]], [date_range('2020-01-01', periods=1, freq='D'), date_range('2020-01-01', periods=2, freq='D'), date_range('2020-01-02', periods=2, freq='D'), date_range('2020-01-03', periods=2, freq='D'), date_range('2020-01-04', periods=2, freq='D')], '2D'), ([[0], [0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]], [date_range('2020-01-01', periods=1, freq='D'), date_range('2020-01-01', periods=2, freq='D'), date_range('2020-01-01', periods=3, freq='D'), date_range('2020-01-02', periods=3, freq='D'), date_range('2020-01-03', periods=3, freq='D')], '3D')])
def test_iter_rolling_datetime(expected: List[List[int]], expected_index: List[DatetimeIndex], window: str) -> None:
    ...

@pytest.mark.parametrize('grouping,_index', [({'level': 0}, MultiIndex.from_tuples([(0, 0), (0, 0), (1, 1), (1, 1), (1, 1)], names=[None, None])), ({'by': 'X'}, MultiIndex.from_tuples([(0, 0), (1, 0), (2, 1), (3, 1), (4, 1)], names=['X', None]))])
def test_rolling_positional_argument(grouping: Dict[str, Any], _index: MultiIndex, raw: bool) -> None:
    ...

@pytest.mark.parametrize('add', [0.0, 2.0])
def test_rolling_numerical_accuracy_kahan_mean(add: float, unit: str) -> None:
    ...

def test_rolling_numerical_accuracy_kahan_sum() -> None:
    ...

def test_rolling_numerical_accuracy_jump() -> None:
    ...

def test_rolling_numerical_accuracy_small_values() -> None:
    ...

def test_rolling_numerical_too_large_numbers() -> None:
    ...

@pytest.mark.parametrize(('index', 'window'), [(period_range(start='2020-01-01 08:00', end='2020-01-01 08:08', freq='min'), '2min'), (period_range(start='2020-01-01 08:00', end='2020-01-01 12:00', freq='30min'), '1h')])
@pytest.mark.parametrize(('func', 'values'), [('min', [np.nan, 0, 0, 1, 2, 3, 4, 5, 6]), ('max', [np.nan, 0, 1, 2, 3, 4, 5, 6, 7]), ('sum', [np.nan, 0, 1, 3, 5, 7, 9, 11, 13])])
def test_rolling_period_index(index: period_range, window: str, func: str, values: List[float]) -> None:
    ...

def test_rolling_sem(frame_or_series: Union[DataFrame, Series]) -> None:
    ...

@pytest.mark.xfail(is_platform_arm() or is_platform_power() or is_platform_riscv64(), reason='GH 38921')
@pytest.mark.parametrize(('func', 'third_value', 'values'), [('var', 1, [5e+33, 0, 0.5, 0.5, 2, 0]), ('std', 1, [7.071068e+16, 0, 0.7071068, 0.7071068, 1.414214, 0]), ('var', 2, [5e+33, 0.5, 0, 0.5, 2, 0]), ('std', 2, [7.071068e+16, 0.7071068, 0, 0.7071068, 1.414214, 0])])
def test_rolling_var_numerical_issues(func: str, third_value: int, values: List[float]) -> None:
    ...

def test_timeoffset_as_window_parameter_for_corr(unit: str) -> None:
    ...

@pytest.mark.parametrize('method', ['var', 'sum', 'mean', 'skew', 'kurt', 'min', 'max'])
def test_rolling_decreasing_indices(method: str) -> None:
    ...

@pytest.mark.parametrize('window,closed,expected', [('2s', 'right', [1.0, 3.0, 5.0, 3.0]), ('2s', 'left', [0.0, 1.0, 3.0, 5.0]), ('2s', 'both', [1.0, 3.0, 6.0, 5.0]), ('2s', 'neither', [0.0, 1.0, 2.0, 3.0]), ('3s', 'right', [1.0, 3.0, 6.0, 5.0]), ('3s', 'left', [1.0, 3.0, 6.0, 5.0]), ('3s', 'both', [1.0, 3.0, 6.0, 5.0]), ('3s', 'neither', [1.0, 3.0, 6.0, 5.0])])
def test_rolling_decreasing_indices_centered(window: str, closed: str, expected: List[float], frame_or_series: Union[DataFrame, Series]) -> None:
    ...

@pytest.mark.parametrize('window,expected', [('1ns', [1.0, 1.0, 1.0, 1.0]), ('3ns', [2.0, 3.0, 3.0, 2.0])])
def test_rolling_center_nanosecond_resolution(window: str, closed: str, expected: List[float], frame_or_series: Union[DataFrame, Series]) -> None:
    ...

@pytest.mark.parametrize('method,expected', [('var', [float('nan'), 43.0, float('nan'), 136.333333, 43.5, 94.966667, 182.0, 318.0]), ('mean', [float('nan'), 7.5, float('nan'), 21.5, 6.0, 9.166667, 13.0, 17.5]), ('sum', [float('nan'), 30.0, float('nan'), 86.0, 30.0, 55.0, 91.0, 140.0]), ('skew', [float('nan'), 0.709296, float('nan'), 0.407073, 0.984656, 0.919184, 0.874674, 0.842418]), ('kurt', [float('nan'), -0.5916711736073559, float('nan'), -1.0028993131317954, -0.06103844629409494, -0.254143227116194, -0.37362637362637585, -0.45439658241367054])])
def test_rolling_non_monotonic(method: str, expected: List[float]) -> None:
    ...

@pytest.mark.parametrize(('index', 'window'), [([0, 1, 2, 3, 4], 2), (date_range('2001-01-01', freq='D', periods=5), '2D')])
def test_rolling_corr_timedelta_index(index: Union[List[int], DatetimeIndex], window: Union[int, str]) -> None:
    ...

@pytest.mark.parametrize('values,method,expected', [([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'first', [float('nan'), float('nan'), 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), ([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan], 'first', [float('nan')] * 10), ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'last', [float('nan'), float('nan'), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), ([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan], 'last', [float('nan')] * 10)])
def test_rolling_first_last(values: List[float], method: str, expected: List[float]) -> None:
    ...

@pytest.mark.parametrize('values,method,expected', [([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'first', [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), ([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan], 'first', [1.0, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0]), ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'last', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), ([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan], 'last', [1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0, 9.0])])
def test_rolling_first_last_no_minp(values: List[float], method: str, expected: List[float]) -> None:
    ...

def test_groupby_rolling_nan_included() -> None:
    ...

@pytest.mark.parametrize('method', ['skew', 'kurt'])
def test_rolling_skew_kurt_numerical_stability(method: str) -> None:
    ...

@pytest.mark.parametrize(('method', 'values'), [('skew', [2.0, 0.854563, 0.0, 1.999984]), ('kurt', [4.0, -1.289256, -1.2, 3.999946])])
def test_rolling_skew_kurt_large_value_range(method: str, values: List[float]) -> None:
    ...

def test_invalid_method() -> None:
    ...

def test_rolling_descending_date_order_with_offset(frame_or_series: Union[DataFrame, Series]) -> None:
    ...

def test_rolling_var_floating_artifact_precision() -> None:
    ...

def test_rolling_std_small_values() -> None:
    ...

@pytest.mark.parametrize('start, exp_values', [(1, [0.03, 0.0155, 0.0155, 0.011, 0.01025]), (2, [0.001, 0.001, 0.0015, 0.00366666])])
def test_rolling_mean_all_nan_window_floating_artifacts(start: int, exp_values: List[float]) -> None:
    ...

def test_rolling_sum_all_nan_window_floating_artifacts() -> None:
    ...

def test_rolling_zero_window() -> None:
    ...

@pytest.mark.parametrize('window', [1, 3, 10, 20])
@pytest.mark.parametrize('method', ['min', 'max', 'average'])
@pytest.mark.parametrize('pct', [True, False])
@pytest.mark.parametrize('ascending', [True, False])
@pytest.mark.parametrize('test_data', ['default', 'duplicates', 'nans'])
def test_rank(window: int, method: str, pct: bool, ascending: bool, test_data: str) -> None:
    ...

def test_rolling_quantile_np_percentile() -> None:
    ...

@pytest.mark.parametrize('quantile', [0.0, 0.1, 0.45, 0.5, 1])
@pytest.mark.parametrize('interpolation', ['linear', 'lower', 'higher', 'nearest', 'midpoint'])
@pytest.mark.parametrize('data', [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [8.0, 1.0, 3.0, 4.0, 5.0, 2.0, 6.0, 7.0], [0.0, np.nan, 0.2, np.nan, 0.4], [np.nan, np.nan, np.nan, np.nan], [np.nan, 0.1, np.nan, 0.3, 0.4, 0.5], [0.5], [np.nan, 0.7, 0.6]])
def test_rolling_quantile_interpolation_options(quantile: float, interpolation: str, data: List[float]) -> None:
    ...

def test_invalid_quantile_value() -> None:
    ...

def test_rolling_std_1obs() -> None:
    ...

def test_rolling_std_neg_sqrt() -> None:
    ...

def test_step_not_integer_raises() -> None:
    ...

def test_step_not_positive_raises() -> None:
    ...

@pytest.mark.parametrize(['values', 'window', 'min_periods', 'expected'], [[[20, 10, 10, np.inf, 1, 1, 2, 3], 3, 1, [np.nan, 50, 100 / 3, 0, 40.5, 0, 1 / 3, 1]], [[20, 10, 10, np.nan, 10, 1, 2, 3], 3, 1, [np.nan, 50, 100 / 3, 0, 0, 40.5, 73 / 3, 1]], [[np.nan, 5, 6, 7, 5, 5, 5], 3, 3, [np.nan] * 3 + [1, 1, 4 / 3, 0]], [[5, 7, 7, 7, np.nan, np.inf, 4, 3, 3, 3], 3, 3, [np.nan] * 2 + [4 / 3, 0] + [np.nan] * 4 + [1 / 3, 0]], [[5, 7, 7, 7, np.nan, np.inf, 7, 3, 3, 3], 3, 3, [np.nan] * 2 + [4 / 3, 0] + [np.nan] * 4 + [16 / 3, 0]], [[5, 7] * 4, 3, 3, [np.nan] * 2 + [4 / 3] * 6], [[5, 7, 5, np.nan, 7, 5, 7], 3, 2, [np.nan, 2, 4 / 3] + [2] * 3 + [4 / 3]]])
def test_rolling_var_same_value_count_logic(values: List[float], window: int, min_periods: int, expected: List[float]) -> None:
    ...

def test_rolling_mean_sum_floating_artifacts() -> None:
    ...

def test_rolling_skew_kurt_floating_artifacts() -> None:
    ...

@pytest.mark.parametrize('kernel', ['corr', 'cov'])
@pytest.mark.parametrize('use_arg', [True, False])
def test_numeric_only_corr_cov_frame(kernel: str, numeric_only: bool, use_arg: bool) -> None:
    ...

@pytest.mark.parametrize('dtype', [int, object])
def test_numeric_only_series(arithmetic_win_operators: str, numeric_only: bool, dtype: type) -> None:
    ...

@pytest.mark.parametrize('kernel', ['corr', 'cov'])
@pytest.mark.parametrize('use_arg', [True, False])
@pytest.mark.parametrize('dtype', [int, object])
def test_numeric_only_corr_cov_series(kernel: str, use_arg: bool, numeric_only: bool, dtype: type) -> None:
    ...

@pytest.mark.parametrize('tz', [None, 'UTC', 'Europe/Prague'])
def test_rolling_timedelta_window_non_nanoseconds(unit: str, tz: Optional[str]) -> None:
    ...
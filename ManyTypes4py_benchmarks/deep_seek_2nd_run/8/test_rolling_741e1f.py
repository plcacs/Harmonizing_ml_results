from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
from pandas.compat import IS64, is_platform_arm, is_platform_power, is_platform_riscv64
from pandas import DataFrame, DatetimeIndex, MultiIndex, Series, Timedelta, Timestamp, date_range, period_range
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay


def test_doc_string() -> None:
    df = DataFrame({'B': [0, 1, 2, np.nan, 4]})
    df
    df.rolling(2).sum()
    df.rolling(2, min_periods=1).sum()


def test_constructor(frame_or_series: Union[DataFrame, Series]) -> None:
    c = frame_or_series(range(5)).rolling
    c(0)
    c(window=2)
    c(window=2, min_periods=1)
    c(window=2, min_periods=1, center=True)
    c(window=2, min_periods=1, center=False)
    msg = 'window must be an integer 0 or greater'
    with pytest.raises(ValueError, match=msg):
        c(-1)


@pytest.mark.parametrize('w', [2.0, 'foo', np.array([2])])
def test_invalid_constructor(frame_or_series: Union[DataFrame, Series], w: Any) -> None:
    c = frame_or_series(range(5)).rolling
    msg = '|'.join(['window must be an integer', 'passed window foo is not compatible with a datetimelike index'])
    with pytest.raises(ValueError, match=msg):
        c(window=w)
    msg = 'min_periods must be an integer'
    with pytest.raises(ValueError, match=msg):
        c(window=2, min_periods=w)
    msg = 'center must be a boolean'
    with pytest.raises(ValueError, match=msg):
        c(window=2, min_periods=1, center=w)


@pytest.mark.parametrize('window', [timedelta(days=3), Timedelta(days=3), '3D', VariableOffsetWindowIndexer(index=date_range('2015-12-25', periods=5), offset=BusinessDay(1))])
def test_freq_window_not_implemented(window: Union[timedelta, Timedelta, str, VariableOffsetWindowIndexer]) -> None:
    df = DataFrame(np.arange(10), index=date_range('2015-12-24', periods=10, freq='D'))
    with pytest.raises(NotImplementedError, match='^step (not implemented|is not supported)'):
        df.rolling(window, step=3).sum()


@pytest.mark.parametrize('agg', ['cov', 'corr'])
def test_step_not_implemented_for_cov_corr(agg: str) -> None:
    roll = DataFrame(range(2)).rolling(1, step=2)
    with pytest.raises(NotImplementedError, match='step not implemented'):
        getattr(roll, agg)()


@pytest.mark.parametrize('window', [timedelta(days=3), Timedelta(days=3)])
def test_constructor_with_timedelta_window(window: Union[timedelta, Timedelta]) -> None:
    n = 10
    df = DataFrame({'value': np.arange(n)}, index=date_range('2015-12-24', periods=n, freq='D'))
    expected_data = np.append([0.0, 1.0], np.arange(3.0, 27.0, 3))
    result = df.rolling(window=window).sum()
    expected = DataFrame({'value': expected_data}, index=date_range('2015-12-24', periods=n, freq='D'))
    tm.assert_frame_equal(result, expected)
    expected = df.rolling('3D').sum()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('window', [timedelta(days=3), Timedelta(days=3), '3D'])
def test_constructor_timedelta_window_and_minperiods(window: Union[timedelta, Timedelta, str], raw: bool) -> None:
    n = 10
    df = DataFrame({'value': np.arange(n)}, index=date_range('2017-08-08', periods=n, freq='D'))
    expected = DataFrame({'value': np.append([np.nan, 1.0], np.arange(3.0, 27.0, 3))}, index=date_range('2017-08-08', periods=n, freq='D'))
    result_roll_sum = df.rolling(window=window, min_periods=2).sum()
    result_roll_generic = df.rolling(window=window, min_periods=2).apply(sum, raw=raw)
    tm.assert_frame_equal(result_roll_sum, expected)
    tm.assert_frame_equal(result_roll_generic, expected)


def test_closed_fixed(closed: str, arithmetic_win_operators: str) -> None:
    func_name = arithmetic_win_operators
    df_fixed = DataFrame({'A': [0, 1, 2, 3, 4]})
    df_time = DataFrame({'A': [0, 1, 2, 3, 4]}, index=date_range('2020', periods=5))
    result = getattr(df_fixed.rolling(2, closed=closed, min_periods=1), func_name)()
    expected = getattr(df_time.rolling('2D', closed=closed, min_periods=1), func_name)().reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('closed,window_selections', [('both', [[True, True, False, False, False], [True, True, True, False, False], [False, True, True, True, False], [False, False, True, True, True], [False, False, False, True, True]]), ('left', [[True, False, False, False, False], [True, True, False, False, False], [False, True, True, False, False], [False, False, True, True, False], [False, False, False, True, True]]), ('right', [[True, True, False, False, False], [False, True, True, False, False], [False, False, True, True, False], [False, False, False, True, True], [False, False, False, False, True]]), ('neither', [[True, False, False, False, False], [False, True, False, False, False], [False, False, True, False, False], [False, False, False, True, False], [False, False, False, False, True]])])
def test_datetimelike_centered_selections(closed: str, window_selections: List[List[bool]], arithmetic_win_operators: str) -> None:
    func_name = arithmetic_win_operators
    df_time = DataFrame({'A': [0.0, 1.0, 2.0, 3.0, 4.0]}, index=date_range('2020', periods=5))
    expected = DataFrame({'A': [getattr(df_time['A'].iloc[s], func_name)() for s in window_selections]}, index=date_range('2020', periods=5))
    if func_name == 'sem':
        kwargs: Dict[str, Any] = {'ddof': 0}
    else:
        kwargs = {}
    result = getattr(df_time.rolling('2D', closed=closed, min_periods=1, center=True), func_name)(**kwargs)
    tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize('window,closed,expected', [('3s', 'right', [3.0, 3.0, 3.0]), ('3s', 'both', [3.0, 3.0, 3.0]), ('3s', 'left', [3.0, 3.0, 3.0]), ('3s', 'neither', [3.0, 3.0, 3.0]), ('2s', 'right', [3.0, 2.0, 2.0]), ('2s', 'both', [3.0, 3.0, 3.0]), ('2s', 'left', [1.0, 3.0, 3.0]), ('2s', 'neither', [1.0, 2.0, 2.0])])
def test_datetimelike_centered_offset_covers_all(window: str, closed: str, expected: List[float], frame_or_series: Union[DataFrame, Series]) -> None:
    index = [Timestamp('20130101 09:00:01'), Timestamp('20130101 09:00:02'), Timestamp('20130101 09:00:02')]
    df = frame_or_series([1, 1, 1], index=index)
    result = df.rolling(window, closed=closed, center=True).sum()
    expected = frame_or_series(expected, index=index)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize('window,closed,expected', [('2D', 'right', [4, 4, 4, 4, 4, 4, 2, 2]), ('2D', 'left', [2, 2, 4, 4, 4, 4, 4, 4]), ('2D', 'both', [4, 4, 6, 6, 6, 6, 4, 4]), ('2D', 'neither', [2, 2, 2, 2, 2, 2, 2, 2])])
def test_datetimelike_nonunique_index_centering(window: str, closed: str, expected: List[int], frame_or_series: Union[DataFrame, Series]) -> None:
    index = DatetimeIndex(['2020-01-01', '2020-01-01', '2020-01-02', '2020-01-02', '2020-01-03', '2020-01-03', '2020-01-04', '2020-01-04'])
    df = frame_or_series([1] * 8, index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)
    result = df.rolling(window, center=True, closed=closed).sum()
    tm.assert_equal(result, expected)


@pytest.mark.parametrize('closed,expected', [('left', [np.nan, np.nan, 1, 1, 1, 10, 14, 14, 18, 21]), ('neither', [np.nan, np.nan, 1, 1, 1, 9, 5, 5, 13, 8]), ('right', [0, 1, 3, 6, 10, 14, 11, 18, 21, 17]), ('both', [0, 1, 3, 6, 10, 15, 20, 27, 26, 30])])
def test_variable_window_nonunique(closed: str, expected: List[float], frame_or_series: Union[DataFrame, Series]) -> None:
    index = DatetimeIndex(['2011-01-01', '2011-01-01', '2011-01-02', '2011-01-02', '2011-01-02', '2011-01-03', '2011-01-04', '2011-01-04', '2011-01-05', '2011-01-06'])
    df = frame_or_series(range(10), index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)
    result = df.rolling('2D', closed=closed).sum()
    tm.assert_equal(result, expected)


@pytest.mark.parametrize('closed,expected', [('left', [np.nan, np.nan, 1, 1, 1, 10, 15, 15, 18, 21]), ('neither', [np.nan, np.nan, 1, 1, 1, 10, 15, 15, 13, 8]), ('right', [0, 1, 3, 6, 10, 15, 21, 28, 21, 17]), ('both', [0, 1, 3, 6, 10, 15, 21, 28, 26, 30])])
def test_variable_offset_window_nonunique(closed: str, expected: List[float], frame_or_series: Union[DataFrame, Series]) -> None:
    index = DatetimeIndex(['2011-01-01', '2011-01-01', '2011-01-02', '2011-01-02', '2011-01-02', '2011-01-03', '2011-01-04', '2011-01-04', '2011-01-05', '2011-01-06'])
    df = frame_or_series(range(10), index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)
    offset = BusinessDay(2)
    indexer = VariableOffsetWindowIndexer(index=index, offset=offset)
    result = df.rolling(indexer, closed=closed, min_periods=1).sum()
    tm.assert_equal(result, expected)


def test_even_number_window_alignment() -> None:
    s = Series(range(3), index=date_range(start='2020-01-01', freq='D', periods=3))
    result = s.rolling(window='2D', min_periods=1, center=True).mean()
    expected = Series([0.5, 1.5, 2], index=s.index)
    tm.assert_series_equal(result, expected)


def test_closed_fixed_binary_col(center: bool, step: int) -> None:
    data = [0, 1, 1, 0, 0, 1, 0, 1]
    df = DataFrame({'binary_col': data}, index=date_range(start='2020-01-01', freq='min', periods=len(data)))
    if center:
        expected_data = [2 / 3, 0.5, 0.4, 0.5, 0.428571, 0.5, 0.571429, 0.5]
    else:
        expected_data = [np.nan, 0, 0.5, 2 / 3, 0.5, 0.4, 0.5, 0.428571]
    expected = DataFrame(expected_data, columns=['binary_col'], index=date_range(start='2020-01-01', freq='min', periods=len(expected_data)))[::step]
    rolling = df.rolling(window=len(df), closed='left', min_periods=1, center=center, step=step)
    result = rolling.mean()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('closed', ['neither', 'left'])
def test_closed_empty(closed: str, arithmetic_win_operators: str) -> None:
    func_name = arithmetic_win_operators
    ser = Series(data=np.arange(5), index=date_range('2000', periods=5, freq='2D'))
    roll = ser.rolling('1D', closed=closed)
    result = getattr(roll, func_name)()
    expected = Series([np.nan] * 5, index=ser.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('func', ['min', 'max'])
def test_closed_one_entry(func: str) -> None:
    ser = Series(data=[2], index=date_range('2000', periods=1))
    result = getattr(ser.rolling('10D', closed='left'), func)()
    tm.assert_series_equal(result, Series([np.nan], index=ser.index))


@pytest.mark.parametrize('func', ['min', 'max'])
def test_closed_one_entry_groupby(func: str) -> None:
    ser = DataFrame(data={'A': [1, 1, 2], 'B': [3, 2, 1]}, index=date_range('2000', periods=3))
    result = getattr(ser.groupby('A', sort=False)['B'].rolling('10D', closed='left'), func)()
    exp_idx = MultiIndex.from_arrays(arrays=[[1, 1, 2], ser.index], names=('A', None))
    expected = Series(data=[np.nan, 3, np.nan], index=exp_idx, name='B')
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('input_dtype', ['int', 'float'])
@pytest.mark.parametrize('func,closed,expected', [('min', 'right', [0.0, 0, 0, 1, 2, 3, 4, 5, 6, 7]), ('min', 'both', [0.0, 0, 0, 0, 1, 2, 3, 4, 5, 6]), ('min', 'neither', [np.nan, 0, 0, 1, 2, 3, 4, 5, 6, 7]), ('min', 'left', [np.nan, 0, 0, 0, 1, 2, 3, 4, 5, 6]), ('max', 'right', [0.0, 1, 2, 3, 4,
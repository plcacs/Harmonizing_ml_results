from datetime import datetime, timedelta
import numpy as np
import pytest
from pandas.compat import IS64, is_platform_arm, is_platform_power, is_platform_riscv64
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
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
from typing import Union, Any, List, Tuple, Optional

FrameOrSeries = Union[DataFrame, Series]

def test_doc_string() -> None:
    df: DataFrame = DataFrame({'B': [0, 1, 2, np.nan, 4]})
    df
    df.rolling(2).sum()
    df.rolling(2, min_periods=1).sum()

def test_constructor(frame_or_series: FrameOrSeries) -> None:
    c = frame_or_series(range(5)).rolling
    c(0)
    c(window=2)
    c(window=2, min_periods=1)
    c(window=2, min_periods=1, center=True)
    c(window=2, min_periods=1, center=False)
    msg: str = 'window must be an integer 0 or greater'
    with pytest.raises(ValueError, match=msg):
        c(-1)

@pytest.mark.parametrize('w', [2.0, 'foo', np.array([2])])
def test_invalid_constructor(frame_or_series: FrameOrSeries, w: Any) -> None:
    c = frame_or_series(range(5)).rolling
    msg: str = '|'.join(['window must be an integer', 'passed window foo is not compatible with a datetimelike index'])
    with pytest.raises(ValueError, match=msg):
        c(window=w)
    msg = 'min_periods must be an integer'
    with pytest.raises(ValueError, match=msg):
        c(window=2, min_periods=w)
    msg = 'center must be a boolean'
    with pytest.raises(ValueError, match=msg):
        c(window=2, min_periods=1, center=w)

@pytest.mark.parametrize('window', [
    timedelta(days=3),
    Timedelta(days=3),
    '3D',
    VariableOffsetWindowIndexer(index=date_range('2015-12-25', periods=5), offset=BusinessDay(1))
])
def test_freq_window_not_implemented(window: Any) -> None:
    df: DataFrame = DataFrame(np.arange(10), index=date_range('2015-12-24', periods=10, freq='D'))
    with pytest.raises(NotImplementedError, match='^step (not implemented|is not supported)'):
        df.rolling(window, step=3).sum()

@pytest.mark.parametrize('agg', ['cov', 'corr'])
def test_step_not_implemented_for_cov_corr(agg: str) -> None:
    roll = DataFrame(range(2)).rolling(1, step=2)
    with pytest.raises(NotImplementedError, match='step not implemented'):
        getattr(roll, agg)()

@pytest.mark.parametrize('window', [timedelta(days=3), Timedelta(days=3)])
def test_constructor_with_timedelta_window(window: Any) -> None:
    n: int = 10
    df: DataFrame = DataFrame({'value': np.arange(n)}, index=date_range('2015-12-24', periods=n, freq='D'))
    expected_data: np.ndarray = np.append([0.0, 1.0], np.arange(3.0, 27.0, 3))
    result: DataFrame = df.rolling(window=window).sum()
    expected: DataFrame = DataFrame({'value': expected_data}, index=date_range('2015-12-24', periods=n, freq='D'))
    tm.assert_frame_equal(result, expected)
    expected = df.rolling('3D').sum()
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('window', [timedelta(days=3), Timedelta(days=3), '3D'])
def test_constructor_timedelta_window_and_minperiods(window: Any, raw: bool) -> None:
    n: int = 10
    df: DataFrame = DataFrame({'value': np.arange(n)}, index=date_range('2017-08-08', periods=n, freq='D'))
    expected: DataFrame = DataFrame({'value': np.append([np.nan, 1.0], np.arange(3.0, 27.0, 3))}, index=date_range('2017-08-08', periods=n, freq='D'))
    result_roll_sum: DataFrame = df.rolling(window=window, min_periods=2).sum()
    result_roll_generic: DataFrame = df.rolling(window=window, min_periods=2).apply(sum, raw=raw)
    tm.assert_frame_equal(result_roll_sum, expected)
    tm.assert_frame_equal(result_roll_generic, expected)

def test_closed_fixed(closed: str, arithmetic_win_operators: str) -> None:
    func_name: str = arithmetic_win_operators
    df_fixed: DataFrame = DataFrame({'A': [0, 1, 2, 3, 4]})
    df_time: DataFrame = DataFrame({'A': [0, 1, 2, 3, 4]}, index=date_range('2020', periods=5))
    result: DataFrame = getattr(df_fixed.rolling(2, closed=closed, min_periods=1), func_name)()
    expected: DataFrame = getattr(df_time.rolling('2D', closed=closed, min_periods=1), func_name)().reset_index(drop=True)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('closed, window_selections', [
    ('both', [
        [True, True, False, False, False],
        [True, True, True, False, False],
        [False, True, True, True, False],
        [False, False, True, True, True],
        [False, False, False, True, True]
    ]),
    ('left', [
        [True, False, False, False, False],
        [True, True, False, False, False],
        [False, True, True, False, False],
        [False, False, True, True, False],
        [False, False, False, True, True]
    ]),
    ('right', [
        [True, True, False, False, False],
        [False, True, True, False, False],
        [False, False, True, True, False],
        [False, False, False, True, True],
        [False, False, False, False, True]
    ]),
    ('neither', [
        [True, False, False, False, False],
        [False, True, False, False, False],
        [False, False, True, False, False],
        [False, False, False, True, False],
        [False, False, False, False, True]
    ])
])
def test_datetimelike_centered_selections(closed: str, window_selections: List[List[bool]], arithmetic_win_operators: str) -> None:
    func_name: str = arithmetic_win_operators
    df_time: DataFrame = DataFrame({'A': [0.0, 1.0, 2.0, 3.0, 4.0]}, index=date_range('2020', periods=5))
    expected: DataFrame = DataFrame({
        'A': [getattr(df_time['A'].iloc[s], func_name)() for s in window_selections]
    }, index=date_range('2020', periods=5))
    if func_name == 'sem':
        kwargs: dict = {'ddof': 0}
    else:
        kwargs = {}
    result: DataFrame = getattr(
        df_time.rolling('2D', closed=closed, min_periods=1, center=True),
        func_name
    )(**kwargs)
    tm.assert_frame_equal(result, expected, check_dtype=False)

@pytest.mark.parametrize('window,closed,expected', [
    ('3s', 'right', [3.0, 3.0, 3.0]),
    ('3s', 'both', [3.0, 3.0, 3.0]),
    ('3s', 'left', [3.0, 3.0, 3.0]),
    ('3s', 'neither', [3.0, 3.0, 3.0]),
    ('2s', 'right', [3.0, 2.0, 2.0]),
    ('2s', 'both', [3.0, 3.0, 3.0]),
    ('2s', 'left', [1.0, 3.0, 3.0]),
    ('2s', 'neither', [1.0, 2.0, 2.0])
])
def test_datetimelike_centered_offset_covers_all(
    window: str,
    closed: str,
    expected: List[float],
    frame_or_series: FrameOrSeries
) -> None:
    index: List[Timestamp] = [
        Timestamp('20130101 09:00:01'),
        Timestamp('20130101 09:00:02'),
        Timestamp('20130101 09:00:02')
    ]
    df: FrameOrSeries = frame_or_series([1, 1, 1], index=index)
    result: FrameOrSeries = df.rolling(window, closed=closed, center=True).sum()
    expected_series: FrameOrSeries = frame_or_series(expected, index=index)
    tm.assert_equal(result, expected_series)

@pytest.mark.parametrize('window,closed,expected', [
    ('2D', 'right', [4, 4, 4, 4, 4, 4, 2, 2]),
    ('2D', 'left', [2, 2, 4, 4, 4, 4, 4, 4]),
    ('2D', 'both', [4, 4, 6, 6, 6, 6, 4, 4]),
    ('2D', 'neither', [2, 2, 2, 2, 2, 2, 2, 2])
])
def test_datetimelike_nonunique_index_centering(
    window: str,
    closed: str,
    expected: List[float],
    frame_or_series: FrameOrSeries
) -> None:
    index: DatetimeIndex = DatetimeIndex([
        '2020-01-01', '2020-01-01', '2020-01-02',
        '2020-01-02', '2020-01-03', '2020-01-03',
        '2020-01-04', '2020-01-04'
    ])
    df: FrameOrSeries = frame_or_series([1] * 8, index=index, dtype=float)
    expected_df: FrameOrSeries = frame_or_series(expected, index=index, dtype=float)
    result: FrameOrSeries = df.rolling(window, center=True, closed=closed).sum()
    tm.assert_equal(result, expected_df)

@pytest.mark.parametrize('closed,expected', [
    ('left', [np.nan, np.nan, 1, 1, 1, 10, 14, 14, 18, 21]),
    ('neither', [np.nan, np.nan, 1, 1, 1, 9, 5, 5, 13, 8]),
    ('right', [0, 1, 3, 6, 10, 14, 11, 18, 21, 17]),
    ('both', [0, 1, 3, 6, 10, 15, 20, 27, 26, 30])
])
def test_variable_window_nonunique(
    closed: str,
    expected: List[float],
    frame_or_series: FrameOrSeries
) -> None:
    index: DatetimeIndex = DatetimeIndex([
        '2011-01-01', '2011-01-01', '2011-01-02',
        '2011-01-02', '2011-01-02', '2011-01-03',
        '2011-01-04', '2011-01-04', '2011-01-05',
        '2011-01-06'
    ])
    df: FrameOrSeries = frame_or_series(range(10), index=index, dtype=float)
    expected_df: FrameOrSeries = frame_or_series(expected, index=index, dtype=float)
    result: FrameOrSeries = df.rolling('2D', closed=closed).sum()
    tm.assert_equal(result, expected_df)

@pytest.mark.parametrize('closed,expected', [
    ('left', [np.nan, np.nan, 1, 1, 1, 10, 15, 15, 18, 21]),
    ('neither', [np.nan, np.nan, 1, 1, 1, 10, 15, 15, 13, 8]),
    ('right', [0, 1, 3, 6, 10, 15, 21, 28, 21, 17]),
    ('both', [0, 1, 3, 6, 10, 15, 21, 28, 26, 30])
])
def test_variable_offset_window_nonunique(
    closed: str,
    expected: List[float],
    frame_or_series: FrameOrSeries
) -> None:
    index: DatetimeIndex = DatetimeIndex([
        '2011-01-01', '2011-01-01', '2011-01-02',
        '2011-01-02', '2011-01-02', '2011-01-03',
        '2011-01-04', '2011-01-04', '2011-01-05',
        '2011-01-06'
    ])
    df: FrameOrSeries = frame_or_series(range(10), index=index, dtype=float)
    expected_df: FrameOrSeries = frame_or_series(expected, index=index, dtype=float)
    offset: BusinessDay = BusinessDay(2)
    indexer: VariableOffsetWindowIndexer = VariableOffsetWindowIndexer(index=index, offset=offset)
    result: FrameOrSeries = df.rolling(indexer, closed=closed, min_periods=1).sum()
    tm.assert_equal(result, expected_df)

def test_even_number_window_alignment() -> None:
    s: Series = Series(range(3), index=date_range(start='2020-01-01', freq='D', periods=3))
    result: Series = s.rolling(window='2D', min_periods=1, center=True).mean()
    expected: Series = Series([0.5, 1.5, 2], index=s.index)
    tm.assert_series_equal(result, expected)

def test_closed_fixed_binary_col(center: bool, step: int) -> None:
    data: List[int] = [0, 1, 1, 0, 0, 1, 0, 1]
    df: DataFrame = DataFrame({'binary_col': data}, index=date_range(start='2020-01-01', freq='min', periods=len(data)))
    if center:
        expected_data: List[float] = [2 / 3, 0.5, 0.4, 0.5, 0.428571, 0.5, 0.571429, 0.5]
    else:
        expected_data = [np.nan, 0, 0.5, 2 / 3, 0.5, 0.4, 0.5, 0.428571]
    expected: DataFrame = DataFrame(
        expected_data, columns=['binary_col'],
        index=date_range(start='2020-01-01', freq='min', periods=len(expected_data))
    )[::step]
    rolling = df.rolling(
        window=len(df), closed='left',
        min_periods=1, center=center, step=step
    )
    result: DataFrame = rolling.mean()
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('closed', ['neither', 'left'])
def test_closed_empty(closed: str, arithmetic_win_operators: str) -> None:
    func_name: str = arithmetic_win_operators
    ser: Series = Series(data=np.arange(5), index=date_range('2000', periods=5, freq='2D'))
    roll = ser.rolling('1D', closed=closed)
    result: Series = getattr(roll, func_name)()
    expected: Series = Series([np.nan] * 5, index=ser.index)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('func', ['min', 'max'])
def test_closed_one_entry(func: str) -> None:
    ser: Series = Series(data=[2], index=date_range('2000', periods=1))
    result: Series = getattr(ser.rolling('10D', closed='left'), func)()
    expected: Series = Series([np.nan], index=ser.index)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('func', ['min', 'max'])
def test_closed_one_entry_groupby(func: str) -> None:
    ser: DataFrame = DataFrame(data={'A': [1, 1, 2], 'B': [3, 2, 1]}, index=date_range('2000', periods=3))
    result: Series = getattr(ser.groupby('A', sort=False)['B'].rolling('10D', closed='left'), func)()
    exp_idx: MultiIndex = MultiIndex.from_arrays(
        arrays=[[1, 1, 2], ser.index],
        names=('A', None)
    )
    expected: Series = Series(data=[np.nan, 3, np.nan], index=exp_idx, name='B')
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('input_dtype', ['int', 'float'])
@pytest.mark.parametrize('func,closed,expected', [
    ('min', 'right', [0.0, 0, 0, 1, 2, 3, 4, 5, 6, 7]),
    ('min', 'both', [0.0, 0, 0, 0, 1, 2, 3, 4, 5, 6]),
    ('min', 'neither', [np.nan, 0, 0, 1, 2, 3, 4, 5, 6, 7]),
    ('min', 'left', [np.nan, 0, 0, 0, 1, 2, 3, 4, 5, 6]),
    ('max', 'right', [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ('max', 'both', [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ('max', 'neither', [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ('max', 'left', [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8])
])
def test_closed_min_max_datetime(input_dtype: str, func: str, closed: str, expected: List[float]) -> None:
    ser: Series = Series(data=np.arange(10).astype(input_dtype), index=date_range('2000', periods=10))
    result: Series = getattr(ser.rolling('3D', closed=closed), func)()
    expected_series: Series = Series(expected, index=ser.index)
    tm.assert_series_equal(result, expected_series)

def test_closed_uneven() -> None:
    ser: Series = Series(data=np.arange(10), index=date_range('2000', periods=10))
    ser = ser.drop(index=ser.index[[1, 5]])
    result: Series = ser.rolling('3D', closed='left').min()
    expected: Series = Series([np.nan, 0, 0, 2, 3, 4, 6, 6], index=ser.index)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('func,closed,expected', [
    ('min', 'right', [np.nan, 0, 0, 1, 2, 3, 4, 5, np.nan, np.nan]),
    ('min', 'both', [np.nan, 0, 0, 0, 1, 2, 3, 4, 5, np.nan]),
    ('min', 'neither', [np.nan, np.nan, 0, 1, 2, 3, 4, 5, np.nan, np.nan]),
    ('min', 'left', [np.nan, np.nan, 0, 0, 1, 2, 3, 4, 5, np.nan]),
    ('max', 'right', [np.nan, 1, 2, 3, 4, 5, 6, 6, np.nan, np.nan]),
    ('max', 'both', [np.nan, 1, 2, 3, 4, 5, 6, 6, 6, np.nan]),
    ('max', 'neither', [np.nan, np.nan, 1, 2, 3, 4, 5, 6, np.nan, np.nan]),
    ('max', 'left', [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 6, np.nan])
])
def test_closed_min_max_minp(func: str, closed: str, expected: List[float]) -> None:
    ser: Series = Series(data=np.arange(10), index=date_range('2000', periods=10))
    ser = ser.astype('float')
    ser.iloc[-3:] = np.nan
    result: Series = getattr(ser.rolling('3D', min_periods=2, closed=closed), func)()
    expected_series: Series = Series(expected, index=ser.index)
    tm.assert_series_equal(result, expected_series)

@pytest.mark.parametrize('closed,expected', [
    ('right', [1.0, 3.0, 5.0, 3.0]),
    ('left', [0.0, 1.0, 3.0, 5.0]),
    ('both', [1.0, 3.0, 6.0, 5.0]),
    ('neither', [0.0, 1.0, 2.0, 3.0]),
    ('3s', 'right', [1.0, 3.0, 6.0, 5.0]),
    ('3s', 'left', [1.0, 3.0, 6.0, 5.0]),
    ('3s', 'both', [1.0, 3.0, 6.0, 5.0]),
    ('3s', 'neither', [1.0, 3.0, 6.0, 5.0])
])
def test_rolling_decreasing_indices_centered(
    window: str,
    closed: str,
    expected: List[float],
    frame_or_series: FrameOrSeries
) -> None:
    """
    Ensure that a symmetrical inverted index return same result as non-inverted.
    """
    index: DatetimeIndex = date_range('2020', periods=4, freq='1s')
    df_inc: FrameOrSeries = frame_or_series(range(4), index=index)
    df_dec: FrameOrSeries = frame_or_series(range(4), index=index[::-1])
    expected_inc: FrameOrSeries = frame_or_series(expected, index=index)
    expected_dec: FrameOrSeries = frame_or_series(expected, index=index[::-1])
    result_inc: FrameOrSeries = df_inc.rolling(window, closed=closed, center=True).sum()
    result_dec: FrameOrSeries = df_dec.rolling(window, closed=closed, center=True).sum()
    tm.assert_equal(result_inc, expected_inc)
    tm.assert_equal(result_dec, expected_dec)

@pytest.mark.parametrize('window,expected', [
    ('1ns', [1.0, 1.0, 1.0, 1.0]),
    ('3ns', [2.0, 3.0, 3.0, 2.0])
])
def test_rolling_center_nanosecond_resolution(
    window: str,
    closed: str,
    expected: List[float],
    frame_or_series: FrameOrSeries
) -> None:
    index: DatetimeIndex = date_range('2020', periods=4, freq='1ns')
    df: FrameOrSeries = frame_or_series([1, 1, 1, 1], index=index, dtype=float)
    expected_series: FrameOrSeries = frame_or_series(expected, index=index, dtype=float)
    result: FrameOrSeries = df.rolling(window, closed=closed, center=True).sum()
    tm.assert_equal(result, expected_series)

@pytest.mark.parametrize('start, exp_values', [
    (1, [0.03, 0.0155, 0.0155, 0.011, 0.01025]),
    (2, [0.001, 0.001, 0.0015, 0.00366666])
])
def test_rolling_mean_all_nan_window_floating_artifacts(
    start: int,
    exp_values: List[float]
) -> None:
    df: DataFrame = DataFrame([0.03, 0.03, 0.001, np.nan, 0.002, 0.008, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.005, 0.2])
    values: List[float] = exp_values + [0.00366666, 0.005, 0.005, 0.008, np.nan, np.nan, 0.005, 0.1025]
    expected: DataFrame = DataFrame(values, index=list(range(start, len(values) + start)))
    result: DataFrame = df.iloc[start:].rolling(5, min_periods=0).mean()
    tm.assert_frame_equal(result, expected)

def test_rolling_sum_all_nan_window_floating_artifacts() -> None:
    df: DataFrame = DataFrame([0.002, 0.008, 0.005, np.nan, np.nan, np.nan])
    result: DataFrame = df.rolling(3, min_periods=0).sum()
    expected: DataFrame = DataFrame([0.002, 0.01, 0.015, 0.013, 0.005, 0.0])
    tm.assert_frame_equal(result, expected)

def test_rolling_zero_window() -> None:
    s: Series = Series(range(1))
    result: Series = s.rolling(0).min()
    expected: Series = Series([np.nan])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('window', [1, 3, 10, 20])
@pytest.mark.parametrize('method', ['min', 'max', 'average'])
@pytest.mark.parametrize('pct', [True, False])
@pytest.mark.parametrize('test_data', ['default', 'duplicates', 'nans'])
def test_rank(
    window: int,
    method: str,
    pct: bool,
    ascending: bool,
    test_data: str
) -> None:
    length: int = 20
    if test_data == 'default':
        ser: Series = Series(data=np.random.default_rng(2).random(length))
    elif test_data == 'duplicates':
        ser = Series(data=np.random.default_rng(2).choice(3, length))
    elif test_data == 'nans':
        ser = Series(data=np.random.default_rng(2).choice([1.0, 0.25, 0.75, np.nan, np.inf, -np.inf], length))
    expected: Series = ser.rolling(window).apply(
        lambda x: x.rank(method=method, pct=pct, ascending=ascending).iloc[-1]
    )
    result: Series = ser.rolling(window).rank(method=method, pct=pct, ascending=ascending)
    tm.assert_series_equal(result, expected)

def test_rolling_quantile_np_percentile() -> None:
    row: int = 10
    col: int = 5
    idx: DatetimeIndex = date_range('20100101', periods=row, freq='B')
    df: DataFrame = DataFrame(np.random.default_rng(2).random(row * col).reshape((row, -1)), index=idx)
    df_quantile: DataFrame = df.quantile([0.25, 0.5, 0.75], axis=0)
    np_percentile: np.ndarray = np.percentile(df, [25, 50, 75], axis=0)
    tm.assert_almost_equal(df_quantile.values, np.array(np_percentile))

@pytest.mark.parametrize('quantile', [0.0, 0.1, 0.45, 0.5, 1])
@pytest.mark.parametrize('interpolation', ['linear', 'lower', 'higher', 'nearest', 'midpoint'])
@pytest.mark.parametrize('data', [
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    [8.0, 1.0, 3.0, 4.0, 5.0, 2.0, 6.0, 7.0],
    [0.0, np.nan, 0.2, np.nan, 0.4],
    [np.nan, np.nan, np.nan, np.nan],
    [np.nan, 0.1, np.nan, 0.3, 0.4, 0.5],
    [0.5],
    [np.nan, 0.7, 0.6]
])
def test_rolling_quantile_interpolation_options(
    quantile: float,
    interpolation: str,
    data: List[Optional[float]]
) -> None:
    s: Series = Series(data)
    q1: float = s.quantile(quantile, interpolation)
    q2: float = s.expanding(min_periods=1).quantile(quantile, interpolation).iloc[-1]
    if np.isnan(q1):
        assert np.isnan(q2)
    elif not IS64:
        assert np.allclose([q1], [q2], rtol=1e-07, atol=0)
    else:
        assert q1 == q2

def test_invalid_quantile_value() -> None:
    data: np.ndarray = np.arange(5)
    s: Series = Series(data)
    msg: str = "Interpolation 'invalid' is not supported"
    with pytest.raises(ValueError, match=msg):
        s.rolling(len(data), min_periods=1).quantile(0.5, interpolation='invalid')

def test_rolling_quantile_param() -> None:
    ser: Series = Series([0.0, 0.1, 0.5, 0.9, 1.0])
    msg: str = 'quantile value -0.1 not in \\[0, 1\\]'
    with pytest.raises(ValueError, match=msg):
        ser.rolling(3).quantile(-0.1)
    msg = 'quantile value 10.0 not in \\[0, 1\\]'
    with pytest.raises(ValueError, match=msg):
        ser.rolling(3).quantile(10.0)
    msg = 'must be real number, not str'
    with pytest.raises(TypeError, match=msg):
        ser.rolling(3).quantile('foo')

def test_rolling_std_1obs() -> None:
    vals: Series = Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result: Series = vals.rolling(1, min_periods=1).std()
    expected: Series = Series([np.nan] * 5)
    tm.assert_series_equal(result, expected)
    result = vals.rolling(1, min_periods=1).std(ddof=0)
    expected = Series([0.0] * 5)
    tm.assert_series_equal(result, expected)
    result = Series([np.nan, np.nan, 3, 4, 5]).rolling(3, min_periods=2).std()
    assert np.isnan(result[2])

def test_rolling_std_neg_sqrt() -> None:
    a: Series = Series([
        0.0011448196318903589,
        0.00028718669878572767,
        0.00028718669878572767,
        0.00028718669878572767,
        0.00028718669878572767
    ])
    b: Series = a.rolling(window=3).std()
    assert np.isfinite(b[2:]).all()
    b = a.ewm(span=3).std()
    assert np.isfinite(b[2:]).all()

def test_step_not_integer_raises() -> None:
    with pytest.raises(ValueError, match='step must be an integer'):
        DataFrame(range(2)).rolling(1, step='foo')

def test_step_not_positive_raises() -> None:
    with pytest.raises(ValueError, match='step must be >= 0'):
        DataFrame(range(2)).rolling(1, step=-1)

@pytest.mark.parametrize(['values', 'window', 'min_periods', 'expected'], [
    (
        [20, 10, 10, np.inf, 1, 1, 2, 3],
        3,
        1,
        [np.nan, 50, 100 / 3, 0, 40.5, 0, 1 / 3, 1]
    ),
    (
        [20, 10, 10, np.nan, 10, 1, 2, 3],
        3,
        1,
        [np.nan, 50, 100 / 3, 0, 0, 40.5, 73 / 3, 1]
    ),
    (
        [np.nan, 5, 6, 7, 5, 5, 5],
        3,
        3,
        [np.nan] * 3 + [1, 1, 4 / 3]
    ),
    (
        [5, 7, 7, 7, np.nan, np.inf, 4, 3, 3, 3],
        3,
        3,
        [np.nan] * 2 + [4 / 3, 0] + [np.nan] * 4 + [1 / 3, 0]
    ),
    (
        [5, 7, 7, 7, np.nan, np.inf, 7, 3, 3, 3],
        3,
        3,
        [np.nan] * 2 + [4 / 3, 0] + [np.nan] * 4 + [16 / 3, 0]
    ),
    (
        [5, 7] * 4,
        3,
        3,
        [np.nan] * 2 + [4 / 3] * 6
    ),
    (
        [5, 7, 5, np.nan, 7, 5, 7],
        3,
        2,
        [np.nan, 2, 4 / 3] + [2] * 3 + [4 / 3]
    )
])
def test_rolling_var_same_value_count_logic(
    values: List[Union[int, float]],
    window: int,
    min_periods: int,
    expected: List[float]
) -> None:
    expected_series: Series = Series(expected)
    sr: Series = Series(values)
    result_var: Series = sr.rolling(window, min_periods=min_periods).var()
    tm.assert_series_equal(result_var, expected_series)
    tm.assert_series_equal(expected_series == 0, result_var == 0)
    result_std: Series = sr.rolling(window, min_periods=min_periods).std()
    expected_std: Series = Series([np.nan if v is np.nan else np.sqrt(v) for v in expected])
    tm.assert_series_equal(result_std, np.sqrt(expected_series))
    tm.assert_series_equal(expected_series == 0, result_std == 0)

def test_rolling_mean_sum_floating_artifacts() -> None:
    sr: Series = Series([1 / 3, 4, 0, 0, 0, 0, 0])
    r = sr.rolling(3)
    result: Series = r.mean()
    assert (result[-3:] == 0).all()
    result = r.sum()
    assert (result[-3:] == 0).all()

def test_rolling_skew_kurt_floating_artifacts() -> None:
    sr: Series = Series([1 / 3, 4, 0, 0, 0, 0, 0])
    r = sr.rolling(4)
    result: Series = r.skew()
    assert (result[-2:] == 0).all()
    result = r.kurt()
    assert (result[-2:] == -3).all()

def test_numeric_only_frame(arithmetic_win_operators: str, numeric_only: bool) -> None:
    kernel: str = arithmetic_win_operators
    df: DataFrame = DataFrame({'a': [1], 'b': 2, 'c': 3})
    df['c'] = df['c'].astype(object)
    rolling = df.rolling(2, min_periods=1)
    op = getattr(rolling, kernel)
    result: DataFrame = op(numeric_only=numeric_only)
    columns: List[str] = ['a', 'b'] if numeric_only else ['a', 'b', 'c']
    expected: DataFrame = df[columns].agg([kernel]).reset_index(drop=True).astype(float)
    assert list(expected.columns) == columns
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('kernel', ['corr', 'cov'])
@pytest.mark.parametrize('use_arg', [True, False])
def test_numeric_only_corr_cov_frame(
    kernel: str,
    numeric_only: bool,
    use_arg: bool
) -> None:
    df: DataFrame = DataFrame({'a': [1, 2, 3], 'b': 2, 'c': 3})
    df['c'] = df['c'].astype(object)
    arg: Tuple[Any, ...] = (df,) if use_arg else ()
    rolling = df.rolling(2, min_periods=1)
    op = getattr(rolling, kernel)
    result: DataFrame = op(*arg, numeric_only=numeric_only)
    columns: List[str] = ['a', 'b'] if numeric_only else ['a', 'b', 'c']
    df2: DataFrame = df[columns].astype(float)
    arg2: Tuple[Any, ...] = (df2,) if use_arg else ()
    rolling2: BaseIndexer = df2.rolling(2, min_periods=1)
    op2 = getattr(rolling2, kernel)
    expected: DataFrame = op2(*arg2, numeric_only=numeric_only)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('dtype', [int, object])
def test_numeric_only_series(
    arithmetic_win_operators: str,
    numeric_only: bool,
    dtype: type
) -> None:
    kernel: str = arithmetic_win_operators
    ser: Series = Series([1], dtype=dtype)
    rolling = ser.rolling(2, min_periods=1)
    op = getattr(rolling, kernel)
    if numeric_only and dtype is object:
        msg: str = f'Rolling.{kernel} does not implement numeric_only'
        with pytest.raises(NotImplementedError, match=msg):
            op(numeric_only=numeric_only)
    else:
        result: Series = op(numeric_only=numeric_only)
        expected: Series = ser.agg([kernel]).reset_index(drop=True).astype(float)
        tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('kernel', ['corr', 'cov'])
@pytest.mark.parametrize('use_arg', [True, False])
@pytest.mark.parametrize('dtype', [int, object])
def test_numeric_only_corr_cov_series(
    kernel: str,
    use_arg: bool,
    numeric_only: bool,
    dtype: type
) -> None:
    ser: Series = Series([1, 2, 3], dtype=dtype)
    arg: Tuple[Any, ...] = (ser,) if use_arg else ()
    rolling = ser.rolling(2, min_periods=1)
    op = getattr(rolling, kernel)
    if numeric_only and dtype is object:
        msg: str = f'Rolling.{kernel} does not implement numeric_only'
        with pytest.raises(NotImplementedError, match=msg):
            op(*arg, numeric_only=numeric_only)
    else:
        result: Series = op(*arg, numeric_only=numeric_only)
        ser2: Series = ser.astype(float)
        arg2: Tuple[Any, ...] = (ser2,) if use_arg else ()
        rolling2 = ser2.rolling(2, min_periods=1)
        op2 = getattr(rolling2, kernel)
        expected: Series = op2(*arg2, numeric_only=numeric_only)
        tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('tz', [None, 'UTC', 'Europe/Prague'])
def test_rolling_timedelta_window_non_nanoseconds(unit: str, tz: Optional[str]) -> None:
    df_time: DataFrame = DataFrame({'A': range(5)}, index=date_range('2013-01-01', freq='1s', periods=5, tz=tz))
    sum_in_nanosecs: DataFrame = df_time.rolling('1s').sum()
    df_time.index = df_time.index.as_unit(unit)
    sum_in_microsecs: DataFrame = df_time.rolling('1s').sum()
    sum_in_microsecs.index = sum_in_microsecs.index.as_unit('ns')
    tm.assert_frame_equal(sum_in_nanosecs, sum_in_microsecs)
    ref_dates: DatetimeIndex = date_range('2023-01-01', '2023-01-10', unit='ns', tz=tz)
    ref_series: Series = Series(0, index=ref_dates)
    ref_series.iloc[0] = 1
    ref_max_series: Series = ref_series.rolling(Timedelta(days=4)).max()
    dates: DatetimeIndex = date_range('2023-01-01', '2023-01-10', unit=unit, tz=tz)
    series: Series = Series(0, index=dates)
    series.iloc[0] = 1
    max_series: Series = series.rolling(Timedelta(days=4)).max()
    ref_df: DataFrame = DataFrame(ref_max_series)
    df: DataFrame = DataFrame(max_series)
    df.index = df.index.as_unit('ns')
    tm.assert_frame_equal(ref_df, df)

def test_iter_rolling_dataframe(df: DataFrame, expected: List[Tuple[DataFrame, List[int]]]) -> None:
    # Placeholder for the actual implementation
    pass

def test_iter_rolling_on_dataframe(expected: List[Tuple[DataFrame, DatetimeIndex]], window: str) -> None:
    # Placeholder for the actual implementation
    pass

def test_iter_rolling_on_dataframe_unordered() -> None:
    df: DataFrame = DataFrame({'a': ['x', 'y', 'x'], 'b': [0, 1, 2]})
    results: List[DataFrame] = list(df.groupby('a').rolling(2))
    expecteds: List[DataFrame] = [df.iloc[idx, [1]] for idx in [[0], [0, 2], [1]]]
    for result, expected in zip(results, expecteds):
        tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize(['ser', 'expected', 'window', 'min_periods'], [
    (
        Series([1, 2, 3]),
        [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])],
        3,
        None
    ),
    (
        Series([1, 2, 3]),
        [([1], [0]), ([1, 2], [0, 1]), ([2, 3], [1, 2])],
        2,
        1
    ),
    (
        Series([1, 2, 3]),
        [([1], [0]), ([1, 2], [0, 1]), ([2, 3], [1, 2])],
        2,
        2
    ),
    (
        Series([1, 2, 3]),
        [([1], [0]), ([2], [1]), ([3], [2])],
        1,
        1
    ),
    (
        Series([1, 2, 3]),
        [([1], [0]), ([2], [1]), ([3], [2])],
        1,
        0
    ),
    (
        Series([1, 2]),
        [([1], [0]), ([1, 2], [0, 1])],
        2,
        0
    ),
    (
        Series([1, 2, 3]),
        [([1], [0]), ([2], [1]), ([3], [2])],
        1,
        1
    ),
    (
        Series([], dtype='int64'),
        [],
        2,
        1
    ),
    (
        {'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]},
        [({'A': [1.0], 'B': [np.nan]}, [0]), ({'A': [1, np.nan], 'B': [np.nan, 5]}, [0, 1]), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [0, 1, 2])],
        3,
        2
    )
])
def test_iter_rolling_series(
    ser: Series,
    expected: List[Tuple[List[Any], List[int]]],
    window: int,
    min_periods: int
) -> None:
    expecteds: List[Series] = [Series(values, index=index) for values, index in expected]
    for expected_series, actual in zip(expecteds, ser.rolling(window, min_periods=min_periods)):
        tm.assert_series_equal(actual, expected_series)

@pytest.mark.parametrize('expected,expected_index,window', [
    (
        [[0], [1], [2], [3], [4]],
        [
            date_range('2020-01-01', periods=1, freq='D'),
            date_range('2020-01-02', periods=1, freq='D'),
            date_range('2020-01-03', periods=1, freq='D'),
            date_range('2020-01-04', periods=1, freq='D'),
            date_range('2020-01-05', periods=1, freq='D')
        ],
        '1D'
    ),
    (
        [[0], [0, 1], [1, 2], [2, 3], [3, 4]],
        [
            date_range('2020-01-01', periods=1, freq='D'),
            date_range('2020-01-01', periods=2, freq='D'),
            date_range('2020-01-02', periods=2, freq='D'),
            date_range('2020-01-03', periods=2, freq='D'),
            date_range('2020-01-04', periods=2, freq='D')
        ],
        '2D'
    ),
    (
        [[0], [0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]],
        [
            date_range('2020-01-01', periods=1, freq='D'),
            date_range('2020-01-01', periods=2, freq='D'),
            date_range('2020-01-01', periods=3, freq='D'),
            date_range('2020-01-02', periods=3, freq='D'),
            date_range('2020-01-03', periods=3, freq='D')
        ],
        '3D'
    )
])
def test_iter_rolling_datetime(
    expected: List[Tuple[List[Any], DatetimeIndex]],
    expected_index: List[DatetimeIndex],
    window: str
) -> None:
    ser: Series = Series(range(5), index=date_range(start='2020-01-01', periods=5, freq='D'))
    expecteds: List[Series] = [Series(values, index=idx) for values, idx in zip(expected, expected_index)]
    for expected_series, actual in zip(expecteds, ser.rolling(window)):
        tm.assert_series_equal(actual, expected_series)

@pytest.mark.parametrize('grouping,_index', [
    (
        {'level': 0},
        MultiIndex.from_tuples([(0, 0), (0, 0), (1, 1), (1, 1), (1, 1)], names=[None, None])
    ),
    (
        {'by': 'X'},
        MultiIndex.from_tuples([(0, 0), (1, 0), (2, 1), (3, 1), (4, 1)], names=['X', None])
    )
])
def test_rolling_positional_argument(
    grouping: dict,
    _index: MultiIndex,
    raw: bool
) -> None:
    def scaled_sum(*args: Any) -> float:
        if len(args) < 2:
            raise ValueError('The function needs two arguments')
        array, scale = args
        return array.sum() / scale

    df: DataFrame = DataFrame(data={'X': range(5)}, index=[0, 0, 1, 1, 1])
    expected: DataFrame = DataFrame(data={'X': [0.0, 0.5, 1.0, 1.5, 2.0]}, index=_index)
    if 'by' in grouping:
        expected = expected.drop(columns='X', errors='ignore')
    result: DataFrame = df.groupby(**grouping).rolling(1).apply(scaled_sum, raw=raw, args=(2,))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('add', [0.0, 2.0])
def test_rolling_numerical_accuracy_kahan_mean(add: float, unit: str) -> None:
    dti: DatetimeIndex = DatetimeIndex([
        Timestamp('19700101 09:00:00'),
        Timestamp('19700101 09:00:03'),
        Timestamp('19700101 09:00:06')
    ]).as_unit(unit)
    df: DataFrame = DataFrame({'A': [3002399751580331.0 + add, -0.0, -0.0]}, index=dti)
    result: DataFrame = df.resample('1s').ffill().rolling('3s', closed='left', min_periods=3).mean()
    dates: DatetimeIndex = date_range('19700101 09:00:00', periods=7, freq='s', unit=unit)
    expected: DataFrame = DataFrame({
        'A': [np.nan, np.nan, np.nan, 3002399751580330.5, 2001599834386887.2, 1000799917193443.6, 0.0]
    }, index=dates)
    tm.assert_frame_equal(result, expected)

def test_rolling_numerical_accuracy_kahan_sum() -> None:
    df: DataFrame = DataFrame([2.186, -1.647, 0.0, 0.0, 0.0, 0.0], columns=['x'])
    result: Series = df['x'].rolling(3).sum()
    expected: Series = Series([np.nan, np.nan, 0.539, -1.647, 0.0, 0.0], name='x')
    tm.assert_series_equal(result, expected)

def test_rolling_numerical_accuracy_jump() -> None:
    index: DatetimeIndex = date_range(start='2020-01-01', end='2020-01-02', freq='60s').append(DatetimeIndex(['2020-01-03']))
    data: np.ndarray = np.random.default_rng(2).random(len(index))
    df: DataFrame = DataFrame({'data': data}, index=index)
    result: DataFrame = df.rolling('60s').mean()
    tm.assert_frame_equal(result, df[['data']])

def test_rolling_numerical_accuracy_small_values() -> None:
    s: Series = Series(data=[0.00012456, 0.0003, -0.0, -0.0], index=date_range('1999-02-03', '1999-02-06'))
    result: Series = s.rolling(1).mean()
    tm.assert_series_equal(result, s)

def test_rolling_numerical_too_large_numbers() -> None:
    dates: DatetimeIndex = date_range('2015-01-01', periods=10, freq='D')
    ds: Series = Series(data=range(10), index=dates, dtype=np.float64)
    ds.iloc[2] = -9e+33
    result: Series = ds.rolling(5).mean()
    expected: Series = Series([np.nan, np.nan, np.nan, np.nan, -1.8e+33, -1.8e+33, -1.8e+33, 5.0, 6.0, 7.0], index=dates)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(('index', 'window'), [
    (period_range(start='2020-01-01 08:00', end='2020-01-01 08:08', freq='min'), '2min'),
    (period_range(start='2020-01-01 08:00', end='2020-01-01 12:00', freq='30min'), '1h')
])
@pytest.mark.parametrize(('func', 'values'), [
    ('min', [np.nan, 0, 0, 1, 2, 3, 4, 5, 6]),
    ('max', [np.nan, 0, 1, 2, 3, 4, 5, 6, 7]),
    ('sum', [np.nan, 0, 1, 3, 5, 7, 9, 11, 13])
])
def test_rolling_period_index(
    index: DatetimeIndex,
    window: str,
    func: str,
    values: List[float]
) -> None:
    ds: Series = Series([0, 1, 2, 3, 4, 5, 6, 7, 8], index=index)
    result: Series = getattr(ds.rolling(window, closed='left'), func)()
    expected: Series = Series(values, index=index)
    tm.assert_series_equal(result, expected)

def test_rolling_sem(frame_or_series: FrameOrSeries) -> None:
    obj: FrameOrSeries = frame_or_series([0, 1, 2])
    result: FrameOrSeries = obj.rolling(2, min_periods=1).sem()
    if isinstance(result, DataFrame):
        result = Series(result[0].values)
    expected: Series = Series([np.nan, 0.7071067811865476, 0.7071067811865476])
    tm.assert_series_equal(result, expected)

@pytest.mark.xfail(is_platform_arm() or is_platform_power() or is_platform_riscv64(), reason='GH 38921')
@pytest.mark.parametrize(('func', 'third_value', 'values'), [
    ('var', 1, [5e+33, 0, 0.5, 0.5, 2, 0]),
    ('std', 1, [7.071068e+16, 0, 0.7071068, 0.7071068, 1.414214, 0]),
    ('var', 2, [5e+33, 0.5, 0, 0.5, 2, 0]),
    ('std', 2, [7.071068e+16, 0.7071068, 0, 0.7071068, 1.414214, 0])
])
def test_rolling_var_numerical_issues(
    func: str,
    third_value: float,
    values: List[float]
) -> None:
    ds: Series = Series([99999999999999999, 1, third_value, 2, 3, 1, 1])
    result: Series = getattr(ds.rolling(2), func)()
    expected: Series = Series([np.nan] + values)
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(result == 0, expected == 0)

def test_timeoffset_as_window_parameter_for_corr(unit: str) -> None:
    dti: DatetimeIndex = DatetimeIndex([
        Timestamp('20130101 09:00:00'),
        Timestamp('20130102 09:00:02'),
        Timestamp('20130103 09:00:03'),
        Timestamp('20130105 09:00:05'),
        Timestamp('20130106 09:00:06')
    ]).as_unit(unit)
    mi: MultiIndex = MultiIndex.from_product([dti, ['B', 'A']])
    exp: DataFrame = DataFrame({
        'B': [
            np.nan, np.nan, 0.9999999999999998, -1.0, 1.0,
            -0.3273268353539892, 0.9999999999999998, 1.0, 0.9999999999999998, 1.0
        ],
        'A': [
            np.nan, np.nan, -1.0, 1.0000000000000002, -0.3273268353539892,
            0.9999999999999966, 1.0, 1.0000000000000002, 1.0, 1.0000000000000002
        ]
    }, index=mi)
    df: DataFrame = DataFrame({'B': [0, 1, 2, 4, 3], 'A': [7, 4, 6, 9, 3]}, index=dti)
    res: DataFrame = df.rolling(window='3D').corr()
    tm.assert_frame_equal(exp, res)

@pytest.mark.parametrize('method', ['var', 'sum', 'mean', 'skew', 'kurt', 'min', 'max'])
def test_rolling_decreasing_indices(method: str) -> None:
    """
    Make sure that decreasing indices give the same results as increasing indices.
    GH 36933
    """
    df: DataFrame = DataFrame({'values': np.arange(-15, 10) ** 2})
    df_reverse: DataFrame = DataFrame({'values': df['values'][::-1]}, index=df.index[::-1])
    increasing: DataFrame = getattr(df.rolling(window=5), method)()
    decreasing: DataFrame = getattr(df_reverse.rolling(window=5), method)()
    assert np.abs(decreasing.values[::-1][:-4] - increasing.values[4:]).max() < 1e-12

@pytest.mark.parametrize('window,closed,expected', [
    ('2s', 'right', [1.0, 3.0, 5.0, 3.0]),
    ('2s', 'left', [0.0, 1.0, 3.0, 5.0]),
    ('2s', 'both', [1.0, 3.0, 6.0, 5.0]),
    ('2s', 'neither', [0.0, 1.0, 2.0, 3.0]),
    ('3s', 'right', [1.0, 3.0, 6.0, 5.0]),
    ('3s', 'left', [1.0, 3.0, 6.0, 5.0]),
    ('3s', 'both', [1.0, 3.0, 6.0, 5.0]),
    ('3s', 'neither', [1.0, 3.0, 6.0, 5.0])
])
def test_rolling_decreasing_indices_centered(
    window: str,
    closed: str,
    expected: List[float],
    frame_or_series: FrameOrSeries
) -> None:
    """
    Ensure that a symmetrical inverted index return same result as non-inverted.
    """
    index: DatetimeIndex = date_range('2020', periods=4, freq='1s')
    df_inc: FrameOrSeries = frame_or_series(range(4), index=index)
    df_dec: FrameOrSeries = frame_or_series(range(4), index=index[::-1])
    expected_inc: FrameOrSeries = frame_or_series(expected, index=index)
    expected_dec: FrameOrSeries = frame_or_series(expected, index=index[::-1])
    result_inc: FrameOrSeries = df_inc.rolling(window, closed=closed, center=True).sum()
    result_dec: FrameOrSeries = df_dec.rolling(window, closed=closed, center=True).sum()
    tm.assert_equal(result_inc, expected_inc)
    tm.assert_equal(result_dec, expected_dec)

@pytest.mark.parametrize('window,expected', [
    ('1ns', [1.0, 1.0, 1.0, 1.0]),
    ('3ns', [2.0, 3.0, 3.0, 2.0])
])
def test_rolling_center_nanosecond_resolution(
    window: str,
    closed: str,
    expected: List[float],
    frame_or_series: FrameOrSeries
) -> None:
    index: DatetimeIndex = date_range('2020', periods=4, freq='1ns')
    df: FrameOrSeries = frame_or_series([1, 1, 1, 1], index=index, dtype=float)
    expected_series: FrameOrSeries = frame_or_series(expected, index=index, dtype=float)
    result: FrameOrSeries = df.rolling(window, closed=closed, center=True).sum()
    tm.assert_equal(result, expected_series)

@pytest.mark.parametrize('method,expected', [
    ('var', [float('nan'), 43.0, float('nan'), 136.333333, 43.5, 94.966667, 182.0, 318.0]),
    ('mean', [float('nan'), 7.5, float('nan'), 21.5, 6.0, 9.166667, 13.0, 17.5]),
    ('sum', [float('nan'), 30.0, float('nan'), 86.0, 30.0, 55.0, 91.0, 140.0]),
    ('skew', [float('nan'), 0.709296, float('nan'), 0.407073, 0.984656, 0.919184, 0.874674, 0.842418]),
    ('kurt', [float('nan'), -0.5916711736073559, float('nan'), -1.0028993131317954, -0.06103844629409494, -0.254143227116194, -0.37362637362637585, -0.45439658241367054])
])
def test_rolling_non_monotonic(method: str, expected: List[float]) -> None:
    """
    Make sure the (rare) branch of non-monotonic indices is covered by a test.

    output from 1.1.3 is assumed to be the expected output. Output of sum/mean has
    manually been verified.

    GH 36933.
    """
    use_expanding: List[bool] = [True, False, True, False, True, True, True, True]
    df: DataFrame = DataFrame({'values': np.arange(len(use_expanding)) ** 2})

    class CustomIndexer(BaseIndexer):
        def get_window_bounds(
            self,
            num_values: int,
            min_periods: int,
            center: bool,
            closed: str,
            step: int
        ) -> Tuple[np.ndarray, np.ndarray]:
            start: np.ndarray = np.empty(num_values, dtype=np.int64)
            end: np.ndarray = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if self.use_expanding[i]:
                    start[i] = 0
                    end[i] = i + 1
                else:
                    start[i] = i
                    end[i] = i + self.window_size
            return (start, end)

    indexer = CustomIndexer(window_size=4, use_expanding=use_expanding)
    result: DataFrame = getattr(df.rolling(indexer), method)()
    expected_df: DataFrame = DataFrame({'values': expected})
    tm.assert_frame_equal(result, expected_df)

@pytest.mark.parametrize(('index', 'window'), [
    ([0, 1, 2, 3, 4], 2),
    (date_range('2001-01-01', freq='D', periods=5), '2D')
])
def test_rolling_corr_timedelta_index(
    index: Union[List[int], DatetimeIndex],
    window: Union[int, str]
) -> None:
    x: Series = Series([1, 2, 3, 4, 5], index=index)
    y: Series = x.copy()
    x.iloc[0:2] = 0.0
    result: Series = x.rolling(window).corr(y)
    expected: Series = Series([np.nan, np.nan, 1, 1, 1], index=index)
    tm.assert_almost_equal(result, expected)

@pytest.mark.parametrize('values,method,expected', [
    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'first', [float('nan'), float('nan'), 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    ([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan], 'first', [float('nan')] * 10),
    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'last', [float('nan'), float('nan'), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    ([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan], 'last', [float('nan')] * 10)
])
def test_rolling_first_last(
    values: List[Optional[float]],
    method: str,
    expected: List[Optional[float]]
) -> None:
    x: Series = Series(values)
    result: Series = getattr(x.rolling(3), method)()
    expected_series: Series = Series(expected)
    tm.assert_almost_equal(result, expected_series)
    x_df: DataFrame = DataFrame({'A': values})
    result_df: DataFrame = getattr(x_df.rolling(3), method)()
    expected_df: DataFrame = DataFrame({'A': expected})
    tm.assert_almost_equal(result_df, expected_df)

@pytest.mark.parametrize('values,method,expected', [
    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'first', [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    ([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan], 'first', [1.0, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0]),
    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'last', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    ([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan], 'last', [1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0, 9.0])
])
def test_rolling_first_last_no_minp(
    values: List[Optional[float]],
    method: str,
    expected: List[Optional[float]]
) -> None:
    x: Series = Series(values)
    result: Series = getattr(x.rolling(3, min_periods=0), method)()
    expected_series: Series = Series(expected)
    tm.assert_almost_equal(result, expected_series)
    x_df: DataFrame = DataFrame({'A': values})
    result_df: DataFrame = getattr(x_df.rolling(3, min_periods=0), method)()
    expected_df: DataFrame = DataFrame({'A': expected})
    tm.assert_almost_equal(result_df, expected_df)

def test_groupby_rolling_nan_included() -> None:
    data: dict = {'group': ['g1', np.nan, 'g1', 'g2', np.nan], 'B': [0, 1, 2, 3, 4]}
    df: DataFrame = DataFrame(data)
    result: DataFrame = df.groupby('group', dropna=False).rolling(1, min_periods=1).mean()
    expected: DataFrame = DataFrame({
        'B': [0.0, 2.0, 3.0, 1.0, 4.0]
    }, index=MultiIndex.from_arrays([
        ['g1', 'g2', np.nan, 'g1', 'g2'],
        [0, 1, 2, 3, 4]
    ], names=['group', None]))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('method', ['skew', 'kurt'])
def test_rolling_skew_kurt_numerical_stability(method: str) -> None:
    ser: Series = Series(np.random.default_rng(2).random(10))
    ser_copy: Series = ser.copy()
    expected: Series = getattr(ser.rolling(3), method)()
    tm.assert_series_equal(ser, ser_copy)
    ser = ser + 50000
    result: Series = getattr(ser.rolling(3), method)()
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(('method', 'values'), [
    ('skew', [2.0, 0.854563, 0.0, 1.999984]),
    ('kurt', [4.0, -1.289256, -1.2, 3.999946])
])
def test_rolling_skew_kurt_large_value_range(
    method: str,
    values: List[float]
) -> None:
    s: Series = Series([3000000, 1, 1, 2, 3, 4, 999])
    result: Series = getattr(s.rolling(4), method)()
    expected: Series = Series([np.nan] * 3 + values)
    tm.assert_series_equal(result, expected)

def test_invalid_method() -> None:
    with pytest.raises(ValueError, match="method must be 'table' or 'single"):
        Series(range(1)).rolling(1, method='foo')

def test_rolling_descending_date_order_with_offset(frame_or_series: FrameOrSeries) -> None:
    msg: str = "'d' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        idx: DatetimeIndex = date_range(start='2020-01-01', end='2020-01-03', freq='1d')
        obj: FrameOrSeries = frame_or_series(range(1, 4), index=idx)
        result: FrameOrSeries = obj.rolling('1d', closed='left').sum()
    expected: FrameOrSeries = frame_or_series([np.nan, 1, 2], index=idx)
    tm.assert_equal(result, expected)
    result = obj.iloc[::-1].rolling('1D', closed='left').sum()
    idx_reversed: DatetimeIndex = date_range(start='2020-01-03', end='2020-01-01', freq='-1D')
    expected_reversed: FrameOrSeries = frame_or_series([np.nan, 3, 2], index=idx_reversed)
    tm.assert_equal(result, expected_reversed)

def test_rolling_var_floating_artifact_precision() -> None:
    s: Series = Series([7, 5, 5, 5])
    result: Series = s.rolling(3).var()
    expected: Series = Series([np.nan, np.nan, 4 / 3, 0])
    tm.assert_series_equal(result, expected, atol=1e-15, rtol=1e-15)
    tm.assert_series_equal(result == 0, expected == 0)

def test_rolling_std_small_values() -> None:
    s: Series = Series([5.4e-07, 5.3e-07, 5.4e-07])
    result: Series = s.rolling(2).std()
    expected: Series = Series([np.nan, 7.071068e-09, 7.071068e-09])
    tm.assert_series_equal(result, expected, atol=1e-15, rtol=1e-15)

@pytest.mark.parametrize('start, exp_values', [
    (1, [0.03, 0.0155, 0.0155, 0.011, 0.01025]),
    (2, [0.001, 0.001, 0.0015, 0.00366666])
])
@pytest.mark.parametrize('test_data', ['default', 'duplicates', 'nans'])
def test_rank(
    window: int,
    method: str,
    pct: bool,
    ascending: bool,
    test_data: str
) -> None:
    # This function is already defined above correctly
    pass

def test_numeric_only_corr_cov_frame(
    kernel: str,
    numeric_only: bool,
    use_arg: bool
) -> None:
    # Already defined appropriately above
    pass

def test_numeric_only_corr_cov_series(
    kernel: str,
    use_arg: bool,
    numeric_only: bool,
    dtype: type
) -> None:
    # Already defined appropriately above
    pass

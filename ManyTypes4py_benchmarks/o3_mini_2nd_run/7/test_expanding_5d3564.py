#!/usr/bin/env python3
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
import numpy as np
import pytest
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex, Series, isna, notna
import pandas._testing as tm

FrameOrSeriesMaker = Callable[[Iterable[Any]], Union[DataFrame, Series]]

def test_doc_string() -> None:
    df = DataFrame({'B': [0, 1, 2, np.nan, 4]})
    df
    df.expanding(2).sum()

def test_constructor(frame_or_series: FrameOrSeriesMaker) -> None:
    c = frame_or_series(range(5)).expanding
    c(min_periods=1)

@pytest.mark.parametrize('w', [2.0, 'foo', np.array([2])])
def test_constructor_invalid(frame_or_series: FrameOrSeriesMaker, w: Union[float, str, np.ndarray]) -> None:
    c = frame_or_series(range(5)).expanding
    msg = 'min_periods must be an integer'
    with pytest.raises(ValueError, match=msg):
        c(min_periods=w)

@pytest.mark.parametrize('expander', [2, pytest.param('ls', marks=pytest.mark.xfail(reason='GH#16425 expanding with offset not supported'))])
def test_empty_df_expanding(expander: Union[int, str]) -> None:
    expected = DataFrame()
    result = DataFrame().expanding(expander).sum()
    tm.assert_frame_equal(result, expected)
    expected = DataFrame(index=DatetimeIndex([]))
    result = DataFrame(index=DatetimeIndex([])).expanding(expander).sum()
    tm.assert_frame_equal(result, expected)

def test_missing_minp_zero() -> None:
    x = Series([np.nan])
    result = x.expanding(min_periods=0).sum()
    expected = Series([0.0])
    tm.assert_series_equal(result, expected)
    result = x.expanding(min_periods=1).sum()
    expected = Series([np.nan])
    tm.assert_series_equal(result, expected)

def test_expanding() -> None:
    df = DataFrame(np.ones((10, 20)))
    expected = DataFrame({i: [np.nan] * 2 + [float(j) for j in range(3, 11)] for i in range(20)})
    result = df.expanding(3).sum()
    tm.assert_frame_equal(result, expected)

def test_expanding_count_with_min_periods(frame_or_series: FrameOrSeriesMaker) -> None:
    result = frame_or_series(range(5)).expanding(min_periods=3).count()
    expected = frame_or_series([np.nan, np.nan, 3.0, 4.0, 5.0])
    tm.assert_equal(result, expected)

def test_expanding_count_default_min_periods_with_null_values(frame_or_series: FrameOrSeriesMaker) -> None:
    values = [1, 2, 3, np.nan, 4, 5, 6]
    expected_counts = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0]
    result = frame_or_series(values).expanding().count()
    expected = frame_or_series(expected_counts)
    tm.assert_equal(result, expected)

def test_expanding_count_with_min_periods_exceeding_series_length(frame_or_series: FrameOrSeriesMaker) -> None:
    result = frame_or_series(range(5)).expanding(min_periods=6).count()
    expected = frame_or_series([np.nan, np.nan, np.nan, np.nan, np.nan])
    tm.assert_equal(result, expected)

@pytest.mark.parametrize(
    'df,expected,min_periods',
    [
        (
            {'A': [1, 2, 3], 'B': [4, 5, 6]},
            [
                ({'A': [1], 'B': [4]}, [0]),
                ({'A': [1, 2], 'B': [4, 5]}, [0, 1]),
                ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [0, 1, 2])
            ],
            3
        ),
        (
            {'A': [1, 2, 3], 'B': [4, 5, 6]},
            [
                ({'A': [1], 'B': [4]}, [0]),
                ({'A': [1, 2], 'B': [4, 5]}, [0, 1]),
                ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [0, 1, 2])
            ],
            2
        ),
        (
            {'A': [1, 2, 3], 'B': [4, 5, 6]},
            [
                ({'A': [1], 'B': [4]}, [0]),
                ({'A': [1, 2], 'B': [4, 5]}, [0, 1]),
                ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [0, 1, 2])
            ],
            1
        ),
        (
            {'A': [1], 'B': [4]},
            [],
            2
        ),
        (
            None,
            [({}, [])],
            1
        ),
        (
            {'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]},
            [
                ({'A': [1.0], 'B': [np.nan]}, [0]),
                ({'A': [1, np.nan], 'B': [np.nan, 5]}, [0, 1]),
                ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [0, 1, 2])
            ],
            3
        ),
        (
            {'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]},
            [
                ({'A': [1.0], 'B': [np.nan]}, [0]),
                ({'A': [1, np.nan], 'B': [np.nan, 5]}, [0, 1]),
                ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [0, 1, 2])
            ],
            2
        ),
        (
            {'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]},
            [
                ({'A': [1.0], 'B': [np.nan]}, [0]),
                ({'A': [1, np.nan], 'B': [np.nan, 5]}, [0, 1]),
                ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [0, 1, 2])
            ],
            1
        ),
    ]
)
def test_iter_expanding_dataframe(
    df: Union[Dict[str, List[Any]], None],
    expected: List[Tuple[Dict[str, List[Any]], List[Any]]],
    min_periods: int
) -> None:
    df_obj = DataFrame(df) if df is not None else DataFrame()
    expecteds = [DataFrame(values, index=index) for values, index in expected]
    for exp, actual in zip(expecteds, df_obj.expanding(min_periods)):
        tm.assert_frame_equal(actual, exp)

@pytest.mark.parametrize(
    'ser,expected,min_periods',
    [
        (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 3),
        (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 2),
        (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 1),
        (Series([1, 2]), [([1], [0]), ([1, 2], [0, 1])], 2),
        (Series([np.nan, 2]), [([np.nan], [0]), ([np.nan, 2], [0, 1])], 2),
        (Series([], dtype='int64'), [], 2)
    ]
)
def test_iter_expanding_series(
    ser: Series,
    expected: List[Tuple[List[Any], List[Any]]],
    min_periods: int
) -> None:
    expecteds = [Series(values, index=index) for values, index in expected]
    for exp, actual in zip(expecteds, ser.expanding(min_periods)):
        tm.assert_series_equal(actual, exp)

def test_center_invalid() -> None:
    df = DataFrame()
    with pytest.raises(TypeError, match='.* got an unexpected keyword'):
        df.expanding(center=True)

def test_expanding_sem(frame_or_series: FrameOrSeriesMaker) -> None:
    obj = frame_or_series([0, 1, 2])
    result = obj.expanding().sem()
    if isinstance(result, DataFrame):
        result = Series(result[0].values)
    expected = Series([np.nan] + [0.707107] * 2)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('method', ['skew', 'kurt'])
def test_expanding_skew_kurt_numerical_stability(method: str) -> None:
    s = Series(np.random.default_rng(2).random(10))
    expected = getattr(s.expanding(3), method)()
    s = s + 5000
    result = getattr(s.expanding(3), method)()
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('window', [1, 3, 10, 20])
@pytest.mark.parametrize('method', ['min', 'max', 'average'])
@pytest.mark.parametrize('pct', [True, False])
@pytest.mark.parametrize('ascending', [True, False])
@pytest.mark.parametrize('test_data', ['default', 'duplicates', 'nans'])
def test_rank(window: int, method: str, pct: bool, ascending: bool, test_data: str) -> None:
    length = 20
    if test_data == 'default':
        ser = Series(data=np.random.default_rng(2).random(length))
    elif test_data == 'duplicates':
        ser = Series(data=np.random.default_rng(2).choice(3, length))
    elif test_data == 'nans':
        ser = Series(data=np.random.default_rng(2).choice([1.0, 0.25, 0.75, np.nan, np.inf, -np.inf], length))
    expected = ser.expanding(window).apply(lambda x: x.rank(method=method, pct=pct, ascending=ascending).iloc[-1])
    result = ser.expanding(window).rank(method=method, pct=pct, ascending=ascending)
    tm.assert_series_equal(result, expected)

def test_expanding_corr(series: Series) -> None:
    A = series.dropna()
    B = (A + np.random.default_rng(2).standard_normal(len(A)))[:-5]
    result = A.expanding().corr(B)
    rolling_result = A.rolling(window=len(A), min_periods=1).corr(B)
    tm.assert_almost_equal(rolling_result, result)

def test_expanding_count(series: Series) -> None:
    result = series.expanding(min_periods=0).count()
    tm.assert_almost_equal(result, series.rolling(window=len(series), min_periods=0).count())

def test_expanding_quantile(series: Series) -> None:
    result = series.expanding().quantile(0.5)
    rolling_result = series.rolling(window=len(series), min_periods=1).quantile(0.5)
    tm.assert_almost_equal(result, rolling_result)

def test_expanding_cov(series: Series) -> None:
    A = series
    B = (A + np.random.default_rng(2).standard_normal(len(A)))[:-5]
    result = A.expanding().cov(B)
    rolling_result = A.rolling(window=len(A), min_periods=1).cov(B)
    tm.assert_almost_equal(rolling_result, result)

def test_expanding_cov_pairwise(frame: DataFrame) -> None:
    result = frame.expanding().cov()
    rolling_result = frame.rolling(window=len(frame), min_periods=1).cov()
    tm.assert_frame_equal(result, rolling_result)

def test_expanding_corr_pairwise(frame: DataFrame) -> None:
    result = frame.expanding().corr()
    rolling_result = frame.rolling(window=len(frame), min_periods=1).corr()
    tm.assert_frame_equal(result, rolling_result)

@pytest.mark.parametrize(
    'func,static_comp',
    [
        ('sum', lambda x: np.sum(x, axis=0)),
        ('mean', lambda x: np.mean(x, axis=0)),
        ('max', lambda x: np.max(x, axis=0)),
        ('min', lambda x: np.min(x, axis=0)),
    ],
    ids=['sum', 'mean', 'max', 'min']
)
def test_expanding_func(func: str, static_comp: Callable[[Any], Any], frame_or_series: FrameOrSeriesMaker) -> None:
    data = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))
    obj = data.expanding(min_periods=1)
    result = getattr(obj, func)()
    assert isinstance(result, type(data))
    expected = static_comp(data[:11])
    if isinstance(data, Series):
        tm.assert_almost_equal(result[10], expected)
    else:
        tm.assert_series_equal(result.iloc[10], expected, check_names=False)

@pytest.mark.parametrize(
    'func,static_comp',
    [
        ('sum', np.sum),
        ('mean', np.mean),
        ('max', np.max),
        ('min', np.min),
    ],
    ids=['sum', 'mean', 'max', 'min']
)
def test_expanding_min_periods(func: str, static_comp: Callable[[Any], Any]) -> None:
    ser = Series(np.random.default_rng(2).standard_normal(50))
    result = getattr(ser.expanding(min_periods=30), func)()
    assert result[:29].isna().all()
    tm.assert_almost_equal(result.iloc[-1], static_comp(ser[:50]))
    result = getattr(ser.expanding(min_periods=15), func)()
    assert isna(result.iloc[13])
    assert notna(result.iloc[14])
    ser2 = Series(np.random.default_rng(2).standard_normal(20))
    result = getattr(ser2.expanding(min_periods=5), func)()
    assert isna(result[3])
    assert notna(result[4])
    result0 = getattr(ser.expanding(min_periods=0), func)()
    result1 = getattr(ser.expanding(min_periods=1), func)()
    tm.assert_almost_equal(result0, result1)
    result = getattr(ser.expanding(min_periods=1), func)()
    tm.assert_almost_equal(result.iloc[-1], static_comp(ser[:50]))

@pytest.mark.parametrize('engine_and_raw', [("cython", True), ("cython", False)])
def test_expanding_apply(engine_and_raw: Tuple[Any, Any], frame_or_series: FrameOrSeriesMaker) -> None:
    engine, raw = engine_and_raw
    data = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))
    result = data.expanding(min_periods=1).apply(lambda x: x.mean(), raw=raw, engine=engine)
    assert isinstance(result, type(data))
    if isinstance(data, Series):
        tm.assert_almost_equal(result[9], np.mean(data[:11], axis=0))
    else:
        tm.assert_series_equal(result.iloc[9], np.mean(data[:11], axis=0), check_names=False)

@pytest.mark.parametrize('engine_and_raw', [("cython", True), ("cython", False)])
def test_expanding_min_periods_apply(engine_and_raw: Tuple[Any, Any]) -> None:
    engine, raw = engine_and_raw
    ser = Series(np.random.default_rng(2).standard_normal(50))
    result = ser.expanding(min_periods=30).apply(lambda x: x.mean(), raw=raw, engine=engine)
    assert result[:29].isna().all()
    tm.assert_almost_equal(result.iloc[-1], np.mean(ser[:50]))
    result = ser.expanding(min_periods=15).apply(lambda x: x.mean(), raw=raw, engine=engine)
    assert isna(result.iloc[13])
    assert notna(result.iloc[14])
    ser2 = Series(np.random.default_rng(2).standard_normal(20))
    result = ser2.expanding(min_periods=5).apply(lambda x: x.mean(), raw=raw, engine=engine)
    assert isna(result[3])
    assert notna(result[4])
    result0 = ser.expanding(min_periods=0).apply(lambda x: x.mean(), raw=raw, engine=engine)
    result1 = ser.expanding(min_periods=1).apply(lambda x: x.mean(), raw=raw, engine=engine)
    tm.assert_almost_equal(result0, result1)
    result = ser.expanding(min_periods=1).apply(lambda x: x.mean(), raw=raw, engine=engine)
    tm.assert_almost_equal(result.iloc[-1], np.mean(ser[:50]))

@pytest.mark.parametrize('f', [
    lambda x: x.expanding(min_periods=5).cov(x, pairwise=True),
    lambda x: x.expanding(min_periods=5).corr(x, pairwise=True)
])
def test_moment_functions_zero_length_pairwise(f: Callable[[Any], DataFrame]) -> None:
    df1 = DataFrame()
    df2 = DataFrame(columns=Index(['a'], name='foo'), index=Index([], name='bar'))
    df2['a'] = df2['a'].astype('float64')
    df1_expected = DataFrame(index=MultiIndex.from_product([df1.index, df1.columns]))
    df2_expected = DataFrame(index=MultiIndex.from_product([df2.index, df2.columns], names=['bar', 'foo']),
                              columns=Index(['a'], name='foo'), dtype='float64')
    df1_result = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)
    df2_result = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)

@pytest.mark.parametrize('f', [
    lambda x: x.expanding().count(),
    lambda x: x.expanding(min_periods=5).cov(x, pairwise=False),
    lambda x: x.expanding(min_periods=5).corr(x, pairwise=False),
    lambda x: x.expanding(min_periods=5).max(),
    lambda x: x.expanding(min_periods=5).min(),
    lambda x: x.expanding(min_periods=5).first(),
    lambda x: x.expanding(min_periods=5).last(),
    lambda x: x.expanding(min_periods=5).sum(),
    lambda x: x.expanding(min_periods=5).mean(),
    lambda x: x.expanding(min_periods=5).std(),
    lambda x: x.expanding(min_periods=5).var(),
    lambda x: x.expanding(min_periods=5).skew(),
    lambda x: x.expanding(min_periods=5).kurt(),
    lambda x: x.expanding(min_periods=5).quantile(0.5),
    lambda x: x.expanding(min_periods=5).median(),
    lambda x: x.expanding(min_periods=5).apply(sum, raw=False),
    lambda x: x.expanding(min_periods=5).apply(sum, raw=True)
])
def test_moment_functions_zero_length(f: Callable[[Union[Series, DataFrame]], Union[Series, DataFrame]]) -> None:
    s = Series(dtype=np.float64)
    s_expected = s
    df1 = DataFrame()
    df1_expected = df1
    df2 = DataFrame(columns=['a'])
    df2['a'] = df2['a'].astype('float64')
    df2_expected = df2
    s_result = f(s)
    tm.assert_series_equal(s_result, s_expected)
    df1_result = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)
    df2_result = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)

@pytest.mark.parametrize('engine_and_raw', [("cython", True), ("cython", False)])
def test_expanding_apply_empty_series(engine_and_raw: Tuple[Any, Any]) -> None:
    engine, raw = engine_and_raw
    ser = Series([], dtype=np.float64)
    tm.assert_series_equal(ser, ser.expanding().apply(lambda x: x.mean(), raw=raw, engine=engine))

@pytest.mark.parametrize('engine_and_raw', [("cython", True), ("cython", False)])
def test_expanding_apply_min_periods_0(engine_and_raw: Tuple[Any, Any]) -> None:
    engine, raw = engine_and_raw
    s = Series([None, None, None])
    result = s.expanding(min_periods=0).apply(lambda x: len(x), raw=raw, engine=engine)
    expected = Series([1.0, 2.0, 3.0])
    tm.assert_series_equal(result, expected)

def test_expanding_cov_diff_index() -> None:
    s1 = Series([1, 2, 3], index=range(3))
    s2 = Series([1, 3], index=range(0, 4, 2))
    result = s1.expanding().cov(s2)
    expected = Series([None, None, 2.0])
    tm.assert_series_equal(result, expected)
    s2a = Series([1, None, 3], index=[0, 1, 2])
    result = s1.expanding().cov(s2a)
    tm.assert_series_equal(result, expected)
    s1 = Series([7, 8, 10], index=[0, 1, 3])
    s2 = Series([7, 9, 10], index=[0, 2, 3])
    result = s1.expanding().cov(s2)
    expected = Series([None, None, None, 4.5], index=list(range(4)))
    tm.assert_series_equal(result, expected)

def test_expanding_corr_diff_index() -> None:
    s1 = Series([1, 2, 3], index=range(3))
    s2 = Series([1, 3], index=range(0, 4, 2))
    result = s1.expanding().corr(s2)
    expected = Series([None, None, 1.0])
    tm.assert_series_equal(result, expected)
    s2a = Series([1, None, 3], index=[0, 1, 2])
    result = s1.expanding().corr(s2a)
    tm.assert_series_equal(result, expected)
    s1 = Series([7, 8, 10], index=[0, 1, 3])
    s2 = Series([7, 9, 10], index=[0, 2, 3])
    result = s1.expanding().corr(s2)
    expected = Series([None, None, None, 1.0], index=list(range(4)))
    tm.assert_series_equal(result, expected)

def test_expanding_cov_pairwise_diff_length() -> None:
    df1 = DataFrame([[1, 5], [3, 2], [3, 9]], columns=Index(['A', 'B'], name='foo'))
    df1a = DataFrame([[1, 5], [3, 9]], index=[0, 2], columns=Index(['A', 'B'], name='foo'))
    df2 = DataFrame([[5, 6], [None, None], [2, 1]], columns=Index(['X', 'Y'], name='foo'))
    df2a = DataFrame([[5, 6], [2, 1]], index=[0, 2], columns=Index(['X', 'Y'], name='foo'))
    result1 = df1.expanding().cov(df2, pairwise=True).loc[2]
    result2 = df1.expanding().cov(df2a, pairwise=True).loc[2]
    result3 = df1a.expanding().cov(df2, pairwise=True).loc[2]
    result4 = df1a.expanding().cov(df2a, pairwise=True).loc[2]
    expected = DataFrame([[-3.0, -6.0], [-5.0, -10.0]], columns=Index(['A', 'B'], name='foo'), index=Index(['X', 'Y'], name='foo'))
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)
    tm.assert_frame_equal(result3, expected)
    tm.assert_frame_equal(result4, expected)

def test_expanding_corr_pairwise_diff_length() -> None:
    df1 = DataFrame([[1, 2], [3, 2], [3, 4]], columns=['A', 'B'], index=Index(range(3), name='bar'))
    df1a = DataFrame([[1, 2], [3, 4]], index=Index([0, 2], name='bar'), columns=['A', 'B'])
    df2 = DataFrame([[5, 6], [None, None], [2, 1]], columns=['X', 'Y'], index=Index(range(3), name='bar'))
    df2a = DataFrame([[5, 6], [2, 1]], index=Index([0, 2], name='bar'), columns=['X', 'Y'])
    result1 = df1.expanding().corr(df2, pairwise=True).loc[2]
    result2 = df1.expanding().corr(df2a, pairwise=True).loc[2]
    result3 = df1a.expanding().corr(df2, pairwise=True).loc[2]
    result4 = df1a.expanding().corr(df2a, pairwise=True).loc[2]
    expected = DataFrame([[-1.0, -1.0], [-1.0, -1.0]], columns=['A', 'B'], index=Index(['X', 'Y']))
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)
    tm.assert_frame_equal(result3, expected)
    tm.assert_frame_equal(result4, expected)

@pytest.mark.parametrize(
    'values,method,expected',
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'first',
            [float('nan'), float('nan'), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ),
        (
            [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan],
            'first',
            [float('nan'), float('nan'), float('nan'), float('nan'), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'last',
            [float('nan'), float('nan'), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ),
        (
            [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan],
            'last',
            [float('nan'), float('nan'), float('nan'), float('nan'), 5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
        )
    ]
)
def test_expanding_first_last(values: List[float], method: str, expected: List[float]) -> None:
    x = Series(values)
    result = getattr(x.expanding(3), method)()
    expected_series = Series(expected)
    tm.assert_almost_equal(result, expected_series)
    x_df = DataFrame({'A': values})
    result_df = getattr(x_df.expanding(3), method)()
    expected_df = DataFrame({'A': expected_series})
    tm.assert_almost_equal(result_df, expected_df)

@pytest.mark.parametrize(
    'values,method,expected',
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'first',
            [1.0] * 10
        ),
        (
            [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan],
            'first',
            [1.0] * 10
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'last',
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ),
        (
            [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan],
            'last',
            [1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
        )
    ]
)
def test_expanding_first_last_no_minp(values: List[float], method: str, expected: List[float]) -> None:
    x = Series(values)
    result = getattr(x.expanding(min_periods=0), method)()
    expected_series = Series(expected)
    tm.assert_almost_equal(result, expected_series)
    x_df = DataFrame({'A': values})
    result_df = getattr(x_df.expanding(min_periods=0), method)()
    expected_df = DataFrame({'A': expected_series})
    tm.assert_almost_equal(result_df, expected_df)

def test_expanding_apply_args_kwargs(engine_and_raw: Tuple[Any, Any]) -> None:
    def mean_w_arg(x: Any, const: float) -> float:
        return np.mean(x) + const
    engine, raw = engine_and_raw
    df = DataFrame(np.random.default_rng(2).random((20, 3)))
    expected = df.expanding().apply(np.mean, engine=engine, raw=raw) + 20.0
    result = df.expanding().apply(mean_w_arg, engine=engine, raw=raw, args=(20,))
    tm.assert_frame_equal(result, expected)
    result = df.expanding().apply(mean_w_arg, raw=raw, kwargs={'const': 20})
    tm.assert_frame_equal(result, expected)

def test_numeric_only_frame(arithmetic_win_operators: str, numeric_only: bool) -> None:
    kernel = arithmetic_win_operators
    df = DataFrame({'a': [1], 'b': 2, 'c': 3})
    df['c'] = df['c'].astype(object)
    expanding = df.expanding()
    op = getattr(expanding, kernel, None)
    if op is not None:
        result = op(numeric_only=numeric_only)
        columns = ['a', 'b'] if numeric_only else ['a', 'b', 'c']
        expected = df[columns].agg([kernel]).reset_index(drop=True).astype(float)
        assert list(expected.columns) == columns
        tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('kernel', ['corr', 'cov'])
@pytest.mark.parametrize('use_arg', [True, False])
def test_numeric_only_corr_cov_frame(kernel: str, numeric_only: bool, use_arg: bool) -> None:
    df = DataFrame({'a': [1, 2, 3], 'b': 2, 'c': 3})
    df['c'] = df['c'].astype(object)
    arg: Tuple[Any, ...] = (df,) if use_arg else ()
    expanding = df.expanding()
    op = getattr(expanding, kernel)
    result = op(*arg, numeric_only=numeric_only)
    columns = ['a', 'b'] if numeric_only else ['a', 'b', 'c']
    df2 = df[columns].astype(float)
    arg2: Tuple[Any, ...] = (df2,) if use_arg else ()
    expanding2 = df2.expanding()
    op2 = getattr(expanding2, kernel)
    expected = op2(*arg2, numeric_only=numeric_only)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('dtype', [int, object])
def test_numeric_only_series(arithmetic_win_operators: str, numeric_only: bool, dtype: Any) -> None:
    kernel = arithmetic_win_operators
    ser = Series([1], dtype=dtype)
    expanding = ser.expanding()
    op = getattr(expanding, kernel)
    if numeric_only and dtype is object:
        msg = f'Expanding.{kernel} does not implement numeric_only'
        with pytest.raises(NotImplementedError, match=msg):
            op(numeric_only=numeric_only)
    else:
        result = op(numeric_only=numeric_only)
        expected = ser.agg([kernel]).reset_index(drop=True).astype(float)
        tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('kernel', ['corr', 'cov'])
@pytest.mark.parametrize('use_arg', [True, False])
@pytest.mark.parametrize('dtype', [int, object])
def test_numeric_only_corr_cov_series(kernel: str, use_arg: bool, numeric_only: bool, dtype: Any) -> None:
    ser = Series([1, 2, 3], dtype=dtype)
    arg: Tuple[Any, ...] = (ser,) if use_arg else ()
    expanding = ser.expanding()
    op = getattr(expanding, kernel)
    if numeric_only and dtype is object:
        msg = f'Expanding.{kernel} does not implement numeric_only'
        with pytest.raises(NotImplementedError, match=msg):
            op(*arg, numeric_only=numeric_only)
    else:
        result = op(*arg, numeric_only=numeric_only)
        ser2 = ser.astype(float)
        arg2: Tuple[Any, ...] = (ser2,) if use_arg else ()
        expanding2 = ser2.expanding()
        op2 = getattr(expanding2, kernel)
        expected = op2(*arg2, numeric_only=numeric_only)
        tm.assert_series_equal(result, expected)
import numpy as np
import pytest
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex, Series
import pandas._testing as tm

def test_doc_string() -> None:
    df: DataFrame = DataFrame({'B': [0, 1, 2, np.nan, 4]})
    df
    df.expanding(2).sum()

def test_constructor(frame_or_series: callable) -> None:
    c = frame_or_series(range(5)).expanding
    c(min_periods=1)

@pytest.mark.parametrize('w', [2.0, 'foo', np.array([2])])
def test_constructor_invalid(frame_or_series: callable, w) -> None:
    c = frame_or_series(range(5)).expanding
    msg = 'min_periods must be an integer'
    with pytest.raises(ValueError, match=msg):
        c(min_periods=w)

@pytest.mark.parametrize('expander', [1, pytest.param('ls', marks=pytest.mark.xfail(reason='GH#16425 expanding with offset not supported'))])
def test_empty_df_expanding(expander) -> None:
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

def test_expanding_count_with_min_periods(frame_or_series: callable) -> None:
    result = frame_or_series(range(5)).expanding(min_periods=3).count()
    expected = frame_or_series([np.nan, np.nan, 3.0, 4.0, 5.0])
    tm.assert_equal(result, expected)

def test_expanding_count_default_min_periods_with_null_values(frame_or_series: callable) -> None:
    values = [1, 2, 3, np.nan, 4, 5, 6]
    expected_counts = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0]
    result = frame_or_series(values).expanding().count()
    expected = frame_or_series(expected_counts)
    tm.assert_equal(result, expected)

def test_expanding_count_with_min_periods_exceeding_series_length(frame_or_series: callable) -> None:
    result = frame_or_series(range(5)).expanding(min_periods=6).count()
    expected = frame_or_series([np.nan, np.nan, np.nan, np.nan, np.nan])
    tm.assert_equal(result, expected)

@pytest.mark.parametrize('df,expected,min_periods', [({'A': [1, 2, 3], 'B': [4, 5, 6]}, [({'A': [1], 'B': [4]}, [0]), ({'A': [1, 2], 'B': [4, 5]}, [0, 1]), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [0, 1, 2])], 3), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [({'A': [1], 'B': [4]}, [0]), ({'A': [1, 2], 'B': [4, 5]}, [0, 1]), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [0, 1, 2])], 2), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [({'A': [1], 'B': [4]}, [0]), ({'A': [1, 2], 'B': [4, 5]}, [0, 1]), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [0, 1, 2])], 1), ({'A': [1], 'B': [4]}, [], 2), (None, [({}, [])], 1), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [({'A': [1.0], 'B': [np.nan]}, [0]), ({'A': [1, np.nan], 'B': [np.nan, 5]}, [0, 1]), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [0, 1, 2])], 3), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [({'A': [1.0], 'B': [np.nan]}, [0]), ({'A': [1, np.nan], 'B': [np.nan, 5]}, [0, 1]), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [0, 1, 2])], 2), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [({'A': [1.0], 'B': [np.nan]}, [0]), ({'A': [1, np.nan], 'B': [np.nan, 5]}, [0, 1]), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [0, 1, 2])], 1)])
def test_iter_expanding_dataframe(df, expected, min_periods) -> None:
    df = DataFrame(df)
    expecteds = [DataFrame(values, index=index) for values, index in expected]
    for expected, actual in zip(expecteds, df.expanding(min_periods)):
        tm.assert_frame_equal(actual, expected)

@pytest.mark.parametrize('ser,expected,min_periods', [(Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 3), (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 2), (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 1), (Series([1, 2]), [([1], [0]), ([1, 2], [0, 1])], 2), (Series([np.nan, 2]), [([np.nan], [0]), ([np.nan, 2], [0, 1])], 2), (Series([], dtype='int64'), [], 2)])
def test_iter_expanding_series(ser, expected, min_periods) -> None:
    expecteds = [Series(values, index=index) for values, index in expected]
    for expected, actual in zip(expecteds, ser.expanding(min_periods)):
        tm.assert_series_equal(actual, expected)

def test_center_invalid() -> None:
    df = DataFrame()
    with pytest.raises(TypeError, match='.* got an unexpected keyword'):
        df.expanding(center=True)

def test_expanding_sem(frame_or_series: callable) -> None:
    obj = frame_or_series([0, 1, 2])
    result = obj.expanding().sem()
    if isinstance(result, DataFrame):
        result = Series(result[0].values)
    expected = Series([np.nan] + [0.707107] * 2)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('method', ['skew', 'kurt'])
def test_expanding_skew_kurt_numerical_stability(method) -> None:
    s = Series(np.random.default_rng(2).random(10))
    expected = getattr(s.expanding(3), method)()
    s = s + 5000
    result = getattr(s.expanding(3), method)()
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('window', [1, 3, 10, 20])
@pytest.mark.parametrize('method', ['min', 'max', 'average'])
@pytest.mark.parametrize('pct', [True, False])
@pytest.mark.parametrize('test_data', ['default', 'duplicates', 'nans'])
def test_rank(window, method, pct, ascending, test_data) -> None:
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

@pytest.mark.parametrize('func,static_comp', [('sum', lambda x: np.sum(x, axis=0)), ('mean', lambda x: np.mean(x, axis=0)), ('max', lambda x: np.max(x, axis=0)), ('min', lambda x: np.min(x, axis=0))], ids=['sum', 'mean', 'max', 'min'])
def test_expanding_func(func, static_comp, frame_or_series: callable) -> None:
    data = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))
    obj = data.expanding(min_periods=1)
    result = getattr(obj, func)()
    assert isinstance(result, frame_or_series)
    expected = static_comp(data[:11])
    if frame_or_series is Series:
        tm.assert_almost_equal(result[10], expected)
    else:
        tm.assert_series_equal(result.iloc[10], expected, check_names=False)

@pytest.mark.parametrize('func,static_comp', [('sum', np.sum), ('mean', np.mean), ('max', np.max), ('min', np.min)], ids=['sum', 'mean', 'max', 'min'])
def test_expanding_min_periods(func, static_comp) -> None:
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

def test_expanding_apply(engine_and_raw, frame_or_series: callable) -> None:
    engine, raw = engine_and_raw
    data = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))
    result = data.expanding(min_periods=1).apply(lambda x: x.mean(), raw=raw, engine=engine)
    assert isinstance(result, frame_or_series)
    if frame_or_series is Series:
        tm.assert_almost_equal(result[9], np.mean(data[:11], axis=0))
    else:
        tm.assert_series_equal(result.iloc[9], np.mean(data[:11], axis=0), check_names=False)

def test_expanding_min_periods_apply(engine_and_raw) -> None:
    engine, raw = engine_and_raw
    ser = Series([None, None, None])
    result = ser.expanding(min_periods=0).apply(lambda x: len(x), raw=raw, engine=engine)
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
    df1 = DataFrame([[1, 2], [3, 2], [3, 4]], columns
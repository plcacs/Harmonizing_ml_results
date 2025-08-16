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
def test_constructor_invalid(frame_or_series: callable, w: any) -> None:
    c = frame_or_series(range(5)).expanding
    msg = 'min_periods must be an integer'
    with pytest.raises(ValueError, match=msg):
        c(min_periods=w)

@pytest.mark.parametrize('expander', [1, pytest.param('ls', marks=pytest.mark.xfail(reason='GH#16425 expanding with offset not supported'))])
def test_empty_df_expanding(expander: any) -> None:
    expected: DataFrame = DataFrame()
    result: DataFrame = DataFrame().expanding(expander).sum()
    tm.assert_frame_equal(result, expected)
    expected: DataFrame = DataFrame(index=DatetimeIndex([]))
    result: DataFrame = DataFrame(index=DatetimeIndex([])).expanding(expander).sum()
    tm.assert_frame_equal(result, expected)

def test_missing_minp_zero() -> None:
    x: Series = Series([np.nan])
    result: Series = x.expanding(min_periods=0).sum()
    expected: Series = Series([0.0])
    tm.assert_series_equal(result, expected)
    result: Series = x.expanding(min_periods=1).sum()
    expected: Series = Series([np.nan])
    tm.assert_series_equal(result, expected)

def test_expanding() -> None:
    df: DataFrame = DataFrame(np.ones((10, 20)))
    expected: DataFrame = DataFrame({i: [np.nan] * 2 + [float(j) for j in range(3, 11)] for i in range(20)})
    result: DataFrame = df.expanding(3).sum()
    tm.assert_frame_equal(result, expected)

def test_expanding_count_with_min_periods(frame_or_series: callable) -> None:
    result: Series = frame_or_series(range(5)).expanding(min_periods=3).count()
    expected: Series = frame_or_series([np.nan, np.nan, 3.0, 4.0, 5.0])
    tm.assert_equal(result, expected)

def test_expanding_count_default_min_periods_with_null_values(frame_or_series: callable) -> None:
    values: list = [1, 2, 3, np.nan, 4, 5, 6]
    expected_counts: list = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0]
    result: Series = frame_or_series(values).expanding().count()
    expected: Series = frame_or_series(expected_counts)
    tm.assert_equal(result, expected)

def test_expanding_count_with_min_periods_exceeding_series_length(frame_or_series: callable) -> None:
    result: Series = frame_or_series(range(5)).expanding(min_periods=6).count()
    expected: Series = frame_or_series([np.nan, np.nan, np.nan, np.nan, np.nan])
    tm.assert_equal(result, expected)

@pytest.mark.parametrize('df,expected,min_periods', [({'A': [1, 2, 3], 'B': [4, 5, 6]}, [({'A': [1], 'B': [4]}, [0]), ({'A': [1, 2], 'B': [4, 5]}, [0, 1]), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [0, 1, 2])], 3), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [({'A': [1], 'B': [4]}, [0]), ({'A': [1, 2], 'B': [4, 5]}, [0, 1]), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [0, 1, 2])], 2), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [({'A': [1], 'B': [4]}, [0]), ({'A': [1, 2], 'B': [4, 5]}, [0, 1]), ({'A': [1, 2, 3], 'B': [4, 5, 6]}, [0, 1, 2])], 1), ({'A': [1], 'B': [4]}, [], 2), (None, [({}, [])], 1), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [({'A': [1.0], 'B': [np.nan]}, [0]), ({'A': [1, np.nan], 'B': [np.nan, 5]}, [0, 1]), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [0, 1, 2])], 3), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [({'A': [1.0], 'B': [np.nan]}, [0]), ({'A': [1, np.nan], 'B': [np.nan, 5]}, [0, 1]), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [0, 1, 2])], 2), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [({'A': [1.0], 'B': [np.nan]}, [0]), ({'A': [1, np.nan], 'B': [np.nan, 5]}, [0, 1]), ({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]}, [0, 1, 2])], 1)])
def test_iter_expanding_dataframe(df: dict, expected: list, min_periods: int) -> None:
    df: DataFrame = DataFrame(df)
    expecteds: list = [DataFrame(values, index=index) for values, index in expected]
    for expected, actual in zip(expecteds, df.expanding(min_periods)):
        tm.assert_frame_equal(actual, expected)

@pytest.mark.parametrize('ser,expected,min_periods', [(Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 3), (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 2), (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 1), (Series([1, 2]), [([1], [0]), ([1, 2], [0, 1])], 2), (Series([np.nan, 2]), [([np.nan], [0]), ([np.nan, 2], [0, 1])], 2), (Series([], dtype='int64'), [], 2)])
def test_iter_expanding_series(ser: Series, expected: list, min_periods: int) -> None:
    expecteds: list = [Series(values, index=index) for values, index in expected]
    for expected, actual in zip(expecteds, ser.expanding(min_periods)):
        tm.assert_series_equal(actual, expected)

def test_center_invalid() -> None:
    df: DataFrame = DataFrame()
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
def test_expanding_skew_kurt_numerical_stability(method: str) -> None:
    s: Series = Series(np.random.default_rng(2).random(10))
    expected = getattr(s.expanding(3), method)()
    s = s + 5000
    result = getattr(s.expanding(3), method)()
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('window', [1, 3, 10, 20])
@pytest.mark.parametrize('method', ['min', 'max', 'average'])
@pytest.mark.parametrize('pct', [True, False])
@pytest.mark.parametrize('test_data', ['default', 'duplicates', 'nans'])
def test_rank(window: int, method: str, pct: bool, ascending: any, test_data: str) -> None:
    length: int = 20
    if test_data == 'default':
        ser: Series = Series(data=np.random.default_rng(2).random(length))
    elif test_data == 'duplicates':
        ser: Series = Series(data=np.random.default_rng(2).choice(3, length))
    elif test_data == 'nans':
        ser: Series = Series(data=np.random.default_rng(2).choice([1.0, 0.25, 0.75, np.nan, np.inf, -np.inf], length))
    expected: Series = ser.expanding(window).apply(lambda x: x.rank(method=method, pct=pct, ascending=ascending).iloc[-1])
    result: Series = ser.expanding(window).rank(method=method, pct=pct, ascending=ascending)
    tm.assert_series_equal(result, expected)

def test_expanding_corr(series: Series) -> None:
    A: Series = series.dropna()
    B: Series = (A + np.random.default_rng(2).standard_normal(len(A)))[:-5]
    result: Series = A.expanding().corr(B)
    rolling_result: Series = A.rolling(window=len(A), min_periods=1).corr(B)
    tm.assert_almost_equal(rolling_result, result)

def test_expanding_count(series: Series) -> None:
    result: Series = series.expanding(min_periods=0).count()
    tm.assert_almost_equal(result, series.rolling(window=len(series), min_periods=0).count())

def test_expanding_quantile(series: Series) -> None:
    result: Series = series.expanding().quantile(0.5)
    rolling_result: Series = series.rolling(window=len(series), min_periods=1).quantile(0.5)
    tm.assert_almost_equal(result, rolling_result)

def test_expanding_cov(series: Series) -> None:
    A: Series = series
    B: Series = (A + np.random.default_rng(2).standard_normal(len(A)))[:-5]
    result: Series = A.expanding().cov(B)
    rolling_result: Series = A.rolling(window=len(A), min_periods=1).cov(B)
    tm.assert_almost_equal(rolling_result, result)

def test_expanding_cov_pairwise(frame: DataFrame) -> None:
    result: DataFrame = frame.expanding().cov()
    rolling_result: DataFrame = frame.rolling(window=len(frame), min_periods=1).cov()
    tm.assert_frame_equal(result, rolling_result)

def test_expanding_corr_pairwise(frame: DataFrame) -> None:
    result: DataFrame = frame.expanding().corr()
    rolling_result: DataFrame = frame.rolling(window=len(frame), min_periods=1).corr()
    tm.assert_frame_equal(result, rolling_result)

@pytest.mark.parametrize('func,static_comp', [('sum', lambda x: np.sum(x, axis=0)), ('mean', lambda x: np.mean(x, axis=0)), ('max', lambda x: np.max(x, axis=0)), ('min', lambda x: np.min(x, axis=0))], ids=['sum', 'mean', 'max', 'min'])
def test_expanding_func(func: str, static_comp: callable, frame_or_series: callable) -> None:
    data: any = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))
    obj: any = data.expanding(min_periods=1)
    result: any = getattr(obj, func)()
    assert isinstance(result, frame_or_series)
    expected: any = static_comp(data[:11])
    if frame_or_series is Series:
        tm.assert_almost_equal(result[10], expected)
    else:
        tm.assert_series_equal(result.iloc[10], expected, check_names=False)

@pytest.mark.parametrize('func,static_comp', [('sum', np.sum), ('mean', np.mean), ('max', np.max), ('min', np.min)], ids=['sum', 'mean', 'max', 'min'])
def test_expanding_min_periods(func: str, static_comp: callable) -> None:
    ser: Series = Series(np.random.default_rng(2).standard_normal(50))
    result: Series = getattr(ser.expanding(min_periods=30), func)()
    assert result[:29].isna().all()
    tm.assert_almost_equal(result.iloc[-1], static_comp(ser[:50]))
    result: Series = getattr(ser.expanding(min_periods=15), func)()
    assert isna(result.iloc[13])
    assert notna(result.iloc[14])
    ser2: Series = Series(np.random.default_rng(2).standard_normal(20))
    result: Series = getattr(ser2.expanding(min_periods=5), func)()
    assert isna(result[3])
    assert notna(result[4])
    result0: Series = getattr(ser.expanding(min_periods=0), func)()
    result1: Series = getattr(ser.expanding(min_periods=1), func)()
    tm.assert_almost_equal(result0, result1)
    result: Series = getattr(ser.expanding(min_periods=1), func)()
    tm.assert_almost_equal(result.iloc[-1], static_comp(ser[:50]))

def test_expanding_apply(engine_and_raw: tuple, frame_or_series: callable) -> None:
    engine, raw = engine_and_raw
    data: any = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))
    result: any = data.expanding(min_periods=1).apply(lambda x: x.mean(), raw=raw, engine=engine)
    assert isinstance(result, frame_or_series)
    if frame_or_series is Series:
        tm.assert_almost_equal(result[9], np.mean(data[:11], axis=0))
    else:
        tm.assert_series_equal(result.iloc[9], np.mean(data[:11], axis=0), check_names=False)

def test_expanding_min_periods_apply(engine_and_raw: tuple) -> None:
    engine, raw = engine_and_raw
    ser: Series = Series([None, None, None])
    result: Series = ser.expanding(min_periods=0).apply(lambda x: len(x), raw=raw, engine=engine)
    expected: Series = Series([1.0, 2.0, 3.0])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('f', [lambda x: x.expanding(min_periods=5).cov(x, pairwise=True), lambda x: x.expanding(min_periods=5).corr(x, pairwise=True)])
def test_moment_functions_zero_length_pairwise(f: callable) -> None:
    df1: DataFrame = DataFrame()
    df2: DataFrame = DataFrame(columns=Index(['a'], name='foo'), index=Index([], name='bar'))
    df2['a'] = df2['a'].astype('float64')
    df1_expected: DataFrame = DataFrame(index=MultiIndex.from_product([df1.index, df1.columns]))
    df2_expected: DataFrame = DataFrame(index=MultiIndex.from_product([df2.index, df2.columns], names=['bar', 'foo']), columns=Index(['a'], name='foo'), dtype='float64')
    df1_result: DataFrame = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)
    df2_result: DataFrame = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)

@pytest.mark.parametrize('f', [lambda x: x.expanding().count(), lambda x: x.expanding(min_periods=5).cov(x, pairwise=False), lambda x: x.expanding(min_periods=5).corr(x, pairwise=False), lambda x: x.expanding(min_periods=5).max(), lambda x: x.expanding(min_periods=5).min(), lambda x: x.expanding(min_periods=5).first(), lambda x: x.expanding(min_periods=5).last(), lambda x: x.expanding(min_periods=5).sum(), lambda x: x.expanding(min_periods=5).mean(), lambda x: x.expanding(min_periods=5).std(), lambda x: x.expanding(min_periods=5).var(), lambda x: x.expanding(min_periods=5).skew(), lambda x: x.expanding(min_periods=5).kurt(), lambda x: x.expanding(min_periods=5).quantile(0.5), lambda x: x.expanding(min_periods=5).median(), lambda x: x.expanding(min_periods=5).apply(sum, raw=False), lambda x: x.expanding(min_periods=5).apply(sum, raw=True)])
def test_moment_functions_zero_length(f: callable) -> None:
    s: Series = Series(dtype=np.float64)
    s_expected: Series = s
    df1: DataFrame = DataFrame()
    df1_expected: DataFrame = df1
    df2: DataFrame = DataFrame(columns=['a'])
    df2['a'] = df2['a'].astype('float64')
    df2_expected: DataFrame = df2
    s_result: Series = f(s)
    tm.assert_series_equal(s_result, s_expected)
    df1_result: DataFrame = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)
    df2_result: DataFrame = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)

def test_expanding_apply_empty_series(engine_and_raw: tuple) -> None:
    engine, raw = engine_and_raw
    ser: Series = Series
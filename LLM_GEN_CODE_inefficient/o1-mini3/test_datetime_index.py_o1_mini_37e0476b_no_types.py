from datetime import datetime
from functools import partial
from typing import Callable, List, Tuple, Union
import zoneinfo
import numpy as np
import pytest
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import DataFrame, Index, Series, Timedelta, Timestamp, isna, notna
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import Period, period_range
from pandas.core.resample import DatetimeIndex, _get_timestamp_range_edges
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute

@pytest.fixture
def simple_date_range_series():
    """
    Series with date range index and random data for test purposes.
    """

    def _simple_date_range_series(start, end, freq='D'):
        rng = date_range(start, end, freq=freq)
        return Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    return _simple_date_range_series

def test_custom_grouper(unit):
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='Min')
    dti = index.as_unit(unit)
    s = Series(np.array([1] * len(dti)), index=dti, dtype='int64')
    b = Grouper(freq=Minute(5))
    g = s.groupby(b)
    g.ohlc()
    funcs = ['sum', 'mean', 'prod', 'min', 'max', 'var']
    for f in funcs:
        g._cython_agg_general(f, alt=None, numeric_only=True)
    b = Grouper(freq=Minute(5), closed='right', label='right')
    g = s.groupby(b)
    g.ohlc()
    funcs = ['sum', 'mean', 'prod', 'min', 'max', 'var']
    for f in funcs:
        g._cython_agg_general(f, alt=None, numeric_only=True)
    assert g.ngroups == 2593
    assert notna(g.mean()).all()
    arr: List[int] = [1] + [5] * 2592
    idx = dti[0:-1:5]
    idx = idx.append(dti[-1:])
    idx = DatetimeIndex(idx, freq='5min').as_unit(unit)
    expect = Series(arr, index=idx)
    result = g.agg('sum')
    tm.assert_series_equal(result, expect)

def test_custom_grouper_df(unit):
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='D')
    b = Grouper(freq=Minute(5), closed='right', label='right')
    dti = index.as_unit(unit)
    df = DataFrame(np.random.default_rng(2).random((len(dti), 10)), index=dti, dtype='float64')
    r = df.groupby(b).agg('sum')
    assert len(r.columns) == 10
    assert len(r.index) == 2593

@pytest.mark.parametrize('closed, expected', [('right', lambda s: Series([s.iloc[0], s[1:6].mean(), s[6:11].mean(), s[11:].mean()], index=date_range('1/1/2000', periods=4, freq='5min', name='index'))), ('left', lambda s: Series([s[:5].mean(), s[5:10].mean(), s[10:].mean()], index=date_range('1/1/2000 00:05', periods=3, freq='5min', name='index')))])
def test_resample_basic(closed, expected, unit):
    index = date_range('1/1/2000 00:00:00', '1/1/2000 00:13:00', freq='Min')
    s = Series(range(len(index)), index=index)
    s.index.name = 'index'
    s.index = s.index.as_unit(unit)
    expected_series = expected(s)
    expected_series.index = expected_series.index.as_unit(unit)
    result = s.resample('5min', closed=closed, label='right').mean()
    tm.assert_series_equal(result, expected_series)

def test_resample_integerarray(unit):
    ts = Series(range(9), index=date_range('1/1/2000', periods=9, freq='min').as_unit(unit), dtype='Int64')
    result = ts.resample('3min').sum()
    expected = Series([3, 12, 21], index=date_range('1/1/2000', periods=3, freq='3min').as_unit(unit), dtype='Int64')
    tm.assert_series_equal(result, expected)
    result = ts.resample('3min').mean()
    expected = Series([1, 4, 7], index=date_range('1/1/2000', periods=3, freq='3min').as_unit(unit), dtype='Float64')
    tm.assert_series_equal(result, expected)

def test_resample_basic_grouper(unit):
    index = date_range('1/1/2000 00:00:00', '1/1/2000 00:13:00', freq='Min')
    s = Series(range(len(index)), index=index)
    s.index.name = 'index'
    s.index = s.index.as_unit(unit)
    result = s.resample('5Min').last()
    grouper = Grouper(freq=Minute(5), closed='left', label='left')
    expected = s.groupby(grouper).agg(lambda x: x.iloc[-1])
    tm.assert_series_equal(result, expected)

@pytest.mark.filterwarnings("ignore:The 'convention' keyword in Series.resample:FutureWarning")
@pytest.mark.parametrize('keyword,value', [('label', 'righttt'), ('closed', 'righttt'), ('convention', 'starttt')])
def test_resample_string_kwargs(keyword, value, unit):
    index = date_range('1/1/2000 00:00:00', '1/1/2000 00:13:00', freq='Min')
    series = Series(range(len(index)), index=index)
    series.index.name = 'index'
    series.index = series.index.as_unit(unit)
    msg = f'Unsupported value {value} for `{keyword}`'
    with pytest.raises(ValueError, match=msg):
        series.resample('5min', **{keyword: value})

def test_resample_how(downsample_method, unit):
    if downsample_method == 'ohlc':
        pytest.skip('covered by test_resample_how_ohlc')
    index = date_range('1/1/2000 00:00:00', '1/1/2000 00:13:00', freq='Min')
    s = Series(range(len(index)), index=index)
    s.index.name = 'index'
    s.index = s.index.as_unit(unit)
    grouplist = np.ones_like(s)
    grouplist[0] = 0
    grouplist[1:6] = 1
    grouplist[6:11] = 2
    grouplist[11:] = 3
    expected = s.groupby(grouplist).agg(downsample_method)
    expected.index = date_range('1/1/2000', periods=4, freq='5min', name='index').as_unit(unit)
    result = getattr(s.resample('5min', closed='right', label='right'), downsample_method)()
    tm.assert_series_equal(result, expected)

def test_resample_how_ohlc(unit):
    index = date_range('1/1/2000 00:00:00', '1/1/2000 00:13:00', freq='Min')
    s = Series(range(len(index)), index=index)
    s.index.name = 'index'
    s.index = s.index.as_unit(unit)
    grouplist = np.ones_like(s)
    grouplist[0] = 0
    grouplist[1:6] = 1
    grouplist[6:11] = 2
    grouplist[11:] = 3

    def _ohlc(group):
        if isna(group).all():
            return [np.nan, np.nan, np.nan, np.nan]
        return [group.iloc[0], group.max(), group.min(), group.iloc[-1]]
    expected = DataFrame(s.groupby(grouplist).agg(_ohlc).values.tolist(), index=date_range('1/1/2000', periods=4, freq='5min', name='index').as_unit(unit), columns=['open', 'high', 'low', 'close'])
    result = s.resample('5min', closed='right', label='right').ohlc()
    tm.assert_frame_equal(result, expected)

def test_resample_how_callables(unit):
    data = np.arange(5, dtype=np.int64)
    msg = "'d' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        ind = date_range(start='2014-01-01', periods=len(data), freq='d').as_unit(unit)
    df = DataFrame({'A': data, 'B': data}, index=ind)

    def fn(x, a=1):
        return str(type(x))

    class FnClass:

        def __call__(self, x):
            return str(type(x))
    df_standard = df.resample('ME').apply(fn)
    df_lambda = df.resample('ME').apply(lambda x: str(type(x)))
    df_partial = df.resample('ME').apply(partial(fn))
    df_partial2 = df.resample('ME').apply(partial(fn, a=2))
    df_class = df.resample('ME').apply(FnClass())
    tm.assert_frame_equal(df_standard, df_lambda)
    tm.assert_frame_equal(df_standard, df_partial)
    tm.assert_frame_equal(df_standard, df_partial2)
    tm.assert_frame_equal(df_standard, df_class)

def test_resample_rounding(unit):
    ts = ['2014-11-08 00:00:01', '2014-11-08 00:00:02', '2014-11-08 00:00:02', '2014-11-08 00:00:03', '2014-11-08 00:00:07', '2014-11-08 00:00:07', '2014-11-08 00:00:08', '2014-11-08 00:00:08', '2014-11-08 00:00:08', '2014-11-08 00:00:09', '2014-11-08 00:00:10', '2014-11-08 00:00:11', '2014-11-08 00:00:11', '2014-11-08 00:00:13', '2014-11-08 00:00:14', '2014-11-08 00:00:15', '2014-11-08 00:00:17', '2014-11-08 00:00:20', '2014-11-08 00:00:21']
    df = DataFrame({'value': [1] * 19}, index=pd.to_datetime(ts))
    df.index = df.index.as_unit(unit)
    result = df.resample('6s').sum()
    expected = DataFrame({'value': [4, 9, 4, 2]}, index=date_range('2014-11-08', freq='6s', periods=4).as_unit(unit))
    tm.assert_frame_equal(result, expected)
    result = df.resample('7s').sum()
    expected = DataFrame({'value': [4, 10, 4, 1]}, index=date_range('2014-11-08', freq='7s', periods=4).as_unit(unit))
    tm.assert_frame_equal(result, expected)
    result = df.resample('11s').sum()
    expected = DataFrame({'value': [11, 8]}, index=date_range('2014-11-08', freq='11s', periods=2).as_unit(unit))
    tm.assert_frame_equal(result, expected)
    result = df.resample('13s').sum()
    expected = DataFrame({'value': [13, 6]}, index=date_range('2014-11-08', freq='13s', periods=2).as_unit(unit))
    tm.assert_frame_equal(result, expected)
    result = df.resample('17s').sum()
    expected = DataFrame({'value': [16, 3]}, index=date_range('2014-11-08', freq='17s', periods=2).as_unit(unit))
    tm.assert_frame_equal(result, expected)

def test_resample_basic_from_daily(unit):
    dti = date_range(start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq='D', name='index').as_unit(unit)
    s = Series(np.random.default_rng(2).random(len(dti)), dti)
    msg = "'w-sun' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.resample('w-sun').last()
    assert len(result) == 3
    assert (result.index.dayofweek == [6, 6, 6]).all()
    assert result.iloc[0] == s['1/2/2005']
    assert result.iloc[1] == s['1/9/2005']
    assert result.iloc[2] == s.iloc[-1]
    result = s.resample('W-MON').last()
    assert len(result) == 2
    assert (result.index.dayofweek == [0, 0]).all()
    assert result.iloc[0] == s['1/3/2005']
    assert result.iloc[1] == s['1/10/2005']
    result = s.resample('W-TUE').last()
    assert len(result) == 2
    assert (result.index.dayofweek == [1, 1]).all()
    assert result.iloc[0] == s['1/4/2005']
    assert result.iloc[1] == s['1/10/2005']
    result = s.resample('W-WED').last()
    assert len(result) == 2
    assert (result.index.dayofweek == [2, 2]).all()
    assert result.iloc[0] == s['1/5/2005']
    assert result.iloc[1] == s['1/10/2005']
    result = s.resample('W-THU').last()
    assert len(result) == 2
    assert (result.index.dayofweek == [3, 3]).all()
    assert result.iloc[0] == s['1/6/2005']
    assert result.iloc[1] == s['1/10/2005']
    result = s.resample('W-FRI').last()
    assert len(result) == 2
    assert (result.index.dayofweek == [4, 4]).all()
    assert result.iloc[0] == s['1/7/2005']
    assert result.iloc[1] == s['1/10/2005']
    result = s.resample('B').last()
    assert len(result) == 7
    assert (result.index.dayofweek == [4, 0, 1, 2, 3, 4, 0]).all()
    assert result.iloc[0] == s['1/2/2005']
    assert result.iloc[1] == s['1/3/2005']
    assert result.iloc[5] == s['1/9/2005']
    assert result.index.name == 'index'

def test_resample_upsample_picked_but_not_correct(unit):
    dates = date_range('01-Jan-2014', '05-Jan-2014', freq='D').as_unit(unit)
    series = Series(1, index=dates)
    result = series.resample('D').mean()
    assert result.index[0] == dates[0]
    s = Series(np.arange(1.0, 6), index=[datetime(1975, 1, i, 12, 0) for i in range(1, 6)])
    s.index = s.index.as_unit(unit)
    expected = Series(np.arange(1.0, 6), index=date_range('19750101', periods=5, freq='D').as_unit(unit))
    result = s.resample('D').count()
    tm.assert_series_equal(result, Series(1, index=expected.index))
    result1 = s.resample('D').sum()
    result2 = s.resample('D').mean()
    tm.assert_series_equal(result1, expected)
    tm.assert_series_equal(result2, expected)

@pytest.mark.parametrize('f', ['sum', 'mean', 'prod', 'min', 'max', 'var'])
def test_resample_frame_basic_cy_funcs(f, unit):
    df = DataFrame(np.random.default_rng(2).standard_normal((50, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=50, freq='B'))
    df.index = df.index.as_unit(unit)
    b = Grouper(freq='ME')
    g = df.groupby(b)
    g._cython_agg_general(f, alt=None, numeric_only=True)

@pytest.mark.parametrize('freq', ['YE', 'ME'])
def test_resample_frame_basic_M_A(freq, unit):
    df = DataFrame(np.random.default_rng(2).standard_normal((50, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=50, freq='B'))
    df.index = df.index.as_unit(unit)
    result = df.resample(freq).mean()
    tm.assert_series_equal(result['A'], df['A'].resample(freq).mean())

def test_resample_upsample(unit):
    dti = date_range(start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq='D', name='index').as_unit(unit)
    s = Series(np.random.default_rng(2).random(len(dti)), dti)
    result = s.resample('Min').ffill()
    assert len(result) == 12961
    assert result.iloc[0] == s.iloc[0]
    assert result.iloc[-1] == s.iloc[-1]
    assert result.index.name == 'index'

def test_resample_how_method(unit):
    s = Series([11, 22], index=[Timestamp('2015-03-31 21:48:52.672000'), Timestamp('2015-03-31 21:49:52.739000')])
    s.index = s.index.as_unit(unit)
    expected = Series([11, np.nan, np.nan, np.nan, np.nan, np.nan, 22], index=DatetimeIndex([Timestamp('2015-03-31 21:48:50'), Timestamp('2015-03-31 21:49:00'), Timestamp('2015-03-31 21:49:10'), Timestamp('2015-03-31 21:49:20'), Timestamp('2015-03-31 21:49:30'), Timestamp('2015-03-31 21:49:40'), Timestamp('2015-03-31 21:49:50')], freq='10s'))
    expected.index = expected.index.as_unit(unit)
    tm.assert_series_equal(s.resample('10s').mean(), expected)

def test_resample_extra_index_point(unit):
    index = date_range(start='20150101', end='20150331', freq='BME').as_unit(unit)
    expected = DataFrame({'A': Series([21, 41, 63], index=index)})
    index = date_range(start='20150101', end='20150331', freq='B').as_unit(unit)
    df = DataFrame({'A': Series(range(len(index)), index=index)}, dtype='int64')
    result = df.resample('BME').last()
    tm.assert_frame_equal(result, expected)

def test_upsample_with_limit(unit):
    rng = date_range('1/1/2000', periods=3, freq='5min').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
    result = ts.resample('min').ffill(limit=2)
    expected = ts.reindex(result.index, method='ffill', limit=2)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('freq', ['1D', '10h', '5Min', '10s'])
@pytest.mark.parametrize('rule', ['YE', '3ME', '15D', '30h', '15Min', '30s'])
def test_nearest_upsample_with_limit(tz_aware_fixture, freq, rule, unit):
    rng = date_range('1/1/2000', periods=3, freq=freq, tz=tz_aware_fixture).as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
    result = ts.resample(rule).nearest(limit=2)
    expected = ts.reindex(result.index, method='nearest', limit=2)
    tm.assert_series_equal(result, expected)

def test_resample_ohlc(unit):
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='Min')
    s = Series(range(len(index)), index=index)
    s.index.name = 'index'
    s.index = s.index.as_unit(unit)
    grouper = Grouper(freq=Minute(5))
    expect = s.groupby(grouper).agg(lambda x: x.iloc[-1])
    result = s.resample('5Min').ohlc()
    assert len(result) == len(expect)
    assert len(result.columns) == 4
    xs = result.iloc[-2]
    assert xs['open'] == s.iloc[-6]
    assert xs['high'] == s[-6:-1].max()
    assert xs['low'] == s[-6:-1].min()
    assert xs['close'] == s.iloc[-2]
    xs = result.iloc[0]
    assert xs['open'] == s.iloc[0]
    assert xs['high'] == s[:5].max()
    assert xs['low'] == s[:5].min()
    assert xs['close'] == s.iloc[4]

def test_resample_ohlc_result(unit):
    index = date_range('1-1-2000', '2-15-2000', freq='h').as_unit(unit)
    index = index.union(date_range('4-15-2000', '5-15-2000', freq='h').as_unit(unit))
    s = Series(range(len(index)), index=index)
    a = s.loc[:'4-15-2000'].resample('30min').ohlc()
    assert isinstance(a, DataFrame)
    b = s.loc[:'4-14-2000'].resample('30min').ohlc()
    assert isinstance(b, DataFrame)

def test_resample_ohlc_result_odd_period(unit):
    rng = date_range('2013-12-30', '2014-01-07').as_unit(unit)
    index = rng.drop([Timestamp('2014-01-01'), Timestamp('2013-12-31'), Timestamp('2014-01-04'), Timestamp('2014-01-05')])
    df = DataFrame(data=np.arange(len(index)), index=index)
    result = df.resample('B').mean()
    expected = df.reindex(index=date_range(rng[0], rng[-1], freq='B').as_unit(unit))
    tm.assert_frame_equal(result, expected)

def test_resample_ohlc_dataframe(unit):
    df = DataFrame({'PRICE': {Timestamp('2011-01-06 10:59:05', tz=None): 24990, Timestamp('2011-01-06 12:43:33', tz=None): 25499, Timestamp('2011-01-06 12:54:09', tz=None): 25499}, 'VOLUME': {Timestamp('2011-01-06 10:59:05', tz=None): 1500000000, Timestamp('2011-01-06 12:43:33', tz=None): 5000000000, Timestamp('2011-01-06 12:54:09', tz=None): 100000000}}).reindex(['VOLUME', 'PRICE'], axis=1)
    df.index = df.index.as_unit(unit)
    df.columns.name = 'Cols'
    res = df.resample('h').ohlc()
    exp = pd.concat([df['VOLUME'].resample('h').ohlc(), df['PRICE'].resample('h').ohlc()], axis=1, keys=df.columns)
    assert exp.columns.names[0] == 'Cols'
    tm.assert_frame_equal(exp, res)
    df.columns = [['a', 'b'], ['c', 'd']]
    res = df.resample('h').ohlc()
    exp.columns = pd.MultiIndex.from_tuples([('a', 'c', 'open'), ('a', 'c', 'high'), ('a', 'c', 'low'), ('a', 'c', 'close'), ('b', 'd', 'open'), ('b', 'd', 'high'), ('b', 'd', 'low'), ('b', 'd', 'close')])
    tm.assert_frame_equal(exp, res)

def test_resample_reresample(unit):
    dti = date_range(start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq='D').as_unit(unit)
    s = Series(np.random.default_rng(2).random(len(dti)), dti)
    bs = s.resample('B', closed='right', label='right').mean()
    result = bs.resample('8h').mean()
    assert len(result) == 25
    assert isinstance(result.index.freq, offsets.DateOffset)
    assert result.index.freq == offsets.Hour(8)

@pytest.mark.parametrize('freq, expected_kwargs', [['YE-DEC', {'start': '1990', 'end': '2000', 'freq': 'Y-DEC'}], ['YE-JUN', {'start': '1990', 'end': '2000', 'freq': 'Y-JUN'}], ['ME', {'start': '1990-01', 'end': '2000-01', 'freq': 'M'}]])
def test_resample_timestamp_to_period(simple_date_range_series, freq, expected_kwargs, unit):
    ts = simple_date_range_series('1/1/1990', '1/1/2000')
    ts.index = ts.index.as_unit(unit)
    result = ts.resample(freq).mean().to_period()
    expected = ts.resample(freq).mean()
    expected.index = period_range(**expected_kwargs)
    tm.assert_series_equal(result, expected)

def test_ohlc_5min(unit):

    def _ohlc(group):
        if isna(group).all():
            return [np.nan, np.nan, np.nan, np.nan]
        return [group.iloc[0], group.max(), group.min(), group.iloc[-1]]
    rng = date_range('1/1/2000 00:00:00', '1/1/2000 5:59:50', freq='10s').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    resampled = ts.resample('5min', closed='right', label='right').ohlc()
    assert (resampled.loc['1/1/2000 00:00'] == ts.iloc[0]).all()
    exp = _ohlc(ts[1:31])
    assert (resampled.loc['1/1/2000 00:05'] == exp).all()
    exp = _ohlc(ts['1/1/2000 5:55:01':])
    assert (resampled.loc['1/1/2000 6:00:00'] == exp).all()

def test_downsample_non_unique(unit):
    rng = date_range('1/1/2000', '2/29/2000').as_unit(unit)
    rng2 = rng.repeat(5).values
    ts = Series(np.random.default_rng(2).standard_normal(len(rng2)), index=rng2)
    result = ts.resample('ME').mean()
    expected = ts.groupby(lambda x: x.month).mean()
    assert len(result) == 2
    tm.assert_almost_equal(result.iloc[0], expected[1])
    tm.assert_almost_equal(result.iloc[1], expected[2])

def test_asfreq_non_unique(unit):
    rng = date_range('1/1/2000', '2/29/2000').as_unit(unit)
    rng2 = rng.repeat(2).values
    ts = Series(np.random.default_rng(2).standard_normal(len(rng2)), index=rng2)
    msg = 'cannot reindex on an axis with duplicate labels'
    with pytest.raises(ValueError, match=msg):
        ts.asfreq('B')

@pytest.mark.parametrize('freq', ['min', '5min', '15min', '30min', '4h', '12h'])
def test_resample_anchored_ticks(freq, unit):
    rng = date_range('1/1/2000 04:00:00', periods=86400, freq='s').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts[:2] = np.nan
    result = ts[2:].resample(freq, closed='left', label='left').mean()
    expected = ts.resample(freq, closed='left', label='left').mean()
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('end', [1, 2])
def test_resample_single_group(end, unit):
    mysum = lambda x: x.sum()
    rng = date_range('2000-1-1', f'2000-{end}-10', freq='D').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    tm.assert_series_equal(ts.resample('ME').sum(), ts.resample('ME').apply(mysum))

def test_resample_single_group_std(unit):
    s = Series([30.1, 31.6], index=[Timestamp('20070915 15:30:00'), Timestamp('20070915 15:40:00')])
    s.index = s.index.as_unit(unit)
    expected = Series([0.75], index=DatetimeIndex([Timestamp('20070915')], freq='D').as_unit(unit))
    result = s.resample('D').apply(lambda x: np.std(x))
    tm.assert_series_equal(result, expected)

def test_resample_offset(unit):
    rng = date_range('1/1/2000 00:00:00', '1/1/2000 02:00', freq='s').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    resampled = ts.resample('5min', offset='2min').mean()
    exp_rng = date_range('12/31/1999 23:57:00', '1/1/2000 01:57', freq='5min').as_unit(unit)
    tm.assert_index_equal(resampled.index, exp_rng)

@pytest.mark.parametrize('kwargs', [{'origin': '1999-12-31 23:57:00'}, {'origin': Timestamp('1970-01-01 00:02:00')}, {'origin': 'epoch', 'offset': '2m'}, {'origin': '1999-12-31 12:02:00'}, {'offset': '-3m'}])
def test_resample_origin(kwargs, unit):
    rng = date_range('2000-01-01 00:00:00', '2000-01-01 02:00', freq='s').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    exp_rng = date_range('1999-12-31 23:57:00', '2000-01-01 01:57', freq='5min').as_unit(unit)
    resampled = ts.resample('5min', **kwargs).mean()
    tm.assert_index_equal(resampled.index, exp_rng)

@pytest.mark.parametrize('origin', ['invalid_value', 'epch', 'startday', 'startt', '2000-30-30', object()])
def test_resample_bad_origin(origin, unit):
    rng = date_range('2000-01-01 00:00:00', '2000-01-01 02:00', freq='s').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    msg = f"'origin' should be equal to 'epoch', 'start', 'start_day', 'end', 'end_day' or should be a Timestamp convertible type. Got '{origin}' instead."
    with pytest.raises(ValueError, match=msg):
        ts.resample('5min', origin=origin)

@pytest.mark.parametrize('offset', ['invalid_value', '12dayys', '2000-30-30', object()])
def test_resample_bad_offset(offset, unit):
    rng = date_range('2000-01-01 00:00:00', '2000-01-01 02:00', freq='s').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    msg = f"'offset' should be a Timedelta convertible type. Got '{offset}' instead."
    with pytest.raises(ValueError, match=msg):
        ts.resample('5min', offset=offset)

def test_resample_origin_prime_freq(unit):
    start, end = ('2000-10-01 23:30:00', '2000-10-02 00:30:00')
    rng = date_range(start, end, freq='7min').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    exp_rng = date_range('2000-10-01 23:14:00', '2000-10-02 00:22:00', freq='17min').as_unit(unit)
    resampled = ts.resample('17min').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample('17min', origin='start_day').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    exp_rng = date_range('2000-10-01 23:30:00', '2000-10-02 00:21:00', freq='17min').as_unit(unit)
    resampled = ts.resample('17min', origin='start').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample('17min', offset='23h30min').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample('17min', origin='start_day', offset='23h30min').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    exp_rng = date_range('2000-10-01 23:18:00', '2000-10-02 00:26:00', freq='17min').as_unit(unit)
    resampled = ts.resample('17min', origin='epoch').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    exp_rng = date_range('2000-10-01 23:24:00', '2000-10-02 00:15:00', freq='17min').as_unit(unit)
    resampled = ts.resample('17min', origin='2000-01-01').mean()
    tm.assert_index_equal(resampled.index, exp_rng)

def test_resample_origin_with_tz(unit):
    msg = 'The origin must have the same timezone as the index.'
    tz = 'Europe/Paris'
    rng = date_range('2000-01-01 00:00:00', '2000-01-01 02:00', freq='s', tz=tz).as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    exp_rng = date_range('1999-12-31 23:57:00', '2000-01-01 01:57', freq='5min', tz=tz).as_unit(unit)
    resampled = ts.resample('5min', origin='1999-12-31 23:57:00+00:00').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample('5min', origin='1999-12-31 12:02:00+03:00').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample('5min', origin='epoch', offset='2m').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    with pytest.raises(ValueError, match=msg):
        ts.resample('5min', origin='12/31/1999 23:57:00').mean()
    rng = date_range('2000-01-01 00:00:00', '2000-01-01 02:00', freq='s').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    with pytest.raises(ValueError, match=msg):
        ts.resample('5min', origin='12/31/1999 23:57:00+03:00').mean()

def test_resample_origin_epoch_with_tz_day_vs_24h(unit):
    start, end = ('2000-10-01 23:30:00+0500', '2000-12-02 00:30:00+0500')
    rng = date_range(start, end, freq='7min').as_unit(unit)
    random_values = np.random.default_rng(2).standard_normal(len(rng))
    ts_1 = Series(random_values, index=rng)
    result_1 = ts_1.resample('D', origin='epoch').mean()
    result_2 = ts_1.resample('24h', origin='epoch').mean()
    tm.assert_series_equal(result_1, result_2)
    ts_no_tz = ts_1.tz_localize(None)
    result_3 = ts_no_tz.resample('D', origin='epoch').mean()
    result_4 = ts_no_tz.resample('24h', origin='epoch').mean()
    tm.assert_series_equal(result_1, result_3.tz_localize(rng.tz), check_freq=False)
    tm.assert_series_equal(result_1, result_4.tz_localize(rng.tz), check_freq=False)
    start, end = ('2000-10-01 23:30:00+0200', '2000-12-02 00:30:00+0200')
    rng = date_range(start, end, freq='7min').as_unit(unit)
    ts_2 = Series(random_values, index=rng)
    result_5 = ts_2.resample('D', origin='epoch').mean()
    result_6 = ts_2.resample('24h', origin='epoch').mean()
    tm.assert_series_equal(result_1.tz_localize(None), result_5.tz_localize(None))
    tm.assert_series_equal(result_1.tz_localize(None), result_6.tz_localize(None))

def test_resample_origin_with_day_freq_on_dst(unit):
    tz = 'America/Chicago'

    def _create_series(values, timestamps, freq='D'):
        return Series(values, index=DatetimeIndex([Timestamp(t, tz=tz) for t in timestamps], freq=freq, ambiguous=True).as_unit(unit))
    start = Timestamp('2013-11-02', tz=tz)
    end = Timestamp('2013-11-03 23:59', tz=tz)
    rng = date_range(start, end, freq='1h').as_unit(unit)
    ts = Series(np.ones(len(rng)), index=rng)
    expected = _create_series([24.0, 25.0], ['2013-11-02', '2013-11-03'])
    for origin in ['epoch', 'start', 'start_day', start, None]:
        result = ts.resample('D', origin=origin).sum()
        tm.assert_series_equal(result, expected)
    start = Timestamp('2013-11-03', tz=tz)
    end = Timestamp('2013-11-03 23:59', tz=tz)
    rng = date_range(start, end, freq='1h').as_unit(unit)
    ts = Series(np.ones(len(rng)), index=rng)
    expected_ts = ['2013-11-02 22:00-05:00', '2013-11-03 22:00-06:00']
    expected = _create_series([23.0, 2.0], expected_ts)
    result = ts.resample('D', origin='start', offset='-2h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 22:00-05:00', '2013-11-03 21:00-06:00']
    expected = _create_series([22.0, 3.0], expected_ts, freq='24h')
    result = ts.resample('24h', origin='start', offset='-2h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 02:00-05:00', '2013-11-03 02:00:00-0500']
    expected = _create_series([3.0, 22.0], expected_ts)
    result = ts.resample('D', origin='start', offset='2h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 23:00-05:00', '2013-11-03 23:00-06:00']
    expected = _create_series([24.0, 1.0], expected_ts)
    result = ts.resample('D', origin='start', offset='-1h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 01:00-05:00', '2013-11-03 01:00:00-0500']
    expected = _create_series([1.0, 24.0], expected_ts)
    result = ts.resample('D', origin='start', offset='1h').sum()
    tm.assert_series_equal(result, expected)

def test_resample_dst_midnight_last_nonexistent():
    ts = Series(1, date_range('2024-04-19', '2024-04-20', tz='Africa/Cairo', freq='15min'))
    expected = Series([len(ts)], index=DatetimeIndex([ts.index[0]], freq='7D'))
    result = ts.resample('7D').sum()
    tm.assert_series_equal(result, expected)

def test_resample_daily_anchored(unit):
    rng = date_range('1/1/2012', '4/1/2012', freq='100min').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts[:2] = np.nan
    result = ts[2:].resample('D', closed='left', label='left').mean()
    expected = ts.resample('D', closed='left', label='left').mean()
    tm.assert_series_equal(result, expected)

def test_resample_to_period_monthly_buglet(unit):
    rng = date_range('1/1/2000', '12/31/2000').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    result = ts.resample('ME').mean().to_period()
    exp_index = period_range('Jan-2000', 'Dec-2000', freq='M')
    tm.assert_index_equal(result.index, exp_index)

def test_period_with_agg():
    s2 = Series(np.random.default_rng(2).integers(0, 5, 50), index=period_range('2012-01-01', freq='h', periods=50), dtype='float64')
    expected = s2.to_timestamp().resample('D').mean().to_period()
    msg = 'Resampling with a PeriodIndex is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = s2.resample('D')
    result = rs.agg(lambda x: x.mean())
    tm.assert_series_equal(result, expected)

def test_resample_segfault(unit):
    all_wins_and_wagers = [(1, datetime(2013, 10, 1, 16, 20), 1, 0), (2, datetime(2013, 10, 1, 16, 10), 1, 0), (2, datetime(2013, 10, 1, 18, 15), 1, 0), (2, datetime(2013, 10, 1, 16, 10, 31), 1, 0)]
    df = DataFrame.from_records(all_wins_and_wagers, columns=('ID', 'timestamp', 'A', 'B')).set_index('timestamp')
    df.index = df.index.as_unit(unit)
    result = df.groupby('ID').resample('5min').sum()
    expected = df.groupby('ID').apply(lambda x: x.resample('5min').sum())
    tm.assert_frame_equal(result, expected)

def test_resample_dtype_preservation(unit):
    df = DataFrame({'date': date_range(start='2016-01-01', periods=4, freq='W').as_unit(unit), 'group': [1, 1, 2, 2], 'val': Series([5, 6, 7, 8], dtype='int32')}).set_index('date')
    result = df.resample('1D').ffill()
    assert result.val.dtype == np.int32
    result = df.groupby('group').resample('1D').ffill()
    assert result.val.dtype == np.int32

def test_resample_dtype_coercion(unit):
    pytest.importorskip('scipy.interpolate')
    df = {'a': [1, 3, 1, 4]}
    df = DataFrame(df, index=date_range('2017-01-01', '2017-01-04').as_unit(unit))
    expected = df.astype('float64').resample('h').mean()['a'].interpolate('cubic')
    result = df.resample('h')['a'].mean().interpolate('cubic')
    tm.assert_series_equal(result, expected)
    result = df.resample('h').mean()['a'].interpolate('cubic')
    tm.assert_series_equal(result, expected)

def test_weekly_resample_buglet(unit):
    rng = date_range('1/1/2000', freq='B', periods=20).as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    resampled = ts.resample('W').mean()
    expected = ts.resample('W-SUN').mean()
    tm.assert_series_equal(resampled, expected)

def test_monthly_resample_error(unit):
    dates = date_range('4/16/2012 20:00', periods=5000, freq='h').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(dates)), index=dates)
    ts.resample('ME')

def test_nanosecond_resample_error(unit):
    start = 1443707890427
    exp_start = 1443707890400
    indx = date_range(start=pd.to_datetime(start), periods=10, freq='100ns')
    ts = Series(range(len(indx)), index=indx)
    r = ts.resample(pd.tseries.offsets.Nano(100))
    result = r.agg('mean')
    exp_indx = date_range(start=pd.to_datetime(exp_start), periods=10, freq='100ns')
    exp = Series(range(len(exp_indx)), index=exp_indx, dtype=float)
    tm.assert_series_equal(result, exp)

def test_resample_anchored_intraday(unit):
    rng = date_range('1/1/2012', '4/1/2012', freq='100min').as_unit(unit)
    df = DataFrame(rng.month, index=rng)
    result = df.resample('ME').mean()
    expected = df.resample('ME').mean().to_period()
    expected = expected.to_timestamp(how='end')
    expected.index += Timedelta(1, 'ns') - Timedelta(1, 'D')
    expected.index = expected.index.as_unit(unit)._with_freq('infer')
    assert expected.index.freq == 'ME'
    tm.assert_frame_equal(result, expected)
    result = df.resample('ME', closed='left').mean()
    exp = df.shift(1, freq='D').resample('ME').mean().to_period()
    exp = exp.to_timestamp(how='end')
    exp.index = exp.index + Timedelta(1, 'ns') - Timedelta(1, 'D')
    exp.index = exp.index.as_unit(unit)._with_freq('infer')
    assert exp.index.freq == 'ME'
    tm.assert_frame_equal(result, exp)

def test_resample_anchored_intraday2(unit):
    rng = date_range('1/1/2012', '4/1/2012', freq='100min').as_unit(unit)
    df = DataFrame(rng.month, index=rng)
    result = df.resample('QE').mean()
    expected = df.resample('QE').mean().to_period()
    expected = expected.to_timestamp(how='end')
    expected.index += Timedelta(1, 'ns') - Timedelta(1, 'D')
    expected.index._data.freq = 'QE'
    expected.index._freq = lib.no_default
    expected.index = expected.index.as_unit(unit)
    tm.assert_frame_equal(result, expected)
    result = df.resample('QE', closed='left').mean()
    expected = df.shift(1, freq='D').resample('QE').mean()
    expected = expected.to_period()
    expected = expected.to_timestamp(how='end')
    expected.index += Timedelta(1, 'ns') - Timedelta(1, 'D')
    expected.index._data.freq = 'QE'
    expected.index._freq = lib.no_default
    expected.index = expected.index.as_unit(unit)
    tm.assert_frame_equal(result, expected)

def test_resample_anchored_intraday3(simple_date_range_series, unit):
    ts = simple_date_range_series('2012-04-29 23:00', '2012-04-30 5:00', freq='h')
    ts.index = ts.index.as_unit(unit)
    resampled = ts.resample('ME').mean()
    assert len(resampled) == 1

@pytest.mark.parametrize('freq', ['MS', 'BMS', 'QS-MAR', 'YS-DEC', 'YS-JUN'])
def test_resample_anchored_monthstart(simple_date_range_series, freq, unit):
    ts = simple_date_range_series('1/1/2000', '12/31/2002')
    ts.index = ts.index.as_unit(unit)
    ts.resample(freq).mean()

@pytest.mark.parametrize('label, sec', [[None, 2.0], ['right', '4.2']])
def test_resample_anchored_multiday(label, sec):
    index1 = date_range('2014-10-14 23:06:23.206', periods=3, freq='400ms')
    index2 = date_range('2014-10-15 23:00:00', periods=2, freq='2200ms')
    index = index1.union(index2)
    s = Series(np.random.default_rng(2).standard_normal(5), index=index)
    result = s.resample('2200ms', label=label).mean()
    assert result.index[-1] == Timestamp(f'2014-10-15 23:00:{sec}00')

def test_corner_cases(unit):
    rng = date_range('1/1/2000', periods=12, freq='min').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    result = ts.resample('5min', closed='right', label='left').mean()
    ex_index = date_range('1999-12-31 23:55', periods=4, freq='5min').as_unit(unit)
    tm.assert_index_equal(result.index, ex_index)

def test_corner_cases_date(simple_date_range_series, unit):
    ts = simple_date_range_series('2000-04-28', '2000-04-30 11:00', freq='h')
    ts.index = ts.index.as_unit(unit)
    result = ts.resample('ME').apply(lambda x: x.mean()).to_period()
    assert len(result) == 1
    assert result.index[0] == Period('2000-04', freq='M')

def test_anchored_lowercase_buglet(unit):
    dates = date_range('4/16/2012 20:00', periods=50000, freq='s').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(dates)), index=dates)
    msg = "'d' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        ts.resample('d').mean()

def test_upsample_apply_functions(unit):
    rng = date_range('2012-06-12', periods=4, freq='h').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    result = ts.resample('20min').aggregate(['mean', 'sum'])
    assert isinstance(result, DataFrame)

def test_resample_not_monotonic(unit):
    rng = date_range('2012-06-12', periods=200, freq='h').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts = ts.take(np.random.default_rng(2).permutation(len(ts)))
    result = ts.resample('D').sum()
    exp = ts.sort_index().resample('D').sum()
    tm.assert_series_equal(result, exp)

@pytest.mark.parametrize('dtype', ['int64', 'int32', 'float64', pytest.param('float32', marks=pytest.mark.xfail(reason='Empty groups cause x.mean() to return float64'))])
def test_resample_median_bug_1688(dtype, unit):
    dti = DatetimeIndex([datetime(2012, 1, 1, 0, 0, 0), datetime(2012, 1, 1, 0, 5, 0)]).as_unit(unit)
    df = DataFrame([1, 2], index=dti, dtype=dtype)
    result = df.resample('min').apply(lambda x: x.mean())
    exp = df.asfreq('min')
    tm.assert_frame_equal(result, exp)
    result = df.resample('min').median()
    exp = df.asfreq('min')
    tm.assert_frame_equal(result, exp)

def test_how_lambda_functions(simple_date_range_series, unit):
    ts = simple_date_range_series('1/1/2000', '4/1/2000')
    ts.index = ts.index.as_unit(unit)
    result = ts.resample('ME').apply(lambda x: x.mean())
    exp = ts.resample('ME').mean()
    tm.assert_series_equal(result, exp)
    foo_exp = ts.resample('ME').mean()
    foo_exp.name = 'foo'
    bar_exp = ts.resample('ME').std()
    bar_exp.name = 'bar'
    result = ts.resample('ME').apply([lambda x: x.mean(), lambda x: x.std(ddof=1)])
    result.columns = ['foo', 'bar']
    tm.assert_series_equal(result['foo'], foo_exp)
    tm.assert_series_equal(result['bar'], bar_exp)
    result = ts.resample('ME').aggregate({'foo': lambda x: x.mean(), 'bar': lambda x: x.std(ddof=1)})
    tm.assert_series_equal(result['foo'], foo_exp, check_names=False)
    tm.assert_series_equal(result['bar'], bar_exp, check_names=False)

def test_resample_unequal_times(unit):
    start = datetime(1999, 3, 1, 5)
    end = datetime(2012, 7, 31, 4)
    bad_ind = date_range(start, end, freq='30min').as_unit(unit)
    df = DataFrame({'A': bad_ind, 'B': np.arange(len(bad_ind))})
    result = df.set_index('A').resample('ME').count()
    exp_idx = DatetimeIndex(['2014-07-31', '2014-08-31', '2014-09-30', '2014-10-31', '2014-11-30'], freq='ME', name='A').as_unit(unit)
    expected = DataFrame({'B': [1, 0, 2, 2, 1]}, index=exp_idx)
    if df['A'].isna().any():
        expected.index = expected.index._with_freq(None)
    tm.assert_frame_equal(result, expected)
    result = df.groupby(Grouper(freq='ME', key='A')).count()
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('dates', [[datetime(2014, 10, 1), datetime(2014, 9, 3), datetime(2014, 11, 5), datetime(2014, 9, 5), datetime(2014, 10, 8), datetime(2014, 7, 15)], [datetime(2014, 10, 1), datetime(2014, 9, 3), pd.NaT, datetime(2014, 11, 5), datetime(2014, 9, 5), pd.NaT, datetime(2014, 10, 8), datetime(2014, 7, 15)], [pd.NaT] + [datetime(2014, 10, 1), datetime(2014, 9, 3), datetime(2014, 11, 5), datetime(2014, 9, 5), datetime(2014, 10, 8), datetime(2014, 7, 15)]])
def test_resample_timegrouper(dates, unit):
    dates = DatetimeIndex(dates).as_unit(unit)
    df = DataFrame({'A': dates, 'B': np.arange(len(dates))})
    result = df.set_index('A').resample('ME').count()
    exp_idx = DatetimeIndex(['2014-07-31', '2014-08-31', '2014-09-30', '2014-10-31', '2014-11-30'], freq='ME', name='A').as_unit(unit)
    expected = DataFrame({'B': [1, 0, 2, 2, 1]}, index=exp_idx)
    if df['A'].isna().any():
        expected.index = expected.index._with_freq(None)
    tm.assert_frame_equal(result, expected)
    result = df.groupby(Grouper(freq='ME', key='A')).count()
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('dates', [[datetime(2014, 10, 1), datetime(2014, 9, 3), datetime(2014, 11, 5), datetime(2014, 9, 5), datetime(2014, 10, 8), datetime(2014, 7, 15)], [datetime(2014, 10, 1), datetime(2014, 9, 3), pd.NaT, datetime(2014, 11, 5), datetime(2014, 9, 5), pd.NaT, datetime(2014, 10, 8), datetime(2014, 7, 15)]])
def test_resample_timegrouper2(dates, unit):
    dates = DatetimeIndex(dates).as_unit(unit)
    df = DataFrame({'A': dates, 'B': np.arange(len(dates)), 'C': np.arange(len(dates))})
    result = df.set_index('A').resample('ME').count()
    exp_idx = DatetimeIndex(['2014-07-31', '2014-08-31', '2014-09-30', '2014-10-31', '2014-11-30'], freq='ME', name='A').as_unit(unit)
    expected = DataFrame({'B': [1, 0, 2, 2, 1], 'C': [1, 0, 2, 2, 1]}, index=exp_idx, columns=['B', 'C'])
    if df['A'].isna().any():
        expected.index = expected.index._with_freq(None)
    tm.assert_frame_equal(result, expected)
    result = df.groupby(Grouper(freq='ME', key='A')).count()
    tm.assert_frame_equal(result, expected)

def test_resample_nunique(unit):
    df = DataFrame({'ID': {Timestamp('2015-06-05 00:00:00'): '0010100903', Timestamp('2015-06-08 00:00:00'): '0010150847'}, 'DATE': {Timestamp('2015-06-05 00:00:00'): '2015-06-05', Timestamp('2015-06-08 00:00:00'): '2015-06-08'}})
    df.index = df.index.as_unit(unit)
    r = df.resample('D')
    g = df.groupby(Grouper(freq='D'))
    expected = df.groupby(Grouper(freq='D')).ID.apply(lambda x: x.nunique())
    assert expected.name == 'ID'
    for t in [r, g]:
        result = t.ID.nunique()
        tm.assert_series_equal(result, expected)
    result = df.ID.resample('D').nunique()
    tm.assert_series_equal(result, expected)
    result = df.ID.groupby(Grouper(freq='D')).nunique()
    tm.assert_series_equal(result, expected)

def test_resample_nunique_preserves_column_level_names(unit):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', end='2000-01-05', freq='B')).abs()
    df.columns = pd.MultiIndex.from_arrays([df.columns.tolist()] * 2, names=['lev0', 'lev1'])
    result = df.resample('1h').nunique()
    tm.assert_index_equal(df.columns, result.columns)

@pytest.mark.parametrize('func', [lambda x: x.nunique(), lambda x: x.agg(Series.nunique), lambda x: x.agg('nunique')], ids=['nunique', 'series_nunique', 'nunique_str'])
def test_resample_nunique_with_date_gap(func, unit):
    index = date_range('1-1-2000', '2-15-2000', freq='h').as_unit(unit)
    index2 = date_range('4-15-2000', '5-15-2000', freq='h').as_unit(unit)
    index3 = index.append(index2)
    s = Series(range(len(index3)), index=index3, dtype='int64')
    r = s.resample('ME')
    result = r.count()
    expected = func(r)
    tm.assert_series_equal(result, expected)

def test_resample_group_info(unit):
    n = 100
    k = 10
    prng = np.random.default_rng(2)
    dr = date_range(start='2015-08-27', periods=n // 10, freq='min').as_unit(unit)
    ts = Series(prng.integers(0, n // k, n).astype('int64'), index=prng.choice(dr, n))
    left = ts.resample('7min').nunique()
    ix = date_range(start=ts.index.min(), end=ts.index.max(), freq='7min').as_unit(unit)
    vals = ts.values
    bins = np.searchsorted(ix.values, ts.index, side='right')
    sorter = np.lexsort((vals, bins))
    vals, bins = (vals[sorter], bins[sorter])
    mask = np.r_[True, vals[1:] != vals[:-1]]
    mask |= np.r_[True, bins[1:] != bins[:-1]]
    arr = np.bincount(bins[mask] - 1, minlength=len(ix)).astype('int64', copy=False)
    right = Series(arr, index=ix)
    tm.assert_series_equal(left, right)

def test_resample_size(unit):
    n = 10000
    dr = date_range('2015-09-19', periods=n, freq='min').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(n), index=np.random.default_rng(2).choice(dr, n))
    left = ts.resample('7min').size()
    ix = date_range(start=left.index.min(), end=ts.index.max(), freq='7min').as_unit(unit)
    bins = np.searchsorted(ix.values, ts.index.values, side='right')
    val = np.bincount(bins, minlength=len(ix) + 1)[1:].astype('int64', copy=False)
    right = Series(val, index=ix)
    tm.assert_series_equal(left, right)

def test_resample_across_dst():
    df1 = DataFrame([1477786980, 1477790580], columns=['ts'])
    dti1 = DatetimeIndex(pd.to_datetime(df1.ts, unit='s').dt.tz_localize('UTC').dt.tz_convert('Europe/Madrid'))
    dti1 = dti1.as_unit('ns')
    df = DataFrame([5, 5], index=dti1)
    expected = Series([5.0, 5.0], index=DatetimeIndex(['2011-01-06 10:59:05', '2011-01-06 12:54:09'], tz='Europe/Madrid').as_unit('ns'))
    result = df.resample('h').ohlc()
    exp = pd.concat([df['VOLUME'].resample('h').ohlc(), df['PRICE'].resample('h').ohlc()], axis=1, keys=df.columns)
    tm.assert_frame_equal(exp, RESULT)

def test_downsample_across_dst(unit):
    tz = zoneinfo.ZoneInfo('Europe/Berlin')
    dt = datetime(2014, 10, 26)
    dates = date_range(dt.astimezone(tz), periods=4, freq='2h').as_unit(unit)
    df = DataFrame({'a': [5, 5, 5, 5]}, index=dates, dtype='int64')
    result = df.resample('h').ffill()
    expected = pd.DataFrame([5.0, np.nan] * 3 + [5.0], index=date_range(dt.astimezone(tz), periods=7, freq='h').as_unit(unit))
    tm.assert_frame_equal(result, expected)

def test_downsample_across_dst_weekly(unit):
    df = DataFrame([11, 12, 13, 14, 15], index=DatetimeIndex(['2017-03-25', '2017-03-26', '2017-03-27', '2017-03-28', '2017-03-29'], tz='Europe/Amsterdam').as_unit(unit))
    result = df.resample('1W-MON').sum()
    expected = DataFrame([23, 42], index=DatetimeIndex(['2017-03-26', '2017-04-02'], tz='Europe/Amsterdam', freq='W').as_unit(unit))
    tm.assert_frame_equal(result, expected)

def test_downsample_across_dst_weekly_2(unit):
    idx = date_range('2013-04-01', '2013-05-01', tz='Europe/London', freq='h').as_unit(unit)
    s = Series(range(len(idx)), index=idx)
    result = s.resample('W').mean()
    expected = Series(np.nan, index=date_range('2013-04-07', freq='W', periods=5, tz='Europe/London').as_unit(unit), dtype='float64')
    tm.assert_series_equal(result, expected)

def test_resample_lowercase_freq(unit):
    ts = Series([1, 2, 3], index=date_range('2021-01-01', periods=3))
    with tm.assert_produces_warning(FutureWarning):
        result = ts.resample('d').mean()
    expected = ts.resample('D').mean()
    tm.assert_series_equal(result, expected)

def test_resample_origin_prime_freq(unit):
    start, end = ('2000-10-01 23:30:00+0500', '2000-12-02 00:30:00+0500')
    rng = date_range(start, end, freq='7min').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    exp_rng = date_range('2000-10-01 23:14:00', '2000-10-02 00:22:00', freq='17min').as_unit(unit)
    resampled = ts.resample('17min').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample('17min', origin='start_day').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    exp_rng = date_range('2000-10-01 23:30:00', '2000-10-02 00:21:00', freq='17min').as_unit(unit)
    resampled = ts.resample('17min', origin='start').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample('17min', offset='23h30min').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    resampled = ts.resample('17min', origin='start_day', offset='23h30min').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    exp_rng = date_range('2000-10-01 23:18:00', '2000-10-02 00:26:00', freq='17min').as_unit(unit)
    resampled = ts.resample('17min', origin='epoch').mean()
    tm.assert_index_equal(resampled.index, exp_rng)
    exp_rng = date_range('2000-10-01 23:24:00', '2000-10-02 00:15:00', freq='17min').as_unit(unit)
    resampled = ts.resample('17min', origin='2000-01-01').mean()
    tm.assert_index_equal(resampled.index, exp_rng)

def test_resample_origin_with_tz_day_vs_24h(unit):
    start, end = ('2000-10-01 23:30:00+0500', '2000-12-02 00:30:00+0500')
    rng = date_range(start, end, freq='7min').as_unit(unit)
    random_values = np.random.default_rng(2).standard_normal(len(rng))
    ts_1 = Series(random_values, index=rng)
    result_1 = ts_1.resample('D', origin='epoch').mean()
    result_2 = ts_1.resample('24h', origin='epoch').mean()
    tm.assert_series_equal(result_1, result_2)
    ts_no_tz = ts_1.tz_localize(None)
    result_3 = ts_no_tz.resample('D', origin='epoch').mean()
    result_4 = ts_no_tz.resample('24h', origin='epoch').mean()
    tm.assert_series_equal(result_1, result_3.tz_localize(rng.tz), check_freq=False)
    tm.assert_series_equal(result_1, result_4.tz_localize(rng.tz), check_freq=False)
    start, end = ('2000-10-01 23:30:00+0200', '2000-12-02 00:30:00+0200')
    rng = date_range(start, end, freq='7min').as_unit(unit)
    ts_2 = Series(random_values, index=rng)
    result_5 = ts_2.resample('D', origin='epoch').mean()
    result_6 = ts_2.resample('24h', origin='epoch').mean()
    tm.assert_series_equal(result_1.tz_localize(None), result_5.tz_localize(None))
    tm.assert_series_equal(result_1.tz_localize(None), result_6.tz_localize(None))

def test_resample_origin_with_day_freq_on_dst_double(unit):
    tz = 'America/Chicago'

    def _create_series(values, timestamps, freq='D'):
        return Series(values, index=DatetimeIndex([Timestamp(t, tz=tz) for t in timestamps], freq=freq, ambiguous=True).as_unit(unit))
    start = Timestamp('2013-11-02', tz=tz)
    end = Timestamp('2013-11-03 23:59', tz=tz)
    rng = date_range(start, end, freq='1h').as_unit(unit)
    ts = Series(np.ones(len(rng)), index=rng)
    expected = _create_series([24.0, 25.0], ['2013-11-02', '2013-11-03'])
    for origin in ['epoch', 'start', 'start_day', start, None]:
        result = ts.resample('D', origin=origin).sum()
        tm.assert_series_equal(result, expected)
    start = Timestamp('2013-11-03', tz=tz)
    end = Timestamp('2013-11-03 23:59', tz=tz)
    rng = date_range(start, end, freq='1h').as_unit(unit)
    ts = Series(np.ones(len(rng)), index=rng)
    expected_ts = ['2013-11-02 22:00-05:00', '2013-11-03 22:00-06:00']
    expected = _create_series([23.0, 2.0], expected_ts)
    result = ts.resample('D', origin='start', offset='-2h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 22:00-05:00', '2013-11-03 21:00-06:00']
    expected = _create_series([22.0, 3.0], expected_ts, freq='24h')
    result = ts.resample('24h', origin='start', offset='-2h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 02:00-05:00', '2013-11-03 02:00:00-0500']
    expected = _create_series([3.0, 22.0], expected_ts)
    result = ts.resample('D', origin='start', offset='2h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 23:00-05:00', '2013-11-03 23:00-06:00']
    expected = _create_series([24.0, 1.0], expected_ts)
    result = ts.resample('D', origin='start', offset='-1h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 01:00-05:00', '2013-11-03 01:00:00-0500']
    expected = _create_series([1.0, 24.0], expected_ts)
    result = ts.resample('D', origin='start', offset='1h').sum()
    tm.assert_series_equal(result, expected)

def test_resample_dst_midnight_last_nonexistent():
    ts = Series(1, date_range('2024-04-19', '2024-04-20', tz='Africa/Cairo', freq='15min'))
    expected = Series([len(ts)], index=DatetimeIndex([ts.index[0]], freq='7D'))
    result = ts.resample('7D').sum()
    tm.assert_series_equal(result, expected)

def test_resample_dtype_preservation(unit):
    df = DataFrame({'date': date_range(start='2016-01-01', periods=4, freq='W').as_unit(unit), 'group': [1, 1, 2, 2], 'val': Series([5, 6, 7, 8], dtype='int32')}).set_index('date')
    result = df.resample('1D').ffill()
    assert result.val.dtype == np.int32
    result = df.groupby('group').resample('1D').ffill()
    assert result.val.dtype == np.int32

def test_resample_segfault(unit):
    all_wins_and_wagers = [(1, datetime(2013, 10, 1, 16, 20), 1, 0), (2, datetime(2013, 10, 1, 16, 10), 1, 0), (2, datetime(2013, 10, 1, 18, 15), 1, 0), (2, datetime(2013, 10, 1, 16, 10, 31), 1, 0)]
    df = DataFrame.from_records(all_wins_and_wagers, columns=('ID', 'timestamp', 'A', 'B')).set_index('timestamp')
    df.index = df.index.as_unit(unit)
    result = df.groupby('ID').resample('5min').sum()
    expected = df.groupby('ID').apply(lambda x: x.resample('5min').sum())
    tm.assert_frame_equal(result, expected)

def test_resample_dtype_preservation(unit):
    df = DataFrame({'date': date_range(start='2016-01-01', periods=4, freq='W').as_unit(unit), 'group': [1, 1, 2, 2], 'val': Series([5, 6, 7, 8], dtype='int32')}).set_index('date')
    result = df.resample('1D').ffill()
    assert result.val.dtype == np.int32
    result = df.groupby('group').resample('1D').ffill()
    assert result.val.dtype == np.int32

def test_resample_dtype_coercion(unit):
    pytest.importorskip('scipy.interpolate')
    df = {'a': [1, 3, 1, 4]}
    df = DataFrame(df, index=date_range('2017-01-01', '2017-01-04').as_unit(unit))
    expected = df.astype('float64').resample('h').mean()['a'].interpolate('cubic')
    result = df.resample('h')['a'].mean().interpolate('cubic')
    tm.assert_series_equal(result, expected)
    result = df.resample('h').mean()['a'].interpolate('cubic')
    tm.assert_series_equal(result, expected)

def test_weekly_resample_buglet(unit):
    rng = date_range('1/1/2000', freq='B', periods=20).as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    resampled = ts.resample('W').mean()
    expected = ts.resample('W-SUN').mean()
    tm.assert_series_equal(resampled, expected)

def test_monthly_resample_error(unit):
    dates = date_range('4/16/2012 20:00', periods=5000, freq='h').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(dates)), index=dates)
    ts.resample('ME')

def test_nanosecond_resample_error(unit):
    start = 1443707890427
    exp_start = 1443707890400
    indx = date_range(start=pd.to_datetime(start), periods=10, freq='100ns')
    ts = Series(range(len(indx)), index=indx)
    r = ts.resample(pd.tseries.offsets.Nano(100))
    result = r.agg('mean')
    exp_indx = date_range(start=pd.to_datetime(exp_start), periods=10, freq='100ns')
    exp = Series(range(len(exp_indx)), index=exp_indx, dtype=float)
    tm.assert_series_equal(result, exp)

def test_resample_anchored_intraday(unit):
    rng = date_range('1/1/2012', '4/1/2012', freq='100min').as_unit(unit)
    df = DataFrame(rng.month, index=rng)
    result = df.resample('ME').mean()
    expected = df.resample('ME').mean().to_period()
    expected = expected.to_timestamp(how='end')
    expected.index += Timedelta(1, 'ns') - Timedelta(1, 'D')
    expected.index = expected.index.as_unit(unit)._with_freq('infer')
    assert expected.index.freq == 'ME'
    tm.assert_frame_equal(result, expected)
    result = df.resample('ME', closed='left').mean()
    exp = df.shift(1, freq='D').resample('ME').mean().to_period()
    exp = exp.to_timestamp(how='end')
    exp.index = exp.index + Timedelta(1, 'ns') - Timedelta(1, 'D')
    exp.index = exp.index.as_unit(unit)._with_freq('infer')
    assert exp.index.freq == 'ME'
    tm.assert_frame_equal(result, exp)

def test_resample_anchored_intraday2(unit):
    rng = date_range('1/1/2012', '4/1/2012', freq='100min').as_unit(unit)
    df = DataFrame(rng.month, index=rng)
    result = df.resample('QE').mean()
    expected = df.resample('QE').mean().to_period()
    expected = expected.to_timestamp(how='end')
    expected.index += Timedelta(1, 'ns') - Timedelta(1, 'D')
    expected.index._data.freq = 'QE'
    expected.index._freq = lib.no_default
    expected.index = expected.index.as_unit(unit)
    tm.assert_frame_equal(result, expected)
    result = df.resample('QE', closed='left').mean()
    expected = df.shift(1, freq='D').resample('QE').mean()
    expected = expected.to_period()
    expected = expected.to_timestamp(how='end')
    expected.index += Timedelta(1, 'ns') - Timedelta(1, 'D')
    expected.index._data.freq = 'QE'
    expected.index._freq = lib.no_default
    expected.index = expected.index.as_unit(unit)
    tm.assert_frame_equal(result, expected)

def test_resample_anchored_intraday3(simple_date_range_series, unit):
    ts = simple_date_range_series('2012-04-29 23:00', '2012-04-30 5:00', freq='h')
    ts.index = ts.index.as_unit(unit)
    resampled = ts.resample('ME').mean()
    assert len(resampled) == 1

@pytest.mark.parametrize('freq', ['MS', 'BMS', 'QS-MAR', 'YS-DEC', 'YS-JUN'])
def test_resample_anchored_monthstart(simple_date_range_series, freq, unit):
    ts = simple_date_range_series('1/1/2000', '12/31/2002')
    ts.index = ts.index.as_unit(unit)
    ts.resample(freq).mean()

@pytest.mark.parametrize('label, sec', [[None, 2.0], ['right', '4.2']])
def test_resample_anchored_multiday(label, sec):
    index1 = date_range('2014-10-14 23:06:23.206', periods=3, freq='400ms')
    index2 = date_range('2014-10-15 23:00:00', periods=2, freq='2200ms')
    index = index1.union(index2)
    s = Series(np.random.default_rng(2).standard_normal(5), index=index)
    result = s.resample('2200ms', label=label).mean()
    assert result.index[-1] == Timestamp(f'2014-10-15 23:00:{sec}00')

def test_resample_origin_with_tz_day_vs_24h(unit):
    start, end = ('2000-10-01 23:30:00+0500', '2000-12-02 00:30:00+0500')
    rng = date_range(start, end, freq='7min').as_unit(unit)
    random_values = np.random.default_rng(2).standard_normal(len(rng))
    ts_1 = Series(random_values, index=rng)
    result_1 = ts_1.resample('D', origin='epoch').mean()
    result_2 = ts_1.resample('24h', origin='epoch').mean()
    tm.assert_series_equal(result_1, result_2)
    ts_no_tz = ts_1.tz_localize(None)
    result_3 = ts_no_tz.resample('D', origin='epoch').mean()
    result_4 = ts_no_tz.resample('24h', origin='epoch').mean()
    tm.assert_series_equal(result_1, result_3.tz_localize(rng.tz), check_freq=False)
    tm.assert_series_equal(result_1, result_4.tz_localize(rng.tz), check_freq=False)
    start, end = ('2000-10-01 23:30:00+0200', '2000-12-02 00:30:00+0200')
    rng = date_range(start, end, freq='7min').as_unit(unit)
    ts_2 = Series(random_values, index=rng)
    result_5 = ts_2.resample('D', origin='epoch').mean()
    result_6 = ts_2.resample('24h', origin='epoch').mean()
    tm.assert_series_equal(result_1.tz_localize(None), result_5.tz_localize(None))
    tm.assert_series_equal(result_1.tz_localize(None), result_6.tz_localize(None))

def test_resample_origin_with_day_freq_on_dst_double(unit):
    tz = 'America/Chicago'

    def _create_series(values, timestamps, freq='D'):
        return Series(values, index=DatetimeIndex([Timestamp(t, tz=tz) for t in timestamps], freq=freq, ambiguous=True).as_unit(unit))
    start = Timestamp('2013-11-02', tz=tz)
    end = Timestamp('2013-11-03 23:59', tz=tz)
    rng = date_range(start, end, freq='1h').as_unit(unit)
    ts = Series(np.ones(len(rng)), index=rng)
    expected = _create_series([24.0, 25.0], ['2013-11-02', '2013-11-03'])
    for origin in ['epoch', 'start', 'start_day', start, None]:
        result = ts.resample('D', origin=origin).sum()
        tm.assert_series_equal(result, expected)
    start = Timestamp('2013-11-03', tz=tz)
    end = Timestamp('2013-11-03 23:59', tz=tz)
    rng = date_range(start, end, freq='1h').as_unit(unit)
    ts = Series(np.ones(len(rng)), index=rng)
    expected_ts = ['2013-11-02 22:00-05:00', '2013-11-03 22:00-06:00']
    expected = _create_series([23.0, 2.0], expected_ts)
    result = ts.resample('D', origin='start', offset='-2h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 22:00-05:00', '2013-11-03 21:00-06:00']
    expected = _create_series([22.0, 3.0], expected_ts, freq='24h')
    result = ts.resample('24h', origin='start', offset='-2h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 02:00-05:00', '2013-11-03 02:00:00-0500']
    expected = _create_series([3.0, 22.0], expected_ts)
    result = ts.resample('D', origin='start', offset='2h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 23:00-05:00', '2013-11-03 23:00-06:00']
    expected = _create_series([24.0, 1.0], expected_ts)
    result = ts.resample('D', origin='start', offset='-1h').sum()
    tm.assert_series_equal(result, expected)
    expected_ts = ['2013-11-02 01:00-05:00', '2013-11-03 01:00:00-0500']
    expected = _create_series([1.0, 24.0], expected_ts)
    result = ts.resample('D', origin='start', offset='1h').sum()
    tm.assert_series_equal(result, expected)

def test_resample_dst_midnight_last_nonexistent():
    ts = Series(1, date_range('2024-04-19', '2024-04-20', tz='Africa/Cairo', freq='15min'))
    expected = Series([len(ts)], index=DatetimeIndex([ts.index[0]], freq='7D'))
    result = ts.resample('7D').sum()
    tm.assert_series_equal(result, expected)

def test_resample_daily_anchored(unit):
    rng = date_range('1/1/2012', '4/1/2012', freq='100min').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts[:2] = np.nan
    result = ts[2:].resample('D', closed='left', label='left').mean()
    expected = ts.resample('D', closed='left', label='left').mean()
    tm.assert_series_equal(result, expected)

def test_resample_to_period_monthly_buglet(unit):
    rng = date_range('1/1/2000', '12/31/2000').as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    result = ts.resample('ME').mean().to_period()
    exp_index = period_range('Jan-2000', 'Dec-2000', freq='M')
    tm.assert_index_equal(result.index, exp_index)

def test_period_with_agg():
    s2 = Series(np.random.default_rng(2).integers(0, 5, 50), index=period_range('2012-01-01', freq='h', periods=50), dtype='float64')
    expected = s2.to_timestamp().resample('D').mean().to_period()
    msg = 'Resampling with a PeriodIndex is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = s2.resample('D')
    result = rs.agg(lambda x: x.mean())
    tm.assert_series_equal(result, expected)

def test_resample_segfault(unit):
    all_wins_and_wagers = [(1, datetime(2013, 10, 1, 16, 20), 1, 0), (2, datetime(2013, 10, 1, 16, 10), 1, 0), (2, datetime(2013, 10, 1, 18, 15), 1, 0), (2, datetime(2013, 10, 1, 16, 10, 31), 1, 0)]
    df = DataFrame.from_records(all_wins_and_wagers, columns=('ID', 'timestamp', 'A', 'B')).set_index('timestamp')
    df.index = df.index.as_unit(unit)
    result = df.groupby('ID').resample('5min').sum()
    expected = df.groupby('ID').apply(lambda x: x.resample('5min').sum())
    tm.assert_frame_equal(result, expected)

def test_resample_dtype_preservation(unit):
    df = DataFrame({'date': date_range(start='2016-01-01', periods=4, freq='W').as_unit(unit), 'group': [1, 1, 2, 2], 'val': Series([5, 6, 7, 8], dtype='int32')}).set_index('date')
    result = df.resample('1D').ffill()
    assert result.val.dtype == np.int32
    result = df.groupby('group').resample('1D').ffill()
    assert result.val.dtype == np.int32

def test_resample_dtype_coercion(unit):
    pytest.importorskip('scipy.interpolate')
    df = {'a': [1, 3, 1, 4]}
    df = DataFrame(df, index=date_range('2017-01-01', '2017-01-04').as_unit(unit))
    expected = df.astype('float64').resample('h').mean()['a'].interpolate('cubic')
    result = df.resample('h')['a'].mean().interpolate('cubic')
    tm.assert_series_equal(result, expected)
    result = df.resample('h').mean()['a'].interpolate('cubic')
    tm.assert_series_equal(result, expected)
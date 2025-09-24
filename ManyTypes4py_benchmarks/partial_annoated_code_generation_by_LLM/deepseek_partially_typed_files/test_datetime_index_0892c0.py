from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
def simple_date_range_series() -> Callable[[str, str, str], Series]:
    """
    Series with date range index and random data for test purposes.
    """

    def _simple_date_range_series(start: str, end: str, freq: str = 'D') -> Series:
        rng: DatetimeIndex = date_range(start, end, freq=freq)
        return Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    return _simple_date_range_series

def test_custom_grouper(unit: str) -> None:
    index: DatetimeIndex = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='Min')
    dti: DatetimeIndex = index.as_unit(unit)
    s: Series = Series(np.array([1] * len(dti)), index=dti, dtype='int64')
    b: Grouper = Grouper(freq=Minute(5))
    g = s.groupby(b)
    g.ohlc()
    funcs: List[str] = ['sum', 'mean', 'prod', 'min', 'max', 'var']
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
    idx: DatetimeIndex = dti[0:-1:5]
    idx = idx.append(dti[-1:])
    idx = DatetimeIndex(idx, freq='5min').as_unit(unit)
    expect: Series = Series(arr, index=idx)
    result: Series = g.agg('sum')
    tm.assert_series_equal(result, expect)

def test_custom_grouper_df(unit: str) -> None:
    index: DatetimeIndex = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='D')
    b: Grouper = Grouper(freq=Minute(5), closed='right', label='right')
    dti: DatetimeIndex = index.as_unit(unit)
    df: DataFrame = DataFrame(np.random.default_rng(2).random((len(dti), 10)), index=dti, dtype='float64')
    r: DataFrame = df.groupby(b).agg('sum')
    assert len(r.columns) == 10
    assert len(r.index) == 2593

@pytest.mark.parametrize('closed, expected', [('right', lambda s: Series([s.iloc[0], s[1:6].mean(), s[6:11].mean(), s[11:].mean()], index=date_range('1/1/2000', periods=4, freq='5min', name='index'))), ('left', lambda s: Series([s[:5].mean(), s[5:10].mean(), s[10:].mean()], index=date_range('1/1/2000 00:05', periods=3, freq='5min', name='index')))])
def test_resample_basic(closed: str, expected: Callable[[Series], Series], unit: str) -> None:
    index: DatetimeIndex = date_range('1/1/2000 00:00:00', '1/1/2000 00:13:00', freq='Min')
    s: Series = Series(range(len(index)), index=index)
    s.index.name = 'index'
    s.index = s.index.as_unit(unit)
    expected_series: Series = expected(s)
    expected_series.index = expected_series.index.as_unit(unit)
    result: Series = s.resample('5min', closed=closed, label='right').mean()
    tm.assert_series_equal(result, expected_series)

def test_resample_integerarray(unit: str) -> None:
    ts: Series = Series(range(9), index=date_range('1/1/2000', periods=9, freq='min').as_unit(unit), dtype='Int64')
    result: Series = ts.resample('3min').sum()
    expected: Series = Series([3, 12, 21], index=date_range('1/1/2000', periods=3, freq='3min').as_unit(unit), dtype='Int64')
    tm.assert_series_equal(result, expected)
    result = ts.resample('3min').mean()
    expected = Series([1, 4, 7], index=date_range('1/1/2000', periods=3, freq='3min').as_unit(unit), dtype='Float64')
    tm.assert_series_equal(result, expected)

def test_resample_basic_grouper(unit: str) -> None:
    index: DatetimeIndex = date_range('1/1/2000 00:00:00', '1/1/2000 00:13:00', freq='Min')
    s: Series = Series(range(len(index)), index=index)
    s.index.name = 'index'
    s.index = s.index.as_unit(unit)
    result: Series = s.resample('5Min').last()
    grouper: Grouper = Grouper(freq=Minute(5), closed='left', label='left')
    expected: Series = s.groupby(grouper).agg(lambda x: x.iloc[-1])
    tm.assert_series_equal(result, expected)

@pytest.mark.filterwarnings("ignore:The 'convention' keyword in Series.resample:FutureWarning")
@pytest.mark.parametrize('keyword,value', [('label', 'righttt'), ('closed', 'righttt'), ('convention', 'starttt')])
def test_resample_string_kwargs(keyword: str, value: str, unit: str) -> None:
    index: DatetimeIndex = date_range('1/1/2000 00:00:00', '1/1/2000 00:13:00', freq='Min')
    series: Series = Series(range(len(index)), index=index)
    series.index.name = 'index'
    series.index = series.index.as_unit(unit)
    msg: str = f'Unsupported value {value} for `{keyword}`'
    with pytest.raises(ValueError, match=msg):
        series.resample('5min', **{keyword: value})

def test_resample_how(downsample_method: str, unit: str) -> None:
    if downsample_method == 'ohlc':
        pytest.skip('covered by test_resample_how_ohlc')
    index: DatetimeIndex = date_range('1/1/2000 00:00:00', '1/1/2000 00:13:00', freq='Min')
    s: Series = Series(range(len(index)), index=index)
    s.index.name = 'index'
    s.index = s.index.as_unit(unit)
    grouplist: np.ndarray = np.ones_like(s)
    grouplist[0] = 0
    grouplist[1:6] = 1
    grouplist[6:11] = 2
    grouplist[11:] = 3
    expected: Series = s.groupby(grouplist).agg(downsample_method)
    expected.index = date_range('1/1/2000', periods=4, freq='5min', name='index').as_unit(unit)
    result: Series = getattr(s.resample('5min', closed='right', label='right'), downsample_method)()
    tm.assert_series_equal(result, expected)

def test_resample_how_ohlc(unit: str) -> None:
    index: DatetimeIndex = date_range('1/1/2000 00:00:00', '1/1/2000 00:13:00', freq='Min')
    s: Series = Series(range(len(index)), index=index)
    s.index.name = 'index'
    s.index = s.index.as_unit(unit)
    grouplist: np.ndarray = np.ones_like(s)
    grouplist[0] = 0
    grouplist[1:6] = 1
    grouplist[6:11] = 2
    grouplist[11:] = 3

    def _ohlc(group: Series) -> List[Union[float, int]]:
        if isna(group).all():
            return np.repeat(np.nan, 4)
        return [group.iloc[0], group.max(), group.min(), group.iloc[-1]]
    expected: DataFrame = DataFrame(s.groupby(grouplist).agg(_ohlc).values.tolist(), index=date_range('1/1/2000', periods=4, freq='5min', name='index').as_unit(unit), columns=['open', 'high', 'low', 'close'])
    result: DataFrame = s.resample('5min', closed='right', label='right').ohlc()
    tm.assert_frame_equal(result, expected)

def test_resample_how_callables(unit: str) -> None:
    data: np.ndarray = np.arange(5, dtype=np.int64)
    msg: str = "'d' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        ind: DatetimeIndex = date_range(start='2014-01-01', periods=len(data), freq='d').as_unit(unit)
    df: DataFrame = DataFrame({'A': data, 'B': data}, index=ind)

    def fn(x: DataFrame, a: int = 1) -> str:
        return str(type(x))

    class FnClass:

        def __call__(self, x: DataFrame) -> str:
            return str(type(x))
    df_standard: DataFrame = df.resample('ME').apply(fn)
    df_lambda: DataFrame = df.resample('ME').apply(lambda x: str(type(x)))
    df_partial: DataFrame = df.resample('ME').apply(partial(fn))
    df_partial2: DataFrame = df.resample('ME').apply(partial(fn, a=2))
    df_class: DataFrame = df.resample('ME').apply(FnClass())
    tm.assert_frame_equal(df_standard, df_lambda)
    tm.assert_frame_equal(df_standard, df_partial)
    tm.assert_frame_equal(df_standard, df_partial2)
    tm.assert_frame_equal(df_standard, df_class)

def test_resample_rounding(unit: str) -> None:
    ts: List[str] = ['2014-11-08 00:00:01', '2014-11-08 00:00:02', '2014-11-08 00:00:02', '2014-11-08 00:00:03', '2014-11-08 00:00:07', '2014-11-08 00:00:07', '2014-11-08 00:00:08', '2014-11-08 00:00:08', '2014-11-08 00:00:08', '2014-11-08 00:00:09', '2014-11-08 00:00:10', '2014-11-08 00:00:11', '2014-11-08 00:00:11', '2014-11-08 00:00:13', '2014-11-08 00:00:14', '2014-11-08 00:00:15', '2014-11-08 00:00:17', '2014-11-08 00:00:20', '2014-11-08 00:00:21']
    df: DataFrame = DataFrame({'value': [1] * 19}, index=pd.to_datetime(ts))
    df.index = df.index.as_unit(unit)
    result: DataFrame = df.resample('6s').sum()
    expected: DataFrame = DataFrame({'value': [4, 9, 4, 2]}, index=date_range('2014-11-08', freq='6s', periods=4).as_unit(unit))
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

def test_resample_basic_from_daily(unit: str) -> None:
    dti: DatetimeIndex = date_range(start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq='D', name='index').as_unit(unit)
    s: Series = Series(np.random.default_rng(2).random(len(dti)), dti)
    msg: str = "'w-sun' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result: Series = s.resample('w-sun').last()
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

def test_resample_upsampling_picked_but_not_correct(unit: str) -> None:
    dates: DatetimeIndex = date_range('01-Jan-2014', '05-Jan-2014', freq='D').as_unit(unit)
    series: Series = Series(1, index=dates)
    result: Series = series.resample('D').mean()
    assert result.index[0] == dates[0]
    s: Series = Series(np.arange(1.0, 6), index=[datetime(1975, 1, i, 12, 0) for
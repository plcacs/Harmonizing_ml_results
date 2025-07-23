from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pytest
from pandas._libs import lib
import pandas as pd
from pandas import DataFrame, NamedAgg, Series
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.resample import DatetimeIndexResampler
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

@pytest.fixture
def dti() -> pd.DatetimeIndex:
    return date_range(start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq='Min')

@pytest.fixture
def _test_series(dti: pd.DatetimeIndex) -> Series:
    return Series(np.random.default_rng(2).random(len(dti)), dti)

@pytest.fixture
def test_frame(dti: pd.DatetimeIndex, _test_series: Series) -> DataFrame:
    return DataFrame({'A': _test_series, 'B': _test_series, 'C': np.arange(len(dti))})

def test_str(_test_series: Series) -> None:
    r = _test_series.resample('h')
    assert 'DatetimeIndexResampler [freq=<Hour>, closed=left, label=left, convention=start, origin=start_day]' in str(r)
    r = _test_series.resample('h', origin='2000-01-01')
    assert 'DatetimeIndexResampler [freq=<Hour>, closed=left, label=left, convention=start, origin=2000-01-01 00:00:00]' in str(r)

def test_api(_test_series: Series) -> None:
    r = _test_series.resample('h')
    result = r.mean()
    assert isinstance(result, Series)
    assert len(result) == 217
    r = _test_series.to_frame().resample('h')
    result = r.mean()
    assert isinstance(result, DataFrame)
    assert len(result) == 217

def test_groupby_resample_api() -> None:
    df = DataFrame({'date': date_range(start='2016-01-01', periods=4, freq='W'), 'group': [1, 1, 2, 2], 'val': [5, 6, 7, 8]}).set_index('date')
    i = date_range('2016-01-03', periods=8).tolist() + date_range('2016-01-17', periods=8).tolist()
    index = pd.MultiIndex.from_arrays([[1] * 8 + [2] * 8, i], names=['group', 'date'])
    expected = DataFrame({'val': [5] * 7 + [6] + [7] * 7 + [8]}, index=index)
    result = df.groupby('group').apply(lambda x: x.resample('1D').ffill())[['val']]
    tm.assert_frame_equal(result, expected)

def test_groupby_resample_on_api() -> None:
    df = DataFrame({'key': ['A', 'B'] * 5, 'dates': date_range('2016-01-01', periods=10), 'values': np.random.default_rng(2).standard_normal(10)})
    expected = df.set_index('dates').groupby('key').resample('D').mean()
    result = df.groupby('key').resample('D', on='dates').mean()
    tm.assert_frame_equal(result, expected)

def test_resample_group_keys() -> None:
    df = DataFrame({'A': 1, 'B': 2}, index=date_range('2000', periods=10))
    expected = df.copy()
    g = df.resample('5D', group_keys=False)
    result = g.apply(lambda x: x)
    tm.assert_frame_equal(result, expected)
    g = df.resample('5D')
    result = g.apply(lambda x: x)
    tm.assert_frame_equal(result, expected)
    expected.index = pd.MultiIndex.from_arrays([pd.to_datetime(['2000-01-01', '2000-01-06']).as_unit('ns').repeat(5), expected.index])
    g = df.resample('5D', group_keys=True)
    result = g.apply(lambda x: x)
    tm.assert_frame_equal(result, expected)

def test_pipe(test_frame: DataFrame, _test_series: Series) -> None:
    r = _test_series.resample('h')
    expected = r.max() - r.mean()
    result = r.pipe(lambda x: x.max() - x.mean())
    tm.assert_series_equal(result, expected)
    r = test_frame.resample('h')
    expected = r.max() - r.mean()
    result = r.pipe(lambda x: x.max() - x.mean())
    tm.assert_frame_equal(result, expected)

def test_getitem(test_frame: DataFrame) -> None:
    r = test_frame.resample('h')
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns)
    r = test_frame.resample('h')['B']
    assert r._selected_obj.name == test_frame.columns[1]
    r = test_frame.resample('h')['A', 'B']
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns[[0, 1]])
    r = test_frame.resample('h')['A', 'B']
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns[[0, 1]])

@pytest.mark.parametrize('key', [['D'], ['A', 'D']])
def test_select_bad_cols(key: List[str], test_frame: DataFrame) -> None:
    g = test_frame.resample('h')
    msg = '^\\"Columns not found: \'D\'\\"$'
    with pytest.raises(KeyError, match=msg):
        g[key]

def test_attribute_access(test_frame: DataFrame) -> None:
    r = test_frame.resample('h')
    tm.assert_series_equal(r.A.sum(), r['A'].sum())

@pytest.mark.parametrize('attr', ['groups', 'ngroups', 'indices'])
def test_api_compat_before_use(attr: str) -> None:
    rng = date_range('1/1/2012', periods=100, freq='s')
    ts = Series(np.arange(len(rng)), index=rng)
    rs = ts.resample('30s')
    getattr(rs, attr)
    rs.mean()
    getattr(rs, attr)

def tests_raises_on_nuisance(test_frame: DataFrame, using_infer_string: bool) -> None:
    df = test_frame
    df['D'] = 'foo'
    r = df.resample('h')
    result = r[['A', 'B']].mean()
    expected = pd.concat([r.A.mean(), r.B.mean()], axis=1)
    tm.assert_frame_equal(result, expected)
    expected = r[['A', 'B', 'C']].mean()
    msg = re.escape('agg function failed [how->mean,dtype->')
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        r.mean()
    result = r.mean(numeric_only=True)
    tm.assert_frame_equal(result, expected)

def test_downsample_but_actually_upsampling() -> None:
    rng = date_range('1/1/2012', periods=100, freq='s')
    ts = Series(np.arange(len(rng), dtype='int64'), index=rng)
    result = ts.resample('20s').asfreq()
    expected = Series([0, 20, 40, 60, 80], index=date_range('2012-01-01 00:00:00', freq='20s', periods=5))
    tm.assert_series_equal(result, expected)

def test_combined_up_downsampling_of_irregular() -> None:
    rng = date_range('1/1/2012', periods=100, freq='s')
    ts = Series(np.arange(len(rng)), index=rng)
    ts2 = ts.iloc[[0, 1, 2, 3, 5, 7, 11, 15, 16, 25, 30]]
    result = ts2.resample('2s').mean().ffill()
    expected = Series([0.5, 2.5, 5.0, 7.0, 7.0, 11.0, 11.0, 15.0, 16.0, 16.0, 16.0, 16.0, 25.0, 25.0, 25.0, 30.0], index=pd.DatetimeIndex(['2012-01-01 00:00:00', '2012-01-01 00:00:02', '2012-01-01 00:00:04', '2012-01-01 00:00:06', '2012-01-01 00:00:08', '2012-01-01 00:00:10', '2012-01-01 00:00:12', '2012-01-01 00:00:14', '2012-01-01 00:00:16', '2012-01-01 00:00:18', '2012-01-01 00:00:20', '2012-01-01 00:00:22', '2012-01-01 00:00:24', '2012-01-01 00:00:26', '2012-01-01 00:00:28', '2012-01-01 00:00:30'], dtype='datetime64[ns]', freq='2s'))
    tm.assert_series_equal(result, expected)

def test_transform_series(_test_series: Series) -> None:
    r = _test_series.resample('20min')
    expected = _test_series.groupby(pd.Grouper(freq='20min')).transform('mean')
    result = r.transform('mean')
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('on', [None, 'date'])
def test_transform_frame(on: Optional[str]) -> None:
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='D')
    index.name = 'date'
    df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=list('AB'), index=index)
    expected = df.groupby(pd.Grouper(freq='20min')).transform('mean')
    if on == 'date':
        expected = expected.reset_index(drop=True)
        df = df.reset_index()
    r = df.resample('20min', on=on)
    result = r.transform('mean')
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('func', [lambda x: x.resample('20min', group_keys=False), lambda x: x.groupby(pd.Grouper(freq='20min'), group_keys=False)], ids=['resample', 'groupby'])
def test_apply_without_aggregation(func: Any, _test_series: Series) -> None:
    t = func(_test_series)
    result = t.apply(lambda x: x)
    tm.assert_series_equal(result, _test_series)

def test_apply_without_aggregation2(_test_series: Series) -> None:
    grouped = _test_series.to_frame(name='foo').resample('20min', group_keys=False)
    result = grouped['foo'].apply(lambda x: x)
    tm.assert_series_equal(result, _test_series.rename('foo'))

def test_agg_consistency() -> None:
    df = DataFrame(np.random.default_rng(2).standard_normal((1000, 3)), index=date_range('1/1/2012', freq='s', periods=1000), columns=['A', 'B', 'C'])
    r = df.resample('3min')
    msg = "Label\\(s\\) \\['r1', 'r2'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        r.agg({'r1': 'mean', 'r2': 'sum'})

def test_agg_consistency_int_str_column_mix() -> None:
    df = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)), index=date_range('1/1/2012', freq='s', periods=1000), columns=[1, 'a'])
    r = df.resample('3min')
    msg = "Label\\(s\\) \\[2, 'b'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        r.agg({2: 'mean', 'b': 'sum'})

@pytest.fixture
def index() -> pd.DatetimeIndex:
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='D')
    index.name = 'date'
    return index

@pytest.fixture
def df(index: pd.DatetimeIndex) -> DataFrame:
    frame = DataFrame(np.random.default_rng(2).random((10, 2)), columns=list('AB'), index=index)
    return frame

@pytest.fixture
def df_col(df: DataFrame) -> DataFrame:
    return df.reset_index()

@pytest.fixture
def df_mult(df_col: DataFrame, index: pd.DatetimeIndex) -> DataFrame:
    df_mult = df_col.copy()
    df_mult.index = pd.MultiIndex.from_arrays([range(10), index], names=['index', 'date'])
    return df_mult

@pytest.fixture
def a_mean(df: DataFrame) -> Series:
    return df.resample('2D')['A'].mean()

@pytest.fixture
def a_std(df: DataFrame) -> Series:
    return df.resample('2D')['A'].std()

@pytest.fixture
def a_sum(df: DataFrame) -> Series:
    return df.resample('2D')['A'].sum()

@pytest.fixture
def b_mean(df: DataFrame) -> Series:
    return df.resample('2D')['B'].mean()

@pytest.fixture
def b_std(df: DataFrame) -> Series:
    return df.resample('2D')['B'].std()

@pytest.fixture
def b_sum(df: DataFrame) -> Series:
    return df.resample('2D')['B'].sum()

@pytest.fixture
def df_resample(df: DataFrame) -> DatetimeIndexResampler:
    return df.resample('2D')

@pytest.fixture
def df_col_resample(df_col: DataFrame) -> DatetimeIndexResampler:
    return df_col.resample('2D', on='date')

@pytest.fixture
def df_mult_resample(df_mult: DataFrame) -> DatetimeIndexResampler:
    return df_mult.resample('2D', level='date')

@pytest.fixture
def df_grouper_resample(df: DataFrame) -> DataFrameGroupBy:
    return df.groupby(pd.Grouper(freq='2D'))

@pytest.fixture(params=['df_resample', 'df_col_resample', 'df_mult_resample', 'df_grouper_resample'])
def cases(request: pytest.FixtureRequest) -> Any:
    return request.getfixturevalue(request.param)

def test_agg_mixed_column_aggregation(cases: Any, a_mean: Series, a_std: Series, b_mean: Series, b_std: Series, request: pytest.FixtureRequest) -> None:
    expected = pd.concat([a_mean, a_std, b_mean, b_std], axis=1)
    expected.columns = pd.MultiIndex.from_product([['A', 'B'], ['mean', '<lambda_0>']])
    if 'df_mult' in request.node.callspec.id:
        date_mean = cases['date'].mean()
        date_std = cases['date'].std()
        expected = pd.concat([date_mean, date_std, expected], axis=1)
        expected.columns = pd.MultiIndex.from_product([['date', 'A', 'B'], ['mean', '<lambda_0>']])
    result = cases.aggregate([np.mean, lambda x: np.std(x, ddof=1)])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('agg', [{'func': {'A': np.mean, 'B': lambda x: np.std(x, ddof=1)}}, {'A': ('A', np.mean), 'B': ('B', lambda x: np.std(x, ddof=1))}, {'A': NamedAgg('A', np.mean), 'B': NamedAgg('B', lambda x: np.std(x, ddof=1))}])
def test_agg_both_mean_std_named_result(cases: Any, a_mean: Series, b_std: Series, agg: Dict[str, Any]) -> None:
    expected = pd.concat([a_mean, b_std], axis=1)
    result = cases.aggregate(**agg)
    tm.assert_frame_equal(result, expected, check_like=True)

def test_agg_both_mean_std_dict_of_list(cases: Any, a_mean: Series, a_std: Series) -> None:
    expected = pd.concat([a_mean, a_std], axis=1)
    expected.columns = pd.MultiIndex.from_tuples([('A', 'mean'), ('A', 'std')])
    result = cases.aggregate({'A': ['mean', 'std']})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('agg', [{'func': ['mean', 'sum']}, {'mean': 'mean', 'sum': 'sum'}])
def test_agg_both_mean_sum(cases: Any, a_mean: Series, a_sum: Series, agg: Dict[str, Any]) -> None:
    expected = pd.concat([a_mean, a_sum], axis=1)
    expected.columns = ['mean', 'sum']
    result = cases['A'].aggregate(**agg)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('agg', [{'A': {'mean': 'mean', 'sum': 'sum'}},
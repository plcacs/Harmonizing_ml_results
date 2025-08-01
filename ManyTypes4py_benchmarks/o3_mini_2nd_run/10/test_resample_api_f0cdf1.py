#!/usr/bin/env python3
from datetime import datetime
import re
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, NamedAgg, Series
from pandas._libs import lib
from pandas.core.indexes.datetimes import date_range
import pandas._testing as tm
import pytest


@pytest.fixture
def dti() -> pd.DatetimeIndex:
    return date_range(start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq='Min')


@pytest.fixture
def _test_series(dti: pd.DatetimeIndex) -> pd.Series:
    return Series(np.random.default_rng(2).random(len(dti)), dti)


@pytest.fixture
def test_frame(dti: pd.DatetimeIndex, _test_series: pd.Series) -> pd.DataFrame:
    return DataFrame({'A': _test_series, 'B': _test_series, 'C': np.arange(len(dti))})


def test_str(_test_series: pd.Series) -> None:
    r = _test_series.resample('h')
    assert 'DatetimeIndexResampler [freq=<Hour>, closed=left, label=left, convention=start, origin=start_day]' in str(r)
    r = _test_series.resample('h', origin='2000-01-01')
    assert 'DatetimeIndexResampler [freq=<Hour>, closed=left, label=left, convention=start, origin=2000-01-01 00:00:00]' in str(r)


def test_api(_test_series: pd.Series) -> None:
    r = _test_series.resample('h')
    result = r.mean()
    assert isinstance(result, Series)
    assert len(result) == 217
    r = _test_series.to_frame().resample('h')
    result = r.mean()
    assert isinstance(result, DataFrame)
    assert len(result) == 217


def test_groupby_resample_api() -> None:
    df = DataFrame({
        'date': date_range(start='2016-01-01', periods=4, freq='W'),
        'group': [1, 1, 2, 2],
        'val': [5, 6, 7, 8]
    }).set_index('date')
    i = date_range('2016-01-03', periods=8).tolist() + date_range('2016-01-17', periods=8).tolist()
    index = pd.MultiIndex.from_arrays([[1] * 8 + [2] * 8, i], names=['group', 'date'])
    expected = DataFrame({'val': [5] * 7 + [6] + [7] * 7 + [8]}, index=index)
    result = df.groupby('group').apply(lambda x: x.resample('1D').ffill())[['val']]
    tm.assert_frame_equal(result, expected)


def test_groupby_resample_on_api() -> None:
    df = DataFrame({
        'key': ['A', 'B'] * 5,
        'dates': date_range('2016-01-01', periods=10),
        'values': np.random.default_rng(2).standard_normal(10)
    })
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
    expected.index = pd.MultiIndex.from_arrays([
        pd.to_datetime(['2000-01-01', '2000-01-06']).as_unit('ns').repeat(5),
        expected.index
    ])
    g = df.resample('5D', group_keys=True)
    result = g.apply(lambda x: x)
    tm.assert_frame_equal(result, expected)


def test_pipe(test_frame: pd.DataFrame, _test_series: pd.Series) -> None:
    r = _test_series.resample('h')
    expected = r.max() - r.mean()
    result = r.pipe(lambda x: x.max() - x.mean())
    tm.assert_series_equal(result, expected)
    r = test_frame.resample('h')
    expected = r.max() - r.mean()
    result = r.pipe(lambda x: x.max() - x.mean())
    tm.assert_frame_equal(result, expected)


def test_getitem(test_frame: pd.DataFrame) -> None:
    r = test_frame.resample('h')
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns)
    r = test_frame.resample('h')['B']
    assert r._selected_obj.name == test_frame.columns[1]
    r = test_frame.resample('h')['A', 'B']
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns[[0, 1]])
    r = test_frame.resample('h')['A', 'B']
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns[[0, 1]])


@pytest.mark.parametrize('key', [['D'], ['A', 'D']])
def test_select_bad_cols(key: List[str], test_frame: pd.DataFrame) -> None:
    g = test_frame.resample('h')
    msg = '^\\"Columns not found: \'D\'\\"$'
    with pytest.raises(KeyError, match=msg):
        g[key]


def test_attribute_access(test_frame: pd.DataFrame) -> None:
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


def tests_raises_on_nuisance(test_frame: pd.DataFrame, using_infer_string: bool) -> None:
    df = test_frame.copy()
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
    expected = Series(
        [0, 20, 40, 60, 80],
        index=date_range('2012-01-01 00:00:00', freq='20s', periods=5)
    )
    tm.assert_series_equal(result, expected)


def test_combined_up_downsampling_of_irregular() -> None:
    rng = date_range('1/1/2012', periods=100, freq='s')
    ts = Series(np.arange(len(rng)), index=rng)
    ts2 = ts.iloc[[0, 1, 2, 3, 5, 7, 11, 15, 16, 25, 30]]
    result = ts2.resample('2s').mean().ffill()
    expected = Series(
        [0.5, 2.5, 5.0, 7.0, 7.0, 11.0, 11.0, 15.0, 16.0, 16.0, 16.0, 16.0, 25.0, 25.0, 25.0, 30.0],
        index=pd.DatetimeIndex([
            '2012-01-01 00:00:00', '2012-01-01 00:00:02', '2012-01-01 00:00:04', '2012-01-01 00:00:06',
            '2012-01-01 00:00:08', '2012-01-01 00:00:10', '2012-01-01 00:00:12', '2012-01-01 00:00:14',
            '2012-01-01 00:00:16', '2012-01-01 00:00:18', '2012-01-01 00:00:20', '2012-01-01 00:00:22',
            '2012-01-01 00:00:24', '2012-01-01 00:00:26', '2012-01-01 00:00:28', '2012-01-01 00:00:30'
        ], dtype='datetime64[ns]', freq='2s')
    )
    tm.assert_series_equal(result, expected)


def test_transform_series(_test_series: pd.Series) -> None:
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


@pytest.mark.parametrize(
    'func',
    [
        lambda x: x.resample('20min', group_keys=False),
        lambda x: x.groupby(pd.Grouper(freq='20min'), group_keys=False)
    ],
    ids=['resample', 'groupby']
)
def test_apply_without_aggregation(func: Callable[[pd.Series], Any], _test_series: pd.Series) -> None:
    t = func(_test_series)
    result = t.apply(lambda x: x)
    tm.assert_series_equal(result, _test_series)


def test_apply_without_aggregation2(_test_series: pd.Series) -> None:
    grouped = _test_series.to_frame(name='foo').resample('20min', group_keys=False)
    result = grouped['foo'].apply(lambda x: x)
    tm.assert_series_equal(result, _test_series.rename('foo'))


def test_agg_consistency() -> None:
    df = DataFrame(
        np.random.default_rng(2).standard_normal((1000, 3)),
        index=date_range('1/1/2012', freq='s', periods=1000),
        columns=['A', 'B', 'C']
    )
    r = df.resample('3min')
    msg = "Label\\(s\\) \\['r1', 'r2'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        r.agg({'r1': 'mean', 'r2': 'sum'})


def test_agg_consistency_int_str_column_mix() -> None:
    df = DataFrame(
        np.random.default_rng(2).standard_normal((1000, 2)),
        index=date_range('1/1/2012', freq='s', periods=1000),
        columns=[1, 'a']
    )
    r = df.resample('3min')
    msg = "Label\\(s\\) \\[2, 'b'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        r.agg({2: 'mean', 'b': 'sum'})


@pytest.fixture
def index() -> pd.DatetimeIndex:
    idx = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='D')
    idx.name = 'date'
    return idx


@pytest.fixture
def df(index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = DataFrame(np.random.default_rng(2).random((10, 2)), columns=list('AB'), index=index)
    return frame


@pytest.fixture
def df_col(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index()


@pytest.fixture
def df_mult(df_col: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    df_mult = df_col.copy()
    df_mult.index = pd.MultiIndex.from_arrays([range(10), index], names=['index', 'date'])
    return df_mult


@pytest.fixture
def a_mean(df: pd.DataFrame) -> pd.Series:
    return df.resample('2D')['A'].mean()


@pytest.fixture
def a_std(df: pd.DataFrame) -> pd.Series:
    return df.resample('2D')['A'].std()


@pytest.fixture
def a_sum(df: pd.DataFrame) -> pd.Series:
    return df.resample('2D')['A'].sum()


@pytest.fixture
def b_mean(df: pd.DataFrame) -> pd.Series:
    return df.resample('2D')['B'].mean()


@pytest.fixture
def b_std(df: pd.DataFrame) -> pd.Series:
    return df.resample('2D')['B'].std()


@pytest.fixture
def b_sum(df: pd.DataFrame) -> pd.Series:
    return df.resample('2D')['B'].sum()


@pytest.fixture
def df_resample(df: pd.DataFrame) -> Any:
    return df.resample('2D')


@pytest.fixture
def df_col_resample(df_col: pd.DataFrame) -> Any:
    return df_col.resample('2D', on='date')


@pytest.fixture
def df_mult_resample(df_mult: pd.DataFrame) -> Any:
    return df_mult.resample('2D', level='date')


@pytest.fixture
def df_grouper_resample(df: pd.DataFrame) -> Any:
    return df.groupby(pd.Grouper(freq='2D'))


@pytest.fixture(params=['df_resample', 'df_col_resample', 'df_mult_resample', 'df_grouper_resample'])
def cases(request: Any) -> Any:
    return request.getfixturevalue(request.param)


def test_agg_mixed_column_aggregation(
    cases: Any,
    a_mean: pd.Series,
    a_std: pd.Series,
    b_mean: pd.Series,
    b_std: pd.Series,
    request: Any
) -> None:
    expected = pd.concat([a_mean, a_std, b_mean, b_std], axis=1)
    expected.columns = pd.MultiIndex.from_product([['A', 'B'], ['mean', '<lambda_0>']])
    if 'df_mult' in request.node.callspec.id:
        date_mean = cases['date'].mean()
        date_std = cases['date'].std()
        expected = pd.concat([date_mean, date_std, expected], axis=1)
        expected.columns = pd.MultiIndex.from_product([['date', 'A', 'B'], ['mean', '<lambda_0>']])
    result = cases.aggregate([np.mean, lambda x: np.std(x, ddof=1)])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'agg',
    [
        {'func': {'A': np.mean, 'B': lambda x: np.std(x, ddof=1)}},
        {'A': ('A', np.mean), 'B': ('B', lambda x: np.std(x, ddof=1))},
        {'A': NamedAgg('A', np.mean), 'B': NamedAgg('B', lambda x: np.std(x, ddof=1))}
    ]
)
def test_agg_both_mean_std_named_result(
    cases: Any, a_mean: pd.Series, b_std: pd.Series, agg: Dict[str, Any]
) -> None:
    expected = pd.concat([a_mean, b_std], axis=1)
    result = cases.aggregate(**agg)
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    'agg',
    [
        {'A': ['mean', 'std']},
        {'A': ['mean', 'std'], 'B': ['mean', 'std']}
    ]
)
def test_agg_both_mean_std_dict_of_list(
    cases: Any, a_mean: pd.Series, a_std: pd.Series
) -> None:
    expected = pd.concat([a_mean, a_std], axis=1)
    expected.columns = pd.MultiIndex.from_tuples([('A', 'mean'), ('A', 'std')])
    result = cases.aggregate({'A': ['mean', 'std']})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'agg',
    [
        {'func': {'A': np.mean, 'B': np.sum}},
        {'A': ('A', np.mean), 'B': ('B', np.sum)}
    ]
)
def test_agg_both_mean_sum(
    cases: Any, a_mean: pd.Series, a_sum: pd.Series, agg: Dict[str, Any]
) -> None:
    expected = pd.concat([a_mean, a_sum], axis=1)
    expected.columns = ['mean', 'sum']
    result = cases['A'].aggregate(**agg)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'agg',
    [
        {'A': {'mean': 'mean', 'sum': 'sum'}},
        {'A': {'mean': 'mean', 'sum': 'sum'}, 'B': {'mean2': 'mean', 'sum2': 'sum'}}
    ]
)
def test_agg_dict_of_dict_specificationerror(cases: Any, agg: Dict[str, Any]) -> None:
    msg = 'nested renamer is not supported'
    with pytest.raises(pd.errors.SpecificationError, match=msg):
        cases.aggregate(agg)


def test_agg_dict_of_lists(
    cases: Any, a_mean: pd.Series, a_std: pd.Series, b_mean: pd.Series, b_std: pd.Series
) -> None:
    expected = pd.concat([a_mean, a_std, b_mean, b_std], axis=1)
    expected.columns = pd.MultiIndex.from_tuples([
        ('A', 'mean'), ('A', 'std'), ('B', 'mean'), ('B', 'std')
    ])
    result = cases.aggregate({'A': ['mean', 'std'], 'B': ['mean', 'std']})
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    'agg',
    [
        {'func': {'A': np.sum, 'B': lambda x: np.std(x, ddof=1)}},
        {'A': ('A', np.sum), 'B': ('B', lambda x: np.std(x, ddof=1))},
        {'A': NamedAgg('A', np.sum), 'B': NamedAgg('B', lambda x: np.std(x, ddof=1))}
    ]
)
def test_agg_with_lambda(cases: Any, agg: Dict[str, Any]) -> None:
    rcustom = cases['B'].apply(lambda x: np.std(x, ddof=1))
    expected = pd.concat([cases['A'].sum(), rcustom], axis=1)
    result = cases.agg(**agg)
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    'agg',
    [
        {'func': {'result1': np.sum, 'result2': np.mean}},
        {'A': ('result1', np.sum), 'B': ('result2', np.mean)},
        {'A': NamedAgg('result1', np.sum), 'B': NamedAgg('result2', np.mean)}
    ]
)
def test_agg_no_column(cases: Any, agg: Dict[str, Any]) -> None:
    msg = "Label\\(s\\) \\['result1', 'result2'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        cases[['A', 'B']].agg(**agg)


@pytest.mark.parametrize(
    'cols, agg',
    [
        (None, {'A': ['sum', 'std'], 'B': ['mean', 'std']}),
        (['A', 'B'], {'A': ['sum', 'std'], 'B': ['mean', 'std']})
    ]
)
def test_agg_specificationerror_nested(
    cases: Any,
    cols: Optional[List[str]],
    agg: Dict[str, Any],
    a_sum: pd.Series,
    a_std: pd.Series,
    b_mean: pd.Series,
    b_std: pd.Series
) -> None:
    expected = pd.concat([a_sum, a_std, b_mean, b_std], axis=1)
    expected.columns = pd.MultiIndex.from_tuples([
        ('A', 'sum'), ('A', 'std'), ('B', 'mean'), ('B', 'std')
    ])
    if cols is not None:
        obj = cases[cols]
    else:
        obj = cases
    result = obj.agg(agg)
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    'agg',
    [
        {'A': ['sum', 'std']},
        {'A': ['sum', 'std'], 'B': ['mean', 'std']}
    ]
)
def test_agg_specificationerror_series(cases: Any, agg: Dict[str, Any]) -> None:
    msg = 'nested renamer is not supported'
    with pytest.raises(pd.errors.SpecificationError, match=msg):
        cases['A'].agg(agg)


def test_agg_specificationerror_invalid_names(cases: Any) -> None:
    msg = "Label\\(s\\) \\['B'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        cases[['A']].agg({'A': ['sum', 'std'], 'B': ['mean', 'std']})


def test_agg_nested_dicts() -> None:
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='D')
    index.name = 'date'
    df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=list('AB'), index=index)
    df_col = df.reset_index()
    df_mult = df_col.copy()
    df_mult.index = pd.MultiIndex.from_arrays([range(10), df.index], names=['index', 'date'])
    r = df.resample('2D')
    cases_list = [
        r,
        df_col.resample('2D', on='date'),
        df_mult.resample('2D', level='date'),
        df.groupby(pd.Grouper(freq='2D'))
    ]
    msg = 'nested renamer is not supported'
    for t in cases_list:
        with pytest.raises(pd.errors.SpecificationError, match=msg):
            t.aggregate({'r1': {'A': ['mean', 'sum']}, 'r2': {'B': ['mean', 'sum']}})
    for t in cases_list:
        with pytest.raises(pd.errors.SpecificationError, match=msg):
            t[['A', 'B']].agg({'A': {'ra': ['mean', 'std']}, 'B': {'rb': ['mean', 'std']}})
        with pytest.raises(pd.errors.SpecificationError, match=msg):
            t.agg({'A': {'ra': ['mean', 'std']}, 'B': {'rb': ['mean', 'std']}})


def test_try_aggregate_non_existing_column() -> None:
    data = [
        {'dt': datetime(2017, 6, 1, 0), 'x': 1.0, 'y': 2.0},
        {'dt': datetime(2017, 6, 1, 1), 'x': 2.0, 'y': 2.0},
        {'dt': datetime(2017, 6, 1, 2), 'x': 3.0, 'y': 1.5}
    ]
    df = DataFrame(data).set_index('dt')
    msg = "Label\\(s\\) \\['z'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        df.resample('30min').agg({'x': ['mean'], 'y': ['median'], 'z': ['sum']})


def test_agg_list_like_func_with_args() -> None:
    df = DataFrame({'x': [1, 2, 3]}, index=date_range('2020-01-01', periods=3, freq='D'))

    def foo1(x: pd.Series, a: int = 1, c: int = 0) -> pd.Series:
        return x + a + c

    def foo2(x: pd.Series, b: int = 2, c: int = 0) -> pd.Series:
        return x + b + c

    msg = "foo1\\(\\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        df.resample('D').agg([foo1, foo2], 3, b=3, c=4)
    result = df.resample('D').agg([foo1, foo2], 3, c=4)
    expected = DataFrame(
        [[8, 8], [9, 9], [10, 10]],
        index=date_range('2020-01-01', periods=3, freq='D'),
        columns=pd.MultiIndex.from_tuples([('x', 'foo1'), ('x', 'foo2')])
    )
    tm.assert_frame_equal(result, expected)


def test_selection_api_validation() -> None:
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='D')
    rng = np.arange(len(index), dtype=np.int64)
    df = DataFrame({'date': index, 'a': rng}, index=pd.MultiIndex.from_arrays([rng, index], names=['v', 'd']))
    df_exp = DataFrame({'a': rng}, index=index)
    msg = "Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an instance of 'Index'"
    with pytest.raises(TypeError, match=msg):
        df.resample('2D', level='v')
    msg = 'The Grouper cannot specify both a key and a level!'
    with pytest.raises(ValueError, match=msg):
        df.resample('2D', on='date', level='d')
    msg = "unhashable type: 'list'"
    with pytest.raises(TypeError, match=msg):
        df.resample('2D', on=['a', 'date'])
    msg = '\\"Level \\[\'a\', \'date\'\\] not found\\"'
    with pytest.raises(KeyError, match=msg):
        df.resample('2D', level=['a', 'date'])
    msg = 'Upsampling from level= or on= selection is not supported, use \\.set_index\\(\\.\\.\\.\\) to explicitly set index to datetime-like'
    with pytest.raises(ValueError, match=msg):
        df.resample('2D', level='d').asfreq()
    with pytest.raises(ValueError, match=msg):
        df.resample('2D', on='date').asfreq()
    exp = df_exp.resample('2D').sum()
    exp.index.name = 'date'
    result = df.resample('2D', on='date').sum()
    tm.assert_frame_equal(exp, result)
    exp.index.name = 'd'
    with pytest.raises(TypeError, match="datetime64 type does not support operation 'sum'"):
        df.resample('2D', level='d').sum()
    result = df.resample('2D', level='d').sum(numeric_only=True)
    tm.assert_frame_equal(exp, result)


@pytest.mark.parametrize('col_name', ['t2', 't2x', 't2q', 'T_2M', 't2p', 't2m', 't2m1', 'T2M'])
def test_agg_with_datetime_index_list_agg_func(col_name: str) -> None:
    df = DataFrame(
        list(range(200)),
        index=date_range(start='2017-01-01', freq='15min', periods=200, tz='Europe/Berlin'),
        columns=[col_name]
    )
    result = df.resample('1D').aggregate(['mean'])
    expected = DataFrame(
        [47.5, 143.5, 195.5],
        index=date_range(start='2017-01-01', freq='D', periods=3, tz='Europe/Berlin'),
        columns=pd.MultiIndex(levels=[[col_name], ['mean']], codes=[[0], [0]])
    )
    tm.assert_frame_equal(result, expected)


def test_resample_agg_readonly() -> None:
    index = date_range('2020-01-01', '2020-01-02', freq='1h')
    arr = np.zeros_like(index)
    arr.setflags(write=False)
    ser = Series(arr, index=index)
    rs = ser.resample('1D')
    expected = Series([pd.Timestamp(0), pd.Timestamp(0)], index=index[::24])
    result = rs.agg('last')
    tm.assert_series_equal(result, expected)
    result = rs.agg('first')
    tm.assert_series_equal(result, expected)
    result = rs.agg('max')
    tm.assert_series_equal(result, expected)
    result = rs.agg('min')
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'start,end,freq,data,resample_freq,origin,closed,exp_data,exp_end,exp_periods',
    [
        ('2000-10-01 23:30:00', '2000-10-02 00:26:00', '7min', [0, 3, 6, 9, 12, 15, 18, 21, 24], '17min', 'end', None, [0, 18, 27, 63], '20001002 00:26:00', 4),
        ('20200101 8:26:35', '20200101 9:31:58', '77s', [1] * 51, '7min', 'end', 'right', [1, 6, 5, 6, 5, 6, 5, 6, 5, 6], '2020-01-01 09:30:45', 10),
        ('2000-10-01 23:30:00', '2000-10-02 00:26:00', '7min', [0, 3, 6, 9, 12, 15, 18, 21, 24], '17min', 'end', 'left', [0, 18, 27, 39, 24], '20001002 00:43:00', 5),
        ('2000-10-01 23:30:00', '2000-10-02 00:26:00', '7min', [0, 3, 6, 9, 12, 15, 18, 21, 24], '17min', 'end_day', None, [3, 15, 45, 45], '2000-10-02 00:29:00', 4)
    ]
)
def test_end_and_end_day_origin(
    start: str,
    end: str,
    freq: str,
    data: List[int],
    resample_freq: str,
    origin: str,
    closed: Optional[str],
    exp_data: List[int],
    exp_end: str,
    exp_periods: int
) -> None:
    rng = date_range(start, end, freq=freq)
    ts = Series(data, index=rng)
    res = ts.resample(resample_freq, origin=origin, closed=closed).sum()
    expected = Series(exp_data, index=date_range(end=exp_end, freq=resample_freq, periods=exp_periods))
    tm.assert_series_equal(res, expected)


@pytest.mark.parametrize(
    'method, numeric_only, expected_data',
    [
        ('sum', True, {'num': [25]}),
        ('sum', False, {'cat': ['cat_1cat_2'], 'num': [25]}),
        ('sum', lib.no_default, {'cat': ['cat_1cat_2'], 'num': [25]}),
        ('prod', True, {'num': [100]}),
        ('prod', False, "can't multiply sequence"),
        ('prod', lib.no_default, "can't multiply sequence"),
        ('min', True, {'num': [5]}),
        ('min', False, {'cat': ['cat_1'], 'num': [5]}),
        ('min', lib.no_default, {'cat': ['cat_1'], 'num': [5]}),
        ('max', True, {'num': [20]}),
        ('max', False, {'cat': ['cat_2'], 'num': [20]}),
        ('max', lib.no_default, {'cat': ['cat_2'], 'num': [20]}),
        ('first', True, {'num': [5]}),
        ('first', False, {'cat': ['cat_1'], 'num': [5]}),
        ('first', lib.no_default, {'cat': ['cat_1'], 'num': [5]}),
        ('last', True, {'num': [20]}),
        ('last', False, {'cat': ['cat_2'], 'num': [20]}),
        ('last', lib.no_default, {'cat': ['cat_2'], 'num': [20]}),
        ('mean', True, {'num': [12.5]}),
        ('mean', False, 'Could not convert'),
        ('mean', lib.no_default, 'Could not convert'),
        ('median', True, {'num': [12.5]}),
        ('median', False, "Cannot convert \\['cat_1' 'cat_2'\\] to numeric"),
        ('median', lib.no_default, "Cannot convert \\['cat_1' 'cat_2'\\] to numeric"),
        ('std', True, {'num': [10.606601717798213]}),
        ('std', False, 'could not convert string to float'),
        ('std', lib.no_default, 'could not convert string to float'),
        ('var', True, {'num': [112.5]}),
        ('var', False, 'could not convert string to float'),
        ('var', lib.no_default, 'could not convert string to float'),
        ('sem', True, {'num': [7.5]}),
        ('sem', False, 'could not convert string to float'),
        ('sem', lib.no_default, 'could not convert string to float')
    ]
)
def test_frame_downsample_method(
    method: str,
    numeric_only: Union[bool, Any],
    expected_data: Any,
    using_infer_string: bool
) -> None:
    index = date_range('2018-01-01', periods=2, freq='D')
    expected_index = date_range('2018-12-31', periods=1, freq='YE')
    df = DataFrame({'cat': ['cat_1', 'cat_2'], 'num': [5, 20]}, index=index)
    resampled = df.resample('YE')
    kwargs = {} if numeric_only is lib.no_default else {'numeric_only': numeric_only}
    func = getattr(resampled, method)
    if isinstance(expected_data, str):
        if method in ('var', 'mean', 'median', 'prod'):
            klass = TypeError
            msg = re.escape(f'agg function failed [how->{method},dtype->')
            if using_infer_string:
                msg = f"dtype 'str' does not support operation '{method}'"
        elif method in ['sum', 'std', 'sem'] and using_infer_string:
            klass = TypeError
            msg = f"dtype 'str' does not support operation '{method}'"
        else:
            klass = ValueError
            msg = expected_data
        with pytest.raises(klass, match=msg):
            _ = func(**kwargs)
    else:
        result = func(**kwargs)
        expected = DataFrame(expected_data, index=expected_index)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'method, numeric_only, expected_data',
    [
        ('sum', True, ()),
        ('sum', False, ['cat_1cat_2']),
        ('sum', lib.no_default, ['cat_1cat_2']),
        ('prod', True, ()),
        ('prod', False, ()),
        ('prod', lib.no_default, ()),
        ('min', True, ()),
        ('min', False, ['cat_1']),
        ('min', lib.no_default, ['cat_1']),
        ('max', True, ()),
        ('max', False, ['cat_2']),
        ('max', lib.no_default, ['cat_2']),
        ('first', True, ()),
        ('first', False, ['cat_1']),
        ('first', lib.no_default, ['cat_1']),
        ('last', True, ()),
        ('last', False, ['cat_2']),
        ('last', lib.no_default, ['cat_2'])
    ]
)
def test_series_downsample_method(
    method: str,
    numeric_only: Union[bool, Any],
    expected_data: Any,
    using_infer_string: bool
) -> None:
    index = date_range('2018-01-01', periods=2, freq='D')
    expected_index = date_range('2018-12-31', periods=1, freq='YE')
    df = Series(['cat_1', 'cat_2'], index=index)
    resampled = df.resample('YE')
    kwargs = {} if numeric_only is lib.no_default else {'numeric_only': numeric_only}
    func = getattr(resampled, method)
    if numeric_only and numeric_only is not lib.no_default:
        msg = f'Cannot use numeric_only=True with SeriesGroupBy\\.{method}'
        with pytest.raises(TypeError, match=msg):
            func(**kwargs)
    elif method == 'prod':
        msg = re.escape('agg function failed [how->prod,dtype->')
        if using_infer_string:
            msg = "dtype 'str' does not support operation 'prod'"
        with pytest.raises(TypeError, match=msg):
            func(**kwargs)
    else:
        result = func(**kwargs)
        expected = Series(expected_data, index=expected_index)
        tm.assert_series_equal(result, expected)


def test_resample_empty() -> None:
    df = DataFrame(index=pd.to_datetime(['2018-01-01 00:00:00', '2018-01-01 12:00:00', '2018-01-02 00:00:00']))
    expected = DataFrame(index=pd.to_datetime(['2018-01-01 00:00:00', '2018-01-01 08:00:00', '2018-01-01 16:00:00', '2018-01-02 00:00:00']))
    result = df.resample('8h').mean()
    tm.assert_frame_equal(result, expected)
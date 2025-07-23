import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series, Timestamp, isna
import pandas._testing as tm
from typing import Any, Dict, List, Optional, Tuple, Union, cast

def test_first_last_nth(df: DataFrame) -> None:
    grouped = df.groupby('A')
    first = grouped.first()
    expected = df.loc[[1, 0], ['B', 'C', 'D']]
    expected.index = Index(['bar', 'foo'], name='A')
    expected = expected.sort_index()
    tm.assert_frame_equal(first, expected)
    nth = grouped.nth(0)
    expected = df.loc[[0, 1]]
    tm.assert_frame_equal(nth, expected)
    last = grouped.last()
    expected = df.loc[[5, 7], ['B', 'C', 'D']]
    expected.index = Index(['bar', 'foo'], name='A')
    tm.assert_frame_equal(last, expected)
    nth = grouped.nth(-1)
    expected = df.iloc[[5, 7]]
    tm.assert_frame_equal(nth, expected)
    nth = grouped.nth(1)
    expected = df.iloc[[2, 3]]
    tm.assert_frame_equal(nth, expected)
    grouped['B'].first()
    grouped['B'].last()
    grouped['B'].nth(0)
    df = df.copy()
    df.loc[df['A'] == 'foo', 'B'] = np.nan
    grouped = df.groupby('A')
    assert isna(grouped['B'].first()['foo'])
    assert isna(grouped['B'].last()['foo'])
    assert isna(grouped['B'].nth(0).iloc[0])
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=['A', 'B'])
    g = df.groupby('A')
    result = g.first()
    expected = df.iloc[[1, 2]].set_index('A')
    tm.assert_frame_equal(result, expected)
    expected = df.iloc[[1, 2]]
    result = g.nth(0, dropna='any')
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('method', ['first', 'last'])
def test_first_last_with_na_object(method: str, nulls_fixture: Any) -> None:
    groups = DataFrame({'a': [1, 1, 2, 2], 'b': [1, 2, 3, nulls_fixture]}).groupby('a')
    result = getattr(groups, method)()
    if method == 'first':
        values = [1, 3]
    else:
        values = [2, 3]
    values = np.array(values, dtype=result['b'].dtype)
    idx = Index([1, 2], name='a')
    expected = DataFrame({'b': values}, index=idx)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('index', [0, -1])
def test_nth_with_na_object(index: int, nulls_fixture: Any) -> None:
    df = DataFrame({'a': [1, 1, 2, 2], 'b': [1, 2, 3, nulls_fixture]})
    groups = df.groupby('a')
    result = groups.nth(index)
    expected = df.iloc[[0, 2]] if index == 0 else df.iloc[[1, 3]]
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('method', ['first', 'last'])
def test_first_last_with_None(method: str) -> None:
    df = DataFrame.from_dict({'id': ['a'], 'value': [None]})
    groups = df.groupby('id', as_index=False)
    result = getattr(groups, method)()
    tm.assert_frame_equal(result, df)

@pytest.mark.parametrize('method', ['first', 'last'])
@pytest.mark.parametrize('df, expected', [(DataFrame({'id': 'a', 'value': [None, 'foo', np.nan]}), DataFrame({'value': ['foo']}, index=Index(['a'], name='id'))), (DataFrame({'id': 'a', 'value': [np.nan]}, dtype=object), DataFrame({'value': [None]}, index=Index(['a'], name='id')))])
def test_first_last_with_None_expanded(method: str, df: DataFrame, expected: DataFrame) -> None:
    result = getattr(df.groupby('id'), method)()
    tm.assert_frame_equal(result, expected)

def test_first_last_nth_dtypes() -> None:
    df = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'], 'C': np.random.default_rng(2).standard_normal(8), 'D': np.array(np.random.default_rng(2).standard_normal(8), dtype='float32')})
    df['E'] = True
    df['F'] = 1
    grouped = df.groupby('A')
    first = grouped.first()
    expected = df.loc[[1, 0], ['B', 'C', 'D', 'E', 'F']]
    expected.index = Index(['bar', 'foo'], name='A')
    expected = expected.sort_index()
    tm.assert_frame_equal(first, expected)
    last = grouped.last()
    expected = df.loc[[5, 7], ['B', 'C', 'D', 'E', 'F']]
    expected.index = Index(['bar', 'foo'], name='A')
    expected = expected.sort_index()
    tm.assert_frame_equal(last, expected)
    nth = grouped.nth(1)
    expected = df.iloc[[2, 3]]
    tm.assert_frame_equal(nth, expected)

def test_first_last_nth_dtypes2() -> None:
    idx = list(range(10))
    idx.append(9)
    ser = Series(data=range(11), index=idx, name='IntCol')
    assert ser.dtype == 'int64'
    f = ser.groupby(level=0).first()
    assert f.dtype == 'int64'

def test_first_last_nth_nan_dtype() -> None:
    df = DataFrame({'data': ['A'], 'nans': Series([None], dtype=object)})
    grouped = df.groupby('data')
    expected = df.set_index('data').nans
    tm.assert_series_equal(grouped.nans.first(), expected)
    tm.assert_series_equal(grouped.nans.last(), expected)
    expected = df.nans
    tm.assert_series_equal(grouped.nans.nth(-1), expected)
    tm.assert_series_equal(grouped.nans.nth(0), expected)

def test_first_strings_timestamps() -> None:
    test = DataFrame({Timestamp('2012-01-01 00:00:00'): ['a', 'b'], Timestamp('2012-01-02 00:00:00'): ['c', 'd'], 'name': ['e', 'e'], 'aaaa': ['f', 'g']})
    result = test.groupby('name').first()
    expected = DataFrame([['a', 'c', 'f']], columns=Index([Timestamp('2012-01-01'), Timestamp('2012-01-02'), 'aaaa']), index=Index(['e'], name='name'))
    tm.assert_frame_equal(result, expected)

def test_nth() -> None:
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=['A', 'B'])
    gb = df.groupby('A')
    tm.assert_frame_equal(gb.nth(0), df.iloc[[0, 2]])
    tm.assert_frame_equal(gb.nth(1), df.iloc[[1]])
    tm.assert_frame_equal(gb.nth(2), df.loc[[]])
    tm.assert_frame_equal(gb.nth(-1), df.iloc[[1, 2]])
    tm.assert_frame_equal(gb.nth(-2), df.iloc[[0]])
    tm.assert_frame_equal(gb.nth(-3), df.loc[[]])
    tm.assert_series_equal(gb.B.nth(0), df.B.iloc[[0, 2]])
    tm.assert_series_equal(gb.B.nth(1), df.B.iloc[[1]])
    tm.assert_frame_equal(gb[['B']].nth(0), df[['B']].iloc[[0, 2]])
    tm.assert_frame_equal(gb.nth(0, dropna='any'), df.iloc[[1, 2]])
    tm.assert_frame_equal(gb.nth(-1, dropna='any'), df.iloc[[1, 2]])
    tm.assert_frame_equal(gb.nth(7, dropna='any'), df.iloc[:0])
    tm.assert_frame_equal(gb.nth(2, dropna='any'), df.iloc[:0])

def test_nth2() -> None:
    df = DataFrame({'color': {0: 'green', 1: 'green', 2: 'red', 3: 'red', 4: 'red'}, 'food': {0: 'ham', 1: 'eggs', 2: 'eggs', 3: 'ham', 4: 'pork'}, 'two': {0: 1.5456590000000001, 1: -0.070345, 2: -2.400454, 3: 0.46206, 4: 0.523508}, 'one': {0: 0.565738, 1: -0.9742360000000001, 2: 1.033801, 3: -0.785435, 4: 0.704228}}).set_index(['color', 'food'])
    result = df.groupby(level=0, as_index=False).nth(2)
    expected = df.iloc[[-1]]
    tm.assert_frame_equal(result, expected)
    result = df.groupby(level=0, as_index=False).nth(3)
    expected = df.loc[[]]
    tm.assert_frame_equal(result, expected)

def test_nth3() -> None:
    df = DataFrame(np.random.default_rng(2).integers(1, 10, (100, 2)), dtype='int64')
    ser = df[1]
    gb = df[0]
    expected = ser.groupby(gb).first()
    expected2 = ser.groupby(gb).apply(lambda x: x.iloc[0])
    tm.assert_series_equal(expected2, expected, check_names=False)
    assert expected.name == 1
    assert expected2.name == 1
    v = ser[gb == 1].iloc[0]
    assert expected.iloc[0] == v
    assert expected2.iloc[0] == v
    with pytest.raises(ValueError, match='For a DataFrame'):
        ser.groupby(gb, sort=False).nth(0, dropna=True)

def test_nth4() -> None:
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=['A', 'B'])
    gb = df.groupby('A')
    result = gb.B.nth(0, dropna='all')
    expected = df.B.iloc[[1, 2]]
    tm.assert_series_equal(result, expected)

def test_nth5() -> None:
    df = DataFrame([[1, np.nan], [1, 3], [1, 4], [5, 6], [5, 7]], columns=['A', 'B'])
    gb = df.groupby('A')
    tm.assert_frame_equal(gb.nth(0), df.iloc[[0, 3]])
    tm.assert_frame_equal(gb.nth([0]), df.iloc[[0, 3]])
    tm.assert_frame_equal(gb.nth([0, 1]), df.iloc[[0, 1, 3, 4]])
    tm.assert_frame_equal(gb.nth([0, -1]), df.iloc[[0, 2, 3, 4]])
    tm.assert_frame_equal(gb.nth([0, 1, 2]), df.iloc[[0, 1, 2, 3, 4]])
    tm.assert_frame_equal(gb.nth([0, 1, -1]), df.iloc[[0, 1, 2, 3, 4]])
    tm.assert_frame_equal(gb.nth([2]), df.iloc[[2]])
    tm.assert_frame_equal(gb.nth([3, 4]), df.loc[[]])

def test_nth_bdays(unit: str) -> None:
    business_dates = pd.date_range(start='4/1/2014', end='6/30/2014', freq='B', unit=unit)
    df = DataFrame(1, index=business_dates, columns=['a', 'b'])
    key = [df.index.year, df.index.month]
    result = df.groupby(key, as_index=False).nth([0, 3, -2, -1])
    expected_dates = pd.to_datetime(['2014/4/1', '2014/4/4', '2014/4/29', '2014/4/30', '2014/5/1', '2014/5/6', '2014/5/29', '2014/5/30', '2014/6/2', '2014/6/5', '2014/6/27', '2014/6/30']).as_unit(unit)
    expected = DataFrame(1, columns=['a', 'b'], index=expected_dates)
    tm.assert_frame_equal(result, expected)

def test_nth_multi_grouper(three_group: DataFrame) -> None:
    grouped = three_group.groupby(['A', 'B'])
    result = grouped.nth(0)
    expected = three_group.iloc[[0, 3, 4, 7]]
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('data, expected_first, expected_last', [({'id': ['A'], 'time': Timestamp('2012-02-01 14:00:00', tz='US/Central'), 'foo': [1]}, {'id': ['A'], 'time': Timestamp('2012-02-01 14:00:00', tz='US/Central'), 'foo': [1]}, {'id': ['A'], 'time': Timestamp('2012-02-01 14:00:00', tz='US/Central'), 'foo': [1]}), ({'id': ['A', 'B', 'A'], 'time': [Timestamp('2012-01-01 13:00:00', tz='America/New_York'), Timestamp('2012-02-01 14:00:00', tz='US/Central'), Timestamp('2012-03-01 12:00:00', tz='Europe/London')], 'foo': [1, 2, 3]}, {'id': ['A', 'B'], 'time': [Timestamp('2012-01-01 13:00:00', tz='America/New_York'), Timestamp('2012-02-01 14:00:00', tz='US/Central')], 'foo': [1, 2]}, {'id': ['A', 'B'], 'time': [Timestamp('2012-03-01 12:00:00', tz='Europe/London'), Timestamp('2012-02-01 14:00:00', tz='US/Central')], 'foo': [3, 2]})])
def test_first_last_tz(data: Dict[str, List[Any]], expected_first: Dict[str, List[Any]], expected_last: Dict[str, List[Any]]) -> None:
    df = DataFrame(data)
    result = df.groupby('id', as_index=False).first()
    expected = DataFrame(expected_first)
    cols = ['id', 'time', 'foo']
    tm.assert_frame_equal(result[cols], expected[cols])
    result = df.groupby('id', as_index=False)['time'].first()
    tm.assert_frame_equal(result, expected[['id', 'time']])
    result = df.groupby('id', as_index=False).last()
    expected = DataFrame(expected_last)
    cols = ['id', 'time', 'foo']
    tm.assert_frame_equal(result[cols], expected[cols])
    result = df.groupby('id', as_index=False)['time'].last()
    tm.assert_frame_equal(result, expected[['id', 'time']])

@pytest.mark.parametrize('method, ts, alpha', [['first', Timestamp('2013-01-01', tz='US/Eastern'), 'a'], ['last', Timestamp('2013-01-02', tz='US/Eastern'), 'b']])
def test_first_last_tz_multi_column(method: str, ts: Timestamp, alpha: str, unit: str) -> None:
    category_string = Series(list('abc')).astype('category')
    dti = pd.date_range('20130101', periods=3, tz='US/Eastern', unit=unit)
    df = DataFrame({'group': [1, 1, 2], 'category_string': category_string, 'datetimetz': dti})
    result = getattr(df.groupby('group'), method)()
    expected = DataFrame({'category_string': pd.Categorical([alpha, 'c'], dtype=category_string.dtype), 'datetimetz': [ts, Timestamp('2013-01-03', tz='US/Eastern')]}, index=Index([1, 2], name='group'))
    expected['datetimetz'] = expected['datetimetz'].dt.as_unit(unit)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('values', [pd.array([True, False], dtype='boolean'), pd.array([1, 2], dtype='Int64'), pd.to_datetime(['2020-01-01', '2020-02-01']), pd.to_timedelta([1, 2], unit='D')
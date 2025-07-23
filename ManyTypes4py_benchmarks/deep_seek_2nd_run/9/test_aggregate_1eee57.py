"""
test .agg behavior / note that .apply is tested generally in test_groupby.py
"""
import datetime
import functools
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series, concat, to_datetime
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping


def test_groupby_agg_no_extra_calls() -> None:
    df = DataFrame({'key': ['a', 'b', 'c', 'c'], 'value': [1, 2, 3, 4]})
    gb = df.groupby('key')['value']

    def dummy_func(x: Series) -> float:
        assert len(x) != 0
        return x.sum()
    gb.agg(dummy_func)


def test_agg_regression1(tsframe: DataFrame) -> None:
    grouped = tsframe.groupby([lambda x: x.year, lambda x: x.month])
    result = grouped.agg('mean')
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)


def test_agg_must_agg(df: DataFrame) -> None:
    grouped = df.groupby('A')['C']
    msg = 'Must produce aggregated value'
    with pytest.raises(Exception, match=msg):
        grouped.agg(lambda x: x.describe())
    with pytest.raises(Exception, match=msg):
        grouped.agg(lambda x: x.index[:2])


def test_agg_ser_multi_key(df: DataFrame) -> None:
    f = lambda x: x.sum()
    results = df.C.groupby([df.A, df.B]).aggregate(f)
    expected = df.groupby(['A', 'B']).sum()['C']
    tm.assert_series_equal(results, expected)


def test_agg_with_missing_values() -> None:
    missing_df = DataFrame({'nan': [np.nan, np.nan, np.nan, np.nan], 'na': [pd.NA, pd.NA, pd.NA, pd.NA], 'nat': [pd.NaT, pd.NaT, pd.NaT, pd.NaT], 'none': [None, None, None, None], 'values': [1, 2, 3, 4]})
    result = missing_df.agg(x=('nan', 'min'), y=('na', 'min'), z=('values', 'sum'))
    expected = DataFrame({'nan': [np.nan, np.nan, np.nan], 'na': [np.nan, np.nan, np.nan], 'values': [np.nan, np.nan, 10.0]}, index=['x', 'y', 'z'])
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_mixed_dtype() -> None:
    expected = DataFrame({'v1': [5, 5, 7, np.nan, 3, 3, 4, 1], 'v2': [55, 55, 77, np.nan, 33, 33, 44, 11]}, index=MultiIndex.from_tuples([(1, 95), (1, 99), (2, 95), (2, 99), ('big', 'damp'), ('blue', 'dry'), ('red', 'red'), ('red', 'wet')], names=['by1', 'by2']))
    df = DataFrame({'v1': [1, 3, 5, 7, 8, 3, 5, np.nan, 4, 5, 7, 9], 'v2': [11, 33, 55, 77, 88, 33, 55, np.nan, 44, 55, 77, 99], 'by1': ['red', 'blue', 1, 2, np.nan, 'big', 1, 2, 'red', 1, np.nan, 12], 'by2': ['wet', 'dry', 99, 95, np.nan, 'damp', 95, 99, 'red', 99, np.nan, np.nan]})
    g = df.groupby(['by1', 'by2'])
    result = g[['v1', 'v2']].mean()
    tm.assert_frame_equal(result, expected)


def test_agg_apply_corner(ts: Series, tsframe: DataFrame) -> None:
    grouped = ts.groupby(ts * np.nan, group_keys=False)
    assert ts.dtype == np.float64
    exp = Series([], dtype=np.float64, index=Index([], dtype=np.float64))
    tm.assert_series_equal(grouped.sum(), exp)
    tm.assert_series_equal(grouped.agg('sum'), exp)
    tm.assert_series_equal(grouped.apply('sum'), exp, check_index_type=False)
    grouped = tsframe.groupby(tsframe['A'] * np.nan, group_keys=False)
    exp_df = DataFrame(columns=tsframe.columns, dtype=float, index=Index([], name='A', dtype=np.float64))
    tm.assert_frame_equal(grouped.sum(), exp_df)
    tm.assert_frame_equal(grouped.agg('sum'), exp_df)
    res = grouped.apply(np.sum, axis=0)
    exp_df = exp_df.reset_index(drop=True)
    tm.assert_frame_equal(res, exp_df)


def test_with_na_groups(any_real_numpy_dtype: str) -> None:
    index = Index(np.arange(10))
    values = Series(np.ones(10), index, dtype=any_real_numpy_dtype)
    labels = Series([np.nan, 'foo', 'bar', 'bar', np.nan, np.nan, 'bar', 'bar', np.nan, 'foo'], index=index)
    grouped = values.groupby(labels)
    agged = grouped.agg(len)
    expected = Series([4, 2], index=['bar', 'foo'])
    tm.assert_series_equal(agged, expected, check_dtype=False)

    def f(x: Series) -> float:
        return float(len(x))
    agged = grouped.agg(f)
    expected = Series([4.0, 2.0], index=['bar', 'foo'])
    tm.assert_series_equal(agged, expected)


def test_agg_grouping_is_list_tuple(ts: Series) -> None:
    df = DataFrame(np.random.default_rng(2).standard_normal((30, 4)), columns=Index(list('ABCD'), dtype=object), index=pd.date_range('2000-01-01', periods=30, freq='B'))
    grouped = df.groupby(lambda x: x.year)
    grouper = grouped._grouper.groupings[0].grouping_vector
    grouped._grouper.groupings[0] = Grouping(ts.index, list(grouper))
    result = grouped.agg('mean')
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)
    grouped._grouper.groupings[0] = Grouping(ts.index, tuple(grouper))
    result = grouped.agg('mean')
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)


def test_agg_python_multiindex(multiindex_dataframe_random_data: DataFrame) -> None:
    grouped = multiindex_dataframe_random_data.groupby(['A', 'B'])
    result = grouped.agg('mean')
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('groupbyfunc', [lambda x: x.weekday(), [lambda x: x.month, lambda x: x.weekday()]])
def test_aggregate_str_func(tsframe: DataFrame, groupbyfunc: Union[Callable, List[Callable]]) -> None:
    grouped = tsframe.groupby(groupbyfunc)
    result = grouped['A'].agg('std')
    expected = grouped['A'].std()
    tm.assert_series_equal(result, expected)
    result = grouped.aggregate('var')
    expected = grouped.var()
    tm.assert_frame_equal(result, expected)
    result = grouped.agg({'A': 'var', 'B': 'std', 'C': 'mean', 'D': 'sem'})
    expected = DataFrame({'A': grouped['A'].var(), 'B': grouped['B'].std(), 'C': grouped['C'].mean(), 'D': grouped['D'].sem()})
    tm.assert_frame_equal(result, expected)


def test_std_masked_dtype(any_numeric_ea_dtype: str) -> None:
    df = DataFrame({'a': [2, 1, 1, 1, 2, 2, 1], 'b': Series([pd.NA, 1, 2, 1, 1, 1, 2], dtype='Float64')})
    result = df.groupby('a').std()
    expected = DataFrame({'b': [0.57735, 0]}, index=Index([1, 2], name='a'), dtype='Float64')
    tm.assert_frame_equal(result, expected)


def test_agg_str_with_kwarg_axis_1_raises(df: DataFrame, reduction_func: str) -> None:
    gb = df.groupby(level=0)
    msg = f'Operation {reduction_func} does not support axis=1'
    with pytest.raises(ValueError, match=msg):
        gb.agg(reduction_func, axis=1)


def test_aggregate_item_by_item(df: DataFrame) -> None:
    grouped = df.groupby('A')
    aggfun_0 = lambda ser: ser.size
    result = grouped.agg(aggfun_0)
    foosum = (df.A == 'foo').sum()
    barsum = (df.A == 'bar').sum()
    K = len(result.columns)
    exp = Series(np.array([foosum] * K), index=list('BCD'), name='foo')
    tm.assert_series_equal(result.xs('foo'), exp)
    exp = Series(np.array([barsum] * K), index=list('BCD'), name='bar')
    tm.assert_almost_equal(result.xs('bar'), exp)

    def aggfun_1(ser: Series) -> int:
        return ser.size
    result = DataFrame().groupby(df.A).agg(aggfun_1)
    assert isinstance(result, DataFrame)
    assert len(result) == 0


def test_wrap_agg_out(three_group: DataFrame) -> None:
    grouped = three_group.groupby(['A', 'B'])

    def func(ser: Series) -> Union[int, float]:
        if ser.dtype == object or ser.dtype == 'string':
            raise TypeError('Test error message')
        return ser.sum()
    with pytest.raises(TypeError, match='Test error message'):
        grouped.aggregate(func)
    result = grouped[['D', 'E', 'F']].aggregate(func)
    exp_grouped = three_group.loc[:, ['A', 'B', 'D', 'E', 'F']]
    expected = exp_grouped.groupby(['A', 'B']).aggregate(func)
    tm.assert_frame_equal(result, expected)


def test_agg_multiple_functions_maintain_order(df: DataFrame) -> None:
    funcs = [('mean', np.mean), ('max', np.max), ('min', np.min)]
    result = df.groupby('A')['C'].agg(funcs)
    exp_cols = Index(['mean', 'max', 'min'])
    tm.assert_index_equal(result.columns, exp_cols)


def test_series_index_name(df: DataFrame) -> None:
    grouped = df.loc[:, ['C']].groupby(df['A'])
    result = grouped.agg(lambda x: x.mean())
    assert result.index.name == 'A'


def test_agg_multiple_functions_same_name() -> None:
    df = DataFrame(np.random.default_rng(2).standard_normal((1000, 3)), index=pd.date_range('1/1/2012', freq='s', periods=1000), columns=['A', 'B', 'C'])
    result = df.resample('3min').agg({'A': [partial(np.quantile, q=0.9999), partial(np.quantile, q=0.1111)]})
    expected_index = pd.date_range('1/1/2012', freq='3min', periods=6)
    expected_columns = MultiIndex.from_tuples([('A', 'quantile'), ('A', 'quantile')])
    expected_values = np.array([df.resample('3min').A.quantile(q=q).values for q in [0.9999, 0.1111]]).T
    expected = DataFrame(expected_values, columns=expected_columns, index=expected_index)
    tm.assert_frame_equal(result, expected)


def test_agg_multiple_functions_same_name_with_ohlc_present() -> None:
    df = DataFrame(np.random.default_rng(2).standard_normal((1000, 3)), index=pd.date_range('1/1/2012', freq='s', periods=1000, name='dti'), columns=Index(['A', 'B', 'C'], name='alpha'))
    result = df.resample('3min').agg({'A': ['ohlc', partial(np.quantile, q=0.9999), partial(np.quantile, q=0.1111)]})
    expected_index = pd.date_range('1/1/2012', freq='3min', periods=6, name='dti')
    expected_columns = MultiIndex.from_tuples([('A', 'ohlc', 'open'), ('A', 'ohlc', 'high'), ('A', 'ohlc', 'low'), ('A', 'ohlc', 'close'), ('A', 'quantile', 'A'), ('A', 'quantile', 'A')], names=['alpha', None, None])
    non_ohlc_expected_values = np.array([df.resample('3min').A.quantile(q=q).values for q in [0.9999, 0.1111]]).T
    expected_values = np.hstack([df.resample('3min').A.ohlc(), non_ohlc_expected_values])
    expected = DataFrame(expected_values, columns=expected_columns, index=expected_index)
    tm.assert_frame_equal(result, expected)


def test_multiple_functions_tuples_and_non_tuples(df: DataFrame) -> None:
    df = df.drop(columns=['B', 'C'])
    funcs = [('foo', 'mean'), 'std']
    ex_funcs = [('foo', 'mean'), ('std', 'std')]
    result = df.groupby('A')['D'].agg(funcs)
    expected = df.groupby('A')['D'].agg(ex_funcs)
    tm.assert_frame_equal(result, expected)
    result = df.groupby('A').agg(funcs)
    expected = df.groupby('A').agg(ex_funcs)
    tm.assert_frame_equal(result, expected)


def test_more_flexible_frame_multi_function(df: DataFrame) -> None:
    grouped = df.groupby('A')
    exmean = grouped.agg({'C': 'mean', 'D': 'mean'})
    exstd = grouped.agg({'C': 'std', 'D': 'std'})
    expected = concat([exmean, exstd], keys=['mean', 'std'], axis=1)
    expected = expected.swaplevel(0, 1, axis=1).sort_index(level=0, axis=1)
    d = {'C': ['mean', 'std'], 'D': ['mean', 'std']}
    result = grouped.aggregate(d)
    tm.assert_frame_equal(result, expected)
    result = grouped.aggregate({'C': 'mean', 'D': ['mean', 'std']})
    expected = grouped.aggregate({'C': 'mean', 'D': ['mean', 'std']})
    tm.assert_frame_equal(result, expected)

    def numpymean(x: Series) -> float:
        return np.mean(x)

    def numpystd(x: Series) -> float:
        return np.std(x, ddof=1)
    msg = 'nested renamer is not supported'
    d = {'C': 'mean', 'D': {'foo': 'mean', 'bar': 'std'}}
    with pytest.raises(SpecificationError, match=msg):
        grouped.aggregate(d)
    d = {'C': ['mean'], 'D': [numpymean, numpystd]}
    grouped.aggregate(d)


def test_multi_function_flexible_mix(df: DataFrame) -> None:
    grouped = df.groupby('A')
    d = {'C': {'foo': 'mean', 'bar': 'std'}, 'D': {'sum': 'sum'}}
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        grouped.aggregate(d)
    d = {'C': {'foo': 'mean', 'bar': 'std'}, 'D': 'sum'}
    with pytest.raises(SpecificationError, match=msg):
        grouped.aggregate(d)
    d = {'C': {'foo': 'mean', 'bar': 'std'}, 'D': 'sum'}
    with pytest.raises(SpecificationError, match=msg):
        grouped.aggregate(d)


def test_groupby_agg_coercing_bools() -> None:
    dat = DataFrame({'a': [1, 1, 2, 2], 'b': [0, 1, 2, 3], 'c': [None, None, 1, 1]})
    gp = dat.groupby('a')
    index = Index([1, 2], name='a')
    result = gp['b'].aggregate(lambda x: (x != 0).all())
    expected = Series([False, True], index=index, name='b')
    tm.assert_series_equal(result, expected)
    result = gp['c'].aggregate(lambda x: x.isnull().all())
    expected = Series([True, False], index=index, name='c')
    tm.assert_series_equal(result, expected)


def test_agg_dict_with_getitem() -> None:
    dat = DataFrame({'A': ['A', 'A', 'B
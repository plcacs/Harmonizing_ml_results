from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import Categorical, DataFrame, Grouper, Index, Interval, MultiIndex, RangeIndex, Series, Timedelta, Timestamp, date_range, to_datetime
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com

def test_repr() -> None:
    result = repr(Grouper(key='A', level='B'))
    expected = "Grouper(key='A', level='B', sort=False, dropna=True)"
    assert result == expected

def test_groupby_nonobject_dtype(multiindex_dataframe_random_data: DataFrame) -> None:
    key = multiindex_dataframe_random_data.index.codes[0]
    grouped = multiindex_dataframe_random_data.groupby(key)
    result = grouped.sum()
    expected = multiindex_dataframe_random_data.groupby(key.astype('O')).sum()
    assert result.index.dtype == np.int8
    assert expected.index.dtype == np.int64
    tm.assert_frame_equal(result, expected, check_index_type=False)

def test_groupby_nonobject_dtype_mixed() -> None:
    df: DataFrame = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'], 'C': np.random.default_rng(2).standard_normal(8), 'D': np.array(np.random.default_rng(2).standard_normal(8), dtype='float32')})
    df['value'] = range(len(df))

    def max_value(group: DataFrame) -> DataFrame:
        return group.loc[group['value'].idxmax()]

    applied = df.groupby('A').apply(max_value)
    result = applied.dtypes
    expected = df.drop(columns='A').dtypes
    tm.assert_series_equal(result, expected)

def test_pass_args_kwargs(ts: DataFrame) -> None:
    def f(x: np.ndarray, q: float = None, axis: int = 0) -> np.ndarray:
        return np.percentile(x, q, axis=axis)

    g = lambda x: np.percentile(x, 80, axis=0)
    ts_grouped = ts.groupby(lambda x: x.month)
    agg_result = ts_grouped.agg(np.percentile, 80, axis=0)
    apply_result = ts_grouped.apply(np.percentile, 80, axis=0)
    trans_result = ts_grouped.transform(np.percentile, 80, axis=0)
    agg_expected = ts_grouped.quantile(0.8)
    trans_expected = ts_grouped.transform(g)
    tm.assert_series_equal(apply_result, agg_expected)
    tm.assert_series_equal(agg_result, agg_expected)
    tm.assert_series_equal(trans_result, trans_expected)
    agg_result = ts_grouped.agg(f, q=80)
    apply_result = ts_grouped.apply(f, q=80)
    trans_result = ts_grouped.transform(f, q=80)
    tm.assert_series_equal(agg_result, agg_expected)
    tm.assert_series_equal(apply_result, agg_expected)
    tm.assert_series_equal(trans_result, trans_expected)

def test_pass_args_kwargs_dataframe(tsframe: DataFrame, as_index: bool) -> None:
    def f(x: np.ndarray, q: float = None, axis: int = 0) -> np.ndarray:
        return np.percentile(x, q, axis=axis)

    df_grouped = tsframe.groupby(lambda x: x.month, as_index=as_index)
    agg_result = df_grouped.agg(np.percentile, 80, axis=0)
    apply_result = df_grouped.apply(DataFrame.quantile, 0.8)
    expected = df_grouped.quantile(0.8)
    tm.assert_frame_equal(apply_result, expected, check_names=False)
    tm.assert_frame_equal(agg_result, expected)
    apply_result = df_grouped.apply(DataFrame.quantile, [0.4, 0.8])
    expected_seq = df_grouped.quantile([0.4, 0.8])
    if not as_index:
        apply_result.index = range(4)
        apply_result.insert(loc=0, column='level_0', value=[1, 1, 2, 2])
        apply_result.insert(loc=1, column='level_1', value=[0.4, 0.8, 0.4, 0.8])
    tm.assert_frame_equal(apply_result, expected_seq, check_names=False)
    agg_result = df_grouped.agg(f, q=80)
    apply_result = df_grouped.apply(DataFrame.quantile, q=0.8)
    tm.assert_frame_equal(agg_result, expected)
    tm.assert_frame_equal(apply_result, expected, check_names=False)

def test_len() -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    grouped = df.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day])
    assert len(grouped) == len(df)
    grouped = df.groupby([lambda x: x.year, lambda x: x.month])
    expected = len({(x.year, x.month) for x in df.index})
    assert len(grouped) == expected

def test_len_nan_group() -> None:
    df: DataFrame = DataFrame({'a': [np.nan] * 3, 'b': [1, 2, 3]})
    assert len(df.groupby('a')) == 0
    assert len(df.groupby('b')) == 3
    assert len(df.groupby(['a', 'b'])) == 0

def test_groupby_timedelta_median() -> None:
    expected = Series(data=Timedelta('1D'), index=['foo'])
    df: DataFrame = DataFrame({'label': ['foo', 'foo'], 'timedelta': [pd.NaT, Timedelta('1D')]})
    gb = df.groupby('label')['timedelta']
    actual = gb.median()
    tm.assert_series_equal(actual, expected, check_names=False)

# ... (rest of the code remains the same)

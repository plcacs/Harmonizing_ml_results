from datetime import date, datetime, timedelta
import re
from typing import Tuple, List, Any, Optional, Union, overload

import numpy as np
import pytest
from pandas.core.dtypes.common import is_object_dtype, is_string_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
)
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import MergeError, merge

def get_test_data(ngroups: int = 8, n: int = 50) -> np.ndarray:
    unique_groups: List[int] = list(range(ngroups))
    arr: np.ndarray = np.asarray(np.tile(unique_groups, n // ngroups))
    if len(arr) < n:
        arr = np.asarray(list(arr) + unique_groups[: n - len(arr)])
    np.random.default_rng(2).shuffle(arr)
    return arr

@pytest.fixture
def dfs_for_indicator() -> Tuple[DataFrame, DataFrame]:
    df1: DataFrame = DataFrame({'col1': [0, 1], 'col_conflict': [1, 2], 'col_left': ['a', 'b']})
    df2: DataFrame = DataFrame({'col1': [1, 2, 3, 4, 5], 'col_conflict': [1, 2, 3, 4, 5], 'col_right': [2, 2, 2, 2, 2]})
    return (df1, df2)

class TestMerge:
    @pytest.fixture
    def df(self) -> DataFrame:
        df_: DataFrame = DataFrame({
            'key1': get_test_data(),
            'key2': get_test_data(),
            'data1': np.random.default_rng(2).standard_normal(50),
            'data2': np.random.default_rng(2).standard_normal(50)
        })
        df_ = df_[df_['key2'] > 1]
        return df_

    @pytest.fixture
    def df2(self) -> DataFrame:
        return DataFrame({
            'key1': get_test_data(n=10),
            'key2': get_test_data(ngroups=4, n=10),
            'value': np.random.default_rng(2).standard_normal(10)
        })

    @pytest.fixture
    def left(self) -> DataFrame:
        return DataFrame({
            'key': ['a', 'b', 'c', 'd', 'e', 'e', 'a'],
            'v1': np.random.default_rng(2).standard_normal(7)
        })

    def test_merge_inner_join_empty(self) -> None:
        df_empty: DataFrame = DataFrame()
        df_a: DataFrame = DataFrame({'a': [1, 2]}, index=[0, 1], dtype='int64')
        result: DataFrame = merge(df_empty, df_a, left_index=True, right_index=True)
        expected: DataFrame = DataFrame({'a': []}, dtype='int64')
        tm.assert_frame_equal(result, expected)

    def test_merge_common(self, df: DataFrame, df2: DataFrame) -> None:
        joined: DataFrame = merge(df, df2)
        exp: DataFrame = merge(df, df2, on=['key1', 'key2'])
        tm.assert_frame_equal(joined, exp)

    def test_merge_non_string_columns(self) -> None:
        left_: DataFrame = DataFrame({0: [1, 0, 1, 0], 1: [0, 1, 0, 0], 2: [0, 0, 2, 0], 3: [1, 0, 0, 3]})
        right_: DataFrame = left_.astype(float)
        expected: DataFrame = left_
        result: DataFrame = merge(left_, right_)
        tm.assert_frame_equal(expected, result)

    def test_merge_index_as_on_arg(self, df: DataFrame, df2: DataFrame) -> None:
        left_: DataFrame = df.set_index('key1')
        right_: DataFrame = df2.set_index('key1')
        result: DataFrame = merge(left_, right_, on='key1')
        expected: DataFrame = merge(df, df2, on='key1').set_index('key1')
        tm.assert_frame_equal(result, expected)

    def test_merge_index_singlekey_right_vs_left(self) -> None:
        left_: DataFrame = DataFrame({'key': ['a', 'b', 'c', 'd', 'e', 'e', 'a'], 'v1': np.random.default_rng(2).standard_normal(7)})
        right_: DataFrame = DataFrame({'v2': np.random.default_rng(2).standard_normal(4)}, index=['d', 'b', 'c', 'a'])
        merged1: DataFrame = merge(left_, right_, left_on='key', right_index=True, how='left', sort=False)
        merged2: DataFrame = merge(right_, left_, right_on='key', left_index=True, how='right', sort=False)
        tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])
        merged1 = merge(left_, right_, left_on='key', right_index=True, how='left', sort=True)
        merged2 = merge(right_, left_, right_on='key', left_index=True, how='right', sort=True)
        tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])

    def test_merge_index_singlekey_inner(self) -> None:
        left_: DataFrame = DataFrame({'key': ['a', 'b', 'c', 'd', 'e', 'e', 'a'], 'v1': np.random.default_rng(2).standard_normal(7)})
        right_: DataFrame = DataFrame({'v2': np.random.default_rng(2).standard_normal(4)}, index=['d', 'b', 'c', 'a'])
        result: DataFrame = merge(left_, right_, left_on='key', right_index=True, how='inner')
        expected: DataFrame = left_.join(right_, on='key').loc[result.index]
        tm.assert_frame_equal(result, expected)
        result = merge(right_, left_, right_on='key', left_index=True, how='inner')
        expected = left_.join(right_, on='key').loc[result.index]
        tm.assert_frame_equal(result, expected.loc[:, result.columns])

    def test_merge_misspecified(self, df: DataFrame, df2: DataFrame, left: DataFrame) -> None:
        right_: DataFrame = DataFrame({'v2': np.random.default_rng(2).standard_normal(4)}, index=['d', 'b', 'c', 'a'])
        msg: str = 'Must pass right_on or right_index=True'
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, right_, left_index=True)
        msg = 'Must pass left_on or left_index=True'
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, right_, right_index=True)
        msg = 'Can only pass argument "on" OR "left_on" and "right_on", not a combination of both'
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, left, left_on='key', on='key')
        msg = 'len\\(right_on\\) must equal len\\(left_on\\)'
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, left_on=['key1'], right_on=['key1', 'key2'])

    def test_index_and_on_parameters_confusion(self, df: DataFrame, df2: DataFrame) -> None:
        msg: str = "right_index parameter must be of type bool, not <class 'list'>"
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, how='left', left_index=False, right_index=['key1', 'key2'])
        msg = "left_index parameter must be of type bool, not <class 'list'>"
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, how='left', left_index=['key1', 'key2'], right_index=False)
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, how='left', left_index=['key1', 'key2'], right_index=['key1', 'key2'])

    def test_merge_overlap(self, left: DataFrame) -> None:
        merged: DataFrame = merge(left, left, on='key')
        exp_len: Any = (left['key'].value_counts() ** 2).sum()
        assert len(merged) == exp_len
        assert 'v1_x' in merged
        assert 'v1_y' in merged

    def test_merge_different_column_key_names(self) -> None:
        left_: DataFrame = DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 4]})
        right_: DataFrame = DataFrame({'rkey': ['foo', 'bar', 'qux', 'foo'], 'value': [5, 6, 7, 8]})
        merged: DataFrame = left_.merge(right_, left_on='lkey', right_on='rkey', how='outer', sort=True)
        exp: Series = Series(['bar', 'baz', 'foo', 'foo', 'foo', 'foo', np.nan], name='lkey')
        tm.assert_series_equal(merged['lkey'], exp)
        exp = Series(['bar', np.nan, 'foo', 'foo', 'foo', 'foo', 'qux'], name='rkey')
        tm.assert_series_equal(merged['rkey'], exp)
        exp = Series([2, 3, 1, 1, 4, 4, np.nan], name='value_x')
        tm.assert_series_equal(merged['value_x'], exp)
        exp = Series([6, np.nan, 5, 8, 5, 8, 7], name='value_y')
        tm.assert_series_equal(merged['value_y'], exp)

    def test_merge_copy(self) -> None:
        left_: DataFrame = DataFrame({'a': 0, 'b': 1}, index=range(10))
        right_: DataFrame = DataFrame({'c': 'foo', 'd': 'bar'}, index=range(10))
        merged: DataFrame = merge(left_, right_, left_index=True, right_index=True)
        merged['a'] = 6
        assert (left_['a'] == 0).all()
        merged['d'] = 'peekaboo'
        assert (right_['d'] == 'bar').all()

    def test_merge_nocopy(self, using_infer_string: bool) -> None:
        left_: DataFrame = DataFrame({'a': 0, 'b': 1}, index=range(10))
        right_: DataFrame = DataFrame({'c': 'foo', 'd': 'bar'}, index=range(10))
        merged: DataFrame = merge(left_, right_, left_index=True, right_index=True)
        assert np.shares_memory(merged['a']._values, left_['a']._values)
        if not using_infer_string:
            assert np.shares_memory(merged['d']._values, right_['d']._values)

    def test_intelligently_handle_join_key(self) -> None:
        left_: DataFrame = DataFrame({'key': [1, 1, 2, 2, 3], 'value': list(range(5))}, columns=['value', 'key'])
        right_: DataFrame = DataFrame({'key': [1, 1, 2, 3, 4, 5], 'rvalue': list(range(6))})
        joined: DataFrame = merge(left_, right_, on='key', how='outer')
        expected: DataFrame = DataFrame({
            'key': [1, 1, 1, 1, 2, 2, 3, 4, 5],
            'value': np.array([0, 0, 1, 1, 2, 3, 4, np.nan, np.nan]),
            'rvalue': [0, 1, 0, 1, 2, 2, 3, 4, 5]
        }, columns=['value', 'key', 'rvalue'])
        tm.assert_frame_equal(joined, expected)

    def test_merge_join_key_dtype_cast(self) -> None:
        df1: DataFrame = DataFrame({'key': [1], 'v1': [10]})
        df2: DataFrame = DataFrame({'key': [2], 'v1': [20]})
        df: DataFrame = merge(df1, df2, how='outer')
        assert df['key'].dtype == 'int64'
        df1 = DataFrame({'key': [True], 'v1': [1]})
        df2 = DataFrame({'key': [False], 'v1': [0]})
        df = merge(df1, df2, how='outer')
        assert df['key'].dtype == 'bool'
        df1 = DataFrame({'val': [1]})
        df2 = DataFrame({'val': [2]})
        lkey: np.ndarray = np.array([1])
        rkey: np.ndarray = np.array([2])
        df = merge(df1, df2, left_on=lkey, right_on=rkey, how='outer')
        assert df['key_0'].dtype == np.dtype(int)

    def test_handle_join_key_pass_array(self) -> None:
        left_: DataFrame = DataFrame({'key': [1, 1, 2, 2, 3], 'value': np.arange(5)}, columns=['value', 'key'], dtype='int64')
        right_: DataFrame = DataFrame({'rvalue': np.arange(6)}, dtype='int64')
        key: np.ndarray = np.array([1, 1, 2, 3, 4, 5], dtype='int64')
        merged: DataFrame = merge(left_, right_, left_on='key', right_on=key, how='outer')
        merged2: DataFrame = merge(right_, left_, left_on=key, right_on='key', how='outer')
        tm.assert_series_equal(merged['key'], merged2['key'])
        assert merged['key'].notna().all()
        assert merged2['key'].notna().all()
        left_ = DataFrame({'value': np.arange(5)}, columns=['value'])
        right_ = DataFrame({'rvalue': np.arange(6)})
        lkey: np.ndarray = np.array([1, 1, 2, 2, 3])
        rkey: np.ndarray = np.array([1, 1, 2, 3, 4, 5])
        merged = merge(left_, right_, left_on=lkey, right_on=rkey, how='outer')
        expected: Series = Series([1, 1, 1, 1, 2, 2, 3, 4, 5], dtype=int, name='key_0')
        tm.assert_series_equal(merged['key_0'], expected)
        left_ = DataFrame({'value': np.arange(3)})
        right_ = DataFrame({'rvalue': np.arange(6)})
        key = np.array([0, 1, 1, 2, 2, 3], dtype=np.int64)
        merged = merge(left_, right_, left_index=True, right_on=key, how='outer')
        tm.assert_series_equal(merged['key_0'], Series(key, name='key_0'))

    def test_no_overlap_more_informative_error(self) -> None:
        dt: datetime = datetime.now()
        df1: DataFrame = DataFrame({'x': ['a']}, index=[dt])
        df2: DataFrame = DataFrame({'y': ['b', 'c']}, index=[dt, dt])
        msg: str = f'No common columns to perform merge on. Merge options: left_on={None}, right_on={None}, left_index={False}, right_index={False}'
        with pytest.raises(MergeError, match=msg):
            merge(df1, df2)

    def test_merge_non_unique_indexes(self) -> None:
        dt: datetime = datetime(2012, 5, 1)
        dt2: datetime = datetime(2012, 5, 2)
        dt3: datetime = datetime(2012, 5, 3)
        dt4: datetime = datetime(2012, 5, 4)
        df1: DataFrame = DataFrame({'x': ['a']}, index=[dt])
        df2: DataFrame = DataFrame({'y': ['b', 'c']}, index=[dt, dt])
        _check_merge(df1, df2)
        df1 = DataFrame({'x': ['a', 'b', 'q']}, index=[dt2, dt, dt4])
        df2 = DataFrame({'y': ['c', 'd', 'e', 'f', 'g', 'h']}, index=[dt3, dt3, dt2, dt2, dt, dt])
        _check_merge(df1, df2)
        df1 = DataFrame({'x': ['a', 'b']}, index=[dt, dt])
        df2 = DataFrame({'y': ['c', 'd']}, index=[dt, dt])
        _check_merge(df1, df2)

    def test_merge_non_unique_index_many_to_many(self) -> None:
        dt: datetime = datetime(2012, 5, 1)
        dt2: datetime = datetime(2012, 5, 2)
        dt3: datetime = datetime(2012, 5, 3)
        df1: DataFrame = DataFrame({'x': ['a', 'b', 'c', 'd']}, index=[dt2, dt2, dt, dt])
        df2: DataFrame = DataFrame({'y': ['e', 'f', 'g', ' h', 'i']}, index=[dt2, dt2, dt3, dt, dt])
        _check_merge(df1, df2)

    def test_left_merge_empty_dataframe(self) -> None:
        left_: DataFrame = DataFrame({'key': [1], 'value': [2]})
        right_: DataFrame = DataFrame({'key': []})
        result: DataFrame = merge(left_, right_, on='key', how='left')
        tm.assert_frame_equal(result, left_)
        result = merge(right_, left_, on='key', how='right')
        tm.assert_frame_equal(result, left_)

    def test_merge_empty_dataframe(self, index: Index, join_type: str) -> None:
        left_: DataFrame = DataFrame([], index=index[:0])
        right_: DataFrame = left_.copy()
        result: DataFrame = left_.join(right_, how=join_type)
        tm.assert_frame_equal(result, left_)

    @pytest.mark.parametrize('kwarg', [
        {'left_index': True, 'right_index': True},
        {'left_index': True, 'right_on': 'x'},
        {'left_on': 'a', 'right_index': True},
        {'left_on': 'a', 'right_on': 'x'}
    ])
    def test_merge_left_empty_right_empty(self, join_type: str, kwarg: dict) -> None:
        left_: DataFrame = DataFrame(columns=['a', 'b', 'c'])
        right_: DataFrame = DataFrame(columns=['x', 'y', 'z'])
        exp_in: DataFrame = DataFrame(columns=['a', 'b', 'c', 'x', 'y', 'z'], dtype=object)
        result: DataFrame = merge(left_, right_, how=join_type, **kwarg)
        tm.assert_frame_equal(result, exp_in)

    def test_merge_left_empty_right_notempty(self) -> None:
        left_: DataFrame = DataFrame(columns=['a', 'b', 'c'])
        right_: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['x', 'y', 'z'])
        exp_out: DataFrame = DataFrame({
            'a': np.array([np.nan] * 3, dtype=object),
            'b': np.array([np.nan] * 3, dtype=object),
            'c': np.array([np.nan] * 3, dtype=object),
            'x': [1, 4, 7],
            'y': [2, 5, 8],
            'z': [3, 6, 9]
        }, columns=['a', 'b', 'c', 'x', 'y', 'z'])
        exp_in: DataFrame = exp_out[0:0]

        def check1(exp: DataFrame, kwarg: dict) -> None:
            result: DataFrame = merge(left_, right_, how='inner', **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left_, right_, how='left', **kwarg)
            tm.assert_frame_equal(result, exp)

        def check2(exp: DataFrame, kwarg: dict) -> None:
            result: DataFrame = merge(left_, right_, how='right', **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left_, right_, how='outer', **kwarg)
            tm.assert_frame_equal(result, exp)
        for kwarg in [{'left_index': True, 'right_index': True}, {'left_index': True, 'right_on': 'x'}]:
            check1(exp_in, kwarg)
            check2(exp_out, kwarg)
        kwarg = {'left_on': 'a', 'right_index': True}
        check1(exp_in, kwarg)
        exp_out['a'] = [0, 1, 2]
        check2(exp_out, kwarg)
        kwarg = {'left_on': 'a', 'right_on': 'x'}
        check1(exp_in, kwarg)
        exp_out['a'] = np.array([np.nan] * 3, dtype=object)
        check2(exp_out, kwarg)

    def test_merge_left_notempty_right_empty(self) -> None:
        left_: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['a', 'b', 'c'])
        right_: DataFrame = DataFrame(columns=['x', 'y', 'z'])
        exp_out: DataFrame = DataFrame({
            'a': [1, 4, 7],
            'b': [2, 5, 8],
            'c': [3, 6, 9],
            'x': np.array([np.nan] * 3, dtype=object),
            'y': np.array([np.nan] * 3, dtype=object),
            'z': np.array([np.nan] * 3, dtype=object)
        }, columns=['a', 'b', 'c', 'x', 'y', 'z'])
        exp_in: DataFrame = exp_out[0:0]
        exp_in.index = exp_in.index.astype(object)

        def check1(exp: DataFrame, kwarg: dict) -> None:
            result: DataFrame = merge(left_, right_, how='inner', **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left_, right_, how='right', **kwarg)
            tm.assert_frame_equal(result, exp)

        def check2(exp: DataFrame, kwarg: dict) -> None:
            result: DataFrame = merge(left_, right_, how='left', **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left_, right_, how='outer', **kwarg)
            tm.assert_frame_equal(result, exp)
        for kwarg in [{'left_index': True, 'right_index': True}, {'left_index': True, 'right_on': 'x'}, {'left_on': 'a', 'right_index': True}, {'left_on': 'a', 'right_on': 'x'}]:
            check1(exp_in, kwarg)
            check2(exp_out, kwarg)

    @pytest.mark.parametrize('series_of_dtype', [
        Series([1], dtype='int64'),
        Series([1], dtype='Int64'),
        Series([1.23]),
        Series(['foo']),
        Series([True]),
        Series([pd.Timestamp('2018-01-01')]),
        Series([pd.Timestamp('2018-01-01', tz='US/Eastern')])
    ])
    @pytest.mark.parametrize('series_of_dtype2', [
        Series([1], dtype='int64'),
        Series([1], dtype='Int64'),
        Series([1.23]),
        Series(['foo']),
        Series([True]),
        Series([pd.Timestamp('2018-01-01')]),
        Series([pd.Timestamp('2018-01-01', tz='US/Eastern')])
    ])
    def test_merge_empty_frame(self, series_of_dtype: Series, series_of_dtype2: Series) -> None:
        df: DataFrame = DataFrame({'key': series_of_dtype, 'value': series_of_dtype2}, columns=['key', 'value'])
        df_empty: DataFrame = df[:0]
        expected: DataFrame = DataFrame({
            'key': Series(dtype=df.dtypes['key']),
            'value_x': Series(dtype=df.dtypes['value']),
            'value_y': Series(dtype=df.dtypes['value'])
        }, columns=['key', 'value_x', 'value_y'])
        actual: DataFrame = df_empty.merge(df, on='key')
        tm.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize('series_of_dtype', [
        Series([1], dtype='int64'),
        Series([1], dtype='Int64'),
        Series([1.23]),
        Series(['foo']),
        Series([True]),
        Series([pd.Timestamp('2018-01-01')]),
        Series([pd.Timestamp('2018-01-01', tz='US/Eastern')])
    ])
    @pytest.mark.parametrize('series_of_dtype_all_na', [
        Series([np.nan], dtype='Int64'),
        Series([np.nan], dtype='float'),
        Series([np.nan], dtype='object'),
        Series([pd.NaT])
    ])
    def test_merge_all_na_column(self, series_of_dtype: Series, series_of_dtype_all_na: Series) -> None:
        df_left: DataFrame = DataFrame({'key': series_of_dtype, 'value': series_of_dtype_all_na}, columns=['key', 'value'])
        df_right: DataFrame = DataFrame({'key': series_of_dtype, 'value': series_of_dtype_all_na}, columns=['key', 'value'])
        expected: DataFrame = DataFrame({
            'key': series_of_dtype,
            'value_x': series_of_dtype_all_na,
            'value_y': series_of_dtype_all_na
        }, columns=['key', 'value_x', 'value_y'])
        actual: DataFrame = df_left.merge(df_right, on='key')
        tm.assert_frame_equal(actual, expected)

    def test_merge_nosort(self) -> None:
        d: dict = {'var1': np.random.default_rng(2).integers(0, 10, size=10),
                   'var2': np.random.default_rng(2).integers(0, 10, size=10),
                   'var3': [datetime(2012, 1, 12),
                            datetime(2011, 2, 4),
                            datetime(2010, 2, 3),
                            datetime(2012, 1, 12),
                            datetime(2011, 2, 4),
                            datetime(2012, 4, 3),
                            datetime(2012, 3, 4),
                            datetime(2008, 5, 1),
                            datetime(2010, 2, 3),
                            datetime(2012, 2, 3)]}
        df: DataFrame = DataFrame.from_dict(d)
        var3: Any = df.var3.unique()
        var3 = np.sort(var3)
        new: DataFrame = DataFrame.from_dict({'var3': var3, 'var8': np.random.default_rng(2).random(7)})
        result: DataFrame = df.merge(new, on='var3', sort=False)
        exp: DataFrame = merge(df, new, on='var3', sort=False)
        tm.assert_frame_equal(result, exp)
        assert (df.var3.unique() == result.var3.unique()).all()

    @pytest.mark.parametrize(('sort', 'values'), [(False, [1, 1, 0, 1, 1]), (True, [0, 1, 1, 1, 1])])
    @pytest.mark.parametrize('how', ['left', 'right'])
    def test_merge_same_order_left_right(self, sort: bool, values: List[int], how: str) -> None:
        df: DataFrame = DataFrame({'a': [1, 0, 1]})
        result: DataFrame = df.merge(df, on='a', how=how, sort=sort)
        expected: DataFrame = DataFrame(values, columns=['a'])
        tm.assert_frame_equal(result, expected)

    def test_merge_nan_right(self) -> None:
        df1: DataFrame = DataFrame({'i1': [0, 1], 'i2': [0, 1]})
        df2: DataFrame = DataFrame({'i1': [0], 'i3': [0]})
        result: DataFrame = df1.join(df2, on='i1', rsuffix='_')
        expected: DataFrame = DataFrame({
            'i1': {0: 0.0, 1: 1},
            'i2': {0: 0, 1: 1},
            'i1_': {0: 0, 1: np.nan},
            'i3': {0: 0.0, 1: np.nan},
            None: {0: 0, 1: 0}
        }, columns=Index(['i1', 'i2', 'i1_', 'i3', None], dtype='object')).set_index(None).reset_index()[['i1', 'i2', 'i1_', 'i3']]
        result.columns = result.columns.astype('object')
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_merge_nan_right2(self) -> None:
        df1: DataFrame = DataFrame({'i1': [0, 1], 'i2': [0.5, 1.5]})
        df2: DataFrame = DataFrame({'i1': [0], 'i3': [0.7]})
        result: DataFrame = df1.join(df2, rsuffix='_', on='i1')
        expected: DataFrame = DataFrame({
            'i1': {0: 0, 1: 1},
            'i1_': {0: 0.0, 1: np.nan},
            'i2': {0: 0.5, 1: 1.5},
            'i3': {0: 0.7, 1: np.nan}
        })[['i1', 'i2', 'i1_', 'i3']]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning')
    def test_merge_type(self, df: DataFrame, df2: DataFrame) -> None:
        class NotADataFrame(DataFrame):
            @property
            def _constructor(self) -> Any:
                return NotADataFrame
        nad: NotADataFrame = NotADataFrame(df)
        result: NotADataFrame = nad.merge(df2, on='key1')
        assert isinstance(result, NotADataFrame)

    def test_join_append_timedeltas(self) -> None:
        d: DataFrame = DataFrame.from_dict({
            'd': [datetime(2013, 11, 5, 5, 56)],
            't': [timedelta(0, 22500)]
        })
        df: DataFrame = DataFrame(columns=list('dt'))
        df = concat([df, d], ignore_index=True)
        result: DataFrame = concat([df, d], ignore_index=True)
        expected: DataFrame = DataFrame({
            'd': [datetime(2013, 11, 5, 5, 56), datetime(2013, 11, 5, 5, 56)],
            't': [timedelta(0, 22500), timedelta(0, 22500)]
        }, dtype=object)
        tm.assert_frame_equal(result, expected)

    def test_join_append_timedeltas2(self) -> None:
        td: np.timedelta64 = np.timedelta64(300000000)
        lhs: DataFrame = DataFrame(Series([td, td], index=['A', 'B']))
        rhs: DataFrame = DataFrame(Series([td], index=['A']))
        result: DataFrame = lhs.join(rhs, rsuffix='r', how='left')
        expected: DataFrame = DataFrame({
            '0': Series([td, td], index=list('AB')),
            '0r': Series([td, pd.NaT], index=list('AB'))
        })
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('unit', ['D', 'h', 'm', 's', 'ms', 'us', 'ns'])
    def test_other_datetime_unit(self, unit: str) -> None:
        df1: DataFrame = DataFrame({'entity_id': [101, 102]})
        ser: Series = Series([None, None], index=[101, 102], name='days')
        dtype: str = f'datetime64[{unit}]'
        if unit in ['D', 'h', 'm']:
            exp_dtype: str = 'datetime64[s]'
        else:
            exp_dtype = dtype
        df2: DataFrame = ser.astype(exp_dtype).to_frame('days')
        assert df2['days'].dtype == exp_dtype
        result: DataFrame = df1.merge(df2, left_on='entity_id', right_index=True)
        days: np.ndarray = np.array(['nat', 'nat'], dtype=exp_dtype)
        days = pd.core.arrays.DatetimeArray._simple_new(days, dtype=days.dtype)
        exp: DataFrame = DataFrame({'entity_id': [101, 102], 'days': days}, columns=['entity_id', 'days'])
        assert exp['days'].dtype == exp_dtype
        tm.assert_frame_equal(result, exp)

    @pytest.mark.parametrize('unit', ['D', 'h', 'm', 's', 'ms', 'us', 'ns'])
    def test_other_timedelta_unit(self, unit: str) -> None:
        df1: DataFrame = DataFrame({'entity_id': [101, 102]})
        ser: Series = Series([None, None], index=[101, 102], name='days')
        dtype: str = f'm8[{unit}]'
        if unit in ['D', 'h', 'm']:
            msg: str = "Supported resolutions are 's', 'ms', 'us', 'ns'"
            with pytest.raises(ValueError, match=msg):
                ser.astype(dtype)
            df2: DataFrame = ser.astype('m8[s]').to_frame('days')
        else:
            df2 = ser.astype(dtype).to_frame('days')
            assert df2['days'].dtype == dtype
        result: DataFrame = df1.merge(df2, left_on='entity_id', right_index=True)
        exp: DataFrame = DataFrame({'entity_id': [101, 102], 'days': np.array(['nat', 'nat'], dtype=dtype)}, columns=['entity_id', 'days'])
        tm.assert_frame_equal(result, exp)

    def test_overlapping_columns_error_message(self) -> None:
        df: DataFrame = DataFrame({'key': [1, 2, 3], 'v1': [4, 5, 6], 'v2': [7, 8, 9]})
        df2: DataFrame = DataFrame({'key': [1, 2, 3], 'v1': [4, 5, 6], 'v2': [7, 8, 9]})
        df.columns = ['key', 'foo', 'foo']
        df2.columns = ['key', 'bar', 'bar']
        expected: DataFrame = DataFrame({
            'key': [1, 2, 3],
            'v1': [4, 5, 6],
            'v2': [7, 8, 9],
            'v3': [4, 5, 6],
            'v4': [7, 8, 9]
        })
        expected.columns = ['key', 'foo', 'foo', 'bar', 'bar']
        tm.assert_frame_equal(merge(df, df2), expected)
        df2.columns = ['key1', 'foo', 'foo']
        msg: str = "Data columns not unique: Index\\(\\['foo'\\], dtype='object|str'\\)"
        with pytest.raises(MergeError, match=msg):
            merge(df, df2)

    def test_merge_on_datetime64tz(self) -> None:
        left: DataFrame = DataFrame({
            'key': pd.date_range('20151010', periods=2, tz='US/Eastern'),
            'value': [1, 2]
        })
        right: DataFrame = DataFrame({
            'key': pd.date_range('20151011', periods=3, tz='US/Eastern'),
            'value': [1, 2, 3]
        })
        expected: DataFrame = DataFrame({
            'key': pd.date_range('20151010', periods=4, tz='US/Eastern'),
            'value_x': [1, 2, np.nan, np.nan],
            'value_y': [np.nan, 1, 2, 3]
        })
        result: DataFrame = merge(left, right, on='key', how='outer')
        tm.assert_frame_equal(result, expected)

    def test_merge_datetime64tz_values(self) -> None:
        left: DataFrame = DataFrame({
            'key': [1, 2],
            'value': pd.date_range('20151010', periods=2, tz='US/Eastern')
        })
        right: DataFrame = DataFrame({
            'key': [2, 3],
            'value': pd.date_range('20151011', periods=2, tz='US/Eastern')
        })
        expected: DataFrame = DataFrame({
            'key': [1, 2, 3],
            'value_x': list(pd.date_range('20151010', periods=2, tz='US/Eastern')) + [pd.NaT],
            'value_y': [pd.NaT] + list(pd.date_range('20151011', periods=2, tz='US/Eastern'))
        })
        result: DataFrame = merge(left, right, on='key', how='outer')
        tm.assert_frame_equal(result, expected)
        assert result['value_x'].dtype == 'datetime64[ns, US/Eastern]'
        assert result['value_y'].dtype == 'datetime64[ns, US/Eastern]'

    def test_merge_on_datetime64tz_empty(self) -> None:
        dtz: pd.DatetimeTZDtype = pd.DatetimeTZDtype(tz='UTC')
        right: DataFrame = DataFrame({
            'date': DatetimeIndex(['2018'], dtype=dtz),
            'value': [4.0],
            'date2': DatetimeIndex(['2019'], dtype=dtz)
        }, columns=['date', 'value', 'date2'])
        left: DataFrame = right[:0]
        result: DataFrame = left.merge(right, on='date')
        expected: DataFrame = DataFrame({
            'date': Series(dtype=dtz),
            'value_x': Series(dtype=float),
            'date2_x': Series(dtype=dtz),
            'value_y': Series(dtype=float),
            'date2_y': Series(dtype=dtz)
        }, columns=['date', 'value_x', 'date2_x', 'value_y', 'date2_y'])
        tm.assert_frame_equal(result, expected)

    def test_merge_datetime64tz_with_dst_transition(self) -> None:
        df1: DataFrame = DataFrame(pd.date_range('2017-10-29 01:00', periods=4, freq='h', tz='Europe/Madrid'), columns=['date'])
        df1['value'] = 1
        df2: DataFrame = DataFrame({
            'date': pd.to_datetime(['2017-10-29 03:00:00', '2017-10-29 04:00:00', '2017-10-29 05:00:00']),
            'value': 2
        })
        df2['date'] = df2['date'].dt.tz_localize('UTC').dt.tz_convert('Europe/Madrid')
        result: DataFrame = merge(df1, df2, how='outer', on='date')
        expected: DataFrame = DataFrame({
            'date': pd.date_range('2017-10-29 01:00', periods=7, freq='h', tz='Europe/Madrid'),
            'value_x': [1] * 4 + [np.nan] * 3,
            'value_y': [np.nan] * 4 + [2] * 3
        })
        tm.assert_frame_equal(result, expected)

    def test_merge_non_unique_period_index(self) -> None:
        index: PeriodIndex = pd.period_range('2016-01-01', periods=16, freq='M')
        df: DataFrame = DataFrame(list(range(len(index))), index=index, columns=['pnum'])
        df2: DataFrame = concat([df, df])
        result: DataFrame = df.merge(df2, left_index=True, right_index=True, how='inner')
        expected: DataFrame = DataFrame(np.tile(np.arange(16, dtype=np.int64).repeat(2).reshape(-1, 1), 2), columns=['pnum_x', 'pnum_y'], index=df2.sort_index().index)
        tm.assert_frame_equal(result, expected)

    def test_merge_on_periods(self) -> None:
        left: DataFrame = DataFrame({'key': pd.period_range('20151010', periods=2, freq='D'), 'value': [1, 2]})
        right: DataFrame = DataFrame({'key': pd.period_range('20151011', periods=3, freq='D'), 'value': [1, 2, 3]})
        expected: DataFrame = DataFrame({
            'key': pd.period_range('20151010', periods=4, freq='D'),
            'value_x': [1, 2, np.nan, np.nan],
            'value_y': [np.nan, 1, 2, 3]
        })
        result: DataFrame = merge(left, right, on='key', how='outer')
        tm.assert_frame_equal(result, expected)

    def test_merge_period_values(self) -> None:
        left: DataFrame = DataFrame({'key': [1, 2], 'value': pd.period_range('20151010', periods=2, freq='D')})
        right: DataFrame = DataFrame({'key': [2, 3], 'value': pd.period_range('20151011', periods=2, freq='D')})
        exp_x: PeriodIndex = pd.period_range('20151010', periods=2, freq='D')
        exp_y: PeriodIndex = pd.period_range('20151011', periods=2, freq='D')
        expected: DataFrame = DataFrame({
            'key': [1, 2, 3],
            'value_x': list(exp_x) + [pd.NaT],
            'value_y': [pd.NaT] + list(exp_y)
        })
        result: DataFrame = merge(left, right, on='key', how='outer')
        tm.assert_frame_equal(result, expected)
        assert result['value_x'].dtype == 'Period[D]'
        assert result['value_y'].dtype == 'Period[D]'

    def test_indicator(self, dfs_for_indicator: Tuple[DataFrame, DataFrame]) -> None:
        df1, df2 = dfs_for_indicator
        df1_copy: DataFrame = df1.copy()
        df2_copy: DataFrame = df2.copy()
        df_result: DataFrame = DataFrame({
            'col1': [0, 1, 2, 3, 4, 5],
            'col_conflict_x': [1, 2, np.nan, np.nan, np.nan, np.nan],
            'col_left': ['a', 'b', np.nan, np.nan, np.nan, np.nan],
            'col_conflict_y': [np.nan, 1, 2, 3, 4, 5],
            'col_right': [np.nan, 2, 2, 2, 2, 2]
        })
        df_result['_merge'] = Categorical(['left_only', 'both', 'right_only', 'right_only', 'right_only', 'right_only'],
                                            categories=['left_only', 'right_only', 'both'])
        df_result = df_result[['col1', 'col_conflict_x', 'col_left', 'col_conflict_y', 'col_right', '_merge']]
        test_: DataFrame = merge(df1, df2, on='col1', how='outer', indicator=True)
        tm.assert_frame_equal(test_, df_result)
        test_ = df1.merge(df2, on='col1', how='outer', indicator=True)
        tm.assert_frame_equal(test_, df_result)
        tm.assert_frame_equal(df1, df1_copy)
        tm.assert_frame_equal(df2, df2_copy)
        df_result_custom_name: DataFrame = df_result.rename(columns={'_merge': 'custom_name'})
        test_custom_name: DataFrame = merge(df1, df2, on='col1', how='outer', indicator='custom_name')
        tm.assert_frame_equal(test_custom_name, df_result_custom_name)
        test_custom_name = df1.merge(df2, on='col1', how='outer', indicator='custom_name')
        tm.assert_frame_equal(test_custom_name, df_result_custom_name)

    def test_merge_indicator_arg_validation(self, dfs_for_indicator: Tuple[DataFrame, DataFrame]) -> None:
        df1, df2 = dfs_for_indicator
        msg: str = 'indicator option can only accept boolean or string arguments'
        with pytest.raises(ValueError, match=msg):
            merge(df1, df2, on='col1', how='outer', indicator=5)
        with pytest.raises(ValueError, match=msg):
            df1.merge(df2, on='col1', how='outer', indicator=5)

    def test_merge_indicator_result_integrity(self, dfs_for_indicator: Tuple[DataFrame, DataFrame]) -> None:
        df1, df2 = dfs_for_indicator
        test2: DataFrame = merge(df1, df2, on='col1', how='left', indicator=True)
        assert (test2._merge != 'right_only').all()
        test2 = df1.merge(df2, on='col1', how='left', indicator=True)
        assert (test2._merge != 'right_only').all()
        test3: DataFrame = merge(df1, df2, on='col1', how='right', indicator=True)
        assert (test3._merge != 'left_only').all()
        test3 = df1.merge(df2, on='col1', how='right', indicator=True)
        assert (test3._merge != 'left_only').all()
        test4: DataFrame = merge(df1, df2, on='col1', how='inner', indicator=True)
        assert (test4._merge == 'both').all()
        test4 = df1.merge(df2, on='col1', how='inner', indicator=True)
        assert (test4._merge == 'both').all()

    def test_merge_indicator_invalid(self, dfs_for_indicator: Tuple[DataFrame, DataFrame]) -> None:
        df1, _ = dfs_for_indicator
        for i in ['_right_indicator', '_left_indicator', '_merge']:
            df_badcolumn: DataFrame = DataFrame({'col1': [1, 2], i: [2, 2]})
            msg: str = f'Cannot use `indicator=True` option when data contains a column named {i}|Cannot use name of an existing column for indicator column'
            with pytest.raises(ValueError, match=msg):
                merge(df1, df_badcolumn, on='col1', how='outer', indicator=True)
            with pytest.raises(ValueError, match=msg):
                df1.merge(df_badcolumn, on='col1', how='outer', indicator=True)
        df_badcolumn = DataFrame({'col1': [1, 2], 'custom_column_name': [2, 2]})
        msg = 'Cannot use name of an existing column for indicator column'
        with pytest.raises(ValueError, match=msg):
            merge(df1, df_badcolumn, on='col1', how='outer', indicator='custom_column_name')
        with pytest.raises(ValueError, match=msg):
            df1.merge(df_badcolumn, on='col1', how='outer', indicator='custom_column_name')

    def test_merge_indicator_multiple_columns(self) -> None:
        df3: DataFrame = DataFrame({'col1': [0, 1], 'col2': ['a', 'b']})
        df4: DataFrame = DataFrame({'col1': [1, 1, 3], 'col2': ['b', 'x', 'y']})
        hand_coded_result: DataFrame = DataFrame({'col1': [0, 1, 1, 3], 'col2': ['a', 'b', 'x', 'y']})
        hand_coded_result['_merge'] = Categorical(['left_only', 'both', 'right_only', 'right_only'], categories=['left_only', 'right_only', 'both'])
        test5: DataFrame = merge(df3, df4, on=['col1', 'col2'], how='outer', indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)
        test5 = df3.merge(df4, on=['col1', 'col2'], how='outer', indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)

    def test_validation(self) -> None:
        left_: DataFrame = DataFrame({'a': ['a', 'b', 'c', 'd'], 'b': ['cat', 'dog', 'weasel', 'horse']}, index=range(4))
        right_: DataFrame = DataFrame({'a': ['a', 'b', 'c', 'd', 'e'], 'c': ['meow', 'bark', 'um... weasel noise?', 'nay', 'chirp']}, index=range(5))
        left_copy: DataFrame = left_.copy()
        right_copy: DataFrame = right_.copy()
        result: DataFrame = merge(left_, right_, left_index=True, right_index=True, validate='1:1')
        tm.assert_frame_equal(left_, left_copy)
        tm.assert_frame_equal(right_, right_copy)
        expected: DataFrame = DataFrame({
            'a_x': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'a_y': ['a', 'b', 'c', 'd'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, index=range(4), columns=['a_x', 'b', 'a_y', 'c'])
        result = merge(left_, right_, left_index=True, right_index=True, validate='one_to_one')
        tm.assert_frame_equal(result, expected)
        expected_2: DataFrame = DataFrame({
            'a': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, index=range(4))
        result = merge(left_, right_, on='a', validate='1:1')
        tm.assert_frame_equal(left_, left_copy)
        tm.assert_frame_equal(right_, right_copy)
        tm.assert_frame_equal(result, expected_2)
        result = merge(left_, right_, on='a', validate='one_to_one')
        tm.assert_frame_equal(result, expected_2)
        expected_3: DataFrame = DataFrame({
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'a': ['a', 'b', 'c', 'd'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, columns=['b', 'a', 'c'], index=range(4))
        left_index_reset: DataFrame = left_.set_index('a')
        result = merge(left_index_reset, right_, left_index=True, right_on='a', validate='one_to_one')
        tm.assert_frame_equal(result, expected_3)
        right_w_dups: DataFrame = concat([right_, DataFrame({'a': ['e'], 'c': ['moo']}, index=[4])])
        merge(left_, right_w_dups, left_index=True, right_index=True, validate='one_to_many')
        msg: str = 'Merge keys are not unique in right dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_, right_w_dups, left_index=True, right_index=True, validate='one_to_one')
        with pytest.raises(MergeError, match=msg):
            merge(left_, right_w_dups, on='a', validate='one_to_one')
        left_w_dups: DataFrame = concat([left_, DataFrame({'a': ['a'], 'c': ['cow']}, index=[3])], sort=True)
        merge(left_w_dups, right_, left_index=True, right_index=True, validate='many_to_one')
        msg = 'Merge keys are not unique in left dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_, left_index=True, right_index=True, validate='one_to_one')
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_, on='a', validate='one_to_one')
        merge(left_w_dups, right_w_dups, on='a', validate='many_to_many')
        msg = 'Merge keys are not unique in right dataset; not a many-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, left_index=True, right_index=True, validate='many_to_one')
        msg = 'Merge keys are not unique in left dataset; not a one-to-many merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, on='a', validate='one_to_many')
        msg = '"jibberish" is not a valid argument. Valid arguments are:\n- "1:1"\n- "1:m"\n- "m:1"\n- "m:m"\n- "one_to_one"\n- "one_to_many"\n- "many_to_one"\n- "many_to_many"'
        with pytest.raises(ValueError, match=msg):
            merge(left_, right_, on='a', validate='jibberish')
        left_ = DataFrame({'a': ['a', 'a', 'b', 'b'], 'b': [0, 1, 0, 1], 'c': ['cat', 'dog', 'weasel', 'horse']}, index=range(4))
        right_ = DataFrame({'a': ['a', 'a', 'b'], 'b': [0, 1, 0], 'd': ['meow', 'bark', 'um... weasel noise?']}, index=range(3))
        expected_multi: DataFrame = DataFrame({
            'a': ['a', 'a', 'b'],
            'b': [0, 1, 0],
            'c': ['cat', 'dog', 'weasel'],
            'd': ['meow', 'bark', 'um... weasel noise?']
        }, index=range(3))
        msg = 'Merge keys are not unique in either left or right dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_, right_, on='a', validate='1:1')
        result = merge(left_, right_, on=['a', 'b'], validate='1:1')
        tm.assert_frame_equal(result, expected_multi)

    def test_merge_two_empty_df_no_division_error(self) -> None:
        a: DataFrame = DataFrame({'a': [], 'b': [], 'c': []})
        with np.errstate(divide='raise'):
            merge(a, a, on=('a', 'b'))

    @pytest.mark.parametrize('how', ['right', 'outer'])
    @pytest.mark.parametrize('index,expected_index', [
        (CategoricalIndex([1, 2, 4]), CategoricalIndex([1, 2, 4, None, None, None])),
        (DatetimeIndex(['2001-01-01', '2002-02-02', '2003-03-03'], dtype='M8[ns]'),
         DatetimeIndex(['2001-01-01', '2002-02-02', '2003-03-03', pd.NaT, pd.NaT, pd.NaT], dtype='M8[ns]')),
        *[(Index([1, 2, 3], dtype=dtyp), Index([1, 2, 3, None, None, None], dtype=np.float64)) for dtyp in tm.ALL_REAL_NUMPY_DTYPES],
        (IntervalIndex.from_tuples([(1, 2), (2, 3), (3, 4)]),
         IntervalIndex.from_tuples([(1, 2), (2, 3), (3, 4), np.nan, np.nan, np.nan])),
        (PeriodIndex(['2001-01-01', '2001-01-02', '2001-01-03'], freq='D'),
         PeriodIndex(['2001-01-01', '2001-01-02', '2001-01-03', pd.NaT, pd.NaT, pd.NaT], freq='D')),
        (TimedeltaIndex(['1D', '2D', '3D']),
         TimedeltaIndex(['1D', '2D', '3D', pd.NaT, pd.NaT, pd.NaT]))
    ])
    def test_merge_on_index_with_more_values(self, how: str, index: Index, expected_index: Index) -> None:
        df1: DataFrame = DataFrame({'a': [0, 1, 2], 'key': [0, 1, 2]}, index=index)
        df2: DataFrame = DataFrame({'b': [0, 1, 2, 3, 4, 5]})
        result: DataFrame = df1.merge(df2, left_on='key', right_index=True, how=how)
        expected: DataFrame = DataFrame([[0, 0, 0], [1, 1, 1], [2, 2, 2], [np.nan, 3, 3], [np.nan, 4, 4], [np.nan, 5, 5]], columns=['a', 'key', 'b'])
        expected.set_index(expected_index, inplace=True)
        tm.assert_frame_equal(result, expected)

    def test_merge_right_index_right(self) -> None:
        left_: DataFrame = DataFrame({'a': [1, 2, 3], 'key': [0, 1, 1]})
        right_: DataFrame = DataFrame({'b': [1, 2, 3]})
        expected: DataFrame = DataFrame({'a': [1, 2, 3, None], 'key': [0, 1, 1, 2], 'b': [1, 2, 2, 3]}, columns=['a', 'key', 'b'], index=[0, 1, 2, np.nan])
        result: DataFrame = left_.merge(right_, left_on='key', right_index=True, how='right')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('how', ['left', 'right'])
    def test_merge_preserves_row_order(self, how: str) -> None:
        left_df: DataFrame = DataFrame({'animal': ['dog', 'pig'], 'max_speed': [40, 11]})
        right_df: DataFrame = DataFrame({'animal': ['quetzal', 'pig'], 'max_speed': [80, 11]})
        result: DataFrame = left_df.merge(right_df, on=['animal', 'max_speed'], how=how)
        if how == 'right':
            expected: DataFrame = DataFrame({'animal': ['quetzal', 'pig'], 'max_speed': [80, 11]})
        else:
            expected = DataFrame({'animal': ['dog', 'pig'], 'max_speed': [40, 11]})
        tm.assert_frame_equal(result, expected)

    def test_merge_take_missing_values_from_index_of_other_dtype(self) -> None:
        left_: DataFrame = DataFrame({'a': [1, 2, 3], 'key': Categorical(['a', 'a', 'b'], categories=list('abc'))})
        right_: DataFrame = DataFrame({'b': [1, 2, 3]}, index=CategoricalIndex(['a', 'b', 'c']))
        result: DataFrame = left_.merge(right_, left_on='key', right_index=True, how='right')
        expected: DataFrame = DataFrame({'a': [1, 2, 3, None], 'key': Categorical(['a', 'a', 'b', 'c']), 'b': [1, 1, 2, 3]}, index=[0, 1, 2, np.nan])
        expected = expected.reindex(columns=['a', 'key', 'b'])
        tm.assert_frame_equal(result, expected)

    def test_merge_readonly(self) -> None:
        data1: DataFrame = DataFrame(np.arange(20).reshape((4, 5)) + 1, columns=['a', 'b', 'c', 'd', 'e'])
        data2: DataFrame = DataFrame(np.arange(20).reshape((5, 4)) + 1, columns=['a', 'b', 'x', 'y'])
        for block in data1._mgr.blocks:
            block.values.flags.writeable = False
        data1.merge(data2)

    def test_merge_how_validation(self) -> None:
        data1: DataFrame = DataFrame(np.arange(20).reshape((4, 5)) + 1, columns=['a', 'b', 'c', 'd', 'e'])
        data2: DataFrame = DataFrame(np.arange(20).reshape((5, 4)) + 1, columns=['a', 'b', 'x', 'y'])
        msg: str = "'full' is not a valid Merge type: left, right, inner, outer, left_anti, right_anti, cross, asof"
        with pytest.raises(ValueError, match=re.escape(msg)):
            data1.merge(data2, how='full')

def _check_merge(x: DataFrame, y: DataFrame) -> None:
    for how in ['inner', 'left', 'outer']:
        for sort in [True, False]:
            result: DataFrame = x.join(y, how=how, sort=sort)
            expected: DataFrame = merge(x.reset_index(), y.reset_index(), how=how, sort=sort)
            expected = expected.set_index('index')
            tm.assert_frame_equal(result, expected, check_names=False)

class TestMergeDtypes:
    @pytest.mark.parametrize('dtype', [object, 'category'])
    def test_different(self, dtype: Union[type, str]) -> None:
        left_: DataFrame = DataFrame({
            'A': ['foo', 'bar'],
            'B': Series(['foo', 'bar']).astype('category'),
            'C': [1, 2],
            'D': [1.0, 2.0],
            'E': Series([1, 2], dtype='uint64'),
            'F': Series([1, 2], dtype='int32')
        })
        right_vals: Series = Series(['foo', 'bar'], dtype=dtype)
        right_: DataFrame = DataFrame({'A': right_vals})
        result: DataFrame = merge(left_, right_, on='A')
        assert is_object_dtype(result.A.dtype) or is_string_dtype(result.A.dtype)

    @pytest.mark.parametrize('d2', [np.int64, np.float64, np.float32, np.float16])
    def test_join_multi_dtypes(self, any_int_numpy_dtype: Any, d2: Any) -> None:
        dtype1: np.dtype = np.dtype(any_int_numpy_dtype)
        dtype2: np.dtype = np.dtype(d2)
        left_: DataFrame = DataFrame({
            'k1': np.array([0, 1, 2] * 8, dtype=dtype1),
            'k2': ['foo', 'bar'] * 12,
            'v': np.array(np.arange(24), dtype=np.int64)
        })
        index: MultiIndex = MultiIndex.from_tuples([(2, 'bar'), (1, 'foo')])
        right_: DataFrame = DataFrame({'v2': np.array([5, 7], dtype=dtype2)}, index=index)
        result: DataFrame = left_.join(right_, on=['k1', 'k2'])
        expected: DataFrame = left_.copy()
        if dtype2.kind == 'i':
            dtype2 = np.dtype('float64')
        expected['v2'] = np.array(np.nan, dtype=dtype2)
        expected.loc[(expected.k1 == 2) & (expected.k2 == 'bar'), 'v2'] = 5
        expected.loc[(expected.k1 == 1) & (expected.k2 == 'foo'), 'v2'] = 7
        tm.assert_frame_equal(result, expected)
        result = left_.join(right_, on=['k1', 'k2'], sort=True)
        expected.sort_values(['k1', 'k2'], kind='mergesort', inplace=True)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('int_vals, float_vals, exp_vals', [
        ([1, 2, 3], [1.0, 2.0, 3.0], {'X': [1, 2, 3], 'Y': [1.0, 2.0, 3.0]}),
        ([1, 2, 3], [1.0, 3.0], {'X': [1, 3], 'Y': [1.0, 3.0]}),
        ([1, 2], [1.0, 2.0, 3.0], {'X': [1, 2], 'Y': [1.0, 2.0]})
    ])
    def test_merge_on_ints_floats(self, int_vals: List[int], float_vals: List[float], exp_vals: dict) -> None:
        A: DataFrame = DataFrame({'X': int_vals})
        B: DataFrame = DataFrame({'Y': float_vals})
        expected: DataFrame = DataFrame(exp_vals)
        result: DataFrame = A.merge(B, left_on='X', right_on='Y')
        tm.assert_frame_equal(result, expected)
        result = B.merge(A, left_on='Y', right_on='X')
        tm.assert_frame_equal(result, expected[['Y', 'X']])

    def test_merge_key_dtype_cast(self) -> None:
        df1: DataFrame = DataFrame({'key': [1.0, 2.0], 'v1': [10, 20]}, columns=['key', 'v1'])
        df2: DataFrame = DataFrame({'key': [2], 'v2': [200]}, columns=['key', 'v2'])
        result: DataFrame = merge(df1, df2, on='key', how='left')
        expected: DataFrame = DataFrame({'key': [1.0, 2.0], 'v1': [10, 20], 'v2': [np.nan, 200.0]}, columns=['key', 'v1', 'v2'])
        tm.assert_frame_equal(result, expected)

    def test_merge_on_ints_floats_warning(self) -> None:
        A: DataFrame = DataFrame({'X': [1, 2, 3]})
        B: DataFrame = DataFrame({'Y': [1.1, 2.5, 3.0]})
        expected: DataFrame = DataFrame({'X': [3], 'Y': [3.0]})
        msg: str = 'the float values are not equal to their int representation'
        with tm.assert_produces_warning(UserWarning, match=msg):
            result: DataFrame = A.merge(B, left_on='X', right_on='Y')
            tm.assert_frame_equal(result, expected)
        with tm.assert_produces_warning(UserWarning, match=msg):
            result = B.merge(A, left_on='Y', right_on='X')
            tm.assert_frame_equal(result, expected[['Y', 'X']])
        B = DataFrame({'Y': [np.nan, np.nan, 3.0]})
        with tm.assert_produces_warning(None):
            result = B.merge(A, left_on='Y', right_on='X')
            tm.assert_frame_equal(result, expected[['Y', 'X']])

    def test_merge_incompat_infer_boolean_object(self) -> None:
        df1: DataFrame = DataFrame({'key': Series([True, False], dtype=object)})
        df2: DataFrame = DataFrame({'key': [True, False]})
        expected: DataFrame = DataFrame({'key': [True, False]}, dtype=object)
        result: DataFrame = merge(df1, df2, on='key')
        tm.assert_frame_equal(result, expected)
        result = merge(df2, df1, on='key')
        tm.assert_frame_equal(result, expected)

    def test_merge_incompat_infer_boolean_object_with_missing(self) -> None:
        df1: DataFrame = DataFrame({'key': Series([True, False, np.nan], dtype=object)})
        df2: DataFrame = DataFrame({'key': [True, False]})
        expected: DataFrame = DataFrame({'key': [True, False]}, dtype=object)
        result: DataFrame = merge(df1, df2, on='key')
        tm.assert_frame_equal(result, expected)
        result = merge(df2, df1, on='key')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('df1_vals, df2_vals', [
        ([0, 1, 2], Series(['a', 'b', 'a']).astype('category')),
        ([0.0, 1.0, 2.0], Series(['a', 'b', 'a']).astype('category')),
        ([0, 1], Series([False, True], dtype=object)),
        ([0, 1], Series([False, True], dtype=bool))
    ])
    def test_merge_incompat_dtypes_are_ok(self, df1_vals: Any, df2_vals: Any) -> None:
        df1: DataFrame = DataFrame({'A': df1_vals})
        df2: DataFrame = DataFrame({'A': df2_vals})
        result: DataFrame = merge(df1, df2, on=['A'])
        assert is_object_dtype(result.A.dtype)
        result = merge(df2, df1, on=['A'])
        assert is_object_dtype(result.A.dtype) or is_string_dtype(result.A.dtype)

    @pytest.mark.parametrize('df1_vals, df2_vals', [
        (Series([1, 2], dtype='uint64'), ['a', 'b', 'c']),
        (Series([1, 2], dtype='int32'), ['a', 'b', 'c']),
        ([0, 1, 2], ['0', '1', '2']),
        ([0.0, 1.0, 2.0], ['0', '1', '2']),
        (pd.date_range('1/1/2011', periods=2, freq='D'), ['2011-01-01', '2011-01-02']),
        (pd.date_range('1/1/2011', periods=2, freq='D'), [0, 1]),
        (pd.date_range('1/1/2011', periods=2, freq='D'), [0.0, 1.0]),
        (pd.date_range('20130101', periods=3), pd.date_range('20130101', periods=3, tz='US/Eastern'))
    ])
    def test_merge_incompat_dtypes_error(self, df1_vals: Any, df2_vals: Any) -> None:
        df1: DataFrame = DataFrame({'A': df1_vals})
        df2: DataFrame = DataFrame({'A': df2_vals})
        msg: str = f"You are trying to merge on {df1['A'].dtype} and {df2['A'].dtype} columns for key 'A'. If you wish to proceed you should use pd.concat"
        msg = re.escape(msg)
        with pytest.raises(ValueError, match=msg):
            merge(df1, df2, on=['A'])
        msg = f"You are trying to merge on {df2['A'].dtype} and {df1['A'].dtype} columns for key 'A'. If you wish to proceed you should use pd.concat"
        msg = re.escape(msg)
        with pytest.raises(ValueError, match=msg):
            merge(df2, df1, on=['A'])
        if len(df1_vals) == len(df2_vals):
            df3: DataFrame = DataFrame({'A': df2_vals, 'B': df1_vals, 'C': df1_vals})
            df4: DataFrame = DataFrame({'A': df2_vals, 'B': df2_vals, 'C': df2_vals})
            msg = f"You are trying to merge on {df3['B'].dtype} and {df4['B'].dtype} columns for key 'B'. If you wish to proceed you should use pd.concat"
            msg = re.escape(msg)
            with pytest.raises(ValueError, match=msg):
                merge(df3, df4)
            msg = f"You are trying to merge on {df3['C'].dtype} and {df4['C'].dtype} columns for key 'C'. If you wish to proceed you should use pd.concat"
            msg = re.escape(msg)
            with pytest.raises(ValueError, match=msg):
                merge(df3, df4, on=['A', 'C'])

    @pytest.mark.parametrize('expected_data, how', [
        ([1, 2], 'outer'),
        ([], 'inner'),
        ([2], 'right'),
        ([1], 'left')
    ])
    def test_merge_EA_dtype(self, any_numeric_ea_dtype: Any, how: str, expected_data: List[Any]) -> None:
        d1: DataFrame = DataFrame([(1,)], columns=['id'], dtype=any_numeric_ea_dtype)
        d2: DataFrame = DataFrame([(2,)], columns=['id'], dtype=any_numeric_ea_dtype)
        result: DataFrame = merge(d1, d2, how=how)
        exp_index: RangeIndex = RangeIndex(len(expected_data))
        expected: DataFrame = DataFrame(expected_data, index=exp_index, columns=['id'], dtype=any_numeric_ea_dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('how, expected_data', [
        (['outer', ['a', 'b']], ('outer', ['a', 'b'])),
        (['inner', []], ('inner', [])),
        (['right', ['b']], ('right', ['b'])),
        (['left', ['a']], ('left', ['a']))
    ])
    @pytest.mark.parametrize('series_of_dtype', [
        Series(['a'], dtype='object'),
        Series(['a'], dtype='string')
    ])
    def test_merge_string_dtype(self, how: str, expected_data: List[Any], any_string_dtype: Any) -> None:
        d1: DataFrame = DataFrame([('a',)], columns=['id'], dtype=any_string_dtype)
        d2: DataFrame = DataFrame([('b',)], columns=['id'], dtype=any_string_dtype)
        result: DataFrame = merge(d1, d2, how=how)
        exp_idx: RangeIndex = RangeIndex(len(expected_data))
        expected: DataFrame = DataFrame(expected_data, index=exp_idx, columns=['id'], dtype=any_string_dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('how, expected_data', [
        ('inner', [[True, 1, 4], [False, 5, 3]]),
        ('outer', [[False, 5, 3], [True, 1, 4]]),
        ('left', [[True, 1, 4], [False, 5, 3]]),
        ('right', [[False, 5, 3], [True, 1, 4]])
    ])
    def test_merge_bool_dtype(self, how: str, expected_data: List[List[Any]]) -> None:
        df1: DataFrame = DataFrame({'A': [True, False], 'B': [1, 5]})
        df2: DataFrame = DataFrame({'A': [False, True], 'C': [3, 4]})
        result: DataFrame = merge(df1, df2, how=how)
        expected: DataFrame = DataFrame(expected_data, columns=['A', 'B', 'C'])
        tm.assert_frame_equal(result, expected)

    def test_merge_ea_with_string(self, join_type: str, string_dtype: Any) -> None:
        df1: DataFrame = DataFrame(data={('lvl0', 'lvl1-a'): ['1', '2', '3', '4', None],
                                          ('lvl0', 'lvl1-b'): ['4', '5', '6', '7', '8']}, dtype=pd.StringDtype())
        df1_copy: DataFrame = df1.copy()
        df2: DataFrame = DataFrame(data={('lvl0', 'lvl1-a'): ['1', '2', '3', pd.NA, '5'],
                                          ('lvl0', 'lvl1-c'): ['7', '8', '9', pd.NA, '11']}, dtype=string_dtype)
        df2_copy: DataFrame = df2.copy()
        merged: DataFrame = merge(left=df1, right=df2, on=[('lvl0', 'lvl1-a')], how=join_type)
        tm.assert_frame_equal(df1, df1_copy)
        tm.assert_frame_equal(df2, df2_copy)
        expected: Series = Series([np.dtype('O'), pd.StringDtype(), np.dtype('O')], index=MultiIndex.from_tuples([('lvl0', 'lvl1-a'), ('lvl0', 'lvl1-b'), ('lvl0', 'lvl1-c')]))
        tm.assert_series_equal(merged.dtypes.sort_index(), expected.sort_index())

    @pytest.mark.parametrize('left_empty, how, exp', [
        (False, 'left', 'left'),
        (False, 'right', 'empty'),
        (False, 'inner', 'empty'),
        (False, 'outer', 'left'),
        (False, 'cross', 'empty_cross'),
        (True, 'left', 'empty'),
        (True, 'right', 'right'),
        (True, 'inner', 'empty'),
        (True, 'outer', 'right'),
        (True, 'cross', 'empty_cross')
    ])
    def test_merge_empty(self, left_empty: bool, how: str, exp: str) -> None:
        left_: DataFrame = DataFrame({'A': [2, 1], 'B': [3, 4]})
        right_: DataFrame = DataFrame({'A': [1], 'C': [5]}, dtype='int64')
        if left_empty:
            left_ = left_.head(0)
        else:
            right_ = right_.head(0)
        result: DataFrame = left_.merge(right_, how=how)
        if exp == 'left':
            expected: DataFrame = DataFrame({'A': [2, 1], 'B': [3, 4], 'C': [np.nan, np.nan]})
        elif exp == 'right':
            expected = DataFrame({'A': [1], 'B': [np.nan], 'C': [5]})
        elif exp == 'empty':
            expected = DataFrame(columns=['A', 'B', 'C'], dtype='int64')
        elif exp == 'empty_cross':
            expected = DataFrame(columns=['A_x', 'B', 'A_y', 'C'], dtype='int64')
        if how == 'outer':
            expected = expected.sort_values('A', ignore_index=True)
        tm.assert_frame_equal(result, expected)

    def test_merge_with_uintc_columns(self) -> None:
        df1: DataFrame = DataFrame({'a': ['foo', 'bar'], 'b': np.array([1, 2], dtype=np.uintc)})
        df2: DataFrame = DataFrame({'a': ['foo', 'baz'], 'b': np.array([3, 4], dtype=np.uintc)})
        result: DataFrame = df1.merge(df2, how='outer')
        expected: DataFrame = DataFrame({'a': ['bar', 'baz', 'foo', 'foo'], 'b': np.array([2, 4, 1, 3], dtype=np.uintc)})
        tm.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_merge_with_intc_columns(self) -> None:
        df1: DataFrame = DataFrame({'a': ['foo', 'bar'], 'b': np.array([1, 2], dtype=np.intc)})
        df2: DataFrame = DataFrame({'a': ['foo', 'baz'], 'b': np.array([3, 4], dtype=np.intc)})
        result: DataFrame = df1.merge(df2, how='outer')
        expected: DataFrame = DataFrame({'a': ['bar', 'baz', 'foo', 'foo'], 'b': np.array([2, 4, 1, 3], dtype=np.intc)})
        tm.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_merge_intc_non_monotonic(self) -> None:
        df: DataFrame = DataFrame({'join_key': Series([0, 2, 1], dtype=np.intc)})
        df_details: DataFrame = DataFrame({'join_key': Series([0, 1, 2], dtype=np.intc), 'value': ['a', 'b', 'c']})
        merged: DataFrame = df.merge(df_details, on='join_key', how='left')
        expected: DataFrame = DataFrame({'join_key': np.array([0, 2, 1], dtype=np.intc), 'value': ['a', 'c', 'b']})
        tm.assert_frame_equal(merged.reset_index(drop=True), expected)

# Fixture for TestMergeCategorical
@pytest.fixture
def left() -> DataFrame:
    return DataFrame({
        'X': Series(np.random.default_rng(2).choice(['foo', 'bar'], size=(10,))).astype(CategoricalDtype(['foo', 'bar'])),
        'Y': np.random.default_rng(2).choice(['one', 'two', 'three'], size=(10,))
    })

@pytest.fixture
def right() -> DataFrame:
    return DataFrame({
        'X': Series(['foo', 'bar']).astype(CategoricalDtype(['foo', 'bar'])),
        'Z': [1, 2]
    })

class TestMergeCategorical:
    def test_identical(self, left: DataFrame, using_infer_string: bool) -> None:
        merged: DataFrame = merge(left, left, on='X')
        result: Any = merged.dtypes.sort_index()
        dtype: Any = np.dtype('O') if not using_infer_string else 'str'
        expected: Series = Series([CategoricalDtype(categories=['foo', 'bar']), dtype, dtype], index=['X', 'Y_x', 'Y_y'])
        tm.assert_series_equal(result, expected)

    def test_basic(self, left: DataFrame, right: DataFrame, using_infer_string: bool) -> None:
        merged: DataFrame = merge(left, right, on='X')
        result: Any = merged.dtypes.sort_index()
        dtype: Any = np.dtype('O') if not using_infer_string else 'str'
        expected: Series = Series([CategoricalDtype(categories=['foo', 'bar']), dtype, np.dtype('int64')], index=['X', 'Y', 'Z'])
        tm.assert_series_equal(result, expected)

    def test_merge_categorical(self) -> None:
        right_: DataFrame = DataFrame({'c': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}, 'd': {0: 'null', 1: 'null', 2: 'null', 3: 'null', 4: 'null'}})
        left_: DataFrame = DataFrame({'a': {0: 'f', 1: 'f', 2: 'f', 3: 'f', 4: 'f'}, 'b': {0: 'g', 1: 'g', 2: 'g', 3: 'g', 4: 'g'}})
        df: DataFrame = merge(left_, right_, how='left', left_on='b', right_on='c')
        expected: DataFrame = df.copy()
        cright: DataFrame = right_.copy()
        cright['d'] = cright['d'].astype('category')
        result: DataFrame = merge(left_, cright, how='left', left_on='b', right_on='c')
        expected['d'] = expected['d'].astype(CategoricalDtype(['null']))
        tm.assert_frame_equal(result, expected)
        cleft: DataFrame = left_.copy()
        cleft['b'] = cleft['b'].astype('category')
        result = merge(cleft, cright, how='left', left_on='b', right_on='c')
        tm.assert_frame_equal(result, expected)
        cright = right_.copy()
        cright['d'] = cright['d'].astype('category')
        cleft = left_.copy()
        cleft['b'] = cleft['b'].astype('category')
        result = merge(cleft, cright, how='left', left_on='b', right_on='c')
        tm.assert_frame_equal(result, expected)

    def tests_merge_categorical_unordered_equal(self) -> None:
        df1: DataFrame = DataFrame({
            'Foo': Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']),
            'Left': ['A0', 'B0', 'C0']
        })
        df2: DataFrame = DataFrame({
            'Foo': Categorical(['C', 'B', 'A'], categories=['C', 'B', 'A']),
            'Right': ['C1', 'B1', 'A1']
        })
        result: DataFrame = merge(df1, df2, on=['Foo'])
        expected: DataFrame = DataFrame({
            'Foo': Categorical(['A', 'B', 'C']),
            'Left': ['A0', 'B0', 'C0'],
            'Right': ['A1', 'B1', 'C1']
        })
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('ordered', [True, False])
    def test_multiindex_merge_with_unordered_categoricalindex(self, ordered: bool) -> None:
        pcat: CategoricalDtype = CategoricalDtype(categories=['P2', 'P1'], ordered=ordered)
        df1: DataFrame = DataFrame({'id': ['C', 'C', 'D'], 'p': Categorical(['P2', 'P1', 'P2'], dtype=pcat), 'a': [0, 1, 2]}).set_index(['id', 'p'])
        df2: DataFrame = DataFrame({'id': ['A', 'C', 'C'], 'p': Categorical(['P2', 'P2', 'P1'], dtype=pcat), 'd1': [10, 11, 12]}).set_index(['id', 'p'])
        result: DataFrame = merge(df1, df2, how='left', left_index=True, right_index=True)
        expected: DataFrame = DataFrame({'id': ['C', 'C', 'D'], 'p': Categorical(['P2', 'P1', 'P2'], dtype=pcat), 'a': [0, 1, 2], 'd1': [11.0, 12.0, np.nan]}).set_index(['id', 'p'])
        tm.assert_frame_equal(result, expected)

    def test_other_columns(self, left: DataFrame, right: DataFrame, using_infer_string: bool) -> None:
        right_ = right.assign(Z=right.Z.astype('category'))
        merged: DataFrame = merge(left, right_, on='X')
        result: Any = merged.dtypes.sort_index()
        dtype: Any = np.dtype('O') if not using_infer_string else 'str'
        expected: Series = Series([CategoricalDtype(categories=['foo', 'bar']), dtype, CategoricalDtype(categories=[1, 2])], index=['X', 'Y', 'Z'])
        tm.assert_series_equal(result, expected)
        assert left.X.values._categories_match_up_to_permutation(merged.X.values)
        assert right_.Z.values._categories_match_up_to_permutation(merged.Z.values)

    @pytest.mark.parametrize('change', [
        lambda x: x,
        lambda x: x.astype(CategoricalDtype(['foo', 'bar', 'bah'])),
        lambda x: x.astype(CategoricalDtype(ordered=True))
    ])
    def test_dtype_on_merged_different(self, change: Any, join_type: str, left: DataFrame, right: DataFrame, using_infer_string: bool) -> None:
        X: Any = change(right.X.astype('object'))
        right_ = right.assign(X=X)
        assert isinstance(left.X.values.dtype, CategoricalDtype)
        merged: DataFrame = merge(left, right_, on='X', how=join_type)
        result: Any = merged.dtypes.sort_index()
        dtype: Any = np.dtype('O') if not using_infer_string else 'str'
        expected: Series = Series([dtype, dtype, np.dtype('int64')], index=['X', 'Y', 'Z'])
        tm.assert_series_equal(result, expected)

    def test_self_join_multiple_categories(self) -> None:
        m: int = 5
        df: DataFrame = DataFrame({
            'a': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'] * m,
            'b': ['t', 'w', 'x', 'y', 'z'] * 2 * m,
            'c': [letter for each in ['m', 'n', 'u', 'p', 'o'] for letter in [each] * 2 * m],
            'd': [letter for each in ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj'] for letter in [each] * m]
        })
        df = df.apply(lambda x: x.astype('category'))
        result: DataFrame = merge(df, df, on=list(df.columns))
        tm.assert_frame_equal(result, df)

    def test_dtype_on_categorical_dates(self) -> None:
        df: DataFrame = DataFrame([[date(2001, 1, 1), 1.1], [date(2001, 1, 2), 1.3]], columns=['date', 'num2'])
        df['date'] = df['date'].astype('category')
        df2: DataFrame = DataFrame([[date(2001, 1, 1), 1.3], [date(2001, 1, 3), 1.4]], columns=['date', 'num4'])
        df2['date'] = df2['date'].astype('category')
        expected_outer: DataFrame = DataFrame([
            [pd.Timestamp('2001-01-01').date(), 1.1, 1.3],
            [pd.Timestamp('2001-01-02').date(), 1.3, np.nan],
            [pd.Timestamp('2001-01-03').date(), np.nan, 1.4]
        ], columns=['date', 'num2', 'num4'])
        result_outer: DataFrame = merge(df, df2, how='outer', on=['date'])
        tm.assert_frame_equal(result_outer, expected_outer)
        expected_inner: DataFrame = DataFrame([[pd.Timestamp('2001-01-01').date(), 1.1, 1.3]], columns=['date', 'num2', 'num4'])
        result_inner: DataFrame = merge(df, df2, how='inner', on=['date'])
        tm.assert_frame_equal(result_inner, expected_inner)

    @pytest.mark.parametrize('ordered', [True, False])
    @pytest.mark.parametrize('category_column,categories,expected_categories', [
        ([False, True, True, False], [True, False], [True, False]),
        ([2, 1, 1, 2], [1, 2], [1, 2]),
        (['False', 'True', 'True', 'False'], ['True', 'False'], ['True', 'False'])
    ])
    def test_merging_with_bool_or_int_cateorical_column(self, category_column: List[Any], categories: List[Any], expected_categories: List[Any], ordered: bool) -> None:
        df1: DataFrame = DataFrame({'id': [1, 2, 3, 4], 'cat': category_column})
        df1['cat'] = df1['cat'].astype(CategoricalDtype(categories, ordered=ordered))
        df2: DataFrame = DataFrame({'id': [2, 4], 'num': [1, 9]})
        result: DataFrame = df1.merge(df2)
        expected: DataFrame = DataFrame({'id': [2, 4], 'cat': expected_categories, 'num': [1, 9]})
        expected['cat'] = expected['cat'].astype(CategoricalDtype(categories, ordered=ordered))
        tm.assert_frame_equal(expected, result)

    def test_merge_on_int_array(self) -> None:
        df: DataFrame = DataFrame({'A': Series([1, 2, np.nan], dtype='Int64'), 'B': 1})
        result: DataFrame = merge(df, df, on='A')
        expected: DataFrame = DataFrame({'A': Series([1, 2, np.nan], dtype='Int64'), 'B_x': 1, 'B_y': 1})
        tm.assert_frame_equal(result, expected)

class TestMergeOnIndexes:
    @pytest.mark.parametrize('how, sort, expected', [
        ('inner', False, DataFrame({'a': [20, 10], 'b': [200, 100]}, index=[2, 1])),
        ('inner', True, DataFrame({'a': [10, 20], 'b': [100, 200]}, index=[1, 2])),
        ('left', False, DataFrame({'a': [20, 10, 0], 'b': [200, 100, np.nan]}, index=[2, 1, 0])),
        ('left', True, DataFrame({'a': [0, 10, 20], 'b': [np.nan, 100, 200]}, index=[0, 1, 2])),
        ('right', False, DataFrame({'a': [np.nan, 10, 20], 'b': [300, 100, 200]}, index=[3, 1, 2])),
        ('right', True, DataFrame({'a': [10, 20, np.nan], 'b': [100, 200, 300]}, index=[1, 2, 3])),
        ('outer', False, DataFrame({'a': [0, 10, 20, np.nan], 'b': [np.nan, 100, 200, 300]}, index=[0, 1, 2, 3])),
        ('outer', True, DataFrame({'a': [0, 10, 20, np.nan], 'b': [np.nan, 100, 200, 300]}, index=[0, 1, 2, 3]))
    ])
    def test_merge_on_indexes(self, how: str, sort: bool, expected: DataFrame) -> None:
        left_df: DataFrame = DataFrame({'a': [20, 10, 0]}, index=[2, 1, 0])
        right_df: DataFrame = DataFrame({'b': [300, 100, 200]}, index=[3, 1, 2])
        result: DataFrame = merge(left_df, right_df, left_index=True, right_index=True, how=how, sort=sort)
        tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('index', [
    Index([1, 2, 4], dtype=dtyp, name='index_col') for dtyp in tm.ALL_REAL_NUMPY_DTYPES
] + [
    CategoricalIndex(['A', 'B', 'C'], categories=['A', 'B', 'C'], name='index_col'),
    RangeIndex(start=0, stop=3, name='index_col'),
    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'], name='index_col')
], ids=lambda x: f'{type(x).__name__}[{x.dtype}]')
def test_merge_index_types(index: Index) -> None:
    left_: DataFrame = DataFrame({'left_data': [1, 2, 3]}, index=index)
    right_: DataFrame = DataFrame({'right_data': [1.0, 2.0, 3.0]}, index=index)
    result: DataFrame = left_.merge(right_, on=['index_col'])
    expected: DataFrame = DataFrame({'left_data': [1, 2, 3], 'right_data': [1.0, 2.0, 3.0]}, index=index)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('on,left_on,right_on,left_index,right_index,nm', [
    (['outer', 'inner'], None, None, False, False, 'B'),
    (None, None, None, True, True, 'B'),
    (None, ['outer', 'inner'], None, False, True, 'B'),
    (None, None, ['outer', 'inner'], True, False, 'B'),
    (['outer', 'inner'], None, None, False, False, None),
    (None, None, None, True, True, None),
    (None, ['outer', 'inner'], None, False, True, None),
    (None, None, ['outer', 'inner'], True, False, None)
])
def test_merge_series(on: Optional[List[Any]], left_on: Optional[Union[str, List[Any]]], right_on: Optional[Union[str, List[Any]]], left_index: bool, right_index: bool, nm: Optional[Any]) -> None:
    a: DataFrame = DataFrame({'A': [1, 2, 3, 4]}, index=MultiIndex.from_product([['a', 'b'], [0, 1]], names=['outer', 'inner']))
    b: Series = Series([1, 2, 3, 4], index=MultiIndex.from_product([['a', 'b'], [1, 2]], names=['outer', 'inner']), name=nm)
    expected: DataFrame = DataFrame({'A': [2, 4], 'B': [1, 3]}, index=MultiIndex.from_product([['a', 'b'], [1]], names=['outer', 'inner']))
    if nm is not None:
        result: DataFrame = merge(a, b, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index)
        tm.assert_frame_equal(result, expected)
    else:
        msg: str = 'Cannot merge a Series without a name'
        with pytest.raises(ValueError, match=msg):
            _ = merge(a, b, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index)

@pytest.mark.parametrize('func', ['merge', 'merge_asof'])
@pytest.mark.parametrize(('kwargs', 'err_msg'), [
    ({'left_on': 'a', 'left_index': True}, ['left_on', 'left_index']),
    ({'right_on': 'a', 'right_index': True}, ['right_on', 'right_index'])
])
def test_merge_join_cols_error_reporting_duplicates(func: str, kwargs: dict, err_msg: List[str]) -> None:
    left_: DataFrame = DataFrame({'a': [1, 2], 'b': [3, 4]})
    right_: DataFrame = DataFrame({'a': [1, 1], 'c': [5, 6]})
    msg: str = f'Can only pass argument "{err_msg[0]}" OR "{err_msg[1]}" not both\\.'
    with pytest.raises(MergeError, match=msg):
        getattr(pd, func)(left_, right_, **kwargs)

@pytest.mark.parametrize('func', ['merge', 'merge_asof'])
@pytest.mark.parametrize(('kwargs', 'err_msg'), [
    ({'left_on': 'a'}, ['right_on', 'right_index']),
    ({'right_on': 'a'}, ['left_on', 'left_index'])
])
def test_merge_join_cols_error_reporting_missing(func: str, kwargs: dict, err_msg: List[str]) -> None:
    left_: DataFrame = DataFrame({'a': [1, 2], 'b': [3, 4]})
    right_: DataFrame = DataFrame({'a': [1, 1], 'c': [5, 6]})
    msg: str = f'Must pass "{err_msg[0]}" OR "{err_msg[1]}"\\.'
    with pytest.raises(MergeError, match=msg):
        getattr(pd, func)(left_, right_, **kwargs)

@pytest.mark.parametrize('func', ['merge', 'merge_asof'])
@pytest.mark.parametrize('kwargs', [{'right_index': True}, {'left_index': True}])
def test_merge_join_cols_error_reporting_on_and_index(func: str, kwargs: dict) -> None:
    left_: DataFrame = DataFrame({'a': [1, 2], 'b': [3, 4]})
    right_: DataFrame = DataFrame({'a': [1, 1], 'c': [5, 6]})
    msg: str = 'Can only pass argument "on" OR "left_index" and "right_index", not a combination of both\\.'
    with pytest.raises(MergeError, match=msg):
        getattr(pd, func)(left_, right_, on='a', **kwargs)

def test_merge_right_left_index() -> None:
    left_: DataFrame = DataFrame({'x': [1, 1], 'z': ['foo', 'foo']})
    right_: DataFrame = DataFrame({'x': [1, 1], 'z': ['foo', 'foo']})
    result: DataFrame = merge(left_, right_, how='right', left_index=True, right_on='x')
    expected: DataFrame = DataFrame({'x': [1, 1], 'x_x': [1, 1], 'z_x': ['foo', 'foo'], 'x_y': [1, 1], 'z_y': ['foo', 'foo']})
    tm.assert_frame_equal(result, expected)

def test_merge_result_empty_index_and_on() -> None:
    df1: DataFrame = DataFrame({'a': [1], 'b': [2]}).set_index(['a', 'b'])
    df2: DataFrame = DataFrame({'b': [1]}).set_index(['b'])
    expected: DataFrame = DataFrame({'a': [], 'b': []}, dtype=np.int64).set_index(['a', 'b'])
    result: DataFrame = merge(df1, df2, left_on=['b'], right_index=True)
    tm.assert_frame_equal(result, expected)
    result = merge(df2, df1, left_index=True, right_on=['b'])
    tm.assert_frame_equal(result, expected)

def test_merge_suffixes_produce_dup_columns_raises() -> None:
    left_: DataFrame = DataFrame({'a': [1, 2, 3], 'b': 1, 'b_x': 2})
    right_: DataFrame = DataFrame({'a': [1, 2, 3], 'b': 2})
    with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
        merge(left_, right_, on='a')
    with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
        merge(right_, left_, on='a', suffixes=('_y', '_x'))

def test_merge_duplicate_columns_with_suffix_no_warning() -> None:
    left_: DataFrame = DataFrame([[1, 1, 1], [2, 2, 2]], columns=['a', 'b', 'b'])
    right_: DataFrame = DataFrame({'a': [1, 3], 'b': 2})
    result: DataFrame = merge(left_, right_, on='a')
    expected: DataFrame = DataFrame([[1, 1, 1, 2]], columns=['a', 'b_x', 'b_x', 'b_y'])
    tm.assert_frame_equal(result, expected)

def test_merge_duplicate_columns_with_suffix_causing_another_duplicate_raises() -> None:
    left_: DataFrame = DataFrame([[1, 1, 1, 1], [2, 2, 2, 2]], columns=['a', 'b', 'b', 'b_x'])
    right_: DataFrame = DataFrame({'a': [1, 3], 'b': 2})
    with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
        merge(left_, right_, on='a')

def test_merge_string_float_column_result() -> None:
    df1: DataFrame = DataFrame([[1, 2], [3, 4]], columns=Index(['a', 114.0]))
    df2: DataFrame = DataFrame([[9, 10], [11, 12]], columns=['x', 'y'])
    result: DataFrame = merge(df2, df1, how='inner', left_index=True, right_index=True)
    expected: DataFrame = DataFrame([[9, 10, 1, 2], [11, 12, 3, 4]], columns=Index(['x', 'y', 'a', 114.0]))
    tm.assert_frame_equal(result, expected)

def test_mergeerror_on_left_index_mismatched_dtypes() -> None:
    df_1: DataFrame = DataFrame(data=['X'], columns=['C'], index=[22])
    df_2: DataFrame = DataFrame(data=['X'], columns=['C'], index=[999])
    with pytest.raises(MergeError, match='Can only pass argument'):
        merge(df_1, df_2, on=['C'], left_index=True)

def test_merge_on_left_categoricalindex() -> None:
    ci: CategoricalIndex = CategoricalIndex(range(3))
    right_: DataFrame = DataFrame({'A': ci, 'B': range(3)})
    left_: DataFrame = DataFrame({'C': range(3, 6)})
    res: DataFrame = merge(left_, right_, left_on=ci, right_on='A')
    expected: DataFrame = merge(left_, right_, left_on=ci._data, right_on='A')
    tm.assert_frame_equal(res, expected)

@pytest.mark.parametrize('dtype', [None, 'Int64'])
def test_merge_outer_with_NaN(dtype: Optional[str]) -> None:
    left_: DataFrame = DataFrame({'key': [1, 2], 'col1': [1, 2]}, dtype=dtype)
    right_: DataFrame = DataFrame({'key': [np.nan, np.nan], 'col2': [3, 4]}, dtype=dtype)
    result: DataFrame = merge(left_, right_, on='key', how='outer')
    expected: DataFrame = DataFrame({
        'key': [1, 2, np.nan, np.nan],
        'col1': [1, 2, np.nan, np.nan],
        'col2': [np.nan, np.nan, 3, 4]
    }, dtype=dtype)
    tm.assert_frame_equal(result, expected)
    result = merge(right_, left_, on='key', how='outer')
    expected = DataFrame({
        'key': [1, 2, np.nan, np.nan],
        'col2': [np.nan, np.nan, 3, 4],
        'col1': [1, 2, np.nan, np.nan]
    }, dtype=dtype)
    tm.assert_frame_equal(result, expected)

def test_merge_different_index_names() -> None:
    left_: DataFrame = DataFrame({'a': [1]}, index=Index([1], name='c'))
    right_: DataFrame = DataFrame({'a': [1]}, index=Index([1], name='d'))
    result: DataFrame = merge(left_, right_, left_on='c', right_on='d')
    expected: DataFrame = DataFrame({'a_x': [1], 'a_y': 1})
    tm.assert_frame_equal(result, expected)

def test_merge_ea(any_numeric_ea_dtype: Any, join_type: str) -> None:
    left_: DataFrame = DataFrame({'a': [1, 2, 3], 'b': 1}, dtype=any_numeric_ea_dtype)
    right_: DataFrame = DataFrame({'a': [1, 2, 3], 'c': 2}, dtype=any_numeric_ea_dtype)
    result: DataFrame = left_.merge(right_, how=join_type)
    expected: DataFrame = DataFrame({'a': [1, 2, 3], 'b': 1, 'c': 2}, dtype=any_numeric_ea_dtype)
    tm.assert_frame_equal(result, expected)

def test_merge_ea_and_non_ea(any_numeric_ea_dtype: Any, join_type: str) -> None:
    left_: DataFrame = DataFrame({'a': [1, 2, 3], 'b': 1}, dtype=any_numeric_ea_dtype)
    right_: DataFrame = DataFrame({'a': [1, 2, 3], 'c': 2}, dtype=any_numeric_ea_dtype.lower())
    result: DataFrame = left_.merge(right_, how=join_type)
    expected: DataFrame = DataFrame({
        'a': Series([1, 2, 3], dtype=any_numeric_ea_dtype),
        'b': Series([1, 1, 1], dtype=any_numeric_ea_dtype),
        'c': Series([2, 2, 2], dtype=any_numeric_ea_dtype.lower())
    })
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('dtype', ['int64', 'int64[pyarrow]'])
def test_merge_arrow_and_numpy_dtypes(dtype: str) -> None:
    pytest.importorskip('pyarrow')
    df: DataFrame = DataFrame({'a': [1, 2]}, dtype=dtype)
    df2: DataFrame = DataFrame({'a': [1, 2]}, dtype='int64[pyarrow]')
    result: DataFrame = df.merge(df2)
    expected: DataFrame = df.copy()
    tm.assert_frame_equal(result, expected)
    result = df2.merge(df)
    expected = df2.copy()
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('tz', [None, 'America/Chicago'])
def test_merge_datetime_different_resolution(tz: Optional[str], join_type: str) -> None:
    vals: List[pd.Timestamp] = [pd.Timestamp(2023, 5, 12, tz=tz), pd.Timestamp(2023, 5, 13, tz=tz), pd.Timestamp(2023, 5, 14, tz=tz)]
    df1: DataFrame = DataFrame({'t': vals[:2], 'a': [1.0, 2.0]})
    df1['t'] = df1['t'].dt.as_unit('ns')
    df2: DataFrame = DataFrame({'t': vals[1:], 'b': [1.0, 2.0]})
    df2['t'] = df2['t'].dt.as_unit('s')
    expected: DataFrame = DataFrame({'t': vals, 'a': [1.0, 2.0, np.nan], 'b': [np.nan, 1.0, 2.0]})
    expected['t'] = expected['t'].dt.as_unit('ns')
    if join_type == 'inner':
        expected = expected.iloc[[1]].reset_index(drop=True)
    elif join_type == 'left':
        expected = expected.iloc[[0, 1]]
    elif join_type == 'right':
        expected = expected.iloc[[1, 2]].reset_index(drop=True)
    result: DataFrame = df1.merge(df2, on='t', how=join_type)
    tm.assert_frame_equal(result, expected)

def test_merge_multiindex_single_level() -> None:
    df: DataFrame = DataFrame({'col': ['A', 'B']})
    df2: DataFrame = DataFrame(data={'b': [100]}, index=MultiIndex.from_tuples([('A',), ('C',)], names=['col']))
    expected: DataFrame = DataFrame({'col': ['A', 'B'], 'b': [100, np.nan]})
    result: DataFrame = df.merge(df2, left_on=['col'], right_index=True, how='left')
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('on_index', [True, False])
@pytest.mark.parametrize('left_unique', [True, False])
@pytest.mark.parametrize('left_monotonic', [True, False])
@pytest.mark.parametrize('right_unique', [True, False])
@pytest.mark.parametrize('right_monotonic', [True, False])
def test_merge_combinations(join_type: str, sort: bool, on_index: bool, left_unique: bool, left_monotonic: bool, right_unique: bool, right_monotonic: bool) -> None:
    how: str = join_type
    left_vals: List[int] = [2, 3]
    if left_unique:
        left_vals.append(4 if left_monotonic else 1)
    else:
        left_vals.append(3 if left_monotonic else 2)
    right_vals: List[int] = [2, 3]
    if right_unique:
        right_vals.append(4 if right_monotonic else 1)
    else:
        right_vals.append(3 if right_monotonic else 2)
    left_df: DataFrame = DataFrame({'key': left_vals})
    right_df: DataFrame = DataFrame({'key': right_vals})
    if on_index:
        left_df = left_df.set_index('key')
        right_df = right_df.set_index('key')
        on_kwargs = {'left_index': True, 'right_index': True}
    else:
        on_kwargs = {'on': 'key'}
    result: DataFrame = merge(left_df, right_df, how=how, sort=sort, **on_kwargs)
    if on_index:
        left_df = left_df.reset_index()
        right_df = right_df.reset_index()
    if how in ['left', 'right', 'inner']:
        if how in ['left', 'inner']:
            expected_df, other, other_unique = (left_df, right_df, right_unique)
        else:
            expected_df, other, other_unique = (right_df, left_df, left_unique)
        if how == 'inner':
            keep_values = set(left_df['key'].values).intersection(right_df['key'].values)
            keep_mask = expected_df['key'].isin(keep_values)
            expected_df = expected_df[keep_mask]
        if sort:
            expected_df = expected_df.sort_values('key')
        if not other_unique:
            other_value_counts = other['key'].value_counts()
            repeats = other_value_counts.reindex(expected_df['key'].values, fill_value=1)
            repeats = repeats.astype(np.intp)
            expected_df = expected_df['key'].repeat(repeats.values)
            expected_df = expected_df.to_frame()
    elif how == 'outer':
        left_counts = left_df['key'].value_counts()
        right_counts = right_df['key'].value_counts()
        expected_counts = left_counts.mul(right_counts, fill_value=1)
        expected_counts = expected_counts.astype(np.intp)
        expected_arr = expected_counts.index.values.repeat(expected_counts.values)
        expected_df = DataFrame({'key': expected_arr})
        expected_df = expected_df.sort_values('key')
    if on_index:
        expected_df = expected_df.set_index('key')
    else:
        expected_df = expected_df.reset_index(drop=True)
    tm.assert_frame_equal(result, expected_df)

def test_merge_ea_int_and_float_numpy() -> None:
    df1: DataFrame = DataFrame([1.0, np.nan], dtype=pd.Int64Dtype())
    df2: DataFrame = DataFrame([1.5])
    expected: DataFrame = DataFrame(columns=[0], dtype='Int64')
    with tm.assert_produces_warning(UserWarning, match='You are merging'):
        result: DataFrame = df1.merge(df2)
    tm.assert_frame_equal(result, expected)
    with tm.assert_produces_warning(UserWarning, match='You are merging'):
        result = df2.merge(df1)
    tm.assert_frame_equal(result, expected.astype('float64'))
    df2 = DataFrame([1.0])
    expected = DataFrame([1], columns=[0], dtype='Int64')
    result = df1.merge(df2)
    tm.assert_frame_equal(result, expected)
    result = df2.merge(df1)
    tm.assert_frame_equal(result, expected.astype('float64'))

def test_merge_arrow_string_index(any_string_dtype: Any) -> None:
    pytest.importorskip('pyarrow')
    left_: DataFrame = DataFrame({'a': ['a', 'b']}, dtype=any_string_dtype)
    right_: DataFrame = DataFrame({'b': 1}, index=Index(['a', 'c'], dtype=any_string_dtype))
    result: DataFrame = left_.merge(right_, left_on='a', right_index=True, how='left')
    expected: DataFrame = DataFrame({'a': Series(['a', 'b'], dtype=any_string_dtype), 'b': [1, np.nan]})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('left_empty', [True, False])
@pytest.mark.parametrize('right_empty', [True, False])
def test_merge_empty_frames_column_order(left_empty: bool, right_empty: bool) -> None:
    df1: DataFrame = DataFrame(1, index=[0], columns=['A', 'B'])
    df2: DataFrame = DataFrame(1, index=[0], columns=['A', 'C', 'D'])
    if left_empty:
        df1 = df1.iloc[:0]
    if right_empty:
        df2 = df2.iloc[:0]
    result: DataFrame = merge(df1, df2, on=['A'], how='outer')
    expected: DataFrame = DataFrame(1, index=range(1), columns=['A', 'B', 'C', 'D'])
    if left_empty and right_empty:
        expected = expected.iloc[:0]
    elif left_empty:
        expected['B'] = np.nan
    elif right_empty:
        expected[['C', 'D']] = np.nan
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('how', ['left', 'right', 'inner', 'outer'])
def test_merge_datetime_and_timedelta(how: str) -> None:
    left_: DataFrame = DataFrame({'key': Series([1, None], dtype='datetime64[ns]')})
    right_: DataFrame = DataFrame({'key': Series([1], dtype='timedelta64[ns]')})
    msg: str = f"You are trying to merge on {left_['key'].dtype} and {right_['key'].dtype} columns for key 'key'. If you wish to proceed you should use pd.concat"
    with pytest.raises(ValueError, match=re.escape(msg)):
        left_.merge(right_, on='key', how=how)
    msg = f"You are trying to merge on {right_['key'].dtype} and {left_['key'].dtype} columns for key 'key'. If you wish to proceed you should use pd.concat"
    with pytest.raises(ValueError, match=re.escape(msg)):
        right_.merge(left_, on='key', how=how)

def test_merge_on_all_nan_column() -> None:
    left_: DataFrame = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'z': [4, 5, 6]})
    right_: DataFrame = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'zz': [4, 5, 6]})
    result: DataFrame = left_.merge(right_, on=['x', 'y'], how='outer')
    expected: DataFrame = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'z': [4, 5, 6], 'zz': [4, 5, 6]})
    tm.assert_frame_equal(result, expected)

# Add type annotations to this module's main functions above.

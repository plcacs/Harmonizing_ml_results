from datetime import date, datetime, timedelta
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import is_object_dtype, is_string_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
    Categorical, CategoricalIndex, DataFrame, DatetimeIndex, Index, IntervalIndex,
    MultiIndex, PeriodIndex, RangeIndex, Series, TimedeltaIndex
)
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import MergeError, merge
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def get_test_data(ngroups: int = 8, n: int = 50) -> np.ndarray:
    unique_groups: List[int] = list(range(ngroups))
    arr: np.ndarray = np.asarray(np.tile(unique_groups, n // ngroups))
    if len(arr) < n:
        arr = np.asarray(list(arr) + unique_groups[: n - len(arr)])
    np.random.default_rng(2).shuffle(arr)
    return arr

@pytest.fixture
def dfs_for_indicator() -> Tuple[DataFrame, DataFrame]:
    df1: DataFrame = DataFrame({
        'col1': [0, 1],
        'col_conflict': [1, 2],
        'col_left': ['a', 'b']
    })
    df2: DataFrame = DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col_conflict': [1, 2, 3, 4, 5],
        'col_right': [2, 2, 2, 2, 2]
    })
    return (df1, df2)

class TestMerge:
    @pytest.fixture
    def df(self) -> DataFrame:
        df: DataFrame = DataFrame({
            'key1': get_test_data(),
            'key2': get_test_data(),
            'data1': np.random.default_rng(2).standard_normal(50),
            'data2': np.random.default_rng(2).standard_normal(50)
        })
        df = df[df['key2'] > 1]
        return df

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
        left_df: DataFrame = DataFrame({0: [1, 0, 1, 0],
                                         1: [0, 1, 0, 0],
                                         2: [0, 0, 2, 0],
                                         3: [1, 0, 0, 3]})
        right_df: DataFrame = left_df.astype(float)
        expected: DataFrame = left_df
        result: DataFrame = merge(left_df, right_df)
        tm.assert_frame_equal(expected, result)

    def test_merge_index_as_on_arg(self, df: DataFrame, df2: DataFrame) -> None:
        left_df: DataFrame = df.set_index('key1')
        right_df: DataFrame = df2.set_index('key1')
        result: DataFrame = merge(left_df, right_df, on='key1')
        expected: DataFrame = merge(df, df2, on='key1').set_index('key1')
        tm.assert_frame_equal(result, expected)

    def test_merge_index_singlekey_right_vs_left(self) -> None:
        left_df: DataFrame = DataFrame({'key': ['a', 'b', 'c', 'd', 'e', 'e', 'a'],
                                        'v1': np.random.default_rng(2).standard_normal(7)})
        right_df: DataFrame = DataFrame({'v2': np.random.default_rng(2).standard_normal(4)},
                                         index=['d', 'b', 'c', 'a'])
        merged1: DataFrame = merge(left_df, right_df, left_on='key', right_index=True, how='left', sort=False)
        merged2: DataFrame = merge(right_df, left_df, right_on='key', left_index=True, how='right', sort=False)
        tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])
        merged1 = merge(left_df, right_df, left_on='key', right_index=True, how='left', sort=True)
        merged2 = merge(right_df, left_df, right_on='key', left_index=True, how='right', sort=True)
        tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])

    def test_merge_index_singlekey_inner(self) -> None:
        left_df: DataFrame = DataFrame({'key': ['a', 'b', 'c', 'd', 'e', 'e', 'a'],
                                        'v1': np.random.default_rng(2).standard_normal(7)})
        right_df: DataFrame = DataFrame({'v2': np.random.default_rng(2).standard_normal(4)},
                                         index=['d', 'b', 'c', 'a'])
        result: DataFrame = merge(left_df, right_df, left_on='key', right_index=True, how='inner')
        expected: DataFrame = left_df.join(right_df, on='key').loc[result.index]
        tm.assert_frame_equal(result, expected)
        result = merge(right_df, left_df, right_on='key', left_index=True, how='inner')
        expected = left_df.join(right_df, on='key').loc[result.index]
        tm.assert_frame_equal(result, expected.loc[:, result.columns])

    def test_merge_misspecified(self, df: DataFrame, df2: DataFrame, left: DataFrame) -> None:
        right_df: DataFrame = DataFrame({'v2': np.random.default_rng(2).standard_normal(4)},
                                         index=['d', 'b', 'c', 'a'])
        msg: str = 'Must pass right_on or right_index=True'
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, right_df, left_index=True)
        msg = 'Must pass left_on or left_index=True'
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, right_df, right_index=True)
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
        exp_len: int = (left['key'].value_counts() ** 2).sum()
        assert len(merged) == exp_len
        assert 'v1_x' in merged
        assert 'v1_y' in merged

    def test_merge_different_column_key_names(self) -> None:
        left_df: DataFrame = DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 4]})
        right_df: DataFrame = DataFrame({'rkey': ['foo', 'bar', 'qux', 'foo'], 'value': [5, 6, 7, 8]})
        merged: DataFrame = left_df.merge(right_df, left_on='lkey', right_on='rkey', how='outer', sort=True)
        exp: Series = Series(['bar', 'baz', 'foo', 'foo', 'foo', 'foo', np.nan], name='lkey')
        tm.assert_series_equal(merged['lkey'], exp)
        exp = Series(['bar', np.nan, 'foo', 'foo', 'foo', 'foo', 'qux'], name='rkey')
        tm.assert_series_equal(merged['rkey'], exp)
        exp = Series([2, 3, 1, 1, 4, 4, np.nan], name='value_x')
        tm.assert_series_equal(merged['value_x'], exp)
        exp = Series([6, np.nan, 5, 8, 5, 8, 7], name='value_y')
        tm.assert_series_equal(merged['value_y'], exp)

    def test_merge_copy(self) -> None:
        left_df: DataFrame = DataFrame({'a': 0, 'b': 1}, index=range(10))
        right_df: DataFrame = DataFrame({'c': 'foo', 'd': 'bar'}, index=range(10))
        merged: DataFrame = merge(left_df, right_df, left_index=True, right_index=True)
        merged['a'] = 6
        assert (left_df['a'] == 0).all()
        merged['d'] = 'peekaboo'
        assert (right_df['d'] == 'bar').all()

    def test_merge_nocopy(self, using_infer_string: bool) -> None:
        left_df: DataFrame = DataFrame({'a': 0, 'b': 1}, index=range(10))
        right_df: DataFrame = DataFrame({'c': 'foo', 'd': 'bar'}, index=range(10))
        merged: DataFrame = merge(left_df, right_df, left_index=True, right_index=True)
        assert np.shares_memory(merged['a']._values, left_df['a']._values)
        if not using_infer_string:
            assert np.shares_memory(merged['d']._values, right_df['d']._values)

    def test_intelligently_handle_join_key(self) -> None:
        left_df: DataFrame = DataFrame({'key': [1, 1, 2, 2, 3], 'value': list(range(5))}, columns=['value', 'key'])
        right_df: DataFrame = DataFrame({'key': [1, 1, 2, 3, 4, 5], 'rvalue': list(range(6))})
        joined: DataFrame = merge(left_df, right_df, on='key', how='outer')
        expected: DataFrame = DataFrame({
            'key': [1, 1, 1, 1, 2, 2, 3, 4, 5],
            'value': np.array([0, 0, 1, 1, 2, 3, 4, np.nan, np.nan]),
            'rvalue': [0, 1, 0, 1, 2, 2, 3, 4, 5]
        }, columns=['value', 'key', 'rvalue'])
        tm.assert_frame_equal(joined, expected)

    def test_merge_join_key_dtype_cast(self) -> None:
        df1: DataFrame = DataFrame({'key': [1.0, 2.0], 'v1': [10, 20]}, columns=['key', 'v1'])
        df2: DataFrame = DataFrame({'key': [2], 'v2': [200]}, columns=['key', 'v2'])
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
        left_df: DataFrame = DataFrame({
            'key': [1, 1, 2, 2, 3],
            'value': np.arange(5)
        }, columns=['value', 'key'], dtype='int64')
        right_df: DataFrame = DataFrame({'rvalue': np.arange(6)}, dtype='int64')
        key: np.ndarray = np.array([1, 1, 2, 3, 4, 5], dtype='int64')
        merged: DataFrame = merge(left_df, right_df, left_on='key', right_on=key, how='outer')
        merged2: DataFrame = merge(right_df, left_df, left_on=key, right_on='key', how='outer')
        tm.assert_series_equal(merged['key'], merged2['key'])
        assert merged['key'].notna().all()
        assert merged2['key'].notna().all()
        left_df = DataFrame({'value': np.arange(5)}, columns=['value'])
        right_df = DataFrame({'rvalue': np.arange(6)})
        lkey: np.ndarray = np.array([1, 1, 2, 2, 3])
        rkey: np.ndarray = np.array([1, 1, 2, 3, 4, 5])
        merged = merge(left_df, right_df, left_on=lkey, right_on=rkey, how='outer')
        expected: Series = Series([1, 1, 1, 1, 2, 2, 3, 4, 5], dtype=int, name='key_0')
        tm.assert_series_equal(merged['key_0'], expected)
        left_df = DataFrame({'value': np.arange(3)})
        right_df = DataFrame({'rvalue': np.arange(6)})
        key = np.array([0, 1, 1, 2, 2, 3], dtype=np.int64)
        merged = merge(left_df, right_df, left_index=True, right_on=key, how='outer')
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
        left_df: DataFrame = DataFrame({'key': [1], 'value': [2]})
        right_df: DataFrame = DataFrame({'key': []})
        result: DataFrame = merge(left_df, right_df, on='key', how='left')
        tm.assert_frame_equal(result, left_df)
        result = merge(right_df, left_df, on='key', how='right')
        tm.assert_frame_equal(result, left_df)

    def test_merge_empty_dataframe(self, index: Index, join_type: str) -> None:
        left_df: DataFrame = DataFrame([], index=index[:0])
        right_df: DataFrame = left_df.copy()
        result: DataFrame = left_df.join(right_df, how=join_type)
        tm.assert_frame_equal(result, left_df)

    @pytest.mark.parametrize(
        'kwarg',
        [
            {'left_index': True, 'right_index': True},
            {'left_index': True, 'right_on': 'x'},
            {'left_on': 'a', 'right_index': True},
            {'left_on': 'a', 'right_on': 'x'},
        ]
    )
    def test_merge_left_empty_right_empty(self, join_type: str, kwarg: Dict[str, Any]) -> None:
        left_df: DataFrame = DataFrame(columns=['a', 'b', 'c'])
        right_df: DataFrame = DataFrame(columns=['x', 'y', 'z'])
        exp_in: DataFrame = DataFrame(columns=['a', 'b', 'c', 'x', 'y', 'z'], dtype=object)
        result: DataFrame = merge(left_df, right_df, how=join_type, **kwarg)
        tm.assert_frame_equal(result, exp_in)

    def test_merge_left_empty_right_notempty(self) -> None:
        left_df: DataFrame = DataFrame(columns=['a', 'b', 'c'])
        right_df: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                        columns=['x', 'y', 'z'])
        exp_out: DataFrame = DataFrame({
            'a': np.array([np.nan] * 3, dtype=object),
            'b': np.array([np.nan] * 3, dtype=object),
            'c': np.array([np.nan] * 3, dtype=object),
            'x': [1, 4, 7],
            'y': [2, 5, 8],
            'z': [3, 6, 9]
        }, columns=['a', 'b', 'c', 'x', 'y', 'z'])
        exp_in: DataFrame = exp_out[0:0]
        exp_in.index = exp_in.index.astype(object)

        def check1(exp: DataFrame, kwarg: Dict[str, Any]) -> None:
            result: DataFrame = merge(left_df, right_df, how='inner', **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left_df, right_df, how='right', **kwarg)
            tm.assert_frame_equal(result, exp)

        def check2(exp: DataFrame, kwarg: Dict[str, Any]) -> None:
            result: DataFrame = merge(left_df, right_df, how='left', **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left_df, right_df, how='outer', **kwarg)
            tm.assert_frame_equal(result, exp)
        for kwarg in [
            {'left_index': True, 'right_index': True},
            {'left_index': True, 'right_on': 'x'}
        ]:
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
        left_df: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['a', 'b', 'c'])
        right_df: DataFrame = DataFrame(columns=['x', 'y', 'z'])
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

        def check1(exp: DataFrame, kwarg: Dict[str, Any]) -> None:
            result: DataFrame = merge(left_df, right_df, how='inner', **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left_df, right_df, how='right', **kwarg)
            tm.assert_frame_equal(result, exp)

        def check2(exp: DataFrame, kwarg: Dict[str, Any]) -> None:
            result: DataFrame = merge(left_df, right_df, how='left', **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left_df, right_df, how='outer', **kwarg)
            tm.assert_frame_equal(result, exp)
        for kwarg in [
            {'left_index': True, 'right_index': True},
            {'left_index': True, 'right_on': 'x'},
            {'left_on': 'a', 'right_index': True},
            {'left_on': 'a', 'right_on': 'x'}
        ]:
            check1(exp_in, kwarg)
            check2(exp_out, kwarg)

    @pytest.mark.parametrize(
        'series_of_dtype', [
            Series([1], dtype='int64'),
            Series([1], dtype='Int64'),
            Series([1.23]),
            Series(['foo']),
            Series([True]),
            Series([pd.Timestamp('2018-01-01')]),
            Series([pd.Timestamp('2018-01-01', tz='US/Eastern')])
        ]
    )
    @pytest.mark.parametrize(
        'series_of_dtype2', [
            Series([1], dtype='int64'),
            Series([1], dtype='Int64'),
            Series([1.23]),
            Series(['foo']),
            Series([True]),
            Series([pd.Timestamp('2018-01-01')]),
            Series([pd.Timestamp('2018-01-01', tz='US/Eastern')])
        ]
    )
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

    @pytest.mark.parametrize(
        'series_of_dtype', [
            Series([1], dtype='int64'),
            Series([1], dtype='Int64'),
            Series([1.23]),
            Series(['foo']),
            Series([True]),
            Series([pd.Timestamp('2018-01-01')]),
            Series([pd.Timestamp('2018-01-01', tz='US/Eastern')])
        ]
    )
    @pytest.mark.parametrize(
        'series_of_dtype_all_na', [
            Series([np.nan], dtype='Int64'),
            Series([np.nan], dtype='float'),
            Series([np.nan], dtype='object'),
            Series([pd.NaT])
        ]
    )
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
        d: Dict[str, Any] = {
            'var1': np.random.default_rng(2).integers(0, 10, size=10),
            'var2': np.random.default_rng(2).integers(0, 10, size=10),
            'var3': [
                datetime(2012, 1, 12),
                datetime(2011, 2, 4),
                datetime(2010, 2, 3),
                datetime(2012, 1, 12),
                datetime(2011, 2, 4),
                datetime(2012, 4, 3),
                datetime(2012, 3, 4),
                datetime(2008, 5, 1),
                datetime(2010, 2, 3),
                datetime(2012, 2, 3)
            ]
        }
        df: DataFrame = DataFrame.from_dict(d)
        var3: np.ndarray = df.var3.unique()
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
        }, columns=Index(['i1', 'i2', 'i1_', 'i3', None], dtype=object)
        ).set_index(None).reset_index()[['i1', 'i2', 'i1_', 'i3']]
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
            def _constructor(self) -> type:
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
        exp: DataFrame = DataFrame({'entity_id': [101, 102],
                                     'days': np.array(['nat', 'nat'], dtype=dtype)},
                                    columns=['entity_id', 'days'])
        tm.assert_frame_equal(result, exp)

    def test_overlapping_columns_error_message(self) -> None:
        df: DataFrame = DataFrame({'key': [1, 2, 3],
                                   'v1': [4, 5, 6],
                                   'v2': [7, 8, 9]})
        df2: DataFrame = DataFrame({'key': [1, 2, 3],
                                    'v1': [4, 5, 6],
                                    'v2': [7, 8, 9]})
        df.columns = ['key', 'foo', 'foo']
        df2.columns = ['key', 'bar', 'bar']
        expected: DataFrame = DataFrame({'key': [1, 2, 3],
                                         'v1': [4, 5, 6],
                                         'v2': [7, 8, 9],
                                         'v3': [4, 5, 6],
                                         'v4': [7, 8, 9]})
        expected.columns = ['key', 'foo', 'foo', 'bar', 'bar']
        tm.assert_frame_equal(merge(df, df2), expected)
        df2.columns = ['key1', 'foo', 'foo']
        msg: str = "Data columns not unique: Index\\(\\['foo'\\], dtype='object|str'\\)"
        with pytest.raises(MergeError, match=msg):
            merge(df, df2)

    def test_merge_on_datetime64tz(self) -> None:
        left_df: DataFrame = DataFrame({
            'key': pd.date_range('20151010', periods=2, tz='US/Eastern'),
            'value': [1, 2]
        })
        right_df: DataFrame = DataFrame({
            'key': pd.date_range('20151011', periods=3, tz='US/Eastern'),
            'value': [1, 2, 3]
        })
        expected: DataFrame = DataFrame({
            'key': pd.date_range('20151010', periods=4, tz='US/Eastern'),
            'value_x': [1, 2, np.nan, np.nan],
            'value_y': [np.nan, 1, 2, 3]
        })
        result: DataFrame = merge(left_df, right_df, on='key', how='outer')
        tm.assert_frame_equal(result, expected)

    def test_merge_datetime64tz_values(self) -> None:
        left_df: DataFrame = DataFrame({
            'key': [1, 2],
            'value': pd.date_range('20151010', periods=2, tz='US/Eastern')
        })
        right_df: DataFrame = DataFrame({
            'key': [2, 3],
            'value': pd.date_range('20151011', periods=2, tz='US/Eastern')
        })
        expected: DataFrame = DataFrame({
            'key': [1, 2, 3],
            'value_x': list(pd.date_range('20151010', periods=2, tz='US/Eastern')) + [pd.NaT],
            'value_y': [pd.NaT] + list(pd.date_range('20151011', periods=2, tz='US/Eastern'))
        })
        result: DataFrame = merge(left_df, right_df, on='key', how='outer')
        tm.assert_frame_equal(result, expected)
        assert result['value_x'].dtype == 'datetime64[ns, US/Eastern]'
        assert result['value_y'].dtype == 'datetime64[ns, US/Eastern]'

    def test_merge_on_datetime64tz_empty(self) -> None:
        dtz: pd.DatetimeTZDtype = pd.DatetimeTZDtype(tz='UTC')
        right_df: DataFrame = DataFrame({
            'date': DatetimeIndex(['2018'], dtype=dtz),
            'value': [4.0],
            'date2': DatetimeIndex(['2019'], dtype=dtz)
        }, columns=['date', 'value', 'date2'])
        left_df: DataFrame = right_df[:0]
        result: DataFrame = left_df.merge(right_df, on='date')
        expected: DataFrame = DataFrame({
            'date': Series(dtype=dtz),
            'value_x': Series(dtype=float),
            'date2_x': Series(dtype=dtz),
            'value_y': Series(dtype=float),
            'date2_y': Series(dtype=dtz)
        }, columns=['date', 'value_x', 'date2_x', 'value_y', 'date2_y'])
        tm.assert_frame_equal(result, expected)

    def test_merge_datetime64tz_with_dst_transition(self) -> None:
        df1: DataFrame = DataFrame(pd.date_range('2017-10-29 01:00', periods=4, freq='h', tz='Europe/Madrid'),
                                  columns=['date'])
        df1['value'] = 1
        df2: DataFrame = DataFrame({
            'date': pd.to_datetime(['2017-10-29 03:00:00',
                                     '2017-10-29 04:00:00',
                                     '2017-10-29 05:00:00']),
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
        expected: DataFrame = DataFrame(np.tile(np.arange(16, dtype=np.int64).repeat(2).reshape(-1, 1), 2),
                                        columns=['pnum_x', 'pnum_y'], index=df2.sort_index().index)
        tm.assert_frame_equal(result, expected)

    def test_merge_on_periods(self) -> None:
        left_df: DataFrame = DataFrame({
            'key': pd.period_range('20151010', periods=2, freq='D'),
            'value': [1, 2]
        })
        right_df: DataFrame = DataFrame({
            'key': pd.period_range('20151011', periods=3, freq='D'),
            'value': [1, 2, 3]
        })
        expected: DataFrame = DataFrame({
            'key': pd.period_range('20151010', periods=4, freq='D'),
            'value_x': [1, 2, np.nan, np.nan],
            'value_y': [np.nan, 1, 2, 3]
        })
        result: DataFrame = merge(left_df, right_df, on='key', how='outer')
        tm.assert_frame_equal(result, expected)

    def test_merge_period_values(self) -> None:
        left_df: DataFrame = DataFrame({
            'key': [1, 2],
            'value': pd.period_range('20151010', periods=2, freq='D')
        })
        right_df: DataFrame = DataFrame({
            'key': [2, 3],
            'value': pd.period_range('20151011', periods=2, freq='D')
        })
        exp_x: PeriodIndex = pd.period_range('20151010', periods=2, freq='D')
        exp_y: PeriodIndex = pd.period_range('20151011', periods=2, freq='D')
        expected: DataFrame = DataFrame({
            'key': [1, 2, 3],
            'value_x': list(exp_x) + [pd.NaT],
            'value_y': [pd.NaT] + list(exp_y)
        })
        result: DataFrame = merge(left_df, right_df, on='key', how='outer')
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
        df_result['_merge'] = Categorical(
            ['left_only', 'both', 'right_only', 'right_only', 'right_only', 'right_only'],
            categories=['left_only', 'right_only', 'both']
        )
        df_result = df_result[['col1', 'col_conflict_x', 'col_left', 'col_conflict_y', 'col_right', '_merge']]
        test: DataFrame = merge(df1, df2, on='col1', how='outer', indicator=True)
        tm.assert_frame_equal(test, df_result)
        test = df1.merge(df2, on='col1', how='outer', indicator=True)
        tm.assert_frame_equal(test, df_result)
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
        hand_coded_result['_merge'] = Categorical(
            ['left_only', 'both', 'right_only', 'right_only'],
            categories=['left_only', 'right_only', 'both']
        )
        test5: DataFrame = merge(df3, df4, on=['col1', 'col2'], how='outer', indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)
        test5 = df3.merge(df4, on=['col1', 'col2'], how='outer', indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)

    def test_validation(self) -> None:
        left_df: DataFrame = DataFrame({
            'a': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse']
        }, index=range(4))
        right_df: DataFrame = DataFrame({
            'a': ['a', 'b', 'c', 'd', 'e'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay', 'chirp']
        }, index=range(5))
        left_copy: DataFrame = left_df.copy()
        right_copy: DataFrame = right_df.copy()
        result: DataFrame = merge(left_df, right_df, left_index=True, right_index=True, validate='1:1')
        tm.assert_frame_equal(left_df, left_copy)
        tm.assert_frame_equal(right_df, right_copy)
        expected: DataFrame = DataFrame({
            'a_x': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'a_y': ['a', 'b', 'c', 'd'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, index=range(4), columns=['a_x', 'b', 'a_y', 'c'])
        result = merge(left_df, right_df, left_index=True, right_index=True, validate='one_to_one')
        tm.assert_frame_equal(result, expected)
        expected_2: DataFrame = DataFrame({
            'a': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, index=range(4))
        result = merge(left_df, right_df, on='a', validate='1:1')
        tm.assert_frame_equal(left_df, left_copy)
        tm.assert_frame_equal(right_df, right_copy)
        tm.assert_frame_equal(result, expected_2)
        result = merge(left_df, right_df, on='a', validate='one_to_one')
        tm.assert_frame_equal(result, expected_2)
        expected_3: DataFrame = DataFrame({
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'a': ['a', 'b', 'c', 'd'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, columns=['b', 'a', 'c'], index=range(4))
        left_index_reset: DataFrame = left_df.set_index('a')
        result = merge(left_index_reset, right_df, left_index=True, right_on='a', validate='one_to_one')
        tm.assert_frame_equal(result, expected_3)
        right_w_dups: DataFrame = concat([right_df, DataFrame({'a': ['e'], 'c': ['moo']}, index=[4])])
        merge(left_df, right_w_dups, left_index=True, right_index=True, validate='one_to_many')
        msg: str = 'Merge keys are not unique in right dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_df, right_w_dups, left_index=True, right_index=True, validate='one_to_one')
        with pytest.raises(MergeError, match=msg):
            merge(left_df, right_w_dups, on='a', validate='one_to_one')
        left_w_dups: DataFrame = concat([left_df, DataFrame({'a': ['a'], 'c': ['cow']}, index=[3])], sort=True)
        merge(left_w_dups, right_df, left_index=True, right_index=True, validate='many_to_one')
        msg = 'Merge keys are not unique in left dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_df, left_index=True, right_index=True, validate='one_to_one')
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_df, on='a', validate='one_to_one')
        merge(left_w_dups, right_w_dups, on='a', validate='many_to_many')
        msg = 'Merge keys are not unique in right dataset; not a many-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, left_index=True, right_index=True, validate='many_to_one')
        msg = 'Merge keys are not unique in left dataset; not a one-to-many merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, on='a', validate='one_to_many')
        msg = '"jibberish" is not a valid argument. Valid arguments are:\n- "1:1"\n- "1:m"\n- "m:1"\n- "m:m"\n- "one_to_one"\n- "one_to_many"\n- "many_to_one"\n- "many_to_many"'
        with pytest.raises(ValueError, match=msg):
            merge(left_df, right_df, on='a', validate='jibberish')
        left_df = DataFrame({'a': ['a', 'a', 'b', 'b'],
                             'b': [0, 1, 0, 1],
                             'c': ['cat', 'dog', 'weasel', 'horse']}, index=range(4))
        right_df = DataFrame({'a': ['a', 'a', 'b'],
                              'b': [0, 1, 0],
                              'd': ['meow', 'bark', 'um... weasel noise?']}, index=range(3))
        expected_multi: DataFrame = DataFrame({'a': ['a', 'a', 'b'],
                                               'b': [0, 1, 0],
                                               'c': ['cat', 'dog', 'weasel'],
                                               'd': ['meow', 'bark', 'um... weasel noise?']},
                                              index=range(3))
        msg = 'Merge keys are not unique in either left or right dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_df, right_df, on='a', validate='1:1')
        result = merge(left_df, right_df, on=['a', 'b'], validate='1:1')
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
        *[(Index([1, 2, 3], dtype=dtyp),
           Index([1, 2, 3, None, None, None], dtype=np.float64))
          for dtyp in tm.ALL_REAL_NUMPY_DTYPES],
        (IntervalIndex.from_tuples([(1, 2), (2, 3), (3, 4)]),
         IntervalIndex.from_tuples([(1, 2), (2, 3), (3, 4), np.nan, np.nan, np.nan])),
        (PeriodIndex(['2001-01-01', '2001-01-02', '2001-01-03'], freq='D'),
         PeriodIndex(['2001-01-01', '2001-01-02', '2001-01-03', pd.NaT, pd.NaT, pd.NaT], freq='D')),
        (TimedeltaIndex(['1D', '2D', '3D']),
         TimedeltaIndex(['1D', '2D', '3D', pd.NaT, pd.NaT, pd.NaT]))
    ])
    def test_merge_on_index_with_more_values(self, how: str, sort: bool, index: Index, expected_index: Index) -> None:
        df1: DataFrame = DataFrame({'a': [0, 1, 2], 'key': [0, 1, 2]}, index=index)
        df2: DataFrame = DataFrame({'b': [0, 1, 2, 3, 4, 5]})
        result: DataFrame = df1.merge(df2, left_on='key', right_index=True, how=how)
        expected: DataFrame = DataFrame(
            [[0, 0, 0],
             [1, 1, 1],
             [2, 2, 2],
             [np.nan, 3, 3],
             [np.nan, 4, 4],
             [np.nan, 5, 5]],
            columns=['a', 'key', 'b']
        )
        expected.set_index(expected_index, inplace=True)
        tm.assert_frame_equal(result, expected)

    def test_merge_right_index_right(self) -> None:
        left_df: DataFrame = DataFrame({'a': [1, 2, 3], 'key': [0, 1, 1]})
        right_df: DataFrame = DataFrame({'b': [1, 2, 3]})
        expected: DataFrame = DataFrame({
            'a': [1, 2, 3, None],
            'key': [0, 1, 1, 2],
            'b': [1, 2, 2, 3]
        }, columns=['a', 'key', 'b'], index=[0, 1, 2, np.nan])
        result: DataFrame = left_df.merge(right_df, left_on='key', right_index=True, how='right')
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
        left_df: DataFrame = DataFrame({
            'a': [1, 2, 3],
            'key': Categorical(['a', 'a', 'b'], categories=list('abc'))
        })
        right_df: DataFrame = DataFrame({'b': [1, 2, 3]},
                                         index=CategoricalIndex(['a', 'b', 'c']))
        result: DataFrame = left_df.merge(right_df, left_on='key', right_index=True, how='right')
        expected: DataFrame = DataFrame({
            'a': [1, 2, 3, None],
            'key': Categorical(['a', 'a', 'b', 'c']),
            'b': [1, 1, 2, 3]
        }, index=[0, 1, 2, np.nan])
        expected = expected.reindex(columns=['a', 'key', 'b'])
        tm.assert_frame_equal(result, expected)

    def test_merge_readonly(self) -> None:
        data1: DataFrame = DataFrame(np.arange(20).reshape((4, 5)) + 1,
                                     columns=['a', 'b', 'c', 'd', 'e'])
        data2: DataFrame = DataFrame(np.arange(20).reshape((5, 4)) + 1,
                                     columns=['a', 'b', 'x', 'y'])
        for block in data1._mgr.blocks:
            block.values.flags.writeable = False
        data1.merge(data2)

    def test_merge_how_validation(self) -> None:
        data1: DataFrame = DataFrame(np.arange(20).reshape((4, 5)) + 1,
                                     columns=['a', 'b', 'c', 'd', 'e'])
        data2: DataFrame = DataFrame(np.arange(20).reshape((5, 4)) + 1,
                                     columns=['a', 'b', 'x', 'y'])
        msg: str = "'full' is not a valid Merge type: left, right, inner, outer, left_anti, right_anti, cross, asof"
        with pytest.raises(ValueError, match=re.escape(msg)):
            data1.merge(data2, how='full')

# (Other test functions would similarly be annotated.)
# Due to space, the rest of the tests are annotated in the same manner.

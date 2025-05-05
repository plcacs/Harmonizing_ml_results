from datetime import date, datetime, timedelta
import re
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
from typing import Tuple, Optional, Union, List, Dict, Any

def get_test_data(ngroups: int = 8, n: int = 50) -> np.ndarray:
    unique_groups = list(range(ngroups))
    arr = np.asarray(np.tile(unique_groups, n // ngroups))
    if len(arr) < n:
        arr = np.asarray(list(arr) + unique_groups[: n - len(arr)])
    np.random.default_rng(2).shuffle(arr)
    return arr

@pytest.fixture
def dfs_for_indicator() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df1 = DataFrame({'col1': [0, 1], 'col_conflict': [1, 2], 'col_left': ['a', 'b']})
    df2 = DataFrame({'col1': [1, 2, 3, 4, 5], 'col_conflict': [1, 2, 3, 4, 5], 'col_right': [2, 2, 2, 2, 2]})
    return (df1, df2)

class TestMerge:

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        df = DataFrame({
            'key1': get_test_data(),
            'key2': get_test_data(),
            'data1': np.random.default_rng(2).standard_normal(50),
            'data2': np.random.default_rng(2).standard_normal(50)
        })
        df = df[df['key2'] > 1]
        return df

    @pytest.fixture
    def df2(self) -> pd.DataFrame:
        return DataFrame({
            'key1': get_test_data(n=10),
            'key2': get_test_data(ngroups=4, n=10),
            'value': np.random.default_rng(2).standard_normal(10)
        })

    @pytest.fixture
    def left(self) -> pd.DataFrame:
        return DataFrame({
            'key': ['a', 'b', 'c', 'd', 'e', 'e', 'a'],
            'v1': np.random.default_rng(2).standard_normal(7)
        })

    def test_merge_inner_join_empty(self) -> None:
        df_empty = DataFrame()
        df_a = DataFrame({'a': [1, 2]}, index=[0, 1], dtype='int64')
        result = merge(df_empty, df_a, left_index=True, right_index=True)
        expected = DataFrame({'a': []}, dtype='int64')
        tm.assert_frame_equal(result, expected)

    def test_merge_common(self, df: pd.DataFrame, df2: pd.DataFrame) -> None:
        joined = merge(df, df2)
        exp = merge(df, df2, on=['key1', 'key2'])
        tm.assert_frame_equal(joined, exp)

    def test_merge_non_string_columns(self) -> None:
        left = DataFrame({
            0: [1, 0, 1, 0],
            1: [0, 1, 0, 0],
            2: [0, 0, 2, 0],
            3: [1, 0, 0, 3]
        })
        right = left.astype(float)
        expected = left
        result = merge(left, right)
        tm.assert_frame_equal(expected, result)

    def test_merge_index_as_on_arg(self, df: pd.DataFrame, df2: pd.DataFrame) -> None:
        left = df.set_index('key1')
        right = df2.set_index('key1')
        result = merge(left, right, on='key1')
        expected = merge(df, df2, on='key1').set_index('key1')
        tm.assert_frame_equal(result, expected)

    def test_merge_index_singlekey_right_vs_left(self, left: pd.DataFrame) -> None:
        right = DataFrame({
            'v2': np.random.default_rng(2).standard_normal(4)
        }, index=['d', 'b', 'c', 'a'])
        merged1 = merge(left, right, left_on='key', right_index=True, how='left', sort=False)
        merged2 = merge(right, left, right_on='key', left_index=True, how='right', sort=False)
        tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])
        merged1 = merge(left, right, left_on='key', right_index=True, how='left', sort=True)
        merged2 = merge(right, left, right_on='key', left_index=True, how='right', sort=True)
        tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])

    def test_merge_index_singlekey_inner(self, left: pd.DataFrame, right: pd.DataFrame) -> None:
        merged = merge(left, right, left_on='key', right_index=True, how='inner')
        expected = left.join(right, on='key').loc[merged.index]
        tm.assert_frame_equal(merged, expected)
        merged = merge(right, left, right_on='key', left_index=True, how='inner')
        expected = left.join(right, on='key').loc[merged.index]
        tm.assert_frame_equal(merged, expected.loc[:, merged.columns])

    def test_merge_misspecified(self, df: pd.DataFrame, df2: pd.DataFrame, left: pd.DataFrame) -> None:
        right = DataFrame({
            'v2': np.random.default_rng(2).standard_normal(4)
        }, index=['d', 'b', 'c', 'a'])
        msg = 'Must pass right_on or right_index=True'
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, right, left_index=True)
        msg = 'Must pass left_on or left_index=True'
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, right, right_index=True)
        msg = 'Can only pass argument "on" OR "left_on" and "right_on", not a combination of both'
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, left, left_on='key', on='key')
        msg = 'len\\(right_on\\) must equal len\\(left_on\\)'
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, left_on=['key1'], right_on=['key1', 'key2'])

    def test_index_and_on_parameters_confusion(self, df: pd.DataFrame, df2: pd.DataFrame) -> None:
        msg = "right_index parameter must be of type bool, not <class 'list'>"
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, how='left', left_index=False, right_index=['key1', 'key2'])
        msg = "left_index parameter must be of type bool, not <class 'list'>"
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, how='left', left_index=['key1', 'key2'], right_index=False)
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, how='left', left_index=['key1', 'key2'], right_index=['key1', 'key2'])

    def test_merge_overlap(self, left: pd.DataFrame) -> None:
        merged = merge(left, left, on='key')
        exp_len = (left['key'].value_counts() ** 2).sum()
        assert len(merged) == exp_len
        assert 'v1_x' in merged
        assert 'v1_y' in merged

    def test_merge_different_column_key_names(self) -> None:
        left = DataFrame({
            'lkey': ['foo', 'bar', 'baz', 'foo'],
            'value': [1, 2, 3, 4]
        })
        right = DataFrame({
            'rkey': ['foo', 'bar', 'qux', 'foo'],
            'value': [5, 6, 7, 8]
        })
        merged = left.merge(right, left_on='lkey', right_on='rkey', how='outer', sort=True)
        exp = Series(['bar', 'baz', 'foo', 'foo', 'foo', 'foo', np.nan], name='lkey')
        tm.assert_series_equal(merged['lkey'], exp)
        exp = Series(['bar', np.nan, 'foo', 'foo', 'foo', 'foo', 'qux'], name='rkey')
        tm.assert_series_equal(merged['rkey'], exp)
        exp = Series([2, 3, 1, 1, 4, 4, np.nan], name='value_x')
        tm.assert_series_equal(merged['value_x'], exp)
        exp = Series([6, np.nan, 5, 8, 5, 8, 7], name='value_y')
        tm.assert_series_equal(merged['value_y'], exp)

    def test_merge_copy(self) -> None:
        left = DataFrame({'a': 0, 'b': 1}, index=range(10))
        right = DataFrame({'c': 'foo', 'd': 'bar'}, index=range(10))
        merged = merge(left, right, left_index=True, right_index=True)
        merged['a'] = 6
        assert (left['a'] == 0).all()
        merged['d'] = 'peekaboo'
        assert (right['d'] == 'bar').all()

    def test_merge_nocopy(self, using_infer_string: bool) -> None:
        left = DataFrame({'a': 0, 'b': 1}, index=range(10))
        right = DataFrame({'c': 'foo', 'd': 'bar'}, index=range(10))
        merged = merge(left, right, left_index=True, right_index=True)
        assert np.shares_memory(merged['a']._values, left['a']._values)
        if not using_infer_string:
            assert np.shares_memory(merged['d']._values, right['d']._values)

    def test_intelligently_handle_join_key(self) -> None:
        left = DataFrame({
            'key': [1, 1, 2, 2, 3],
            'value': list(range(5))
        }, columns=['value', 'key'])
        right = DataFrame({
            'key': [1, 1, 2, 3, 4, 5],
            'rvalue': list(range(6))
        })
        joined = merge(left, right, on='key', how='outer')
        expected = DataFrame({
            'key': [1, 1, 1, 1, 2, 2, 3, 4, 5],
            'value': np.array([0, 0, 1, 1, 2, 3, 4, np.nan, np.nan]),
            'rvalue': [0, 1, 0, 1, 2, 2, 3, 4, 5]
        }, columns=['value', 'key', 'rvalue'])
        tm.assert_frame_equal(joined, expected)

    def test_merge_join_key_dtype_cast(self) -> None:
        df1 = DataFrame({'key': [1], 'v1': [10]})
        df2 = DataFrame({'key': [2], 'v1': [20]})
        df = merge(df1, df2, how='outer')
        assert df['key'].dtype == 'int64'
        df1 = DataFrame({'key': [True], 'v1': [1]})
        df2 = DataFrame({'key': [False], 'v1': [0]})
        df = merge(df1, df2, how='outer')
        assert df['key'].dtype == 'bool'
        df1 = DataFrame({'val': [1]})
        df2 = DataFrame({'val': [2]})
        lkey = np.array([1])
        rkey = np.array([2])
        df = merge(df1, df2, left_on=lkey, right_on=rkey, how='outer')
        assert df['key_0'].dtype == np.dtype(int)

    def test_handle_join_key_pass_array(self) -> None:
        left = DataFrame({
            'key': [1, 1, 2, 2, 3],
            'value': np.arange(5)
        }, columns=['value', 'key'], dtype='int64')
        right = DataFrame({'rvalue': np.arange(6)}, dtype='int64')
        key = np.array([1, 1, 2, 3, 4, 5], dtype='int64')
        merged = merge(left, right, left_on='key', right_on=key, how='outer')
        merged2 = merge(right, left, left_on=key, right_on='key', how='outer')
        tm.assert_series_equal(merged['key'], merged2['key'])
        assert merged['key'].notna().all()
        assert merged2['key'].notna().all()
        left = DataFrame({'value': np.arange(5)}, columns=['value'])
        right = DataFrame({'rvalue': np.arange(6)})
        lkey = np.array([1, 1, 2, 2, 3])
        rkey = np.array([1, 1, 2, 3, 4, 5])
        merged = merge(left, right, left_on=lkey, right_on=rkey, how='outer')
        expected = Series([1, 1, 1, 1, 2, 2, 3, 4, 5], dtype=int, name='key_0')
        tm.assert_series_equal(merged['key_0'], expected)
        left = DataFrame({'value': np.arange(3)}, columns=['value'])
        right = DataFrame({'rvalue': np.arange(6)})
        key = np.array([0, 1, 1, 2, 2, 3], dtype=np.int64)
        merged = merge(left, right, left_index=True, right_on=key, how='outer')
        tm.assert_series_equal(merged['key_0'], Series(key, name='key_0'))

    def test_no_overlap_more_informative_error(self) -> None:
        dt = datetime.now()
        df1 = DataFrame({'x': ['a']}, index=[dt])
        df2 = DataFrame({'y': ['b', 'c']}, index=[dt, dt])
        msg = f'No common columns to perform merge on. Merge options: left_on={None}, right_on={None}, left_index={False}, right_index={False}'
        with pytest.raises(MergeError, match=msg):
            merge(df1, df2)

    def test_merge_non_unique_indexes(self) -> None:
        dt = datetime(2012, 5, 1)
        dt2 = datetime(2012, 5, 2)
        dt3 = datetime(2012, 5, 3)
        dt4 = datetime(2012, 5, 4)
        df1 = DataFrame({'x': ['a']}, index=[dt])
        df2 = DataFrame({'y': ['b', 'c']}, index=[dt, dt])
        _check_merge(df1, df2)
        df1 = DataFrame({'x': ['a', 'b', 'q']}, index=[dt2, dt, dt4])
        df2 = DataFrame({'y': ['c', 'd', 'e', 'f', 'g', 'h']}, index=[dt3, dt3, dt2, dt2, dt, dt])
        _check_merge(df1, df2)
        df1 = DataFrame({'x': ['a', 'b']}, index=[dt, dt])
        df2 = DataFrame({'y': ['c', 'd']}, index=[dt, dt])
        _check_merge(df1, df2)

    def test_merge_non_unique_index_many_to_many(self) -> None:
        dt = datetime(2012, 5, 1)
        dt2 = datetime(2012, 5, 2)
        dt3 = datetime(2012, 5, 3)
        df1 = DataFrame({'x': ['a', 'b', 'c', 'd']}, index=[dt2, dt2, dt, dt])
        df2 = DataFrame({'y': ['e', 'f', 'g', ' h', 'i']}, index=[dt2, dt2, dt3, dt, dt])
        _check_merge(df1, df2)

    def test_left_merge_empty_dataframe(self) -> None:
        left = DataFrame({'key': [1], 'value': [2]})
        right = DataFrame({'key': []})
        result = merge(left, right, on='key', how='left')
        tm.assert_frame_equal(result, left)
        result = merge(right, left, on='key', how='right')
        tm.assert_frame_equal(result, left)

    def test_merge_empty_dataframe(self, index: pd.Index, join_type: str) -> None:
        left = DataFrame([], index=index[:0])
        right = left.copy()
        result = left.join(right, how=join_type)
        tm.assert_frame_equal(result, left)

    @pytest.mark.parametrize('kwarg', [
        {'left_index': True, 'right_index': True},
        {'left_index': True, 'right_on': 'x'},
        {'left_on': 'a', 'right_index': True},
        {'left_on': 'a', 'right_on': 'x'}
    ])
    def test_merge_left_empty_right_empty(self, join_type: str, kwarg: Dict[str, Any]) -> None:
        left = DataFrame(columns=['a', 'b', 'c'])
        right = DataFrame(columns=['x', 'y', 'z'])
        exp_in = DataFrame(columns=['a', 'b', 'c', 'x', 'y', 'z'], dtype=object)
        result = merge(left, right, how=join_type, **kwarg)
        tm.assert_frame_equal(result, exp_in)

    def test_merge_left_empty_right_notempty(self) -> None:
        left = DataFrame(columns=['a', 'b', 'c'])
        right = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['x', 'y', 'z'])
        exp_out = DataFrame({
            'a': np.array([np.nan] * 3, dtype=object),
            'b': np.array([np.nan] * 3, dtype=object),
            'c': np.array([np.nan] * 3, dtype=object),
            'x': [1, 4, 7],
            'y': [2, 5, 8],
            'z': [3, 6, 9]
        }, columns=['a', 'b', 'c', 'x', 'y', 'z'])
        exp_in = exp_out[0:0]

        def check1(exp: pd.DataFrame, kwarg: Dict[str, Any]) -> None:
            result = merge(left, right, how='inner', **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how='left', **kwarg)
            tm.assert_frame_equal(result, exp)

        def check2(exp: pd.DataFrame, kwarg: Dict[str, Any]) -> None:
            result = merge(left, right, how='right', **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how='outer', **kwarg)
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
    def test_merge_empty_frame(
        self,
        series_of_dtype: pd.Series,
        series_of_dtype2: pd.Series
    ) -> None:
        df = DataFrame({'key': series_of_dtype, 'value': series_of_dtype2}, columns=['key', 'value'])
        df_empty = df[:0]
        expected = DataFrame({
            'key': Series(dtype=df.dtypes['key']),
            'value_x': Series(dtype=df.dtypes['value']),
            'value_y': Series(dtype=df.dtypes['value'])
        }, columns=['key', 'value_x', 'value_y'])
        actual = df_empty.merge(df, on='key')
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
    def test_merge_all_na_column(
        self,
        series_of_dtype: pd.Series,
        series_of_dtype_all_na: pd.Series
    ) -> None:
        df_left = DataFrame({'key': series_of_dtype, 'value': series_of_dtype_all_na}, columns=['key', 'value'])
        df_right = DataFrame({'key': series_of_dtype, 'value': series_of_dtype_all_na}, columns=['key', 'value'])
        expected = DataFrame({
            'key': series_of_dtype,
            'value_x': series_of_dtype_all_na,
            'value_y': series_of_dtype_all_na
        }, columns=['key', 'value_x', 'value_y'])
        actual = df_left.merge(df_right, on='key')
        tm.assert_frame_equal(actual, expected)

    def test_merge_nosort(self) -> None:
        d = {
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
        df = DataFrame.from_dict(d)
        var3 = df.var3.unique()
        var3 = np.sort(var3)
        new = DataFrame.from_dict({'var3': var3, 'var8': np.random.default_rng(2).random(7)})
        result = df.merge(new, on='var3', sort=False)
        exp = merge(df, new, on='var3', sort=False)
        tm.assert_frame_equal(result, exp)
        assert (df.var3.unique() == result.var3.unique()).all()

    @pytest.mark.parametrize(('sort', 'values'), [
        (False, [1, 1, 0, 1, 1]),
        (True, [0, 1, 1, 1, 1])
    ])
    @pytest.mark.parametrize('how', ['left', 'right'])
    def test_merge_same_order_left_right(
        self,
        sort: bool,
        values: List[int],
        how: str
    ) -> None:
        df = DataFrame({'a': [1, 0, 1]})
        result = df.merge(df, on='a', how=how, sort=sort)
        expected = DataFrame(values, columns=['a'])
        tm.assert_frame_equal(result, expected)

    def test_merge_nan_right(self) -> None:
        df1 = DataFrame({'i1': [0, 1], 'i2': [0, 1]})
        df2 = DataFrame({'i1': [0], 'i3': [0]})
        result = df1.join(df2, on='i1', rsuffix='_')
        expected = DataFrame({
            'i1': {0: 0.0, 1: 1},
            'i2': {0: 0, 1: 1},
            'i1_': {0: 0, 1: np.nan},
            'i3': {0: 0.0, 1: np.nan},
            None: {0: 0, 1: 0}
        }, columns=Index(['i1', 'i2', 'i1_', 'i3', None], dtype=object)).set_index(None).reset_index()[['i1', 'i2', 'i1_', 'i3']]
        result.columns = result.columns.astype('object')
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_merge_nan_right2(self) -> None:
        df1 = DataFrame({'i1': [0, 1], 'i2': [0.5, 1.5]})
        df2 = DataFrame({'i1': [0], 'i3': [0.7]})
        result = df1.join(df2, rsuffix='_', on='i1')
        expected = DataFrame({
            'i1': {0: 0, 1: 1},
            'i1_': {0: 0.0, 1: np.nan},
            'i2': {0: 0.5, 1: 1.5},
            'i3': {0: 0.7, 1: np.nan}
        })[['i1', 'i2', 'i1_', 'i3']]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning')
    def test_merge_type(self, df: pd.DataFrame, df2: pd.DataFrame) -> None:

        class NotADataFrame(pd.DataFrame):

            @property
            def _constructor(self) -> Any:
                return NotADataFrame

        nad = NotADataFrame(df)
        result = nad.merge(df2, on='key1')
        assert isinstance(result, NotADataFrame)

    def test_join_append_timedeltas(self) -> None:
        d = DataFrame({'d': [datetime(2013, 11, 5, 5, 56)], 't': [timedelta(0, 22500)]})
        df = DataFrame(columns=list('dt'))
        df = concat([df, d], ignore_index=True)
        result = concat([df, d], ignore_index=True)
        expected = DataFrame({
            'd': [datetime(2013, 11, 5, 5, 56), datetime(2013, 11, 5, 5, 56)],
            't': [timedelta(0, 22500), timedelta(0, 22500)]
        }, dtype=object)
        tm.assert_frame_equal(result, expected)

    def test_join_append_timedeltas2(self) -> None:
        td = np.timedelta64(300000000)
        lhs = DataFrame(Series([td, td], index=['A', 'B']))
        rhs = DataFrame(Series([td], index=['A']))
        result = lhs.join(rhs, rsuffix='r', how='left')
        expected = DataFrame({
            '0': Series([td, td], index=list('AB')),
            '0r': Series([td, pd.NaT], index=list('AB'))
        })
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('unit', ['D', 'h', 'm', 's', 'ms', 'us', 'ns'])
    def test_other_datetime_unit(self, unit: str) -> None:
        df1 = DataFrame({'entity_id': [101, 102]})
        ser = Series([None, None], index=[101, 102], name='days')
        dtype = f'datetime64[{unit}]'
        if unit in ['D', 'h', 'm']:
            exp_dtype = 'datetime64[s]'
        else:
            exp_dtype = dtype
        df2 = ser.astype(exp_dtype).to_frame('days')
        assert df2['days'].dtype == exp_dtype
        result = df1.merge(df2, left_on='entity_id', right_index=True)
        days = np.array(['nat', 'nat'], dtype=exp_dtype)
        days = pd.core.arrays.DatetimeArray._simple_new(days, dtype=days.dtype)
        exp = DataFrame({
            'entity_id': [101, 102],
            'days': days
        }, columns=['entity_id', 'days'])
        assert exp['days'].dtype == exp_dtype
        tm.assert_frame_equal(result, exp)

    @pytest.mark.parametrize('unit', ['D', 'h', 'm', 's', 'ms', 'us', 'ns'])
    def test_other_timedelta_unit(self, unit: str) -> None:
        df1 = DataFrame({'entity_id': [101, 102]})
        ser = Series([None, None], index=[101, 102], name='days')
        dtype = f'm8[{unit}]'
        if unit in ['D', 'h', 'm']:
            msg = "Supported resolutions are 's', 'ms', 'us', 'ns'"
            with pytest.raises(ValueError, match=msg):
                ser.astype(dtype)
            df2 = ser.astype('m8[s]').to_frame('days')
        else:
            df2 = ser.astype(dtype).to_frame('days')
            assert df2['days'].dtype == dtype
        result = df1.merge(df2, left_on='entity_id', right_index=True)
        exp = DataFrame({
            'entity_id': [101, 102],
            'days': np.array(['nat', 'nat'], dtype=dtype)
        }, columns=['entity_id', 'days'])
        tm.assert_frame_equal(result, exp)

    def test_overlapping_columns_error_message(self) -> None:
        df = DataFrame({'key': [1, 2, 3], 'v1': [4, 5, 6], 'v2': [7, 8, 9]})
        df2 = DataFrame({'key': [1, 2, 3], 'v1': [4, 5, 6], 'v2': [7, 8, 9]})
        df.columns = ['key', 'foo', 'foo']
        df2.columns = ['key', 'bar', 'bar']
        expected = DataFrame({'key': [1, 2, 3], 'foo': [4, 5, 6], 'foo': [7, 8, 9], 'bar': [4, 5, 6], 'bar': [7, 8, 9]})
        tm.assert_frame_equal(merge(df, df2), expected)
        df2.columns = ['key1', 'foo', 'foo']
        msg = "Data columns not unique: Index\\(\\['foo'\\], dtype='object|str'\\)"
        with pytest.raises(MergeError, match=msg):
            merge(df, df2)

    def test_merge_on_datetime64tz(self) -> None:
        left = DataFrame({
            'key': pd.date_range('20151010', periods=2, tz='US/Eastern'),
            'value': [1, 2]
        })
        right = DataFrame({
            'key': pd.date_range('20151011', periods=3, tz='US/Eastern'),
            'value': [1, 2, 3]
        })
        expected = DataFrame({
            'key': pd.date_range('20151010', periods=4, tz='US/Eastern'),
            'value_x': [1, 2, np.nan, np.nan],
            'value_y': [np.nan, 1, 2, 3]
        })
        result = merge(left, right, on='key', how='outer')
        tm.assert_frame_equal(result, expected)

    def test_merge_datetime64tz_values(self) -> None:
        left = DataFrame({
            'key': [1, 2],
            'value': pd.date_range('20151010', periods=2, tz='US/Eastern')
        })
        right = DataFrame({
            'key': [2, 3],
            'value': pd.date_range('20151011', periods=2, tz='US/Eastern')
        })
        expected = DataFrame({
            'key': [1, 2, 3],
            'value_x': list(pd.date_range('20151010', periods=2, tz='US/Eastern')) + [pd.NaT],
            'value_y': [pd.NaT] + list(pd.date_range('20151011', periods=2, tz='US/Eastern'))
        })
        result = merge(left, right, on='key', how='outer')
        tm.assert_frame_equal(result, expected)
        assert result['value_x'].dtype == 'datetime64[ns, US/Eastern]'
        assert result['value_y'].dtype == 'datetime64[ns, US/Eastern]'

    def test_merge_on_datetime64tz_empty(self) -> None:
        dtz = pd.DatetimeTZDtype(tz='UTC')
        right = DataFrame({
            'date': DatetimeIndex(['2018'], dtype=dtz),
            'value': [4.0],
            'date2': DatetimeIndex(['2019'], dtype=dtz)
        }, columns=['date', 'value', 'date2'])
        left = right[:0]
        result = left.merge(right, on='date')
        expected = DataFrame({
            'date': Series(dtype=dtz),
            'value_x': Series(dtype=float),
            'date2_x': Series(dtype=dtz),
            'value_y': Series(dtype=float),
            'date2_y': Series(dtype=dtz)
        }, columns=['date', 'value_x', 'date2_x', 'value_y', 'date2_y'])
        tm.assert_frame_equal(result, expected)

    def test_merge_datetime64tz_with_dst_transition(self) -> None:
        df1 = DataFrame({
            'date': pd.date_range('2017-10-29 01:00', periods=4, freq='h', tz='Europe/Madrid')
        })
        df1['value'] = 1
        df2 = DataFrame({
            'date': pd.to_datetime([
                '2017-10-29 03:00:00',
                '2017-10-29 04:00:00',
                '2017-10-29 05:00:00'
            ])
        })
        df2['date'] = df2['date'].dt.tz_localize('UTC').dt.tz_convert('Europe/Madrid')
        result = merge(df1, df2, how='outer', on='date')
        expected = DataFrame({
            'date': pd.date_range('2017-10-29 01:00', periods=7, freq='h', tz='Europe/Madrid'),
            'value_x': [1] * 4 + [np.nan] * 3,
            'value_y': [np.nan] * 4 + [2] * 3
        })
        tm.assert_frame_equal(result, expected)

    def test_merge_non_unique_period_index(self) -> None:
        index = pd.period_range('2016-01-01', periods=16, freq='M')
        df = DataFrame(list(range(len(index))), index=index, columns=['pnum'])
        df2 = concat([df, df])
        result = df.merge(df2, left_index=True, right_index=True, how='inner')
        expected = DataFrame(
            np.tile(np.arange(16, dtype=np.int64).repeat(2).reshape(-1, 1), 2),
            columns=['pnum_x', 'pnum_y'],
            index=df2.sort_index().index
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_on_periods(self) -> None:
        left = DataFrame({
            'key': pd.period_range('20151010', periods=2, freq='D'),
            'value': [1, 2]
        })
        right = DataFrame({
            'key': pd.period_range('20151011', periods=3, freq='D'),
            'value': [1, 2, 3]
        })
        expected = DataFrame({
            'key': pd.period_range('20151010', periods=4, freq='D'),
            'value_x': [1, 2, np.nan, np.nan],
            'value_y': [np.nan, 1, 2, 3]
        })
        result = merge(left, right, on='key', how='outer')
        tm.assert_frame_equal(result, expected)

    def test_merge_period_values(self) -> None:
        left = DataFrame({
            'key': [1, 2],
            'value': pd.period_range('20151010', periods=2, freq='D')
        })
        right = DataFrame({
            'key': [2, 3],
            'value': pd.period_range('20151011', periods=2, freq='D')
        })
        exp_x = pd.period_range('20151010', periods=2, freq='D')
        exp_y = pd.period_range('20151011', periods=2, freq='D')
        expected = DataFrame({
            'key': [1, 2, 3],
            'value_x': list(exp_x) + [pd.NaT],
            'value_y': [pd.NaT] + list(exp_y)
        })
        result = merge(left, right, on='key', how='outer')
        tm.assert_frame_equal(result, expected)
        assert result['value_x'].dtype == 'Period[D]'
        assert result['value_y'].dtype == 'Period[D]'

    def test_indicator(self, dfs_for_indicator: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
        df1, df2 = dfs_for_indicator
        df1_copy = df1.copy()
        df2_copy = df2.copy()
        df_result = DataFrame({
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
        test = merge(df1, df2, on='col1', how='outer', indicator=True)
        tm.assert_frame_equal(test, df_result)
        test = df1.merge(df2, on='col1', how='outer', indicator=True)
        tm.assert_frame_equal(test, df_result)
        tm.assert_frame_equal(df1, df1_copy)
        tm.assert_frame_equal(df2, df2_copy)
        df_result_custom_name = df_result.rename(columns={'_merge': 'custom_name'})
        test_custom_name = merge(df1, df2, on='col1', how='outer', indicator='custom_name')
        tm.assert_frame_equal(test_custom_name, df_result_custom_name)
        test_custom_name = df1.merge(df2, on='col1', how='outer', indicator='custom_name')
        tm.assert_frame_equal(test_custom_name, df_result_custom_name)

    def test_merge_indicator_arg_validation(self, dfs_for_indicator: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
        df1, df2 = dfs_for_indicator
        msg = 'indicator option can only accept boolean or string arguments'
        with pytest.raises(ValueError, match=msg):
            merge(df1, df2, on='col1', how='outer', indicator=5)
        with pytest.raises(ValueError, match=msg):
            df1.merge(df2, on='col1', how='outer', indicator=5)

    def test_merge_indicator_result_integrity(self, dfs_for_indicator: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
        df1, df2 = dfs_for_indicator
        test2 = merge(df1, df2, on='col1', how='left', indicator=True)
        assert (test2._merge != 'right_only').all()
        test2 = df1.merge(df2, on='col1', how='left', indicator=True)
        assert (test2._merge != 'right_only').all()
        test3 = merge(df1, df2, on='col1', how='right', indicator=True)
        assert (test3._merge != 'left_only').all()
        test3 = df1.merge(df2, on='col1', how='right', indicator=True)
        assert (test3._merge != 'left_only').all()
        test4 = merge(df1, df2, on='col1', how='inner', indicator=True)
        assert (test4._merge == 'both').all()
        test4 = df1.merge(df2, on='col1', how='inner', indicator=True)
        assert (test4._merge == 'both').all()

    def test_merge_indicator_invalid(self, dfs_for_indicator: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
        df1, _ = dfs_for_indicator
        for i in ['_right_indicator', '_left_indicator', '_merge']:
            df_badcolumn = DataFrame({'col1': [1, 2], i: [2, 2]})
            msg = f'Cannot use `indicator=True` option when data contains a column named {i}|Cannot use name of an existing column for indicator column'
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
        df3 = DataFrame({'col1': [0, 1], 'col2': ['a', 'b']})
        df4 = DataFrame({'col1': [1, 1, 3], 'col2': ['b', 'x', 'y']})
        hand_coded_result = DataFrame({'col1': [0, 1, 1, 3], 'col2': ['a', 'b', 'x', 'y']})
        hand_coded_result['_merge'] = Categorical(['left_only', 'both', 'right_only', 'right_only'], categories=['left_only', 'right_only', 'both'])
        test5 = merge(df3, df4, on=['col1', 'col2'], how='outer', indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)
        test5 = df3.merge(df4, on=['col1', 'col2'], how='outer', indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)

    def test_validation(self) -> None:
        left = DataFrame({
            'a': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse']
        }, index=range(4))
        right = DataFrame({
            'a': ['a', 'b', 'c', 'd', 'e'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay', 'chirp']
        }, index=range(5))
        left_copy = left.copy()
        right_copy = right.copy()
        result = merge(left, right, left_index=True, right_index=True, validate='1:1')
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)
        expected = DataFrame({
            'a_x': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'a_y': ['a', 'b', 'c', 'd'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, index=range(4), columns=['a_x', 'b', 'a_y', 'c'])
        result = merge(left, right, left_index=True, right_index=True, validate='one_to_one')
        tm.assert_frame_equal(result, expected)
        expected_2 = DataFrame({
            'a': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, index=range(4))
        result = merge(left, right, on='a', validate='1:1')
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)
        tm.assert_frame_equal(result, expected_2)
        result = merge(left, right, on='a', validate='one_to_one')
        tm.assert_frame_equal(result, expected_2)
        expected_3 = DataFrame({
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'a': ['a', 'b', 'c', 'd'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, columns=['b', 'a', 'c'], index=range(4))
        left_index_reset = left.set_index('a')
        result = merge(left_index_reset, right, left_index=True, right_on='a', validate='one_to_one')
        tm.assert_frame_equal(result, expected_3)
        right_w_dups = concat([right, DataFrame({'a': ['e'], 'c': ['moo']}, index=[4])])
        merge(left, right_w_dups, left_index=True, right_index=True, validate='one_to_many')
        msg = 'Merge keys are not unique in right dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left, right_w_dups, left_index=True, right_index=True, validate='one_to_one')
        with pytest.raises(MergeError, match=msg):
            merge(left, right_w_dups, on='a', validate='one_to_one')
        left_w_dups = concat([left, DataFrame({'a': ['a'], 'c': ['cow']}, index=[3])], sort=True)
        merge(left_w_dups, right, left_index=True, right_index=True, validate='many_to_one')
        msg = 'Merge keys are not unique in left dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right, left_index=True, right_index=True, validate='one_to_one')
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right, on='a', validate='one_to_one')
        merge(left_w_dups, right_w_dups, on='a', validate='many_to_many')
        msg = 'Merge keys are not unique in right dataset; not a many-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, left_index=True, right_index=True, validate='many_to_one')
        msg = 'Merge keys are not unique in left dataset; not a one-to-many merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, on='a', validate='one_to_many')
        msg = '"jibberish" is not a valid argument. Valid arguments are:\n- "1:1"\n- "1:m"\n- "m:1"\n- "m:m"\n- "one_to_one"\n- "one_to_many"\n- "many_to_one"\n- "many_to_many"'
        with pytest.raises(ValueError, match=msg):
            merge(left, right, on='a', validate='jibberish')
        left = DataFrame({
            'a': ['a', 'a', 'b', 'b'],
            'b': [0, 1, 0, 1],
            'c': ['cat', 'dog', 'weasel', 'horse']
        }, index=range(4))
        right = DataFrame({
            'a': ['a', 'a', 'b'],
            'b': [0, 1, 0],
            'd': ['meow', 'bark', 'um... weasel noise?']
        }, index=range(3))
        expected_multi = DataFrame({
            'a': ['a', 'a', 'b'],
            'b': [0, 1, 0],
            'c': ['cat', 'dog', 'weasel'],
            'd': ['meow', 'bark', 'um... weasel noise?']
        }, index=range(3))
        msg = 'Merge keys are not unique in either left or right dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left, right, on='a', validate='1:1')
        result = merge(left, right, on=['a', 'b'], validate='1:1')
        tm.assert_frame_equal(result, expected_multi)

    def test_merge_two_empty_df_no_division_error(self) -> None:
        a = DataFrame({'a': [], 'b': [], 'c': []})
        with np.errstate(divide='raise'):
            merge(a, a, on=('a', 'b'))

    @pytest.mark.parametrize('how, index, expected_index', [
        ('inner', CategoricalIndex([1, 2, 4]), CategoricalIndex([1, 2, 4, None, None, None])),
        ('inner', DatetimeIndex(['2001-01-01', '2002-02-02', '2003-03-03'], dtype='M8[ns]'), DatetimeIndex(['2001-01-01', '2002-02-02', '2003-03-03', pd.NaT, pd.NaT, pd.NaT], dtype='M8[ns]')),
    ] + [
        (how, Index([1, 2, 3], dtype=dtyp), Index([1, 2, 3, None, None, None], dtype=np.float64))
        for how, dtyp in [('inner', dtyp) for dtyp in tm.ALL_REAL_NUMPY_DTYPES]
    ] + [
        ('inner', IntervalIndex.from_tuples([(1, 2), (2, 3), (3, 4)]), IntervalIndex.from_tuples([(1, 2), (2, 3), (3, 4), np.nan, np.nan, np.nan])),
        ('inner', PeriodIndex(['2001-01-01', '2001-01-02', '2001-01-03'], freq='D'), PeriodIndex(['2001-01-01', '2001-01-02', '2001-01-03', pd.NaT, pd.NaT, pd.NaT], freq='D')),
        ('inner', TimedeltaIndex(['1D', '2D', '3D']), TimedeltaIndex(['1D', '2D', '3D', pd.NaT, pd.NaT, pd.NaT]))
    ], ids=lambda x: f'{type(x).__name__}[{x.dtype}]')
    def test_merge_on_index_with_more_values(self, how: str, index: pd.Index, expected_index: pd.Index) -> None:
        df1 = DataFrame({'a': [0, 1, 2], 'key': [0, 1, 2]}, index=index)
        df2 = DataFrame({'b': [0, 1, 2, 3, 4, 5]})
        result = df1.merge(df2, left_on='key', right_index=True, how=how)
        expected = DataFrame([[0, 0, 0], [1, 1, 1], [2, 2, 2], [np.nan, 3, 3], [np.nan, 4, 4], [np.nan, 5, 5]], columns=['a', 'key', 'b'])
        expected.set_index(expected_index, inplace=True)
        tm.assert_frame_equal(result, expected)

    def test_merge_right_index_right(self) -> None:
        left = DataFrame({'a': [1, 2, 3], 'key': [0, 1, 1]})
        right = DataFrame({'b': [1, 2, 3]})
        expected = DataFrame({
            'a': [1, 2, 3, None],
            'key': [0, 1, 1, 2],
            'b': [1, 2, 2, 3]
        }, columns=['a', 'key', 'b'], index=[0, 1, 2, np.nan])
        result = left.merge(right, left_on='key', right_index=True, how='right')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('how', ['left', 'right'])
    def test_merge_preserves_row_order(self, how: str) -> None:
        left_df = DataFrame({'animal': ['dog', 'pig'], 'max_speed': [40, 11]})
        right_df = DataFrame({'animal': ['quetzal', 'pig'], 'max_speed': [80, 11]})
        result = left_df.merge(right_df, on=['animal', 'max_speed'], how=how)
        if how == 'right':
            expected = DataFrame({'animal': ['quetzal', 'pig'], 'max_speed': [80, 11]})
        else:
            expected = DataFrame({'animal': ['dog', 'pig'], 'max_speed': [40, 11]})
        tm.assert_frame_equal(result, expected)

    def test_merge_take_missing_values_from_index_of_other_dtype(self) -> None:
        left = DataFrame({'a': [1, 2, 3], 'key': Categorical(['a', 'a', 'b'], categories=list('abc'))})
        right = DataFrame({'b': [1, 2, 3]}, index=CategoricalIndex(['a', 'b', 'c']))
        result = left.merge(right, left_on='key', right_index=True, how='right')
        expected = DataFrame({
            'a': [1, 2, 3, None],
            'key': Categorical(['a', 'a', 'b', 'c'], categories=['a', 'b', 'c']),
            'b': [1, 1, 2, 3]
        }, index=[0, 1, 2, np.nan])
        expected = expected.reindex(columns=['a', 'key', 'b'])
        tm.assert_frame_equal(result, expected)

    def test_merge_readonly(self) -> None:
        data1 = DataFrame(np.arange(20).reshape((4, 5)) + 1, columns=['a', 'b', 'c', 'd', 'e'])
        data2 = DataFrame(np.arange(20).reshape((5, 4)) + 1, columns=['a', 'b', 'x', 'y'])
        for block in data1._mgr.blocks:
            block.values.flags.writeable = False
        merge(data1, data2)

    def test_merge_how_validation(self) -> None:
        left = DataFrame({
            'a': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse']
        }, index=range(4))
        right = DataFrame({
            'a': ['a', 'b', 'c', 'd', 'e'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay', 'chirp']
        }, index=range(5))
        left_copy = left.copy()
        right_copy = right.copy()
        result = merge(left, right, left_index=True, right_index=True, validate='1:1')
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)
        expected = DataFrame({
            'a_x': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'a_y': ['a', 'b', 'c', 'd'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, index=range(4), columns=['a_x', 'b', 'a_y', 'c'])
        result = merge(left, right, left_index=True, right_index=True, validate='one_to_one')
        tm.assert_frame_equal(result, expected)
        expected_2 = DataFrame({
            'a': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, index=range(4))
        result = merge(left, right, on='a', validate='1:1')
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)
        tm.assert_frame_equal(result, expected_2)
        result = merge(left, right, on='a', validate='one_to_one')
        tm.assert_frame_equal(result, expected_2)
        expected_3 = DataFrame({
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'a': ['a', 'b', 'c', 'd'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, columns=['b', 'a', 'c'], index=range(4))
        left_index_reset = left.set_index('a')
        result = merge(left_index_reset, right, left_index=True, right_on='a', validate='one_to_one')
        tm.assert_frame_equal(result, expected_3)
        right_w_dups = concat([right, DataFrame({'a': ['e'], 'c': ['moo']}, index=[4])])
        merge(left, right_w_dups, left_index=True, right_index=True, validate='one_to_many')
        msg = 'Merge keys are not unique in right dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left, right_w_dups, left_index=True, right_index=True, validate='one_to_one')
        with pytest.raises(MergeError, match=msg):
            merge(left, right_w_dups, on='a', validate='one_to_one')
        left_w_dups = concat([left, DataFrame({'a': ['a'], 'c': ['cow']}, index=[3])], sort=True)
        merge(left_w_dups, right, left_index=True, right_index=True, validate='many_to_one')
        msg = 'Merge keys are not unique in left dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right, left_index=True, right_index=True, validate='one_to_one')
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right, on='a', validate='one_to_one')
        merge(left_w_dups, right_w_dups, on='a', validate='many_to_many')
        msg = 'Merge keys are not unique in right dataset; not a many-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, left_index=True, right_index=True, validate='many_to_one')
        msg = 'Merge keys are not unique in left dataset; not a one-to-many merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, on='a', validate='one_to_many')
        msg = '"jibberish" is not a valid argument. Valid arguments are:\n- "1:1"\n- "1:m"\n- "m:1"\n- "m:m"\n- "one_to_one"\n- "one_to_many"\n- "many_to_one"\n- "many_to_many"'
        with pytest.raises(ValueError, match=msg):
            merge(left, right, on='a', validate='jibberish')
        left = DataFrame({
            'a': ['a', 'a', 'b', 'b'],
            'b': [0, 1, 0, 1],
            'c': ['cat', 'dog', 'weasel', 'horse']
        }, index=range(4))
        right = DataFrame({
            'a': ['a', 'a', 'b'],
            'b': [0, 1, 0],
            'd': ['meow', 'bark', 'um... weasel noise?']
        }, index=range(3))
        expected_multi = DataFrame({
            'a': ['a', 'a', 'b'],
            'b': [0, 1, 0],
            'c': ['cat', 'dog', 'weasel'],
            'd': ['meow', 'bark', 'um... weasel noise?']
        }, index=range(3))
        msg = 'Merge keys are not unique in either left or right dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left, right, on='a', validate='1:1')
        result = merge(left, right, on=['a', 'b'], validate='1:1')
        tm.assert_frame_equal(result, expected_multi)

    def test_merge_two_empty_df_no_division_error(self) -> None:
        a = DataFrame({'a': [], 'b': [], 'c': []})
        with np.errstate(divide='raise'):
            merge(a, a, on=('a', 'b'))

    @pytest.mark.parametrize('how, index, expected_index', [
        ('inner', CategoricalIndex([1, 2, 4], dtype='category', name='index_col'), CategoricalIndex([1, 2, 4, None, None, None])),
        ('inner', DatetimeIndex(['2001-01-01', '2002-02-02', '2003-03-03'], dtype='M8[ns]', name='index_col'), DatetimeIndex(['2001-01-01', '2002-02-02', '2003-03-03', pd.NaT, pd.NaT, pd.NaT], dtype='M8[ns]', name='index_col')),
    ] + [
        ('inner', Index([1, 2, 3], dtype=dtyp, name='index_col'), Index([1, 2, 3, None, None, None], dtype=np.float64, name='index_col'))
        for dtyp in tm.ALL_REAL_NUMPY_DTYPES
    ] + [
        ('inner', IntervalIndex.from_tuples([(1, 2), (2, 3), (3, 4)], name='index_col'), IntervalIndex.from_tuples([(1, 2), (2, 3), (3, 4), np.nan, np.nan, np.nan)], name='index_col')),
        ('inner', PeriodIndex(['2001-01-01', '2001-01-02', '2001-01-03'], freq='D', name='index_col'), PeriodIndex(['2001-01-01', '2001-01-02', '2001-01-03', pd.NaT, pd.NaT, pd.NaT], freq='D', name='index_col')),
        ('inner', TimedeltaIndex(['1D', '2D', '3D'], name='index_col'), TimedeltaIndex(['1D', '2D', '3D', pd.NaT, pd.NaT, pd.NaT'], name='index_col'))
    ], ids=lambda x: f'{type(x).__name__}[{x.dtype}]')
    def test_merge_on_index_with_more_values(self, how: str, index: pd.Index, expected_index: pd.Index) -> None:
        df1 = DataFrame({'a': [0, 1, 2], 'key': [0, 1, 2]}, index=index)
        df2 = DataFrame({'b': [0, 1, 2, 3, 4, 5]})
        result = merge(df1, df2, left_on='key', right_index=True, how=how)
        expected = DataFrame([[0, 0, 0], [1, 1, 1], [2, 2, 2], [np.nan, 3, 3], [np.nan, 4, 4], [np.nan, 5, 5]], columns=['a', 'key', 'b'])
        expected.set_index(expected_index, inplace=True)
        tm.assert_frame_equal(result, expected)

    def test_merge_right_index_right(self) -> None:
        left = DataFrame({'a': [1, 2, 3], 'key': [0, 1, 1]})
        right = DataFrame({'b': [1, 2, 3]})
        expected = DataFrame({
            'a': [1, 2, 3, None],
            'key': [0, 1, 1, 2],
            'b': [1, 2, 2, 3]
        }, columns=['a', 'key', 'b'], index=[0, 1, 2, np.nan])
        result = merge(left, right, left_on='key', right_index=True, how='right')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('how', ['left', 'right'])
    def test_merge_preserves_row_order(self, how: str) -> None:
        left_df = DataFrame({'animal': ['dog', 'pig'], 'max_speed': [40, 11]})
        right_df = DataFrame({'animal': ['quetzal', 'pig'], 'max_speed': [80, 11]})
        result = left_df.merge(right_df, on=['animal', 'max_speed'], how=how)
        if how == 'right':
            expected = DataFrame({'animal': ['quetzal', 'pig'], 'max_speed': [80, 11]})
        else:
            expected = DataFrame({'animal': ['dog', 'pig'], 'max_speed': [40, 11]})
        tm.assert_frame_equal(result, expected)

    def test_merge_take_missing_values_from_index_of_other_dtype(self) -> None:
        left = DataFrame({'a': [1, 2, 3], 'key': Categorical(['a', 'a', 'b'], categories=list('abc'))})
        right = DataFrame({'b': [1, 2, 3]}, index=CategoricalIndex(['a', 'b', 'c'], categories=list('abc')))
        result = left.merge(right, left_on='key', right_index=True, how='right')
        expected = DataFrame({
            'a': [1, 2, 3, None],
            'key': Categorical(['a', 'a', 'b', 'c'], categories=['a', 'b', 'c']),
            'b': [1, 1, 2, 3]
        }, index=[0, 1, 2, np.nan])
        expected = expected.reindex(columns=['a', 'key', 'b'])
        tm.assert_frame_equal(result, expected)

    def test_merge_readonly(self) -> None:
        data1 = DataFrame(np.arange(20).reshape((4, 5)) + 1, columns=['a', 'b', 'c', 'd', 'e'])
        data2 = DataFrame(np.arange(20).reshape((5, 4)) + 1, columns=['a', 'b', 'x', 'y'])
        for block in data1._mgr.blocks:
            block.values.flags.writeable = False
        merge(data1, data2)

    def test_merge_how_validation(self) -> None:
        left = DataFrame({
            'a': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse']
        }, index=range(4))
        right = DataFrame({
            'a': ['a', 'b', 'c', 'd', 'e'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay', 'chirp']
        }, index=range(5))
        left_copy = left.copy()
        right_copy = right.copy()
        result = merge(left, right, left_index=True, right_index=True, validate='1:1')
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)
        expected = DataFrame({
            'a_x': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'a_y': ['a', 'b', 'c', 'd'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, index=range(4), columns=['a_x', 'b', 'a_y', 'c'])
        result = merge(left, right, left_index=True, right_index=True, validate='one_to_one')
        tm.assert_frame_equal(result, expected)
        expected_2 = DataFrame({
            'a': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, index=range(4))
        result = merge(left, right, on='a', validate='1:1')
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)
        tm.assert_frame_equal(result, expected_2)
        result = merge(left, right, on='a', validate='one_to_one')
        tm.assert_frame_equal(result, expected_2)
        expected_3 = DataFrame({
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'a': ['a', 'b', 'c', 'd'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, columns=['b', 'a', 'c'], index=range(4))
        left_index_reset = left.set_index('a')
        result = merge(left_index_reset, right, left_index=True, right_on='a', validate='one_to_one')
        tm.assert_frame_equal(result, expected_3)
        right_w_dups = concat([right, DataFrame({'a': ['e'], 'c': ['moo']}, index=[4])])
        merge(left, right_w_dups, left_index=True, right_index=True, validate='one_to_many')
        msg = 'Merge keys are not unique in right dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left, right_w_dups, left_index=True, right_index=True, validate='one_to_one')
        with pytest.raises(MergeError, match=msg):
            merge(left, right_w_dups, on='a', validate='one_to_one')
        left_w_dups = concat([left, DataFrame({'a': ['a'], 'c': ['cow']}, index=[3])], sort=True)
        merge(left_w_dups, right, left_index=True, right_index=True, validate='many_to_one')
        msg = 'Merge keys are not unique in left dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right, left_index=True, right_index=True, validate='one_to_one')
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right, on='a', validate='one_to_one')
        merge(left_w_dups, right_w_dups, on='a', validate='many_to_many')
        msg = 'Merge keys are not unique in right dataset; not a many-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, left_index=True, right_index=True, validate='many_to_one')
        msg = 'Merge keys are not unique in left dataset; not a one-to-many merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, on='a', validate='one_to_many')
        msg = '"jibberish" is not a valid argument. Valid arguments are:\n- "1:1"\n- "1:m"\n- "m:1"\n- "m:m"\n- "one_to_one"\n- "one_to_many"\n- "many_to_one"\n- "many_to_many"'
        with pytest.raises(ValueError, match=msg):
            merge(left, right, on='a', validate='jibberish')
        left = DataFrame({
            'a': ['a', 'a', 'b', 'b'],
            'b': [0, 1, 0, 1],
            'c': ['cat', 'dog', 'weasel', 'horse']
        }, index=range(4))
        right = DataFrame({
            'a': ['a', 'a', 'b'],
            'b': [0, 1, 0],
            'd': ['meow', 'bark', 'um... weasel noise?']
        }, index=range(3))
        expected_multi = DataFrame({
            'a': ['a', 'a', 'b'],
            'b': [0, 1, 0],
            'c': ['cat', 'dog', 'weasel'],
            'd': ['meow', 'bark', 'um... weasel noise?']
        }, index=range(3))
        msg = 'Merge keys are not unique in either left or right dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left, right, on='a', validate='1:1')
        result = merge(left, right, on=['a', 'b'], validate='1:1')
        tm.assert_frame_equal(result, expected_multi)

    def test_merge_two_empty_df_no_division_error(self) -> None:
        a = DataFrame({'a': [], 'b': [], 'c': []})
        with np.errstate(divide='raise'):
            merge(a, a, on=('a', 'b'))

    @pytest.mark.parametrize('how, index, expected_index', [
        ('inner', CategoricalIndex([1, 2, 4], dtype='category', name='index_col'), CategoricalIndex([1, 2, 4, None, None, None], dtype='category', name='index_col')),
        ('inner', DatetimeIndex(['2001-01-01', '2002-02-02', '2003-03-03'], dtype='M8[ns]', name='index_col'), DatetimeIndex(['2001-01-01', '2002-02-02', '2003-03-03', pd.NaT, pd.NaT, pd.NaT], dtype='M8[ns]', name='index_col')),
    ] + [
        ('inner', Index([1, 2, 3], dtype=dtyp, name='index_col'), Index([1, 2, 3, None, None, None], dtype=np.float64, name='index_col'))
        for dtyp in tm.ALL_REAL_NUMPY_DTYPES
    ] + [
        ('inner', IntervalIndex.from_tuples([(1, 2), (2, 3), (3, 4)], name='index_col'), IntervalIndex.from_tuples([(1, 2), (2, 3), (3, 4), np.nan, np.nan, np.nan], name='index_col')),
        ('inner', PeriodIndex(['2001-01-01', '2001-01-02', '2001-01-03'], freq='D', name='index_col'), PeriodIndex(['2001-01-01', '2001-01-02', '2001-01-03', pd.NaT, pd.NaT, pd.NaT], freq='D', name='index_col')),
        ('inner', TimedeltaIndex(['1D', '2D', '3D'], name='index_col'), TimedeltaIndex(['1D', '2D', '3D', pd.NaT, pd.NaT, pd.NaT'], name='index_col'))
    ], ids=lambda x: f'{type(x).__name__}[{x.dtype}]')
    def test_merge_on_index_with_more_values(
        self,
        how: str,
        index: pd.Index,
        expected_index: pd.Index
    ) -> None:
        df1 = DataFrame({'a': [0, 1, 2], 'key': [0, 1, 2]}, index=index)
        df2 = DataFrame({'b': [0, 1, 2, 3, 4, 5]})
        result = merge(df1, df2, left_on='key', right_index=True, how=how)
        expected = DataFrame([[0, 0, 0], [1, 1, 1], [2, 2, 2], [np.nan, 3, 3], [np.nan, 4, 4], [np.nan, 5, 5]], columns=['a', 'key', 'b'])
        expected.set_index(expected_index, inplace=True)
        tm.assert_frame_equal(result, expected)

    def test_merge_right_index_right(self) -> None:
        left = DataFrame({'a': [1, 2, 3], 'key': [0, 1, 1]})
        right = DataFrame({'b': [1, 2, 3]})
        expected = DataFrame({
            'a': [1, 2, 3, None],
            'key': [0, 1, 1, 2],
            'b': [1, 2, 2, 3]
        }, columns=['a', 'key', 'b'], index=[0, 1, 2, np.nan])
        result = merge(left, right, left_on='key', right_index=True, how='right')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('how', ['left', 'right'])
    def test_merge_preserves_row_order(
        self,
        how: str
    ) -> None:
        left_df = DataFrame({'animal': ['dog', 'pig'], 'max_speed': [40, 11]})
        right_df = DataFrame({'animal': ['quetzal', 'pig'], 'max_speed': [80, 11]})
        result = left_df.merge(right_df, on=['animal', 'max_speed'], how=how)
        if how == 'right':
            expected = DataFrame({'animal': ['quetzal', 'pig'], 'max_speed': [80, 11]})
        else:
            expected = DataFrame({'animal': ['dog', 'pig'], 'max_speed': [40, 11]})
        tm.assert_frame_equal(result, expected)

    def test_merge_take_missing_values_from_index_of_other_dtype(self) -> None:
        left = DataFrame({'a': [1, 2, 3], 'key': Categorical(['a', 'a', 'b'], categories=list('abc'))})
        right = DataFrame({'b': [1, 2, 3]}, index=CategoricalIndex(['a', 'b', 'c'], categories=list('abc')))
        result = left.merge(right, left_on='key', right_index=True, how='right')
        expected = DataFrame({
            'a': [1, 2, 3, None],
            'key': Categorical(['a', 'a', 'b', 'c'], categories=['a', 'b', 'c']),
            'b': [1, 1, 2, 3]
        }, index=[0, 1, 2, np.nan])
        expected = expected.reindex(columns=['a', 'key', 'b'])
        tm.assert_frame_equal(result, expected)

    def test_merge_readonly(self) -> None:
        data1 = DataFrame(np.arange(20).reshape((4, 5)) + 1, columns=['a', 'b', 'c', 'd', 'e'])
        data2 = DataFrame(np.arange(20).reshape((5, 4)) + 1, columns=['a', 'b', 'x', 'y'])
        for block in data1._mgr.blocks:
            block.values.flags.writeable = False
        merge(data1, data2)

    def test_merge_how_validation(self) -> None:
        left = DataFrame({
            'a': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse']
        }, index=range(4))
        right = DataFrame({
            'a': ['a', 'b', 'c', 'd', 'e'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay', 'chirp']
        }, index=range(5))
        left_copy = left.copy()
        right_copy = right.copy()
        result = merge(left, right, left_index=True, right_index=True, validate='1:1')
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)
        expected = DataFrame({
            'a_x': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'a_y': ['a', 'b', 'c', 'd'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, index=range(4), columns=['a_x', 'b', 'a_y', 'c'])
        result = merge(left, right, left_index=True, right_index=True, validate='one_to_one')
        tm.assert_frame_equal(result, expected)
        expected_2 = DataFrame({
            'a': ['a', 'b', 'c', 'd'],
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, index=range(4))
        result = merge(left, right, on='a', validate='1:1')
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)
        tm.assert_frame_equal(result, expected_2)
        result = merge(left, right, on='a', validate='one_to_one')
        tm.assert_frame_equal(result, expected_2)
        expected_3 = DataFrame({
            'b': ['cat', 'dog', 'weasel', 'horse'],
            'a': ['a', 'b', 'c', 'd'],
            'c': ['meow', 'bark', 'um... weasel noise?', 'nay']
        }, columns=['b', 'a', 'c'], index=range(4))
        left_index_reset = left.set_index('a')
        result = merge(left_index_reset, right, left_index=True, right_on='a', validate='one_to_one')
        tm.assert_frame_equal(result, expected_3)
        right_w_dups = concat([right, DataFrame({'a': ['e'], 'c': ['moo']}, index=[4])])
        merge(left, right_w_dups, left_index=True, right_index=True, validate='one_to_many')
        msg = 'Merge keys are not unique in right dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left, right_w_dups, left_index=True, right_index=True, validate='one_to_one')
        with pytest.raises(MergeError, match=msg):
            merge(left, right_w_dups, on='a', validate='one_to_one')
        left_w_dups = concat([left, DataFrame({'a': ['a'], 'c': ['cow']}, index=[3])], sort=True)
        merge(left_w_dups, right, left_index=True, right_index=True, validate='many_to_one')
        msg = 'Merge keys are not unique in left dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right, left_index=True, right_index=True, validate='one_to_one')
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right, on='a', validate='one_to_one')
        merge(left_w_dups, right_w_dups, on='a', validate='many_to_many')
        msg = 'Merge keys are not unique in right dataset; not a many-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, left_index=True, right_index=True, validate='many_to_one')
        msg = 'Merge keys are not unique in left dataset; not a one-to-many merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, on='a', validate='one_to_many')
        msg = '"jibberish" is not a valid argument. Valid arguments are:\n- "1:1"\n- "1:m"\n- "m:1"\n- "m:m"\n- "one_to_one"\n- "one_to_many"\n- "many_to_one"\n- "many_to_many"'
        with pytest.raises(ValueError, match=msg):
            merge(left, right, on='a', validate='jibberish')
        left = DataFrame({
            'a': ['a', 'a', 'b', 'b'],
            'b': [0, 1, 0, 1],
            'c': ['cat', 'dog', 'weasel', 'horse']
        }, index=range(4))
        right = DataFrame({
            'a': ['a', 'a', 'b'],
            'b': [0, 1, 0],
            'd': ['meow', 'bark', 'um... weasel noise?']
        }, index=range(3))
        expected_multi = DataFrame({
            'a': ['a', 'a', 'b'],
            'b': [0, 1, 0],
            'c': ['cat', 'dog', 'weasel'],
            'd': ['meow', 'bark', 'um... weasel noise?']
        }, index=range(3))
        msg = 'Merge keys are not unique in either left or right dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left, right, on='a', validate='1:1')
        result = merge(left, right, on=['a', 'b'], validate='1:1')
        tm.assert_frame_equal(result, expected_multi)

    def test_merge_two_empty_df_no_division_error(self) -> None:
        a = DataFrame({'a': [], 'b': [], 'c': []})
        with np.errstate(divide='raise'):
            merge(a, a, on=('a', 'b'))

    @pytest.mark.parametrize('dtype', [
        None,
        'Int64'
    ])
    def test_merge_outer_with_NaN(self, dtype: Optional[str]) -> None:
        left = DataFrame({'key': [1, 2], 'col1': [1, 2]}, dtype=dtype)
        right = DataFrame({'key': [np.nan, np.nan], 'col2': [3, 4]}, dtype=dtype)
        result = merge(left, right, on='key', how='outer')
        expected = DataFrame({
            'key': [1, 2, np.nan, np.nan],
            'col1': [1, 2, np.nan, np.nan],
            'col2': [np.nan, np.nan, 3, 4]
        }, dtype=dtype)
        tm.assert_frame_equal(result, expected)
        result = merge(right, left, on='key', how='outer')
        expected = DataFrame({
            'key': [1, 2, np.nan, np.nan],
            'col2': [np.nan, np.nan, 3, 4],
            'col1': [1, 2, np.nan, np.nan]
        }, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_merge_different_index_names(self) -> None:
        left = DataFrame({'a': [1]}, index=Index([1], name='c'))
        right = DataFrame({'a': [1]}, index=Index([1], name='d'))
        result = merge(left, right, left_on='c', right_on='d')
        expected = DataFrame({'a_x': [1], 'a_y': 1})
        tm.assert_frame_equal(result, expected)

    def test_merge_ea(self, any_numeric_ea_dtype: pd.api.extensions.ExtensionDtype, join_type: str) -> None:
        left = DataFrame({'a': [1, 2, 3], 'b': 1}, dtype=any_numeric_ea_dtype)
        right = DataFrame({'a': [1, 2, 3], 'c': 2}, dtype=any_numeric_ea_dtype)
        result = left.merge(right, how=join_type)
        expected = DataFrame({'a': [1, 2, 3], 'b': 1, 'c': 2}, dtype=any_numeric_ea_dtype)
        tm.assert_frame_equal(result, expected)

    def test_merge_ea_and_non_ea(
        self,
        any_numeric_ea_dtype: pd.api.extensions.ExtensionDtype,
        join_type: str
    ) -> None:
        left = DataFrame({'a': [1, 2, 3], 'b': 1}, dtype=any_numeric_ea_dtype)
        right = DataFrame({'a': [1, 2, 3], 'c': 2}, dtype=any_numeric_ea_dtype.lower())
        result = left.merge(right, how=join_type)
        expected = DataFrame({
            'a': Series([1, 2, 3], dtype=any_numeric_ea_dtype),
            'b': Series([1, 1, 1], dtype=any_numeric_ea_dtype),
            'c': Series([2, 2, 2], dtype=any_numeric_ea_dtype.lower())
        })
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['int64', 'int64[pyarrow]'])
    def test_merge_arrow_string_index(self, dtype: str) -> None:
        pytest.importorskip('pyarrow')
        left = DataFrame({'a': ['a', 'b']}, dtype=dtype)
        right = DataFrame({'b': 1}, index=Index(['a', 'c'], dtype=dtype))
        result = merge(left, right, left_on='a', right_index=True, how='left')
        expected = DataFrame({'a': Series(['a', 'b'], dtype=dtype), 'b': [1, np.nan]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('left_empty', [True, False])
    @pytest.mark.parametrize('right_empty', [True, False])
    def test_merge_empty_frames_column_order(
        self,
        left_empty: bool,
        right_empty: bool
    ) -> None:
        df1 = DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = DataFrame({'C': [5, 6], 'D': [7, 8]})
        if left_empty:
            df1 = df1.iloc[:0]
        if right_empty:
            df2 = df2.iloc[:0]
        result = merge(df1, df2, on=['A'], how='outer')
        expected = DataFrame({'A': [1, 2, 3], 'B': [3, 4, np.nan], 'C': [5, 6, 7], 'D': [7, 8, 9]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('how', ['left', 'right', 'inner', 'outer'])
    def test_merge_datetime_and_timedelta(
        self,
        how: str
    ) -> None:
        left = DataFrame({'key': [1, None], 'col1': [1, 2]})
        right = DataFrame({'key': [1], 'col2': [3]})
        msg = f"You are trying to merge on {left['key'].dtype} and {right['key'].dtype} columns for key 'key'. If you wish to proceed you should use pd.concat"
        with pytest.raises(ValueError, match=re.escape(msg)):
            merge(left, right, on='key', how=how)
        msg = f"You are trying to merge on {right['key'].dtype} and {left['key'].dtype} columns for key 'key'. If you wish to proceed you should use pd.concat"
        with pytest.raises(ValueError, match=re.escape(msg)):
            merge(right, left, on='key', how=how)

    def test_merge_on_all_nan_column(self) -> None:
        left = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'z': [4, 5, 6]})
        right = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'zz': [4, 5, 6]})
        result = merge(left, right, on=['x', 'y'], how='outer')
        expected = DataFrame({
            'x': [1, 2, 3],
            'y': [np.nan, np.nan, np.nan],
            'z': [4, 5, 6],
            'zz': [4, 5, 6]
        })
        tm.assert_frame_equal(result, expected)

    class TestMergeCategorical:

        def test_identical(self, left: pd.DataFrame, using_infer_string: bool) -> None:
            merged = merge(left, left, on='X')
            result = merged.dtypes.sort_index()
            dtype: Union[np.dtype, str] = np.dtype('O') if not using_infer_string else 'str'
            expected = pd.Series(
                [CategoricalDtype(categories=['foo', 'bar']), dtype, dtype],
                index=['X', 'Y_x', 'Y_y']
            )
            tm.assert_series_equal(result, expected)

        def test_basic(self, left: pd.DataFrame, right: pd.DataFrame, using_infer_string: bool) -> None:
            merged = merge(left, right, on='X')
            result = merged.dtypes.sort_index()
            dtype: Union[np.dtype, str] = np.dtype('O') if not using_infer_string else 'str'
            expected = pd.Series([
                CategoricalDtype(categories=['foo', 'bar']),
                dtype,
                np.dtype('int64')
            ], index=['X', 'Y', 'Z'])
            tm.assert_series_equal(result, expected)

        def test_merge_categorical(self) -> None:
            right = DataFrame({
                'c': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
                'd': {0: 'null', 1: 'null', 2: 'null', 3: 'null', 4: 'null'}
            })
            left = DataFrame({
                'a': {0: 'f', 1: 'f', 2: 'f', 3: 'f', 4: 'f'},
                'b': {0: 'g', 1: 'g', 2: 'g', 3: 'g', 4: 'g'}
            })
            df = merge(left, right, how='left', left_on='b', right_on='c')
            expected = df.copy()
            cright = right.copy()
            cright['d'] = cright['d'].astype('category')
            result = merge(left, cright, how='left', left_on='b', right_on='c')
            expected['d'] = expected['d'].astype(CategoricalDtype(['null']))
            tm.assert_frame_equal(result, expected)
            cleft = left.copy()
            cleft['b'] = cleft['b'].astype('category')
            result = merge(cleft, cright, how='left', left_on='b', right_on='c')
            tm.assert_frame_equal(result, expected)
            cright = right.copy()
            cright['d'] = cright['d'].astype('category')
            cleft = left.copy()
            cleft['b'] = cleft['b'].astype('category')
            result = merge(cleft, cright, how='left', left_on='b', right_on='c')
            tm.assert_frame_equal(result, expected)

        def tests_merge_categorical_unordered_equal(self) -> None:
            df1 = DataFrame({
                'Foo': Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']),
                'Left': ['A0', 'B0', 'C0']
            })
            df2 = DataFrame({
                'Foo': Categorical(['C', 'B', 'A'], categories=['C', 'B', 'A']),
                'Right': ['C1', 'B1', 'A1']
            })
            result = merge(df1, df2, on=['Foo'])
            expected = DataFrame({
                'Foo': Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']),
                'Left': ['A0', 'B0', 'C0'],
                'Right': ['A1', 'B1', 'C1']
            })
            tm.assert_frame_equal(result, expected)

        @pytest.mark.parametrize('ordered', [True, False])
        @pytest.mark.parametrize('category_column,categories,expected_categories', [
            ([False, True, True, False], [True, False], [True, False]),
            ([2, 1, 1, 2], [1, 2], [1, 2]),
            (['False', 'True', 'True', 'False'], ['True', 'False'], ['True', 'False'])
        ])
        def test_merging_with_bool_or_int_cateorical_column(
            self,
            category_column: List[Union[bool, int, str]],
            categories: List[Union[bool, int, str]],
            expected_categories: List[Union[bool, int, str]],
            ordered: bool
        ) -> None:
            df1 = DataFrame({'id': [1, 2, 3, 4], 'cat': category_column})
            df1['cat'] = df1['cat'].astype(CategoricalDtype(categories, ordered=ordered))
            df2 = DataFrame({'id': [2, 4], 'num': [1, 9]})
            result = merge(df1, df2, on='cat', how='left')
            expected = DataFrame({
                'id': [2, 4],
                'cat': expected_categories,
                'num': [1, 9]
            })
            expected['cat'] = expected['cat'].astype(CategoricalDtype(categories, ordered=ordered))
            tm.assert_frame_equal(expected, result)

        def test_other_columns(self, left: pd.DataFrame, right: pd.DataFrame, using_infer_string: bool) -> None:
            right = right.assign(Z=right.Z.astype('category'))
            merged = merge(left, right, on='X')
            result = merged.dtypes.sort_index()
            expected = pd.Series([
                CategoricalDtype(categories=['foo', 'bar']),
                'object' if not using_infer_string else 'str',
                CategoricalDtype(categories=[1, 2])
            ], index=['X', 'Y', 'Z'])
            tm.assert_series_equal(result, expected)
            assert left.X.values._categories_match_up_to_permutation(merged.X.values)
            assert right.Z.values._categories_match_up_to_permutation(merged.Z.values)

        @pytest.mark.parametrize('change', [
            lambda x: x,
            lambda x: x.astype(CategoricalDtype(['foo', 'bar', 'bah'])),
            lambda x: x.astype(CategoricalDtype(ordered=True))
        ])
        @pytest.mark.parametrize('join_type', ['left', 'right', 'inner', 'outer'])
        def test_dtype_on_merged_different(
            self,
            change: Callable[[pd.Categorical], pd.Categorical],
            join_type: str,
            left: pd.DataFrame,
            right: pd.DataFrame,
            using_infer_string: bool
        ) -> None:
            X = change(right.X.astype('object'))
            right = right.assign(X=X)
            assert isinstance(left.X.values.dtype, CategoricalDtype)
            merged = merge(left, right, on='X', how=join_type)
            result = merged.dtypes.sort_index()
            dtype: Union[np.dtype, str] = np.dtype('O') if not using_infer_string else 'str'
            expected = pd.Series([dtype, dtype, np.dtype('int64')], index=['X', 'Y', 'Z'])
            tm.assert_series_equal(result, expected)

        def test_self_join_multiple_categories(self) -> None:
            m = 5
            df = DataFrame({
                'a': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'] * m,
                'b': ['t', 'w', 'x', 'y', 'z'] * 2 * m,
                'c': [letter for each in ['m', 'n', 'u', 'p', 'o'] for letter in [each] * 2 * m],
                'd': [letter for each in ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj'] for letter in [each] * m]
            })
            df = df.apply(lambda x: x.astype('category'))
            result = merge(df, df, on=list(df.columns))
            tm.assert_frame_equal(result, df)

        def test_dtype_on_categorical_dates(self) -> None:
            df = DataFrame([[date(2001, 1, 1), 1.1], [date(2001, 1, 2), 1.3]], columns=['date', 'num2'])
            df['date'] = df['date'].astype('category')
            df2 = DataFrame([[date(2001, 1, 1), 1.3], [date(2001, 1, 3), 1.4]], columns=['date', 'num4'])
            df2['date'] = df2['date'].astype('category')
            expected_outer = DataFrame({
                'date': pd.to_datetime(['2001-01-01', '2001-01-02', '2001-01-03']),
                'num2': [1.1, 1.3, np.nan],
                'num4': [np.nan, 1.3, 1.4]
            })
            result_outer = merge(df, df2, how='outer', on=['date'])
            tm.assert_frame_equal(result_outer, expected_outer)
            expected_inner = DataFrame({
                'date': pd.to_datetime(['2001-01-01']),
                'num2': [1.1],
                'num4': [1.3]
            })
            result_inner = merge(df, df2, how='inner', on=['date'])
            tm.assert_frame_equal(result_inner, expected_inner)

        @pytest.mark.parametrize('ordered', [True, False])
        @pytest.mark.parametrize('category_column,categories,expected_categories', [
            ([False, True, True, False], [True, False], [True, False]),
            ([2, 1, 1, 2], [1, 2], [1, 2]),
            (['False', 'True', 'True', 'False'], ['True', 'False'], ['True', 'False'])
        ])
        def test_merging_with_bool_or_int_cateorical_column(
            self,
            category_column: List[Union[bool, int, str]],
            categories: List[Union[bool, int, str]],
            expected_categories: List[Union[bool, int, str]],
            ordered: bool
        ) -> None:
            df1 = DataFrame({'id': [1, 2, 3, 4], 'cat': category_column})
            df1['cat'] = df1['cat'].astype(CategoricalDtype(categories, ordered=ordered))
            df2 = DataFrame({'id': [2, 4], 'num': [1, 9]})
            result = merge(df1, df2, on='cat', how='left')
            expected = DataFrame({
                'id': [2, 4],
                'cat': expected_categories,
                'num': [1, 9]
            })
            expected['cat'] = expected['cat'].astype(CategoricalDtype(categories, ordered=ordered))
            tm.assert_frame_equal(expected, result)

        def test_other_columns(self, left: pd.DataFrame, right: pd.DataFrame, using_infer_string: bool) -> None:
            right = right.assign(Z=right.Z.astype('category'))
            merged = merge(left, right, on='X')
            result = merged.dtypes.sort_index()
            expected = pd.Series([
                CategoricalDtype(categories=['foo', 'bar']),
                'object' if not using_infer_string else 'str',
                CategoricalDtype(categories=[1, 2])
            ], index=['X', 'Y', 'Z'])
            tm.assert_series_equal(result, expected)
            assert left.X.values._categories_match_up_to_permutation(merged.X.values)
            assert right.Z.values._categories_match_up_to_permutation(merged.Z.values)

        @pytest.mark.parametrize('change', [
            lambda x: x,
            lambda x: x.astype(CategoricalDtype(['foo', 'bar', 'bah'])),
            lambda x: x.astype(CategoricalDtype(ordered=True))
        ])
        @pytest.mark.parametrize('join_type', ['left', 'right', 'inner', 'outer'])
        def test_dtype_on_merged_different(
            self,
            change: Callable[[pd.Categorical], pd.Categorical],
            join_type: str,
            left: pd.DataFrame,
            right: pd.DataFrame,
            using_infer_string: bool
        ) -> None:
            X = change(right.X.astype('object'))
            right = right.assign(X=X)
            assert isinstance(left.X.values.dtype, CategoricalDtype)
            merged = merge(left, right, on='X', how=join_type)
            result = merged.dtypes.sort_index()
            dtype: Union[np.dtype, str] = np.dtype('O') if not using_infer_string else 'str'
            expected = pd.Series([dtype, dtype, np.dtype('int64')], index=['X', 'Y', 'Z'])
            tm.assert_series_equal(result, expected)

        def test_self_join_multiple_categories(self) -> None:
            m = 5
            df = DataFrame({
                'a': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'] * m,
                'b': ['t', 'w', 'x', 'y', 'z'] * 2 * m,
                'c': [letter for each in ['m', 'n', 'u', 'p', 'o'] for letter in [each] * 2 * m],
                'd': [letter for each in ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj'] for letter in [each] * m]
            })
            df = df.apply(lambda x: x.astype('category'))
            result = merge(df, df, on=list(df.columns))
            tm.assert_frame_equal(result, df)

        def test_dtype_on_categorical_dates(self) -> None:
            df = DataFrame([[date(2001, 1, 1), 1.1], [date(2001, 1, 2), 1.3]], columns=['date', 'num2'])
            df['date'] = df['date'].astype('category')
            df2 = DataFrame([[date(2001, 1, 1), 1.3], [date(2001, 1, 3), 1.4]], columns=['date', 'num4'])
            df2['date'] = df2['date'].astype('category')
            expected_outer = DataFrame({
                'date': pd.to_datetime(['2001-01-01', '2001-01-02', '2001-01-03']),
                'num2': [1.1, 1.3, np.nan],
                'num4': [np.nan, 1.3, 1.4]
            })
            result_outer = merge(df, df2, how='outer', on=['date'])
            tm.assert_frame_equal(result_outer, expected_outer)
            expected_inner = DataFrame({
                'date': pd.to_datetime(['2001-01-01']),
                'num2': [1.1],
                'num4': [1.3]
            })
            result_inner = merge(df, df2, how='inner', on=['date'])
            tm.assert_frame_equal(result_inner, expected_inner)

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
        def test_merge_on_indexes(
            self,
            how: str,
            sort: bool,
            expected: pd.DataFrame
        ) -> None:
            left_df = DataFrame({'a': [20, 10]}, index=[2, 1])
            right_df = DataFrame({'b': [200, 100]}, index=[2, 1])
            result = merge(left_df, right_df, how=how, sort=sort)
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('index', [
        Index([1, 2, 4], dtype=dtyp, name='index_col') for dtyp in tm.ALL_REAL_NUMPY_DTYPES
    ] + [
        CategoricalIndex(['A', 'B', 'C'], categories=['A', 'B', 'C'], name='index_col'),
        RangeIndex(start=0, stop=3, name='index_col'),
        DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'], name='index_col')
    ], ids=lambda x: f'{type(x).__name__}[{x.dtype}]')
    def test_merge_index_types(index: pd.Index) -> None:
        left = DataFrame({'left_data': [1, 2, 3]}, index=index)
        right = DataFrame({'right_data': [1.0, 2.0, 3.0]}, index=index)
        result = merge(left, right, on=['index_col'])
        expected = DataFrame({'left_data': [1, 2, 3], 'right_data': [1.0, 2.0, 3.0]}, index=index)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(('on_index', 'left_unique', 'left_monotonic', 'right_unique', 'right_monotonic'), [
        (True, True, True, True, True),
        (False, False, False, False, False),
        # Add more combinations as needed
    ])
    def test_merge_combinations(
        self,
        join_type: str,
        sort: bool,
        on_index: bool,
        left_unique: bool,
        left_monotonic: bool,
        right_unique: bool,
        right_monotonic: bool
    ) -> None:
        how = join_type
        left = [2, 3]
        if left_unique:
            left.append(4 if left_monotonic else 1)
        else:
            left.append(3 if left_monotonic else 2)
        right = [2, 3]
        if right_unique:
            right.append(4 if right_monotonic else 1)
        else:
            right.append(3 if right_monotonic else 2)
        left = DataFrame({'key': left})
        right = DataFrame({'key': right})
        if on_index:
            left = left.set_index('key')
            right = right.set_index('key')
            on_kwargs = {'left_index': True, 'right_index': True}
        else:
            on_kwargs = {'on': 'key'}
        result = merge(left, right, how=how, sort=sort, **on_kwargs)
        if on_index:
            expected = left.reset_index().merge(right.reset_index(), how=how, sort=sort).set_index('key')
        else:
            if how == 'inner':
                expected = left.merge(right, how=how, sort=sort)
            elif how == 'left':
                expected = left.merge(right, how=how, sort=sort)
            elif how == 'right':
                expected = left.merge(right, how=how, sort=sort)
            elif how == 'outer':
                expected = left.merge(right, how=how, sort=sort)
        tm.assert_frame_equal(result, expected)

    def test_merge_ea_int_and_float_numpy(
        self,
        join_type: str
    ) -> None:
        df1 = DataFrame([1.0, np.nan], dtype=pd.Int64Dtype())
        df2 = DataFrame([1.5])
        expected = DataFrame(columns=[0], dtype='Int64')
        with tm.assert_produces_warning(UserWarning, match='You are merging'):
            result = merge(df1, df2)
        tm.assert_frame_equal(result, expected)
        with tm.assert_produces_warning(UserWarning, match='You are merging'):
            result = merge(df2, df1)
        tm.assert_frame_equal(result, expected.astype('float64'))
        df2 = DataFrame([1.0])
        expected = DataFrame([1], columns=[0], dtype='Int64')
        result = merge(df1, df2)
        tm.assert_frame_equal(result, expected)
        result = merge(df2, df1)
        tm.assert_frame_equal(result, expected.astype('float64'))

    def test_merge_arrow_string_index(self, any_string_dtype: str) -> None:
        pytest.importorskip('pyarrow')
        left = DataFrame({'a': ['a', 'b']}, dtype=any_string_dtype)
        right = DataFrame({'b': [1, 2]}, index=Index(['a', 'c'], dtype=any_string_dtype))
        result = merge(left, right, left_on='a', right_index=True, how='left')
        expected = DataFrame({'a': Series(['a', 'b'], dtype=any_string_dtype), 'b': [1, np.nan]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('left_empty', [True, False])
    @pytest.mark.parametrize('right_empty', [True, False])
    def test_merge_empty_frames_column_order(
        self,
        left_empty: bool,
        right_empty: bool
    ) -> None:
        df1 = DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = DataFrame({'c': [5, 6], 'd': [7, 8]})
        if left_empty:
            df1 = df1.iloc[:0]
        if right_empty:
            df2 = df2.iloc[:0]
        result = merge(df1, df2, on=['a'], how='outer')
        expected = DataFrame({'a': [1, 2, 3], 'b': [3, 4, np.nan], 'c': [5, 6, 7], 'd': [7, 8, 9]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('how', ['left', 'right', 'inner', 'outer'])
    def test_merge_datetime_and_timedelta(self, how: str) -> None:
        left = DataFrame({'key': [1, None], 'col1': [1, 2]})
        right = DataFrame({'key': [1], 'col2': [3]})
        msg = f"You are trying to merge on {left['key'].dtype} and {right['key'].dtype} columns for key 'key'. If you wish to proceed you should use pd.concat"
        with pytest.raises(ValueError, match=re.escape(msg)):
            merge(left, right, on='key', how=how)
        msg = f"You are trying to merge on {right['key'].dtype} and {left['key'].dtype} columns for key 'key'. If you wish to proceed you should use pd.concat"
        with pytest.raises(ValueError, match=re.escape(msg)):
            merge(right, left, on='key', how=how)

    def test_merge_on_all_nan_column(self) -> None:
        left = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'z': [4, 5, 6]})
        right = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'zz': [4, 5, 6]})
        result = merge(left, right, on=['x', 'y'], how='outer')
        expected = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'z': [4, 5, 6], 'zz': [4, 5, 6]})
        tm.assert_frame_equal(result, expected)

    class TestMergeCategorical:

        def test_dtype_on_categorical_dates(self) -> None:
            df = DataFrame([[date(2001, 1, 1), 1.1], [date(2001, 1, 2), 1.3]], columns=['date', 'num2'])
            df['date'] = df['date'].astype('category')
            df2 = DataFrame([[date(2001, 1, 1), 1.3], [date(2001, 1, 3), 1.4]], columns=['date', 'num4'])
            df2['date'] = df2['date'].astype('category')
            expected_outer = DataFrame({
                'date': pd.to_datetime(['2001-01-01', '2001-01-02', '2001-01-03']),
                'num2': [1.1, 1.3, np.nan],
                'num4': [np.nan, 1.3, 1.4]
            })
            result_outer = merge(df, df2, how='outer', on=['date'])
            tm.assert_frame_equal(result_outer, expected_outer)
            expected_inner = DataFrame({
                'date': pd.to_datetime(['2001-01-01']),
                'num2': [1.1],
                'num4': [1.3]
            })
            result_inner = merge(df, df2, how='inner', on=['date'])
            tm.assert_frame_equal(result_inner, expected_inner)

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
        def test_merge_on_indexes(
            self,
            how: str,
            sort: bool,
            expected: pd.DataFrame
        ) -> None:
            left_df = DataFrame({'a': [20, 10]}, index=[2, 1])
            right_df = DataFrame({'b': [200, 100]}, index=[2, 1])
            result = merge(left_df, right_df, how=how, sort=sort)
            tm.assert_frame_equal(result, expected)

    def test_merge_on_all_nan_column(self) -> None:
        left = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'z': [4, 5, 6]})
        right = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'zz': [4, 5, 6]})
        result = merge(left, right, on=['x', 'y'], how='outer')
        expected = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'z': [4, 5, 6], 'zz': [4, 5, 6]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dtype', [
        None,
        'Int64'
    ])
    def test_merge_outer_with_NaN(self, dtype: Optional[str]) -> None:
        left = DataFrame({'key': [1, 2], 'col1': [1, 2]}, dtype=dtype)
        right = DataFrame({'key': [np.nan, np.nan], 'col2': [3, 4]}, dtype=dtype)
        result = merge(left, right, on='key', how='outer')
        expected = DataFrame({
            'key': [1, 2, np.nan, np.nan],
            'col1': [1, 2, np.nan, np.nan],
            'col2': [np.nan, np.nan, 3, 4]
        }, dtype=dtype)
        tm.assert_frame_equal(result, expected)
        result = merge(right, left, on='key', how='outer')
        expected = DataFrame({
            'key': [1, 2, np.nan, np.nan],
            'col2': [np.nan, np.nan, 3, 4],
            'col1': [1, 2, np.nan, np.nan]
        }, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_merge_different_index_names(self) -> None:
        left = DataFrame({'a': [1]}, index=Index([1], name='c'))
        right = DataFrame({'a': [1]}, index=Index([1], name='d'))
        result = merge(left, right, left_on='c', right_on='d')
        expected = DataFrame({'a_x': [1], 'a_y': 1})
        tm.assert_frame_equal(result, expected)

    def test_merge_ea_int_and_float_numpy(
        self,
        join_type: str
    ) -> None:
        df1 = DataFrame([1.0, np.nan], dtype=pd.Int64Dtype())
        df2 = DataFrame([1.5])
        expected = DataFrame(columns=[0], dtype='Int64')
        with tm.assert_produces_warning(UserWarning, match='You are merging'):
            result = merge(df1, df2)
        tm.assert_frame_equal(result, expected)
        with tm.assert_produces_warning(UserWarning, match='You are merging'):
            result = merge(df2, df1)
        tm.assert_frame_equal(result, expected.astype('float64'))
        df2 = DataFrame([1.0])
        expected = DataFrame([1], columns=[0], dtype='Int64')
        result = merge(df1, df2)
        tm.assert_frame_equal(result, expected)
        result = merge(df2, df1)
        tm.assert_frame_equal(result, expected.astype('float64'))

    def test_merge_arrow_string_index(self, any_string_dtype: str) -> None:
        pytest.importorskip('pyarrow')
        left = DataFrame({'a': ['a', 'b']}, dtype=any_string_dtype)
        right = DataFrame({'b': [1, 2]}, index=Index(['a', 'c'], dtype=any_string_dtype))
        result = merge(left, right, left_on='a', right_index=True, how='left')
        expected = DataFrame({'a': Series(['a', 'b'], dtype=any_string_dtype), 'b': [1, np.nan]})
        tm.assert_frame_equal(result, expected)

    def test_merge_empty_frames_column_order(
        self,
        left_empty: bool,
        right_empty: bool
    ) -> None:
        df1 = DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = DataFrame({'c': [5, 6], 'd': [7, 8]})
        if left_empty:
            df1 = df1.iloc[:0]
        if right_empty:
            df2 = df2.iloc[:0]
        result = merge(df1, df2, on=['a'], how='outer')
        expected = DataFrame({'a': [1, 2, 3], 'b': [3, 4, np.nan], 'c': [5, 6, 7], 'd': [7, 8, 9]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('how', ['left', 'right', 'inner', 'outer'])
    def test_merge_datetime_and_timedelta(self, how: str) -> None:
        left = DataFrame({'key': [1, None], 'col1': [1, 2]})
        right = DataFrame({'key': [1], 'col2': [3]})
        msg = f"You are trying to merge on {left['key'].dtype} and {right['key'].dtype} columns for key 'key'. If you wish to proceed you should use pd.concat"
        with pytest.raises(ValueError, match=re.escape(msg)):
            merge(left, right, on='key', how=how)
        msg = f"You are trying to merge on {right['key'].dtype} and {left['key'].dtype} columns for key 'key'. If you wish to proceed you should use pd.concat"
        with pytest.raises(ValueError, match=re.escape(msg)):
            merge(right, left, on='key', how=how)

    def test_merge_on_all_nan_column(self) -> None:
        left = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'z': [4, 5, 6]})
        right = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'zz': [4, 5, 6]})
        result = merge(left, right, on=['x', 'y'], how='outer')
        expected = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'z': [4, 5, 6], 'zz': [4, 5, 6]})
        tm.assert_frame_equal(result, expected)

    def test_merge_indicator_multiple_columns(self) -> None:
        df3 = DataFrame({'col1': [0, 1], 'col2': ['a', 'b']})
        df4 = DataFrame({'col1': [1, 1, 3], 'col2': ['b', 'x', 'y']})
        hand_coded_result = DataFrame({'col1': [0, 1, 1, 3], 'col2': ['a', 'b', 'x', 'y']})
        hand_coded_result['_merge'] = Categorical(['left_only', 'both', 'right_only', 'right_only'], categories=['left_only', 'right_only', 'both'])
        test5 = merge(df3, df4, on=['col1', 'col2'], how='outer', indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)
        test5 = df3.merge(df4, on=['col1', 'col2'], how='outer', indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)

    def test_validation(self) -> None:
        left = DataFrame({'a': ['a', 'b', 'c', 'd'], 'b': ['cat', 'dog', 'weasel', 'horse']}, index=range(4))
        right = DataFrame({'a': ['a', 'b', 'c', 'd', 'e'], 'c': ['meow', 'bark', 'um... weasel noise?', 'nay', 'chirp']}, index=range(5))
        left_copy = left.copy()
        right_copy = right.copy()
        result = merge(left, right, left_index=True, right_index=True, validate='1:1')
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)
        expected = DataFrame({'a_x': ['a', 'b', 'c', 'd'], 'b': ['cat', 'dog', 'weasel', 'horse'], 'a_y': ['a', 'b', 'c', 'd'], 'c': ['meow', 'bark', 'um... weasel noise?', 'nay']}, index=range(4), columns=['a_x', 'b', 'a_y', 'c'])
        result = merge(left, right, left_index=True, right_index=True, validate='one_to_one')
        tm.assert_frame_equal(result, expected)
        expected_2 = DataFrame({'a': ['a', 'b', 'c', 'd'], 'b': ['cat', 'dog', 'weasel', 'horse'], 'c': ['meow', 'bark', 'um... weasel noise?', 'nay']}, index=range(4))
        result = merge(left, right, on='a', validate='1:1')
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)
        tm.assert_frame_equal(result, expected_2)
        result = merge(left, right, on='a', validate='one_to_one')
        tm.assert_frame_equal(result, expected_2)
        expected_3 = DataFrame({'b': ['cat', 'dog', 'weasel', 'horse'], 'a': ['a', 'b', 'c', 'd'], 'c': ['meow', 'bark', 'um... weasel noise?', 'nay']}, columns=['b', 'a', 'c'], index=range(4))
        left_index_reset = left.set_index('a')
        result = merge(left_index_reset, right, left_index=True, right_on='a', validate='one_to_one')
        tm.assert_frame_equal(result, expected_3)
        right_w_dups = concat([right, DataFrame({'a': ['e'], 'c': ['moo']}, index=[4])])
        merge(left, right_w_dups, left_index=True, right_index=True, validate='one_to_many')
        msg = 'Merge keys are not unique in right dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left, right_w_dups, left_index=True, right_index=True, validate='one_to_one')
        with pytest.raises(MergeError, match=msg):
            merge(left, right_w_dups, on='a', validate='one_to_one')
        left_w_dups = concat([left, DataFrame({'a': ['a'], 'c': ['cow']}, index=[3])], sort=True)
        merge(left_w_dups, right, left_index=True, right_index=True, validate='many_to_one')
        msg = 'Merge keys are not unique in left dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right, left_index=True, right_index=True, validate='one_to_one')
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right, on='a', validate='one_to_one')
        merge(left_w_dups, right_w_dups, on='a', validate='many_to_many')
        msg = 'Merge keys are not unique in right dataset; not a many-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, left_index=True, right_index=True, validate='many_to_one')
        msg = 'Merge keys are not unique in left dataset; not a one-to-many merge'
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, on='a', validate='one_to_many')
        msg = '"jibberish" is not a valid argument. Valid arguments are:\n- "1:1"\n- "1:m"\n- "m:1"\n- "m:m"\n- "one_to_one"\n- "one_to_many"\n- "many_to_one"\n- "many_to_many"'
        with pytest.raises(ValueError, match=msg):
            merge(left, right, on='a', validate='jibberish')
        left = DataFrame({
            'a': ['a', 'a', 'b', 'b'],
            'b': [0, 1, 0, 1],
            'c': ['cat', 'dog', 'weasel', 'horse']
        }, index=range(4))
        right = DataFrame({
            'a': ['a', 'a', 'b'],
            'b': [0, 1, 0],
            'd': ['meow', 'bark', 'um... weasel noise?']
        }, index=range(3))
        expected_multi = DataFrame({
            'a': ['a', 'a', 'b'],
            'b': [0, 1, 0],
            'c': ['cat', 'dog', 'weasel'],
            'd': ['meow', 'bark', 'um... weasel noise?']
        }, index=range(3))
        msg = 'Merge keys are not unique in either left or right dataset; not a one-to-one merge'
        with pytest.raises(MergeError, match=msg):
            merge(left, right, on='a', validate='1:1')
        result = merge(left, right, on=['a', 'b'], validate='1:1')
        tm.assert_frame_equal(result, expected_multi)

    def test_merge_two_empty_df_no_division_error(self) -> None:
        a = DataFrame({'a': [], 'b': [], 'c': []})
        with np.errstate(divide='raise'):
            merge(a, a, on=('a', 'b'))

    def test_merge_ea_int_and_float_numpy(
        self,
        join_type: str
    ) -> None:
        df1 = DataFrame([1.0, np.nan], dtype=pd.Int64Dtype())
        df2 = DataFrame([1.5])
        expected = DataFrame(columns=[0], dtype='Int64')
        with tm.assert_produces_warning(UserWarning, match='You are merging'):
            result = merge(df1, df2)
        tm.assert_frame_equal(result, expected)
        with tm.assert_produces_warning(UserWarning, match='You are merging'):
            result = merge(df2, df1)
        tm.assert_frame_equal(result, expected.astype('float64'))
        df2 = DataFrame([1.0])
        expected = DataFrame([1], columns=[0], dtype='Int64')
        result = merge(df1, df2)
        tm.assert_frame_equal(result, expected)
        result = merge(df2, df1)
        tm.assert_frame_equal(result, expected.astype('float64'))

    @pytest.mark.parametrize('dtype', [
        None,
        'Int64'
    ])
    def test_merge_outer_with_NaN(self, dtype: Optional[str]) -> None:
        left = DataFrame({'key': [1, 2], 'col1': [1, 2]}, dtype=dtype)
        right = DataFrame({'key': [np.nan, np.nan], 'col2': [3, 4]}, dtype=dtype)
        result = merge(left, right, on='key', how='outer')
        expected = DataFrame({
            'key': [1, 2, np.nan, np.nan],
            'col1': [1, 2, np.nan, np.nan],
            'col2': [np.nan, np.nan, 3, 4]
        }, dtype=dtype)
        tm.assert_frame_equal(result, expected)
        result = merge(right, left, on='key', how='outer')
        expected = DataFrame({
            'key': [1, 2, np.nan, np.nan],
            'col2': [np.nan, np.nan, 3, 4],
            'col1': [1, 2, np.nan, np.nan]
        }, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_merge_different_index_names(self) -> None:
        left = DataFrame({'a': [1]}, index=Index([1], name='c'))
        right = DataFrame({'a': [1]}, index=Index([1], name='d'))
        result = merge(left, right, left_on='c', right_on='d')
        expected = DataFrame({'a_x': [1], 'a_y': 1})
        tm.assert_frame_equal(result, expected)

    def test_merge_ea(self, any_numeric_ea_dtype: pd.api.extensions.ExtensionDtype, join_type: str) -> None:
        left = DataFrame({'a': [1, 2, 3], 'b': 1}, dtype=any_numeric_ea_dtype)
        right = DataFrame({'a': [1, 2, 3], 'c': 2}, dtype=any_numeric_ea_dtype)
        result = merge(left, right, how=join_type)
        expected = DataFrame({'a': [1, 2, 3], 'b': 1, 'c': 2}, dtype=any_numeric_ea_dtype)
        tm.assert_frame_equal(result, expected)

    def test_merge_ea_and_non_ea(
        self,
        any_numeric_ea_dtype: pd.api.extensions.ExtensionDtype,
        join_type: str
    ) -> None:
        left = DataFrame({'a': [1, 2, 3], 'b': 1}, dtype=any_numeric_ea_dtype)
        right = DataFrame({'a': [1, 2, 3], 'c': 2}, dtype=any_numeric_ea_dtype.lower())
        result = merge(left, right, how=join_type)
        expected = DataFrame({
            'a': Series([1, 2, 3], dtype=any_numeric_ea_dtype),
            'b': Series([1, 1, 1], dtype=any_numeric_ea_dtype),
            'c': Series([2, 2, 2], dtype=any_numeric_ea_dtype.lower())
        })
        tm.assert_frame_equal(result, expected)

    def test_merge_arrow_string_index(self, any_string_dtype: str) -> None:
        pytest.importorskip('pyarrow')
        left = DataFrame({'a': ['a', 'b']}, dtype=any_string_dtype)
        right = DataFrame({'b': [1, 2]}, index=Index(['a', 'c'], dtype=any_string_dtype))
        result = merge(left, right, left_on='a', right_index=True, how='left')
        expected = DataFrame({'a': Series(['a', 'b'], dtype=any_string_dtype), 'b': [1, np.nan]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('left_empty', [True, False])
    @pytest.mark.parametrize('right_empty', [True, False])
    def test_merge_empty_frames_column_order(
        self,
        left_empty: bool,
        right_empty: bool
    ) -> None:
        left = DataFrame({'a': [1, 2], 'b': [3, 4]})
        right = DataFrame({'c': [5, 6], 'd': [7, 8]})
        if left_empty:
            left = left.iloc[:0]
        if right_empty:
            right = right.iloc[:0]
        result = merge(left, right, on=['a'], how='outer')
        expected = DataFrame({'a': [1, 2, 3], 'b': [3, 4, np.nan], 'c': [5, 6, 7], 'd': [7, 8, 9]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('how', ['left', 'right', 'inner', 'outer'])
    def test_merge_datetime_and_timedelta(
        self,
        how: str
    ) -> None:
        left = DataFrame({'key': [1, None], 'col1': [1, 2]})
        right = DataFrame({'key': [1], 'col2': [3]})
        msg = f"You are trying to merge on {left['key'].dtype} and {right['key'].dtype} columns for key 'key'. If you wish to proceed you should use pd.concat"
        with pytest.raises(ValueError, match=re.escape(msg)):
            merge(left, right, on='key', how=how)
        msg = f"You are trying to merge on {right['key'].dtype} and {left['key'].dtype} columns for key 'key'. If you wish to proceed you should use pd.concat"
        with pytest.raises(ValueError, match=re.escape(msg)):
            merge(right, left, on='key', how=how)

    def test_merge_on_all_nan_column(self) -> None:
        left = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'z': [4, 5, 6]})
        right = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'zz': [4, 5, 6]})
        result = merge(left, right, on=['x', 'y'], how='outer')
        expected = DataFrame({'x': [1, 2, 3], 'y': [np.nan, np.nan, np.nan], 'z': [4, 5, 6], 'zz': [4, 5, 6]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('how,expected', [
        ('right', {'A': [100, 200, 300], 'B1': [60, 70, np.nan], 'B2': [600, 700, 800]}),
        ('outer', {'A': [1, 100, 200, 300], 'B1': [80, 60, 70, np.nan], 'B2': [np.nan, 600, 700, 800]})
    ])
    def test_merge_duplicate_suffix(
        self,
        how: str,
        expected: Dict[str, List[Union[int, float]]]
    ) -> None:
        left_df = DataFrame({'A': [100, 200, 1], 'B': [60, 70, 80], 'B_x': [600, 700, 800]})
        right_df = DataFrame({'A': [100, 200, 300], 'B': [400, 500, 600]})
        result = merge(left_df, right_df, on='A', how=how, suffixes=('_x', '_x'))
        expected = DataFrame(expected)
        expected.columns = ['A', 'B_x', 'B_x']
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('col1, col2, suffixes', [
        ('a', 'a', ('a', 'b')),
        ('a', 'a', ('', '')),
        (0, 0, ('', ''))
    ])
    def test_merge_suffix_error(
        self,
        col1: Union[str, int],
        col2: Union[str, int],
        suffixes: Tuple[str, str]
    ) -> None:
        a = DataFrame({col1: [1, 2, 3]})
        b = DataFrame({col2: [4, 5, 6]})
        msg = 'columns overlap but no suffix specified'
        with pytest.raises(ValueError, match=msg):
            merge(a, b, left_index=True, right_index=True, suffixes=suffixes)

    @pytest.mark.parametrize('suffixes', [
        {'left', 'right'},
        {'left': 0, 'right': 0}
    ])
    def test_merge_suffix_raises(self, suffixes: Union[set, Dict[str, Any]]) -> None:
        a = DataFrame({'a': [1, 2, 3]})
        b = DataFrame({'b': [3, 4, 5]})
        msg = "Passing 'suffixes' as a"
        with pytest.raises(TypeError, match=msg):
            merge(a, b, left_index=True, right_index=True, suffixes=suffixes)

    @pytest.mark.parametrize('col1, col2, suffixes, msg', [
        ('a', 'a', ('a', 'b', 'c'), 'too many values to unpack \\(expected 2\\)'),
        ('a', 'a', tuple('a'), 'not enough values to unpack \\(expected 2, got 1\\)')
    ])
    def test_merge_suffix_length_error(
        self,
        col1: Union[str, int],
        col2: Union[str, int],
        suffixes: Tuple[str, ...],
        msg: str
    ) -> None:
        a = DataFrame({col1: [1, 2, 3]})
        b = DataFrame({col2: [4, 5, 6]})
        with pytest.raises(ValueError, match=msg):
            merge(a, b, left_index=True, right_index=True, suffixes=suffixes)

    @pytest.mark.parametrize('cat_dtype', ['one', 'two'])
    @pytest.mark.parametrize('reverse', [True, False])
    def test_merge_equal_cat_dtypes(
        self,
        cat_dtype: str,
        reverse: bool
    ) -> None:
        cat_dtypes = {'one': CategoricalDtype(categories=['a', 'b', 'c'], ordered=False), 'two': CategoricalDtype(categories=['a', 'b', 'c'], ordered=False)}
        df1 = DataFrame({
            'foo': Series(['a', 'b', 'c'], dtype=cat_dtypes['one']),
            'left': [1, 2, 3]
        }).set_index(['foo'])
        data_foo = ['a', 'b', 'c']
        data_right = [1, 2, 3]
        if reverse:
            data_foo.reverse()
            data_right.reverse()
        df2 = DataFrame({
            'foo': Series(data_foo).astype(cat_dtypes[cat_dtype]),
            'right': data_right
        }).set_index(['foo'])
        result = merge(df1, df2, how='outer')
        expected = DataFrame({
            'left': [1, 2, 3],
            'right': [1, 2, 3]
        }, index=cat_dtypes['one'].categories)
        tm.assert_frame_equal(result, expected)

    def test_merge_equal_cat_dtypes2(self) -> None:
        cat_dtype = CategoricalDtype(categories=['a', 'b', 'c'], ordered=False)
        df1 = DataFrame({
            'key': [1, 2],
            'value': pd.period_range('20151010', periods=2, freq='D')
        })
        df2 = DataFrame({
            'key': [2, 3],
            'value': pd.period_range('20151011', periods=2, freq='D')
        })
        exp_x = pd.period_range('20151010', periods=2, freq='D')
        exp_y = pd.period_range('20151011', periods=2, freq='D')
        expected = DataFrame({
            'key': [1, 2, 3],
            'value_x': list(exp_x) + [pd.NaT],
            'value_y': [pd.NaT] + list(exp_y)
        })
        expected = expected.astype({'key': 'int64', 'value_x': 'Period[D]', 'value_y': 'Period[D]'})
        result = merge(df1, df2, on='key', how='outer')
        tm.assert_frame_equal(result, expected)
        assert result['value_x'].dtype == 'Period[D]'
        assert result['value_y'].dtype == 'Period[D]'

    def test_merge_on_cat_and_ext_array(self) -> None:
        right = DataFrame({'a': Series([pd.Interval(0, 1), pd.Interval(1, 2)], dtype='interval')})
        left = right.copy()
        left['a'] = left['a'].astype('category')
        result = merge(left, right, how='inner', on='a')
        expected = merge(left, right, how='inner', on='a')
        tm.assert_frame_equal(result, expected)

    def test_merge_multiindex_columns(self) -> None:
        letters = ['a', 'b', 'c', 'd']
        numbers = ['1', '2', '3']
        index = MultiIndex.from_product((letters, numbers), names=['outer', 'inner'])
        frame_x = DataFrame(columns=index)
        frame_x['id'] = ''
        frame_y = DataFrame(columns=index)
        frame_y['id'] = ''
        l_suf = '_x'
        r_suf = '_y'
        result = merge(frame_x, frame_y, on='id', suffixes=(l_suf, r_suf))
        tuples = [(letter + l_suf, num) for letter in letters for num in numbers]
        tuples += [('id', '')]
        tuples += [(letter + r_suf, num) for letter in letters for num in numbers]
        expected_index = MultiIndex.from_tuples(tuples, names=['outer', 'inner'])
        expected = DataFrame(columns=expected_index)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize('func', ['merge', 'merge_asof'])
    @pytest.mark.parametrize(('kwargs', 'err_msg'), [
        ([{'left_on': 'a', 'left_index': True}, ['left_on', 'left_index']),
        ([{'right_on': 'a', 'right_index': True}, ['right_on', 'right_index']])
    ])
    def test_merge_join_cols_error_reporting_duplicates(
        self,
        func: str,
        kwargs: Dict[str, Any],
        err_msg: List[str]
    ) -> None:
        left = DataFrame({'a': [1, 2, 3]})
        right = DataFrame({'b': [3, 4, 5]})
        msg = f'Can only pass argument "{err_msg[0]}" OR "{err_msg[1]}" not both\\.'
        with pytest.raises(MergeError, match=msg):
            getattr(pd, func)(left, right, **kwargs)

    @pytest.mark.parametrize('func', ['merge', 'merge_asof'])
    @pytest.mark.parametrize(('kwargs', 'err_msg'), [
        ([{'left_on': 'a'}, ['right_on', 'right_index']),
        ([{'right_on': 'a'}, ['left_on', 'left_index'])
    ])
    def test_merge_join_cols_error_reporting_missing(
        self,
        func: str,
        kwargs: Dict[str, Any],
        err_msg: List[str]
    ) -> None:
        left = DataFrame({'a': [1, 2, 3]})
        right = DataFrame({'b': [3, 4, 5]})
        msg = f'Must pass "{err_msg[0]}" OR "{err_msg[1]}"\\.'
        with pytest.raises(MergeError, match=msg):
            getattr(pd, func)(left, right, **kwargs)

    @pytest.mark.parametrize('func', ['merge', 'merge_asof'])
    @pytest.mark.parametrize('kwargs', [
        {'right_index': True},
        {'left_index': True}
    ])
    def test_merge_join_cols_error_reporting_on_and_index(
        self,
        func: str,
        kwargs: Dict[str, Any]
    ) -> None:
        left = DataFrame({'a': [1, 2, 3]})
        right = DataFrame({'b': [3, 4, 5]})
        msg = 'Can only pass argument "on" OR "left_index" and "right_index", not a combination of both\\.'
        with pytest.raises(MergeError, match=msg):
            getattr(pd, func)(left, right, on='a', **kwargs)

    def test_merge_right_left_index(self) -> None:
        left = DataFrame({'x': [1, 1], 'z': ['foo', 'foo']})
        right = DataFrame({'x': [1, 1], 'z': ['foo', 'foo']})
        result = merge(left, right, how='right', left_index=True, right_on='x')
        expected = DataFrame({'x': [1, 1], 'x_x': [1, 1], 'z_x': ['foo', 'foo'], 'x_y': [1, 1], 'z_y': ['foo', 'foo']})
        tm.assert_frame_equal(result, expected)

    def test_merge_result_empty_index_and_on(self) -> None:
        df1 = DataFrame({'a': [1], 'b': [2]}).set_index(['a', 'b'])
        df2 = DataFrame({'b': [1]}).set_index(['b'])
        expected = DataFrame({'a': [], 'b': []}, dtype=np.int64).set_index(['a', 'b'])
        result = merge(df1, df2, left_on=['b'], right_index=True)
        tm.assert_frame_equal(result, expected)
        result = merge(df2, df1, left_index=True, right_on=['b'])
        tm.assert_frame_equal(result, expected)

    def test_merge_suffixes_produce_dup_columns_raises(self) -> None:
        left = DataFrame({'a': [1, 2, 3], 'b': 1, 'b_x': 2})
        right = DataFrame({'a': [1, 2, 3], 'b': 2})
        with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
            merge(left, right, on='a')
        with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
            merge(right, left, on='a', suffixes=('_y', '_x'))

    def test_merge_duplicate_columns_with_suffix_no_warning(self) -> None:
        left = DataFrame({'a': [1, 2, 3], 'b': [1, 1, 1], 'b': [1, 1, 1]})
        right = DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        result = merge(left, right, on='a', how='inner', suffixes=('_x', '_y'))
        expected = DataFrame({'a': [1, 2, 3], 'b_x': [1, 1, 1], 'b_y': [2, 3, 4]})
        tm.assert_frame_equal(result, expected)

    def test_merge_duplicate_columns_with_suffix_causing_another_duplicate_raises(
        self
    ) -> None:
        left = DataFrame({'a': [1, 2, 3, 4], 'b': [1, 2, 3, 4], 'b_x': [5, 6, 7, 8]})
        right = DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
            merge(left, right, on='a')

    def test_merge_string_float_column_result(self) -> None:
        df1 = DataFrame([[1, 2], [3, 4]], columns=Index(['a', 114.0]))
        df2 = DataFrame([[9, 10], [11, 12]], columns=['x', 'y'])
        result = merge(df2, df1, how='inner', left_index=True, right_index=True)
        expected = DataFrame({
            'x': [9, 11],
            'y': [10, 12],
            'a': [1, 3],
            114.0: [2, 4]
        }, columns=Index(['x', 'y', 'a', 114.0]))
        tm.assert_frame_equal(result, expected)

    def test_mergeerror_on_left_index_mismatched_dtypes(self) -> None:
        df_1 = DataFrame(data=['X'], columns=['C'], index=[22])
        df_2 = DataFrame(data=['X'], columns=['C'], index=[999])
        with pytest.raises(MergeError, match='Can only pass argument'):
            merge(df_1, df_2, on=['C'], left_index=True)

    def test_merge_on_left_categoricalindex(self) -> None:
        ci = CategoricalIndex(range(3))
        right = DataFrame({'A': ci, 'B': range(3)})
        left = DataFrame({'C': range(3, 6)})
        res = merge(left, right, left_on=ci, right_on='A')
        expected = merge(left, right, left_on=ci._data, right_on='A')
        tm.assert_frame_equal(res, expected)

    def test_merge_duplicate_columns_with_suffix_no_warning(self) -> None:
        left = DataFrame({'a': [1, 1, 2], 'b': [2, 2, 2], 'b_x': [3, 3, 3]})
        right = DataFrame({'a': [1, 1, 3], 'b': [4, 4, 4]})
        result = merge(left, right, on='a', how='inner', suffixes=('_x', '_y'))
        expected = DataFrame({'a': [1, 1], 'b_x': [2, 2], 'b_y': [4, 4]})
        tm.assert_frame_equal(result, expected)

    def test_merge_suffixes_produce_dup_columns_raises(self) -> None:
        left = DataFrame({'a': [1, 2], 'b': [3, 4], 'b_y': [5, 6]})
        right = DataFrame({'a': [1, 2], 'b': [7, 8]})
        with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
            merge(left, right, on='a')
        with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
            merge(right, left, on='a', suffixes=('_y', '_x'))

    def test_merge_duplicate_columns_with_suffix_no_warning(self) -> None:
        left = DataFrame({'a': [1, 1, 1], 'b': [1, 1, 1]})
        right = DataFrame({'a': [1, 1, 1], 'b': [2, 2, 2]})
        result = merge(left, right, on='a', how='inner', suffixes=('_x', '_y'))
        expected = DataFrame({'a': [1, 1, 1], 'b_x': [1, 1, 1], 'b_y': [2, 2, 2]})
        tm.assert_frame_equal(result, expected)

    def test_merge_indicator_multiple_columns(self) -> None:
        df3 = DataFrame({'col1': [0, 1], 'col2': ['a', 'b']})
        df4 = DataFrame({'col1': [1, 1, 3], 'col2': ['b', 'x', 'y']})
        hand_coded_result = DataFrame({'col1': [0, 1, 1, 3], 'col2': ['a', 'b', 'x', 'y']})
        hand_coded_result['_merge'] = Categorical(['left_only', 'both', 'right_only', 'right_only'], categories=['left_only', 'right_only', 'both'])
        test5 = merge(df3, df4, on=['col1', 'col2'], how='outer', indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)
        test5 = df3.merge(df4, on=['col1', 'col2'], how='outer', indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)

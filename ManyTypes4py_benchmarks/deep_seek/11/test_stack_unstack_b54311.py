from datetime import datetime
import itertools
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest
from pandas._libs import lib
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Period, Series, Timedelta, date_range
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib

@pytest.fixture(params=[True, False])
def future_stack(request: pytest.FixtureRequest) -> bool:
    return request.param

class TestDataFrameReshape:

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unstack(self, float_frame: DataFrame, future_stack: bool) -> None:
        df = float_frame.copy()
        df[:] = np.arange(np.prod(df.shape)).reshape(df.shape)
        stacked = df.stack(future_stack=future_stack)
        stacked_df = DataFrame({'foo': stacked, 'bar': stacked})
        unstacked = stacked.unstack()
        unstacked_df = stacked_df.unstack()
        tm.assert_frame_equal(unstacked, df)
        tm.assert_frame_equal(unstacked_df['bar'], df)
        unstacked_cols = stacked.unstack(0)
        unstacked_cols_df = stacked_df.unstack(0)
        tm.assert_frame_equal(unstacked_cols.T, df)
        tm.assert_frame_equal(unstacked_cols_df['bar'].T, df)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_mixed_level(self, future_stack: bool) -> None:
        levels = [range(3), [3, 'a', 'b'], [1, 2]]
        df = DataFrame(1, index=levels[0], columns=levels[1])
        result = df.stack(future_stack=future_stack)
        expected = Series(1, index=MultiIndex.from_product(levels[:2]))
        tm.assert_series_equal(result, expected)
        df = DataFrame(1, index=levels[0], columns=MultiIndex.from_product(levels[1:]))
        result = df.stack(1, future_stack=future_stack)
        expected = DataFrame(1, index=MultiIndex.from_product([levels[0], levels[2]]), columns=levels[1])
        tm.assert_frame_equal(result, expected)
        result = df[['a', 'b']].stack(1, future_stack=future_stack)
        expected = expected[['a', 'b']]
        tm.assert_frame_equal(result, expected)

    def test_unstack_not_consolidated(self) -> None:
        df = DataFrame({'x': [1, 2, np.nan], 'y': [3.0, 4, np.nan]})
        df2 = df[['x']]
        df2['y'] = df['y']
        assert len(df2._mgr.blocks) == 2
        res = df2.unstack()
        expected = df.unstack()
        tm.assert_series_equal(res, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_fill(self, future_stack: bool) -> None:
        data = Series([1, 2, 4, 5], dtype=np.int16)
        data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
        result = data.unstack(fill_value=-1)
        expected = DataFrame({'a': [1, -1, 5], 'b': [2, 4, -1]}, index=['x', 'y', 'z'], dtype=np.int16)
        tm.assert_frame_equal(result, expected)
        result = data.unstack(fill_value=0.5)
        expected = DataFrame({'a': [1, 0.5, 5], 'b': [2, 4, 0.5]}, index=['x', 'y', 'z'], dtype=float)
        tm.assert_frame_equal(result, expected)
        df = DataFrame({'x': ['a', 'a', 'b'], 'y': ['j', 'k', 'j'], 'z': [0, 1, 2], 'w': [0, 1, 2]}).set_index(['x', 'y', 'z'])
        unstacked = df.unstack(['x', 'y'], fill_value=0)
        key = ('w', 'b', 'j')
        expected = unstacked[key]
        result = Series([0, 0, 2], index=unstacked.index, name=key)
        tm.assert_series_equal(result, expected)
        stacked = unstacked.stack(['x', 'y'], future_stack=future_stack)
        stacked.index = stacked.index.reorder_levels(df.index.names)
        stacked = stacked.astype(np.int64)
        result = stacked.loc[df.index]
        tm.assert_frame_equal(result, df)
        s = df['w']
        result = s.unstack(['x', 'y'], fill_value=0)
        expected = unstacked['w']
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame(self) -> None:
        rows = [[1, 2], [3, 4], [5, 6], [7, 8]]
        df = DataFrame(rows, columns=list('AB'), dtype=np.int32)
        df.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
        result = df.unstack(fill_value=-1)
        rows = [[1, 3, 2, 4], [-1, 5, -1, 6], [7, -1, 8, -1]]
        expected = DataFrame(rows, index=list('xyz'), dtype=np.int32)
        expected.columns = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')])
        tm.assert_frame_equal(result, expected)
        df['A'] = df['A'].astype(np.int16)
        df['B'] = df['B'].astype(np.float64)
        result = df.unstack(fill_value=-1)
        expected['A'] = expected['A'].astype(np.int16)
        expected['B'] = expected['B'].astype(np.float64)
        tm.assert_frame_equal(result, expected)
        result = df.unstack(fill_value=0.5)
        rows = [[1, 3, 2, 4], [0.5, 5, 0.5, 6], [7, 0.5, 8, 0.5]]
        expected = DataFrame(rows, index=list('xyz'), dtype=float)
        expected.columns = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')])
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_datetime(self) -> None:
        dv = date_range('2012-01-01', periods=4).values
        data = Series(dv)
        data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
        result = data.unstack()
        expected = DataFrame({'a': [dv[0], pd.NaT, dv[3]], 'b': [dv[1], dv[2], pd.NaT]}, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)
        result = data.unstack(fill_value=dv[0])
        expected = DataFrame({'a': [dv[0], dv[0], dv[3]], 'b': [dv[1], dv[2], dv[0]]}, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_timedelta(self) -> None:
        td = [Timedelta(days=i) for i in range(4)]
        data = Series(td)
        data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
        result = data.unstack()
        expected = DataFrame({'a': [td[0], pd.NaT, td[3]], 'b': [td[1], td[2], pd.NaT]}, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)
        result = data.unstack(fill_value=td[1])
        expected = DataFrame({'a': [td[0], td[1], td[3]], 'b': [td[1], td[2], td[1]]}, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_period(self) -> None:
        periods = [Period('2012-01'), Period('2012-02'), Period('2012-03'), Period('2012-04')]
        data = Series(periods)
        data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
        result = data.unstack()
        expected = DataFrame({'a': [periods[0], None, periods[3]], 'b': [periods[1], periods[2], None]}, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)
        result = data.unstack(fill_value=periods[1])
        expected = DataFrame({'a': [periods[0], periods[1], periods[3]], 'b': [periods[1], periods[2], periods[1]]}, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_categorical(self) -> None:
        data = Series(['a', 'b', 'c', 'a'], dtype='category')
        data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
        result = data.unstack()
        expected = DataFrame({'a': pd.Categorical(list('axa'), categories=list('abc')), 'b': pd.Categorical(list('bcx'), categories=list('abc'))}, index=list('xyz'))
        tm.assert_frame_equal(result, expected)
        msg = 'Cannot setitem on a Categorical with a new category \\(d\\)'
        with pytest.raises(TypeError, match=msg):
            data.unstack(fill_value='d')
        result = data.unstack(fill_value='c')
        expected = DataFrame({'a': pd.Categorical(list('aca'), categories=list('abc')), 'b': pd.Categorical(list('bcc'), categories=list('abc'))}, index=list('xyz'))
        tm.assert_frame_equal(result, expected)

    def test_unstack_tuplename_in_multiindex(self) -> None:
        idx = MultiIndex.from_product([['a', 'b', 'c'], [1, 2, 3]], names=[('A', 'a'), ('B', 'b')])
        df = DataFrame({'d': [1] * 9, 'e': [2] * 9}, index=idx)
        result = df.unstack(('A', 'a'))
        expected = DataFrame([[1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2]], columns=MultiIndex.from_tuples([('d', 'a'), ('d', 'b'), ('d', 'c'), ('e', 'a'), ('e', 'b'), ('e', 'c')], names=[None, ('A', 'a')]), index=Index([1, 2, 3], name=('B', 'b')))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('unstack_idx, expected_values, expected_index, expected_columns', [(('A', 'a'), [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]], MultiIndex.from_tuples([(1, 3), (1, 4), (2, 3), (2, 4)], names=['B', 'C']), MultiIndex.from_tuples([('d', 'a'), ('d', 'b'), ('e', 'a'), ('e', 'b')], names=[None, ('A', 'a')])), ((('A', 'a'), 'B'), [[1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 2, 2, 2, 2]], Index([3, 4], name='C'), MultiIndex.from_tuples([('d', 'a', 1), ('d', 'a', 2), ('d', 'b', 1), ('d', 'b', 2), ('e', 'a', 1), ('e', 'a', 2), ('e', 'b', 1), ('e', 'b', 2)], names=[None, ('A', 'a'), 'B']))])
    def test_unstack_mixed_type_name_in_multiindex(self, unstack_idx: Union[str, Tuple[str, str]], expected_values: List[List[Union[int, float]]], expected_index: Index, expected_columns: MultiIndex) -> None:
        idx = MultiIndex.from_product([['a', 'b'], [1, 2], [3, 4]], names=[('A', 'a'), 'B', 'C'])
        df = DataFrame({'d': [1] * 8, 'e': [2] * 8}, index=idx)
        result = df.unstack(unstack_idx)
        expected = DataFrame(expected_values, columns=expected_columns, index=expected_index)
        tm.assert_frame_equal(result, expected)

    def test_unstack_preserve_dtypes(self) -> None:
        df = DataFrame({'state': ['IL', 'MI', 'NC'], 'index': ['a', 'b', 'c'], 'some_categories': Series(['a', 'b', 'c']).astype('category'), 'A': np.random.default_rng(2).random(3), 'B': 1, 'C': 'foo', 'D': pd.Timestamp('20010102'), 'E': Series([1.0, 50.0, 100.0]).astype('float32'), 'F': Series([3.0, 4.0, 5.0]).astype('float64'), 'G': False, 'H': Series([1, 200, 923442]).astype('int8')})

        def unstack_and_compare(df: DataFrame, column_name: str) -> None:
            unstacked1 = df.unstack([column_name])
            unstacked2 = df.unstack(column_name)
            tm.assert_frame_equal(unstacked1, unstacked2)
        df1 = df.set_index(['state', 'index'])
        unstack_and_compare(df1, 'index')
        df1 = df.set_index(['state', 'some_categories'])
        unstack_and_compare(df1, 'some_categories')
        df1 = df.set_index(['F', 'C'])
        unstack_and_compare(df1, 'F')
        df1 = df.set_index(['G', 'B', 'state'])
        unstack_and_compare(df1, 'B')
        df1 = df.set_index(['E', 'A'])
        unstack_and_compare(df1, 'E')
        df1 = df.set_index(['state', 'index'])
        s = df1['A']
        unstack_and_compare(s, 'index')

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_ints(self, future_stack: bool) -> None:
        columns = MultiIndex.from_tuples(list(itertools.product(range(3), repeat=3)))
        df = DataFrame(np.random.default_rng(2).standard_normal((30, 27)), columns=columns)
        tm.assert_frame_equal(df.stack(level=[1, 2], future_stack=future_stack), df.stack(level=1, future_stack=future_stack).stack(level=1, future_stack=future_stack))
        tm.assert_frame_equal(df.stack(level=[-2, -1], future_stack=future_stack), df.stack(level=1, future_stack=future_stack).stack(level=1, future_stack=future_stack))
        df_named = df.copy()
        return_value = df_named.columns.set_names(range(3), inplace=True)
        assert return_value is None
        tm.assert_frame_equal(df_named.stack(level=[1, 2], future_stack=future_stack), df_named.stack(level=1, future_stack=future_stack).stack(level=1, future_stack=future_stack))

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_mixed_levels(self, future_stack: bool) -> None:
        columns = MultiIndex.from_tuples([('A', 'cat', 'long'), ('B', 'cat', 'long'), ('A', 'dog', 'short'), ('B', 'dog', 'short')], names=['exp', 'animal', 'hair_length'])
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), columns=columns)
        animal_hair_stacked = df.stack(level=['animal', 'hair_length'], future_stack=future_stack)
        exp_hair_stacked = df.stack(level=['exp', 'hair_length'], future_stack=future_stack)
        df2 = df.copy()
        df2.columns.names = ['exp', 'animal', 1]
        tm.assert_frame_equal(df2.stack(level=['animal', 1], future_stack=future_stack), animal_hair_stacked, check_names=False)
        tm.assert_frame_equal(df2.stack(level=['exp', 1], future_stack=future_stack), exp_hair_stacked, check_names=False)
        msg = 'level should contain all level names or all level numbers, not a mixture of
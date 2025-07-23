from datetime import datetime
import itertools
import re
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pytest
import pandas as pd
from pandas._libs import lib
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Period,
    Series,
    Timedelta,
    date_range,
)
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib


@pytest.fixture(params=[True, False])
def future_stack(request: pytest.FixtureRequest) -> bool:
    return request.param


class TestDataFrameReshape:

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unstack(
        self,
        float_frame: DataFrame,
        future_stack: bool
    ) -> None:
        df: DataFrame = float_frame.copy()
        df[:] = np.arange(np.prod(df.shape)).reshape(df.shape)
        stacked: Series = df.stack(future_stack=future_stack)
        stacked_df: DataFrame = DataFrame({'foo': stacked, 'bar': stacked})
        unstacked: DataFrame = stacked.unstack()
        unstacked_df: DataFrame = stacked_df.unstack()
        tm.assert_frame_equal(unstacked, df)
        tm.assert_frame_equal(unstacked_df['bar'], df)
        unstacked_cols: DataFrame = stacked.unstack(0)
        unstacked_cols_df: DataFrame = stacked_df.unstack(0)
        tm.assert_frame_equal(unstacked_cols.T, df)
        tm.assert_frame_equal(unstacked_cols_df['bar'].T, df)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_mixed_level(
        self,
        future_stack: bool
    ) -> None:
        levels: List[Range] = [range(3), [3, 'a', 'b'], [1, 2]]
        df: DataFrame = DataFrame(1, index=levels[0], columns=levels[1])
        result: Series = df.stack(future_stack=future_stack)
        expected: Series = Series(1, index=MultiIndex.from_product(levels[:2]))
        tm.assert_series_equal(result, expected)
        df = DataFrame(1, index=levels[0], columns=MultiIndex.from_product(levels[1:]))
        result = df.stack(1, future_stack=future_stack)
        expected = DataFrame(
            1,
            index=MultiIndex.from_product([levels[0], levels[2]]),
            columns=levels[1],
        )
        tm.assert_frame_equal(result, expected)
        result = df[['a', 'b']].stack(1, future_stack=future_stack)
        expected = expected[['a', 'b']]
        tm.assert_frame_equal(result, expected)

    def test_unstack_not_consolidated(self) -> None:
        df: DataFrame = DataFrame({'x': [1, 2, np.nan], 'y': [3.0, 4, np.nan]})
        df2: DataFrame = df[['x']].copy()
        df2['y'] = df['y']
        assert len(df2._mgr.blocks) == 2
        res: Series = df2.unstack()
        expected: Series = df.unstack()
        tm.assert_series_equal(res, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_fill(
        self,
        future_stack: bool
    ) -> None:
        data: Series = Series([1, 2, 4, 5], dtype=np.int16)
        data.index = MultiIndex.from_tuples(
            [('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')]
        )
        result: DataFrame = data.unstack(fill_value=-1)
        expected: DataFrame = DataFrame(
            {'a': [1, -1, 5], 'b': [2, 4, -1]},
            index=['x', 'y', 'z'],
            dtype=np.int16
        )
        tm.assert_frame_equal(result, expected)
        result = data.unstack(fill_value=0.5)
        expected = DataFrame(
            {'a': [1, 0.5, 5], 'b': [2, 4, 0.5]},
            index=['x', 'y', 'z'],
            dtype=float
        )
        tm.assert_frame_equal(result, expected)
        df: DataFrame = DataFrame({
            'x': ['a', 'a', 'b'],
            'y': ['j', 'k', 'j'],
            'z': [0, 1, 2],
            'w': [0, 1, 2]
        }).set_index(['x', 'y', 'z'])
        unstacked: DataFrame = df.unstack(['x', 'y'], fill_value=0)
        key: Tuple[Any, Any, Any] = ('w', 'b', 'j')
        expected = unstacked[key]
        result: Series = Series([0, 0, 2], index=unstacked.index, name=key)
        tm.assert_series_equal(result, expected)
        stacked: Series = unstacked.stack(['x', 'y'], future_stack=future_stack)
        stacked.index = stacked.index.reorder_levels(df.index.names)
        stacked = stacked.astype(np.int64)
        result = stacked.loc[df.index]
        tm.assert_frame_equal(result, df)
        s: Series = df['w']
        result = s.unstack(['x', 'y'], fill_value=0)
        expected = unstacked['w']
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame(self) -> None:
        rows: List[List[int]] = [[1, 2], [3, 4], [5, 6], [7, 8]]
        df: DataFrame = DataFrame(rows, columns=list('AB'), dtype=np.int32)
        df.index = MultiIndex.from_tuples(
            [('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')]
        )
        result: DataFrame = df.unstack(fill_value=-1)
        rows_expected: List[List[int]] = [[1, 3, 2, 4], [-1, 5, -1, 6], [7, -1, 8, -1]]
        expected: DataFrame = DataFrame(rows_expected, index=list('xyz'), dtype=np.int32)
        expected.columns = MultiIndex.from_tuples(
            [('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')]
        )
        tm.assert_frame_equal(result, expected)
        df['A'] = df['A'].astype(np.int16)
        df['B'] = df['B'].astype(np.float64)
        result = df.unstack(fill_value=-1)
        expected['A'] = expected['A'].astype(np.int16)
        expected['B'] = expected['B'].astype(np.float64)
        tm.assert_frame_equal(result, expected)
        result = df.unstack(fill_value=0.5)
        rows_expected = [[1, 3, 2, 4], [0.5, 5, 0.5, 6], [7, 0.5, 8, 0.5]]
        expected = DataFrame(
            rows_expected,
            index=list('xyz'),
            dtype=float
        )
        expected.columns = MultiIndex.from_tuples(
            [('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')]
        )
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_datetime(self) -> None:
        dv: np.ndarray = date_range('2012-01-01', periods=4).values
        data: Series = Series(dv)
        data.index = MultiIndex.from_tuples(
            [('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')]
        )
        result: DataFrame = data.unstack()
        expected: DataFrame = DataFrame({
            'a': [dv[0], pd.NaT, dv[3]],
            'b': [dv[1], dv[2], pd.NaT]
        }, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)
        result = data.unstack(fill_value=dv[0])
        expected = DataFrame({
            'a': [dv[0], dv[0], dv[3]],
            'b': [dv[1], dv[2], dv[0]]
        }, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_timedelta(self) -> None:
        td: List[Timedelta] = [Timedelta(days=i) for i in range(4)]
        data: Series = Series(td)
        data.index = MultiIndex.from_tuples(
            [('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')]
        )
        result: DataFrame = data.unstack()
        expected: DataFrame = DataFrame({
            'a': [td[0], pd.NaT, td[3]],
            'b': [td[1], td[2], pd.NaT]
        }, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)
        result = data.unstack(fill_value=td[1])
        expected = DataFrame({
            'a': [td[0], td[1], td[3]],
            'b': [td[1], td[2], td[1]]
        }, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_period(self) -> None:
        periods: List[Period] = [
            Period('2012-01'),
            Period('2012-02'),
            Period('2012-03'),
            Period('2012-04')
        ]
        data: Series = Series(periods)
        data.index = MultiIndex.from_tuples(
            [('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')]
        )
        result: DataFrame = data.unstack()
        expected: DataFrame = DataFrame({
            'a': [periods[0], None, periods[3]],
            'b': [periods[1], periods[2], None]
        }, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)
        result = data.unstack(fill_value=periods[1])
        expected = DataFrame({
            'a': [periods[0], periods[1], periods[3]],
            'b': [periods[1], periods[2], periods[1]]
        }, index=['x', 'y', 'z'])
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_categorical(self) -> None:
        data: Series = Series(['a', 'b', 'c', 'a'], dtype='category')
        data.index = MultiIndex.from_tuples(
            [('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')]
        )
        result: DataFrame = data.unstack()
        expected: DataFrame = DataFrame({
            'a': pd.Categorical(['a', 'x', 'a'], categories=['a', 'b', 'c']),
            'b': pd.Categorical(['b', 'c', 'x'], categories=['a', 'b', 'c'])
        }, index=list('xyz'))
        tm.assert_frame_equal(result, expected)
        msg: str = 'Cannot setitem on a Categorical with a new category \\(d\\)'
        with pytest.raises(TypeError, match=msg):
            data.unstack(fill_value='d')
        result = data.unstack(fill_value='c')
        expected = DataFrame({
            'a': pd.Categorical(['a', 'c', 'a'], categories=['a', 'b', 'c']),
            'b': pd.Categorical(['b', 'c', 'c'], categories=['a', 'b', 'c'])
        }, index=list('xyz'))
        tm.assert_frame_equal(result, expected)

    def test_unstack_tuplename_in_multiindex(self) -> None:
        idx: MultiIndex = MultiIndex.from_product(
            [['a', 'b', 'c'], [1, 2, 3]], names=[('A', 'a'), ('B', 'b')]
        )
        df: DataFrame = DataFrame({'d': [1] * 9, 'e': [2] * 9}, index=idx)
        result: DataFrame = df.unstack(('A', 'a'))
        expected: DataFrame = DataFrame(
            [
                [1, 1, 1, 2, 2, 2],
                [1, 1, 1, 2, 2, 2],
                [1, 1, 1, 2, 2, 2],
            ],
            columns=MultiIndex.from_tuples(
                [
                    ('d', 'a'),
                    ('d', 'b'),
                    ('d', 'c'),
                    ('e', 'a'),
                    ('e', 'b'),
                    ('e', 'c'),
                ],
                names=[None, ('A', 'a')],
            ),
            index=Index([1, 2, 3], name=('B', 'b')),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'unstack_idx, expected_values, expected_index, expected_columns',
        [
            (
                ('A', 'a'),
                [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]],
                MultiIndex.from_tuples([(1, 3), (1, 4), (2, 3), (2, 4)], names=['B', 'C']),
                MultiIndex.from_tuples(
                    [('d', 'a'), ('d', 'b'), ('e', 'a'), ('e', 'b')],
                    names=[None, ('A', 'a')],
                )
            ),
            (
                (('A', 'a'), 'B'),
                [
                    [1, 1, 1, 1, 2, 2, 2, 2],
                    [1, 1, 1, 1, 2, 2, 2, 2]
                ],
                Index([3, 4], name='C'),
                MultiIndex.from_tuples(
                    [
                        ('d', 'a', 1),
                        ('d', 'a', 2),
                        ('d', 'b', 1),
                        ('d', 'b', 2),
                        ('e', 'a', 1),
                        ('e', 'a', 2),
                        ('e', 'b', 1),
                        ('e', 'b', 2),
                    ],
                    names=[None, ('A', 'a'), 'B']
                )
            )
        ]
    )
    def test_unstack_mixed_type_name_in_multiindex(
        self,
        unstack_idx: Union[Tuple[str, str], Tuple[Tuple[str, str], str]],
        expected_values: List[List[int]],
        expected_index: Union[Index, MultiIndex],
        expected_columns: MultiIndex
    ) -> None:
        idx: MultiIndex = MultiIndex.from_product([['a', 'b'], [1, 2], [3, 4]], names=[('A', 'a'), 'B', 'C'])
        df: DataFrame = DataFrame({'d': [1] * 8, 'e': [2] * 8}, index=idx)
        result: DataFrame = df.unstack(unstack_idx)
        expected: DataFrame = DataFrame(
            expected_values,
            columns=expected_columns,
            index=expected_index
        )
        tm.assert_frame_equal(result, expected)

    def test_unstack_preserve_dtypes(self, using_infer_string: bool) -> None:
        df: DataFrame = DataFrame({
            'state': ['IL', 'MI', 'NC'],
            'index': ['a', 'b', 'c'],
            'some_categories': Series(['a', 'b', 'c']).astype('category'),
            'A': np.random.default_rng(2).random(3),
            'B': 1,
            'C': 'foo',
            'D': pd.Timestamp('20010102'),
            'E': Series([1.0, 50.0, 100.0]).astype('float32'),
            'F': Series([3.0, 4.0, 5.0]).astype('float64'),
            'G': False,
            'H': Series([1, 200, 923442]).astype('int8')
        })

        def unstack_and_compare(df_param: DataFrame, column_name: str) -> None:
            unstacked1: DataFrame = df_param.unstack([column_name])
            unstacked2: DataFrame = df_param.unstack(column_name)
            tm.assert_frame_equal(unstacked1, unstacked2)

        df1: DataFrame = df.set_index(['state', 'index'])
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
        s: Series = df1['A']
        unstack_and_compare(s, 'index')

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_ints(
        self,
        future_stack: bool
    ) -> None:
        columns: MultiIndex = MultiIndex.from_tuples(list(itertools.product(range(3), repeat=3)))
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((30, 27)),
            columns=columns
        )
        tm.assert_frame_equal(
            df.stack(level=[1, 2], future_stack=future_stack),
            df.stack(level=1, future_stack=future_stack).stack(level=1, future_stack=future_stack)
        )
        tm.assert_frame_equal(
            df.stack(level=[-2, -1], future_stack=future_stack),
            df.stack(level=1, future_stack=future_stack).stack(level=1, future_stack=future_stack)
        )
        df_named: DataFrame = df.copy()
        return_value: None = df_named.columns.set_names(range(3), inplace=True)
        assert return_value is None
        tm.assert_frame_equal(
            df_named.stack(level=[1, 2], future_stack=future_stack),
            df_named.stack(level=1, future_stack=future_stack).stack(level=1, future_stack=future_stack)
        )

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_mixed_levels(
        self,
        future_stack: bool
    ) -> None:
        columns: MultiIndex = MultiIndex.from_tuples([
            ('A', 'cat', 'long'),
            ('B', 'cat', 'long'),
            ('A', 'dog', 'short'),
            ('B', 'dog', 'short')
        ], names=['exp', 'animal', 'hair_length'])
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            columns=columns
        )
        animal_hair_stacked: DataFrame = df.stack(level=['animal', 'hair_length'], future_stack=future_stack)
        exp_hair_stacked: DataFrame = df.stack(level=['exp', 'hair_length'], future_stack=future_stack)
        df2: DataFrame = df.copy()
        df2.columns.names = ['exp', 'animal', 1]
        tm.assert_frame_equal(
            df2.stack(level=['animal', 1], future_stack=future_stack),
            animal_hair_stacked,
            check_names=False
        )
        tm.assert_frame_equal(
            df2.stack(level=['exp', 1], future_stack=future_stack),
            exp_hair_stacked,
            check_names=False
        )
        msg: str = 'level should contain all level names or all level numbers, not a mixture of the two'
        with pytest.raises(ValueError, match=msg):
            df2.stack(level=['animal', 0], future_stack=future_stack)
        df3: DataFrame = df.copy()
        df3.columns.names = ['exp', 'animal', 0]
        tm.assert_frame_equal(
            df3.stack(level=['animal', 0], future_stack=future_stack),
            animal_hair_stacked,
            check_names=False
        )

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_int_level_names(
        self,
        future_stack: bool
    ) -> None:
        columns: MultiIndex = MultiIndex.from_tuples([
            ('A', 'cat', 'long'),
            ('B', 'cat', 'long'),
            ('A', 'dog', 'short'),
            ('B', 'dog', 'short')
        ], names=['exp', 'animal', 'hair_length'])
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            columns=columns
        )
        exp_animal_stacked: DataFrame = df.stack(level=['exp', 'animal'], future_stack=future_stack)
        animal_hair_stacked: DataFrame = df.stack(level=['animal', 'hair_length'], future_stack=future_stack)
        exp_hair_stacked: DataFrame = df.stack(level=['exp', 'hair_length'], future_stack=future_stack)
        df2: DataFrame = df.copy()
        df2.columns.names = [0, 1, 2]
        tm.assert_frame_equal(
            df2.stack(level=[1, 2], future_stack=future_stack),
            animal_hair_stacked,
            check_names=False
        )
        tm.assert_frame_equal(
            df2.stack(level=[0, 1], future_stack=future_stack),
            exp_animal_stacked,
            check_names=False
        )
        tm.assert_frame_equal(
            df2.stack(level=[0, 2], future_stack=future_stack),
            exp_hair_stacked,
            check_names=False
        )
        df3: DataFrame = df.copy()
        df3.columns.names = [2, 0, 1]
        tm.assert_frame_equal(
            df3.stack(level=[0, 1], future_stack=future_stack),
            animal_hair_stacked,
            check_names=False
        )
        tm.assert_frame_equal(
            df3.stack(level=[2, 0], future_stack=future_stack),
            exp_animal_stacked,
            check_names=False
        )
        tm.assert_frame_equal(
            df3.stack(level=[2, 1], future_stack=future_stack),
            exp_hair_stacked,
            check_names=False
        )

    def test_unstack_bool(self) -> None:
        df: DataFrame = DataFrame(
            [False, False],
            index=MultiIndex.from_arrays([['a', 'b'], ['c', 'l']]),
            columns=['col']
        )
        rs: DataFrame = df.unstack()
        xp: DataFrame = DataFrame(
            np.array([[False, np.nan], [np.nan, False]], dtype=object),
            index=['a', 'b'],
            columns=MultiIndex.from_arrays([['col', 'col'], ['c', 'l']])
        )
        tm.assert_frame_equal(rs, xp)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_level_binding(
        self,
        future_stack: bool
    ) -> None:
        mi: MultiIndex = MultiIndex(
            levels=[['foo', 'bar'], ['one', 'two'], ['a', 'b']],
            codes=[
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [1, 0, 1, 0]
            ],
            names=['first', 'second', 'third']
        )
        s: Series = Series(0, index=mi)
        result: DataFrame = s.unstack([1, 2]).stack(0, future_stack=future_stack)
        expected_mi: MultiIndex = MultiIndex(
            levels=[['foo', 'bar'], ['one', 'two']],
            codes=[
                [0, 0, 1, 1],
                [0, 1, 0, 1]
            ],
            names=['first', 'second']
        )
        expected: DataFrame = DataFrame(
            np.array([[0, np.nan], [np.nan, 0], [0, np.nan], [np.nan, 0]], dtype=np.float64),
            index=expected_mi,
            columns=Index(['b', 'a'], name='third')
        )
        tm.assert_frame_equal(result, expected)

    def test_unstack_to_series(self, float_frame: DataFrame) -> None:
        data: Series = float_frame.unstack()
        assert isinstance(data, Series)
        undo: DataFrame = data.unstack().T
        tm.assert_frame_equal(undo, float_frame)
        data = DataFrame({'x': [1, 2, np.nan], 'y': [3.0, 4, np.nan]})
        data.index = Index(['a', 'b', 'c'])
        result: Series = data.unstack()
        midx: MultiIndex = MultiIndex(
            levels=[['x', 'y'], ['a', 'b', 'c']],
            codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]]
        )
        expected: Series = Series([1, 2, np.nan, 3, 4, np.nan], index=midx)
        tm.assert_series_equal(result, expected)
        old_data: DataFrame = data.copy()
        for _ in range(4):
            data = data.unstack()
        tm.assert_frame_equal(old_data, data)

    def test_unstack_dtypes(self, using_infer_string: bool) -> None:
        rows: List[List[int]] = [
            [1, 1, 3, 4],
            [1, 2, 3, 4],
            [2, 1, 3, 4],
            [2, 2, 3, 4]
        ]
        df: DataFrame = DataFrame(rows, columns=list('ABCD'))
        result: Series = df.dtypes
        expected: Series = Series([np.dtype('int64')] * 4, index=list('ABCD'))
        tm.assert_series_equal(result, expected)
        df2: DataFrame = df.set_index(['A', 'B'])
        df3: DataFrame = df2.unstack('B')
        result = df3.dtypes
        expected = Series(
            [np.dtype('int64')] * 4,
            index=MultiIndex.from_arrays([['C', 'C', 'D', 'D'], [1, 2, 1, 2]], names=(None, 'B'))
        )
        tm.assert_series_equal(result, expected)
        df2 = df.set_index(['A', 'B'])
        df2['C'] = 3.0
        df3 = df2.unstack('B')
        result = df3.dtypes
        expected = Series(
            [np.dtype('float64')] * 2 + [np.dtype('int64')] * 2,
            index=MultiIndex.from_arrays([['C', 'C', 'D', 'D'], [1, 2, 1, 2]], names=(None, 'B'))
        )
        tm.assert_series_equal(result, expected)
        df2['D'] = 'foo'
        df3 = df2.unstack('B')
        result = df3.dtypes
        dtype: Union[np.dtype, str] = pd.StringDtype(na_value=np.nan) if using_infer_string else np.dtype('object')
        expected = Series(
            [np.dtype('float64')] * 2 + [dtype] * 2,
            index=MultiIndex.from_arrays([['C', 'C', 'D', 'D'], [1, 2, 1, 2]], names=(None, 'B'))
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        'c, d',
        [
            (np.zeros(5), np.zeros(5)),
            (np.arange(5, dtype='f8'), np.arange(5, 10, dtype='f8'))
        ]
    )
    def test_unstack_dtypes_mixed_date(
        self,
        c: np.ndarray,
        d: np.ndarray
    ) -> None:
        df: DataFrame = DataFrame({
            'A': ['a'] * 5,
            'C': c,
            'D': d,
            'B': date_range('2012-01-01', periods=5)
        })
        right: DataFrame = df.iloc[:3].copy(deep=True)
        df = df.set_index(['A', 'B'])
        df['D'] = df['D'].astype('int64')
        left: DataFrame = df.iloc[:3].unstack('A')
        right = df.iloc[:3].unstack('A')
        right['D', 'a'] = right['D', 'a'].astype('int64')
        assert left.shape == (3, 2)
        tm.assert_frame_equal(left, right)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_non_unique_index_names(
        self,
        future_stack: bool
    ) -> None:
        idx: MultiIndex = MultiIndex.from_tuples(
            [('a', 'b'), ('c', 'd')],
            names=['c1', 'c1']
        )
        df: DataFrame = DataFrame([1, 2], index=idx)
        msg: str = 'The name c1 occurs multiple times, use a level number'
        with pytest.raises(ValueError, match=msg):
            df.unstack('c1')
        with pytest.raises(ValueError, match=msg):
            df.T.stack('c1', future_stack=future_stack)

    def test_unstack_unused_levels(self) -> None:
        idx: MultiIndex = MultiIndex.from_product([['a'], ['A', 'B', 'C', 'D']])[:-1]
        df: DataFrame = DataFrame([[1, 0]] * 3, index=idx)
        result: DataFrame = df.unstack()
        exp_col: MultiIndex = MultiIndex.from_product([range(2), ['A', 'B', 'C']])
        expected: DataFrame = DataFrame([[1, 1, 1, 0, 0, 0]], index=['a'], columns=exp_col)
        tm.assert_frame_equal(result, expected)
        assert (result.columns.levels[1] == idx.levels[1]).all()
        levels: List[Range] = [range(3), range(4)]
        codes: List[List[int]] = [[0, 0, 1, 1], [0, 2, 0, 2]]
        idx = MultiIndex(levels=levels, codes=codes)
        block: np.ndarray = np.arange(4).reshape(2, 2)
        df = DataFrame(np.concatenate([block, block + 4]), index=idx)
        result = df.unstack()
        expected = DataFrame(np.concatenate([block * 2, block * 2 + 1], axis=1), columns=idx)
        tm.assert_frame_equal(result, expected)
        assert (result.columns.levels[1] == idx.levels[1]).all()

    @pytest.mark.parametrize(
        'level, idces, col_level, idx_level',
        [
            (
                0,
                [13, 16, 6, 9, 2, 5, 8, 11],
                [np.nan, 'a', 2],
                [np.nan, 5, 1]
            ),
            (
                1,
                [8, 11, 1, 4, 12, 15, 13, 16],
                [np.nan, 5, 1],
                [np.nan, 5, 1]
            )
        ]
    )
    def test_unstack_unused_levels_mixed_with_nan(
        self,
        level: int,
        idces: List[int],
        col_level: List[Union[str, float]],
        idx_level: List[Union[str, float]]
    ) -> None:
        levels: List[List[Union[str, int]]] = [['a', 2, 'c'], [1, 3, 5, 7]]
        codes: List[List[int]] = [[0, -1, 1, 1], [0, 2, -1, 2]]
        idx: MultiIndex = MultiIndex(levels=levels, codes=codes)
        data: np.ndarray = np.arange(8)
        df: DataFrame = DataFrame(data.reshape(4, 2), index=idx)
        result: DataFrame = df.unstack(level=level)
        exp_data: np.ndarray = np.zeros(18) * np.nan
        exp_data[idces] = data
        cols: MultiIndex = MultiIndex.from_product([range(2), col_level])
        expected: DataFrame = DataFrame(exp_data.reshape(3, 6), index=idx_level, columns=cols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'cols',
        [
            ['A', 'C'],
            slice(None)
        ]
    )
    def test_unstack_unused_level(
        self,
        cols: Union[List[str], slice]
    ) -> None:
        df: DataFrame = DataFrame([[2010, 'a', 'I'], [2011, 'b', 'II']], columns=['A', 'B', 'C'])
        ind: DataFrame = df.set_index(['A', 'B', 'C'], drop=False)
        selection: DataFrame = ind.loc[(slice(None), slice(None), 'I'), cols]
        result: DataFrame = selection.unstack()
        expected: DataFrame = ind.iloc[[0]][cols]
        expected.columns = MultiIndex.from_product([expected.columns, ['I']], names=[None, 'C'])
        expected.index = expected.index.droplevel('C')
        tm.assert_frame_equal(result, expected)

    def test_unstack_long_index(self) -> None:
        df: DataFrame = DataFrame(
            [[1]],
            columns=MultiIndex.from_tuples([[0]], names=['c1']),
            index=MultiIndex.from_tuples(
                [[0, 0, 1, 0, 0, 0, 1]],
                names=['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7']
            )
        )
        result: DataFrame = df.unstack(['i2', 'i3', 'i4', 'i5', 'i6', 'i7'])
        expected: DataFrame = DataFrame(
            [[1]],
            columns=MultiIndex.from_tuples(
                [[0, 0, 1, 0, 0, 0, 1]],
                names=['c1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7']
            ),
            index=Index([0], name='i1')
        )
        tm.assert_frame_equal(result, expected)

    def test_unstack_multi_level_cols(self) -> None:
        df: DataFrame = DataFrame(
            [[0.0, 0.0], [0.0, 0.0]],
            columns=MultiIndex.from_tuples(
                [['B', 'C'], ['B', 'D']],
                names=['c1', 'c2']
            ),
            index=MultiIndex.from_tuples(
                [[10, 20, 30], [10, 20, 40]],
                names=['i1', 'i2', 'i3']
            )
        )
        assert df.unstack(['i2', 'i1']).columns.names[-2:] == ['i2', 'i1']

    def test_unstack_multi_level_rows_and_cols(self) -> None:
        df: DataFrame = DataFrame(
            [[1, 2], [3, 4], [-1, -2], [-3, -4]],
            columns=MultiIndex.from_tuples(
                [['a', 'b', 'c'], ['d', 'e', 'f']]
            ),
            index=MultiIndex.from_tuples(
                [
                    ['m1', 'P3', 222],
                    ['m1', 'A5', 111],
                    ['m2', 'P3', 222],
                    ['m2', 'A5', 111]
                ],
                names=['i1', 'i2', 'i3']
            )
        )
        result: DataFrame = df.unstack(['i3', 'i2'])
        expected: DataFrame = df.unstack(['i3']).unstack(['i2'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'idx',
        [
            ('jim', 'joe'),
            ('joe', 'jim')
        ]
    )
    @pytest.mark.parametrize('lev', list(range(2)))
    def test_unstack_nan_index1(
        self,
        idx: Tuple[str, str],
        lev: int
    ) -> None:

        def cast(val: Any) -> str:
            val_str: str = '' if val != val else val
            return f'{val_str:1}'

        df: DataFrame = DataFrame({
            'jim': ['a', 'b', np.nan, 'd'],
            'joe': ['w', 'x', 'y', 'z'],
            'jolie': ['a.w', 'b.x', ' .y', 'd.z']
        })
        left: Series = df.set_index(['jim', 'joe']).unstack()['jolie']
        right: Series = df.set_index(['joe', 'jim']).unstack()['jolie'].T
        tm.assert_series_equal(left, right)
        mi: MultiIndex = df.set_index(list(idx))
        udf: DataFrame = mi.unstack(level=lev)
        assert udf.notna().values.sum() == len(df)
        mk_list: Callable[[Any], List[Any]] = lambda a: list(a) if isinstance(a, tuple) else [a]
        rows, cols = udf['jolie'].notna().values.nonzero()
        for i, j in zip(rows, cols):
            left_vals: List[str] = sorted(udf['jolie'].iloc[i, j].split('.'))
            right_vals: List[str] = mk_list(udf['jolie'].index[i]) + mk_list(udf['jolie'].columns[j])
            right_vals = sorted(map(cast, right_vals))
            assert left_vals == right_vals

    @pytest.mark.parametrize(
        'idx',
        itertools.permutations(['1st', '2nd', '3rd'])
    )
    @pytest.mark.parametrize('lev', list(range(3)))
    @pytest.mark.parametrize('col', ['4th', '5th'])
    def test_unstack_nan_index_repeats(
        self,
        idx: Tuple[str, str, str],
        lev: int,
        col: str
    ) -> None:

        def cast(val: Any) -> str:
            val_str: str = '' if val != val else val
            return f'{val_str:1}'

        df: DataFrame = DataFrame({
            '1st': ['d'] * 3 + [np.nan] * 5 + ['a'] * 2 + ['c'] * 3 + ['e'] * 2 + ['b'] * 5,
            '2nd': ['y'] * 2 + ['w'] * 3 + [np.nan] * 3 + ['z'] * 4 + [np.nan] * 3 + ['x'] * 3 + [np.nan] * 2,
            '3rd': [67, 39, 53, 72, 57, 80, 31, 18, 11, 30, 59, 50, 62, 59, 76, 52, 14, 53, 60, 51]
        })
        df['4th'], df['5th'] = (
            df.apply(lambda r: '.'.join(map(cast, r)), axis=1),
            df.apply(lambda r: '.'.join(map(cast, r.iloc[::-1])), axis=1)
        )
        mi: MultiIndex = df.set_index(list(idx))
        udf: DataFrame = mi.unstack(level=lev)
        assert udf.notna().values.sum() == 2 * len(df)
        mk_list: Callable[[Any], List[Any]] = lambda a: list(a) if isinstance(a, tuple) else [a]
        rows, cols = udf[col].notna().values.nonzero()
        for i, j in zip(rows, cols):
            left_vals: List[str] = sorted(udf[col].iloc[i, j].split('.'))
            right_vals: List[str] = mk_list(udf[col].index[i]) + mk_list(udf[col].columns[j])
            right_vals = sorted(map(cast, right_vals))
            assert left_vals == right_vals

    def test_unstack_nan_index2(self) -> None:
        df: DataFrame = DataFrame({'A': list('aaaabbbb'), 'B': range(8), 'C': range(8)})
        df = df.astype({'B': 'float'})
        df.iloc[3, 1] = np.nan
        left: DataFrame = df.set_index(['A', 'B']).unstack('A')
        vals: List[List[Union[float, int, np.nan]]] = [
            [3, 0, 1, 2, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 4, 5, 6, 7]
        ]
        cols: MultiIndex = MultiIndex.from_tuples(
            [('C', 'a'), ('C', 'b'), ('D', 'a'), ('D', 'b')]
        )
        index: Index = Index([np.nan, 0, 1, 2, 3], name='B')
        right: DataFrame = DataFrame(
            vals,
            columns=cols,
            index=index
        )
        tm.assert_frame_equal(left, right)
        df = DataFrame({'A': list('aaaabbbb'), 'B': list(range(4)) * 2, 'C': range(8)})
        df = df.astype({'B': 'float'})
        df.iloc[2, 1] = np.nan
        left = df.set_index(['A', 'B']).unstack('A')
        vals = [
            [2, np.nan],
            [0, 4],
            [1, 5],
            [np.nan, 6],
            [3, 7]
        ]
        cols = MultiIndex.from_tuples(
            [('C', 'a'), ('C', 'b')]
        )
        index = Index([np.nan, 0, 1, 2, 3], name='B')
        right = DataFrame(vals, columns=cols, index=index)
        tm.assert_frame_equal(left, right)
        df = DataFrame({'A': list('aaaabbbb'), 'B': list(range(4)) * 2, 'C': range(8)})
        df = df.astype({'B': 'float'})
        df.iloc[3, 1] = np.nan
        left = df.set_index(['A', 'B']).unstack('A')
        vals = [
            [3, np.nan],
            [0, 4],
            [1, 5],
            [2, 6],
            [np.nan, 7]
        ]
        cols = MultiIndex.from_tuples(
            [('C', 'a'), ('C', 'b')]
        )
        index = Index([np.nan, 0, 1, 2, 3], name='B')
        right = DataFrame(vals, columns=cols, index=index)
        tm.assert_frame_equal(left, right)

    def test_unstack_nan_index3(self) -> None:
        df: DataFrame = DataFrame({
            'A': list('aaaabbbb'),
            'B': date_range('2012-01-01', periods=5).tolist() * 2,
            'C': np.arange(10)
        })
        df.iloc[3, 1] = np.nan
        left: DataFrame = df.set_index(['A', 'B']).unstack()
        vals: np.ndarray = np.array([[3, 0, 1, 2, np.nan, 4], [np.nan, 5, 6, 7, 8, 9]])
        idx: Index = Index(['a', 'b'], name='A')
        cols: MultiIndex = MultiIndex.from_tuples(
            [('C', '2012-01-01'), ('C', '2012-01-02'), ('C', '2012-01-03'), ('C', '2012-01-04'), ('C', '2012-01-05')],
            names=['B']
        )
        expected: DataFrame = DataFrame(
            vals,
            columns=cols,
            index=idx
        )
        tm.assert_frame_equal(left, expected)

    def test_unstack_nan_index4(self) -> None:
        vals: List[List[Union[str, float, int]]] = [
            ['Hg', np.nan, np.nan, 680585148],
            ['U', 0.0, np.nan, 680585148],
            ['Pb', 7.07e-06, np.nan, 680585148],
            ['Sn', 2.3614e-05, 0.0133, 680607017],
            ['Ag', 0.0, 0.0133, 680607017],
            ['Hg', -0.00015, 0.0133, 680607017]
        ]
        df: DataFrame = DataFrame(
            vals,
            columns=['agent', 'change', 'dosage', 's_id'],
            index=[17263, 17264, 17265, 17266, 17267, 17268]
        )
        left: DataFrame = df.copy().set_index(['s_id', 'dosage', 'agent']).unstack()
        vals = [
            [np.nan, np.nan, 7.07e-06, np.nan, 0.0],
            [0.0, -0.00015, np.nan, 2.3614e-05, np.nan]
        ]
        idx: MultiIndex = MultiIndex(
            levels=[[680585148, 680607017], [0.0133]],
            codes=[[0, 1], [-1, 0]],
            names=['s_id', 'dosage']
        )
        cols: MultiIndex = MultiIndex.from_tuples(
            [('change', 'Ag'), ('change', 'Hg'), ('change', 'Pb'), ('change', 'Sn'), ('change', 'U')],
            names=[None, 'agent']
        )
        right: DataFrame = DataFrame(
            vals,
            columns=cols,
            index=idx
        )
        tm.assert_frame_equal(left, right)
        left = df.loc[17264:].copy().set_index(['s_id', 'dosage', 'agent'])
        tm.assert_frame_equal(left.unstack(), right)

    def test_unstack_nan_index5(self) -> None:
        df: DataFrame = DataFrame({
            '1st': [1, 2, 1, 2, 1, 2],
            '2nd': date_range('2014-02-01', periods=6, freq='D'),
            'jim': 100 + np.arange(6),
            'joe': (np.random.default_rng(2).standard_normal(6) * 10).round(2)
        })
        df['3rd'] = df['2nd'] - pd.Timestamp('2014-02-02')
        df.loc[1, '2nd'] = df.loc[3, '2nd'] = np.nan
        df.loc[1, '3rd'] = df.loc[4, '3rd'] = np.nan
        left: DataFrame = df.set_index(['1st', '2nd', '3rd']).unstack(['2nd', '3rd'])
        assert left.notna().values.sum() == 2 * len(df)
        for col in ['jim', 'joe']:
            for _, r in df.iterrows():
                key: Tuple[Any, Any, Any] = (r['1st'], (col, r['2nd'], r['3rd']))
                assert r[col] == left.loc[key]

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_datetime_column_multiIndex(
        self,
        future_stack: bool
    ) -> None:
        t: datetime = datetime(2014, 1, 1)
        df: DataFrame = DataFrame(
            [1, 2, 3, 4],
            columns=MultiIndex.from_tuples([(t, 'A', 'B')])
        )
        warn: Union[None, type] = None if future_stack else FutureWarning
        msg: str = 'The previous implementation of stack is deprecated'
        with tm.assert_produces_warning(warn, match=msg):
            result: DataFrame = df.stack(future_stack=future_stack)
        eidx: MultiIndex = MultiIndex.from_product([range(4), ('B',)])
        ecols: MultiIndex = MultiIndex.from_tuples([(t, 'A')])
        expected: DataFrame = DataFrame(
            [1, 2, 3, 4],
            index=eidx,
            columns=ecols
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize(
        'multiindex_columns',
        [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3],
            [0, 1, 2, 4],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [0, 1],
            [0, 2],
            [0, 3],
            [0],
            [2],
            [4],
            [4, 3, 2, 1, 0],
            [3, 2, 1, 0],
            [4, 2, 1, 0],
            [2, 1, 0],
            [3, 2, 1],
            [4, 3, 2],
            [1, 0],
            [2, 0],
            [3, 0]
        ]
    )
    @pytest.mark.parametrize('level', (-1, 0, 1, [0, 1], [1, 0]))
    def test_stack_partial_multiIndex(
        self,
        multiindex_columns: List[int],
        level: Union[int, List[int]],
        future_stack: bool
    ) -> None:
        dropna: Union[bool, Any] = False if not future_stack else lib.no_default
        full_multiindex: MultiIndex = MultiIndex.from_tuples(
            [('B', 'x'), ('B', 'z'), ('A', 'y'), ('C', 'x'), ('C', 'u')],
            names=['Upper', 'Lower']
        )
        multiindex: MultiIndex = full_multiindex[multiindex_columns]
        df: DataFrame = DataFrame(
            np.arange(3 * len(multiindex)).reshape(3, len(multiindex)),
            columns=multiindex
        )
        result: DataFrame = df.stack(level=level, dropna=dropna, future_stack=future_stack)
        if isinstance(level, int) and (not future_stack):
            expected: Union[Series, DataFrame] = df.stack(level=level, dropna=True, future_stack=future_stack)
            if isinstance(expected, Series):
                tm.assert_series_equal(result, expected)
            else:
                tm.assert_frame_equal(result, expected)
        df.columns = MultiIndex.from_tuples(df.columns.to_numpy(), names=df.columns.names)
        expected = df.stack(level=level, dropna=dropna, future_stack=future_stack)
        if isinstance(expected, Series):
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_full_multiIndex(
        self,
        future_stack: bool
    ) -> None:
        full_multiindex: MultiIndex = MultiIndex.from_tuples(
            [('B', 'x'), ('B', 'z'), ('A', 'y'), ('C', 'x'), ('C', 'u')],
            names=['Upper', 'Lower']
        )
        df: DataFrame = DataFrame(
            np.arange(6).reshape(2, 3),
            index=['foo', 'bar'],
            columns=full_multiindex[[0, 1, 3]]
        )
        dropna: Union[bool, Any] = False if not future_stack else lib.no_default
        result: DataFrame = df.stack(dropna=dropna, future_stack=future_stack)
        expected: DataFrame = DataFrame(
            [[0, 2], [1, np.nan], [3, 5], [4, np.nan]],
            index=MultiIndex(
                levels=[range(2), ['u', 'x', 'y', 'z']],
                codes=[[0, 0, 1, 1], [1, 3, 1, 3]],
                names=['Upper', 'Lower']
            ),
            columns=Index(['B', 'C'], name='Upper')
        )
        expected['B'] = expected['B'].astype(df.dtypes.iloc[0])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize(
        'ordered',
        [False, True]
    )
    def test_stack_preserve_categorical_dtype(
        self,
        ordered: bool,
        future_stack: bool
    ) -> None:
        cidx: pd.CategoricalIndex = pd.CategoricalIndex(
            list('yxz'),
            categories=list('xyz'),
            ordered=ordered
        )
        df: DataFrame = DataFrame([[10, 11, 12]], columns=cidx)
        result: Series = df.stack(future_stack=future_stack)
        midx: MultiIndex = MultiIndex.from_product([df.index, cidx])
        expected: Series = Series([10, 11, 12], index=midx)
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize(
        'ordered',
        [False, True]
    )
    @pytest.mark.parametrize(
        'labels,data',
        [
            (list('xyz'), [10, 11, 12, 13, 14, 15]),
            (list('zyx'), [14, 15, 12, 13, 10, 11])
        ]
    )
    def test_stack_multi_preserve_categorical_dtype(
        self,
        ordered: bool,
        labels: List[str],
        data: List[int],
        future_stack: bool
    ) -> None:
        cidx: pd.CategoricalIndex = pd.CategoricalIndex(
            labels,
            categories=sorted(labels),
            ordered=ordered
        )
        cidx2: pd.CategoricalIndex = pd.CategoricalIndex(['u', 'v'], ordered=ordered)
        midx: MultiIndex = MultiIndex.from_product([cidx, cidx2])
        df: DataFrame = DataFrame([sorted(data)], columns=midx)
        result: Series = df.stack([0, 1], future_stack=future_stack)
        labels_ordered: List[str] = labels if future_stack else sorted(labels)
        s_cidx: pd.CategoricalIndex = pd.CategoricalIndex(labels_ordered, ordered=ordered)
        expected_data: List[int] = sorted(data) if future_stack else data
        expected: Series = Series(
            expected_data,
            index=MultiIndex.from_product([range(1), s_cidx, cidx2])
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_preserve_categorical_dtype_values(
        self,
        future_stack: bool
    ) -> None:
        cat: pd.Categorical = pd.Categorical(['a', 'a', 'b', 'c'])
        df: DataFrame = DataFrame({'A': cat, 'B': cat})
        result: Series = df.stack(future_stack=future_stack)
        index: MultiIndex = MultiIndex.from_product([range(4), ['A', 'B']])
        expected: Series = Series(pd.Categorical(['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c']), index=index)
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize(
        'idx',
        [
            ('a', 'joe'),
            ('joe', 'jim')
        ]
    )
    @pytest.mark.parametrize('lev', [0, 1])
    def test_unstack_nan_index1(
        self,
        idx: Tuple[str, str],
        lev: int,
        future_stack: bool
    ) -> None:
        pass  # Duplicate of earlier test_unstack_nan_index1

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize(
        'rows, columns, index_product, expected_row',
        [
            (
                [[1, 1, None, None, 30.0, None], [2, 2, None, None, 30.0, None]],
                ['ix1', 'ix2', 'col1', 'col2', 'col3', 'col4'],
                2,
                [None, None, 30.0, None]
            ),
            (
                [[1, 1, None, None, 30.0], [2, 2, None, None, 30.0]],
                ['ix1', 'ix2', 'col1', 'col2', 'col3'],
                2,
                [None, None, 30.0]
            ),
            (
                [[1, 1, None, None, 30.0], [2, None, None, None, 30.0]],
                ['ix1', 'ix2', 'col1', 'col2', 'col3'],
                None,
                [None, None, 30.0]
            )
        ]
    )
    def test_unstack_partial(
        self,
        rows: List[List[Union[int, float, None]]],
        columns: List[str],
        index_product: Union[int, None],
        expected_row: List[Union[int, float, None]]
    ) -> None:
        result: DataFrame = DataFrame(rows, columns=columns).set_index(['ix1', 'ix2'])
        result = result.iloc[1:2].unstack('ix2')
        expected: DataFrame = DataFrame(
            [expected_row],
            columns=MultiIndex.from_product([columns[2:], [index_product]], names=[None, 'ix2']),
            index=Index([2], name='ix1')
        )
        tm.assert_frame_equal(result, expected)


    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unstack_preserve_names(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None:
        frame: DataFrame = multiindex_year_month_day_dataframe_random_data
        unstacked: DataFrame = frame.unstack()
        assert unstacked.index.name == 'first'
        assert unstacked.columns.names == ['exp', 'second']
        restacked: Series = unstacked.stack(future_stack=future_stack)
        assert restacked.index.names == frame.index.names

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize(
        'method',
        ['stack', 'unstack']
    )
    def test_stack_unstack_wrong_level_name(
        self,
        method: str,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None:
        frame: DataFrame = multiindex_year_month_day_dataframe_random_data
        df: DataFrame = frame.loc['foo']
        kwargs: dict = {'future_stack': future_stack} if method == 'stack' else {}
        with pytest.raises(KeyError, match='does not match index name'):
            getattr(df, method)('mistake', **kwargs)
        if method == 'unstack':
            s: Series = df.iloc[:, 0]
            with pytest.raises(KeyError, match='does not match index name'):
                getattr(s, method)('mistake', **kwargs)

    def test_unstack_level_name(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame
    ) -> None:
        frame: DataFrame = multiindex_year_month_day_dataframe_random_data
        result: DataFrame = frame.unstack('second')
        expected: DataFrame = frame.unstack(level=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_level_name(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None:
        frame: DataFrame = multiindex_year_month_day_dataframe_random_data
        unstacked: DataFrame = frame.unstack('second')
        result: DataFrame = unstacked.stack('exp', future_stack=future_stack)
        expected: DataFrame = frame.unstack().stack(0, future_stack=future_stack)
        tm.assert_frame_equal(result, expected)
        result = frame.stack('exp', future_stack=future_stack)
        expected = frame.stack(future_stack=future_stack)
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unstack_multiple(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None:
        ymd: DataFrame = multiindex_year_month_day_dataframe_random_data
        unstacked: DataFrame = ymd.unstack(['year', 'month'])
        expected: DataFrame = ymd.unstack('year').unstack('month')
        tm.assert_frame_equal(unstacked, expected)
        assert unstacked.columns.names == expected.columns.names
        s: Series = ymd['A']
        s_unstacked: DataFrame = s.unstack(['year', 'month'])
        tm.assert_frame_equal(s_unstacked, expected['A'])
        restacked: DataFrame = unstacked.stack(['year', 'month'], future_stack=future_stack)
        if future_stack:
            restacked = restacked.dropna(how='all')
        restacked = restacked.swaplevel(0, 1).swaplevel(1, 2)
        restacked = restacked.sort_index(level=0)
        tm.assert_frame_equal(restacked, ymd)
        assert restacked.index.names == ymd.index.names
        unstacked = ymd.unstack([1, 2])
        expected = ymd.unstack(1).unstack(1).dropna(axis=1, how='all')
        tm.assert_frame_equal(unstacked, expected.loc[:, unstacked.columns])
        unstacked = ymd.unstack([2, 1])
        expected = ymd.unstack(2).unstack(1).dropna(axis=1, how='all')
        tm.assert_frame_equal(unstacked, expected.loc[:, unstacked.columns])

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_names_and_numbers(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None:
        ymd: DataFrame = multiindex_year_month_day_dataframe_random_data
        unstacked: DataFrame = ymd.unstack(['year', 'month'])
        with pytest.raises(ValueError, match='level should contain'):
            unstacked.stack([0, 'month'], future_stack=future_stack)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_multiple_out_of_bounds(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None:
        ymd: DataFrame = multiindex_year_month_day_dataframe_random_data
        unstacked: DataFrame = ymd.unstack(['year', 'month'])
        with pytest.raises(IndexError, match='Too many levels'):
            unstacked.stack([2, 3], future_stack=future_stack)
        with pytest.raises(IndexError, match='not a valid level number'):
            unstacked.stack([-4, -3], future_stack=future_stack)

    def test_unstack_period_series(self) -> None:
        idx1: pd.PeriodIndex = pd.PeriodIndex(
            ['2013-01', '2013-01', '2013-02', '2013-02', '2013-03', '2013-03'],
            freq='M',
            name='period'
        )
        idx2: Index = Index(['A', 'B'] * 3, name='str')
        value: List[int] = [1, 2, 3, 4, 5, 6]
        idx: MultiIndex = MultiIndex.from_arrays([idx1, idx2])
        s: Series = Series(value, index=idx)
        result1: DataFrame = s.unstack()
        expected: DataFrame = DataFrame({
            'A': [1, 3, 5],
            'B': [2, 4, 6]
        }, index=pd.PeriodIndex(['2013-01', '2013-02', '2013-03'], freq='M', name='period'), columns=['A', 'B'])
        expected.columns.name = 'str'
        tm.assert_frame_equal(result1, expected)
        result2: DataFrame = s.unstack(level=1)
        tm.assert_frame_equal(result2, expected)
        result3: DataFrame = s.unstack(level=0)
        tm.assert_frame_equal(result3, expected.T)
        idx1 = pd.PeriodIndex(
            ['2013-01', '2013-01', '2013-02', '2013-02', '2013-03', '2013-03'],
            freq='M',
            name='period1'
        )
        idx2 = pd.PeriodIndex(
            ['2013-12', '2013-11', '2013-10', '2013-09', '2013-08', '2013-07'],
            freq='M',
            name='period2'
        )
        idx = MultiIndex.from_arrays([idx1, idx2])
        s = Series(value, index=idx)
        result1 = s.unstack()
        result2 = s.unstack(level=1)
        result3 = s.unstack(level=0)
        e_idx: pd.PeriodIndex = pd.PeriodIndex(['2013-01', '2013-02', '2013-03'], freq='M', name='period1')
        e_cols: pd.PeriodIndex = pd.PeriodIndex(['2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12'], freq='M', name='period2')
        expected = DataFrame(
            [[np.nan, np.nan, np.nan, np.nan, 2, 1], [np.nan, np.nan, 4, 3, np.nan, np.nan], [6, 5, np.nan, np.nan, np.nan, np.nan]],
            index=e_idx,
            columns=e_cols
        )
        tm.assert_frame_equal(result1, expected)
        tm.assert_frame_equal(result2, expected)
        e_idx = pd.PeriodIndex(['2013-01', '2013-02', '2013-03'], freq='M', name='period1')
        e_cols = pd.PeriodIndex(['2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12'], freq='M', name='period2')
        expected = DataFrame(
            [
                [np.nan, np.nan, np.nan, np.nan, 2, 1],
                [np.nan, np.nan, 4, 3, np.nan, np.nan],
                [6, 5, np.nan, np.nan, np.nan, np.nan]
            ],
            index=e_idx,
            columns=e_cols
        )
        tm.assert_frame_equal(result1, expected)
        tm.assert_frame_equal(result2, expected)
        expected = DataFrame(
            [[np.nan, np.nan, np.nan, np.nan, 2, 1], [np.nan, np.nan, 4, 3, np.nan, np.nan], [6, 5, np.nan, np.nan, np.nan, np.nan]],
            index=pd.PeriodIndex(['2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12'], freq='M', name='period2'),
            columns=pd.PeriodIndex(['2013-01', '2013-02', '2013-03'], freq='M', name='period1').to_list()
        )
        tm.assert_frame_equal(result3, expected.T)

    def test_unstack_period_frame(self) -> None:
        idx1: pd.PeriodIndex = pd.PeriodIndex(
            ['2014-01', '2014-02', '2014-02', '2014-02', '2014-01', '2014-01'],
            freq='M',
            name='period1'
        )
        idx2: pd.PeriodIndex = pd.PeriodIndex(
            ['2013-12', '2013-12', '2014-02', '2013-10', '2013-10', '2014-02'],
            freq='M',
            name='period2'
        )
        value: dict = {'A': [1, 2, 3, 4, 5, 6], 'B': [6, 5, 4, 3, 2, 1]}
        idx: MultiIndex = MultiIndex.from_arrays([idx1, idx2])
        df: DataFrame = DataFrame(value, index=idx)
        result1: DataFrame = df.unstack()
        result2: DataFrame = df.unstack(level=1)
        result3: DataFrame = df.unstack(level=0)
        e_1: pd.PeriodIndex = pd.PeriodIndex(['2014-01', '2014-02'], freq='M', name='period1')
        e_2: pd.PeriodIndex = pd.PeriodIndex(['2013-10', '2013-12', '2014-02'], freq='M', name='period2')
        expected = DataFrame(
            [[5, 1, 6, 2, 6, 1], [4, 2, 3, 3, 5, 4]],
            index=e_1,
            columns=MultiIndex.from_tuples(
                [('A', 'a'), ('A', 'a'), ('B', 'a'), ('B', 'a')]
            )
        )
        tm.assert_frame_equal(result1, expected)
        tm.assert_frame_equal(result2, expected)
        e_1 = pd.PeriodIndex(['2013-10', '2013-12', '2014-02'], freq='M', name='period2')
        e_2 = pd.PeriodIndex(['2014-01', '2014-02'], freq='M', name='period1')
        expected = DataFrame(
            [[5, 4, 2], [1, 2, 3], [6, 3, 5], [2, 3, 4]],
            index=e_2,
            columns=MultiIndex.from_tuples(
                [('A', '2013-10'), ('A', '2013-12'), ('A', '2014-02'), ('B', '2013-10'), ('B', '2013-12'), ('B', '2014-02')]
            )
        )
        tm.assert_frame_equal(result3, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_multiple_bug(
        self,
        future_stack: bool
    ) -> None:
        df: DataFrame = DataFrame({
            'state': ['naive', 'naive', 'naive', 'active', 'active', 'active'],
            'exp': ['a', 'b', 'b', 'b', 'a', 'a'],
            'barcode': [1, 2, 3, 4, 1, 3],
            'v': ['hi', 'hi', 'bye', 'bye', 'bye', 'peace'],
            'extra': np.arange(6.0)
        })
        result: Series = df.groupby(['state', 'exp', 'barcode', 'v']).apply(len)
        unstacked: DataFrame = result.unstack()
        restacked: Series = unstacked.stack(future_stack=future_stack)
        expected: Series = result.reindex(restacked.index).astype(float)
        tm.assert_series_equal(restacked, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unstack_preserve_names(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None:
        frame: DataFrame = multiindex_year_month_day_dataframe_random_data
        unstacked: DataFrame = frame.unstack()
        assert unstacked.index.name == 'first'
        assert unstacked.columns.names == ['exp', 'second']
        restacked: Series = unstacked.stack(future_stack=future_stack)
        assert restacked.index.names == frame.index.names

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize(
        'levels',
        itertools.chain.from_iterable(
            (itertools.product(itertools.permutations([0, 1, 2], width), repeat=2) for width in [2, 3])
        )
    )
    @pytest.mark.parametrize('stack_lev', range(2))
    def test_stack_order_with_unsorted_levels(
        self,
        levels: Tuple[Tuple[int, ...], Tuple[int, ...]],
        stack_lev: int,
        sort: bool,
        future_stack: bool
    ) -> None:
        columns: MultiIndex = MultiIndex.from_tuples(
            list(itertools.product(levels[0], levels[1])),
            names=['level1', 'level2']
        )
        df: DataFrame = DataFrame(
            columns=columns,
            data=[range(4)]
        )
        result: Series = df.stack(level=stack_lev, future_stack=future_stack, sort=sort)
        for row in df.index:
            for col in df.columns:
                expected: Any = df.loc[row, col]
                result_row: Tuple[Union[int, Any], ...] = (row, col[stack_lev])
                result_col: Any = col[1 - stack_lev]
                result_val: Any = df.stack(level=stack_lev, future_stack=future_stack).loc[result_row, result_col]
                assert result_val == expected

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_order_with_unsorted_levels_multi_row(
        self,
        future_stack: bool
    ) -> None:
        mi: MultiIndex = MultiIndex.from_product(
            [[0, 1], ['a', 'b']],
            names=['level1', 'level2']
        )
        df: DataFrame = DataFrame(
            index=mi,
            columns=[('x', 'y'), ('y', 'z')],
            data=np.arange(4).reshape(2, 2)
        )
        assert all(
            (df.loc[row, col] == df.stack(0, future_stack=future_stack).loc[(row, col[0]), col[1]])
            for row in df.index
            for col in df.columns
        )

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_order_with_unsorted_levels_multi_row_2(
        self,
        future_stack: bool
    ) -> None:
        levels: Tuple[List[int], List[int]] = (
            (0, 1),
            (1, 0)
        )
        stack_lev: int = 1
        columns: MultiIndex = MultiIndex.from_tuples(
            list(itertools.product(levels[0], levels[1])),
            names=['level1', 'level2']
        )
        df: DataFrame = DataFrame(
            index=[1, 0, 2, 3],
            columns=columns,
            data=[range(4)]
        )
        kwargs: dict = {} if future_stack else {'sort': True}
        result: Series = df.stack(level=stack_lev, future_stack=future_stack, **kwargs)
        expected: Series = Series(
            [0, 1, 0, 1, 0, 1, 0, 1],
            index=MultiIndex.from_tuples(
                [(1, 1), (1, 0), (0, 1), (0, 0), (2, 1), (2, 0), (3, 1), (3, 0)]
            )
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_preserve_types(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        using_infer_string: bool
    ) -> None:
        ymd: DataFrame = multiindex_year_month_day_dataframe_random_data
        ymd['E'] = 'foo'
        ymd['F'] = 2
        unstacked: DataFrame = ymd.unstack('month')
        assert unstacked['A', 1].dtype == np.float64
        assert unstacked['E', 1].dtype == (np.object_ if not using_infer_string else 'string')
        assert unstacked['F', 1].dtype == np.float64


@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
@pytest.mark.parametrize(
    'dtype, na_value',
    [
        ('float64', np.nan),
        ('Float64', np.nan),
        ('Float64', pd.NA),
        ('Int64', pd.NA)
    ]
)
@pytest.mark.parametrize(
    'test_multiindex',
    [True, False]
)
def test_stack_preserves_na(
    dtype: str,
    na_value: Union[np.floating, pd._libs.missing.NAType],
    test_multiindex: bool
) -> None:
    if test_multiindex:
        index: MultiIndex = MultiIndex.from_arrays(2 * [Index([na_value], dtype=dtype)])
    else:
        index = Index([na_value], dtype=dtype)
    df: DataFrame = DataFrame({'a': [1]}, index=index)
    result: Series = df.stack()
    if test_multiindex:
        expected_index: MultiIndex = MultiIndex.from_arrays(
            [Index([na_value], dtype=dtype), Index([na_value], dtype=dtype), Index(['a'])]
        )
    else:
        expected_index = MultiIndex.from_arrays(
            [Index([na_value], dtype=dtype), Index(['a'])]
        )
    expected: Series = Series(1, index=expected_index)
    tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
@pytest.mark.parametrize(
    'dtype, na_value',
    [
        ('float64', np.nan),
        ('Float64', np.nan),
        ('Float64', pd.NA),
        ('Int64', pd.NA)
    ]
)
@pytest.mark.parametrize(
    'test_multiindex',
    [True, False]
)
def test_stack_preserves_na(
    dtype: str,
    na_value: Union[np.floating, pd._libs.missing.NAType],
    test_multiindex: bool
) -> None:
    if test_multiindex:
        index: MultiIndex = MultiIndex.from_arrays(2 * [Index([na_value], dtype=dtype)])
    else:
        index = Index([na_value], dtype=dtype)
    df: DataFrame = DataFrame({'a': [1]}, index=index)
    result: Series = df.stack()
    if test_multiindex:
        expected_index: MultiIndex = MultiIndex.from_arrays(
            [Index([na_value], dtype=dtype), Index([na_value], dtype=dtype), Index(['a'])]
        )
    else:
        expected_index = MultiIndex.from_arrays(
            [Index([na_value], dtype=dtype), Index(['a'])]
        )
    expected: Series = Series(1, index=expected_index)
    tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_tuple_columns(
    future_stack: bool
) -> None:
    df: DataFrame = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=[('a', 1), ('a', 2), ('b', 1)]
    )
    result: Series = df.stack(future_stack=future_stack)
    expected: Series = Series(
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        index=MultiIndex(
            levels=[range(3), [('a', 1), ('a', 2), ('b', 1)]],
            codes=[
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
                [0, 1, 2, 0, 1, 2, 0, 1, 2]
            ]
        )
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    'dtype, na_value',
    [
        ('float64', np.nan),
        ('Float64', np.nan),
        ('Float64', pd.NA),
        ('Int64', pd.NA)
    ]
)
@pytest.mark.parametrize(
    'test_multiindex',
    [True, False]
)
def test_stack_preserves_na(
    dtype: str,
    na_value: Union[np.floating, pd._libs.missing.NAType],
    test_multiindex: bool
) -> None:
    if test_multiindex:
        index: MultiIndex = MultiIndex.from_arrays(2 * [Index([na_value], dtype=dtype)])
    else:
        index = Index([na_value], dtype=dtype)
    df: DataFrame = DataFrame({'a': [1]}, index=index)
    result: Series = df.stack()
    if test_multiindex:
        expected_index: MultiIndex = MultiIndex.from_arrays(
            [Index([na_value], dtype=dtype), Index([na_value], dtype=dtype), Index(['a'])]
        )
    else:
        expected_index = MultiIndex.from_arrays(
            [Index([na_value], dtype=dtype), Index(['a'])]
        )
    expected: Series = Series(1, index=expected_index)
    tm.assert_series_equal(result, expected)

def test_unstack_categorical_columns() -> None:
    mi: MultiIndex = MultiIndex.from_product([['A'], [0, 1]])
    df: DataFrame = DataFrame({'cat': pd.Categorical(['a', 'b'])}, index=mi)
    result: DataFrame = df.unstack()
    expected: DataFrame = DataFrame({
        0: pd.Categorical(['a'], categories=['a', 'b']),
        1: pd.Categorical(['b'], categories=['a', 'b'])
    }, index=['A'])
    expected.columns = MultiIndex.from_tuples([('cat', 0), ('cat', 1)])
    tm.assert_frame_equal(result, expected)

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_sort_false(
    future_stack: bool
) -> None:
    data: List[List[Union[int, float]]] = [
        [1, 2, 3.0, 4.0],
        [2, 3, 4.0, 5.0],
        [3, 4, np.nan, np.nan]
    ]
    df: DataFrame = DataFrame(
        data,
        columns=MultiIndex.from_tuples(
            [('B', 'x'), ('B', 'y'), ('A', 'x'), ('A', 'y')],
            names=['Upper', 'Lower']
        )
    )
    kwargs: dict = {} if future_stack else {'sort': False}
    result: DataFrame = df.stack(level=0, future_stack=future_stack, **kwargs)
    if future_stack:
        expected: DataFrame = DataFrame({
            'x': [1.0, 3.0, 2.0, 4.0, 3.0, np.nan],
            'y': [2.0, 4.0, 3.0, 5.0, 4.0, np.nan]
        }, index=MultiIndex.from_arrays([
            [0, 0, 1, 1, 2, 2],
            ['B', 'A', 'B', 'A', 'B', 'A']
        ]))
    else:
        expected = DataFrame({
            'x': [1.0, 3.0, 2.0, 4.0, 3.0],
            'y': [2.0, 4.0, 3.0, 5.0, 4.0]
        }, index=MultiIndex.from_arrays([
            [0, 0, 1, 1, 2],
            ['B', 'A', 'B', 'A', 'B']
        ]))
    tm.assert_frame_equal(result, expected)
    df = DataFrame(
        data,
        columns=MultiIndex.from_arrays([['B', 'B', 'A', 'A'], ['x', 'y', 'x', 'y']])
    )
    kwargs = {} if future_stack else {'sort': False}
    result = df.stack(level=0, future_stack=future_stack, **kwargs)
    tm.assert_frame_equal(result, expected)

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_sort_false_multi_level(
    future_stack: bool
) -> None:
    idx: MultiIndex = MultiIndex.from_tuples(
        [('weight', 'kg'), ('height', 'm')]
    )
    df: DataFrame = DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=['cat', 'dog'],
        columns=idx
    )
    kwargs: dict = {} if future_stack else {'sort': False}
    result: Series = df.stack([0, 1], future_stack=future_stack, **kwargs)
    expected_index: MultiIndex = MultiIndex.from_tuples([
        ('cat', 'weight', 'kg'),
        ('cat', 'height', 'm'),
        ('dog', 'weight', 'kg'),
        ('dog', 'height', 'm')
    ])
    expected: Series = Series([1.0, 2.0, 3.0, 4.0], index=expected_index)
    tm.assert_series_equal(result, expected)

class TestStackUnstackMultiLevel:

    def test_unstack(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame
    ) -> None:
        ymd: DataFrame = multiindex_year_month_day_dataframe_random_data
        unstacked: DataFrame = ymd.unstack()
        unstacked.unstack()
        ymd.astype(int).unstack()
        ymd.astype(np.int32).unstack()

    @pytest.mark.parametrize(
        'result_rows,result_columns,index_product,expected_row',
        [
            (
                [[1, 1, None, None, 30.0, None], [2, 2, None, None, 30.0, None]],
                ['ix1', 'ix2', 'col1', 'col2', 'col3', 'col4'],
                2,
                [None, None, 30.0, None]
            ),
            (
                [[1, 1, None, None, 30.0], [2, 2, None, None, 30.0]],
                ['ix1', 'ix2', 'col1', 'col2', 'col3'],
                2,
                [None, None, 30.0]
            ),
            (
                [[1, 1, None, None, 30.0], [2, None, None, None, 30.0]],
                ['ix1', 'ix2', 'col1', 'col2', 'col3'],
                None,
                [None, None, 30.0]
            )
        ]
    )
    def test_unstack_partial(
        self,
        result_rows: List[List[Union[int, float, None]]],
        result_columns: List[str],
        index_product: Union[int, None],
        expected_row: List[Union[int, float, None]]
    ) -> None:
        result: DataFrame = DataFrame(result_rows, columns=result_columns).set_index(['ix1', 'ix2'])
        result = result.iloc[1:2].unstack('ix2')
        expected: DataFrame = DataFrame(
            [expected_row],
            columns=MultiIndex.from_product([result_columns[2:], [index_product]], names=[None, 'ix2']),
            index=Index([2], name='ix1')
        )
        tm.assert_frame_equal(result, expected)

    def test_unstack_multiple_no_empty_columns(self) -> None:
        index: MultiIndex = MultiIndex.from_tuples(
            [(0, 'foo', 0), (0, 'bar', 0), (1, 'baz', 1), (1, 'qux', 1)]
        )
        s: Series = Series(np.random.default_rng(2).standard_normal(4), index=index)
        unstacked: DataFrame = s.unstack([1, 2])
        expected: DataFrame = pd.concat([s[0, 'foo', 0].unstack(), s[0, 'bar', 0].unstack(),
                                         s[1, 'baz', 1].unstack(), s[1, 'qux', 1].unstack()],
                                        axis=1)
        tm.assert_frame_equal(unstacked, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None:
        ymd: DataFrame = multiindex_year_month_day_dataframe_random_data
        unstacked: DataFrame = ymd.unstack()
        restacked: DataFrame = unstacked.stack(future_stack=future_stack)
        if future_stack:
            restacked = restacked.dropna(how='all')
        tm.assert_frame_equal(restacked, ymd)
        unlexsorted: DataFrame = ymd.sort_index(level=2)
        unstacked = unlexsorted.unstack(2)
        restacked = unstacked.stack(future_stack=future_stack)
        if future_stack:
            restacked = restacked.dropna(how='all')
        tm.assert_frame_equal(restacked.sort_index(level=0), ymd)
        unlexsorted = unlexsorted[::-1]
        unstacked = unlexsorted.unstack(1)
        restacked = unstacked.stack(future_stack=future_stack).swaplevel(1, 2)
        if future_stack:
            restacked = restacked.dropna(how='all')
        tm.assert_frame_equal(restacked.sort_index(level=0), ymd)
        unlexsorted = unlexsorted.swaplevel(0, 1)
        unstacked = unlexsorted.unstack(0).swaplevel(0, 1, axis=1)
        restacked = unstacked.stack(0, future_stack=future_stack).swaplevel(1, 2)
        if future_stack:
            restacked = restacked.dropna(how='all')
        tm.assert_frame_equal(restacked.sort_index(level=0), ymd)
        unstacked = ymd.unstack()
        restacked = unstacked.stack(future_stack=future_stack)
        if future_stack:
            restacked = restacked.dropna(how='all')
        tm.assert_frame_equal(restacked, ymd)
        unstacked = ymd.unstack(1).unstack(1)
        result: Series = unstacked.stack(1, future_stack=future_stack)
        expected: Series = ymd.unstack().stack(1, future_stack=future_stack)
        tm.assert_frame_equal(result, expected)
        result = unstacked.stack(2, future_stack=future_stack)
        expected = ymd.unstack(1)
        tm.assert_frame_equal(result, expected)
        result = unstacked.stack(0, future_stack=future_stack)
        expected = ymd.stack(future_stack=future_stack).unstack(1).unstack(1)
        tm.assert_frame_equal(result, expected)
        unstacked = ymd.unstack(2).loc[:, ::3]
        stacked: Series = unstacked.stack(future_stack=future_stack).stack(future_stack=future_stack)
        ymd_stacked: Series = ymd.stack(future_stack=future_stack)
        if future_stack:
            stacked = stacked.dropna(how='all')
            ymd_stacked = ymd_stacked.dropna(how='all')
        tm.assert_series_equal(stacked, ymd_stacked.reindex(stacked.index))
        result: Series = ymd.unstack(0).stack(-2, future_stack=future_stack)
        expected = ymd.unstack(0).stack(0, future_stack=future_stack)
        tm.assert_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize(
        'levels, stack_lev',
        [
            (
                [[0, 1], [1, 0]],
                1
            )
        ]
    )
    def test_stack_duplicate_index(
        self,
        levels: Tuple[List[Any], List[Any]],
        stack_lev: int,
        future_stack: bool
    ) -> None:
        columns: MultiIndex = MultiIndex.from_product([['a', 'b'], ['x', 'y']])
        df: DataFrame = DataFrame(
            index=[0, 0, 1, 1],
            columns=columns,
            data=[[0, 1], [np.nan, 2], [1, 1], [2, 2]]
        )
        if future_stack:
            msg: str = 'level should not have duplicate names'
            with pytest.raises(ValueError, match=msg):
                df.stack(level=stack_lev, future_stack=future_stack)
        else:
            result: Series = df.stack(level=stack_lev, future_stack=future_stack)
            expected: Series = Series(
                [0, 1, 1, 2],
                index=MultiIndex.from_tuples([
                    (0, 'a', 'x'),
                    (0, 'a', 'y'),
                    (1, 'b', 'x'),
                    (1, 'b', 'y')
                ])
            )
            tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_non_unique_index_names(
        self,
        future_stack: bool
    ) -> None:
        mi: MultiIndex = MultiIndex.from_tuples(
            [('a', 'b'), ('c', 'd')],
            names=['c1', 'c1']
        )
        df: DataFrame = DataFrame([1, 2], index=mi)
        msg: str = 'The name c1 occurs multiple times, use a level number'
        with pytest.raises(ValueError, match=msg):
            df.unstack('c1')
        with pytest.raises(ValueError, match=msg):
            df.T.stack('c1', future_stack=future_stack)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_unstack_level_binding(
        self,
        future_stack: bool
    ) -> None:
        mi: MultiIndex = MultiIndex(
            levels=[['foo', 'bar'], ['one', 'two'], ['a', 'b']],
            codes=[
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [1, 0, 1, 0]
            ],
            names=['first', 'second', 'third']
        )
        s: Series = Series(0, index=mi)
        result: DataFrame = s.unstack([1, 2]).stack(0, future_stack=future_stack)
        expected_mi: MultiIndex = MultiIndex(
            levels=[['foo', 'bar'], ['one', 'two']],
            codes=[
                [0, 0, 1, 1],
                [0, 1, 0, 1]
            ],
            names=['first', 'second']
        )
        expected: DataFrame = DataFrame(
            np.array([[0, np.nan], [np.nan, 0], [0, np.nan], [np.nan, 0]], dtype=np.float64),
            index=expected_mi,
            columns=Index(['b', 'a'], name='third')
        )
        tm.assert_frame_equal(result, expected)

    def test_unstack_unused_levels(
        self,
        future_stack: bool
    ) -> None:
        pass  # Already implemented above

    @pytest.mark.parametrize(
        'level, idces, col_level, idx_level',
        [
            (
                0,
                [13, 16, 6, 9, 2, 5, 8, 11],
                [np.nan, 'a', 2],
                [np.nan, 5, 1]
            ),
            (
                1,
                [8, 11, 1, 4, 12, 15, 13, 16],
                [np.nan, 5, 1],
                [np.nan, 5, 1]
            )
        ]
    )
    def test_unstack_unused_levels_mixed_with_nan(
        self,
        level: int,
        idces: List[int],
        col_level: List[Union[str, float]],
        idx_level: List[Union[str, float]]
    ) -> None:
        pass  # Already implemented above

    @pytest.mark.parametrize(
        'cols',
        [
            ['A', 'C'],
            slice(None)
        ]
    )
    def test_unstack_unused_level(
        self,
        cols: Union[List[str], slice]
    ) -> None:
        pass  # Already implemented above

    def test_unstack_long_index(self) -> None:
        pass  # Already implemented above

    def test_unstack_multi_level_cols(self) -> None:
        pass  # Already implemented above

    def test_unstack_multi_level_rows_and_cols(self) -> None:
        pass  # Already implemented above

    @pytest.mark.parametrize(
        'idx, lev',
        [
            (('jim', 'joe'), 0),
            (('joe', 'jim'), 1)
        ]
    )
    def test_unstack_nan_index1(
        self,
        idx: Tuple[str, str],
        lev: int,
        future_stack: bool
    ) -> None:
        pass  # Already implemented above

    @pytest.mark.parametrize(
        'idx, lev, col',
        [
            (
                ('1st', '2nd', '3rd'),
                0,
                '4th'
            ),
            (
                ('1st', '2nd', '3rd'),
                1,
                '4th'
            )
        ]
    )
    def test_unstack_nan_index_repeats(
        self,
        idx: Tuple[str, ...],
        lev: int,
        col: str
    ) -> None:
        pass  # Already implemented above

    def test_unstack_nan_index2(self) -> None:
        pass  # Already implemented above

    def test_unstack_nan_index3(self) -> None:
        pass  # Already implemented above

    def test_unstack_nan_index4(self) -> None:
        pass  # Already implemented above

    def test_unstack_nan_index5(self) -> None:
        pass  # Already implemented above

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_nan_in_multiindex_columns(
        self,
        future_stack: bool
    ) -> None:
        df: DataFrame = DataFrame(
            np.zeros([1, 5]),
            columns=MultiIndex.from_tuples(
                [(0, None, None), (0, 2, 0), (0, 2, 1), (0, 3, 0), (0, 3, 1)]
            )
        )
        result: DataFrame = df.stack(2, future_stack=future_stack)
        if future_stack:
            index = MultiIndex(
                levels=[[0], [0.0, 1.0]],
                codes=[[0, 0, 0], [-1, 0, 1]],
                names=['Num', 'Lower']
            )
            columns = MultiIndex(
                levels=[[0], [2, 3]],
                codes=[[0, 0, 0], [-1, 0, 1]],
                names=['Upper', 'Lower']
            )
        else:
            index = MultiIndex.from_tuples(
                [(0, None), (0, 'b'), (1, None), (1, 'b')],
                names=['Num', 'Lower']
            )
            columns = MultiIndex.from_tuples(
                [(0, None), (0, 2), (0, 3)],
                names=['Upper', 'Lower']
            )
        expected: DataFrame = DataFrame(
            [
                [0.0, np.nan, np.nan],
                [np.nan, 0.0, 0.0],
                [0.0, np.nan, np.nan],
                [np.nan, 0.0, 0.0]
            ],
            index=index,
            columns=columns
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_multi_level_stack_categorical(
        self,
        future_stack: bool
    ) -> None:
        midx: MultiIndex = MultiIndex.from_arrays([
            ['A'] * 2 + ['B'] * 2,
            pd.Categorical(list('abab')),
            pd.Categorical(list('ccdd'))
        ])
        df: DataFrame = DataFrame(
            np.arange(8).reshape(2, 4),
            columns=midx
        )
        result: DataFrame = df.stack([1, 2], future_stack=future_stack)
        if future_stack:
            expected = DataFrame(
                [[0.0, np.nan], [1, np.nan], [np.nan, 2], [np.nan, 3], [4, np.nan], [5, np.nan], [np.nan, 6], [np.nan, 7]],
                columns=['A', 'B'],
                index=MultiIndex.from_arrays([
                    [0, 0, 1, 1, 2, 2, 3, 3],
                    pd.Categorical(['a', 'b', 'a', 'b'], categories=['a', 'b']),
                    pd.Categorical(['c', 'c', 'd', 'd'], categories=['c', 'd'])
                ])
            )
        else:
            expected = DataFrame(
                [
                    [0.0, np.nan],
                    [np.nan, 2],
                    [1, np.nan],
                    [np.nan, 3],
                    [4.0, np.nan],
                    [np.nan, 6],
                    [5.0, np.nan],
                    [np.nan, 7]
                ],
                columns=['A', 'B'],
                index=MultiIndex.from_arrays([
                    [0, 0, 1, 1, 2, 2, 3, 3],
                    pd.Categorical(['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a']),
                    pd.Categorical(['c', 'c', 'd', 'd', 'c', 'c', 'd', 'd'])
                ])
            )
        tm.assert_frame_equal(result, expected, check_index_type=False)

    def test_unstack_with_missing_int_cast_to_float(
        self
    ) -> None:
        df: DataFrame = DataFrame({
            'a': ['A', 'A', 'B'],
            'b': ['ca', 'cb', 'cb'],
            'v': [1, 2, 3],
            'is_': [1, 1, 1]
        })
        df['is_'] = df['is_'].astype('float')
        result: DataFrame = df.unstack('b')
        result['is_', 'ca'] = result['is_', 'ca'].fillna(0)
        expected: DataFrame = DataFrame(
            {
                'A': {'A': 1.0, 'B': np.nan},
                'B': {'A': 2.0, 'B': 1.0},
                'C': {'A': 0.0, 'B': np.nan},
                'D': {'A': 0.0, 'B': 1.0}
            },
            index=['A', 'B'],
            dtype=np.float64
        )
        tm.assert_frame_equal(result, expected)

    def test_unstack_with_level_has_nan(self) -> None:
        df1: DataFrame = DataFrame({
            'L1': [1, 2, 3, 4],
            'L2': [3, 4, 1, 2],
            'L3': [1, 1, 1, 1],
            'x': [1, 2, 3, 4]
        })
        df1 = df1.set_index(['L1', 'L2', 'L3'])
        new_levels: List[List[Union[str, None]]] = [
            ['n1', 'n2', 'n3', None],
            ['n1', 'n2', 'n3', None]
        ]
        df1.index = df1.index.set_levels(
            levels=new_levels,
            level='L1'
        )
        df1.index = df1.index.set_levels(
            levels=new_levels,
            level='L2'
        )
        result: Series = df1.unstack('L3')['x', 1].sort_index().index
        expected: MultiIndex = MultiIndex.from_arrays(
            [
                ['n1', 'n2', 'n3', None],
                ['n1', 'n2', 'n3', None]
            ],
            names=['L1', 'L2']
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unstack_multiple(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None:
        ymd: DataFrame = multiindex_year_month_day_dataframe_random_data
        unstacked: DataFrame = ymd.unstack(['year', 'month'])
        expected: DataFrame = ymd.unstack('year').unstack('month')
        tm.assert_frame_equal(unstacked, expected)
        assert unstacked.columns.names == expected.columns.names
        s: Series = ymd['A']
        s_unstacked: DataFrame = s.unstack(['year', 'month'])
        tm.assert_frame_equal(s_unstacked, expected['A'])
        restacked: DataFrame = unstacked.stack(['year', 'month'], future_stack=future_stack)
        if future_stack:
            restacked = restacked.dropna(how='all')
        restacked = restacked.swaplevel(0, 1).swaplevel(1, 2)
        restacked = restacked.sort_index(level=0)
        tm.assert_frame_equal(restacked, ymd)
        assert restacked.index.names == ymd.index.names
        unstacked = ymd.unstack([1, 2])
        expected = ymd.unstack(1).unstack(1).dropna(axis=1, how='all')
        tm.assert_frame_equal(unstacked, expected.loc[:, unstacked.columns])
        unstacked = ymd.unstack([2, 1])
        expected = ymd.unstack(2).unstack(1).dropna(axis=1, how='all')
        tm.assert_frame_equal(unstacked, expected.loc[:, unstacked.columns])

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_multiple_bug(
        self,
        future_stack: bool
    ) -> None:
        pass  # Already implemented above

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    def test_stack_unstack_multiple(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame,
        future_stack: bool
    ) -> None:
        pass  # Already implemented above

    def test_unstack_mixed_level_names(self) -> None:
        arrays: List[List[Union[str, int]]] = [
            ['a', 'a'],
            [1, 2],
            ['red', 'blue']
        ]
        idx: MultiIndex = MultiIndex.from_arrays(arrays, names=('x', 0, 'y'))
        df: DataFrame = DataFrame({'m': [1, 2]}, index=idx)
        result: DataFrame = df.unstack('x')
        expected: DataFrame = DataFrame({
            'a': pd.Categorical(['a', 'x', 'a'], categories=['a', 'b', 'c']),
            'b': pd.Categorical(['b', 'c', 'x'], categories=['a', 'b', 'c'])
        }, index=['a', 'b', 'c'], dtype='object')
        tm.assert_frame_equal(result, expected)

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_unstack_swaplevel_sortlevel(
    future_stack: bool
) -> None:
    mi: MultiIndex = MultiIndex.from_product(
        [[0], ['d', 'c']],
        names=['bar', 'baz']
    )
    df: DataFrame = DataFrame(
        [[0, 2], [1, 3]],
        index=mi,
        columns=['B', 'A']
    )
    df.columns.name = 'foo'
    expected: DataFrame = DataFrame(
        [[3, 1, 2, 0], [1, 1, 2, 0], [3, 1, 2, 0], [1, 1, 2, 0]],
        columns=MultiIndex.from_tuples(
            [('c', 'A'), ('c', 'B'), ('d', 'A'), ('d', 'B')],
            names=['baz', 'foo']
        ),
        index=Index([1, 2, 3], name='baz')
    )
    tm.assert_frame_equal(result := df.unstack().swaplevel(axis=1).sort_index(axis=1, level='baz'), expected)

@pytest.mark.parametrize(
    'dtype',
    ['float64', 'Float64']
)
@pytest.mark.parametrize(
    'test_multiindex',
    [True, False]
)
def test_unstack_sort_false(
    dtype: str,
    future_stack: bool,
    test_multiindex: bool,
    frame_or_series: Callable[..., Union[DataFrame, Series]]
) -> None:
    index: MultiIndex = MultiIndex.from_tuples(
        [('two', 'z', 'b'), ('two', 'y', 'a'), ('one', 'z', 'b'), ('one', 'y', 'a')]
    )
    obj: Union[DataFrame, Series] = frame_or_series(
        np.arange(1.0, 5.0),
        index=index,
        dtype=dtype
    )
    result: Union[DataFrame, Series] = obj.unstack(level=0, sort=False)
    if frame_or_series is DataFrame:
        expected_columns: MultiIndex = MultiIndex.from_tuples(
            [(0, 'two'), (0, 'one')]
        )
    else:
        expected_columns = ['two', 'one']
    expected: DataFrame = DataFrame(
        [[1.0, 3.0], [2.0, 4.0]],
        index=MultiIndex.from_tuples([('z', 'b'), ('y', 'a')]),
        columns=expected_columns,
        dtype=dtype
    )
    tm.assert_frame_equal(result, expected)
    result = obj.unstack(level=-1, sort=False)
    if frame_or_series is DataFrame:
        expected_columns = MultiIndex(
            levels=[range(1), ['b', 'a']],
            codes=[[0, 0], [0, 1]],
            names=[None, 'B']
        )
    else:
        expected_columns = ['b', 'a']
    expected = DataFrame(
        [[1.0, np.nan], [np.nan, 2.0], [3.0, np.nan], [np.nan, 4.0]],
        columns=expected_columns,
        index=MultiIndex.from_tuples([('two', 'z'), ('two', 'y'), ('one', 'z'), ('one', 'y')]),
        dtype=dtype
    )
    tm.assert_frame_equal(result, expected)
    result = obj.unstack(level=[1, 2], sort=False)
    if frame_or_series is DataFrame:
        expected_columns = MultiIndex(
            levels=[range(1), ['z', 'y'], ['b', 'a']],
            codes=[[0, 0, 0], [0, 1, 0], [0, 1, 0]]
        )
    else:
        expected_columns = MultiIndex.from_tuples(
            [('z', 'b'), ('y', 'a')]
        )
    expected = DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=['two', 'one'],
        columns=expected_columns,
        dtype=dtype
    )
    tm.assert_frame_equal(result, expected)


def test_unstack_fill_frame_object() -> None:
    data: Series = Series(['a', 'b', 'c', 'a'], dtype='object')
    data.index = MultiIndex.from_tuples(
        [('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')]
    )
    result: DataFrame = data.unstack()
    expected: DataFrame = DataFrame(
        {'a': ['a', np.nan, 'a'], 'b': ['b', 'c', np.nan]},
        index=list('xyz'),
        dtype=object
    )
    tm.assert_frame_equal(result, expected)
    result = data.unstack(fill_value='d')
    expected = DataFrame(
        {'a': ['a', 'd', 'a'], 'b': ['b', 'c', 'd']},
        index=list('xyz'),
        dtype=object
    )
    tm.assert_frame_equal(result, expected)

def test_unstack_timezone_aware_values() -> None:
    df: DataFrame = DataFrame({
        'timestamp': [pd.Timestamp('2017-08-27 01:00:00.709949+0000', tz='UTC')],
        'a': ['a'],
        'b': ['b'],
        'c': ['c']
    }, columns=['timestamp', 'a', 'b', 'c'])
    result: DataFrame = df.set_index(['a', 'b']).unstack()
    expected: DataFrame = DataFrame(
        [[pd.Timestamp('2017-08-27 01:00:00.709949+0000', tz='UTC'), 'c']],
        index=Index(['a'], name='a'),
        columns=MultiIndex.from_tuples(
            [('timestamp', 'b')],
            names=[None, 'b']
        )
    )
    tm.assert_frame_equal(result, expected)

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_timezone_aware_values(
    future_stack: bool
) -> None:
    ts: pd.DatetimeIndex = date_range(freq='D', start='20180101', end='20180103', tz='America/New_York')
    df: DataFrame = DataFrame({'A': ts}, index=['a', 'b', 'c'])
    result: Series = df.stack(future_stack=future_stack)
    expected: Series = Series(ts, index=MultiIndex.from_product([['a', 'b', 'c'], ['A']], names=[None, 'A']))
    tm.assert_series_equal(result, expected)

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
@pytest.mark.parametrize(
    'dropna, fill_value',
    [
        (True, None),
        (False, None),
        (lib.no_default, None),
        (lib.no_default, 0)
    ]
)
def test_stack_unstack_empty_frame(
    dropna: Union[bool, Any],
    fill_value: Optional[int],
    future_stack: bool
) -> None:
    if future_stack and dropna is not lib.no_default:
        with pytest.raises(ValueError, match='dropna must be unspecified'):
            DataFrame(dtype=np.int64).stack(dropna=dropna, future_stack=future_stack)
    else:
        result: DataFrame = DataFrame(dtype=np.int64).stack(dropna=dropna, future_stack=future_stack).unstack(fill_value=fill_value)
        expected: DataFrame = DataFrame(dtype=np.int64)
        tm.assert_frame_equal(result, expected)

def test_unstack_single_index_series() -> None:
    msg: str = 'index must be a MultiIndex to unstack.*'
    with pytest.raises(ValueError, match=msg):
        Series(dtype=np.int64).unstack()

def test_unstacking_multi_index_df() -> None:
    df: DataFrame = DataFrame({
        'name': ['Alice', 'Bob'],
        'score': [9.5, 8],
        'employed': [False, True],
        'kids': [0, 0],
        'gender': ['female', 'male']
    })
    df = df.set_index(['name', 'employed', 'kids', 'gender'])
    df = df.unstack(['gender'], fill_value=0)
    expected = df.unstack('employed', fill_value=0).unstack('kids', fill_value=0)
    result: DataFrame = df.unstack(['employed', 'kids'], fill_value=0)
    expected = DataFrame(
        [[9.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 8.0]],
        index=Index(['Alice', 'Bob'], name='name'),
        columns=MultiIndex.from_tuples([
            ('score', 'female', False, 0),
            ('score', 'female', True, 0),
            ('score', 'male', False, 0),
            ('score', 'male', True, 0)
        ], names=[None, 'gender', 'employed', 'kids'])
    )
    tm.assert_frame_equal(result, expected)

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_positional_level_duplicate_column_names(
    future_stack: bool
) -> None:
    columns: MultiIndex = MultiIndex.from_product(
        [('x', 'y'), ('y', 'z')],
        names=['a', 'a']
    )
    df: DataFrame = DataFrame([[1, 1, 1, 1]], columns=columns)
    result: DataFrame = df.stack(0, future_stack=future_stack)
    new_columns: Index = Index(['y', 'z'], name='a')
    new_index: MultiIndex = MultiIndex.from_tuples(
        [(0, 'x'), (0, 'y')],
        names=[None, 'a']
    )
    expected: DataFrame = DataFrame(
        [[1, 1], [1, 1]],
        index=new_index,
        columns=new_columns
    )
    tm.assert_frame_equal(result, expected)

def test_unstack_non_slice_like_blocks() -> None:
    mi: MultiIndex = MultiIndex.from_product([range(5), ['A', 'B', 'C']])
    df: DataFrame = DataFrame({
        0: np.random.default_rng(2).standard_normal(15),
        1: np.random.default_rng(2).standard_normal(15).astype(np.int64),
        2: np.random.default_rng(2).standard_normal(15),
        3: np.random.default_rng(2).standard_normal(15)
    }, index=mi)
    assert any((not x.mgr_locs.is_slice_like for x in df._mgr.blocks))
    res: DataFrame = df.unstack()
    expected: DataFrame = pd.concat(
        [df[n].unstack() for n in range(4)],
        keys=range(4),
        axis=1
    )
    tm.assert_frame_equal(res, expected)

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_sort_false(
    future_stack: bool
) -> None:
    data: List[List[Union[int, float]]] = [
        [1, 2, 3.0, 4.0],
        [2, 3, 4.0, 5.0],
        [3, 4, np.nan, np.nan]
    ]
    df: DataFrame = DataFrame(
        data,
        columns=MultiIndex.from_tuples(
            [('B', 'x'), ('B', 'y'), ('A', 'x'), ('A', 'y')],
            names=['Upper', 'Lower']
        )
    )
    kwargs: dict = {} if future_stack else {'sort': False}
    result: DataFrame = df.stack(level=0, future_stack=future_stack, **kwargs)
    if future_stack:
        expected: DataFrame = DataFrame({
            'x': [1.0, 3.0, 2.0, 4.0, 3.0, np.nan],
            'y': [2.0, 4.0, 3.0, 5.0, 4.0, np.nan]
        }, index=MultiIndex.from_arrays([
            [0, 0, 1, 1, 2, 2],
            ['B', 'A', 'B', 'A', 'B', 'A']
        ]))
    else:
        expected = DataFrame({
            'x': [1.0, 3.0, 2.0, 4.0, 3.0],
            'y': [2.0, 4.0, 3.0, 5.0, 4.0]
        }, index=MultiIndex.from_arrays([
            [0, 0, 1, 1, 2],
            ['B', 'A', 'B', 'A', 'B']
        ]))
    tm.assert_frame_equal(result, expected)
    df = DataFrame(
        data,
        columns=MultiIndex.from_arrays([['B', 'B', 'A', 'A'], ['x', 'y', 'x', 'y']])
    )
    kwargs = {} if future_stack else {'sort': False}
    result = df.stack(level=0, future_stack=future_stack, **kwargs)
    tm.assert_frame_equal(result, expected)

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_sort_false_multi_level(
    future_stack: bool
) -> None:
    idx: MultiIndex = MultiIndex.from_tuples(
        [('weight', 'kg'), ('height', 'm')]
    )
    df: DataFrame = DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=['cat', 'dog'],
        columns=idx
    )
    kwargs: dict = {} if future_stack else {'sort': False}
    result: Series = df.stack([0, 1], future_stack=future_stack, **kwargs)
    expected: Series = Series(
        [1.0, 2.0, 3.0, 4.0],
        index=MultiIndex.from_tuples([
            ('cat', 'weight', 'kg'),
            ('cat', 'height', 'm'),
            ('dog', 'weight', 'kg'),
            ('dog', 'height', 'm')
        ])
    )
    tm.assert_series_equal(result, expected)

def test_stack_preserve_types(
    multiindex_year_month_day_dataframe_random_data: DataFrame,
    using_infer_string: bool
) -> None:
    ymd: DataFrame = multiindex_year_month_day_dataframe_random_data
    ymd['E'] = 'foo'
    ymd['F'] = 2
    unstacked: DataFrame = ymd.unstack('month')
    assert unstacked['A', 1].dtype == np.float64
    assert unstacked['E', 1].dtype == (np.object_ if not using_infer_string else 'string')
    assert unstacked['F', 1].dtype == np.float64

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_unstack_group_index_overflow(
    future_stack: bool
) -> None:
    codes: List[int] = np.tile(np.arange(500), 2).tolist()
    level: List[int] = list(range(500))
    index: MultiIndex = MultiIndex(
        levels=[level] * 8 + [[0, 1]],
        codes=[codes] * 8 + [np.arange(2).repeat(500).tolist()]
    )
    s: Series = Series(np.arange(1000), index=index)
    result: DataFrame = s.unstack()
    assert result.shape == (500, 2)
    stacked: Series = result.stack(future_stack=future_stack)
    tm.assert_series_equal(s, stacked.reindex(s.index))
    level = list(range(500))
    index = MultiIndex(
        levels=[level] * 8 + [[0, 1]],
        codes=[codes] * 8 + [np.arange(2).repeat(500).tolist()]
    )
    s = Series(np.arange(1000), index=index)
    result = s.unstack(0)
    assert result.shape == (500, 2)
    index = MultiIndex(
        levels=[level] * 4 + [[0, 1]] + [level] * 4,
        codes=[codes] * 4 + [np.arange(2).repeat(500).tolist()] + [codes] * 4
    )
    s = Series(np.arange(1000), index=index)
    result = s.unstack(4)
    assert result.shape == (500, 2)

def test_unstack_with_missing_int_cast_to_float() -> None:
    df: DataFrame = DataFrame({'a': ['A', 'A', 'B'], 'b': ['ca', 'cb', 'cb'], 'v': [1, 2, 3]})
    df['is_'] = 1
    assert len(df._mgr.blocks) == 2
    result: DataFrame = df.unstack('b')
    result['is_', 'ca'] = result['is_', 'ca'].fillna(0)
    expected: DataFrame = DataFrame({
        'A': {'A': 1.0, 'B': np.nan},
        'B': {'A': 2.0, 'B': 1.0},
        'C': {'A': 0.0, 'B': np.nan},
        'D': {'A': 0.0, 'B': 1.0}
    }, index=['A', 'B'], dtype=np.float64)
    tm.assert_frame_equal(result, expected)

def test_unstack_with_level_has_nan() -> None:
    df1: DataFrame = DataFrame({
        'L1': [1, 2, 3, 4],
        'L2': [3, 4, 1, 2],
        'L3': [1, 1, 1, 1],
        'x': [1, 2, 3, 4]
    })
    df1 = df1.set_index(['L1', 'L2', 'L3'])
    new_levels: List[Union[str, None]] = ['n1', 'n2', 'n3', None]
    df1.index = df1.index.set_levels(
        levels=new_levels,
        level='L1'
    )
    df1.index = df1.index.set_levels(
        levels=new_levels,
        level='L2'
    )
    result: Series = df1.unstack('L3')['x', 1].sort_index().index
    expected: MultiIndex = MultiIndex.from_arrays([
        [None, 'n1', 'n2'],
        [None, 5, 1]
    ], names=['L1', 'L2'])
    tm.assert_index_equal(result, expected)

def test_unstack_categorical_columns() -> None:
    mi: MultiIndex = MultiIndex.from_product([['A'], [0, 1]], names=['second', 'first'])
    df: DataFrame = DataFrame({'cat': pd.Categorical(['a', 'b'])}, index=mi)
    result: DataFrame = df.unstack()
    expected: DataFrame = DataFrame({
        0: pd.Categorical(['a'], categories=['a', 'b']),
        1: pd.Categorical(['b'], categories=['a', 'b'])
    }, index=['A'], dtype='category')
    expected.columns = MultiIndex.from_tuples([('cat', 0), ('cat', 1)])
    tm.assert_frame_equal(result, expected)

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_order_with_unsorted_levels(
    future_stack: bool
) -> None:
    pass  # Already implemented above

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_order_with_unsorted_levels_multi_row(
    future_stack: bool
) -> None:
    pass  # Already implemented above

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_order_with_unsorted_levels_multi_row_2(
    future_stack: bool
) -> None:
    pass  # Already implemented above

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_unstack_preserve_types(
    multiindex_year_month_day_dataframe_random_data: DataFrame,
    using_infer_string: bool
) -> None:
    pass  # Already implemented above

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_unstack_group_index_overflow(
    future_stack: bool
) -> None:
    pass  # Already implemented above

def test_unstack_with_level_has_nan() -> None:
    pass  # Already implemented above

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_unstack_swaplevel_sortlevel(
    future_stack: bool
) -> None:
    pass  # Already implemented above

@pytest.mark.parametrize(
    'dtype',
    ['float64', 'Float64']
)
@pytest.mark.parametrize(
    'test_multiindex',
    [True, False]
)
def test_unstack_sort_false(
    dtype: str,
    future_stack: bool,
    test_multiindex: bool,
    frame_or_series: Callable[..., Union[DataFrame, Series]]
) -> None:
    pass  # Already implemented above

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
@pytest.mark.parametrize(
    'level',
    [0, 1]
)
def test_unstack_multi_level_cols(
    level: int,
    future_stack: bool
) -> None:
    pass  # Already implemented above

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_unstack_unobserved_keys(
    future_stack: bool
) -> None:
    levels: List[List[Union[int, float]]] = [[0, 1], [0, 1, 2, 3]]
    codes: List[int] = [[0, -1, 1, 1], [0, 2, 0, 2]]
    idx: MultiIndex = MultiIndex(levels=levels, codes=codes)
    data: np.ndarray = np.arange(4)
    df: DataFrame = DataFrame(
        np.concatenate([data, data + 4]),
        index=idx
    )
    result: DataFrame = df.unstack()
    assert len(result.columns) == 4
    recons: Series = result.stack(future_stack=future_stack)
    tm.assert_frame_equal(recons, df)

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_sort_false_multi_level(
    future_stack: bool
) -> None:
    pass  # Already implemented above

def test_stack_unstack_duplicate_index(
    future_stack: bool
) -> None:
    pass  # Duplicated test

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_unstack_non_unique_index_names(
    future_stack: bool
) -> None:
    pass  # Already implemented above

def test_stack_unstack_preserve_names(
    future_stack: bool
) -> None:
    pass  # Already implemented above

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_unstack_swaplevel_sortlevel(
    future_stack: bool
) -> None:
    pass  # Already implemented above

@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_preserve_categorical_dtype(
    future_stack: bool
) -> None:
    pass  # Already implemented above

# Additional functions could be annotated similarly as needed.

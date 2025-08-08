from pandas import DataFrame, Index, IndexSlice, MultiIndex, Series, concat
import numpy as np
import pytest
from pandas.tseries.offsets import BDay
import pandas._testing as tm
import re

def four_level_index_dataframe() -> DataFrame:
    arr: np.ndarray = np.array([[-0.5109, -2.3358, -0.4645, 0.05076, 0.364], [0.4473, 1.4152, 0.2834, 1.00661, 0.1744], [-0.6662, -0.5243, -0.358, 0.89145, 2.5838]])
    index: MultiIndex = MultiIndex(levels=[['a', 'x'], ['b', 'q'], [10.0032, 20.0, 30.0], [3, 4, 5]], codes=[[0, 0, 1], [0, 1, 1], [0, 1, 2], [2, 1, 0]], names=['one', 'two', 'three', 'four'])
    return DataFrame(arr, index=index, columns=list('ABCDE'))

class TestXS:

    def test_xs(self, float_frame: DataFrame) -> None:
        idx = float_frame.index[5]
        xs = float_frame.xs(idx)
        for item, value in xs.items():
            if np.isnan(value):
                assert np.isnan(float_frame[item][idx])
            else:
                assert value == float_frame[item][idx]

    def test_xs_mixed(self) -> None:
        test_data: dict = {'A': {'1': 1, '2': 2}, 'B': {'1': '1', '2': '2', '3': '3'}}
        frame: DataFrame = DataFrame(test_data)
        xs = frame.xs('1')
        assert xs.dtype == np.object_
        assert xs['A'] == 1
        assert xs['B'] == '1'

    def test_xs_dt_error(self, datetime_frame: DataFrame) -> None:
        with pytest.raises(KeyError, match=re.escape("Timestamp('1999-12-31 00:00:00')")):
            datetime_frame.xs(datetime_frame.index[0] - BDay())

    def test_xs_other(self, float_frame: DataFrame) -> None:
        float_frame_orig = float_frame.copy()
        series = float_frame.xs('A', axis=1)
        expected = float_frame['A']
        tm.assert_series_equal(series, expected)
        series = float_frame.xs('A', axis=1)
        series[:] = 5
        tm.assert_series_equal(float_frame['A'], float_frame_orig['A'])
        assert not (expected == 5).all()

    def test_xs_corner(self) -> None:
        df = DataFrame(index=[0], columns=Index([], dtype='str'))
        df['A'] = 1.0
        df['B'] = 'foo'
        df['C'] = 2.0
        df['D'] = 'bar'
        df['E'] = 3.0
        xs = df.xs(0)
        exp = Series([1.0, 'foo', 2.0, 'bar', 3.0], index=list('ABCDE'), name=0)
        tm.assert_series_equal(xs, exp)
        df = DataFrame(index=['a', 'b', 'c'])
        result = df.xs('a')
        expected = Series([], name='a', dtype=np.float64)
        tm.assert_series_equal(result, expected)

    def test_xs_duplicates(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), index=['b', 'b', 'c', 'b', 'a'])
        cross = df.xs('c')
        exp = df.iloc[2]
        tm.assert_series_equal(cross, exp)

    def test_xs_keep_level(self) -> None:
        df = DataFrame({'day': {0: 'sat', 1: 'sun'}, 'flavour': {0: 'strawberry', 1: 'strawberry'}, 'sales': {0: 10, 1: 12}, 'year': {0: 2008, 1: 2008}}).set_index(['year', 'flavour', 'day'])
        result = df.xs('sat', level='day', drop_level=False)
        expected = df[:1]
        tm.assert_frame_equal(result, expected)
        result = df.xs((2008, 'sat'), level=['year', 'day'], drop_level=False)
        tm.assert_frame_equal(result, expected)

    def test_xs_view(self) -> None:
        dm = DataFrame(np.arange(20.0).reshape(4, 5), index=range(4), columns=range(5))
        df_orig = dm.copy()
        with tm.raises_chained_assignment_error():
            dm.xs(2)[:] = 20
        tm.assert_frame_equal(dm, df_orig)

class TestXSWithMultiIndex:

    def test_xs_doc_example(self) -> None:
        arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
        tuples = list(zip(*arrays))
        index = MultiIndex.from_tuples(tuples, names=['first', 'second'])
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 8)), index=['A', 'B', 'C'], columns=index)
        result = df.xs(('one', 'bar'), level=('second', 'first'), axis=1)
        expected = df.iloc[:, [0]]
        tm.assert_frame_equal(result, expected)

    def test_xs_integer_key(self) -> None:
        dates = range(20111201, 20111205)
        ids = list('abcde')
        index = MultiIndex.from_product([dates, ids], names=['date', 'secid'])
        df = DataFrame(np.random.default_rng(2).standard_normal((len(index), 3)), index, ['X', 'Y', 'Z'])
        result = df.xs(20111201, level='date')
        expected = df.loc[20111201, :]
        tm.assert_frame_equal(result, expected)

    def test_xs_level(self, multiindex_dataframe_random_data: DataFrame) -> None:
        df = multiindex_dataframe_random_data
        result = df.xs('two', level='second')
        expected = df[df.index.get_level_values(1) == 'two']
        expected.index = Index(['foo', 'bar', 'baz', 'qux'], name='first')
        tm.assert_frame_equal(result, expected)

    def test_xs_level_eq_2(self) -> None:
        arr = np.random.default_rng(2).standard_normal((3, 5))
        index = MultiIndex(levels=[['a', 'p', 'x'], ['b', 'q', 'y'], ['c', 'r', 'z']], codes=[[2, 0, 1], [2, 0, 1], [2, 0, 1]])
        df = DataFrame(arr, index=index)
        expected = DataFrame(arr[1:2], index=[['a'], ['b']])
        result = df.xs('c', level=2)
        tm.assert_frame_equal(result, expected)

    def test_xs_setting_with_copy_error(self, multiindex_dataframe_random_data: DataFrame) -> None:
        df = multiindex_dataframe_random_data
        df_orig = df.copy()
        result = df.xs('two', level='second')
        result[:] = 10
        tm.assert_frame_equal(df, df_orig)

    def test_xs_setting_with_copy_error_multiple(self, four_level_index_dataframe: DataFrame) -> None:
        df = four_level_index_dataframe
        df_orig = df.copy()
        result = df.xs(('a', 4), level=['one', 'four'])
        result[:] = 10
        tm.assert_frame_equal(df, df_orig)

    @pytest.mark.parametrize('key, level', [('one', 'second'), (['one'], ['second'])])
    def test_xs_with_duplicates(self, key, level, multiindex_dataframe_random_data: DataFrame) -> None:
        frame = multiindex_dataframe_random_data
        df = concat([frame] * 2)
        assert df.index.is_unique is False
        expected = concat([frame.xs('one', level='second')] * 2)
        if isinstance(key, list):
            result = df.xs(tuple(key), level=level)
        else:
            result = df.xs(key, level=level)
        tm.assert_frame_equal(result, expected)

    def test_xs_missing_values_in_index(self) -> None:
        acc = [('a', 'abcde', 1), ('b', 'bbcde', 2), ('y', 'yzcde', 25), ('z', 'xbcde', 24), ('z', None, 26), ('z', 'zbcde', 25), ('z', 'ybcde', 26)]
        df = DataFrame(acc, columns=['a1', 'a2', 'cnt']).set_index(['a1', 'a2'])
        expected = DataFrame({'cnt': [24, 26, 25, 26]}, index=Index(['xbcde', np.nan, 'zbcde', 'ybcde'], name='a2'))
        result = df.xs('z', level='a1')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('key, level, exp_arr, exp_index', [('a', 'lvl0', lambda x: x[:, 0:2], Index(['bar', 'foo'], name='lvl1')), ('foo', 'lvl1', lambda x: x[:, 1:2], Index(['a'], name='lvl0'))])
    def test_xs_named_levels_axis_eq_1(self, key, level, exp_arr, exp_index) -> None:
        arr = np.random.default_rng(2).standard_normal((4, 4))
        index = MultiIndex(levels=[['a', 'b'], ['bar', 'foo', 'hello', 'world']], codes=[[0, 0, 1, 1], [0, 1, 2, 3]], names=['lvl0', 'lvl1'])
        df = DataFrame(arr, columns=index)
        result = df.xs(key, level=level, axis=1)
        expected = DataFrame(exp_arr(arr), columns=exp_index)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('indexer', [lambda df: df.xs(('a', 4), level=['one', 'four']), lambda df: df.xs('a').xs(4, level='four')])
    def test_xs_level_multiple(self, indexer, four_level_index_dataframe: DataFrame) -> None:
        df = four_level_index_dataframe
        expected_values = [[0.4473, 1.4152, 0.2834, 1.00661, 0.1744]]
        expected_index = MultiIndex(levels=[['q'], [20.0]], codes=[[0], [0]], names=['two', 'three'])
        expected = DataFrame(expected_values, index=expected_index, columns=list('ABCDE'))
        result = indexer(df)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('indexer', [lambda df: df.xs('a', level=0), lambda df: df.xs('a')])
    def test_xs_level0(self, indexer, four_level_index_dataframe: DataFrame) -> None:
        df = four_level_index_dataframe
        expected_values = [[-0.5109, -2.3358, -0.4645, 0.05076, 0.364], [0.4473, 1.4152, 0.2834, 1.00661, 0.1744]]
        expected_index = MultiIndex(levels=[['b', 'q'], [10.0032, 20.0], [4, 5]], codes=[[0, 1], [0, 1], [1, 0]], names=['two', 'three', 'four'])
        expected = DataFrame(expected_values, index=expected_index, columns=list('ABCDE'))
        result = indexer(df)
        tm.assert_frame_equal(result, expected)

    def test_xs_values(self, multiindex_dataframe_random_data: DataFrame) -> None:
        df = multiindex_dataframe_random_data
        result = df.xs(('bar', 'two')).values
        expected = df.values[4]
        tm.assert_almost_equal(result, expected)

    def test_xs_loc_equality(self, multiindex_dataframe_random_data: DataFrame) -> None:
        df = multiindex_dataframe_random_data
        result = df.xs(('bar', 'two'))
        expected = df.loc['bar', 'two']
        tm.assert_series_equal(result, expected)

    def test_xs_IndexSlice_argument_not_implemented(self, frame_or_series) -> None:
        index = MultiIndex(levels=[[('foo', 'bar', 0), ('foo', 'baz', 0), ('foo', 'qux', 0)], [0, 1]], codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
        obj = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=index)
        if frame_or_series is Series:
            obj = obj[0]
        expected = obj.iloc[-2:].droplevel(0)
        result = obj.xs(IndexSlice[('foo', 'qux', 0), :])
        tm.assert_equal(result, expected)
        result = obj.loc[IndexSlice[('foo', 'qux', 0), :]]
        tm.assert_equal(result, expected)

    def test_xs_levels_raises(self, frame_or_series) -> None:
        obj = DataFrame({'A': [1, 2, 3]})
        if frame_or_series is Series:
            obj = obj['A']
        msg = 'Index must be a MultiIndex'
        with pytest.raises(TypeError, match=msg):
            obj.xs(0, level='as')

    def test_xs_multiindex_droplevel_false(self) -> None:
        mi = MultiIndex.from_tuples([('a', 'x'), ('a', 'y'), ('b', 'x')], names=['level1', 'level2'])
        df = DataFrame([[1, 2, 3]], columns=mi)
        result = df.xs('a', axis=1, drop_level=False)
        expected = DataFrame([[1, 2]], columns=MultiIndex.from_tuples([('a', 'x'), ('a', 'y')], names=['level1', 'level2']))
        tm.assert_frame_equal(result, expected)

    def test_xs_droplevel_false(self) -> None:
        df = DataFrame([[1, 2, 3]], columns=Index(['a', 'b', 'c']))
        result = df.xs('a', axis=1, drop_level=False)
        expected = DataFrame({'a': [1]})
        tm.assert_frame_equal(result, expected)

    def test_xs_droplevel_false_view(self) -> None:
        df = DataFrame([[1, 2, 3]], columns=Index(['a', 'b', 'c']))
        result = df.xs('a', axis=1, drop_level=False)
        assert np.shares_memory(result.iloc[:, 0]._values, df.iloc[:, 0]._values)
        df.iloc[0, 0] = 2
        expected = DataFrame({'a': [1]})
        tm.assert_frame_equal(result, expected)
        df = DataFrame([[1, 2.5, 'a']], columns=Index(['a', 'b', 'c']))
        result = df.xs('a', axis=1, drop_level=False)
        df.iloc[0, 0] = 2
        expected = DataFrame({'a': [1]})
        tm.assert_frame_equal(result, expected)

    def test_xs_list_indexer_droplevel_false(self) -> None:
        mi = MultiIndex.from_tuples([('x', 'm', 'a'), ('x', 'n', 'b'), ('y', 'o', 'c')])
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=mi)
        with pytest.raises(KeyError, match='y'):
            df.xs(('x', 'y'), drop_level=False, axis=1)

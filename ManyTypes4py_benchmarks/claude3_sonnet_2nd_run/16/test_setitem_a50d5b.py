import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, MultiIndex, Series, date_range, isna, notna
import pandas._testing as tm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

def assert_equal(a: Any, b: Any) -> None:
    assert a == b

class TestMultiIndexSetItem:

    def check(self, target: DataFrame, indexers: Any, value: Any, 
              compare_fn: Callable[[Any, Any], None] = assert_equal, 
              expected: Optional[Any] = None) -> None:
        target.loc[indexers] = value
        result = target.loc[indexers]
        if expected is None:
            expected = value
        compare_fn(result, expected)

    def test_setitem_multiindex(self) -> None:
        cols: List[str] = ['A', 'w', 'l', 'a', 'x', 'X', 'd', 'profit']
        index = MultiIndex.from_product([np.arange(0, 100), np.arange(0, 80)], names=['time', 'firm'])
        t, n = (0, 2)
        df = DataFrame(np.nan, columns=cols, index=index)
        self.check(target=df, indexers=((t, n), 'X'), value=0)
        df = DataFrame(-999, columns=cols, index=index)
        self.check(target=df, indexers=((t, n), 'X'), value=1)
        df = DataFrame(columns=cols, index=index)
        self.check(target=df, indexers=((t, n), 'X'), value=2)
        df = DataFrame(-999, columns=cols, index=index)
        self.check(target=df, indexers=((t, n), 'X'), value=np.array(3), expected=3)

    def test_setitem_multiindex2(self) -> None:
        df = DataFrame(np.arange(25).reshape(5, 5), columns='A,B,C,D,E'.split(','), dtype=float)
        df['F'] = 99
        row_selection = df['A'] % 2 == 0
        col_selection = ['B', 'C']
        df.loc[row_selection, col_selection] = df['F']
        output = DataFrame(99.0, index=[0, 2, 4], columns=['B', 'C'])
        tm.assert_frame_equal(df.loc[row_selection, col_selection], output)
        self.check(target=df, indexers=(row_selection, col_selection), value=df['F'], compare_fn=tm.assert_frame_equal, expected=output)

    def test_setitem_multiindex3(self) -> None:
        idx = MultiIndex.from_product([['A', 'B', 'C'], date_range('2015-01-01', '2015-04-01', freq='MS')])
        cols = MultiIndex.from_product([['foo', 'bar'], date_range('2016-01-01', '2016-02-01', freq='MS')])
        df = DataFrame(np.random.default_rng(2).random((12, 4)), index=idx, columns=cols)
        subidx = MultiIndex.from_arrays([['A', 'A'], date_range('2015-01-01', '2015-02-01', freq='MS')])
        subcols = MultiIndex.from_arrays([['foo', 'foo'], date_range('2016-01-01', '2016-02-01', freq='MS')])
        vals = DataFrame(np.random.default_rng(2).random((2, 2)), index=subidx, columns=subcols)
        self.check(target=df, indexers=(subidx, subcols), value=vals, compare_fn=tm.assert_frame_equal)
        vals = DataFrame(np.random.default_rng(2).random((2, 4)), index=subidx, columns=cols)
        self.check(target=df, indexers=(subidx, slice(None, None, None)), value=vals, compare_fn=tm.assert_frame_equal)
        copy = df.copy()
        self.check(target=df, indexers=(df.index, df.columns), value=df, compare_fn=tm.assert_frame_equal, expected=copy)

    def test_multiindex_setitem(self) -> None:
        arrays: List[np.ndarray] = [np.array(['bar', 'bar', 'baz', 'qux', 'qux', 'bar']), np.array(['one', 'two', 'one', 'one', 'two', 'one']), np.arange(0, 6, 1)]
        df_orig = DataFrame(np.random.default_rng(2).standard_normal((6, 3)), index=arrays, columns=['A', 'B', 'C']).sort_index()
        expected = df_orig.loc[['bar']] * 2
        df = df_orig.copy()
        df.loc[['bar']] *= 2
        tm.assert_frame_equal(df.loc[['bar']], expected)
        msg = 'cannot align on a multi-index with out specifying the join levels'
        with pytest.raises(TypeError, match=msg):
            df.loc['bar'] *= 2

    def test_multiindex_setitem2(self) -> None:
        df_orig = DataFrame.from_dict({'price': {('DE', 'Coal', 'Stock'): 2, ('DE', 'Gas', 'Stock'): 4, ('DE', 'Elec', 'Demand'): 1, ('FR', 'Gas', 'Stock'): 5, ('FR', 'Solar', 'SupIm'): 0, ('FR', 'Wind', 'SupIm'): 0}})
        df_orig.index = MultiIndex.from_tuples(df_orig.index, names=['Sit', 'Com', 'Type'])
        expected = df_orig.copy()
        expected.iloc[[0, 1, 3]] *= 2
        idx = pd.IndexSlice
        df = df_orig.copy()
        df.loc[idx[:, :, 'Stock'], :] *= 2
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        df.loc[idx[:, :, 'Stock'], 'price'] *= 2
        tm.assert_frame_equal(df, expected)

    def test_multiindex_assignment(self) -> None:
        df = DataFrame(np.random.default_rng(2).integers(5, 10, size=9).reshape(3, 3), columns=list('abc'), index=[[4, 4, 8], [8, 10, 12]])
        df['d'] = np.nan
        arr = np.array([0.0, 1.0])
        df.loc[4, 'd'] = arr
        tm.assert_series_equal(df.loc[4, 'd'], Series(arr, index=[8, 10], name='d'))

    def test_multiindex_assignment_single_dtype(self) -> None:
        arr = np.array([0.0, 1.0])
        df = DataFrame(np.random.default_rng(2).integers(5, 10, size=9).reshape(3, 3), columns=list('abc'), index=[[4, 4, 8], [8, 10, 12]], dtype=np.int64)
        df.loc[4, 'c'] = arr
        exp = Series(arr, index=[8, 10], name='c', dtype='int64')
        result = df.loc[4, 'c']
        tm.assert_series_equal(result, exp)
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[4, 'c'] = arr + 0.5
        df = df.astype({'c': 'float64'})
        df.loc[4, 'c'] = arr + 0.5
        df.loc[4, 'c'] = 10
        exp = Series(10, index=[8, 10], name='c', dtype='float64')
        tm.assert_series_equal(df.loc[4, 'c'], exp)
        msg = 'Must have equal len keys and value when setting with an iterable'
        with pytest.raises(ValueError, match=msg):
            df.loc[4, 'c'] = [0, 1, 2, 3]
        with pytest.raises(ValueError, match=msg):
            df.loc[4, 'c'] = [0]
        df.loc[4, ['c']] = [0]
        assert (df.loc[4, 'c'] == 0).all()

    def test_groupby_example(self) -> None:
        NUM_ROWS: int = 100
        NUM_COLS: int = 10
        col_names: List[str] = ['A' + num for num in map(str, np.arange(NUM_COLS).tolist())]
        index_cols: List[str] = col_names[:5]
        df = DataFrame(np.random.default_rng(2).integers(5, size=(NUM_ROWS, NUM_COLS)), dtype=np.int64, columns=col_names)
        df = df.set_index(index_cols).sort_index()
        grp = df.groupby(level=index_cols[:4])
        df['new_col'] = np.nan
        for name, df2 in grp:
            new_vals = np.arange(df2.shape[0])
            df.loc[name, 'new_col'] = new_vals

    def test_series_setitem(self, multiindex_year_month_day_dataframe_random_data: DataFrame) -> None:
        ymd = multiindex_year_month_day_dataframe_random_data
        s = ymd['A']
        s[2000, 3] = np.nan
        assert isna(s.values[42:65]).all()
        assert notna(s.values[:42]).all()
        assert notna(s.values[65:]).all()
        s[2000, 3, 10] = np.nan
        assert isna(s.iloc[49])
        with pytest.raises(KeyError, match='49'):
            s[49]

    def test_frame_getitem_setitem_boolean(self, multiindex_dataframe_random_data: DataFrame) -> None:
        frame = multiindex_dataframe_random_data
        df = frame.T.copy()
        values = df.values.copy()
        result = df[df > 0]
        expected = df.where(df > 0)
        tm.assert_frame_equal(result, expected)
        df[df > 0] = 5
        values[values > 0] = 5
        tm.assert_almost_equal(df.values, values)
        df[df == 5] = 0
        values[values == 5] = 0
        tm.assert_almost_equal(df.values, values)
        df[df[:-1] < 0] = 2
        np.putmask(values[:-1], values[:-1] < 0, 2)
        tm.assert_almost_equal(df.values, values)
        with pytest.raises(TypeError, match='boolean values only'):
            df[df * 0] = 2

    def test_frame_getitem_setitem_multislice(self) -> None:
        levels: List[List[str]] = [['t1', 't2'], ['a', 'b', 'c']]
        codes: List[List[int]] = [[0, 0, 0, 1, 1], [0, 1, 2, 0, 1]]
        midx = MultiIndex(codes=codes, levels=levels, names=[None, 'id'])
        df = DataFrame({'value': [1, 2, 3, 7, 8]}, index=midx)
        result = df.loc[:, 'value']
        tm.assert_series_equal(df['value'], result)
        result = df.loc[df.index[1:3], 'value']
        tm.assert_series_equal(df['value'][1:3], result)
        result = df.loc[:, :]
        tm.assert_frame_equal(df, result)
        result = df
        df.loc[:, 'value'] = 10
        result['value'] = 10
        tm.assert_frame_equal(df, result)
        df.loc[:, :] = 10
        tm.assert_frame_equal(df, result)

    def test_frame_setitem_multi_column(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=[['a', 'a', 'b', 'b'], [0, 1, 0, 1]])
        cp = df.copy()
        cp['a'] = cp['b']
        tm.assert_frame_equal(cp['a'], cp['b'])
        cp = df.copy()
        cp['a'] = cp['b'].values
        tm.assert_frame_equal(cp['a'], cp['b'])

    def test_frame_setitem_multi_column2(self) -> None:
        columns = MultiIndex.from_tuples([('A', '1'), ('A', '2'), ('B', '1')])
        df = DataFrame(index=[1, 3, 5], columns=columns)
        df['A'] = 0.0
        assert (df['A'].values == 0).all()
        df['B', '1'] = [1, 2, 3]
        df['A'] = df['B', '1']
        sliced_a1 = df['A', '1']
        sliced_a2 = df['A', '2']
        sliced_b1 = df['B', '1']
        tm.assert_series_equal(sliced_a1, sliced_b1, check_names=False)
        tm.assert_series_equal(sliced_a2, sliced_b1, check_names=False)
        assert sliced_a1.name == ('A', '1')
        assert sliced_a2.name == ('A', '2')
        assert sliced_b1.name == ('B', '1')

    def test_loc_getitem_tuple_plus_columns(self, multiindex_year_month_day_dataframe_random_data: DataFrame) -> None:
        ymd = multiindex_year_month_day_dataframe_random_data
        df = ymd[:5]
        result = df.loc[(2000, 1, 6), ['A', 'B', 'C']]
        expected = df.loc[2000, 1, 6][['A', 'B', 'C']]
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
    def test_loc_getitem_setitem_slice_integers(self, frame_or_series: Any) -> None:
        index = MultiIndex(levels=[[0, 1, 2], [0, 2]], codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
        obj = DataFrame(np.random.default_rng(2).standard_normal((len(index), 4)), index=index, columns=['a', 'b', 'c', 'd'])
        obj = tm.get_obj(obj, frame_or_series)
        res = obj.loc[1:2]
        exp = obj.reindex(obj.index[2:])
        tm.assert_equal(res, exp)
        obj.loc[1:2] = 7
        assert (obj.loc[1:2] == 7).values.all()

    def test_setitem_change_dtype(self, multiindex_dataframe_random_data: DataFrame) -> None:
        frame = multiindex_dataframe_random_data
        dft = frame.T
        s = dft['foo', 'two']
        dft['foo', 'two'] = s > s.median()
        tm.assert_series_equal(dft['foo', 'two'], s > s.median())
        reindexed = dft.reindex(columns=[('foo', 'two')])
        tm.assert_series_equal(reindexed['foo', 'two'], s > s.median())

    def test_set_column_scalar_with_loc(self, multiindex_dataframe_random_data: DataFrame) -> None:
        frame = multiindex_dataframe_random_data
        subset = frame.index[[1, 4, 5]]
        frame.loc[subset] = 99
        assert (frame.loc[subset].values == 99).all()
        frame_original = frame.copy()
        col = frame['B']
        col[subset] = 97
        tm.assert_frame_equal(frame, frame_original)

    def test_nonunique_assignment_1750(self) -> None:
        df = DataFrame([[1, 1, 'x', 'X'], [1, 1, 'y', 'Y'], [1, 2, 'z', 'Z']], columns=list('ABCD'))
        df = df.set_index(['A', 'B'])
        mi = MultiIndex.from_tuples([(1, 1)])
        df.loc[mi, 'C'] = '_'
        assert (df.xs((1, 1))['C'] == '_').all()

    def test_astype_assignment_with_dups(self) -> None:
        cols = MultiIndex.from_tuples([('A', '1'), ('B', '1'), ('A', '2')])
        df = DataFrame(np.arange(3).reshape((1, 3)), columns=cols, dtype=object)
        index = df.index.copy()
        df['A'] = df['A'].astype(np.float64)
        tm.assert_index_equal(df.index, index)

    def test_setitem_nonmonotonic(self) -> None:
        index = MultiIndex.from_tuples([('a', 'c'), ('b', 'x'), ('a', 'd')], names=['l1', 'l2'])
        df = DataFrame(data=[0, 1, 2], index=index, columns=['e'])
        df.loc['a', 'e'] = np.arange(99, 101, dtype='int64')
        expected = DataFrame({'e': [99, 1, 100]}, index=index)
        tm.assert_frame_equal(df, expected)

class TestSetitemWithExpansionMultiIndex:

    def test_setitem_new_column_mixed_depth(self) -> None:
        arrays: List[List[str]] = [['a', 'top', 'top', 'routine1', 'routine1', 'routine2'], ['', 'OD', 'OD', 'result1', 'result2', 'result1'], ['', 'wx', 'wy', '', '', '']]
        tuples = sorted(zip(*arrays))
        index = MultiIndex.from_tuples(tuples)
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)
        result = df.copy()
        expected = df.copy()
        result['b'] = [1, 2, 3, 4]
        expected['b', '', ''] = [1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

    def test_setitem_new_column_all_na(self) -> None:
        mix = MultiIndex.from_tuples([('1a', '2a'), ('1a', '2b'), ('1a', '2c')])
        df = DataFrame([[1, 2], [3, 4], [5, 6]], index=mix)
        s = Series({(1, 1): 1, (1, 2): 2})
        df['new'] = s
        assert df['new'].isna().all()

    def test_setitem_enlargement_keep_index_names(self) -> None:
        mi = MultiIndex.from_tuples([(1, 2, 3)], names=['i1', 'i2', 'i3'])
        df = DataFrame(data=[[10, 20, 30]], index=mi, columns=['A', 'B', 'C'])
        df.loc[0, 0, 0] = df.loc[1, 2, 3]
        mi_expected = MultiIndex.from_tuples([(1, 2, 3), (0, 0, 0)], names=['i1', 'i2', 'i3'])
        expected = DataFrame(data=[[10, 20, 30], [10, 20, 30]], index=mi_expected, columns=['A', 'B', 'C'])
        tm.assert_frame_equal(df, expected)

def test_frame_setitem_view_direct(multiindex_dataframe_random_data: DataFrame) -> None:
    df = multiindex_dataframe_random_data.T
    with pytest.raises(ValueError, match='read-only'):
        df['foo'].values[:] = 0
    assert (df['foo'].values != 0).all()

def test_frame_setitem_copy_raises(multiindex_dataframe_random_data: DataFrame) -> None:
    df = multiindex_dataframe_random_data.T
    with tm.raises_chained_assignment_error():
        df['foo']['one'] = 2

def test_frame_setitem_copy_no_write(multiindex_dataframe_random_data: DataFrame) -> None:
    frame = multiindex_dataframe_random_data.T
    expected = frame
    df = frame.copy()
    with tm.raises_chained_assignment_error():
        df['foo']['one'] = 2
    tm.assert_frame_equal(df, expected)

def test_frame_setitem_partial_multiindex() -> None:
    df = DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5], 'c': 6, 'd': 7}).set_index(['a', 'b', 'c'])
    ser = Series(8, index=df.index.droplevel('c'))
    result = df.copy()
    result['d'] = ser
    expected = df.copy()
    expected['d'] = 8
    tm.assert_frame_equal(result, expected)

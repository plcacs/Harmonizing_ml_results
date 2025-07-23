"""test fancy indexing & misc"""
import array
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import re
import weakref
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_float_dtype, is_integer_dtype, is_object_dtype
import pandas as pd
from pandas import DataFrame, Index, NaT, Series, date_range, offsets, timedelta_range
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj

class TestFancy:
    """pure get/set item & fancy indexing"""

    def test_setitem_ndarray_1d(self) -> None:
        df = DataFrame(index=Index(np.arange(1, 11), dtype=np.int64))
        df['foo'] = np.zeros(10, dtype=np.float64)
        df['bar'] = np.zeros(10, dtype=complex)
        msg = 'Must have equal len keys and value when setting with an iterable'
        with pytest.raises(ValueError, match=msg):
            df.loc[df.index[2:5], 'bar'] = np.array([2.33j, 1.23 + 0.1j, 2.2, 1.0])
        df.loc[df.index[2:6], 'bar'] = np.array([2.33j, 1.23 + 0.1j, 2.2, 1.0])
        result = df.loc[df.index[2:6], 'bar']
        expected = Series([2.33j, 1.23 + 0.1j, 2.2, 1.0], index=[3, 4, 5, 6], name='bar')
        tm.assert_series_equal(result, expected)

    def test_setitem_ndarray_1d_2(self) -> None:
        df = DataFrame(index=Index(np.arange(1, 11)))
        df['foo'] = np.zeros(10, dtype=np.float64)
        df['bar'] = np.zeros(10, dtype=complex)
        msg = 'Must have equal len keys and value when setting with an iterable'
        with pytest.raises(ValueError, match=msg):
            df[2:5] = np.arange(1, 4) * 1j

    @pytest.mark.filterwarnings('ignore:Series.__getitem__ treating keys as positions is deprecated:FutureWarning')
    def test_getitem_ndarray_3d(self, index: Index, frame_or_series: Callable, indexer_sli: Callable) -> None:
        obj = gen_obj(frame_or_series, index)
        idxr = indexer_sli(obj)
        nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))
        msgs: List[str] = []
        if frame_or_series is Series and indexer_sli in [tm.setitem, tm.iloc]:
            msgs.append('Wrong number of dimensions. values.ndim > ndim \\[3 > 1\\]')
        if frame_or_series is Series or indexer_sli is tm.iloc:
            msgs.append('Buffer has wrong number of dimensions \\(expected 1, got 3\\)')
        if indexer_sli is tm.loc or (frame_or_series is Series and indexer_sli is tm.setitem):
            msgs.append('Cannot index with multidimensional key')
        if frame_or_series is DataFrame and indexer_sli is tm.setitem:
            msgs.append('Index data must be 1-dimensional')
        if isinstance(index, pd.IntervalIndex) and indexer_sli is tm.iloc:
            msgs.append('Index data must be 1-dimensional')
        if isinstance(index, (pd.TimedeltaIndex, pd.DatetimeIndex, pd.PeriodIndex)):
            msgs.append('Data must be 1-dimensional')
        if len(index) == 0 or isinstance(index, pd.MultiIndex):
            msgs.append('positional indexers are out-of-bounds')
        if type(index) is Index and (not isinstance(index._values, np.ndarray)):
            msgs.append('values must be a 1D array')
            msgs.append('only handle 1-dimensional arrays')
        msg = '|'.join(msgs)
        potential_errors = (IndexError, ValueError, NotImplementedError)
        with pytest.raises(potential_errors, match=msg):
            idxr[nd3]

    @pytest.mark.filterwarnings('ignore:Series.__setitem__ treating keys as positions is deprecated:FutureWarning')
    def test_setitem_ndarray_3d(self, index: Index, frame_or_series: Callable, indexer_sli: Callable) -> None:
        obj = gen_obj(frame_or_series, index)
        idxr = indexer_sli(obj)
        nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))
        if indexer_sli is tm.iloc:
            err = ValueError
            msg = f'Cannot set values with ndim > {obj.ndim}'
        else:
            err = ValueError
            msg = '|'.join(['Buffer has wrong number of dimensions \\(expected 1, got 3\\)', 'Cannot set values with ndim > 1', 'Index data must be 1-dimensional', 'Data must be 1-dimensional', 'Array conditional must be same shape as self'])
        with pytest.raises(err, match=msg):
            idxr[nd3] = 0

    def test_getitem_ndarray_0d(self) -> None:
        key = np.array(0)
        df = DataFrame([[1, 2], [3, 4]])
        result = df[key]
        expected = Series([1, 3], name=0)
        tm.assert_series_equal(result, expected)
        ser = Series([1, 2])
        result = ser[key]
        assert result == 1

    def test_inf_upcast(self) -> None:
        df = DataFrame(columns=[0])
        df.loc[1] = 1
        df.loc[2] = 2
        df.loc[np.inf] = 3
        assert df.loc[np.inf, 0] == 3
        result = df.index
        expected = Index([1, 2, np.inf], dtype=np.float64)
        tm.assert_index_equal(result, expected)

    def test_setitem_dtype_upcast(self) -> None:
        df = DataFrame([{'a': 1}, {'a': 3, 'b': 2}])
        df['c'] = np.nan
        assert df['c'].dtype == np.float64
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[0, 'c'] = 'foo'

    @pytest.mark.parametrize('val', [3.14, 'wxyz'])
    def test_setitem_dtype_upcast2(self, val: Union[float, str]) -> None:
        df = DataFrame(np.arange(6, dtype='int64').reshape(2, 3), index=list('ab'), columns=['foo', 'bar', 'baz'])
        left = df.copy()
        with pytest.raises(TypeError, match='Invalid value'):
            left.loc['a', 'bar'] = val

    def test_setitem_dtype_upcast3(self) -> None:
        left = DataFrame(np.arange(6, dtype='int64').reshape(2, 3) / 10.0, index=list('ab'), columns=['foo', 'bar', 'baz'])
        with pytest.raises(TypeError, match='Invalid value'):
            left.loc['a', 'bar'] = 'wxyz'

    def test_dups_fancy_indexing(self) -> None:
        df = DataFrame(np.eye(3), columns=['a', 'a', 'b'])
        result = df[['b', 'a']].columns
        expected = Index(['b', 'a', 'a'])
        tm.assert_index_equal(result, expected)

    def test_dups_fancy_indexing_across_dtypes(self) -> None:
        df = DataFrame([[1, 2, 1.0, 2.0, 3.0, 'foo', 'bar']], columns=list('aaaaaaa'))
        result = DataFrame([[1, 2, 1.0, 2.0, 3.0, 'foo', 'bar']])
        result.columns = list('aaaaaaa')
        df.iloc[:, 4]
        result.iloc[:, 4]
        tm.assert_frame_equal(df, result)

    def test_dups_fancy_indexing_not_in_order(self) -> None:
        df = DataFrame({'test': [5, 7, 9, 11], 'test1': [4.0, 5, 6, 7], 'other': list('abcd')}, index=['A', 'A', 'B', 'C'])
        rows = ['C', 'B']
        expected = DataFrame({'test': [11, 9], 'test1': [7.0, 6], 'other': ['d', 'c']}, index=rows)
        result = df.loc[rows]
        tm.assert_frame_equal(result, expected)
        result = df.loc[Index(rows)]
        tm.assert_frame_equal(result, expected)
        rows = ['C', 'B', 'E']
        with pytest.raises(KeyError, match='not in index'):
            df.loc[rows]
        rows = ['F', 'G', 'H', 'C', 'B', 'E']
        with pytest.raises(KeyError, match='not in index'):
            df.loc[rows]

    def test_dups_fancy_indexing_only_missing_label(self, using_infer_string: bool) -> None:
        dfnu = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), index=list('AABCD'))
        if using_infer_string:
            with pytest.raises(KeyError, match=re.escape('"None of [Index([\'E\'], dtype=\'str\')] are in the [index]"')):
                dfnu.loc[['E']]
        else:
            with pytest.raises(KeyError, match=re.escape('"None of [Index([\'E\'], dtype=\'object\')] are in the [index]"')):
                dfnu.loc[['E']]

    @pytest.mark.parametrize('vals', [[0, 1, 2], list('abc')])
    def test_dups_fancy_indexing_missing_label(self, vals: List) -> None:
        df = DataFrame({'A': vals})
        with pytest.raises(KeyError, match='not in index'):
            df.loc[[0, 8, 0]]

    def test_dups_fancy_indexing_non_unique(self) -> None:
        df = DataFrame({'test': [5, 7, 9, 11]}, index=['A', 'A', 'B', 'C'])
        with pytest.raises(KeyError, match='not in index'):
            df.loc[['A', 'A', 'E']]

    def test_dups_fancy_indexing2(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)), columns=['A', 'B', 'B', 'B', 'A'])
        with pytest.raises(KeyError, match='not in index'):
            df.loc[:, ['A', 'B', 'C']]

    def test_dups_fancy_indexing3(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((9, 2)), index=[1, 1, 1, 2, 2, 2, 3, 3, 3], columns=['a', 'b'])
        expected = df.iloc[0:6]
        result = df.loc[[1, 2]]
        tm.assert_frame_equal(result, expected)
        expected = df
        result = df.loc[:, ['a', 'b']]
        tm.assert_frame_equal(result, expected)
        expected = df.iloc[0:6, :]
        result = df.loc[[1, 2], ['a', 'b']]
        tm.assert_frame_equal(result, expected)

    def test_duplicate_int_indexing(self, indexer_sl: Callable) -> None:
        ser = Series(range(3), index=[1, 1, 3])
        expected = Series(range(2), index=[1, 1])
        result = indexer_sl(ser)[[1]]
        tm.assert_series_equal(result, expected)

    def test_indexing_mixed_frame_bug(self) -> None:
        df = DataFrame({'a': {1: 'aaa', 2: 'bbb', 3: 'ccc'}, 'b': {1: 111, 2: 222, 3: 333}})
        df['test'] = df['a'].apply(lambda x: '_' if x == 'aaa' else x)
        idx = df['test'] == '_'
        temp = df.loc[idx, 'a'].apply(lambda x: '-----' if x == 'aaa' else x)
        df.loc[idx, 'test'] = temp
        assert df.iloc[0, 2] == '-----'

    def test_multitype_list_index_access(self) -> None:
        df = DataFrame(np.random.default_rng(2).random((10, 5)), columns=['a'] + [20, 21, 22, 23])
        with pytest.raises(KeyError, match=re.escape("'[26, -8] not in index'")):
            df[[22, 26, -8]]
        assert df[21].shape[0] == df.shape[0]

    def test_set_index_nan(self) -> None:
        df = DataFrame({'PRuid': {17: 'nonQC', 18: 'nonQC', 19: 'nonQC', 20: '10', 21: '11', 22: '12', 23: '13', 24: '24', 25: '35', 26: '46', 27: '47', 28: '48', 29: '59', 30: '10'}, 'QC': {17: 0.0, 18: 0.0, 19: 0.0, 20: np.nan, 21: np.nan, 22: np.nan, 23: np.nan, 24: 1.0, 25: np.nan, 26: np.nan, 27: np.nan, 28: np.nan, 29: np.nan, 30: np.nan}, 'data': {17: 7.95449, 18: 8.014261, 19: 7.859152000000001, 20: 0.8614035, 21: 0.8785311, 22: 0.8427041999999999, 23: 0.785877, 24: 0.7306246, 25: 0.8166856, 26: 0.8192708000000001, 27: 0.8070501, 28: 0.8144024000000001, 29: 0.8014085, 30: 0.8130774000000001}, 'year': {17: 2006, 18: 2007, 19: 2008, 20: 1985, 21: 1985, 22: 1985, 23: 1985, 24: 1985, 25: 1985, 26: 1985, 27: 1985, 28: 1985, 29: 1985, 30: 1986}}).reset_index()
        result = df.set_index(['year', 'PRuid', 'QC']).reset_index().reindex(columns=df.columns)
        tm.assert_frame_equal(result, df)

    def test_multi_assign(self) -> None:
        df = DataFrame({'FC': ['a', 'b', 'a', 'b', 'a', 'b'], 'PF': [0, 0, 0, 0, 1, 1], 'col1': list(range(6)), 'col2': list(range(6, 12))}).astype({'col2': 'float64'})
        df.iloc[1, 0] = np.nan
        df2 = df.copy()
        mask = ~df2.FC.isna()
        cols = ['col1', 'col2']
        dft = df2 * 2
        dft.iloc[3, 3] = np.nan
        expected = DataFrame({'FC': ['a', np.nan, 'a', 'b', 'a', 'b'], 'PF': [0, 0, 0, 0, 1, 1], 'col1': Series([0, 1, 4, 6, 8, 10]), 'col2': [12, 7, 16, np.nan, 20, 22]})
        df2.loc[mask, cols] = dft.loc[mask, cols]
        tm.assert_frame_equal(df2, expected)
        expected = DataFrame({'FC': ['a', np.nan, 'a', 'b', 'a', 'b'], 'PF': [0, 0, 0, 0, 1, 1], 'col1': [0, 1, 4, 6, 8, 10], 'col2': [12, 7, 16, np.nan, 20, 22]})
        df2 = df.copy()
        df2.loc[mask, cols] = dft.loc[mask, cols].values
        tm.assert_frame_equal(df2, expected)

    def test_multi_assign_broadcasting_rhs(self) -> None:
        df = DataFrame({'A': [1, 2, 0, 0, 0], 'B': [0, 0, 0, 10, 11], 'C': [0, 0, 0, 10, 11], 'D': [3, 4, 5, 6, 7]})
        expected = df.copy()
        mask = expected['A'] == 0
        for col in ['A', 'B']:
            expected.loc[mask, col] = df['D']
        df.loc[df['A'] == 0, ['A', 'B']] = df['D'].copy()
        tm.assert_frame_equal(df, expected)

    def test_setitem_list(self) -> None:
        df = DataFrame(index=[0, 1], columns=[0])
        df.iloc[1, 0] = [1, 2, 3]
        df.iloc[1, 0] = [1, 2]
        result = DataFrame(index=[0, 1], columns=[0])
        result.iloc[1, 0] = [1, 2]
        tm.assert_frame_equal(result, df)

    def test_string_slice(self) -> None:
        df = DataFrame([1], Index([pd.Timestamp('2011-01-01')], dtype=object))
        assert df.index._is_all_dates
        with pytest.raises(KeyError, match="'2011'"):
            df['2011']
        with pytest.raises(KeyError, match="'2011'"):
            df.loc['2011', 0]

    def test_string_slice_empty(self) -> None:
        df = DataFrame()
        assert not df.index._is_all_dates
        with pytest.raises(KeyError, match="'2011'"):
            df['2011']
        with pytest.raises(KeyError, match='^0$'):
            df.loc['2011', 0]

    def test_astype_assignment(self, using_infer_string: bool) -> None:
        df_orig = DataFrame([['1', '2', '3', '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
        df_orig[list('ABCDG')] = df_orig[list('ABCDG')].astype(object)
        df = df_orig.copy()
        df.iloc[:, 0:2] = df.iloc[:, 0:2].astype(np.int64)
        expected = DataFrame([[1, 2, '3', '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
        expected[list('CDG')] = expected[list('CDG')].astype(object)
        expected['A'] = expected['A'].astype(object)
        expected['B'] = expected['B'].astype(object)
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        df.loc[:, 'A'] = df.loc[:, 'A'].astype(np.int64)
        expected = DataFrame([[1, '2', '3', '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
        expected[list('ABCDG')] = expected[list('ABCDG')].astype(object)
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        df.loc[:, ['B', 'C']] = df.loc[:, ['B', 'C']].astype(np.int64)
        expected = DataFrame([['1', 2, 3, '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
        expected[list('ABCDG')] = expected[list('ABCDG')].astype(object)
        tm.assert_frame_equal(df, expected)

    def test_astype_assignment_full_replacements(self) -> None:
        df = DataFrame({'A': [1.0, 2.0, 3.0, 4.0]})
        df.iloc[:, 0] = df['A'].astype(np.int64)
        expected = DataFrame({'A': [1.0, 2.0, 3.0, 4.0]})
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'A': [1.0, 2.0, 3.0, 4.0]})
        df.loc[:, 'A'] = df['A'].astype(np.int64)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('indexer', [tm.getitem, tm.loc])
    def test_index_type_coercion(self, indexer: Callable) -> None:
        for s in [Series(range(5)), Series(range(5), index=range(1, 6))]:
            assert is_integer_dtype(s.index)
            s2 = s.copy()
            indexer(s2)[0.1] = 0
            assert is_float_dtype(s2.index)
            assert indexer(s2)[0.1] == 0
            s2 = s.copy()
            indexer(s2)[0.0] = 0
            exp = s.index
            if 0 not in s:
                exp = Index(s.index.tolist() + [0])
            tm.assert_index_equal(s2.index, exp)
            s2 = s.copy()
            indexer(s2)['0'] = 0
            assert is_object_dtype(s2.index)
        for s in [Series(range(5), index=np.arange(5.0))]:
            assert is_float_dtype(s.index)
            s2 = s.copy()
            indexer(s2)[0.1] = 0
            assert is_float_dtype(s2.index)
            assert indexer(s2)[0.1] == 0
            s2 = s.copy()
            indexer(s2)[0.0] = 0
            tm.assert_index_equal(s2.index, s.index)
            s2 = s.copy()
            indexer(s2)['0'] = 0
            assert is_object_dtype(s2.index)

class TestMisc:

    def test_float_index_to_mixed(self) -> None:
        df = DataFrame({0.0: np.random.default_rng(2).random(10), 1.0: np.random.default_rng(2).random(10)})
        df['a'] = 10
        expected = DataFrame({0.0: df[0.0], 1.0: df[1.0], 'a': [10] * 10})
        tm.assert_frame_equal(expected, df)

    def test_float_index_non_scalar_assignment(self) -> None:
        df = DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]}, index=[1.0, 2.0, 3.0])
        df.loc[df.index[:2]] = 1
        expected = DataFrame({'a': [1, 1, 3], 'b': [1, 1, 5]}, index=df.index)
        tm.assert_frame_equal(expected, df)

    def test_loc_setitem_fullindex_views(self) -> None:
        df = DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]}, index=[1.0, 2.0, 3.0])
        df2 = df.copy()
        df.loc[df.index] = df.loc[df.index]
        tm.assert_frame_equal(df, df2)

    def test_rhs_alignment(self) -> None:

        def run_tests(df: DataFrame, rhs: DataFrame, right_loc: DataFrame, right_iloc: DataFrame) -> None:
            lbl_one, idx_one, slice_one = (list('bcd'), [1, 2, 3], slice(1, 4))
            lbl_two, idx_two, slice_two = (['joe', 'jolie'], [1, 2], slice(1, 3))
            left = df.copy()
            left.loc[lbl_one, lbl_two] = rhs
            tm.assert_frame_equal(left, right_loc)
            left = df.copy()
            left.iloc[idx_one, idx_two] = rhs
            tm.assert_frame_equal(left, right_iloc)
            left = df.copy()
            left.iloc[slice_one, slice_two] = rhs
            tm.assert_frame_equal(left, right_iloc)
        xs = np.arange(20).reshape(5, 4)
        cols = ['jim', 'joe', 'jolie', 'joline']
        df = DataFrame(xs, columns=cols, index=list('abcde'), dtype='int64')
        rhs = -2 * df.iloc[3:0:-1, 2:0:-1]
        right_iloc = df.copy()
        right_iloc['joe'] = [1, 14, 10, 6, 17]
        right_iloc['jolie'] = [2, 13, 9, 5, 18]
        right_iloc.iloc[1:4, 1:3] *= -2
        right_loc = df.copy()
        right_loc.iloc[1:4, 1:3] *= -2
        run_tests(df, rhs, right_loc, right_iloc)
        for frame in [df, rhs, right_loc, right_iloc]:
            frame['joe'] = frame['joe'].astype('float64')
            frame['jolie'] = frame['jolie'].map(lambda x: f'@{x}')
        right_iloc['joe'] = [1.0, '@-28', '@-20', '@-12', 17.0]
        right_iloc['jolie'] = ['@2', -26.0, -18.0, -10.0, '@18']
        with pytest.raises(TypeError, match='Invalid value'):
            run_tests(df, rhs, right_loc, right_iloc)

    @pytest.mark.parametrize('idx', [_mklbl('A', 20), np.arange(20) + 100, np.linspace(100, 150, 20)])
    def test_str_label_slicing_with_negative_step(self, idx: Union[List[str], np.ndarray]) -> None:
        SLC = pd.IndexSlice
        idx = Index(idx)
        ser = Series(np.arange(20), index=idx)
        tm.assert_indexing_slices_equivalent(ser, SLC[idx[9]::-1], SLC[9::-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[:idx[9]:-1], SLC[:8:-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[idx[13]:idx[9]:-1], SLC[13:8:-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[idx[9]:idx[13]:-1], SLC[:0])

    def test_slice_with_zero_step_raises(self, index: Index, indexer_sl: Callable, frame_or_series: Callable) -> None:
        obj = frame_or_series(np.arange(len(index)), index=index)
        with pytest.raises(ValueError, match='slice step cannot be zero'):
            indexer_sl(obj)[::0]

    def test_loc_setitem_indexing_assignment_dict_already_exists(self) -> None:
        index = Index([-5, 0, 5], name='z')
        df = DataFrame({'x': [1, 2, 6], 'y': [2, 2, 8]}, index=index)
        expected = df.copy()
        rhs = {'x': 9, 'y': 99}
        df.loc[5] = rhs
        expected.loc[5] = [9, 99]
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'x': [1, 2, 6], 'y': [2.0, 2.0, 8.0]}, index=index)
        df.loc[5] = rhs
        expected = DataFrame({'x': [1, 2, 9], 'y': [2.0, 2.0, 99.0]}, index=index)
        tm.assert_frame_equal(df, expected)

    def test_iloc_getitem_indexing_dtypes_on_empty(self) -> None:
        df = DataFrame({'a': [1, 2, 3], 'b': ['b', 'b2', 'b3']})
        df2 = df.iloc[[], :]
        assert df2.loc[:, 'a'].dtype == np.int64
        tm.assert_series_equal(df2.loc[:, 'a'], df2.iloc[:, 0])

    @pytest.mark.parametrize('size', [5, 999999, 1000000])
    def test_loc_range_in_series_indexing(self, size: int) -> None:
        s = Series(index=range(size), dtype=np.float64)
        s.loc[range(1)] = 42
        tm.assert_series_equal(s.loc[range(1)], Series(42.0, index=range(1)))
        s.loc[range(2)] = 43
        tm.assert_series_equal(s.loc[range(2)], Series(43.0, index=range(2)))

    def test_partial_boolean_frame_indexing(self) -> None:
        df = DataFrame(np.arange(9.0).reshape(3, 3), index=list('abc'), columns=list('ABC'))
        index_df = DataFrame(1, index=list('ab'), columns=list('AB'))
        result = df[index_df.notnull()]
        expected = DataFrame(np.array([[0.0, 1.0, np.nan], [3.0, 4.0, np.nan], [np.nan] * 3]), index=list('abc'), columns=list('ABC'))
        tm.assert_frame_equal(result, expected)

    def test_no_reference_cycle(self) -> None:
        df = DataFrame({'a': [0, 1], 'b': [2, 3]})
        for name in ('loc', 'iloc', 'at', 'iat'):
            getattr(df, name)
        wr = weakref.ref(df)
        del df
        assert wr() is None

    def test_label_indexing_on_nan(self, nulls_fixture: Any) -> None:
        df = Series([1, '{1,2}', 1, nulls_fixture])
        vc = df.value_counts(dropna=False)
        result1 = vc.loc[nulls_fixture]
        result2 = vc[nulls_fixture]
        expected = 1
        assert result1 == expected
        assert result2 == expected

class TestDataframeNoneCoercion:
    EXPECTED_SINGLE_ROW_RESULTS: List[Tuple[List, List, Optional[Type[Warning]]]] = [
        ([1, 2, 3], [np.nan, 2, 3], FutureWarning),
        ([1.0, 2.0, 3.0], [np.nan, 2.0, 3.0], None),
        ([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], [NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)], None),
        (['foo', 'bar', 'baz'], [None, 'bar', 'baz'], None)
    ]

    @pytest.mark.parametrize('expected', EXPECTED_SINGLE_ROW_RESULTS)
    def test_coercion_with_loc(self, expected: Tuple[List, List, Optional[Type[Warning]]]) -> None:
        start_data, expected_result, warn = expected
        start_dataframe = DataFrame({'foo': start_data})
        start_dataframe.loc[0, ['foo']] = None
        expected_dataframe = DataFrame({'foo': expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    @pytest.mark.parametrize('expected', EXPECTED_SINGLE_ROW_RESULTS)
    def test_coercion_with_setitem_and_dataframe(self, expected: Tuple[List, List, Optional[Type[Warning]]]) -> None:
        start_data, expected_result, warn = expected
        start_dataframe = DataFrame({'foo': start_data})
        start_dataframe[start_dataframe['foo'] == start_dataframe['foo'][0]] = None
        expected_dataframe = DataFrame({'foo': expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    @pytest.mark.parametrize('expected', EXPECTED_SINGLE_ROW_RESULTS)
    def test_none_coercion_loc_and_dataframe(self, expected: Tuple[List, List, Optional[Type[Warning]]]) -> None:
        start_data, expected_result, warn = expected
        start_dataframe = DataFrame({'foo': start_data})
        start_dataframe.loc[start_dataframe['foo'] == start_dataframe['foo'][0]] = None
        expected_dataframe = DataFrame({'foo': expected_result})
        tm.assert_frame_equal(start_dataframe, expected_dataframe)

    def test_none_coercion_mixed_dtypes(self) -> None:
        start_dataframe = DataFrame({'a': [1, 2, 3], 'b': [1.0, 2.0, 3.0], 'c': [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], 'd': ['a', 'b', 'c']})
        start_dataframe.iloc[0] = None
        exp = DataFrame({'a': [np.nan, 2, 3], 'b': [np.nan, 2.0, 3.0], 'c': [NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)], 'd': [None, 'b', 'c']})
        tm.assert_frame_equal(start_dataframe, exp)

class TestDatetimelikeCoercion:

    def test_setitem_dt64_string_scalar(self, tz_naive_fixture: Optional[str], indexer_sli: Callable) -> None:
        tz = tz_naive_fixture
        dti = date_range('2016-01-01', periods=3, tz=tz)
        ser = Series(dti.copy(deep=True))
        values = ser._values
        newval = '2018-01-01'
        values._validate_setitem_value(newval)
        indexer_sli(ser)[0] = newval
        if tz is None:
            assert ser.dtype == dti.dtype
            assert ser._values._ndarray is values._ndarray
        else:
            assert ser._values is values

    @pytest.mark.parametrize('box', [list, np.array, pd.array, pd.Categorical, Index])
    @pytest.mark.parametrize('key', [[0, 1], slice(0, 2), np.array([True, True, False])])
    def test_setitem_dt64_string_values(self, tz_naive_fixture: Optional[str], indexer_sli: Callable, key: Union[List[int], slice, np.ndarray], box: Callable) -> None:
        tz = tz_naive_fixture
        if isinstance(key, slice) and indexer_sli is tm.loc:
            key = slice(0, 1)
        dti = date_range('2016-01-01', periods=3, tz=tz)
        ser = Series(dti.copy(deep=True))
        values = ser._values
        newvals = box(['2019-01-01', '2010-01-02'])
        values._validate_setitem_value(newvals)
        indexer_sli(ser)[key] = newvals
        if tz is None:
            assert ser.dtype == dti.dtype
            assert ser._values._ndarray is values._ndarray
        else:
            assert ser._values is values

    @pytest.mark.parametrize('scalar', ['3 Days', offsets.Hour(4)])
    def test_setitem_td64_scalar(self, indexer_sli: Callable, scalar: Union[str, offsets.Hour]) -> None:
        tdi = timedelta_range('1 Day', periods=3)
        ser = Series(tdi.copy(deep=True))
        values = ser._values
        values._validate_setitem_value(scalar)
        indexer_sli(ser)[0] = scalar
        assert ser._values._ndarray is values._ndarray

    @pytest.mark.parametrize('box', [list, np.array, pd.array, pd.Categorical, Index])
    @pytest.mark.parametrize('key', [[0, 1], slice(0, 2), np.array([True, True, False])])
    def test_setitem_td64_string_values(self, indexer_sli: Callable, key: Union[List[int], slice, np.ndarray], box: Callable) -> None:
        if isinstance(key, slice) and indexer_sli is tm.loc:
            key = slice(0, 1)
        tdi = timedelta_range('1 Day', periods=3)
        ser = Series(tdi.copy(deep=True))
        values = ser._values
        newvals = box(['10 Days', '44 hours'])
        values._validate_setitem_value(newvals)
        indexer_sli(ser)[key] = newvals
        assert ser._values._ndarray is values._ndarray

def test_extension_array_cross_section() -> None:
    df = DataFrame({'A': pd.array([1, 2], dtype='Int64'), 'B': pd.array([3, 4], dtype='Int64')}, index=['a', 'b'])
    expected = Series(pd.array([1, 3], dtype='Int64'), index=['A', 'B'], name='a')
    result = df.loc['a']
    tm.assert_series_equal(result, expected)
    result = df.iloc[0]
    tm.assert_series_equal(result, expected)

def test_extension_array_cross_section_converts() -> None:
    df = DataFrame({'A': pd.array([1, 2], dtype='Int64'), 'B': np.array([1, 2], dtype='int64')}, index=['a', 'b'])
    result = df.loc['a']
    expected = Series([1, 1], dtype='Int64', index=['A', 'B'], name='a')
    tm.assert_series_equal(result, expected)
    result = df.iloc[0]
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': pd.array([1, 2], dtype='Int64'), 'B': np.array(['a', 'b'])}, index=['a', 'b'])
    result = df.loc['a']
    expected = Series([1, 'a'], dtype=object, index=['A', 'B'], name='a')
    tm.assert_series_equal(result, expected)
    result = df.iloc[0]
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('ser, keys', [(Series([10]), (0, 0)), (Series([1, 2, 3], index=list('abc')), (0, 1))])
def test_ser_tup_indexer_exceeds_dimensions(ser: Series, keys: Tuple[int, int], indexer_li: Callable) -> None:
    exp_err, exp_msg = (IndexingError, 'Too many indexers')
    with pytest.raises(exp_err, match=exp_msg):
        indexer_li(ser)[keys]
    if indexer_li == tm.iloc:
        exp_err, exp_msg = (IndexError, 'too many indices for array')
    with pytest.raises(exp_err, match=exp_msg):
        indexer_li(ser)[keys] = 0

def test_ser_list_indexer_exceeds_dimensions(indexer_li: Callable) -> None:
    ser = Series([10])
    res = indexer_li(ser)[[0, 0]]
    exp = Series([10, 10], index=Index([0, 0]))
    tm.assert_series_equal(res, exp)

@pytest.mark.parametrize('value', [(0, 1), [0, 1], np.array([0, 1]), array.array('b', [0, 1])])
def test_scalar_setitem_with_nested_value(value: Union[Tuple[int, int], List[int], np.ndarray, array.array]) -> None:
    df = DataFrame({'A': [1, 2, 3]})
    msg = '|'.join(['Must have equal len keys and value', 'setting an array element with a sequence'])
    with pytest.raises(ValueError, match=msg):
        df.loc[0, 'B'] = value
    df = DataFrame({'A': [1, 2, 3], 'B': np.array([1, 'a', 'b'], dtype=object)})
    with pytest.raises(ValueError, match='Must have equal len keys and value'):
        df.loc[0, 'B'] = value

@pytest.mark.parametrize('value', [(0, 1), [0, 1], np.array([0, 1]), array.array('b', [0, 1])])
def test_scalar_setitem_series_with_nested_value(value: Union[Tuple[int, int], List[int], np.ndarray, array.array], indexer_sli: Callable) -> None:
    ser = Series([1, 2, 3])
    with pytest.raises(ValueError, match='setting an array element with a sequence'):
        indexer_sli(ser)[0] = value
    ser = Series([1, 'a', 'b'], dtype=object)
    indexer_sli(ser)[0] = value
    if isinstance(value, np.ndarray):
        assert (ser.loc[0] == value).all()
    else:
        assert ser.loc[0] == value

@pytest.mark.parametrize('value', [(0.0,), [0.0], np.array([0.0]), array.array('d', [0.0])])
def test_scalar_setitem_with_nested_value_length1(value: Union[Tuple[float], List[float], np.ndarray, array.array]) -> None:
    df = DataFrame({'A': [1, 2, 3]})
    df.loc[0, 'B'] = value
    expected = DataFrame({'A': [1, 2, 3], 'B': [0.0, np.nan, np.nan]})
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'A': [1, 2, 3], 'B': np.array([1, 'a', 'b'], dtype=object)})
    df.loc[0, 'B'] = value
    if isinstance(value, np.ndarray):
        assert (df.loc[0, 'B'] == value).all()
    else:
        assert df.loc[0, 'B'] == value

@pytest.mark.parametrize('value', [(0.0,), [0.0], np.array([0.0]), array.array('d', [0.0])])
def test_scalar_setitem_series_with_nested_value_length1(value: Union[Tuple[float], List[float], np.ndarray, array.array], indexer_sli: Callable) -> None:
    ser = Series([1.0, 2.0, 3.0])
    if isinstance(value, np.ndarray):
        indexer_sli(ser)[0] = value
        expected = Series([0.0, 2.0, 3.0])
        tm.assert_series_equal(ser, expected)
    else:
        with pytest.raises(ValueError, match='setting an array element with a sequence'):
            indexer_sli(ser)[0] = value
    ser = Series([1, 'a', 'b'], dtype=object)
    indexer_sli(ser)[0] = value
    if isinstance(value, np.ndarray):
        assert (ser.loc[0] == value).all()
    else:
        assert ser.loc[0] == value

def test_object_dtype_series_set_series_element() -> None:
    s1 = Series(dtype='O', index=['a', 'b'])
    s1['a'] = Series()
    s1.loc['b'] = Series()
    tm.assert_series_equal(s1.loc['a'], Series())
    tm.assert_series_equal(s1.loc['b'], Series())
    s2 = Series(dtype='O', index=['a', 'b'])
    s2.iloc[1] = Series()
    tm.assert_series_equal(s2.iloc[1], Series())

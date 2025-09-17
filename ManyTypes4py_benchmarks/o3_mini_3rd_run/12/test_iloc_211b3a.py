#!/usr/bin/env python3
"""test positional based indexing with iloc"""
from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (Categorical, CategoricalDtype, DataFrame, Index, Interval,
                    NaT, Series, Timestamp, array, concat, date_range, interval_range,
                    isna, to_datetime)
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
from typing import Any, Callable, List, Union, Sequence
from numpy import ndarray

_slice_iloc_msg = re.escape('only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices')


class TestiLoc:

    @pytest.mark.parametrize('key', [2, -1, [0, 1, 2]])
    @pytest.mark.parametrize('index', [
        Index(list('abcd'), dtype=object),
        Index([2, 4, 'null', 8], dtype=object),
        date_range('20130101', periods=4),
        Index(range(0, 8, 2), dtype=np.float64),
        Index([])
    ])
    def test_iloc_getitem_int_and_list_int(self, key: Union[int, List[int]],
                                           frame_or_series: Callable[[Sequence[int], Index], Union[DataFrame, Series]],
                                           index: Index,
                                           request: Any) -> None:
        obj = frame_or_series(range(len(index)), index=index)
        check_indexing_smoketest_or_raises(obj, 'iloc', key, fails=IndexError)

        
class TestiLocBaseIndependent:
    """Tests Independent Of Base Class"""

    @pytest.mark.parametrize('key', [slice(None), slice(3), range(3), [0, 1, 2], Index(range(3)), np.asarray([0, 1, 2])])
    def test_iloc_setitem_fullcol_categorical(self, indexer_li: Callable[[DataFrame], Any],
                                              key: Union[slice, range, List[int], Index, ndarray]) -> None:
        df: DataFrame = DataFrame({0: list(range(3))}, dtype=object)
        cat = Categorical(['alpha', 'beta', 'gamma'])
        assert df._mgr.blocks[0]._can_hold_element(cat)
        df_copy: DataFrame = df.copy()
        orig_vals: ndarray = df_copy.values
        indexer_li(df_copy)[key, 0] = cat
        expected: DataFrame = DataFrame({0: cat}).astype(object)
        assert np.shares_memory(df_copy[0].values, orig_vals)
        tm.assert_frame_equal(df_copy, expected)
        df_copy.iloc[0, 0] = 'gamma'
        assert cat[0] != 'gamma'
        frame: DataFrame = DataFrame({0: np.array([0, 1, 2], dtype=object), 1: list(range(3))})
        df_copy = frame.copy()
        indexer_li(df_copy)[key, 0] = cat
        expected = DataFrame({0: Series(cat.astype(object), dtype=object), 1: list(range(3))})
        tm.assert_frame_equal(df_copy, expected)

    def test_iloc_setitem_ea_inplace(self, frame_or_series: Callable[[Any], Union[DataFrame, Series]],
                                     index_or_series_or_array: Callable[[Any], Any]) -> None:
        arr = array([1, 2, 3, 4])
        obj = frame_or_series(arr.to_numpy('i8'))
        if frame_or_series is Series:
            values = obj.values
        else:
            values = obj._mgr.blocks[0].values
        if frame_or_series is Series:
            obj.iloc[:2] = index_or_series_or_array(arr[2:])
        else:
            obj.iloc[:2, 0] = index_or_series_or_array(arr[2:])
        expected = frame_or_series(np.array([3, 4, 3, 4], dtype='i8'))
        tm.assert_equal(obj, expected)
        if frame_or_series is Series:
            assert obj.values is not values
            assert np.shares_memory(obj.values, values)
        else:
            assert np.shares_memory(obj[0].values, values)

    def test_is_scalar_access(self) -> None:
        index: Index = Index([1, 2, 1])
        ser: Series = Series(range(3), index=index)
        assert ser.iloc._is_scalar_access((1,))
        df: DataFrame = ser.to_frame()
        assert df.iloc._is_scalar_access((1, 0))

    def test_iloc_exceeds_bounds(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((20, 5)), columns=list('ABCDE'))
        msg: str = 'positional indexers are out-of-bounds'
        with pytest.raises(IndexError, match=msg):
            df.iloc[:, [0, 1, 2, 3, 4, 5]]
        with pytest.raises(IndexError, match=msg):
            df.iloc[[1, 30]]
        with pytest.raises(IndexError, match=msg):
            df.iloc[[1, -30]]
        with pytest.raises(IndexError, match=msg):
            df.iloc[[100]]
        s: Series = df['A']
        with pytest.raises(IndexError, match=msg):
            s.iloc[[100]]
        with pytest.raises(IndexError, match=msg):
            s.iloc[[-100]]
        msg = 'single positional indexer is out-of-bounds'
        with pytest.raises(IndexError, match=msg):
            df.iloc[30]
        with pytest.raises(IndexError, match=msg):
            df.iloc[-30]
        with pytest.raises(IndexError, match=msg):
            s.iloc[30]
        with pytest.raises(IndexError, match=msg):
            s.iloc[-30]
        result: DataFrame = df.iloc[:, 4:10]
        expected: DataFrame = df.iloc[:, 4:]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[:, -4:-10]
        expected = df.iloc[:, :0]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[:, 10:4:-1]
        expected = df.iloc[:, :4:-1]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[:, 4:-10:-1]
        expected = df.iloc[:, 4::-1]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[:, -10:4]
        expected = df.iloc[:, :4]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[:, 10:4]
        expected = df.iloc[:, :0]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[:, -10:-11:-1]
        expected = df.iloc[:, :0]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[:, 10:11]
        expected = df.iloc[:, :0]
        tm.assert_frame_equal(result, expected)
        result = s.iloc[18:30]
        expected = s.iloc[18:]
        tm.assert_series_equal(result, expected)
        result = s.iloc[30:]
        expected = s.iloc[:0]
        tm.assert_series_equal(result, expected)
        result = s.iloc[30::-1]
        expected = s.iloc[::-1]
        tm.assert_series_equal(result, expected)
        dfl: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), columns=list('AB'))
        tm.assert_frame_equal(dfl.iloc[:, 2:3],
                              DataFrame(index=dfl.index, columns=Index([], dtype=dfl.columns.dtype)))
        tm.assert_frame_equal(dfl.iloc[:, 1:3], dfl.iloc[:, [1]])
        tm.assert_frame_equal(dfl.iloc[4:6], dfl.iloc[[4]])
        msg = 'positional indexers are out-of-bounds'
        with pytest.raises(IndexError, match=msg):
            dfl.iloc[[4, 5, 6]]
        msg = 'single positional indexer is out-of-bounds'
        with pytest.raises(IndexError, match=msg):
            dfl.iloc[:, 4]

    @pytest.mark.parametrize('index,columns', [(np.arange(20), list('ABCDE'))])
    @pytest.mark.parametrize('index_vals,column_vals',
                             [[slice(None), ['A', 'D']], (['1', '2'], slice(None)), ([datetime(2019, 1, 1)], slice(None))])
    def test_iloc_non_integer_raises(self, index: Any, columns: List[Any],
                                      index_vals: Any, column_vals: Any) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(index), len(columns))), index=index, columns=columns)
        msg: str = '.iloc requires numeric indexers, got'
        with pytest.raises(IndexError, match=msg):
            df.iloc[index_vals, column_vals]

    def test_iloc_getitem_invalid_scalar(self, frame_or_series: Callable[[Any], Union[DataFrame, Series]]) -> None:
        obj: DataFrame = DataFrame(np.arange(100).reshape(10, 10))
        obj = tm.get_obj(obj, frame_or_series)
        with pytest.raises(TypeError, match='Cannot index by location index'):
            obj.iloc['a']

    def test_iloc_array_not_mutating_negative_indices(self) -> None:
        array_with_neg_numbers: ndarray = np.array([1, 2, -1])
        array_copy: ndarray = array_with_neg_numbers.copy()
        df: DataFrame = DataFrame({'A': [100, 101, 102],
                                   'B': [103, 104, 105],
                                   'C': [106, 107, 108]}, index=[1, 2, 3])
        df.iloc[array_with_neg_numbers]
        tm.assert_numpy_array_equal(array_with_neg_numbers, array_copy)
        df.iloc[:, array_with_neg_numbers]
        tm.assert_numpy_array_equal(array_with_neg_numbers, array_copy)

    def test_iloc_getitem_neg_int_can_reach_first_index(self) -> None:
        df: DataFrame = DataFrame({'A': [2, 3, 5], 'B': [7, 11, 13]})
        s: Series = df['A']
        expected: Series = df.iloc[0]
        result: Series = df.iloc[-3]
        tm.assert_series_equal(result, expected)
        expected_df: DataFrame = df.iloc[[0]]
        result_df: DataFrame = df.iloc[[-3]]
        tm.assert_frame_equal(result_df, expected_df)
        expected_scalar: Any = s.iloc[0]
        result_scalar: Any = s.iloc[-3]
        assert result_scalar == expected_scalar
        expected_series: Series = s.iloc[[0]]
        result_series: Series = s.iloc[[-3]]
        tm.assert_series_equal(result_series, expected_series)
        expected: Series = Series(['a'], index=['A'])
        result: Series = expected.iloc[[-1]]
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_dups(self) -> None:
        df1: DataFrame = DataFrame([{'A': None, 'B': 1}, {'A': 2, 'B': 2}])
        df2: DataFrame = DataFrame([{'A': 3, 'B': 3}, {'A': 4, 'B': 4}])
        df: DataFrame = concat([df1, df2], axis=1)
        result = df.iloc[0, 0]
        assert isna(result)
        result = df.iloc[0, :]
        expected: Series = Series([np.nan, 1, 3, 3], index=['A', 'B', 'A', 'B'], name=0)
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_array(self) -> None:
        df: DataFrame = DataFrame([{'A': 1, 'B': 2, 'C': 3},
                                    {'A': 100, 'B': 200, 'C': 300},
                                    {'A': 1000, 'B': 2000, 'C': 3000}])
        expected: DataFrame = DataFrame([{'A': 1, 'B': 2, 'C': 3}])
        tm.assert_frame_equal(df.iloc[[0]], expected)
        expected = DataFrame([{'A': 1, 'B': 2, 'C': 3},
                              {'A': 100, 'B': 200, 'C': 300}])
        tm.assert_frame_equal(df.iloc[[0, 1]], expected)
        expected = DataFrame([{'B': 2, 'C': 3},
                              {'B': 2000, 'C': 3000}], index=[0, 2])
        result = df.iloc[[0, 2], [1, 2]]
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_bool(self) -> None:
        df: DataFrame = DataFrame([{'A': 1, 'B': 2, 'C': 3},
                                   {'A': 100, 'B': 200, 'C': 300},
                                   {'A': 1000, 'B': 2000, 'C': 3000}])
        expected: DataFrame = DataFrame([{'A': 1, 'B': 2, 'C': 3},
                                         {'A': 100, 'B': 200, 'C': 300}])
        result = df.iloc[[True, True, False]]
        tm.assert_frame_equal(result, expected)
        expected = DataFrame([{'A': 1, 'B': 2, 'C': 3},
                              {'A': 1000, 'B': 2000, 'C': 3000}], index=[0, 2])
        result = df.iloc[lambda x: x.index % 2 == 0]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('index', [[True, False], [True, False, True, False]])
    def test_iloc_getitem_bool_diff_len(self, index: List[bool]) -> None:
        s: Series = Series([1, 2, 3])
        msg: str = f'Boolean index has wrong length: {len(index)} instead of {len(s)}'
        with pytest.raises(IndexError, match=msg):
            s.iloc[index]

    def test_iloc_getitem_slice(self) -> None:
        df: DataFrame = DataFrame([{'A': 1, 'B': 2, 'C': 3},
                                   {'A': 100, 'B': 200, 'C': 300},
                                   {'A': 1000, 'B': 2000, 'C': 3000}])
        expected: DataFrame = DataFrame([{'A': 1, 'B': 2, 'C': 3},
                                         {'A': 100, 'B': 200, 'C': 300}])
        result = df.iloc[:2]
        tm.assert_frame_equal(result, expected)
        expected = DataFrame([{'A': 100, 'B': 200}], index=[1])
        result = df.iloc[1:2, 0:2]
        tm.assert_frame_equal(result, expected)
        expected = DataFrame([{'A': 1, 'C': 3},
                              {'A': 100, 'C': 300},
                              {'A': 1000, 'C': 3000}])
        result = df.iloc[:, lambda df_: [0, 2]]
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_slice_dups(self) -> None:
        df1: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=['A', 'A', 'B', 'B'])
        df2: DataFrame = DataFrame(np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2), columns=['A', 'C'])
        df: DataFrame = concat([df1, df2], axis=1)
        tm.assert_frame_equal(df.iloc[:, :4], df1)
        tm.assert_frame_equal(df.iloc[:, 4:], df2)
        df = concat([df2, df1], axis=1)
        tm.assert_frame_equal(df.iloc[:, :2], df2)
        tm.assert_frame_equal(df.iloc[:, 2:], df1)
        exp: DataFrame = concat([df2, df1.iloc[:, [0]]], axis=1)
        tm.assert_frame_equal(df.iloc[:, 0:3], exp)
        df = concat([df, df], axis=0)
        tm.assert_frame_equal(df.iloc[0:10, :2], df2)
        tm.assert_frame_equal(df.iloc[0:10, 2:], df1)
        tm.assert_frame_equal(df.iloc[10:, :2], df2)
        tm.assert_frame_equal(df.iloc[10:, 2:], df1)

    def test_iloc_setitem(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((4, 4)),
                                   index=np.arange(0, 8, 2),
                                   columns=np.arange(0, 12, 3))
        df.iloc[1, 1] = 1
        result: Any = df.iloc[1, 1]
        assert result == 1
        df.iloc[:, 2:3] = 0
        expected: DataFrame = df.iloc[:, 2:3]
        result = df.iloc[:, 2:3]
        tm.assert_frame_equal(result, expected)
        s: Series = Series(0, index=[4, 5, 6])
        s.iloc[1:2] += 1
        expected = Series([0, 1, 0], index=[4, 5, 6])
        tm.assert_series_equal(s, expected)

    def test_iloc_setitem_axis_argument(self) -> None:
        df: DataFrame = DataFrame([[6, 'c', 10], [7, 'd', 11], [8, 'e', 12]])
        df[1] = df[1].astype(object)
        expected: DataFrame = DataFrame([[6, 'c', 10], [7, 'd', 11], [5, 5, 5]])
        expected[1] = expected[1].astype(object)
        df.iloc(axis=0)[2] = 5
        tm.assert_frame_equal(df, expected)
        df = DataFrame([[6, 'c', 10], [7, 'd', 11], [8, 'e', 12]])
        df[1] = df[1].astype(object)
        expected = DataFrame([[6, 'c', 5], [7, 'd', 5], [8, 'e', 5]])
        expected[1] = expected[1].astype(object)
        df.iloc(axis=1)[2] = 5
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_list(self) -> None:
        df: DataFrame = DataFrame(np.arange(9).reshape((3, 3)),
                                   index=['A', 'B', 'C'],
                                   columns=['A', 'B', 'C'])
        df.iloc[[0, 1], [1, 2]]
        df.iloc[[0, 1], [1, 2]] += 100
        expected: DataFrame = DataFrame(np.array([0, 101, 102, 3, 104, 105, 6, 7, 8]).reshape((3, 3)),
                                        index=['A', 'B', 'C'],
                                        columns=['A', 'B', 'C'])
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_pandas_object(self) -> None:
        s_orig: Series = Series([0, 1, 2, 3])
        expected: Series = Series([0, -1, -2, 3])
        s: Series = s_orig.copy()
        s.iloc[Series([1, 2])] = [-1, -2]
        tm.assert_series_equal(s, expected)
        s = s_orig.copy()
        s.iloc[Index([1, 2])] = [-1, -2]
        tm.assert_series_equal(s, expected)

    def test_iloc_setitem_dups(self) -> None:
        df1: DataFrame = DataFrame([{'A': None, 'B': 1}, {'A': 2, 'B': 2}])
        df2: DataFrame = DataFrame([{'A': 3, 'B': 3}, {'A': 4, 'B': 4}])
        df: DataFrame = concat([df1, df2], axis=1)
        expected: DataFrame = df.fillna(3)
        inds = np.isnan(df.iloc[:, 0])
        mask = inds[inds].index
        df.iloc[mask, 0] = df.iloc[mask, 2]
        tm.assert_frame_equal(df, expected)
        expected = DataFrame({0: [1, 2], 1: [3, 4]})
        expected.columns = ['B', 'B']
        del df['A']
        tm.assert_frame_equal(df, expected)
        df.iloc[[0, 1], [0, 1]] = df.iloc[[0, 1], [0, 1]]
        tm.assert_frame_equal(df, expected)
        df.iloc[[1, 0], [0, 1]] = df.iloc[[1, 0], [0, 1]].reset_index(drop=True)
        df.iloc[[1, 0], [0, 1]] = df.iloc[[1, 0], [0, 1]].reset_index(drop=True)
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_frame_duplicate_columns_multiple_blocks(self) -> None:
        df: DataFrame = DataFrame([[0, 1], [2, 3]], columns=['B', 'B'])
        df.iloc[:, 0] = df.iloc[:, 0].astype('f8')
        assert len(df._mgr.blocks) == 1
        with pytest.raises(TypeError, match='Invalid value'):
            df.iloc[:, 0] = df.iloc[:, 0] + 0.5
        expected: DataFrame = df.copy()
        df.iloc[[0, 1], [0, 1]] = df.iloc[[0, 1], [0, 1]]
        tm.assert_frame_equal(df, expected)

    def test_iloc_getitem_frame(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                   index=range(0, 20, 2),
                                   columns=range(0, 8, 2))
        result: Series = df.iloc[2]
        exp: Series = df.loc[4]
        tm.assert_series_equal(result, exp)
        result = df.iloc[2, 2]
        exp = df.loc[4, 4]
        assert result == exp
        result = df.iloc[4:8]
        expected = df.loc[8:14]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[:, 2:3]
        expected = df.loc[:, 4:5]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[[0, 1, 3]]
        expected = df.loc[[0, 2, 6]]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[[0, 1, 3], [0, 1]]
        expected = df.loc[[0, 2, 6], [0, 2]]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[[-1, 1, 3], [-1, 1]]
        expected = df.loc[[18, 2, 6], [6, 2]]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[[-1, -1, 1, 3], [-1, 1]]
        expected = df.loc[[18, 18, 2, 6], [6, 2]]
        tm.assert_frame_equal(result, expected)
        s: Series = Series(index=range(1, 5), dtype=object)
        result = df.iloc[s.index]
        expected = df.loc[[2, 4, 6, 8]]
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_labelled_frame(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                   index=list('abcdefghij'), columns=list('ABCD'))
        result = df.iloc[1, 1]
        exp = df.loc['b', 'B']
        assert result == exp
        result = df.iloc[:, 2:3]
        expected = df.loc[:, ['C']]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[-1, -1]
        exp = df.loc['j', 'D']
        assert result == exp
        msg: str = 'index 5 is out of bounds for axis 0 with size 4|index out of bounds'
        with pytest.raises(IndexError, match=msg):
            df.iloc[10, 5]
        msg = 'Location based indexing can only have \\[integer, integer slice \\(START point is INCLUDED, END point is EXCLUDED\\), listlike of integers, boolean array\\] types'
        with pytest.raises(ValueError, match=msg):
            df.iloc['j', 'D']

    def test_iloc_getitem_doc_issue(self) -> None:
        arr: ndarray = np.random.default_rng(2).standard_normal((6, 4))
        index = date_range('20130101', periods=6)
        columns: List[str] = list('ABCD')
        df: DataFrame = DataFrame(arr, index=index, columns=columns)
        df.describe()
        result: DataFrame = df.iloc[3:5, 0:2]
        expected: DataFrame = DataFrame(arr[3:5, 0:2], index=index[3:5], columns=columns[0:2])
        tm.assert_frame_equal(result, expected)
        df.columns = list('aaaa')
        result = df.iloc[3:5, 0:2]
        expected = DataFrame(arr[3:5, 0:2], index=index[3:5], columns=list('aa'))
        tm.assert_frame_equal(result, expected)
        arr = np.random.default_rng(2).standard_normal((6, 4))
        index = list(range(0, 12, 2))
        columns = list(range(0, 8, 2))
        df = DataFrame(arr, index=index, columns=columns)
        df._mgr.blocks[0].mgr_locs
        result = df.iloc[1:5, 2:4]
        expected = DataFrame(arr[1:5, 2:4], index=index[1:5], columns=columns[2:4])
        tm.assert_frame_equal(result, expected)

    def test_iloc_setitem_series(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                   index=list('abcdefghij'), columns=list('ABCD'))
        df.iloc[1, 1] = 1
        result: Any = df.iloc[1, 1]
        assert result == 1
        df.iloc[:, 2:3] = 0
        expected: DataFrame = df.iloc[:, 2:3]
        result = df.iloc[:, 2:3]
        tm.assert_frame_equal(result, expected)
        s: Series = Series(np.random.default_rng(2).standard_normal(10), index=range(0, 20, 2))
        s.iloc[1] = 1
        result = s.iloc[1]
        assert result == 1
        s.iloc[:4] = 0
        expected = s.iloc[:4]
        result = s.iloc[:4]
        tm.assert_series_equal(result, expected)
        s = Series([-1] * 6)
        s.iloc[0::2] = [0, 2, 4]
        s.iloc[1::2] = [1, 3, 5]
        result = s
        expected = Series([0, 1, 2, 3, 4, 5])
        tm.assert_series_equal(result, expected)

    def test_iloc_setitem_list_of_lists(self) -> None:
        df: DataFrame = DataFrame({'A': np.arange(5, dtype='int64'), 'B': np.arange(5, 10, dtype='int64')})
        df.iloc[2:4] = [[10, 11], [12, 13]]
        expected: DataFrame = DataFrame({'A': [0, 1, 10, 12, 4], 'B': [5, 6, 11, 13, 9]})
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'A': ['a', 'b', 'c', 'd', 'e'], 'B': np.arange(5, 10, dtype='int64')})
        df.iloc[2:4] = [['x', 11], ['y', 13]]
        expected = DataFrame({'A': ['a', 'b', 'x', 'y', 'e'], 'B': [5, 6, 11, 13, 9]})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('indexer', [[0], slice(None, 1, None), np.array([0])])
    @pytest.mark.parametrize('value', [['Z'], np.array(['Z'])])
    def test_iloc_setitem_with_scalar_index(self, indexer: Union[List[int], slice, ndarray],
                                            value: Union[List[str], ndarray]) -> None:
        df: DataFrame = DataFrame([[1, 2], [3, 4]], columns=['A', 'B']).astype({'A': object})
        df.iloc[0, indexer] = value
        result = df.iloc[0, 0]
        assert is_scalar(result) and result == 'Z'

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_iloc_mask(self) -> None:
        df: DataFrame = DataFrame(list(range(5)), index=list('ABCDE'), columns=['a'])
        mask: Series = df.a % 2 == 0
        msg: str = 'iLocation based boolean indexing cannot use an indexable as a mask'
        with pytest.raises(ValueError, match=msg):
            df.iloc[mask]
        mask.index = list(range(len(mask)))
        msg = 'iLocation based boolean indexing on an integer type is not available'
        with pytest.raises(NotImplementedError, match=msg):
            df.iloc[mask]
        result: DataFrame = df.iloc[np.array([True] * len(mask), dtype=bool)]
        tm.assert_frame_equal(result, df)
        locs: ndarray = np.arange(4)
        nums: ndarray = 2 ** locs
        reps: List[str] = [bin(num) for num in nums]
        df = DataFrame({'locs': locs, 'nums': nums}, index=reps)
        expected = {(None, ''): '0b1100', (None, '.loc'): '0b1100', (None, '.iloc'): '0b1100',
                    ('index', ''): '0b11', ('index', '.loc'): '0b11',
                    ('index', '.iloc'): 'iLocation based boolean indexing cannot use an indexable as a mask',
                    ('locs', ''): 'Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match).',
                    ('locs', '.loc'): 'Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match).',
                    ('locs', '.iloc'): 'iLocation based boolean indexing on an integer type is not available'}
        for idx in [None, 'index', 'locs']:
            mask = (df.nums > 2).values
            if idx:
                mask_index = getattr(df, idx)[::-1]
                mask = Series(mask, list(mask_index))
            for method in ['', '.loc', '.iloc']:
                try:
                    if method:
                        accessor = getattr(df, method[1:])
                    else:
                        accessor = df
                    answer = str(bin(accessor[mask]['nums'].sum()))
                except (ValueError, IndexingError, NotImplementedError) as err:
                    answer = str(err)
                key = (idx, method)
                r = expected.get(key)
                if r != answer:
                    raise AssertionError(f'[{key}] does not match [{answer}], received [{r}]')

    def test_iloc_non_unique_indexing(self) -> None:
        df: DataFrame = DataFrame({'A': [0.1] * 3000, 'B': [1] * 3000})
        idx: ndarray = np.arange(30) * 99
        expected: DataFrame = df.iloc[idx]
        df3: DataFrame = concat([df, 2 * df, 3 * df])
        result: DataFrame = df3.iloc[idx]
        tm.assert_frame_equal(result, expected)
        df2: DataFrame = DataFrame({'A': [0.1] * 1000, 'B': [1] * 1000})
        df2 = concat([df2, 2 * df2, 3 * df2])
        with pytest.raises(KeyError, match='not in index'):
            df2.loc[idx]

    def test_iloc_empty_list_indexer_is_ok(self) -> None:
        df: DataFrame = DataFrame(np.ones((5, 2)),
                                  index=Index([f'i-{i}' for i in range(5)], name='a'),
                                  columns=Index([f'i-{i}' for i in range(2)], name='a'))
        tm.assert_frame_equal(df.iloc[:, []], df.iloc[:, :0], check_index_type=True, check_column_type=True)
        tm.assert_frame_equal(df.iloc[[], :], df.iloc[:0, :], check_index_type=True, check_column_type=True)
        tm.assert_frame_equal(df.iloc[[]], df.iloc[:0, :], check_index_type=True, check_column_type=True)

    def test_identity_slice_returns_new_object(self) -> None:
        original_df: DataFrame = DataFrame({'a': [1, 2, 3]})
        sliced_df: DataFrame = original_df.iloc[:]
        assert sliced_df is not original_df
        assert np.shares_memory(original_df['a'], sliced_df['a'])
        original_df.loc[:, 'a'] = [4, 4, 4]
        assert (sliced_df['a'] == [1, 2, 3]).all()
        original_series: Series = Series([1, 2, 3, 4, 5, 6])
        sliced_series: Series = original_series.iloc[:]
        assert sliced_series is not original_series
        original_series[:3] = [7, 8, 9]
        assert all(sliced_series[:3] == [1, 2, 3])

    def test_indexing_zerodim_np_array(self) -> None:
        df: DataFrame = DataFrame([[1, 2], [3, 4]])
        result: Series = df.iloc[np.array(0)]
        s: Series = Series([1, 2], name=0)
        tm.assert_series_equal(result, s)

    def test_series_indexing_zerodim_np_array(self) -> None:
        s: Series = Series([1, 2])
        result: Any = s.iloc[np.array(0)]
        assert result == 1

    def test_iloc_setitem_categorical_updates_inplace(self) -> None:
        cat = Categorical(['A', 'B', 'C'])
        df: DataFrame = DataFrame({1: cat, 2: [1, 2, 3]}, copy=False)
        assert tm.shares_memory(df[1], cat)
        df.iloc[:, 0] = cat[::-1]
        assert tm.shares_memory(df[1], cat)
        expected = Categorical(['C', 'B', 'A'], categories=['A', 'B', 'C'])
        tm.assert_categorical_equal(cat, expected)

    def test_iloc_with_boolean_operation(self) -> None:
        result: DataFrame = DataFrame([[0, 1], [2, 3], [4, 5], [6, np.nan]])
        result.iloc[result.index <= 2] *= 2
        expected: DataFrame = DataFrame([[0, 2], [4, 6], [8, 10], [6, np.nan]])
        tm.assert_frame_equal(result, expected)
        result.iloc[result.index > 2] *= 2
        expected = DataFrame([[0, 2], [4, 6], [8, 10], [12, np.nan]])
        tm.assert_frame_equal(result, expected)
        result.iloc[[True, True, False, False]] *= 2
        expected = DataFrame([[0, 4], [8, 12], [8, 10], [12, np.nan]])
        tm.assert_frame_equal(result, expected)
        result.iloc[[False, False, True, True]] /= 2
        expected = DataFrame([[0, 4.0], [8, 12.0], [4, 5.0], [6, np.nan]])
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_singlerow_slice_categoricaldtype_gives_series(self) -> None:
        df: DataFrame = DataFrame({'x': Categorical('a b c d e'.split())})
        result: Series = df.iloc[0]
        raw_cat = Categorical(['a'], categories=['a', 'b', 'c', 'd', 'e'])
        expected: Series = Series(raw_cat, index=['x'], name=0, dtype='category')
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_categorical_values(self) -> None:
        ser: Series = Series([1, 2, 3]).astype('category')
        result = ser.iloc[0:2]
        expected = Series([1, 2]).astype(CategoricalDtype([1, 2, 3]))
        tm.assert_series_equal(result, expected)
        result = ser.iloc[[0, 1]]
        expected = Series([1, 2]).astype(CategoricalDtype([1, 2, 3]))
        tm.assert_series_equal(result, expected)
        result = ser.iloc[[True, False, False]]
        expected = Series([1]).astype(CategoricalDtype([1, 2, 3]))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('value', [None, NaT, np.nan])
    def test_iloc_setitem_td64_values_cast_na(self, value: Any) -> None:
        series: Series = Series([0, 1, 2], dtype='timedelta64[ns]')
        series.iloc[0] = value
        expected: Series = Series([NaT, 1, 2], dtype='timedelta64[ns]')
        tm.assert_series_equal(series, expected)

    @pytest.mark.parametrize('not_na', [Interval(0, 1), 'a', 1.0])
    def test_setitem_mix_of_nan_and_interval(self, not_na: Any, nulls_fixture: Any) -> None:
        dtype = CategoricalDtype(categories=[not_na])
        ser: Series = Series([nulls_fixture, nulls_fixture, nulls_fixture, nulls_fixture], dtype=dtype)
        ser.iloc[:3] = [nulls_fixture, not_na, nulls_fixture]
        exp: Series = Series([nulls_fixture, not_na, nulls_fixture, nulls_fixture], dtype=dtype)
        tm.assert_series_equal(ser, exp)

    def test_iloc_setitem_empty_frame_raises_with_3d_ndarray(self) -> None:
        idx: Index = Index([])
        obj: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(idx), len(idx))),
                                     index=idx, columns=idx)
        nd3: ndarray = np.random.default_rng(2).integers(5, size=(2, 2, 2))
        msg: str = f'Cannot set values with ndim > {obj.ndim}'
        with pytest.raises(ValueError, match=msg):
            obj.iloc[nd3] = 0

    def test_iloc_getitem_read_only_values(self, indexer_li: Callable[[DataFrame], Any]) -> None:
        rw_array: ndarray = np.eye(10)
        rw_df: DataFrame = DataFrame(rw_array)
        ro_array: ndarray = np.eye(10)
        ro_array.setflags(write=False)
        ro_df: DataFrame = DataFrame(ro_array)
        tm.assert_frame_equal(indexer_li(rw_df)[[1, 2, 3]], indexer_li(ro_df)[[1, 2, 3]])
        tm.assert_frame_equal(indexer_li(rw_df)[[1]], indexer_li(ro_df)[[1]])
        tm.assert_series_equal(indexer_li(rw_df)[1], indexer_li(ro_df)[1])
        tm.assert_frame_equal(indexer_li(rw_df)[1:3], indexer_li(ro_df)[1:3])

    def test_iloc_getitem_readonly_key(self) -> None:
        df: DataFrame = DataFrame({'data': np.ones(100, dtype='float64')})
        indices: ndarray = np.array([1, 3, 6])
        indices.flags.writeable = False
        result: DataFrame = df.iloc[indices]
        expected: DataFrame = df.loc[[1, 3, 6]]
        tm.assert_frame_equal(result, expected)
        result = df['data'].iloc[indices]
        expected = df['data'].loc[[1, 3, 6]]
        tm.assert_series_equal(result, expected)

    def test_iloc_assign_series_to_df_cell(self) -> None:
        df: DataFrame = DataFrame(columns=['a'], index=[0])
        df.iloc[0, 0] = Series([1, 2, 3])
        expected: DataFrame = DataFrame({'a': [Series([1, 2, 3])]}, columns=['a'], index=[0])
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('klass', [list, np.array])
    def test_iloc_setitem_bool_indexer(self, klass: Callable[[List[Any]], Any]) -> None:
        df: DataFrame = DataFrame({'flag': ['x', 'y', 'z'], 'value': [1, 3, 4]})
        indexer = klass([True, False, True])
        df.iloc[indexer, 1] = df.iloc[indexer, 1] * 2
        expected: DataFrame = DataFrame({'flag': ['x', 'y', 'z'], 'value': [2, 3, 4]})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('indexer', [[1], slice(1, 2)])
    def test_iloc_setitem_pure_position_based(self, indexer: Union[List[int], slice]) -> None:
        df1: DataFrame = DataFrame({'a2': [11, 12, 13], 'b2': [14, 15, 16]})
        df2: DataFrame = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        df2.iloc[:, indexer] = df1.iloc[:, [0]]
        expected: DataFrame = DataFrame({'a': [1, 2, 3], 'b': [11, 12, 13], 'c': [7, 8, 9]})
        tm.assert_frame_equal(df2, expected)

    def test_iloc_setitem_dictionary_value(self) -> None:
        df: DataFrame = DataFrame({'x': [1, 2], 'y': [2, 2]})
        rhs: dict = {'x': 9, 'y': 99}
        df.iloc[1] = rhs
        expected: DataFrame = DataFrame({'x': [1, 9], 'y': [2, 99]})
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'x': [1, 2], 'y': [2.0, 2.0]})
        df.iloc[1] = rhs
        expected = DataFrame({'x': [1, 9], 'y': [2.0, 99.0]})
        tm.assert_frame_equal(df, expected)

    def test_iloc_getitem_float_duplicates(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((3, 3)),
                                   index=[0.1, 0.2, 0.2],
                                   columns=list('abc'))
        expect: DataFrame = df.iloc[1:]
        tm.assert_frame_equal(df.loc[0.2], expect)
        expect = df.iloc[1:, 0]
        tm.assert_series_equal(df.loc[0.2, 'a'], expect)
        df.index = [1, 0.2, 0.2]
        expect = df.iloc[1:]
        tm.assert_frame_equal(df.loc[0.2], expect)
        expect = df.iloc[1:, 0]
        tm.assert_series_equal(df.loc[0.2, 'a'], expect)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((4, 3)),
                                   index=[1, 0.2, 0.2, 1],
                                   columns=list('abc'))
        expect = df.iloc[1:-1]
        tm.assert_frame_equal(df.loc[0.2], expect)
        expect = df.iloc[1:-1, 0]
        tm.assert_series_equal(df.loc[0.2, 'a'], expect)
        df.index = [0.1, 0.2, 2, 0.2]
        expect = df.iloc[[1, -1]]
        tm.assert_frame_equal(df.loc[0.2], expect)
        expect = df.iloc[[1, -1], 0]
        tm.assert_series_equal(df.loc[0.2, 'a'], expect)

    def test_iloc_setitem_custom_object(self) -> None:

        class TO:
            def __init__(self, value: Any) -> None:
                self.value = value

            def __str__(self) -> str:
                return f'[{self.value}]'
            __repr__ = __str__

            def __eq__(self, other: Any) -> bool:
                return self.value == other.value

            def view(self) -> Any:
                return self

        df: DataFrame = DataFrame(index=[0, 1], columns=[0])
        df.iloc[1, 0] = TO(1)
        df.iloc[1, 0] = TO(2)
        result: DataFrame = DataFrame(index=[0, 1], columns=[0])
        result.iloc[1, 0] = TO(2)
        tm.assert_frame_equal(result, df)
        df = DataFrame(index=[0, 1], columns=[0])
        df.iloc[1, 0] = TO(1)
        df.iloc[1, 0] = np.nan
        result = DataFrame(index=[0, 1], columns=[0])
        tm.assert_frame_equal(result, df)

    def test_iloc_getitem_with_duplicates(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((3, 3)),
                                   columns=list('ABC'),
                                   index=list('aab'))
        result = df.iloc[0]
        assert isinstance(result, Series)
        tm.assert_almost_equal(result.values, df.values[0])
        result = df.T.iloc[:, 0]
        assert isinstance(result, Series)
        tm.assert_almost_equal(result.values, df.values[0])

    def test_iloc_getitem_with_duplicates2(self) -> None:
        df: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6]], columns=[1, 1, 2])
        result = df.iloc[:, [0]]
        expected = df.take([0], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_iloc_interval(self) -> None:
        df: DataFrame = DataFrame({Interval(1, 2): [1, 2]})
        result = df.iloc[0]
        expected: Series = Series({Interval(1, 2): 1}, name=0)
        tm.assert_series_equal(result, expected)
        result = df.iloc[:, 0]
        expected = Series([1, 2], name=Interval(1, 2))
        tm.assert_series_equal(result, expected)
        result = df.copy()
        result.iloc[:, 0] += 1
        expected = DataFrame({Interval(1, 2): [2, 3]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('indexing_func', [list, np.array])
    @pytest.mark.parametrize('rhs_func', [list, np.array])
    def test_loc_setitem_boolean_list(self, rhs_func: Callable[[List[Any]], Any],
                                      indexing_func: Callable[[List[Any]], Any]) -> None:
        ser: Series = Series([0, 1, 2])
        ser.iloc[indexing_func([True, False, True])] = rhs_func([5, 10])
        expected: Series = Series([5, 1, 10])
        tm.assert_series_equal(ser, expected)
        df: DataFrame = DataFrame({'a': [0, 1, 2]})
        df.iloc[indexing_func([True, False, True])] = rhs_func([[5], [10]])
        expected = DataFrame({'a': [5, 1, 10]})
        tm.assert_frame_equal(df, expected)

    def test_iloc_getitem_slice_negative_step_ea_block(self) -> None:
        df: DataFrame = DataFrame({'A': [1, 2, 3]}, dtype='Int64')
        res: DataFrame = df.iloc[:, ::-1]
        tm.assert_frame_equal(res, df)
        df['B'] = 'foo'
        res = df.iloc[:, ::-1]
        expected = DataFrame({'B': df['B'], 'A': df['A']})
        tm.assert_frame_equal(res, expected)

    def test_iloc_setitem_2d_ndarray_into_ea_block(self) -> None:
        df: DataFrame = DataFrame({'status': ['a', 'b', 'c']}, dtype='category')
        df.iloc[np.array([0, 1]), np.array([0])] = np.array([['a'], ['a']])
        expected: DataFrame = DataFrame({'status': ['a', 'a', 'c']}, dtype=df['status'].dtype)
        tm.assert_frame_equal(df, expected)

    def test_iloc_getitem_int_single_ea_block_view(self) -> None:
        arr: ndarray = interval_range(1, 10.0)._values
        df: DataFrame = DataFrame(arr)
        ser: Series = df.iloc[2]
        assert arr[2] != arr[-1]
        arr[2] = arr[-1]
        assert ser[0] == arr[-1]

    def test_iloc_setitem_multicolumn_to_datetime(self, using_infer_string: bool) -> None:
        df: DataFrame = DataFrame({'A': ['2022-01-01', '2022-01-02'], 'B': ['2021', '2022']})
        if using_infer_string:
            with pytest.raises(TypeError, match='Invalid value'):
                df.iloc[:, [0]] = DataFrame({'A': to_datetime(['2021', '2022'])})
        else:
            df.iloc[:, [0]] = DataFrame({'A': to_datetime(['2021', '2022'])})
            expected = DataFrame({'A': [Timestamp('2021-01-01 00:00:00'), Timestamp('2022-01-01 00:00:00')],
                                  'B': ['2021', '2022']})
            tm.assert_frame_equal(df, expected, check_dtype=False)


class TestILocErrors:

    def test_iloc_float_raises(self, series_with_simple_index: Series,
                                frame_or_series: Callable[[Any], Union[DataFrame, Series]]) -> None:
        obj: Union[Series, DataFrame] = series_with_simple_index
        if frame_or_series is DataFrame:
            obj = obj.to_frame()
        msg: str = 'Cannot index by location index with a non-integer key'
        with pytest.raises(TypeError, match=msg):
            obj.iloc[3.0]
        with pytest.raises(IndexError, match=_slice_iloc_msg):
            obj.iloc[3.0] = 0

    def test_iloc_getitem_setitem_fancy_exceptions(self, float_frame: DataFrame) -> None:
        with pytest.raises(IndexingError, match='Too many indexers'):
            float_frame.iloc[:, :, :]
        with pytest.raises(IndexError, match='too many indices for array'):
            float_frame.iloc[:, :, :] = 1

    def test_iloc_frame_indexer(self) -> None:
        df: DataFrame = DataFrame({'a': [1, 2, 3]})
        indexer: DataFrame = DataFrame({'a': [True, False, True]})
        msg: str = 'DataFrame indexer for .iloc is not supported. Consider using .loc'
        with pytest.raises(TypeError, match=msg):
            df.iloc[indexer] = 1
        msg = 'DataFrame indexer is not allowed for .iloc\nConsider using .loc for automatic alignment.'
        with pytest.raises(IndexError, match=msg):
            df.iloc[indexer]


class TestILocSetItemDuplicateColumns:

    def test_iloc_setitem_scalar_duplicate_columns(self) -> None:
        df1: DataFrame = DataFrame([{'A': None, 'B': 1}, {'A': 2, 'B': 2}])
        df2: DataFrame = DataFrame([{'A': 3, 'B': 3}, {'A': 4, 'B': 4}])
        df: DataFrame = concat([df1, df2], axis=1)
        df.iloc[0, 0] = -1
        assert df.iloc[0, 0] == -1
        assert df.iloc[0, 2] == 3
        assert df.dtypes.iloc[2] == np.int64

    def test_iloc_setitem_list_duplicate_columns(self) -> None:
        df: DataFrame = DataFrame([[0, 'str', 'str2']], columns=['a', 'b', 'b'])
        df.iloc[:, 2] = ['str3']
        expected: DataFrame = DataFrame([[0, 'str', 'str3']], columns=['a', 'b', 'b'])
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_series_duplicate_columns(self) -> None:
        df: DataFrame = DataFrame(np.arange(8, dtype=np.int64).reshape(2, 4), columns=['A', 'B', 'A', 'B'])
        df.iloc[:, 0] = df.iloc[:, 0].astype(np.float64)
        assert df.dtypes.iloc[2] == np.int64

    @pytest.mark.parametrize(['dtypes', 'init_value', 'expected_value'], [('int64', '0', 0), ('float', '1.2', 1.2)])
    def test_iloc_setitem_dtypes_duplicate_columns(self, dtypes: str, init_value: str, expected_value: Union[int, float]) -> None:
        df: DataFrame = DataFrame([[init_value, 'str', 'str2']], columns=['a', 'b', 'b'], dtype=object)
        df.iloc[:, 0] = df.iloc[:, 0].astype(dtypes)
        expected_df: DataFrame = DataFrame([[expected_value, 'str', 'str2']], columns=['a', 'b', 'b'], dtype=object)
        tm.assert_frame_equal(df, expected_df)


class TestILocCallable:

    def test_frame_iloc_getitem_callable(self) -> None:
        df: DataFrame = DataFrame({'X': [1, 2, 3, 4], 'Y': list('aabb')}, index=list('ABCD'))
        res: DataFrame = df.iloc[lambda x: [1, 3]]
        tm.assert_frame_equal(res, df.iloc[[1, 3]])
        res = df.iloc[lambda x: [1, 3], :]
        tm.assert_frame_equal(res, df.iloc[[1, 3], :])
        res = df.iloc[lambda x: [1, 3], lambda x: 0]
        tm.assert_series_equal(res, df.iloc[[1, 3], 0])
        res = df.iloc[lambda x: [1, 3], lambda x: [0]]
        tm.assert_frame_equal(res, df.iloc[[1, 3], [0]])
        res = df.iloc[[1, 3], lambda x: 0]
        tm.assert_series_equal(res, df.iloc[[1, 3], 0])
        res = df.iloc[[1, 3], lambda x: [0]]
        tm.assert_frame_equal(res, df.iloc[[1, 3], [0]])
        res = df.iloc[lambda x: [1, 3], 0]
        tm.assert_series_equal(res, df.iloc[[1, 3], 0])
        res = df.iloc[lambda x: [1, 3], [0]]
        tm.assert_frame_equal(res, df.iloc[[1, 3], [0]])

    def test_frame_iloc_setitem_callable(self) -> None:
        df: DataFrame = DataFrame({'X': [1, 2, 3, 4], 'Y': Series(list('aabb'), dtype=object)}, index=list('ABCD'))
        res: DataFrame = df.copy()
        res.iloc[lambda x: [1, 3]] = 0
        exp: DataFrame = df.copy()
        exp.iloc[[1, 3]] = 0
        tm.assert_frame_equal(res, exp)
        res = df.copy()
        res.iloc[lambda x: [1, 3], :] = -1
        exp = df.copy()
        exp.iloc[[1, 3], :] = -1
        tm.assert_frame_equal(res, exp)
        res = df.copy()
        res.iloc[lambda x: [1, 3], lambda x: 0] = 5
        exp = df.copy()
        exp.iloc[[1, 3], 0] = 5
        tm.assert_frame_equal(res, exp)
        res = df.copy()
        res.iloc[lambda x: [1, 3], lambda x: [0]] = 25
        exp = df.copy()
        exp.iloc[[1, 3], [0]] = 25
        tm.assert_frame_equal(res, exp)
        res = df.copy()
        res.iloc[[1, 3], lambda x: 0] = -3
        exp = df.copy()
        exp.iloc[[1, 3], 0] = -3
        tm.assert_frame_equal(res, exp)
        res = df.copy()
        res.iloc[[1, 3], lambda x: [0]] = -5
        exp = df.copy()
        exp.iloc[[1, 3], [0]] = -5
        tm.assert_frame_equal(res, exp)
        res = df.copy()
        res.iloc[lambda x: [1, 3], 0] = 10
        exp = df.copy()
        exp.iloc[[1, 3], 0] = 10
        tm.assert_frame_equal(res, exp)
        res = df.copy()
        res.iloc[lambda x: [1, 3], [0]] = [-5, -5]
        exp = df.copy()
        exp.iloc[[1, 3], [0]] = [-5, -5]
        tm.assert_frame_equal(res, exp)


class TestILocSeries:

    def test_iloc(self) -> None:
        ser: Series = Series(np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2)))
        ser_original: Series = ser.copy()
        for i in range(len(ser)):
            result = ser.iloc[i]
            exp = ser[ser.index[i]]
            tm.assert_almost_equal(result, exp)
        result = ser.iloc[slice(1, 3)]
        expected = ser.loc[2:4]
        tm.assert_series_equal(result, expected)
        with tm.assert_produces_warning(None):
            result[:] = 0
        tm.assert_series_equal(ser, ser_original)
        result = ser.iloc[[0, 2, 3, 4, 5]]
        expected = ser.reindex(ser.index[[0, 2, 3, 4, 5]])
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_nonunique(self) -> None:
        ser: Series = Series([0, 1, 2], index=[0, 1, 0])
        assert ser.iloc[2] == 2

    def test_iloc_setitem_pure_position_based(self) -> None:
        ser1: Series = Series([1, 2, 3])
        ser2: Series = Series([4, 5, 6], index=[1, 0, 2])
        ser1.iloc[1:3] = ser2.iloc[1:3]
        expected: Series = Series([1, 5, 6])
        tm.assert_series_equal(ser1, expected)

    def test_iloc_nullable_int64_size_1_nan(self) -> None:
        result: DataFrame = DataFrame({'a': ['test'], 'b': [np.nan]})
        with pytest.raises(TypeError, match='Invalid value'):
            result.loc[:, 'b'] = result.loc[:, 'b'].astype('Int64')
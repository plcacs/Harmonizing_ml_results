"""test positional based indexing with iloc"""
from datetime import datetime
import re
from typing import Any, List, Optional, Sequence, Union, cast

import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import Categorical, CategoricalDtype, DataFrame, Index, Interval, NaT, Series, Timestamp, array, concat, date_range, interval_range, isna, to_datetime
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises

_slice_iloc_msg = re.escape('only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices')

class TestiLoc:

    @pytest.mark.parametrize('key', [2, -1, [0, 1, 2]])
    @pytest.mark.parametrize('index', [Index(list('abcd'), dtype=object), Index([2, 4, 'null', 8], dtype=object), date_range('20130101', periods=4), Index(range(0, 8, 2), dtype=np.float64), Index([])])
    def test_iloc_getitem_int_and_list_int(self, key: Union[int, List[int]], frame_or_series: Any, index: Index, request: Any) -> None:
        obj = frame_or_series(range(len(index)), index=index)
        check_indexing_smoketest_or_raises(obj, 'iloc', key, fails=IndexError)

class TestiLocBaseIndependent:
    """Tests Independent Of Base Class"""

    @pytest.mark.parametrize('key', [slice(None), slice(3), range(3), [0, 1, 2], Index(range(3)), np.asarray([0, 1, 2])])
    def test_iloc_setitem_fullcol_categorical(self, indexer_li: Any, key: Any) -> None:
        frame = DataFrame({0: range(3)}, dtype=object)
        cat = Categorical(['alpha', 'beta', 'gamma'])
        assert frame._mgr.blocks[0]._can_hold_element(cat)
        df = frame.copy()
        orig_vals = df.values
        indexer_li(df)[key, 0] = cat
        expected = DataFrame({0: cat}).astype(object)
        assert np.shares_memory(df[0].values, orig_vals)
        tm.assert_frame_equal(df, expected)
        df.iloc[0, 0] = 'gamma'
        assert cat[0] != 'gamma'
        frame = DataFrame({0: np.array([0, 1, 2], dtype=object), 1: range(3)})
        df = frame.copy()
        indexer_li(df)[key, 0] = cat
        expected = DataFrame({0: Series(cat.astype(object), dtype=object), 1: range(3)})
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_ea_inplace(self, frame_or_series: Any, index_or_series_or_array: Any) -> None:
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
        index = Index([1, 2, 1])
        ser = Series(range(3), index=index)
        assert ser.iloc._is_scalar_access((1,))
        df = ser.to_frame()
        assert df.iloc._is_scalar_access((1, 0))

    def test_iloc_exceeds_bounds(self) -> None:
        df = DataFrame(np.random.default_rng(2).random((20, 5)), columns=list('ABCDE'))
        msg = 'positional indexers are out-of-bounds'
        with pytest.raises(IndexError, match=msg):
            df.iloc[:, [0, 1, 2, 3, 4, 5]]
        with pytest.raises(IndexError, match=msg):
            df.iloc[[1, 30]]
        with pytest.raises(IndexError, match=msg):
            df.iloc[[1, -30]]
        with pytest.raises(IndexError, match=msg):
            df.iloc[[100]]
        s = df['A']
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
        result = df.iloc[:, 4:10]
        expected = df.iloc[:, 4:]
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
        dfl = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), columns=list('AB'))
        tm.assert_frame_equal(dfl.iloc[:, 2:3], DataFrame(index=dfl.index, columns=Index([], dtype=dfl.columns.dtype)))
        tm.assert_frame_equal(dfl.iloc[:, 1:3], dfl.iloc[:, [1]])
        tm.assert_frame_equal(dfl.iloc[4:6], dfl.iloc[[4]])
        msg = 'positional indexers are out-of-bounds'
        with pytest.raises(IndexError, match=msg):
            dfl.iloc[[4, 5, 6]]
        msg = 'single positional indexer is out-of-bounds'
        with pytest.raises(IndexError, match=msg):
            dfl.iloc[:, 4]

    @pytest.mark.parametrize('index,columns', [(np.arange(20), list('ABCDE'))])
    @pytest.mark.parametrize('index_vals,column_vals', [[slice(None), ['A', 'D']], (['1', '2'], slice(None)), ([datetime(2019, 1, 1)], slice(None))])
    def test_iloc_non_integer_raises(self, index: Any, columns: Any, index_vals: Any, column_vals: Any) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((len(index), len(columns))), index=index, columns=columns)
        msg = '.iloc requires numeric indexers, got'
        with pytest.raises(IndexError, match=msg):
            df.iloc[index_vals, column_vals]

    def test_iloc_getitem_invalid_scalar(self, frame_or_series: Any) -> None:
        obj = DataFrame(np.arange(100).reshape(10, 10))
        obj = tm.get_obj(obj, frame_or_series)
        with pytest.raises(TypeError, match='Cannot index by location index'):
            obj.iloc['a']

    def test_iloc_array_not_mutating_negative_indices(self) -> None:
        array_with_neg_numbers = np.array([1, 2, -1])
        array_copy = array_with_neg_numbers.copy()
        df = DataFrame({'A': [100, 101, 102], 'B': [103, 104, 105], 'C': [106, 107, 108]}, index=[1, 2, 3])
        df.iloc[array_with_neg_numbers]
        tm.assert_numpy_array_equal(array_with_neg_numbers, array_copy)
        df.iloc[:, array_with_neg_numbers]
        tm.assert_numpy_array_equal(array_with_neg_numbers, array_copy)

    def test_iloc_getitem_neg_int_can_reach_first_index(self) -> None:
        df = DataFrame({'A': [2, 3, 5], 'B': [7, 11, 13]})
        s = df['A']
        expected = df.iloc[0]
        result = df.iloc[-3]
        tm.assert_series_equal(result, expected)
        expected = df.iloc[[0]]
        result = df.iloc[[-3]]
        tm.assert_frame_equal(result, expected)
        expected = s.iloc[0]
        result = s.iloc[-3]
        assert result == expected
        expected = s.iloc[[0]]
        result = s.iloc[[-3]]
        tm.assert_series_equal(result, expected)
        expected = Series(['a'], index=['A'])
        result = expected.iloc[[-1]]
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_dups(self) -> None:
        df1 = DataFrame([{'A': None, 'B': 1}, {'A': 2, 'B': 2}])
        df2 = DataFrame([{'A': 3, 'B': 3}, {'A': 4, 'B': 4}])
        df = concat([df1, df2], axis=1)
        result = df.iloc[0, 0]
        assert isna(result)
        result = df.iloc[0, :]
        expected = Series([np.nan, 1, 3, 3], index=['A', 'B', 'A', 'B'], name=0)
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_array(self) -> None:
        df = DataFrame([{'A': 1, 'B': 2, 'C': 3}, {'A': 100, 'B': 200, 'C': 300}, {'A': 1000, 'B': 2000, 'C': 3000}])
        expected = DataFrame([{'A': 1, 'B': 2, 'C': 3}])
        tm.assert_frame_equal(df.iloc[[0]], expected)
        expected = DataFrame([{'A': 1, 'B': 2, 'C': 3}, {'A': 100, 'B': 200, 'C': 300}])
        tm.assert_frame_equal(df.iloc[[0, 1]], expected)
        expected = DataFrame([{'B': 2, 'C': 3}, {'B': 2000, 'C': 3000}], index=[0, 2])
        result = df.iloc[[0, 2], [1, 2]]
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_bool(self) -> None:
        df = DataFrame([{'A': 1, 'B': 2, 'C': 3}, {'A': 100, 'B': 200, 'C': 300}, {'A': 1000, 'B': 2000, 'C': 3000}])
        expected = DataFrame([{'A': 1, 'B': 2, 'C': 3}, {'A': 100, 'B': 200, 'C': 300}])
        result = df.iloc[[True, True, False]]
        tm.assert_frame_equal(result, expected)
        expected = DataFrame([{'A': 1, 'B': 2, 'C': 3}, {'A': 1000, 'B': 2000, 'C': 3000}], index=[0, 2])
        result = df.iloc[lambda x: x.index % 2 == 0]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('index', [[True, False], [True, False, True, False]])
    def test_iloc_getitem_bool_diff_len(self, index: List[bool]) -> None:
        s = Series([1, 2, 3])
        msg = f'Boolean index has wrong length: {len(index)} instead of {len(s)}'
        with pytest.raises(IndexError, match=msg):
            s.iloc[index]

    def test_iloc_getitem_slice(self) -> None:
        df = DataFrame([{'A': 1, 'B': 2, 'C': 3}, {'A': 100, 'B': 200, 'C': 300}, {'A': 1000, 'B': 2000, 'C': 3000}])
        expected = DataFrame([{'A': 1, 'B': 2, 'C': 3}, {'A': 100, 'B': 200, 'C': 300}])
        result = df.iloc[:2]
        tm.assert_frame_equal(result, expected)
        expected = DataFrame([{'A': 100, 'B': 200}], index=[1])
        result = df.iloc[1:2, 0:2]
        tm.assert_frame_equal(result, expected)
        expected = DataFrame([{'A': 1, 'C': 3}, {'A': 100, 'C': 300}, {'A': 1000, 'C': 3000}])
        result = df.iloc[:, lambda df: [0, 2]]
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_slice_dups(self) -> None:
        df1 = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=['A', 'A', 'B', 'B'])
        df2 = DataFrame(np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2), columns=['A', 'C'])
        df = concat([df1, df2], axis=1)
        tm.assert_frame_equal(df.iloc[:, :4], df1)
        tm.assert_frame_equal(df.iloc[:, 4:], df2)
        df = concat([df2, df1], axis=1)
        tm.assert_frame_equal(df.iloc[:, :2], df2)
        tm.assert_frame_equal(df.iloc[:, 2:], df1)
        exp = concat([df2, df1.iloc[:, [0]]], axis=1)
        tm.assert_frame_equal(df.iloc[:, 0:3], exp)
        df = concat([df, df], axis=0)
        tm.assert_frame_equal(df.iloc[0:10, :2], df2)
        tm.assert_frame_equal(df.iloc[0:10, 2:], df1)
        tm.assert_frame_equal(df.iloc[10:, :2], df2)
        tm.assert_frame_equal(df.iloc[10:, 2:], df1)

    def test_iloc_setitem(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=np.arange(0, 8, 2), columns=np.arange(0, 12, 3))
        df.iloc[1, 1] = 1
        result = df.iloc[1, 1]
        assert result == 1
        df.iloc[:, 2:3] = 0
        expected = df.iloc[:, 2:3]
        result = df.iloc[:, 2:3]
        tm.assert_frame_equal(result, expected)
        s = Series(0, index=[4, 5, 6])
        s.iloc[1:2] += 1
        expected = Series([0, 1, 0], index=[4, 5, 6])
        tm.assert_series_equal(s, expected)

    def test_iloc_setitem_axis_argument(self) -> None:
        df = DataFrame([[6, 'c', 10], [7, 'd', 11], [8, 'e', 12]])
        df[1] = df[1].astype(object)
        expected = DataFrame([[6, 'c', 10], [7, 'd', 11], [5, 5, 5]])
        expected[1] = expected[1].astype(object)
        df.iloc(axis=0)[2] = 5
        tm.assert_frame_equal(df, expected)
        df = DataFrame([[6, 'c', 10], [7, 'd', 11], [8, 'e', 12]])

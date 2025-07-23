from collections import namedtuple
from datetime import timedelta
import re
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import Categorical, DataFrame, Index, MultiIndex, date_range
import pandas._testing as tm

class TestSliceLocs:

    def test_slice_locs_partial(self, idx: MultiIndex) -> None:
        sorted_idx, _ = idx.sortlevel(0)
        result = sorted_idx.slice_locs(('foo', 'two'), ('qux', 'one'))
        assert result == (1, 5)
        result = sorted_idx.slice_locs(None, ('qux', 'one'))
        assert result == (0, 5)
        result = sorted_idx.slice_locs(('foo', 'two'), None)
        assert result == (1, len(sorted_idx))
        result = sorted_idx.slice_locs('bar', 'baz')
        assert result == (2, 4)

    def test_slice_locs(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((50, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=50, freq='B'))
        stacked = df.stack()
        idx = stacked.index
        slob = slice(*idx.slice_locs(df.index[5], df.index[15]))
        sliced = stacked[slob]
        expected = df[5:16].stack()
        tm.assert_almost_equal(sliced.values, expected.values)
        slob = slice(*idx.slice_locs(df.index[5] + timedelta(seconds=30), df.index[15] - timedelta(seconds=30)))
        sliced = stacked[slob]
        expected = df[6:15].stack()
        tm.assert_almost_equal(sliced.values, expected.values)

    def test_slice_locs_with_type_mismatch(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        stacked = df.stack()
        idx = stacked.index
        with pytest.raises(TypeError, match='^Level type mismatch'):
            idx.slice_locs((1, 3))
        with pytest.raises(TypeError, match='^Level type mismatch'):
            idx.slice_locs(df.index[5] + timedelta(seconds=30), (5, 2))
        df = DataFrame(np.ones((5, 5)), index=Index([f'i-{i}' for i in range(5)], name='a'), columns=Index([f'i-{i}' for i in range(5)], name='a'))
        stacked = df.stack()
        idx = stacked.index
        with pytest.raises(TypeError, match='^Level type mismatch'):
            idx.slice_locs(timedelta(seconds=30))
        with pytest.raises(TypeError, match='^Level type mismatch'):
            idx.slice_locs(df.index[1], (16, 'a'))

    def test_slice_locs_not_sorted(self) -> None:
        index = MultiIndex(levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))], codes=[np.array([0, 0, 1, 2, 2, 2, 3, 3]), np.array([0, 1, 0, 0, 0, 1, 0, 1]), np.array([1, 0, 1, 1, 0, 0, 1, 0])])
        msg = '[Kk]ey length.*greater than MultiIndex lexsort depth'
        with pytest.raises(KeyError, match=msg):
            index.slice_locs((1, 0, 1), (2, 1, 0))
        sorted_index, _ = index.sortlevel(0)
        sorted_index.slice_locs((1, 0, 1), (2, 1, 0))

    def test_slice_locs_not_contained(self) -> None:
        index = MultiIndex(levels=[[0, 2, 4, 6], [0, 2, 4]], codes=[[0, 0, 0, 1, 1, 2, 3, 3, 3], [0, 1, 2, 1, 2, 2, 0, 1, 2]])
        result = index.slice_locs((1, 0), (5, 2))
        assert result == (3, 6)
        result = index.slice_locs(1, 5)
        assert result == (3, 6)
        result = index.slice_locs((2, 2), (5, 2))
        assert result == (3, 6)
        result = index.slice_locs(2, 5)
        assert result == (3, 6)
        result = index.slice_locs((1, 0), (6, 3))
        assert result == (3, 8)
        result = index.slice_locs(-1, 10)
        assert result == (0, len(index))

    @pytest.mark.parametrize('index_arr,expected,start_idx,end_idx', [([[np.nan, 'a', 'b'], ['c', 'd', 'e']], (0, 3), np.nan, None), ([[np.nan, 'a', 'b'], ['c', 'd', 'e']], (0, 3), np.nan, 'b'), ([[np.nan, 'a', 'b'], ['c', 'd', 'e']], (0, 3), np.nan, ('b', 'e')), ([['a', 'b', 'c'], ['d', np.nan, 'e']], (1, 3), ('b', np.nan), None), ([['a', 'b', 'c'], ['d', np.nan, 'e']], (1, 3), ('b', np.nan), 'c'), ([['a', 'b', 'c'], ['d', np.nan, 'e']], (1, 3), ('b', np.nan), ('c', 'e'))])
    def test_slice_locs_with_missing_value(self, index_arr: List[List[Any]], expected: Tuple[int, int], start_idx: Any, end_idx: Any) -> None:
        idx = MultiIndex.from_arrays(index_arr)
        result = idx.slice_locs(start=start_idx, end=end_idx)
        assert result == expected

class TestPutmask:

    def test_putmask_with_wrong_mask(self, idx: MultiIndex) -> None:
        msg = 'putmask: mask and data must be the same size'
        with pytest.raises(ValueError, match=msg):
            idx.putmask(np.ones(len(idx) + 1, np.bool_), 1)
        with pytest.raises(ValueError, match=msg):
            idx.putmask(np.ones(len(idx) - 1, np.bool_), 1)
        with pytest.raises(ValueError, match=msg):
            idx.putmask('foo', 1)

    def test_putmask_multiindex_other(self) -> None:
        left = MultiIndex.from_tuples([(np.nan, 6), (np.nan, 6), ('a', 4)])
        right = MultiIndex.from_tuples([('a', 1), ('a', 1), ('d', 1)])
        mask = np.array([True, True, False])
        result = left.putmask(mask, right)
        expected = MultiIndex.from_tuples([right[0], right[1], left[2]])
        tm.assert_index_equal(result, expected)

    def test_putmask_keep_dtype(self, any_numeric_ea_dtype: Any) -> None:
        midx = MultiIndex.from_arrays([pd.Series([1, 2, 3], dtype=any_numeric_ea_dtype), [10, 11, 12]])
        midx2 = MultiIndex.from_arrays([pd.Series([5, 6, 7], dtype=any_numeric_ea_dtype), [-1, -2, -3]])
        result = midx.putmask([True, False, False], midx2)
        expected = MultiIndex.from_arrays([pd.Series([5, 2, 3], dtype=any_numeric_ea_dtype), [-1, 11, 12]])
        tm.assert_index_equal(result, expected)

    def test_putmask_keep_dtype_shorter_value(self, any_numeric_ea_dtype: Any) -> None:
        midx = MultiIndex.from_arrays([pd.Series([1, 2, 3], dtype=any_numeric_ea_dtype), [10, 11, 12]])
        midx2 = MultiIndex.from_arrays([pd.Series([5], dtype=any_numeric_ea_dtype), [-1]])
        result = midx.putmask([True, False, False], midx2)
        expected = MultiIndex.from_arrays([pd.Series([5, 2, 3], dtype=any_numeric_ea_dtype), [-1, 11, 12]])
        tm.assert_index_equal(result, expected)

class TestGetIndexer:

    def test_get_indexer(self) -> None:
        major_axis = Index(np.arange(4))
        minor_axis = Index(np.arange(2))
        major_codes = np.array([0, 0, 1, 2, 2, 3, 3], dtype=np.intp)
        minor_codes = np.array([0, 1, 0, 0, 1, 0, 1], dtype=np.intp)
        index = MultiIndex(levels=[major_axis, minor_axis], codes=[major_codes, minor_codes])
        idx1 = index[:5]
        idx2 = index[[1, 3, 5]]
        r1 = idx1.get_indexer(idx2)
        tm.assert_almost_equal(r1, np.array([1, 3, -1], dtype=np.intp))
        r1 = idx2.get_indexer(idx1, method='pad')
        e1 = np.array([-1, 0, 0, 1, 1], dtype=np.intp)
        tm.assert_almost_equal(r1, e1)
        r2 = idx2.get_indexer(idx1[::-1], method='pad')
        tm.assert_almost_equal(r2, e1[::-1])
        rffill1 = idx2.get_indexer(idx1, method='ffill')
        tm.assert_almost_equal(r1, rffill1)
        r1 = idx2.get_indexer(idx1, method='backfill')
        e1 = np.array([0, 0, 1, 1, 2], dtype=np.intp)
        tm.assert_almost_equal(r1, e1)
        r2 = idx2.get_indexer(idx1[::-1], method='backfill')
        tm.assert_almost_equal(r2, e1[::-1])
        rbfill1 = idx2.get_indexer(idx1, method='bfill')
        tm.assert_almost_equal(r1, rbfill1)
        r1 = idx1.get_indexer(idx2.values)
        rexp1 = idx1.get_indexer(idx2)
        tm.assert_almost_equal(r1, rexp1)
        r1 = idx1.get_indexer([1, 2, 3])
        assert (r1 == [-1, -1, -1]).all()
        idx1 = Index(list(range(10)) + list(range(10)))
        idx2 = Index(list(range(20)))
        msg = 'Reindexing only valid with uniquely valued Index objects'
        with pytest.raises(InvalidIndexError, match=msg):
            idx1.get_indexer(idx2)

    def test_get_indexer_nearest(self) -> None:
        midx = MultiIndex.from_tuples([('a', 1), ('b', 2)])
        msg = "method='nearest' not implemented yet for MultiIndex; see GitHub issue 9365"
        with pytest.raises(NotImplementedError, match=msg):
            midx.get_indexer(['a'], method='nearest')
        msg = 'tolerance not implemented yet for MultiIndex'
        with pytest.raises(NotImplementedError, match=msg):
            midx.get_indexer(['a'], method='pad', tolerance=2)

    def test_get_indexer_categorical_time(self) -> None:
        midx = MultiIndex.from_product([Categorical(['a', 'b', 'c']), Categorical(date_range('2012-01-01', periods=3, freq='h'))])
        result = midx.get_indexer(midx)
        tm.assert_numpy_array_equal(result, np.arange(9, dtype=np.intp))

    @pytest.mark.parametrize('index_arr,labels,expected', [([[1, np.nan, 2], [3, 4, 5]], [1, np.nan, 2], np.array([-1, -1, -1], dtype=np.intp)), ([[1, np.nan, 2], [3, 4, 5]], [(np.nan, 4)], np.array([1], dtype=np.intp)), ([[1, 2, 3], [np.nan, 4, 5]], [(1, np.nan)], np.array([0], dtype=np.intp)), ([[1, 2, 3], [np.nan, 4, 5]], [np.nan, 4, 5], np.array([-1, -1, -1], dtype=np.intp))])
    def test_get_indexer_with_missing_value(self, index_arr: List[List[Any]], labels: List[Any], expected: np.ndarray) -> None:
        idx = MultiIndex.from_arrays(index_arr)
        result = idx.get_indexer(labels)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_methods(self) -> None:
        mult_idx_1 = MultiIndex.from_product([[-1, 0, 1], [0, 2, 3, 4]])
        mult_idx_2 = MultiIndex.from_product([[0], [1, 3, 4]])
        indexer = mult_idx_1.get_indexer(mult_idx_2)
        expected = np.array([-1, 6, 7], dtype=indexer.dtype)
        tm.assert_almost_equal(expected, indexer)
        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method='backfill')
        expected = np.array([5, 6, 7], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)
        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method='bfill')
        expected = np.array([5, 6, 7], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)
        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method='pad')
        expected = np.array([4, 6, 7], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)
        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method='ffill')
        expected = np.array([4, 6, 7], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)

    @pytest.mark.parametrize('method', ['pad', 'ffill', 'backfill', 'bfill', 'nearest'])
    def test_get_indexer_methods_raise_for_non_monotonic(self, method: str) -> None:
        mi = MultiIndex.from_arrays([[0, 4, 2], [0, 4, 2]])
        if method == 'nearest':
            err = NotImplementedError
            msg = 'not implemented yet for MultiIndex'
        else:
            err = ValueError
            msg = 'index must be monotonic increasing or decreasing'
        with pytest.raises(err, match=msg):
            mi.get_indexer([(1, 1)], method=method)

    def test_get_indexer_three_or_more_levels(self) -> None:
        mult_idx_1 = MultiIndex.from_product([[1, 3], [2, 4, 6], [5, 7]])
        mult_idx_2 = MultiIndex.from_tuples([(1, 1, 8), (1, 5, 9), (1, 6, 7), (2, 1, 6), (2, 7, 7), (2, 7, 8), (3, 6, 8)])
        assert mult_idx_1.is_monotonic_increasing
        assert mult_idx_1.is_unique
        assert mult_idx_2.is_monotonic_increasing
        assert mult_idx_2.is_unique
        assert mult_idx_2[0] < mult_idx_1[0]
        assert mult_idx_1[3] < mult_idx_2[1] < mult_idx_1[4]
        assert mult_idx_1[5] == mult_idx_2[2]
        assert mult_idx_1[5] < mult_idx_2[3] < mult_idx_1[6]
        assert mult_idx_1[5] < mult_idx_2[4] < mult_idx_1[6]
        assert mult_idx_1[5] < mult_idx_2[5] < mult_idx_1[6]
        assert mult_idx_1[-1] < mult_idx_2[6]
        indexer_no_fill = mult_idx_1.get_indexer(mult_idx_2)
        expected = np.array([-1, -1, 5, -1, -1, -1, -1], dtype=indexer_no_fill.dtype)
        tm.assert_almost_equal(expected, indexer_no_fill)
        indexer_backfilled = mult_idx_1.get_indexer(mult_idx_2, method='backfill')
        expected = np.array([0, 4, 5, 6, 6, 6, -1],
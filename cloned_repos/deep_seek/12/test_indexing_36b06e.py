import re
from typing import Any, List, Tuple, Union, cast

import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import NA, CategoricalIndex, DatetimeIndex, Index, Interval, IntervalIndex, MultiIndex, NaT, Timedelta, Timestamp, array, date_range, interval_range, isna, period_range, timedelta_range
import pandas._testing as tm

class TestGetItem:

    def test_getitem(self, closed: str) -> None:
        idx = IntervalIndex.from_arrays((0, 1, np.nan), (1, 2, np.nan), closed=closed)
        assert idx[0] == Interval(0.0, 1.0, closed=closed)
        assert idx[1] == Interval(1.0, 2.0, closed=closed)
        assert isna(idx[2])
        result = idx[0:1]
        expected = IntervalIndex.from_arrays((0.0,), (1.0,), closed=closed)
        tm.assert_index_equal(result, expected)
        result = idx[0:2]
        expected = IntervalIndex.from_arrays((0.0, 1), (1.0, 2.0), closed=closed)
        tm.assert_index_equal(result, expected)
        result = idx[1:3]
        expected = IntervalIndex.from_arrays((1.0, np.nan), (2.0, np.nan), closed=closed)
        tm.assert_index_equal(result, expected)

    def test_getitem_2d_deprecated(self) -> None:
        idx = IntervalIndex.from_breaks(range(11), closed='right')
        with pytest.raises(ValueError, match='multi-dimensional indexing not allowed'):
            idx[:, None]
        with pytest.raises(ValueError, match='multi-dimensional indexing not allowed'):
            idx[True]
        with pytest.raises(ValueError, match='multi-dimensional indexing not allowed'):
            idx[False]

class TestWhere:

    def test_where(self, listlike_box: Any) -> None:
        klass = listlike_box
        idx = IntervalIndex.from_breaks(range(11), closed='right')
        cond = [True] * len(idx)
        expected = idx
        result = expected.where(klass(cond))
        tm.assert_index_equal(result, expected)
        cond = [False] + [True] * len(idx[1:])
        expected = IntervalIndex([np.nan] + idx[1:].tolist())
        result = idx.where(klass(cond))
        tm.assert_index_equal(result, expected)

class TestTake:

    def test_take(self, closed: str) -> None:
        index = IntervalIndex.from_breaks(range(11), closed=closed)
        result = index.take(range(10))
        tm.assert_index_equal(result, index)
        result = index.take([0, 0, 1])
        expected = IntervalIndex.from_arrays([0, 0, 1], [1, 1, 2], closed=closed)
        tm.assert_index_equal(result, expected)

class TestGetLoc:

    @pytest.mark.parametrize('side', ['right', 'left', 'both', 'neither'])
    def test_get_loc_interval(self, closed: str, side: str) -> None:
        idx = IntervalIndex.from_tuples([(0, 1), (2, 3)], closed=closed)
        for bound in [[0, 1], [1, 2], [2, 3], [3, 4], [0, 2], [2.5, 3], [-1, 4]]:
            msg = re.escape(f"Interval({bound[0]}, {bound[1]}, closed='{side}')")
            if closed == side:
                if bound == [0, 1]:
                    assert idx.get_loc(Interval(0, 1, closed=side)) == 0
                elif bound == [2, 3]:
                    assert idx.get_loc(Interval(2, 3, closed=side)) == 1
                else:
                    with pytest.raises(KeyError, match=msg):
                        idx.get_loc(Interval(*bound, closed=side))
            else:
                with pytest.raises(KeyError, match=msg):
                    idx.get_loc(Interval(*bound, closed=side))

    @pytest.mark.parametrize('scalar', [-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    def test_get_loc_scalar(self, closed: str, scalar: float) -> None:
        correct = {'right': {0.5: 0, 1: 0, 2.5: 1, 3: 1}, 'left': {0: 0, 0.5: 0, 2: 1, 2.5: 1}, 'both': {0: 0, 0.5: 0, 1: 0, 2: 1, 2.5: 1, 3: 1}, 'neither': {0.5: 0, 2.5: 1}}
        idx = IntervalIndex.from_tuples([(0, 1), (2, 3)], closed=closed)
        if scalar in correct[closed].keys():
            assert idx.get_loc(scalar) == correct[closed][scalar]
        else:
            with pytest.raises(KeyError, match=str(scalar)):
                idx.get_loc(scalar)

    @pytest.mark.parametrize('scalar', [-1, 0, 0.5, 3, 4.5, 5, 6])
    def test_get_loc_length_one_scalar(self, scalar: float, closed: str) -> None:
        index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
        if scalar in index[0]:
            result = index.get_loc(scalar)
            assert result == 0
        else:
            with pytest.raises(KeyError, match=str(scalar)):
                index.get_loc(scalar)

    @pytest.mark.parametrize('left, right', [(0, 5), (-1, 4), (-1, 6), (6, 7)])
    def test_get_loc_length_one_interval(self, left: float, right: float, closed: str, other_closed: str) -> None:
        index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
        interval = Interval(left, right, closed=other_closed)
        if interval == index[0]:
            result = index.get_loc(interval)
            assert result == 0
        else:
            with pytest.raises(KeyError, match=re.escape(f"Interval({left}, {right}, closed='{other_closed}')")):
                index.get_loc(interval)

    @pytest.mark.parametrize('breaks', [date_range('20180101', periods=4), date_range('20180101', periods=4, tz='US/Eastern'), timedelta_range('0 days', periods=4)], ids=lambda x: str(x.dtype))
    def test_get_loc_datetimelike_nonoverlapping(self, breaks: Any) -> None:
        index = IntervalIndex.from_breaks(breaks)
        value = index[0].mid
        result = index.get_loc(value)
        expected = 0
        assert result == expected
        interval = Interval(index[0].left, index[0].right)
        result = index.get_loc(interval)
        expected = 0
        assert result == expected

    @pytest.mark.parametrize('arrays', [(date_range('20180101', periods=4), date_range('20180103', periods=4)), (date_range('20180101', periods=4, tz='US/Eastern'), date_range('20180103', periods=4, tz='US/Eastern')), (timedelta_range('0 days', periods=4), timedelta_range('2 days', periods=4))], ids=lambda x: str(x[0].dtype))
    def test_get_loc_datetimelike_overlapping(self, arrays: Any) -> None:
        index = IntervalIndex.from_arrays(*arrays)
        value = index[0].mid + Timedelta('12 hours')
        result = index.get_loc(value)
        expected = slice(0, 2, None)
        assert result == expected
        interval = Interval(index[0].left, index[0].right)
        result = index.get_loc(interval)
        expected = 0
        assert result == expected

    @pytest.mark.parametrize('values', [date_range('2018-01-04', periods=4, freq='-1D'), date_range('2018-01-04', periods=4, freq='-1D', tz='US/Eastern'), timedelta_range('3 days', periods=4, freq='-1D'), np.arange(3.0, -1.0, -1.0), np.arange(3, -1, -1)], ids=lambda x: str(x.dtype))
    def test_get_loc_decreasing(self, values: Any) -> None:
        index = IntervalIndex.from_arrays(values[1:], values[:-1])
        result = index.get_loc(index[0])
        expected = 0
        assert result == expected

    @pytest.mark.parametrize('key', [[5], (2, 3)])
    def test_get_loc_non_scalar_errors(self, key: Any) -> None:
        idx = IntervalIndex.from_tuples([(1, 3), (2, 4), (3, 5), (7, 10), (3, 10)])
        msg = str(key)
        with pytest.raises(InvalidIndexError, match=msg):
            idx.get_loc(key)

    def test_get_indexer_with_nans(self) -> None:
        index = IntervalIndex([np.nan, Interval(1, 2), np.nan])
        expected = np.array([True, False, True])
        for key in [None, np.nan, NA]:
            assert key in index
            result = index.get_loc(key)
            tm.assert_numpy_array_equal(result, expected)
        for key in [NaT, np.timedelta64('NaT', 'ns'), np.datetime64('NaT', 'ns')]:
            with pytest.raises(KeyError, match=str(key)):
                index.get_loc(key)

class TestGetIndexer:

    @pytest.mark.parametrize('query, expected', [([Interval(2, 4, closed='right')], [1]), ([Interval(2, 4, closed='left')], [-1]), ([Interval(2, 4, closed='both')], [-1]), ([Interval(2, 4, closed='neither')], [-1]), ([Interval(1, 4, closed='right')], [-1]), ([Interval(0, 4, closed='right')], [-1]), ([Interval(0.5, 1.5, closed='right')], [-1]), ([Interval(2, 4, closed='right'), Interval(0, 1, closed='right')], [1, -1]), ([Interval(2, 4, closed='right'), Interval(2, 4, closed='right')], [1, 1]), ([Interval(5, 7, closed='right'), Interval(2, 4, closed='right')], [2, 1]), ([Interval(2, 4, closed='right'), Interval(2, 4, closed='left')], [1, -1])])
    def test_get_indexer_with_interval(self, query: Any, expected: List[int]) -> None:
        tuples = [(0, 2), (2, 4), (5, 7)]
        index = IntervalIndex.from_tuples(tuples, closed='right')
        result = index.get_indexer(query)
        expected = np.array(expected, dtype='intp')
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('query, expected', [([-0.5], [-1]), ([0], [-1]), ([0.5], [0]), ([1], [0]), ([1.5], [1]), ([2], [1]), ([2.5], [-1]), ([3], [-1]), ([3.5], [2]), ([4], [2]), ([4.5], [-1]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, -1]), ([1, 2, 3, 4], [0, 1, -1, 2]), ([1, 2, 3, 4, 2], [0, 1, -1, 2, 1])])
    def test_get_indexer_with_int_and_float(self, query: Any, expected: List[int]) -> None:
        tuples = [(0, 1), (1, 2), (3, 4)]
        index = IntervalIndex.from_tuples(tuples, closed='right')
        result = index.get_indexer(query)
        expected = np.array(expected, dtype='intp')
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('item', [[3], np.arange(0.5, 5, 0.5)])
    def test_get_indexer_length_one(self, item: Any, closed: str) -> None:
        index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
        result = index.get_indexer(item)
        expected = np.array([0] * len(item), dtype='intp')
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('size', [1, 5])
    def test_get_indexer_length_one_interval(self, size: int, closed: str) -> None:
        index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
        result = index.get_indexer([Interval(0, 5, closed)] * size)
        expected = np.array([0] * size, dtype='intp')
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('target', [IntervalIndex.from_tuples([(7, 8), (1, 2), (3, 4), (0, 1)]), IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4), np.nan]), IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)], closed='both'), [-1, 0, 0.5, 1, 2, 2.5, np.nan], ['foo', 'foo', 'bar', 'baz']])
    def test_get_indexer_categorical(self, target: Any, ordered: bool) -> None:
        index = IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)])
        categorical_target = CategoricalIndex(target, ordered=ordered)
        result = index.get_indexer(categorical_target)
        expected = index.get_indexer(target)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:invalid value encountered in cast:RuntimeWarning')
    def test_get_indexer_categorical_with_nans(self) -> None:
        ii = IntervalIndex.from_breaks(range(5))
        ii2 = ii.append(IntervalIndex([np.nan]))
        ci2 = CategoricalIndex(ii2)
        result = ii2.get_indexer(ci2)
        expected = np.arange(5, dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        result = ii2[1:].get_indexer(ci2[::-1])
        expected = np.array([3, 2, 1, 0, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        result = ii2.get_indexer(ci2.append(ci2))
        expected = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_datetime(self) -> None:
        ii = IntervalIndex.from_breaks(date_range('2018-01-01', periods=4))
        target = DatetimeIndex(['2018-01-02'], dtype='M8[ns]')
        result = ii.get_indexer(target)
        expected = np.array([0], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        result = ii.get_indexer(target.astype(str))
        tm.assert_numpy_array_equal(result, expected)
        result = ii.get_indexer(target.asi8)
        expected = np.array([-1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('tuples, closed', [([(0, 2), (1, 3), (3, 4)], 'neither'), ([(0, 5), (1, 4), (6, 7)], 'left'), ([(0, 1), (0, 1), (1, 2)], 'right'), ([(0, 1), (2, 3), (3, 4)], 'both')])
    def test_get_indexer_errors(self, tuples: List[Tuple[int, int]], closed: str) -> None:
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        msg = 'cannot handle overlapping indices; use IntervalIndex.get_indexer_non_unique'
        with pytest.raises(InvalidIndexError, match=msg):
            index.get_indexer([0, 2])

    @pytest.mark.parametrize('query, expected', [([-0.5], ([-1], [0])), ([0], ([0], [])), ([0.5], ([0], [])), ([1], ([0, 1], [])), ([1.5], ([0, 1], [])), ([2], ([0, 1, 2], [])), ([2.5], ([1, 2], [])), ([3], ([2], [])), ([3.5], ([2
import re
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

class TestWhere:
    def test_where(self) -> None:
        klass: type = ...
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

class TestGetIndexer:
    @pytest.mark.parametrize('query, expected', ...)
    def test_get_indexer_with_interval(self, query, expected) -> None:
        tuples = [(0, 2), (1, 3), (3, 4)]
        index = IntervalIndex.from_tuples(tuples, closed='right')
        result = index.get_indexer(query)
        expected = np.array(expected, dtype='intp')
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('query, expected', ...)
    def test_get_indexer_with_int_and_float(self, query, expected) -> None:
        tuples = [(0, 2.5), (1, 3), (2, 4)]
        index = IntervalIndex.from_tuples(tuples, closed='left')
        result = index.get_indexer(query)
        expected = np.array(expected, dtype='intp')
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('query, expected', ...)
    def test_get_indexer_errors(self, query, expected) -> None:
        index = IntervalIndex.from_tuples([(0, 2), (1, 3), (3, 4)])
        msg = 'cannot handle overlapping indices; use IntervalIndex.get_indexer_non_unique'
        with pytest.raises(InvalidIndexError, match=msg):
            index.get_indexer([0, 2])

class TestSliceLocs:
    @pytest.mark.parametrize('query', ...)
    @pytest.mark.parametrize('tuples', ...)
    def test_slice_locs_with_interval(self, query, tuples) -> None:
        index = IntervalIndex.from_tuples(tuples)
        result = index.slice_locs(query[0], query[1])
        expected = (0, 3)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('query', ...)
    @pytest.mark.parametrize('tuples', ...)
    def test_slice_locs_with_ints_and_floats_errors(self, query, tuples) -> None:
        index = IntervalIndex.from_tuples(tuples)
        with pytest.raises(KeyError, match="'can only get slices from an IntervalIndex if bounds are non-overlapping and all monotonic increasing or decreasing'"):
            index.slice_locs(query[0], query[1])

class TestPutmask:
    @pytest.mark.parametrize('tz', ...)
    def test_putmask_dt64(self, tz) -> None:
        dti = date_range('2016-01-01', periods=9, tz=tz)
        idx = IntervalIndex.from_breaks(dti)
        mask = np.zeros(idx.shape, dtype=bool)
        mask[0:3] = True
        result = idx.putmask(mask, idx[-1])
        expected = IntervalIndex([idx[-1]] * 3 + list(idx[3:]))
        tm.assert_index_equal(result, expected)

class TestContains:
    def test_contains_dunder(self) -> None:
        index = IntervalIndex.from_arrays([0, 1], [1, 2], closed='right')
        assert 0 not in index
        assert 1 not in index
        assert 2 not in index
        assert Interval(0, 1, closed='right') in index
        assert Interval(0, 2, closed='right') not in index
        assert Interval(0, 0.5, closed='right') not in index
        assert Interval(3, 5, closed='right') not in index
        assert Interval(-1, 0, closed='left') not in index
        assert Interval(0, 1, closed='left') not in index
        assert Interval(0, 1, closed='both') not in index

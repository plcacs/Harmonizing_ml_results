from itertools import permutations
import re
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    Interval,
    IntervalIndex,
    Timedelta,
    Timestamp,
    date_range,
    interval_range,
    isna,
    notna,
    timedelta_range,
)
import pandas._testing as tm
import pandas.core.common as com


class TestIntervalIndex:
    index: IntervalIndex = IntervalIndex.from_arrays([0, 1], [1, 2])

    def create_index(self, closed: str = "right") -> IntervalIndex:
        return IntervalIndex.from_breaks(range(11), closed=closed)

    def create_index_with_nan(self, closed: str = "right") -> IntervalIndex:
        mask: List[bool] = [True, False] + [True] * 8
        return IntervalIndex.from_arrays(
            np.where(mask, np.arange(10), np.nan),
            np.where(mask, np.arange(1, 11), np.nan),
            closed=closed,
        )

    def test_properties(self, closed: str) -> None:
        index: IntervalIndex = self.create_index(closed=closed)
        assert len(index) == 10
        assert index.size == 10
        assert index.shape == (10,)

        tm.assert_index_equal(index.left, Index(np.arange(10, dtype=np.int64)))
        tm.assert_index_equal(index.right, Index(np.arange(1, 11, dtype=np.int64)))
        tm.assert_index_equal(index.mid, Index(np.arange(0.5, 10.5, dtype=np.float64)))

        assert index.closed == closed

        ivs: List[Interval] = [
            Interval(left, right, closed)
            for left, right in zip(range(10), range(1, 11))
        ]
        expected: np.ndarray = np.array(ivs, dtype=object)
        tm.assert_numpy_array_equal(np.asarray(index), expected)

        # with nans
        index = self.create_index_with_nan(closed=closed)
        assert len(index) == 10
        assert index.size == 10
        assert index.shape == (10,)

        expected_left: Index = Index([0, np.nan, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_right: Index = expected_left + 1
        expected_mid: Index = expected_left + 0.5
        tm.assert_index_equal(index.left, expected_left)
        tm.assert_index_equal(index.right, expected_right)
        tm.assert_index_equal(index.mid, expected_mid)

        assert index.closed == closed

        ivs = [
            Interval(left, right, closed) if notna(left) else np.nan
            for left, right in zip(expected_left, expected_right)
        ]
        expected = np.array(ivs, dtype=object)
        tm.assert_numpy_array_equal(np.asarray(index), expected)

    @pytest.mark.parametrize(
        "breaks",
        [
            [1, 1, 2, 5, 15, 53, 217, 1014, 5335, 31240, 201608],
            [-np.inf, -100, -10, 0.5, 1, 1.5, 3.8, 101, 202, np.inf],
            date_range("2017-01-01", "2017-01-04"),
            pytest.param(
                date_range("2017-01-01", "2017-01-04", unit="s"),
                marks=pytest.mark.xfail(reason="mismatched result unit"),
            ),
            pd.to_timedelta(["1ns", "2ms", "3s", "4min", "5h", "6D"]),
        ],
    )
    def test_length(self, closed: str, breaks: List[Union[int, float, Timestamp, Timedelta]]) -> None:
        # GH 18789
        index: IntervalIndex = IntervalIndex.from_breaks(breaks, closed=closed)
        result: Index = index.length
        expected: Index = Index(iv.length for iv in index)
        tm.assert_index_equal(result, expected)

        # with NA
        index = index.insert(1, np.nan)
        result = index.length
        expected = Index(iv.length if notna(iv) else iv for iv in index)
        tm.assert_index_equal(result, expected)

    def test_with_nans(self, closed: str) -> None:
        index: IntervalIndex = self.create_index(closed=closed)
        assert index.hasnans is False

        result: np.ndarray = index.isna()
        expected: np.ndarray = np.zeros(len(index), dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        result = index.notna()
        expected = np.ones(len(index), dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        index = self.create_index_with_nan(closed=closed)
        assert index.hasnans is True

        result = index.isna()
        expected = np.array([False, True] + [False] * (len(index) - 2))
        tm.assert_numpy_array_equal(result, expected)

        result = index.notna()
        expected = np.array([True, False] + [True] * (len(index) - 2))
        tm.assert_numpy_array_equal(result, expected)

    def test_copy(self, closed: str) -> None:
        expected: IntervalIndex = self.create_index(closed=closed)

        result: IntervalIndex = expected.copy()
        assert result.equals(expected)

        result = expected.copy(deep=True)
        assert result.equals(expected)
        assert result.left is not expected.left

    def test_ensure_copied_data(self, closed: str) -> None:
        # exercise the copy flag in the constructor

        # not copying
        index: IntervalIndex = self.create_index(closed=closed)
        result: IntervalIndex = IntervalIndex(index, copy=False)
        tm.assert_numpy_array_equal(
            index.left.values, result.left.values, check_same="same"
        )
        tm.assert_numpy_array_equal(
            index.right.values, result.right.values, check_same="same"
        )

        # by-definition make a copy
        result = IntervalIndex(np.array(index), copy=False)
        tm.assert_numpy_array_equal(
            index.left.values, result.left.values, check_same="copy"
        )
        tm.assert_numpy_array_equal(
            index.right.values, result.right.values, check_same="copy"
        )

    def test_delete(self, closed: str) -> None:
        breaks: np.ndarray = np.arange(1, 11, dtype=np.int64)
        expected: IntervalIndex = IntervalIndex.from_breaks(breaks, closed=closed)
        result: IntervalIndex = self.create_index(closed=closed).delete(0)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "data",
        [
            interval_range(0, periods=10, closed="neither"),
            interval_range(1.7, periods=8, freq=2.5, closed="both"),
            interval_range(Timestamp("20170101"), periods=12, closed="left"),
            interval_range(Timedelta("1 day"), periods=6, closed="right"),
        ],
    )
    def test_insert(self, data: IntervalIndex) -> None:
        item: Interval = data[0]
        idx_item: IntervalIndex = IntervalIndex([item])

        # start
        expected: IntervalIndex = idx_item.append(data)
        result: IntervalIndex = data.insert(0, item)
        tm.assert_index_equal(result, expected)

        # end
        expected = data.append(idx_item)
        result = data.insert(len(data), item)
        tm.assert_index_equal(result, expected)

        # mid
        expected = data[:3].append(idx_item).append(data[3:])
        result = data.insert(3, item)
        tm.assert_index_equal(result, expected)

        # invalid type
        res: IntervalIndex = data.insert(1, "foo")
        expected = data.astype(object).insert(1, "foo")
        tm.assert_index_equal(res, expected)

        msg = "can only insert Interval objects and NA into an IntervalArray"
        with pytest.raises(TypeError, match=msg):
            data._data.insert(1, "foo")

        # invalid closed
        msg = "'value.closed' is 'left', expected 'right'."
        for closed in {"left", "right", "both", "neither"} - {item.closed}:
            msg = f"'value.closed' is '{closed}', expected '{item.closed}'."
            bad_item: Interval = Interval(item.left, item.right, closed=closed)
            res = data.insert(1, bad_item)
            expected = data.astype(object).insert(1, bad_item)
            tm.assert_index_equal(res, expected)
            with pytest.raises(ValueError, match=msg):
                data._data.insert(1, bad_item)

        # GH 18295 (test missing)
        na_idx: IntervalIndex = IntervalIndex([np.nan], closed=data.closed)
        for na in [np.nan, None, pd.NA]:
            expected = data[:1].append(na_idx).append(data[1:])
            result = data.insert(1, na)
            tm.assert_index_equal(result, expected)

        if data.left.dtype.kind not in ["m", "M"]:
            # trying to insert pd.NaT into a numeric-dtyped Index should cast
            expected = data.astype(object).insert(1, pd.NaT)

            msg = "can only insert Interval objects and NA into an IntervalArray"
            with pytest.raises(TypeError, match=msg):
                data._data.insert(1, pd.NaT)

        result = data.insert(1, pd.NaT)
        tm.assert_index_equal(result, expected)

    def test_is_unique_interval(self, closed: str) -> None:
        """
        Interval specific tests for is_unique in addition to base class tests
        """
        # unique overlapping - distinct endpoints
        idx: IntervalIndex = IntervalIndex.from_tuples([(0, 1), (0.5, 1.5)], closed=closed)
        assert idx.is_unique is True

        # unique overlapping - shared endpoints
        idx = IntervalIndex.from_tuples([(1, 2), (1, 3), (2, 3)], closed=closed)
        assert idx.is_unique is True

        # unique nested
        idx = IntervalIndex.from_tuples([(-1, 1), (-2, 2)], closed=closed)
        assert idx.is_unique is True

        # unique NaN
        idx = IntervalIndex.from_tuples([(np.nan, np.nan)], closed=closed)
        assert idx.is_unique is True

        # non-unique NaN
        idx = IntervalIndex.from_tuples(
            [(np.nan, np.nan), (np.nan, np.nan)], closed=closed
        )
        assert idx.is_unique is False

    def test_monotonic(self, closed: str) -> None:
        # increasing non-overlapping
        idx: IntervalIndex = IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)], closed=closed)
        assert idx.is_monotonic_increasing is True
        assert idx._is_strictly_monotonic_increasing is True
        assert idx.is_monotonic_decreasing is False
        assert idx._is_strictly_monotonic_decreasing is False

        # decreasing non-overlapping
        idx = IntervalIndex.from_tuples([(4, 5), (2, 3), (1, 2)], closed=closed)
        assert idx.is_monotonic_increasing is False
        assert idx._is_strictly_monotonic_increasing is False
        assert idx.is_monotonic_decreasing is True
        assert idx._is_strictly_monotonic_decreasing is True

        # unordered non-overlapping
        idx = IntervalIndex.from_tuples([(0, 1), (4, 5), (2, 3)], closed=closed)
        assert idx.is_monotonic_increasing is False
        assert idx._is_strictly_monotonic_increasing is False
        assert idx.is_monotonic_decreasing is False
        assert idx._is_strictly_monotonic_decreasing is False

        # increasing overlapping
        idx = IntervalIndex.from_tuples([(0, 2), (0.5, 2.5), (1, 3)], closed=closed)
        assert idx.is_monotonic_increasing is True
        assert idx._is_strictly_monotonic_increasing is True
        assert idx.is_monotonic_decreasing is False
        assert idx._is_strictly_monotonic_decreasing is False

        # decreasing overlapping
        idx = IntervalIndex.from_tuples([(1, 3), (0.5, 2.5), (0, 2)], closed=closed)
        assert idx.is_monotonic_increasing is False
        assert idx._is_strictly_monotonic_increasing is False
        assert idx.is_monotonic_decreasing is True
        assert idx._is_strictly_monotonic_decreasing is True

        # unordered overlapping
        idx = IntervalIndex.from_tuples([(0.5, 2.5), (0, 2), (1, 3)], closed=closed)
        assert idx.is_monotonic_increasing is False
        assert idx._is_strictly_monotonic_increasing is False
        assert idx.is_monotonic_decreasing is False
        assert idx._is_strictly_monotonic_decreasing is False

        # increasing overlapping shared endpoints
        idx = IntervalIndex.from_tuples([(1, 2), (1, 3), (2, 3)], closed=closed)
        assert idx.is_monotonic_increasing is True
        assert idx._is_strictly_monotonic_increasing is True
        assert idx.is_monotonic_decreasing is False
        assert idx._is_strictly_monotonic_decreasing is False

        # decreasing overlapping shared endpoints
        idx = IntervalIndex.from_tuples([(2, 3), (1, 3), (1, 2)], closed=closed)
        assert idx.is_monotonic_increasing is False
        assert idx._is_strictly_monotonic_increasing is False
        assert idx.is_monotonic_decreasing is True
        assert idx._is_strictly_monotonic_decreasing is True

        # stationary
        idx = IntervalIndex.from_tuples([(0, 1), (0, 1)], closed=closed)
        assert idx.is_monotonic_increasing is True
        assert idx._is_strictly_monotonic_increasing is False
        assert idx.is_monotonic_decreasing is True
        assert idx._is_strictly_monotonic_decreasing is False

        # empty
        idx = IntervalIndex([], closed=closed)
        assert idx.is_monotonic_increasing is True
        assert idx._is_strictly_monotonic_increasing is True
        assert idx.is_monotonic_decreasing is True
        assert idx._is_strictly_monotonic_decreasing is True

    def test_is_monotonic_with_nans(self) -> None:
        # GH#41831
        index: IntervalIndex = IntervalIndex([np.nan, np.nan])

        assert not index.is_monotonic_increasing
        assert not index._is_strictly_monotonic_increasing
        assert not index.is_monotonic_increasing
        assert not index._is_strictly_monotonic_decreasing
        assert not index.is_monotonic_decreasing

    @pytest.mark.parametrize(
        "breaks",
        [
            date_range("20180101", periods=4),
            date_range("20180101", periods=4, tz="US/Eastern"),
            timedelta_range("0 days", periods=4),
        ],
        ids=lambda x: str(x.dtype),
    )
    def test_maybe_convert_i8(self, breaks: Union[pd.DatetimeIndex, pd.TimedeltaIndex]) -> None:
        # GH 20636
        index: IntervalIndex = IntervalIndex.from_breaks(breaks)

        # intervalindex
        result: IntervalIndex = index._maybe_convert_i8(index)
        expected: IntervalIndex = IntervalIndex.from_breaks(breaks.asi8)
        tm.assert_index_equal(result, expected)

        # interval
        interval: Interval = Interval(breaks[0], breaks[1])
        result: Interval = index._maybe_convert_i8(interval)
        expected: Interval = Interval(breaks[0]._value, breaks[1]._value)
        assert result == expected

        # datetimelike index
        result: Index = index._maybe_convert_i8(breaks)
        expected: Index = Index(breaks.asi8)
        tm.assert_index_equal(result, expected)

        # datetimelike scalar
        result: int = index._maybe_convert_i8(breaks[0])
        expected: int = breaks[0]._value
        assert result == expected

        # list-like of datetimelike scalars
        result: Index = index._maybe_convert_i8(list(breaks))
        expected: Index = Index(breaks.asi8)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "breaks",
        [date_range("2018-01-01", periods=5), timedelta_range("0 days", periods=5)],
    )
    def test_maybe_convert_i8_nat(self, breaks: Union[pd.DatetimeIndex, pd.TimedeltaIndex]) -> None:
        # GH 20636
        index: IntervalIndex = IntervalIndex.from_breaks(breaks)

        to_convert: Union[pd.DatetimeIndex, pd.TimedeltaIndex] = breaks._constructor([pd.NaT] * 3).as_unit("ns")
        expected: Index = Index([np.nan] * 3, dtype=np.float64)
        result: Index = index._maybe_convert_i8(to_convert)
        tm.assert_index_equal(result, expected)

        to_convert = to_convert.insert(0, breaks[0])
        expected = expected.insert(0, float(breaks[0]._value))
        result = index._maybe_convert_i8(to_convert)
        tm.assert_index_equal(result, expected)

    @pytest.mark.
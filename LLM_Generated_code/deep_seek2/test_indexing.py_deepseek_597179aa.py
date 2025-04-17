from collections import namedtuple
from datetime import timedelta
import re
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pytest

from pandas._libs import index as libindex
from pandas.errors import InvalidIndexError

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    date_range,
)
import pandas._testing as tm


class TestSliceLocs:
    def test_slice_locs_partial(self, idx: MultiIndex) -> None:
        sorted_idx, _ = idx.sortlevel(0)

        result = sorted_idx.slice_locs(("foo", "two"), ("qux", "one"))
        assert result == (1, 5)

        result = sorted_idx.slice_locs(None, ("qux", "one"))
        assert result == (0, 5)

        result = sorted_idx.slice_locs(("foo", "two"), None)
        assert result == (1, len(sorted_idx))

        result = sorted_idx.slice_locs("bar", "baz")
        assert result == (2, 4)

    def test_slice_locs(self) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((50, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=50, freq="B"),
        )
        stacked = df.stack()
        idx = stacked.index

        slob = slice(*idx.slice_locs(df.index[5], df.index[15]))
        sliced = stacked[slob]
        expected = df[5:16].stack()
        tm.assert_almost_equal(sliced.values, expected.values)

        slob = slice(
            *idx.slice_locs(
                df.index[5] + timedelta(seconds=30),
                df.index[15] - timedelta(seconds=30),
            )
        )
        sliced = stacked[slob]
        expected = df[6:15].stack()
        tm.assert_almost_equal(sliced.values, expected.values)

    def test_slice_locs_with_type_mismatch(self) -> None:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        stacked = df.stack()
        idx = stacked.index
        with pytest.raises(TypeError, match="^Level type mismatch"):
            idx.slice_locs((1, 3))
        with pytest.raises(TypeError, match="^Level type mismatch"):
            idx.slice_locs(df.index[5] + timedelta(seconds=30), (5, 2))
        df = DataFrame(
            np.ones((5, 5)),
            index=Index([f"i-{i}" for i in range(5)], name="a"),
            columns=Index([f"i-{i}" for i in range(5)], name="a"),
        )
        stacked = df.stack()
        idx = stacked.index
        with pytest.raises(TypeError, match="^Level type mismatch"):
            idx.slice_locs(timedelta(seconds=30))
        # TODO: Try creating a UnicodeDecodeError in exception message
        with pytest.raises(TypeError, match="^Level type mismatch"):
            idx.slice_locs(df.index[1], (16, "a"))

    def test_slice_locs_not_sorted(self) -> None:
        index = MultiIndex(
            levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))],
            codes=[
                np.array([0, 0, 1, 2, 2, 2, 3, 3]),
                np.array([0, 1, 0, 0, 0, 1, 0, 1]),
                np.array([1, 0, 1, 1, 0, 0, 1, 0]),
            ],
        )
        msg = "[Kk]ey length.*greater than MultiIndex lexsort depth"
        with pytest.raises(KeyError, match=msg):
            index.slice_locs((1, 0, 1), (2, 1, 0))

        # works
        sorted_index, _ = index.sortlevel(0)
        # should there be a test case here???
        sorted_index.slice_locs((1, 0, 1), (2, 1, 0))

    def test_slice_locs_not_contained(self) -> None:
        # some searchsorted action

        index = MultiIndex(
            levels=[[0, 2, 4, 6], [0, 2, 4]],
            codes=[[0, 0, 0, 1, 1, 2, 3, 3, 3], [0, 1, 2, 1, 2, 2, 0, 1, 2]],
        )

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

    @pytest.mark.parametrize(
        "index_arr,expected,start_idx,end_idx",
        [
            ([[np.nan, "a", "b"], ["c", "d", "e"]], (0, 3), np.nan, None),
            ([[np.nan, "a", "b"], ["c", "d", "e"]], (0, 3), np.nan, "b"),
            ([[np.nan, "a", "b"], ["c", "d", "e"]], (0, 3), np.nan, ("b", "e")),
            ([["a", "b", "c"], ["d", np.nan, "e"]], (1, 3), ("b", np.nan), None),
            ([["a", "b", "c"], ["d", np.nan, "e"]], (1, 3), ("b", np.nan), "c"),
            ([["a", "b", "c"], ["d", np.nan, "e"]], (1, 3), ("b", np.nan), ("c", "e")),
        ],
    )
    def test_slice_locs_with_missing_value(
        self, index_arr: List[List[Union[str, float]]], expected: Tuple[int, int], start_idx: Union[float, Tuple[float, str]], end_idx: Union[str, Tuple[str, str]]
    ) -> None:
        # issue 19132
        idx = MultiIndex.from_arrays(index_arr)
        result = idx.slice_locs(start=start_idx, end=end_idx)
        assert result == expected


class TestPutmask:
    def test_putmask_with_wrong_mask(self, idx: MultiIndex) -> None:
        # GH18368

        msg = "putmask: mask and data must be the same size"
        with pytest.raises(ValueError, match=msg):
            idx.putmask(np.ones(len(idx) + 1, np.bool_), 1)

        with pytest.raises(ValueError, match=msg):
            idx.putmask(np.ones(len(idx) - 1, np.bool_), 1)

        with pytest.raises(ValueError, match=msg):
            idx.putmask("foo", 1)

    def test_putmask_multiindex_other(self) -> None:
        # GH#43212 `value` is also a MultiIndex

        left = MultiIndex.from_tuples([(np.nan, 6), (np.nan, 6), ("a", 4)])
        right = MultiIndex.from_tuples([("a", 1), ("a", 1), ("d", 1)])
        mask = np.array([True, True, False])

        result = left.putmask(mask, right)

        expected = MultiIndex.from_tuples([right[0], right[1], left[2]])
        tm.assert_index_equal(result, expected)

    def test_putmask_keep_dtype(self, any_numeric_ea_dtype: str) -> None:
        # GH#49830
        midx = MultiIndex.from_arrays(
            [pd.Series([1, 2, 3], dtype=any_numeric_ea_dtype), [10, 11, 12]]
        )
        midx2 = MultiIndex.from_arrays(
            [pd.Series([5, 6, 7], dtype=any_numeric_ea_dtype), [-1, -2, -3]]
        )
        result = midx.putmask([True, False, False], midx2)
        expected = MultiIndex.from_arrays(
            [pd.Series([5, 2, 3], dtype=any_numeric_ea_dtype), [-1, 11, 12]]
        )
        tm.assert_index_equal(result, expected)

    def test_putmask_keep_dtype_shorter_value(self, any_numeric_ea_dtype: str) -> None:
        # GH#49830
        midx = MultiIndex.from_arrays(
            [pd.Series([1, 2, 3], dtype=any_numeric_ea_dtype), [10, 11, 12]]
        )
        midx2 = MultiIndex.from_arrays(
            [pd.Series([5], dtype=any_numeric_ea_dtype), [-1]]
        )
        result = midx.putmask([True, False, False], midx2)
        expected = MultiIndex.from_arrays(
            [pd.Series([5, 2, 3], dtype=any_numeric_ea_dtype), [-1, 11, 12]]
        )
        tm.assert_index_equal(result, expected)


class TestGetIndexer:
    def test_get_indexer(self) -> None:
        major_axis = Index(np.arange(4))
        minor_axis = Index(np.arange(2))

        major_codes = np.array([0, 0, 1, 2, 2, 3, 3], dtype=np.intp)
        minor_codes = np.array([0, 1, 0, 0, 0, 1, 0, 1], dtype=np.intp)

        index = MultiIndex(
            levels=[major_axis, minor_axis], codes=[major_codes, minor_codes]
        )
        idx1 = index[:5]
        idx2 = index[[1, 3, 5]]

        r1 = idx1.get_indexer(idx2)
        tm.assert_almost_equal(r1, np.array([1, 3, -1], dtype=np.intp))

        r1 = idx2.get_indexer(idx1, method="pad")
        e1 = np.array([-1, 0, 0, 1, 1], dtype=np.intp)
        tm.assert_almost_equal(r1, e1)

        r2 = idx2.get_indexer(idx1[::-1], method="pad")
        tm.assert_almost_equal(r2, e1[::-1])

        rffill1 = idx2.get_indexer(idx1, method="ffill")
        tm.assert_almost_equal(r1, rffill1)

        r1 = idx2.get_indexer(idx1, method="backfill")
        e1 = np.array([0, 0, 1, 1, 2], dtype=np.intp)
        tm.assert_almost_equal(r1, e1)

        r2 = idx2.get_indexer(idx1[::-1], method="backfill")
        tm.assert_almost_equal(r2, e1[::-1])

        rbfill1 = idx2.get_indexer(idx1, method="bfill")
        tm.assert_almost_equal(r1, rbfill1)

        # pass non-MultiIndex
        r1 = idx1.get_indexer(idx2.values)
        rexp1 = idx1.get_indexer(idx2)
        tm.assert_almost_equal(r1, rexp1)

        r1 = idx1.get_indexer([1, 2, 3])
        assert (r1 == [-1, -1, -1]).all()

        # create index with duplicates
        idx1 = Index(list(range(10)) + list(range(10)))
        idx2 = Index(list(range(20)))

        msg = "Reindexing only valid with uniquely valued Index objects"
        with pytest.raises(InvalidIndexError, match=msg):
            idx1.get_indexer(idx2)

    def test_get_indexer_nearest(self) -> None:
        midx = MultiIndex.from_tuples([("a", 1), ("b", 2)])
        msg = (
            "method='nearest' not implemented yet for MultiIndex; see GitHub issue 9365"
        )
        with pytest.raises(NotImplementedError, match=msg):
            midx.get_indexer(["a"], method="nearest")
        msg = "tolerance not implemented yet for MultiIndex"
        with pytest.raises(NotImplementedError, match=msg):
            midx.get_indexer(["a"], method="pad", tolerance=2)

    def test_get_indexer_categorical_time(self) -> None:
        # https://github.com/pandas-dev/pandas/issues/21390
        midx = MultiIndex.from_product(
            [
                Categorical(["a", "b", "c"]),
                Categorical(date_range("2012-01-01", periods=3, freq="h")),
            ]
        )
        result = midx.get_indexer(midx)
        tm.assert_numpy_array_equal(result, np.arange(9, dtype=np.intp))

    @pytest.mark.parametrize(
        "index_arr,labels,expected",
        [
            (
                [[1, np.nan, 2], [3, 4, 5]],
                [1, np.nan, 2],
                np.array([-1, -1, -1], dtype=np.intp),
            ),
            ([[1, np.nan, 2], [3, 4, 5]], [(np.nan, 4)], np.array([1], dtype=np.intp)),
            ([[1, 2, 3], [np.nan, 4, 5]], [(1, np.nan)], np.array([0], dtype=np.intp)),
            (
                [[1, 2, 3], [np.nan, 4, 5]],
                [np.nan, 4, 5],
                np.array([-1, -1, -1], dtype=np.intp),
            ),
        ],
    )
    def test_get_indexer_with_missing_value(self, index_arr: List[List[Union[int, float]]], labels: List[Union[int, float, Tuple[float, int]]], expected: np.ndarray) -> None:
        # issue 19132
        idx = MultiIndex.from_arrays(index_arr)
        result = idx.get_indexer(labels)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_methods(self) -> None:
        # https://github.com/pandas-dev/pandas/issues/29896
        # test getting an indexer for another index with different methods
        # confirms that getting an indexer without a filling method, getting an
        # indexer and backfilling, and getting an indexer and padding all behave
        # correctly in the case where all of the target values fall in between
        # several levels in the MultiIndex into which they are getting an indexer
        #
        # visually, the MultiIndexes used in this test are:
        # mult_idx_1:
        #  0: -1 0
        #  1:    2
        #  2:    3
        #  3:    4
        #  4:  0 0
        #  5:    2
        #  6:    3
        #  7:    4
        #  8:  1 0
        #  9:    2
        # 10:    3
        # 11:    4
        #
        # mult_idx_2:
        #  0: 0 1
        #  1:   3
        #  2:   4
        mult_idx_1 = MultiIndex.from_product([[-1, 0, 1], [0, 2, 3, 4]])
        mult_idx_2 = MultiIndex.from_product([[0], [1, 3, 4]])

        indexer = mult_idx_1.get_indexer(mult_idx_2)
        expected = np.array([-1, 6, 7], dtype=indexer.dtype)
        tm.assert_almost_equal(expected, indexer)

        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method="backfill")
        expected = np.array([5, 6, 7], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)

        # ensure the legacy "bfill" option functions identically to "backfill"
        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method="bfill")
        expected = np.array([5, 6, 7], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)

        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method="pad")
        expected = np.array([4, 6, 7], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)

        # ensure the legacy "ffill" option functions identically to "pad"
        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method="ffill")
        expected = np.array([4, 6, 7], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)

    @pytest.mark.parametrize("method", ["pad", "ffill", "backfill", "bfill", "nearest"])
    def test_get_indexer_methods_raise_for_non_monotonic(self, method: str) -> None:
        # 53452
        mi = MultiIndex.from_arrays([[0, 4, 2], [0, 4, 2]])
        if method == "nearest":
            err = NotImplementedError
            msg = "not implemented yet for MultiIndex"
        else:
            err = ValueError
            msg = "index must be monotonic increasing or decreasing"
        with pytest.raises(err, match=msg):
            mi.get_indexer([(1, 1)], method=method)

    def test_get_indexer_three_or_more_levels(self) -> None:
        # https://github.com/pandas-dev/pandas/issues/29896
        # tests get_indexer() on MultiIndexes with 3+ levels
        # visually, these are
        # mult_idx_1:
        #  0: 1 2 5
        #  1:     7
        #  2:   4 5
        #  3:     7
        #  4:   6 5
        #  5:     7
        #  6: 3 2 5
        #  7:     7
        #  8:   4 5
        #  9:     7
        # 10:   6 5
        # 11:     7
        #
        # mult_idx_2:
        #  0: 1 1 8
        #  1: 1 5 9
        #  2: 1 6 7
        #  3: 2 1 6
        #  4: 2 7 6
        #  5: 2 7 8
        #  6: 3 6 8
        mult_idx_1 = MultiIndex.from_product([[1, 3], [2, 4, 6], [5, 7]])
        mult_idx_2 = MultiIndex.from_tuples(
            [
                (1, 1, 8),
                (1, 5, 9),
                (1, 6, 7),
                (2, 1, 6),
                (2, 7, 7),
                (2, 7, 8),
                (3, 6, 8),
            ]
        )
        # sanity check
        assert mult_idx_1.is_monotonic_increasing
        assert mult_idx_1.is_unique
        assert mult_idx_2.is_monotonic_increasing
        assert mult_idx_2.is_unique

        # show the relationships between the two
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

        # test with backfilling
        indexer_backfilled = mult_idx_1.get_indexer(mult_idx_2, method="backfill")
        expected = np.array([0, 4, 5, 6, 6, 6, -1], dtype=indexer_backfilled.dtype)
        tm.assert_almost_equal(expected, indexer_backfilled)

        # now, the same thing, but forward-filled (aka "padded")
        indexer_padded = mult_idx_1.get_indexer(mult_idx_2, method="pad")
        expected = np.array([-1, 3, 5, 5, 5, 5, 11], dtype=indexer_padded.dtype)
        tm.assert_almost_equal(expected, indexer_padded)

        # now, do the indexing in the other direction
        assert mult_idx_2[0] < mult_idx_1[0] < mult_idx_2[1]
        assert mult_idx_2[0] < mult_idx_1[1] < mult_idx_2[1]
        assert mult_idx_2[0] < mult_idx_1[2] < mult_idx_2[1]
        assert mult_idx_2[0] < mult_idx_1[3] < mult_idx_2[1]
        assert mult_idx_2[1] < mult_idx_1[4] < mult_idx_2[2]
        assert mult_idx_2[2] == mult_idx_1[5]
        assert mult_idx_2[5] < mult_idx_1[6] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[7] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[8] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[9] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[10] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[11] < mult_idx_2[6]

        indexer = mult_idx_2.get_indexer(mult_idx_1)
        expected = np.array(
            [-1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1], dtype=indexer.dtype
        )
        tm.assert_almost_equal(expected, indexer)

        backfill_indexer = mult_idx_2.get_indexer(mult_idx_1, method="bfill")
        expected = np.array(
            [1, 1, 1, 1, 2, 2, 6, 6, 6, 6, 6, 6], dtype=backfill_indexer.dtype
        )
        tm.assert_almost_equal(expected, backfill_indexer)

        pad_indexer = mult_idx_2.get_indexer(mult_idx_1, method="pad")
        expected = np.array(
            [0, 0, 0, 0, 1, 2, 5, 5, 5, 5, 5, 5], dtype=pad_indexer.dtype
        )
        tm.assert_almost_equal(expected, pad_indexer)

    def test_get_indexer_crossing_levels(self) -> None:
        # https://github.com/pandas-dev/pandas/issues/29896
        # tests a corner case with get_indexer() with MultiIndexes where, when we
        # need to "carry" across levels, proper tuple ordering is respected
        #
        # the MultiIndexes used in this test, visually, are:
        # mult_idx_1:
        #  0: 1 1 1 1
        #  1:       2
        #  2:     2 1
        #  3:       2
        #  4: 1 2 1 1
        #  5:       2
        #  6:     2 1
        #  7:       2
        #  8: 2 1 1 1
        #  9:       2
        # 10:     2 1
        # 11:       2
        # 12: 2 2 1 1
        # 13:       2
        # 14:     2 1
        # 15:       2
        #
        # mult_idx_2:
        #  0: 1 3 2 2
        #  1: 2 3 2 2
        mult_idx_1 = MultiIndex.from_product([[1, 2]] * 4)
        mult_idx_2 = MultiIndex.from_tuples([(1, 3, 2, 2), (2, 3, 2, 2)])

        # show the tuple orderings, which get_indexer() should respect
        assert mult_idx_1[7] < mult_idx_2[0] < mult_idx_1[8]
        assert mult_idx_1[-1] < mult_idx_2[1]

        indexer = mult_idx_1.get_indexer(mult_idx_2)
        expected = np.array([-1, -1], dtype=indexer.dtype)
        tm.assert_almost_equal(expected, indexer)

        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method="bfill")
        expected = np.array([8, -1], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)

        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method="ffill")
        expected = np.array([7, 15], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)

    def test_get_indexer_kwarg_validation(self) -> None:
        # GH#41918
        mi = MultiIndex.from_product([range(3), ["A", "B"]])

        msg = "limit argument only valid if doing pad, backfill or nearest"
        with pytest.raises(ValueError, match=msg):
            mi.get_indexer(mi[:-1], limit=4)

        msg = "tolerance argument only valid if doing pad, backfill or nearest"
        with pytest.raises(ValueError, match=msg):
            mi.get_indexer(mi[:-1], tolerance="piano")

    def test_get_indexer_nan(self) -> None:
        # GH#37222
        idx1 = MultiIndex.from_product([["A"], [1.0, 2.0]], names=["id1", "id2"])
        idx2 = MultiIndex.from_product([["A"], [np.nan, 2.0]], names=["id1", "id2"])
        expected = np.array([-1, 1])
        result = idx2.get_indexer(idx1)
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)
        result = idx1.get_indexer(idx2)
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)


def test_getitem(idx: MultiIndex) -> None:
    # scalar
    assert idx[2] == ("bar", "one")

    # slice
    result = idx[2:5]
    expected = idx[[2, 3, 4]]
    assert result.equals(expected)

    # boolean
    result = idx[[True, False, True, False, True, True]]
    result2 = idx[np.array([True, False, True, False, True, True])]
    expected = idx[[0, 2, 4, 5]]
    assert result.equals(expected)
    assert result2.equals(expected)


def test_getitem_group_select(idx: MultiIndex) -> None:
    sorted_idx, _ = idx.sortlevel(0)
    assert sorted_idx.get_loc("baz") == slice(3, 4)
    assert sorted_idx.get_loc("foo") == slice(0, 2)


@pytest.mark.parametrize("box", [list, Index])
def test_getitem_bool_index_all(box: Union[type, Index]) -> None:
    # GH#22533
    ind1 = box([True] * 5)
    idx = MultiIndex.from_tuples([(10, 1), (20, 2), (30, 3), (40, 4), (50, 5)])
    tm.assert_index_equal(idx[ind1], idx)

    ind2 = box([True, False, True, False, False])
    expected = MultiIndex.from_tuples([(10, 1), (30, 3)])
    tm.assert_index_equal(idx[ind2], expected)


@pytest.mark.parametrize("box", [list, Index])
def test_getitem_bool_index_single(box: Union[type, Index]) -> None:
    # GH#22533
    ind1 = box([True])
    idx = MultiIndex.from_tuples([(10, 1)])
    tm.assert_index_equal(idx[ind1], idx)

    ind2 = box([False])
    expected = MultiIndex(
        levels=[np.array([], dtype=np.int64), np.array([], dtype=np.int64)],
        codes=[[], []],
    )
    tm.assert_index_equal(idx[ind2], expected)


class TestGetLoc:
    def test_get_loc(self, idx: MultiIndex) -> None:
        assert idx.get_loc(("foo", "two")) == 1
        assert idx.get_loc(("baz", "two")) == 3
        with pytest.raises(KeyError, match=r"^\('bar', 'two'\)$"):
            idx.get_loc(("bar", "two"))
        with pytest.raises(KeyError, match=r"^'quux'$"):
            idx.get_loc("quux")

        # 3 levels
        index = MultiIndex(
            levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))],
            codes=[
                np.array([0, 0, 1, 2, 2, 2, 3, 3]),
                np.array([0, 1, 0, 0, 0, 1, 0, 1]),
                np.array([1, 0, 1, 1, 0, 0, 1, 0]),
            ],
        )
        with pytest.raises(KeyError, match=r"^\(1, 1\)$"):
            index.get_loc((1, 1))
        assert index.get_loc((2, 0)) == slice(3, 5)

    def test_get_loc_duplicates(self) -> None:
        index = Index([2, 2, 2, 2])
        result = index.get_loc(2)
        expected = slice(0, 4)
        assert result == expected

        index = Index(["c", "a", "a", "b", "b"])
        rs = index.get_loc("c")
        xp = 0
        assert rs == xp

        with pytest.raises(KeyError, match="2"):
            index.get_loc(2)

    def test_get_loc_level(self) -> None:
        index = MultiIndex(
            levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))],
            codes=[
                np.array([0, 0, 1, 2, 2, 2, 3, 3]),
                np.array([0, 1, 0, 0, 0, 1, 0, 1]),
                np.array([1, 0, 1, 1, 0, 0, 1, 0]),
            ],
        )
        loc, new_index = index.get_loc_level((0, 1))
        expected = slice(1, 2)
        exp_index = index[expected].droplevel(0).droplevel(0)
        assert loc == expected
        assert new_index.equals(exp_index)

        loc, new_index = index.get_loc_level((0, 1, 0))
        expected = 1
        assert loc == expected
        assert new_index is None

        with pytest.raises(KeyError, match=r"^\(2, 2\)$"):
            index.get_loc_level((2, 2))
        # GH 22221: unused label
        with pytest.raises(KeyError, match=r"^2$"):
            index.drop(2).get_loc_level(2)
        # Unused label on unsorted level:
        with pytest.raises(KeyError, match=r"^2$"):
            index.drop(1, level=2).get_loc_level(2, level=2)

        index = MultiIndex(
            levels=[[2000], list(range(4))],
            codes=[np.array([0, 0, 0, 0]), np.array([0, 1, 2, 3])],
        )
        result, new_index = index.get_loc_level((2000, slice(None, None)))
        expected = slice(None, None)
        assert result == expected
        assert new_index.equals(index.droplevel(0))

    @pytest.mark.parametrize("dtype1", [int, float, bool, str])
    @pytest.mark.parametrize("dtype2", [int, float, bool, str])
    def test_get_loc_multiple_dtypes(self, dtype1: type, dtype2: type) -> None:
        # GH 18520
        levels = [np.array([0, 1]).astype(dtype1), np.array([0, 1]).astype(dtype2)]
        idx = MultiIndex.from_product(levels)
        assert idx.get_loc(idx[2]) == 2

    @pytest.mark.parametrize("level", [0, 1])
    @pytest.mark.parametrize("dtypes", [[int, float], [float, int]])
    def test_get_loc_implicit_cast(self, level: int,
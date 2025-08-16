import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    NaT,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.util.version import Version
from typing import Any, List, Optional, Union


class TestDataFrameSortValues:
    @pytest.mark.parametrize("dtype", [np.uint8, bool])
    def test_sort_values_sparse_no_warning(self, dtype: Union[np.uint8, bool]) -> None:
        # GH#45618
        ser = pd.Series(Categorical(["a", "b", "a"], categories=["a", "b", "c"]))
        df = pd.get_dummies(ser, dtype=dtype, sparse=True)

        with tm.assert_produces_warning(None):
            # No warnings about constructing Index from SparseArray
            df.sort_values(by=df.columns.tolist())

    def test_sort_values(self) -> None:
        frame = DataFrame(
            [[1, 1, 2], [3, 1, 0], [4, 5, 6]], index=[1, 2, 3], columns=list("ABC")
        )

        # by column (axis=0)
        sorted_df = frame.sort_values(by="A")
        indexer = frame["A"].argsort().values
        expected = frame.loc[frame.index[indexer]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by="A", ascending=False)
        indexer = indexer[::-1]
        expected = frame.loc[frame.index[indexer]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by="A", ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

        # GH4839
        sorted_df = frame.sort_values(by=["A"], ascending=[False])
        tm.assert_frame_equal(sorted_df, expected)

        # multiple bys
        sorted_df = frame.sort_values(by=["B", "C"])
        expected = frame.loc[[2, 1, 3]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=["B", "C"], ascending=False)
        tm.assert_frame_equal(sorted_df, expected[::-1])

        sorted_df = frame.sort_values(by=["B", "A"], ascending=[True, False])
        tm.assert_frame_equal(sorted_df, expected)

        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            frame.sort_values(by=["A", "B"], axis=2, inplace=True)

        # by row (axis=1): GH#10806
        sorted_df = frame.sort_values(by=3, axis=1)
        expected = frame
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=3, axis=1, ascending=False)
        expected = frame.reindex(columns=["C", "B", "A"])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 2], axis="columns")
        expected = frame.reindex(columns=["B", "A", "C"])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 3], axis=1, ascending=[True, False])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 3], axis=1, ascending=False)
        expected = frame.reindex(columns=["C", "B", "A"])
        tm.assert_frame_equal(sorted_df, expected)

        msg = r"Length of ascending \(5\) != length of by \(2\)"
        with pytest.raises(ValueError, match=msg):
            frame.sort_values(by=["A", "B"], axis=0, ascending=[True] * 5)

    def test_sort_values_by_empty_list(self) -> None:
        # https://github.com/pandas-dev/pandas/issues/40258
        expected = DataFrame({"a": [1, 4, 2, 5, 3, 6]})
        result = expected.sort_values(by=[])
        tm.assert_frame_equal(result, expected)
        assert result is not expected

    def test_sort_values_inplace(self) -> None:
        frame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[1, 2, 3, 4],
            columns=["A", "B", "C", "D"],
        )

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by="A", inplace=True)
        assert return_value is None
        expected = frame.sort_values(by="A")
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by=1, axis=1, inplace=True)
        assert return_value is None
        expected = frame.sort_values(by=1, axis=1)
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(by="A", ascending=False, inplace=True)
        assert return_value is None
        expected = frame.sort_values(by="A", ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.copy()
        return_value = sorted_df.sort_values(
            by=["A", "B"], ascending=False, inplace=True
        )
        assert return_value is None
        expected = frame.sort_values(by=["A", "B"], ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_multicolumn(self) -> None:
        A = np.arange(5).repeat(20)
        B = np.tile(np.arange(5), 20)
        np.random.default_rng(2).shuffle(A)
        np.random.default_rng(2).shuffle(B)
        frame = DataFrame(
            {"A": A, "B": B, "C": np.random.default_rng(2).standard_normal(100)}
        )

        result = frame.sort_values(by=["A", "B"])
        indexer = np.lexsort((frame["B"], frame["A"]))
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)

        result = frame.sort_values(by=["A", "B"], ascending=False)
        indexer = np.lexsort(
            (frame["B"].rank(ascending=False), frame["A"].rank(ascending=False))
        )
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)

        result = frame.sort_values(by=["B", "A"])
        indexer = np.lexsort((frame["A"], frame["B"]))
        expected = frame.take(indexer)
        tm.assert_frame_equal(result, expected)

    def test_sort_values_multicolumn_uint64(self) -> None:
        # GH#9918
        # uint64 multicolumn sort

        df = DataFrame(
            {
                "a": pd.Series([18446637057563306014, 1162265347240853609]),
                "b": pd.Series([1, 2]),
            }
        )
        df["a"] = df["a"].astype(np.uint64)
        result = df.sort_values(["a", "b"])

        expected = DataFrame(
            {
                "a": pd.Series([18446637057563306014, 1162265347240853609]),
                "b": pd.Series([1, 2]),
            },
            index=range(1, -1, -1),
        )

        tm.assert_frame_equal(result, expected)

    def test_sort_values_nan(self) -> None:
        # GH#3917
        df = DataFrame(
            {"A": [1, 2, np.nan, 1, 6, 8, 4], "B": [9, np.nan, 5, 2, 5, 4, 5]}
        )

        # sort one column only
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 9, 2, np.nan, 5, 5, 4]},
            index=[2, 0, 3, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A"], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        expected = DataFrame(
            {"A": [np.nan, 8, 6, 4, 2, 1, 1], "B": [5, 4, 5, 5, np.nan, 9, 2]},
            index=[2, 5, 4, 6, 1, 0, 3],
        )
        sorted_df = df.sort_values(["A"], na_position="first", ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

        expected = df.reindex(columns=["B", "A"])
        sorted_df = df.sort_values(by=1, axis=1, na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='last', order
        expected = DataFrame(
            {"A": [1, 1, 2, 4, 6, 8, np.nan], "B": [2, 9, np.nan, 5, 5, 4, 5]},
            index=[3, 0, 1, 6, 4, 5, 2],
        )
        sorted_df = df.sort_values(["A", "B"])
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='first', order
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 2, 9, np.nan, 5, 5, 4]},
            index=[2, 3, 0, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A", "B"], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='first', not order
        expected = DataFrame(
            {"A": [np.nan, 1, 1, 2, 4, 6, 8], "B": [5, 9, 2, np.nan, 5, 5, 4]},
            index=[2, 0, 3, 1, 6, 4, 5],
        )
        sorted_df = df.sort_values(["A", "B"], ascending=[1, 0], na_position="first")
        tm.assert_frame_equal(sorted_df, expected)

        # na_position='last', not order
        expected = DataFrame(
            {"A": [8, 6, 4, 2, 1, 1, np.nan], "B": [4, 5, 5, np.nan, 2, 9, 5]},
            index=[5, 4, 6, 1, 3, 0, 2],
        )
        sorted_df = df.sort_values(["A", "B"], ascending=[0, 1], na_position="last")
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_stable_descending_sort(self) -> None:
        # GH#6399
        df = DataFrame(
            [[2, "first"], [2, "second"], [1, "a"], [1, "b"]],
            columns=["sort_col", "order"],
        )
        sorted_df = df.sort_values(by="sort_col", kind="mergesort", ascending=False)
        tm.assert_frame_equal(df, sorted_df)

    @pytest.mark.parametrize(
        "expected_idx_non_na, ascending",
        [
            [
                [3, 4, 5, 0, 1, 8, 6, 9, 7, 10, 13, 14],
                [True, True],
            ],
            [
                [0, 3, 4, 5, 1, 8, 6, 7, 10, 13, 14, 9],
                [True, False],
            ],
            [
                [9, 7, 10, 13, 14, 6, 8, 1, 3, 4, 5, 0],
                [False, True],
            ],
            [
                [7, 10, 13, 14, 9, 6, 8, 1, 0, 3, 4, 5],
                [False, False],
            ],
        ],
    )
    @pytest.mark.parametrize("na_position", ["first", "last"])
    def test_sort_values_stable_multicolumn_sort(
        self,
        expected_idx_non_na: List[int],
        ascending: List[bool],
        na_position: str,
    ) -> None:
        # GH#38426 Clarify sort_values with mult. columns / labels is stable
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 1, 1, 1, 6, 8, 4, 8, 8, np.nan, np.nan, 8, 8],
                "B": [9, np.nan, 5, 2, 2, 2, 5, 4, 5, 3, 4, np.nan, np.nan, 4, 4],
            }
        )
        # All rows with NaN in col "B" only have unique values in "A", therefore,
        # only the rows with NaNs in "A" have to be treated individually:
        expected_idx = (
            [11, 12, 2] + expected_idx_non_na
            if na_position == "first"
            else expected_idx_non_na + [2, 11, 12]
        )
        expected = df.take(expected_idx)
        sorted_df = df.sort_values(
            ["A", "B"], ascending=ascending, na_position=na_position
        )
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_stable_categorial(self) -> None:
        # GH#16793
        df = DataFrame({"x": Categorical(np.repeat([1, 2, 3, 4], 5), ordered=True)})
        expected = df.copy()
        sorted_df = df.sort_values("x", kind="mergesort")
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_values_datetimes(self) -> None:
        # GH#3461, argsort / lexsort differences for a datetime column
        df = DataFrame(
            ["a", "a", "a", "b", "c", "d", "e", "f", "g"],
            columns=["A"],
            index=date_range("20130101", periods=9),
        )
        dts = [
            Timestamp(x)
            for x in [
                "2004-02-11",
                "2004-01-21",
                "2004-01-26",
                "2005-09-20",
                "2010-10-04",
                "2009-05-12",
                "2008-11-12",
                "2010-09-28",
                "2010-09-28",
            ]
        ]
        df["B"] = dts[::2] + dts[1::2]
        df["C"] = 2.0
        df["A1"] = 3.0

        df1 = df.sort_values(by="A")
        df2 = df.sort_values(by=["A"])
        tm.assert_frame_equal(df1, df2)

        df1 = df.sort_values(by="B")
        df2 = df.sort_values(by=["B"])
        tm.assert_frame_equal(df1, df2)

        df1 = df.sort_values(by="B")

        df2 = df.sort_values(by=["C", "B"])
        tm.assert_frame_equal(df1, df2)

    def test_sort_values_frame_column_inplace_sort_exception(self, float_frame: DataFrame) -> None:
        s = float_frame["A"]
        float_frame_orig = float_frame.copy()
        # INFO(CoW) Series is a new object, so can be changed inplace
        # without modifying original datafame
        s.sort_values(inplace=True)
        tm.assert_series_equal(s, float_frame_orig["A"].sort_values())
        # column in dataframe is not changed
        tm.assert_frame_equal(float_frame, float_frame_orig)

        cp = s.copy()
        cp.sort_values()  # it works!

    def test_sort_values_nat_values_in_int_column(self) -> None:
        # GH#14922: "sorting with large float and multiple columns incorrect"

        # cause was that the int64 value NaT was considered as "na". Which is
        # only correct for datetime64 columns.

        int_values = (2, int(NaT._value))
        float_values = (2.0, -1.797693e308)

        df = DataFrame(
            {"int": int_values, "float": float_values}, columns=["int", "float"]
        )

        df_reversed = DataFrame(
            {"int": int_values[::-1], "float": float_values[::-1]},
            columns=["int", "float"],
            index=range(1, -1, -1),
        )

        # NaT is not a "na" for int64 columns, so na_position must not
        # influence the result:
        df_sorted = df.sort_values(["int", "float"], na_position="last")
        tm.assert_frame_equal(df_sorted, df_reversed)

        df_sorted = df.sort_values(["int", "float"], na_position="first")
        tm.assert_frame_equal(df_sorted, df_reversed)

        # reverse sorting order
        df_sorted = df.sort_values(["int", "float"], ascending=False)
        tm.assert_frame_equal(df_sorted, df)

        # and now check if NaT is still considered as "na" for datetime64
        # columns:
        df = DataFrame(
            {"datetime": [Timestamp("2016-01-01"), NaT], "float": float_values},
            columns=["datetime", "float"],
        )

        df_reversed = DataFrame(
            {"datetime": [NaT, Timestamp("2016-01
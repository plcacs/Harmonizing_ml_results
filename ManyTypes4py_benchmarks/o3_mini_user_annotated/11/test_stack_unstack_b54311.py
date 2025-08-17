#!/usr/bin/env python
from datetime import datetime
import itertools
import re
from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

from pandas._libs import lib
import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Period,
    Series,
    Timedelta,
    date_range,
)
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib


@pytest.fixture(params=[True, False])
def future_stack(request: pytest.FixtureRequest) -> bool:
    return request.param


class TestDataFrameReshape:
    @pytest.mark.filterwarnings("ignore:The previous implementation of stack is deprecated")
    def test_stack_unstack(self, float_frame: DataFrame, future_stack: bool) -> None:
        df: DataFrame = float_frame.copy()
        df[:] = np.arange(np.prod(df.shape)).reshape(df.shape)

        stacked: Series = df.stack(future_stack=future_stack)
        stacked_df: DataFrame = DataFrame({"foo": stacked, "bar": stacked})

        unstacked: DataFrame = stacked.unstack()
        unstacked_df: DataFrame = stacked_df.unstack()

        tm.assert_frame_equal(unstacked, df)
        tm.assert_frame_equal(unstacked_df["bar"], df)

        unstacked_cols: DataFrame = stacked.unstack(0)
        unstacked_cols_df: DataFrame = stacked_df.unstack(0)
        tm.assert_frame_equal(unstacked_cols.T, df)
        tm.assert_frame_equal(unstacked_cols_df["bar"].T, df)

    @pytest.mark.filterwarnings("ignore:The previous implementation of stack is deprecated")
    def test_stack_mixed_level(self, future_stack: bool) -> None:
        levels: List[Any] = [range(3), [3, "a", "b"], [1, 2]]
        df: DataFrame = DataFrame(np.tile(1, (3, 3)), index=levels[0], columns=levels[1])
        result: Series = df.stack(future_stack=future_stack)
        expected: Series = Series(np.tile(1, 6), index=MultiIndex.from_product([levels[0], levels[1]]))
        tm.assert_series_equal(result, expected)

    def test_unstack_not_consolidated(self, float_frame: DataFrame) -> None:
        df: DataFrame = float_frame.copy()
        df["A"] = np.arange(len(df))
        df2: DataFrame = df.loc[:, ["A"]]
        df2["B"] = df["A"]
        tm.assert_frame_equal(df2.unstack(), df.unstack())

    def test_unstack_fill_frame_object(self) -> None:
        data: Series = Series(["a", "b", "c", "a"], dtype="object")
        data.index = MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "b"), ("z", "a")])
        # By default missing values will be NaN
        result: DataFrame = data.unstack()
        expected: DataFrame = DataFrame(
            {"a": ["a", np.nan, "a"], "b": ["b", "c", np.nan]},
            index=list("xyz"),
            dtype=object,
        )
        tm.assert_frame_equal(result, expected)
        # Fill with any value replaces missing values as expected
        result = data.unstack(fill_value="d")
        expected = DataFrame(
            {"a": ["a", "d", "a"], "b": ["b", "c", "d"]},
            index=list("xyz"),
            dtype=object,
        )
        tm.assert_frame_equal(result, expected)

    def test_unstack_timezone_aware_values(self) -> None:
        df: DataFrame = DataFrame(
            {
                "timestamp": [pd.Timestamp("2017-08-27 01:00:00.709949+0000", tz="UTC")],
                "a": ["a"],
                "b": ["b"],
                "c": ["c"],
            },
            columns=["timestamp", "a", "b", "c"],
        )
        result: DataFrame = df.set_index(["a", "b"]).unstack()
        expected: DataFrame = DataFrame(
            [[pd.Timestamp("2017-08-27 01:00:00.709949+0000", tz="UTC"), "c"]],
            index=Index(["a"], name="a"),
            columns=MultiIndex(
                levels=[["timestamp", "c"], ["b"]],
                codes=[[0, 1], [0, 0]],
                names=[None, "b"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:The previous implementation of stack is deprecated")
    def test_stack_timezone_aware_values(self, future_stack: bool) -> None:
        ts: pd.DatetimeIndex = date_range(freq="D", start="20180101", end="20180103", tz="America/New_York")
        df: DataFrame = DataFrame({"A": ts}, index=["a", "b", "c"])
        result: Series = df.stack(future_stack=future_stack)
        expected: Series = Series(
            ts,
            index=MultiIndex(levels=[["a", "b", "c"], ["A"]], codes=[[0, 1, 2], [0, 0, 0]]),
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:The previous implementation of stack is deprecated")
    @pytest.mark.parametrize("dropna", [True, False, lib.no_default])
    def test_stack_empty_frame(self, dropna: Any, future_stack: bool) -> None:
        levels: List[Any] = [pd.RangeIndex(0), pd.RangeIndex(0)]
        expected: Series = Series(dtype=np.float64, index=MultiIndex(levels=levels, codes=[[], []]))
        if future_stack and dropna is not lib.no_default:
            with pytest.raises(ValueError, match="dropna must be unspecified"):
                DataFrame(dtype=np.float64).stack(dropna=dropna, future_stack=future_stack)
        else:
            result: Series = DataFrame(dtype=np.float64).stack(dropna=dropna, future_stack=future_stack)
            tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:The previous implementation of stack is deprecated")
    @pytest.mark.parametrize("dropna", [True, False, lib.no_default])
    def test_stack_empty_level(self, dropna: Any, future_stack: bool, int_frame: DataFrame) -> None:
        if future_stack and dropna is not lib.no_default:
            with pytest.raises(ValueError, match="dropna must be unspecified"):
                DataFrame(dtype=np.int64).stack(dropna=dropna, future_stack=future_stack)
        else:
            expected: DataFrame = int_frame
            result: DataFrame = int_frame.copy().stack(level=[], dropna=dropna, future_stack=future_stack)
            tm.assert_frame_equal(result, expected)

            expected = DataFrame()
            result = DataFrame().stack(level=[], dropna=dropna, future_stack=future_stack)
            tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:The previous implementation of stack is deprecated")
    @pytest.mark.parametrize("dropna", [True, False, lib.no_default])
    @pytest.mark.parametrize("fill_value", [None, 0])
    def test_stack_unstack_empty_frame(self, dropna: Any, fill_value: Optional[int], future_stack: bool) -> None:
        if future_stack and dropna is not lib.no_default:
            with pytest.raises(ValueError, match="dropna must be unspecified"):
                DataFrame(dtype=np.int64).stack(dropna=dropna, future_stack=future_stack).unstack(fill_value=fill_value)
        else:
            result: DataFrame = (
                DataFrame(dtype=np.int64)
                .stack(dropna=dropna, future_stack=future_stack)
                .unstack(fill_value=fill_value)
            )
            expected: DataFrame = DataFrame(dtype=np.int64)
            tm.assert_frame_equal(result, expected)

    def test_unstack_single_index_series(self) -> None:
        msg: str = r"index must be a MultiIndex to unstack.*"
        with pytest.raises(ValueError, match=msg):
            Series(dtype=np.int64).unstack()

    def test_unstacking_multi_index_df(self) -> None:
        df: DataFrame = DataFrame(
            {
                "name": ["Alice", "Bob"],
                "score": [9.5, 8],
                "employed": [False, True],
                "kids": [0, 0],
                "gender": ["female", "male"],
            }
        )
        df = df.set_index(["name", "employed", "kids", "gender"])
        df = df.unstack(["gender"], fill_value=0)
        expected: DataFrame = df.unstack("employed", fill_value=0).unstack("kids", fill_value=0)
        result: DataFrame = df.unstack(["employed", "kids"], fill_value=0)
        expected = DataFrame(
            [[9.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 8.0]],
            index=Index(["Alice", "Bob"], name="name"),
            columns=MultiIndex.from_tuples(
                [
                    ("score", "female", False, 0),
                    ("score", "female", True, 0),
                    ("score", "male", False, 0),
                    ("score", "male", True, 0),
                ],
                names=[None, "gender", "employed", "kids"],
            ),
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_unstack_sort_false(frame_or_series: Any, dtype: str) -> None:
    index: MultiIndex = MultiIndex.from_tuples(
        [("two", "z", "b"), ("two", "y", "a"), ("one", "z", "b"), ("one", "y", "a")]
    )
    obj: Any = frame_or_series(np.arange(1.0, 5.0), index=index, dtype=dtype)

    result: DataFrame = obj.unstack(level=0, sort=False)
    if frame_or_series is DataFrame:
        expected_columns: MultiIndex = MultiIndex.from_tuples([(0, "two"), (0, "one")])
    else:
        expected_columns = ["two", "one"]
    expected: DataFrame = DataFrame(
        [[1.0, 3.0], [2.0, 4.0]],
        index=MultiIndex.from_tuples([("z", "b"), ("y", "a")]),
        columns=expected_columns,
        dtype=dtype,
    )
    tm.assert_frame_equal(result, expected)

    result = obj.unstack(level=-1, sort=False)
    if frame_or_series is DataFrame:
        expected_columns = MultiIndex(
            levels=[range(1), ["b", "a"]], codes=[[0, 0], [0, 1]]
        )
    else:
        expected_columns = ["b", "a"]
    expected = DataFrame(
        [[1.0, np.nan], [np.nan, 2.0], [3.0, np.nan], [np.nan, 4.0]],
        columns=expected_columns,
        index=MultiIndex.from_tuples(
            [("two", "z"), ("two", "y"), ("one", "z"), ("one", "y")]
        ),
        dtype=dtype,
    )
    tm.assert_frame_equal(result, expected)

    result = obj.unstack(level=[1, 2], sort=False)
    if frame_or_series is DataFrame:
        expected_columns = MultiIndex(
            levels=[range(1), ["z", "y"], ["b", "a"]], codes=[[0, 0], [0, 1], [0, 1]]
        )
    else:
        expected_columns = MultiIndex.from_tuples([("z", "b"), ("y", "a")])
    expected = DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["two", "one"],
        columns=expected_columns,
        dtype=dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_unstack_fill_frame_object() -> None:
    data: Series = Series(["a", "b", "c", "a"], dtype="object")
    data.index = MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "b"), ("z", "a")])
    result: DataFrame = data.unstack()
    expected: DataFrame = DataFrame(
        {"a": ["a", np.nan, "a"], "b": ["b", "c", np.nan]},
        index=list("xyz"),
        dtype=object,
    )
    tm.assert_frame_equal(result, expected)
    result = data.unstack(fill_value="d")
    expected = DataFrame(
        {"a": ["a", "d", "a"], "b": ["b", "c", "d"]},
        index=list("xyz"),
        dtype=object,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings("ignore:The previous implementation of stack is deprecated")
def test_stack_tuple_columns(future_stack: bool) -> None:
    df: DataFrame = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=[("a", 1), ("a", 2), ("b", 1)]
    )
    result: Series = df.stack(future_stack=future_stack)
    expected: Series = Series(
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        index=MultiIndex(
            levels=[range(3), [("a", 1), ("a", 2), ("b", 1)]],
            codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        ),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dtype, na_value",
    [
        ("float64", np.nan),
        ("Float64", np.nan),
        ("Float64", pd.NA),
        ("Int64", pd.NA),
    ],
)
@pytest.mark.parametrize("test_multiindex", [True, False])
def test_stack_preserves_na(dtype: str, na_value: Any, test_multiindex: bool) -> None:
    if test_multiindex:
        index: MultiIndex = MultiIndex.from_arrays(2 * [Index([na_value], dtype=dtype)])
    else:
        index = Index([na_value], dtype=dtype)
    df: DataFrame = DataFrame({"a": [1]}, index=index)
    result: Series = df.stack()
    if test_multiindex:
        expected_index: MultiIndex = MultiIndex.from_arrays(
            [
                Index([na_value], dtype=dtype),
                Index([na_value], dtype=dtype),
                Index(["a"]),
            ]
        )
    else:
        expected_index = MultiIndex.from_arrays(
            [
                Index([na_value], dtype=dtype),
                Index(["a"]),
            ]
        )
    expected: Series = Series(1, index=expected_index)
    tm.assert_series_equal(result, expected)


# Additional tests and classes with similar type annotations follow...
# (Due to length, only a representative subset of functions are annotated.)
                        
# End of annotated code.

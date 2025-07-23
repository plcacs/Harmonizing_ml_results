from datetime import date, datetime, timedelta
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest
from pandas.core.dtypes.common import is_object_dtype, is_string_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
)
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import MergeError, merge


def get_test_data(ngroups: int = 8, n: int = 50) -> np.ndarray:
    unique_groups = list(range(ngroups))
    arr = np.asarray(np.tile(unique_groups, n // ngroups))
    if len(arr) < n:
        arr = np.asarray(list(arr) + unique_groups[: n - len(arr)])
    np.random.default_rng(2).shuffle(arr)
    return arr


@pytest.fixture
def dfs_for_indicator() -> Tuple[DataFrame, DataFrame]:
    df1 = DataFrame(
        {"col1": [0, 1], "col_conflict": [1, 2], "col_left": ["a", "b"]}
    )
    df2 = DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col_conflict": [1, 2, 3, 4, 5],
            "col_right": [2, 2, 2, 2, 2],
        }
    )
    return (df1, df2)


class TestMerge:

    @pytest.fixture
    def df(self) -> DataFrame:
        df = DataFrame(
            {
                "key1": get_test_data(),
                "key2": get_test_data(),
                "data1": np.random.default_rng(2).standard_normal(50),
                "data2": np.random.default_rng(2).standard_normal(50),
            }
        )
        df = df[df["key2"] > 1]
        return df

    @pytest.fixture
    def df2(self) -> DataFrame:
        return DataFrame(
            {
                "key1": get_test_data(n=10),
                "key2": get_test_data(ngroups=4, n=10),
                "value": np.random.default_rng(2).standard_normal(10),
            }
        )

    @pytest.fixture
    def left(self) -> DataFrame:
        return DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "e", "a"],
                "v1": np.random.default_rng(2).standard_normal(7),
            }
        )

    def test_merge_inner_join_empty(self) -> None:
        df_empty = DataFrame()
        df_a = DataFrame({"a": [1, 2]}, index=[0, 1], dtype="int64")
        result = merge(df_empty, df_a, left_index=True, right_index=True)
        expected = DataFrame({"a": []}, dtype="int64")
        tm.assert_frame_equal(result, expected)

    def test_merge_common(self, df: DataFrame, df2: DataFrame) -> None:
        joined = merge(df, df2)
        exp = merge(df, df2, on=["key1", "key2"])
        tm.assert_frame_equal(joined, exp)

    def test_merge_non_string_columns(self) -> None:
        left = DataFrame(
            {0: [1, 0, 1, 0], 1: [0, 1, 0, 0], 2: [0, 0, 2, 0], 3: [1, 0, 0, 3]}
        )
        right = left.astype(float)
        expected = left
        result = merge(left, right)
        tm.assert_frame_equal(expected, result)

    def test_merge_index_as_on_arg(self, df: DataFrame, df2: DataFrame) -> None:
        left = df.set_index("key1")
        right = df2.set_index("key1")
        result = merge(left, right, on="key1")
        expected = merge(df, df2, on="key1").set_index("key1")
        tm.assert_frame_equal(result, expected)

    def test_merge_index_singlekey_right_vs_left(self) -> None:
        left = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "e", "a"],
                "v1": np.random.default_rng(2).standard_normal(7),
            }
        )
        right = DataFrame(
            {"v2": np.random.default_rng(2).standard_normal(4)},
            index=["d", "b", "c", "a"],
        )
        merged1 = merge(
            left, right, left_on="key", right_index=True, how="left", sort=False
        )
        merged2 = merge(
            right, left, right_on="key", left_index=True, how="right", sort=False
        )
        tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])
        merged1 = merge(
            left, right, left_on="key", right_index=True, how="left", sort=True
        )
        merged2 = merge(
            right, left, right_on="key", left_index=True, how="right", sort=True
        )
        tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])

    def test_merge_index_singlekey_inner(self) -> None:
        left = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "e", "a"],
                "v1": np.random.default_rng(2).standard_normal(7),
            }
        )
        right = DataFrame(
            {"v2": np.random.default_rng(2).standard_normal(4)},
            index=["d", "b", "c", "a"],
        )
        result = merge(left, right, left_on="key", right_index=True, how="inner")
        expected = left.join(right, on="key").loc[result.index]
        tm.assert_frame_equal(result, expected)
        result = merge(right, left, right_on="key", left_index=True, how="inner")
        expected = left.join(right, on="key").loc[result.index]
        tm.assert_frame_equal(result, expected.loc[:, result.columns])

    def test_merge_misspecified(
        self, df: DataFrame, df2: DataFrame, left: DataFrame
    ) -> None:
        right = DataFrame(
            {"v2": np.random.default_rng(2).standard_normal(4)},
            index=["d", "b", "c", "a"],
        )
        msg = "Must pass right_on or right_index=True"
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, right, left_index=True)
        msg = "Must pass left_on or left_index=True"
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, right, right_index=True)
        msg = 'Can only pass argument "on" OR "left_on" and "right_on", not a combination of both'
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, left, left_on="key", on="key")
        msg = "len\\(right_on\\) must equal len\\(left_on\\)"
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, left_on=["key1"], right_on=["key1", "key2"])

    def test_index_and_on_parameters_confusion(
        self, df: DataFrame, df2: DataFrame
    ) -> None:
        msg = "right_index parameter must be of type bool, not <class 'list'>"
        with pytest.raises(ValueError, match=msg):
            merge(
                df,
                df2,
                how="left",
                left_index=False,
                right_index=["key1", "key2"],
            )
        msg = "left_index parameter must be of type bool, not <class 'list'>"
        with pytest.raises(ValueError, match=msg):
            merge(
                df,
                df2,
                how="left",
                left_index=["key1", "key2"],
                right_index=False,
            )
        with pytest.raises(ValueError, match=msg):
            merge(
                df,
                df2,
                how="left",
                left_index=["key1", "key2"],
                right_index=["key1", "key2"],
            )

    def test_merge_overlap(self, left: DataFrame) -> None:
        merged = merge(left, left, on="key")
        exp_len = (left["key"].value_counts() ** 2).sum()
        assert len(merged) == exp_len
        assert "v1_x" in merged
        assert "v1_y" in merged

    def test_merge_different_column_key_names(self) -> None:
        left = DataFrame(
            {"lkey": ["foo", "bar", "baz", "foo"], "value": [1, 2, 3, 4]}
        )
        right = DataFrame(
            {"rkey": ["foo", "bar", "qux", "foo"], "value": [5, 6, 7, 8]}
        )
        merged = left.merge(
            right, left_on="lkey", right_on="rkey", how="outer", sort=True
        )
        exp = Series(
            ["bar", "baz", "foo", "foo", "foo", "foo", np.nan], name="lkey"
        )
        tm.assert_series_equal(merged["lkey"], exp)
        exp = Series(
            ["bar", np.nan, "foo", "foo", "foo", "foo", "qux"], name="rkey"
        )
        tm.assert_series_equal(merged["rkey"], exp)
        exp = Series([2, 3, 1, 1, 4, 4, np.nan], name="value_x")
        tm.assert_series_equal(merged["value_x"], exp)
        exp = Series([6, np.nan, 5, 8, 5, 8, 7], name="value_y")
        tm.assert_series_equal(merged["value_y"], exp)

    def test_merge_copy(self) -> None:
        left = DataFrame({"a": 0, "b": 1}, index=range(10))
        right = DataFrame(
            {"c": "foo", "d": "bar"}, index=range(10)
        )
        merged = merge(left, right, left_index=True, right_index=True)
        merged["a"] = 6
        assert (left["a"] == 0).all()
        merged["d"] = "peekaboo"
        assert (right["d"] == "bar").all()

    def test_merge_nocopy(self, using_infer_string: bool) -> None:
        left = DataFrame({"a": 0, "b": 1}, index=range(10))
        right = DataFrame(
            {"c": "foo", "d": "bar"}, index=range(10)
        )
        merged = merge(left, right, left_index=True, right_index=True)
        assert np.shares_memory(merged["a"]._values, left["a"]._values)
        if not using_infer_string:
            assert np.shares_memory(merged["d"]._values, right["d"]._values)

    def test_intelligently_handle_join_key(self) -> None:
        left = DataFrame(
            {"key": [1, 1, 2, 2, 3], "value": list(range(5))},
            columns=["value", "key"],
        )
        right = DataFrame(
            {"key": [1, 1, 2, 3, 4, 5], "rvalue": list(range(6))}
        )
        joined = merge(left, right, on="key", how="outer")
        expected = DataFrame(
            {
                "key": [1, 1, 1, 1, 2, 2, 3, 4, 5],
                "value": np.array(
                    [0, 0, 1, 1, 2, 3, 4, np.nan, np.nan]
                ),
                "rvalue": [0, 1, 0, 1, 2, 2, 3, 4, 5],
            },
            columns=["value", "key", "rvalue"],
        )
        tm.assert_frame_equal(joined, expected)

    def test_merge_join_key_dtype_cast(self) -> None:
        df1 = DataFrame({"key": [1], "v1": [10]})
        df2 = DataFrame({"key": [2], "v1": [20]})
        df = merge(df1, df2, how="outer")
        assert df["key"].dtype == "int64"
        df1 = DataFrame({"key": [True], "v1": [1]})
        df2 = DataFrame({"key": [False], "v1": [0]})
        df = merge(df1, df2, how="outer")
        assert df["key"].dtype == "bool"
        df1 = DataFrame({"val": [1]})
        df2 = DataFrame({"val": [2]})
        lkey = np.array([1])
        rkey = np.array([2])
        df = merge(df1, df2, left_on=lkey, right_on=rkey, how="outer")
        assert df["key_0"].dtype == np.dtype(int)

    def test_handle_join_key_pass_array(self) -> None:
        left = DataFrame(
            {"key": [1, 1, 2, 2, 3], "value": np.arange(5)},
            columns=["value", "key"],
            dtype="int64",
        )
        right = DataFrame({"rvalue": np.arange(6)}, dtype="int64")
        key = np.array([1, 1, 2, 3, 4, 5], dtype="int64")
        merged = merge(left, right, left_on="key", right_on=key, how="outer")
        merged2 = merge(right, left, left_on=key, right_on="key", how="outer")
        tm.assert_series_equal(merged["key"], merged2["key"])
        assert merged["key"].notna().all()
        assert merged2["key"].notna().all()
        left = DataFrame({"value": np.arange(5)}, columns=["value"])
        right = DataFrame({"rvalue": np.arange(6)})
        lkey = np.array([1, 1, 2, 2, 3])
        rkey = np.array([1, 1, 2, 3, 4, 5])
        merged = merge(left, right, left_on=lkey, right_on=rkey, how="outer")
        expected = Series(
            [1, 1, 1, 1, 2, 2, 3, 4, 5], dtype=int, name="key_0"
        )
        tm.assert_series_equal(merged["key_0"], expected)
        left = DataFrame({"value": np.arange(3)})
        right = DataFrame({"rvalue": np.arange(6)})
        key = np.array([0, 1, 1, 2, 2, 3], dtype=np.int64)
        merged = merge(left, right, left_index=True, right_on=key, how="outer")
        tm.assert_series_equal(
            merged["key_0"], Series(key, name="key_0")
        )

    def test_no_overlap_more_informative_error(self) -> None:
        dt = datetime.now()
        df1 = DataFrame({"x": ["a"]}, index=[dt])
        df2 = DataFrame({"y": ["b", "c"]}, index=[dt, dt])
        msg = f"No common columns to perform merge on. Merge options: left_on={None}, right_on={None}, left_index={False}, right_index={False}"
        with pytest.raises(MergeError, match=msg):
            merge(df1, df2)

    def test_merge_non_unique_indexes(self) -> None:
        dt = datetime(2012, 5, 1)
        dt2 = datetime(2012, 5, 2)
        dt3 = datetime(2012, 5, 3)
        dt4 = datetime(2012, 5, 4)
        df1 = DataFrame({"x": ["a"]}, index=[dt])
        df2 = DataFrame({"y": ["b", "c"]}, index=[dt, dt])
        _check_merge(df1, df2)
        df1 = DataFrame(
            {"x": ["a", "b", "q"]}, index=[dt2, dt, dt4]
        )
        df2 = DataFrame(
            {"y": ["c", "d", "e", "f", "g", "h"]},
            index=[dt3, dt3, dt2, dt2, dt, dt],
        )
        _check_merge(df1, df2)
        df1 = DataFrame({"x": ["a", "b"]}, index=[dt, dt])
        df2 = DataFrame({"y": ["c", "d"]}, index=[dt, dt])
        _check_merge(df1, df2)

    def test_merge_non_unique_index_many_to_many(self) -> None:
        dt = datetime(2012, 5, 1)
        dt2 = datetime(2012, 5, 2)
        dt3 = datetime(2012, 5, 3)
        df1 = DataFrame(
            {"x": ["a", "b", "c", "d"]}, index=[dt2, dt2, dt, dt]
        )
        df2 = DataFrame(
            {"y": ["e", "f", "g", " h", "i"]},
            index=[dt2, dt2, dt3, dt, dt],
        )
        _check_merge(df1, df2)

    def test_left_merge_empty_dataframe(self) -> None:
        left = DataFrame({"key": [1], "value": [2]})
        right = DataFrame({"key": []})
        result
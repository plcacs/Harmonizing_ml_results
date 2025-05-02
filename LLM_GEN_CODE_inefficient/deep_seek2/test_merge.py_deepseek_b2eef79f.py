from datetime import date, datetime, timedelta
import re
from typing import Any, Dict, List, Optional, Tuple, Union

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
    df1 = DataFrame({"col1": [0, 1], "col_conflict": [1, 2], "col_left": ["a", "b"]})
    df2 = DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col_conflict": [1, 2, 3, 4, 5],
            "col_right": [2, 2, 2, 2, 2],
        }
    )
    return df1, df2


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

        # exclude a couple keys for fun
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
        # GH 15328
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
        # https://github.com/pandas-dev/pandas/issues/17962
        # Checks that method runs for non string column names
        left = DataFrame(
            {0: [1, 0, 1, 0], 1: [0, 1, 0, 0], 2: [0, 0, 2, 0], 3: [1, 0, 0, 3]}
        )

        right = left.astype(float)
        expected = left
        result = merge(left, right)
        tm.assert_frame_equal(expected, result)

    def test_merge_index_as_on_arg(self, df: DataFrame, df2: DataFrame) -> None:
        # GH14355

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

        # inner join
        result = merge(left, right, left_on="key", right_index=True, how="inner")
        expected = left.join(right, on="key").loc[result.index]
        tm.assert_frame_equal(result, expected)

        result = merge(right, left, right_on="key", left_index=True, how="inner")
        expected = left.join(right, on="key").loc[result.index]
        tm.assert_frame_equal(result, expected.loc[:, result.columns])

    def test_merge_misspecified(self, df: DataFrame, df2: DataFrame, left: DataFrame) -> None:
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

        msg = (
            'Can only pass argument "on" OR "left_on" and "right_on", not '
            "a combination of both"
        )
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, left, left_on="key", on="key")

        msg = r"len\(right_on\) must equal len\(left_on\)"
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, left_on=["key1"], right_on=["key1", "key2"])

    def test_index_and_on_parameters_confusion(self, df: DataFrame, df2: DataFrame) -> None:
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
        left = DataFrame({"lkey": ["foo", "bar", "baz", "foo"], "value": [1, 2, 3, 4]})
        right = DataFrame({"rkey": ["foo", "bar", "qux", "foo"], "value": [5, 6, 7, 8]})

        merged = left.merge(
            right, left_on="lkey", right_on="rkey", how="outer", sort=True
        )

        exp = Series(["bar", "baz", "foo", "foo", "foo", "foo", np.nan], name="lkey")
        tm.assert_series_equal(merged["lkey"], exp)

        exp = Series(["bar", np.nan, "foo", "foo", "foo", "foo", "qux"], name="rkey")
        tm.assert_series_equal(merged["rkey"], exp)

        exp = Series([2, 3, 1, 1, 4, 4, np.nan], name="value_x")
        tm.assert_series_equal(merged["value_x"], exp)

        exp = Series([6, np.nan, 5, 8, 5, 8, 7], name="value_y")
        tm.assert_series_equal(merged["value_y"], exp)

    def test_merge_copy(self) -> None:
        left = DataFrame({"a": 0, "b": 1}, index=range(10))
        right = DataFrame({"c": "foo", "d": "bar"}, index=range(10))

        merged = merge(left, right, left_index=True, right_index=True)

        merged["a"] = 6
        assert (left["a"] == 0).all()

        merged["d"] = "peekaboo"
        assert (right["d"] == "bar").all()

    def test_merge_nocopy(self, using_infer_string: bool) -> None:
        left = DataFrame({"a": 0, "b": 1}, index=range(10))
        right = DataFrame({"c": "foo", "d": "bar"}, index=range(10))

        merged = merge(left, right, left_index=True, right_index=True)

        assert np.shares_memory(merged["a"]._values, left["a"]._values)
        if not using_infer_string:
            assert np.shares_memory(merged["d"]._values, right["d"]._values)

    def test_intelligently_handle_join_key(self) -> None:
        # #733, be a bit more 1337 about not returning unconsolidated DataFrame

        left = DataFrame(
            {"key": [1, 1, 2, 2, 3], "value": list(range(5))}, columns=["value", "key"]
        )
        right = DataFrame({"key": [1, 1, 2, 3, 4, 5], "rvalue": list(range(6))})

        joined = merge(left, right, on="key", how="outer")
        expected = DataFrame(
            {
                "key": [1, 1, 1, 1, 2, 2, 3, 4, 5],
                "value": np.array([0, 0, 1, 1, 2, 3, 4, np.nan, np.nan]),
                "rvalue": [0, 1, 0, 1, 2, 2, 3, 4, 5],
            },
            columns=["value", "key", "rvalue"],
        )
        tm.assert_frame_equal(joined, expected)

    def test_merge_join_key_dtype_cast(self) -> None:
        # #8596

        df1 = DataFrame({"key": [1], "v1": [10]})
        df2 = DataFrame({"key": [2], "v1": [20]})
        df = merge(df1, df2, how="outer")
        assert df["key"].dtype == "int64"

        df1 = DataFrame({"key": [True], "v1": [1]})
        df2 = DataFrame({"key": [False], "v1": [0]})
        df = merge(df1, df2, how="outer")

        # GH13169
        # GH#40073
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
        expected = Series([1, 1, 1, 1, 2, 2, 3, 4, 5], dtype=int, name="key_0")
        tm.assert_series_equal(merged["key_0"], expected)

        left = DataFrame({"value": np.arange(3)})
        right = DataFrame({"rvalue": np.arange(6)})

        key = np.array([0, 1, 1, 2, 2, 3], dtype=np.int64)
        merged = merge(left, right, left_index=True, right_on=key, how="outer")
        tm.assert_series_equal(merged["key_0"], Series(key, name="key_0"))

    def test_no_overlap_more_informative_error(self) -> None:
        dt = datetime.now()
        df1 = DataFrame({"x": ["a"]}, index=[dt])

        df2 = DataFrame({"y": ["b", "c"]}, index=[dt, dt])

        msg = (
            "No common columns to perform merge on. "
            f"Merge options: left_on={None}, right_on={None}, "
            f"left_index={False}, right_index={False}"
        )

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

        # Not monotonic
        df1 = DataFrame({"x": ["a", "b", "q"]}, index=[dt2, dt, dt4])
        df2 = DataFrame(
            {"y": ["c", "d", "e", "f", "g", "h"]}, index=[dt3, dt3, dt2, dt2, dt, dt]
        )
        _check_merge(df1, df2)

        df1 = DataFrame({"x": ["a", "b"]}, index=[dt, dt])
        df2 = DataFrame({"y": ["c", "d"]}, index=[dt, dt])
        _check_merge(df1, df2)

    def test_merge_non_unique_index_many_to_many(self) -> None:
        dt = datetime(2012, 5, 1)
        dt2 = datetime(2012, 5, 2)
        dt3 = datetime(2012, 5, 3)
        df1 = DataFrame({"x": ["a", "b", "c", "d"]}, index=[dt2, dt2, dt, dt])
        df2 = DataFrame(
            {"y": ["e", "f", "g", " h", "i"]}, index=[dt2, dt2, dt3, dt, dt]
        )
        _check_merge(df1, df2)

    def test_left_merge_empty_dataframe(self) -> None:
        left = DataFrame({"key": [1], "value": [2]})
        right = DataFrame({"key": []})

        result = merge(left, right, on="key", how="left")
        tm.assert_frame_equal(result, left)

        result = merge(right, left, on="key", how="right")
        tm.assert_frame_equal(result, left)

    def test_merge_empty_dataframe(self, index: Index, join_type: str) -> None:
        # GH52777
        left = DataFrame([], index=index[:0])
        right = left.copy()

        result = left.join(right, how=join_type)
        tm.assert_frame_equal(result, left)

    @pytest.mark.parametrize(
        "kwarg",
        [
            {"left_index": True, "right_index": True},
            {"left_index": True, "right_on": "x"},
            {"left_on": "a", "right_index": True},
            {"left_on": "a", "right_on": "x"},
        ],
    )
    def test_merge_left_empty_right_empty(self, join_type: str, kwarg: Dict[str, Any]) -> None:
        # GH 10824
        left = DataFrame(columns=["a", "b", "c"])
        right = DataFrame(columns=["x", "y", "z"])

        exp_in = DataFrame(columns=["a", "b", "c", "x", "y", "z"], dtype=object)

        result = merge(left, right, how=join_type, **kwarg)
        tm.assert_frame_equal(result, exp_in)

    def test_merge_left_empty_right_notempty(self) -> None:
        # GH 10824
        left = DataFrame(columns=["a", "b", "c"])
        right = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["x", "y", "z"])

        exp_out = DataFrame(
            {
                "a": np.array([np.nan] * 3, dtype=object),
                "b": np.array([np.nan] * 3, dtype=object),
                "c": np.array([np.nan] * 3, dtype=object),
                "x": [1, 4, 7],
                "y": [2, 5, 8],
                "z": [3, 6, 9],
            },
            columns=["a", "b", "c", "x", "y", "z"],
        )
        exp_in = exp_out[0:0]  # make empty DataFrame keeping dtype

        def check1(exp: DataFrame, kwarg: Dict[str, Any]) -> None:
            result = merge(left, right, how="inner", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="left", **kwarg)
            tm.assert_frame_equal(result, exp)

        def check2(exp: DataFrame, kwarg: Dict[str, Any]) -> None:
            result = merge(left, right, how="right", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="outer", **kwarg)
            tm.assert_frame_equal(result, exp)

        for kwarg in [
            {"left_index": True, "right_index": True},
            {"left_index": True, "right_on": "x"},
        ]:
            check1(exp_in, kwarg)
            check2(exp_out, kwarg)

        kwarg = {"left_on": "a", "right_index": True}
        check1(exp_in, kwarg)
        exp_out["a"] = [0, 1, 2]
        check2(exp_out, kwarg)

        kwarg = {"left_on": "a", "right_on": "x"}
        check1(exp_in, kwarg)
        exp_out["a"] = np.array([np.nan] * 3, dtype=object)
        check2(exp_out, kwarg)

    def test_merge_left_notempty_right_empty(self) -> None:
        # GH 10824
        left = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"])
        right = DataFrame(columns=["x", "y", "z"])

        exp_out = DataFrame(
            {
                "a": [1, 4, 7],
                "b": [2, 5, 8],
                "c": [3, 6, 9],
                "x": np.array([np.nan] * 3, dtype=object),
                "y": np.array([np.nan] * 3, dtype=object),
                "z": np.array([np.nan] * 3, dtype=object),
            },
            columns=["a", "b", "c", "x", "y", "z"],
        )
        exp_in = exp_out[0:0]  # make empty DataFrame keeping dtype
        # result will have object dtype
        exp_in.index = exp_in.index.astype(object)

        def check1(exp: DataFrame, kwarg: Dict[str, Any]) -> None:
            result = merge(left, right, how="inner", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="right", **kwarg)
            tm.assert_frame_equal(result, exp)

        def check2(exp: DataFrame, kwarg: Dict[str, Any]) -> None:
            result = merge(left, right, how="left", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="outer", **kwarg)
            tm.assert_frame_equal(result, exp)

            # TODO: might reconsider current raise behaviour, see issue 24782
            for kwarg in [
                {"left_index": True, "right_index": True},
                {"left_index": True, "right_on": "x"},
                {"left_on": "a", "right_index": True},
                {"left_on": "a", "right_on": "x"},
            ]:
                check1(exp_in, kwarg)
                check2(exp_out, kwarg)

    @pytest.mark.parametrize(
        "series_of_dtype",
        [
            Series([1], dtype="int64"),
            Series([1], dtype="Int64"),
            Series([1.23]),
            Series(["foo"]),
            Series([True]),
            Series([pd.Timestamp("2018-01-01")]),
            Series([pd.Timestamp("2018-01-01", tz="US/Eastern")]),
        ],
    )
    @pytest.mark.parametrize(
        "series_of_dtype2",
        [
            Series([1], dtype="int64"),
            Series([1], dtype="Int64"),
            Series([1.23]),
            Series(["foo"]),
            Series([True]),
            Series([pd.Timestamp("2018-01-01")]),
            Series([pd.Timestamp("2018-01-01", tz="US/Eastern")]),
        ],
    )
    def test_merge_empty_frame(
        self, series_of_dtype: Series, series_of_dtype2: Series
    ) -> None:
        # GH 25183
        df = DataFrame(
            {"key": series_of_dtype, "value": series_of_dtype2},
            columns=["key", "value"],
        )
        df_empty = df[:0]
        expected = DataFrame(
            {
                "key": Series(dtype=df.dtypes["key"]),
                "value_x": Series(dtype=df.dtypes["value"]),
                "value_y": Series(dtype=df.dtypes["value"]),
            },
            columns=["key", "value_x", "value_y"],
        )
        actual = df_empty.merge(df, on="key")
        tm.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "series_of_dtype",
        [
            Series([1], dtype="int64"),
            Series([1], dtype="Int64"),
            Series([1.23]),
            Series(["foo"]),
            Series([True]),
            Series([pd.Timestamp("2018-01-01")]),
            Series([pd.Timestamp("2018-01-01", tz="US/Eastern")]),
        ],
    )
    @pytest.mark.parametrize(
        "series_of_dtype_all_na",
        [
            Series([np.nan], dtype="Int64"),
            Series([np.nan], dtype="float"),
            Series([np.nan], dtype="object"),
            Series([pd.NaT]),
        ],
    )
    def test_merge_all_na_column(
        self, series_of_dtype: Series, series_of_dtype_all_na: Series
    ) -> None:
        # GH 25183
        df_left = DataFrame(
            {"key": series_of_dtype, "value": series_of_dtype_all_na},
            columns=["key", "value"],
        )
        df_right = DataFrame(
            {"key": series_of_dtype, "value": series_of_dtype_all_na},
            columns=["key", "value"],
        )
        expected = DataFrame(
            {
                "key": series_of_dtype,
                "value_x": series_of_dtype_all_na,
                "value_y": series_of_dtype_all_na,
            },
            columns=["key", "value_x", "value_y"],
        )
        actual = df_left.merge(df_right, on="key")
        tm.assert_frame_equal(actual, expected)

    def test_merge_nosort(self) -> None:
        # GH#2098

        d = {
            "var1": np.random.default_rng(2).integers(0, 10, size=10),
            "var2": np.random.default_rng(2).integers(0, 10, size=10),
            "var3": [
                datetime(2012, 1, 12),
                datetime(2011, 2, 4),
                datetime(2010, 2, 3),
                datetime(2012, 1, 12),
                datetime(2011, 2, 4),
                datetime(2012, 4, 3),
                datetime(2012, 3, 4),
                datetime(2008, 5, 1),
                datetime(2010, 2, 3),
                datetime(2012, 2, 3),
            ],
        }
        df = DataFrame.from_dict(d)
        var3 = df.var3.unique()
        var3 = np.sort(var3)
        new = DataFrame.from_dict(
            {"var3": var3, "var8": np.random.default_rng(2).random(7)}
        )

        result = df.merge(new, on="var3", sort=False)
        exp = merge(df, new, on="var3", sort=False)
        tm.assert_frame_equal(result, exp)

        assert (df.var3.unique() == result.var3.unique()).all()

    @pytest.mark.parametrize(
        ("sort", "values"), [(False, [1, 1, 0, 1, 1]), (True, [0, 1, 1, 1, 1])]
    )
    @pytest.mark.parametrize("how", ["left", "right"])
    def test_merge_same_order_left_right(
        self, sort: bool, values: List[int], how: str
    ) -> None:
        # GH#35382
        df = DataFrame({"a": [1, 0, 1]})

        result = df.merge(df, on="a", how=how, sort=sort)
        expected = DataFrame(values, columns=["a"])
        tm.assert_frame_equal(result, expected)

    def test_merge_nan_right(self) -> None:
        df1 = DataFrame({"i1": [0, 1], "i2": [0, 1]})
        df2 = DataFrame({"i1": [0], "i3": [0]})
        result = df1.join(df2, on="i1", rsuffix="_")
        expected = (
            DataFrame(
                {
                    "i1": {0: 0.0, 1: 1},
                    "i2": {0: 0, 1: 1},
                    "i1_": {0: 0, 1: np.nan},
                    "i3": {0: 0.0, 1: np.nan},
                    None: {0: 0, 1: 0},
                },
                columns=Index(["i1", "i2", "i1_", "i3", None], dtype=object),
            )
            .set_index(None)
            .reset_index()[["i1", "i2", "i1_", "i3"]]
        )
        result.columns = result.columns.astype("object")
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_merge_nan_right2(self) -> None:
        df1 = DataFrame({"i1": [0, 1], "i2": [0.5, 1.5]})
        df2 = DataFrame({"i1": [0], "i3": [0.7]})
        result = df1.join(df2, rsuffix="_", on="i1")
        expected = DataFrame(
            {
                "i1": {0: 0, 1: 1},
                "i1_": {0: 0.0, 1: np.nan},
                "i2": {0: 0.5, 1: 1.5},
                "i3": {0: 0.69999999999999996, 1: np.nan},
            }
        )[["i1", "i2", "i1_", "i3"]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
    )
    def test_merge_type(self, df: DataFrame, df2: DataFrame) -> None:
        class NotADataFrame(DataFrame):
            @property
            def _constructor(self) -> type:
                return NotADataFrame

        nad = NotADataFrame(df)
        result = nad.merge(df2, on="key1")

        assert isinstance(result, NotADataFrame)

    def test_join_append_timedeltas(self) -> None:
        # timedelta64 issues with join/merge
        # GH 5695

        d = DataFrame.from_dict(
            {"d": [datetime(2013, 11, 5, 5, 56)], "t": [timedelta(0, 22500)]}
        )
        df = DataFrame(columns=list("dt"))
        df = concat([df, d], ignore_index=True)
        result = concat([df, d], ignore_index=True)
        expected = DataFrame(
            {
                "d": [datetime(2013, 11, 5, 5, 56), datetime(2013, 11, 5, 5, 56)],
                "t": [timedelta(0, 22500), timedelta(0, 22500)],
            },
            dtype=object,
        )
        tm.assert_frame_equal(result, expected)

    def test_join_append_timedeltas2(self) -> None:
        # timedelta64 issues with join/merge
        # GH 5695
        td = np.timedelta64(300000000)
        lhs = DataFrame(Series([td, td], index=["A", "B"]))
        rhs = DataFrame(Series([td], index=["A"]))

        result = lhs.join(rhs, rsuffix="r", how="left")
        expected = DataFrame(
            {
                "0": Series([td, td], index=list("AB")),
                "0r": Series([td, pd.NaT], index=list("AB")),
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
    def test_other_datetime_unit(self, unit: str) -> None:
        # GH 13389
        df1 = DataFrame({"entity_id": [101, 102]})
        ser = Series([None, None], index=[101, 102], name="days")

        dtype = f"datetime64[{unit}]"

        if unit in ["D", "h", "m"]:
            # not supported so we cast to the nearest supported unit, seconds
            exp_dtype = "datetime64[s]"
        else:
            exp_dtype = dtype
        df2 = ser.astype(exp_dtype).to_frame("days")
        assert df2["days"].dtype == exp_dtype

        result = df1.merge(df2, left_on="entity_id", right_index=True)

        days = np.array(["nat", "nat"], dtype=exp_dtype)
        days = pd.core.arrays.DatetimeArray._simple_new(days, dtype=days.dtype)
        exp = DataFrame(
            {
                "entity_id": [101, 102],
                "days": days,
            },
            columns=["entity_id", "days"],
        )
        assert exp["days"].dtype == exp_dtype
        tm.assert_frame_equal(result, exp)

    @pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "
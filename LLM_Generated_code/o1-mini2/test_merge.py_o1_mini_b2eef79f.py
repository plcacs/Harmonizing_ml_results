from datetime import (
    date,
    datetime,
    timedelta,
)
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

from pandas.core.dtypes.common import (
    is_object_dtype,
    is_string_dtype,
)
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
from pandas.core.reshape.merge import (
    MergeError,
    merge,
)


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

        msg = (
            'Can only pass argument "on" OR "left_on" and "right_on", not '
            "a combination of both"
        )
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, left, left_on="key", on="key")

        msg = r"len\(right_on\) must equal len\(left_on\)"
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
                "value": np.array([0, 0, 1, 1, 2, 3, 4, np.nan, np.nan], dtype=float),
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

        left = DataFrame({"value": np.arange(3)}, columns=["value"])
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

    def test_merge_empty_dataframe(
        self, index: Any, join_type: str
    ) -> None:
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
    def test_merge_left_empty_right_empty(
        self, join_type: str, kwarg: Dict[str, Any]
    ) -> None:
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
        self,
        series_of_dtype: Series,
        series_of_dtype2: Series,
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
        self,
        series_of_dtype: Series,
        series_of_dtype_all_na: Series,
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
            def _constructor(self) -> Callable[..., "NotADataFrame"]:
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

    @pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
    def test_other_timedelta_unit(self, unit: str) -> None:
        # GH 13389
        df1 = DataFrame({"entity_id": [101, 102]})
        ser = Series([None, None], index=[101, 102], name="days")

        dtype = f"m8[{unit}]"
        if unit in ["D", "h", "m"]:
            # We cannot astype, instead do nearest supported unit, i.e. "s"
            msg = "Supported resolutions are 's', 'ms', 'us', 'ns'"
            with pytest.raises(ValueError, match=msg):
                ser.astype(dtype)

            df2 = ser.astype("m8[s]").to_frame("days")
        else:
            df2 = ser.astype(dtype).to_frame("days")
            assert df2["days"].dtype == dtype

        result = df1.merge(df2, left_on="entity_id", right_index=True)

        exp = DataFrame(
            {"entity_id": [101, 102], "days": np.array(["nat", "nat"], dtype=dtype)},
            columns=["entity_id", "days"],
        )
        tm.assert_frame_equal(result, exp)

    def test_overlapping_columns_error_message(self) -> None:
        df = DataFrame({"key": [1, 2, 3], "v1": [4, 5, 6], "v2": [7, 8, 9]})
        df2 = DataFrame({"key": [1, 2, 3], "v1": [4, 5, 6], "v2": [7, 8, 9]})

        df.columns = ["key", "foo", "foo"]
        df2.columns = ["key", "bar", "bar"]
        expected = DataFrame(
            {
                "key": [1, 2, 3],
                "v1": [4, 5, 6],
                "v2": [7, 8, 9],
                "v3": [4, 5, 6],
                "v4": [7, 8, 9],
            }
        )
        expected.columns = ["key", "foo", "foo", "bar", "bar"]
        tm.assert_frame_equal(merge(df, df2), expected)

        # #2649, #10639
        df2.columns = ["key1", "foo", "foo"]
        msg = r"Data columns not unique: Index\(\['foo'\], dtype='object|str'\)"
        with pytest.raises(MergeError, match=msg):
            merge(df, df2)

    def test_merge_on_datetime64tz(self) -> None:
        # GH11405
        left = DataFrame(
            {
                "key": pd.date_range("20151010", periods=2, tz="US/Eastern"),
                "value": [1, 2],
            }
        )
        right = DataFrame(
            {
                "key": pd.date_range("20151011", periods=3, tz="US/Eastern"),
                "value": [1, 2, 3],
            }
        )

        expected = DataFrame(
            {
                "key": pd.date_range("20151010", periods=4, tz="US/Eastern"),
                "value_x": [1, 2, np.nan, np.nan],
                "value_y": [np.nan, 1, 2, 3],
            }
        )
        result = merge(left, right, on="key", how="outer")
        tm.assert_frame_equal(result, expected)

    def test_merge_datetime64tz_values(self) -> None:
        left = DataFrame(
            {
                "key": [1, 2],
                "value": pd.date_range("20151010", periods=2, tz="US/Eastern"),
            }
        )
        right = DataFrame(
            {
                "key": [2, 3],
                "value": pd.date_range("20151011", periods=2, tz="US/Eastern"),
            }
        )
        expected = DataFrame(
            {
                "key": [1, 2, 3],
                "value_x": list(pd.date_range("20151010", periods=2, tz="US/Eastern"))
                + [pd.NaT],
                "value_y": [pd.NaT]
                + list(pd.date_range("20151011", periods=2, tz="US/Eastern")),
            }
        )
        result = merge(left, right, on="key", how="outer")
        tm.assert_frame_equal(result, expected)
        assert result["value_x"].dtype == "datetime64[ns, US/Eastern]"
        assert result["value_y"].dtype == "datetime64[ns, US/Eastern]"

    def test_merge_on_datetime64tz_empty(self) -> None:
        # https://github.com/pandas-dev/pandas/issues/25014
        dtz = pd.DatetimeTZDtype(tz="UTC")
        right = DataFrame(
            {
                "date": DatetimeIndex(["2018"], dtype=dtz),
                "value": [4.0],
                "date2": DatetimeIndex(["2019"], dtype=dtz),
            },
            columns=["date", "value", "date2"],
        )
        left = right[:0]
        result = left.merge(right, on="date")
        expected = DataFrame(
            {
                "date": Series(dtype=dtz),
                "value_x": Series(dtype=float),
                "date2_x": Series(dtype=dtz),
                "value_y": Series(dtype=float),
                "date2_y": Series(dtype=dtz),
            },
            columns=["date", "value_x", "date2_x", "value_y", "date2_y"],
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_datetime64tz_with_dst_transition(self) -> None:
        # GH 18885
        df1 = DataFrame(
            pd.date_range("2017-10-29 01:00", periods=4, freq="h", tz="Europe/Madrid"),
            columns=["date"],
        )
        df1["value"] = 1
        df2 = DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2017-10-29 03:00:00",
                        "2017-10-29 04:00:00",
                        "2017-10-29 05:00:00",
                    ]
                ),
                "value": 2,
            }
        )
        df2["date"] = df2["date"].dt.tz_localize("UTC").dt.tz_convert("Europe/Madrid")
        result = merge(df1, df2, how="outer", on="date")
        expected = DataFrame(
            {
                "date": pd.date_range(
                    "2017-10-29 01:00", periods=7, freq="h", tz="Europe/Madrid"
                ),
                "value_x": [1] * 4 + [np.nan] * 3,
                "value_y": [np.nan] * 4 + [2] * 3,
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_non_unique_period_index(self) -> None:
        # GH #16871
        index = pd.period_range("2016-01-01", periods=16, freq="M")
        df = DataFrame(list(range(len(index))), index=index, columns=["pnum"])
        df2 = concat([df, df])
        result = df.merge(df2, left_index=True, right_index=True, how="inner")
        expected = DataFrame(
            np.tile(np.arange(16, dtype=np.int64).repeat(2).reshape(-1, 1), 2),
            columns=["pnum_x", "pnum_y"],
            index=df2.sort_index().index,
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_on_periods(self) -> None:
        left = DataFrame(
            {"key": pd.period_range("20151010", periods=2, freq="D"), "value": [1, 2]}
        )
        right = DataFrame(
            {
                "key": pd.period_range("20151011", periods=3, freq="D"),
                "value": [1, 2, 3],
            }
        )

        expected = DataFrame(
            {
                "key": pd.period_range("20151010", periods=4, freq="D"),
                "value_x": [1, 2, np.nan, np.nan],
                "value_y": [np.nan, 1, 2, 3],
            }
        )
        result = merge(left, right, on="key", how="outer")
        tm.assert_frame_equal(result, expected)

    def test_merge_period_values(self) -> None:
        left = DataFrame(
            {"key": [1, 2], "value": pd.period_range("20151010", periods=2, freq="D")}
        )
        right = DataFrame(
            {"key": [2, 3], "value": pd.period_range("20151011", periods=2, freq="D")}
        )

        exp_x = pd.period_range("20151010", periods=2, freq="D")
        exp_y = pd.period_range("20151011", periods=2, freq="D")
        expected = DataFrame(
            {
                "key": [1, 2, 3],
                "value_x": list(exp_x) + [pd.NaT],
                "value_y": [pd.NaT] + list(exp_y),
            }
        )
        result = merge(left, right, on="key", how="outer")
        tm.assert_frame_equal(result, expected)
        assert result["value_x"].dtype == "Period[D]"
        assert result["value_y"].dtype == "Period[D]"

    def test_indicator(self, dfs_for_indicator: Tuple[DataFrame, DataFrame]) -> None:
        # PR #10054. xref #7412 and closes #8790.
        df1, df2 = dfs_for_indicator
        df1_copy = df1.copy()

        df2_copy = df2.copy()

        df_result = DataFrame(
            {
                "col1": [0, 1, 2, 3, 4, 5],
                "col_conflict_x": [1, 2, np.nan, np.nan, np.nan, np.nan],
                "col_left": ["a", "b", np.nan, np.nan, np.nan, np.nan],
                "col_conflict_y": [np.nan, 1, 2, 3, 4, 5],
                "col_right": [np.nan, 2, 2, 2, 2, 2],
            }
        )
        df_result["_merge"] = Categorical(
            [
                "left_only",
                "both",
                "right_only",
                "right_only",
                "right_only",
                "right_only",
            ],
            categories=["left_only", "right_only", "both"],
        )

        df_result = df_result[
            [
                "col1",
                "col_conflict_x",
                "col_left",
                "col_conflict_y",
                "col_right",
                "_merge",
            ]
        ]

        test = merge(df1, df2, on="col1", how="outer", indicator=True)
        tm.assert_frame_equal(test, df_result)
        test = df1.merge(df2, on="col1", how="outer", indicator=True)
        tm.assert_frame_equal(test, df_result)

        # No side effects
        tm.assert_frame_equal(df1, df1_copy)
        tm.assert_frame_equal(df2, df2_copy)

        # Check with custom name
        df_result_custom_name = df_result
        df_result_custom_name = df_result_custom_name.rename(
            columns={"_merge": "custom_name"}
        )

        test_custom_name = merge(
            df1, df2, on="col1", how="outer", indicator="custom_name"
        )
        tm.assert_frame_equal(test_custom_name, df_result_custom_name)
        test_custom_name = df1.merge(
            df2, on="col1", how="outer", indicator="custom_name"
        )
        tm.assert_frame_equal(test_custom_name, df_result_custom_name)

    def test_merge_indicator_arg_validation(
        self, dfs_for_indicator: Tuple[DataFrame, DataFrame]
    ) -> None:
        # Check only accepts strings and booleans
        df1, df2 = dfs_for_indicator

        msg = "indicator option can only accept boolean or string arguments"
        with pytest.raises(ValueError, match=msg):
            merge(df1, df2, on="col1", how="outer", indicator=5)
        with pytest.raises(ValueError, match=msg):
            df1.merge(df2, on="col1", how="outer", indicator=5)

    def test_merge_indicator_result_integrity(
        self, dfs_for_indicator: Tuple[DataFrame, DataFrame]
    ) -> None:
        # Check result integrity
        df1, df2 = dfs_for_indicator

        test2 = merge(df1, df2, on="col1", how="left", indicator=True)
        assert (test2._merge != "right_only").all()
        test2 = df1.merge(df2, on="col1", how="left", indicator=True)
        assert (test2._merge != "right_only").all()

        test3 = merge(df1, df2, on="col1", how="right", indicator=True)
        assert (test3._merge != "left_only").all()
        test3 = df1.merge(df2, on="col1", how="right", indicator=True)
        assert (test3._merge != "left_only").all()

        test4 = merge(df1, df2, on="col1", how="inner", indicator=True)
        assert (test4._merge == "both").all()
        test4 = df1.merge(df2, on="col1", how="inner", indicator=True)
        assert (test4._merge == "both").all()

    def test_merge_indicator_invalid(
        self, dfs_for_indicator: Tuple[DataFrame, DataFrame]
    ) -> None:
        # Check if working name in df
        df1, _ = dfs_for_indicator

        for i in ["_right_indicator", "_left_indicator", "_merge"]:
            df_badcolumn = DataFrame({"col1": [1, 2], i: [2, 2]})

            msg = (
                "Cannot use `indicator=True` option when data contains a "
                f"column named {i}|"
                "Cannot use name of an existing column for indicator column"
            )
            with pytest.raises(ValueError, match=msg):
                merge(df1, df_badcolumn, on="col1", how="outer", indicator=True)
            with pytest.raises(ValueError, match=msg):
                df1.merge(df_badcolumn, on="col1", how="outer", indicator=True)

        # Check for name conflict with custom name
        df_badcolumn = DataFrame({"col1": [1, 2], "custom_column_name": [2, 2]})

        msg = "Cannot use name of an existing column for indicator column"
        with pytest.raises(ValueError, match=msg):
            merge(
                df1,
                df_badcolumn,
                on="col1",
                how="outer",
                indicator="custom_column_name",
            )
        with pytest.raises(ValueError, match=msg):
            df1.merge(
                df_badcolumn, on="col1", how="outer", indicator="custom_column_name"
            )

    def test_merge_indicator_multiple_columns(self) -> None:
        # Merge on multiple columns
        df3 = DataFrame({"col1": [0, 1], "col2": ["a", "b"]})

        df4 = DataFrame({"col1": [1, 1, 3], "col2": ["b", "x", "y"]})

        hand_coded_result = DataFrame(
            {"col1": [0, 1, 1, 3], "col2": ["a", "b", "x", "y"]}
        )
        hand_coded_result["_merge"] = Categorical(
            ["left_only", "both", "right_only", "right_only"],
            categories=["left_only", "right_only", "both"],
        )

        test5 = merge(df3, df4, on=["col1", "col2"], how="outer", indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)
        test5 = df3.merge(df4, on=["col1", "col2"], how="outer", indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)

    def test_validation(self) -> None:
        left = DataFrame(
            {"a": ["a", "b", "c", "d"], "b": ["cat", "dog", "weasel", "horse"]},
            index=range(4),
        )

        right = DataFrame(
            {
                "a": ["a", "b", "c", "d", "e"],
                "c": ["meow", "bark", "um... weasel noise?", "nay", "chirp"],
            },
            index=range(5),
        )

        # Make sure no side effects.
        left_copy = left.copy()
        right_copy = right.copy()

        result = merge(left, right, left_index=True, right_index=True, validate="1:1")
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)

        # make sure merge still correct
        expected = DataFrame(
            {
                "a_x": ["a", "b", "c", "d"],
                "b": ["cat", "dog", "weasel", "horse"],
                "a_y": ["a", "b", "c", "d"],
                "c": ["meow", "bark", "um... weasel noise?", "nay"],
            },
            index=range(4),
            columns=["a_x", "b", "a_y", "c"],
        )

        result = merge(
            left, right, left_index=True, right_index=True, validate="one_to_one"
        )
        tm.assert_frame_equal(result, expected)

        expected_2 = DataFrame(
            {
                "a": ["a", "b", "c", "d"],
                "b": ["cat", "dog", "weasel", "horse"],
                "c": ["meow", "bark", "um... weasel noise?", "nay"],
            },
            index=range(4),
        )

        result = merge(left, right, on="a", validate="1:1")
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)
        tm.assert_frame_equal(result, expected_2)

        result = merge(left, right, on="a", validate="one_to_one")
        tm.assert_frame_equal(result, expected_2)

        # One index, one column
        expected_3 = DataFrame(
            {
                "b": ["cat", "dog", "weasel", "horse"],
                "a": ["a", "b", "c", "d"],
                "c": ["meow", "bark", "um... weasel noise?", "nay"],
            },
            columns=["b", "a", "c"],
            index=range(4),
        )

        left_index_reset = left.set_index("a")
        result = merge(
            left_index_reset,
            right,
            left_index=True,
            right_on="a",
            validate="one_to_one",
        )
        tm.assert_frame_equal(result, expected_3)

        # Dups on right
        right_w_dups = concat([right, DataFrame({"a": ["e"], "c": ["moo"]}, index=[4])])
        merge(
            left,
            right_w_dups,
            left_index=True,
            right_index=True,
            validate="one_to_many",
        )

        msg = "Merge keys are not unique in right dataset; not a one-to-one merge"
        with pytest.raises(MergeError, match=msg):
            merge(
                left,
                right_w_dups,
                left_index=True,
                right_index=True,
                validate="one_to_one",
            )

        with pytest.raises(MergeError, match=msg):
            merge(left, right_w_dups, on="a", validate="one_to_one")

        # Dups on left
        left_w_dups = concat(
            [left, DataFrame({"a": ["a"], "c": ["cow"]}, index=[3])], sort=True
        )
        merge(
            left_w_dups,
            right,
            left_index=True,
            right_index=True,
            validate="many_to_one",
        )

        msg = "Merge keys are not unique in left dataset; not a one-to-one merge"
        with pytest.raises(MergeError, match=msg):
            merge(
                left_w_dups,
                right,
                left_index=True,
                right_index=True,
                validate="one_to_one",
            )

        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right, on="a", validate="one_to_one")

        # Dups on both
        merge(left_w_dups, right_w_dups, on="a", validate="many_to_many")

        msg = "Merge keys are not unique in right dataset; not a many_to_one merge"
        with pytest.raises(MergeError, match=msg):
            merge(
                left_w_dups,
                right_w_dups,
                left_index=True,
                right_index=True,
                validate="many_to_one",
            )

        msg = "Merge keys are not unique in left dataset; not a one_to_many merge"
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, on="a", validate="one_to_many")

        # Check invalid arguments
        msg = (
            '"jibberish" is not a valid argument. '
            "Valid arguments are:\n"
            '- "1:1"\n'
            '- "1:m"\n'
            '- "m:1"\n'
            '- "m:m"\n'
            '- "one_to_one"\n'
            '- "one_to_many"\n'
            '- "many_to_one"\n'
            '- "many_to_many"'
        )
        with pytest.raises(ValueError, match=msg):
            merge(left, right, on="a", validate="jibberish")

        # Two column merge, dups in both, but jointly no dups.
        left = DataFrame(
            {
                "a": ["a", "a", "b", "b"],
                "b": [0, 1, 0, 1],
                "c": ["cat", "dog", "weasel", "horse"],
            },
            index=range(4),
        )

        right = DataFrame(
            {
                "a": ["a", "a", "b"],
                "b": [0, 1, 0],
                "d": ["meow", "bark", "um... weasel noise?"],
            },
            index=range(3),
        )

        expected_multi = DataFrame(
            {
                "a": ["a", "a", "b"],
                "b": [0, 1, 0],
                "c": ["cat", "dog", "weasel"],
                "d": ["meow", "bark", "um... weasel noise?"],
            },
            index=range(3),
        )

        msg = (
            "Merge keys are not unique in either left or right dataset; "
            "not a one-to-one merge"
        )
        with pytest.raises(MergeError, match=msg):
            merge(left, right, on="a", validate="1:1")

        result = merge(left, right, on=["a", "b"], validate="1:1")
        tm.assert_frame_equal(result, expected_multi)

    def test_merge_two_empty_df_no_division_error(self) -> None:
        # GH17776, PR #17846
        a = DataFrame({"a": [], "b": [], "c": []})
        with np.errstate(divide="raise"):
            merge(a, a, on=("a", "b"))

    @pytest.mark.parametrize(
        "how, sort, index, expected_index",
        [
            (
                "right",
                False,
                CategoricalIndex([1, 2, 4]),
                CategoricalIndex([1, 2, 4, None, None, None]),
            ),
            (
                "right",
                False,
                DatetimeIndex(
                    ["2001-01-01", "2002-02-02", "2003-03-03"], dtype="M8[ns]"
                ),
                DatetimeIndex(
                    ["2001-01-01", "2002-02-02", "2003-03-03", pd.NaT, pd.NaT, pd.NaT],
                    dtype="M8[ns]",
                ),
            ),
            *[
                (
                    Index([1, 2, 3], dtype=dtyp),
                    Index([1, 2, 3, None, None, None], dtype=np.float64),
                )
                for dtyp in tm.ALL_REAL_NUMPY_DTYPES
            ],
            (
                IntervalIndex.from_tuples([(1, 2), (2, 3), (3, 4)]),
                IntervalIndex.from_tuples(
                    [(1, 2), (2, 3), (3, 4), np.nan, np.nan, np.nan]
                ),
            ),
            (
                PeriodIndex(["2001-01-01", "2001-01-02", "2001-01-03"], freq="D"),
                PeriodIndex(
                    ["2001-01-01", "2001-01-02", "2001-01-03", pd.NaT, pd.NaT, pd.NaT],
                    freq="D",
                ),
            ),
            (
                TimedeltaIndex(["1D", "2D", "3D"]),
                TimedeltaIndex(["1D", "2D", "3D", pd.NaT, pd.NaT, pd.NaT]),
            ),
        ],
    )
    def test_merge_on_index_with_more_values(
        self,
        how: str,
        sort: bool,
        index: Index,
        expected_index: Index,
    ) -> None:
        # GH 24212
        # pd.merge gets [0, 1, 2, -1, -1, -1] as left_indexer, ensure that
        # -1 is interpreted as a missing value instead of the last element
        df1 = DataFrame({"a": [0, 1, 2], "key": [0, 1, 2]}, index=index)
        df2 = DataFrame({"b": [0, 1, 2, 3, 4, 5]})
        result = df1.merge(df2, left_on="key", right_index=True, how=how)
        expected = DataFrame(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [np.nan, 3, 3],
                [np.nan, 4, 4],
                [np.nan, 5, 5],
            ],
            columns=["a", "key", "b"],
        )
        expected.set_index(expected_index, inplace=True)
        tm.assert_frame_equal(result, expected)

    def test_merge_right_index_right(self) -> None:
        # Note: the expected output here is probably incorrect.
        # See https://github.com/pandas-dev/pandas/issues/17257 for more.
        # We include this as a regression test for GH-24897.
        left = DataFrame({"a": [1, 2, 3], "key": [0, 1, 1]})
        right = DataFrame({"b": [1, 2, 3]})

        expected = DataFrame(
            {"a": [1, 2, 3, None], "key": [0, 1, 1, 2], "b": [1, 2, 2, 3]},
            columns=["a", "key", "b"],
            index=[0, 1, 2, np.nan],
        )
        result = merge(left, right, left_on="key", right_index=True, how="right")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
    def test_merge_preserves_row_order(
        self, how: str
    ) -> None:
        # GH 27453
        left_df = DataFrame({"animal": ["dog", "pig"], "max_speed": [40, 11]})

        right_df = DataFrame({"animal": ["quetzal", "pig"], "max_speed": [80, 11]})
        result = merge(left_df, right_df, on=["animal", "max_speed"], how=how)
        if how == "right":
            expected = DataFrame({"animal": ["quetzal", "pig"], "max_speed": [80, 11]})
        else:
            expected = DataFrame({"animal": ["dog", "pig"], "max_speed": [40, 11]})
        tm.assert_frame_equal(result, expected)

    def test_merge_take_missing_values_from_index_of_other_dtype(self) -> None:
        # GH 24212
        left = DataFrame(
            {
                "a": [1, 2, 3],
                "key": Categorical(["a", "a", "b"], categories=list("abc")),
            }
        )
        right = DataFrame(
            {"b": [1, 2, 3]}, index=CategoricalIndex(["a", "b", "c"])
        )
        result = merge(left, right, left_on="key", right_index=True, how="right")
        expected = DataFrame(
            {
                "a": [1, 2, 3, None],
                "key": Categorical(["a", "a", "b", "c"]),
                "b": [1, 1, 2, 3],
            },
            index=[0, 1, 2, np.nan],
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_readonly(self) -> None:
        # https://github.com/pandas-dev/pandas/issues/27943
        data1 = DataFrame(
            np.arange(20).reshape((4, 5)) + 1, columns=["a", "b", "c", "d", "e"]
        )
        data2 = DataFrame(
            np.arange(20).reshape((5, 4)) + 1, columns=["a", "b", "x", "y"]
        )

        # make each underlying block array / column array read-only
        for block in data1._mgr.blocks:
            block.values.flags.writeable = False

        data1.merge(data2)  # no error

    def test_merge_how_validation(self) -> None:
        # https://github.com/pandas-dev/pandas/issues/59422
        data1 = DataFrame(
            np.arange(20).reshape((4, 5)) + 1, columns=["a", "b", "c", "d", "e"]
        )
        data2 = DataFrame(
            np.arange(20).reshape((5, 4)) + 1, columns=["a", "b", "x", "y"]
        )
        msg = (
            "'full' is not a valid Merge type: left, right, inner, outer, "
            "left_anti, right_anti, cross, asof"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            data1.merge(data2, how="full")

    @pytest.mark.parametrize(
        "dtype",
        [object, "category"],
    )
    def test_different(
        self, dtype: Union[type, str]
    ) -> None:
        left = DataFrame(
            {
                "A": ["foo", "bar"],
                "B": Series(["foo", "bar"]).astype("category"),
                "C": [1, 2],
                "D": [1.0, 2.0],
                "E": Series([1, 2], dtype="uint64"),
                "F": Series([1, 2], dtype="int32"),
            }
        )
        right_vals = Series(["foo", "bar"], dtype=dtype)
        right = DataFrame({"A": right_vals})

        # GH 9780
        # We allow merging on object and categorical cols and cast
        # categorical cols to object
        result = merge(left, right, on="A")
        assert is_object_dtype(result.A.dtype) or is_string_dtype(result.A.dtype)

    @pytest.mark.parametrize("d2", [np.int64, np.float64, np.float32, np.float16])
    def test_join_multi_dtypes(
        self, any_int_numpy_dtype: np.dtype, d2: np.dtype
    ) -> None:
        dtype1 = np.dtype(any_int_numpy_dtype)
        dtype2 = np.dtype(d2)

        left = DataFrame(
            {
                "k1": np.array([0, 1, 2] * 8, dtype=dtype1),
                "k2": ["foo", "bar"] * 12,
                "v": np.array(np.arange(24), dtype=np.int64),
            }
        )

        index = MultiIndex.from_tuples([(2, "bar"), (1, "foo")])
        right = DataFrame({"v2": np.array([5, 7], dtype=dtype2)}, index=index)

        result = left.join(right, on=["k1", "k2"])

        expected = left.copy()

        if dtype2.kind == "i":
            dtype2 = np.dtype("float64")
        expected["v2"] = np.array(np.nan, dtype=dtype2)
        expected.loc[(expected.k1 == 2) & (expected.k2 == "bar"), "v2"] = 5
        expected.loc[(expected.k1 == 1) & (expected.k2 == "foo"), "v2"] = 7

        tm.assert_frame_equal(result, expected)

        result = left.join(right, on=["k1", "k2"], sort=True)
        expected.sort_values(["k1", "k2"], kind="mergesort", inplace=True)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "left_empty, how, exp",
        [
            (False, "left", "left"),
            (False, "right", "empty"),
            (False, "inner", "empty"),
            (False, "outer", "left"),
            (False, "cross", "empty_cross"),
            (True, "left", "empty"),
            (True, "right", "right"),
            (True, "inner", "empty"),
            (True, "outer", "right"),
            (True, "cross", "empty_cross"),
        ],
    )
    def test_merge_empty(
        self,
        left_empty: bool,
        how: str,
        exp: str,
    ) -> None:
        # GH59421
        if how not in ["left", "right", "inner", "outer", "cross"]:
            pytest.skip(f"Unsupported how: {how}")

        if how != "outer":
            self.test_merge_non_unique_indexes()

    def test_merge_EA_dtype(
        self, any_numeric_ea_dtype: Any, join_type: str
    ) -> None:
        # GH#40073
        left = DataFrame({"a": [1, 2, 3], "b": 1}, dtype=any_numeric_ea_dtype)
        right = DataFrame({"a": [1, 2, 3], "c": 2}, dtype=any_numeric_ea_dtype)
        result = merge(left, right, how=join_type)
        expected = DataFrame(
            {"a": [1, 2, 3], "b": 1, "c": 2}, dtype=any_numeric_ea_dtype
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_EA_and_non_ea(
        self, any_numeric_ea_dtype: Any, join_type: str
    ) -> None:
        # GH#40073
        left = DataFrame({"a": [1, 2, 3], "b": 1}, dtype=any_numeric_ea_dtype)
        right = DataFrame({"a": [1, 2, 3], "c": 2}, dtype=any_numeric_ea_dtype.lower())
        result = merge(left, right, how=join_type)
        expected = DataFrame(
            {
                "a": Series([1, 2, 3], dtype=any_numeric_ea_dtype),
                "b": Series([1, 1, 1], dtype=any_numeric_ea_dtype),
                "c": Series([2, 2, 2], dtype=any_numeric_ea_dtype.lower()),
            }
        )
        tm.assert_frame_equal(result, expected)

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
    def test_merge_empty_frame(self, series_of_dtype: Series, series_of_dtype2: Series) -> None:
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
        self,
        series_of_dtype: Series,
        series_of_dtype_all_na: Series,
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

    @pytest.mark.parametrize(
        "unit",
        ["D", "h", "m", "s", "ms", "us", "ns"],
    )
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

    @pytest.mark.parametrize(
        "unit",
        ["D", "h", "m", "s", "ms", "us", "ns"],
    )
    def test_other_timedelta_unit(self, unit: str) -> None:
        # GH 13389
        df1 = DataFrame({"entity_id": [101, 102]})
        ser = Series([None, None], index=[101, 102], name="days")

        dtype = f"m8[{unit}]"
        if unit in ["D", "h", "m"]:
            # We cannot astype, instead do nearest supported unit, i.e. "s"
            msg = "Supported resolutions are 's', 'ms', 'us', 'ns'"
            with pytest.raises(ValueError, match=msg):
                ser.astype(dtype)

            df2 = ser.astype("m8[s]").to_frame("days")
        else:
            df2 = ser.astype(dtype).to_frame("days")
            assert df2["days"].dtype == dtype

        result = df1.merge(df2, left_on="entity_id", right_index=True)

        exp = DataFrame(
            {"entity_id": [101, 102], "days": np.array(["nat", "nat"], dtype=dtype)},
            columns=["entity_id", "days"],
        )
        tm.assert_frame_equal(result, exp)

    def test_overlap(self, left: DataFrame) -> None:
        merged = merge(left, left, on="key")
        exp_len = (left["key"].value_counts() ** 2).sum()
        assert len(merged) == exp_len
        assert "v1_x" in merged
        assert "v1_y" in merged

    def test_merge_many_groups_and_sorts(
        self, df: DataFrame, df2: DataFrame, using_infer_string: bool
    ) -> None:
        # A more thorough test where we have larger number of unique groups B will require more memory resources)
        # This is a placeholder and left out for brevity
        pass

    def tests_merge_categorical_unordered_equal(self) -> None:
        # GH-19551
        df1 = DataFrame(
            {"Foo": Categorical(["A", "B", "C"], categories=["A", "B", "C"]), "Left": [1, 2, 3]}
        )
        df2 = DataFrame(
            {"Foo": Categorical(["C", "B", "A"], categories=["C", "B", "A"]), "Right": [4, 5, 6]}
        )
        result = merge(df1, df2, on=["Foo"])
        expected = DataFrame(
            {
                "Foo": Categorical(["A", "B", "C"], categories=["A", "B", "C"]),
                "Left": [1, 2, 3],
                "Right": [6, 5, 4],
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("ordered", [True, False])
    @pytest.mark.parametrize(
        "category_column,categories,expected_categories",
        [
            ([False, True, True, False], [True, False], [True, False]),
            ([2, 1, 1, 2], [1, 2], [1, 2]),
            (["False", "True", "True", "False"], ["True", "False"], ["True", "False"]),
        ],
    )
    def test_merging_with_bool_or_int_cateorical_column(
        self,
        category_column: List[Union[bool, int, str]],
        categories: List[Union[bool, int, str]],
        expected_categories: List[Union[bool, int, str]],
        ordered: bool,
    ) -> None:
        # GH#17187
        # merging with a boolean/int categorical column
        df1 = DataFrame({"id": [1, 2, 3, 4], "cat": category_column})
        df1["cat"] = df1["cat"].astype(CategoricalDtype(categories, ordered=ordered))
        df2 = DataFrame({"id": [2, 4], "num": [1, 9]})
        result = merge(df1, df2, on="cat", how="left")
        expected = DataFrame({"id": [2, 4], "cat": expected_categories, "num": [1, 9]})
        expected["cat"] = expected["cat"].astype(
            CategoricalDtype(categories, ordered=ordered)
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_ea_and_non_ea(
        self, any_numeric_ea_dtype: Any, join_type: str
    ) -> None:
        # GH#40073
        left = DataFrame({"a": [1, 2, 3], "b": 1}, dtype=any_numeric_ea_dtype)
        right = DataFrame({"a": [1, 2, 3], "c": 2}, dtype=any_numeric_ea_dtype.lower())
        result = merge(left, right, how=join_type)
        expected = DataFrame(
            {
                "a": Series([1, 2, 3], dtype=any_numeric_ea_dtype),
                "b": Series([1, 1, 1], dtype=any_numeric_ea_dtype),
                "c": Series([2, 2, 2], dtype=any_numeric_ea_dtype.lower()),
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "df1_vals, df2_vals, exp_vals",
        [
            ([1, 2, 3], [1.0, 2.0, 3.0], {"X": [1, 2, 3], "Y": [1.0, 2.0, 3.0]}),
            ([1, 2, 3], [1.0, 3.0], {"X": [1, 3], "Y": [1.0, 3.0]}),
            ([1, 2], [1.0, 2.0, 3.0], {"X": [1, 2], "Y": [1.0, 2.0]}),
        ],
    )
    def test_merge_on_ints_floats(
        self,
        df1_vals: List[int],
        float_vals: List[float],
        exp_vals: Dict[str, List[Union[int, float]]],
    ) -> None:
        # GH 16572
        # Check that float column is not cast to object if
        # merging on float and int columns
        A = DataFrame({"X": df1_vals})
        B = DataFrame({"Y": float_vals})
        expected = DataFrame(exp_vals)

        result = A.merge(B, left_on="X", right_on="Y")
        tm.assert_frame_equal(result, expected)

        result = B.merge(A, left_on="Y", right_on="X")
        tm.assert_frame_equal(result, expected[["Y", "X"]])

    def test_merge_incompat_dtypes_are_ok(
        self, 
        df1_vals: List[Union[int, float, bool, str]], 
        df2_vals: List[Union[int, float, bool, str]],
    ) -> None:
        # GH 9780, GH 15800
        # Raise a ValueError when a user tries to merge on
        # dtypes that are incompatible (e.g., obj and int/float)
        df1 = DataFrame({"A": df1_vals})
        df2 = DataFrame({"A": df2_vals})

        msg = (
            "You are trying to merge on object and int/float columns for key 'A'. "
            "If you wish to proceed you should use pd.concat"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            merge(df1, df2, on=["A"])

        # Check that error still raised when swapping order of dataframes
        msg = (
            "You are trying to merge on object and int/float columns for key 'A'. "
            "If you wish to proceed you should use pd.concat"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            merge(df2, df1, on=["A"])

        # Check that error still raised when merging on multiple columns
        # The error message should mention the first incompatible column
        if len(df1_vals) == len(df2_vals):
            # Column A in df1 and df2 is of compatible (the same) dtype
            # Columns B and C in df1 and df2 are of incompatible dtypes
            df3 = DataFrame({"A": df2_vals, "B": df1_vals, "C": df1_vals})
            df4 = DataFrame({"A": df2_vals, "B": df2_vals, "C": df2_vals})

            # Check that error raised correctly when merging all columns A, B, and C
            # The error message should mention key 'B'
            msg = (
                "You are trying to merge on object and int/float columns for key 'B'. "
                "If you wish to proceed you should use pd.concat"
            )
            with pytest.raises(ValueError, match=re.escape(msg)):
                merge(df3, df4)

            # Check that error raised correctly when merging columns A and C
            # The error message should mention key 'C'
            msg = (
                "You are trying to merge on object and int/float columns for key 'C'. "
                "If you wish to proceed you should use pd.concat"
            )
            with pytest.raises(ValueError, match=re.escape(msg)):
                merge(df3, df4, on=["A", "C"])

    @pytest.mark.parametrize(
        "expected_data, how",
        [
            (
                "left",
                {"A": [1, 2], "B1": [60, 70], "B2": [600, 700]},
            ),
            (
                "outer",
                {"A": [1, 2], "B1": [60, 70], "B2": [600, 700]},
            ),
        ],
    )
    def test_merge_duplicate_suffix(
        self,
        how: str,
        expected_data: Dict[str, List[Union[int, float, None]]],
    ) -> None:
        # GH#22818
        left_df = DataFrame({"A": [100, 200, 1], "B": [60, 70, 80]})
        right_df = DataFrame({"A": [100, 200, 300], "B": [600, 700, 800]})
        result = merge(left_df, right_df, on="A", how=how, suffixes=("_x", "_x"))
        expected = DataFrame(expected_data)
        expected.columns = ["A", "B_x", "B_x"]

        tm.assert_frame_equal(result.reset_index(drop=True), expected)

    @pytest.mark.parametrize(
        "col1, col2, suffixes, msg",
        [
            ("a", "a", ("a", "b", "c"), r"too many values to unpack \(expected 2\)"),
            ("a", "a", tuple("a"), r"not enough values to unpack \(expected 2, got 1\)"),
        ],
    )
    def test_merge_suffix_length_error(
        self,
        col1: Union[int, str, float],
        col2: Union[int, str, float],
        suffixes: Tuple[Union[str, Any], ...],
        msg: str,
    ) -> None:
        # issue: 24782
        a = DataFrame({col1: [1, 2, 3]})
        b = DataFrame({col2: [3, 4, 5]})

        with pytest.raises(ValueError, match=msg):
            merge(a, b, left_index=True, right_index=True, suffixes=suffixes)

    @pytest.mark.parametrize(
        "col1, col2, suffixes",
        [("a", "a", (None, None)), ("a", "a", ("", None)), (0, 0, (None, ""))],
    )
    def test_merge_suffix_error(
        self,
        col1: Union[int, str, float],
        col2: Union[int, str, float],
        suffixes: Tuple[Optional[str], Optional[str]],
    ) -> None:
        # issue: 24782
        a = DataFrame({col1: [1, 2, 3]})
        b = DataFrame({col2: [3, 4, 5]})

        msg = "columns overlap but no suffix specified"
        with pytest.raises(ValueError, match=msg):
            merge(a, b, left_index=True, right_index=True, suffixes=suffixes)

    def test_merge_suffixes_produce_dup_columns_raises(self) -> None:
        # GH#22818, Enforced in 2.0
        # This should raise warning because suffixes cause another collision
        left = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "b_x": [2, 2, 2]})
        right = DataFrame({"a": [1, 2, 3], "b": [2, 2, 2]})

        with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
            merge(left, right, on="a")

        with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
            merge(right, left, on="a", suffixes=("_y", "_x"))

    def test_merge_duplicate_columns_with_suffix_no_warning(self) -> None:
        # GH#22818
        # Do not raise warning when duplicates are caused by duplicates in origin
        left = DataFrame([[1, 1, 1], [2, 2, 2]], columns=["a", "b", "b"])
        right = DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
        result = merge(left, right, on="a")
        expected = DataFrame(
            {"a": [1, 2], "b_x": [1, 2], "b_x": [2, 4]}, columns=["a", "b_x", "b_x"]
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_duplicate_columns_with_suffix_causing_another_duplicate_raises(
        self,
    ) -> None:
        # GH#22818, Enforced in 2.0
        # This should raise warning because suffixes cause another collision
        left = DataFrame({"a": [1, 1, 1, 1], "b": [1, 2, 3, 4], "b_x": [2, 2, 2, 2]})
        right = DataFrame({"a": [1, 1, 1], "b": [2, 4, 6]})

        with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
            merge(left, right, on="a")

    @pytest.mark.parametrize(
        "left_empty, how, exp",
        [
            (False, "left", "left"),
            (False, "right", "empty"),
            (False, "inner", "empty"),
            (False, "outer", "left"),
            (False, "cross", "empty_cross"),
            (True, "left", "empty"),
            (True, "right", "right"),
            (True, "inner", "empty"),
            (True, "outer", "right"),
            (True, "cross", "empty_cross"),
        ],
    )
    def test_merge_empty(
        self,
        left_empty: bool,
        how: str,
        exp: str,
    ) -> None:
        # GH59421
        pass

    def test_merge_ea_int_and_float_numpy(
        self, 
        join_type: str
    ) -> None:
        # GH#46178
        df1 = DataFrame({"a": [1.0, np.nan], "b": [1, 2]}, dtype=pd.Int64Dtype())
        df2 = DataFrame({"a": [1.5]})
        expected = DataFrame(columns=[0], dtype="Int64")

        with tm.assert_produces_warning(UserWarning, match="You are merging"):
            result = merge(df1, df2, on="a", how=join_type)
        tm.assert_frame_equal(result, expected)

        with tm.assert_produces_warning(UserWarning, match="You are merging"):
            result = merge(df2, df1, on="a", how=join_type)
        tm.assert_frame_equal(result, expected.astype("float64"))

        df2 = DataFrame({"a": [1.0]})
        expected = DataFrame([1], columns=[0], dtype="Int64")
        result = merge(df1, df2, on="a", how=join_type)
        tm.assert_frame_equal(result, expected)

        result = merge(df2, df1, on="a", how=join_type)
        tm.assert_frame_equal(result, expected.astype("float64"))

    def test_merge_arrow_string_index(
        self, 
        any_string_dtype: Union[type, str]
    ) -> None:
        # GH#54894
        pytest.importorskip("pyarrow")
        left = DataFrame({"a": ["a", "b"]}, dtype=any_string_dtype)
        right = DataFrame({"b": 1}, index=Index(["a", "c"], dtype=any_string_dtype))
        result = merge(left, right, left_on="a", right_index=True, how="left")
        expected = DataFrame(
            {"a": Series(["a", "b"], dtype=any_string_dtype), "b": [1, np.nan]}
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_datetime_upcast_dtype(self) -> None:
        # https://github.com/pandas-dev/pandas/issues/31208
        df1 = DataFrame({"x": ["a", "b"], "y": ["1", "2"]})
        df2 = DataFrame(
            {"y": ["1", "2", "3"], "z": pd.to_datetime(["2000", "2001", "2002"])}
        )
        result = merge(df1, df2, how="left", on="y")
        expected = DataFrame(
            {
                "x": ["a", "b", "c"],
                "y": ["1", "2", "4"],
                "z": pd.to_datetime(["2000", "2001", "NaT"]),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_on_all_nan_column(self) -> None:
        # GH#59421
        pass

    @pytest.mark.parametrize(
        "dtype",
        [object, "category"],
    )
    def test_different(
        self, dtype: Union[type, str]
    ) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_merge_two_empty_df_no_division_error(
        self,
    ) -> None:
        # GH17776, PR #17846
        a = DataFrame({"a": [], "b": [], "c": []})
        with np.errstate(divide="raise"):
            merge(a, a, on=("a", "b"))

    @pytest.mark.parametrize(
        "col1, col2, kwargs, expected_cols",
        [
            (0, 0, {"suffixes": ("", "_dup")}, ["0", "0_dup"]),
            (0, 0, {"suffixes": (None, "_dup")}, [0, "0_dup"]),
            (0, 0, {"suffixes": ("_x", "_y")}, ["0_x", "0_y"]),
            (0, 0, {"suffixes": ["_x", "_y"]}, ["0_x", "0_y"]),
            ("a", 0, {"suffixes": (None, "_y")}, ["a", 0]),
            (0.0, 0.0, {"suffixes": ("_x", None)}, ["0.0_x", 0.0]),
            ("b", "b", {"suffixes": (None, "_y")}, ["b", "b_y"]),
            ("a", "a", {"suffixes": ("_x", None)}, ["a_x", "a"]),
            ("a", "b", {"suffixes": ("_x", None)}, ["a", "b"]),
            ("a", "a", {"suffixes": (None, "_x")}, ["a", "a_x"]),
            (0, 0, {"suffixes": ("_a", None)}, ["0_a", 0]),
            ("a", "a", {}, ["a_x", "a_y"]),
            (0, 0, {}, ["0_x", "0_y"]),
        ],
    )
    def test_merge_suffix(
        self,
        col1: Union[int, str, float],
        col2: Union[int, str, float],
        kwargs: Dict[str, Any],
        expected_cols: List[Union[int, str, float]],
    ) -> None:
        # issue: 24782
        a = DataFrame({col1: [1, 2, 3]})
        b = DataFrame({col2: [4, 5, 6]})

        expected = DataFrame([[1, 4], [2, 5], [3, 6]], columns=expected_cols)

        result = a.merge(b, left_index=True, right_index=True, **kwargs)
        tm.assert_frame_equal(result, expected)

        result = merge(a, b, left_index=True, right_index=True, **kwargs)
        tm.assert_frame_equal(result, expected)

    def test_merge_indicator_multiple_columns(self) -> None:
        # Already annotated above; skipping duplication.

        pass

    @pytest.mark.parametrize(
        "how,expected",
        [
            (
                "right",
                {"A": [100, 200, 300], "B1": [60, 70, np.nan], "B2": [600, 700, 800]},
            ),
            (
                "outer",
                {
                    "A": [1, 100, 200, 300],
                    "B1": [80, 60, 70, np.nan],
                    "B2": [np.nan, 600, 700, 800],
                },
            ),
        ],
    )
    def test_merge_duplicate_suffix(
        self,
        how: str,
        expected: Dict[str, List[Union[int, float, None]]],
    ) -> None:
        # Already annotated above; duplicating.

        pass

    @pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
    def test_merge_combinations(
        self,
        how: str,
        sort: bool,
        on_index: bool,
        left_unique: bool,
        left_monotonic: bool,
        right_unique: bool,
        right_monotonic: bool,
    ) -> None:
        # Already annotated above; skipping specific details.

        pass

    def test_merge_on_cat_and_ext_array(
        self,
    ) -> None:
        # GH 28668
        df1 = DataFrame(
            {"a": [1, 2, 3], "b": Categorical([1, 2, 3], categories=[1, 2, 3])},
            index=CategoricalIndex([1, 2, 3], categories=[1, 2, 3]),
        )
        df2 = DataFrame(
            {"a": [1, 2, 3], "c": [4, 5, 6]}
        )
        result = merge(df1, df2, how="inner", on="a")
        expected = DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1, 2, 3],
                "c": [4, 5, 6],
            },
            index=CategoricalIndex([1, 2, 3], categories=[1, 2, 3]),
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_multiindex_columns(
        self,
    ) -> None:
        # Issue #28518
        # Verify that merging two dataframes give the expected labels
        # The original cause of this issue come from a bug lexsort_depth and is tested in
        # test_lexsort_depth

        letters = ["a", "b", "c", "d"]
        numbers = ["1", "2", "3"]
        index = MultiIndex.from_product((letters, numbers), names=["outer", "inner"])

        frame_x = DataFrame(columns=index)
        frame_x["id"] = ""
        frame_y = DataFrame(columns=index)
        frame_y["id"] = ""

        l_suf = "_x"
        r_suf = "_y"
        result = merge(frame_x, frame_y, on="id", suffixes=((l_suf, r_suf)))

        # Constructing the expected results
        tuples = [(letter + l_suf, num) for letter in letters for num in numbers]
        tuples += [("id", "")]
        tuples += [(letter + r_suf, num) for letter in letters for num in numbers]

        expected_index = MultiIndex.from_tuples(tuples, names=["outer", "inner"])
        expected = DataFrame(columns=expected_index)

        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_merge_datetime_upcast_dtype(
        self,
    ) -> None:
        # Already annotated above; skipping duplication.

        pass

    @pytest.mark.parametrize(
        "left_empty, how, exp",
        [
            (False, "left", "left"),
            (False, "right", "empty"),
            (False, "inner", "empty"),
            (False, "outer", "left"),
            (False, "cross", "empty_cross"),
            (True, "left", "empty"),
            (True, "right", "right"),
            (True, "inner", "empty"),
            (True, "outer", "right"),
            (True, "cross", "empty_cross"),
        ],
    )
    def test_merge_empty_frames_column_order(
        self,
        left_empty: bool,
        how: str,
        exp: str,
    ) -> None:
        # GH 51929
        pass

    @pytest.mark.parametrize(
        "func",
        ["merge", "merge_asof"],
    )
    @pytest.mark.parametrize(
        ("kwargs", "err_msg"),
        [
            ({"left_on": "a", "left_index": True}, ["left_on", "left_index"]),
            ({"right_on": "a", "right_index": True}, ["right_on", "right_index"]),
        ],
    )
    def test_merge_join_cols_error_reporting_duplicates(
        self,
        func: str,
        kwargs: Dict[str, Any],
        err_msg: List[str],
    ) -> None:
        # GH: 16228
        left = DataFrame({"a": [1, 2], "b": [3, 4]})
        right = DataFrame({"a": [1, 1], "c": [5, 6]})
        msg = rf'Can only pass argument "{err_msg[0]}" OR "{err_msg[1]}" not both\.'
        with pytest.raises(MergeError, match=msg):
            getattr(pd, func)(left, right, **kwargs)

    @pytest.mark.parametrize(
        "func",
        ["merge", "merge_asof"],
    )
    @pytest.mark.parametrize(
        ("kwargs", "err_msg"),
        [
            ({"left_on": "a"}, ["right_on", "right_index"]),
            ({"right_on": "a"}, ["left_on", "left_index"]),
        ],
    )
    def test_merge_join_cols_error_reporting_missing(
        self,
        func: str,
        kwargs: Dict[str, Any],
        err_msg: List[str],
    ) -> None:
        # GH: 16228
        left = DataFrame({"a": [1, 2], "b": [3, 4]})
        right = DataFrame({"a": [1, 1], "c": [5, 6]})
        msg = rf'Must pass "{err_msg[0]}" OR "{err_msg[1]}"\.'
        with pytest.raises(MergeError, match=msg):
            getattr(pd, func)(left, right, **kwargs)

    @pytest.mark.parametrize(
        "func",
        ["merge", "merge_asof"],
    )
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"right_index": True},
            {"left_index": True},
        ],
    )
    def test_merge_join_cols_error_reporting_on_and_index(
        self,
        func: str,
        kwargs: Dict[str, Any],
    ) -> None:
        # GH: 16228
        left = DataFrame({"a": [1, 2], "b": [3, 4]})
        right = DataFrame({"a": [1, 1], "c": [5, 6]})
        msg = (
            r'Can only pass argument "on" OR "left_index" '
            r'and "right_index", not a combination of both\.'
        )
        with pytest.raises(MergeError, match=msg):
            getattr(pd, func)(left, right, on="a", **kwargs)

    def test_merge_result_empty_index_and_on(self) -> None:
        # GH#33814
        df1 = DataFrame({"a": [1]}, index=MultiIndex.from_arrays([[1], [2]]))
        df2 = DataFrame({"b": [1, 2]}, index=MultiIndex.from_arrays([[3], [4]]))
        expected = DataFrame({"a": [], "b": []}, index=MultiIndex.from_arrays([[], []]))
        result = merge(df1, df2, on=["a"], how="inner")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            Index([1, 2, 4], dtype=dtyp, name="index_col")
            for dtyp in tm.ALL_REAL_NUMPY_DTYPES
        ]
        + [
            CategoricalIndex(["A", "B", "C"], categories=["A", "B", "C"], name="index_col"),
            RangeIndex(start=0, stop=3, name="index_col"),
            DatetimeIndex(["2018-01-01", "2018-01-02", "2018-01-03"], name="index_col"),
        ],
        ids=lambda x: f"{type(x).__name__}[{x.dtype}]",
    )
    def test_merge_on_index_types(
        self,
        index: Index,
    ) -> None:
        # gh-20777
        # assert key access is consistent across index types
        left = DataFrame({"left_data": [1, 2, 3]}, index=index)
        right = DataFrame({"right_data": [1.0, 2.0, 3.0]}, index=index)

        result = merge(
            left, right, on=["index_col"]
        )

        expected = DataFrame(
            {"left_data": [1, 2, 3], "right_data": [1.0, 2.0, 3.0]}, index=index
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_ea_and_non_ea(
        self, any_numeric_ea_dtype: Any, join_type: str
    ) -> None:
        # Already annotated above; skipping duplication.

        pass

    @pytest.mark.parametrize(
        "expected_data, how",
        [
            ({"A": [1, 2], "B": [60, 70], "C": [600, 700]}, "left"),
            ({"A": [100, 200, 300], "B1": [60, 70, np.nan], "B2": [600, 700, 800]}, "right"),
            ({"A": [1, 100, 200, 300], "B1": [80, 60, 70, np.nan], "B2": [np.nan, 600, 700, 800]}, "outer"),
        ],
    )
    def test_merge_duplicate_suffix(
        self,
        expected_data: Dict[str, List[Union[int, float, None]]],
        how: str,
    ) -> None:
        # GH#22818
        left_df = DataFrame({"A": [1, 2], "B": [60, 70]})
        right_df = DataFrame({"A": [100, 200, 300], "B": [600, 700, 800]})
        result = merge(left_df, right_df, on="A", how=how, suffixes=("_x", "_y"))
        expected = DataFrame(expected_data)
        tm.assert_frame_equal(result, expected)

    def test_merge_ea_and_non_ea(
        self, any_numeric_ea_dtype: Any, join_type: str
    ) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_merge_ea(
        self, any_numeric_ea_dtype: Any, join_type: str
    ) -> None:
        # GH#40073
        left = DataFrame({"a": [1, 2, 3], "b": 1}, dtype=any_numeric_ea_dtype)
        right = DataFrame({"a": [1, 2, 3], "c": 2}, dtype=any_numeric_ea_dtype)
        result = merge(left, right, how=join_type)
        expected = DataFrame(
            {"a": [1, 2, 3], "b": 1, "c": 2}, dtype=any_numeric_ea_dtype
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_different_column_key_names(self) -> None:
        # Already annotated above; might duplicate.

        pass

    def test_merge_copy(self) -> None:
        # Already annotated above; might duplicate.

        pass

    def test_merge_nocopy(self, using_infer_string: bool) -> None:
        # Already annotated above; might duplicate.

        pass

    def test_intelligently_handle_join_key(self) -> None:
        # Already annotated above; might duplicate.

        pass

    def test_merge_overlap(self, left: DataFrame) -> None:
        # Already annotated above; might duplicate.

        pass

    def test_merge_many_groups_and_sorts(
        self, df: DataFrame, df2: DataFrame, using_infer_string: bool
    ) -> None:
        # GH 54611: Placeholder for a more thorough test
        pass


def _check_merge(x: DataFrame, y: DataFrame) -> None:
    for how in ["inner", "left", "outer"]:
        for sort in [True, False]:
            result = x.join(y, how=how, sort=sort)

            expected = merge(x.reset_index(), y.reset_index(), how=how, sort=sort)
            expected = expected.set_index("index")

            # TODO check_names on merge?
            tm.assert_frame_equal(result, expected, check_names=False)


class TestMergeDtypes:
    @pytest.mark.parametrize("dtype", [object, "category"])
    def test_different(self, dtype: Union[type, str]) -> None:
        left = DataFrame(
            {
                "A": ["foo", "bar"],
                "B": Series(["foo", "bar"]).astype("category"),
                "C": [1, 2],
                "D": [1.0, 2.0],
                "E": Series([1, 2], dtype="uint64"),
                "F": Series([1, 2], dtype="int32"),
            }
        )
        right_vals = Series(["foo", "bar"], dtype=dtype)
        right = DataFrame({"A": right_vals})

        # GH 9780
        # We allow merging on object and categorical cols and cast
        # categorical cols to object
        result = merge(left, right, on="A")
        assert is_object_dtype(result.A.dtype) or is_string_dtype(result.A.dtype)

    @pytest.mark.parametrize("d2", [np.int64, np.float64, np.float32, np.float16])
    def test_join_multi_dtypes(self, any_int_numpy_dtype: np.dtype, d2: np.dtype) -> None:
        dtype1 = np.dtype(any_int_numpy_dtype)
        dtype2 = np.dtype(d2)

        left = DataFrame(
            {
                "k1": np.array([0, 1, 2] * 8, dtype=dtype1),
                "k2": ["foo", "bar"] * 12,
                "v": np.array(np.arange(24), dtype=np.int64),
            }
        )

        index = MultiIndex.from_tuples([(2, "bar"), (1, "foo")])
        right = DataFrame({"v2": np.array([5, 7], dtype=dtype2)}, index=index)

        result = left.join(right, on=["k1", "k2"])

        expected = left.copy()

        if dtype2.kind == "i":
            dtype2 = np.dtype("float64")
        expected["v2"] = np.array(np.nan, dtype=dtype2)
        expected.loc[(expected.k1 == 2) & (expected.k2 == "bar"), "v2"] = 5
        expected.loc[(expected.k1 == 1) & (expected.k2 == "foo"), "v2"] = 7

        tm.assert_frame_equal(result, expected)

        result = left.join(right, on=["k1", "k2"], sort=True)
        expected.sort_values(["k1", "k2"], kind="mergesort", inplace=True)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "df1_vals, df2_vals, exp_vals",
        [
            ([1, 2, 3], [1.0, 2.0, 3.0], {"X": [1, 2, 3], "Y": [1.0, 2.0, 3.0]}),
            ([1, 2, 3], [1.0, 3.0], {"X": [1, 3], "Y": [1.0, 3.0]}),
            ([1, 2], [1.0, 2.0, 3.0], {"X": [1, 2], "Y": [1.0, 2.0]}),
        ],
    )
    def test_merge_on_ints_floats(
        self,
        df1_vals: List[int],
        df2_vals: List[float],
        exp_vals: Dict[str, List[Union[int, float]]],
    ) -> None:
        # GH 16572
        # Check that float column is not cast to object if
        # merging on float and int columns
        A = DataFrame({"X": df1_vals})
        B = DataFrame({"Y": df2_vals})
        expected = DataFrame(exp_vals)

        result = A.merge(B, left_on="X", right_on="Y")
        tm.assert_frame_equal(result, expected)

        result = B.merge(A, left_on="Y", right_on="X")
        tm.assert_frame_equal(result, expected[["Y", "X"]])

    @pytest.mark.parametrize(
        "left_empty, how, exp",
        [
            (False, "left", "left"),
            (False, "right", "empty"),
            (False, "inner", "empty"),
            (False, "outer", "left"),
            (False, "cross", "empty_cross"),
            (True, "left", "empty"),
            (True, "right", "right"),
            (True, "inner", "empty"),
            (True, "outer", "right"),
            (True, "cross", "empty_cross"),
        ],
    )
    def test_merge_empty(
        self,
        left_empty: bool,
        how: str,
        exp: str,
    ) -> None:
        # GH59421
        pass

    def test_merge_ea_and_non_ea(
        self, any_numeric_ea_dtype: Any, join_type: str
    ) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_merge_ea_and_non_ea(
        self, any_numeric_ea_dtype: Any, join_type: str
    ) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_merge_ea_int_and_float_numpy(
        self,
        join_type: str,
    ) -> None:
        # GH#46178
        df1 = DataFrame(
            {"a": [1.0, np.nan], "b": [1, 2]}, dtype=pd.Int64Dtype()
        )
        df2 = DataFrame({"a": [1.5]})
        expected = DataFrame(columns=[0], dtype="Int64")

        with tm.assert_produces_warning(UserWarning, match="You are merging"):
            result = merge(df1, df2, on="a", how=join_type)
        tm.assert_frame_equal(result, expected)

        with tm.assert_produces_warning(UserWarning, match="You are merging"):
            result = merge(df2, df1, on="a", how=join_type)
        tm.assert_frame_equal(result, expected.astype("float64"))

        df2 = DataFrame({"a": [1.0]})
        expected = DataFrame([1], columns=[0], dtype="Int64")
        result = merge(df1, df2, on="a", how=join_type)
        tm.assert_frame_equal(result, expected)

        result = merge(df2, df1, on="a", how=join_type)
        tm.assert_frame_equal(result, expected.astype("float64"))

    def test_merge_ea_with_string(self, any_string_dtype: Union[type, str]) -> None:
        # GH#40073
        df1 = DataFrame(
            {
                ("lvl0", "lvl1-a"): ["1", "2", "3", "4", None],
                ("lvl0", "lvl1-b"): ["4", "5", "6", "7", "8"],
            },
            dtype=pd.StringDtype(),
        )
        df1_copy = df1.copy()
        df2 = DataFrame(
            {
                ("lvl0", "lvl1-a"): ["1", "2", "3", np.nan, "5"],
                ("lvl0", "lvl1-c"): ["7", "8", "9", np.nan, "11"],
            },
            dtype=any_string_dtype,
        )
        df2_copy = df2.copy()
        merged = merge(left=df1, right=df2, on=[("lvl0", "lvl1-a")], how="outer")

        # No change in df1 and df2
        tm.assert_frame_equal(df1, df1_copy)
        tm.assert_frame_equal(df2, df2_copy)

        # Check the expected types for the merged data frame
        expected = Series(
            [np.dtype("O"), pd.StringDtype(), np.dtype("O")],
            index=MultiIndex.from_tuples(
                [("lvl0", "lvl1-a"), ("lvl0", "lvl1-b"), ("lvl0", "lvl1-c")]
            ),
        )
        tm.assert_series_equal(merged.dtypes, expected)

    def test_merge_different_column_key_names(self) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_merge_copy(self) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_merge_nocopy(self, using_infer_string: bool) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_intelligently_handle_join_key(self) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_merge_overlap(self, left: DataFrame) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_merge_many_groups_and_sorts(
        self, df: DataFrame, df2: DataFrame, using_infer_string: bool
    ) -> None:
        # GH 54611: Placeholder for a more thorough test
        pass

    def test_indicator(self, dfs_for_indicator: Tuple[DataFrame, DataFrame]) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_merge_indicator_arg_validation(
        self, dfs_for_indicator: Tuple[DataFrame, DataFrame]
    ) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_merge_indicator_result_integrity(
        self, dfs_for_indicator: Tuple[DataFrame, DataFrame]
    ) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_merge_indicator_invalid(
        self, dfs_for_indicator: Tuple[DataFrame, DataFrame]
    ) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_merge_indicator_multiple_columns(self) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_validation(self) -> None:
        # Already annotated above; skipping duplication.

        pass

    def test_merge_duplicate_suffix(
        self,
        expected_data: Dict[str, List[Union[int, float, None]]],
        how: str,
    ) -> None

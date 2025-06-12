import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    bdate_range,
    concat,
    merge,
    option_context,
)
import pandas._testing as tm
from pandas.core.dtypes.common import is_scalar


def get_test_data(ngroups: int = 8, n: int = 50) -> np.ndarray:
    unique_groups = list(range(ngroups))
    arr = np.asarray(np.tile(unique_groups, n // ngroups))
    if len(arr) < n:
        arr = np.asarray(list(arr) + unique_groups[: n - len(arr)])
    np.random.default_rng(2).shuffle(arr)
    return arr


class TestJoin:
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
    def target_source(self) -> Tuple[DataFrame, DataFrame]:
        data = {
            "A": [0.0, 1.0, 2.0, 3.0, 4.0],
            "B": [0.0, 1.0, 0.0, 1.0, 0.0],
            "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
            "D": bdate_range("1/1/2009", periods=5),
        }
        target = DataFrame(
            data, index=Index(["a", "b", "c", "d", "e"], dtype=object)
        )
        source = DataFrame(
            {"MergedA": data["A"], "MergedD": data["D"]}, index=data["C"]
        )
        return (target, source)

    def test_left_outer_join(self, df: DataFrame, df2: DataFrame) -> None:
        joined_key2 = merge(df, df2, on="key2")
        _check_join(df, df2, joined_key2, ["key2"], how="left")
        joined_both = merge(df, df2)
        _check_join(df, df2, joined_both, ["key1", "key2"], how="left")

    def test_right_outer_join(self, df: DataFrame, df2: DataFrame) -> None:
        joined_key2 = merge(df, df2, on="key2", how="right")
        _check_join(df, df2, joined_key2, ["key2"], how="right")
        joined_both = merge(df, df2, how="right")
        _check_join(df, df2, joined_both, ["key1", "key2"], how="right")

    def test_full_outer_join(self, df: DataFrame, df2: DataFrame) -> None:
        joined_key2 = merge(df, df2, on="key2", how="outer")
        _check_join(df, df2, joined_key2, ["key2"], how="outer")
        joined_both = merge(df, df2, how="outer")
        _check_join(df, df2, joined_both, ["key1", "key2"], how="outer")

    def test_inner_join(self, df: DataFrame, df2: DataFrame) -> None:
        joined_key2 = merge(df, df2, on="key2", how="inner")
        _check_join(df, df2, joined_key2, ["key2"], how="inner")
        joined_both = merge(df, df2, how="inner")
        _check_join(df, df2, joined_both, ["key1", "key2"], how="inner")

    def test_handle_overlap(self, df: DataFrame, df2: DataFrame) -> None:
        joined = merge(df, df2, on="key2", suffixes=(".foo", ".bar"))
        assert "key1.foo" in joined
        assert "key1.bar" in joined

    def test_handle_overlap_arbitrary_key(
        self, df: DataFrame, df2: DataFrame
    ) -> None:
        joined = merge(
            df,
            df2,
            left_on="key2",
            right_on="key1",
            suffixes=(".foo", ".bar"),
        )
        assert "key1.foo" in joined
        assert "key2.bar" in joined

    @pytest.mark.parametrize(
        "infer_string",
        [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))],
    )
    def test_join_on(
        self, target_source: Tuple[DataFrame, DataFrame], infer_string: bool
    ) -> None:
        target, source = target_source
        merged = target.join(source, on="C")
        tm.assert_series_equal(
            merged["MergedA"], target["A"], check_names=False
        )
        tm.assert_series_equal(
            merged["MergedD"], target["D"], check_names=False
        )
        df = DataFrame({"key": ["a", "a", "b", "b", "c"]})
        df2 = DataFrame({"value": [0, 1, 2]}, index=["a", "b", "c"])
        joined = df.join(df2, on="key")
        expected = DataFrame(
            {"key": ["a", "a", "b", "b", "c"], "value": [0, 0, 1, 1, 2]}
        )
        tm.assert_frame_equal(joined, expected)
        df_a = DataFrame(
            [[1], [2], [3]], index=["a", "b", "c"], columns=["one"]
        )
        df_b = DataFrame([["foo"], ["bar"]], index=[1, 2], columns=["two"])
        df_c = DataFrame([[1], [2]], index=[1, 2], columns=["three"])
        joined = df_a.join(df_b, on="one")
        joined = joined.join(df_c, on="one")
        assert np.isnan(joined["two"]["c"])
        assert np.isnan(joined["three"]["c"])
        with pytest.raises(KeyError, match="^'E'$"):
            target.join(source, on="E")
        msg = "You are trying to merge on float64 and object|str columns for key 'A'. If you wish to proceed you should use pd.concat"
        with pytest.raises(ValueError, match=msg):
            target.join(source, on="A")

    def test_join_on_fails_with_different_right_index(self) -> None:
        df = DataFrame(
            {
                "a": np.random.default_rng(2).choice(["m", "f"], size=3),
                "b": np.random.default_rng(2).standard_normal(3),
            }
        )
        df2 = DataFrame(
            {
                "a": np.random.default_rng(2).choice(["m", "f"], size=10),
                "b": np.random.default_rng(2).standard_normal(10),
            },
            index=MultiIndex.from_product([range(5), ["A", "B"]]),
        )
        msg = 'len\\(left_on\\) must equal the number of levels in the index of "right"'
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, left_on="a", right_index=True)

    def test_join_on_fails_with_different_left_index(self) -> None:
        df = DataFrame(
            {
                "a": np.random.default_rng(2).choice(["m", "f"], size=3),
                "b": np.random.default_rng(2).standard_normal(3),
            },
            index=MultiIndex.from_arrays([range(3), list("abc")]),
        )
        df2 = DataFrame(
            {
                "a": np.random.default_rng(2).choice(["m", "f"], size=10),
                "b": np.random.default_rng(2).standard_normal(10),
            }
        )
        msg = 'len\\(right_on\\) must equal the number of levels in the index of "left"'
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, right_on="b", left_index=True)

    def test_join_on_fails_with_different_column_counts(self) -> None:
        df = DataFrame(
            {
                "a": np.random.default_rng(2).choice(["m", "f"], size=3),
                "b": np.random.default_rng(2).standard_normal(3),
            }
        )
        df2 = DataFrame(
            {
                "a": np.random.default_rng(2).choice(["m", "f"], size=10),
                "b": np.random.default_rng(2).standard_normal(10),
            },
            index=MultiIndex.from_product([range(5), ["A", "B"]]),
        )
        msg = "len\\(right_on\\) must equal len\\(left_on\\)"
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, right_on="a", left_on=["a", "b"])

    @pytest.mark.parametrize("wrong_type", [2, "str", None, np.array([0, 1])])
    def test_join_on_fails_with_wrong_object_type(
        self, wrong_type: Any
    ) -> None:
        df = DataFrame({"a": [1, 1]})
        msg = f"Can only merge Series or DataFrame objects, a {type(wrong_type)} was passed"
        with pytest.raises(TypeError, match=msg):
            merge(wrong_type, df, left_on="a", right_on="a")
        with pytest.raises(TypeError, match=msg):
            merge(df, wrong_type, left_on="a", right_on="a")

    def test_join_on_pass_vector(
        self, target_source: Tuple[DataFrame, DataFrame]
    ) -> None:
        target, source = target_source
        expected = target.join(source, on="C")
        expected = expected.rename(columns={"C": "key_0"})
        expected = expected[["key_0", "A", "B", "D", "MergedA", "MergedD"]]
        join_col = target.pop("C")
        result = target.join(source, on=join_col)
        tm.assert_frame_equal(result, expected)

    def test_join_with_len0(
        self, target_source: Tuple[DataFrame, DataFrame]
    ) -> None:
        target, source = target_source
        merged = target.join(source.reindex([]), on="C")
        for col in source:
            assert col in merged
            assert merged[col].isna().all()
        merged2 = target.join(source.reindex([]), on="C", how="inner")
        tm.assert_index_equal(merged2.columns, merged.columns)
        assert len(merged2) == 0

    def test_join_on_inner(self) -> None:
        df = DataFrame({"key": ["a", "a", "d", "b", "b", "c"]})
        df2 = DataFrame({"value": [0, 1]}, index=["a", "b"])
        joined = df.join(df2, on="key", how="inner")
        expected = df.join(df2, on="key")
        expected = expected[expected["value"].notna()]
        tm.assert_series_equal(joined["key"], expected["key"])
        tm.assert_series_equal(
            joined["value"], expected["value"], check_dtype=False
        )
        tm.assert_index_equal(joined.index, expected.index)

    def test_join_on_singlekey_list(self) -> None:
        df = DataFrame({"key": ["a", "a", "b", "b", "c"]})
        df2 = DataFrame({"value": [0, 1, 2]}, index=["a", "b", "c"])
        joined = df.join(df2, on=["key"])
        expected = df.join(df2, on="key")
        tm.assert_frame_equal(joined, expected)

    def test_join_on_series(
        self, target_source: Tuple[DataFrame, DataFrame]
    ) -> None:
        target, source = target_source
        result = target.join(source["MergedA"], on="C")
        expected = target.join(source[["MergedA"]], on="C")
        tm.assert_frame_equal(result, expected)

    def test_join_on_series_buglet(self) -> None:
        df = DataFrame({"a": [1, 1]})
        ds = Series([2], index=[1], name="b")
        result = df.join(ds, on="a")
        expected = DataFrame({"a": [1, 1], "b": [2, 2]}, index=df.index)
        tm.assert_frame_equal(result, expected)

    def test_join_index_mixed(self, join_type: str) -> None:
        df1 = DataFrame(index=np.arange(10))
        df1["bool"] = True
        df1["string"] = "foo"
        df2 = DataFrame(index=np.arange(5, 15))
        df2["int"] = 1
        df2["float"] = 1.0
        joined = df1.join(df2, how=join_type)
        expected = _join_by_hand(df1, df2, how=join_type)
        tm.assert_frame_equal(joined, expected)
        joined = df2.join(df1, how=join_type)
        expected = _join_by_hand(df2, df1, how=join_type)
        tm.assert_frame_equal(joined, expected)

    def test_join_index_mixed_overlap(self) -> None:
        df1 = DataFrame(
            {"A": 1.0, "B": 2, "C": "foo", "D": True},
            index=np.arange(10),
            columns=["A", "B", "C", "D"],
        )
        assert df1["B"].dtype == np.int64
        assert df1["D"].dtype == np.bool_
        df2 = DataFrame(
            {"A": 1.0, "B": 2, "C": "foo", "D": True},
            index=np.arange(0, 10, 2),
            columns=["A", "B", "C", "D"],
        )
        joined = df1.join(df2, lsuffix="_one", rsuffix="_two")
        expected_columns = [
            "A_one",
            "B_one",
            "C_one",
            "D_one",
            "A_two",
            "B_two",
            "C_two",
            "D_two",
        ]
        df1.columns = expected_columns[:4]
        df2.columns = expected_columns[4:]
        expected = _join_by_hand(df1, df2)
        tm.assert_frame_equal(joined, expected)

    def test_join_empty_bug(self) -> None:
        x = DataFrame()
        x.join(DataFrame([3], index=[0], columns=["A"]), how="outer")

    def test_join_unconsolidated(self) -> None:
        a = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)), columns=["a", "b"]
        )
        c = Series(np.random.default_rng(2).standard_normal(30))
        a["c"] = c
        d = DataFrame(
            np.random.default_rng(2).standard_normal((30, 1)), columns=["q"]
        )
        a.join(d)
        d.join(a)

    def test_join_multiindex(self) -> None:
        index1 = MultiIndex.from_arrays(
            [["a", "a", "a", "b", "b", "b"], [1, 2, 3, 1, 2, 3]],
            names=["first", "second"],
        )
        index2 = MultiIndex.from_arrays(
            [["b", "b", "b", "c", "c", "c"], [1, 2, 3, 1, 2, 3]],
            names=["first", "second"],
        )
        df1 = DataFrame(
            data=np.random.default_rng(2).standard_normal(6),
            index=index1,
            columns=["var X"],
        )
        df2 = DataFrame(
            data=np.random.default_rng(2).standard_normal(6),
            index=index2,
            columns=["var Y"],
        )
        df1 = df1.sort_index(level=0)
        df2 = df2.sort_index(level=0)
        joined = df1.join(df2, how="outer")
        ex_index = Index(index1.values).union(Index(index2.values))
        expected = df1.reindex(ex_index).join(df2.reindex(ex_index))
        expected.index.names = index1.names
        tm.assert_frame_equal(joined, expected)
        assert joined.index.names == index1.names
        df1 = df1.sort_index(level=1)
        df2 = df2.sort_index(level=1)
        joined = df1.join(df2, how="outer").sort_index(level=0)
        ex_index = Index(index1.values).union(Index(index2.values))
        expected = df1.reindex(ex_index).join(df2.reindex(ex_index))
        expected.index.names = index1.names
        tm.assert_frame_equal(joined, expected)
        assert joined.index.names == index1.names

    def test_join_inner_multiindex(
        self, lexsorted_two_level_string_multiindex: MultiIndex
    ) -> None:
        key1 = [
            "bar",
            "bar",
            "bar",
            "foo",
            "foo",
            "baz",
            "baz",
            "qu
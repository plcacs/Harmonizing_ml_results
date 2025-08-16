import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    date_range,
    lreshape,
    melt,
    wide_to_long,
)
import pandas._testing as tm


@pytest.fixture
def df() -> DataFrame:
    res = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    res["id1"] = (res["A"] > 0).astype(np.int64)
    res["id2"] = (res["B"] > 0).astype(np.int64)
    return res


@pytest.fixture
def df1() -> DataFrame:
    res = DataFrame(
        [
            [1.067683, -1.110463, 0.20867],
            [-1.321405, 0.368915, -1.055342],
            [-0.807333, 0.08298, -0.873361],
        ]
    )
    res.columns = [list("ABC"), list("abc")]
    res.columns.names = ["CAP", "low"]
    return res


@pytest.fixture
def var_name() -> str:
    return "var"


@pytest.fixture
def value_name() -> str:
    return "val"


class TestMelt:
    def test_top_level_method(self, df: DataFrame) -> None:
        result = melt(df)
        assert result.columns.tolist() == ["variable", "value"]

    def test_method_signatures(
        self, df: DataFrame, df1: DataFrame, var_name: str, value_name: str
    ) -> None:
        tm.assert_frame_equal(df.melt(), melt(df))

        tm.assert_frame_equal(
            df.melt(id_vars=["id1", "id2"], value_vars=["A", "B"]),
            melt(df, id_vars=["id1", "id2"], value_vars=["A", "B"]),
        )

        tm.assert_frame_equal(
            df.melt(var_name=var_name, value_name=value_name),
            melt(df, var_name=var_name, value_name=value_name),
        )

        tm.assert_frame_equal(df1.melt(col_level=0), melt(df1, col_level=0))

    def test_default_col_names(self, df: DataFrame) -> None:
        result = df.melt()
        assert result.columns.tolist() == ["variable", "value"]

        result1 = df.melt(id_vars=["id1"])
        assert result1.columns.tolist() == ["id1", "variable", "value"]

        result2 = df.melt(id_vars=["id1", "id2"])
        assert result2.columns.tolist() == ["id1", "id2", "variable", "value"]

    def test_value_vars(self, df: DataFrame) -> None:
        result3 = df.melt(id_vars=["id1", "id2"], value_vars="A")
        assert len(result3) == 10

        result4 = df.melt(id_vars=["id1", "id2"], value_vars=["A", "B"])
        expected4 = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,
                "id2": df["id2"].tolist() * 2,
                "variable": ["A"] * 10 + ["B"] * 10,
                "value": (df["A"].tolist() + df["B"].tolist()),
            },
            columns=["id1", "id2", "variable", "value"],
        )
        tm.assert_frame_equal(result4, expected4)

    @pytest.mark.parametrize("type_", (tuple, list, np.array))
    def test_value_vars_types(self, type_: Any, df: DataFrame) -> None:
        # GH 15348
        expected = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,
                "id2": df["id2"].tolist() * 2,
                "variable": ["A"] * 10 + ["B"] * 10,
                "value": (df["A"].tolist() + df["B"].tolist()),
            },
            columns=["id1", "id2", "variable", "value"],
        )
        result = df.melt(id_vars=["id1", "id2"], value_vars=type_(("A", "B")))
        tm.assert_frame_equal(result, expected)

    def test_vars_work_with_multiindex(self, df1: DataFrame) -> None:
        expected = DataFrame(
            {
                ("A", "a"): df1[("A", "a")],
                "CAP": ["B"] * len(df1),
                "low": ["b"] * len(df1),
                "value": df1[("B", "b")],
            },
            columns=[("A", "a"), "CAP", "low", "value"],
        )

        result = df1.melt(id_vars=[("A", "a")], value_vars=[("B", "b")])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "id_vars, value_vars, col_level, expected",
        [
            (
                ["A"],
                ["B"],
                0,
                {
                    "A": {0: 1.067683, 1: -1.321405, 2: -0.807333},
                    "CAP": {0: "B", 1: "B", 2: "B"},
                    "value": {0: -1.110463, 1: 0.368915, 2: 0.08298},
                },
            ),
            (
                ["a"],
                ["b"],
                1,
                {
                    "a": {0: 1.067683, 1: -1.321405, 2: -0.807333},
                    "low": {0: "b", 1: "b", 2: "b"},
                    "value": {0: -1.110463, 1: 0.368915, 2: 0.08298},
                },
            ),
        ],
    )
    def test_single_vars_work_with_multiindex(
        self,
        id_vars: List[str],
        value_vars: List[str],
        col_level: int,
        expected: Dict[str, Any],
        df1: DataFrame,
    ) -> None:
        result = df1.melt(id_vars, value_vars, col_level=col_level)
        expected = DataFrame(expected)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "id_vars, value_vars",
        [
            [("A", "a"), [("B", "b")]],
            [[("A", "a")], ("B", "b")],
            [("A", "a"), ("B", "b")],
        ],
    )
    def test_tuple_vars_fail_with_multiindex(
        self, id_vars: Any, value_vars: Any, df1: DataFrame
    ) -> None:
        # melt should fail with an informative error message if
        # the columns have a MultiIndex and a tuple is passed
        # for id_vars or value_vars.
        msg = r"(id|value)_vars must be a list of tuples when columns are a MultiIndex"
        with pytest.raises(ValueError, match=msg):
            df1.melt(id_vars=id_vars, value_vars=value_vars)

    def test_custom_var_name(self, df: DataFrame, var_name: str) -> None:
        result5 = df.melt(var_name=var_name)
        assert result5.columns.tolist() == ["var", "value"]

        result6 = df.melt(id_vars=["id1"], var_name=var_name)
        assert result6.columns.tolist() == ["id1", "var", "value"]

        result7 = df.melt(id_vars=["id1", "id2"], var_name=var_name)
        assert result7.columns.tolist() == ["id1", "id2", "var", "value"]

        result8 = df.melt(id_vars=["id1", "id2"], value_vars="A", var_name=var_name)
        assert result8.columns.tolist() == ["id1", "id2", "var", "value"]

        result9 = df.melt(
            id_vars=["id1", "id2"], value_vars=["A", "B"], var_name=var_name
        )
        expected9 = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,
                "id2": df["id2"].tolist() * 2,
                var_name: ["A"] * 10 + ["B"] * 10,
                "value": (df["A"].tolist() + df["B"].tolist()),
            },
            columns=["id1", "id2", var_name, "value"],
        )
        tm.assert_frame_equal(result9, expected9)

    def test_custom_value_name(self, df: DataFrame, value_name: str) -> None:
        result10 = df.melt(value_name=value_name)
        assert result10.columns.tolist() == ["variable", "val"]

        result11 = df.melt(id_vars=["id1"], value_name=value_name)
        assert result11.columns.tolist() == ["id1", "variable", "val"]

        result12 = df.melt(id_vars=["id1", "id2"], value_name=value_name)
        assert result12.columns.tolist() == ["id1", "id2", "variable", "val"]

        result13 = df.melt(
            id_vars=["id1", "id2"], value_vars="A", value_name=value_name
        )
        assert result13.columns.tolist() == ["id1", "id2", "variable", "val"]

        result14 = df.melt(
            id_vars=["id1", "id2"], value_vars=["A", "B"], value_name=value_name
        )
        expected14 = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,
                "id2": df["id2"].tolist() * 2,
                "variable": ["A"] * 10 + ["B"] * 10,
                value_name: (df["A"].tolist() + df["B"].tolist()),
            },
            columns=["id1", "id2", "variable", value_name],
        )
        tm.assert_frame_equal(result14, expected14)

    def test_custom_var_and_value_name(
        self, df: DataFrame, value_name: str, var_name: str
    ) -> None:
        result15 = df.melt(var_name=var_name, value_name=value_name)
        assert result15.columns.tolist() == ["var", "val"]

        result16 = df.melt(id_vars=["id1"], var_name=var_name, value_name=value_name)
        assert result16.columns.tolist() == ["id1", "var", "val"]

        result17 = df.melt(
            id_vars=["id1", "id2"], var_name=var_name, value_name=value_name
        )
        assert result17.columns.tolist() == ["id1", "id2", "var", "val"]

        result18 = df.melt(
            id_vars=["id1", "id2"],
            value_vars="A",
            var_name=var_name,
            value_name=value_name,
        )
        assert result18.columns.tolist() == ["id1", "id2", "var", "val"]

        result19 = df.melt(
            id_vars=["id1", "id2"],
            value_vars=["A", "B"],
            var_name=var_name,
            value_name=value_name,
        )
        expected19 = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,
                "id2": df["id2"].tolist() * 2,
                var_name: ["A"] * 10 + ["B"] * 10,
                value_name: (df["A"].tolist() + df["B"].tolist()),
            },
            columns=["id1", "id2", var_name, value_name],
        )
        tm.assert_frame_equal(result19, expected19)

        df20 = df.copy()
        df20.columns.name = "foo"
        result20 = df20.melt()
        assert result20.columns.tolist() == ["foo", "value"]

    @pytest.mark.parametrize("col_level", [0, "CAP"])
    def test_col_level(self, col_level: Union[int, str], df1: DataFrame) -> None:
        res = df1.melt(col_level=col_level)
        assert res.columns.tolist() == ["CAP", "value"]

    def test_multiindex(self, df1: DataFrame) -> None:
        res = df1.melt()
        assert res.columns.tolist() == ["CAP", "low", "value"]

    @pytest.mark.parametrize(
        "col",
        [
            date_range("2010", periods=5, tz="US/Pacific"),
            pd.Categorical(["a", "b", "c", "a", "d"]),
            [0, 1, 0, 0, 0],
        ],
    )
    def test_pandas_dtypes(self, col: Any) -> None:
        # GH 15785
        col = pd.Series(col)
        df = DataFrame(
            {"klass": range(5), "col": col, "attr1": [1, 0, 0, 0, 0], "attr2": col}
        )
        expected_value = pd.concat([pd.Series([1, 0, 0, 0, 0]), col], ignore_index=True)
        result = melt(
            df, id_vars=["klass", "col"], var_name="attribute", value_name="value"
        )
        expected = DataFrame(
            {
                0: list(range(5)) * 2,
                1: pd.concat([col] * 2, ignore_index=True),
                2: ["attr1"] * 5 + ["attr2"] * 5,
                3: expected_value,
            }
        )
        expected.columns = ["klass", "col", "attribute", "value"]
        tm.assert_frame_equal(result, expected)

    def test_preserve_category(self) -> None:
        # GH 15853
        data = DataFrame({"A": [1, 2], "B": pd.Categorical(["X", "Y"])})
        result = melt(data, ["B"], ["A"])
        expected = DataFrame(
            {"B": pd.Categorical(["X", "Y"]), "variable": ["A", "A"], "value": [1, 2]}
        )

        tm.assert_frame_equal(result, expected)

    def test_melt_missing_columns_raises(self) -> None:
        # GH-23575
        # This test is to ensure that pandas raises an error if melting is
        # attempted with column names absent from the dataframe

        # Generate data
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)), columns=list("abcd")
        )

        # Try to melt with missing `value_vars` column name
        msg = "The following id_vars or value_vars are not present in the DataFrame:"
        with pytest.raises(KeyError, match=msg):
            df.melt(["a", "b"], ["C", "d"])

        # Try to melt with missing `id_vars` column name
        with pytest.raises(KeyError, match=msg):
            df.melt(["A", "b"], ["c", "d"])

        # Multiple missing
        with pytest.raises(
            KeyError,
            match=msg,
        ):
            df.melt(["a", "b", "not_here", "or_there"], ["c", "d"])

        # Multiindex melt fails if column is missing from multilevel melt
        df.columns = [list("ABCD"), list("abcd")]
        with pytest.raises(KeyError, match=msg):
            df.melt([("E", "a")], [("B", "b")])
        # Multiindex fails if column is missing from single level melt
        with pytest.raises(KeyError, match=msg):
            df.melt(["A"], ["F"], col_level=0)

    def test_melt_mixed_int_str_id_vars(self) -> None:
        # GH 29718
        df = DataFrame({0: ["foo"], "a": ["bar"], "b": [1], "d": [2]})
        result = melt(df, id_vars=[0, "a"], value_vars=["b", "d"])
        expected = DataFrame(
            {0: ["foo"] * 2, "a": ["bar"] * 2, "variable": list("bd"), "value": [1, 2]}
        )
        # the df's columns are mixed type and thus object -> preserves object dtype
        expected["variable"] = expected["variable"].astype(object)
        tm.assert_frame_equal(result, expected)

    def test_melt_mixed_int_str_value_vars(self) -> None:
        # GH 29718
        df = DataFrame({0: ["foo"], "a": ["bar"]})
        result = melt(df, value_vars=[0, "a"])
        expected = DataFrame({"variable": [0, "a"], "value": ["foo", "bar"]})
        tm.assert_frame_equal(result, expected)

    def test_ignore_index(self) -> None:
        # GH 17440
        df = DataFrame({"foo": [0], "bar": [1]}, index=["first"])
        result = melt(df, ignore_index=False)
        expected = DataFrame(
            {"variable": ["foo", "bar"], "value": [0, 1]}, index=["first", "first"]
        )
        tm.assert_frame_equal(result, expected)

    def test_ignore_multiindex(self) -> None:
        # GH 17440
        index = pd.MultiIndex.from_tuples(
            [("first", "second"), ("first", "third")], names=["baz", "foobar"]
        )
        df = DataFrame
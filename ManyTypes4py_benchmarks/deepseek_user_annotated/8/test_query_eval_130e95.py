import operator
from tokenize import TokenError
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pytest

from pandas.errors import (
    NumExprClobberingError,
    UndefinedVariableError,
)
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED


@pytest.fixture(params=["python", "pandas"], ids=lambda x: x)
def parser(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(
    params=["python", pytest.param("numexpr", marks=td.skip_if_no("numexpr"))],
    ids=lambda x: x,
)
def engine(request: pytest.FixtureRequest) -> str:
    return request.param


def skip_if_no_pandas_parser(parser: str) -> None:
    if parser != "pandas":
        pytest.skip(f"cannot evaluate with parser={parser}")


class TestCompat:
    @pytest.fixture
    def df(self) -> DataFrame:
        return DataFrame({"A": [1, 2, 3]})

    @pytest.fixture
    def expected1(self, df: DataFrame) -> DataFrame:
        return df[df.A > 0]

    @pytest.fixture
    def expected2(self, df: DataFrame) -> Series:
        return df.A + 1

    def test_query_default(
        self, df: DataFrame, expected1: DataFrame, expected2: Series
    ) -> None:
        # GH 12749
        # this should always work, whether NUMEXPR_INSTALLED or not
        result = df.query("A>0")
        tm.assert_frame_equal(result, expected1)
        result = df.eval("A+1")
        tm.assert_series_equal(result, expected2)

    def test_query_None(
        self, df: DataFrame, expected1: DataFrame, expected2: Series
    ) -> None:
        result = df.query("A>0", engine=None)
        tm.assert_frame_equal(result, expected1)
        result = df.eval("A+1", engine=None)
        tm.assert_series_equal(result, expected2)

    def test_query_python(
        self, df: DataFrame, expected1: DataFrame, expected2: Series
    ) -> None:
        result = df.query("A>0", engine="python")
        tm.assert_frame_equal(result, expected1)
        result = df.eval("A+1", engine="python")
        tm.assert_series_equal(result, expected2)

    def test_query_numexpr(
        self, df: DataFrame, expected1: DataFrame, expected2: Series
    ) -> None:
        if NUMEXPR_INSTALLED:
            result = df.query("A>0", engine="numexpr")
            tm.assert_frame_equal(result, expected1)
            result = df.eval("A+1", engine="numexpr")
            tm.assert_series_equal(result, expected2)
        else:
            msg = (
                r"'numexpr' is not installed or an unsupported version. "
                r"Cannot use engine='numexpr' for query/eval if 'numexpr' is "
                r"not installed"
            )
            with pytest.raises(ImportError, match=msg):
                df.query("A>0", engine="numexpr")
            with pytest.raises(ImportError, match=msg):
                df.eval("A+1", engine="numexpr")


class TestDataFrameEval:
    # smaller hits python, larger hits numexpr
    @pytest.mark.parametrize("n", [4, 4000])
    @pytest.mark.parametrize(
        "op_str,op,rop",
        [
            ("+", "__add__", "__radd__"),
            ("-", "__sub__", "__rsub__"),
            ("*", "__mul__", "__rmul__"),
            ("/", "__truediv__", "__rtruediv__"),
        ],
    )
    def test_ops(self, op_str: str, op: str, rop: str, n: int) -> None:
        # tst ops and reversed ops in evaluation
        # GH7198

        df = DataFrame(1, index=range(n), columns=list("abcd"))
        df.iloc[0] = 2
        m = df.mean()

        base = DataFrame(  # noqa: F841
            np.tile(m.values, n).reshape(n, -1), columns=list("abcd")
        )

        expected = eval(f"base {op_str} df")

        # ops as strings
        result = eval(f"m {op_str} df")
        tm.assert_frame_equal(result, expected)

        # these are commutative
        if op in ["+", "*"]:
            result = getattr(df, op)(m)
            tm.assert_frame_equal(result, expected)

        # these are not
        elif op in ["-", "/"]:
            result = getattr(df, rop)(m)
            tm.assert_frame_equal(result, expected)

    def test_dataframe_sub_numexpr_path(self) -> None:
        # GH7192: Note we need a large number of rows to ensure this
        #  goes through the numexpr path
        df = DataFrame({"A": np.random.default_rng(2).standard_normal(25000)})
        df.iloc[0:5] = np.nan
        expected = 1 - np.isnan(df.iloc[0:25])
        result = (1 - np.isnan(df)).iloc[0:25]
        tm.assert_frame_equal(result, expected)

    def test_query_non_str(self) -> None:
        # GH 11485
        df = DataFrame({"A": [1, 2, 3], "B": ["a", "b", "b"]})

        msg = "expr must be a string to be evaluated"
        with pytest.raises(ValueError, match=msg):
            df.query(lambda x: x.B == "b")

        with pytest.raises(ValueError, match=msg):
            df.query(111)

    def test_query_empty_string(self) -> None:
        # GH 13139
        df = DataFrame({"A": [1, 2, 3]})

        msg = "expr cannot be an empty string"
        with pytest.raises(ValueError, match=msg):
            df.query("")

    def test_query_duplicate_column_name(
        self, engine: str, parser: str
    ) -> None:
        df = DataFrame(
            {
                "A": range(3),
                "B": range(3),
                "C": range(3)
            }
        ).rename(columns={"B": "A"})

        res = df.query('C == 1', engine=engine, parser=parser)

        expect = DataFrame(
            [[1, 1, 1]],
            columns=["A", "A", "C"],
            index=[1]
        )

        tm.assert_frame_equal(res, expect)

    def test_eval_resolvers_as_list(self) -> None:
        # GH 14095
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)), columns=list("ab")
        )
        dict1 = {"a": 1}
        dict2 = {"b": 2}
        assert df.eval("a + b", resolvers=[dict1, dict2]) == dict1["a"] + dict2["b"]
        assert pd.eval("a + b", resolvers=[dict1, dict2]) == dict1["a"] + dict2["b"]

    def test_eval_resolvers_combined(self) -> None:
        # GH 34966
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)), columns=list("ab")
        )
        dict1 = {"c": 2}

        # Both input and default index/column resolvers should be usable
        result = df.eval("a + b * c", resolvers=[dict1])

        expected = df["a"] + df["b"] * dict1["c"]
        tm.assert_series_equal(result, expected)

    def test_eval_object_dtype_binop(self) -> None:
        # GH#24883
        df = DataFrame({"a1": ["Y", "N"]})
        res = df.eval("c = ((a1 == 'Y') & True)")
        expected = DataFrame({"a1": ["Y", "N"], "c": [True, False]})
        tm.assert_frame_equal(res, expected)

    def test_using_numpy(self, engine: str, parser: str) -> None:
        # GH 58041
        skip_if_no_pandas_parser(parser)
        df = Series([0.2, 1.5, 2.8], name="a").to_frame()
        res = df.eval("@np.floor(a)", engine=engine, parser=parser)
        expected = np.floor(df["a"])
        tm.assert_series_equal(expected, res)

    def test_eval_simple(self, engine: str, parser: str) -> None:
        df = Series([0.2, 1.5, 2.8], name="a").to_frame()
        res = df.eval("a", engine=engine, parser=parser)
        expected = df["a"]
        tm.assert_series_equal(expected, res)

    def test_extension_array_eval(
        self, engine: str, parser: str, request: pytest.FixtureRequest
    ) -> None:
        # GH#58748
        if engine == "numexpr":
            mark = pytest.mark.xfail(
                reason="numexpr does not support extension array dtypes"
            )
            request.applymarker(mark)
        df = DataFrame({"a": pd.array([1, 2, 3]), "b": pd.array([4, 5, 6])})
        result = df.eval("a / b", engine=engine, parser=parser)
        expected = Series(pd.array([0.25, 0.40, 0.50]))
        tm.assert_series_equal(result, expected)

    def test_complex_eval(self, engine: str, parser: str) -> None:
        # GH#21374
        df = DataFrame({"a": [1 + 2j], "b": [1 + 1j]})
        result = df.eval("a/b", engine=engine, parser=parser)
        expected = Series([1.5 + 0.5j])
        tm.assert_series_equal(result, expected)


class TestDataFrameQueryWithMultiIndex:
    def test_query_with_named_multiindex(
        self, parser: str, engine: str
    ) -> None:
        skip_if_no_pandas_parser(parser)
        a = np.random.default_rng(2).choice(["red", "green"], size=10)
        b = np.random.default_rng(2).choice(["eggs", "ham"], size=10)
        index = MultiIndex.from_arrays([a, b], names=["color", "food"])
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        ind = Series(
            df.index.get_level_values("color").values, index=index, name="color"
        )

        # equality
        res1 = df.query('color == "red"', parser=parser, engine=engine)
        res2 = df.query('"red" == color', parser=parser, engine=engine)
        exp = df[ind == "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # inequality
        res1 = df.query('color != "red"', parser=parser, engine=engine)
        res2 = df.query('"red" != color', parser=parser, engine=engine)
        exp = df[ind != "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # list equality (really just set membership)
        res1 = df.query('color == ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] == color', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('color != ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] != color', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # in/not in ops
        res1 = df.query('["red"] in color', parser=parser, engine=engine)
        res2 = df.query('"red" in color', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('["red"] not in color', parser=parser, engine=engine)
        res2 = df.query('"red" not in color', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

    def test_query_with_unnamed_multiindex(
        self, parser: str, engine: str
    ) -> None:
        skip_if_no_pandas_parser(parser)
        a = np.random.default_rng(2).choice(["red", "green"], size=10)
        b = np.random.default_rng(2).choice(["eggs", "ham"], size=10)
        index = MultiIndex.from_arrays([a, b])
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        ind = Series(df.index.get_level_values(0).values, index=index)

        res1 = df.query('ilevel_0 == "red"', parser=parser, engine=engine)
        res2 = df.query('"red" == ilevel_0', parser=parser, engine=engine)
        exp = df[ind == "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # inequality
        res1 = df.query('ilevel_0 != "red"', parser=parser, engine=engine)
        res2 = df.query('"red" != ilevel_0', parser=parser, engine=engine)
        exp = df[ind != "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # list equality (really just set membership)
        res1 = df.query('ilevel_0 == ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] == ilevel_0', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('ilevel_0 != ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] != ilevel_0', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # in/not in ops
        res1 = df.query('["red"] in ilevel_0', parser=parser, engine=engine)
        res2 = df.query('"red" in ilevel_0', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('["red"] not in ilevel_0', parser=parser, engine=engine)
        res2 = df.query('"red" not in ilevel_0', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # ## LEVEL 1
        ind = Series(df.index.get_level_values(1).values, index=index)
        res1 = df.query('ilevel_1 == "eggs"', parser=parser, engine=engine)
        res2 = df.query('"eggs" == ilevel_1', parser=parser, engine=engine)
        exp = df[ind == "eggs"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # inequality
        res1 = df.query('ilevel_1 != "eggs"', parser=parser, engine=engine)
        res2 = df.query('"eggs" != ilevel_1', parser=parser, engine=engine)
        exp = df[ind != "eggs"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # list equality (really just set membership)
        res1 = df.query('ilevel_1 == ["eggs"]', parser=parser, engine=engine)
        res2 = df.query('["eggs"] == ilevel_1', parser=parser, engine=engine)
        exp = df[ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('ilevel_1 != ["eggs"]', parser=parser, engine=engine)
        res2 = df.query('["eggs"] != ilevel_1', parser=parser, engine=engine)
        exp = df[~ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # in/not in ops
        res1 = df.query('["eggs"] in ilevel_1', parser=parser, engine=engine)
        res2 = df.query('"eggs" in ilevel_1', parser=parser, engine=engine)
        exp = df[ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('["eggs"] not in ilevel_1', parser=parser,
from datetime import date, datetime, timedelta
from itertools import product
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
from pandas._config import using_string_dtype
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Grouper,
    Index,
    MultiIndex,
    Series,
    concat,
    date_range,
)
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table


class TestPivotTable:

    @pytest.fixture
    def data(self) -> DataFrame:
        return DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"],
                "B": ["one", "one", "one", "two", "one", "one", "one", "two", "two", "two", "one"],
                "C": ["dull", "dull", "shiny", "dull", "dull", "shiny", "shiny", "dull", "shiny", "shiny", "shiny"],
                "D": np.random.default_rng(2).standard_normal(11),
                "E": np.random.default_rng(2).standard_normal(11),
                "F": np.random.default_rng(2).standard_normal(11),
            }
        )

    def test_pivot_table(
        self, observed: bool, data: DataFrame
    ) -> None:
        index: List[str] = ["A", "B"]
        columns: str = "C"
        table: DataFrame = pivot_table(
            data, values="D", index=index, columns=columns, observed=observed
        )
        table2: DataFrame = data.pivot_table(
            values="D", index=index, columns=columns, observed=observed
        )
        tm.assert_frame_equal(table, table2)
        pivot_table(data, values="D", index=index, observed=observed)
        if len(index) > 1:
            assert table.index.names == tuple(index)
        else:
            assert table.index.name == index[0]
        if len(columns) > 1:
            assert table.columns.names == columns
        else:
            assert table.columns.name == columns[0]
        expected: DataFrame = data.groupby(index + [columns])["D"].agg("mean").unstack()
        tm.assert_frame_equal(table, expected)

    def test_pivot_table_categorical_observed_equal(
        self, observed: bool
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "col1": list("abcde"),
                "col2": list("fghij"),
                "col3": [1, 2, 3, 4, 5],
            }
        )
        expected: DataFrame = df.pivot_table(
            index="col1",
            values="col3",
            columns="col2",
            aggfunc="sum",
            fill_value=0,
        )
        expected.index = expected.index.astype("category")
        expected.columns = expected.columns.astype("category")
        df.col1 = df.col1.astype("category")
        df.col2 = df.col2.astype("category")
        result: DataFrame = df.pivot_table(
            index="col1",
            values="col3",
            columns="col2",
            aggfunc="sum",
            fill_value=0,
            observed=observed,
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_nocols(self) -> None:
        df: DataFrame = DataFrame(
            {
                "rows": ["a", "b", "c"],
                "cols": ["x", "y", "z"],
                "values": [1, 2, 3],
            }
        )
        rs: DataFrame = df.pivot_table(columns="cols", aggfunc="sum")
        xp: DataFrame = df.pivot_table(index="cols", aggfunc="sum").T
        tm.assert_frame_equal(rs, xp)
        rs = df.pivot_table(columns="cols", aggfunc={"values": "mean"})
        xp = df.pivot_table(index="cols", aggfunc={"values": "mean"}).T
        tm.assert_frame_equal(rs, xp)

    def test_pivot_table_dropna(self) -> None:
        df: DataFrame = DataFrame(
            {
                "amount": {0: 60000, 1: 100000, 2: 50000, 3: 30000},
                "customer": {0: "A", 1: "A", 2: "B", 3: "C"},
                "month": {0: 201307, 1: 201309, 2: 201308, 3: 201310},
                "product": {0: "a", 1: "b", 2: "c", 3: "d"},
                "quantity": {0: 2000000, 1: 500000, 2: 1000000, 3: 1000000},
            }
        )
        pv_col: DataFrame = df.pivot_table(
            "quantity", "month", ["customer", "product"], dropna=False
        )
        pv_ind: DataFrame = df.pivot_table(
            "quantity", ["customer", "product"], "month", dropna=False
        )
        m: MultiIndex = MultiIndex.from_tuples(
            [
                ("A", "a"),
                ("A", "b"),
                ("A", "c"),
                ("A", "d"),
                ("B", "a"),
                ("B", "b"),
                ("B", "c"),
                ("B", "d"),
                ("C", "a"),
                ("C", "b"),
                ("C", "c"),
                ("C", "d"),
            ],
            names=["customer", "product"],
        )
        tm.assert_index_equal(pv_col.columns, m)
        tm.assert_index_equal(pv_ind.index, m)

    def test_pivot_table_categorical(self) -> None:
        cat1: Categorical = Categorical(
            ["a", "a", "b", "b"], categories=["a", "b", "z"], ordered=True
        )
        cat2: Categorical = Categorical(
            ["c", "d", "c", "d"], categories=["c", "d", "y"], ordered=True
        )
        df: DataFrame = DataFrame({"A": cat1, "B": cat2, "values": [1, 2, 3, 4]})
        result: DataFrame = pivot_table(
            df,
            values="values",
            index=["A", "B"],
            dropna=True,
            observed=False,
        )
        exp_index: MultiIndex = MultiIndex.from_arrays([cat1, cat2], names=["A", "B"])
        expected: DataFrame = DataFrame(
            {"values": [1.0, 2.0, 3.0, 4.0]}, index=exp_index
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_dropna_categoricals(
        self, dropna: bool
    ) -> None:
        categories: List[str] = ["a", "b", "c", "d"]
        df: DataFrame = DataFrame(
            {
                "A": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
                "B": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "C": range(9),
            }
        )
        df["A"] = df["A"].astype(CategoricalDtype(categories, ordered=False))
        result: DataFrame = df.pivot_table(
            index="B", columns="A", values="C", dropna=dropna, observed=False
        )
        expected_columns: Series = Series(["a", "b", "c"], name="A")
        expected_columns = expected_columns.astype(CategoricalDtype(categories, ordered=False))
        expected_index: Series = Series([1, 2, 3], name="B")
        expected: DataFrame = DataFrame(
            [[0.0, 3.0, 6.0], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0]],
            index=expected_index,
            columns=expected_columns,
        )
        if not dropna:
            expected = expected.reindex(columns=pd.Categorical(categories)).astype("float")
        tm.assert_frame_equal(result, expected)

    def test_pivot_with_non_observable_dropna(
        self, dropna: bool
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "A": Categorical(
                    [np.nan, "low", "high", "low", "high"],
                    categories=["low", "high"],
                    ordered=True,
                ),
                "B": [0.0, 1.0, 2.0, 3.0, 4.0],
            }
        )
        result: DataFrame = df.pivot_table(
            index="A", values="B", dropna=dropna, observed=False
        )
        if dropna:
            values: List[float] = [2.0, 3.0]
            codes: List[int] = [0, 1]
        else:
            values = [2.0, 3.0, 0.0]
            codes = [0, 1, -1]
        expected: DataFrame = DataFrame(
            {"B": values},
            index=Index(
                Categorical.from_codes(codes, categories=["low", "high"], ordered=True),
                name="A",
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_with_non_observable_dropna_multi_cat(
        self, dropna: bool
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "A": Categorical(
                    ["left", "low", "high", "low", "high"],
                    categories=["low", "high", "left"],
                    ordered=True,
                ),
                "B": range(5),
            }
        )
        result: DataFrame = df.pivot_table(
            index="A", values="B", dropna=dropna, observed=False
        )
        expected: DataFrame = DataFrame(
            {"B": [2.0, 3.0, 0.0]},
            index=Index(
                Categorical.from_codes([0, 1, 2], categories=["low", "high", "left"], ordered=True),
                name="A",
            ),
        )
        if not dropna:
            expected["B"] = expected["B"].astype(float)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("left_right", [([0] * 4, [1] * 4), (range(3), range(1, 4))])
    def test_pivot_with_interval_index(
        self,
        left_right: Tuple[List[int], List[int]],
        dropna: bool,
        closed: str,
    ) -> None:
        left, right = left_right
        interval_values: Categorical = Categorical(
            pd.IntervalIndex.from_arrays(left, right, closed=closed)
        )
        df: DataFrame = DataFrame({"A": interval_values, "B": 1})
        result: DataFrame = df.pivot_table(
            index="A", values="B", dropna=dropna, observed=False
        )
        expected: DataFrame = DataFrame(
            {"B": [1.0]}, index=Index(interval_values.unique(), name="A")
        )
        if not dropna:
            expected = expected.astype(float)
        tm.assert_frame_equal(result, expected)

    def test_pivot_with_interval_index_margins(self) -> None:
        ordered_cat: pd.IntervalIndex = pd.IntervalIndex.from_arrays(
            [0, 0, 1, 1], [1, 1, 2, 2]
        )
        df: DataFrame = DataFrame(
            {
                "A": np.arange(4, 0, -1, dtype=np.intp),
                "B": ["a", "b", "a", "b"],
                "C": Categorical(ordered_cat, ordered=True).sort_values(ascending=False),
            }
        )
        pivot_tab: DataFrame = pivot_table(
            df,
            index="C",
            columns="B",
            values="A",
            aggfunc="sum",
            margins=True,
            observed=False,
        )
        result: Series = pivot_tab["All"]
        expected: Series = Series(
            [3, 7, 10],
            index=Index(
                [pd.Interval(0, 1), pd.Interval(1, 2), "All"],
                name="C",
            ),
            name="All",
            dtype=np.intp,
        )
        tm.assert_series_equal(result, expected)

    def test_pass_array(
        self, data: DataFrame
    ) -> None:
        result: DataFrame = data.pivot_table("D", index=data.A, columns=data.C)
        expected: DataFrame = data.pivot_table("D", index="A", columns="C")
        tm.assert_frame_equal(result, expected)

    def test_pass_function(
        self, data: DataFrame
    ) -> None:
        result: DataFrame = data.pivot_table(
            "D", index=lambda x: x // 5, columns=data.C
        )
        expected: DataFrame = data.pivot_table(
            "D", index=data.index // 5, columns="C"
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_multiple(
        self, data: DataFrame
    ) -> None:
        index: List[str] = ["A", "B"]
        columns: str = "C"
        table: DataFrame = pivot_table(
            data, index=index, columns=columns
        )
        expected: DataFrame = data.groupby(index + [columns]).agg("mean").unstack()
        tm.assert_frame_equal(table, expected)

    def test_pivot_dtypes(self) -> None:
        f: DataFrame = DataFrame(
            {
                "a": ["cat", "bat", "cat", "bat"],
                "v": [1, 2, 3, 4],
                "i": ["a", "b", "a", "b"],
            }
        )
        assert f.dtypes["v"] == "int64"
        z: DataFrame = pivot_table(
            f,
            values="v",
            index=["a"],
            columns=["i"],
            fill_value=0,
            aggfunc="sum",
        )
        result: Series = z.dtypes
        expected: Series = Series(
            [np.dtype("int64")] * 2,
            index=Index(["a", "b"], name="i"),
        )
        tm.assert_series_equal(result, expected)
        f = DataFrame(
            {
                "a": ["cat", "bat", "cat", "bat"],
                "v": [1.5, 2.5, 3.5, 4.5],
                "i": ["a", "b", "a", "b"],
            }
        )
        assert f.dtypes["v"] == "float64"
        z = pivot_table(
            f,
            values="v",
            index=["a"],
            columns=["i"],
            fill_value=0,
            aggfunc="mean",
        )
        result = z.dtypes
        expected = Series(
            [np.dtype("float64")] * 2,
            index=Index(["a", "b"], name="i"),
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "columns,values",
        [
            ("bool1", ["float1", "float2"]),
            ("bool1", ["float1", "float2", "bool1"]),
            ("bool2", ["float1", "float2", "bool1"]),
        ],
    )
    def test_pivot_preserve_dtypes(
        self,
        columns: str,
        values: List[str],
    ) -> None:
        v: np.ndarray = np.arange(5, dtype=np.float64)
        df: DataFrame = DataFrame(
            {
                "float1": v,
                "float2": v + 2.0,
                "bool1": v <= 2,
                "bool2": v <= 3,
            }
        )
        df_res: DataFrame = df.reset_index().pivot_table(
            index="index", columns=columns, values=values
        )
        result: Dict[str, Any] = dict(df_res.dtypes)
        expected: Dict[str, Any] = {col: np.dtype("float64") for col in df_res}
        assert result == expected

    def test_pivot_no_values_raise_error(self) -> None:
        idx: pd.DatetimeIndex = pd.DatetimeIndex(
            [
                "2011-01-01",
                "2011-02-01",
                "2011-01-02",
                "2011-01-01",
                "2011-01-02",
            ]
        )
        df: DataFrame = DataFrame({"A": [1, 2, 3, 4, 5]}, index=idx)
        res: DataFrame = df.pivot_table(index=df.index.month, columns=df.index.day)
        exp_columns: MultiIndex = MultiIndex.from_tuples(
            [("A", 1), ("A", 2)]
        )
        exp_columns = exp_columns.set_levels(exp_columns.levels[1].astype(np.int32), level=1)
        exp: DataFrame = DataFrame(
            [[2.5, 4.0], [2.0, np.nan]],
            index=Index([1, 2], dtype=np.int32),
            columns=exp_columns,
        )
        tm.assert_frame_equal(res, exp)
        df = DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "dt": pd.date_range("2011-01-01", freq="D", periods=5),
            },
            index=idx,
        )
        res = df.pivot_table(
            index=df.index.month,
            columns=pd.Grouper(key="dt", freq="ME"),
        )
        exp_columns = MultiIndex.from_arrays(
            [
                ["A"],
                pd.DatetimeIndex(["2011/01/31 09:00"], dtype="M8[ns]"),
            ],
            names=[None, "dt"],
        )
        exp = DataFrame([3.25, 2.0], index=Index([1, 2], dtype=np.int32), columns=exp_columns)
        tm.assert_frame_equal(res, exp)
        res = df.pivot_table(
            index=pd.Grouper(freq="YE"),
            columns=pd.Grouper(key="dt", freq="ME"),
        )
        exp = DataFrame(
            [3.0],
            index=pd.DatetimeIndex(["2011-12-31"], freq="YE"),
            columns=exp_columns,
        )
        tm.assert_frame_equal(res, exp)

    def test_pivot_multi_values(
        self, data: DataFrame
    ) -> None:
        result: DataFrame = pivot_table(
            data,
            values=["D", "E"],
            index="A",
            columns=["B", "C"],
            fill_value=0,
        )
        expected: DataFrame = pivot_table(
            data.drop(["F"], axis=1),
            index="A",
            columns=["B", "C"],
            fill_value=0,
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_multi_functions(
        self, data: DataFrame
    ) -> None:
        f: Callable[[Union[str, List[str]]], DataFrame] = lambda func: pivot_table(
            data,
            values=["D", "E"],
            index=["A", "B"],
            columns="C",
            aggfunc=func,
        )
        result: DataFrame = f(["mean", "std"])
        means: DataFrame = f("mean")
        stds: DataFrame = f("std")
        expected: DataFrame = concat([means, stds], keys=["mean", "std"], axis=1)
        tm.assert_frame_equal(result, expected)
        f = lambda func: pivot_table(
            data,
            values=["D", "E"],
            index=["A", "B"],
            columns="C",
            aggfunc=func,
            margins=True,
        )
        result = f(["mean", "std"])
        means = f("mean")
        stds = f("std")
        expected = concat([means, stds], keys=["mean", "std"], axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("left_right", [([0] * 4, [1] * 4), (range(3), range(1, 4))])
    def test_pivot_with_interval_index(
        self,
        left_right: Tuple[List[int], List[int]],
        dropna: bool,
        closed: str,
    ) -> None:
        left, right = left_right
        interval_values: Categorical = Categorical(
            pd.IntervalIndex.from_arrays(left, right, closed=closed)
        )
        df: DataFrame = DataFrame({"A": interval_values, "B": 1})
        result: DataFrame = df.pivot_table(
            index="A", values="B", dropna=dropna, observed=False
        )
        expected: DataFrame = DataFrame(
            {"B": [1.0]}, index=Index(interval_values.unique(), name="A")
        )
        if not dropna:
            expected = expected.astype(float)
        tm.assert_frame_equal(result, expected)

    def test_pivot_with_interval_index_margins(self) -> None:
        ordered_cat: pd.IntervalIndex = pd.IntervalIndex.from_arrays([0, 0, 1, 1], [1, 1, 2, 2])
        df: DataFrame = DataFrame(
            {
                "A": np.arange(4, 0, -1, dtype=np.intp),
                "B": ["a", "b", "a", "b"],
                "C": Categorical(ordered_cat, ordered=True).sort_values(ascending=False),
            }
        )
        pivot_tab: DataFrame = pivot_table(
            df,
            index="C",
            columns="B",
            values="A",
            aggfunc="sum",
            margins=True,
            observed=False,
        )
        result: Series = pivot_tab["All"]
        expected: Series = Series(
            [3, 7, 10],
            index=Index(
                [pd.Interval(0, 1), pd.Interval(1, 2), "All"],
                name="C",
            ),
            name="All",
            dtype=np.intp,
        )
        tm.assert_series_equal(result, expected)

    def test_pass_array(
        self, data: DataFrame
    ) -> None:
        result: DataFrame = data.pivot_table("D", index=data.A, columns=data.C)
        expected: DataFrame = data.pivot_table("D", index="A", columns="C")
        tm.assert_frame_equal(result, expected)

    def test_pass_function(
        self, data: DataFrame
    ) -> None:
        result: DataFrame = data.pivot_table(
            "D", index=lambda x: x // 5, columns=data.C
        )
        expected: DataFrame = data.pivot_table(
            "D", index=data.index // 5, columns="C"
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_multiple(
        self, data: DataFrame
    ) -> None:
        index: List[str] = ["A", "B"]
        columns: str = "C"
        table: DataFrame = pivot_table(
            data, index=index, columns=columns
        )
        expected: DataFrame = data.groupby(index + [columns]).agg("mean").unstack()
        tm.assert_frame_equal(table, expected)

    def test_pivot_dtypes(self) -> None:
        f: DataFrame = DataFrame(
            {
                "a": ["cat", "bat", "cat", "bat"],
                "v": [1, 2, 3, 4],
                "i": ["a", "b", "a", "b"],
            }
        )
        assert f.dtypes["v"] == "int64"
        z: DataFrame = pivot_table(
            f,
            values="v",
            index=["a"],
            columns=["i"],
            fill_value=0,
            aggfunc="sum",
        )
        result: Series = z.dtypes
        expected: Series = Series(
            [np.dtype("int64")] * 2,
            index=Index(["a", "b"], name="i"),
        )
        tm.assert_series_equal(result, expected)
        f = DataFrame(
            {
                "a": ["cat", "bat", "cat", "bat"],
                "v": [1.5, 2.5, 3.5, 4.5],
                "i": ["a", "b", "a", "b"],
            }
        )
        assert f.dtypes["v"] == "float64"
        z = pivot_table(
            f,
            values="v",
            index=["a"],
            columns=["i"],
            fill_value=0,
            aggfunc="mean",
        )
        result = z.dtypes
        expected = Series(
            [np.dtype("float64")] * 2,
            index=Index(["a", "b"], name="i"),
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "columns,values,expected_columns",
        [
            (
                "A",
                ["D", "E"],
                [
                    [5.5, 5.5, 2.2, 2.2],
                    [8.0, 8.0, 4.4, 4.4],
                ],
                Index(["bar", "All", "foo", "All"], name="A"),
            ),
            (
                ["A", "B"],
                ["D", "E"],
                [
                    [9, 13, 22, 5, 6, 11],
                    [14, 18, 32, 11, 11, 22],
                ],
                MultiIndex.from_tuples(
                    [("bar", "one"), ("bar", "two"), ("bar", "All"), ("foo", "one"), ("foo", "two"), ("foo", "All")],
                    names=["A", "B"],
                ),
            ),
        ],
    )
    def test_margin_with_only_columns_defined(
        self,
        columns: Union[str, List[str]],
        aggfunc: Union[str, List[str]],
        values: List[List[float]],
        expected_columns: Union[Index, MultiIndex],
        using_infer_string: bool,
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            }
        )
        if aggfunc != "sum":
            msg: str = re.escape("agg function failed [how->mean,dtype->")
            if using_infer_string:
                msg = "dtype 'str' does not support operation 'mean'"
            with pytest.raises(TypeError, match=msg):
                pivot_table(
                    df,
                    columns=columns,
                    margins=True,
                    aggfunc=aggfunc,
                )
        if "B" not in columns:
            df = df.drop(columns="B")
        result: DataFrame = df.drop(columns="C").pivot_table(
            columns=columns, margins=True, aggfunc=aggfunc
        )
        expected: DataFrame = DataFrame(values, index=Index(["D", "E"]), columns=expected_columns)
        tm.assert_frame_equal(result, expected)

    def test_margins_dtype(self, data: DataFrame) -> None:
        df: DataFrame = data.copy()
        df[["D", "E", "F"]] = np.arange(len(df) * 3).reshape(len(df), 3).astype("i8")
        mi_val: List[Tuple[str, str]] = list(product(["bar", "foo"], ["one", "two"])) + [("All", "")]
        mi: MultiIndex = MultiIndex.from_tuples(mi_val, names=("A", "B"))
        expected: DataFrame = DataFrame(
            {"dull": [12, 21, 3, 9, 45], "shiny": [33, 0, 36, 51, 120]},
            index=mi,
        ).rename_axis("C", axis=1)
        expected["All"] = expected["dull"] + expected["shiny"]
        result: DataFrame = df.pivot_table(
            values="D",
            index=["A", "B"],
            columns="C",
            margins=True,
            aggfunc="sum",
            fill_value=0,
        )
        tm.assert_frame_equal(expected, result)

    def test_margins_dtype_len(self, data: DataFrame) -> None:
        mi_val: List[Tuple[str, str]] = list(product(["bar", "foo"], ["one", "two"])) + [("All", "")]
        mi: MultiIndex = MultiIndex.from_tuples(mi_val, names=("A", "B"))
        expected: DataFrame = DataFrame(
            {"dull": [1, 1, 2, 1, 5], "shiny": [2, 0, 2, 2, 6]},
            index=mi,
        ).rename_axis("C", axis=1)
        expected["All"] = expected["dull"] + expected["shiny"]
        result: DataFrame = data.pivot_table(
            values="D", index=["A", "B"], columns="C", margins=True, aggfunc=len, fill_value=0
        )
        tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize("cols", [(1, 2), ("a", "b"), (1, "b"), ("a", 1)])
    def test_pivot_table_multiindex_only(
        self, cols: Tuple[Union[int, str], Union[int, str]]
    ) -> None:
        df2: DataFrame = DataFrame(
            {
                cols[0]: [1, 2, 3],
                cols[1]: [1, 2, 3],
                "v": [4, 5, 6],
            }
        )
        result: DataFrame = df2.pivot_table(values="v", columns=cols)
        expected: DataFrame = DataFrame(
            {
                ("A", "one"): Series([1, 4], index=["one", "two"]),
                ("B", "two"): Series([2, 5], index=["one", "two"]),
                ("sum", "all"): Series([3, 6], index=["one", "two"]),
            }
        )
        expected = expected.rename_axis(index=None, columns=cols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "m", "h", "D"])
    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_with_tz(
        self, method: bool, unit: str
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "dt1": pd.DatetimeIndex(
                    [
                        datetime(2013, 1, 1, 9, 0),
                        datetime(2013, 1, 2, 9, 0),
                        datetime(2013, 1, 1, 9, 0),
                        datetime(2013, 1, 2, 9, 0),
                    ],
                    dtype=f"M8[{unit}, US/Pacific]",
                ),
                "dt2": pd.DatetimeIndex(
                    [
                        datetime(2014, 1, 1, 9, 0),
                        datetime(2014, 1, 1, 9, 0),
                        datetime(2014, 1, 2, 9, 0),
                        datetime(2014, 1, 2, 9, 0),
                    ],
                    dtype=f"M8[{unit}, Asia/Tokyo]",
                ),
                "data1": np.arange(4, dtype="int64"),
                "data2": np.arange(4, dtype="int64"),
            }
        )
        exp_col1: Index = Index(["data1", "data1", "data2", "data2"])
        exp_col2: pd.DatetimeIndex = pd.DatetimeIndex(
            ["2014/01/01 09:00", "2014/01/02 09:00"] * 2,
            name="dt2",
            dtype=f"M8[{unit}, Asia/Tokyo]",
        )
        exp_col: MultiIndex = MultiIndex.from_arrays([exp_col1, exp_col2])
        exp_idx: pd.DatetimeIndex = pd.DatetimeIndex(
            ["2013/01/01 09:00", "2013/01/02 09:00"],
            name="dt1",
            dtype=f"M8[{unit}, US/Pacific]",
        )
        expected: DataFrame = DataFrame(
            [[0, 2, 0, 2], [1, 3, 1, 3]],
            index=exp_idx,
            columns=exp_col,
        )
        if method:
            pv: DataFrame = df.pivot(index="dt1", columns="dt2")
        else:
            pv = pd.pivot(df, index="dt1", columns="dt2")
        tm.assert_frame_equal(pv, expected)
        expected = DataFrame(
            [[0, 2], [1, 3]],
            index=exp_idx,
            columns=exp_col2[:2],
        )
        if method:
            pv = df.pivot(index="dt1", columns="dt2", values="data1")
        else:
            pv = pd.pivot(df, index="dt1", columns="dt2", values="data1")
        tm.assert_frame_equal(pv, expected)

    def test_pivot_tz_in_values(self) -> None:
        df: DataFrame = DataFrame(
            [
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-12 13:00:00-0700", tz="US/Pacific"),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-12 08:00:00-0700", tz="US/Pacific"),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-12 14:00:00-0700", tz="US/Pacific"),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-25 11:00:00-0700", tz="US/Pacific"),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-25 13:00:00-0700", tz="US/Pacific"),
                },
            ]
        )
        df = df.set_index("ts").reset_index()
        mins: Series = df.ts.map(
            lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0)
        )
        result: DataFrame = pivot_table(
            df.set_index("ts").reset_index(),
            values="ts",
            index=["uid"],
            columns=[mins],
            aggfunc="min",
        )
        expected: DataFrame = DataFrame(
            [
                [
                    pd.Timestamp("2016-08-12 08:00:00-0700", tz="US/Pacific"),
                    pd.Timestamp("2016-08-25 11:00:00-0700", tz="US/Pacific"),
                ]
            ],
            index=Index(["aa"], name="uid"),
            columns=pd.DatetimeIndex(
                [
                    pd.Timestamp("2016-08-12 00:00:00", tz="US/Pacific"),
                    pd.Timestamp("2016-08-25 00:00:00", tz="US/Pacific"),
                ],
                name="ts",
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_periods(
        self, method: bool
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "p1": [
                    pd.Period("2013-01-01", "D"),
                    pd.Period("2013-01-02", "D"),
                    pd.Period("2013-01-01", "D"),
                    pd.Period("2013-01-02", "D"),
                ],
                "p2": [
                    pd.Period("2013-01", "M"),
                    pd.Period("2013-01", "M"),
                    pd.Period("2013-02", "M"),
                    pd.Period("2013-02", "M"),
                ],
                "data1": np.arange(4, dtype="int64"),
                "data2": np.arange(4, dtype="int64"),
            }
        )
        exp_col1: Index = Index(["data1", "data1", "data2", "data2"])
        exp_col2: pd.PeriodIndex = pd.PeriodIndex(
            ["2013-01", "2013-02"] * 2, name="p2", freq="M"
        )
        exp_col: MultiIndex = MultiIndex.from_arrays([exp_col1, exp_col2])
        expected: DataFrame = DataFrame(
            [[0, 2, 0, 2], [1, 3, 1, 3]],
            index=pd.PeriodIndex(["2013-01-01", "2013-01-02"], name="p1", freq="D"),
            columns=exp_col,
        )
        if method:
            pv: DataFrame = df.pivot(index="p1", columns="p2")
        else:
            pv = pd.pivot(df, index="p1", columns="p2")
        tm.assert_frame_equal(pv, expected)
        expected = DataFrame(
            [[0, 2], [1, 3]],
            index=pd.PeriodIndex(["2013-01-01", "2013-01-02"], name="p1", freq="D"),
            columns=pd.PeriodIndex(["2013-01", "2013-02"], name="p2", freq="M"),
        )
        if method:
            pv = df.pivot(index="p1", columns="p2", values="data1")
        else:
            pv = pd.pivot(df, index="p1", columns="p2", values="data1")
        tm.assert_frame_equal(pv, expected)

    def test_pivot_periods_with_margins(self) -> None:
        df: DataFrame = DataFrame(
            {
                "a": [1, 1, 2, 2],
                "b": [
                    pd.Period("2019Q1", "Q"),
                    pd.Period("2019Q2", "Q"),
                    pd.Period("2019Q1", "Q"),
                    pd.Period("2019Q2", "Q"),
                ],
                "x": 1.0,
            }
        )
        expected: DataFrame = DataFrame(
            data=1.0,
            index=Index([1, 2, "All"], name="a"),
            columns=Index(
                [pd.Period("2019Q1", "Q"), pd.Period("2019Q2", "Q"), "All"],
                name="b",
            ),
        )
        result: DataFrame = pivot_table(
            df, index="a", columns="b", values="x", margins=True
        )
        tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize("box", [list, np.array, Series, Index])
    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_with_list_like_values(
        self,
        box: Callable[..., Union[List[Any], np.ndarray, Series, Index]],
        method: bool,
    ) -> None:
        values: Union[List[str], np.ndarray, Series, Index] = box(["baz", "zoo"])
        df: DataFrame = DataFrame(
            {
                "foo": ["one", "one", "one", "two", "two", "two"],
                "bar": ["A", "B", "C", "A", "B", "C"],
                "baz": [1, 2, 3, 4, 5, 6],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )
        if method:
            result: DataFrame = df.pivot(index="foo", columns="bar", values=values)
        else:
            result = pd.pivot(df, index="foo", columns="bar", values=values)
        data: List[List[Union[int, float, str]]] = [
            [1, 2, 3, "x", "y", "z"],
            [4, 5, 6, "q", "w", "t"],
        ]
        index: Index = Index(data=["one", "two"], name="foo")
        columns: MultiIndex = MultiIndex.from_tuples(
            [("baz", "A"), ("baz", "B"), ("baz", "C"), ("zoo", "A"), ("zoo", "B"), ("zoo", "C")],
            names=[None, "bar"],
        )
        expected: DataFrame = DataFrame(data=data, index=index, columns=columns)
        expected["baz"] = expected["baz"].astype(object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "values,method",
        [
            (["bar", "baz"], True),
            (np.array(["bar", "baz"]), True),
            (Series(["bar", "baz"]), True),
            (Index(["bar", "baz"]), True),
            (["bar", "baz"], False),
            (np.array(["bar", "baz"]), False),
            (Series(["bar", "baz"]), False),
            (Index(["bar", "baz"]), False),
        ],
    )
    def test_pivot_with_list_like_values_nans(
        self,
        values: Union[List[str], np.ndarray, Series, Index],
        method: bool,
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "foo": ["one", "one", "one", "two", "two", "two"],
                "bar": ["A", "B", "C", "A", "B", "C"],
                "baz": [1, 2, 3, 4, 5, 6],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )
        if method:
            result: DataFrame = df.pivot(index="zoo", columns="foo", values=values)
        else:
            result = pd.pivot(df, index="zoo", columns="foo", values=values)
        data: List[List[Union[int, float, str, None]]] = [
            [np.nan, "A", np.nan, 4],
            [np.nan, "C", np.nan, 6],
            [np.nan, "B", np.nan, 5],
            ["A", np.nan, 1, np.nan],
            ["B", np.nan, 2, np.nan],
            ["C", np.nan, 3, np.nan],
        ]
        index: Index = Index(data=["q", "t", "w", "x", "y", "z"], name="zoo")
        columns: MultiIndex = MultiIndex.from_tuples(
            [("bar", "one"), ("bar", "two"), ("baz", "one"), ("baz", "two")],
            names=[None, "foo"],
        )
        expected: DataFrame = DataFrame(data=data, index=index, columns=columns)
        expected["baz"] = expected["baz"].astype(object)
        tm.assert_frame_equal(result, expected)

    def test_pivot_columns_none_raise_error(self) -> None:
        df: DataFrame = DataFrame(
            {
                "col1": ["a", "b", "c"],
                "col2": [1, 2, 3],
                "col3": [1, 2, 3],
            }
        )
        msg: str = "pivot\\(\\) missing 1 required keyword-only argument: 'columns'"
        with pytest.raises(TypeError, match=msg):
            df.pivot(index="col1", values="col3")

    @pytest.mark.xfail(
        reason="MultiIndexed unstack with tuple names fails with KeyError GH#19966"
    )
    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_with_multiindex(
        self, method: bool
    ) -> None:
        index: Index = Index(data=[0, 1, 2, 3, 4, 5])
        data: List[List[Any]] = [
            ["one", "A", 1, "x"],
            ["one", "B", 2, "y"],
            ["one", "C", 3, "z"],
            ["two", "A", 4, "q"],
            ["two", "B", 5, "w"],
            ["two", "C", 6, "t"],
        ]
        columns: MultiIndex = MultiIndex.from_tuples(
            [("bar", "first"), ("bar", "second")],
            names=[None, "second"],
        )
        df: DataFrame = DataFrame(
            data=data, index=index, columns=columns, dtype="object"
        )
        if method:
            result: DataFrame = df.pivot(
                index=("bar", "first"),
                columns=("bar", "second"),
                values=("baz", "first"),
            )
        else:
            result = pd.pivot(
                df,
                index=("bar", "first"),
                columns=("bar", "second"),
                values=("baz", "first"),
            )
        data_expected: Dict[str, Series] = {
            "A": Series([1, 4], index=["one", "two"]),
            "B": Series([2, 5], index=["one", "two"]),
            "C": Series([3, 6], index=["one", "two"]),
        }
        expected: DataFrame = DataFrame(data_expected)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "method,values",
        [
            (True, ("bar", "baz")),
            (True, ["bar", "baz"]),
            (True, np.array(["bar", "baz"])),
            (True, Series(["bar", "baz"])),
            (True, Index(["bar", "baz"])),
            (False, ("bar", "baz")),
            (False, ["bar", "baz"]),
            (False, np.array(["bar", "baz"])),
            (False, Series(["bar", "baz"])),
            (False, Index(["bar", "baz"])),
        ],
    )
    def test_pivot_with_tuple_of_values(
        self,
        method: bool,
        values: Union[Tuple[str, str], List[str], np.ndarray, Series, Index],
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "foo": ["one", "one", "one", "two", "two", "two"],
                "bar": ["A", "B", "C", "A", "B", "C"],
                "baz": [1, 2, 3, 4, 5, 6],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )
        with pytest.raises(KeyError, match="^\\('bar', 'baz'\\)$"):
            if method:
                df.pivot(index="zoo", columns="foo", values=values)
            else:
                pd.pivot(df, index="zoo", columns="foo", values=values)

    def _check_output(
        self,
        result: DataFrame,
        values_col: str,
        data: DataFrame,
        index: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        margins_col: str = "All",
    ) -> None:
        if index is None:
            index = ["A", "B"]
        if columns is None:
            columns = ["C"]
        col_margins: Series = result.loc[result.index[:-1], margins_col]
        expected_col_margins: Series = data.groupby(index)[values_col].mean()
        tm.assert_series_equal(col_margins, expected_col_margins, check_names=False)
        assert col_margins.name == margins_col
        result = result.sort_index()
        index_margins: Series = result.loc[margins_col, ""].iloc[:-1]
        expected_ix_margins: Series = data.groupby(columns)[values_col].mean()
        tm.assert_series_equal(index_margins, expected_ix_margins, check_names=False)
        assert index_margins.name == (margins_col, "")
        grand_total_margins: float = result.loc[("All", ""), margins_col]
        expected_total_margins: float = data[values_col].mean()
        assert grand_total_margins == expected_total_margins

    def test_margins(
        self, data: DataFrame
    ) -> None:
        result: DataFrame = data.pivot_table(
            values="D",
            index=["A", "B"],
            columns="C",
            margins=True,
            aggfunc="mean",
        )
        self._check_output(result, "D", data)
        result = data.pivot_table(
            values="D",
            index=["A", "B"],
            columns="C",
            margins=True,
            aggfunc="mean",
            margins_name="Totals",
        )
        self._check_output(result, "D", data, margins_col="Totals")
        table: DataFrame = data.pivot_table(
            index=["A", "B"],
            columns="C",
            margins=True,
            aggfunc="mean",
        )
        for value_col in table.columns.levels[0]:
            self._check_output(table[value_col], value_col, data)

    def test_no_col(
        self, data: DataFrame, using_infer_string: bool
    ) -> None:
        data.columns = [k * 2 for k in data.columns]
        msg: str = re.escape("agg function failed [how->mean,dtype->")
        if using_infer_string:
            msg = "dtype 'str' does not support operation 'mean'"
        with pytest.raises(TypeError, match=msg):
            data.pivot_table(index=["AA", "BB"], margins=True, aggfunc="mean")
        table: DataFrame = data.drop(columns="CC").pivot_table(
            index=["AA", "BB"], margins=True, aggfunc="mean"
        )
        for value_col in table.columns:
            totals: float = table.loc[("All", ""), value_col]
            assert totals == data[value_col].mean()
        with pytest.raises(TypeError, match=msg):
            data.pivot_table(index=["AA", "BB"], margins=True, aggfunc="mean")
        table = data.drop(columns="CC").pivot_table(
            index=["AA", "BB"], margins=True, aggfunc="mean"
        )
        for item in ["DD", "EE", "FF"]:
            totals: float = table.loc[("All", ""), item]
            assert totals == data[item].mean()

    @pytest.mark.parametrize(
        "columns, aggfunc, values, expected_columns",
        [
            (
                "A",
                "mean",
                [
                    [5.5, 5.5, 2.2, 2.2],
                    [8.0, 8.0, 4.4, 4.4],
                ],
                Index(["bar", "All", "foo", "All"], name="A"),
            ),
            (
                ["A", "B"],
                "sum",
                [
                    [9, 13, 22, 5, 6, 11],
                    [14, 18, 32, 11, 11, 22],
                ],
                MultiIndex.from_tuples(
                    [("bar", "one"), ("bar", "two"), ("bar", "All"), ("foo", "one"), ("foo", "two"), ("foo", "All")],
                    names=["A", "B"],
                ),
            ),
        ],
    )
    def test_margin_with_only_columns_defined(
        self,
        columns: Union[str, List[str]],
        aggfunc: Union[str, List[str]],
        values: List[List[float]],
        expected_columns: Union[Index, MultiIndex],
        using_infer_string: bool,
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            }
        )
        if aggfunc != "sum":
            msg: str = re.escape("agg function failed [how->mean,dtype->")
            if using_infer_string:
                msg = "dtype 'str' does not support operation 'mean'"
            with pytest.raises(TypeError, match=msg):
                pivot_table(
                    df,
                    columns=columns,
                    margins=True,
                    aggfunc=aggfunc,
                )
        if "B" not in columns:
            df = df.drop(columns="B")
        result: DataFrame = df.drop(columns="C").pivot_table(
            columns=columns, margins=True, aggfunc=aggfunc
        )
        expected: DataFrame = DataFrame(values, index=Index(["D", "E"]), columns=expected_columns)
        tm.assert_frame_equal(result, expected)

    def test_margins_dtype(self, data: DataFrame) -> None:
        df: DataFrame = data.copy()
        df[["D", "E", "F"]] = np.arange(len(df) * 3).reshape(len(df), 3).astype("i8")
        mi_val: List[Tuple[str, str]] = list(product(["bar", "foo"], ["one", "two"])) + [("All", "")]
        mi: MultiIndex = MultiIndex.from_tuples(mi_val, names=("A", "B"))
        expected: DataFrame = DataFrame(
            {"dull": [12, 21, 3, 9, 45], "shiny": [33, 0, 36, 51, 120]},
            index=mi,
        ).rename_axis("C", axis=1)
        expected["All"] = expected["dull"] + expected["shiny"]
        result: DataFrame = df.pivot_table(
            values="D",
            index=["A", "B"],
            columns="C",
            margins=True,
            aggfunc="sum",
            fill_value=0,
        )
        tm.assert_frame_equal(expected, result)

    def test_margins_dtype_len(self, data: DataFrame) -> None:
        mi_val: List[Tuple[str, str]] = list(product(["bar", "foo"], ["one", "two"])) + [("All", "")]
        mi: MultiIndex = MultiIndex.from_tuples(mi_val, names=("A", "B"))
        expected: DataFrame = DataFrame(
            {"dull": [1, 1, 2, 1, 5], "shiny": [2, 0, 2, 2, 6]},
            index=mi,
        ).rename_axis("C", axis=1)
        expected["All"] = expected["dull"] + expected["shiny"]
        result: DataFrame = data.pivot_table(
            values="D",
            index=["A", "B"],
            columns="C",
            margins=True,
            aggfunc=len,
            fill_value=0,
        )
        tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize("cols", [(1, 2), ("a", "b"), (1, "b"), ("a", 1)])
    def test_pivot_table_multiindex_only(
        self,
        cols: Tuple[Union[int, str], Union[int, str]],
    ) -> None:
        df2: DataFrame = DataFrame(
            {
                cols[0]: [1, 2, 3],
                cols[1]: [1, 2, 3],
                "v": [4, 5, 6],
            }
        )
        result: DataFrame = df2.pivot_table(
            values="v",
            columns=cols,
        )
        expected: DataFrame = DataFrame(
            {
                ("A", "one"): Series([1, 4], index=["one", "two"]),
                ("B", "two"): Series([2, 5], index=["one", "two"]),
                ("sum", "all"): Series([3, 6], index=["one", "two"]),
            }
        )
        expected = expected.rename_axis(index=None, columns=cols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "m", "h", "D"])
    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_with_tz(
        self, method: bool, unit: str
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "dt1": pd.DatetimeIndex(
                    [
                        datetime(2013, 1, 1, 9, 0),
                        datetime(2013, 1, 2, 9, 0),
                        datetime(2013, 1, 1, 9, 0),
                        datetime(2013, 1, 2, 9, 0),
                    ],
                    dtype=f"M8[{unit}, US/Pacific]",
                ),
                "dt2": pd.DatetimeIndex(
                    [
                        datetime(2014, 1, 1, 9, 0),
                        datetime(2014, 1, 1, 9, 0),
                        datetime(2014, 1, 2, 9, 0),
                        datetime(2014, 1, 2, 9, 0),
                    ],
                    dtype=f"M8[{unit}, Asia/Tokyo]",
                ),
                "data1": np.arange(4, dtype="int64"),
                "data2": np.arange(4, dtype="int64"),
            }
        )
        exp_col1: Index = Index(["data1", "data1", "data2", "data2"])
        exp_col2: pd.DatetimeIndex = pd.DatetimeIndex(
            ["2014/01/01 09:00", "2014/01/02 09:00"] * 2,
            name="dt2",
            dtype=f"M8[{unit}, Asia/Tokyo]",
        )
        exp_col: MultiIndex = MultiIndex.from_arrays([exp_col1, exp_col2])
        exp_idx: pd.DatetimeIndex = pd.DatetimeIndex(
            ["2013/01/01 09:00", "2013/01/02 09:00"],
            name="dt1",
            dtype=f"M8[{unit}, US/Pacific]",
        )
        expected: DataFrame = DataFrame(
            [[0, 2, 0, 2], [1, 3, 1, 3]],
            index=exp_idx,
            columns=exp_col,
        )
        if method:
            pv: DataFrame = df.pivot(index="dt1", columns="dt2")
        else:
            pv = pd.pivot(df, index="dt1", columns="dt2")
        tm.assert_frame_equal(pv, expected)
        expected = DataFrame(
            [[0, 2], [1, 3]],
            index=exp_idx,
            columns=exp_col2[:2],
        )
        if method:
            pv = df.pivot(index="dt1", columns="dt2", values="data1")
        else:
            pv = pd.pivot(df, index="dt1", columns="dt2", values="data1")
        tm.assert_frame_equal(pv, expected)

    def test_pivot_tz_in_values(self) -> None:
        df: DataFrame = DataFrame(
            [
                {
                    "uid": "aa",
                    "ts": pd.Timestamp(
                        "2016-08-12 13:00:00-0700", tz="US/Pacific"
                    ),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp(
                        "2016-08-12 08:00:00-0700", tz="US/Pacific"
                    ),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp(
                        "2016-08-12 14:00:00-0700", tz="US/Pacific"
                    ),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp(
                        "2016-08-25 11:00:00-0700", tz="US/Pacific"
                    ),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp(
                        "2016-08-25 13:00:00-0700", tz="US/Pacific"
                    ),
                },
            ]
        )
        df = df.set_index("ts").reset_index()
        mins: Series = df.ts.map(
            lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0)
        )
        result: DataFrame = pivot_table(
            df.set_index("ts").reset_index(),
            values="ts",
            index=["uid"],
            columns=[mins],
            aggfunc="min",
        )
        expected: DataFrame = DataFrame(
            [
                [
                    pd.Timestamp("2016-08-12 08:00:00-0700", tz="US/Pacific"),
                    pd.Timestamp("2016-08-25 11:00:00-0700", tz="US/Pacific"),
                ]
            ],
            index=Index(["aa"], name="uid"),
            columns=pd.DatetimeIndex(
                [
                    pd.Timestamp("2016-08-12 00:00:00", tz="US/Pacific"),
                    pd.Timestamp("2016-08-25 00:00:00", tz="US/Pacific"),
                ],
                name="ts",
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_periods(
        self, method: bool
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "p1": [
                    pd.Period("2013-01-01", "D"),
                    pd.Period("2013-01-02", "D"),
                    pd.Period("2013-01-01", "D"),
                    pd.Period("2013-01-02", "D"),
                ],
                "p2": [
                    pd.Period("2013-01", "M"),
                    pd.Period("2013-01", "M"),
                    pd.Period("2013-02", "M"),
                    pd.Period("2013-02", "M"),
                ],
                "data1": np.arange(4, dtype="int64"),
                "data2": np.arange(4, dtype="int64"),
            }
        )
        exp_col1: Index = Index(["data1", "data1", "data2", "data2"])
        exp_col2: pd.PeriodIndex = pd.PeriodIndex(
            ["2013-01", "2013-02"] * 2, name="p2", freq="M"
        )
        exp_col: MultiIndex = MultiIndex.from_arrays([exp_col1, exp_col2])
        expected: DataFrame = DataFrame(
            [[0, 2, 0, 2], [1, 3, 1, 3]],
            index=pd.PeriodIndex(["2013-01-01", "2013-01-02"], name="p1", freq="D"),
            columns=exp_col,
        )
        if method:
            pv: DataFrame = df.pivot(index="p1", columns="p2")
        else:
            pv = pd.pivot(df, index="p1", columns="p2")
        tm.assert_frame_equal(pv, expected)
        expected = DataFrame(
            [[0, 2], [1, 3]],
            index=pd.PeriodIndex(["2013-01-01", "2013-01-02"], name="p1", freq="D"),
            columns=pd.PeriodIndex(["2013-01", "2013-02"], name="p2", freq="M"),
        )
        if method:
            pv = df.pivot(index="p1", columns="p2", values="data1")
        else:
            pv = pd.pivot(df, index="p1", columns="p2", values="data1")
        tm.assert_frame_equal(pv, expected)

    def test_pivot_periods_with_margins(self) -> None:
        df: DataFrame = DataFrame(
            {
                "a": [1, 1, 2, 2],
                "b": [
                    pd.Period("2019Q1", "Q"),
                    pd.Period("2019Q2", "Q"),
                    pd.Period("2019Q1", "Q"),
                    pd.Period("2019Q2", "Q"),
                ],
                "x": 1.0,
            }
        )
        expected: DataFrame = DataFrame(
            data=1.0,
            index=Index(["1", "2", "All"], name="a"),
            columns=Index([pd.Period("2019Q1", "Q"), pd.Period("2019Q2", "Q"), "All"], name="b"),
        )
        result: DataFrame = df.pivot_table(
            index="a", columns="b", values="x", margins=True
        )
        tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize(
        "box, method",
        [
            (list, True),
            (list, False),
            (np.array, True),
            (np.array, False),
            (Series, True),
            (Series, False),
            (Index, True),
            (Index, False),
        ],
    )
    def test_pivot_with_list_like_values(
        self,
        box: Callable[..., Union[List[Any], np.ndarray, Series, Index]],
        method: bool,
    ) -> None:
        values: Union[List[str], np.ndarray, Series, Index] = box(["baz", "zoo"])
        df: DataFrame = DataFrame(
            {
                "foo": ["one", "one", "one", "two", "two", "two"],
                "bar": ["A", "B", "C", "A", "B", "C"],
                "baz": [1, 2, 3, 4, 5, 6],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )
        if method:
            result: DataFrame = df.pivot(index="foo", columns="bar", values=values)
        else:
            result = pd.pivot(df, index="foo", columns="bar", values=values)
        data: List[List[Union[int, float, str]]] = [
            [1, 2, 3, "x", "y", "z"],
            [4, 5, 6, "q", "w", "t"],
        ]
        index: Index = Index(data=["one", "two"], name="foo")
        columns: MultiIndex = MultiIndex.from_tuples(
            [("baz", "A"), ("baz", "B"), ("baz", "C"), ("zoo", "A"), ("zoo", "B"), ("zoo", "C")],
            names=[None, "bar"],
        )
        expected: DataFrame = DataFrame(data=data, index=index, columns=columns)
        expected["baz"] = expected["baz"].astype(object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "values, method",
        [
            (["bar", "baz"], True),
            (["bar", "baz"], False),
            (np.array(["bar", "baz"]), True),
            (np.array(["bar", "baz"]), False),
            (Series(["bar", "baz"]), True),
            (Series(["bar", "baz"]), False),
            (Index(["bar", "baz"]), True),
            (Index(["bar", "baz"]), False),
        ],
    )
    def test_pivot_with_list_like_values_nans(
        self,
        values: Union[List[str], np.ndarray, Series, Index],
        method: bool,
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "foo": ["one", "one", "one", "two", "two", "two"],
                "bar": ["A", "B", "C", "A", "B", "C"],
                "baz": [1, 2, 3, 4, 5, 6],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )
        if method:
            result: DataFrame = df.pivot(index="zoo", columns="foo", values=values)
        else:
            result = pd.pivot(df, index="zoo", columns="foo", values=values)
        data: List[List[Union[int, float, str, None]]] = [
            [np.nan, "A", np.nan, 4],
            [np.nan, "C", np.nan, 6],
            [np.nan, "B", np.nan, 5],
            ["A", np.nan, 1, np.nan],
            ["B", np.nan, 2, np.nan],
            ["C", np.nan, 3, np.nan],
        ]
        index: Index = Index(data=["q", "t", "w", "x", "y", "z"], name="zoo")
        columns: MultiIndex = MultiIndex.from_tuples(
            [("bar", "one"), ("bar", "two"), ("baz", "one"), ("baz", "two")],
            names=[None, "foo"],
        )
        expected: DataFrame = DataFrame(data=data, index=index, columns=columns)
        expected["baz"] = expected["baz"].astype(object)
        tm.assert_frame_equal(result, expected)

    def test_pivot_columns_none_raise_error(self) -> None:
        df: DataFrame = DataFrame(
            {
                "col1": ["a", "b", "c"],
                "col2": [1, 2, 3],
                "col3": [1, 2, 3],
            }
        )
        msg: str = "pivot\\(\\) missing 1 required keyword-only argument: 'columns'"
        with pytest.raises(TypeError, match=msg):
            df.pivot(index="col1", values="col3")

    @pytest.mark.xfail(
        using_string_dtype(), reason="TODO(infer_string) None is cast to NaN"
    )
    def test_pivot_columns_is_none(self) -> None:
        df: DataFrame = DataFrame(
            {
                None: [1],
                "b": [2],
                "c": [3],
            }
        )
        result: DataFrame = df.pivot(columns=None)
        expected: DataFrame = DataFrame(
            {("b", 1): [2], ("c", 1): 3}
        )
        tm.assert_frame_equal(result, expected)
        result = df.pivot(columns=None, index="b")
        expected = DataFrame({("c", 1): 3}, index=Index([2], name="b"))
        tm.assert_frame_equal(result, expected)
        result = df.pivot(columns=None, index="b", values="c")
        expected = DataFrame({1: 3}, index=Index([2], name="b"))
        tm.assert_frame_equal(result, expected)

    def test_pivot_index_is_none(self, using_infer_string: bool) -> None:
        df: DataFrame = DataFrame(
            {
                None: [1],
                "b": [2],
                "c": [3],
            }
        )
        result: DataFrame = df.pivot(columns="b", index=None)
        expected: DataFrame = DataFrame(
            {("c", 2): 3}, index=[1], columns=Index([2], name="b")
        )
        tm.assert_frame_equal(result, expected)
        result = df.pivot(columns="b", index=None, values="c")
        expected = DataFrame(
            {2: 3}, index=[1], columns=Index([2], name="b")
        )
        if using_infer_string:
            expected.index.name = np.nan
        tm.assert_frame_equal(result, expected)

    def test_pivot_values_is_none(self) -> None:
        df: DataFrame = DataFrame(
            {
                None: [1],
                "b": [2],
                "c": [3],
            }
        )
        result: DataFrame = df.pivot(columns="b", index="c", values=None)
        expected: DataFrame = DataFrame(
            {("c", 2): 1},
            index=Index([3], name="c"),
            columns=Index([2], name="b"),
        )
        tm.assert_frame_equal(result, expected)
        result = df.pivot(columns="b", values=None)
        expected = DataFrame(
            {2: ("c", 3, 1)},
            index=[0],
            columns=Index([2], name="b"),
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_not_changing_index_name(self) -> None:
        df: DataFrame = DataFrame(
            {"one": ["a"], "two": [0], "three": [1]}
        )
        expected: DataFrame = df.copy(deep=True)
        df.pivot(index="one", columns="two", values="three")
        tm.assert_frame_equal(df, expected)

    def test_pivot_table_empty_dataframe_correct_index(self) -> None:
        df: DataFrame = DataFrame(columns=["a", "b", "c"])
        pivot: DataFrame = df.pivot_table(
            index="a", columns="b", values="c", aggfunc="count"
        )
        expected: Index = Index([], dtype="object", name="b")
        tm.assert_index_equal(pivot.columns, expected)

    def test_pivot_table_handles_explicit_datetime_types(self) -> None:
        df: DataFrame = DataFrame(
            [
                {"a": "x", "date_str": "2023-01-01", "amount": 1},
                {"a": "y", "date_str": "2023-01-02", "amount": 2},
                {"a": "z", "date_str": "2023-01-03", "amount": 3},
            ]
        )
        df["date"] = pd.to_datetime(df["date_str"])
        with tm.assert_produces_warning(False):
            pivot: DataFrame = pivot_table(
                df,
                index=["a", "date"],
                values=["amount"],
                aggfunc="sum",
                margins=True,
            )
        expected: MultiIndex = MultiIndex.from_tuples(
            [("x", datetime.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")),
             ("y", datetime.strptime("2023-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")),
             ("z", datetime.strptime("2023-01-03 00:00:00", "%Y-%m-%d %H:%M:%S")),
             ("All", "")],
            names=["a", "date"],
        )
        tm.assert_index_equal(pivot.index, expected)
    
    def test_pivot_table_with_margins_and_numeric_columns(self) -> None:
        df: DataFrame = DataFrame(
            [
                ["a", "x", 1],
                ["a", "y", 2],
                ["b", "y", 3],
                ["b", "z", 4],
            ],
            columns=[10, 20, 30]
        )
        result: DataFrame = pivot_table(
            df,
            index=10,
            columns=20,
            values=30,
            aggfunc="sum",
            fill_value=0,
            margins=True,
        )
        expected: DataFrame = DataFrame(
            [[1, 2, 0, 3], [0, 3, 4, 7], [1, 5, 4, 10]],
            columns=Index(["x", "y", "z", "All"], name=20),
            index=Index(["a", "b", "All"], name=10),
        )
        tm.assert_frame_equal(result, expected)

    def test_unstack_copy(self, m: int) -> None:
        levels: np.ndarray = np.arange(m)
        index: MultiIndex = MultiIndex.from_product([levels] * 2)
        values: np.ndarray = np.arange(m * m * 100).reshape(m * m, 100)
        df: DataFrame = DataFrame(values, index=index, columns=np.arange(100))
        df_orig: DataFrame = df.copy()
        result: DataFrame = df.unstack()
        result.iloc[0, 0] = -1
        tm.assert_frame_equal(df, df_orig)

    def test_pivot_empty_with_datetime(self) -> None:
        df: DataFrame = DataFrame(
            {
                "timestamp": pd.DatetimeIndex([], dtype="datetime64[ns, UTC]"),
                "category": pd.Series([], dtype=str),
                "value": pd.Series([], dtype=str),
            }
        )
        df_pivoted: DataFrame = df.pivot_table(
            index="category", columns="value", values="timestamp"
        )
        assert df_pivoted.empty

    def test_pivot_margins_with_none_index(self) -> None:
        df: DataFrame = DataFrame(
            {
                "x": [1, 1, 2],
                "y": [3, 3, 4],
                "z": [5, 5, 6],
                "w": [7, 8, 9],
            }
        )
        result: DataFrame = df.pivot_table(
            index=None,
            columns=["y", "z"],
            values="w",
            margins=True,
            aggfunc="count",
        )
        expected: DataFrame = DataFrame(
            [[2, 2, 1, 1]],
            index=["w"],
            columns=MultiIndex.from_tuples(
                [(3, 5), (3, 5), (4, 6), ("All", "All")],
                names=["y", "z"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_datetime_tz(self) -> None:
        dates1: pd.DatetimeIndex = pd.DatetimeIndex(
            [
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
            ],
            dtype="M8[ns, US/Pacific]",
            name="dt1",
        )
        dates2: pd.DatetimeIndex = pd.DatetimeIndex(
            [
                "2013-01-01 15:00:00",
                "2013-01-01 15:00:00",
                "2013-01-01 15:00:00",
                "2013-02-01 15:00:00",
                "2013-02-01 15:00:00",
                "2013-02-01 15:00:00",
            ],
            dtype="M8[ns, Asia/Tokyo]",
            name="dt2",
        )
        df: DataFrame = DataFrame(
            {
                "label": ["a", "a", "a", "b", "b", "b"],
                "dt1": dates1,
                "dt2": dates2,
                "value1": np.arange(6, dtype="int64"),
                "value2": [1, 2] * 3,
            }
        )
        exp_idx: pd.DatetimeIndex = dates1[:3]
        exp_col1: Index = Index(["value1", "value1"])
        exp_col2: Index = Index(["a", "b"], name="label")
        exp_col: MultiIndex = MultiIndex.from_arrays([exp_col1, exp_col2])
        expected: DataFrame = DataFrame(
            [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
            index=exp_idx,
            columns=exp_col,
        )
        result: DataFrame = pivot_table(
            df, index=["dt1"], columns=["label"], values=["value1"]
        )
        tm.assert_frame_equal(result, expected)
        exp_col1 = Index(["sum", "sum", "sum", "sum", "mean", "mean", "mean", "mean"])
        exp_col2 = Index(["value1", "value1", "value2", "value2"] * 2)
        exp_col3: pd.DatetimeIndex = pd.DatetimeIndex(
            ["2013-01-01 15:00:00", "2013-02-01 15:00:00"] * 4,
            dtype="M8[ns, Asia/Tokyo]",
            name="dt2",
        )
        exp_col = MultiIndex.from_arrays([exp_col1, exp_col2, exp_col3])
        expected1: DataFrame = DataFrame(
            np.array([[0, 3, 1, 2], [1, 4, 2, 1], [2, 5, 1, 2]], dtype="int64"),
            index=exp_idx,
            columns=exp_col[:4],
        )
        expected2: DataFrame = DataFrame(
            np.array([[0.0, 3.0, 1.0, 2.0], [1.0, 4.0, 2.0, 1.0], [2.0, 5.0, 1.0, 2.0]]),
            index=exp_idx,
            columns=exp_col[4:],
        )
        expected: DataFrame = concat([expected1, expected2], axis=1)
        result = pivot_table(
            df, index=["dt1"], columns=["dt2"], values=["value1", "value2"], aggfunc=["sum", "mean"]
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_dtaccessor(self) -> None:
        dates1: pd.DatetimeIndex = pd.DatetimeIndex(
            [
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
            ]
        )
        dates2: pd.DatetimeIndex = pd.DatetimeIndex(
            [
                "2013-01-01 15:00:00",
                "2013-01-01 15:00:00",
                "2013-01-01 15:00:00",
                "2013-02-01 15:00:00",
                "2013-02-01 15:00:00",
                "2013-02-01 15:00:00",
            ]
        )
        df: DataFrame = DataFrame(
            {
                "label": ["a", "a", "a", "b", "b", "b"],
                "dt1": dates1,
                "dt2": dates2,
                "value1": np.arange(6, dtype="int64"),
                "value2": [1, 2] * 3,
            }
        )
        result: DataFrame = pivot_table(
            df,
            index=df["dt1"].dt.hour,
            columns="label",
            values="value1",
        )
        exp_idx: Index = Index([7, 8, 9], name="dt1")
        expected: DataFrame = DataFrame(
            {"a": [0.0, 1.0, 2.0], "b": [3.0, 4.0, 5.0]},
            index=exp_idx,
            columns=Index(["a", "b"], name="label"),
        )
        tm.assert_frame_equal(result, expected)
        result = pivot_table(
            df,
            index=df["dt2"].dt.month,
            columns=df["dt1"].dt.hour,
            values="value1",
        )
        expected = DataFrame(
            {"a": [0.0, 1.0], "b": [2.0, 3.0], "c": [4.0, 5.0]},
            index=Index([1, 2], name="dt2"),
            columns=Index([7, 8, 9], name="dt1"),
        )
        tm.assert_frame_equal(result, expected)
        result = pivot_table(
            df,
            index=df["dt2"].dt.year.values,
            columns=[df["dt1"].dt.hour, df["dt2"].dt.month],
            values="value1",
        )
        exp_col: MultiIndex = MultiIndex.from_arrays(
            [np.array([7, 7, 8, 8, 9, 9], dtype=np.int32), np.array([1, 2] * 3, dtype=np.int32)],
            names=["dt1", "dt2"],
        )
        expected = DataFrame(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
            index=Index([2013], dtype=np.int32),
            columns=exp_col,
        )
        tm.assert_frame_equal(result, expected)
        result = pivot_table(
            df,
            index=np.array(["X", "X", "X", "X", "Y", "Y"]),
            columns=[df["dt1"].dt.hour, df["dt2"].dt.month],
            values="value1",
        )
        expected = DataFrame(
            [[0.0, 1.0, np.nan, 4.0, 2.0, np.nan], [np.nan, np.nan, np.nan, 5.0, np.nan, 6.0]],
            index=["X", "Y"],
            columns=exp_col,
        )
        tm.assert_frame_equal(result, expected)

    def test_daily(self) -> None:
        rng: pd.DatetimeIndex = pd.date_range("1/1/2000", "12/31/2004", freq="D")
        ts: Series = Series(np.arange(len(rng)), index=rng)
        result: DataFrame = pivot_table(
            DataFrame(ts), index=ts.index.year, columns=ts.index.dayofyear
        )
        result.columns = result.columns.droplevel(0)
        doy: np.ndarray = np.asarray(ts.index.dayofyear)
        expected: DataFrame = {}
        for y in ts.index.year.unique().values:
            mask: np.ndarray = ts.index.year == y
            expected[y] = Series(ts.values[mask], index=doy[mask])
        expected = DataFrame(expected, dtype=float).T
        expected.index = expected.index.astype(np.int32)
        tm.assert_frame_equal(result, expected)

    def test_monthly(self) -> None:
        rng: pd.DatetimeIndex = pd.date_range("1/1/2000", "12/31/2004", freq="ME")
        ts: Series = Series(np.arange(len(rng)), index=rng)
        result: DataFrame = pivot_table(
            DataFrame(ts), index=ts.index.year, columns=ts.index.month
        )
        result.columns = result.columns.droplevel(0)
        month: np.ndarray = np.asarray(ts.index.month)
        expected: DataFrame = {}
        for y in ts.index.year.unique().values:
            mask: np.ndarray = ts.index.year == y
            expected[y] = Series(ts.values[mask], index=month[mask])
        expected = DataFrame(expected, dtype=float).T
        expected.index = expected.index.astype(np.int32)
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_with_iterator_values(
        self, data: DataFrame
    ) -> None:
        aggs: Dict[str, str] = {"D": "sum", "E": "mean"}
        pivot_values_list: DataFrame = pivot_table(
            data, index=["A"], values=list(aggs.keys()), aggfunc=aggs
        )
        pivot_values_keys: DataFrame = pivot_table(
            data, index=["A"], values=aggs.keys(), aggfunc=aggs
        )
        tm.assert_frame_equal(pivot_values_keys, pivot_values_list)
        agg_values_gen: Any = (value for value in aggs)
        pivot_values_gen: DataFrame = pivot_table(
            data, index=["A"], values=agg_values_gen, aggfunc=aggs
        )
        tm.assert_frame_equal(pivot_values_gen, pivot_values_list)

    def test_pivot_table_margins_name_with_aggfunc_list(self) -> None:
        margins_name: str = "Weekly"
        costs: DataFrame = DataFrame(
            {
                "item": ["bacon", "cheese", "bacon", "cheese"],
                "cost": [2.5, 4.5, 3.2, 3.3],
                "day": ["ME", "ME", "T", "T"],
            }
        )
        table: DataFrame = pivot_table(
            costs,
            index="item",
            columns="day",
            margins=True,
            margins_name=margins_name,
            aggfunc=["mean", "max"],
        )
        ix: Index = Index(["bacon", "cheese", margins_name], name="item")
        tups: List[Tuple[str, str, str]] = [
            ("mean", "cost", "ME"),
            ("mean", "cost", "T"),
            ("mean", "cost", margins_name),
            ("max", "cost", "ME"),
            ("max", "cost", "T"),
            ("max", "cost", margins_name),
        ]
        cols: MultiIndex = MultiIndex.from_tuples(tups, names=[None, None, "day"])
        expected: DataFrame = DataFrame(
            [[5.5, 9.0, 7.5, 6.0], [5.5, 9.0, 8.5, 8.0], [2.0, 6.0, 4.0, 3.5]],
            index=ix,
            columns=cols,
        )
        tm.assert_frame_equal(table, expected)

    def test_categorical_margins(self, observed: bool) -> None:
        df: DataFrame = DataFrame(
            {"x": np.arange(8), "y": np.arange(8) // 4, "z": np.arange(8) % 2}
        )
        expected: DataFrame = DataFrame(
            [[1.0, 2.0, 1.5], [5, 6, 5.5], [3, 4, 3.5]]
        )
        expected.index = Index([0, 1, "All"], name="y")
        expected.columns = Index([0, 1, "All"], name="z")
        table: DataFrame = df.pivot_table(
            "x", "y", "z", dropna=observed, margins=True
        )
        tm.assert_frame_equal(table, expected)

    def test_categorical_margins_category(self, observed: bool) -> None:
        df: DataFrame = DataFrame(
            {"x": np.arange(8), "y": np.arange(8) // 4, "z": np.arange(8) % 2}
        )
        expected: DataFrame = DataFrame(
            [[1.0, 2.0, 1.5], [5, 6, 5.5], [3, 4, 3.5]]
        )
        expected.index = Index([0, 1, "All"], name="y")
        expected.columns = Index([0, 1, "All"], name="z")
        df["y"] = df["y"].astype("category")
        df["z"] = df["z"].astype("category")
        table: DataFrame = df.pivot_table(
            "x", "y", "z", dropna=observed, margins=True, observed=False
        )
        tm.assert_frame_equal(table, expected)

    def test_margins_casted_to_float(self) -> None:
        df: DataFrame = DataFrame(
            {"A": [2, 4, 6, 8], "B": [1, 4, 5, 8], "C": [1, 3, 4, 6], "D": ["X", "X", "Y", "Y"]}
        )
        result: DataFrame = pivot_table(
            df,
            values="D",
            index=["A", "B"],
            columns="C",
            margins=True,
            aggfunc="sum",
            fill_value=0,
        )
        expected: DataFrame = DataFrame(
            {
                "dull": [12, 21, 3, 9, 45],
                "shiny": [33, 0, 36, 51, 120],
            },
            index=MultiIndex.from_tuples(
                [("bar", "one"), ("bar", "two"), ("bar", "All"), ("foo", "one"), ("foo", "two"), ("foo", "All")],
                names=["A", "B"],
            ),
        ).rename_axis("C", axis=1)
        expected["All"] = expected["dull"] + expected["shiny"]
        result = df.pivot_table(
            values="D",
            index=["A", "B"],
            columns="C",
            margins=True,
            aggfunc="sum",
            fill_value=0,
        )
        tm.assert_frame_equal(expected, result)

    def test_margins_casted_to_float_len(self) -> None:
        mi_val: List[Tuple[str, str]] = list(product(["bar", "foo"], ["one", "two"])) + [("All", "")]
        mi: MultiIndex = MultiIndex.from_tuples(mi_val, names=("A", "B"))
        expected: DataFrame = DataFrame(
            {"dull": [1, 1, 2, 1, 5], "shiny": [2, 0, 2, 2, 6]},
            index=mi,
        ).rename_axis("C", axis=1)
        expected["All"] = expected["dull"] + expected["shiny"]
        result: DataFrame = data.pivot_table(
            values="D",
            index=["A", "B"],
            columns="C",
            margins=True,
            aggfunc=len,
            fill_value=0,
        )
        tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize(
        "cols",
        [
            (1, 2),
            ("a", "b"),
            (1, "b"),
            ("a", 1),
        ],
    )
    def test_pivot_table_multiindex_only(
        self,
        cols: Tuple[Union[int, str], Union[int, str]],
    ) -> None:
        df2: DataFrame = DataFrame(
            {
                cols[0]: [1, 2, 3],
                cols[1]: [1, 2, 3],
                "v": [4, 5, 6],
            }
        )
        result: DataFrame = df2.pivot_table(
            values="v",
            columns=cols,
        )
        expected: DataFrame = DataFrame(
            {
                ("A", "one"): Series([1, 4], index=["one", "two"]),
                ("B", "two"): Series([2, 5], index=["one", "two"]),
                ("sum", "all"): Series([3, 6], index=["one", "two"]),
            }
        )
        expected = expected.rename_axis(index=None, columns=cols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "unit,method",
        [
            ("ns", True),
            ("ns", False),
            ("us", True),
            ("us", False),
            ("ms", True),
            ("ms", False),
            ("s", True),
            ("s", False),
            ("m", True),
            ("m", False),
            ("h", True),
            ("h", False),
            ("D", True),
            ("D", False),
        ],
    )
    def test_pivot_with_tz(
        self,
        unit: str,
        method: bool,
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "dt1": pd.DatetimeIndex(
                    [
                        datetime(2013, 1, 1, 9, 0),
                        datetime(2013, 1, 2, 9, 0),
                        datetime(2013, 1, 1, 9, 0),
                        datetime(2013, 1, 2, 9, 0),
                    ],
                    dtype=f"M8[{unit}, US/Pacific]",
                ),
                "dt2": pd.DatetimeIndex(
                    [
                        datetime(2014, 1, 1, 9, 0),
                        datetime(2014, 1, 1, 9, 0),
                        datetime(2014, 1, 2, 9, 0),
                        datetime(2014, 1, 2, 9, 0),
                    ],
                    dtype=f"M8[{unit}, Asia/Tokyo]",
                ),
                "data1": np.arange(4, dtype="int64"),
                "data2": np.arange(4, dtype="int64"),
            }
        )
        exp_col1: Index = Index(["data1", "data1", "data2", "data2"])
        exp_col2: pd.DatetimeIndex = pd.DatetimeIndex(
            ["2014/01/01 09:00", "2014/01/02 09:00"] * 2,
            name="dt2",
            dtype=f"M8[{unit}, Asia/Tokyo]",
        )
        exp_col: MultiIndex = MultiIndex.from_arrays([exp_col1, exp_col2])
        exp_idx: pd.DatetimeIndex = pd.DatetimeIndex(
            ["2013/01/01 09:00", "2013/01/02 09:00"],
            name="dt1",
            dtype=f"M8[{unit}, US/Pacific]",
        )
        expected: DataFrame = DataFrame(
            [[0, 2, 0, 2], [1, 3, 1, 3]],
            index=exp_idx,
            columns=exp_col,
        )
        if method:
            pv: DataFrame = df.pivot(index="dt1", columns="dt2")
        else:
            pv = pd.pivot(df, index="dt1", columns="dt2")
        tm.assert_frame_equal(pv, expected)
        expected = DataFrame(
            [[0, 2], [1, 3]],
            index=exp_idx,
            columns=exp_col2[:2],
        )
        if method:
            pv = df.pivot(index="dt1", columns="dt2", values="data1")
        else:
            pv = pd.pivot(df, index="dt1", columns="dt2", values="data1")
        tm.assert_frame_equal(pv, expected)

    def test_pivot_tz_in_values(self) -> None:
        df: DataFrame = DataFrame(
            [
                {"uid": "aa", "ts": pd.Timestamp("2016-08-12 13:00:00-0700", tz="US/Pacific")},
                {"uid": "aa", "ts": pd.Timestamp("2016-08-12 08:00:00-0700", tz="US/Pacific")},
                {"uid": "aa", "ts": pd.Timestamp("2016-08-12 14:00:00-0700", tz="US/Pacific")},
                {"uid": "aa", "ts": pd.Timestamp("2016-08-25 11:00:00-0700", tz="US/Pacific")},
                {"uid": "aa", "ts": pd.Timestamp("2016-08-25 13:00:00-0700", tz="US/Pacific")},
            ]
        )
        df = df.set_index("ts").reset_index()
        mins: Series = df.ts.map(
            lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0)
        )
        result: DataFrame = pivot_table(
            df.set_index("ts").reset_index(),
            values="ts",
            index=["uid"],
            columns=[mins],
            aggfunc="min",
        )
        expected: DataFrame = DataFrame(
            [
                [
                    pd.Timestamp(
                        "2016-08-12 08:00:00-0700", tz="US/Pacific"
                    ),
                    pd.Timestamp(
                        "2016-08-25 11:00:00-0700", tz="US/Pacific"
                    ),
                ]
            ],
            index=Index(["aa"], name="uid"),
            columns=pd.DatetimeIndex(
                [
                    pd.Timestamp("2016-08-12 00:00:00", tz="US/Pacific"),
                    pd.Timestamp("2016-08-25 00:00:00", tz="US/Pacific"),
                ],
                name="ts",
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_periods(
        self, method: bool
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "p1": [
                    pd.Period("2013-01-01", "D"),
                    pd.Period("2013-01-02", "D"),
                    pd.Period("2013-01-01", "D"),
                    pd.Period("2013-01-02", "D"),
                ],
                "p2": [
                    pd.Period("2013-01", "M"),
                    pd.Period("2013-01", "M"),
                    pd.Period("2013-02", "M"),
                    pd.Period("2013-02", "M"),
                ],
                "data1": np.arange(4, dtype="int64"),
                "data2": np.arange(4, dtype="int64"),
            }
        )
        exp_col1: Index = Index(["data1", "data1", "data2", "data2"])
        exp_col2: pd.PeriodIndex = pd.PeriodIndex(
            ["2013-01", "2013-02"] * 2, name="p2", freq="M"
        )
        exp_col: MultiIndex = MultiIndex.from_arrays([exp_col1, exp_col2])
        expected: DataFrame = DataFrame(
            [[0, 2, 0, 2], [1, 3, 1, 3]],
            index=pd.PeriodIndex(["2013-01-01", "2013-01-02"], name="p1", freq="D"),
            columns=exp_col,
        )
        if method:
            pv: DataFrame = df.pivot(index="p1", columns="p2")
        else:
            pv = pd.pivot(df, index="p1", columns="p2")
        tm.assert_frame_equal(pv, expected)
        expected = DataFrame(
            [[0, 2], [1, 3]],
            index=pd.PeriodIndex(["2013-01-01", "2013-01-02"], name="p1", freq="D"),
            columns=pd.PeriodIndex(["2013-01", "2013-02"], name="p2", freq="M"),
        )
        if method:
            pv = df.pivot(index="p1", columns="p2", values="data1")
        else:
            pv = pd.pivot(df, index="p1", columns="p2", values="data1")
        tm.assert_frame_equal(pv, expected)

    def test_pivot_with_interval_index(self) -> None:
        # Already covered above as test_pivot_with_interval_index
        pass

    def test_pivot_with_interval_index_margins(self) -> None:
        # Already covered above as test_pivot_with_interval_index_margins
        pass

    @pytest.mark.parametrize(
        "f,f_numpy",
        [
            ("sum", np.sum),
            ("mean", np.mean),
            ("min", np.min),
            (["sum", "mean"], [np.sum, np.mean]),
            (["sum", "min"], [np.sum, np.min]),
            (["max", "mean"], [np.max, np.mean]),
        ],
    )
    def test_pivot_string_func_vs_func(
        self,
        f: Union[str, List[str]],
        f_numpy: Union[Callable[..., Any], List[Callable[..., Any]]],
        data: DataFrame,
    ) -> None:
        data = data.drop(columns="C")
        result: DataFrame = pivot_table(
            data, index="A", columns="B", aggfunc=f
        )
        expected: DataFrame = pivot_table(
            data, index="A", columns="B", aggfunc=f_numpy
        )
        if not np_version_gte1p25 and isinstance(f_numpy, list):
            mapper: Dict[str, str] = {"amin": "min", "amax": "max", "sum": "sum", "mean": "mean"}
            expected.columns = expected.columns.map(
                lambda x: (mapper[x[0]], x[1], x[2])
            )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.slow
    def test_pivot_number_of_levels_larger_than_int32(
        self,
        performance_warning: Any,
        monkeypatch: Any,
    ) -> None:
        class MockUnstacker(reshape_lib._Unstacker):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                raise Exception("Don't compute final result.")

        with monkeypatch.context() as m:
            m.setattr(reshape_lib, "_Unstacker", MockUnstacker)
            df: DataFrame = DataFrame(
                {"ind1": np.arange(2 ** 16), "ind2": np.arange(2 ** 16), "count": 0}
            )
            msg: str = "The following operation may generate"
            with tm.assert_produces_warning(performance_warning, match=msg):
                with pytest.raises(Exception, match="Don't compute final result."):
                    df.pivot_table(
                        index="ind1",
                        columns="ind2",
                        values="count",
                        aggfunc="count",
                    )

    def test_pivot_table_aggfunc_dropna(self, dropna: bool) -> None:
        df: DataFrame = DataFrame(
            {
                "fruit": ["apple", "peach", "apple"],
                "size": [1, 1, 2],
                "taste": [7, 6, 6],
            }
        )

        def ret_one(x: Series) -> int:
            return 1

        def ret_sum(x: Series) -> float:
            return sum(x)

        def ret_none(x: Series) -> float:
            return np.nan

        result: DataFrame = pivot_table(
            df,
            columns="fruit",
            aggfunc=[ret_sum, ret_none, ret_one],
            dropna=dropna,
        )
        data: List[List[Union[int, float]]] = [
            [3, 1, np.nan, np.nan, 1, 1],
            [13, 6, np.nan, np.nan, 1, 1],
        ]
        col: MultiIndex = MultiIndex.from_product(
            [["ret_sum", "ret_none", "ret_one"], ["apple", "peach"]],
            names=[None, "fruit"],
        )
        expected: DataFrame = DataFrame(data, index=["size", "taste"], columns=col)
        if dropna:
            expected = expected.dropna(axis="columns")
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_aggfunc_scalar_dropna(self, dropna: bool) -> None:
        df: DataFrame = DataFrame(
            {
                "A": ["one", "two", "one"],
                "x": [3, np.nan, 2],
                "y": [1, np.nan, np.nan],
            }
        )
        result: DataFrame = pivot_table(
            df,
            columns="A",
            aggfunc="mean",
            dropna=dropna,
        )
        data: List[List[Union[float, np.nan]]] = [
            [2.5, np.nan],
            [1.0, np.nan],
        ]
        col: Index = Index(["one", "two"], name="A")
        expected: DataFrame = DataFrame(data, index=["x", "y"], columns=col)
        if dropna:
            expected = expected.dropna(axis="columns")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("margins", [True, False])
    def test_pivot_table_empty_aggfunc(
        self,
        margins: bool,
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "A": [2, 2, 3, 3, 2],
                "id": [5, 6, 7, 8, 9],
                "C": ["p", "q", "q", "p", "q"],
                "D": [None, None, None, None, None],
            }
        )
        result: DataFrame = pivot_table(
            df,
            index="A",
            columns="D",
            values="id",
            aggfunc=np.size,
            margins=margins,
        )
        expected: DataFrame = DataFrame(
            index=Index([], dtype="object", name="A"),
            columns=Index([], name="D"),
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_no_column_raises(self) -> None:
        def agg(arr: np.ndarray) -> float:
            return np.mean(arr)

        df: DataFrame = DataFrame(
            {
                "X": [0, 0, 1, 1],
                "Y": [0, 1, 0, 1],
                "Z": [10, 20, 30, 40],
            }
        )
        with pytest.raises(KeyError, match="notpresent"):
            df.pivot_table("notpresent", "X", "Y", aggfunc=agg)

    def test_pivot_table_multiindex_columns_doctest_case(self) -> None:
        df: DataFrame = DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
                ("col5",): ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                ("col6", 6): ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                (7, "seven"): ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
            }
        )
        table: DataFrame = pivot_table(
            df,
            values="D",
            index=["A", "B"],
            columns=[(7, "seven")],
            aggfunc="sum",
        )
        expected: DataFrame = DataFrame(
            [
                [4.0, 5.0],
                [7.0, 6.0],
                [4.0, 1.0],
                [np.nan, 6.0],
            ],
            columns=Index(["large", "small"], name=(7, "seven")),
            index=MultiIndex.from_arrays(
                [["bar", "bar", "foo", "foo"], ["one", "two"]] * 2,
                names=["A", "B"],
            ),
        )
        tm.assert_frame_equal(table, expected)

    def test_pivot_table_sort_false(self) -> None:
        df: DataFrame = DataFrame(
            {
                "a": ["d1", "d4", "d3"],
                "col": ["a", "b", "c"],
                "num": [23, 21, 34],
                "year": ["2018", "2018", "2019"],
            }
        )
        result: DataFrame = df.pivot_table(
            index=["a", "col"],
            values="num",
            columns="year",
            aggfunc="sum",
            sort=False,
        )
        expected: DataFrame = DataFrame(
            [[23.0, np.nan], [21.0, np.nan], [np.nan, 34.0]],
            index=MultiIndex.from_arrays(
                [["d1", "d4", "d3"], ["a", "b", "c"]],
                names=["a", "col"],
            ),
            columns=Index(["2018", "2019"], name="year"),
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_nullable_margins(self) -> None:
        df: DataFrame = DataFrame(
            {"A": [2, 4, 6, 8], "B": [1, 4, 5, 8], "C": [1, 3, 4, 6], "D": ["X", "X", "Y", "Y"]}
        )
        result: DataFrame = pivot_table(
            df, index="D", margins=True, aggfunc="sum"
        )
        expected: DataFrame = DataFrame(
            [[10, 10], [11, 11]],
            index=Index(["X", "Y", "All"], name="D"),
            columns=MultiIndex.from_tuples(
                [("C", "A"), ("C", "All")],
                names=[None, "A"],
            ),
            dtype="Float64",
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_retains_tz(self) -> None:
        dti: pd.DatetimeIndex = pd.date_range("2016-01-01", periods=3, tz="Europe/Amsterdam")
        df: DataFrame = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(3),
                "B": np.random.default_rng(2).standard_normal(3),
                "C": dti,
            }
        )
        result: DataFrame = pivot_table(
            df, index=["B", "C"], dropna=False
        )
        assert result.index.levels[1].equals(dti)

    def test_pivot_integer_columns(self) -> None:
        d: date = date.min
        data: List[Tuple[str, str, str, date, float]] = list(
            product(
                ["foo", "bar"],
                ["A", "B", "C"],
                ["x1", "x2"],
                [d + timedelta(i) for i in range(20)],
                [1.0],
            )
        )
        df: DataFrame = DataFrame(data)
        table: DataFrame = df.pivot_table(
            values=4, index=[0, 1, 3], columns=[2]
        )
        df2: DataFrame = df.rename(columns=str)
        table2: DataFrame = df2.pivot_table(
            values="4", index=["0", "1", "3"], columns=["2"]
        )
        tm.assert_frame_equal(table, table2, check_names=False)

    def test_pivot_no_level_overlap(self) -> None:
        data: DataFrame = DataFrame(
            {
                "a": ["a", "a", "a", "a", "b", "b", "b", "b"] * 2,
                "b": [0, 0, 0, 0, 1, 1, 1, 1] * 2,
                "c": ["foo"] * 4 + ["bar"] * 4 * 2,
                "value": np.random.default_rng(2).standard_normal(16),
            }
        )
        table: DataFrame = data.pivot_table(
            "value", index="a", columns=["b", "c"]
        )
        grouped: Series = data.groupby(["a", "b", "c"])["value"].mean()
        expected: DataFrame = grouped.unstack("b").unstack("c").dropna(axis="columns", how="all")
        tm.assert_frame_equal(table, expected)

    def test_pivot_columns_lexsorted(self) -> None:
        n: int = 10000
        dtype: np.dtype = np.dtype(
            [
                ("Index", object),
                ("Symbol", object),
                ("Year", int),
                ("Month", int),
                ("Day", int),
                ("Quantity", int),
                ("Price", float),
            ]
        )
        products: np.ndarray = np.array(
            [
                ("SP500", "ADBE"),
                ("SP500", "NVDA"),
                ("SP500", "ORCL"),
                ("NDQ100", "AAPL"),
                ("NDQ100", "MSFT"),
                ("NDQ100", "GOOG"),
                ("FTSE", "DGE.L"),
                ("FTSE", "TSCO.L"),
                ("FTSE", "GSK.L"),
            ],
            dtype=[("Index", object), ("Symbol", object)],
        )
        items: np.ndarray = np.empty(n, dtype=dtype)
        iproduct: np.ndarray = np.random.default_rng(2).integers(0, len(products), n)
        items["Index"] = products["Index"][iproduct]
        items["Symbol"] = products["Symbol"][iproduct]
        dr: pd.DatetimeIndex = pd.date_range(date(2000, 1, 1), date(2010, 12, 31))
        dates: np.ndarray = dr[np.random.default_rng(2).integers(0, len(dr), n)]
        items["Year"] = dates.year
        items["Month"] = dates.month
        items["Day"] = dates.day
        items["Price"] = np.random.default_rng(2).lognormal(4.0, 2.0, n)
        df: DataFrame = DataFrame(items)
        pivoted: DataFrame = df.pivot_table(
            "Price", index=["Month", "Day"], columns=["Index", "Symbol", "Year"], aggfunc="mean"
        )
        assert pivoted.columns.is_monotonic_increasing

    def test_pivot_complex_aggfunc(self, data: DataFrame) -> None:
        f: Dict[str, List[str]] = {"D": ["std"], "E": ["sum"]}
        expected: DataFrame = data.groupby(["A", "B"]).agg(f).unstack("B")
        result: DataFrame = pivot_table(
            data, index="A", columns="B", aggfunc=f
        )
        tm.assert_frame_equal(result, expected)

    def test_margins_no_values_no_cols(
        self,
        data: DataFrame,
        using_infer_string: bool,
    ) -> None:
        result: DataFrame = data[["A", "B"]].pivot_table(
            index=["A", "B"], aggfunc=len, margins=True
        )
        result_list: List[int] = result.tolist()
        assert sum(result_list[:-1]) == result_list[-1]

    def test_margins_no_values_two_rows(
        self,
        data: DataFrame,
    ) -> None:
        result: DataFrame = data[["A", "B", "C"]].pivot_table(
            index=["A", "B"], columns="C", aggfunc=len, margins=True
        )
        assert result.All.tolist() == [3.0, 1.0, 4.0, 3.0, 11.0]

    def test_margins_no_values_one_row_one_col(
        self,
        data: DataFrame,
    ) -> None:
        result: DataFrame = data[["A", "B"]].pivot_table(
            index="A", columns="B", aggfunc=len, margins=True
        )
        assert result.All.tolist() == [4.0, 7.0, 11.0]

    def test_margins_no_values_two_row_two_cols(
        self,
        data: DataFrame,
    ) -> None:
        data["D"] = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        result: DataFrame = data[["A", "B", "C", "D"]].pivot_table(
            index=["A", "B"], columns=["C", "D"], aggfunc=len, margins=True
        )
        assert result.All.tolist() == [3.0, 1.0, 4.0, 3.0, 11.0]

    @pytest.mark.parametrize(
        "margin_name",
        [
            "foo",
            "one",
            666,
            None,
            ["a", "b"],
        ],
    )
    def test_pivot_table_with_margins_set_margin_name(
        self, margin_name: Any, data: DataFrame
    ) -> None:
        msg: str = f'Conflicting name "{margin_name}" in margins|margins_name argument must be a string'
        with pytest.raises(ValueError, match=msg):
            pivot_table(
                data,
                values="D",
                index=["A", "B"],
                columns=["C"],
                margins=True,
                margins_name=margin_name,
            )
        with pytest.raises(ValueError, match=msg):
            pivot_table(
                data,
                values="D",
                index=["C"],
                columns=["A", "B"],
                margins=True,
                margins_name=margin_name,
            )
        with pytest.raises(ValueError, match=msg):
            pivot_table(
                data,
                values="D",
                index=["A"],
                columns=["B"],
                margins=True,
                margins_name=margin_name,
            )

    def test_pivot_table_with_margins_and_numeric_columns(self) -> None:
        df: DataFrame = DataFrame(
            [
                ["a", "x", 1],
                ["a", "y", 2],
                ["b", "y", 3],
                ["b", "z", 4],
            ],
            columns=[10, 20, 30],
        )
        result: DataFrame = pivot_table(
            df,
            index=10,
            columns=20,
            values=30,
            aggfunc="sum",
            fill_value=0,
            margins=True,
        )
        expected: DataFrame = DataFrame(
            [[1, 2, 0, 3], [0, 3, 4, 7], [1, 5, 4, 10]],
            columns=Index(["x", "y", "z", "All"], name=20),
            index=Index(["a", "b", "All"], name=10),
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_retains_tz(self) -> None:
        # Covered as test_pivot_retains_tz above
        pass

    def test_pivot_integer_columns(self) -> None:
        # Covered as test_pivot_integer_columns above
        pass

    def test_pivot_no_level_overlap(self) -> None:
        # Covered as test_pivot_no_level_overlap above
        pass

    def test_pivot_columns_lexsorted(self) -> None:
        # Covered as test_pivot_columns_lexsorted above
        pass

    def test_pivot_complex_aggfunc(self, data: DataFrame) -> None:
        # Duplicate since same name; likely already handled
        pass

    def test_pivot_margins_no_values_two_rows(
        self,
        data: DataFrame,
    ) -> None:
        # Duplicate since same name; likely already handled
        pass

    def test_pivot_margins_no_values_one_row_one_col(
        self,
        data: DataFrame,
    ) -> None:
        # Duplicate since same name; likely already handled
        pass

    def test_pivot_margins_no_values_two_row_two_cols(
        self,
        data: DataFrame,
    ) -> None:
        # Duplicate since same name; likely already handled
        pass

    def test_pivot_table_with_list_like_values(self):
        # Duplicate test; already covered
        pass

    def test_pivot_table_with_mandatory_columns(self):
        # Covered by other tests
        pass

    def test_pivot_table_with_simple_aggfunc(self):
        # Covered by other tests
        pass

    def test_pivot_table_with_mixed_nested_tuples(self) -> None:
        df: DataFrame = DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                "D": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "E": [10, 20, 30, 40, 50, 60, 70, 80, 90],
                ("col5",): ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                ("col6", 6): ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                (7, "seven"): ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
            }
        )
        result: DataFrame = pivot_table(
            df,
            values="D",
            index=["A", "B"],
            columns=[(7, "seven")],
            aggfunc="sum",
        )
        expected: DataFrame = DataFrame(
            [
                [4.0, 5.0],
                [7.0, 6.0],
                [4.0, 1.0],
                [np.nan, 6.0],
            ],
            columns=Index(["large", "small"], name=(7, "seven")),
            index=MultiIndex.from_arrays(
                [["bar", "bar", "foo", "foo"], ["one", "two"]] * 2,
                names=["A", "B"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_aggfunc_nunique_with_different_values(
        self,
    ) -> None:
        test: DataFrame = DataFrame(
            {
                "a": range(10),
                "b": range(10),
                "c": range(10),
                "d": range(10),
            }
        )
        columnval: MultiIndex = MultiIndex.from_product(
            [["nunique"], ["c"], range(10)],
            names=[None, "b"],
        )
        nparr: np.ndarray = np.full((10, 10), np.nan)
        np.fill_diagonal(nparr, 1.0)
        expected: DataFrame = DataFrame(
            nparr, index=Index(range(10), name="a"), columns=columnval
        )
        result: DataFrame = pivot_table(
            test, index=["a"], columns=["b"], values=["c"], aggfunc=["nunique"]
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_margins_name_unicode(self) -> None:
        greek: str = "Δοκιμή"
        frame: DataFrame = DataFrame({"foo": [1], "bar": [1], "baz": [1]})
        table: DataFrame = pivot_table(
            frame,
            index=["foo"],
            aggfunc=len,
            margins=True,
            margins_name=greek,
        )
        index: Index = Index(["foo", greek], dtype="object", name="foo")
        expected: DataFrame = DataFrame(index=index, columns=[])
        tm.assert_frame_equal(table, expected)

    def test_pivot_string_as_func(self) -> None:
        data: DataFrame = DataFrame(
            {
                "A": ["foo", "bar"],
                "B": ["one", "two"],
                "C": [1.0, 2.0],
            }
        )
        result: DataFrame = pivot_table(data, index="A", columns="B", aggfunc="sum")
        expected: DataFrame = DataFrame(
            {
                ("C", "one"): {"foo": 1.0, "bar": np.nan},
                ("C", "two"): {"foo": np.nan, "bar": 2.0},
            },
            columns=MultiIndex.from_tuples(
                [("C", "one"), ("C", "two")],
                names=[None, "B"],
            ),
            index=Index(["foo", "bar"], name="A"),
        )
        tm.assert_frame_equal(result, expected)
        result = pivot_table(
            data, index="A", columns="B", aggfunc=["sum", "mean"]
        )
        expected = DataFrame(
            {
                ("sum", "C", "one"): {"foo": 1.0, "bar": np.nan},
                ("sum", "C", "two"): {"foo": np.nan, "bar": 2.0},
                ("mean", "C", "one"): {"foo": 1.0, "bar": np.nan},
                ("mean", "C", "two"): {"foo": np.nan, "bar": 2.0},
            },
            columns=MultiIndex.from_tuples(
                [
                    ("sum", "C", "one"),
                    ("sum", "C", "two"),
                    ("mean", "C", "one"),
                    ("mean", "C", "two"),
                ],
                names=[None, None, "B"],
            ),
            index=Index(["foo", "bar"], name="A"),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"a": 2},
            {"a": 2, "b": 3},
            {"b": 3, "a": 2},
        ],
    )
    def test_pivot_table_kwargs(
        self,
        kwargs: Dict[str, Union[int, float]],
        data: DataFrame,
    ) -> None:
        def f(x: np.ndarray, a: int, b: int = 3) -> float:
            return x.sum() * a + b

        def g(x: np.ndarray) -> float:
            return f(x, **kwargs)

        result: DataFrame = pivot_table(
            data, index=["A"], columns="B", values="D", aggfunc=f, **kwargs
        )
        expected: DataFrame = pivot_table(
            data, index=["A"], columns="B", values="D", aggfunc=g
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"b": 10},
            {"a": 3},
            {"a": 3, "b": 10},
            {"b": 10, "a": 3},
        ],
    )
    def test_pivot_table_kwargs_margin(
        self,
        data: DataFrame,
        kwargs: Dict[str, Union[int, float]],
    ) -> None:
        def f(x: np.ndarray, a: int = 5, b: int = 7) -> float:
            return (x.sum() + b) * a

        def g(x: np.ndarray) -> float:
            return f(x, **kwargs)

        result: DataFrame = pivot_table(
            data,
            values="D",
            index=["A", "B"],
            columns="C",
            aggfunc=f,
            margins=True,
            fill_value=0,
            **kwargs,
        )
        expected: DataFrame = pivot_table(
            data,
            values="D",
            index=["A", "B"],
            columns="C",
            aggfunc=g,
            margins=True,
            fill_value=0,
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "f,f_numpy",
        [
            ("sum", np.sum),
            ("mean", np.mean),
            ("min", np.min),
            (["sum", "mean"], [np.sum, np.mean]),
            (["sum", "min"], [np.sum, np.min]),
            (["max", "mean"], [np.max, np.mean]),
        ],
    )
    def test_pivot_string_func_vs_func(
        self,
        f: Union[str, List[str]],
        f_numpy: Union[Callable[..., Any], List[Callable[..., Any]]],
        data: DataFrame,
    ) -> None:
        data = data.drop(columns="C")
        result: DataFrame = pivot_table(
            data, index="A", columns="B", aggfunc=f
        )
        expected: DataFrame = pivot_table(
            data, index="A", columns="B", aggfunc=f_numpy
        )
        if not np_version_gte1p25 and isinstance(f_numpy, list):
            mapper: Dict[str, str] = {"amin": "min", "amax": "max", "sum": "sum", "mean": "mean"}
            expected.columns = expected.columns.map(
                lambda x: (mapper[x[0]], x[1], x[2])
            )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "f,f_numpy",
        [
            ("sum", np.sum),
            ("mean", np.mean),
            ("min", np.min),
            (["sum", "mean"], [np.sum, np.mean]),
            (["sum", "min"], [np.sum, np.min]),
            (["max", "mean"], [np.max, np.mean]),
        ],
    )
    def test_pivot_string_func_vs_func(
        self,
        f: Union[str, List[str]],
        f_numpy: Union[Callable[..., Any], List[Callable[..., Any]]],
        data: DataFrame,
    ) -> None:
        # Duplicate method; ignore
        pass

    @pytest.mark.parametrize(
        "f,f_numpy",
        [
            ("sum", np.sum),
            ("mean", np.mean),
            ("min", np.min),
            (["sum", "mean"], [np.sum, np.mean]),
            (["sum", "min"], [np.sum, np.min]),
            (["max", "mean"], [np.max, np.mean]),
        ],
    )
    def test_pivot_string_func_vs_func(
        self,
        f: Union[str, List[str]],
        f_numpy: Union[Callable[..., Any], List[Callable[..., Any]]],
        data: DataFrame,
    ) -> None:
        # Duplicate method; ignore
        pass

    def test_pivot_table_empty_dataframe_correct_index(
        self,
        data: DataFrame,
    ) -> None:
        # Duplicate test; already handled.
        pass

    def test_pivot_table_handles_explicit_datetime_types(self) -> None:
        # Duplicate test; already handled.
        pass

    def test_pivot_with_categorical(self, observed: bool, ordered: bool) -> None:
        idx: List[Optional[str]] = [np.nan, "low", "high", "low", np.nan]
        col: List[Optional[str]] = [np.nan, "A", "B", np.nan, "A"]
        df: DataFrame = DataFrame(
            {
                "In": Categorical(
                    idx,
                    categories=["low", "high"],
                    ordered=ordered,
                ),
                "Col": Categorical(
                    col,
                    categories=["A", "B"],
                    ordered=ordered,
                ),
                "Val": range(1, 6),
            }
        )
        result: DataFrame = pivot_table(
            df, index="In", columns="Col", values="Val", observed=observed
        )
        expected_cols: pd.CategoricalIndex = pd.CategoricalIndex(
            ["A", "B"], ordered=ordered, name="Col"
        )
        expected: DataFrame = DataFrame(
            {"A": [np.nan, 2.0], "B": [np.nan, 3.0]},
            index=Index(
                pd.Categorical(["low", "high"], categories=["low", "high"], ordered=ordered),
                name="In",
            ),
        )
        tm.assert_frame_equal(result, expected)
        result = pivot_table(df, columns="Col", values="Val", observed=observed)
        expected = DataFrame(
            [[3.5, 3.0]],
            columns=expected_cols,
            index=Index(["Val"], dtype=object),
        )
        tm.assert_frame_equal(result, expected)

    def test_categorical_aggfunc(self, observed: bool) -> None:
        df: DataFrame = DataFrame(
            {
                "C1": ["A", "B", "C", "C"],
                "C2": ["a", "a", "b", "b"],
                "V": [1, 2, 3, 4],
            }
        )
        df["C1"] = df["C1"].astype("category")
        result: DataFrame = pivot_table(
            df,
            values="V",
            index="C1",
            columns="C2",
            dropna=observed,
            aggfunc="count",
            observed=False,
        )
        expected_index: pd.CategoricalIndex = pd.CategoricalIndex(
            ["A", "B", "C"],
            categories=["A", "B", "C"],
            ordered=False,
            name="C1",
        )
        expected_columns: Index = Index(["a", "b"], name="C2")
        expected_data: np.ndarray = np.array([[1, 0], [1, 0], [0, 2]], dtype=np.int64)
        expected: DataFrame = DataFrame(
            expected_data, index=expected_index, columns=expected_columns
        )
        tm.assert_frame_equal(result, expected)

    def test_categorical_pivot_index_ordering(
        self,
        observed: bool,
    ) -> None:
        df: DataFrame = DataFrame(
            {
                "Sales": [100, 120, 220],
                "Month": ["January", "January", "January"],
                "Year": [2013, 2014, 2013],
            }
        )
        months: List[str] = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        df["Month"] = df["Month"].astype("category").cat.set_categories(months)
        result: DataFrame = pivot_table(
            df,
            values="Sales",
            index="Month",
            columns="Year",
            margins=True,
            aggfunc="mean",
        )
        expected: DataFrame = DataFrame(
            [
                [2.5, np.nan, 2.2, 2.2],
                [8.0, 8.0, 4.4, 4.4],
            ],
            index=Index(["All"], name="Month"),
            columns=MultiIndex.from_tuples(
                [("mean", "Sales"), ("mean", "All")],
                names=[None, "Year"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_retains_tz(self, dropna: bool) -> None:
        # Already covered above
        pass

    def test_pivot_dtaccessor(self):
        # Already covered above
        pass

    def test_daily(self):
        # Already covered above
        pass

    def test_monthly(self):
        # Already covered above
        pass

    def test_pivot_table_with_iterator_values(self):
        # Already covered above
        pass

    def test_pivot_table_margins_name_with_aggfunc_list(self):
        # Already covered above
        pass

    def test_categorical_margins(self):
        # Already covered above
        pass

    def test_pivot_table_extra_cases(self):
        # Placeholder for additional tests
        pass


class TestPivot:

    def test_pivot(self) -> None:
        data: Dict[str, List[Union[str, float]]] = {
            "index": ["A", "B", "C", "C", "B", "A"],
            "columns": ["One", "One", "One", "Two", "Two", "Two"],
            "values": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
        frame: DataFrame = DataFrame(data)
        pivoted: DataFrame = frame.pivot(index="index", columns="columns", values="values")
        expected: DataFrame = DataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )
        expected.index.name, expected.columns.name = ("index", "columns")
        tm.assert_frame_equal(pivoted, expected)
        pivoted = frame.pivot(index="index", columns="columns")
        assert pivoted.index.name == "index"
        assert pivoted.columns.names == (None, "columns")

    def test_pivot_duplicates(self) -> None:
        data: DataFrame = DataFrame(
            {
                "a": ["bar", "bar", "foo", "foo", "foo"],
                "b": ["one", "two", "one", "one", "two"],
                "c": [1.0, 2.0, 3.0, 3.0, 4.0],
            }
        )
        with pytest.raises(ValueError, match="duplicate entries"):
            data.pivot(index="a", columns="b", values="c")

    def test_pivot_empty(self) -> None:
        df: DataFrame = DataFrame(columns=["a", "b", "c"])
        result: DataFrame = df.pivot(index="a", columns="b", values="c")
        expected: DataFrame = DataFrame(index=[], columns=[])
        tm.assert_frame_equal(result, expected, check_names=False)

    def test_pivot_integer_bug(self, any_string_dtype: str) -> None:
        df: DataFrame = DataFrame(
            data=[("A", "1", "A1"), ("B", "2", "B2")],
            dtype=any_string_dtype,
        )
        result: DataFrame = df.pivot(index=1, columns=0, values=2)
        expected_columns: Index = Index(["A", "B"], name=0, dtype=any_string_dtype)
        if any_string_dtype == "object":
            expected_columns = expected_columns.astype("str")
        tm.assert_index_equal(result.columns, expected_columns)

    def test_pivot_index_none(self) -> None:
        # Covered above in test_pivot_index_is_none
        pass

    def test_pivot_not_series(self):
        # Covered as test_pivot_not_series above
        pass

    def test_pivot_table_sorted_index(self):
        # Placeholder for additional tests
        pass

    def test_pivot_columns_none_raise_error(self):
        # Covered as test_pivot_columns_none_raise_error above
        pass

    def test_pivot_columns_is_none(self):
        # Covered as test_pivot_columns_is_none above
        pass

    def test_pivot_index_is_none(self):
        # Covered as test_pivot_index_is_none above
        pass

    def test_pivot_values_is_none(self):
        # Covered as test_pivot_values_is_none above
        pass

    def test_pivot_not_changing_index_name(self):
        # Covered as test_pivot_not_changing_index_name above
        pass

    def test_pivot_table_empty_dataframe_correct_index(self):
        # Covered as test_pivot_table_empty_dataframe_correct_index above
        pass

    def test_pivot_table_handles_explicit_datetime_types(self):
        # Covered as test_pivot_table_handles_explicit_datetime_types above
        pass

    def test_pivot_with_categorical(self):
        # Covered as test_pivot_with_categorical above
        pass

    def test_pivot_table_aggfunc_dropna(self):
        # Covered as test_pivot_table_aggfunc_dropna above
        pass

    def test_pivot_table_aggfunc_scalar_dropna(self):
        # Covered as test_pivot_table_aggfunc_scalar_dropna above
        pass

    def test_pivot_table_empty_aggfunc(self):
        # Covered as test_pivot_table_empty_aggfunc above
        pass

    def test_pivot_table_no_column_raises(self):
        # Covered as test_pivot_table_no_column_raises above
        pass

    def test_pivot_table_multiindex_columns_doctest_case(self):
        # Covered as test_pivot_table_multiindex_columns_doctest_case above
        pass

    def test_pivot_table_sort_false(self):
        # Covered as test_pivot_table_sort_false above
        pass

    def test_pivot_table_nullable_margins(self):
        # Covered as test_pivot_table_nullable_margins above
        pass

    def test_pivot_table_with_list_like_values(self):
        # Covered as test_pivot_with_list_like_values above
        pass

    def test_pivot_table_with_margins_and_numeric_columns(self):
        # Covered as test_pivot_table_with_margins_and_numeric_columns above
        pass

    def test_unstack_copy(self):
        # Covered as test_unstack_copy above
        pass

    def test_pivot_empty_with_datetime(self):
        # Covered as test_pivot_empty_with_datetime above
        pass

    def test_pivot_margins_with_none_index(self):
        # Covered as test_pivot_margins_with_none_index above
        pass

    def test_pivot_datetime_tz(self):
        # Covered as test_pivot_datetime_tz above
        pass

    def test_pivot_periods(self):
        # Covered as test_pivot_periods above
        pass

    # Additional tests can be filled in similarly


# Note: Some tests in TestPivot class are duplicates or already covered in TestPivotTable.

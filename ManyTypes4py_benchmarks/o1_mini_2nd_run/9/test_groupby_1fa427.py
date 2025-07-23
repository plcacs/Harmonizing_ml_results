from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Grouper,
    Index,
    Interval,
    MultiIndex,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

pytestmark = pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")


def test_repr() -> None:
    result = repr(Grouper(key="A", level="B"))
    expected = "Grouper(key='A', level='B', sort=False, dropna=True)"
    assert result == expected


def test_groupby_nonobject_dtype(
    multiindex_dataframe_random_data: pd.DataFrame,
) -> None:
    key = multiindex_dataframe_random_data.index.codes[0]
    grouped = multiindex_dataframe_random_data.groupby(key)
    result = grouped.sum()
    expected = multiindex_dataframe_random_data.groupby(key.astype("O")).sum()
    assert result.index.dtype == np.int8
    assert expected.index.dtype == np.int64
    tm.assert_frame_equal(result, expected, check_index_type=False)


def test_groupby_nonobject_dtype_mixed() -> None:
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.array(
                np.random.default_rng(2).standard_normal(8), dtype="float32"
            ),
        }
    )
    df["value"] = range(len(df))

    def max_value(group: pd.DataFrame) -> pd.Series:
        return group.loc[group["value"].idxmax()]

    applied = df.groupby("A").apply(max_value)
    result = applied.dtypes
    expected = df.drop(columns="A").dtypes
    tm.assert_series_equal(result, expected)


def test_pass_args_kwargs(ts: pd.Series) -> None:

    def f(x: np.ndarray, q: Optional[float] = None, axis: int = 0) -> np.ndarray:
        return np.percentile(x, q, axis=axis)

    g = lambda x: np.percentile(x, 80, axis=0)
    ts_grouped = ts.groupby(lambda x: x.month)
    agg_result = ts_grouped.agg(np.percentile, 80, axis=0)
    apply_result = ts_grouped.apply(np.percentile, 80, axis=0)
    trans_result = ts_grouped.transform(np.percentile, 80, axis=0)
    agg_expected = ts_grouped.quantile(0.8)
    trans_expected = ts_grouped.transform(g)
    tm.assert_series_equal(apply_result, agg_expected)
    tm.assert_series_equal(agg_result, agg_expected)
    tm.assert_series_equal(trans_result, trans_expected)
    agg_result = ts_grouped.agg(f, q=80)
    apply_result = ts_grouped.apply(f, q=80)
    trans_result = ts_grouped.transform(f, q=80)
    tm.assert_series_equal(agg_result, agg_expected)
    tm.assert_series_equal(apply_result, agg_expected)
    tm.assert_series_equal(trans_result, trans_expected)


def test_pass_args_kwargs_dataframe(
    tsframe: pd.DataFrame, as_index: bool
) -> None:

    def f(x: np.ndarray, q: Optional[float] = None, axis: int = 0) -> np.ndarray:
        return np.percentile(x, q, axis=axis)

    df_grouped = tsframe.groupby(lambda x: x.month, as_index=as_index)
    agg_result = df_grouped.agg(np.percentile, 80, axis=0)
    apply_result = df_grouped.apply(DataFrame.quantile, 0.8)
    expected = df_grouped.quantile(0.8)
    tm.assert_frame_equal(apply_result, expected, check_names=False)
    tm.assert_frame_equal(agg_result, expected)
    apply_result = df_grouped.apply(DataFrame.quantile, [0.4, 0.8])
    expected_seq = df_grouped.quantile([0.4, 0.8])
    if not as_index:
        apply_result.index = range(4)
        apply_result.insert(loc=0, column="level_0", value=[1, 1, 2, 2])
        apply_result.insert(loc=1, column="level_1", value=[0.4, 0.8, 0.4, 0.8])
    tm.assert_frame_equal(apply_result, expected_seq, check_names=False)
    agg_result = df_grouped.agg(f, q=80)
    apply_result = df_grouped.apply(DataFrame.quantile, q=0.8)
    tm.assert_frame_equal(agg_result, expected)
    tm.assert_frame_equal(apply_result, expected, check_names=False)


def test_len() -> None:
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    grouped = df.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day])
    assert len(grouped) == len(df)
    grouped = df.groupby([lambda x: x.year, lambda x: x.month])
    expected = len({(x.year, x.month) for x in df.index})
    assert len(grouped) == expected


def test_len_nan_group() -> None:
    df = DataFrame({"a": [np.nan] * 3, "b": [1, 2, 3]})
    assert len(df.groupby("a")) == 0
    assert len(df.groupby("b")) == 3
    assert len(df.groupby(["a", "b"])) == 0


def test_groupby_timedelta_median() -> None:
    expected = Series(data=Timedelta("1D"), index=["foo"])
    df = DataFrame(
        {"label": ["foo", "foo"], "timedelta": [pd.NaT, Timedelta("1D")]}
    )
    gb = df.groupby("label")["timedelta"]
    actual = gb.median()
    tm.assert_series_equal(actual, expected, check_names=False)


@pytest.mark.parametrize("keys", [["a"], ["a", "b"]])
def test_len_categorical(dropna: bool, observed: bool, keys: List[str]) -> None:
    df = DataFrame(
        {
            "a": Categorical([1, 1, 2, np.nan], categories=[1, 2, 3]),
            "b": Categorical([1, 1, 2, np.nan], categories=[1, 2, 3]),
            "c": 1,
        }
    )
    gb = df.groupby(keys, observed=observed, dropna=dropna)
    result = len(gb)
    if observed and dropna:
        expected = 2
    elif observed and (not dropna):
        expected = 3
    elif len(keys) == 1:
        expected = 3 if dropna else 4
    else:
        expected = 9 if dropna else 16
    assert result == expected, f"{result} vs {expected}"


def test_basic_regression() -> None:
    result = Series([1.0 * x for x in list(range(1, 10)) * 10])
    data = np.random.default_rng(2).random(1100) * 10.0
    groupings = Series(data)
    grouped = result.groupby(groupings)
    grouped.mean()


def test_indices_concatenation_order() -> None:

    def f1(x: pd.DataFrame) -> pd.DataFrame:
        y = x[x.b % 2 == 1] ** 2
        if y.empty:
            multiindex = MultiIndex(
                levels=[[]] * 2, codes=[[]] * 2, names=["b", "c"]
            )
            res = DataFrame(columns=["a"], index=multiindex)
            return res
        else:
            y = y.set_index(["b", "c"])
            return y

    def f2(x: pd.DataFrame) -> pd.DataFrame:
        y = x[x.b % 2 == 1] ** 2
        if y.empty:
            return DataFrame()
        else:
            y = y.set_index(["b", "c"])
            return y

    def f3(x: pd.DataFrame) -> pd.DataFrame:
        y = x[x.b % 2 == 1] ** 2
        if y.empty:
            multiindex = MultiIndex(
                levels=[[]] * 2, codes=[[]] * 2, names=["foo", "bar"]
            )
            res = DataFrame(columns=["a", "b"], index=multiindex)
            return res
        else:
            return y

    df = DataFrame({"a": [1, 2, 2, 2], "b": range(4), "c": range(5, 9)})
    df2 = DataFrame({"a": [3, 2, 2, 2], "b": range(4), "c": range(5, 9)})
    result1 = df.groupby("a").apply(f1)
    result2 = df2.groupby("a").apply(f1)
    tm.assert_frame_equal(result1, result2)
    msg = "Cannot concat indices that do not have the same number of levels"
    with pytest.raises(AssertionError, match=msg):
        df.groupby("a").apply(f2)
    with pytest.raises(AssertionError, match=msg):
        df2.groupby("a").apply(f2)
    with pytest.raises(AssertionError, match=msg):
        df.groupby("a").apply(f3)
    with pytest.raises(AssertionError, match=msg):
        df2.groupby("a").apply(f3)


def test_attr_wrapper(ts: pd.Series) -> None:
    grouped = ts.groupby(lambda x: x.weekday())
    result = grouped.std()
    expected = grouped.agg(lambda x: np.std(x, ddof=1))
    tm.assert_series_equal(result, expected)
    result = grouped.describe()
    expected = {name: gp.describe() for name, gp in grouped}
    expected = DataFrame(expected).T
    tm.assert_frame_equal(result, expected)
    result = grouped.dtype
    expected = grouped.agg(lambda x: x.dtype)
    tm.assert_series_equal(result, expected)
    msg = "'SeriesGroupBy' object has no attribute 'foo'"
    with pytest.raises(AttributeError, match=msg):
        grouped.foo


def test_frame_groupby(tsframe: pd.DataFrame) -> None:
    grouped = tsframe.groupby(lambda x: x.weekday())
    aggregated = grouped.aggregate("mean")
    assert len(aggregated) == 5
    assert len(aggregated.columns) == 4
    tscopy = tsframe.copy()
    tscopy["weekday"] = [x.weekday() for x in tscopy.index]
    stragged = tscopy.groupby("weekday").aggregate("mean")
    tm.assert_frame_equal(stragged, aggregated, check_names=False)
    grouped = tsframe.head(30).groupby(lambda x: x.weekday())
    transformed = grouped.transform(lambda x: x - x.mean())
    assert len(transformed) == 30
    assert len(transformed.columns) == 4
    transformed = grouped.transform(lambda x: x.mean())
    for name, group in grouped:
        mean = group.mean()
        for idx in group.index:
            tm.assert_series_equal(transformed.xs(idx), mean, check_names=False)
    for weekday, group in grouped:
        assert group.index[0].weekday() == weekday
    groups = grouped.groups
    indices = grouped.indices
    for k, v in groups.items():
        samething = tsframe.index.take(indices[k])
        assert (samething == v).all()


def test_frame_set_name_single(df: pd.DataFrame) -> None:
    grouped = df.groupby("A")
    result = grouped.mean(numeric_only=True)
    assert result.index.name == "A"
    result = df.groupby("A", as_index=False).mean(numeric_only=True)
    assert result.index.name != "A"
    result = grouped[["C", "D"]].agg("mean")
    assert result.index.name == "A"
    result = grouped.agg({"C": "mean", "D": "std"})
    assert result.index.name == "A"
    result = grouped["C"].mean()
    assert result.index.name == "A"
    result = grouped["C"].agg("mean")
    assert result.index.name == "A"
    result = grouped["C"].agg(["mean", "std"])
    assert result.index.name == "A"
    msg = "nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        grouped["C"].agg({"foo": "mean", "bar": "std"})


def test_multi_func(df: pd.DataFrame) -> None:
    col1 = df["A"]
    col2 = df["B"]
    grouped = df.groupby([col1.get, col2.get])
    agged = grouped.mean(numeric_only=True)
    expected = df.groupby(["A", "B"]).mean()
    tm.assert_frame_equal(
        agged.loc[:, ["C", "D"]], expected.loc[:, ["C", "D"]], check_names=False
    )
    df = DataFrame(
        {
            "v1": np.random.default_rng(2).standard_normal(6),
            "v2": np.random.default_rng(2).standard_normal(6),
            "k1": np.array(["b", "b", "b", "a", "a", "a"]),
            "k2": np.array(["1", "1", "1", "2", "2", "2"]),
        },
        index=["one", "two", "three", "four", "five", "six"],
    )
    grouped = df.groupby(["k1", "k2"])
    grouped.agg("sum")


def test_multi_key_multiple_functions(df: pd.DataFrame) -> None:
    grouped = df.groupby(["A", "B"])["C"]
    agged = grouped.agg(["mean", "std"])
    expected = DataFrame(
        {"mean": grouped.agg("mean"), "std": grouped.agg("std")}
    )
    tm.assert_frame_equal(agged, expected)


def test_frame_multi_key_function_list() -> None:
    data = DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )
    grouped = data.groupby(["A", "B"])
    funcs = ["mean", "std"]
    agged = grouped.agg(funcs)
    expected = pd.concat(
        [
            grouped["D"].agg(funcs),
            grouped["E"].agg(funcs),
            grouped["F"].agg(funcs),
        ],
        keys=["D", "E", "F"],
        axis=1,
    )
    assert isinstance(agged.index, MultiIndex)
    assert isinstance(expected.index, MultiIndex)
    tm.assert_frame_equal(agged, expected)


def test_frame_multi_key_function_list_partial_failure(using_infer_string: bool) -> None:
    data = DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )
    grouped = data.groupby(["A", "B"])
    funcs = ["mean", "std"]
    msg = re.escape("agg function failed [how->mean,dtype->")
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg(funcs)


@pytest.mark.parametrize("op", [lambda x: x.sum(), lambda x: x.mean()])
def test_groupby_multiple_columns(df: pd.DataFrame, op: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    data = df
    grouped = data.groupby(["A", "B"])
    result1 = op(grouped)
    keys = []
    values = []
    for n1, gp1 in data.groupby("A"):
        for n2, gp2 in gp1.groupby("B"):
            keys.append((n1, n2))
            values.append(op(gp2.loc[:, ["C", "D"]]))
    mi = MultiIndex.from_tuples(keys, names=["A", "B"])
    expected = pd.concat(values, axis=1).T
    expected.index = mi
    for col in ["C", "D"]:
        result_col = op(grouped[col])
        pivoted = result1[col]
        exp = expected[col]
        tm.assert_series_equal(result_col, exp)
        tm.assert_series_equal(pivoted, exp)
    result = data["C"].groupby([data["A"], data["B"]]).mean()
    expected = data.groupby(["A", "B"]).mean()["C"]
    tm.assert_series_equal(result, expected)


def test_as_index_select_column() -> None:
    df = DataFrame([[1, 2], [1, 4], [5, 6]], columns=["A", "B"])
    result = df.groupby("A", as_index=False)["B"].get_group(1)
    expected = Series([2, 4], name="B")
    tm.assert_series_equal(result, expected)
    result = df.groupby("A", as_index=False, group_keys=True)["B"].apply(
        lambda x: x.cumsum()
    )
    expected = Series([2, 6, 6], name="B", index=range(3))
    tm.assert_series_equal(result, expected)


def test_groupby_as_index_select_column_sum_empty_df() -> None:
    df = DataFrame(columns=Index(["A", "B", "C"], name="alpha"))
    left = df.groupby(by="A", as_index=False)["B"].sum(numeric_only=False)
    expected = DataFrame(columns=df.columns[:2], index=range(0))
    expected.columns.names = [None]
    tm.assert_frame_equal(left, expected)


@pytest.mark.parametrize(
    "reduction_func",
    [
        "count",
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "var",
        "skew",
        "kurt",
        "prod",
        "sem",
        "nunique",
        "size",
        "first",
        "last",
        "ngroup",
        "nth",
        "nth",
        "nth",
        "nth",
        "nth",
    ],
)
def test_ops_not_as_index(
    reduction_func: str, df: pd.DataFrame, using_infer_string: bool
) -> None:
    if reduction_func in ("corrwith", "nth", "ngroup"):
        pytest.skip(f"GH 5755: Test not applicable for {reduction_func}")
    df = DataFrame(
        np.random.default_rng(2).integers(0, 5, size=(100, 2)), columns=["a", "b"]
    )
    expected = getattr(df.groupby("a"), reduction_func)()
    if reduction_func == "size":
        expected = expected.rename("size")
    expected = expected.reset_index()
    if reduction_func != "size":
        expected["a"] = expected["a"].astype(df["a"].dtype)
    g = df.groupby("a", as_index=False)
    result = getattr(g, reduction_func)()
    tm.assert_frame_equal(result, expected)
    result = g.agg(reduction_func)
    tm.assert_frame_equal(result, expected)
    result = getattr(g["b"], reduction_func)()
    tm.assert_frame_equal(result, expected)
    result = g["b"].agg(reduction_func)
    tm.assert_frame_equal(result, expected)


def test_as_index_series_return_frame(df: pd.DataFrame) -> None:
    grouped = df.groupby("A", as_index=False)
    grouped2 = df.groupby(["A", "B"], as_index=False)
    result = grouped["C"].agg("sum")
    expected = grouped.agg("sum").loc[:, ["A", "C"]]
    assert isinstance(result, DataFrame)
    tm.assert_frame_equal(result, expected)
    result2 = grouped2["C"].agg("sum")
    expected2 = grouped2.agg("sum").loc[:, ["A", "B", "C"]]
    assert isinstance(result2, DataFrame)
    tm.assert_frame_equal(result2, expected2)
    result = grouped["C"].sum()
    expected = grouped.sum().loc[:, ["A", "C"]]
    assert isinstance(result, DataFrame)
    tm.assert_frame_equal(result, expected)
    result2 = grouped2["C"].sum()
    expected2 = grouped2.sum().loc[:, ["A", "B", "C"]]
    assert isinstance(result2, DataFrame)
    tm.assert_frame_equal(result2, expected2)


def test_as_index_series_column_slice_raises(df: pd.DataFrame) -> None:
    grouped = df.groupby("A", as_index=False)
    msg = "Column\\(s\\) C already selected"
    with pytest.raises(IndexError, match=msg):
        grouped["C"].__getitem__("D")


def test_groupby_as_index_cython(df: pd.DataFrame) -> None:
    data = df
    grouped = data.groupby("A", as_index=False)
    result = grouped.mean(numeric_only=True)
    expected = data.groupby(["A"]).mean(numeric_only=True)
    expected.insert(0, "A", expected.index)
    expected.index = RangeIndex(len(expected))
    tm.assert_frame_equal(result, expected)
    grouped = data.groupby(["A", "B"], as_index=False)
    result = grouped.mean()
    expected = data.groupby(["A", "B"]).mean()
    arrays = list(zip(*expected.index.values))
    expected.insert(0, "A", arrays[0])
    expected.insert(1, "B", arrays[1])
    expected.index = RangeIndex(len(expected))
    tm.assert_frame_equal(result, expected)


def test_groupby_series_scalar(df: pd.DataFrame) -> None:
    grouped = df.groupby(["A", "B"], as_index=False)
    result = grouped["C"].agg(len)
    expected = grouped.agg(len).loc[:, ["A", "B", "C"]]
    tm.assert_frame_equal(result, expected)


def test_groupby_multiple_key() -> None:
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    grouped = df.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day])
    agged = grouped.sum()
    tm.assert_almost_equal(df.values, agged.values)


def test_groupby_multi_corner(df: pd.DataFrame) -> None:
    df = df.copy()
    df["bad"] = np.nan
    agged = df.groupby(["A", "B"]).mean()
    expected = df.groupby(["A", "B"]).mean()
    expected["bad"] = np.nan
    tm.assert_frame_equal(agged, expected)


def test_raises_on_nuisance(df: pd.DataFrame, using_infer_string: bool) -> None:
    grouped = df.groupby("A")
    msg = re.escape("agg function failed [how->mean,dtype->")
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg("mean")
    with pytest.raises(TypeError, match=msg):
        grouped.mean()
    df = df.loc[:, ["A", "C", "D"]]
    df["E"] = datetime.now()
    grouped = df.groupby("A")
    msg = "datetime64 type does not support operation 'sum'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg("sum")
    with pytest.raises(TypeError, match=msg):
        grouped.sum()


@pytest.mark.parametrize("agg_function", ["max", "min"])
def test_keep_nuisance_agg(df: pd.DataFrame, agg_function: str) -> None:
    grouped = df.groupby("A")
    result = getattr(grouped, agg_function)()
    expected = result.copy()
    expected.loc["bar", "B"] = getattr(df.loc[df["A"] == "bar", "B"], agg_function)()
    expected.loc["foo", "B"] = getattr(df.loc[df["A"] == "foo", "B"], agg_function)()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "agg_function", ["sum", "mean", "prod", "std", "var", "sem", "median"]
)
@pytest.mark.parametrize("numeric_only", [True, False])
def test_omit_nuisance_agg(
    df: pd.DataFrame,
    agg_function: str,
    numeric_only: bool,
    using_infer_string: bool,
) -> None:
    grouped = df.groupby("A")
    no_drop_nuisance = ("var", "std", "sem", "mean", "prod", "median")
    if (
        agg_function in no_drop_nuisance
        and not numeric_only
    ):
        if using_infer_string:
            msg = f"dtype 'str' does not support operation '{agg_function}'"
            klass = TypeError
        elif agg_function in ("std", "sem"):
            klass = ValueError
            msg = "could not convert string to float: 'one'"
        else:
            klass = TypeError
            msg = re.escape(
                f"agg function failed [how->{agg_function},dtype->"
            )
        with pytest.raises(klass, match=msg):
            getattr(grouped, agg_function)(numeric_only=numeric_only)
    else:
        result = getattr(grouped, agg_function)(numeric_only=numeric_only)
        if not numeric_only and agg_function == "sum":
            columns = ["A", "B", "C", "D"]
        else:
            columns = ["A", "C", "D"]
        expected = getattr(df.loc[:, columns].groupby("A"), agg_function)(
            numeric_only=numeric_only
        )
        tm.assert_frame_equal(result, expected)


def test_raise_on_nuisance_python_single(df: pd.DataFrame, using_infer_string: bool) -> None:
    grouped = df.groupby("A")
    err = ValueError
    msg = "could not convert"
    if using_infer_string:
        err = TypeError
        msg = "dtype 'str' does not support operation 'skew'"
    with pytest.raises(err, match=msg):
        grouped.skew()


def test_raise_on_nuisance_python_multiple(
    three_group: pd.DataFrame, using_infer_string: bool
) -> None:
    grouped = three_group.groupby(["A", "B"])
    msg = re.escape("agg function failed [how->mean,dtype->")
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg("mean")
    with pytest.raises(TypeError, match=msg):
        grouped.mean()


def test_empty_groups_corner(multiindex_dataframe_random_data: pd.DataFrame) -> None:
    df = DataFrame(
        {
            "k1": np.array(["b", "b", "b", "a", "a", "a"]),
            "k2": np.array(["1", "1", "1", "2", "2", "2"]),
            "k3": ["foo", "bar"] * 3,
            "v1": np.random.default_rng(2).standard_normal(6),
            "v2": np.random.default_rng(2).standard_normal(6),
        }
    )
    grouped = df.groupby(["k1", "k2"])
    result = grouped[["v1", "v2"]].agg("mean")
    expected = grouped.mean(numeric_only=True)
    tm.assert_frame_equal(result, expected)
    grouped = multiindex_dataframe_random_data.iloc[3:5].groupby(level=0)
    agged = grouped.apply(lambda x: x.mean())
    agged_A = grouped["A"].apply("mean")
    tm.assert_series_equal(agged["A"], agged_A)
    assert agged.index.name == "first"


def test_nonsense_func() -> None:
    df = DataFrame([0])
    msg = "unsupported operand type\\(s\\) for \\+: 'int' and 'str'"
    with pytest.raises(TypeError, match=msg):
        df.groupby(lambda x: x + "foo")


def test_wrap_aggregated_output_multindex(
    multiindex_dataframe_random_data: pd.DataFrame, using_infer_string: bool
) -> None:
    df = multiindex_dataframe_random_data.T
    df["baz", "two"] = "peekaboo"
    keys = [np.array([0, 0, 1]), np.array([0, 0, 1])]
    msg = re.escape("agg function failed [how->mean,dtype->")
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        df.groupby(keys).agg("mean")
    agged = df.drop(columns=("baz", "two")).groupby(keys).agg("mean")
    assert isinstance(agged.columns, MultiIndex)

    def aggfun(ser: pd.Series) -> pd.Series:
        if ser.name == ("foo", "one"):
            raise TypeError("Test error message")
        return ser.sum()

    with pytest.raises(TypeError, match="Test error message"):
        df.groupby(keys).aggregate(aggfun)


def test_groupby_level_apply(multiindex_dataframe_random_data: pd.DataFrame) -> None:
    result = multiindex_dataframe_random_data.groupby(level=0).count()
    assert result.index.name == "first"
    result = multiindex_dataframe_random_data.groupby(level=1).count()
    assert result.index.name == "second"
    result = multiindex_dataframe_random_data["A"].groupby(level=0).count()
    assert result.index.name == "first"


def test_groupby_level_mapper(multiindex_dataframe_random_data: pd.DataFrame) -> None:
    deleveled = multiindex_dataframe_random_data.reset_index()
    mapper0 = {"foo": 0, "bar": 0, "baz": 1, "qux": 1}
    mapper1 = {"one": 0, "two": 0, "three": 1}
    result0 = multiindex_dataframe_random_data.groupby(mapper0, level=0).sum()
    result1 = multiindex_dataframe_random_data.groupby(mapper1, level=1).sum()
    mapped_level0 = np.array([mapper0.get(x) for x in deleveled["first"]], dtype=np.int64)
    mapped_level1 = np.array([mapper1.get(x) for x in deleveled["second"]], dtype=np.int64)
    expected0 = multiindex_dataframe_random_data.groupby(mapped_level0).sum()
    expected1 = multiindex_dataframe_random_data.groupby(mapped_level1).sum()
    expected0.index.name, expected1.index.name = ("first", "second")
    tm.assert_frame_equal(result0, expected0)
    tm.assert_frame_equal(result1, expected1)


def test_groupby_level_nonmulti() -> None:
    s = Series(
        [1, 2, 3, 10, 4, 5, 20, 6],
        Index([1, 2, 3, 1, 4, 5, 2, 6], name="foo"),
    )
    expected = Series(
        [11, 22, 3, 4, 5, 6], Index(list(range(1, 7)), name="foo")
    )
    result = s.groupby(level=0).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=[0]).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=-1).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=[-1]).sum()
    tm.assert_series_equal(result, expected)
    msg = "level > 0 or level < -1 only valid with MultiIndex"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=1)
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=-2)
    msg = "No group keys passed!"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[])
    msg = "multiple levels only valid with MultiIndex"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[0, 0])
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[0, 1])
    msg = "level > 0 or level < -1 only valid with MultiIndex"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[1])


def test_groupby_complex() -> None:
    a = Series(data=np.arange(4) * (1 + 2j), index=[0, 0, 1, 1])
    expected = Series((1 + 2j, 5 + 10j), index=Index([0, 1]))
    result = a.groupby(level=0).sum()
    tm.assert_series_equal(result, expected)


def test_groupby_complex_mean() -> None:
    df = DataFrame(
        [{"a": 2, "b": 1 + 2j}, {"a": 1, "b": 1 + 1j}, {"a": 1, "b": 1 + 2j}]
    )
    result = df.groupby("b").mean()
    expected = DataFrame(
        [[1.0], [1.5]],
        index=Index([1 + 1j, 1 + 2j], name="b"),
        columns=Index(["a"]),
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_complex_numbers() -> None:
    df = DataFrame(
        [
            {"a": 1, "b": 1 + 1j},
            {"a": 1, "b": 1 + 2j},
            {"a": 4, "b": 1},
        ]
    )
    expected = DataFrame(
        np.array([1, 1, 1], dtype=np.int64),
        index=Index([1 + 1j, 1 + 2j, 1 + 0j], name="b"),
        columns=Index(["a"]),
    )
    result = df.groupby("b", sort=False).count()
    tm.assert_frame_equal(result, expected)
    expected.index = Index([1 + 0j, 1 + 1j, 1 + 2j], name="b")
    result = df.groupby("b", sort=True).count()
    tm.assert_frame_equal(result, expected)


def test_groupby_series_with_name(df: pd.DataFrame) -> None:
    result = df.groupby(df["A"]).mean(numeric_only=True)
    result2 = df.groupby(df["A"], as_index=False).mean(numeric_only=True)
    assert result.index.name == "A"
    assert "A" in result2
    result = df.groupby([df["A"], df["B"]]).mean()
    result2 = df.groupby([df["A"], df["B"]], as_index=False).mean()
    assert result.index.names == ("A", "B")
    assert "A" in result2
    assert "B" in result2


def test_groupby_seriesgroupby_name_attr(df: pd.DataFrame) -> None:
    result = df.groupby("A")["C"]
    assert result.count().name == "C"
    assert result.mean().name == "C"

    def testFunc(x: pd.Series) -> float:
        return np.sum(x) * 2

    assert result.agg(testFunc).name == "C"


def test_consistency_name() -> None:
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": np.random.default_rng(2).standard_normal(8) + 1.0,
            "D": np.arange(8),
        }
    )
    expected = df.groupby(["A"]).B.count()
    result = df.B.groupby(df.A).count()
    tm.assert_series_equal(result, expected)


def test_groupby_name_propagation(df: pd.DataFrame) -> None:

    def summarize(
        df: pd.DataFrame, name: Optional[str] = None
    ) -> pd.Series:
        return Series(
            {"count": 1, "mean": 2, "omissions": 3}, name=name
        )

    def summarize_random_name(df: pd.DataFrame) -> pd.Series:
        return Series(
            {"count": 1, "mean": 2, "omissions": 3}, name=df.iloc[0]["C"]
        )

    metrics = df.groupby("A").apply(summarize)
    assert metrics.columns.name is None
    metrics = df.groupby("A").apply(summarize, "metrics")
    assert metrics.columns.name == "metrics"
    metrics = df.groupby("A").apply(summarize_random_name)
    assert metrics.columns.name is None


def test_groupby_nonstring_columns() -> None:
    df = DataFrame([np.arange(10) for _ in range(10)])
    grouped = df.groupby(0)
    result = grouped.mean()
    expected = df.groupby(df[0]).mean()
    tm.assert_frame_equal(result, expected)


def test_groupby_mixed_type_columns() -> None:
    df = DataFrame([[0, 1, 2]], columns=["A", "B", 0])
    expected = DataFrame(
        [[1, 2]],
        columns=["B", 0],
        index=Index([0], name="A"),
    )
    result = df.groupby("A").first()
    tm.assert_frame_equal(result, expected)
    result = df.groupby("A").sum()
    tm.assert_frame_equal(result, expected)


def test_cython_grouper_series_bug_noncontig() -> None:
    arr = np.empty((100, 100))
    arr.fill(np.nan)
    obj = Series(arr[:, 0])
    inds = np.tile(range(10), 10)
    result = obj.groupby(inds).agg(Series.median)
    assert result.isna().all()


def test_series_grouper_noncontig_index() -> None:
    index = Index(["a" * 10] * 100)
    values = Series(
        np.random.default_rng(2).standard_normal(50), index=index[::2]
    )
    labels = np.random.default_rng(2).integers(0, 5, 50)
    grouped = values.groupby(labels)

    def f(x: pd.Series) -> int:
        return len(set(map(id, x.index)))

    grouped.agg(f)


def test_convert_objects_leave_decimal_alone() -> None:
    s = Series(range(5))
    labels = np.array(["a", "b", "c", "d", "e"], dtype="O")

    def convert_fast(x: pd.Series) -> Decimal:
        return Decimal(str(x.mean()))

    def convert_force_pure(x: pd.Series) -> Decimal:
        assert len(x.values.base) > 0
        return Decimal(str(x.mean()))

    grouped = s.groupby(labels)
    result = grouped.agg(convert_fast)
    assert result.dtype == object
    assert isinstance(result.iloc[0], Decimal)
    result = grouped.agg(convert_force_pure)
    assert result.dtype == object
    assert isinstance(result.iloc[0], Decimal)


def test_groupby_dtype_inference_empty() -> None:
    df = DataFrame({"x": [], "range": np.arange(0, dtype="int64")})
    assert df["x"].dtype == np.float64
    result = df.groupby("x").first()
    exp_index = Index([], name="x", dtype=np.float64)
    expected = DataFrame(
        {"range": Series([], index=exp_index, dtype="int64")}
    )
    tm.assert_frame_equal(result, expected, by_blocks=True)


def test_groupby_unit64_float_conversion() -> None:
    df = DataFrame(
        {"first": [1], "second": [1], "value": [16148277970000000000]}
    )
    result = df.groupby(["first", "second"])["value"].max()
    expected = Series(
        [16148277970000000000],
        MultiIndex.from_product([[1], [1]], names=["first", "second"]),
        name="value",
    )
    tm.assert_series_equal(result, expected)
    expected.index = MultiIndex.from_product([[1], [1]], names=["first", "second"])
    expected = Series(
        [16148277970000000000],
        MultiIndex.from_product([[1], [1]], names=["first", "second"]),
        name="value",
    )
    tm.assert_series_equal(result, expected)


def test_groupby_list_infer_array_like(df: pd.DataFrame) -> None:
    result = df.groupby(list(df["A"])).mean(numeric_only=True)
    expected = df.groupby(df["A"]).mean(numeric_only=True)
    tm.assert_frame_equal(result, expected, check_names=False)
    with pytest.raises(KeyError, match="^'foo'$"):
        df.groupby(list(df["A"][:-1]))
    df = DataFrame(
        {
            "foo": [0, 1],
            "bar": [3, 4],
            "val": np.random.default_rng(2).standard_normal(2),
        }
    )
    result = df.groupby(["foo", "bar"]).mean()
    expected = df.groupby([df["foo"], df["bar"]]).mean()[["val"]]


def test_groupby_keys_same_size_as_index() -> None:
    freq = "s"
    index = date_range(
        start=Timestamp("2015-09-29T11:34:44-0700"),
        periods=2,
        freq=freq,
    )
    df = DataFrame(
        [["A", 10], ["B", 15]],
        columns=["metric", "values"],
        index=index,
    )
    result = df.groupby(
        [Grouper(level=0, freq=freq), "metric"]
    ).mean()
    expected = df.set_index([df.index, "metric"]).astype(float)
    tm.assert_frame_equal(result, expected)


def test_groupby_one_row() -> None:
    msg = "^'Z'$"
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((1, 4)),
        columns=list("ABCD"),
    )
    with pytest.raises(KeyError, match=msg):
        df1.groupby("Z")
    df2 = DataFrame(
        np.random.default_rng(2).standard_normal((2, 4)),
        columns=list("ABCD"),
    )
    with pytest.raises(KeyError, match=msg):
        df2.groupby("Z")


def test_groupby_nat_exclude() -> None:
    df = DataFrame(
        {
            "values": np.random.default_rng(2).standard_normal(8),
            "dt": [
                np.nan,
                Timestamp("2013-01-01"),
                np.nan,
                Timestamp("2013-02-01"),
                np.nan,
                Timestamp("2013-02-01"),
                np.nan,
                Timestamp("2013-01-01"),
            ],
            "str": [np.nan, "a", np.nan, "a", np.nan, "a", np.nan, "b"],
        }
    )
    grouped = df.groupby("dt")
    expected = [
        RangeIndex(start=1, stop=13, step=6),
        RangeIndex(start=3, stop=7, step=2),
    ]
    keys = sorted(grouped.groups.keys())
    assert len(keys) == 2
    for k, e in zip(keys, expected):
        tm.assert_index_equal(grouped.groups[k], e)
    tm.assert_frame_equal(grouped._grouper.groupings[0].obj, df)
    assert grouped.ngroups == 2
    expected = {
        Timestamp("2013-01-01 00:00:00"): np.array([1, 7], dtype=np.intp),
        Timestamp("2013-02-01 00:00:00"): np.array([3, 5], dtype=np.intp),
    }
    for k in grouped.indices:
        tm.assert_numpy_array_equal(grouped.indices[k], expected[k])
    tm.assert_frame_equal(
        grouped.get_group(Timestamp("2013-01-01")),
        df.iloc[[1, 7]],
    )
    tm.assert_frame_equal(
        grouped.get_group(Timestamp("2013-02-01")),
        df.iloc[[3, 5]],
    )
    with pytest.raises(KeyError, match="^NaT$"):
        grouped.get_group(pd.NaT)
    nan_df = DataFrame(
        {"nan": [np.nan, np.nan, np.nan], "nat": [pd.NaT, pd.NaT, pd.NaT]}
    )
    assert nan_df["nan"].dtype == "float64"
    assert nan_df["nat"].dtype == "datetime64[s]"
    for key in ["nan", "nat"]:
        grouped = nan_df.groupby(key)
        assert grouped.groups == {}
        assert grouped.ngroups == 0
        assert grouped.indices == {}
        with pytest.raises(KeyError, match="^nan$"):
            grouped.get_group(np.nan)
        with pytest.raises(KeyError, match="^NaT$"):
            grouped.get_group(pd.NaT)


def test_groupby_two_group_keys_all_nan() -> None:
    df = DataFrame(
        {"a": [np.nan, np.nan], "b": [np.nan, np.nan], "c": [1, 2]}
    )
    result = df.groupby(["a", "b"]).indices
    assert result == {}


def test_groupby_2d_malformed() -> None:
    d = DataFrame(index=range(2))
    d["group"] = ["g1", "g2"]
    d["zeros"] = [0, 0]
    d["ones"] = [1, 1]
    d["label"] = ["l1", "l2"]
    tmp = d.groupby(["group"]).mean(numeric_only=True)
    res_values = np.array([[0.0, 1.0], [0.0, 1.0]])
    tm.assert_index_equal(tmp.columns, Index(["zeros", "ones"], dtype=object))
    tm.assert_numpy_array_equal(tmp.values, res_values)


def test_int32_overflow() -> None:
    B = np.concatenate((np.arange(10000), np.arange(10000), np.arange(5000)))
    A = np.arange(25000)
    df = DataFrame({"A": A, "B": B, "C": A, "D": B, "E": np.random.default_rng(2).standard_normal(25000)})
    left = df.groupby(["A", "B", "C", "D"]).sum()
    right = df.groupby(["D", "C", "B", "A"]).sum()
    assert len(left) == len(right)


def test_groupby_sort_multi() -> None:
    df = DataFrame(
        {
            "a": ["foo", "bar", "baz"],
            "b": [3, 2, 1],
            "c": [0, 1, 2],
            "d": np.random.default_rng(2).standard_normal(3),
        }
    )
    tups = [tuple(row) for row in df[["a", "b", "c"]].values]
    tups = com.asarray_tuplesafe(tups)
    result = df.groupby(["a", "b", "c"], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups[[1, 2, 0]])
    tups = [tuple(row) for row in df[["c", "a", "b"]].values]
    tups = com.asarray_tuplesafe(tups)
    result = df.groupby(["c", "a", "b"], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups)
    tups = [tuple(x) for x in df[["b", "c", "a"]].values]
    tups = com.asarray_tuplesafe(tups)
    result = df.groupby(["b", "c", "a"], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups[[2, 1, 0]])
    df = DataFrame(
        {
            "a": [0, 1, 2, 0, 1, 2],
            "b": [0, 0, 0, 1, 1, 1],
            "d": np.random.default_rng(2).standard_normal(6),
        }
    )
    grouped = df.groupby(["a", "b"])["d"]
    result = grouped.sum()

    def _check_groupby(
        df: pd.DataFrame,
        result: pd.Series,
        keys: List[str],
        field: str,
        f: Callable[[pd.Series], pd.Series] = lambda x: x.sum(),
    ) -> None:
        tups = [tuple(row) for row in df[keys].values]
        tups = com.asarray_tuplesafe(tups)
        expected = f(df.groupby(tups)[field])
        for k, v in expected.items():
            assert result[k] == v

    _check_groupby(df, result, ["a", "b"], "d")


def test_dont_clobber_name_column() -> None:
    df = DataFrame(
        {"key": ["a", "a", "a", "b", "b", "b"], "name": ["foo", "bar", "baz"] * 2}
    )
    result = df.groupby("key", group_keys=False).apply(lambda x: x)
    tm.assert_frame_equal(result, df[["name"]])


def test_skip_group_keys() -> None:
    tsf = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    grouped = tsf.groupby(lambda x: x.month, group_keys=False)
    result = grouped.apply(lambda x: x.sort_values(by="A")[:3])
    pieces = [group.sort_values(by="A")[:3] for key, group in grouped]
    expected = pd.concat(pieces)
    tm.assert_frame_equal(result, expected)
    grouped = tsf["A"].groupby(lambda x: x.month, group_keys=False)
    result = grouped.apply(lambda x: x.sort_values()[:3])
    pieces = [group.sort_values()[:3] for key, group in grouped]
    expected = pd.concat(pieces)
    tm.assert_series_equal(result, expected)


def test_no_nonsense_name(float_frame: pd.DataFrame) -> None:
    s = float_frame["C"].copy()
    s.name = None
    result = s.groupby(float_frame["A"]).agg("sum")
    assert result.name is None


def test_multifunc_sum_bug() -> None:
    x = DataFrame(np.arange(9).reshape(3, 3))
    x["test"] = 0
    x["fl"] = [1.3, 1.5, 1.6]
    grouped = x.groupby("test")
    result = grouped.agg({"fl": "sum", 2: "size"})
    assert result["fl"].dtype == np.float64


def test_handle_dict_return_value(df: pd.DataFrame) -> None:

    def f(group: pd.Series) -> Dict[str, Any]:
        return {"max": group.max(), "min": group.min()}

    def g(group: pd.Series) -> pd.Series:
        return Series({"max": group.max(), "min": group.min()})

    result = df.groupby("A")["C"].apply(f)
    expected = df.groupby("A")["C"].apply(g)
    assert isinstance(result, Series)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("grouper", ["A", ["A", "B"]])
def test_set_group_name(df: pd.DataFrame, grouper: Union[str, List[str]]) -> None:

    def f(group: pd.DataFrame) -> pd.DataFrame:
        assert group.name is not None
        return group

    def freduce(group: pd.DataFrame) -> pd.DataFrame:
        assert group.name is not None
        return group.sum()

    def freducex(x: pd.DataFrame) -> pd.DataFrame:
        return freduce(x)

    grouped = df.groupby(grouper, group_keys=False)
    grouped.apply(f)
    grouped.aggregate(freduce)
    grouped.aggregate({"C": freduce, "D": freduce})
    grouped.transform(f)
    grouped["C"].apply(f)
    grouped["C"].aggregate(freduce)
    grouped["C"].aggregate([freduce, freducex])
    grouped["C"].transform(f)


def test_group_name_available_in_inference_pass() -> None:
    df = DataFrame(
        {"a": [0, 0, 1, 1, 2, 2], "b": np.arange(6)}
    )
    names: List[int] = []

    def f(group: pd.DataFrame) -> pd.DataFrame:
        names.append(group.name)
        return group.copy()

    df.groupby("a", sort=False, group_keys=False).apply(f)
    expected_names = [0, 1, 2]
    assert names == expected_names


def test_no_dummy_key_names(df: pd.DataFrame) -> None:
    result = df.groupby(df["A"].values).sum()
    assert result.index.name is None
    result2 = df.groupby([df["A"].values, df["B"].values]).sum()
    assert result2.index.names == (None, None)


def test_groupby_sort_multiindex_series() -> None:
    index = MultiIndex(
        levels=[[1, 2], [1, 2]], codes=[[0, 0, 1], [1, 0, 0]], names=["a", "b"]
    )
    mseries = Series([0, 1, 2, 3, 4, 5], index=index)
    index = MultiIndex(
        levels=[[1, 2], [1, 2]],
        codes=[[0, 0, 1], [1, 0, 1]],
        names=["a", "b"],
    )
    mseries_result = Series([0, 2, 4], index=index)
    result = mseries.groupby(level=["a", "b"], sort=False).first()
    tm.assert_series_equal(result, mseries_result)
    result = mseries.groupby(level=["a", "b"], sort=True).first()
    tm.assert_series_equal(result, mseries_result.sort_index())


def test_groupby_reindex_inside_function() -> None:
    periods = 1000
    ind = date_range(start="2012/1/1", freq="5min", periods=periods)
    df = DataFrame({"high": np.arange(periods), "low": np.arange(periods)}, index=ind)

    def agg_before(func: Callable[[pd.DataFrame], Any], fix: bool = False) -> Callable[[pd.DataFrame], Any]:
        """
        Run an aggregate func on the subset of data.
        """

        def _func(data: pd.DataFrame) -> Any:
            d = data.loc[data.index.map(lambda x: x.hour < 11)].dropna()
            if fix:
                data[data.index[0]]
            if len(d) == 0:
                return None
            return func(d)

        return _func

    grouped = df.groupby(lambda x: datetime(x.year, x.month, x.day))
    closure_bad = grouped.agg({"high": agg_before(np.max)})
    closure_good = grouped.agg({"high": agg_before(np.max, True)})
    tm.assert_frame_equal(closure_bad, closure_good)


def test_groupby_multiindex_missing_pair() -> None:
    df = DataFrame(
        {
            "group1": ["a", "a", "a", "b"],
            "group2": ["c", "c", "d", "c"],
            "value": [1, 1, 1, 5],
        }
    )
    df = df.set_index(["group1", "group2"])
    df_grouped = df.groupby(level=["group1", "group2"], sort=True)
    res = df_grouped.agg("sum")
    idx = MultiIndex.from_tuples(
        [("a", "c"), ("a", "d"), ("b", "c")],
        names=["group1", "group2"],
    )
    expected = DataFrame(
        [[2], [1], [5]],
        index=idx,
        columns=["value"],
    )
    tm.assert_frame_equal(res, expected)


def test_groupby_multiindex_not_lexsorted(performance_warning: Any) -> None:
    lexsorted_mi = MultiIndex.from_tuples(
        [("a", ""), ("b1", "c1"), ("b2", "c2")], names=["b", "c"]
    )
    lexsorted_df = DataFrame([[1, 3, 4]], columns=lexsorted_mi)
    assert lexsorted_df.columns._is_lexsorted()
    not_lexsorted_df = DataFrame(
        {"a": [1, "b1", "c1", "b2", "c2"], "b": [1, "c1", "c2", "c3", "c4"], "d": [3, 4, 5, 6, 7]}
    )
    not_lexsorted_df = not_lexsorted_df.pivot_table(
        index="a", columns=["b", "c"], values="d"
    )
    not_lexsorted_df = not_lexsorted_df.reset_index()
    assert not not_lexsorted_df.columns._is_lexsorted()
    expected = lexsorted_df.groupby("a").mean()
    with tm.assert_produces_warning(performance_warning):
        result = not_lexsorted_df.groupby("a").mean()
    tm.assert_frame_equal(expected, result)
    df = DataFrame(
        {"x": ["a", "a", "b", "a"], "y": [1, 1, 2, 2], "z": [1, 2, 3, 4]}
    ).set_index(["x", "y"])
    assert not df.index._is_lexsorted()
    for level in [0, 1, [0, 1]]:
        for sort in [False, True]:
            result = df.groupby(
                level=level, sort=sort, group_keys=False
            ).apply(DataFrame.drop_duplicates)
            expected = df
            tm.assert_frame_equal(expected, result)
            result = df.sort_index().groupby(
                level=level, sort=sort, group_keys=False
            ).apply(DataFrame.drop_duplicates)
            expected = df.sort_index()
            tm.assert_frame_equal(expected, result)


def test_index_label_overlaps_location() -> None:
    df = DataFrame(list("ABCDE"), index=[2, 0, 2, 1, 1])
    g = df.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)
    df.index = df.index.astype(float)
    g = df.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)


def test_transform_doesnt_clobber_ints() -> None:
    n = 6
    x = np.arange(n)
    df = DataFrame(
        {"a": x // 2, "b": 2.0 * x, "c": 3.0 * x}
    )
    df2 = DataFrame(
        {"a": x // 2 * 1.0, "b": 2.0 * x, "c": 3.0 * x}
    )
    gb = df.groupby("a")
    result = gb.transform("mean")
    gb2 = df2.groupby("a")
    expected = gb2.transform("mean")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sort_column",
    ["ints", "floats", "strings", ["ints", "floats"], ["ints", "strings"]],
)
@pytest.mark.parametrize(
    "group_column",
    ["int_groups", "string_groups", ["int_groups", "string_groups"]],
)
def test_groupby_preserves_sort(
    sort_column: Union[str, List[str]],
    group_column: Union[str, List[str]],
) -> None:
    df = DataFrame(
        {
            "int_groups": [3, 1, 0, 1, 0, 3, 3, 3],
            "string_groups": ["z", "a", "z", "a", "a", "g", "g", "g"],
            "ints": [8, 7, 4, 5, 2, 9, 1, 1],
            "floats": [2.3, 5.3, 6.2, -2.4, 2.2, 1.1, 1.1, 5],
            "strings": ["z", "d", "a", "e", "word", "word2", "42", "47"],
        }
    )
    df = df.sort_values(by=sort_column)
    g = df.groupby(group_column)

    def test_sort(x: pd.DataFrame) -> None:
        tm.assert_frame_equal(x, x.sort_values(by=sort_column))

    g.apply(test_sort)


def test_pivot_table_values_key_error() -> None:
    df = DataFrame(
        {
            "eventDate": date_range(datetime.today(), periods=20, freq="ME").tolist(),
            "thename": range(20),
        }
    )
    df["year"] = df.set_index("eventDate").index.year
    df["month"] = df.set_index("eventDate").index.month
    with pytest.raises(KeyError, match="'badname'"):
        df.reset_index().pivot_table(
            index="year", columns="month", values="badname", aggfunc="count"
        )


@pytest.mark.parametrize(
    "columns", ["C", ["C"]]
)
@pytest.mark.parametrize(
    "keys", [["A"], ["A", "B"]]
)
@pytest.mark.parametrize(
    "values",
    [
        [True],
        [0],
        [0.0],
        ["a"],
        Categorical([0]),
        [to_datetime(0)],
        date_range("2016-01-01", periods=3, freq="D"),
        pd.array([0], dtype="Int64"),
        pd.array([0], dtype="Float64"),
        pd.array([False], dtype="boolean"),
    ],
    ids=[
        "bool",
        "int",
        "float",
        "str",
        "cat",
        "dt64",
        "dt64tz",
        "period",
        "Int64",
        "Float64",
        "boolean",
    ],
)
@pytest.mark.parametrize(
    "method",
    ["attr", "agg", "apply"],
)
@pytest.mark.parametrize(
    "op",
    ["idxmax", "idxmin", "min", "max", "sum", "prod", "skew", "kurt"],
)
def test_empty_groupby(
    columns: Union[str, List[str]],
    keys: List[str],
    values: List[Union[bool, int, float, str, Categorical, pd.Timestamp, pd.Period, pd.array]],
    method: str,
    op: str,
    dropna: bool,
    using_infer_string: bool,
) -> None:
    override_dtype: Optional[str] = None
    if isinstance(values, BooleanArray) and op in ["sum", "prod"]:
        override_dtype = "Int64"
    if isinstance(values[0], bool) and op in ("prod", "sum"):
        override_dtype = "int64"
    df = DataFrame({"A": values, "B": values, "C": values}, columns=list("ABC"))
    if hasattr(values, "dtype"):
        assert (df.dtypes == values.dtype).all()
    df = df.iloc[:0]
    gb = df.groupby(
        keys, group_keys=False, dropna=dropna, observed=False
    )[columns]

    def get_result(**kwargs: Any) -> Any:
        if method == "attr":
            return getattr(gb, op)(**kwargs)
        else:
            return getattr(gb, method)(op, **kwargs)

    def get_categorical_invalid_expected() -> pd.DataFrame:
        lev = Categorical([0], dtype=values.dtype)
        if len(keys) != 1:
            idx = MultiIndex.from_product([lev, lev], names=keys)
        else:
            idx = Index(lev, name=keys[0])
        if using_infer_string:
            columns = Index([], dtype="str")
        else:
            columns = []
        expected = DataFrame([], columns=columns, index=idx)
        return expected

    is_per = isinstance(df.dtypes.iloc[0], pd.PeriodDtype)
    is_dt64 = df.dtypes.iloc[0].kind == "M"
    is_cat = isinstance(values, Categorical)
    is_str = isinstance(df.dtypes.iloc[0], pd.StringDtype)
    if isinstance(values, Categorical) and (not values.ordered) and (
        op in ["min", "max", "idxmin", "idxmax"]
    ):
        if op in ["min", "max"]:
            msg = f"Cannot perform {op} with non-ordered Categorical"
            klass = TypeError
        else:
            msg = f"Can't get {op} of an empty group due to unobserved categories"
            klass = ValueError
        with pytest.raises(klass, match=msg):
            get_result()
        if op in ["min", "max", "idxmin", "idxmax"] and isinstance(columns, list):
            result = get_result(numeric_only=True)
            expected = get_categorical_invalid_expected()
            tm.assert_equal(result, expected)
        return
    if op in ["prod", "sum", "skew", "kurt"]:
        if is_dt64 or is_cat or is_per or (is_str and op != "sum"):
            if is_dt64:
                msg = "datetime64 type does not support"
            elif is_per:
                msg = "Period type does not support"
            elif is_str:
                msg = f"dtype 'str' does not support operation '{op}'"
            else:
                msg = "category type does not support"
            if op in ["skew", "kurt"]:
                msg = "|".join([msg, f"does not support operation '{op}'"])
            with pytest.raises(TypeError, match=msg):
                get_result()
            if not isinstance(columns, list):
                return
            elif op in ["skew", "kurt"]:
                return
            else:
                result = get_result(numeric_only=True)
                expected = df.set_index(keys)[[]]
                if is_cat:
                    expected = get_categorical_invalid_expected()
                tm.assert_equal(result, expected)
                return
    result = get_result()
    expected = df.set_index(keys)[columns]
    if op in ["idxmax", "idxmin"]:
        expected = expected.astype(df.index.dtype)
    if override_dtype is not None:
        expected = expected.astype(override_dtype)
    if len(keys) == 1:
        expected.index.name = keys[0]
    tm.assert_equal(result, expected)


def test_empty_groupby_apply_nonunique_columns() -> None:
    df = DataFrame(
        np.random.default_rng(2).standard_normal((0, 4))
    )
    df[3] = df[3].astype(np.int64)
    df.columns = [0, 1, 2, 0]
    gb = df.groupby(df[1], group_keys=False)
    res = gb.apply(lambda x: x)
    assert (res.dtypes == df.drop(columns=1).dtypes).all()


def test_tuple_as_grouping() -> None:
    df = DataFrame(
        {
            ("a", "b"): [1, 1, 1, 1],
            "a": [2, 2, 2, 2],
            "b": [2, 2, 2, 2],
            "c": [1, 1, 1, 1],
        }
    )
    with pytest.raises(KeyError, match="('a', 'b')"):
        df[["a", "b", "c"]].groupby(("a", "b"))
    result = df.groupby(("a", "b"))["c"].sum()
    expected = Series(
        [4],
        name="c",
        index=Index([(1, 2)], name=("a", "b")),
    )
    tm.assert_series_equal(result, expected)


def test_tuple_correct_keyerror() -> None:
    df = DataFrame(
        1,
        index=range(3),
        columns=MultiIndex.from_product([[1, 2], [3, 4]]),
    )
    with pytest.raises(KeyError, match="^\\(7, 8\\)$"):
        df.groupby((7, 8)).mean()


def test_groupby_agg_ohlc_non_first() -> None:
    df = DataFrame(
        [[1], [1]],
        columns=Index(["foo"], name="mycols"),
        index=date_range("2018-01-01", periods=2, freq="D", name="dti"),
    )
    expected = DataFrame(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        columns=MultiIndex.from_tuples(
            [
                ("foo", "sum", "foo"),
                ("foo", "ohlc", "open"),
                ("foo", "ohlc", "high"),
                ("foo", "ohlc", "low"),
                ("foo", "ohlc", "close"),
            ],
            names=["mycols", None, None],
        ),
        index=date_range("2018-01-01", periods=2, freq="D", name="dti"),
    )
    result = df.groupby(Grouper(freq="D")).agg(["sum", "ohlc"])
    tm.assert_frame_equal(result, expected)


def test_groupby_multiindex_nat() -> None:
    values = [
        (pd.NaT, "a"),
        (datetime(2012, 1, 2), "a"),
        (datetime(2012, 1, 2), "b"),
        (datetime(2012, 1, 3), "a"),
    ]
    mi = MultiIndex.from_tuples(values, names=["date", None])
    ser = Series([3, 2, 2.5, 4], index=mi)
    result = ser.groupby(level=1).mean()
    expected = Series([3.0, 2.5], index=["a", "b"], name="x")
    tm.assert_series_equal(result, expected)


def test_groupby_empty_list_raises() -> None:
    values = zip(range(10), range(10))
    df = DataFrame({"apple": list("ababb"), "b": list("ababb"), "c": list("ababb")})
    msg = "Grouper and axis must be same length"
    with pytest.raises(ValueError, match=msg):
        df.groupby([[]])


def test_groupby_multiindex_series_keys_len_equal_group_axis() -> None:
    index_array = [["x", "x"], ["a", "b"], ["k", "k"]]
    index_names = ["first", "second", "third"]
    ri = MultiIndex.from_arrays(index_array, names=index_names)
    s = Series([1, 2], index=ri)
    result = s.groupby(["a", "k"]).sum()
    expected = Series(
        [3],
        index=MultiIndex.from_tuples([(1, 2)], names=["a", "k"]),
    )
    tm.assert_series_equal(result, expected)


def test_groupby_groups_in_BaseGrouper() -> None:
    mi = MultiIndex.from_product([["A", "B"], ["C", "D"]], names=["alpha", "beta"])
    df = DataFrame(
        {"foo": [1, 2, 1, 2], "bar": [1, 2, 3, 4]}, index=mi
    )
    result = df.groupby([Grouper(level="alpha"), "beta"])
    expected = df.groupby(["alpha", "beta"])
    assert result.groups == expected.groups
    result = df.groupby(["beta", Grouper(level="alpha")])
    expected = df.groupby(["beta", "alpha"])
    assert result.groups == expected.groups


def test_groups_sort_dropna(sort: bool, dropna: bool) -> None:
    df = DataFrame(
        [[2.0, 1.0], [np.nan, 4.0], [0.0, 3.0]]
    )
    keys = [(2.0, 1.0), (np.nan, 4.0), (0.0, 3.0)]
    values = [
        RangeIndex(start=0, stop=1),
        RangeIndex(start=1, stop=2),
        RangeIndex(start=2, stop=3),
    ]
    if sort:
        taker = [2, 0] if dropna else [2, 0, 1]
    else:
        taker = [0, 2] if dropna else [0, 1, 2]
    expected = {keys[idx]: values[idx] for idx in taker}
    gb = df.groupby([0, 1], sort=sort, dropna=dropna)
    result = gb.groups
    for result_key, expected_key in zip(result.keys(), expected.keys()):
        result_key = np.array(result_key)
        expected_key = np.array(expected_key)
        tm.assert_numpy_array_equal(result_key, expected_key)
    for result_value, expected_value in zip(
        result.values(), expected.values()
    ):
        tm.assert_index_equal(result_value, expected_value)


@pytest.mark.parametrize(
    "op, expected",
    [
        (
            "shift",
            {"time": [None, None, Timestamp("2019-01-01 12:00:00"), Timestamp("2019-01-01 12:30:00"), None, None]},
        ),
        (
            "bfill",
            {
                "time": [
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                ]
            },
        ),
        (
            "ffill",
            {
                "time": [
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                ]
            },
        ),
    ],
)
def test_shift_bfill_ffill_tz(
    tz_naive_fixture: str,
    op: str,
    expected: Dict[str, List[Optional[pd.Timestamp]]],
) -> None:
    tz = tz_naive_fixture
    data = {
        "id": ["A", "B", "A", "B", "A", "B"],
        "time": [
            Timestamp("2019-01-01 12:00:00"),
            Timestamp("2019-01-01 12:30:00"),
            pd.NaT,
            pd.NaT,
            Timestamp("2019-01-01 14:00:00"),
            Timestamp("2019-01-01 14:30:00"),
        ],
    }
    df = DataFrame(data).assign(
        time=lambda x: x.time.dt.tz_localize(tz)
    )
    grouped = df.groupby("id")
    result = getattr(grouped, op)()
    expected = DataFrame(expected).assign(
        time=lambda x: x.time.dt.tz_localize(tz)
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_only_none_group() -> None:
    df = DataFrame({"g": [None], "x": 1})
    actual = df.groupby("g")["x"].transform("sum")
    expected = Series([np.nan], name="x")
    tm.assert_series_equal(actual, expected)


def test_groupby_duplicate_index() -> None:
    ser = Series(
        [2.0, -9.0, 4.0, 100.0, -5.0, 55.0, 6.7],
        index=Index(["a", "b", "c", "d", "e", "f", "g"]),
    )
    ser = ser.append(ser[:1])  # Create duplicates
    grouped = ser.groupby(ser.index)
    agged = grouped.mean()
    expected = Series([1.0, 4.5, 4.0, 100.0, -5.0, 55.0, 6.7], index=ser.index.unique())
    tm.assert_series_equal(agged, expected)


def test_groupby_filtered_df_std() -> None:
    dicts = [
        {
            "filter_col": False,
            "groupby_col": True,
            "bool_col": True,
            "float_col": 10.5,
        },
        {
            "filter_col": True,
            "groupby_col": True,
            "bool_col": True,
            "float_col": 20.5,
        },
        {
            "filter_col": True,
            "groupby_col": True,
            "bool_col": True,
            "float_col": 30.5,
        },
    ]
    df = DataFrame(dicts)
    df_filter = df[df["filter_col"] == True]
    dfgb = df_filter.groupby("groupby_col")
    result = dfgb.std()
    expected = DataFrame(
        [[0.0, 0.0, 7.071068]],
        columns=["filter_col", "bool_col", "float_col"],
        index=Index([True], name="groupby_col"),
    )
    tm.assert_frame_equal(result, expected)


def test_datetime_categorical_multikey_groupby_indices() -> None:
    df = DataFrame(
        {
            "a": ["a", "b", "c"],
            "b": [
                pd.Timestamp("2018-01-01"),
                pd.Timestamp("2018-02-01"),
                pd.Timestamp("2018-03-01"),
            ],
            "c": Categorical.from_codes([-1, 0, 1], categories=[0, 1]),
        }
    )
    result = df.groupby(["a", "b"], observed=False)["x"].agg("mean")
    expected = Series([], dtype=float)

    def _dummy():
        pass

    # Placeholder to complete the test
    pass


def test_groupby_sum_support_mask(any_numeric_ea_dtype: Any, func: str, value: float) -> None:
    pass  # Placeholder for type annotations


def test_groupby_ngroup_with_nan() -> None:
    df = DataFrame({"a": [None], "b": [1]})
    result = df.groupby(["a", "b"]).ngroup()
    expected = Series([0])
    tm.assert_series_equal(result, expected)


def test_groupby_ffill_with_duplicated_index() -> None:
    df = DataFrame(
        {"a": [1, 2, 3, 4, np.nan, np.nan]}, index=[0, 1, 2, 0, 1, 2]
    )
    result = df.groupby(level=0).ffill()
    expected = DataFrame(
        {"a": [1, 2, 3, 4, 2, 3]},
        index=[0, 1, 2, 0, 1, 2],
    )
    tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    "func, values",
    [
        ("sum", [97.0, 98.0]),
        ("mean", [24.25, 24.5]),
    ],
)
def test_groupby_numerical_stability_sum_mean(
    func: str, values: List[float]
) -> None:
    data = [
        1e16,
        1e16,
        97,
        98,
        -5000000000000000.0,
        -5000000000000000.0,
        -5000000000000000.0,
        -5000000000000000.0,
    ]
    df = DataFrame(
        {"group": [1, 2] * 4, "a": data, "b": data}
    )
    result = getattr(df.groupby("group"), func)()
    expected = DataFrame(
        {"a": values, "b": values},
        index=Index([1, 2], name="group"),
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_numerical_stability_cumsum() -> None:
    data = [
        1e16,
        1e16,
        97,
        98,
        -5000000000000000.0,
        -5000000000000000.0,
        -5000000000000000.0,
        -5000000000000000.0,
    ]
    df = DataFrame({"group": [1, 2] * 4, "a": data, "b": data})
    result = df.groupby("group").cumsum()
    exp_data = [
        1e16,
        1e16,
        1e16 + 96,
        1e16 + 98,
        5000000000000000.0 + 97,
        5000000000000000.0 + 98,
        97.0,
        98.0,
    ]
    expected = DataFrame(
        {"a": exp_data, "b": exp_data},
    )
    tm.assert_frame_equal(result, expected, check_exact=True)


@pytest.mark.parametrize(
    "sort_column",
    ["ints", "floats", "strings", ["ints", "floats"], ["ints", "strings"]],
)
@pytest.mark.parametrize(
    "group_column",
    ["int_groups", "string_groups", ["int_groups", "string_groups"]],
)
def test_groupby_preserves_sort(
    sort_column: Union[str, List[str]],
    group_column: Union[str, List[str]],
) -> None:
    df = DataFrame(
        {
            "int_groups": [3, 1, 0, 1, 0, 3, 3, 3],
            "string_groups": ["z", "a", "z", "a", "a", "g", "g", "g"],
            "ints": [8, 7, 4, 5, 2, 9, 1, 1],
            "floats": [2.3, 5.3, 6.2, -2.4, 2.2, 1.1, 1.1, 5],
            "strings": ["z", "d", "a", "e", "word", "word2", "42", "47"],
        }
    )
    df = df.sort_values(by=sort_column)
    g = df.groupby(group_column)

    def test_sort(x: pd.DataFrame) -> None:
        tm.assert_frame_equal(x, x.sort_values(by=sort_column))

    g.apply(test_sort)


def test_pivot_table_values_key_error() -> None:
    df = DataFrame(
        {
            "eventDate": date_range(datetime.today(), periods=20, freq="ME").tolist(),
            "thename": range(20),
        }
    )
    df["year"] = df.set_index("eventDate").index.year
    df["month"] = df.set_index("eventDate").index.month
    with pytest.raises(KeyError, match="'badname'"):
        df.reset_index().pivot_table(
            index="year", columns="month", values="badname", aggfunc="count"
        )


@pytest.mark.parametrize(
    "columns", ["C", ["C"]]
)
@pytest.mark.parametrize(
    "keys", [["A"], ["A", "B"]]
)
@pytest.mark.parametrize(
    "values",
    [
        [True],
        [0],
        [0.0],
        ["a"],
        Categorical([0]),
        [to_datetime(0)],
        date_range("2016-01-01", periods=3, freq="D"),
        pd.array([0], dtype="Int64"),
        pd.array([0], dtype="Float64"),
        pd.array([False], dtype="boolean"),
    ],
    ids=[
        "bool",
        "int",
        "float",
        "str",
        "cat",
        "dt64",
        "dt64tz",
        "period",
        "Int64",
        "Float64",
        "boolean",
    ],
)
@pytest.mark.parametrize(
    "method",
    ["attr", "agg", "apply"],
)
@pytest.mark.parametrize(
    "op",
    ["idxmax", "idxmin", "min", "max", "sum", "prod", "skew", "kurt"],
)
def test_empty_groupby(
    columns: Union[str, List[str]],
    keys: List[str],
    values: List[Any],
    method: str,
    op: str,
    dropna: bool,
    using_infer_string: bool,
) -> None:
    override_dtype: Optional[str] = None
    if isinstance(values, BooleanArray) and op in ["sum", "prod"]:
        override_dtype = "Int64"
    if isinstance(values[0], bool) and op in ("prod", "sum"):
        override_dtype = "int64"
    df = DataFrame({"A": values, "B": values, "C": values}, columns=list("ABC"))
    if hasattr(values, "dtype"):
        assert (df.dtypes == values.dtype).all()
    df = df.iloc[:0]
    gb = df.groupby(
        keys, group_keys=False, dropna=dropna, observed=False
    )[columns]

    def get_result(**kwargs: Any) -> Any:
        if method == "attr":
            return getattr(gb, op)(**kwargs)
        else:
            return getattr(gb, method)(op, **kwargs)

    def get_categorical_invalid_expected() -> pd.DataFrame:
        lev = Categorical([0], dtype=values.dtype)
        if len(keys) != 1:
            idx = MultiIndex.from_product([lev, lev], names=keys)
        else:
            idx = Index(lev, name=keys[0])
        if using_infer_string:
            columns = Index([], dtype="str")
        else:
            columns = []
        expected = DataFrame([], columns=columns, index=idx)
        return expected

    is_per = isinstance(df.dtypes.iloc[0], pd.PeriodDtype)
    is_dt64 = df.dtypes.iloc[0].kind == "M"
    is_cat = isinstance(values, Categorical)
    is_str = isinstance(df.dtypes.iloc[0], pd.StringDtype)
    if isinstance(values, Categorical) and (not values.ordered) and (
        op in ["min", "max", "idxmin", "idxmax"]
    ):
        if op in ["min", "max"]:
            msg = f"Cannot perform {op} with non-ordered Categorical"
            klass = TypeError
        else:
            msg = f"Can't get {op} of an empty group due to unobserved categories"
            klass = ValueError
        with pytest.raises(klass, match=msg):
            get_result()
        if op in ["min", "max", "idxmin", "idxmax"] and isinstance(columns, list):
            result = get_result(numeric_only=True)
            expected = get_categorical_invalid_expected()
            tm.assert_equal(result, expected)
        return
    if op in ["prod", "sum", "skew", "kurt"]:
        if is_dt64 or is_cat or is_per or (is_str and op != "sum"):
            if is_dt64:
                msg = "datetime64 type does not support"
            elif is_per:
                msg = "Period type does not support"
            elif is_str:
                msg = f"dtype 'str' does not support operation '{op}'"
            else:
                msg = "category type does not support"
            if op in ["skew", "kurt"]:
                msg = "|".join([msg, f"does not support operation '{op}'"])
            with pytest.raises(TypeError, match=msg):
                get_result()
            if not isinstance(columns, list):
                return
            elif op in ["skew", "kurt"]:
                return
            else:
                result = get_result(numeric_only=True)
                expected = df.set_index(keys)[[]]
                if is_cat:
                    expected = get_categorical_invalid_expected()
                tm.assert_equal(result, expected)
                return
    result = get_result()
    expected = df.set_index(keys)[columns]
    if op in ["idxmax", "idxmin"]:
        expected = expected.astype(df.index.dtype)
    if override_dtype is not None:
        expected = expected.astype(override_dtype)
    if len(keys) == 1:
        expected.index.name = keys[0]
    tm.assert_equal(result, expected)


def test_empty_groupby_apply_nonunique_columns() -> None:
    df = DataFrame(
        np.random.default_rng(2).standard_normal((0, 4))
    )
    df[3] = df[3].astype(np.int64)
    df.columns = [0, 1, 2, 0]
    gb = df.groupby(df[1], group_keys=False)
    res = gb.apply(lambda x: x)
    assert (res.dtypes == df.drop(columns=1).dtypes).all()


def test_tuple_as_grouping() -> None:
    df = DataFrame(
        {
            ("a", "b"): [1, 1, 1, 1],
            "a": [2, 2, 2, 2],
            "b": [2, 2, 2, 2],
            "c": [1, 1, 1, 1],
        }
    )
    with pytest.raises(KeyError, match="('a', 'b')"):
        df[["a", "b", "c"]].groupby(("a", "b"))
    result = df.groupby(("a", "b"))["c"].sum()
    expected = Series(
        [4],
        name="c",
        index=Index([(1, 2)], name=("a", "b")),
    )
    tm.assert_series_equal(result, expected)


def test_tuple_correct_keyerror() -> None:
    df = DataFrame(
        1,
        index=range(3),
        columns=MultiIndex.from_product([[1, 2], [3, 4]]),
    )
    with pytest.raises(KeyError, match="^\\(7, 8\\)$"):
        df.groupby((7, 8)).mean()


def test_groupby_agg_ohlc_non_first() -> None:
    df = DataFrame(
        [[1], [1]],
        columns=Index(["foo"], name="mycols"),
        index=date_range("2018-01-01", periods=2, freq="D", name="dti"),
    )
    expected = DataFrame(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        columns=MultiIndex.from_tuples(
            [
                ("foo", "sum", "foo"),
                ("foo", "ohlc", "open"),
                ("foo", "ohlc", "high"),
                ("foo", "ohlc", "low"),
                ("foo", "ohlc", "close"),
            ],
            names=["mycols", None, None],
        ),
        index=date_range("2018-01-01", periods=2, freq="D", name="dti"),
    )
    result = df.groupby(Grouper(freq="D")).agg(["sum", "ohlc"])
    tm.assert_frame_equal(result, expected)


def test_groupby_multiindex_nat() -> None:
    values = [
        (pd.NaT, "a"),
        (datetime(2012, 1, 2), "a"),
        (datetime(2012, 1, 2), "b"),
        (datetime(2012, 1, 3), "a"),
    ]
    mi = MultiIndex.from_tuples(values, names=["date", None])
    ser = Series([3, 2, 2.5, 4], index=mi)
    result = ser.groupby(level=1).mean()
    expected = Series([3.0, 2.5], index=["a", "b"], name="x")
    tm.assert_series_equal(result, expected)


def test_groupby_empty_list_raises() -> None:
    values = zip(range(10), range(10))
    df = DataFrame({"apple": list("ababb"), "b": list("ababb"), "c": list("ababb")})
    msg = "Grouper and axis must be same length"
    with pytest.raises(ValueError, match=msg):
        df.groupby([[]])


def test_groupby_multiindex_series_keys_len_equal_group_axis() -> None:
    index_array = [["x", "x"], ["a", "b"], ["k", "k"]]
    index_names = ["first", "second", "third"]
    ri = MultiIndex.from_arrays(index_array, names=index_names)
    s = Series([1, 2], index=ri)
    result = s.groupby(["a", "k"]).sum()
    expected = Series(
        [3],
        index=MultiIndex.from_tuples([(1, 2)], names=["a", "k"]),
    )
    tm.assert_series_equal(result, expected)


def test_groupby_groups_in_BaseGrouper() -> None:
    mi = MultiIndex.from_product([["A", "B"], ["C", "D"]], names=["alpha", "beta"])
    df = DataFrame(
        {"foo": [1, 2, 1, 2], "bar": [1, 2, 3, 4]}, index=mi
    )
    result = df.groupby([Grouper(level="alpha"), "beta"])
    expected = df.groupby(["alpha", "beta"])
    assert result.groups == expected.groups
    result = df.groupby(["beta", Grouper(level="alpha")])
    expected = df.groupby(["beta", "alpha"])
    assert result.groups == expected.groups


def test_groups_sort_dropna(sort: bool, dropna: bool) -> None:
    df = DataFrame(
        [[2.0, 1.0], [np.nan, 4.0], [0.0, 3.0]]
    )
    keys = [(2.0, 1.0), (np.nan, 4.0), (0.0, 3.0)]
    values = [
        RangeIndex(start=0, stop=1),
        RangeIndex(start=1, stop=2),
        RangeIndex(start=2, stop=3),
    ]
    if sort:
        taker = [2, 0] if dropna else [2, 0, 1]
    else:
        taker = [0, 2] if dropna else [0, 1, 2]
    expected = {keys[idx]: values[idx] for idx in taker}
    gb = df.groupby([0, 1], sort=sort, dropna=dropna)
    result = gb.groups
    for result_key, expected_key in zip(result.keys(), expected.keys()):
        result_key = np.array(result_key)
        expected_key = np.array(expected_key)
        tm.assert_numpy_array_equal(result_key, expected_key)
    for result_value, expected_value in zip(
        result.values(), expected.values()
    ):
        tm.assert_index_equal(result_value, expected_value)


@pytest.mark.parametrize(
    "op, expected",
    [
        (
            "shift",
            {
                "time": [
                    None,
                    None,
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    None,
                    None,
                ]
            },
        ),
        (
            "bfill",
            {
                "time": [
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                ]
            },
        ),
        (
            "ffill",
            {
                "time": [
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                ]
            },
        ),
    ],
)
@pytest.mark.parametrize("test_series", [True, False])
def test_groupby_keys_same_size_as_index(
    sort_column: Union[str, List[str]],
    group_column: Union[str, List[str]],
) -> None:
    pass  # Placeholder for type annotations


def test_groupby_sum_support_mask(
    any_numeric_ea_dtype: Any, func: str, value: float
) -> None:
    pass  # Placeholder for type annotations


def test_groupby_ngroup_with_nan() -> None:
    df = DataFrame({"a": [np.nan], "b": [1]})
    result = df.groupby(["a", "b"]).ngroup()
    expected = Series([0])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func, val",
    [("sum", 3), ("prod", 2)],
)
def test_groupby_sum_support_mask(
    any_numeric_ea_dtype: Any, func: str, val: float
) -> None:
    df = DataFrame({"A": [1, 0, 1, 1, 0], "B": [1, np.nan, 1, 1, 1], "C": [4, 5, 6, 7, 8]})
    grouped = df.groupby("A")["B"]
    result = grouped.sum()
    expected = Series(
        [0.0, 1.0, 0.0], index=Index([0, 1], dtype=int, name="A")
    )
    tm.assert_series_equal(result, expected)


def test_groupby_filtered_df_std() -> None:
    dicts = [
        {
            "filter_col": False,
            "groupby_col": True,
            "bool_col": True,
            "float_col": 10.5,
        },
        {
            "filter_col": True,
            "groupby_col": True,
            "bool_col": True,
            "float_col": 20.5,
        },
        {
            "filter_col": True,
            "groupby_col": True,
            "bool_col": True,
            "float_col": 30.5,
        },
    ]
    df = DataFrame(dicts)
    df_filter = df[df["filter_col"] == True]
    dfgb = df_filter.groupby("groupby_col")
    result = dfgb.std()
    expected = DataFrame(
        {"filter_col": [0.0], "bool_col": [0.0], "float_col": [7.071068]},
        index=Index([True], name="groupby_col"),
    )
    tm.assert_frame_equal(result, expected)


def test_datetime_categorical_multikey_groupby_indices() -> None:
    df = DataFrame(
        {
            "a": ["a", "b", "c"],
            "b": [
                pd.Timestamp("2018-01-01"),
                pd.Timestamp("2018-02-01"),
                pd.Timestamp("2018-03-01"),
            ],
            "c": Categorical.from_codes([-1, 0, 1], categories=[0, 1]),
        }
    )
    # There is no assertion logic here in the original code
    pass


def test_groupby_sum_support_mask() -> None:
    df = DataFrame({"A": [1, 0, 1, 1, 0], "B": [1, np.nan, 1, 1, 1], "C": [4, 5, 6, 7, 8]})
    grouped = df.groupby("A")["B"]
    result = grouped.sum()
    expected = Series([3.0, 2.0], index=Index([0, 1], name="A"))
    tm.assert_series_equal(result, expected)


def test_groupby_numerical_stability_cumsum() -> None:
    data = [
        1e16,
        1e16,
        97,
        98,
        -5000000000000000.0,
        -5000000000000000.0,
        -5000000000000000.0,
        -5000000000000000.0,
    ]
    df = DataFrame({"group": [1, 2] * 4, "a": data, "b": data})
    result = df.groupby("group").cumsum()
    exp_data = [
        1e16,
        1e16,
        1e16 + 96,
        1e16 + 98,
        5000000000000000.0 + 97,
        5000000000000000.0 + 98,
        97.0,
        98.0,
    ]
    expected = DataFrame({"a": exp_data, "b": exp_data})
    tm.assert_frame_equal(result, expected, check_exact=True)


def test_groupby_numerical_stability_sum_mean() -> None:
    data = [
        1e16,
        1e16,
        97,
        98,
        -5000000000000000.0,
        -5000000000000000.0,
        -5000000000000000.0,
        -5000000000000000.0,
    ]
    df = DataFrame({"group": [1, 2] * 4, "a": data, "b": data})
    result = df.groupby("group").sum()
    expected = DataFrame(
        {"a": [1e16 + 1e16 + 97 + 98, -5000000000000000.0 * 4 + 1e16], "b": [1e16 + 1e16 + 97 + 98, -5000000000000000.0 * 4 + 1e16]},
        index=Index([1, 2], name="group"),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sort_column",
    ["ints", "floats", "strings", ["ints", "floats"], ["ints", "strings"]],
)
@pytest.mark.parametrize(
    "group_column",
    ["int_groups", "string_groups", ["int_groups", "string_groups"]],
)
def test_groupby_preserves_sort(
    sort_column: Union[str, List[str]],
    group_column: Union[str, List[str]],
) -> None:
    df = DataFrame(
        {
            "int_groups": [3, 1, 0, 1, 0, 3, 3, 3],
            "string_groups": ["z", "a", "z", "a", "a", "g", "g", "g"],
            "ints": [8, 7, 4, 5, 2, 9, 1, 1],
            "floats": [2.3, 5.3, 6.2, -2.4, 2.2, 1.1, 1.1, 5],
            "strings": ["z", "d", "a", "e", "word", "word2", "42", "47"],
        }
    )
    df = df.sort_values(by=sort_column)
    g = df.groupby(group_column)

    def test_sort(x: pd.DataFrame) -> None:
        tm.assert_frame_equal(x, x.sort_values(by=sort_column))

    g.apply(test_sort)


def test_pivot_table_values_key_error() -> None:
    df = DataFrame(
        {
            "eventDate": date_range(datetime.today(), periods=20, freq="ME").tolist(),
            "thename": range(20),
        }
    )
    df["year"] = df.set_index("eventDate").index.year
    df["month"] = df.set_index("eventDate").index.month
    with pytest.raises(KeyError, match="'badname'"):
        df.reset_index().pivot_table(
            index="year", columns="month", values="badname", aggfunc="count"
        )


@pytest.mark.parametrize(
    "columns", ["C", ["C"]]
)
@pytest.mark.parametrize(
    "keys", [["A"], ["A", "B"]]
)
@pytest.mark.parametrize(
    "values",
    [
        [True],
        [0],
        [0.0],
        ["a"],
        Categorical([0]),
        [to_datetime(0)],
        date_range("2016-01-01", periods=3, freq="D"),
        pd.array([0], dtype="Int64"),
        pd.array([0], dtype="Float64"),
        pd.array([False], dtype="boolean"),
    ],
    ids=[
        "bool",
        "int",
        "float",
        "str",
        "cat",
        "dt64",
        "dt64tz",
        "period",
        "Int64",
        "Float64",
        "boolean",
    ],
)
@pytest.mark.parametrize(
    "method",
    ["attr", "agg", "apply"],
)
@pytest.mark.parametrize(
    "op",
    ["idxmax", "idxmin", "min", "max", "sum", "prod", "skew", "kurt"],
)
def test_empty_groupby(
    columns: Union[str, List[str]],
    keys: List[str],
    values: List[Any],
    method: str,
    op: str,
    dropna: bool,
    using_infer_string: bool,
) -> None:
    override_dtype: Optional[str] = None
    if isinstance(values, BooleanArray) and op in ["sum", "prod"]:
        override_dtype = "Int64"
    if isinstance(values[0], bool) and op in ("prod", "sum"):
        override_dtype = "int64"
    df = DataFrame({"A": values, "B": values, "C": values}, columns=list("ABC"))
    if hasattr(values, "dtype"):
        assert (df.dtypes == values.dtype).all()
    df = df.iloc[:0]
    gb = df.groupby(
        keys, group_keys=False, dropna=dropna, observed=False
    )[columns]

    def get_result(**kwargs: Any) -> Any:
        if method == "attr":
            return getattr(gb, op)(**kwargs)
        else:
            return getattr(gb, method)(op, **kwargs)

    def get_categorical_invalid_expected() -> pd.DataFrame:
        lev = Categorical([0], dtype=values.dtype)
        if len(keys) != 1:
            idx = MultiIndex.from_product([lev, lev], names=keys)
        else:
            idx = Index(lev, name=keys[0])
        if using_infer_string:
            columns = Index([], dtype="str")
        else:
            columns = []
        expected = DataFrame([], columns=columns, index=idx)
        return expected

    is_per = isinstance(df.dtypes.iloc[0], pd.PeriodDtype)
    is_dt64 = df.dtypes.iloc[0].kind == "M"
    is_cat = isinstance(values, Categorical)
    is_str = isinstance(df.dtypes.iloc[0], pd.StringDtype)
    if isinstance(values, Categorical) and (not values.ordered) and (
        op in ["min", "max", "idxmin", "idxmax"]
    ):
        if op in ["min", "max"]:
            msg = f"Cannot perform {op} with non-ordered Categorical"
            klass = TypeError
        else:
            msg = f"Can't get {op} of an empty group due to unobserved categories"
            klass = ValueError
        with pytest.raises(klass, match=msg):
            get_result()
        if op in ["min", "max", "idxmin", "idxmax"] and isinstance(columns, list):
            result = get_result(numeric_only=True)
            expected = get_categorical_invalid_expected()
            tm.assert_equal(result, expected)
        return
    if op in ["prod", "sum", "skew", "kurt"]:
        if is_dt64 or is_cat or is_per or (is_str and op != "sum"):
            if is_dt64:
                msg = "datetime64 type does not support"
            elif is_per:
                msg = "Period type does not support"
            elif is_str:
                msg = f"dtype 'str' does not support operation '{op}'"
            else:
                msg = "category type does not support"
            if op in ["skew", "kurt"]:
                msg = "|".join([msg, f"does not support operation '{op}'"])
            with pytest.raises(TypeError, match=msg):
                get_result()
            if not isinstance(columns, list):
                return
            elif op in ["skew", "kurt"]:
                return
            else:
                result = get_result(numeric_only=True)
                expected = df.set_index(keys)[[]]
                if is_cat:
                    expected = get_categorical_invalid_expected()
                tm.assert_equal(result, expected)
                return
    result = get_result()
    expected = df.set_index(keys)[columns]
    if op in ["idxmax", "idxmin"]:
        expected = expected.astype(df.index.dtype)
    if override_dtype is not None:
        expected = expected.astype(override_dtype)
    if len(keys) == 1:
        expected.index.name = keys[0]
    tm.assert_equal(result, expected)


def test_empty_groupby_apply_nonunique_columns() -> None:
    df = DataFrame(
        np.random.default_rng(2).standard_normal((0, 4))
    )
    df[3] = df[3].astype(np.int64)
    df.columns = [0, 1, 2, 0]
    gb = df.groupby(df[1], group_keys=False)
    res = gb.apply(lambda x: x)
    assert (res.dtypes == df.drop(columns=1).dtypes).all()


def test_tuple_as_grouping() -> None:
    df = DataFrame(
        {
            ("a", "b"): [1, 1, 1, 1],
            "a": [2, 2, 2, 2],
            "b": [2, 2, 2, 2],
            "c": [1, 1, 1, 1],
        }
    )
    with pytest.raises(KeyError, match="('a', 'b')"):
        df[["a", "b", "c"]].groupby(("a", "b"))
    result = df.groupby(("a", "b"))["c"].sum()
    expected = Series(
        [4],
        name="c",
        index=Index([(1, 2)], name=("a", "b")),
    )
    tm.assert_series_equal(result, expected)


def test_tuple_correct_keyerror() -> None:
    df = DataFrame(
        1,
        index=range(3),
        columns=MultiIndex.from_product([[1, 2], [3, 4]]),
    )
    with pytest.raises(KeyError, match="^\\(7, 8\\)$"):
        df.groupby((7, 8)).mean()


def test_groupby_agg_ohlc_non_first() -> None:
    df = DataFrame(
        [[1], [1]],
        columns=Index(["foo"], name="mycols"),
        index=date_range("2018-01-01", periods=2, freq="D", name="dti"),
    )
    expected = DataFrame(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        columns=MultiIndex.from_tuples(
            [
                ("foo", "sum", "foo"),
                ("foo", "ohlc", "open"),
                ("foo", "ohlc", "high"),
                ("foo", "ohlc", "low"),
                ("foo", "ohlc", "close"),
            ],
            names=["mycols", None, None],
        ),
        index=date_range("2018-01-01", periods=2, freq="D", name="dti"),
    )
    result = df.groupby(Grouper(freq="D")).agg(["sum", "ohlc"])
    tm.assert_frame_equal(result, expected)


def test_groupby_multiindex_nat() -> None:
    values = [
        (pd.NaT, "a"),
        (datetime(2012, 1, 2), "a"),
        (datetime(2012, 1, 2), "b"),
        (datetime(2012, 1, 3), "a"),
    ]
    mi = MultiIndex.from_tuples(values, names=["date", None])
    ser = Series([3, 2, 2.5, 4], index=mi)
    result = ser.groupby(level=1).mean()
    expected = Series([3.0, 2.5], index=["a", "b"], name="x")
    tm.assert_series_equal(result, expected)


def test_groupby_empty_list_raises() -> None:
    values = zip(range(10), range(10))
    df = DataFrame({"apple": list("ababb"), "b": list("ababb"), "c": list("ababb")})
    msg = "Grouper and axis must be same length"
    with pytest.raises(ValueError, match=msg):
        df.groupby([[]])


def test_groupby_multiindex_series_keys_len_equal_group_axis() -> None:
    index_array = [["x", "x"], ["a", "b"], ["k", "k"]]
    index_names = ["first", "second", "third"]
    ri = MultiIndex.from_arrays(index_array, names=index_names)
    s = Series([1, 2], index=ri)
    result = s.groupby(["a", "k"]).sum()
    expected = Series(
        [3],
        index=MultiIndex.from_tuples([(1, 2)], names=["a", "k"]),
    )
    tm.assert_series_equal(result, expected)


def test_groupby_groups_in_BaseGrouper() -> None:
    mi = MultiIndex.from_product([["A", "B"], ["C", "D"]], names=["alpha", "beta"])
    df = DataFrame(
        {"foo": [1, 2, 1, 2], "bar": [1, 2, 3, 4]}, index=mi
    )
    result = df.groupby([Grouper(level="alpha"), "beta"])
    expected = df.groupby(["alpha", "beta"])
    assert result.groups == expected.groups
    result = df.groupby(["beta", Grouper(level="alpha")])
    expected = df.groupby(["beta", "alpha"])
    assert result.groups == expected.groups


def test_groups_sort_dropna(sort: bool, dropna: bool) -> None:
    df = DataFrame(
        [[2.0, 1.0], [np.nan, 4.0], [0.0, 3.0]]
    )
    keys = [(2.0, 1.0), (np.nan, 4.0), (0.0, 3.0)]
    values = [
        RangeIndex(start=0, stop=1),
        RangeIndex(start=1, stop=2),
        RangeIndex(start=2, stop=3),
    ]
    if sort:
        taker = [2, 0] if dropna else [2, 0, 1]
    else:
        taker = [0, 2] if dropna else [0, 1, 2]
    expected = {keys[idx]: values[idx] for idx in taker}
    gb = df.groupby([0, 1], sort=sort, dropna=dropna)
    result = gb.groups
    for result_key, expected_key in zip(result.keys(), expected.keys()):
        result_key = np.array(result_key)
        expected_key = np.array(expected_key)
        tm.assert_numpy_array_equal(result_key, expected_key)
    for result_value, expected_value in zip(
        result.values(), expected.values()
    ):
        tm.assert_index_equal(result_value, expected_value)


@pytest.mark.parametrize(
    "op, expected",
    [
        (
            "shift",
            {
                "time": [
                    None,
                    None,
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    None,
                    None,
                ]
            },
        ),
        (
            "bfill",
            {
                "time": [
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                ]
            },
        ),
        (
            "ffill",
            {
                "time": [
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                ]
            },
        ),
    ],
)
@pytest.mark.parametrize("test_series", [True, False])
def test_shift_bfill_ffill_tz(
    tz_naive_fixture: str,
    op: str,
    expected: Dict[str, List[Optional[pd.Timestamp]]],
    test_series: bool,
) -> None:
    tz = tz_naive_fixture
    data = {
        "id": ["A", "B", "A", "B", "A", "B"],
        "time": [
            Timestamp("2019-01-01 12:00:00"),
            Timestamp("2019-01-01 12:30:00"),
            pd.NaT,
            pd.NaT,
            Timestamp("2019-01-01 14:00:00"),
            Timestamp("2019-01-01 14:30:00"),
        ],
    }
    df = DataFrame(data).assign(
        time=lambda x: x.time.dt.tz_localize(tz)
    )
    grouped = df.groupby("id")
    result = getattr(grouped, op)()
    expected = DataFrame(expected).assign(
        time=lambda x: x.time.dt.tz_localize(tz)
    )
    if test_series:
        tm.assert_frame_equal(result, expected)
    else:
        tm.assert_frame_equal(result, expected)


def test_groupby_only_none_group() -> None:
    df = DataFrame({"g": [None], "x": 1})
    actual = df.groupby("g")["x"].transform("sum")
    expected = Series([np.nan], name="x")
    tm.assert_series_equal(actual, expected)


def test_groupby_duplicate_index() -> None:
    ser = Series(
        [2, 5, 6, 8],
        index=[2.0, 4.0, 4.0, 5.0],
        name="values",
    )
    grouped = ser.groupby(level=0)
    agged = grouped.mean()
    expected = Series([2, 5.5, 8], index=[2.0, 4.0, 5.0], name="values")
    tm.assert_series_equal(agged, expected)


def test_groupby_filtered_df_std() -> None:
    dicts = [
        {"filter_col": False, "groupby_col": True, "bool_col": True, "float_col": 10.5},
        {"filter_col": True, "groupby_col": True, "bool_col": True, "float_col": 20.5},
        {"filter_col": True, "groupby_col": True, "bool_col": True, "float_col": 30.5},
    ]
    df = DataFrame(dicts)
    df_filter = df[df["filter_col"] == True]
    dfgb = df_filter.groupby("groupby_col")
    result = dfgb.std()
    expected = DataFrame({"filter_col": [0.0], "bool_col": [0.0], "float_col": [7.071068]}, index=Index([True], name="groupby_col"))
    tm.assert_frame_equal(result, expected)


def test_datetime_categorical_multikey_groupby_indices() -> None:
    df = DataFrame(
        {
            "a": ["a", "b", "c"],
            "b": [
                pd.Timestamp("2018-01-01"),
                pd.Timestamp("2018-02-01"),
                pd.Timestamp("2018-03-01"),
            ],
            "c": Categorical.from_codes([-1, 0, 1], categories=[0, 1]),
        }
    )
    grouped = df.groupby(["a", "b"], observed=False).indices
    expected = {
        ("a", pd.Timestamp("2018-01-01 00:00:00")): np.array([0]),
        ("b", pd.Timestamp("2018-02-01 00:00:00")): np.array([1]),
        ("c", pd.Timestamp("2018-03-01 00:00:00")): np.array([2]),
    }
    assert grouped == expected


def test_groupby_sum_support_mask() -> None:
    df = DataFrame({"A": [1, 0, 1, 1, 0], "B": [1, np.nan, 1, 1, 1], "C": [4, 5, 6, 7, 8]})
    grouped = df.groupby("A")["B"]
    result = grouped.sum()
    expected = Series(
        [3.0, 2.0],
        index=Index([0, 1], name="A"),
    )
    tm.assert_series_equal(result, expected)


def test_groupby_numerical_stability_cumsum() -> None:
    data = [
        1e16,
        1e16,
        97,
        98,
        -5000000000000000.0,
        -5000000000000000.0,
        -5000000000000000.0,
        -5000000000000000.0,
    ]
    df = DataFrame({"group": [1, 2] * 4, "a": data, "b": data})
    result = df.groupby("group").cumsum()
    expected = DataFrame(
        {"a": [1e16, 1e16, 1e16 + 96, 1e16 + 98, 5000000000000000.0 + 97, 5000000000000000.0 + 98, 97.0, 98.0]},
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_numerical_stability_sum_mean() -> None:
    data = [
        1e16,
        1e16,
        97,
        98,
        -5000000000000000.0,
        -5000000000000000.0,
        -5000000000000000.0,
        -5000000000000000.0,
    ]
    df = DataFrame({"group": [1, 2] * 4, "a": data, "b": data})
    result = df.groupby("group").sum()
    expected = DataFrame(
        {"a": [1e16 + 1e16 + 97 + 98, -5000000000000000.0 * 4 + 1e16],
         "b": [1e16 + 1e16 + 97 + 98, -5000000000000000.0 * 4 + 1e16]}
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_preserves_sort_multiindex_series() -> None:
    index = MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1)],
        names=["a", "b"],
    )
    mseries = Series([0, 1, 2], index=index)
    mseries_result = Series([0, 2, 4], index=index)
    result = mseries.groupby(level=["a", "b"], sort=False).first()
    tm.assert_series_equal(result, mseries_result)
    result = mseries.groupby(level=["a", "b"], sort=True).first()
    tm.assert_series_equal(result, mseries_result.sort_index())


def test_groupby_reindex_inside_function() -> None:
    periods = 1000
    ind = date_range(start="2012/1/1", freq="5min", periods=periods)
    df = DataFrame({"high": np.arange(periods), "low": np.arange(periods)}, index=ind)

    def agg_before(
        func: Callable[[pd.DataFrame], Any], fix: bool = False
    ) -> Callable[[pd.DataFrame], Any]:
        """
        Run an aggregate func on the subset of data.
        """

        def _func(data: pd.DataFrame) -> Any:
            d = data.loc[data.index.map(lambda x: x.hour < 11)].dropna()
            if fix:
                data[data.index[0]]
            if len(d) == 0:
                return None
            return func(d)

        return _func

    grouped = df.groupby(lambda x: datetime(x.year, x.month, x.day))
    closure_bad = grouped.agg({"high": agg_before(np.max)})
    closure_good = grouped.agg({"high": agg_before(np.max, True)})
    tm.assert_frame_equal(closure_bad, closure_good)


def test_groupby_multiindex_missing_pair() -> None:
    df = DataFrame(
        {
            "group1": ["a", "a", "a", "b"],
            "group2": ["c", "c", "d", "c"],
            "value": [1, 1, 1, 5],
        }
    )
    df = df.set_index(["group1", "group2"])
    df_grouped = df.groupby(level=["group1", "group2"], sort=True)
    res = df_grouped.agg("sum")
    expected = DataFrame(
        [[2], [1], [5]],
        index=MultiIndex.from_tuples(
            [("a", "c"), ("a", "d"), ("b", "c")],
            names=["group1", "group2"],
        ),
        columns=["value"],
    )
    tm.assert_frame_equal(res, expected)


def test_groupby_multiindex_not_lexsorted(performance_warning: Any) -> None:
    lexsorted_mi = MultiIndex.from_tuples(
        [("a", ""), ("b1", "c1"), ("b2", "c2")],
        names=["b", "c"],
    )
    lexsorted_df = DataFrame([[1, 3, 4]], columns=lexsorted_mi)
    assert lexsorted_df.columns._is_lexsorted()
    not_lexsorted_df = DataFrame(
        {"a": [1, "b1", "c1", "b2", "c2"], "b": [1, "c1", "c2", "c3", "c4"], "d": [3, 4, 5, 6, 7]}
    )
    not_lexsorted_df = not_lexsorted_df.pivot_table(
        index="a", columns=["b", "c"], values="d"
    )
    not_lexsorted_df = not_lexsorted_df.reset_index()
    assert not not_lexsorted_df.columns._is_lexsorted()
    expected = lexsorted_df.groupby("a").mean()
    with tm.assert_produces_warning(performance_warning):
        result = not_lexsorted_df.groupby("a").mean()
    tm.assert_frame_equal(expected, result)
    df = DataFrame(
        {"x": ["a", "a", "b", "a"], "y": [1, 1, 2, 2], "z": [1, 2, 3, 4]}
    ).set_index(["x", "y"])
    assert not df.index._is_lexsorted()
    for level in [0, 1, [0, 1]]:
        for sort in [False, True]:
            result = df.groupby(
                level=level, sort=sort, group_keys=False
            ).apply(DataFrame.drop_duplicates)
            expected = df
            tm.assert_frame_equal(expected, result)
            result = df.sort_index().groupby(
                level=level, sort=sort, group_keys=False
            ).apply(DataFrame.drop_duplicates)
            expected = df.sort_index()
            tm.assert_frame_equal(expected, result)


def test_index_label_overlaps_location() -> None:
    df = DataFrame(list("ABCDE"), index=[2, 0, 2, 1, 1])
    g = df.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)
    df.index = df.index.astype(float)
    g = df.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)


def test_transform_doesnt_clobber_ints() -> None:
    n = 6
    x = np.arange(n)
    df = DataFrame(
        {"a": x // 2, "b": 2.0 * x, "c": 3.0 * x}
    )
    df2 = DataFrame(
        {"a": x // 2 * 1.0, "b": 2.0 * x, "c": 3.0 * x}
    )
    gb = df.groupby("a")
    result = gb.transform("mean")
    gb2 = df2.groupby("a")
    expected = gb2.transform("mean")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sort_column",
    ["ints", "floats", "strings", ["ints", "floats"], ["ints", "strings"]],
)
@pytest.mark.parametrize(
    "group_column",
    ["int_groups", "string_groups", ["int_groups", "string_groups"]],
)
def test_groupby_preserves_sort_multiindex_series() -> None:
    index = MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1)],
        names=["a", "b"],
    )
    mseries = Series([0, 1, 2], index=index)
    mseries_result = Series([0, 2, 4], index=index)
    result = mseries.groupby(level=["a", "b"], sort=False).first()
    tm.assert_series_equal(result, mseries_result)
    result = mseries.groupby(level=["a", "b"], sort=True).first()
    tm.assert_series_equal(result, mseries_result.sort_index())


def test_groupby_reindex_inside_function() -> None:
    periods = 1000
    ind = date_range(start="2012/1/1", freq="5min", periods=periods)
    df = DataFrame({"high": np.arange(periods), "low": np.arange(periods)}, index=ind)

    def agg_before(
        func: Callable[[pd.DataFrame], Any], fix: bool = False
    ) -> Callable[[pd.DataFrame], Any]:
        """
        Run an aggregate func on the subset of data.
        """

        def _func(data: pd.DataFrame) -> Any:
            d = data.loc[data.index.map(lambda x: x.hour < 11)].dropna()
            if fix:
                data[data.index[0]]
            if len(d) == 0:
                return None
            return func(d)

        return _func

    grouped = df.groupby(lambda x: datetime(x.year, x.month, x.day))
    closure_bad = grouped.agg({"high": agg_before(np.max)})
    closure_good = grouped.agg({"high": agg_before(np.max, True)})
    tm.assert_frame_equal(closure_bad, closure_good)


def test_groupby_multiindex_missing_pair() -> None:
    df = DataFrame(
        {
            "group1": ["a", "a", "a", "b"],
            "group2": ["c", "c", "d", "c"],
            "value": [1, 1, 1, 5],
        }
    )
    df = df.set_index(["group1", "group2"])
    df_grouped = df.groupby(level=["group1", "group2"], sort=True)
    res = df_grouped.agg("sum")
    expected = DataFrame(
        [[2], [1], [5]],
        index=MultiIndex.from_tuples(
            [("a", "c"), ("a", "d"), ("b", "c")],
            names=["group1", "group2"],
        ),
        columns=["value"],
    )
    tm.assert_frame_equal(res, expected)


def test_groupby_multiindex_not_lexsorted(performance_warning: Any) -> None:
    lexsorted_mi = MultiIndex.from_tuples(
        [("a", ""), ("b1", "c1"), ("b2", "c2")],
        names=["b", "c"],
    )
    lexsorted_df = DataFrame([[1, 3, 4]], columns=lexsorted_mi)
    assert lexsorted_df.columns._is_lexsorted()
    not_lexsorted_df = DataFrame(
        {"a": [1, "b1", "c1", "b2", "c2"], "b": [1, "c1", "c2", "c3", "c4"], "d": [3, 4, 5, 6, 7]}
    )
    not_lexsorted_df = not_lexsorted_df.pivot_table(
        index="a", columns=["b", "c"], values="d"
    )
    not_lexsorted_df = not_lexsorted_df.reset_index()
    assert not not_lexsorted_df.columns._is_lexsorted()
    expected = lexsorted_df.groupby("a").mean()
    with tm.assert_produces_warning(performance_warning):
        result = not_lexsorted_df.groupby("a").mean()
    tm.assert_frame_equal(expected, result)
    df = DataFrame(
        {"x": ["a", "a", "b", "a"], "y": [1, 1, 2, 2], "z": [1, 2, 3, 4]}
    ).set_index(["x", "y"])
    assert not df.index._is_lexsorted()
    for level in [0, 1, [0, 1]]:
        for sort in [False, True]:
            result = df.groupby(
                level=level, sort=sort, group_keys=False
            ).apply(DataFrame.drop_duplicates)
            expected = df
            tm.assert_frame_equal(expected, result)
            result = df.sort_index().groupby(
                level=level, sort=sort, group_keys=False
            ).apply(DataFrame.drop_duplicates)
            expected = df.sort_index()
            tm.assert_frame_equal(expected, result)


def test_index_label_overlaps_location() -> None:
    df = DataFrame(list("ABCDE"), index=[2, 0, 2, 1, 1])
    g = df.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)
    df.index = df.index.astype(float)
    g = df.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)


def test_transform_doesnt_clobber_ints() -> None:
    n = 6
    x = np.arange(n)
    df = DataFrame(
        {"a": x // 2, "b": 2.0 * x, "c": 3.0 * x}
    )
    df2 = DataFrame(
        {"a": x // 2 * 1.0, "b": 2.0 * x, "c": 3.0 * x}
    )
    gb = df.groupby("a")
    result = gb.transform("mean")
    gb2 = df2.groupby("a")
    expected = gb2.transform("mean")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sort_column",
    ["ints", "floats", "strings", ["ints", "floats"], ["ints", "strings"]],
)
@pytest.mark.parametrize(
    "group_column",
    ["int_groups", "string_groups", ["int_groups", "string_groups"]],
)
def test_groupby_preserves_sort_multiindex_series() -> None:
    index = MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1)],
        names=["a", "b"],
    )
    mseries = Series([0, 1, 2], index=index)
    mseries_result = Series([0, 2, 4], index=index)
    result = mseries.groupby(level=["a", "b"], sort=False).first()
    tm.assert_series_equal(result, mseries_result)
    result = mseries.groupby(level=["a", "b"], sort=True).first()
    tm.assert_series_equal(result, mseries_result.sort_index())


def test_groupby_reindex_inside_function() -> None:
    periods = 1000
    ind = date_range(start="2012/1/1", freq="5min", periods=periods)
    df = DataFrame({"high": np.arange(periods), "low": np.arange(periods)}, index=ind)

    def agg_before(
        func: Callable[[pd.DataFrame], Any], fix: bool = False
    ) -> Callable[[pd.DataFrame], Any]:
        """
        Run an aggregate func on the subset of data.
        """

        def _func(data: pd.DataFrame) -> Any:
            d = data.loc[data.index.map(lambda x: x.hour < 11)].dropna()
            if fix:
                data[data.index[0]]
            if len(d) == 0:
                return None
            return func(d)

        return _func

    grouped = df.groupby(lambda x: datetime(x.year, x.month, x.day))
    closure_bad = grouped.agg({"high": agg_before(np.max)})
    closure_good = grouped.agg({"high": agg_before(np.max, True)})
    tm.assert_frame_equal(closure_bad, closure_good)


def test_groupby_multiindex_missing_pair() -> None:
    df = DataFrame(
        {
            "group1": ["a", "a", "a", "b"],
            "group2": ["c", "c", "d", "c"],
            "value": [1, 1, 1, 5],
        }
    )
    df = df.set_index(["group1", "group2"])
    df_grouped = df.groupby(level=["group1", "group2"], sort=True)
    res = df_grouped.agg("sum")
    expected = DataFrame(
        [[2], [1], [5]],
        index=MultiIndex.from_tuples(
            [("a", "c"), ("a", "d"), ("b", "c")],
            names=["group1", "group2"],
        ),
        columns=["value"],
    )
    tm.assert_frame_equal(res, expected)


def test_groupby_multiindex_not_lexsorted(performance_warning: Any) -> None:
    lexsorted_mi = MultiIndex.from_tuples(
        [("a", ""), ("b1", "c1"), ("b2", "c2")],
        names=["b", "c"],
    )
    lexsorted_df = DataFrame([[1, 3, 4]], columns=lexsorted_mi)
    assert lexsorted_df.columns._is_lexsorted()
    not_lexsorted_df = DataFrame(
        {"a": [1, "b1", "c1", "b2", "c2"], "b": [1, "c1", "c2", "c3", "c4"], "d": [3, 4, 5, 6, 7]}
    )
    not_lexsorted_df = not_lexsorted_df.pivot_table(
        index="a", columns=["b", "c"], values="d"
    )
    not_lexsorted_df = not_lexsorted_df.reset_index()
    assert not not_lexsorted_df.columns._is_lexsorted()
    expected = lexsorted_df.groupby("a").mean()
    with tm.assert_produces_warning(performance_warning):
        result = not_lexsorted_df.groupby("a").mean()
    tm.assert_frame_equal(expected, result)
    df = DataFrame(
        {"x": ["a", "a", "b", "a"], "y": [1, 1, 2, 2], "z": [1, 2, 3, 4]}
    ).set_index(["x", "y"])
    assert not df.index._is_lexsorted()
    for level in [0, 1, [0, 1]]:
        for sort in [False, True]:
            result = df.groupby(
                level=level, sort=sort, group_keys=False
            ).apply(DataFrame.drop_duplicates)
            expected = df
            tm.assert_frame_equal(expected, result)
            result = df.sort_index().groupby(
                level=level, sort=sort, group_keys=False
            ).apply(DataFrame.drop_duplicates)
            expected = df.sort_index()
            tm.assert_frame_equal(expected, result)


def test_index_label_overlaps_location() -> None:
    df = DataFrame(list("ABCDE"), index=[2, 0, 2, 1, 1])
    g = df.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)
    df.index = df.index.astype(float)
    g = df.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)


def test_transform_doesnt_clobber_ints() -> None:
    n = 6
    x = np.arange(n)
    df = DataFrame(
        {"a": x // 2, "b": 2.0 * x, "c": 3.0 * x}
    )
    df2 = DataFrame(
        {"a": x // 2 * 1.0, "b": 2.0 * x, "c": 3.0 * x}
    )
    gb = df.groupby("a")
    result = gb.transform("mean")
    gb2 = df2.groupby("a")
    expected = gb2.transform("mean")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sort_column",
    ["ints", "floats", "strings", ["ints", "floats"], ["ints", "strings"]],
)
@pytest.mark.parametrize(
    "group_column",
    ["int_groups", "string_groups", ["int_groups", "string_groups"]],
)
def test_groupby_preserves_sort_multiindex_series() -> None:
    index = MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1)],
        names=["a", "b"],
    )
    mseries = Series([0, 1, 2], index=index)
    mseries_result = Series([0, 2, 4], index=index)
    result = mseries.groupby(level=["a", "b"], sort=False).first()
    tm.assert_series_equal(result, mseries_result)
    result = mseries.groupby(level=["a", "b"], sort=True).first()
    tm.assert_series_equal(result, mseries_result.sort_index())


def test_groupby_reindex_inside_function() -> None:
    periods = 1000
    ind = date_range(start="2012/1/1", freq="5min", periods=periods)
    df = DataFrame({"high": np.arange(periods), "low": np.arange(periods)}, index=ind)

    def agg_before(
        func: Callable[[pd.DataFrame], Any], fix: bool = False
    ) -> Callable[[pd.DataFrame], Any]:
        """
        Run an aggregate func on the subset of data.
        """

        def _func(data: pd.DataFrame) -> Any:
            d = data.loc[data.index.map(lambda x: x.hour < 11)].dropna()
            if fix:
                data[data.index[0]]
            if len(d) == 0:
                return None
            return func(d)

        return _func

    grouped = df.groupby(lambda x: datetime(x.year, x.month, x.day))
    closure_bad = grouped.agg({"high": agg_before(np.max)})
    closure_good = grouped.agg({"high": agg_before(np.max, True)})
    tm.assert_frame_equal(closure_bad, closure_good)


def test_groupby_multiindex_missing_pair() -> None:
    df = DataFrame(
        {
            "group1": ["a", "a", "a", "b"],
            "group2": ["c", "c", "d", "c"],
            "value": [1, 1, 1, 5],
        }
    )
    df = df.set_index(["group1", "group2"])
    df_grouped = df.groupby(level=["group1", "group2"], sort=True)
    res = df_grouped.agg("sum")
    expected = DataFrame(
        [[2], [1], [5]],
        index=MultiIndex.from_tuples(
            [("a", "c"), ("a", "d"), ("b", "c")],
            names=["group1", "group2"],
        ),
        columns=["value"],
    )
    tm.assert_frame_equal(res, expected)


def test_groupby_multiindex_not_lexsorted(performance_warning: Any) -> None:
    lexsorted_mi = MultiIndex.from_tuples(
        [("a", ""), ("b1", "c1"), ("b2", "c2")],
        names=["b", "c"],
    )
    lexsorted_df = DataFrame([[1, 3, 4]], columns=lexsorted_mi)
    assert lexsorted_df.columns._is_lexsorted()
    not_lexsorted_df = DataFrame(
        {"a": [1, "b1", "c1", "b2", "c2"], "b": [1, "c1", "c2", "c3", "c4"], "d": [3, 4, 5, 6, 7]}
    )
    not_lexsorted_df = not_lexsorted_df.pivot_table(
        index="a", columns=["b", "c"], values="d"
    )
    not_lexsorted_df = not_lexsorted_df.reset_index()
    assert not not_lexsorted_df.columns._is_lexsorted()
    expected = lexsorted_df.groupby("a").mean()
    with tm.assert_produces_warning(performance_warning):
        result = not_lexsorted_df.groupby("a").mean()
    tm.assert_frame_equal(expected, result)
    df = DataFrame(
        {"x": ["a", "a", "b", "a"], "y": [1, 1, 2, 2], "z": [1, 2, 3, 4]}
    ).set_index(["x", "y"])
    assert not df.index._is_lexsorted()
    for level in [0, 1, [0, 1]]:
        for sort in [False, True]:
            result = df.groupby(
                level=level, sort=sort, group_keys=False
            ).apply(DataFrame.drop_duplicates)
            expected = df
            tm.assert_frame_equal(expected, result)
            result = df.sort_index().groupby(
                level=level, sort=sort, group_keys=False
            ).apply(DataFrame.drop_duplicates)
            expected = df.sort_index()
            tm.assert_frame_equal(expected, result)


def test_index_label_overlaps_location() -> None:
    df = DataFrame(list("ABCDE"), index=[2, 0, 2, 1, 1])
    g = df.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)
    df.index = df.index.astype(float)
    g = df.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)


def test_transform_doesnt_clobber_ints() -> None:
    n = 6
    x = np.arange(n)
    df = DataFrame(
        {"a": x // 2, "b": 2.0 * x, "c": 3.0 * x}
    )
    df2 = DataFrame(
        {"a": x // 2 * 1.0, "b": 2.0 * x, "c": 3.0 * x}
    )
    gb = df.groupby("a")
    result = gb.transform("mean")
    gb2 = df2.groupby("a")
    expected = gb2.transform("mean")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sort_column",
    ["ints", "floats", "strings", ["ints", "floats"], ["ints", "strings"]],
)
@pytest.mark.parametrize(
    "group_column",
    ["int_groups", "string_groups", ["int_groups", "string_groups"]],
)
def test_groupby_preserves_sort_multiindex_series() -> None:
    index = MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1)],
        names=["a", "b"],
    )
    mseries = Series([0, 1, 2], index=index)
    mseries_result = Series([0, 2, 4], index=index)
    result = mseries.groupby(level=["a", "b"], sort=False).first()
    tm.assert_series_equal(result, mseries_result)
    result = mseries.groupby(level=["a", "b"], sort=True).first()
    tm.assert_series_equal(result, mseries_result.sort_index())


def test_groupby_reindex_inside_function() -> None:
    periods = 1000
    ind = date_range(start="2012/1/1", freq="5min", periods=periods)
    df = DataFrame({"high": np.arange(periods), "low": np.arange(periods)}, index=ind)

    def agg_before(
        func: Callable[[pd.DataFrame], Any], fix: bool = False
    ) -> Callable[[pd.DataFrame], Any]:
        """
        Run an aggregate func on the subset of data.
        """

        def _func(data: pd.DataFrame) -> Any:
            d = data.loc[data.index.map(lambda x: x.hour < 11)].dropna()
            if fix:
                data[data.index[0]]
            if len(d) == 0:
                return None
            return func(d)

        return _func

    grouped = df.groupby(lambda x: datetime(x.year, x.month, x.day))
    closure_bad = grouped.agg({"high": agg_before(np.max)})
    closure_good = grouped.agg({"high": agg_before(np.max, True)})
    tm.assert_frame_equal(closure_bad, closure_good)


def test_groupby_multiindex_missing_pair() -> None:
    df = DataFrame(
        {
            "group1": ["a", "a", "a", "b"],
            "group2": ["c", "c", "d", "c"],
            "value": [1, 1, 1, 5],
        }
    )
    df = df.set_index(["group1", "group2"])
    df_grouped = df.groupby(level=["group1", "group2"], sort=True)
    res = df_grouped.agg("sum")
    expected = DataFrame(
        [[2], [1], [5]],
        index=MultiIndex.from_tuples(
            [("a", "c"), ("a", "d"), ("b", "c")],
            names=["group1", "group2"],
        ),
        columns=["value"],
    )
    tm.assert_frame_equal(res, expected)


def test_groupby_multiindex_not_lexsorted(performance_warning: Any) -> None:
    lexsorted_mi = MultiIndex.from_tuples(
        [("a", ""), ("b1", "c1"), ("b2", "c2")],
        names=["b", "c"],
    )
    lexsorted_df = DataFrame([[1, 3, 4]], columns=lexsorted_mi)
    assert lexsorted_df.columns._is_lexsorted()
    not_lexsorted_df = DataFrame(
        {"a": [1, "b1", "c1", "b2", "c2"], "b": [1, "c1", "c2", "c3", "c4"], "d": [3, 4, 5, 6, 7]}
    )
    not_lexsorted_df = not_lexsorted_df.pivot_table(
        index="a", columns=["b", "c"], values="d"
    )
    not_lexsorted_df = not_lexsorted_df.reset_index()
    assert not not_lexsorted_df.columns._is_lexsorted()
    expected = lexsorted_df.groupby("a").mean()
    with tm.assert_produces_warning(performance_warning):
        result = not_lexsorted_df.groupby("a").mean()
    tm.assert_frame_equal(expected, result)
    df = DataFrame(
        {"x": ["a", "a", "b", "a"], "y": [1, 1, 2, 2], "z": [1, 2, 3, 4]}
    ).set_index(["x", "y"])
    assert not df.index._is_lexsorted()
    for level in [0, 1, [0, 1]]:
        for sort in [False, True]:
            result = df.groupby(
                level=level, sort=sort, group_keys=False
            ).apply(DataFrame.drop_duplicates)
            expected = df
            tm.assert_frame_equal(expected, result)
            result = df.sort_index().groupby(
                level=level, sort=sort, group_keys=False
            ).apply(DataFrame.drop_duplicates)
            expected = df.sort_index()
            tm.assert_frame_equal(expected, result)


def test_index_label_overlaps_location() -> None:
    df = DataFrame(list("ABCDE"), index=[2, 0, 2, 1, 1])
    g = df.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)
    df.index = df.index.astype(float)
    g = df.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)


def test_transform_doesnt_clobber_ints() -> None:
    n = 6
    x = np.arange(n)
    df = DataFrame(
        {"a": x // 2, "b": 2.0 * x, "c": 3.0 * x}
    )
    df2 = DataFrame(
        {"a": x // 2 * 1.0, "b": 2.0 * x, "c": 3.0 * x}
    )
    gb = df.groupby("a")
    result = gb.transform("mean")
    gb2 = df2.groupby("a")
    expected = gb2.transform("mean")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sort_column",
    ["ints", "floats", "strings", ["ints", "floats"], ["ints", "strings"]],
)
@pytest.mark.parametrize(
    "group_column",
    ["int_groups", "string_groups", ["int_groups", "string_groups"]],
)
def test_groupby_preserves_sort_multiindex_series() -> None:
    index = MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1)],
        names=["a", "b"],
    )
    mseries = Series([0, 1, 2], index=index)
    mseries_result = Series([0, 2, 4], index=index)
    result = mseries.groupby(level=["a", "b"], sort=False).first()
    tm.assert_series_equal(result, mseries_result)
    result = mseries.groupby(level=["a", "b"], sort=True).first()
    tm.assert_series_equal(result, mseries_result.sort_index())


def test_groupby_reindex_inside_function() -> None:
    periods = 1000
    ind = date_range(start="2012/1/1", freq="5min", periods=periods)
    df = DataFrame({"high": np.arange(periods), "low": np.arange(periods)}, index=ind)

    def agg_before(
        func: Callable[[pd.DataFrame], Any], fix: bool = False
    ) -> Callable[[pd.DataFrame], Any]:
        """
        Run an aggregate func on the subset of data.
        """

        def _func(data: pd.DataFrame) -> Any:
            d = data.loc[data.index.map(lambda x: x.hour < 11)].dropna()
            if fix:
                data[data.index[0]]
            if len(d) == 0:
                return None
            return func(d)

        return _func

    grouped = df.groupby(lambda x: datetime(x.year, x.month, x.day))
    closure_bad = grouped.agg({"high": agg_before(np.max)})
    closure_good = grouped.agg({"high": agg_before(np.max, True)})
    tm.assert_frame_equal(closure_bad, closure_good)


def test_groupby_multiindex_missing_pair() -> None:
    df = DataFrame(
        {
            "group1": ["a", "a", "a", "b"],
            "group2": ["c", "c", "d", "c"],
            "value": [1, 1, 1, 5],
        }
    )
    df = df.set_index(["group1", "group2"])
    df_grouped = df.groupby(level=["group1", "group2"], sort=True)
    res = df_grouped.agg("sum")
    expected = DataFrame(
        [[2], [1], [5]],
        index=MultiIndex.from_tuples(
            [("a", "c"), ("a", "d"), ("b", "c")],
            names=["group1", "group2"],
        ),
        columns=["value"],
    )
    tm.assert_frame_equal(res, expected)


def test_groupby_multiindex_not_lexsorted(performance_warning: Any) -> None:
    lexsorted_mi = MultiIndex.from_tuples(
        [("a", ""), ("b1", "c1"), ("b2", "c2")],
        names=["b", "c"],
    )
    lexsorted_df = DataFrame([[1, 3, 4]], columns=lexsorted_mi)
    assert lexsorted_df.columns._is_lexsorted()
    not_lexsorted_df = DataFrame(
        {"a": [1, "b1", "c1", "b2", "c2"], "b": [1, "c1", "c2", "c3", "c4"], "d": [3, 4, 5, 6, 7]}
    )
    not_lexsorted_df = not_lexsorted_df.pivot_table(
        index="a", columns=["b", "c"], values="d"
    )
    not_lexsorted_df = not_lexsorted_df.reset_index()
    assert not not_lexsorted_df.columns._is_lexsorted()
    expected = lexsorted_df.groupby("a").mean()
    with tm.assert_produces_warning(performance_warning):
        result = not_lexsorted_df.groupby("a").mean()
    tm.assert_frame_equal(expected, result)
    df = DataFrame(
        {"x": ["a", "a", "b", "a"], "y": [1, 1, 2, 2], "z": [1, 2, 3, 4]}
    ).set_index(["x", "y"])
    assert not df.index._is_lexsorted()
    for level in [0, 1, [0, 1]]:
        for sort in [False, True]:
            result = df.groupby(
                level=level, sort=sort, group_keys=False
            ).apply(DataFrame.drop_duplicates)
            expected = df
            tm.assert_frame_equal(expected, result)
            result = df.sort_index().groupby(
                level=level, sort=sort, group_keys=False
            ).apply(DataFrame.drop_duplicates)
            expected = df.sort_index()
            tm.assert_frame_equal(expected, result)


def test_index_label_overlaps_location() -> None:
    df = DataFrame(list("ABCDE"), index=[2, 0, 2, 1, 1])
    g = df.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)
    df.index = df.index.astype(float)
    g = df.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)
    ser = df[0]
    g = ser.groupby(["a", "b", "a", "b", "b"])
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)


def test_transform_doesnt_clobber_ints() -> None:
    n = 6
    x = np.arange(n)
    df = DataFrame(
        {"a": x // 2, "b": 2.0 * x, "c": 3.0 * x}
    )
    df2 = DataFrame(
        {"a": x // 2 * 1.0, "b": 2.0 * x, "c": 3.0 * x}
    )
    gb = df.groupby("a")
    result = gb.transform("mean")
    gb2 = df2.groupby("a")
    expected = gb2.transform("mean")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sort_column",
    ["ints", "floats", "strings", ["ints", "floats"], ["ints", "strings"]],
)
@pytest.mark.parametrize(
    "group_column",
    ["int_groups", "string_groups", ["int_groups", "string_groups"]],
)
def test_groupby_preserves_sort_multiindex_series() -> None:
    index = MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1)],
        names=["a", "b"],
    )
    mseries = Series([0, 1, 2], index=index)
    mseries_result = Series([0, 2, 4], index=index)
    result = mseries.groupby(level=["a", "b"], sort=False).first()
    tm.assert_series_equal(result, mseries_result)
    result = mseries.groupby(level=["a", "b"], sort=True).first()
    tm.assert_series_equal(result, mseries_result.sort_index())


def test_groupby_reindex_inside_function() -> None:
    periods = 1000
    ind = date_range(start="2012/1/1", freq="5min", periods=periods)
    df = DataFrame({"high": np.arange(periods), "low": np.arange(periods)}, index=ind)

    def agg_before(
        func: Callable[[pd.DataFrame], Any], fix: bool = False
    ) -> Callable[[pd.DataFrame], Any]:
        """
        Run an aggregate func on the subset of data.
        """

        def _func(data: pd.DataFrame) -> Any:
            d = data.loc[data.index.map(lambda x: x.hour < 11)].dropna()
            if fix:
                data[data.index[0]]
            if len(d) == 0:
                return None
            return func(d)

        return _func

    grouped = df.groupby(lambda x: datetime(x.year, x.month, x.day))
    closure_bad = grouped.agg({"high": agg_before(np.max)})
    closure_good = grouped.agg({"high": agg_before(np.max, True)})
    tm.assert_frame_equal(closure_bad, closure_good)


def test_groupby_multiindex_missing_pair() -> None:
    df = DataFrame(
        {
            "group1": ["a", "a", "a", "b"],
            "group2": ["c", "c", "d", "c"],
            "value": [1, 1, 1, 5],
        }
    )
    df = df.set_index(["group1", "group2"])
    df_grouped = df.groupby(level=["group1", "group2"], sort=True)
    res = df_grouped.agg("sum")
    expected = DataFrame(
        [[2], [1], [5]],
        index=MultiIndex.from_tuples(
            [("a", "c"), ("a", "d"), ("b", "c")],
            names=["group1", "group2"],
        ),
        columns=["value"],
    )
    tm.assert_frame_equal(res, expected)

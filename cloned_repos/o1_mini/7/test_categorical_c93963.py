from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    qcut,
)
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args


def cartesian_product_for_groupers(
    result: DataFrame,
    args: List[Any],
    names: List[str],
    fill_value: Any = np.nan,
) -> DataFrame:
    """Reindex to a cartesian production for the groupers,
    preserving the nature (Categorical) of each grouper
    """

    def f(a: Any) -> Any:
        if isinstance(a, (CategoricalIndex, Categorical)):
            categories = a.categories
            a = Categorical.from_codes(
                np.arange(len(categories)), categories=categories, ordered=a.ordered
            )
        return a

    index = MultiIndex.from_product(map(f, args), names=names)
    return result.reindex(index, fill_value=fill_value).sort_index()


_results_for_groupbys_with_missing_categories: Dict[str, Union[bool, float, int]] = {
    "all": True,
    "any": False,
    "count": 0,
    "corrwith": np.nan,
    "first": np.nan,
    "idxmax": np.nan,
    "idxmin": np.nan,
    "last": np.nan,
    "max": np.nan,
    "mean": np.nan,
    "median": np.nan,
    "min": np.nan,
    "nth": np.nan,
    "nunique": 0,
    "prod": 1,
    "quantile": np.nan,
    "sem": np.nan,
    "size": 0,
    "skew": np.nan,
    "kurt": np.nan,
    "std": np.nan,
    "sum": 0,
    "var": np.nan,
}


def test_apply_use_categorical_name(df: DataFrame) -> None:
    cats = qcut(df.C, 4)

    def get_stats(group: Series) -> Dict[str, float]:
        return {
            "min": group.min(),
            "max": group.max(),
            "count": group.count(),
            "mean": group.mean(),
        }

    result: Series = df.groupby(cats, observed=False).D.apply(get_stats)
    assert result.index.names[0] == "C"


def test_basic() -> None:
    cats = Categorical(
        ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        categories=["a", "b", "c", "d"],
        ordered=True,
    )
    data = DataFrame({"a": [1, 1, 1, 2, 2, 2, 3, 4, 5], "b": cats})
    exp_index: CategoricalIndex = CategoricalIndex(
        list("abcd"), name="b", ordered=True
    )
    expected: DataFrame = DataFrame({"a": [1, 2, 4, np.nan]}, index=exp_index)
    result: DataFrame = data.groupby("b", observed=False).mean()
    tm.assert_frame_equal(result, expected)


def test_basic_single_grouper() -> None:
    cat1 = Categorical(
        ["a", "a", "b", "b"],
        categories=["a", "b", "z"],
        ordered=True,
    )
    cat2 = Categorical(
        ["c", "d", "c", "d"],
        categories=["c", "d", "y"],
        ordered=True,
    )
    df: DataFrame = DataFrame({"A": cat1, "B": cat2, "values": [1, 2, 3, 4]})
    gb: SeriesGroupBy = df.groupby("A", observed=False)
    exp_idx: CategoricalIndex = CategoricalIndex(
        ["a", "b", "z"], name="A", ordered=True
    )
    expected: DataFrame = DataFrame(
        {"values": Series([3, 7, 0], index=exp_idx)}
    )
    result: DataFrame = gb.sum(numeric_only=True)
    tm.assert_frame_equal(result, expected)


def test_basic_string(using_infer_string: bool) -> None:
    x: DataFrame = DataFrame(
        [[1, "John P. Doe"], [2, "Jane Dove"], [1, "John P. Doe"]],
        columns=["person_id", "person_name"],
    )
    x["person_name"] = Categorical(x.person_name)
    g: SeriesGroupBy = x.groupby(["person_id"], observed=False)
    result: DataFrame = g.transform(lambda x: x)
    tm.assert_frame_equal(result, x[["person_name"]])
    result = x.drop_duplicates("person_name")
    expected: DataFrame = x.iloc[[0, 1]]
    tm.assert_frame_equal(result, expected)

    def f(x: DataFrame) -> Series:
        return x.drop_duplicates("person_name").iloc[0]

    result = g.apply(f)
    expected = x[["person_name"]].iloc[[0, 1]]
    expected.index = Index([1, 2], name="person_id")
    dtype: Union[str, Any] = "str" if using_infer_string else object
    expected["person_name"] = expected["person_name"].astype(dtype)
    tm.assert_frame_equal(result, expected)


def test_basic_monotonic() -> None:
    df: DataFrame = DataFrame({"a": [5, 15, 25]})
    c: Categorical = pd.cut(df.a, bins=[0, 10, 20, 30, 40])
    result: Series = df.a.groupby(c, observed=False).transform(sum)
    tm.assert_series_equal(result, df["a"])
    tm.assert_series_equal(
        df.a.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df["a"]
    )
    result = df.groupby(c, observed=False).transform(sum)
    expected: DataFrame = df[["a"]]
    tm.assert_frame_equal(result, expected)
    gbc: pd.core.groupby.DataFrameGroupBy = df.groupby(c, observed=False)
    result = gbc.transform(lambda xs: np.max(xs, axis=0))
    tm.assert_frame_equal(result, df[["a"]])
    result2: DataFrame = gbc.transform(lambda xs: np.max(xs, axis=0))
    result3: DataFrame = gbc.transform(max)
    result4: DataFrame = gbc.transform(np.maximum.reduce)
    result5: DataFrame = gbc.transform(lambda xs: np.maximum.reduce(xs))
    tm.assert_frame_equal(result2, df[["a"]], check_dtype=False)
    tm.assert_frame_equal(result3, df[["a"]], check_dtype=False)
    tm.assert_frame_equal(result4, df[["a"]])
    tm.assert_frame_equal(result5, df[["a"]])
    tm.assert_series_equal(
        df.a.groupby(c, observed=False).filter(np.all), df["a"]
    )
    tm.assert_frame_equal(df.groupby(c, observed=False).filter(np.all), df)


def test_basic_non_monotonic() -> None:
    df: DataFrame = DataFrame({"a": [5, 15, 25, -5]})
    c: Categorical = pd.cut(df.a, bins=[-10, 0, 10, 20, 30, 40])
    result: Series = df.a.groupby(c, observed=False).transform(sum)
    tm.assert_series_equal(result, df["a"])
    tm.assert_series_equal(
        df.a.groupby(c, observed=False).transform(lambda xs: np.sum(xs)),
        df["a"],
    )
    result = df.groupby(c, observed=False).transform(sum)
    expected: DataFrame = df[["a"]]
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(
        df.groupby(c, observed=False).transform(lambda xs: np.sum(xs)),
        df[["a"]],
    )


def test_basic_cut_grouping() -> None:
    df: DataFrame = DataFrame({"a": [1, 0, 0, 0]})
    c: Categorical = pd.cut(
        df.a, [0, 1, 2, 3, 4], labels=Categorical(list("abcd"))
    )
    result: Series = df.groupby(c, observed=False).apply(len)
    exp_index: CategoricalIndex = CategoricalIndex(
        c.values.categories, ordered=c.values.ordered
    )
    expected: Series = Series([1, 0, 0, 0], index=exp_index)
    expected.index.name = "a"
    tm.assert_series_equal(result, expected)


def test_more_basic() -> None:
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    exp_idx: CategoricalIndex = CategoricalIndex(levels, categories=cats.categories, ordered=True)
    expected = expected.reindex(exp_idx)
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = np.asarray(cats).take(idx)
    ord_data: DataFrame = data.take(idx)
    exp_cats: Categorical = Categorical(
        ord_labels, ordered=True, categories=["foo", "bar", "baz", "qux"]
    )
    expected = ord_data.groupby(exp_cats, sort=False, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    expc: Categorical = Categorical.from_codes(np.arange(4).repeat(8), levels, ordered=True)
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(desc_result.stack().index.get_level_values(0), exp)
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(desc_result.stack().index.get_level_values(1), exp)


def test_level_get_group(observed: bool) -> None:
    df: DataFrame = DataFrame(
        data=np.arange(2, 22, 2),
        index=MultiIndex(
            levels=[CategoricalIndex(["a", "b"]), range(10)],
            codes=[[0] * 5 + [1] * 5, list(range(10))],
            names=["Index1", "Index2"],
        ),
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby(level=["Index1"], observed=observed)
    expected: DataFrame = DataFrame(
        data=np.arange(2, 12, 2),
        index=MultiIndex(
            levels=[CategoricalIndex(["a", "b"]), range(5)],
            codes=[[0] * 5, list(range(5))],
            names=["Index1", "Index2"],
        ),
    )
    result: DataFrame = g.get_group(("a",))
    tm.assert_frame_equal(result, expected)


def test_sorting_with_different_categoricals() -> None:
    df: DataFrame = DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "dose": ["high", "med", "low"] * 4,
            "outcomes": np.arange(12.0),
        }
    )
    df.dose = Categorical(df.dose, categories=["low", "med", "high"], ordered=True)
    result: Series = df.groupby("group")["dose"].value_counts()
    result = result.sort_index(level=0, sort_remaining=True)
    index_labels: List[str] = ["low", "med", "high", "low", "med", "high"]
    index_cat: Categorical = Categorical(
        index_labels, categories=["low", "med", "high"], ordered=True
    )
    index: MultiIndex = MultiIndex.from_arrays(
        [["A", "A", "A", "B", "B", "B"], CategoricalIndex(index_cat)],
        names=["group", "dose"],
    )
    expected: Series = Series([2] * 6, index=index, name="count")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
def test_apply(ordered: bool) -> None:
    dense: Categorical = Categorical(list("abc"), ordered=ordered)
    missing: Categorical = Categorical(
        list("aaa"), categories=["a", "b"], ordered=ordered
    )
    values: np.ndarray = np.arange(len(dense))
    df: DataFrame = DataFrame(
        {"missing": missing, "dense": dense, "values": values}
    )
    grouped: pd.core.groupby.DataFrameGroupBy = df.groupby(
        ["missing", "dense"], observed=True
    )
    idx: MultiIndex = MultiIndex.from_arrays(
        [missing, dense], names=["missing", "dense"]
    )
    expected: DataFrame = DataFrame(
        [0, 1, 2.0], index=idx, columns=["values"]
    )
    result: DataFrame = grouped.apply(lambda x: np.mean(x, axis=0))
    tm.assert_frame_equal(result, expected)
    result = grouped.mean()
    tm.assert_frame_equal(result, expected)
    result = grouped.agg(np.mean)
    tm.assert_frame_equal(result, expected)
    idx = MultiIndex.from_arrays([missing, dense], names=["missing", "dense"])
    expected: Series = Series(1, index=idx)
    result = grouped.apply(lambda x: 1)
    tm.assert_series_equal(result, expected)


def test_observed(request: Any, using_infer_string: bool, observed: bool) -> None:
    if using_infer_string and (not observed):
        request.applymarker(pytest.mark.xfail(reason="TODO(infer_string)"))
    cat1: Categorical = Categorical(
        ["a", "a", "b", "b"],
        categories=["a", "b", "z"],
        ordered=True,
    )
    cat2: Categorical = Categorical(
        ["c", "d", "c", "d"],
        categories=["c", "d", "y"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        {"A": cat1, "B": cat2, "values": [1, 2, 3, 4]}
    )
    df["C"] = ["foo", "bar"] * 2
    gb: pd.core.groupby.DataFrameGroupBy = df.groupby(
        ["A", "B", "C"], observed=observed
    )
    exp_index: MultiIndex = MultiIndex.from_arrays(
        [cat1, cat2, ["foo", "bar"] * 2], names=["A", "B", "C"]
    )
    expected: DataFrame = DataFrame(
        {"values": Series([1, 2, 3, 4], index=exp_index)}
    ).sort_index()
    result: DataFrame = gb.sum()
    if not observed:
        expected = cartesian_product_for_groupers(
            expected,
            [cat1, cat2, ["foo", "bar"]],
            list("ABC"),
            fill_value=0,
        )
    tm.assert_frame_equal(result, expected)
    gb = df.groupby(["A", "B"], observed=observed)
    exp_index = MultiIndex.from_arrays([cat1, cat2], names=["A", "B"])
    expected = DataFrame(
        {"values": [1, 2, 3, 4], "C": ["foo", "bar", "foo", "bar"]},
        index=exp_index,
    )
    result = gb.sum()
    if not observed:
        expected = cartesian_product_for_groupers(
            expected, [cat1, cat2], list("AB"), fill_value=0
        )
    tm.assert_frame_equal(result, expected)


def test_observed_single_column(observed: bool) -> None:
    d: Dict[str, Any] = {
        "cat": Categorical(
            ["a", "b", "a", "b"], categories=["a", "b", "c"], ordered=True
        ),
        "ints": [1, 1, 2, 2],
        "val": [10, 20, 30, 40],
    }
    df: DataFrame = DataFrame(d)
    groups_single_key: pd.core.groupby.DataFrameGroupBy = df.groupby(
        "cat", observed=observed
    )
    result: DataFrame = groups_single_key.mean()
    exp_index: CategoricalIndex = CategoricalIndex(
        list("ab"),
        name="cat",
        categories=list("abc"),
        ordered=True,
    )
    expected: DataFrame = DataFrame(
        {"ints": [1.5, 1.5], "val": [20.0, 30]},
        index=exp_index,
    )
    if not observed:
        index: CategoricalIndex = CategoricalIndex(
            list("abc"), name="cat", categories=list("abc"), ordered=True
        )
        expected = expected.reindex(index)
    tm.assert_frame_equal(result, expected)


def test_observed_two_columns(observed: bool) -> None:
    d: Dict[str, Any] = {
        "cat": Categorical(
            ["a", "b", "a", "b"], categories=["a", "b", "c"], ordered=True
        ),
        "ints": [1, 1, 2, 2],
        "val": [10, 20, 30, 40],
    }
    df: DataFrame = DataFrame(d)
    groups_double_key: pd.core.groupby.DataFrameGroupBy = df.groupby(
        ["cat", "ints"], observed=observed
    )
    result: DataFrame = groups_double_key.agg("mean")
    expected: DataFrame = DataFrame(
        {
            "val": [10.0, 30.0, 20.0, 40.0],
            "cat": Categorical(
                ["a", "a", "b", "b"],
                categories=["a", "b", "c"],
                ordered=True,
            ),
            "ints": [1, 2, 1, 2],
        }
    ).set_index(["cat", "ints"])
    if not observed:
        expected = cartesian_product_for_groupers(
            expected, [df.cat.values, [1, 2]], ["cat", "ints"]
        )
    tm.assert_frame_equal(result, expected)
    for key in [("a", 1), ("b", 2), ("b", 1), ("a", 2)]:
        c, i = key
        result = groups_double_key.get_group(key)
        expected = df[(df.cat == c) & (df.ints == i)]
        tm.assert_frame_equal(result, expected)


def test_observed_with_as_index(observed: bool) -> None:
    d: Dict[str, Any] = {
        "foo": [10, 8, 4, 8, 4, 1, 1],
        "bar": [10, 20, 30, 40, 50, 60, 70],
        "baz": ["d", "c", "e", "a", "a", "d", "c"],
    }
    df: DataFrame = DataFrame(d)
    cat: Categorical = pd.cut(df["foo"], np.linspace(0, 10, 3))
    df["range"] = cat
    groups: pd.core.groupby.DataFrameGroupBy = df.groupby(
        ["range", "baz"], as_index=False, observed=observed
    )
    result: DataFrame = groups.agg("mean")
    groups2: pd.core.groupby.DataFrameGroupBy = df.groupby(
        ["range", "baz"], as_index=True, observed=observed
    )
    expected: DataFrame = groups2.agg("mean").reset_index()
    tm.assert_frame_equal(result, expected)


def test_observed_codes_remap(observed: bool) -> None:
    d: Dict[str, Any] = {
        "C1": [3, 3, 4, 5],
        "C2": [1, 2, 3, 4],
        "C3": [10, 100, 200, 34],
    }
    df: DataFrame = DataFrame(d)
    values: Categorical = pd.cut(df["C1"], [1, 2, 3, 6])
    values.name = "cat"
    groups_double_key: pd.core.groupby.DataFrameGroupBy = df.groupby(
        [values, "C2"], observed=observed
    )
    idx: MultiIndex = MultiIndex.from_arrays(
        [values, [1, 2, 3, 4]], names=["cat", "C2"]
    )
    expected: DataFrame = DataFrame(
        {"C1": [3.0, 3.0, 4.0, 5.0], "C3": [10.0, 100.0, 200.0, 34.0]},
        index=idx,
    )
    if not observed:
        expected = cartesian_product_for_groupers(
            expected,
            [values.values, [1, 2, 3, 4]],
            ["cat", "C2"],
            fill_value=0,
        )
    result: DataFrame = groups_double_key.agg("mean")
    tm.assert_frame_equal(result, expected)


def test_observed_perf() -> None:
    df: DataFrame = DataFrame(
        {
            "cat": np.random.default_rng(2).integers(0, 255, size=30000),
            "int_id": np.random.default_rng(2).integers(0, 255, size=30000),
            "other_id": np.random.default_rng(2).integers(0, 10000, size=30000),
            "foo": 0,
        }
    )
    df["cat"] = df.cat.astype(str).astype("category")
    grouped: pd.core.groupby.DataFrameGroupBy = df.groupby(
        ["cat", "int_id", "other_id"], observed=True
    )
    result: DataFrame = grouped.count()
    assert result.index.levels[0].nunique() == df.cat.nunique()
    assert result.index.levels[1].nunique() == df.int_id.nunique()
    assert result.index.levels[2].nunique() == df.other_id.nunique()


def test_observed_groups(observed: bool) -> None:
    cat: Categorical = Categorical(["a", "c", "a"], categories=["a", "b", "d"])
    df: DataFrame = DataFrame({"cat": cat, "vals": [1, 2, 3]})
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("cat", observed=observed)
    result: Dict[str, Index] = g.groups
    if observed:
        expected: Dict[str, Index] = {
            "a": Index([0, 2], dtype="int64"),
            "c": Index([1], dtype="int64"),
        }
    else:
        expected = {
            "a": Index([0, 2], dtype="int64"),
            "b": Index([], dtype="int64"),
            "c": Index([1], dtype="int64"),
        }
    tm.assert_dict_equal(result, expected)


@pytest.mark.parametrize(
    "keys, expected_values, expected_index_levels",
    [
        (
            "a",
            [15, 9, 0],
            CategoricalIndex([1, 2, 3], name="a"),
        ),
        (
            ["a", "b"],
            [7, 8, 0, 0, 0, 9, 0, 0, 0],
            [
                CategoricalIndex([1, 2, 3], name="a"),
                Index([4, 5, 6]),
            ],
        ),
        (
            ["a", "a2"],
            [15, 0, 0, 0, 9, 0, 0, 0, 0],
            [
                CategoricalIndex([1, 2, 3], name="a"),
                CategoricalIndex([1, 2, 3], name="a"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("test_series", [True, False])
def test_unobserved_in_index(
    keys: Union[str, List[str]],
    expected_values: List[int],
    expected_index_levels: Union[List[CategoricalIndex], CategoricalIndex],
    test_series: bool,
) -> None:
    df: DataFrame = DataFrame(
        {
            "a": Categorical([1, 1, 2], categories=[1, 2, 3]),
            "a2": Categorical([1, 1, 2], categories=[1, 2, 3]),
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        }
    ).set_index(["a", "a2"])
    if "b" not in keys:
        df = df.drop(columns="b")
    gb: pd.core.groupby.DataFrameGroupBy = df.groupby(keys, observed=False)
    if test_series:
        gb = gb["c"]
    result: Union[Series, DataFrame] = gb.sum()
    if len(keys) == 1:
        index: Union[List[CategoricalIndex], CategoricalIndex] = expected_index_levels  # type: ignore[misc]
    else:
        codes: List[np.ndarray] = [
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 1, 2] * 3),
        ]
        index = MultiIndex(
            expected_index_levels, codes=codes, names=keys
        )
    expected = DataFrame({"c": expected_values}, index=index)  # type: ignore[misc]
    if test_series:
        expected = expected["c"]  # type: ignore[misc]
    tm.assert_equal(result, expected)


def test_observed_groups_with_nan(observed: bool) -> None:
    df: DataFrame = DataFrame(
        {
            "cat": Categorical(["a", np.nan, "a"], categories=["a", "b", "d"]),
            "vals": [1, 2, 3],
        }
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("cat", observed=observed)
    result: Dict[str, Index] = g.groups
    if observed:
        expected: Dict[str, Index] = {"a": Index([0, 2], dtype="int64")}
    else:
        expected = {
            "a": Index([0, 2], dtype="int64"),
            "b": Index([], dtype="int64"),
            "d": Index([], dtype="int64"),
        }
    tm.assert_dict_equal(result, expected)


def test_observed_nth() -> None:
    cat: Categorical = Categorical(
        ["a", np.nan, np.nan], categories=["a", "b", "c"]
    )
    ser: Series = Series([1, 2, 3])
    df: DataFrame = DataFrame({"cat": cat, "ser": ser})
    result: Series = df.groupby("cat", observed=False)["ser"].nth(0)
    expected: Series = df["ser"].iloc[[0]]
    tm.assert_series_equal(result, expected)


def test_dataframe_categorical_with_nan(observed: bool) -> None:
    s1: Categorical = Categorical([np.nan, "a", np.nan, "a"], categories=["a", "b", "c"])
    s2: Series = Series([1, 2, 3, 4])
    df: DataFrame = DataFrame({"s1": s1, "s2": s2})
    result: DataFrame = df.groupby("s1", observed=observed).first().reset_index()
    if observed:
        expected: DataFrame = DataFrame(
            {
                "s1": Categorical(
                    ["a"], categories=["a", "b", "c"], ordered=False
                ),
                "s2": [2],
            }
        )
    else:
        expected = DataFrame(
            {
                "s1": Categorical(
                    ["a", "b", "c"], categories=["a", "b", "c"], ordered=False
                ),
                "s2": [2, np.nan, np.nan],
            }
        )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
def test_dataframe_categorical_ordered_observed_sort(
    ordered: bool, observed: bool, sort: bool
) -> None:
    label: Categorical = Categorical(
        ["d", "a", "b", "a", "d", "b"],
        categories=["a", "b", "missing", "d"],
        ordered=ordered,
    )
    val: Series = Series(["d", "a", "b", "a", "d", "b"])
    df: DataFrame = DataFrame({"label": label, "val": val})
    result: Series = df.groupby(
        "label", observed=observed, sort=sort
    )["val"].aggregate("first")
    label_series: Series = Series(result.index.array, dtype="object")
    aggr: Series = Series(result.array)
    if not observed:
        aggr[aggr.isna()] = "missing"
    if not all(label_series == aggr):
        msg: str = (
            f"Labels and aggregation results not consistently sorted\nfor "
            f"(ordered={ordered}, observed={observed}, sort={sort})\nResult:\n{result}"
        )
        pytest.fail(msg)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(["artist", "medium"], observed=False)["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
def test_preserve_on_ordered_ops(
    func: str, values: List[str], ordered: bool
) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series(values, dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_observed(request: Any, using_infer_string: bool, observed: bool) -> None:
    if using_infer_string and not observed:
        request.applymarker(pytest.mark.xfail(reason="TODO(infer_string)"))
    cat1: Categorical = Categorical(
        ["a", "a", "b", "b"],
        categories=["a", "b", "z"],
        ordered=True,
    )
    cat2: Categorical = Categorical(
        ["c", "d", "c", "d"],
        categories=["c", "d", "y"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        {"A": cat1, "B": cat2, "values": [1, 2, 3, 4]}
    )
    df["C"] = ["foo", "bar"] * 2
    gb: pd.core.groupby.DataFrameGroupBy = df.groupby(
        ["A", "B", "C"], observed=observed
    )
    exp_index: MultiIndex = MultiIndex.from_arrays(
        [cat1, cat2, ["foo", "bar"] * 2], names=["A", "B", "C"]
    )
    expected: DataFrame = DataFrame(
        {"values": Series([1, 2, 3, 4], index=exp_index)}
    ).sort_index()
    result: DataFrame = gb.sum()
    if not observed:
        expected = cartesian_product_for_groupers(
            expected,
            [cat1, cat2, ["foo", "bar"]],
            list("ABC"),
            fill_value=0,
        )
    tm.assert_frame_equal(result, expected)
    gb = df.groupby(["A", "B"], observed=observed)
    exp_index = MultiIndex.from_arrays([cat1, cat2], names=["A", "B"])
    expected = DataFrame(
        {"values": [1, 2, 3, 4], "C": ["foo", "bar", "foo", "bar"]},
        index=exp_index,
    )
    result = gb.sum()
    if not observed:
        expected = cartesian_product_for_groupers(
            expected, [cat1, cat2], list("AB"), fill_value=0
        )
    tm.assert_frame_equal(result, expected)


def test_observed_perf() -> None:
    df: DataFrame = DataFrame(
        {
            "cat": np.random.default_rng(2).integers(0, 255, size=30000),
            "int_id": np.random.default_rng(2).integers(0, 255, size=30000),
            "other_id": np.random.default_rng(2).integers(0, 10000, size=30000),
            "foo": 0,
        }
    )
    df["cat"] = df.cat.astype(str).astype("category")
    grouped: pd.core.groupby.DataFrameGroupBy = df.groupby(
        ["cat", "int_id", "other_id"], observed=True
    )
    result: DataFrame = grouped.count()
    assert result.index.levels[0].nunique() == df.cat.nunique()
    assert result.index.levels[1].nunique() == df.int_id.nunique()
    assert result.index.levels[2].nunique() == df.other_id.nunique()


def test_observed_groups(observed: bool) -> None:
    cat: Categorical = Categorical(["a", "c", "a"], categories=["a", "b", "d"])
    df: DataFrame = DataFrame({"cat": cat, "vals": [1, 2, 3]})
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("cat", observed=observed)
    result: Dict[str, Index] = g.groups
    if observed:
        expected: Dict[str, Index] = {
            "a": Index([0, 2], dtype="int64"),
            "c": Index([1], dtype="int64"),
        }
    else:
        expected = {
            "a": Index([0, 2], dtype="int64"),
            "b": Index([], dtype="int64"),
            "c": Index([1], dtype="int64"),
        }
    tm.assert_dict_equal(result, expected)


@pytest.mark.parametrize(
    "keys, expected_values, expected_index_levels",
    [
        (
            "a",
            [15, 9, 0],
            CategoricalIndex([1, 2, 3], name="a"),
        ),
        (
            ["a", "b"],
            [7, 8, 0, 0, 0, 9, 0, 0, 0],
            [
                CategoricalIndex([1, 2, 3], name="a"),
                Index([4, 5, 6]),
            ],
        ),
        (
            ["a", "a2"],
            [15, 0, 0, 0, 9, 0, 0, 0, 0],
            [
                CategoricalIndex([1, 2, 3], name="a"),
                CategoricalIndex([1, 2, 3], name="a"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("test_series", [True, False])
def test_unobserved_in_index(
    keys: Union[str, List[str]],
    expected_values: List[int],
    expected_index_levels: Union[List[CategoricalIndex], CategoricalIndex],
    test_series: bool,
) -> None:
    df: DataFrame = DataFrame(
        {
            "a": Categorical([1, 1, 2], categories=[1, 2, 3]),
            "a2": Categorical([1, 1, 2], categories=[1, 2, 3]),
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        }
    ).set_index(["a", "a2"])
    if "b" not in keys:
        df = df.drop(columns="b")
    gb: pd.core.groupby.DataFrameGroupBy = df.groupby(keys, observed=False)
    if test_series:
        gb = gb["c"]
    result: Union[Series, DataFrame] = gb.sum()
    if len(keys) == 1:
        index: Union[List[CategoricalIndex], CategoricalIndex] = expected_index_levels  # type: ignore[misc]
    else:
        codes: List[np.ndarray] = [
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 1, 2] * 3),
        ]
        index = MultiIndex(
            expected_index_levels, codes=codes, names=keys
        )
    expected = DataFrame({"c": expected_values}, index=index)  # type: ignore[misc]
    if test_series:
        expected = expected["c"]  # type: ignore[misc]
    tm.assert_equal(result, expected)


def test_observed_groups_with_nan(observed: bool) -> None:
    df: DataFrame = DataFrame(
        {
            "cat": Categorical(["a", np.nan, "a"], categories=["a", "b", "d"]),
            "vals": [1, 2, 3],
        }
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("cat", observed=observed)
    result: Dict[str, Index] = g.groups
    if observed:
        expected: Dict[str, Index] = {"a": Index([0, 2], dtype="int64")}
    else:
        expected = {
            "a": Index([0, 2], dtype="int64"),
            "b": Index([], dtype="int64"),
            "d": Index([], dtype="int64"),
        }
    tm.assert_dict_equal(result, expected)


def test_observed_nth() -> None:
    cat: Categorical = Categorical(
        ["a", np.nan, np.nan], categories=["a", "b", "c"]
    )
    ser: Series = Series([1, 2, 3])
    df: DataFrame = DataFrame({"cat": cat, "ser": ser})
    result: Series = df.groupby("cat", observed=False)["ser"].nth(0)
    expected: Series = df["ser"].iloc[[0]]
    tm.assert_series_equal(result, expected)


def test_dataframe_categorical_with_nan(observed: bool) -> None:
    s1: Categorical = Categorical(
        [np.nan, "a", np.nan, "a"], categories=["a", "b", "c"]
    )
    s2: Series = Series([1, 2, 3, 4])
    df: DataFrame = DataFrame({"s1": s1, "s2": s2})
    result: DataFrame = df.groupby("s1", observed=observed).first().reset_index()
    if observed:
        expected: DataFrame = DataFrame(
            {
                "s1": Categorical(
                    ["a"], categories=["a", "b", "c"], ordered=False
                ),
                "s2": [2],
            }
        )
    else:
        expected = DataFrame(
            {
                "s1": Categorical(
                    ["a", "b", "c"], categories=["a", "b", "c"], ordered=False
                ),
                "s2": [2, np.nan, np.nan],
            }
        )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
def test_dataframe_categorical_ordered_observed_sort(
    ordered: bool, observed: bool, sort: bool
) -> None:
    label: Categorical = Categorical(
        ["d", "a", "b", "a", "d", "b"],
        categories=["a", "b", "missing", "d"],
        ordered=ordered,
    )
    val: Series = Series(["d", "a", "b", "a", "d", "b"])
    df: DataFrame = DataFrame({"label": label, "val": val})
    result: Series = df.groupby(
        "label", observed=observed, sort=sort
    )["val"].aggregate("first")
    label_series: Series = Series(result.index.array, dtype="object")
    aggr: Series = Series(result.array)
    if not observed:
        aggr[aggr.isna()] = "missing"
    if not all(label_series == aggr):
        msg: str = (
            f"Labels and aggregation results not consistently sorted\nfor "
            f"(ordered={ordered}, observed={observed}, sort={sort})\nResult:\n{result}"
        )
        pytest.fail(msg)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(["artist", "medium"], observed=False)["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, values: List[str], ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series(values, dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series(func, dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, expected_values: List[str], ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series(expected_values, dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series([func], dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series([func], dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series([func], dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series([func], dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series([func], dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series(["first"], dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series([func], dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series([func], dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series([func], dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series([func], dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series([func], dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series([func], dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series([func], dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("func", ["first", "last"])
def test_preserve_on_ordered_ops(func: str, ordered: bool) -> None:
    c: Categorical = Categorical(
        ["first", "second", "third", "fourth"], ordered=True
    )
    df: DataFrame = DataFrame(
        {"payload": [-1, -2, -1, -2], "col": c}
    )
    g: pd.core.groupby.DataFrameGroupBy = df.groupby("payload")
    result: DataFrame = getattr(g, func)()
    expected: DataFrame = DataFrame(
        {"payload": [-2, -1], "col": Series([func], dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)
    sgb: pd.core.groupby.SeriesGroupBy = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_datetime() -> None:
    levels: pd.DatetimeIndex = pd.date_range("2014-01-01", periods=4)
    codes: np.ndarray = np.random.default_rng(2).integers(0, 4, size=10)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    data: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result: DataFrame = data.groupby(cats, observed=False).mean()
    expected: DataFrame = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )
    tm.assert_frame_equal(result, expected)
    grouped: pd.core.groupby.DataFrameGroupBy = data.groupby(cats, observed=False)
    desc_result: DataFrame = grouped.describe()
    idx: np.ndarray = cats.codes.argsort()
    ord_labels: np.ndarray = cats.take(idx)
    ord_data: DataFrame = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )
    expc: Categorical = Categorical.from_codes(
        np.arange(4).repeat(8), levels, ordered=True
    )
    exp: CategoricalIndex = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def test_categorical_index() -> None:
    s: np.random.Generator = np.random.default_rng(2)
    levels: List[str] = ["foo", "bar", "baz", "qux"]
    codes: np.ndarray = s.integers(0, 4, size=20)
    cats: Categorical = Categorical.from_codes(codes, levels, ordered=True)
    df: DataFrame = DataFrame(
        np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd")
    )
    df["cats"] = cats
    result: DataFrame = df.set_index("cats").groupby(
        level=0, observed=False
    ).sum()
    expected: DataFrame = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(
        cats.codes, observed=False
    ).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True),
        name="cats",
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns() -> None:
    cats: CategoricalIndex = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((20, 4)), columns=cats
    )
    result: DataFrame = df.groupby([1, 2, 3, 4] * 5).describe()
    tm.assert_index_equal(result.stack().columns, cats)
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical() -> None:
    df: DataFrame = DataFrame(
        {
            "a": range(10),
            "medium": ["A", "B"] * 5,
            "artist": list("XYXXY") * 2,
        }
    )
    df["medium"] = df["medium"].astype("category")
    gcat: DataFrame = df.groupby(
        ["artist", "medium"], observed=False
    )["a"].count().unstack()
    result: DataFrame = gcat.describe()
    exp_columns: CategoricalIndex = CategoricalIndex(
        ["A", "B"], ordered=False, name="medium"
    )
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)
    result = gcat["A"] + gcat["B"]
    expected: Series = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "operation, expected_values",
    [
        ("agg", [15, 9, 0]),
        ("apply", [15, 9, 0]),
    ],
)
def test_groupby_agg_observed_true_single_column(
    as_index: bool,
    expected: Series,
    operation: str,
) -> None:
    df: DataFrame = DataFrame(
        {"a": [1, 1, 2], "b": [1, 1, 2], "x": [1, 2, 3]}
    ).set_index(["a", "b"])
    result: Series = getattr(df.groupby(["a", "b"], as_index=True, observed=True)['x'], operation)()
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "fill_value",
    [None, np.nan, pd.NaT],
)
def test_shift(fill_value: Optional[Union[float, Any]]) -> None:
    ct: Categorical = Categorical(
        ["a", "b", "c", "d"], categories=["a", "b", "c", "d"], ordered=False
    )
    expected: Categorical = Categorical(
        [None, "a", "b", "c"],
        categories=["a", "b", "c", "d"],
        ordered=False,
    )
    res: Categorical = ct.shift(1, fill_value=fill_value)
    tm.assert_equal(res, expected)


@pytest.fixture
def df_cat(df: DataFrame) -> DataFrame:
    """
    DataFrame with multiple categorical columns and a column of integers.
    Shortened so as not to contain all possible combinations of categories.
    Useful for testing `observed` kwarg functionality on GroupBy objects.

    Parameters
    ----------
    df: DataFrame
        Non-categorical, longer DataFrame from another fixture, used to derive
        this one

    Returns
    -------
    df_cat: DataFrame
    """
    df_cat: DataFrame = df.copy()[:4]
    df_cat["A"] = df_cat["A"].astype("category")
    df_cat["B"] = df_cat["B"].astype("category")
    df_cat["C"] = Series([1, 2, 3, 4])
    df_cat = df_cat.drop(["D"], axis=1)
    return df_cat


@pytest.mark.parametrize(
    "operation",
    ["agg", "apply"],
)
def test_seriesgroupby_observed_true(
    df_cat: DataFrame, operation: str
) -> None:
    lev_a: Index = Index(
        ["bar", "bar", "foo", "foo"],
        dtype=df_cat["A"].dtype,
        name="A",
    )
    lev_b: Index = Index(
        ["one", "three", "one", "two"],
        dtype=df_cat["B"].dtype,
        name="B",
    )
    index: MultiIndex = MultiIndex.from_arrays([lev_a, lev_b])
    expected: Series = Series(
        data=[2, 4, 1, 3], index=index, name="C"
    ).sort_index()
    grouped: pd.core.groupby.SeriesGroupBy = df_cat.groupby(
        ["A", "B"], observed=True
    )["C"]
    result: Series = getattr(grouped, operation)(sum)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "operation",
    ["agg", "apply"],
)
@pytest.mark.parametrize(
    "observed",
    [False, None],
)
def test_seriesgroupby_observed_false_or_none(
    df_cat: DataFrame,
    observed: bool,
    operation: str,
) -> None:
    index: MultiIndex = MultiIndex.from_product(
        [
            CategoricalIndex(["bar", "foo"], ordered=False),
            CategoricalIndex(["one", "three", "two"], ordered=False),
            Index(["min", "max"]),
        ],
        names=["A", "B", None],
    )
    expected: Series = Series(
        data=[2, 4, 0, 1, 0, 3],
        index=index,
        name="C",
    )
    grouped: pd.core.groupby.SeriesGroupBy = df_cat.groupby(
        ["A", "B"], observed=observed
    )["C"]
    result: Series = getattr(grouped, operation)(sum)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "observed, index, data",
    [
        (
            True,
            MultiIndex.from_arrays(
                [
                    Index(["bar"] * 4 + ["foo"] * 4, dtype="category", name="A"),
                    Index(["one", "one", "three", "three", "one", "one", "two", "two"], dtype="category", name="B"),
                    Index(["min", "max"] * 4),
                ]
            ),
            [2, 2, 4, 4, 1, 1, 3, 3],
        ),
        (
            False,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
        (
            None,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
    ],
)
def test_seriesgroupby_observed_apply_dict(
    df_cat: DataFrame,
    observed: bool,
    index: MultiIndex,
    data: List[Union[int, float]],
) -> None:
    expected: Series = Series(
        data=data, index=index, name="C"
    )
    result: Series = df_cat.groupby(["A", "B"], observed=observed)["C"].apply(
        lambda x: {"min": x.min(), "max": x.max()}
    )
    tm.assert_series_equal(result, expected)


def test_groupby_categorical_series_dataframe_consistent(
    df_cat: DataFrame,
) -> None:
    expected: Series = df_cat.groupby(["A", "B"], observed=False)["C"].mean()
    result: Series = df_cat.groupby(["A", "B"], observed=False).mean()["C"]
    tm.assert_series_equal(result, expected)


def test_groupby_cat_preserves_structure(
    observed: bool, ordered: bool
) -> None:
    df: DataFrame = DataFrame(
        {"Name": Categorical(["Bob", "Greg"], ordered=ordered), "Item": [1, 2]}
    )
    expected: DataFrame = df.copy()
    result: DataFrame = df.groupby("Name", observed=observed).agg(
        DataFrame.sum, skipna=True
    ).reset_index()
    tm.assert_frame_equal(result, expected)


def test_get_nonexistent_category() -> None:
    df: DataFrame = DataFrame(
        {"var": ["a", "a", "b", "b"], "val": range(4)}
    )
    with pytest.raises(KeyError, match="'vau'"):
        df.groupby("var").apply(
            lambda rows: DataFrame({"val": [rows.iloc[-1]["vau"]]})
        )


def test_series_groupby_on_2_categoricals_unobserved(
    reduction_func: str, observed: bool
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C", "D"]),
            "cat_2": Categorical(["A", "B", "A", "B"], categories=["A", "B", "C", "D"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    series_groupby: pd.core.groupby.SeriesGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=observed
    )["value"]
    if reduction_func == "corrwith":
        assert not hasattr(series_groupby, reduction_func)
        return
    agg: Callable = getattr(series_groupby, reduction_func)
    if not observed and reduction_func in ["idxmin", "idxmax"]:
        with pytest.raises(ValueError, match="empty group due to unobserved categories"):
            agg(*args)
        return
    result: Union[Series, DataFrame] = agg(*args)
    assert len(result) == (4 if observed else 16)


def test_series_groupby_on_2_categoricals_unobserved_zeroes_or_nans(
    reduction_func: str, request: Any
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")
    if reduction_func == "corrwith":
        mark = pytest.mark.xfail(reason="TODO: implemented SeriesGroupBy.corrwith. See GH 32293")
        request.applymarker(mark)
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C"]),
            "cat_2": Categorical(["A", "B", "A", "B"], categories=["A", "B", "C"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved: List[Tuple[str, str]] = [("A", "C"), ("B", "C"), ("C", "A"), ("C", "B"), ("C", "C")]
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    series_groupby: pd.core.groupby.SeriesGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=False
    )["value"]
    agg: Callable = getattr(series_groupby, reduction_func)
    if reduction_func in ["idxmin", "idxmax"]:
        with pytest.raises(ValueError, match="empty group due to unobserved categories"):
            agg(*args)
        return
    result: Union[Series, DataFrame] = agg(*args)
    missing_fillin: Union[float, int, Any] = _results_for_groupbys_with_missing_categories[reduction_func]
    for idx in unobserved:
        val: Any = result.loc[idx]
        if pd.isna(missing_fillin) and pd.isna(val):
            continue
        assert val == missing_fillin
    if missing_fillin == 0:
        if reduction_func in ["count", "nunique", "size"]:
            assert np.issubdtype(result.dtype, np.integer)
        else:
            assert reduction_func in ["sum", "any"]


def test_dataframe_groupby_on_2_categoricals_when_observed_is_true(
    reduction_func: str
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup does not return the Categories on the index")
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C"]),
            "cat_2": Categorical(["1", "1", "1", "1"], categories=["1", "2"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved_cats: List[Tuple[str, str]] = [
        ("A", "2"),
        ("B", "2"),
        ("C", "1"),
        ("C", "2"),
    ]
    df_grp: pd.core.groupby.DataFrameGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=True
    )
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    if reduction_func == "corrwith":
        warn: Optional[Warning] = FutureWarning
        warn_msg: str = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        warn_msg = ""
    with tm.assert_produces_warning(warn, match=warn_msg):
        res: Union[Series, DataFrame] = getattr(df_grp, reduction_func)(*args)
    for cat in unobserved_cats:
        assert cat not in res.index


@pytest.mark.parametrize(
    "observed, index, data",
    [
        (
            True,
            MultiIndex.from_arrays(
                [
                    Index(["bar"] * 4 + ["foo"] * 4, dtype="category", name="A"),
                    Index(["one", "one", "three", "three", "one", "one", "two", "two"], dtype="category", name="B"),
                    Index(["min", "max"] * 4),
                ]
            ),
            [2, 2, 4, 4, 1, 1, 3, 3],
        ),
        (
            False,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
        (
            None,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
    ],
)
def test_seriesgroupby_observed_apply_dict(
    df_cat: DataFrame,
    observed: bool,
    index: MultiIndex,
    data: List[Union[int, float]],
) -> None:
    expected: Series = Series(
        data=data, index=index, name="C"
    )
    result: Series = df_cat.groupby(["A", "B"], observed=observed)["C"].apply(
        lambda x: {"min": x.min(), "max": x.max()}
    )
    tm.assert_series_equal(result, expected)


def test_groupby_categorical_series_dataframe_consistent(
    df_cat: DataFrame,
) -> None:
    expected: Series = df_cat.groupby(["A", "B"], observed=False)["C"].mean()
    result: Series = df_cat.groupby(["A", "B"], observed=False).mean()["C"]
    tm.assert_series_equal(result, expected)


def test_groupby_cat_preserves_structure(
    observed: bool, ordered: bool
) -> None:
    df: DataFrame = DataFrame(
        {"Name": Categorical(["Bob", "Greg"], ordered=ordered), "Item": [1, 2]}
    )
    expected: DataFrame = df.copy()
    result: DataFrame = df.groupby("Name", observed=observed).agg(
        DataFrame.sum, skipna=True
    ).reset_index()
    tm.assert_frame_equal(result, expected)


def test_get_nonexistent_category() -> None:
    df: DataFrame = DataFrame(
        {"var": ["a", "a", "b", "b"], "val": range(4)}
    )
    with pytest.raises(KeyError, match="'vau'"):
        df.groupby("var").apply(
            lambda rows: DataFrame({"val": [rows.iloc[-1]["vau"]]})
        )


def test_series_groupby_on_2_categoricals_unobserved(
    reduction_func: str, observed: bool
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C", "D"]),
            "cat_2": Categorical(["A", "B", "A", "B"], categories=["A", "B", "C", "D"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    series_groupby: pd.core.groupby.SeriesGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=observed
    )["value"]
    if reduction_func == "corrwith":
        assert not hasattr(series_groupby, reduction_func)
        return
    agg: Callable = getattr(series_groupby, reduction_func)
    if not observed and reduction_func in ["idxmin", "idxmax"]:
        with pytest.raises(ValueError, match="empty group due to unobserved categories"):
            agg(*args)
        return
    result: Union[Series, DataFrame] = agg(*args)
    assert len(result) == (4 if observed else 16)


def test_series_groupby_on_2_categoricals_unobserved_zeroes_or_nans(
    reduction_func: str, request: Any
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")
    if reduction_func == "corrwith":
        mark = pytest.mark.xfail(reason="TODO: implemented SeriesGroupBy.corrwith. See GH 32293")
        request.applymarker(mark)
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C"]),
            "cat_2": Categorical(["A", "B", "A", "B"], categories=["A", "B", "C"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved_cats: List[Tuple[str, str]] = [("A", "C"), ("B", "C"), ("C", "A"), ("C", "B"), ("C", "C")]
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    series_groupby: pd.core.groupby.SeriesGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=False
    )["value"]
    agg: Callable = getattr(series_groupby, reduction_func)
    result: Union[Series, DataFrame] = agg(*args)
    missing_fillin: Union[float, int, Any] = _results_for_groupbys_with_missing_categories[reduction_func]
    for idx in unobserved_cats:
        val: Any = result.loc[idx]
        if pd.isna(missing_fillin) and pd.isna(val):
            continue
        assert val == missing_fillin
    if missing_fillin == 0:
        if reduction_func in ["count", "nunique", "size"]:
            assert np.issubdtype(result.dtype, np.integer)
        else:
            assert reduction_func in ["sum", "any"]


def test_dataframe_groupby_on_2_categoricals_when_observed_is_true(
    reduction_func: str
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup does not return the Categories on the index")
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C"]),
            "cat_2": Categorical(["1", "1", "1", "1"], categories=["1", "2"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved_cats: List[Tuple[str, str]] = [
        ("A", "2"),
        ("B", "2"),
        ("C", "1"),
        ("C", "2"),
    ]
    df_grp: pd.core.groupby.DataFrameGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=True
    )
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    if reduction_func == "corrwith":
        warn: Optional[Warning] = FutureWarning
        warn_msg: str = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        warn_msg = ""
    with tm.assert_produces_warning(warn, match=warn_msg):
        res: Union[Series, DataFrame] = getattr(df_grp, reduction_func)(*args)
    for cat in unobserved_cats:
        assert cat not in res.index


@pytest.mark.parametrize(
    "observed, index, data",
    [
        (
            True,
            MultiIndex.from_arrays(
                [
                    Index(["bar"] * 4 + ["foo"] * 4, dtype="category", name="A"),
                    Index(["one", "one", "three", "three", "one", "one", "two", "two"], dtype="category", name="B"),
                    Index(["min", "max"] * 4),
                ]
            ),
            [2, 2, 4, 4, 1, 1, 3, 3],
        ),
        (
            False,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
        (
            None,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
    ],
)
def test_seriesgroupby_observed_apply_dict(
    df_cat: DataFrame,
    observed: bool,
    index: MultiIndex,
    data: List[Union[int, float]],
) -> None:
    expected: Series = Series(
        data=data, index=index, name="C"
    )
    result: Series = df_cat.groupby(["A", "B"], observed=observed)["C"].apply(
        lambda x: {"min": x.min(), "max": x.max()}
    )
    tm.assert_series_equal(result, expected)


def test_groupby_categorical_series_dataframe_consistent(
    df_cat: DataFrame,
) -> None:
    expected: Series = df_cat.groupby(["A", "B"], observed=False)["C"].mean()
    result: Series = df_cat.groupby(["A", "B"], observed=False).mean()["C"]
    tm.assert_series_equal(result, expected)


def test_groupby_cat_preserves_structure(
    observed: bool, ordered: bool
) -> None:
    df: DataFrame = DataFrame(
        {"Name": Categorical(["Bob", "Greg"], ordered=ordered), "Item": [1, 2]}
    )
    expected: DataFrame = df.copy()
    result: DataFrame = df.groupby("Name", observed=observed).agg(
        DataFrame.sum, skipna=True
    ).reset_index()
    tm.assert_frame_equal(result, expected)


def test_get_nonexistent_category() -> None:
    df: DataFrame = DataFrame(
        {"var": ["a", "a", "b", "b"], "val": range(4)}
    )
    with pytest.raises(KeyError, match="'vau'"):
        df.groupby("var").apply(
            lambda rows: DataFrame({"val": [rows.iloc[-1]["vau"]]})
        )


def test_series_groupby_on_2_categoricals_unobserved(
    reduction_func: str, observed: bool
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C", "D"]),
            "cat_2": Categorical(["A", "B", "A", "B"], categories=["A", "B", "C", "D"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    series_groupby: pd.core.groupby.SeriesGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=observed
    )["value"]
    if reduction_func == "corrwith":
        assert not hasattr(series_groupby, reduction_func)
        return
    agg: Callable = getattr(series_groupby, reduction_func)
    if not observed and reduction_func in ["idxmin", "idxmax"]:
        with pytest.raises(ValueError, match="empty group due to unobserved categories"):
            agg(*args)
        return
    result: Union[Series, DataFrame] = agg(*args)
    assert len(result) == (4 if observed else 16)


def test_series_groupby_on_2_categoricals_unobserved_zeroes_or_nans(
    reduction_func: str, request: Any
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")
    if reduction_func == "corrwith":
        mark = pytest.mark.xfail(reason="TODO: implemented SeriesGroupBy.corrwith. See GH 32293")
        request.applymarker(mark)
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C"]),
            "cat_2": Categorical(["A", "B", "A", "B"], categories=["A", "B", "C"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved_cats: List[Tuple[str, str]] = [("A", "C"), ("B", "C"), ("C", "A"), ("C", "B"), ("C", "C")]
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    series_groupby: pd.core.groupby.SeriesGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=False
    )["value"]
    agg: Callable = getattr(series_groupby, reduction_func)
    result: Union[Series, DataFrame] = agg(*args)
    missing_fillin: Union[float, int, Any] = _results_for_groupbys_with_missing_categories[reduction_func]
    for idx in unobserved_cats:
        val: Any = result.loc[idx]
        if pd.isna(missing_fillin) and pd.isna(val):
            continue
        assert val == missing_fillin
    if missing_fillin == 0:
        if reduction_func in ["count", "nunique", "size"]:
            assert np.issubdtype(result.dtype, np.integer)
        else:
            assert reduction_func in ["sum", "any"]


def test_dataframe_groupby_on_2_categoricals_when_observed_is_true(
    reduction_func: str
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup does not return the Categories on the index")
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C"]),
            "cat_2": Categorical(["1", "1", "1", "1"], categories=["1", "2"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved_cats: List[Tuple[str, str]] = [
        ("A", "2"),
        ("B", "2"),
        ("C", "1"),
        ("C", "2"),
    ]
    df_grp: pd.core.groupby.DataFrameGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=True
    )
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    if reduction_func == "corrwith":
        warn: Optional[Warning] = FutureWarning
        warn_msg: str = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        warn_msg = ""
    with tm.assert_produces_warning(warn, match=warn_msg):
        res: Union[Series, DataFrame] = getattr(df_grp, reduction_func)(*args)
    for cat in unobserved_cats:
        assert cat not in res.index


@pytest.mark.parametrize(
    "observed, index, data",
    [
        (
            True,
            MultiIndex.from_arrays(
                [
                    Index(["bar"] * 4 + ["foo"] * 4, dtype="category", name="A"),
                    Index(["one", "one", "three", "three", "one", "one", "two", "two"], dtype="category", name="B"),
                    Index(["min", "max"] * 4),
                ]
            ),
            [2, 2, 4, 4, 1, 1, 3, 3],
        ),
        (
            False,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
        (
            None,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
    ],
)
def test_seriesgroupby_observed_apply_dict(
    df_cat: DataFrame,
    observed: bool,
    index: MultiIndex,
    data: List[Union[int, float]],
) -> None:
    expected: Series = Series(
        data=data, index=index, name="C"
    )
    result: Series = df_cat.groupby(["A", "B"], observed=observed)["C"].apply(
        lambda x: {"min": x.min(), "max": x.max()}
    )
    tm.assert_series_equal(result, expected)


def test_groupby_categorical_series_dataframe_consistent(
    df_cat: DataFrame,
) -> None:
    expected: Series = df_cat.groupby(["A", "B"], observed=False)["C"].mean()
    result: Series = df_cat.groupby(["A", "B"], observed=False).mean()["C"]
    tm.assert_series_equal(result, expected)


def test_groupby_cat_preserves_structure(
    observed: bool, ordered: bool
) -> None:
    df: DataFrame = DataFrame(
        {"Name": Categorical(["Bob", "Greg"], ordered=ordered), "Item": [1, 2]}
    )
    expected: DataFrame = df.copy()
    result: DataFrame = df.groupby("Name", observed=observed).agg(
        DataFrame.sum, skipna=True
    ).reset_index()
    tm.assert_frame_equal(result, expected)


def test_get_nonexistent_category() -> None:
    df: DataFrame = DataFrame(
        {"var": ["a", "a", "b", "b"], "val": range(4)}
    )
    with pytest.raises(KeyError, match="'vau'"):
        df.groupby("var").apply(
            lambda rows: DataFrame({"val": [rows.iloc[-1]["vau"]]})
        )


def test_series_groupby_on_2_categoricals_unobserved(
    reduction_func: str, observed: bool
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C", "D"]),
            "cat_2": Categorical(["A", "B", "A", "B"], categories=["A", "B", "C", "D"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    series_groupby: pd.core.groupby.SeriesGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=observed
    )["value"]
    if reduction_func == "corrwith":
        assert not hasattr(series_groupby, reduction_func)
        return
    agg: Callable = getattr(series_groupby, reduction_func)
    if not observed and reduction_func in ["idxmin", "idxmax"]:
        with pytest.raises(ValueError, match="empty group due to unobserved categories"):
            agg(*args)
        return
    result: Union[Series, DataFrame] = agg(*args)
    assert len(result) == (4 if observed else 16)


def test_series_groupby_on_2_categoricals_unobserved_zeroes_or_nans(
    reduction_func: str, request: Any
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")
    if reduction_func == "corrwith":
        mark = pytest.mark.xfail(reason="TODO: implemented SeriesGroupBy.corrwith. See GH 32293")
        request.applymarker(mark)
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C"]),
            "cat_2": Categorical(["A", "B", "A", "B"], categories=["A", "B", "C"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved_cats: List[Tuple[str, str]] = [("A", "C"), ("B", "C"), ("C", "A"), ("C", "B"), ("C", "C")]
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    series_groupby: pd.core.groupby.SeriesGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=False
    )["value"]
    agg: Callable = getattr(series_groupby, reduction_func)
    result: Union[Series, DataFrame] = agg(*args)
    missing_fillin: Union[float, int, Any] = _results_for_groupbys_with_missing_categories[reduction_func]
    for idx in unobserved_cats:
        val: Any = result.loc[idx]
        if pd.isna(missing_fillin) and pd.isna(val):
            continue
        assert val == missing_fillin
    if missing_fillin == 0:
        if reduction_func in ["count", "nunique", "size"]:
            assert np.issubdtype(result.dtype, np.integer)
        else:
            assert reduction_func in ["sum", "any"]


def test_dataframe_groupby_on_2_categoricals_when_observed_is_true(
    reduction_func: str
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup does not return the Categories on the index")
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C"]),
            "cat_2": Categorical(["1", "1", "1", "1"], categories=["1", "2"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved_cats: List[Tuple[str, str]] = [
        ("A", "2"),
        ("B", "2"),
        ("C", "1"),
        ("C", "2"),
    ]
    df_grp: pd.core.groupby.DataFrameGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=True
    )
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    if reduction_func == "corrwith":
        warn: Optional[Warning] = FutureWarning
        warn_msg: str = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        warn_msg = ""
    with tm.assert_produces_warning(warn, match=warn_msg):
        res: Union[Series, DataFrame] = getattr(df_grp, reduction_func)(*args)
    for cat in unobserved_cats:
        assert cat not in res.index


@pytest.mark.parametrize(
    "observed, index, data",
    [
        (
            True,
            MultiIndex.from_arrays(
                [
                    Index(["bar"] * 4 + ["foo"] * 4, dtype="category", name="A"),
                    Index(["one", "one", "three", "three", "one", "one", "two", "two"], dtype="category", name="B"),
                    Index(["min", "max"] * 4),
                ]
            ),
            [2, 2, 4, 4, 1, 1, 3, 3],
        ),
        (
            False,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
        (
            None,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
    ],
)
def test_seriesgroupby_observed_apply_dict(
    df_cat: DataFrame,
    observed: bool,
    index: MultiIndex,
    data: List[Union[int, float]],
) -> None:
    expected: Series = Series(
        data=data, index=index, name="C"
    )
    result: Series = df_cat.groupby(["A", "B"], observed=observed)["C"].apply(
        lambda x: {"min": x.min(), "max": x.max()}
    )
    tm.assert_series_equal(result, expected)


def test_groupby_categorical_series_dataframe_consistent(
    df_cat: DataFrame,
) -> None:
    expected: Series = df_cat.groupby(["A", "B"], observed=False)["C"].mean()
    result: Series = df_cat.groupby(["A", "B"], observed=False).mean()["C"]
    tm.assert_series_equal(result, expected)


def test_groupby_cat_preserves_structure(
    observed: bool, ordered: bool
) -> None:
    df: DataFrame = DataFrame(
        {"Name": Categorical(["Bob", "Greg"], ordered=ordered), "Item": [1, 2]}
    )
    expected: DataFrame = df.copy()
    result: DataFrame = df.groupby("Name", observed=observed).agg(
        DataFrame.sum, skipna=True
    ).reset_index()
    tm.assert_frame_equal(result, expected)


def test_get_nonexistent_category() -> None:
    df: DataFrame = DataFrame(
        {"var": ["a", "a", "b", "b"], "val": range(4)}
    )
    with pytest.raises(KeyError, match="'vau'"):
        df.groupby("var").apply(
            lambda rows: DataFrame({"val": [rows.iloc[-1]["vau"]]})
        )


def test_series_groupby_on_2_categoricals_unobserved(
    reduction_func: str, observed: bool
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C", "D"]),
            "cat_2": Categorical(["A", "B", "A", "B"], categories=["A", "B", "C", "D"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    series_groupby: pd.core.groupby.SeriesGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=observed
    )["value"]
    if reduction_func == "corrwith":
        assert not hasattr(series_groupby, reduction_func)
        return
    agg: Callable = getattr(series_groupby, reduction_func)
    if not observed and reduction_func in ["idxmin", "idxmax"]:
        with pytest.raises(ValueError, match="empty group due to unobserved categories"):
            agg(*args)
        return
    result: Union[Series, DataFrame] = agg(*args)
    assert len(result) == (4 if observed else 16)


def test_series_groupby_on_2_categoricals_unobserved_zeroes_or_nans(
    reduction_func: str, request: Any
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")
    if reduction_func == "corrwith":
        mark = pytest.mark.xfail(reason="TODO: implemented SeriesGroupBy.corrwith. See GH 32293")
        request.applymarker(mark)
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C"]),
            "cat_2": Categorical(["A", "B", "A", "B"], categories=["A", "B", "C"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved_cats: List[Tuple[str, str]] = [("A", "C"), ("B", "C"), ("C", "A"), ("C", "B"), ("C", "C")]
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    series_groupby: pd.core.groupby.SeriesGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=False
    )["value"]
    agg: Callable = getattr(series_groupby, reduction_func)
    result: Union[Series, DataFrame] = agg(*args)
    missing_fillin: Union[float, int, Any] = _results_for_groupbys_with_missing_categories[reduction_func]
    for idx in unobserved_cats:
        val: Any = result.loc[idx]
        if pd.isna(missing_fillin) and pd.isna(val):
            continue
        assert val == missing_fillin
    if missing_fillin == 0:
        if reduction_func in ["count", "nunique", "size"]:
            assert np.issubdtype(result.dtype, np.integer)
        else:
            assert reduction_func in ["sum", "any"]


def test_dataframe_groupby_on_2_categoricals_when_observed_is_true(
    reduction_func: str
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup does not return the Categories on the index")
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C"]),
            "cat_2": Categorical(["1", "1", "1", "1"], categories=["1", "2"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved_cats: List[Tuple[str, str]] = [
        ("A", "2"),
        ("B", "2"),
        ("C", "1"),
        ("C", "2"),
    ]
    df_grp: pd.core.groupby.DataFrameGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=True
    )
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    if reduction_func == "corrwith":
        warn: Optional[Warning] = FutureWarning
        warn_msg: str = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        warn_msg = ""
    with tm.assert_produces_warning(warn, match=warn_msg):
        res: Union[Series, DataFrame] = getattr(df_grp, reduction_func)(*args)
    for cat in unobserved_cats:
        assert cat not in res.index


@pytest.mark.parametrize(
    "observed, index, data",
    [
        (
            True,
            MultiIndex.from_arrays(
                [
                    Index(["bar"] * 4 + ["foo"] * 4, dtype="category", name="A"),
                    Index(["one", "one", "three", "three", "one", "one", "two", "two"], dtype="category", name="B"),
                    Index(["min", "max"] * 4),
                ]
            ),
            [2, 2, 4, 4, 1, 1, 3, 3],
        ),
        (
            False,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
        (
            None,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
    ],
)
def test_seriesgroupby_observed_apply_dict(
    df_cat: DataFrame,
    observed: bool,
    index: MultiIndex,
    data: List[Union[int, float]],
) -> None:
    expected: Series = Series(
        data=data, index=index, name="C"
    )
    result: Series = df_cat.groupby(["A", "B"], observed=observed)["C"].apply(
        lambda x: {"min": x.min(), "max": x.max()}
    )
    tm.assert_series_equal(result, expected)


def test_groupby_categorical_series_dataframe_consistent(
    df_cat: DataFrame,
) -> None:
    expected: Series = df_cat.groupby(["A", "B"], observed=False)["C"].mean()
    result: Series = df_cat.groupby(["A", "B"], observed=False).mean()["C"]
    tm.assert_series_equal(result, expected)


def test_groupby_cat_preserves_structure(
    observed: bool, ordered: bool
) -> None:
    df: DataFrame = DataFrame(
        {"Name": Categorical(["Bob", "Greg"], ordered=ordered), "Item": [1, 2]}
    )
    expected: DataFrame = df.copy()
    result: DataFrame = df.groupby("Name", observed=observed).agg(
        DataFrame.sum, skipna=True
    ).reset_index()
    tm.assert_frame_equal(result, expected)


def test_get_nonexistent_category() -> None:
    df: DataFrame = DataFrame(
        {"var": ["a", "a", "b", "b"], "val": range(4)}
    )
    with pytest.raises(KeyError, match="'vau'"):
        df.groupby("var").apply(
            lambda rows: DataFrame({"val": [rows.iloc[-1]["vau"]]})
        )


def test_series_groupby_on_2_categoricals_unobserved(
    reduction_func: str, observed: bool
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C", "D"]),
            "cat_2": Categorical(["A", "B", "A", "B"], categories=["A", "B", "C", "D"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    series_groupby: pd.core.groupby.SeriesGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=observed
    )["value"]
    if reduction_func == "corrwith":
        assert not hasattr(series_groupby, reduction_func)
        return
    agg: Callable = getattr(series_groupby, reduction_func)
    if not observed and reduction_func in ["idxmin", "idxmax"]:
        with pytest.raises(ValueError, match="empty group due to unobserved categories"):
            agg(*args)
        return
    result: Union[Series, DataFrame] = agg(*args)
    assert len(result) == (4 if observed else 16)


def test_series_groupby_on_2_categoricals_unobserved_zeroes_or_nans(
    reduction_func: str, request: Any
) -> None:
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")
    if reduction_func == "corrwith":
        mark = pytest.mark.xfail(reason="TODO: implemented SeriesGroupBy.corrwith. See GH 32293")
        request.applymarker(mark)
    df: DataFrame = DataFrame(
        {
            "cat_1": Categorical(["A", "A", "B", "B"], categories=["A", "B", "C"]),
            "cat_2": Categorical(["A", "B", "A", "B"], categories=["A", "B", "C"]),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved_cats: List[Tuple[str, str]] = [("A", "C"), ("B", "C"), ("C", "A"), ("C", "B"), ("C", "C")]
    args: List[Any] = get_groupby_method_args(reduction_func, df)
    series_groupby: pd.core.groupby.SeriesGroupBy = df.groupby(
        ["cat_1", "cat_2"], observed=False
    )["value"]
    agg: Callable = getattr(series_groupby, reduction_func)
    result: Union[Series, DataFrame] = agg(*args)
    missing_fillin: Union[float, int, Any] = _results_for_groupbys_with_missing_categories[reduction_func]
    for idx in unobserved_cats:
        val: Any = result.loc[idx]
        if pd.isna(missing_fillin) and pd.isna(val):
            continue
        assert val == missing_fillin
    if missing_fillin == 0:
        if reduction_func in ["count", "nunique", "size"]:
            assert np.issubdtype(result.dtype, np.integer)
        else:
            assert reduction_func in ["sum", "any"]

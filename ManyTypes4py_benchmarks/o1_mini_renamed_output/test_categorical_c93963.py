from datetime import datetime
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


def func_ln7ovfd6(
    result: Union[pd.DataFrame, pd.Series],
    args: Iterable[Any],
    names: List[str],
    fill_value: Any = np.nan,
) -> Union[pd.DataFrame, pd.Series]:
    """Reindex to a cartesian production for the groupers,
    preserving the nature (Categorical) of each grouper
    """

    def func_pffbycyp(a: Any) -> Any:
        if isinstance(a, (CategoricalIndex, Categorical)):
            categories = a.categories
            a = Categorical.from_codes(
                np.arange(len(categories)),
                categories=categories,
                ordered=a.ordered,
            )
        return a

    index = MultiIndex.from_product(map(func_pffbycyp, args), names=names)
    return result.reindex(index, fill_value=fill_value).sort_index()


_results_for_groupbys_with_missing_categories: Dict[str, Any] = {
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


def func_i73gteal(df: pd.DataFrame) -> None:
    cats = qcut(df.C, 4)

    def func_51n4zqfi(group: pd.Series) -> Dict[str, Any]:
        return {
            "min": group.min(),
            "max": group.max(),
            "count": group.count(),
            "mean": group.mean(),
        }

    result = df.groupby(cats, observed=False).D.apply(get_groupby_method_args)
    assert result.index.names[0] == "C"


def func_ovvpmi8n() -> None:
    cats = Categorical(
        ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        categories=["a", "b", "c", "d"],
        ordered=True,
    )
    data = DataFrame({"a": [1, 1, 1, 2, 2, 2, 3, 4, 5], "b": cats})
    exp_index = CategoricalIndex(list("abcd"), name="b", ordered=True)
    expected = DataFrame({"a": [1, 2, 4, np.nan]}, index=exp_index)
    result = data.groupby("b", observed=False).mean()
    tm.assert_frame_equal(result, expected)


def func_g378453b() -> None:
    cat1 = Categorical(
        ["a", "a", "b", "b"], categories=["a", "b", "z"], ordered=True
    )
    cat2 = Categorical(
        ["c", "d", "c", "d"], categories=["c", "d", "y"], ordered=True
    )
    df = DataFrame({"A": cat1, "B": cat2, "values": [1, 2, 3, 4]})
    gb = df.groupby("A", observed=False)
    exp_idx = CategoricalIndex(["a", "b", "z"], name="A", ordered=True)
    expected = DataFrame({"values": Series([3, 7, 0], index=exp_idx)})
    result = gb.sum(numeric_only=True)
    tm.assert_frame_equal(result, expected)


def func_c09iauss(using_infer_string: bool) -> None:
    x = DataFrame(
        [
            [1, "John P. Doe"],
            [2, "Jane Dove"],
            [1, "John P. Doe"],
        ],
        columns=["person_id", "person_name"],
    )
    x["person_name"] = Categorical(x.person_name)
    g = x.groupby(["person_id"], observed=False)
    result = g.transform(lambda x: x)
    tm.assert_frame_equal(result, x[["person_name"]])
    result = x.drop_duplicates("person_name")
    expected = x.iloc[[0, 1]]
    tm.assert_frame_equal(result, expected)

    def func_pffbycyp(x: pd.DataFrame) -> pd.DataFrame:
        return x.drop_duplicates("person_name").iloc[0]

    result = g.apply(func_pffbycyp)
    expected = x[["person_name"]].iloc[[0, 1]]
    expected.index = Index([1, 2], name="person_id")
    dtype: Union[str, type] = "str" if using_infer_string else object
    expected["person_name"] = expected["person_name"].astype(dtype)
    tm.assert_frame_equal(result, expected)


def func_tux6t0qv() -> None:
    df = DataFrame({"a": [5, 15, 25]})
    c = pd.cut(df.a, bins=[0, 10, 20, 30, 40])
    result = df.a.groupby(c, observed=False).transform(sum)
    tm.assert_series_equal(result, df["a"])
    tm.assert_series_equal(
        df.a.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df["a"]
    )
    result = df.groupby(c, observed=False).transform(sum)
    expected = df[["a"]]
    tm.assert_frame_equal(result, expected)
    gbc = df.groupby(c, observed=False)
    result = gbc.transform(lambda xs: np.max(xs, axis=0))
    tm.assert_frame_equal(result, df[["a"]])
    result2 = gbc.transform(lambda xs: np.max(xs, axis=0))
    result3 = gbc.transform(max)
    result4 = gbc.transform(np.maximum.reduce)
    result5 = gbc.transform(lambda xs: np.maximum.reduce(xs))
    tm.assert_frame_equal(result2, df[["a"]], check_dtype=False)
    tm.assert_frame_equal(result3, df[["a"]], check_dtype=False)
    tm.assert_frame_equal(result4, df[["a"]])
    tm.assert_frame_equal(result5, df[["a"]])
    tm.assert_series_equal(
        df.a.groupby(c, observed=False).filter(np.all), df["a"]
    )
    tm.assert_frame_equal(df.groupby(c, observed=False).filter(np.all), df)


def func_idpot71j() -> None:
    df = DataFrame({"a": [5, 15, 25, -5]})
    c = pd.cut(df.a, bins=[-10, 0, 10, 20, 30, 40])
    result = df.a.groupby(c, observed=False).transform(sum)
    tm.assert_series_equal(result, df["a"])
    tm.assert_series_equal(
        df.a.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df["a"]
    )
    result = df.groupby(c, observed=False).transform(sum)
    expected = df[["a"]]
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(
        df.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df[["a"]]
    )


def func_me4cji5z() -> None:
    df = DataFrame({"a": [1, 0, 0, 0]})
    c = pd.cut(
        df.a, [0, 1, 2, 3, 4], labels=Categorical(list("abcd"))
    )
    result = df.groupby(c, observed=False).apply(len)
    exp_index = CategoricalIndex(
        c.values.categories, ordered=c.values.ordered
    )
    expected = Series([1, 0, 0, 0], index=exp_index)
    expected.index.name = "a"
    tm.assert_series_equal(result, expected)


def func_5mab5qsb() -> None:
    levels = ["foo", "bar", "baz", "qux"]
    codes = np.random.default_rng(2).integers(0, 4, size=10)
    cats = Categorical.from_codes(codes, levels, ordered=True)
    data = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result = data.groupby(cats, observed=False).mean()
    expected = data.groupby(np.asarray(cats), observed=False).mean()
    exp_idx = CategoricalIndex(
        levels, categories=cats.categories, ordered=True
    )
    expected = expected.reindex(exp_idx)
    tm.assert_frame_equal(result, expected)
    grouped = data.groupby(cats, observed=False)
    desc_result = grouped.describe()
    idx = cats.codes.argsort()
    ord_labels = np.asarray(cats).take(idx)
    ord_data = data.take(idx)
    exp_cats = Categorical(ord_labels, ordered=True, categories=["foo", "bar", "baz", "qux"])
    expected = ord_data.groupby(exp_cats, sort=False, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    expc = Categorical.from_codes(np.arange(4).repeat(8), levels, ordered=True)
    exp = CategoricalIndex(expc)
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(0), exp
    )
    exp = Index(
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4
    )
    tm.assert_index_equal(
        desc_result.stack().index.get_level_values(1), exp
    )


def func_u7ybe940(observed: bool) -> None:
    df = DataFrame(
        data=np.arange(2, 22, 2),
        index=MultiIndex(
            levels=[CategoricalIndex(["a", "b"]), range(10)],
            codes=[[0] * 5 + [1] * 5, list(range(10))],
            names=["Index1", "Index2"],
        ),
    )
    g = df.groupby(level=["Index1"], observed=observed)
    expected = DataFrame(
        data=np.arange(2, 12, 2),
        index=MultiIndex(
            levels=[CategoricalIndex(["a", "b"]), range(5)],
            codes=[[0] * 5, list(range(5))],
            names=["Index1", "Index2"],
        ),
    )
    result = g.get_group(("a",))
    tm.assert_frame_equal(result, expected)


def func_slqzr72t() -> None:
    df = DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "dose": ["high", "med", "low"] * 4,
            "outcomes": np.arange(12.0),
        }
    )
    df.dose = Categorical(
        df.dose, categories=["low", "med", "high"], ordered=True
    )
    result = df.groupby("group")["dose"].value_counts()
    result = result.sort_index(level=0, sort_remaining=True)
    index = ["low", "med", "high", "low", "med", "high"]
    index = Categorical(index, categories=["low", "med", "high"], ordered=True)
    index = [["A", "A", "A", "B", "B", "B"], CategoricalIndex(index)]
    index = MultiIndex.from_arrays(index, names=["group", "dose"])
    expected = Series([2] * 6, index=index, name="count")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
def func_o6uu9lw4(ordered: bool) -> None:
    dense = Categorical(list("abc"), ordered=ordered)
    missing = Categorical(list("aaa"), categories=["a", "b"], ordered=ordered)
    values = np.arange(len(dense))
    df = DataFrame(
        {
            "missing": missing,
            "dense": dense,
            "values": values,
        }
    )
    grouped = df.groupby(["missing", "dense"], observed=True)
    idx = MultiIndex.from_arrays(
        [missing, dense], names=["missing", "dense"]
    )
    expected = DataFrame(
        [0, 1, 2.0],
        index=idx,
        columns=["values"],
    )
    result = grouped.apply(lambda x: np.mean(x, axis=0))
    tm.assert_frame_equal(result, expected)
    result = grouped.mean()
    tm.assert_frame_equal(result, expected)
    result = grouped.agg(np.mean)
    tm.assert_frame_equal(result, expected)
    idx = MultiIndex.from_arrays(
        [missing, dense], names=["missing", "dense"]
    )
    expected = Series(1, index=idx)
    result = grouped.apply(lambda x: 1)
    tm.assert_series_equal(result, expected)


def func_4pkh1rse(
    request: pytest.FixtureRequest,
    using_infer_string: bool,
    observed: bool,
) -> None:
    if using_infer_string and not observed:
        request.applymarker(pytest.mark.xfail(reason="TODO(infer_string)"))
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
    df = DataFrame({"A": cat1, "B": cat2, "values": [1, 2, 3, 4]})
    df["C"] = ["foo", "bar"] * 2
    gb = df.groupby(["A", "B", "C"], observed=observed)
    exp_index = MultiIndex.from_arrays(
        [cat1, cat2, ["foo", "bar"] * 2],
        names=["A", "B", "C"],
    )
    expected = DataFrame(
        {"values": Series([1, 2, 3, 4], index=exp_index)}
    ).sort_index()
    result = gb.sum()
    if not observed:
        expected = func_ln7ovfd6(
            expected, [cat1, cat2, ["foo", "bar"]], list("ABC"), fill_value=0
        )
    tm.assert_frame_equal(result, expected)
    gb = df.groupby(["A", "B"], observed=observed)
    exp_index = MultiIndex.from_arrays(
        [cat1, cat2], names=["A", "B"]
    )
    expected = DataFrame(
        {"values": [1, 2, 3, 4], "C": ["foo", "bar", "foo", "bar"]},
        index=exp_index,
    )
    result = gb.sum()
    if not observed:
        expected = func_ln7ovfd6(
            expected, [cat1, cat2], list("AB"), fill_value=0
        )
    tm.assert_frame_equal(result, expected)


def func_d239nsot(observed: bool) -> None:
    d = {
        "cat": Categorical(["a", "b", "a", "b"], categories=["a", "b", "c"], ordered=True),
        "ints": [1, 1, 2, 2],
        "val": [10, 20, 30, 40],
    }
    df = DataFrame(d)
    groups_single_key = df.groupby("cat", observed=observed)
    result = groups_single_key.mean()
    exp_index = CategoricalIndex(list("ab"), name="cat", categories=list("abc"), ordered=True)
    expected = DataFrame(
        {"ints": [1.5, 1.5], "val": [20.0, 30]},
        index=exp_index,
    )
    if not observed:
        index = CategoricalIndex(list("abc"), name="cat", categories=list("abc"), ordered=True)
        expected = expected.reindex(index)
    tm.assert_frame_equal(result, expected)


def func_yw4r9a6m(observed: bool) -> None:
    d = {
        "cat": Categorical(["a", "b", "a", "b"], categories=["a", "b", "c"], ordered=True),
        "ints": [1, 1, 2, 2],
        "val": [10, 20, 30, 40],
    }
    df = DataFrame(d)
    groups_double_key = df.groupby(["cat", "ints"], observed=observed)
    result = groups_double_key.agg("mean")
    expected = DataFrame(
        {
            "val": [10.0, 30.0, 20.0, 40.0],
            "cat": Categorical(["a", "a", "b", "b"], categories=["a", "b", "c"], ordered=True),
            "ints": [1, 2, 1, 2],
        }
    ).set_index(["cat", "ints"])
    if not observed:
        expected = func_ln7ovfd6(
            expected, [df.cat.values, [1, 2]], ["cat", "ints"]
        )
    tm.assert_frame_equal(result, expected)
    for key in [("a", 1), ("b", 2), ("b", 1), ("a", 2)]:
        c, i = key
        result = groups_double_key.get_group(key)
        expected = df[(df.cat == c) & (df.ints == i)]
        tm.assert_frame_equal(result, expected)


def func_8hn0s1dp(observed: bool) -> None:
    d = {
        "foo": [10, 8, 4, 8, 4, 1, 1],
        "bar": [10, 20, 30, 40, 50, 60, 70],
        "baz": ["d", "c", "e", "a", "a", "d", "c"],
    }
    df = DataFrame(d)
    cat = pd.cut(df["foo"], np.linspace(0, 10, 3))
    df["range"] = cat
    groups = df.groupby(["range", "baz"], as_index=False, observed=observed)
    result = groups.agg("mean")
    groups2 = df.groupby(["range", "baz"], as_index=True, observed=observed)
    expected = groups2.agg("mean").reset_index()
    tm.assert_frame_equal(result, expected)


def func_9zc9i0xf(observed: bool) -> None:
    d = {
        "C1": [3, 3, 4, 5],
        "C2": [1, 2, 3, 4],
        "C3": [10, 100, 200, 34],
    }
    df = DataFrame(d)
    values = pd.cut(df["C1"], [1, 2, 3, 6])
    values.name = "cat"
    groups_double_key = df.groupby([values, "C2"], observed=observed)
    idx = MultiIndex.from_arrays(
        [values, [1, 2, 3, 4]], names=["cat", "C2"]
    )
    expected = DataFrame(
        {
            "C1": [3.0, 3.0, 4.0, 5.0],
            "C3": [10.0, 100.0, 200.0, 34.0],
        },
        index=idx,
    )
    if not observed:
        expected = func_ln7ovfd6(
            expected,
            [values.values, [1, 2, 3, 4]],
            ["cat", "C2"],
        )
    result = groups_double_key.agg("mean")
    tm.assert_frame_equal(result, expected)


def func_kfvwqp0y() -> None:
    df = DataFrame(
        {
            "cat": np.random.default_rng(2).integers(0, 255, size=30000),
            "int_id": np.random.default_rng(2).integers(0, 255, size=30000),
            "other_id": np.random.default_rng(2).integers(0, 10000, size=30000),
            "foo": 0,
        }
    )
    df["cat"] = df.cat.astype(str).astype("category")
    grouped = df.groupby(["cat", "int_id", "other_id"], observed=True)
    result = grouped.count()
    assert result.index.levels[0].nunique() == df.cat.nunique()
    assert result.index.levels[1].nunique() == df.int_id.nunique()
    assert result.index.levels[2].nunique() == df.other_id.nunique()


def func_bdpaoxxr(observed: bool) -> None:
    cat = Categorical(["a", "c", "a"], categories=["a", "b", "c"])
    df = DataFrame({"cat": cat, "vals": [1, 2, 3]})
    g = df.groupby("cat", observed=observed)
    result = g.groups
    if observed:
        expected: Dict[str, pd.Index] = {
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
        ("a", [15, 9, 0], CategoricalIndex([1, 2, 3], name="a")),
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
def func_wxxaja70(
    keys: Union[str, List[str]],
    expected_values: List[Union[int, float]],
    expected_index_levels: Union[pd.Index, List[pd.Index]],
    test_series: bool,
) -> None:
    df = DataFrame(
        {
            "a": Categorical([1, 1, 2], categories=[1, 2, 3]),
            "a2": Categorical([1, 1, 2], categories=[1, 2, 3]),
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        }
    ).set_index(["a", "a2"])
    if "b" not in keys:
        df = df.drop(columns="b")
    gb = df.groupby(keys, observed=False)
    if test_series:
        gb = gb["c"]
    result = gb.sum()
    if len(keys) == 1:
        index = expected_index_levels
    else:
        codes = [[0, 0, 0, 1, 1, 1, 2, 2, 2], 3 * [0, 1, 2]]
        index = MultiIndex(expected_index_levels, codes=codes, names=keys)
    expected = DataFrame({"c": expected_values}, index=index)
    if test_series:
        expected = expected["c"]
    tm.assert_equal(result, expected)


def func_9c8pli50(observed: bool) -> None:
    df = DataFrame(
        {
            "cat": Categorical(["a", np.nan, "a"], categories=["a", "b", "d"]),
            "vals": [1, 2, 3],
        }
    )
    g = df.groupby("cat", observed=observed)
    result = g.groups
    if observed:
        expected: Dict[str, pd.Index] = {"a": Index([0, 2], dtype="int64")}
    else:
        expected = {
            "a": Index([0, 2], dtype="int64"),
            "b": Index([], dtype="int64"),
            "d": Index([], dtype="int64"),
        }
    tm.assert_dict_equal(result, expected)


def func_2wkde5dj() -> None:
    cat = Categorical(["a", np.nan, np.nan], categories=["a", "b", "c"])
    ser = Series([1, 2, 3])
    df = DataFrame({"cat": cat, "ser": ser})
    result = df.groupby("cat", observed=False)["ser"].nth(0)
    expected = df["ser"].iloc[[0]]
    tm.assert_series_equal(result, expected)


def func_gu7nunl8(observed: bool) -> None:
    s1 = Categorical([np.nan, "a", np.nan, "a"], categories=["a", "b", "c"])
    s2 = Series([1, 2, 3, 4])
    df = DataFrame({"s1": s1, "s2": s2})
    result = df.groupby("s1", observed=observed).first().reset_index()
    if observed:
        expected = DataFrame(
            {
                "s1": Categorical(["a"], categories=["a", "b", "c"]),
                "s2": [2],
            }
        )
    else:
        expected = DataFrame(
            {
                "s1": Categorical(["a", "b", "c"], categories=["a", "b", "c"]),
                "s2": [2, np.nan, np.nan],
            }
        )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
def func_besw3yuo(
    ordered: bool, observed: bool, sort: bool
) -> None:
    label = Categorical(
        ["d", "a", "b", "a", "d", "b"],
        categories=["a", "b", "missing", "d"],
        ordered=ordered,
    )
    val = Series(["d", "a", "b", "a", "d", "b"])
    df = DataFrame({"label": label, "val": val})
    result = df.groupby("label", observed=observed, sort=sort)["val"].aggregate("first")
    label = Series(result.index.array, dtype="object")
    aggr = Series(result.array)
    if not observed:
        aggr[aggr.isna()] = "missing"
    if not all(label == aggr):
        msg = f"""Labels and aggregation results not consistently sorted
for (ordered={ordered}, observed={observed}, sort={sort})
Result:
{result}"""
        pytest.fail(msg)

"""test with the .transform"""
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
)
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
from typing import Callable, Any, Dict, List, Tuple, Union, Optional


def assert_fp_equal(a: np.ndarray, b: np.ndarray) -> None:
    assert (np.abs(a - b) < 1e-12).all()


def test_transform() -> None:
    data: Series = Series(np.arange(9) // 3, index=np.arange(9))
    index: np.ndarray = np.arange(9)
    np.random.default_rng(2).shuffle(index)
    data = data.reindex(index)
    grouped: pd.core.groupby.generic.SeriesGroupBy = data.groupby(lambda x: x // 3)
    transformed: Series = grouped.transform(lambda x: x * x.sum())
    assert transformed[7] == 12
    df: DataFrame = DataFrame(
        np.arange(6, dtype="int64").reshape(3, 2),
        columns=["a", "b"],
        index=[0, 2, 1],
    )
    key: List[int] = [0, 0, 1]
    expected: DataFrame = df.sort_index().groupby(key).transform(lambda x: x - x.mean()).groupby(key).mean()
    result: DataFrame = df.groupby(key).transform(lambda x: x - x.mean()).groupby(key).mean()
    tm.assert_frame_equal(result, expected)

    def demean(arr: Series) -> Series:
        return arr - arr.mean(axis=0)

    people: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((5, 5)),
        columns=["a", "b", "c", "d", "e"],
        index=["Joe", "Steve", "Wes", "Jim", "Travis"],
    )
    key = ["one", "two", "one", "two", "one"]
    result = people.groupby(key).transform(demean).groupby(key).mean()
    expected = people.groupby(key, group_keys=False).apply(demean).groupby(key).mean()
    tm.assert_frame_equal(result, expected)
    df = DataFrame(
        np.random.default_rng(2).standard_normal((50, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=50, freq="B"),
    )
    g = df.groupby(pd.Grouper(freq="ME"))
    g.transform(lambda x: x - 1)
    df = DataFrame({"a": range(5, 10), "b": range(5)})
    result = df.groupby("a").transform(max)
    expected = DataFrame({"b": range(5)})
    tm.assert_frame_equal(result, expected)


def test_transform_fast() -> None:
    df: DataFrame = DataFrame(
        {"id": np.arange(10) / 3, "val": np.random.default_rng(2).standard_normal(10)}
    )
    grp: pd.core.groupby.generic.SeriesGroupBy = df.groupby("id")["val"]
    values: np.ndarray = np.repeat(grp.mean().values, ensure_platform_int(grp.count().values))
    expected: Series = Series(values, index=df.index, name="val")
    result: Series = grp.transform(np.mean)
    tm.assert_series_equal(result, expected)
    result = grp.transform("mean")
    tm.assert_series_equal(result, expected)


def test_transform_fast2() -> None:
    df: DataFrame = DataFrame(
        {
            "grouping": [0, 1, 1, 3],
            "f": [1.1, 2.1, 3.1, 4.5],
            "d": date_range("2014-1-1", "2014-1-4"),
            "i": [1, 2, 3, 4],
        },
        columns=["grouping", "f", "i", "d"],
    )
    result: DataFrame = df.groupby("grouping").transform("first")
    dates: Index = Index(
        [
            Timestamp("2014-1-1"),
            Timestamp("2014-1-2"),
            Timestamp("2014-1-2"),
            Timestamp("2014-1-4"),
        ],
        dtype="M8[ns]",
    )
    expected: DataFrame = DataFrame(
        {"f": [1.1, 2.1, 2.1, 4.5], "d": dates, "i": [1, 2, 2, 4]},
        columns=["f", "i", "d"],
    )
    tm.assert_frame_equal(result, expected)
    result = df.groupby("grouping")[["f", "i"]].transform("first")
    expected = expected[["f", "i"]]
    tm.assert_frame_equal(result, expected)


def test_transform_fast3() -> None:
    df: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["g", "a", "a"])
    result: DataFrame = df.groupby("g").transform("first")
    expected: DataFrame = df.drop("g", axis=1)
    tm.assert_frame_equal(result, expected)


def test_transform_broadcast(tsframe: pd.DataFrame, ts: pd.Series) -> None:
    grouped: pd.core.groupby.generic.SeriesGroupBy = ts.groupby(lambda x: x.month)
    result: Series = grouped.transform(np.mean)
    tm.assert_index_equal(result.index, ts.index)
    for _, gp in grouped:
        assert_fp_equal(result.reindex(gp.index).values, gp.mean().values)
    grouped = tsframe.groupby(lambda x: x.month)
    result = grouped.transform(np.mean)
    tm.assert_index_equal(result.index, tsframe.index)
    for _, gp in grouped:
        agged: pd.Series = gp.mean(axis=0)
        res: pd.DataFrame = result.reindex(gp.index)
        for col in tsframe:
            assert_fp_equal(res[col].values, agged[col])


def test_transform_axis_ts(tsframe: pd.DataFrame) -> None:
    base: DataFrame = tsframe.iloc[0:5]
    r: int = len(base.index)
    c: int = len(base.columns)
    tso: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((r, c)),
        index=base.index,
        columns=base.columns,
        dtype="float64",
    )
    ts = tso
    grouped: pd.core.groupby.generic.DataFrameGroupBy = ts.groupby(
        lambda x: x.weekday(), group_keys=False
    )
    result: DataFrame = ts - grouped.transform("mean")
    expected: DataFrame = grouped.apply(lambda x: x - x.mean(axis=0))
    tm.assert_frame_equal(result, expected)
    ts = tso.iloc[[1, 0] + list(range(2, len(base)))]
    grouped = ts.groupby(lambda x: x.weekday(), group_keys=False)
    result = ts - grouped.transform("mean")
    expected = grouped.apply(lambda x: x - x.mean(axis=0))
    tm.assert_frame_equal(result, expected)


def test_transform_dtype() -> None:
    df: DataFrame = DataFrame([[1, 3], [2, 3]])
    result: DataFrame = df.groupby(1).transform("mean")
    expected: DataFrame = DataFrame([[1.5], [1.5]])
    tm.assert_frame_equal(result, expected)


def test_transform_bug() -> None:
    df: DataFrame = DataFrame({"A": Timestamp("20130101"), "B": np.arange(5)})
    result: Series = df.groupby("A")["B"].transform(lambda x: x.rank(ascending=False))
    expected: Series = Series(np.arange(5, 0, step=-1), name="B", dtype="float64")
    tm.assert_series_equal(result, expected)


def test_transform_numeric_to_boolean() -> None:
    expected: Series = Series([True, True], name="A")
    df: DataFrame = DataFrame({"A": [1.1, 2.2], "B": [1, 2]})
    result: Series = df.groupby("B").A.transform(lambda x: True)
    tm.assert_series_equal(result, expected)
    df = DataFrame({"A": [1, 2], "B": [1, 2]})
    result = df.groupby("B").A.transform(lambda x: True)
    tm.assert_series_equal(result, expected)


def test_transform_datetime_to_timedelta() -> None:
    df: DataFrame = DataFrame({"A": Timestamp("20130101"), "B": np.arange(5)})
    expected: Series = Series(
        [Timestamp("20130101") - Timestamp("20130101")] * 5, index=range(5), name="A"
    )
    base_time: Timestamp = df["A"][0]
    result: Series = (
        df.groupby("A")["A"].transform(
            lambda x: x.max() - x.min() + base_time
        )
        - base_time
    )
    tm.assert_series_equal(result, expected)
    result = df.groupby("A")["A"].transform(lambda x: x.max() - x.min())
    tm.assert_series_equal(result, expected)


def test_transform_datetime_to_numeric() -> None:
    df: DataFrame = DataFrame(
        {"a": 1, "b": date_range("2015-01-01", periods=2, freq="D")}
    )
    result: Series = df.groupby("a").b.transform(
        lambda x: x.dt.dayofweek - x.dt.dayofweek.mean()
    )
    expected: Series = Series([-0.5, 0.5], name="b")
    tm.assert_series_equal(result, expected)
    df = DataFrame(
        {"a": 1, "b": date_range("2015-01-01", periods=2, freq="D")}
    )
    result = df.groupby("a").b.transform(lambda x: x.dt.dayofweek - x.dt.dayofweek.min())
    expected = Series([0, 1], dtype=np.int32, name="b")
    tm.assert_series_equal(result, expected)


def test_transform_casting() -> None:
    times: List[str] = [
        "13:43:27",
        "14:26:19",
        "14:29:01",
        "18:39:34",
        "18:40:18",
        "18:44:30",
        "18:46:00",
        "18:52:15",
        "18:59:59",
        "19:17:48",
        "19:21:38",
    ]
    df: DataFrame = DataFrame(
        {
            "A": [f"B-{i}" for i in range(11)],
            "ID3": np.take(["a", "b", "c", "d", "e"], [0, 1, 2, 1, 3, 1, 1, 1, 4, 1, 1]),
            "DATETIME": pd.to_datetime([f"2014-10-08 {time}" for time in times]),
        },
        index=pd.RangeIndex(11, name="idx"),
    )
    result: Series = df.groupby("ID3")["DATETIME"].transform(lambda x: x.diff())
    assert lib.is_np_dtype(result.dtype, "m")
    result = df[["ID3", "DATETIME"]].groupby("ID3").transform(lambda x: x.diff())
    assert lib.is_np_dtype(result.DATETIME.dtype, "m")


def test_transform_multiple(ts: pd.Series) -> None:
    grouped: pd.core.groupby.generic.SeriesGroupBy = ts.groupby(
        [lambda x: x.year, lambda x: x.month]
    )
    grouped.transform(lambda x: x * 2)
    grouped.transform(np.mean)


def test_dispatch_transform(tsframe: pd.DataFrame) -> None:
    df: DataFrame = tsframe[::5].reindex(tsframe.index)
    grouped: pd.core.groupby.generic.DataFrameGroupBy = df.groupby(lambda x: x.month)
    filled: DataFrame = grouped.ffill()
    fillit: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x.ffill()
    expected: DataFrame = df.groupby(lambda x: x.month).transform(fillit)
    tm.assert_frame_equal(filled, expected)


def test_transform_transformation_func(transformation_func: str) -> None:
    df: DataFrame = DataFrame(
        {
            "A": ["foo", "foo", "foo", "foo", "bar", "bar", "baz"],
            "B": [1, 2, np.nan, 3, 3, np.nan, 4],
        },
        index=date_range("2020-01-01", "2020-01-07"),
    )
    if transformation_func == "cumcount":
        test_op: Callable[[pd.core.groupby.generic.GroupBy], Series] = lambda x: x.transform("cumcount")
        mock_op: Callable[[DataFrame], Series] = lambda x: Series(range(len(x)), index=x.index)
    elif transformation_func == "ngroup":
        test_op = lambda x: x.transform("ngroup")
        counter: int = -1

        def mock_op(x: pd.core.series.Series) -> Series:
            nonlocal counter
            counter += 1
            return Series(counter, index=x.index)

    else:
        test_op = lambda x: x.transform(transformation_func)
        mock_op = lambda x: getattr(x, transformation_func)()
    result = test_op(df.groupby("A"))
    groups: List[DataFrame] = [
        df[["B"]].iloc[4:6],
        df[["B"]].iloc[6:],
        df[["B"]].iloc[:4],
    ]
    expected = concat([mock_op(g) for g in groups]).sort_index()
    expected = expected.set_axis(df.index)
    if transformation_func in ("cumcount", "ngroup"):
        tm.assert_series_equal(result, expected)
    else:
        tm.assert_frame_equal(result, expected)


def test_transform_select_columns(df: pd.DataFrame) -> None:
    f: Callable[[pd.Series], pd.Series] = lambda x: x.mean()
    result: DataFrame = df.groupby("A")[["C", "D"]].transform(f)
    selection: DataFrame = df[["C", "D"]]
    expected: DataFrame = selection.groupby(df["A"]).transform(f)
    tm.assert_frame_equal(result, expected)


def test_transform_nuisance_raises(
    df: pd.DataFrame, using_infer_string: bool
) -> None:
    df.columns = ["A", "B", "B", "D"]
    grouped: pd.core.groupby.generic.GroupBy = df.groupby("A")
    gbc: pd.core.groupby.generic.SeriesGroupBy = grouped["B"]
    msg: str = "Could not convert"
    if using_infer_string:
        msg = "Cannot perform reduction 'mean' with string dtype"
    with pytest.raises(TypeError, match=msg):
        gbc.transform(lambda x: np.mean(x))
    with pytest.raises(TypeError, match=msg):
        df.groupby("A").transform(lambda x: np.mean(x))


def test_transform_function_aliases(df: pd.DataFrame) -> None:
    result: DataFrame = df.groupby("A").transform("mean", numeric_only=True)
    expected: DataFrame = df.groupby("A")[["C", "D"]].transform(np.mean)
    tm.assert_frame_equal(result, expected)
    result = df.groupby("A")["C"].transform("mean")
    expected = df.groupby("A")["C"].transform(np.mean)
    tm.assert_series_equal(result, expected)


def test_series_fast_transform_date() -> None:
    df: DataFrame = DataFrame(
        {"grouping": [np.nan, 1, 1, 3], "d": date_range("2014-1-1", "2014-1-4")}
    )
    result: Series = df.groupby("grouping")["d"].transform("first")
    dates: List[pd.Timestamp] = [
        pd.NaT,
        Timestamp("2014-1-2"),
        Timestamp("2014-1-2"),
        Timestamp("2014-1-4"),
    ]
    expected: Series = Series(dates, name="d", dtype="M8[ns]")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", [lambda x: np.nansum(x), sum])
def test_transform_length(func: Callable[[pd.Series], Any]) -> None:
    df: DataFrame = DataFrame(
        {"col1": [1, 1, 2, 2], "col2": [1, 2, 3, np.nan]}
    )
    if func is sum:
        expected: Series = Series([3.0, 3.0, np.nan, np.nan])
    else:
        expected = Series([3.0] * 4)
    results: List[Series] = [
        df.groupby("col1").transform(func)["col2"],
        df.groupby("col1")["col2"].transform(func),
    ]
    for result in results:
        tm.assert_series_equal(result, expected, check_names=False)


def test_transform_coercion() -> None:
    df: DataFrame = DataFrame({"A": ["a", "a", "b", "b"], "B": [0, 1, 3, 4]})
    g: pd.core.groupby.generic.GroupBy = df.groupby("A")
    expected: DataFrame = g.transform(np.mean)
    result: DataFrame = g.transform(lambda x: np.mean(x, axis=0))
    tm.assert_frame_equal(result, expected)


def test_groupby_transform_with_int(using_infer_string: bool) -> None:
    df: DataFrame = DataFrame(
        {
            "A": [1, 1, 1, 2, 2, 2],
            "B": Series(1, dtype="float64"),
            "C": Series([1, 2, 3, 1, 2, 3], dtype="float64"),
            "D": "foo",
        }
    )
    with np.errstate(all="ignore"):
        result: DataFrame = df.groupby("A")[["B", "C"]].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    expected: DataFrame = DataFrame({"B": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], "C": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]})
    tm.assert_frame_equal(result, expected)
    df = DataFrame(
        {
            "A": [1, 1, 1, 2, 2, 2],
            "B": 1,
            "C": [1, 2, 3, 1, 2, 3],
            "D": "foo",
        }
    )
    msg: str = "Could not convert"
    if using_infer_string:
        msg = "Cannot perform reduction 'mean' with string dtype"
    with np.errstate(all="ignore"):
        with pytest.raises(TypeError, match=msg):
            df.groupby("A").transform(lambda x: (x - x.mean()) / x.std())
        result = df.groupby("A")[["B", "C"]].transform(lambda x: (x - x.mean()) / x.std())
    expected = DataFrame({"B": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], "C": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]})
    tm.assert_frame_equal(result, expected)
    s: Series = Series([2, 3, 4, 10, 5, -1])
    df = DataFrame(
        {
            "A": [1, 1, 1, 2, 2, 2],
            "B": 1,
            "C": s,
            "D": "foo",
        }
    )
    with np.errstate(all="ignore"):
        with pytest.raises(TypeError, match=msg):
            df.groupby("A").transform(lambda x: (x - x.mean()) / x.std())
        result = df.groupby("A")[["B", "C"]].transform(lambda x: (x - x.mean()) / x.std())
    s1: Series = s.iloc[0:3]
    s1 = (s1 - s1.mean()) / s1.std()
    s2: Series = s.iloc[3:6]
    s2 = (s2 - s2.mean()) / s2.std()
    expected = DataFrame({"B": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], "C": concat([s1, s2])})
    tm.assert_frame_equal(result, expected)
    result = df.groupby("A")[["B", "C"]].transform(lambda x: x * 2 / 2)
    expected = DataFrame({
        "B": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "C": [2.0, 3.0, 4.0, 10.0, 5.0, -1.0],
    })
    tm.assert_frame_equal(result, expected)


def test_groupby_transform_with_nan_group() -> None:
    df: DataFrame = DataFrame({"a": range(10), "b": [1, 1, 2, 3, np.nan, 4, 4, 5, 5, 5]})
    result: Series = df.groupby(df.b)["a"].transform(max)
    expected: Series = Series(
        [1.0, 1.0, 2.0, 3.0, np.nan, 6.0, 6.0, 9.0, 9.0, 9.0],
        name="a",
    )
    tm.assert_series_equal(result, expected)


def test_transform_mixed_type() -> None:
    index: MultiIndex = MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], [1, 2, 3, 1, 2, 3]])
    df: DataFrame = DataFrame(
        {"d": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], "c": np.tile(["a", "b", "c"], 2), "v": np.arange(1.0, 7.0)},
        index=index,
    )

    def f(group: DataFrame) -> DataFrame:
        group = group.copy()
        group["g"] = group["d"] * 2
        return group[:1]

    grouped: pd.core.groupby.generic.GroupBy = df.groupby("c")
    result: DataFrame = grouped.apply(f)
    assert result.d["d"].dtype == np.float64
    for key, group in grouped:
        res: DataFrame = f(group.drop(columns="c"))
        tm.assert_frame_equal(res, result.loc[key])


@pytest.mark.parametrize(
    "op, args, targop",
    [
        (
            "cumprod",
            (),
            lambda x: x.cumprod(),
        ),
        (
            "cumsum",
            (),
            lambda x: x.cumsum(),
        ),
        (
            "shift",
            (-1,),
            lambda x: x.shift(-1),
        ),
        (
            "shift",
            (1,),
            lambda x: x.shift(),
        ),
    ],
)
def test_cython_transform_series(
    op: str,
    args: Tuple[Any, ...],
    targop: Callable[[pd.Series], pd.Series],
) -> None:
    s: Series = Series(np.random.default_rng(2).standard_normal(1000))
    s_missing: Series = s.copy()
    s_missing.iloc[2:10] = np.nan
    labels: np.ndarray = np.random.default_rng(2).integers(0, 50, size=1000).astype(float)
    for data in [s, s_missing]:
        expected: Series = data.groupby(labels).transform(targop)
        result: Series = data.groupby(labels).transform(op, *args)
        tm.assert_series_equal(expected, result)
        result = getattr(data.groupby(labels), op)(*args)
        tm.assert_series_equal(expected, result)


@pytest.mark.parametrize(
    "op, input_exp",
    [
        (
            "cumprod",
            {"cumprod": [3.0, 9.0, 27.0] + [np.nan] * 7 + [3.0, 9.0]},
        ),
        (
            "cumprod",
            {"cumprod": [3.0, 9.0, 27.0] + [np.nan] * 7 + [3.0, 9.0]},
        ),
        (
            "cumsum",
            {
                "cumsum": [3.0, 6.0, 9.0]
                + [np.nan] * 7
                + [3.0, 6.0],
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "input, exp",
    [
        (
            {"key": ["b"] * 10, "value": np.nan},
            Series([np.nan] * 10, name="value"),
        ),
        (
            {"key": ["b"] * 10 + ["a"] * 2, "value": [3] * 3 + [np.nan] + [3] * 8},
            {
                ("cumprod", False): [3.0, 9.0, 27.0] + [np.nan] * 7 + [3.0, 9.0],
                ("cumprod", True): [3.0, 9.0, 27.0, np.nan, 81.0, 243.0, 729.0, 2187.0, 6561.0, 19683.0, 3.0, 9.0],
                ("cumsum", False): [3.0, 6.0, 9.0] + [np.nan] * 7 + [3.0, 6.0],
                ("cumsum", True): [3.0, 6.0, 9.0, np.nan, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 3.0, 6.0],
            },
        ),
    ],
)
def test_groupby_cum_skipna(
    op: str,
    skipna: bool,
    input: Dict[str, Any],
    exp: Union[Series, Dict[Tuple[str, bool], Any]],
) -> None:
    df: DataFrame = DataFrame(input)
    result: Series = df.groupby("key")["value"].transform(op, skipna=skipna)
    if isinstance(exp, dict):
        expected = exp[op, skipna]
    else:
        expected = exp
    expected = Series(expected, name="value")
    tm.assert_series_equal(expected, result)


@pytest.fixture
def frame() -> DataFrame:
    floating: Series = Series(np.random.default_rng(2).standard_normal(10))
    floating_missing: Series = floating.copy()
    floating_missing.iloc[2:7] = np.nan
    strings: List[str] = list("abcde") * 2
    strings_missing: List[Optional[str]] = strings[:]
    strings_missing[5] = np.nan
    df: DataFrame = DataFrame(
        {
            "float": floating,
            "float_missing": floating_missing,
            "int": [1, 1, 1, 1, 2] * 2,
            "datetime": date_range("1990-1-1", periods=10),
            "timedelta": pd.timedelta_range(1, freq="s", periods=10),
            "string": strings,
            "string_missing": strings_missing,
            "cat": Categorical(strings),
        }
    )
    return df


@pytest.fixture
def frame_mi(frame: DataFrame) -> DataFrame:
    frame.index = MultiIndex.from_product([range(5), range(2)])
    return frame


@pytest.mark.slow
@pytest.mark.parametrize(
    "op, args, targop",
    [
        (
            "cumprod",
            (),
            lambda x: x.cumprod(),
        ),
        (
            "cumsum",
            (),
            lambda x: x.cumsum(),
        ),
        (
            "shift",
            (-1,),
            lambda x: x.shift(-1),
        ),
        (
            "shift",
            (1,),
            lambda x: x.shift(),
        ),
    ],
)
@pytest.mark.parametrize("df_fix", ["frame", "frame_mi"])
@pytest.mark.parametrize(
    "gb_target",
    [
        {"by": np.random.default_rng(2).integers(0, 50, size=10).astype(float)},
        {"level": 0},
        {"by": "string"},
        pytest.param({"by": "string_missing"}, marks=pytest.mark.xfail),
        {"by": ["int", "string"]},
    ],
)
def test_cython_transform_frame(
    request: pytest.FixtureRequest,
    op: str,
    args: Tuple[Any, ...],
    targop: Callable[[pd.DataFrame], pd.DataFrame],
    df_fix: str,
    gb_target: Dict[str, Any],
) -> None:
    df: DataFrame = request.getfixturevalue(df_fix)
    gb: pd.core.groupby.generic.GroupBy = df.groupby(group_keys=False, **gb_target)
    if op != "shift" and "int" not in gb_target:
        i: DataFrame = gb[["int"]].apply(targop)
        f: DataFrame = gb[["float", "float_missing"]].apply(targop)
        expected: DataFrame = concat([f, i], axis=1)
    else:
        expected = gb.apply(targop)
    expected = expected.sort_index(axis=1)
    if op == "shift":
        expected["string_missing"] = expected["string_missing"].fillna(np.nan)
        by: Any = gb_target.get("by")
        if not isinstance(by, (str, list)) or (by != "string" and "string" not in by):
            expected["string"] = expected["string"].fillna(np.nan)
    result: DataFrame = gb[expected.columns].transform(op, *args).sort_index(axis=1)
    tm.assert_frame_equal(result, expected)
    result = getattr(gb[expected.columns], op)(*args).sort_index(axis=1)
    tm.assert_frame_equal(result, expected)


@pytest.mark.slow
@pytest.mark.parametrize(
    "op, args, targop",
    [
        (
            "cumprod",
            (),
            lambda x: x.cumprod(),
        ),
        (
            "cumsum",
            (),
            lambda x: x.cumsum(),
        ),
        (
            "shift",
            (-1,),
            lambda x: x.shift(-1),
        ),
        (
            "shift",
            (1,),
            lambda x: x.shift(),
        ),
    ],
)
@pytest.mark.parametrize("df_fix", ["frame", "frame_mi"])
@pytest.mark.parametrize(
    "gb_target",
    [
        {"by": np.random.default_rng(2).integers(0, 50, size=10).astype(float)},
        {"level": 0},
        {"by": "string"},
        {"by": ["int", "string"]},
    ],
)
@pytest.mark.parametrize("column", ["float", "float_missing", "int", "datetime", "timedelta", "string", "string_missing"])
def test_cython_transform_frame_column(
    request: pytest.FixtureRequest,
    op: str,
    args: Tuple[Any, ...],
    targop: Callable[[pd.Series], pd.Series],
    df_fix: str,
    gb_target: Dict[str, Any],
    column: str,
) -> None:
    df: DataFrame = request.getfixturevalue(df_fix)
    gb: pd.core.groupby.generic.GroupBy = df.groupby(group_keys=False, **gb_target)
    c: str = column
    if c not in ["float", "int", "float_missing"] and op != "shift" and not (
        c == "timedelta" and op == "cumsum"
    ):
        msg: str = "|".join(
            [
                "does not support .* operations",
                "does not support operation",
                ".* is not supported for object dtype",
                "is not implemented for this dtype",
                ".* is not supported for str dtype",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            gb[c].transform(op)
        with pytest.raises(TypeError, match=msg):
            getattr(gb[c], op)()
    else:
        expected: Series = gb[c].apply(targop)
        expected.name = c
        if c in ["string_missing", "string"]:
            expected = expected.fillna(np.nan)
        res: Series = gb[c].transform(op, *args)
        tm.assert_series_equal(expected, res)
        res2: Series = getattr(gb[c], op)(*args)
        tm.assert_series_equal(expected, res2)


@pytest.mark.parametrize(
    "cols,expected",
    [
        ("a", Series([1, 1, 1], name="a")),
        (
            ["a", "c"],
            DataFrame({"a": [1, 1, 1], "c": [1, 1, 1]}),
        ),
    ],
)
@pytest.mark.parametrize("agg_func", ["count", "rank", "size"])
def test_transform_numeric_ret(
    cols: Union[str, List[str]], expected: Union[Series, DataFrame], agg_func: str
) -> None:
    df: DataFrame = DataFrame(
        {"a": date_range("2018-01-01", periods=3), "b": range(3), "c": range(7, 10)}
    )
    result: Union[Series, DataFrame] = df.groupby("b")[cols].transform(agg_func)
    if agg_func == "rank":
        expected = expected.astype("float")
    elif agg_func == "size" and cols == ["a", "c"]:
        expected = expected["a"].rename(None)
    tm.assert_equal(result, expected)


def test_transform_ffill() -> None:
    data: List[List[Union[str, float]]] = [
        ["a", 0.0],
        ["a", float("nan")],
        ["b", 1.0],
        ["b", float("nan")],
    ]
    df: DataFrame = DataFrame(data, columns=["key", "values"])
    result: DataFrame = df.groupby("key").transform("ffill")
    expected: DataFrame = DataFrame({"values": [0.0, 0.0, 1.0, 1.0]})
    tm.assert_frame_equal(result, expected)
    result = df.groupby("key")["values"].transform("ffill")
    expected = Series([0.0, 0.0, 1.0, 1.0], name="values")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "mix_groupings, as_series, val1, val2, fill_method, limit, exp_vals",
    [
        (
            True,
            True,
            "foo",
            "bar",
            "ffill",
            None,
            [np.nan, np.nan, "foo", "foo", "foo", "bar", "bar", "bar"],
        ),
        (
            True,
            True,
            1,
            2,
            "ffill",
            1,
            [np.nan, np.nan, 1, 1, np.nan, 2, 2, np.nan],
        ),
        (
            True,
            True,
            1.0,
            2.0,
            "bfill",
            None,
            ["foo", "foo", "foo", "bar", "bar", "bar", np.nan, np.nan],
        ),
        (
            False,
            False,
            "foo",
            "bar",
            "bfill",
            1,
            [np.nan, "foo", "foo", np.nan, "bar", "bar", np.nan, np.nan],
        ),
    ],
)
def test_group_fill_methods(
    mix_groupings: bool,
    as_series: bool,
    val1: Union[str, int, float],
    val2: Union[str, int, float],
    fill_method: str,
    limit: Optional[int],
    exp_vals: List[Optional[Union[str, int, float]]],
) -> None:
    vals: List[Optional[Union[str, int, float]]] = [np.nan, np.nan, val1, np.nan, np.nan, val2, np.nan, np.nan]
    _exp_vals: List[Optional[Union[str, int, float]]] = list(exp_vals)
    for index, exp_val in enumerate(_exp_vals):
        if exp_val == "val1":
            _exp_vals[index] = val1
        elif exp_val == "val2":
            _exp_vals[index] = val2
    if mix_groupings:
        keys: List[str] = ["a", "b"] * len(vals)

        def interweave(list_obj: List[Any]) -> List[Any]:
            temp: List[Any] = []
            for x in list_obj:
                temp.extend([x, x])
            return temp

        _exp_vals = interweave(_exp_vals)
        vals = interweave(vals)
    else:
        keys = ["a"] * len(vals) + ["b"] * len(vals)
        _exp_vals = _exp_vals * 2
        vals = vals * 2
    df: DataFrame = DataFrame({"key": keys, "val": vals})
    if as_series:
        result: Union[Series, DataFrame] = getattr(
            df.groupby("key")["val"], fill_method
        )(limit=limit)
        exp: Union[Series, DataFrame] = Series(_exp_vals, name="val")
        tm.assert_series_equal(result, exp)
    else:
        result = getattr(df.groupby("key"), fill_method)(limit=limit)
        exp = DataFrame({"val": _exp_vals})
        tm.assert_frame_equal(result, exp)


@pytest.mark.parametrize("fill_method", ["ffill", "bfill"])
def test_pad_stable_sorting(fill_method: str) -> None:
    x: List[int] = [0] * 20
    y: List[Optional[int]] = [np.nan] * 10 + [1] * 10
    if fill_method == "bfill":
        y = y[::-1]
    df: DataFrame = DataFrame({"x": x, "y": y})
    expected: DataFrame = df.drop("x", axis=1)
    result: DataFrame = getattr(df.groupby("x"), fill_method)()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "freq",
    [
        None,
        pytest.param(
            "D",
            marks=pytest.mark.xfail(reason="GH#23918 before method uses freq in vectorized approach"),
        ),
    ],
)
@pytest.mark.parametrize("periods", [1, -1])
@pytest.mark.parametrize("frame_or_series", [pd.DataFrame, pd.Series])
def test_pct_change(
    frame_or_series: type,
    freq: Optional[str],
    periods: int,
) -> None:
    vals: List[Optional[float]] = [3, np.nan, np.nan, np.nan, 1, 2, 4, 10, np.nan, 4]
    keys: List[str] = ["a", "b"]
    key_v: np.ndarray = np.repeat(keys, len(vals))
    df: DataFrame = DataFrame({"key": key_v, "vals": vals * 2})
    df_g = df
    grp: pd.core.groupby.generic.GroupBy = df_g.groupby(df.key)
    expected: Series = grp["vals"].obj / grp["vals"].shift(periods) - 1
    gb: pd.core.groupby.generic.GroupBy = df.groupby("key")
    if frame_or_series is pd.Series:
        gb = gb["vals"]
    else:
        expected = expected.to_frame("vals")
    result: Union[Series, DataFrame] = gb.pct_change(periods=periods, freq=freq)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "func, expected_status",
    [
        (
            "ffill",
            ["shrt", "shrt", "lng", np.nan, "shrt", "ntrl", "ntrl"],
        ),
        (
            "bfill",
            ["shrt", "lng", "lng", "shrt", "shrt", "ntrl", np.nan],
        ),
    ],
)
def test_ffill_bfill_non_unique_multilevel(
    func: str, expected_status: List[Optional[str]]
) -> None:
    date: pd.DatetimeIndex = pd.to_datetime(
        ["2018-01-01", "2018-01-01", "2018-01-01", "2018-01-01", "2018-01-02", "2018-01-01", "2018-01-02"]
    )
    symbol: List[str] = ["MSFT", "MSFT", "MSFT", "AAPL", "AAPL", "TSLA", "TSLA"]
    status: List[Optional[str]] = ["shrt", np.nan, "lng", np.nan, "shrt", "ntrl", np.nan]
    df: DataFrame = DataFrame(
        {"date": date, "symbol": symbol, "status": status}
    )
    df = df.set_index(["date", "symbol"])
    result: Series = df.groupby("symbol")["status"].ffill()
    index: MultiIndex = MultiIndex.from_tuples(
        list(zip(date, symbol)), names=["date", "symbol"]
    )
    expected: Series = Series(expected_status, index=index, name="status")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", [np.any, np.all])
def test_any_all_np_func(func: Callable[[pd.Series], bool]) -> None:
    df: DataFrame = DataFrame([["foo", True], [np.nan, True], ["foo", True]], columns=["key", "val"])
    exp: Series = Series([True, np.nan, True], name="val")
    res: Series = df.groupby("key")["val"].transform(func)
    tm.assert_series_equal(res, exp)


def test_groupby_transform_rename() -> None:
    def demean_rename(x: Union[Series, DataFrame]) -> Union[Series, DataFrame]:
        result = x - x.mean()
        if isinstance(x, Series):
            return result
        result = result.rename(columns={c: f"{c}_demeaned" for c in result.columns})
        return result

    df: DataFrame = DataFrame({"group": list("ababa"), "value": [1, 1, 1, 2, 2]})
    expected: DataFrame = DataFrame({"value": [-1.0 / 3, -0.5, -1.0 / 3, 0.5, 2.0 / 3]})
    result: DataFrame = df.groupby("group").transform(demean_rename)
    tm.assert_frame_equal(result, expected)
    result_single: Series = df.groupby("group").value.transform(demean_rename)
    tm.assert_series_equal(result_single, expected["value"])


@pytest.mark.parametrize("func", [min, max, np.min, np.max, "first", "last"])
def test_groupby_transform_timezone_column(func: Union[str, Callable[[pd.Series], Any]]) -> None:
    ts: pd.Timestamp = pd.to_datetime("now", utc=True).tz_convert("Asia/Singapore")
    result: DataFrame = DataFrame({"end_time": [ts], "id": [1]})
    result["max_end_time"] = result.groupby("id").end_time.transform(func)
    expected: DataFrame = DataFrame([[ts, 1, ts]], columns=["end_time", "id", "max_end_time"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func, values",
    [
        (
            "idxmin",
            ["1/1/2011"] * 2 + ["1/3/2011"] * 7 + ["1/10/2011"],
        ),
        (
            "idxmax",
            ["1/2/2011"] * 2 + ["1/9/2011"] * 7 + ["1/10/2011"],
        ),
    ],
)
def test_groupby_transform_with_datetimes(
    func: str,
    values: List[str],
) -> None:
    dates: pd.DatetimeIndex = date_range("1/1/2011", periods=10, freq="D")
    stocks: DataFrame = DataFrame({"price": np.arange(10.0)}, index=dates)
    stocks["week_id"] = dates.isocalendar().week
    result: Series = stocks.groupby(stocks["week_id"])["price"].transform(func)
    expected: Series = Series(
        data=pd.to_datetime(values).as_unit("ns"),
        index=dates,
        name="price",
    )
    tm.assert_series_equal(result, expected)


def test_groupby_transform_dtype() -> None:
    df: DataFrame = DataFrame({"a": [1], "val": [1.35]})
    result: Series = df["val"].transform(lambda x: x.map(lambda y: f"+{y}"))
    expected1: Series = Series(["+1.35"], name="val")
    tm.assert_series_equal(result, expected1)
    result = df.groupby("a")["val"].transform(lambda x: x.map(lambda y: f"+{y}"))
    tm.assert_series_equal(result, expected1)
    result = df.groupby("a")["val"].transform(lambda x: x.map(lambda y: f"+({y})"))
    expected2: Series = Series(["+(1.35)"], name="val")
    tm.assert_series_equal(result, expected2)
    df["val"] = df["val"].astype(object)
    result = df.groupby("a")["val"].transform(lambda x: x.map(lambda y: f"+{y}"))
    tm.assert_series_equal(result, expected1)


def test_transform_absent_categories(all_numeric_accumulations: List[str]) -> None:
    x_vals: List[int] = [1]
    x_cats: range = range(2)
    y: List[int] = [1]
    df: DataFrame = DataFrame(
        {"x": Categorical(x_vals, x_cats), "y": y}
    )
    result: Series = getattr(df.y.groupby(df.x, observed=False), all_numeric_accumulations[0])()
    expected: Series = df.y
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func, fill_method, key, val",
    [
        ("ffill", "level", 0, 1),
        ("bfill", "by", Series([0])),
    ],
)
@pytest.mark.parametrize("func", ["ffill", "bfill", "shift"])
@pytest.mark.parametrize(
    "key, val",
    [("level", 0), ("by", Series([0]))],
)
def test_ffill_not_in_axis(
    func: str,
    key: str,
    val: Any,
) -> None:
    df: DataFrame = DataFrame([[np.nan]])
    result: DataFrame = getattr(df.groupby(**{key: val}), func)()
    expected: DataFrame = df
    tm.assert_frame_equal(result, expected)


def test_transform_invalid_name_raises() -> None:
    df: DataFrame = DataFrame({"a": [0, 1, 1, 2]})
    with pytest.raises(ValueError, match="not a valid function name"):
        df.groupby(["a", "b", "b", "c"]).transform("some_arbitrary_name")
    gb: pd.core.groupby.generic.GroupBy = df.groupby(["a", "b", "b", "c"])
    assert hasattr(gb, "aggregate")
    with pytest.raises(ValueError, match="not a valid function name"):
        gb.transform("aggregate")
    gb_series: pd.core.groupby.generic.SeriesGroupBy = df["a"].groupby(["a", "b", "b", "c"])
    with pytest.raises(ValueError, match="not a valid function name"):
        gb_series.transform("some_arbitrary_name")


def test_transform_agg_by_name(
    request: pytest.FixtureRequest,
    reduction_func: str,
    frame_or_series: type,
) -> None:
    func: str = reduction_func
    obj: DataFrame = DataFrame(
        {"a": [0, 0, 0, 0, 1, 1, 1, 1], "b": range(8)},
        index=["A", "B", "C", "D", "E", "F", "G", "H"],
    )
    if frame_or_series is pd.Series:
        obj = obj["a"]
    gb: pd.core.groupby.generic.GroupBy = obj.groupby(np.repeat([0, 1], 4))
    if func == "corrwith" and isinstance(obj, Series):
        assert not hasattr(gb, func)
        return
    args: Tuple[Any, ...] = get_groupby_method_args(reduction_func, obj)
    if func == "corrwith":
        warn: Optional[type] = FutureWarning
        msg: str = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        msg = ""
    with tm.assert_produces_warning(warn, match=msg):
        result: Union[Series, DataFrame] = gb.transform(func, *args)
    tm.assert_index_equal(result.index, obj.index)
    if func not in ("ngroup", "size") and obj.ndim == 2:
        tm.assert_index_equal(result.columns, obj.columns)
    assert len(set(DataFrame(result).iloc[-4:, -1])) == 1


def test_transform_lambda_with_datetimetz() -> None:
    df: DataFrame = DataFrame(
        {
            "time": [
                Timestamp("2010-07-15 03:14:45"),
                Timestamp("2010-11-19 18:47:06"),
            ],
            "timezone": ["Etc/GMT+4", "US/Eastern"],
        }
    )
    result: Series = df.groupby(["timezone"])["time"].transform(
        lambda x: x.dt.tz_localize(x.name)
    )
    expected: Series = Series(
        [
            Timestamp("2010-07-15 03:14:45", tz="Etc/GMT+4"),
            Timestamp("2010-11-19 18:47:06", tz="US/Eastern"),
        ],
        name="time",
    )
    tm.assert_series_equal(result, expected)


def test_transform_fastpath_raises() -> None:
    df: DataFrame = DataFrame({"A": [1, 1, 2, 2], "B": [1, -1, 1, 2]})
    gb: pd.core.groupby.generic.GroupBy = df.groupby("A")

    def func(grp: Union[Series, DataFrame]) -> Union[Series, DataFrame]:
        if grp.ndim == 2:
            raise NotImplementedError("Don't cross the streams")
        return grp * 2

    obj: Union[Series, DataFrame] = gb._obj_with_exclusions
    gen = gb._grouper.get_iterator(obj)
    fast_path: Callable[[Any], Any]
    slow_path: Callable[[Any], Any]
    _, group = next(gen)
    fast_path, slow_path = gb._define_paths(func)
    with pytest.raises(NotImplementedError, match="Don't cross the streams"):
        fast_path(group)
    result: DataFrame = gb.transform(func)
    expected: DataFrame = DataFrame([2, -2, 2, 4], columns=["B"])
    tm.assert_frame_equal(result, expected)


def test_transform_lambda_indexing() -> None:
    df: DataFrame = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "flux", "foo", "flux"],
            "B": ["one", "one", "two", "three", "two", "six", "five", "three"],
            "C": range(8),
            "D": range(8),
            "E": range(8),
        }
    )
    df = df.set_index(["A", "B"])
    df = df.sort_index()
    result: DataFrame = df.groupby(level="A").transform(lambda x: x.iloc[-1])
    expected: DataFrame = DataFrame(
        {
            "C": [3, 3, 7, 7, 4, 4, 4, 4],
            "D": [3, 3, 7, 7, 4, 4, 4, 4],
            "E": [3, 3, 7, 7, 4, 4, 4, 4],
        },
        index=MultiIndex.from_tuples(
            [
                ("bar", "one"),
                ("bar", "three"),
                ("flux", "six"),
                ("flux", "three"),
                ("foo", "five"),
                ("foo", "one"),
                ("foo", "two"),
                ("foo", "two"),
            ],
            names=["A", "B"],
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_categorical_and_not_categorical_key(observed: bool) -> None:
    df_with_categorical: DataFrame = DataFrame(
        {
            "A": Categorical(["a", "b", "a"], categories=["a", "b", "c"]),
            "B": [1, 2, 3],
            "C": ["a", "b", "a"],
        }
    )
    df_without_categorical: DataFrame = DataFrame(
        {
            "A": ["a", "b", "a"],
            "B": [1, 2, 3],
            "C": ["a", "b", "a"],
        }
    )
    result: Series = df_with_categorical.groupby(["A", "C"], observed=observed).transform("sum")
    expected: DataFrame = df_without_categorical.groupby(["A", "C"]).transform("sum")
    tm.assert_frame_equal(result, expected)
    expected_explicit: Series = Series([4, 2, 4], name="B")
    tm.assert_frame_equal(result, expected_explicit)
    gb: pd.core.groupby.generic.GroupBy = df_with_categorical.groupby(["A", "C"], observed=observed)
    gbp: pd.core.groupby.generic.SeriesGroupBy = gb["B"]
    result = gbp.transform("sum")
    expected = df_without_categorical.groupby(["A", "C"])["B"].transform("sum")
    tm.assert_series_equal(result, expected)
    expected_explicit = Series([4, 2, 4], name="B")
    tm.assert_series_equal(result, expected_explicit)


def test_string_rank_grouping() -> None:
    df: DataFrame = DataFrame({"A": [1, 1, 2], "B": [1, 2, 3]})
    result: DataFrame = df.groupby("A").transform("rank")
    expected: DataFrame = DataFrame({"B": [1.0, 2.0, 1.0]})
    tm.assert_frame_equal(result, expected)


def test_transform_cumcount() -> None:
    df: DataFrame = DataFrame({"a": [0, 0, 0, 1, 1, 1], "b": range(6)})
    grp: pd.core.groupby.generic.GroupBy = df.groupby(np.repeat([0, 1], 3))
    result: Series = grp.cumcount()
    expected: Series = Series([0, 1, 2, 0, 1, 2])
    tm.assert_series_equal(result, expected)
    result = grp.transform("cumcount")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "keys",
    [["A1"], ["A1", "A2"]],
)
def test_null_group_lambda_self(
    sort: bool,
    dropna: bool,
    keys: List[str],
) -> None:
    size: int = 50
    nulls1: np.ndarray = np.random.default_rng(2).choice([False, True], size)
    nulls2: np.ndarray = np.random.default_rng(2).choice([False, True], size)
    nulls_grouper: np.ndarray = nulls1 if len(keys) == 1 else nulls1 | nulls2
    a1: np.ndarray = np.random.default_rng(2).integers(0, 5, size=size).astype(float)
    a1[nulls1] = np.nan
    a2: np.ndarray = np.random.default_rng(2).integers(0, 5, size=size).astype(float)
    a2[nulls2] = np.nan
    values: np.ndarray = np.random.default_rng(2).integers(0, 5, size=a1.shape)
    df: DataFrame = DataFrame(
        {"A1": a1, "A2": a2, "B": values}
    )
    expected_values: np.ndarray = values
    if dropna and nulls_grouper.any():
        expected_values = expected_values.astype(float)
        expected_values[nulls_grouper] = np.nan
    expected: DataFrame = DataFrame(expected_values, columns=["B"])
    gb: pd.core.groupby.generic.GroupBy = df.groupby(keys, dropna=dropna, sort=sort)
    result: DataFrame = gb[["B"]].transform(lambda x: x)
    tm.assert_frame_equal(result, expected)


def test_null_group_str_reducer(
    request: pytest.FixtureRequest,
    dropna: bool,
    reduction_func: str,
) -> None:
    if reduction_func == "corrwith":
        msg: str = "incorrectly raises"
        request.applymarker(pytest.mark.xfail(reason=msg))
    index: List[int] = [1, 2, 3, 4]
    df: DataFrame = DataFrame(
        {"A": [1, 1, np.nan, np.nan], "B": [1, 2, 2, 3]},
        index=index,
    )
    gb: pd.core.groupby.generic.GroupBy = df.groupby("A", dropna=dropna)
    args: Tuple[Any, ...] = get_groupby_method_args(reduction_func, df)
    if reduction_func == "first":
        expected: DataFrame = DataFrame({"B": [1, 1, 2, 2]}, index=index)
    elif reduction_func == "last":
        expected = DataFrame({"B": [2, 2, 3, 3]}, index=index)
    elif reduction_func == "nth":
        expected = DataFrame({"B": [1, 1, 2, 2]}, index=index)
    elif reduction_func == "size":
        expected = Series([2, 2, 2, 2], index=index)
    elif reduction_func == "corrwith":
        expected = DataFrame({"B": [1.0, 1.0, 1.0, 1.0]}, index=index)
    else:
        expected_gb = df.groupby("A", dropna=False)
        buffer: List[pd.Series] = []
        for idx, group in expected_gb:
            res = getattr(group["B"], reduction_func)()
            buffer.append(Series(res, index=group.index))
        expected = concat(buffer).to_frame("B")
    if dropna:
        dtype: Union[type, Any] = object if reduction_func in ("any", "all") else float
        expected = expected.astype(dtype)
        if expected.ndim == 2:
            expected.iloc[[2, 3], 0] = np.nan
        else:
            expected.iloc[[2, 3]] = np.nan
    result: Union[Series, DataFrame] = gb.transform(reduction_func, *args)
    tm.assert_equal(result, expected)


def test_null_group_str_transformer(
    dropna: bool,
    transformation_func: str,
) -> None:
    df: DataFrame = DataFrame(
        {"A": [1, 1, np.nan], "B": [1, 2, 2]},
        index=[1, 2, 3],
    )
    args: Tuple[Any, ...] = get_groupby_method_args(transformation_func, df)
    gb: pd.core.groupby.generic.GroupBy = df.groupby("A", dropna=dropna)
    buffer: List[DataFrame] = []
    for k, (idx, group) in enumerate(gb):
        if transformation_func == "cumcount":
            res: DataFrame = DataFrame({"B": range(len(group))}, index=group.index)
        elif transformation_func == "ngroup":
            res = DataFrame([k] * len(group), index=group.index, columns=["B"])
        else:
            res = getattr(group[["B"]], transformation_func)(*args)
        buffer.append(res)
    if dropna:
        dtype: Optional[type] = object if transformation_func in ("any", "all") else None
        buffer.append(DataFrame([[np.nan]], index=[3], dtype=dtype, columns=["B"]))
    expected: DataFrame = concat(buffer)
    if transformation_func in ("cumcount", "ngroup"):
        expected = expected["B"].rename(None)
    result: Union[Series, DataFrame] = gb.transform(transformation_func, *args)
    tm.assert_equal(result, expected)


def test_null_group_str_reducer_series(
    request: pytest.FixtureRequest,
    dropna: bool,
    reduction_func: str,
) -> None:
    index: List[int] = [1, 2, 3, 4]
    ser: Series = Series([1, 2, 2, 3], index=index)
    gb: pd.core.groupby.generic.GroupBy = ser.groupby([1, 1, np.nan, np.nan], dropna=dropna)
    if reduction_func == "corrwith":
        expected = Series([1, 1, 2, 2], name="B")
    elif reduction_func == "first":
        expected = Series([1, 1, 2, 2], index=index)
    elif reduction_func == "last":
        expected = Series([2, 2, 3, 3], index=index)
    elif reduction_func == "nth":
        expected = Series([1, 1, 2, 2], index=index)
    elif reduction_func == "size":
        expected = Series([2, 2, 2, 2], index=index)
    else:
        expected_gb = ser.groupby([1, 1, np.nan, np.nan], dropna=False)
        buffer: List[Series] = []
        for idx, group in expected_gb:
            res = getattr(group, reduction_func)()
            buffer.append(Series(res, index=group.index))
        expected = concat(buffer)
    if dropna:
        dtype: Union[type, Any] = object if reduction_func in ("any", "all") else float
        expected = expected.astype(dtype)
        expected.iloc[[2, 3]] = np.nan
    result: Series = gb.transform(reduction_func, *get_groupby_method_args(reduction_func, ser))
    tm.assert_series_equal(result, expected)


def test_null_group_str_transformer_series(
    dropna: bool,
    transformation_func: str,
) -> None:
    ser: Series = Series([1, 2, 2], index=[1, 2, 3])
    args: Tuple[Any, ...] = get_groupby_method_args(transformation_func, ser)
    gb: pd.core.groupby.generic.GroupBy = ser.groupby([1, 1, np.nan], dropna=dropna)
    buffer: List[Series] = []
    for k, (idx, group) in enumerate(gb):
        if transformation_func == "cumcount":
            res: Series = Series(range(len(group)), index=group.index)
        elif transformation_func == "ngroup":
            res = Series(k, index=group.index)
        else:
            res = getattr(group, transformation_func)(*args)
        buffer.append(res)
    if dropna:
        dtype: Optional[type] = object if transformation_func in ("any", "all") else None
        buffer.append(Series([np.nan], index=[3], dtype=dtype))
    expected: Series = concat(buffer)
    result: Series = gb.transform(transformation_func, *args)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "func, expected_values",
    [
        (
            Series.sort_values,
            [5, 4, 3, 2, 1],
        ),
        (
            lambda x: x.head(1),
            [5.0, np.nan, 3, 2, np.nan],
        ),
    ],
)
@pytest.mark.parametrize("keys", [["a1"], ["a1", "a2"]])
@pytest.mark.parametrize("keys_in_index", [True, False])
@pytest.mark.parametrize("frame_or_series", [pd.DataFrame, pd.Series])
def test_transform_aligns(
    func: Callable[[pd.Series], Any],
    frame_or_series: type,
    expected_values: List[Optional[Union[int, float]]],
    keys: List[str],
    keys_in_index: bool,
) -> None:
    df: DataFrame = DataFrame({"a1": [1, 1, 3, 2, 2], "b": [5, 4, 3, 2, 1]})
    if "a2" in keys:
        df["a2"] = df["a1"]
    if keys_in_index:
        df = df.set_index(keys, append=True)
    gb: pd.core.groupby.generic.GroupBy = df.groupby(keys)
    if frame_or_series is pd.Series:
        gb = gb["b"]
    result: Union[Series, DataFrame] = gb.transform(func)
    expected: Union[Series, DataFrame] = DataFrame({"b": expected_values}, index=df.index)
    if frame_or_series is pd.Series:
        expected = expected["b"]
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "how, expected",
    [
        (
            "idxmin",
            ["1/1/2011"] * 2 + ["1/3/2011"] * 7 + ["1/10/2011"],
        ),
        (
            "idxmax",
            ["1/2/2011"] * 2 + ["1/9/2011"] * 7 + ["1/10/2011"],
        ),
    ],
)
@pytest.mark.parametrize(
    "numeric_only",
    [True, False],
)
def test_idxmin_idxmax_transform_args(
    how: str,
    skipna: bool,
    numeric_only: bool,
) -> None:
    df: DataFrame = DataFrame(
        {"a": [1, 1, 1, 2], "b": [3.0, 4.0, np.nan, 6.0], "c": list("abcd")}
    )
    gb: pd.core.groupby.generic.GroupBy = df.groupby("a")
    if skipna:
        result: Union[Series, DataFrame] = gb.transform(
            how, skipna=skipna, numeric_only=numeric_only
        )
        expected: Union[Series, DataFrame] = gb.transform(
            how, skipna=skipna, numeric_only=numeric_only
        )
        tm.assert_frame_equal(result, expected)
    else:
        msg: str = f"DataFrameGroupBy.{how} with skipna=False encountered an NA value"
        with pytest.raises(ValueError, match=msg):
            gb.transform(how, skipna, numeric_only)


def test_transform_sum_one_column_no_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0]})
    series: Series = Series(["Y"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_no_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "B", "C"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "B", "A"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [4936.0, -93204, 4936.0]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_missing_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "C"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0, -93204, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_matching_labels_and_missing_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "A"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [-93203.0, -93203.0, np.nan]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_min_one_unobserved_category_no_type_coercion(dtype: str) -> None:
    df: DataFrame = DataFrame(
        {"A": Categorical([1], categories=[1, 2, 3]), "B": [3, 4, 5]}
    )
    df["B"] = df["B"].astype(dtype)
    gb: pd.core.groupby.generic.GroupBy = df.groupby("A", observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"B": [3, 3, 5]}, dtype=dtype)
    tm.assert_frame_equal(expected, result)


def test_min_all_empty_data_no_type_coercion() -> None:
    df: DataFrame = DataFrame(
        {"X": Categorical([], categories=[1, "randomcat", 100]), "Y": []}
    )
    df["Y"] = df["Y"].astype("int32")
    gb: pd.core.groupby.generic.GroupBy = df.groupby("X", observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"Y": []}, dtype="int32")
    tm.assert_frame_equal(expected, result)


def test_min_one_dim_no_type_coercion() -> None:
    df: DataFrame = DataFrame({"Y": [9435, -5465765, 5055, 0, 954960]})
    df["Y"] = df["Y"].astype("int32")
    categories: Categorical = Categorical([1, 2, 2, 5, 1], categories=[1, 2, 3, 4, 5])
    gb: pd.core.groupby.generic.GroupBy = df.groupby(categories, observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"Y": [9435, -5465765, -5465765, 0, 9435]}, dtype="int32")
    tm.assert_frame_equal(expected, result)


def test_nan_in_cumsum_group_label() -> None:
    df: DataFrame = DataFrame(
        {"A": [1, None], "B": [2, 3]},
        dtype="Int16",
    )
    gb: pd.core.groupby.generic.GroupBy = df.groupby("A")["B"]
    result: Series = gb.cumsum()
    expected: Series = Series([2, None], dtype="Int16", name="B")
    tm.assert_series_equal(expected, result)


@pytest.mark.parametrize(
    "func, expected_values",
    [
        (
            Series.sort_values,
            [5, 4, 3, 2, 1],
        ),
        (
            lambda x: x.head(1),
            [5.0, np.nan, 3, 2, np.nan],
        ),
    ],
)
@pytest.mark.parametrize(
    "keys",
    [["a1"], ["a1", "a2"]],
)
@pytest.mark.parametrize("keys_in_index", [True, False])
@pytest.mark.parametrize("frame_or_series", [pd.DataFrame, pd.Series])
def test_transform_aligns(
    func: Callable[[pd.Series], Any],
    frame_or_series: type,
    expected_values: List[Optional[Union[int, float]]],
    keys: List[str],
    keys_in_index: bool,
) -> None:
    df: DataFrame = DataFrame({"a1": [1, 1, 3, 2, 2], "b": [5, 4, 3, 2, 1]})
    if "a2" in keys:
        df["a2"] = df["a1"]
    if keys_in_index:
        df = df.set_index(keys, append=True)
    gb: pd.core.groupby.generic.GroupBy = df.groupby(keys)
    if frame_or_series is pd.Series:
        gb = gb["b"]
    result: Union[Series, DataFrame] = gb.transform(func)
    expected: Union[Series, DataFrame] = DataFrame({"b": expected_values}, index=df.index)
    if frame_or_series is pd.Series:
        expected = expected["b"]
    tm.assert_equal(result, expected)


def test_transform_sum_one_column_no_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0]})
    series: Series = Series(["Y"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_no_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "B", "C"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "B", "A"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [4936.0, -93204, 4936.0]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_missing_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "C"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0, -93204, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_matching_labels_and_missing_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "A"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [-93203.0, -93203.0, np.nan]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_min_one_unobserved_category_no_type_coercion(dtype: str) -> None:
    df: DataFrame = DataFrame(
        {"A": Categorical([1], categories=[1, 2, 3]), "B": [3, 4, 5]}
    )
    df["B"] = df["B"].astype(dtype)
    gb: pd.core.groupby.generic.GroupBy = df.groupby("A", observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"B": [3, 3, 5]}, dtype=dtype)
    tm.assert_frame_equal(expected, result)


def test_min_all_empty_data_no_type_coercion() -> None:
    df: DataFrame = DataFrame(
        {"X": Categorical([], categories=[1, "randomcat", 100]), "Y": []}
    )
    df["Y"] = df["Y"].astype("int32")
    gb: pd.core.groupby.generic.GroupBy = df.groupby("X", observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"Y": []}, dtype="int32")
    tm.assert_frame_equal(expected, result)


def test_min_one_dim_no_type_coercion() -> None:
    df: DataFrame = DataFrame({"Y": [9435, -5465765, 5055, 0, 954960]})
    df["Y"] = df["Y"].astype("int32")
    categories: Categorical = Categorical([1, 2, 2, 5, 1], categories=[1, 2, 3, 4, 5])
    gb: pd.core.groupby.generic.GroupBy = df.groupby(categories, observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"Y": [9435, -5465765, -5465765, 0, 9435]}, dtype="int32")
    tm.assert_frame_equal(expected, result)


def test_nan_in_cumsum_group_label() -> None:
    df: DataFrame = DataFrame(
        {"A": [1, None], "B": [2, 3]},
        dtype="Int16",
    )
    gb: pd.core.groupby.generic.SeriesGroupBy = df.groupby("A")["B"]
    result: Series = gb.cumsum()
    expected: Series = Series([2, None], dtype="Int16", name="B")
    tm.assert_series_equal(expected, result)


@pytest.mark.parametrize(
    "mix_groupings, as_series, val1, val2, fill_method, limit, exp_vals",
    [
        (
            True,
            True,
            "foo",
            "bar",
            "ffill",
            None,
            [np.nan, np.nan, "foo", "foo", "foo", "bar", "bar", "bar"],
        ),
        (
            True,
            True,
            1,
            2,
            "ffill",
            1,
            [np.nan, np.nan, 1, 1, np.nan, 2, 2, np.nan],
        ),
        (
            True,
            True,
            1.0,
            2.0,
            "bfill",
            None,
            ["foo", "foo", "foo", "bar", "bar", "bar", np.nan, np.nan],
        ),
        (
            False,
            False,
            "foo",
            "bar",
            "bfill",
            1,
            [np.nan, "foo", "foo", np.nan, "bar", "bar", np.nan, np.nan],
        ),
    ],
)
@pytest.mark.parametrize("func", ["ffill", "bfill", "shift"])
@pytest.mark.parametrize(
    "key, val",
    [("level", 0), ("by", Series([0]))],
)
def test_group_fill_methods(
    mix_groupings: bool,
    as_series: bool,
    val1: Union[str, int, float],
    val2: Union[str, int, float],
    fill_method: str,
    limit: Optional[int],
    exp_vals: List[Optional[Union[str, int, float]]],
    key: str,
    val: Any,
) -> None:
    vals: List[Optional[Union[str, int, float]]] = [np.nan, np.nan, val1, np.nan, np.nan, val2, np.nan, np.nan]
    _exp_vals: List[Optional[Union[str, int, float]]] = list(exp_vals)
    for index, exp_val in enumerate(_exp_vals):
        if exp_val == "val1":
            _exp_vals[index] = val1
        elif exp_val == "val2":
            _exp_vals[index] = val2
    if mix_groupings:
        keys_list: List[str] = ["a", "b"] * len(vals)

        def interweave(list_obj: List[Any]) -> List[Any]:
            temp: List[Any] = []
            for x in list_obj:
                temp.extend([x, x])
            return temp

        _exp_vals = interweave(_exp_vals)
        vals = interweave(vals)
    else:
        keys_list = ["a"] * len(vals) + ["b"] * len(vals)
        _exp_vals = _exp_vals * 2
        vals = vals * 2
    df: DataFrame = DataFrame({"key": keys_list, "val": vals})
    if as_series:
        result: Union[Series, DataFrame] = getattr(
            df.groupby("key")["val"], fill_method
        )(limit=limit)
        exp: Union[Series, DataFrame] = Series(_exp_vals, name="val")
        tm.assert_series_equal(result, exp)
    else:
        result = getattr(df.groupby("key"), fill_method)(limit=limit)
        exp = DataFrame({"val": _exp_vals})
        tm.assert_frame_equal(result, exp)


@pytest.mark.parametrize("func", ["ffill", "bfill"])
def test_pad_stable_sorting(fill_method: str) -> None:
    x: List[int] = [0] * 20
    y: List[Optional[int]] = [np.nan] * 10 + [1] * 10
    if fill_method == "bfill":
        y = y[::-1]
    df: DataFrame = DataFrame({"x": x, "y": y})
    expected: DataFrame = df.drop("x", axis=1)
    result: DataFrame = getattr(df.groupby("x"), fill_method)()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "freq",
    [
        None,
        pytest.param(
            "D",
            marks=pytest.mark.xfail(reason="GH#23918 before method uses freq in vectorized approach"),
        ),
    ],
)
@pytest.mark.parametrize("periods", [1, -1])
@pytest.mark.parametrize("frame_or_series", [pd.DataFrame, pd.Series])
def test_pct_change(
    frame_or_series: type,
    freq: Optional[str],
    periods: int,
) -> None:
    vals: List[Optional[float]] = [3, np.nan, np.nan, np.nan, 1, 2, 4, 10, np.nan, 4]
    keys: List[str] = ["a", "b"]
    key_v: np.ndarray = np.repeat(keys, len(vals))
    df: DataFrame = DataFrame({"key": key_v, "vals": vals * 2})
    df_g = df
    grp: pd.core.groupby.generic.GroupBy = df_g.groupby(df.key)
    expected: Series = grp["vals"].obj / grp["vals"].shift(periods) - 1
    gb: pd.core.groupby.generic.GroupBy = df.groupby("key")
    if frame_or_series is pd.Series:
        gb = gb["vals"]
    else:
        expected = expected.to_frame("vals")
    result: Union[Series, DataFrame] = gb.pct_change(periods=periods, freq=freq)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "func, expected_status",
    [
        (
            "ffill",
            ["shrt", "shrt", "lng", np.nan, "shrt", "ntrl", "ntrl"],
        ),
        (
            "bfill",
            ["shrt", "lng", "lng", "shrt", "shrt", "ntrl", np.nan],
        ),
    ],
)
def test_ffill_bfill_non_unique_multilevel(
    func: str,
    expected_status: List[Optional[str]],
) -> None:
    date: pd.DatetimeIndex = pd.to_datetime(
        ["2018-01-01", "2018-01-01", "2018-01-01", "2018-01-01", "2018-01-02", "2018-01-01", "2018-01-02"]
    )
    symbol: List[str] = ["MSFT", "MSFT", "MSFT", "AAPL", "AAPL", "TSLA", "TSLA"]
    status: List[Optional[str]] = ["shrt", np.nan, "lng", np.nan, "shrt", "ntrl", np.nan]
    df: DataFrame = DataFrame(
        {"date": date, "symbol": symbol, "status": status}
    )
    df = df.set_index(["date", "symbol"])
    result: Series = df.groupby("symbol")["status"].transform(func)
    index: MultiIndex = MultiIndex.from_tuples(
        list(zip(date, symbol)), names=["date", "symbol"]
    )
    expected: Series = Series(expected_status, index=index, name="status")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", [np.any, np.all])
def test_any_all_np_func(func: Callable[[pd.Series], bool]) -> None:
    df: DataFrame = DataFrame([["foo", True], [np.nan, True], ["foo", True]], columns=["key", "val"])
    exp: Series = Series([True, np.nan, True], name="val")
    res: Series = df.groupby("key")["val"].transform(func)
    tm.assert_series_equal(res, exp)


def test_groupby_transform_rename() -> None:
    def demean_rename(x: Union[Series, DataFrame]) -> Union[Series, DataFrame]:
        result = x - x.mean()
        if isinstance(x, Series):
            return result
        result = result.rename(columns={c: f"{c}_demeaned" for c in result.columns})
        return result

    df: DataFrame = DataFrame({"group": list("ababa"), "value": [1, 1, 1, 2, 2]})
    expected: DataFrame = DataFrame({"value": [-1.0 / 3, -0.5, -1.0 / 3, 0.5, 2.0 / 3]})
    result: DataFrame = df.groupby("group").transform(demean_rename)
    tm.assert_frame_equal(result, expected)
    result_single: Series = df.groupby("group").value.transform(demean_rename)
    tm.assert_series_equal(result_single, expected["value"])


@pytest.mark.parametrize("func", [min, max, np.min, np.max, "first", "last"])
def test_groupby_transform_timezone_column(
    func: Union[str, Callable[[pd.Series], Any]]
) -> None:
    ts: pd.Timestamp = pd.to_datetime("now", utc=True).tz_convert("Asia/Singapore")
    result: DataFrame = DataFrame({"end_time": [ts], "id": [1]})
    result["max_end_time"] = result.groupby("id").end_time.transform(func)
    expected: DataFrame = DataFrame([[ts, 1, ts]], columns=["end_time", "id", "max_end_time"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func, values",
    [
        (
            "idxmin",
            ["1/1/2011"] * 2 + ["1/3/2011"] * 7 + ["1/10/2011"],
        ),
        (
            "idxmax",
            ["1/2/2011"] * 2 + ["1/9/2011"] * 7 + ["1/10/2011"],
        ),
    ],
)
def test_groupby_transform_with_datetimes(
    func: str,
    values: List[str],
) -> None:
    dates: pd.DatetimeIndex = date_range("1/1/2011", periods=10, freq="D")
    stocks: DataFrame = DataFrame({"price": np.arange(10.0)}, index=dates)
    stocks["week_id"] = dates.isocalendar().week
    result: Series = stocks.groupby(stocks["week_id"])["price"].transform(func)
    expected: Series = Series(
        data=pd.to_datetime(values).as_unit("ns"),
        index=dates,
        name="price",
    )
    tm.assert_series_equal(result, expected)


def test_transform_sum_one_column_no_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0]})
    series: Series = Series(["Y"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_no_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "B", "C"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "B", "A"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [4936.0, -93204, 4936.0]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_missing_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "C"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0, -93204, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_matching_labels_and_missing_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "A"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [-93203.0, -93203.0, np.nan]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_min_one_unobserved_category_no_type_coercion(dtype: str) -> None:
    df: DataFrame = DataFrame(
        {"A": Categorical([1], categories=[1, 2, 3]), "B": [3, 4, 5]}
    )
    df["B"] = df["B"].astype(dtype)
    gb: pd.core.groupby.generic.GroupBy = df.groupby("A", observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"B": [3, 3, 5]}, dtype=dtype)
    tm.assert_frame_equal(expected, result)


def test_min_all_empty_data_no_type_coercion() -> None:
    df: DataFrame = DataFrame(
        {"X": Categorical([], categories=[1, "randomcat", 100]), "Y": []}
    )
    df["Y"] = df["Y"].astype("int32")
    gb: pd.core.groupby.generic.GroupBy = df.groupby("X", observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"Y": []}, dtype="int32")
    tm.assert_frame_equal(expected, result)


def test_min_one_dim_no_type_coercion() -> None:
    df: DataFrame = DataFrame({"Y": [9435, -5465765, 5055, 0, 954960]})
    df["Y"] = df["Y"].astype("int32")
    categories: Categorical = Categorical([1, 2, 2, 5, 1], categories=[1, 2, 3, 4, 5])
    gb: pd.core.groupby.generic.GroupBy = df.groupby(categories, observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"Y": [9435, -5465765, -5465765, 0, 9435]}, dtype="int32")
    tm.assert_frame_equal(expected, result)


def test_nan_in_cumsum_group_label() -> None:
    df: DataFrame = DataFrame(
        {"A": [1, None], "B": [2, 3]},
        dtype="Int16",
    )
    gb: pd.core.groupby.generic.SeriesGroupBy = df.groupby("A")["B"]
    result: Series = gb.cumsum()
    expected: Series = Series([2, None], dtype="Int16", name="B")
    tm.assert_series_equal(expected, result)


@pytest.mark.parametrize(
    "mix_groupings, as_series, val1, val2, fill_method, limit, exp_vals",
    [
        (
            True,
            True,
            "foo",
            "bar",
            "ffill",
            None,
            [np.nan, np.nan, "foo", "foo", "foo", "bar", "bar", "bar"],
        ),
        (
            True,
            True,
            1,
            2,
            "ffill",
            1,
            [np.nan, np.nan, 1, 1, np.nan, 2, 2, np.nan],
        ),
        (
            True,
            True,
            1.0,
            2.0,
            "bfill",
            None,
            ["foo", "foo", "foo", "bar", "bar", "bar", np.nan, np.nan],
        ),
        (
            False,
            False,
            "foo",
            "bar",
            "bfill",
            1,
            [np.nan, "foo", "foo", np.nan, "bar", "bar", np.nan, np.nan],
        ),
    ],
)
@pytest.mark.parametrize("func", ["ffill", "bfill", "shift"])
@pytest.mark.parametrize(
    "key, val",
    [("level", 0), ("by", Series([0]))],
)
@pytest.mark.parametrize(
    "fill_method, limit, exp_vals",
    [
        ("ffill", None, [np.nan, np.nan, "val1", "val1", "val1", "val2", "val2", "val2"]),
        (
            "ffill",
            1,
            [np.nan, np.nan, "val1", "val1", np.nan, "val2", "val2", np.nan],
        ),
        (
            "bfill",
            None,
            ["val1", "val1", "val1", "val2", "val2", "val2", np.nan, np.nan],
        ),
        (
            "bfill",
            1,
            [np.nan, "val1", "val1", np.nan, "val2", "val2", np.nan, np.nan],
        ),
    ],
)
def test_group_fill_methods(
    mix_groupings: bool,
    as_series: bool,
    val1: Union[str, int, float],
    val2: Union[str, int, float],
    fill_method: str,
    limit: Optional[int],
    exp_vals: List[Optional[Union[str, int, float]]],
    key: str,
    val: Any,
) -> None:
    pass  # Placeholder to avoid duplication


def test_transform_invalid_name_raises() -> None:
    df: DataFrame = DataFrame({"a": [0, 1, 1, 2]})
    gb: pd.core.groupby.generic.GroupBy = df.groupby(["a", "b", "b", "c"])
    with pytest.raises(ValueError, match="not a valid function name"):
        gb.transform("some_arbitrary_name")
    assert hasattr(gb, "aggregate")
    with pytest.raises(ValueError, match="not a valid function name"):
        gb.transform("aggregate")
    gb_series: pd.core.groupby.generic.SeriesGroupBy = df["a"].groupby(["a", "b", "b", "c"])
    with pytest.raises(ValueError, match="not a valid function name"):
        gb_series.transform("some_arbitrary_name")


def test_transform_agg_by_name(
    request: pytest.FixtureRequest,
    reduction_func: str,
    frame_or_series: type,
) -> None:
    func: str = reduction_func
    obj: DataFrame = DataFrame(
        {"a": [0, 0, 0, 0, 1, 1, 1, 1], "b": range(8)},
        index=["A", "B", "C", "D", "E", "F", "G", "H"],
    )
    if frame_or_series is pd.Series:
        obj = obj["a"]
    gb: pd.core.groupby.generic.GroupBy = obj.groupby(np.repeat([0, 1], 4))
    if func == "corrwith" and isinstance(obj, Series):
        assert not hasattr(gb, func)
        return
    args: Tuple[Any, ...] = get_groupby_method_args(reduction_func, obj)
    if func == "corrwith":
        warn: Optional[type] = FutureWarning
        msg: str = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        msg = ""
    with tm.assert_produces_warning(warn, match=msg):
        result: Union[Series, DataFrame] = gb.transform(func, *args)
    tm.assert_index_equal(result.index, obj.index)
    if func not in ("ngroup", "size") and obj.ndim == 2:
        tm.assert_index_equal(result.columns, obj.columns)
    assert len(set(DataFrame(result).iloc[-4:, -1])) == 1


def test_transform_lambda_with_datetimetz() -> None:
    df: DataFrame = DataFrame(
        {
            "time": [
                Timestamp("2010-07-15 03:14:45"),
                Timestamp("2010-11-19 18:47:06"),
            ],
            "timezone": ["Etc/GMT+4", "US/Eastern"],
        }
    )
    result: Series = df.groupby(["timezone"])["time"].transform(
        lambda x: x.dt.tz_localize(x.name)
    )
    expected: Series = Series(
        [
            Timestamp("2010-07-15 03:14:45", tz="Etc/GMT+4"),
            Timestamp("2010-11-19 18:47:06", tz="US/Eastern"),
        ],
        name="time",
    )
    tm.assert_series_equal(result, expected)


def test_transform_fastpath_raises() -> None:
    df: DataFrame = DataFrame({"A": [1, 1, 2, 2], "B": [1, -1, 1, 2]})
    gb: pd.core.groupby.generic.GroupBy = df.groupby("A")

    def func(grp: Union[Series, DataFrame]) -> Union[Series, DataFrame]:
        if grp.ndim == 2:
            raise NotImplementedError("Don't cross the streams")
        return grp * 2

    obj: Union[Series, DataFrame] = gb._obj_with_exclusions
    gen = gb._grouper.get_iterator(obj)
    fast_path: Callable[[Any], Any]
    slow_path: Callable[[Any], Any]
    _, group = next(gen)
    fast_path, slow_path = gb._define_paths(func)
    with pytest.raises(NotImplementedError, match="Don't cross the streams"):
        fast_path(group)
    result: DataFrame = gb.transform(func)
    expected: DataFrame = DataFrame([2, -2, 2, 4], columns=["B"])
    tm.assert_frame_equal(result, expected)


def test_transform_lambda_indexing() -> None:
    df: DataFrame = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "flux", "foo", "flux"],
            "B": ["one", "one", "two", "three", "two", "six", "five", "three"],
            "C": range(8),
            "D": range(8),
            "E": range(8),
        }
    )
    df = df.set_index(["A", "B"])
    df = df.sort_index()
    result: DataFrame = df.groupby(level="A").transform(lambda x: x.iloc[-1])
    expected: DataFrame = DataFrame(
        {
            "C": [3, 3, 7, 7, 4, 4, 4, 4],
            "D": [3, 3, 7, 7, 4, 4, 4, 4],
            "E": [3, 3, 7, 7, 4, 4, 4, 4],
        },
        index=MultiIndex.from_tuples(
            [
                ("bar", "one"),
                ("bar", "three"),
                ("flux", "six"),
                ("flux", "three"),
                ("foo", "five"),
                ("foo", "one"),
                ("foo", "two"),
                ("foo", "two"),
            ],
            names=["A", "B"],
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_nan_in_cumsum_group_label() -> None:
    df: DataFrame = DataFrame(
        {"A": [1, None], "B": [2, 3]},
        dtype="Int16",
    )
    gb: pd.core.groupby.generic.SeriesGroupBy = df.groupby("A")["B"]
    result: Series = gb.cumsum()
    expected: Series = Series([2, None], dtype="Int16", name="B")
    tm.assert_series_equal(expected, result)


@pytest.mark.parametrize(
    "keys",
    [["A1"], ["A1", "A2"]],
)
@pytest.mark.parametrize(
    "sort, dropna",
    [
        (True, True),
        (False, False),
    ],
)
def test_null_group_lambda_self(
    sort: bool,
    dropna: bool,
    keys: List[str],
) -> None:
    size: int = 50
    nulls1: np.ndarray = np.random.default_rng(2).choice([False, True], size)
    nulls2: np.ndarray = np.random.default_rng(2).choice([False, True], size)
    nulls_grouper: np.ndarray = nulls1 if len(keys) == 1 else nulls1 | nulls2
    a1: np.ndarray = np.random.default_rng(2).integers(0, 5, size=size).astype(float)
    a1[nulls1] = np.nan
    a2: np.ndarray = np.random.default_rng(2).integers(0, 5, size=size).astype(float)
    a2[nulls2] = np.nan
    values: np.ndarray = np.random.default_rng(2).integers(0, 5, size=a1.shape)
    df: DataFrame = DataFrame(
        {"A1": a1, "A2": a2, "B": values}
    )
    expected_values: np.ndarray = values
    if dropna and nulls_grouper.any():
        expected_values = expected_values.astype(float)
        expected_values[nulls_grouper] = np.nan
    expected: DataFrame = DataFrame(expected_values, columns=["B"])
    gb: pd.core.groupby.generic.GroupBy = df.groupby(keys, dropna=dropna, sort=sort)
    result: DataFrame = gb[["B"]].transform(lambda x: x)
    tm.assert_frame_equal(result, expected)


def test_null_group_str_reducer(
    request: pytest.FixtureRequest,
    dropna: bool,
    reduction_func: str,
) -> None:
    if reduction_func == "corrwith":
        msg: str = "incorrectly raises"
        request.applymarker(pytest.mark.xfail(reason=msg))
    index: List[int] = [1, 2, 3, 4]
    df: DataFrame = DataFrame(
        {"A": [1, 1, np.nan, np.nan], "B": [1, 2, 2, 3]},
        index=index,
    )
    gb: pd.core.groupby.generic.GroupBy = df.groupby("A", dropna=dropna)
    args: Tuple[Any, ...] = get_groupby_method_args(reduction_func, df)
    if reduction_func == "first":
        expected: DataFrame = DataFrame({"B": [1, 1, 2, 2]}, index=index)
    elif reduction_func == "last":
        expected = DataFrame({"B": [2, 2, 3, 3]}, index=index)
    elif reduction_func == "nth":
        expected = DataFrame({"B": [1, 1, 2, 2]}, index=index)
    elif reduction_func == "size":
        expected = Series([2, 2, 2, 2], index=index)
    elif reduction_func == "corrwith":
        expected = DataFrame({"B": [1.0, 1.0, 1.0, 1.0]}, index=index)
    else:
        expected_gb: pd.core.groupby.generic.GroupBy = df.groupby("A", dropna=False)
        buffer: List[Series] = []
        for idx, group in expected_gb:
            res: Any = getattr(group["B"], reduction_func)()
            buffer.append(Series(res, index=group.index))
        expected = concat(buffer).to_frame("B")
    if dropna:
        dtype: Union[type, Any] = object if reduction_func in ("any", "all") else float
        expected = expected.astype(dtype)
        if expected.ndim == 2:
            expected.iloc[[2, 3], 0] = np.nan
        else:
            expected.iloc[[2, 3]] = np.nan
    result: Union[Series, DataFrame] = gb.transform(reduction_func, *args)
    tm.assert_equal(result, expected)


def test_null_group_str_transformer(
    dropna: bool,
    transformation_func: str,
) -> None:
    df: DataFrame = DataFrame(
        {"A": [1, 1, np.nan], "B": [1, 2, 2]},
        index=[1, 2, 3],
    )
    args: Tuple[Any, ...] = get_groupby_method_args(transformation_func, df)
    gb: pd.core.groupby.generic.GroupBy = df.groupby("A", dropna=dropna)
    buffer: List[DataFrame] = []
    for k, (idx, group) in enumerate(gb):
        if transformation_func == "cumcount":
            res: DataFrame = DataFrame({"B": range(len(group))}, index=group.index)
        elif transformation_func == "ngroup":
            res = DataFrame([k] * len(group), index=group.index, columns=["B"])
        else:
            res = getattr(group[["B"]], transformation_func)(*args)
        buffer.append(res)
    if dropna:
        dtype: Optional[type] = object if transformation_func in ("any", "all") else None
        buffer.append(DataFrame([[np.nan]], index=[3], dtype=dtype, columns=["B"]))
    expected: DataFrame = concat(buffer)
    if transformation_func in ("cumcount", "ngroup"):
        expected = expected["B"].rename(None)
    result: Union[Series, DataFrame] = gb.transform(transformation_func, *args)
    tm.assert_equal(result, expected)


def test_null_group_str_reducer_series(
    request: pytest.FixtureRequest,
    dropna: bool,
    reduction_func: str,
) -> None:
    index: List[int] = [1, 2, 3, 4]
    ser: Series = Series([1, 2, 2, 3], index=index)
    gb: pd.core.groupby.generic.GroupBy = ser.groupby([1, 1, np.nan, np.nan], dropna=dropna)
    if reduction_func == "corrwith":
        expected = Series([1, 1, 2, 2], name="B")
    elif reduction_func == "first":
        expected = Series([1, 1, 2, 2], index=index)
    elif reduction_func == "last":
        expected = Series([2, 2, 3, 3], index=index)
    elif reduction_func == "nth":
        expected = Series([1, 1, 2, 2], index=index)
    elif reduction_func == "size":
        expected = Series([2, 2, 2, 2], index=index)
    elif reduction_func == "corrwith":
        expected = Series([1.0, 1.0, 2.0, 2.0], name="B")
    else:
        expected_gb: pd.core.groupby.generic.GroupBy = ser.groupby([1, 1, np.nan, np.nan], dropna=False)
        buffer: List[Series] = []
        for idx, group in expected_gb:
            res: Any = getattr(group, reduction_func)()
            buffer.append(Series(res, index=group.index))
        expected = concat(buffer)
    if dropna:
        dtype: Union[type, Any] = object if reduction_func in ("any", "all") else float
        expected = expected.astype(dtype)
        expected.iloc[[2, 3]] = np.nan
    result: Series = gb.transform(reduction_func, *get_groupby_method_args(reduction_func, ser))
    tm.assert_series_equal(result, expected)


def test_null_group_str_transformer_series(
    dropna: bool,
    transformation_func: str,
) -> None:
    ser: Series = Series([1, 2, 2], index=[1, 2, 3])
    args: Tuple[Any, ...] = get_groupby_method_args(transformation_func, ser)
    gb: pd.core.groupby.generic.GroupBy = ser.groupby([1, 1, np.nan], dropna=dropna)
    buffer: List[Series] = []
    for k, (idx, group) in enumerate(gb):
        if transformation_func == "cumcount":
            res: Series = Series(range(len(group)), index=group.index)
        elif transformation_func == "ngroup":
            res = Series(k, index=group.index)
        else:
            res = getattr(group, transformation_func)(*args)
        buffer.append(res)
    if dropna:
        dtype: Optional[type] = object if transformation_func in ("any", "all") else None
        buffer.append(Series([np.nan], index=[3], dtype=dtype))
    expected: Series = concat(buffer)
    result: Series = gb.transform(transformation_func, *args)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "func, expected_values",
    [
        (
            Series.sort_values,
            [5, 4, 3, 2, 1],
        ),
        (
            lambda x: x.head(1),
            [5.0, np.nan, 3, 2, np.nan],
        ),
    ],
)
@pytest.mark.parametrize(
    "keys",
    [["a1"], ["a1", "a2"]],
)
@pytest.mark.parametrize("keys_in_index", [True, False])
@pytest.mark.parametrize("frame_or_series", [pd.DataFrame, pd.Series])
def test_transform_aligns(
    func: Callable[[pd.Series], Any],
    frame_or_series: type,
    expected_values: List[Optional[Union[int, float]]],
    keys: List[str],
    keys_in_index: bool,
) -> None:
    df: DataFrame = DataFrame({"a1": [1, 1, 3, 2, 2], "b": [5, 4, 3, 2, 1]})
    if "a2" in keys:
        df["a2"] = df["a1"]
    if keys_in_index:
        df = df.set_index(keys, append=True)
    gb: pd.core.groupby.generic.GroupBy = df.groupby(keys)
    if frame_or_series is pd.Series:
        gb = gb["b"]
    result: Union[Series, DataFrame] = gb.transform(func)
    expected: Union[Series, DataFrame] = DataFrame({"b": expected_values}, index=df.index)
    if frame_or_series is pd.Series:
        expected = expected["b"]
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "func, expected",
    [
        (
            "idxmin",
            ["1/1/2011"] * 2 + ["1/3/2011"] * 7 + ["1/10/2011"],
        ),
        (
            "idxmax",
            ["1/2/2011"] * 2 + ["1/9/2011"] * 7 + ["1/10/2011"],
        ),
    ],
)
def test_groupby_transform_with_datetimes(
    func: str,
    values: List[str],
) -> None:
    dates: pd.DatetimeIndex = date_range("1/1/2011", periods=10, freq="D")
    stocks: DataFrame = DataFrame({"price": np.arange(10.0)}, index=dates)
    stocks["week_id"] = dates.isocalendar().week
    result: Series = stocks.groupby(stocks["week_id"])["price"].transform(func)
    expected: Series = Series(
        data=pd.to_datetime(values).as_unit("ns"),
        index=dates,
        name="price",
    )
    tm.assert_series_equal(result, expected)


def test_transform_sum_one_column_no_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0]})
    series: Series = Series(["Y"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_no_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "B", "C"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "B", "A"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [4936.0, -93204, 4936.0]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_missing_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "C"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0, -93204, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_matching_labels_and_missing_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "A"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [-93203.0, -93203.0, np.nan]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype",
    ["int32", "float32"],
)
def test_min_one_unobserved_category_no_type_coercion(dtype: str) -> None:
    df: DataFrame = DataFrame(
        {"A": Categorical([1], categories=[1, 2, 3]), "B": [3, 4, 5]}
    )
    df["B"] = df["B"].astype(dtype)
    gb: pd.core.groupby.generic.GroupBy = df.groupby("A", observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"B": [3, 3, 5]}, dtype=dtype)
    tm.assert_frame_equal(expected, result)


def test_min_all_empty_data_no_type_coercion() -> None:
    df: DataFrame = DataFrame(
        {"X": Categorical([], categories=[1, "randomcat", 100]), "Y": []}
    )
    df["Y"] = df["Y"].astype("int32")
    gb: pd.core.groupby.generic.GroupBy = df.groupby("X", observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"Y": []}, dtype="int32")
    tm.assert_frame_equal(expected, result)


def test_min_one_dim_no_type_coercion() -> None:
    df: DataFrame = DataFrame({"Y": [9435, -5465765, 5055, 0, 954960]})
    df["Y"] = df["Y"].astype("int32")
    categories: Categorical = Categorical([1, 2, 2, 5, 1], categories=[1, 2, 3, 4, 5])
    gb: pd.core.groupby.generic.GroupBy = df.groupby(categories, observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"Y": [9435, -5465765, -5465765, 0, 9435]}, dtype="int32")
    tm.assert_frame_equal(expected, result)


def test_nan_in_cumsum_group_label() -> None:
    df: DataFrame = DataFrame(
        {"A": [1, None], "B": [2, 3]},
        dtype="Int16",
    )
    gb: pd.core.groupby.generic.SeriesGroupBy = df.groupby("A")["B"]
    result: Series = gb.cumsum()
    expected: Series = Series([2, None], dtype="Int16", name="B")
    tm.assert_series_equal(expected, result)


@pytest.mark.parametrize(
    "keys",
    [["A1"], ["A1", "A2"]],
)
@pytest.mark.parametrize(
    "sort, dropna",
    [
        (True, True),
        (False, False),
    ],
)
def test_null_group_lambda_self(
    sort: bool,
    dropna: bool,
    keys: List[str],
) -> None:
    size: int = 50
    nulls1: np.ndarray = np.random.default_rng(2).choice([False, True], size)
    nulls2: np.ndarray = np.random.default_rng(2).choice([False, True], size)
    nulls_grouper: np.ndarray = nulls1 if len(keys) == 1 else nulls1 | nulls2
    a1: np.ndarray = np.random.default_rng(2).integers(0, 5, size=size).astype(float)
    a1[nulls1] = np.nan
    a2: np.ndarray = np.random.default_rng(2).integers(0, 5, size=size).astype(float)
    a2[nulls2] = np.nan
    values: np.ndarray = np.random.default_rng(2).integers(0, 5, size=a1.shape)
    df: DataFrame = DataFrame(
        {"A1": a1, "A2": a2, "B": values}
    )
    expected_values: np.ndarray = values
    if dropna and nulls_grouper.any():
        expected_values = expected_values.astype(float)
        expected_values[nulls_grouper] = np.nan
    expected: DataFrame = DataFrame(expected_values, columns=["B"])
    gb: pd.core.groupby.generic.GroupBy = df.groupby(keys, dropna=dropna, sort=sort)
    result: DataFrame = gb[["B"]].transform(lambda x: x)
    tm.assert_frame_equal(result, expected)


def test_null_group_str_reducer(
    request: pytest.FixtureRequest,
    dropna: bool,
    reduction_func: str,
) -> None:
    if reduction_func == "corrwith":
        msg: str = "incorrectly raises"
        request.applymarker(pytest.mark.xfail(reason=msg))
    index: List[int] = [1, 2, 3, 4]
    df: DataFrame = DataFrame(
        {"A": [1, 1, np.nan, np.nan], "B": [1, 2, 2, 3]},
        index=index,
    )
    gb: pd.core.groupby.generic.GroupBy = df.groupby("A", dropna=dropna)
    args: Tuple[Any, ...] = get_groupby_method_args(reduction_func, df)
    if reduction_func == "first":
        expected: DataFrame = DataFrame({"B": [1, 1, 2, 2]}, index=index)
    elif reduction_func == "last":
        expected = DataFrame({"B": [2, 2, 3, 3]}, index=index)
    elif reduction_func == "nth":
        expected = DataFrame({"B": [1, 1, 2, 2]}, index=index)
    elif reduction_func == "size":
        expected = Series([2, 2, 2, 2], index=index)
    elif reduction_func == "corrwith":
        expected = DataFrame({"B": [1.0, 1.0, 1.0, 1.0]}, index=index)
    else:
        expected_gb: pd.core.groupby.generic.GroupBy = df.groupby("A", dropna=False)
        buffer: List[Series] = []
        for idx, group in expected_gb:
            res: Any = getattr(group["B"], reduction_func)()
            buffer.append(Series(res, index=group.index))
        expected = concat(buffer).to_frame("B")
    if dropna:
        dtype: Union[type, Any] = object if reduction_func in ("any", "all") else float
        expected = expected.astype(dtype)
        if expected.ndim == 2:
            expected.iloc[[2, 3], 0] = np.nan
        else:
            expected.iloc[[2, 3]] = np.nan
    result: Union[Series, DataFrame] = gb.transform(reduction_func, *args)
    tm.assert_equal(result, expected)


def test_null_group_str_transformer(
    dropna: bool,
    transformation_func: str,
) -> None:
    df: DataFrame = DataFrame(
        {"A": [1, 1, np.nan], "B": [1, 2, 2]},
        index=[1, 2, 3],
    )
    args: Tuple[Any, ...] = get_groupby_method_args(transformation_func, df)
    gb: pd.core.groupby.generic.GroupBy = df.groupby("A", dropna=dropna)
    buffer: List[DataFrame] = []
    for k, (idx, group) in enumerate(gb):
        if transformation_func == "cumcount":
            res: DataFrame = DataFrame({"B": range(len(group))}, index=group.index)
        elif transformation_func == "ngroup":
            res = DataFrame([k] * len(group), index=group.index, columns=["B"])
        else:
            res = getattr(group[["B"]], transformation_func)(*args)
        buffer.append(res)
    if dropna:
        dtype: Optional[type] = object if transformation_func in ("any", "all") else None
        buffer.append(DataFrame([[np.nan]], index=[3], dtype=dtype, columns=["B"]))
    expected: DataFrame = concat(buffer)
    if transformation_func in ("cumcount", "ngroup"):
        expected = expected["B"].rename(None)
    result: Union[Series, DataFrame] = gb.transform(transformation_func, *args)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "func, min_index",
    [
        ("idxmin", "1/1/2011"),
        ("idxmax", "1/2/2011"),
    ],
)
@pytest.mark.parametrize("numeric_only", [True, False])
def test_idxmin_idxmax_transform_args(
    how: str,
    min_index: str,
    numeric_only: bool,
) -> None:
    df: DataFrame = DataFrame(
        {"a": [1, 1, 1, 2], "b": [3.0, 4.0, np.nan, 6.0], "c": list("abcd")}
    )
    gb: pd.core.groupby.generic.GroupBy = df.groupby("a")
    if numeric_only:
        result: DataFrame = gb.transform(
            how, skipna=True, numeric_only=numeric_only
        )
        expected: DataFrame = gb.transform(
            how, skipna=True, numeric_only=numeric_only
        )
    else:
        msg: str = f"DataFrameGroupBy.{how} with skipna=False encountered an NA value"
        with pytest.raises(ValueError, match=msg):
            gb.transform(how, skipna=False, numeric_only=numeric_only)


def test_transform_sum_one_column_no_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0]})
    series: Series = Series(["Y"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_no_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "B", "C"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_matching_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "B", "A"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [4936.0, -93204, 4936.0]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_missing_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "C"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [1.0, -93204, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_transform_sum_one_column_with_matching_labels_and_missing_labels() -> None:
    df: DataFrame = DataFrame({"X": [1.0, -93204, 4935]})
    series: Series = Series(["A", "A"])
    result: DataFrame = df.groupby(series, as_index=False).transform("sum")
    expected: DataFrame = DataFrame({"X": [-93203.0, -93203.0, np.nan]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype",
    ["int32", "float32"],
)
def test_min_one_unobserved_category_no_type_coercion(dtype: str) -> None:
    df: DataFrame = DataFrame(
        {"A": Categorical([1], categories=[1, 2, 3]), "B": [3, 4, 5]}
    )
    df["B"] = df["B"].astype(dtype)
    gb: pd.core.groupby.generic.GroupBy = df.groupby("A", observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"B": [3, 3, 5]}, dtype=dtype)
    tm.assert_frame_equal(expected, result)


def test_min_all_empty_data_no_type_coercion() -> None:
    df: DataFrame = DataFrame(
        {"X": Categorical([], categories=[1, "randomcat", 100]), "Y": []}
    )
    df["Y"] = df["Y"].astype("int32")
    gb: pd.core.groupby.generic.GroupBy = df.groupby("X", observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"Y": []}, dtype="int32")
    tm.assert_frame_equal(expected, result)


def test_min_one_dim_no_type_coercion() -> None:
    df: DataFrame = DataFrame({"Y": [9435, -5465765, 5055, 0, 954960]})
    df["Y"] = df["Y"].astype("int32")
    categories: Categorical = Categorical([1, 2, 2, 5, 1], categories=[1, 2, 3, 4, 5])
    gb: pd.core.groupby.generic.GroupBy = df.groupby(categories, observed=False)
    result: DataFrame = gb.transform("min")
    expected: DataFrame = DataFrame({"Y": [9435, -5465765, -5465765, 0, 9435]}, dtype="int32")
    tm.assert_frame_equal(expected, result)


def test_nan_in_cumsum_group_label() -> None:
    df: DataFrame = DataFrame(
        {"A": [1, None], "B": [2, 3]},
        dtype="Int16",
    )
    gb: pd.core.groupby.generic.SeriesGroupBy = df.groupby("A")["B"]
    result: Series = gb.cumsum()
    expected: Series = Series([2, None], dtype="Int16", name="B")
    tm.assert_series_equal(expected, result)

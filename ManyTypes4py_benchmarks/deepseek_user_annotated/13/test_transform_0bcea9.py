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


def assert_fp_equal(a: np.ndarray, b: np.ndarray) -> None:
    assert (np.abs(a - b) < 1e-12).all()


def test_transform() -> None:
    data = Series(np.arange(9) // 3, index=np.arange(9))

    index = np.arange(9)
    np.random.default_rng(2).shuffle(index)
    data = data.reindex(index)

    grouped = data.groupby(lambda x: x // 3)

    transformed = grouped.transform(lambda x: x * x.sum())
    assert transformed[7] == 12

    # GH 8046
    # make sure that we preserve the input order

    df = DataFrame(
        np.arange(6, dtype="int64").reshape(3, 2), columns=["a", "b"], index=[0, 2, 1]
    )
    key = [0, 0, 1]
    expected = (
        df.sort_index()
        .groupby(key)
        .transform(lambda x: x - x.mean())
        .groupby(key)
        .mean()
    )
    result = df.groupby(key).transform(lambda x: x - x.mean()).groupby(key).mean()
    tm.assert_frame_equal(result, expected)

    def demean(arr: np.ndarray) -> np.ndarray:
        return arr - arr.mean(axis=0)

    people = DataFrame(
        np.random.default_rng(2).standard_normal((5, 5)),
        columns=["a", "b", "c", "d", "e"],
        index=["Joe", "Steve", "Wes", "Jim", "Travis"],
    )
    key = ["one", "two", "one", "two", "one"]
    result = people.groupby(key).transform(demean).groupby(key).mean()
    expected = people.groupby(key, group_keys=False).apply(demean).groupby(key).mean()
    tm.assert_frame_equal(result, expected)

    # GH 8430
    df = DataFrame(
        np.random.default_rng(2).standard_normal((50, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=50, freq="B"),
    )
    g = df.groupby(pd.Grouper(freq="ME"))
    g.transform(lambda x: x - 1)

    # GH 9700
    df = DataFrame({"a": range(5, 10), "b": range(5)})
    result = df.groupby("a").transform(max)
    expected = DataFrame({"b": range(5)})
    tm.assert_frame_equal(result, expected)


def test_transform_fast() -> None:
    df = DataFrame(
        {
            "id": np.arange(10) / 3,
            "val": np.random.default_rng(2).standard_normal(10),
        }
    )

    grp = df.groupby("id")["val"]

    values = np.repeat(grp.mean().values, ensure_platform_int(grp.count().values))
    expected = Series(values, index=df.index, name="val")

    result = grp.transform(np.mean)
    tm.assert_series_equal(result, expected)

    result = grp.transform("mean")
    tm.assert_series_equal(result, expected)


def test_transform_fast2() -> None:
    # GH 12737
    df = DataFrame(
        {
            "grouping": [0, 1, 1, 3],
            "f": [1.1, 2.1, 3.1, 4.5],
            "d": date_range("2014-1-1", "2014-1-4"),
            "i": [1, 2, 3, 4],
        },
        columns=["grouping", "f", "i", "d"],
    )
    result = df.groupby("grouping").transform("first")

    dates = Index(
        [
            Timestamp("2014-1-1"),
            Timestamp("2014-1-2"),
            Timestamp("2014-1-2"),
            Timestamp("2014-1-4"),
        ],
        dtype="M8[ns]",
    )
    expected = DataFrame(
        {"f": [1.1, 2.1, 2.1, 4.5], "d": dates, "i": [1, 2, 2, 4]},
        columns=["f", "i", "d"],
    )
    tm.assert_frame_equal(result, expected)

    # selection
    result = df.groupby("grouping")[["f", "i"]].transform("first")
    expected = expected[["f", "i"]]
    tm.assert_frame_equal(result, expected)


def test_transform_fast3() -> None:
    # dup columns
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["g", "a", "a"])
    result = df.groupby("g").transform("first")
    expected = df.drop("g", axis=1)
    tm.assert_frame_equal(result, expected)


def test_transform_broadcast(tsframe: DataFrame, ts: Series) -> None:
    grouped = ts.groupby(lambda x: x.month)
    result = grouped.transform(np.mean)

    tm.assert_index_equal(result.index, ts.index)
    for _, gp in grouped:
        assert_fp_equal(result.reindex(gp.index), gp.mean())

    grouped = tsframe.groupby(lambda x: x.month)
    result = grouped.transform(np.mean)
    tm.assert_index_equal(result.index, tsframe.index)
    for _, gp in grouped:
        agged = gp.mean(axis=0)
        res = result.reindex(gp.index)
        for col in tsframe:
            assert_fp_equal(res[col], agged[col])


def test_transform_axis_ts(tsframe: DataFrame) -> None:
    # make sure that we are setting the axes correctly
    # in the presence of a non-monotonic indexer
    # GH12713

    base = tsframe.iloc[0:5]
    r = len(base.index)
    c = len(base.columns)
    tso = DataFrame(
        np.random.default_rng(2).standard_normal((r, c)),
        index=base.index,
        columns=base.columns,
        dtype="float64",
    )
    # monotonic
    ts = tso
    grouped = ts.groupby(lambda x: x.weekday(), group_keys=False)
    result = ts - grouped.transform("mean")
    expected = grouped.apply(lambda x: x - x.mean(axis=0))
    tm.assert_frame_equal(result, expected)

    # non-monotonic
    ts = tso.iloc[[1, 0] + list(range(2, len(base)))]
    grouped = ts.groupby(lambda x: x.weekday(), group_keys=False)
    result = ts - grouped.transform("mean")
    expected = grouped.apply(lambda x: x - x.mean(axis=0))
    tm.assert_frame_equal(result, expected)


def test_transform_dtype() -> None:
    # GH 9807
    # Check transform dtype output is preserved
    df = DataFrame([[1, 3], [2, 3]])
    result = df.groupby(1).transform("mean")
    expected = DataFrame([[1.5], [1.5]])
    tm.assert_frame_equal(result, expected)


def test_transform_bug() -> None:
    # GH 5712
    # transforming on a datetime column
    df = DataFrame({"A": Timestamp("20130101"), "B": np.arange(5)})
    result = df.groupby("A")["B"].transform(lambda x: x.rank(ascending=False))
    expected = Series(np.arange(5, 0, step=-1), name="B", dtype="float64")
    tm.assert_series_equal(result, expected)


def test_transform_numeric_to_boolean() -> None:
    # GH 16875
    # inconsistency in transforming boolean values
    expected = Series([True, True], name="A")

    df = DataFrame({"A": [1.1, 2.2], "B": [1, 2]})
    result = df.groupby("B").A.transform(lambda x: True)
    tm.assert_series_equal(result, expected)

    df = DataFrame({"A": [1, 2], "B": [1, 2]})
    result = df.groupby("B").A.transform(lambda x: True)
    tm.assert_series_equal(result, expected)


def test_transform_datetime_to_timedelta() -> None:
    # GH 15429
    # transforming a datetime to timedelta
    df = DataFrame({"A": Timestamp("20130101"), "B": np.arange(5)})
    expected = Series(
        Timestamp("20130101") - Timestamp("20130101"), index=range(5), name="A"
    )

    # this does date math without changing result type in transform
    base_time = df["A"][0]
    result = (
        df.groupby("A")["A"].transform(lambda x: x.max() - x.min() + base_time)
        - base_time
    )
    tm.assert_series_equal(result, expected)

    # this does date math and causes the transform to return timedelta
    result = df.groupby("A")["A"].transform(lambda x: x.max() - x.min())
    tm.assert_series_equal(result, expected)


def test_transform_datetime_to_numeric() -> None:
    # GH 10972
    # convert dt to float
    df = DataFrame({"a": 1, "b": date_range("2015-01-01", periods=2, freq="D")})
    result = df.groupby("a").b.transform(
        lambda x: x.dt.dayofweek - x.dt.dayofweek.mean()
    )

    expected = Series([-0.5, 0.5], name="b")
    tm.assert_series_equal(result, expected)

    # convert dt to int
    df = DataFrame({"a": 1, "b": date_range("2015-01-01", periods=2, freq="D")})
    result = df.groupby("a").b.transform(
        lambda x: x.dt.dayofweek - x.dt.dayofweek.min()
    )

    expected = Series([0, 1], dtype=np.int32, name="b")
    tm.assert_series_equal(result, expected)


def test_transform_casting() -> None:
    # 13046
    times = [
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
    df = DataFrame(
        {
            "A": [f"B-{i}" for i in range(11)],
            "ID3": np.take(
                ["a", "b", "c", "d", "e"], [0, 1, 2, 1, 3, 1, 1, 1, 4, 1, 1]
            ),
            "DATETIME": pd.to_datetime([f"2014-10-08 {time}" for time in times]),
        },
        index=pd.RangeIndex(11, name="idx"),
    )

    result = df.groupby("ID3")["DATETIME"].transform(lambda x: x.diff())
    assert lib.is_np_dtype(result.dtype, "m")

    result = df[["ID3", "DATETIME"]].groupby("ID3").transform(lambda x: x.diff())
    assert lib.is_np_dtype(result.DATETIME.dtype, "m")


def test_transform_multiple(ts: Series) -> None:
    grouped = ts.groupby([lambda x: x.year, lambda x: x.month])
    grouped.transform(lambda x: x * 2)
    grouped.transform(np.mean)


def test_dispatch_transform(tsframe: DataFrame) -> None:
    df = tsframe[::5].reindex(tsframe.index)

    grouped = df.groupby(lambda x: x.month)

    filled = grouped.ffill()
    fillit = lambda x: x.ffill()
    expected = df.groupby(lambda x: x.month).transform(fillit)
    tm.assert_frame_equal(filled, expected)


def test_transform_transformation_func(transformation_func: str) -> None:
    # GH 30918
    df = DataFrame(
        {
            "A": ["foo", "foo", "foo", "foo", "bar", "bar", "baz"],
            "B": [1, 2, np.nan, 3, 3, np.nan, 4],
        },
        index=date_range("2020-01-01", "2020-01-07"),
    )
    if transformation_func == "cumcount":
        test_op = lambda x: x.transform("cumcount")
        mock_op = lambda x: Series(range(len(x)), x.index)
    elif transformation_func == "ngroup":
        counter = -1

        def mock_op(x):
            nonlocal counter
            counter += 1
            return Series(counter, index=x.index)

        test_op = lambda x: x.transform("ngroup")
    else:
        test_op = lambda x: x.transform(transformation_func)
        mock_op = lambda x: getattr(x, transformation_func)()

    result = test_op(df.groupby("A"))

    # pass the group in same order as iterating `for ... in df.groupby(...)`
    # but reorder to match df's index since this is a transform
    groups = [df[["B"]].iloc[4:6], df[["B"]].iloc[6:], df[["B"]].iloc[:4]]
    expected = concat([mock_op(g) for g in groups]).sort_index()
    # sort_index does not preserve the freq
    expected = expected.set_axis(df.index)

    if transformation_func in ("cumcount", "ngroup"):
        tm.assert_series_equal(result, expected)
    else:
        tm.assert_frame_equal(result, expected)


def test_transform_select_columns(df: DataFrame) -> None:
    f = lambda x: x.mean()
    result = df.groupby("A")[["C", "D"]].transform(f)

    selection = df[["C", "D"]]
    expected = selection.groupby(df["A"]).transform(f)

    tm.assert_frame_equal(result, expected)


def test_transform_nuisance_raises(df: DataFrame, using_infer_string: bool) -> None:
    # case that goes through _transform_item_by_item

    df.columns = ["A", "B", "B", "D"]

    # this also tests orderings in transform between
    # series/frame to make sure it's consistent
    grouped = df.groupby("A")

    gbc = grouped["B"]
    msg = "Could not convert"
    if using_infer_string:
        msg = "Cannot perform reduction 'mean' with string dtype"
    with pytest.raises(TypeError, match=msg):
        gbc.transform(lambda x: np.mean(x))

    with pytest.raises(TypeError, match=msg):
        df.groupby("A").transform(lambda x: np.mean(x))


def test_transform_function_aliases(df: DataFrame) -> None:
    result = df.groupby("A").transform("mean", numeric_only=True)
    expected = df.groupby("A")[["C", "D"]].transform(np.mean)
    tm.assert_frame_equal(result, expected)

    result = df.groupby("A")["C"].transform("mean")
    expected = df.groupby("A")["C"].transform(np.mean)
    tm.assert_series_equal(result, expected)


def test_series_fast_transform_date() -> None:
    # GH 13191
    df = DataFrame(
        {"grouping": [np.nan, 1, 1, 3], "d": date_range("2014-1-1", "2014-1-4")}
    )
    result = df.groupby("grouping")["d"].transform("first")
    dates = [
        pd.NaT,
        Timestamp("2014-1-2"),
        Timestamp("2014-1-2"),
        Timestamp("2014-1-4"),
    ]
    expected = Series(dates, name="d", dtype="M8[ns]")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", [lambda x: np.nansum(x), sum])
def test_transform_length(func: callable) -> None:
    # GH 9697
    df = DataFrame({"col1": [1, 1, 2, 2], "col2": [1, 2, 3, np.nan]})
    if func is sum:
        expected = Series([3.0, 3.0, np.nan, np.nan])
    else:
        expected = Series([3.0] * 4)

    results = [
        df.groupby("col1").transform(func)["col2"],
        df.groupby("col1")["col2"].transform(func),
    ]
    for result in results:
        tm.assert_series_equal(result, expected, check_names=False)


def test_transform_coercion() -> None:
    # 14457
    # when we are transforming be sure to not coerce
    # via assignment
    df = DataFrame({"A": ["a", "a", "b", "b"], "B": [0, 1, 3, 4]})
    g = df.groupby("A")

    expected = g.transform(np.mean)
    result = g.transform(lambda x: np.mean(x, axis=0))
    tm.assert_frame_equal(result, expected)


def test_groupby_transform_with_int(using_infer_string: bool) -> None:
    # GH 3740, make sure that we might upcast on item-by-item transform

    # floats
   
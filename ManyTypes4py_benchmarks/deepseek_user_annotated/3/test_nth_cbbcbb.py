import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    isna,
)
import pandas._testing as tm
from typing import Any, Dict, List, Optional, Union, cast


def test_first_last_nth(df: DataFrame) -> None:
    # tests for first / last / nth
    grouped = df.groupby("A")
    first = grouped.first()
    expected = df.loc[[1, 0], ["B", "C", "D"]]
    expected.index = Index(["bar", "foo"], name="A")
    expected = expected.sort_index()
    tm.assert_frame_equal(first, expected)

    nth = grouped.nth(0)
    expected = df.loc[[0, 1]]
    tm.assert_frame_equal(nth, expected)

    last = grouped.last()
    expected = df.loc[[5, 7], ["B", "C", "D"]]
    expected.index = Index(["bar", "foo"], name="A")
    tm.assert_frame_equal(last, expected)

    nth = grouped.nth(-1)
    expected = df.iloc[[5, 7]]
    tm.assert_frame_equal(nth, expected)

    nth = grouped.nth(1)
    expected = df.iloc[[2, 3]]
    tm.assert_frame_equal(nth, expected)

    # it works!
    grouped["B"].first()
    grouped["B"].last()
    grouped["B"].nth(0)

    df = df.copy()
    df.loc[df["A"] == "foo", "B"] = np.nan
    grouped = df.groupby("A")
    assert isna(grouped["B"].first()["foo"])
    assert isna(grouped["B"].last()["foo"])
    assert isna(grouped["B"].nth(0).iloc[0])

    # v0.14.0 whatsnew
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])
    g = df.groupby("A")
    result = g.first()
    expected = df.iloc[[1, 2]].set_index("A")
    tm.assert_frame_equal(result, expected)

    expected = df.iloc[[1, 2]]
    result = g.nth(0, dropna="any")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("method", ["first", "last"])
def test_first_last_with_na_object(method: str, nulls_fixture: Any) -> None:
    # https://github.com/pandas-dev/pandas/issues/32123
    groups = DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, nulls_fixture]}).groupby("a")
    result = getattr(groups, method)()

    if method == "first":
        values = [1, 3]
    else:
        values = [2, 3]

    values = np.array(values, dtype=result["b"].dtype)
    idx = Index([1, 2], name="a")
    expected = DataFrame({"b": values}, index=idx)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index", [0, -1])
def test_nth_with_na_object(index: int, nulls_fixture: Any) -> None:
    # https://github.com/pandas-dev/pandas/issues/32123
    df = DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, nulls_fixture]})
    groups = df.groupby("a")
    result = groups.nth(index)
    expected = df.iloc[[0, 2]] if index == 0 else df.iloc[[1, 3]]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("method", ["first", "last"])
def test_first_last_with_None(method: str) -> None:
    # https://github.com/pandas-dev/pandas/issues/32800
    # None should be preserved as object dtype
    df = DataFrame.from_dict({"id": ["a"], "value": [None]})
    groups = df.groupby("id", as_index=False)
    result = getattr(groups, method)()

    tm.assert_frame_equal(result, df)


@pytest.mark.parametrize("method", ["first", "last"])
@pytest.mark.parametrize(
    "df, expected",
    [
        (
            DataFrame({"id": "a", "value": [None, "foo", np.nan]}),
            DataFrame({"value": ["foo"]}, index=Index(["a"], name="id")),
        ),
        (
            DataFrame({"id": "a", "value": [np.nan]}, dtype=object),
            DataFrame({"value": [None]}, index=Index(["a"], name="id")),
        ),
    ],
)
def test_first_last_with_None_expanded(
    method: str, df: DataFrame, expected: DataFrame
) -> None:
    # GH 32800, 38286
    result = getattr(df.groupby("id"), method)()
    tm.assert_frame_equal(result, expected)


def test_first_last_nth_dtypes() -> None:
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.array(np.random.default_rng(2).standard_normal(8), dtype="float32"),
        }
    )
    df["E"] = True
    df["F"] = 1

    # tests for first / last / nth
    grouped = df.groupby("A")
    first = grouped.first()
    expected = df.loc[[1, 0], ["B", "C", "D", "E", "F"]]
    expected.index = Index(["bar", "foo"], name="A")
    expected = expected.sort_index()
    tm.assert_frame_equal(first, expected)

    last = grouped.last()
    expected = df.loc[[5, 7], ["B", "C", "D", "E", "F"]]
    expected.index = Index(["bar", "foo"], name="A")
    expected = expected.sort_index()
    tm.assert_frame_equal(last, expected)

    nth = grouped.nth(1)
    expected = df.iloc[[2, 3]]
    tm.assert_frame_equal(nth, expected)


def test_first_last_nth_dtypes2() -> None:
    # GH 2763, first/last shifting dtypes
    idx = list(range(10))
    idx.append(9)
    ser = Series(data=range(11), index=idx, name="IntCol")
    assert ser.dtype == "int64"
    f = ser.groupby(level=0).first()
    assert f.dtype == "int64"


def test_first_last_nth_nan_dtype() -> None:
    # GH 33591
    df = DataFrame({"data": ["A"], "nans": Series([None], dtype=object)})
    grouped = df.groupby("data")

    expected = df.set_index("data").nans
    tm.assert_series_equal(grouped.nans.first(), expected)
    tm.assert_series_equal(grouped.nans.last(), expected)

    expected = df.nans
    tm.assert_series_equal(grouped.nans.nth(-1), expected)
    tm.assert_series_equal(grouped.nans.nth(0), expected)


def test_first_strings_timestamps() -> None:
    # GH 11244
    test = DataFrame(
        {
            Timestamp("2012-01-01 00:00:00"): ["a", "b"],
            Timestamp("2012-01-02 00:00:00"): ["c", "d"],
            "name": ["e", "e"],
            "aaaa": ["f", "g"],
        }
    )
    result = test.groupby("name").first()
    expected = DataFrame(
        [["a", "c", "f"]],
        columns=Index([Timestamp("2012-01-01"), Timestamp("2012-01-02"), "aaaa"]),
        index=Index(["e"], name="name"),
    )
    tm.assert_frame_equal(result, expected)


def test_nth() -> None:
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])
    gb = df.groupby("A")

    tm.assert_frame_equal(gb.nth(0), df.iloc[[0, 2]])
    tm.assert_frame_equal(gb.nth(1), df.iloc[[1]])
    tm.assert_frame_equal(gb.nth(2), df.loc[[]])
    tm.assert_frame_equal(gb.nth(-1), df.iloc[[1, 2]])
    tm.assert_frame_equal(gb.nth(-2), df.iloc[[0]])
    tm.assert_frame_equal(gb.nth(-3), df.loc[[]])
    tm.assert_series_equal(gb.B.nth(0), df.B.iloc[[0, 2]])
    tm.assert_series_equal(gb.B.nth(1), df.B.iloc[[1]])
    tm.assert_frame_equal(gb[["B"]].nth(0), df[["B"]].iloc[[0, 2]])

    tm.assert_frame_equal(gb.nth(0, dropna="any"), df.iloc[[1, 2]])
    tm.assert_frame_equal(gb.nth(-1, dropna="any"), df.iloc[[1, 2]])

    tm.assert_frame_equal(gb.nth(7, dropna="any"), df.iloc[:0])
    tm.assert_frame_equal(gb.nth(2, dropna="any"), df.iloc[:0])


def test_nth2() -> None:
    # out of bounds, regression from 0.13.1
    # GH 6621
    df = DataFrame(
        {
            "color": {0: "green", 1: "green", 2: "red", 3: "red", 4: "red"},
            "food": {0: "ham", 1: "eggs", 2: "eggs", 3: "ham", 4: "pork"},
            "two": {
                0: 1.5456590000000001,
                1: -0.070345000000000005,
                2: -2.4004539999999999,
                3: 0.46206000000000003,
                4: 0.52350799999999997,
            },
            "one": {
                0: 0.56573799999999996,
                1: -0.9742360000000001,
                2: 1.033801,
                3: -0.78543499999999999,
                4: 0.70422799999999997,
            },
        }
    ).set_index(["color", "food"])

    result = df.groupby(level=0, as_index=False).nth(2)
    expected = df.iloc[[-1]]
    tm.assert_frame_equal(result, expected)

    result = df.groupby(level=0, as_index=False).nth(3)
    expected = df.loc[[]]
    tm.assert_frame_equal(result, expected)


def test_nth3() -> None:
    # GH 7559
    # from the vbench
    df = DataFrame(np.random.default_rng(2).integers(1, 10, (100, 2)), dtype="int64")
    ser = df[1]
    gb = df[0]
    expected = ser.groupby(gb).first()
    expected2 = ser.groupby(gb).apply(lambda x: x.iloc[0])
    tm.assert_series_equal(expected2, expected, check_names=False)
    assert expected.name == 1
    assert expected2.name == 1

    # validate first
    v = ser[gb == 1].iloc[0]
    assert expected.iloc[0] == v
    assert expected2.iloc[0] == v

    with pytest.raises(ValueError, match="For a DataFrame"):
        ser.groupby(gb, sort=False).nth(0, dropna=True)


def test_nth4() -> None:
    # doc example
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])
    gb = df.groupby("A")
    result = gb.B.nth(0, dropna="all")
    expected = df.B.iloc[[1, 2]]
    tm.assert_series_equal(result, expected)


def test_nth5() -> None:
    # test multiple nth values
    df = DataFrame([[1, np.nan], [1, 3], [1, 4], [5, 6], [5, 7]], columns=["A", "B"])
    gb = df.groupby("A")

    tm.assert_frame_equal(gb.nth(0), df.iloc[[0, 3]])
    tm.assert_frame_equal(gb.nth([0]), df.iloc[[0, 3]])
    tm.assert_frame_equal(gb.nth([0, 1]), df.iloc[[0, 1, 3, 4]])
    tm.assert_frame_equal(gb.nth([0, -1]), df.iloc[[0, 2, 3, 4]])
    tm.assert_frame_equal(gb.nth([0, 1, 2]), df.iloc[[0, 1, 2, 3, 4]])
    tm.assert_frame_equal(gb.nth([0, 1, -1]), df.iloc[[0, 1, 2, 3, 4]])
    tm.assert_frame_equal(gb.nth([2]), df.iloc[[2]])
    tm.assert_frame_equal(gb.nth([3, 4]), df.loc[[]])


def test_nth_bdays(unit: str) -> None:
    business_dates = pd.date_range(
        start="4/1/2014", end="6/30/2014", freq="B", unit=unit
    )
    df = DataFrame(1, index=business_dates, columns=["a", "b"])
    # get the first, fourth and last two business days for each month
    key = [df.index.year, df.index.month]
    result = df.groupby(key, as_index=False).nth([0, 3, -2, -1])
    expected_dates = pd.to_datetime(
        [
            "2014/4/1",
            "2014/4/4",
            "2014/4/29",
            "2014/4/30",
            "2014/5/1",
            "2014/5/6",
            "2014/5/29",
            "2014/5/30",
            "2014/6/2",
            "2014/6/5",
            "2014/6/27",
            "2014/6/30",
        ]
    ).as_unit(unit)
    expected = DataFrame(1, columns=["a", "b"], index=expected_dates)
    tm.assert_frame_equal(result, expected)


def test_nth_multi_grouper(three_group: DataFrame) -> None:
    # PR 9090, related to issue 8979
    # test nth on multiple groupers
    grouped = three_group.groupby(["A", "B"])
    result = grouped.nth(0)
    expected = three_group.iloc[[0, 3, 4, 7]]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected_first, expected_last",
    [
        (
            {
                "id": ["A"],
                "time": Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                "foo": [1],
            },
            {
                "id": ["A"],
                "time": Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                "foo": [1],
            },
            {
                "id": ["A"],
                "time": Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                "foo": [1],
            },
        ),
        (
            {
                "id": ["A", "B", "A"],
                "time": [
                    Timestamp("2012-01-01 13:00:00", tz="America/New_York"),
                    Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                    Timestamp("2012-03-01 12:00:00", tz="Europe/London"),
                ],
                "foo": [1, 2, 3],
            },
            {
                "id": ["A", "B"],
                "time": [
                    Timestamp("2012-01-01 13:00:00", tz="America/New_York"),
                    Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                ],
                "foo": [1, 2],
            },
            {
                "id": ["A", "B"],
                "time": [
                    Timestamp("2012-03-01 12:00:00", tz="Europe/London"),
                    Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                ],
                "foo": [3, 2],
            },
        ),
    ],
)
def test_first_last_tz(
    data: Dict[str, Any], expected_first: Dict[str, Any], expected_last: Dict[str, Any]
) -> None:
    # GH15884
    # Test that the timezone is retained when calling first
    # or last on groupby with as_index=False

    df = DataFrame(data)

    result = df.groupby("id", as_index=False).first()
    expected = DataFrame(expected_first)
    cols = ["id", "time", "foo"]
    tm.assert_frame_equal(result[cols], expected[cols])

    result = df.groupby("id", as_index=False)["time"].first()
    tm.assert_frame_equal(result, expected[["id",
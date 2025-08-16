import numpy as np
import pytest
from typing import Any, List, Optional, Tuple, Union

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
)
import pandas._testing as tm


@pytest.fixture(
    params=[["linear", "single"], ["nearest", "table"]], ids=lambda x: "-".join(x)
)
def interp_method(request: pytest.FixtureRequest) -> List[str]:
    """(interpolation, method) arguments for quantile"""
    return request.param


class TestDataFrameQuantile:
    @pytest.mark.parametrize(
        "df,expected",
        [
            [
                DataFrame(
                    {
                        0: Series(pd.arrays.SparseArray([1, 2])),
                        1: Series(pd.arrays.SparseArray([3, 4])),
                    }
                ),
                Series([1.5, 3.5], name=0.5),
            ],
            [
                DataFrame(Series([0.0, None, 1.0, 2.0], dtype="Sparse[float]")),
                Series([1.0], name=0.5),
            ],
        ],
    )
    def test_quantile_sparse(self, df: DataFrame, expected: Series) -> None:
        # GH#17198
        # GH#24600
        result = df.quantile()
        expected = expected.astype("Sparse[float]")
        tm.assert_series_equal(result, expected)

    def test_quantile(self, datetime_frame: DataFrame, interp_method: List[str], request: pytest.FixtureRequest) -> None:
        interpolation, method = interp_method
        df = datetime_frame
        result = df.quantile(
            0.1, axis=0, numeric_only=True, interpolation=interpolation, method=method
        )
        expected = Series(
            [np.percentile(df[col], 10) for col in df.columns],
            index=df.columns,
            name=0.1,
        )
        if interpolation == "linear":
            # np.percentile values only comparable to linear interpolation
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_index_equal(result.index, expected.index)
            assert result.name == expected.name

        result = df.quantile(
            0.9, axis=1, numeric_only=True, interpolation=interpolation, method=method
        )
        expected = Series(
            [np.percentile(df.loc[date], 90) for date in df.index],
            index=df.index,
            name=0.9,
        )
        if interpolation == "linear":
            # np.percentile values only comparable to linear interpolation
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_index_equal(result.index, expected.index)
            assert result.name == expected.name

    def test_empty(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        q = DataFrame({"x": [], "y": []}).quantile(
            0.1, axis=0, numeric_only=True, interpolation=interpolation, method=method
        )
        assert np.isnan(q["x"]) and np.isnan(q["y"])

    def test_non_numeric_exclusion(self, interp_method: List[str], request: pytest.FixtureRequest) -> None:
        interpolation, method = interp_method
        df = DataFrame({"col1": ["A", "A", "B", "B"], "col2": [1, 2, 3, 4]})
        rs = df.quantile(
            0.5, numeric_only=True, interpolation=interpolation, method=method
        )
        xp = df.median(numeric_only=True).rename(0.5)
        if interpolation == "nearest":
            xp = (xp + 0.5).astype(np.int64)
        tm.assert_series_equal(rs, xp)

    def test_axis(self, interp_method: List[str]) -> None:
        # axis
        interpolation, method = interp_method
        df = DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}, index=[1, 2, 3])
        result = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)
        expected = Series([1.5, 2.5, 3.5], index=[1, 2, 3], name=0.5)
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)

        result = df.quantile(
            [0.5, 0.75], axis=1, interpolation=interpolation, method=method
        )
        expected = DataFrame(
            {1: [1.5, 1.75], 2: [2.5, 2.75], 3: [3.5, 3.75]}, index=[0.5, 0.75]
        )
        if interpolation == "nearest":
            expected.iloc[0, :] -= 0.5
            expected.iloc[1, :] += 0.25
            expected = expected.astype(np.int64)
        tm.assert_frame_equal(result, expected, check_index_type=True)

    def test_axis_numeric_only_true(self, interp_method: List[str]) -> None:
        # We may want to break API in the future to change this
        # so that we exclude non-numeric along the same axis
        # See GH #7312
        interpolation, method = interp_method
        df = DataFrame([[1, 2, 3], ["a", "b", 4]])
        result = df.quantile(
            0.5, axis=1, numeric_only=True, interpolation=interpolation, method=method
        )
        expected = Series([3.0, 4.0], index=range(2), name=0.5)
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)

    def test_quantile_date_range(self, interp_method: List[str]) -> None:
        # GH 2460
        interpolation, method = interp_method
        dti = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")
        ser = Series(dti)
        df = DataFrame(ser)

        result = df.quantile(
            numeric_only=False, interpolation=interpolation, method=method
        )
        expected = Series(
            ["2016-01-02 00:00:00"], name=0.5, dtype="datetime64[ns, US/Pacific]"
        )

        tm.assert_series_equal(result, expected)

    def test_quantile_axis_mixed(self, interp_method: List[str]) -> None:
        # mixed on axis=1
        interpolation, method = interp_method
        df = DataFrame(
            {
                "A": [1, 2, 3],
                "B": [2.0, 3.0, 4.0],
                "C": pd.date_range("20130101", periods=3),
                "D": ["foo", "bar", "baz"],
            }
        )
        result = df.quantile(
            0.5, axis=1, numeric_only=True, interpolation=interpolation, method=method
        )
        expected = Series([1.5, 2.5, 3.5], name=0.5)
        if interpolation == "nearest":
            expected -= 0.5
        tm.assert_series_equal(result, expected)

        # must raise
        msg = "'<' not supported between instances of 'Timestamp' and 'float'"
        with pytest.raises(TypeError, match=msg):
            df.quantile(0.5, axis=1, numeric_only=False)

    def test_quantile_axis_parameter(self, interp_method: List[str]) -> None:
        # GH 9543/9544
        interpolation, method = interp_method
        df = DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}, index=[1, 2, 3])

        result = df.quantile(0.5, axis=0, interpolation=interpolation, method=method)

        expected = Series([2.0, 3.0], index=["A", "B"], name=0.5)
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)

        expected = df.quantile(
            0.5, axis="index", interpolation=interpolation, method=method
        )
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)

        result = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)

        expected = Series([1.5, 2.5, 3.5], index=[1, 2, 3], name=0.5)
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)

        result = df.quantile(
            0.5, axis="columns", interpolation=interpolation, method=method
        )
        tm.assert_series_equal(result, expected)

        msg = "No axis named -1 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.quantile(0.1, axis=-1, interpolation=interpolation, method=method)
        msg = "No axis named column for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.quantile(0.1, axis="column")

    def test_quantile_interpolation(self) -> None:
        # see gh-10174

        # interpolation method other than default linear
        df = DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}, index=[1, 2, 3])
        result = df.quantile(0.5, axis=1, interpolation="nearest")
        expected = Series([1, 2, 3], index=[1, 2, 3], name=0.5)
        tm.assert_series_equal(result, expected)

        # cross-check interpolation=nearest results in original dtype
        exp = np.percentile(
            np.array([[1, 2, 3], [2, 3, 4]]),
            0.5,
            axis=0,
            method="nearest",
        )
        expected = Series(exp, index=[1, 2, 3], name=0.5, dtype="int64")
        tm.assert_series_equal(result, expected)

        # float
        df = DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]}, index=[1, 2, 3])
        result = df.quantile(0.5, axis=1, interpolation="nearest")
        expected = Series([1.0, 2.0, 3.0], index=[1, 2, 3], name=0.5)
        tm.assert_series_equal(result, expected)
        exp = np.percentile(
            np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
            0.5,
            axis=0,
            method="nearest",
        )
        expected = Series(exp, index=[1, 2, 3], name=0.5, dtype="float64")
        tm.assert_series_equal(result, expected)

        # axis
        result = df.quantile([0.5, 0.75], axis=1, interpolation="lower")
        expected = DataFrame(
            {1: [1.0, 1.0], 2: [2.0, 2.0], 3: [3.0, 3.0]}, index=[0.5, 0.75]
        )
        tm.assert_frame_equal(result, expected)

        # test degenerate case
        df = DataFrame({"x": [], "y": []})
        q = df.quantile(0.1, axis=0, interpolation="higher")
        assert np.isnan(q["x"]) and np.isnan(q["y"])

        # multi
        df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=["a", "b", "c"])
        result = df.quantile([0.25, 0.5], interpolation="midpoint")

        # https://github.com/numpy/numpy/issues/7163
        expected = DataFrame(
            [[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]],
            index=[0.25, 0.5],
            columns=["a", "b", "c"],
        )
        tm.assert_frame_equal(result, expected)

    def test_quantile_interpolation_datetime(self, datetime_frame: DataFrame) -> None:
        # see gh-10174

        # interpolation = linear (default case)
        df = datetime_frame
        q = df.quantile(0.1, axis=0, numeric_only=True, interpolation="linear")
        assert q["A"] == np.percentile(df["A"], 10)

    def test_quantile_interpolation_int(self, int_frame: DataFrame) -> None:
        # see gh-10174

        df = int_frame
        # interpolation = linear (default case)
        q = df.quantile(0.1)
        assert q["A"] == np.percentile(df["A"], 10)

        # test with and without interpolation keyword
        q1 = df.quantile(0.1, axis=0, interpolation="linear")
        assert q1["A"] == np.percentile(df["A"], 10)
        tm.assert_series_equal(q, q1)

    def test_quantile_multi(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=["a", "b", "c"])
        result = df.quantile([0.25, 0.5], interpolation=interpolation, method=method)
        expected = DataFrame(
            [[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]],
            index=[0.25, 0.5],
            columns=["a", "b", "c"],
        )
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        tm.assert_frame_equal(result, expected)

    def test_quantile_multi_axis_1(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=["a", "b", "c"])
        result = df.quantile(
            [0.25, 0.5], axis=1, interpolation=interpolation, method=method
        )
        expected = DataFrame(
            [[1.0, 2.0, 3.0]] * 2, index=[0.25, 0.5], columns=[0, 1, 2]
        )
        if interpolation == "nearest":
            expected = expected.astype(np.int64)
        tm.assert_frame_equal(result, expected)

    def test_quantile_multi_empty(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        result = DataFrame({"x": [], "y": []}).quantile(
            [0.1, 0.9], axis=0, interpolation=interpolation, method=method
        )
        expected = DataFrame(
            {"x": [np.nan, np.nan], "y": [np.nan, np.nan]}, index=[0.1, 0.9]
        )
        tm.assert_frame_equal(result, expected)

    def test_quantile_datetime(self, unit: str) -> None:
        dti = pd.to_datetime(["2010", "2011"]).as_unit(unit)
        df = DataFrame({"a": dti, "b": [0, 5]})

        # exclude datetime
        result = df.quantile(0.5, numeric_only=True)
        expected = Series([2.5], index=["b"], name=0.5)
        tm.assert_series_equal(result, expected)

        # datetime
        result = df.quantile(0.5, numeric_only=False)
        expected = Series(
            [Timestamp("2010-07-02 12:00:00"), 2.5], index=["a", "b"], name=0.5
        )
        tm.assert_series_equal(result, expected)

        # datetime w/ multi
        result = df.quantile([0.5], numeric_only=False)
        expected = DataFrame(
            {"a": Timestamp("2010-07-02 12:00:00").as_unit(unit), "b": 2.5},
            index=[0.5],
        )
        tm.assert_frame_equal(result, expected)

        # axis = 1
        df["c"] = pd.to_datetime(["2011", "2012"]).as_unit(unit)
        result = df[["a", "c"]].quantile(0.5, axis=1, numeric_only=False)
        expected = Series(
            [Timestamp("2010-07-02 12:00:00"), Timestamp("2011-07-02 12:00:00")],
            index=[0, 1],
            name=0.5,
            dtype=f"M8[{unit}]",
        )
        tm.assert_series_equal(result, expected)

        result = df[["a", "c"]].quantile([0.5], axis=1, numeric_only=False)
        expected = DataFrame(
            [[Timestamp("
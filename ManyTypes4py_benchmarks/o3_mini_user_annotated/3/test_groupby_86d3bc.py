from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    NamedAgg,
    Series,
    Timestamp,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby


@pytest.fixture
def times_frame() -> DataFrame:
    """Frame for testing times argument in EWM groupby."""
    return DataFrame(
        {
            "A": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
            "B": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
            "C": to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-10",
                    "2020-01-22",
                    "2020-01-03",
                    "2020-01-23",
                    "2020-01-23",
                    "2020-01-04",
                ]
            ),
        }
    )


@pytest.fixture
def roll_frame() -> DataFrame:
    return DataFrame({"A": [1] * 20 + [2] * 12 + [3] * 8, "B": np.arange(40)})


class TestRolling:
    def test_groupby_unsupported_argument(self, roll_frame: DataFrame) -> None:
        msg: str = r"groupby\(\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            roll_frame.groupby("A", foo=1)

    def test_getitem(self, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby("A")
        g_mutated = get_groupby(roll_frame, by="A")

        expected: Series = g_mutated.B.apply(lambda x: x.rolling(2).mean())

        result: Series = g.rolling(2).mean().B
        tm.assert_series_equal(result, expected)

        result = g.rolling(2).B.mean()
        tm.assert_series_equal(result, expected)

        result = g.B.rolling(2).mean()
        tm.assert_series_equal(result, expected)

        result = roll_frame.B.groupby(roll_frame.A).rolling(2).mean()
        tm.assert_series_equal(result, expected)

    def test_getitem_multiple(self, roll_frame: DataFrame) -> None:
        # GH 13174
        g = roll_frame.groupby("A")
        r = g.rolling(2, min_periods=0)
        g_mutated = get_groupby(roll_frame, by="A")
        expected: Series = g_mutated.B.apply(lambda x: x.rolling(2, min_periods=0).count())

        result: Series = r.B.count()
        tm.assert_series_equal(result, expected)

        result = r.B.count()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "f",
        [
            "sum",
            "mean",
            "min",
            "max",
            "first",
            "last",
            "count",
            "kurt",
            "skew",
        ],
    )
    def test_rolling(self, f: str, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby("A", group_keys=False)
        r = g.rolling(window=4)

        result: DataFrame = getattr(r, f)()
        expected: DataFrame = g.apply(lambda x: getattr(x.rolling(4), f)())
        # GH 39732
        expected_index: MultiIndex = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["std", "var"])
    def test_rolling_ddof(self, f: str, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby("A", group_keys=False)
        r = g.rolling(window=4)

        result: DataFrame = getattr(r, f)(ddof=1)
        expected: DataFrame = g.apply(lambda x: getattr(x.rolling(4), f)(ddof=1))
        # GH 39732
        expected_index: MultiIndex = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
    )
    def test_rolling_quantile(self, interpolation: str, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby("A", group_keys=False)
        r = g.rolling(window=4)

        result: DataFrame = r.quantile(0.4, interpolation=interpolation)
        expected: DataFrame = g.apply(
            lambda x: x.rolling(4).quantile(0.4, interpolation=interpolation)
        )
        # GH 39732
        expected_index: MultiIndex = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f, expected_val", [["corr", 1], ["cov", 0.5]])
    def test_rolling_corr_cov_other_same_size_as_groups(self, f: str, expected_val: float) -> None:
        # GH 42915
        df = DataFrame(
            {"value": list(range(10)), "idx1": [1] * 5 + [2] * 5, "idx2": [1, 2, 3, 4, 5] * 2}
        ).set_index(["idx1", "idx2"])
        other: DataFrame = DataFrame({"value": list(range(5)), "idx2": [1, 2, 3, 4, 5]}).set_index("idx2")
        result: DataFrame = getattr(df.groupby(level=0).rolling(2), f)(other)
        expected_data: List[Any] = ([np.nan] + [expected_val] * 4) * 2
        expected: DataFrame = DataFrame(
            expected_data,
            columns=["value"],
            index=MultiIndex.from_arrays(
                [
                    [1] * 5 + [2] * 5,
                    [1] * 5 + [2] * 5,
                    list(range(1, 6)) * 2,
                ],
                names=["idx1", "idx1", "idx2"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["corr", "cov"])
    def test_rolling_corr_cov_other_diff_size_as_groups(self, f: str, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby("A")
        r = g.rolling(window=4)

        result: DataFrame = getattr(r, f)(roll_frame)

        def func(x: DataFrame) -> DataFrame:
            return getattr(x.rolling(4), f)(roll_frame)

        expected: DataFrame = g.apply(func)
        # GH 39591: The grouped column should be all np.nan
        expected["A"] = np.nan
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["corr", "cov"])
    def test_rolling_corr_cov_pairwise(self, f: str, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby("A")
        r = g.rolling(window=4)

        result: Series = getattr(r.B, f)(pairwise=True)

        def func(x: DataFrame) -> Series:
            return getattr(x.B.rolling(4), f)(pairwise=True)

        expected: Series = g.apply(func)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "func, expected_values",
        [("cov", [[1.0, 1.0], [1.0, 4.0]]), ("corr", [[1.0, 0.5], [0.5, 1.0]])],
    )
    def test_rolling_corr_cov_unordered(self, func: str, expected_values: List[List[float]]) -> None:
        # GH 43386
        df = DataFrame(
            {
                "a": ["g1", "g2", "g1", "g1"],
                "b": [0, 0, 1, 2],
                "c": [2, 0, 6, 4],
            }
        )
        rol = df.groupby("a").rolling(3)
        result: DataFrame = getattr(rol, func)()
        expected: DataFrame = DataFrame(
            {
                "b": 4 * [np.nan] + expected_values[0] + 2 * [np.nan],
                "c": 4 * [np.nan] + expected_values[1] + 2 * [np.nan],
            },
            index=MultiIndex.from_tuples(
                [
                    ("g1", 0, "b"),
                    ("g1", 0, "c"),
                    ("g1", 2, "b"),
                    ("g1", 2, "c"),
                    ("g1", 3, "b"),
                    ("g1", 3, "c"),
                    ("g2", 1, "b"),
                    ("g2", 1, "c"),
                ],
                names=["a", None, None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_rolling_apply(self, raw: bool, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby("A", group_keys=False)
        r = g.rolling(window=4)

        # reduction
        result: DataFrame = r.apply(lambda x: x.sum(), raw=raw)
        expected: DataFrame = g.apply(lambda x: x.rolling(4).apply(lambda y: y.sum(), raw=raw))
        # GH 39732
        expected_index: MultiIndex = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    def test_rolling_apply_mutability(self) -> None:
        # GH 14013
        df = DataFrame({"A": ["foo"] * 3 + ["bar"] * 3, "B": [1] * 6})
        g = df.groupby("A")

        mi: MultiIndex = MultiIndex.from_tuples(
            [("bar", 3), ("bar", 4), ("bar", 5), ("foo", 0), ("foo", 1), ("foo", 2)]
        )
        mi.names = ["A", None]
        # Grouped column should not be a part of the output
        expected: DataFrame = DataFrame([np.nan, 2.0, 2.0] * 2, columns=["B"], index=mi)

        result: DataFrame = g.rolling(window=2).sum()
        tm.assert_frame_equal(result, expected)

        # Call an arbitrary function on the groupby
        g.sum()

        # Make sure nothing has been mutated
        result = g.rolling(window=2).sum()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("expected_value,raw_value", [[1.0, True], [0.0, False]])
    def test_groupby_rolling(self, expected_value: float, raw_value: bool) -> None:
        # GH 31754

        def isnumpyarray(x: Any) -> int:
            return int(isinstance(x, np.ndarray))

        df = DataFrame({"id": [1, 1, 1], "value": [1, 2, 3]})
        result: Series = df.groupby("id").value.rolling(1).apply(isnumpyarray, raw=raw_value)
        expected: Series = Series(
            [expected_value] * 3,
            index=MultiIndex.from_tuples(((1, 0), (1, 1), (1, 2)), names=["id", None]),
            name="value",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_center_center(self) -> None:
        # GH 35552
        series: Series = Series(range(1, 6))
        result: Series = series.groupby(series).rolling(center=True, window=3).mean()
        expected: Series = Series(
            [np.nan] * 5,
            index=MultiIndex.from_tuples(((1, 0), (2, 1), (3, 2), (4, 3), (5, 4))),
        )
        tm.assert_series_equal(result, expected)

        series = Series(range(1, 5))
        result = series.groupby(series).rolling(center=True, window=3).mean()
        expected = Series(
            [np.nan] * 4,
            index=MultiIndex.from_tuples(((1, 0), (2, 1), (3, 2), (4, 3))),
        )
        tm.assert_series_equal(result, expected)

        df = DataFrame({"a": ["a"] * 5 + ["b"] * 6, "b": range(11)})
        result = df.groupby("a").rolling(center=True, window=3).mean()
        expected = DataFrame(
            [np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, 8, 9, np.nan],
            index=MultiIndex.from_tuples(
                (
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                    ("a", 4),
                    ("b", 5),
                    ("b", 6),
                    ("b", 7),
                    ("b", 8),
                    ("b", 9),
                    ("b", 10),
                ),
                names=["a", None],
            ),
            columns=["b"],
        )
        tm.assert_frame_equal(result, expected)

        df = DataFrame({"a": ["a"] * 5 + ["b"] * 5, "b": range(10)})
        result = df.groupby("a").rolling(center=True, window=3).mean()
        expected = DataFrame(
            [np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, 8, np.nan],
            index=MultiIndex.from_tuples(
                (
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                    ("a", 4),
                    ("b", 5),
                    ("b", 6),
                    ("b", 7),
                    ("b", 8),
                    ("b", 9),
                ),
                names=["a", None],
            ),
            columns=["b"],
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_center_on(self) -> None:
        # GH 37141
        df = DataFrame(
            data={
                "Date": date_range("2020-01-01", "2020-01-10"),
                "gb": ["group_1"] * 6 + ["group_2"] * 4,
                "value": list(range(10)),
            }
        )
        result: Series = (
            df.groupby("gb")
            .rolling(6, on="Date", center=True, min_periods=1)
            .value.mean()
        )
        mi: MultiIndex = MultiIndex.from_arrays([df["gb"], df["Date"]], names=["gb", "Date"])
        expected: Series = Series(
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 7.0, 7.5, 7.5, 7.5],
            name="value",
            index=mi,
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("min_periods", [5, 4, 3])
    def test_groupby_rolling_center_min_periods(self, min_periods: int) -> None:
        # GH 36040
        df = DataFrame({"group": ["A"] * 10 + ["B"] * 10, "data": list(range(20))})

        window_size: int = 5
        result: DataFrame = (
            df.groupby("group")
            .rolling(window_size, center=True, min_periods=min_periods)
            .mean()
        )
        result = result.reset_index()[["group", "data"]]

        grp_A_mean: List[float] = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0]
        grp_B_mean: List[float] = [x + 10.0 for x in grp_A_mean]

        num_nans: int = max(0, min_periods - 3)  # For window_size of 5
        nans: List[Union[float, None]] = [np.nan] * num_nans
        grp_A_expected: List[Union[float, None]] = nans + grp_A_mean[num_nans: 10 - num_nans] + nans
        grp_B_expected: List[Union[float, None]] = nans + grp_B_mean[num_nans: 10 - num_nans] + nans

        expected: DataFrame = DataFrame(
            {"group": ["A"] * 10 + ["B"] * 10, "data": grp_A_expected + grp_B_expected}
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_subselect_rolling(self) -> None:
        # GH 35486
        df = DataFrame(
            {"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0], "c": [10, 20, 30, 20]}
        )
        result: DataFrame = df.groupby("a")[["b"]].rolling(2).max()
        expected: DataFrame = DataFrame(
            [np.nan, np.nan, 2.0, np.nan],
            columns=["b"],
            index=MultiIndex.from_tuples(
                ((1, 0), (2, 1), (2, 3), (3, 2)), names=["a", None]
            ),
        )
        tm.assert_frame_equal(result, expected)

        result = df.groupby("a")["b"].rolling(2).max()
        expected_series: Series = Series(
            [np.nan, np.nan, 2.0, np.nan],
            index=MultiIndex.from_tuples(
                ((1, 0), (2, 1), (2, 3), (3, 2)), names=["a", None]
            ),
            name="b",
        )
        tm.assert_series_equal(result, expected_series)

    def test_groupby_rolling_custom_indexer(self) -> None:
        # GH 35557
        class SimpleIndexer(BaseIndexer):
            window_size: int

            def __init__(self, window_size: int) -> None:
                self.window_size = window_size

            def get_window_bounds(
                self,
                num_values: int = 0,
                min_periods: Union[int, None] = None,
                center: Union[bool, None] = None,
                closed: Any = None,
                step: Any = None,
            ) -> Tuple[np.ndarray, np.ndarray]:
                min_periods = self.window_size if min_periods is None else 0
                end: np.ndarray = np.arange(num_values, dtype=np.int64) + 1
                start: np.ndarray = end - self.window_size
                start[start < 0] = min_periods
                return start, end

        df = DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0] * 3}, index=[0] * 5 + [1] * 5 + [2] * 5
        )
        result: DataFrame = (
            df.groupby(df.index)
            .rolling(SimpleIndexer(window_size=3), min_periods=1)
            .sum()
        )
        expected: DataFrame = df.groupby(df.index).rolling(window=3, min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_subset_with_closed(self) -> None:
        # GH 35549
        df = DataFrame(
            {
                "column1": list(range(8)),
                "column2": list(range(8)),
                "group": ["A"] * 4 + ["B"] * 4,
                "date": [
                    Timestamp(date)
                    for date in ["2019-01-01", "2019-01-01", "2019-01-02", "2019-01-02"]
                ]
                * 2,
            }
        )
        result: Series = (
            df.groupby("group").rolling("1D", on="date", closed="left")["column1"].sum()
        )
        expected: Series = Series(
            [np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 9.0, 9.0],
            index=MultiIndex.from_frame(
                df[["group", "date"]],
                names=["group", "date"],
            ),
            name="column1",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_agg_namedagg(self) -> None:
        # GH#28333
        df = DataFrame(
            {
                "kind": ["cat", "dog", "cat", "dog", "cat", "dog"],
                "height": [9.1, 6.0, 9.5, 34.0, 12.0, 8.0],
                "weight": [7.9, 7.5, 9.9, 198.0, 10.0, 42.0],
            }
        )
        result: DataFrame = (
            df.groupby("kind")
            .rolling(2)
            .agg(
                total_weight=NamedAgg(column="weight", aggfunc=sum),
                min_height=NamedAgg(column="height", aggfunc=min),
            )
        )
        expected: DataFrame = DataFrame(
            {
                "total_weight": [np.nan, 17.8, 19.9, np.nan, 205.5, 240.0],
                "min_height": [np.nan, 9.1, 9.5, np.nan, 6.0, 8.0],
            },
            index=MultiIndex(
                [["cat", "dog"], [0, 1, 2, 3, 4, 5]],
                [[0, 0, 0, 1, 1, 1], [0, 2, 4, 1, 3, 5]],
                names=["kind", None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_subset_rolling_subset_with_closed(self) -> None:
        # GH 35549
        df = DataFrame(
            {
                "column1": list(range(8)),
                "column2": list(range(8)),
                "group": ["A"] * 4 + ["B"] * 4,
                "date": [
                    Timestamp(date)
                    for date in ["2019-01-01", "2019-01-01", "2019-01-02", "2019-01-02"]
                ]
                * 2,
            }
        )

        result: Series = (
            df.groupby("group")[["column1", "date"]]
            .rolling("1D", on="date", closed="left")["column1"]
            .sum()
        )
        expected: Series = Series(
            [np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 9.0, 9.0],
            index=MultiIndex.from_frame(
                df[["group", "date"]],
                names=["group", "date"],
            ),
            name="column1",
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("func", ["max", "min"])
    def test_groupby_rolling_index_changed(self, func: str) -> None:
        # GH: #36018 nlevels of MultiIndex changed
        ds: Series = Series(
            [1, 2, 2],
            index=MultiIndex.from_tuples(
                [("a", "x"), ("a", "y"), ("c", "z")], names=["1", "2"]
            ),
            name="a",
        )

        result: Series = getattr(ds.groupby(ds).rolling(2), func)()
        expected: Series = Series(
            [np.nan, np.nan, 2.0],
            index=MultiIndex.from_tuples(
                [(1, "a", "x"), (2, "a", "y"), (2, "c", "z")], names=["a", "1", "2"]
            ),
            name="a",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_empty_frame(self) -> None:
        # GH 36197
        expected: DataFrame = DataFrame({"s1": []})
        result: DataFrame = expected.groupby("s1").rolling(window=1).sum()
        # GH 32262
        expected = expected.drop(columns="s1")
        expected.index = MultiIndex.from_product(
            [Index([], dtype="float64"), Index([], dtype="int64")], names=["s1", None]
        )
        tm.assert_frame_equal(result, expected)

        expected = DataFrame({"s1": [], "s2": []})
        result = expected.groupby(["s1", "s2"]).rolling(window=1).sum()
        expected = expected.drop(columns=["s1", "s2"])
        expected.index = MultiIndex.from_product(
            [
                Index([], dtype="float64"),
                Index([], dtype="float64"),
                Index([], dtype="int64"),
            ],
            names=["s1", "s2", None],
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_string_index(self) -> None:
        # GH: 36727
        df = DataFrame(
            [
                ["A", "group_1", Timestamp(2019, 1, 1, 9)],
                ["B", "group_1", Timestamp(2019, 1, 2, 9)],
                ["Z", "group_2", Timestamp(2019, 1, 3, 9)],
                ["H", "group_1", Timestamp(2019, 1, 6, 9)],
                ["E", "group_2", Timestamp(2019, 1, 20, 9)],
            ],
            columns=["index", "group", "eventTime"],
        ).set_index("index")

        groups = df.groupby("group")
        df["count_to_date"] = groups.cumcount()
        rolling_groups = groups.rolling("10D", on="eventTime")
        result: DataFrame = rolling_groups.apply(lambda df: df.shape[0])
        expected: DataFrame = DataFrame(
            [
                ["A", "group_1", Timestamp(2019, 1, 1, 9), 1.0],
                ["B", "group_1", Timestamp(2019, 1, 2, 9), 2.0],
                ["H", "group_1", Timestamp(2019, 1, 6, 9), 3.0],
                ["Z", "group_2", Timestamp(2019, 1, 3, 9), 1.0],
                ["E", "group_2", Timestamp(2019, 1, 20, 9), 1.0],
            ],
            columns=["index", "group", "eventTime", "count_to_date"],
        ).set_index(["group", "index"])
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_no_sort(self) -> None:
        # GH 36889
        result: DataFrame = (
            DataFrame({"foo": [2, 1], "bar": [2, 1]})
            .groupby("foo", sort=False)
            .rolling(1)
            .min()
        )
        expected: DataFrame = DataFrame(
            np.array([[2.0, 2.0], [1.0, 1.0]]),
            columns=["foo", "bar"],
            index=MultiIndex.from_tuples([(2, 0), (1, 1)], names=["foo", None]),
        )
        expected = expected.drop(columns="foo")
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_count_closed_on(self, unit: str) -> None:
        # GH 35869
        df = DataFrame(
            {
                "column1": list(range(6)),
                "column2": list(range(6)),
                "group": 3 * ["A", "B"],
                "date": date_range(end="20190101", periods=6, unit=unit),
            }
        )
        msg: str = "'d' is deprecated and will be removed in a future version."

        with tm.assert_produces_warning(FutureWarning, match=msg):
            result: Series = (
                df.groupby("group")
                .rolling("3d", on="date", closed="left")["column1"]
                .count()
            )
        dti: DatetimeIndex = DatetimeIndex(
            [
                "2018-12-27",
                "2018-12-29",
                "2018-12-31",
                "2018-12-28",
                "2018-12-30",
                "2019-01-01",
            ],
            dtype=f"M8[{unit}]",
        )
        mi: MultiIndex = MultiIndex.from_arrays(
            [
                ["A", "A", "A", "B", "B", "B"],
                dti,
            ],
            names=["group", "date"],
        )
        expected: Series = Series(
            [np.nan, 1.0, 1.0, np.nan, 1.0, 1.0],
            name="column1",
            index=mi,
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        ("func", "kwargs"),
        [("rolling", {"window": 2, "min_periods": 1}), ("expanding", {})],
    )
    def test_groupby_rolling_sem(self, func: str, kwargs: Dict[str, Any]) -> None:
        # GH: 26476
        df = DataFrame(
            [["a", 1], ["a", 2], ["b", 1], ["b", 2], ["b", 3]], columns=["a", "b"]
        )
        result: DataFrame = getattr(df.groupby("a"), func)(**kwargs).sem()
        expected: DataFrame = DataFrame(
            {"a": [np.nan] * 5, "b": [np.nan, 0.70711, np.nan, 0.70711, 0.70711]},
            index=MultiIndex.from_tuples(
                [("a", 0), ("a", 1), ("b", 2), ("b", 3), ("b", 4)], names=["a", None]
            ),
        )
        expected = expected.drop(columns="a")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        ("rollings", "key"), [({"on": "a"}, "a"), ({"on": None}, "index")]
    )
    def test_groupby_rolling_nans_in_index(self, rollings: Dict[str, Any], key: str) -> None:
        # GH: 34617
        df = DataFrame(
            {
                "a": to_datetime(["2020-06-01 12:00", "2020-06-01 14:00", np.nan]),
                "b": [1, 2, 3],
                "c": [1, 1, 1],
            }
        )
        if key == "index":
            df = df.set_index("a")
        with pytest.raises(ValueError, match=f"{key} values must not have NaT"):
            df.groupby("c").rolling("60min", **rollings)

    @pytest.mark.parametrize("group_keys", [True, False])
    def test_groupby_rolling_group_keys(self, group_keys: bool) -> None:
        # GH 37641
        arrays: List[List[str]] = [["val1", "val1", "val2"], ["val1", "val1", "val2"]]
        index: MultiIndex = MultiIndex.from_arrays(arrays, names=("idx1", "idx2"))

        s: Series = Series([1, 2, 3], index=index)
        result: Series = s.groupby(["idx1", "idx2"], group_keys=group_keys).rolling(1).mean()
        expected: Series = Series(
            [1.0, 2.0, 3.0],
            index=MultiIndex.from_tuples(
                [
                    ("val1", "val1", "val1", "val1"),
                    ("val1", "val1", "val1", "val1"),
                    ("val2", "val2", "val2", "val2"),
                ],
                names=["idx1", "idx2", "idx1", "idx2"],
            ),
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_index_level_and_column_label(self) -> None:
        arrays = [["val1", "val1", "val2"], ["val1", "val1", "val2"]]
        index: MultiIndex = MultiIndex.from_arrays(arrays, names=("idx1", "idx2"))
        df = DataFrame({"A": [1, 1, 2], "B": list(range(3))}, index=index)
        result: DataFrame = df.groupby(["idx1", "A"]).rolling(1).mean()
        expected: DataFrame = DataFrame(
            {"B": [0.0, 1.0, 2.0]},
            index=MultiIndex.from_tuples(
                [
                    ("val1", 1, "val1", "val1"),
                    ("val1", 1, "val1", "val1"),
                    ("val2", 2, "val2", "val2"),
                ],
                names=["idx1", "A", "idx1", "idx2"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_resulting_multiindex(self) -> None:
        # a few different cases checking the created MultiIndex of the result
        # https://github.com/pandas-dev/pandas/pull/38057

        # grouping by 1 columns -> 2-level MI as result
        df = DataFrame({"a": np.arange(8.0), "b": [1, 2] * 4})
        result: DataFrame = df.groupby("b").rolling(3).mean()
        expected_index: MultiIndex = MultiIndex.from_tuples(
            [(1, 0), (1, 2), (1, 4), (1, 6), (2, 1), (2, 3), (2, 5), (2, 7)],
            names=["b", None],
        )
        tm.assert_index_equal(result.index, expected_index)

    def test_groupby_rolling_resulting_multiindex2(self) -> None:
        # grouping by 2 columns -> 3-level MI as result
        df = DataFrame({"a": np.arange(12.0), "b": [1, 2] * 6, "c": [1, 2, 3, 4] * 3})
        result: DataFrame = df.groupby(["b", "c"]).rolling(2).sum()
        expected_index: MultiIndex = MultiIndex.from_tuples(
            [
                (1, 1, 0),
                (1, 1, 4),
                (1, 1, 8),
                (1, 3, 2),
                (1, 3, 6),
                (1, 3, 10),
                (2, 2, 1),
                (2, 2, 5),
                (2, 2, 9),
                (2, 4, 3),
                (2, 4, 7),
                (2, 4, 11),
            ],
            names=["b", "c", None],
        )
        tm.assert_index_equal(result.index, expected_index)

    def test_groupby_rolling_resulting_multiindex3(self) -> None:
        # grouping with 1 level on dataframe with 2-level MI -> 3-level MI as result
        df = DataFrame({"a": np.arange(8.0), "b": [1, 2] * 4, "c": [1, 2, 3, 4] * 2})
        df = df.set_index("c", append=True)
        result: DataFrame = df.groupby("b").rolling(3).mean()
        expected_index: MultiIndex = MultiIndex.from_tuples(
            [
                (1, 0, 1),
                (1, 2, 3),
                (1, 4, 1),
                (1, 6, 3),
                (2, 1, 2),
                (2, 3, 4),
                (2, 5, 2),
                (2, 7, 4),
            ],
            names=["b", None, "c"],
        )
        tm.assert_index_equal(result.index, expected_index, exact="equiv")

    def test_groupby_rolling_object_doesnt_affect_groupby_apply(self, roll_frame: DataFrame) -> None:
        # GH 39732
        g = roll_frame.groupby("A", group_keys=False)
        expected: MultiIndex = g.apply(lambda x: x.rolling(4).sum()).index
        _ = g.rolling(window=4)
        result: MultiIndex = g.apply(lambda x: x.rolling(4).sum()).index
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        ("window", "min_periods", "closed", "expected"),
        [
            (2, 0, "left", [None, 0.0, 1.0, 1.0, None, 0.0, 1.0, 1.0]),
            (2, 2, "left", [None, None, 1.0, 1.0, None, None, 1.0, 1.0]),
            (4, 4, "left", [None, None, None, None, None, None, None, None]),
            (4, 4, "right", [None, None, None, 5.0, None, None, None, 5.0]),
        ],
    )
    def test_groupby_rolling_var(
        self, window: int, min_periods: int, closed: str, expected: List[Any]
    ) -> None:
        df = DataFrame([1, 2, 3, 4, 5, 6, 7, 8])
        result: DataFrame = (
            df.groupby([1, 2, 1, 2, 1, 2, 1, 2])
            .rolling(window=window, min_periods=min_periods, closed=closed)
            .var(0)
        )
        expected_result: DataFrame = DataFrame(
            np.array(expected, dtype="float64"),
            index=MultiIndex(
                levels=[np.array([1, 2]), list(range(8))],
                codes=[[0, 0, 0, 0, 1, 1, 1, 1], [0, 2, 4, 6, 1, 3, 5, 7]],
            ),
        )
        tm.assert_frame_equal(result, expected_result)

    @pytest.mark.parametrize(
        "columns", [MultiIndex.from_tuples([("A", ""), ("B", "C")]), ["A", "B"]]
    )
    def test_by_column_not_in_values(self, columns: Union[List[str], MultiIndex]) -> None:
        # GH 32262
        df = DataFrame([[1, 0]] * 20 + [[2, 0]] * 12 + [[3, 0]] * 8, columns=columns)
        g = df.groupby("A")
        original_obj: DataFrame = g.obj.copy(deep=True)
        r = g.rolling(4)
        result: DataFrame = r.sum()
        assert "A" not in result.columns
        tm.assert_frame_equal(g.obj, original_obj)

    def test_groupby_level(self) -> None:
        # GH 38523, 38787
        arrays: List[List[str]] = [
            ["Falcon", "Falcon", "Parrot", "Parrot"],
            ["Captive", "Wild", "Captive", "Wild"],
        ]
        index: MultiIndex = MultiIndex.from_arrays(arrays, names=("Animal", "Type"))
        df = DataFrame({"Max Speed": [390.0, 350.0, 30.0, 20.0]}, index=index)
        result: Series = df.groupby(level=0)["Max Speed"].rolling(2).sum()
        expected: Series = Series(
            [np.nan, 740.0, np.nan, 50.0],
            index=MultiIndex.from_tuples(
                [
                    ("Falcon", "Falcon", "Captive"),
                    ("Falcon", "Falcon", "Wild"),
                    ("Parrot", "Parrot", "Captive"),
                    ("Parrot", "Parrot", "Wild"),
                ],
                names=["Animal", "Animal", "Type"],
            ),
            name="Max Speed",
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "by, expected_data",
        [
            [["id"], {"num": [100.0, 150.0, 150.0, 200.0]}],
            [
                ["id", "index"],
                {
                    "date": [
                        Timestamp("2018-01-01"),
                        Timestamp("2018-01-02"),
                        Timestamp("2018-01-01"),
                        Timestamp("2018-01-02"),
                    ],
                    "num": [100.0, 200.0, 150.0, 250.0],
                },
            ],
        ],
    )
    def test_as_index_false(self, by: List[str], expected_data: Dict[str, List[Any]], unit: str) -> None:
        # GH 39433
        data: List[List[Any]] = [
            ["A", "2018-01-01", 100.0],
            ["A", "2018-01-02", 200.0],
            ["B", "2018-01-01", 150.0],
            ["B", "2018-01-02", 250.0],
        ]
        df = DataFrame(data, columns=["id", "date", "num"])
        df["date"] = df["date"].astype(f"M8[{unit}]")
        df = df.set_index(["date"])

        gp_by: List[Series] = [getattr(df, attr) for attr in by]
        result: DataFrame = (
            df.groupby(gp_by, as_index=False).rolling(window=2, min_periods=1).mean()
        )

        expected_dict: Dict[str, Any] = {"id": ["A", "A"]}
        expected_dict.update(expected_data)
        expected: DataFrame = DataFrame(
            expected_dict,
            index=df.index,
        )
        if "date" in expected_data:
            expected["date"] = expected["date"].astype(f"M8[{unit}]")
        tm.assert_frame_equal(result, expected)

    def test_nan_and_zero_endpoints(self, any_int_numpy_dtype: Any) -> None:
        # https://github.com/twosigma/pandas/issues/53
        typ = np.dtype(any_int_numpy_dtype).type
        size: int = 1000
        idx = np.repeat(typ(0), size)
        idx[-1] = 1

        val: float = 5e25
        arr: np.ndarray = np.repeat(val, size)
        arr[0] = np.nan
        arr[-1] = 0

        df = DataFrame(
            {
                "index": idx,
                "adl2": arr,
            }
        ).set_index("index")
        result: Series = df.groupby("index")["adl2"].rolling(window=10, min_periods=1).mean()
        expected: Series = Series(
            arr,
            name="adl2",
            index=MultiIndex.from_arrays(
                [
                    Index([0] * 999 + [1], dtype=typ, name="index"),
                    Index([0] * 999 + [1], dtype=typ, name="index"),
                ],
            ),
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_non_monotonic(self) -> None:
        # GH 43909

        shuffled: List[int] = [3, 0, 1, 2]
        sec: int = 1_000
        df = DataFrame(
            [{"t": Timestamp(2 * x * sec), "x": x + 1, "c": 42} for x in shuffled]
        )
        with pytest.raises(ValueError, match=r".* must be monotonic"):
            df.groupby("c").rolling(on="t", window="3s")

    def test_groupby_monotonic(self) -> None:
        # GH 15130
        data: List[List[Any]] = [
            ["David", "1/1/2015", 100],
            ["David", "1/5/2015", 500],
            ["David", "5/30/2015", 50],
            ["David", "7/25/2015", 50],
            ["Ryan", "1/4/2014", 100],
            ["Ryan", "1/19/2015", 500],
            ["Ryan", "3/31/2016", 50],
            ["Joe", "7/1/2015", 100],
            ["Joe", "9/9/2015", 500],
            ["Joe", "10/15/2015", 50],
        ]

        df = DataFrame(data=data, columns=["name", "date", "amount"])
        df["date"] = to_datetime(df["date"])
        df = df.sort_values("date")

        expected: Any = (
            df.set_index("date")
            .groupby("name")
            .apply(lambda x: x.rolling("180D")["amount"].sum())
        )
        result: Series = df.groupby("name").rolling("180D", on="date")["amount"].sum()
        tm.assert_series_equal(result, expected)

    def test_datelike_on_monotonic_within_each_group(self) -> None:
        # GH 13966 (similar to #15130, closed by #15175)
        dates = date_range(start="2016-01-01 09:30:00", periods=20, freq="s")
        df = DataFrame(
            {
                "A": [1] * 20 + [2] * 12 + [3] * 8,
                "B": np.concatenate((dates, dates)),
                "C": np.arange(40),
            }
        )
        expected: Any = (
            df.set_index("B").groupby("A").apply(lambda x: x.rolling("4s")["C"].mean())
        )
        result: Series = df.groupby("A").rolling("4s", on="B").C.mean()
        tm.assert_series_equal(result, expected)

    def test_datelike_on_not_monotonic_within_each_group(self) -> None:
        # GH 46061
        df = DataFrame(
            {
                "A": [1] * 3 + [2] * 3,
                "B": [Timestamp(year, 1, 1) for year in [2020, 2021, 2019]] * 2,
                "C": list(range(6)),
            }
        )
        with pytest.raises(ValueError, match="Each group within B must be monotonic."):
            df.groupby("A").rolling("365D", on="B")


class TestExpanding:
    @pytest.fixture
    def frame(self) -> DataFrame:
        return DataFrame({"A": [1] * 20 + [2] * 12 + [3] * 8, "B": np.arange(40)})

    @pytest.mark.parametrize(
        "f", ["sum", "mean", "min", "max", "first", "last", "count", "kurt", "skew"]
    )
    def test_expanding(self, f: str, frame: DataFrame) -> None:
        g = frame.groupby("A", group_keys=False)
        r = g.expanding()

        result: DataFrame = getattr(r, f)()
        expected: DataFrame = g.apply(lambda x: getattr(x.expanding(), f)())
        # GH 39732
        expected_index: MultiIndex = MultiIndex.from_arrays([frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["std", "var"])
    def test_expanding_ddof(self, f: str, frame: DataFrame) -> None:
        g = frame.groupby("A", group_keys=False)
        r = g.expanding()

        result: DataFrame = getattr(r, f)(ddof=0)
        expected: DataFrame = g.apply(lambda x: getattr(x.expanding(), f)(ddof=0))
        # GH 39732
        expected_index: MultiIndex = MultiIndex.from_arrays([frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
    )
    def test_expanding_quantile(self, interpolation: str, frame: DataFrame) -> None:
        g = frame.groupby("A", group_keys=False)
        r = g.expanding()

        result: DataFrame = r.quantile(0.4, interpolation=interpolation)
        expected: DataFrame = g.apply(
            lambda x: x.expanding().quantile(0.4, interpolation=interpolation)
        )
        # GH 39732
        expected_index: MultiIndex = MultiIndex.from_arrays([frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["corr", "cov"])
    def test_expanding_corr_cov(self, f: str, frame: DataFrame) -> None:
        g = frame.groupby("A")
        r = g.expanding()

        result: DataFrame = getattr(r, f)(frame)

        def func_0(x: DataFrame) -> DataFrame:
            return getattr(x.expanding(), f)(frame)

        expected: DataFrame = g.apply(func_0)
        # GH 39591: groupby.apply returns 1 instead of nan for windows
        # with all nan values
        null_idx: List[int] = list(range(20, 61)) + list(range(72, 113))
        expected.iloc[null_idx, 1] = np.nan
        # GH 39591: The grouped column should be all np.nan
        expected["A"] = np.nan
        tm.assert_frame_equal(result, expected)

        result = getattr(r.B, f)(pairwise=True)

        def func_1(x: DataFrame) -> Series:
            return getattr(x.B.expanding(), f)(pairwise=True)

        expected_series: Series = g.apply(func_1)
        tm.assert_series_equal(result, expected_series)

    def test_expanding_apply(self, raw: bool, frame: DataFrame) -> None:
        g = frame.groupby("A", group_keys=False)
        r = g.expanding()

        # reduction
        result: DataFrame = r.apply(lambda x: x.sum(), raw=raw)
        expected: DataFrame = g.apply(lambda x: x.expanding().apply(lambda y: y.sum(), raw=raw))
        # GH 39732
        expected_index: MultiIndex = MultiIndex.from_arrays([frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    def test_groupby_expanding_agg_namedagg(self) -> None:
        # GH#28333
        df = DataFrame(
            {
                "kind": ["cat", "dog", "cat", "dog", "cat", "dog"],
                "height": [9.1, 6.0, 9.5, 34.0, 12.0, 8.0],
                "weight": [7.9, 7.5, 9.9, 198.0, 10.0, 42.0],
            }
        )
        result: DataFrame = (
            df.groupby("kind")
            .expanding(1)
            .agg(
                total_weight=NamedAgg(column="weight", aggfunc=sum),
                min_height=NamedAgg(column="height", aggfunc=min),
            )
        )
        expected: DataFrame = DataFrame(
            {
                "total_weight": [7.9, 17.8, 27.8, 7.5, 205.5, 247.5],
                "min_height": [9.1, 9.1, 9.1, 6.0, 6.0, 6.0],
            },
            index=MultiIndex(
                [["cat", "dog"], [0, 1, 2, 3, 4, 5]],
                [[0, 0, 0, 1, 1, 1], [0, 2, 4, 1, 3, 5]],
                names=["kind", None],
            ),
        )
        tm.assert_frame_equal(result, expected)


class TestEWM:
    @pytest.mark.parametrize(
        "method, expected_data",
        [
            ["mean", [0.0, 0.6666666666666666, 1.4285714285714286, 2.2666666666666666]],
            ["std", [np.nan, 0.707107, 0.963624, 1.177164]],
            ["var", [np.nan, 0.5, 0.9285714285714286, 1.3857142857142857]],
        ],
    )
    def test_methods(self, method: str, expected_data: List[Union[float, None]]) -> None:
        # GH 16037
        df = DataFrame({"A": ["a"] * 4, "B": list(range(4))})
        result: DataFrame = getattr(df.groupby("A").ewm(com=1.0), method)()
        expected: DataFrame = DataFrame(
            {"B": expected_data},
            index=MultiIndex.from_tuples(
                [
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                ],
                names=["A", None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_ewm_agg_namedagg(self) -> None:
        # GH#28333
        df = DataFrame({"A": ["a"] * 4, "B": list(range(4))})
        result: DataFrame = (
            df.groupby("A")
            .ewm(com=1.0)
            .agg(
                B_mean=NamedAgg(column="B", aggfunc="mean"),
                B_std=NamedAgg(column="B", aggfunc="std"),
                B_var=NamedAgg(column="B", aggfunc="var"),
            )
        )
        expected: DataFrame = DataFrame(
            {
                "B_mean": [0.0, 0.6666666666666666, 1.4285714285714286, 2.2666666666666666],
                "B_std": [np.nan, 0.707107, 0.963624, 1.177164],
                "B_var": [np.nan, 0.5, 0.9285714285714286, 1.3857142857142857],
            },
            index=MultiIndex.from_tuples(
                [
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                ],
                names=["A", None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "method, expected_data",
        [["corr", [np.nan, 1.0, 1.0, 1]], ["cov", [np.nan, 0.5, 0.928571, 1.385714]]],
    )
    def test_pairwise_methods(self, method: str, expected_data: List[Union[float, None]]) -> None:
        # GH 16037
        df = DataFrame({"A": ["a"] * 4, "B": list(range(4))})
        result: DataFrame = getattr(df.groupby("A").ewm(com=1.0), method)()
        expected: DataFrame = DataFrame(
            {"B": expected_data},
            index=MultiIndex.from_tuples(
                [
                    ("a", 0, "B"),
                    ("a", 1, "B"),
                    ("a", 2, "B"),
                    ("a", 3, "B"),
                ],
                names=["A", None, None],
            ),
        )
        tm.assert_frame_equal(result, expected)

        expected2: DataFrame = df.groupby("A")[["B"]].apply(
            lambda x: getattr(x.ewm(com=1.0), method)()
        )
        tm.assert_frame_equal(result, expected2)

    def test_times(self, times_frame: DataFrame) -> None:
        # GH 40951
        halflife: str = "23 days"
        times: Series = times_frame.pop("C")
        result: DataFrame = times_frame.groupby("A").ewm(halflife=halflife, times=times).mean()
        expected: DataFrame = DataFrame(
            {
                "B": [
                    0.0,
                    0.507534,
                    1.020088,
                    1.537661,
                    0.0,
                    0.567395,
                    1.221209,
                    0.0,
                    0.653141,
                    1.195003,
                ]
            },
            index=MultiIndex.from_tuples(
                [
                    ("a", 0),
                    ("a", 3),
                    ("a", 6),
                    ("a", 9),
                    ("b", 1),
                    ("b", 4),
                    ("b", 7),
                    ("c", 2),
                    ("c", 5),
                    ("c", 8),
                ],
                names=["A", None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_times_array(self, times_frame: DataFrame) -> None:
        # GH 40951
        halflife: str = "23 days"
        times: Series = times_frame.pop("C")
        gb = times_frame.groupby("A")
        result: DataFrame = gb.ewm(halflife=halflife, times=times).mean()
        expected: DataFrame = gb.ewm(halflife=halflife, times=times.values).mean()
        tm.assert_frame_equal(result, expected)

    def test_dont_mutate_obj_after_slicing(self) -> None:
        # GH 43355
        df = DataFrame(
            {
                "id": ["a", "a", "b", "b", "b"],
                "timestamp": date_range("2021-9-1", periods=5, freq="h"),
                "y": list(range(5)),
            }
        )
        grp = df.groupby("id").rolling("1h", on="timestamp")
        result: DataFrame = grp.count()
        expected_df: DataFrame = DataFrame(
            {
                "timestamp": date_range("2021-9-1", periods=5, freq="h"),
                "y": [1.0] * 5,
            },
            index=MultiIndex.from_arrays(
                [["a", "a", "b", "b", "b"], list(range(5))], names=["id", None]
            ),
        )
        tm.assert_frame_equal(result, expected_df)

        result = grp["y"].count()
        expected_series: Series = Series(
            [1.0] * 5,
            index=MultiIndex.from_arrays(
                [
                    ["a", "a", "b", "b", "b"],
                    date_range("2021-9-1", periods=5, freq="h"),
                ],
                names=["id", "timestamp"],
            ),
            name="y",
        )
        tm.assert_series_equal(result, expected_series)
        result = grp.count()
        tm.assert_frame_equal(result, expected_df)


def test_rolling_corr_with_single_integer_in_index() -> None:
    # GH 44078
    df = DataFrame({"a": [(1,), (1,), (1,)], "b": [4, 5, 6]})
    gb = df.groupby(["a"])
    result: DataFrame = gb.rolling(2).corr(other=df)
    index: MultiIndex = MultiIndex.from_tuples([((1,), 0), ((1,), 1), ((1,), 2)], names=["a", None])
    expected: DataFrame = DataFrame(
        {"a": [np.nan, np.nan, np.nan], "b": [np.nan, 1.0, 1.0]}, index=index
    )
    tm.assert_frame_equal(result, expected)


def test_rolling_corr_with_tuples_in_index() -> None:
    # GH 44078
    df = DataFrame(
        {
            "a": [
                (1, 2),
                (1, 2),
                (1, 2),
            ],
            "b": [4, 5, 6],
        }
    )
    gb = df.groupby(["a"])
    result: DataFrame = gb.rolling(2).corr(other=df)
    index: MultiIndex = MultiIndex.from_tuples(
        [((1, 2), 0), ((1, 2), 1), ((1, 2), 2)], names=["a", None]
    )
    expected: DataFrame = DataFrame(
        {"a": [np.nan, np.nan, np.nan], "b": [np.nan, 1.0, 1.0]}, index=index
    )
    tm.assert_frame_equal(result, expected)
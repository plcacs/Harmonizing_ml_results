#!/usr/bin/env python3
from typing import Any, Callable, List, Tuple, Union, Sequence, Optional, Dict
import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    concat,
    date_range,
)
import pandas._testing as tm
from pandas.api.indexers import (
    BaseIndexer,
    FixedForwardWindowIndexer,
)
from pandas.core.indexers.objects import (
    ExpandingIndexer,
    FixedWindowIndexer,
    VariableOffsetWindowIndexer,
)
from pandas.tseries.offsets import BusinessDay


def test_bad_get_window_bounds_signature() -> None:
    class BadIndexer(BaseIndexer):
        def get_window_bounds(
            self,
            num_values: int,
            min_periods: int,
            center: bool,
            closed: str,
            step: int = 1
        ) -> Tuple[np.ndarray, np.ndarray]:
            return None  # type: ignore

    indexer = BadIndexer()
    with pytest.raises(ValueError, match="BadIndexer does not implement"):
        Series(range(5)).rolling(indexer)


def test_expanding_indexer() -> None:
    s: Series = Series(range(10))
    indexer: ExpandingIndexer = ExpandingIndexer()
    result: Series = s.rolling(indexer).mean()
    expected: Series = s.expanding().mean()
    tm.assert_series_equal(result, expected)


def test_indexer_constructor_arg() -> None:
    # Example found in computation.rst
    use_expanding: List[bool] = [True, False, True, False, True]
    df: DataFrame = DataFrame({"values": range(5)})

    class CustomIndexer(BaseIndexer):
        def __init__(self, window_size: int, use_expanding: List[bool]) -> None:
            self.window_size = window_size
            self.use_expanding = use_expanding

        def get_window_bounds(
            self,
            num_values: int,
            min_periods: int,
            center: bool,
            closed: str,
            step: int = 1
        ) -> Tuple[np.ndarray, np.ndarray]:
            start: np.ndarray = np.empty(num_values, dtype=np.int64)
            end: np.ndarray = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if self.use_expanding[i]:
                    start[i] = 0
                    end[i] = i + 1
                else:
                    start[i] = i
                    end[i] = i + self.window_size
            return start, end

    indexer = CustomIndexer(window_size=1, use_expanding=use_expanding)
    result: DataFrame = df.rolling(indexer).sum()
    expected: DataFrame = DataFrame({"values": [0.0, 1.0, 3.0, 3.0, 10.0]})
    tm.assert_frame_equal(result, expected)


def test_indexer_accepts_rolling_args() -> None:
    df: DataFrame = DataFrame({"values": range(5)})

    class CustomIndexer(BaseIndexer):
        def __init__(self, window_size: int) -> None:
            self.window_size = window_size

        def get_window_bounds(
            self,
            num_values: int,
            min_periods: int,
            center: bool,
            closed: str,
            step: int = 1
        ) -> Tuple[np.ndarray, np.ndarray]:
            start: np.ndarray = np.empty(num_values, dtype=np.int64)
            end: np.ndarray = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if (
                    center
                    and min_periods == 1
                    and closed == "both"
                    and step == 1
                    and i == 2
                ):
                    start[i] = 0
                    end[i] = num_values
                else:
                    start[i] = i
                    end[i] = i + self.window_size
            return start, end

    indexer = CustomIndexer(window_size=1)
    result: DataFrame = df.rolling(indexer, center=True, min_periods=1, closed="both", step=1).sum()
    expected: DataFrame = DataFrame({"values": [0.0, 1.0, 10.0, 3.0, 4.0]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func,np_func,expected,np_kwargs",
    [
        ("count", len, [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, np.nan], {}),
        ("min", np.min, [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 6.0, 7.0, 8.0, np.nan], {}),
        (
            "max",
            np.max,
            [2.0, 3.0, 4.0, 100.0, 100.0, 100.0, 8.0, 9.0, 9.0, np.nan],
            {},
        ),
        (
            "std",
            np.std,
            [
                1.0,
                1.0,
                1.0,
                55.71654452,
                54.85739087,
                53.9845657,
                1.0,
                1.0,
                0.70710678,
                np.nan,
            ],
            {"ddof": 1},
        ),
        (
            "var",
            np.var,
            [
                1.0,
                1.0,
                1.0,
                3104.333333,
                3009.333333,
                2914.333333,
                1.0,
                1.0,
                0.500000,
                np.nan,
            ],
            {"ddof": 1},
        ),
        (
            "median",
            np.median,
            [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 7.0, 8.0, 8.5, np.nan],
            {},
        ),
    ],
)
def test_rolling_forward_window(
    frame_or_series: Callable[[np.ndarray], Union[Series, DataFrame]],
    func: str,
    np_func: Callable[..., Any],
    expected: Sequence[Any],
    np_kwargs: Dict[str, Any],
    step: int
) -> None:
    # GH 32865
    values: np.ndarray = np.arange(10.0)
    values[5] = 100.0

    indexer: FixedForwardWindowIndexer = FixedForwardWindowIndexer(window_size=3)

    match: str = "Forward-looking windows can't have center=True"
    rolling = frame_or_series(values).rolling(window=indexer, center=True)
    with pytest.raises(ValueError, match=match):
        getattr(rolling, func)()

    match = "Forward-looking windows don't support setting the closed argument"
    rolling = frame_or_series(values).rolling(window=indexer, closed="right")
    with pytest.raises(ValueError, match=match):
        getattr(rolling, func)()

    rolling = frame_or_series(values).rolling(window=indexer, min_periods=2, step=step)
    result = getattr(rolling, func)()

    # Check that the function output matches the explicitly provided array
    expected_series = frame_or_series(np.array(expected))[::step]
    tm.assert_equal(result, expected_series)

    # Check that the rolling function output matches applying an alternative
    # function to the rolling window object
    expected2 = frame_or_series(rolling.apply(lambda x: np_func(x, **np_kwargs)))
    tm.assert_equal(result, expected2)

    # Check that the function output matches applying an alternative function
    # if min_periods isn't specified
    # GH 39604: After count-min_periods deprecation, apply(lambda x: len(x))
    # is equivalent to count after setting min_periods=0
    min_periods: Optional[int] = 0 if func == "count" else None
    rolling3 = frame_or_series(values).rolling(window=indexer, min_periods=min_periods)
    result3 = getattr(rolling3, func)()
    expected3 = frame_or_series(rolling3.apply(lambda x: np_func(x, **np_kwargs)))
    tm.assert_equal(result3, expected3)


def test_rolling_forward_skewness(
    frame_or_series: Callable[[np.ndarray], Union[Series, DataFrame]], step: int
) -> None:
    values: np.ndarray = np.arange(10.0)
    values[5] = 100.0

    indexer: FixedForwardWindowIndexer = FixedForwardWindowIndexer(window_size=5)
    rolling = frame_or_series(values).rolling(window=indexer, min_periods=3, step=step)
    result = rolling.skew()

    expected = frame_or_series(
        [
            0.0,
            2.232396,
            2.229508,
            2.228340,
            2.229091,
            2.231989,
            0.0,
            0.0,
            np.nan,
            np.nan,
        ]
    )[::step]
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "func,expected",
    [
        ("cov", [2.0, 2.0, 2.0, 97.0, 2.0, -93.0, 2.0, 2.0, np.nan, np.nan]),
        (
            "corr",
            [
                1.0,
                1.0,
                1.0,
                0.8704775290207161,
                0.018229084250926637,
                -0.861357304646493,
                1.0,
                1.0,
                np.nan,
                np.nan,
            ],
        ),
    ],
)
def test_rolling_forward_cov_corr(
    func: str, expected: Sequence[Any]
) -> None:
    values1: np.ndarray = np.arange(10).reshape(-1, 1)
    values2: np.ndarray = values1 * 2
    values1[5, 0] = 100
    values: np.ndarray = np.concatenate([values1, values2], axis=1)

    indexer: FixedForwardWindowIndexer = FixedForwardWindowIndexer(window_size=3)
    rolling = DataFrame(values).rolling(window=indexer, min_periods=3)
    # We are interested in checking only pairwise covariance / correlation
    result = getattr(rolling, func)().loc[(slice(None), 1), 0]
    result = result.reset_index(drop=True)
    expected_series: Series = Series(expected).reset_index(drop=True)
    expected_series.name = result.name
    tm.assert_equal(result, expected_series)


@pytest.mark.parametrize(
    "closed,expected_data",
    [
        ["right", [0.0, 1.0, 2.0, 3.0, 7.0, 12.0, 6.0, 7.0, 8.0, 9.0]],
        ["left", [0.0, 0.0, 1.0, 2.0, 5.0, 9.0, 5.0, 6.0, 7.0, 8.0]],
    ],
)
def test_non_fixed_variable_window_indexer(closed: str, expected_data: Sequence[Union[float, int]]) -> None:
    index = date_range("2020", periods=10)
    df: DataFrame = DataFrame(range(10), index=index)
    offset: BusinessDay = BusinessDay(1)
    indexer: VariableOffsetWindowIndexer = VariableOffsetWindowIndexer(index=index, offset=offset)
    result = df.rolling(indexer, closed=closed).sum()
    expected: DataFrame = DataFrame(expected_data, index=index)
    tm.assert_frame_equal(result, expected)


def test_variableoffsetwindowindexer_not_dti() -> None:
    # GH 54379
    with pytest.raises(ValueError, match="index must be a DatetimeIndex."):
        VariableOffsetWindowIndexer(index="foo", offset=BusinessDay(1))  # type: ignore


def test_variableoffsetwindowindexer_not_offset() -> None:
    # GH 54379
    idx = date_range("2020", periods=10)
    with pytest.raises(ValueError, match="offset must be a DateOffset-like object."):
        VariableOffsetWindowIndexer(index=idx, offset="foo")  # type: ignore


def test_fixed_forward_indexer_count(step: int) -> None:
    # GH: 35579
    df: DataFrame = DataFrame({"b": [None, None, None, 7]})
    indexer: FixedForwardWindowIndexer = FixedForwardWindowIndexer(window_size=2)
    result = df.rolling(window=indexer, min_periods=0, step=step).count()
    expected = DataFrame({"b": [0.0, 0.0, 1.0, 1.0]})[::step]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("end_value", "values"), [(1, [0.0, 1, 1, 3, 2]), (-1, [0.0, 1, 0, 3, 1])]
)
@pytest.mark.parametrize(("func", "args"), [("median", []), ("quantile", [0.5])])
def test_indexer_quantile_sum(
    end_value: int, values: List[Union[int, float]], func: str, args: List[Any]
) -> None:
    # GH 37153
    class CustomIndexer(BaseIndexer):
        def __init__(self, window_size: int, use_expanding: List[bool]) -> None:
            self.window_size = window_size
            self.use_expanding = use_expanding

        def get_window_bounds(
            self,
            num_values: int,
            min_periods: int,
            center: bool,
            closed: str,
            step: int = 1
        ) -> Tuple[np.ndarray, np.ndarray]:
            start: np.ndarray = np.empty(num_values, dtype=np.int64)
            end_arr: np.ndarray = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if self.use_expanding[i]:
                    start[i] = 0
                    end_arr[i] = max(i + end_value, 1)
                else:
                    start[i] = i
                    end_arr[i] = i + self.window_size
            return start, end_arr

    use_expanding: List[bool] = [True, False, True, False, True]
    df: DataFrame = DataFrame({"values": range(5)})

    indexer: CustomIndexer = CustomIndexer(window_size=1, use_expanding=use_expanding)
    result = getattr(df.rolling(indexer), func)(*args)
    expected = DataFrame({"values": values})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "indexer_class", [FixedWindowIndexer, FixedForwardWindowIndexer, ExpandingIndexer]
)
@pytest.mark.parametrize("window_size", [1, 2, 12])
@pytest.mark.parametrize(
    "df_data",
    [
        {"a": [1, 1], "b": [0, 1]},
        {"a": [1, 2], "b": [0, 1]},
        {"a": [1] * 16, "b": [np.nan, 1, 2, np.nan] + list(range(4, 16))},
    ],
)
def test_indexers_are_reusable_after_groupby_rolling(
    indexer_class: Any, window_size: int, df_data: Dict[str, Any]
) -> None:
    # GH 43267
    df: DataFrame = DataFrame(df_data)
    num_trials: int = 3
    indexer = indexer_class(window_size=window_size)
    original_window_size: int = indexer.window_size
    for i in range(num_trials):
        df.groupby("a")["b"].rolling(window=indexer, min_periods=1).mean()
        assert indexer.window_size == original_window_size


@pytest.mark.parametrize(
    "window_size, num_values, expected_start, expected_end",
    [
        (1, 1, [0], [1]),
        (1, 2, [0, 1], [1, 2]),
        (2, 1, [0], [1]),
        (2, 2, [0, 1], [2, 2]),
        (5, 12, list(range(12)), list(range(5, 12)) + [12] * 5),
        (12, 5, list(range(5)), [5] * 5),
        (0, 0, np.array([]), np.array([])),
        (1, 0, np.array([]), np.array([])),
        (0, 1, [0], [0]),
    ],
)
def test_fixed_forward_indexer_bounds(
    window_size: int,
    num_values: int,
    expected_start: Union[Sequence[int], np.ndarray],
    expected_end: Union[Sequence[int], np.ndarray],
    step: int
) -> None:
    # GH 43267
    indexer: FixedForwardWindowIndexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)

    tm.assert_numpy_array_equal(
        start, np.array(expected_start[::step]), check_dtype=False
    )
    tm.assert_numpy_array_equal(end, np.array(expected_end[::step]), check_dtype=False)
    assert len(start) == len(end)


@pytest.mark.parametrize(
    "df, window_size, expected",
    [
        (
            DataFrame({"b": [0, 1, 2], "a": [1, 2, 2]}),
            2,
            Series(
                [0, 1.5, 2.0],
                index=MultiIndex.from_arrays([[1, 2, 2], list(range(3))], names=["a", None]),
                name="b",
                dtype=np.float64,
            ),
        ),
        (
            DataFrame(
                {
                    "b": [np.nan, 1, 2, np.nan] + list(range(4, 18)),
                    "a": [1] * 7 + [2] * 11,
                    "c": list(range(18)),
                }
            ),
            12,
            Series(
                [
                    3.6,
                    3.6,
                    4.25,
                    5.0,
                    5.0,
                    5.5,
                    6.0,
                    12.0,
                    12.5,
                    13.0,
                    13.5,
                    14.0,
                    14.5,
                    15.0,
                    15.5,
                    16.0,
                    16.5,
                    17.0,
                ],
                index=MultiIndex.from_arrays(
                    [[1] * 7 + [2] * 11, list(range(18))], names=["a", None]
                ),
                name="b",
                dtype=np.float64,
            ),
        ),
    ],
)
def test_rolling_groupby_with_fixed_forward_specific(
    df: DataFrame, window_size: int, expected: Series
) -> None:
    # GH 43267
    indexer: FixedForwardWindowIndexer = FixedForwardWindowIndexer(window_size=window_size)
    result: Series = df.groupby("a")["b"].rolling(window=indexer, min_periods=1).mean()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "group_keys",
    [
        (1,),
        (1, 2),
        (2, 1),
        (1, 1, 2),
        (1, 2, 1),
        (1, 1, 2, 2),
        (1, 2, 3, 2, 3),
        (1, 1, 2) * 4,
        (1, 2, 3) * 5,
    ],
)
@pytest.mark.parametrize("window_size", [1, 2, 3, 4, 5, 8, 20])
def test_rolling_groupby_with_fixed_forward_many(
    group_keys: Tuple[Any, ...], window_size: int
) -> None:
    # GH 43267
    df: DataFrame = DataFrame(
        {
            "a": np.array(list(group_keys)),
            "b": np.arange(len(group_keys), dtype=np.float64) + 17,
            "c": np.arange(len(group_keys), dtype=np.int64),
        }
    )

    indexer: FixedForwardWindowIndexer = FixedForwardWindowIndexer(window_size=window_size)
    result: Series = df.groupby("a")["b"].rolling(window=indexer, min_periods=1).sum()
    result.index.names = ["a", "c"]

    groups = df.groupby("a")[["a", "b", "c"]]
    manual: DataFrame = concat(
        [
            g.assign(
                b=[
                    g["b"].iloc[i : i + window_size].sum(min_count=1)
                    for i in range(len(g))
                ]
            )
            for _, g in groups
        ]
    )
    manual = manual.set_index(["a", "c"])["b"]

    tm.assert_series_equal(result, manual)


def test_unequal_start_end_bounds() -> None:
    class CustomIndexer(BaseIndexer):
        def get_window_bounds(
            self,
            num_values: int,
            min_periods: int,
            center: bool,
            closed: str,
            step: int = 1
        ) -> Tuple[np.ndarray, np.ndarray]:
            return np.array([1]), np.array([1, 2])

    indexer = CustomIndexer()
    roll = Series(1).rolling(indexer)
    match: str = "start"
    with pytest.raises(ValueError, match=match):
        roll.mean()

    with pytest.raises(ValueError, match=match):
        next(iter(roll))

    with pytest.raises(ValueError, match=match):
        roll.corr(pairwise=True)

    with pytest.raises(ValueError, match=match):
        roll.cov(pairwise=True)


def test_unequal_bounds_to_object() -> None:
    # GH 44470
    class CustomIndexer(BaseIndexer):
        def get_window_bounds(
            self,
            num_values: int,
            min_periods: int,
            center: bool,
            closed: str,
            step: int = 1
        ) -> Tuple[np.ndarray, np.ndarray]:
            return np.array([1]), np.array([2])

    indexer = CustomIndexer()
    roll = Series([1, 1]).rolling(indexer)
    match: str = "start and end"
    with pytest.raises(ValueError, match=match):
        roll.mean()

    with pytest.raises(ValueError, match=match):
        next(iter(roll))

    with pytest.raises(ValueError, match=match):
        roll.corr(pairwise=True)

    with pytest.raises(ValueError, match=match):
        roll.cov(pairwise=True)

"""Tests for Interval-Interval operations, such as overlaps, contains, etc."""
import numpy as np
import pytest
from pandas import Interval, IntervalIndex, Timedelta, Timestamp
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
from typing import Type, Tuple, Union, Any

@pytest.fixture(params=[IntervalArray, IntervalIndex])
def constructor(request: pytest.FixtureRequest) -> Union[Type[IntervalArray], Type[IntervalIndex]]:
    """
    Fixture for testing both interval container classes.
    """
    return request.param

@pytest.fixture(
    params=[
        (Timedelta("0 days"), Timedelta("1 day")),
        (Timestamp("2018-01-01"), Timedelta("1 day")),
        (0, 1),
    ],
    ids=lambda x: type(x[0]).__name__,
)
def start_shift(
    request: pytest.FixtureRequest,
) -> Union[Tuple[Timedelta, Timedelta], Tuple[Timestamp, Timedelta], Tuple[int, int]]:
    """
    Fixture for generating intervals of different types from a start value
    and a shift value that can be added to start to generate an endpoint.
    """
    return request.param

class TestOverlaps:

    def test_overlaps_interval(
        self,
        constructor: Union[Type[IntervalArray], Type[IntervalIndex]],
        start_shift: Union[Tuple[Timedelta, Timedelta], Tuple[Timestamp, Timedelta], Tuple[int, int]],
        closed: str,
        other_closed: str,
    ) -> None:
        start, shift = start_shift
        interval = Interval(start, start + 3 * shift, other_closed)
        tuples: list[Union[Tuple[Any, Any], Any]] = [
            (start, start + 3 * shift),
            (start + shift, start + 2 * shift),
            (start - shift, start + 4 * shift),
            (start + 2 * shift, start + 4 * shift),
            (start + 3 * shift, start + 4 * shift),
            (start + 4 * shift, start + 5 * shift),
        ]
        interval_container = constructor.from_tuples(tuples, closed=closed)
        adjacent = interval.closed_right and interval_container.closed_left
        expected: np.ndarray = np.array([True, True, True, True, adjacent, False])
        result: np.ndarray = interval_container.overlaps(interval)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("other_constructor", [IntervalArray, IntervalIndex])
    def test_overlaps_interval_container(
        self,
        constructor: Union[Type[IntervalArray], Type[IntervalIndex]],
        other_constructor: Union[Type[IntervalArray], Type[IntervalIndex]],
    ) -> None:
        interval_container = constructor.from_breaks(range(5))
        other_container = other_constructor.from_breaks(range(5))
        with pytest.raises(NotImplementedError, match="^$"):
            interval_container.overlaps(other_container)

    def test_overlaps_na(
        self,
        constructor: Union[Type[IntervalArray], Type[IntervalIndex]],
        start_shift: Union[Tuple[Timedelta, Timedelta], Tuple[Timestamp, Timedelta], Tuple[int, int]],
    ) -> None:
        """NA values are marked as False"""
        start, shift = start_shift
        interval = Interval(start, start + shift)
        tuples: list[Union[Tuple[Any, Any], float]] = [
            (start, start + shift),
            np.nan,
            (start + 2 * shift, start + 3 * shift),
        ]
        interval_container = constructor.from_tuples(tuples)
        expected: np.ndarray = np.array([True, False, False])
        result: np.ndarray = interval_container.overlaps(interval)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [10, True, "foo", Timedelta("1 day"), Timestamp("2018-01-01")],
        ids=lambda x: type(x).__name__,
    )
    def test_overlaps_invalid_type(
        self,
        constructor: Union[Type[IntervalArray], Type[IntervalIndex]],
        other: Any,
    ) -> None:
        interval_container = constructor.from_breaks(range(5))
        msg: str = f"`other` must be Interval-like, got {type(other).__name__}"
        with pytest.raises(TypeError, match=msg):
            interval_container.overlaps(other)

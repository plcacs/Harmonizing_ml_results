from typing import Any, List, Tuple, Union, Type, Protocol, TypeVar

import numpy as np
import pytest

from pandas import (
    Interval,
    IntervalIndex,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import IntervalArray

T = TypeVar("T", int, Timedelta, Timestamp)


class IntervalContainer(Protocol):
    closed: str

    @classmethod
    def from_tuples(
        cls: Type["IntervalContainer"],
        data: List[Union[Tuple[Any, Any], Any]],
        closed: str = "left",
    ) -> "IntervalContainer":
        ...

    @classmethod
    def from_breaks(
        cls: Type["IntervalContainer"],
        breaks: Any,
        closed: str = "left",
    ) -> "IntervalContainer":
        ...

    def overlaps(self, other: Interval) -> np.ndarray:
        ...


@pytest.fixture(params=[IntervalArray, IntervalIndex])
def constructor(request: Any) -> Type[IntervalContainer]:
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
def start_shift(request: Any) -> Tuple[T, T]:
    """
    Fixture for generating intervals of different types from a start value
    and a shift value that can be added to start to generate an endpoint.
    """
    return request.param


class TestOverlaps:
    def test_overlaps_interval(
        self,
        constructor: Type[IntervalContainer],
        start_shift: Tuple[T, T],
        closed: str,
        other_closed: str,
    ) -> None:
        start, shift = start_shift
        interval = Interval(start, start + 3 * shift, other_closed)

        # intervals: identical, nested, spanning, partial, adjacent, disjoint
        tuples: List[Union[Tuple[Any, Any], Any]] = [
            (start, start + 3 * shift),
            (start + shift, start + 2 * shift),
            (start - shift, start + 4 * shift),
            (start + 2 * shift, start + 4 * shift),
            (start + 3 * shift, start + 4 * shift),
            (start + 4 * shift, start + 5 * shift),
        ]
        interval_container: IntervalContainer = constructor.from_tuples(tuples, closed)

        adjacent: bool = interval.closed_right and interval_container.closed_left
        expected: np.ndarray = np.array([True, True, True, True, adjacent, False])
        result: np.ndarray = interval_container.overlaps(interval)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("other_constructor", [IntervalArray, IntervalIndex])
    def test_overlaps_interval_container(
        self,
        constructor: Type[IntervalContainer],
        other_constructor: Type[IntervalContainer],
    ) -> None:
        # TODO: modify this test when implemented
        interval_container: IntervalContainer = constructor.from_breaks(range(5))
        other_container: IntervalContainer = other_constructor.from_breaks(range(5))
        with pytest.raises(NotImplementedError, match="^$"):
            interval_container.overlaps(other_container)  # type: ignore

    def test_overlaps_na(
        self, constructor: Type[IntervalContainer], start_shift: Tuple[T, T]
    ) -> None:
        """NA values are marked as False"""
        start, shift = start_shift
        interval = Interval(start, start + shift)

        tuples: List[Union[Tuple[Any, Any], Any]] = [
            (start, start + shift),
            np.nan,
            (start + 2 * shift, start + 3 * shift),
        ]
        interval_container: IntervalContainer = constructor.from_tuples(tuples)

        expected: np.ndarray = np.array([True, False, False])
        result: np.ndarray = interval_container.overlaps(interval)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [10, True, "foo", Timedelta("1 day"), Timestamp("2018-01-01")],
        ids=lambda x: type(x).__name__,
    )
    def test_overlaps_invalid_type(
        self, constructor: Type[IntervalContainer], other: Union[int, bool, str, Timedelta, Timestamp]
    ) -> None:
        interval_container: IntervalContainer = constructor.from_breaks(range(5))
        msg: str = f"`other` must be Interval-like, got {type(other).__name__}"
        with pytest.raises(TypeError, match=msg):
            interval_container.overlaps(other)  # type: ignore

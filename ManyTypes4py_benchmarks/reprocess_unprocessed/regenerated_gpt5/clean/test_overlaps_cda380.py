"""Tests for Interval-Interval operations, such as overlaps, contains, etc."""
import numpy as np
import pytest
from pandas import Interval, IntervalIndex, Timedelta, Timestamp
import pandas._testing as tm
from pandas.core.arrays import IntervalArray

# Type aliases
IntervalContainer = IntervalArray | IntervalIndex
StartType = Timedelta | Timestamp | int
ShiftType = Timedelta | int


@pytest.fixture(params=[IntervalArray, IntervalIndex])
def constructor(request):
    """
    Fixture for testing both interval container classes.
    """
    constructor_cls: type[IntervalArray] | type[IntervalIndex] = request.param
    return constructor_cls


@pytest.fixture(
    params=[
        (Timedelta('0 days'), Timedelta('1 day')),
        (Timestamp('2018-01-01'), Timedelta('1 day')),
        (0, 1)
    ],
    ids=lambda x: type(x[0]).__name__
)
def start_shift(request):
    """
    Fixture for generating intervals of different types from a start value
    and a shift value that can be added to start to generate an endpoint.
    """
    pair: tuple[StartType, ShiftType] = request.param
    return pair


class TestOverlaps:

    def test_overlaps_interval(self, constructor, start_shift, closed, other_closed):
        start: StartType
        shift: ShiftType
        start, shift = start_shift
        interval: Interval = Interval(start, start + 3 * shift, other_closed)
        tuples: list[tuple[StartType, StartType]] = [
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

    @pytest.mark.parametrize('other_constructor', [IntervalArray, IntervalIndex])
    def test_overlaps_interval_container(self, constructor, other_constructor):
        interval_container: IntervalContainer = constructor.from_breaks(range(5))
        other_constructor_cls: type[IntervalArray] | type[IntervalIndex] = other_constructor
        other_container: IntervalContainer = other_constructor_cls.from_breaks(range(5))
        with pytest.raises(NotImplementedError, match='^$'):
            interval_container.overlaps(other_container)

    def test_overlaps_na(self, constructor, start_shift):
        """NA values are marked as False"""
        start: StartType
        shift: ShiftType
        start, shift = start_shift
        interval: Interval = Interval(start, start + shift)
        tuples: list[tuple[StartType, StartType] | float] = [
            (start, start + shift),
            np.nan,
            (start + 2 * shift, start + 3 * shift),
        ]
        interval_container: IntervalContainer = constructor.from_tuples(tuples)
        expected: np.ndarray = np.array([True, False, False])
        result: np.ndarray = interval_container.overlaps(interval)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        'other',
        [10, True, 'foo', Timedelta('1 day'), Timestamp('2018-01-01')],
        ids=lambda x: type(x).__name__,
    )
    def test_overlaps_invalid_type(self, constructor, other):
        interval_container: IntervalContainer = constructor.from_breaks(range(5))
        msg: str = f'`other` must be Interval-like, got {type(other).__name__}'
        with pytest.raises(TypeError, match=msg):
            interval_container.overlaps(other)
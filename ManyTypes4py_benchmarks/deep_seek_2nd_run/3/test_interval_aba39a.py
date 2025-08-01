import numpy as np
import pytest
from pandas import Interval, Timedelta, Timestamp
from typing import Any, Tuple, Union

@pytest.fixture
def interval() -> Interval:
    return Interval(0, 1)

class TestInterval:

    def test_properties(self, interval: Interval) -> None:
        assert interval.closed == 'right'
        assert interval.left == 0
        assert interval.right == 1
        assert interval.mid == 0.5

    def test_hash(self, interval: Interval) -> None:
        hash(interval)

    @pytest.mark.parametrize('left, right, expected', [
        (0, 5, 5), 
        (-2, 5.5, 7.5), 
        (10, 10, 0), 
        (10, np.inf, np.inf), 
        (-np.inf, -5, np.inf), 
        (-np.inf, np.inf, np.inf), 
        (Timedelta('0 days'), Timedelta('5 days'), Timedelta('5 days')), 
        (Timedelta('10 days'), Timedelta('10 days'), Timedelta('0 days')), 
        (Timedelta('1h10min'), Timedelta('5h5min'), Timedelta('3h55min')), 
        (Timedelta('5s'), Timedelta('1h'), Timedelta('59min55s'))
    ])
    def test_length(self, left: Union[int, float, Timedelta], right: Union[int, float, Timedelta], expected: Union[int, float, Timedelta]) -> None:
        iv = Interval(left, right)
        result = iv.length
        assert result == expected

    @pytest.mark.parametrize('left, right, expected', [
        ('2017-01-01', '2017-01-06', '5 days'), 
        ('2017-01-01', '2017-01-01 12:00:00', '12 hours'), 
        ('2017-01-01 12:00', '2017-01-01 12:00:00', '0 days'), 
        ('2017-01-01 12:01', '2017-01-05 17:31:00', '4 days 5 hours 30 min')
    ])
    @pytest.mark.parametrize('tz', (None, 'UTC', 'CET', 'US/Eastern'))
    def test_length_timestamp(self, tz: Union[str, None], left: str, right: str, expected: str) -> None:
        iv = Interval(Timestamp(left, tz=tz), Timestamp(right, tz=tz))
        result = iv.length
        expected = Timedelta(expected)
        assert result == expected

    @pytest.mark.parametrize('left, right', [
        (0, 1), 
        (Timedelta('0 days'), Timedelta('1 day')), 
        (Timestamp('2018-01-01'), Timestamp('2018-01-02')), 
        (Timestamp('2018-01-01', tz='US/Eastern'), Timestamp('2018-01-02', tz='US/Eastern'))
    ])
    def test_is_empty(self, left: Any, right: Any, closed: str) -> None:
        iv = Interval(left, right, closed)
        assert iv.is_empty is False
        iv = Interval(left, left, closed)
        result = iv.is_empty
        expected = closed != 'both'
        assert result is expected

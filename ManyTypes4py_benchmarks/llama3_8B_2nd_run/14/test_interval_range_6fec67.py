from datetime import timedelta
import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer
from pandas import DateOffset, Interval, IntervalIndex, Timedelta, Timestamp, date_range, interval_range, timedelta_range
import pandas._testing as tm
from pandas.tseries.offsets import Day

@pytest.fixture(params=[None, 'foo'])
def name(request) -> str:
    return request.param

class TestIntervalRange:
    @pytest.mark.parametrize('freq, periods', [(1, 100), (2.5, 40), (5, 20), (25, 4)])
    def test_constructor_numeric(self, closed: str, name: str, freq: float, periods: int) -> None:
        start, end = (0, 100)
        breaks = np.arange(101, step=freq)
        expected = IntervalIndex.from_breaks(breaks, name=name, closed=closed)
        result = interval_range(start=start, end=end, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(start=start, periods=periods, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(end=end, periods=periods, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(start=start, end=end, periods=periods, name=name, closed=closed)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    @pytest.mark.parametrize('freq, periods', [('D', 364), ('2D', 182), ('22D18h', 16), ('ME', 11)])
    def test_constructor_timestamp(self, closed: str, name: str, freq: str, periods: int, tz: str) -> None:
        start, end = (Timestamp('20180101', tz=tz), Timestamp('20181231', tz=tz))
        breaks = date_range(start=start, end=end, freq=freq)
        expected = IntervalIndex.from_breaks(breaks, name=name, closed=closed)
        result = interval_range(start=start, end=end, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(start=start, periods=periods, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(end=end, periods=periods, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        if not breaks.freq.n == 1 and tz is None:
            result = interval_range(start=start, end=end, periods=periods, name=name, closed=closed)
            tm.assert_index_equal(result, expected)

    # ... and so on for the rest of the methods

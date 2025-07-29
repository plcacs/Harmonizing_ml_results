from datetime import timedelta
from typing import Any, Optional, Union

import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer
from pandas import DateOffset, Interval, IntervalIndex, Timedelta, Timestamp, date_range, interval_range, timedelta_range
import pandas._testing as tm
from pandas.tseries.offsets import Day

@pytest.fixture(params=[None, 'foo'])
def name(request: pytest.FixtureRequest) -> Optional[str]:
    return request.param

class TestIntervalRange:

    @pytest.mark.parametrize('freq, periods', [(1, 100), (2.5, 40), (5, 20), (25, 4)])
    def test_constructor_numeric(self, closed: str, name: Optional[str], freq: Union[int, float], periods: int) -> None:
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
    def test_constructor_timestamp(self, closed: str, name: Optional[str], freq: str, periods: int, tz: Optional[str]) -> None:
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

    @pytest.mark.parametrize('freq, periods', [('D', 100), ('2D12h', 40), ('5D', 20), ('25D', 4)])
    def test_constructor_timedelta(self, closed: str, name: Optional[str], freq: str, periods: int) -> None:
        start, end = (Timedelta('0 days'), Timedelta('100 days'))
        breaks = timedelta_range(start=start, end=end, freq=freq)
        expected = IntervalIndex.from_breaks(breaks, name=name, closed=closed)
        result = interval_range(start=start, end=end, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(start=start, periods=periods, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(end=end, periods=periods, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(start=start, end=end, periods=periods, name=name, closed=closed)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('start, end, freq, expected_endpoint', [(0, 10, 3, 9), (0, 10, 1.5, 9), (0.5, 10, 3, 9.5), (Timedelta('0D'), Timedelta('10D'), '2D4h', Timedelta('8D16h')), (Timestamp('2018-01-01'), Timestamp('2018-02-09'), 'MS', Timestamp('2018-02-01')), (Timestamp('2018-01-01', tz='US/Eastern'), Timestamp('2018-01-20', tz='US/Eastern'), '5D12h', Timestamp('2018-01-17 12:00:00', tz='US/Eastern'))])
    def test_early_truncation(self, start: Union[int, float, Timestamp, Timedelta], end: Union[int, float, Timestamp, Timedelta], freq: Union[int, float, str], expected_endpoint: Union[int, float, Timestamp, Timedelta]) -> None:
        result = interval_range(start=start, end=end, freq=freq)
        result_endpoint = result.right[-1]
        assert result_endpoint == expected_endpoint

    @pytest.mark.parametrize('start, end, freq', [(0.5, None, None), (None, 4.5, None), (0.5, None, 1.5), (None, 6.5, 1.5)])
    def test_no_invalid_float_truncation(self, start: Optional[float], end: Optional[float], freq: Optional[float]) -> None:
        if freq is None:
            breaks = [0.5, 1.5, 2.5, 3.5, 4.5]
        else:
            breaks = [0.5, 2.0, 3.5, 5.0, 6.5]
        expected = IntervalIndex.from_breaks(breaks)
        result = interval_range(start=start, end=end, periods=4, freq=freq)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('start, mid, end', [(Timestamp('2018-03-10', tz='US/Eastern'), Timestamp('2018-03-10 23:30:00', tz='US/Eastern'), Timestamp('2018-03-12', tz='US/Eastern')), (Timestamp('2018-11-03', tz='US/Eastern'), Timestamp('2018-11-04 00:30:00', tz='US/Eastern'), Timestamp('2018-11-05', tz='US/Eastern'))])
    def test_linspace_dst_transition(self, start: Timestamp, mid: Timestamp, end: Timestamp) -> None:
        start = start.as_unit('ns')
        mid = mid.as_unit('ns')
        end = end.as_unit('ns')
        result = interval_range(start=start, end=end, periods=2)
        expected = IntervalIndex.from_breaks([start, mid, end])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('freq', [2, 2.0])
    @pytest.mark.parametrize('end', [10, 10.0])
    @pytest.mark.parametrize('start', [0, 0.0])
    def test_float_subtype(self, start: Union[int, float], end: Union[int, float], freq: Union[int, float]) -> None:
        index = interval_range(start=start, end=end, freq=freq)
        result = index.dtype.subtype
        expected = 'int64' if is_integer(start + end + freq) else 'float64'
        assert result == expected
        index = interval_range(start=start, periods=5, freq=freq)
        result = index.dtype.subtype
        expected = 'int64' if is_integer(start + freq) else 'float64'
        assert result == expected
        index = interval_range(end=end, periods=5, freq=freq)
        result = index.dtype.subtype
        expected = 'int64' if is_integer(end + freq) else 'float64'
        assert result == expected
        index = interval_range(start=start, end=end, periods=5)
        result = index.dtype.subtype
        expected = 'int64' if is_integer(start + end) else 'float64'
        assert result == expected

    @pytest.mark.parametrize('start, end, expected', [(np.int8(1), np.int8(10), np.dtype('int8')), (np.int8(1), np.float16(10), np.dtype('float64')), (np.float32(1), np.float32(10), np.dtype('float32')), (1, 10, np.dtype('int64')), (1, 10.0, np.dtype('float64'))])
    def test_interval_dtype(self, start: Union[int, np.number], end: Union[int, float, np.number], expected: np.dtype) -> None:
        result = interval_range(start=start, end=end).dtype.subtype
        assert result == expected

    def test_interval_range_fractional_period(self) -> None:
        msg = 'periods must be an integer, got 10.5'
        ts = Timestamp('2024-03-25')
        with pytest.raises(TypeError, match=msg):
            interval_range(ts, periods=10.5)

    def test_constructor_coverage(self) -> None:
        start, end = (Timestamp('2017-01-01'), Timestamp('2017-01-15'))
        expected = interval_range(start=start, end=end)
        result = interval_range(start=start.to_pydatetime(), end=end.to_pydatetime())
        tm.assert_index_equal(result, expected)
        result = interval_range(start=start.asm8, end=end.asm8)
        tm.assert_index_equal(result, expected)
        equiv_freq: list[Union[str, Day, Timedelta, timedelta, DateOffset]] = ['D', Day(), Timedelta(days=1), timedelta(days=1), DateOffset(days=1)]
        for freq in equiv_freq:
            result = interval_range(start=start, end=end, freq=freq)
            tm.assert_index_equal(result, expected)
        start, end = (Timedelta(days=1), Timedelta(days=10))
        expected = interval_range(start=start, end=end)
        result = interval_range(start=start.to_pytimedelta(), end=end.to_pytimedelta())
        tm.assert_index_equal(result, expected)
        result = interval_range(start=start.asm8, end=end.asm8)
        tm.assert_index_equal(result, expected)
        equiv_freq = ['D', Day(), Timedelta(days=1), timedelta(days=1)]
        for freq in equiv_freq:
            result = interval_range(start=start, end=end, freq=freq)
            tm.assert_index_equal(result, expected)

    def test_errors(self) -> None:
        msg = 'Of the four parameters: start, end, periods, and freq, exactly three must be specified'
        with pytest.raises(ValueError, match=msg):
            interval_range(start=0)
        with pytest.raises(ValueError, match=msg):
            interval_range(end=5)
        with pytest.raises(ValueError, match=msg):
            interval_range(periods=2)
        with pytest.raises(ValueError, match=msg):
            interval_range()
        with pytest.raises(ValueError, match=msg):
            interval_range(start=0, end=5, periods=6, freq=1.5)
        msg = 'start, end, freq need to be type compatible'
        with pytest.raises(TypeError, match=msg):
            interval_range(start=0, end=Timestamp('20130101'), freq=2)
        with pytest.raises(TypeError, match=msg):
            interval_range(start=0, end=Timedelta('1 day'), freq=2)
        with pytest.raises(TypeError, match=msg):
            interval_range(start=0, end=10, freq='D')
        with pytest.raises(TypeError, match=msg):
            interval_range(start=Timestamp('20130101'), end=10, freq='D')
        with pytest.raises(TypeError, match=msg):
            interval_range(start=Timestamp('20130101'), end=Timedelta('1 day'), freq='D')
        with pytest.raises(TypeError, match=msg):
            interval_range(start=Timestamp('20130101'), end=Timestamp('20130110'), freq=2)
        with pytest.raises(TypeError, match=msg):
            interval_range(start=Timedelta('1 day'), end=10, freq='D')
        with pytest.raises(TypeError, match=msg):
            interval_range(start=Timedelta('1 day'), end=Timestamp('20130110'), freq='D')
        with pytest.raises(TypeError, match=msg):
            interval_range(start=Timedelta('1 day'), end=Timedelta('10 days'), freq=2)
        msg = 'periods must be an integer, got foo'
        with pytest.raises(TypeError, match=msg):
            interval_range(start=0, periods='foo')  # type: ignore
        msg = 'start must be numeric or datetime-like, got foo'
        with pytest.raises(ValueError, match=msg):
            interval_range(start='foo', periods=10)  # type: ignore
        msg = 'end must be numeric or datetime-like, got \\(0, 1\\]'
        with pytest.raises(ValueError, match=msg):
            interval_range(end=Interval(0, 1), periods=10)  # type: ignore
        msg = 'freq must be numeric or convertible to DateOffset, got foo'
        with pytest.raises(ValueError, match=msg):
            interval_range(start=0, end=10, freq='foo')
        with pytest.raises(ValueError, match=msg):
            interval_range(start=Timestamp('20130101'), periods=10, freq='foo')
        with pytest.raises(ValueError, match=msg):
            interval_range(end=Timedelta('1 day'), periods=10, freq='foo')
        start = Timestamp('2017-01-01', tz='US/Eastern')
        end = Timestamp('2017-01-07', tz='US/Pacific')
        msg = 'Start and end cannot both be tz-aware with different timezones'
        with pytest.raises(TypeError, match=msg):
            interval_range(start=start, end=end)

    def test_float_freq(self) -> None:
        result = interval_range(0, 1, freq=0.1)
        expected = IntervalIndex.from_breaks([0 + 0.1 * n for n in range(11)])
        tm.assert_index_equal(result, expected)
        result = interval_range(0, 1, freq=0.6)
        expected = IntervalIndex.from_breaks([0, 0.6])
        tm.assert_index_equal(result, expected)

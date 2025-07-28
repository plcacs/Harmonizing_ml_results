from datetime import timedelta
from typing import Optional, Union, Any
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
    def test_constructor_numeric(
        self,
        closed: Optional[str],
        name: Optional[str],
        freq: Union[int, float],
        periods: int
    ) -> None:
        start: int = 0
        end: int = 100
        breaks: np.ndarray = np.arange(101, step=freq)
        expected: IntervalIndex = IntervalIndex.from_breaks(breaks, name=name, closed=closed)
        result: IntervalIndex = interval_range(start=start, end=end, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(start=start, periods=periods, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(end=end, periods=periods, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(start=start, end=end, periods=periods, name=name, closed=closed)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    @pytest.mark.parametrize('freq, periods', [('D', 364), ('2D', 182), ('22D18h', 16), ('ME', 11)])
    def test_constructor_timestamp(
        self,
        closed: Optional[str],
        name: Optional[str],
        freq: str,
        periods: int,
        tz: Optional[str]
    ) -> None:
        start: Timestamp = Timestamp('20180101', tz=tz)
        end: Timestamp = Timestamp('20181231', tz=tz)
        breaks = date_range(start=start, end=end, freq=freq)
        expected: IntervalIndex = IntervalIndex.from_breaks(breaks, name=name, closed=closed)
        result: IntervalIndex = interval_range(start=start, end=end, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(start=start, periods=periods, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(end=end, periods=periods, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        if not breaks.freq.n == 1 and tz is None:
            result = interval_range(start=start, end=end, periods=periods, name=name, closed=closed)
            tm.assert_index_equal(result, expected)

    def test_constructor_timedelta(
        self,
        closed: Optional[str],
        name: Optional[str],
        freq: str,
        periods: int
    ) -> None:
        start: Timedelta = Timedelta('0 days')
        end: Timedelta = Timedelta('100 days')
        breaks = timedelta_range(start=start, end=end, freq=freq)
        expected: IntervalIndex = IntervalIndex.from_breaks(breaks, name=name, closed=closed)
        result: IntervalIndex = interval_range(start=start, end=end, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(start=start, periods=periods, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(end=end, periods=periods, freq=freq, name=name, closed=closed)
        tm.assert_index_equal(result, expected)
        result = interval_range(start=start, end=end, periods=periods, name=name, closed=closed)
        tm.assert_index_equal(result, expected)

    def test_early_truncation(
        self,
        start: Union[int, float, Timedelta, Timestamp],
        end: Union[int, float, Timedelta, Timestamp],
        freq: Union[int, float, str],
        expected_endpoint: Union[int, float, Timedelta, Timestamp]
    ) -> None:
        result: IntervalIndex = interval_range(start=start, end=end, freq=freq)
        result_endpoint: Union[int, float, Timedelta, Timestamp] = result.right[-1]
        assert result_endpoint == expected_endpoint

    @pytest.mark.parametrize('start, end, freq', [
        (0.5, None, None),
        (None, 4.5, None),
        (0.5, None, 1.5),
        (None, 6.5, 1.5)
    ])
    def test_no_invalid_float_truncation(
        self,
        start: Optional[Union[int, float]],
        end: Optional[Union[int, float]],
        freq: Optional[Union[int, float]]
    ) -> None:
        if freq is None:
            breaks = [0.5, 1.5, 2.5, 3.5, 4.5]
        else:
            breaks = [0.5, 2.0, 3.5, 5.0, 6.5]
        expected: IntervalIndex = IntervalIndex.from_breaks(breaks)
        result: IntervalIndex = interval_range(start=start, end=end, periods=4, freq=freq)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('start, mid, end', [
        (Timestamp('2018-03-10', tz='US/Eastern'),
         Timestamp('2018-03-10 23:30:00', tz='US/Eastern'),
         Timestamp('2018-03-12', tz='US/Eastern')),
        (Timestamp('2018-11-03', tz='US/Eastern'),
         Timestamp('2018-11-04 00:30:00', tz='US/Eastern'),
         Timestamp('2018-11-05', tz='US/Eastern'))
    ])
    def test_linspace_dst_transition(
        self,
        start: Timestamp,
        mid: Timestamp,
        end: Timestamp
    ) -> None:
        start = start.as_unit('ns')
        mid = mid.as_unit('ns')
        end = end.as_unit('ns')
        result: IntervalIndex = interval_range(start=start, end=end, periods=2)
        expected: IntervalIndex = IntervalIndex.from_breaks([start, mid, end])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('freq', [2, 2.0])
    @pytest.mark.parametrize('end', [10, 10.0])
    @pytest.mark.parametrize('start', [0, 0.0])
    def test_float_subtype(
        self,
        start: Union[int, float],
        end: Union[int, float],
        freq: Union[int, float]
    ) -> None:
        index: IntervalIndex = interval_range(start=start, end=end, freq=freq)
        result: str = index.dtype.subtype
        expected: str = 'int64' if is_integer(start + end + freq) else 'float64'
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

    @pytest.mark.parametrize('start, end, expected', [
        (np.int8(1), np.int8(10), np.dtype('int8')),
        (np.int8(1), np.float16(10), np.dtype('float64')),
        (np.float32(1), np.float32(10), np.dtype('float32')),
        (1, 10, np.dtype('int64')),
        (1, 10.0, np.dtype('float64'))
    ])
    def test_interval_dtype(
        self,
        start: Union[np.int8, np.float16, np.float32, int, float],
        end: Union[np.int8, np.float16, np.float32, int, float],
        expected: np.dtype
    ) -> None:
        result = interval_range(start=start, end=end).dtype.subtype
        assert result == expected

    def test_interval_range_fractional_period(self) -> None:
        msg: str = 'periods must be an integer, got 10.5'
        ts: Timestamp = Timestamp('2024-03-25')
        with pytest.raises(TypeError, match=msg):
            interval_range(ts, periods=10.5)

    def test_constructor_coverage(self) -> None:
        start: Timestamp = Timestamp('2017-01-01')
        end: Timestamp = Timestamp('2017-01-15')
        expected: IntervalIndex = interval_range(start=start, end=end)
        result: IntervalIndex = interval_range(start=start.to_pydatetime(), end=end.to_pydatetime())
        tm.assert_index_equal(result, expected)
        result = interval_range(start=start.asm8, end=end.asm8)
        tm.assert_index_equal(result, expected)
        equiv_freq: list[Any] = ['D', Day(), Timedelta(days=1), timedelta(days=1), DateOffset(days=1)]
        for freq in equiv_freq:
            result = interval_range(start=start, end=end, freq=freq)
            tm.assert_index_equal(result, expected)
        start = Timedelta(days=1)
        end = Timedelta(days=10)
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
        msg: str = 'Of the four parameters: start, end, periods, and freq, exactly three must be specified'
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
            interval_range(start=0, periods='foo')
        msg = 'start must be numeric or datetime-like, got foo'
        with pytest.raises(ValueError, match=msg):
            interval_range(start='foo', periods=10)
        msg = 'end must be numeric or datetime-like, got \\(0, 1\\]'
        with pytest.raises(ValueError, match=msg):
            interval_range(end=Interval(0, 1), periods=10)
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
        result: IntervalIndex = interval_range(0, 1, freq=0.1)
        expected: IntervalIndex = IntervalIndex.from_breaks([0 + 0.1 * n for n in range(11)])
        tm.assert_index_equal(result, expected)
        result = interval_range(0, 1, freq=0.6)
        expected = IntervalIndex.from_breaks([0, 0.6])
        tm.assert_index_equal(result, expected)
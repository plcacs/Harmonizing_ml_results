from datetime import timedelta
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas._testing as tm
from pandas import DateOffset, Interval, IntervalIndex, Timedelta, Timestamp
from pandas._libs.tslibs.offsets import BaseOffset
from pandas.core.dtypes.common import is_integer
from pandas.tseries.offsets import Day

def interval_range(
    start: Optional[Union[int, float, Timestamp, Timedelta, np.integer, np.floating]] = None,
    end: Optional[Union[int, float, Timestamp, Timedelta, np.integer, np.floating]] = None,
    periods: Optional[int] = None,
    freq: Optional[Union[int, float, str, BaseOffset, timedelta, DateOffset, Timedelta, Day]] = None,
    name: Optional[str] = None,
    closed: Literal["left", "right", "both", "neither"] = "right"
) -> IntervalIndex: ...

@pytest.fixture(params=[None, 'foo'])
def name(request: pytest.FixtureRequest) -> Optional[str]: ...

class TestIntervalRange:
    @pytest.mark.parametrize('freq, periods', [(1, 100), (2.5, 40), (5, 20), (25, 4)])
    def test_constructor_numeric(
        self,
        closed: Literal["left", "right", "both", "neither"],
        name: Optional[str],
        freq: Union[int, float],
        periods: int
    ) -> None: ...
    
    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    @pytest.mark.parametrize('freq, periods', [('D', 364), ('2D', 182), ('22D18h', 16), ('ME', 11)])
    def test_constructor_timestamp(
        self,
        closed: Literal["left", "right", "both", "neither"],
        name: Optional[str],
        freq: str,
        periods: int,
        tz: Optional[str]
    ) -> None: ...
    
    @pytest.mark.parametrize('freq, periods', [('D', 100), ('2D12h', 40), ('5D', 20), ('25D', 4)])
    def test_constructor_timedelta(
        self,
        closed: Literal["left", "right", "both", "neither"],
        name: Optional[str],
        freq: str,
        periods: int
    ) -> None: ...
    
    @pytest.mark.parametrize('start, end, freq, expected_endpoint', [
        (0, 10, 3, 9),
        (0, 10, 1.5, 9),
        (0.5, 10, 3, 9.5),
        (Timedelta('0D'), Timedelta('10D'), '2D4h', Timedelta('8D16h')),
        (Timestamp('2018-01-01'), Timestamp('2018-02-09'), 'MS', Timestamp('2018-02-01')),
        (Timestamp('2018-01-01', tz='US/Eastern'), Timestamp('2018-01-20', tz='US/Eastern'), '5D12h', Timestamp('2018-01-17 12:00:00', tz='US/Eastern'))
    ])
    def test_early_truncation(
        self,
        start: Union[int, float, Timestamp, Timedelta],
        end: Union[int, float, Timestamp, Timedelta],
        freq: Union[int, float, str],
        expected_endpoint: Union[int, float, Timestamp, Timedelta]
    ) -> None: ...
    
    @pytest.mark.parametrize('start, end, freq', [
        (0.5, None, None),
        (None, 4.5, None),
        (0.5, None, 1.5),
        (None, 6.5, 1.5)
    ])
    def test_no_invalid_float_truncation(
        self,
        start: Optional[float],
        end: Optional[float],
        freq: Optional[float]
    ) -> None: ...
    
    @pytest.mark.parametrize('start, mid, end', [
        (Timestamp('2018-03-10', tz='US/Eastern'), Timestamp('2018-03-10 23:30:00', tz='US/Eastern'), Timestamp('2018-03-12', tz='US/Eastern')),
        (Timestamp('2018-11-03', tz='US/Eastern'), Timestamp('2018-11-04 00:30:00', tz='US/Eastern'), Timestamp('2018-11-05', tz='US/Eastern'))
    ])
    def test_linspace_dst_transition(
        self,
        start: Timestamp,
        mid: Timestamp,
        end: Timestamp
    ) -> None: ...
    
    @pytest.mark.parametrize('freq', [2, 2.0])
    @pytest.mark.parametrize('end', [10, 10.0])
    @pytest.mark.parametrize('start', [0, 0.0])
    def test_float_subtype(
        self,
        start: Union[int, float],
        end: Union[int, float],
        freq: Union[int, float]
    ) -> None: ...
    
    @pytest.mark.parametrize('start, end, expected', [
        (np.int8(1), np.int8(10), np.dtype('int8')),
        (np.int8(1), np.float16(10), np.dtype('float64')),
        (np.float32(1), np.float32(10), np.dtype('float32')),
        (1, 10, np.dtype('int64')),
        (1, 10.0, np.dtype('float64'))
    ])
    def test_interval_dtype(
        self,
        start: Union[int, float, np.integer, np.floating],
        end: Union[int, float, np.integer, np.floating],
        expected: np.dtype
    ) -> None: ...
    
    def test_interval_range_fractional_period(self) -> None: ...
    
    def test_constructor_coverage(self) -> None: ...
    
    def test_errors(self) -> None: ...
    
    def test_float_freq(self) -> None: ...
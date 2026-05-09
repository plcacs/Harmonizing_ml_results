import numpy as np
from datetime import timedelta
from typing import Any, Optional, Union, overload
from pandas import DateOffset, Interval, IntervalIndex, Timedelta, Timestamp
from pandas.tseries.offsets import Day

def interval_range(
    start: Any = ...,
    end: Any = ...,
    periods: Optional[int] = ...,
    freq: Any = ...,
    name: Optional[str] = ...,
    closed: Optional[str] = ...,
) -> IntervalIndex: ...

class TestIntervalRange:
    def test_constructor_numeric(
        self, 
        closed: str, 
        name: Optional[str], 
        freq: Union[int, float], 
        periods: int
    ) -> None: ...

    def test_constructor_timestamp(
        self, 
        closed: str, 
        name: Optional[str], 
        freq: str, 
        periods: int, 
        tz: Optional[str]
    ) -> None: ...

    def test_constructor_timedelta(
        self, 
        closed: str, 
        name: Optional[str], 
        freq: str, 
        periods: int
    ) -> None: ...

    def test_early_truncation(
        self, 
        start: Union[int, float, Timedelta, Timestamp], 
        end: Union[int, float, Timedelta, Timestamp], 
        freq: Union[int, float, str], 
        expected_endpoint: Union[int, float, Timedelta, Timestamp]
    ) -> None: ...

    def test_no_invalid_float_truncation(
        self, 
        start: Optional[float], 
        end: Optional[float], 
        freq: Optional[float]
    ) -> None: ...

    def test_linspace_dst_transition(
        self, 
        start: Timestamp, 
        mid: Timestamp, 
        end: Timestamp
    ) -> None: ...

    def test_float_subtype(
        self, 
        start: Union[int, float], 
        end: Union[int, float], 
        freq: Union[int, float]
    ) -> None: ...

    def test_interval_dtype(
        self, 
        start: Union[np.generic, int, float], 
        end: Union[np.generic, int, float], 
        expected: np.dtype
    ) -> None: ...

    def test_interval_range_fractional_period(self) -> None: ...

    def test_constructor_coverage(self) -> None: ...

    def test_errors(self) -> None: ...

    def test_float_freq(self) -> None: ...
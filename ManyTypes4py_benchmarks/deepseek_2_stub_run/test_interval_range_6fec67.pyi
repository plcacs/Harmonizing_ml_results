```python
from datetime import timedelta
from typing import Any, Literal, overload
from pandas import DateOffset, Interval, IntervalIndex, Timedelta, Timestamp
from pandas._libs.tslibs.offsets import BaseOffset
import numpy as np

def interval_range(
    start: Any = ...,
    end: Any = ...,
    periods: int | None = ...,
    freq: Any = ...,
    name: str | None = ...,
    closed: Literal["left", "right", "both", "neither"] = ...
) -> IntervalIndex: ...

@pytest.fixture
def name(request: Any) -> Any: ...

class TestIntervalRange:
    def test_constructor_numeric(
        self,
        closed: Any,
        name: Any,
        freq: Any,
        periods: Any
    ) -> None: ...
    
    def test_constructor_timestamp(
        self,
        closed: Any,
        name: Any,
        freq: Any,
        periods: Any,
        tz: Any
    ) -> None: ...
    
    def test_constructor_timedelta(
        self,
        closed: Any,
        name: Any,
        freq: Any,
        periods: Any
    ) -> None: ...
    
    def test_early_truncation(
        self,
        start: Any,
        end: Any,
        freq: Any,
        expected_endpoint: Any
    ) -> None: ...
    
    def test_no_invalid_float_truncation(
        self,
        start: Any,
        end: Any,
        freq: Any
    ) -> None: ...
    
    def test_linspace_dst_transition(
        self,
        start: Any,
        mid: Any,
        end: Any
    ) -> None: ...
    
    def test_float_subtype(
        self,
        start: Any,
        end: Any,
        freq: Any
    ) -> None: ...
    
    def test_interval_dtype(
        self,
        start: Any,
        end: Any,
        expected: Any
    ) -> None: ...
    
    def test_interval_range_fractional_period(self) -> None: ...
    
    def test_constructor_coverage(self) -> None: ...
    
    def test_errors(self) -> None: ...
    
    def test_float_freq(self) -> None: ...
```
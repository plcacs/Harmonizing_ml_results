```python
from _typeshed import Incomplete
from datetime import timedelta
import numpy as np
from pandas import DateOffset, Interval, IntervalIndex, Timedelta, Timestamp
from pandas.tseries.offsets import Day
from typing import Any, Optional, Union

@pytest.fixture
def name(request: Any) -> Any: ...

class TestIntervalRange:
    @pytest.mark.parametrize
    def test_constructor_numeric(self, closed: Any, name: Any, freq: Any, periods: Any) -> None: ...
    
    @pytest.mark.parametrize
    def test_constructor_timestamp(self, closed: Any, name: Any, freq: Any, periods: Any, tz: Any) -> None: ...
    
    @pytest.mark.parametrize
    def test_constructor_timedelta(self, closed: Any, name: Any, freq: Any, periods: Any) -> None: ...
    
    @pytest.mark.parametrize
    def test_early_truncation(self, start: Any, end: Any, freq: Any, expected_endpoint: Any) -> None: ...
    
    @pytest.mark.parametrize
    def test_no_invalid_float_truncation(self, start: Any, end: Any, freq: Any) -> None: ...
    
    @pytest.mark.parametrize
    def test_linspace_dst_transition(self, start: Any, mid: Any, end: Any) -> None: ...
    
    @pytest.mark.parametrize
    def test_float_subtype(self, start: Any, end: Any, freq: Any) -> None: ...
    
    @pytest.mark.parametrize
    def test_interval_dtype(self, start: Any, end: Any, expected: Any) -> None: ...
    
    def test_interval_range_fractional_period(self) -> None: ...
    
    def test_constructor_coverage(self) -> None: ...
    
    def test_errors(self) -> None: ...
    
    def test_float_freq(self) -> None: ...

def interval_range(
    start: Optional[Union[int, float, Timestamp, Timedelta, np.generic]] = ...,
    end: Optional[Union[int, float, Timestamp, Timedelta, np.generic]] = ...,
    periods: Optional[int] = ...,
    freq: Optional[Union[int, float, str, timedelta, Timedelta, Day, DateOffset]] = ...,
    name: Any = ...,
    closed: Any = ...
) -> IntervalIndex: ...

# Re-exported imports for type checking
pytest: Incomplete
tm: Incomplete
date_range: Incomplete
timedelta_range: Incomplete
```
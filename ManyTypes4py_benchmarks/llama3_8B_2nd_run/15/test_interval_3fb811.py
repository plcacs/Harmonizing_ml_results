from typing import Any, List
import pandas as pd
import numpy as np
import pytest

class TestIntervalIndex:
    def test_properties(self, closed: str) -> None:
        # ... existing code ...

    @pytest.mark.parametrize('breaks', [date_range('20180101', periods=4), date_range('20180101', periods=4, tz='US/Eastern'), timedelta_range('0 days', periods=4)], ids=lambda x: str(x.dtype))
    @pytest.mark.parametrize('make_key', [IntervalIndex.from_breaks, lambda breaks: Interval(breaks[0], breaks[1]), lambda breaks: breaks, lambda breaks: breaks[0], list], ids=['IntervalIndex', 'Interval', 'Index', 'scalar', 'list'])
    def test_maybe_convert_i8_errors(self, breaks1: Any, breaks2: Any, make_key: Any) -> None:
        # ... existing code ...

    def test_append(self, closed: str) -> None:
        # ... existing code ...

    @pytest.mark.parametrize('start, shift, na_value', [(0, 1, np.nan), (Timestamp('2018-01-01'), Timedelta('1 day'), pd.NaT), (Timedelta('0 days'), Timedelta('1 day'), pd.NaT)])
    def test_is_overlapping(self, start: Any, shift: Any, na_value: Any, closed: str) -> None:
        # ... existing code ...

    def test_to_tuples(self, tuples: List[Any]) -> None:
        # ... existing code ...

    @pytest.mark.parametrize('tuples',